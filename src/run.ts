import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { isAbsolute, join, resolve } from "node:path";
import { deserialize } from "node:v8";
import { runProcess } from "./export";
import { GraphPath, NativeModel, RunInputSource, RunResult } from "./graph";
import { PromptSession } from "./promptSession";

/** Panel-owned prediction session; engine/model/stage/settings changes release it. */
export class RunSession {
  generation = 0;
  private model?: NativeModel;
  private key?: string;
  private worker?: PromptSession;
  private closing?: Promise<void>;

  constructor(
    private readonly reportError: (error: Error) => void = console.error,
  ) {}

  get alive(): boolean {
    return this.worker?.alive === true;
  }

  async ensure(
    model: NativeModel,
    path: GraphPath,
    engine: string,
    settings: Record<string, unknown>,
    python: string,
    runner: string,
    signal: AbortSignal,
  ): Promise<number> {
    if (this.closing) await this.closing;
    const key = JSON.stringify([path, engine, settings, python, runner]);
    if (this.model === model && this.key === key) return this.generation;
    await this.close();
    signal.throwIfAborted();
    this.model = model;
    this.key = key;
    if (engine === "onnxruntime" || Number(settings.verbose) > 0) {
      const worker = new PromptSession(
        (error) => {
          if (this.worker !== worker) return;
          void this.close().catch(this.reportError);
          this.reportError(error);
        },
        {
          executable: engine === "onnxruntime" ? python : process.execPath,
          args:
            engine === "onnxruntime"
              ? ["-u", runner, "--worker"]
              : [
                  resolve(__dirname, "..", "media", "run_model.cjs"),
                  "--worker",
                ],
          stderrEncoding:
            engine === "onnxruntime" && process.platform === "win32"
              ? "utf16le"
              : "utf8",
          label: engine === "onnxruntime" ? "ONNX Runtime" : "onnx-light",
        },
      );
      this.worker = worker;
    }
    return this.generation;
  }

  run(
    directory: string,
    signal: AbortSignal,
    progress: (output: string) => void,
  ): Promise<void> {
    signal.throwIfAborted();
    if (!this.worker || !this.key)
      throw new Error("The ONNX Runtime session is not configured.");
    return this.worker.run(
      directory,
      join(directory, "result.json"),
      this.key,
      signal,
      progress,
      () => {},
    );
  }

  close(): Promise<void> {
    if (this.closing) return this.closing;
    this.generation++;
    const worker = this.worker;
    this.model?.releaseRunSession();
    this.model = undefined;
    this.key = undefined;
    this.closing = (worker?.close() ?? Promise.resolve())
      .then(() => {
        if (this.worker === worker) this.worker = undefined;
      })
      .finally(() => {
        this.closing = undefined;
      });
    return this.closing;
  }
}

export async function runModel(
  model: NativeModel,
  path: GraphPath,
  request: Record<string, unknown>,
  pythonPath: string,
  runner: string,
  progress: (output: string) => void,
  signal: AbortSignal = new AbortController().signal,
  session?: RunSession,
): Promise<RunResult> {
  let generation = session?.generation;
  let releasing: Promise<void> | undefined;
  const release = () => {
    if (!releasing && session?.generation === generation)
      releasing = session?.close();
    return releasing;
  };
  const abort = () => {
    void release()?.catch((error: unknown) => {
      progress(
        `Could not release the Run session: ${error instanceof Error ? error.message : String(error)}`,
      );
    });
  };
  signal.addEventListener("abort", abort, { once: true });
  try {
    signal.throwIfAborted();
    const engine = request.engine ?? "onnx-light";
    if (engine !== "onnx-light" && engine !== "onnxruntime")
      throw new Error("Choose a supported execution engine.");
    if (!Array.isArray(request.inputs))
      throw new Error("Expected model inputs.");
    if (
      request.keepLoaded !== undefined &&
      typeof request.keepLoaded !== "boolean"
    )
      throw new Error("Expected a boolean Keep session loaded setting.");
    const dimensions = request.dimensions ?? {};
    if (
      !dimensions ||
      typeof dimensions !== "object" ||
      Array.isArray(dimensions) ||
      Object.values(dimensions).some(
        (value) => !Number.isSafeInteger(value) || (value as number) < 0,
      )
    )
      throw new Error(
        "Specify non-negative safe integers for dynamic dimensions.",
      );
    const settings = request.settings ?? {};
    if (!settings || typeof settings !== "object" || Array.isArray(settings))
      throw new Error("Invalid engine logging settings.");
    const logging = settings as Record<string, unknown>;
    if (
      logging.enableProfiling !== undefined &&
      typeof logging.enableProfiling !== "boolean"
    )
      throw new Error("Expected a boolean Enable profiling setting.");
    const enableProfiling =
      engine === "onnxruntime" && logging.enableProfiling === true;
    const keepLoaded = request.keepLoaded === true && !enableProfiling;
    const retainedSession = keepLoaded ? session : undefined;
    if (keepLoaded && !retainedSession)
      throw new Error("Retained execution requires a panel-owned Run session.");
    for (const [key, maximum] of engine === "onnx-light"
      ? ([["verbose", 10]] as const)
      : ([
          ["logSeverity", 4],
          ["logVerbosity", 2147483647],
        ] as const)) {
      const value = logging[key] ?? (key === "logSeverity" ? 2 : 0);
      if (
        !Number.isSafeInteger(value) ||
        (value as number) < 0 ||
        (value as number) > maximum
      )
        throw new Error(`Invalid ${key} logging setting.`);
    }
    const metadata = model.runInputs(path);
    for (const input of metadata)
      input.shape?.forEach((dimension, axis) => {
        if (
          input.dynamicAxes
            ? !input.dynamicAxes.includes(axis)
            : dimension !== null && /^\d+$/.test(dimension)
        )
          return;
        const key =
          dimension === null
            ? `axis:${input.name}:${axis}`
            : `symbol:${dimension}`;
        if (!Object.hasOwn(dimensions, key))
          throw new Error(`Specify dynamic dimension: ${key}`);
      });
    const inputs: RunInputSource[] = [];
    for (const input of request.inputs) {
      if (typeof input === "string") {
        inputs.push(input);
        continue;
      }
      if (
        !input ||
        typeof input !== "object" ||
        Array.isArray(input) ||
        typeof input.name !== "string"
      )
        throw new Error("Expected input tensor text or a named input source.");
      if (!metadata.some((value) => value.name === input.name))
        throw new Error(`Unknown model input: ${input.name}`);
      if (input.source === "random")
        inputs.push({ name: input.name, source: "random" });
      else if (input.source === "protobuf") {
        if (
          typeof input.path !== "string" ||
          !isAbsolute(input.path) ||
          input.path.includes("\0")
        )
          throw new Error("Choose an absolute TensorProto file path.");
        inputs.push({
          name: input.name,
          source: "protobuf",
          data: await readFile(input.path),
        });
      } else
        throw new Error(
          "Choose tensor text, a TensorProto file or random input.",
        );
    }
    const options = {
      dimensions: dimensions as Record<string, number>,
      verbose:
        engine === "onnx-light"
          ? ((logging.verbose as number | undefined) ?? 0)
          : 0,
    };
    signal.throwIfAborted();
    if (retainedSession)
      generation = await retainedSession.ensure(
        model,
        path,
        engine,
        logging,
        pythonPath,
        runner,
        signal,
      );
    else if (session) {
      const closing = session.close();
      generation = session.generation;
      await closing;
    }
    signal.throwIfAborted();
    if (engine === "onnx-light" && options.verbose === 0) {
      const { log, ...result } = await model.run(inputs, path, {
        ...options,
        keepLoaded,
      });
      signal.throwIfAborted();
      if (log) progress(log);
      return result;
    }
    if (engine === "onnx-light") {
      const directory = await mkdtemp(join(tmpdir(), "onnx-light-run-"));
      let result: RunResult;
      try {
        if (!retainedSession?.alive) {
          const snapshot = await model.prepareRun([], path, options);
          signal.throwIfAborted();
          await writeFile(join(directory, "model.onnx"), snapshot.model);
        }
        const sources = [];
        for (const [index, input] of inputs.entries()) {
          if (typeof input !== "string" && input.source === "protobuf") {
            await writeFile(join(directory, `input-${index}.pb`), input.data);
            sources.push({ name: input.name, source: input.source });
          } else sources.push(input);
        }
        await writeFile(
          join(directory, "request.json"),
          JSON.stringify({
            inputs: sources,
            options: { ...options, keepLoaded },
          }),
        );
        signal.throwIfAborted();
        if (retainedSession)
          await retainedSession.run(directory, signal, progress);
        else
          await runProcess(
            {
              executable: process.execPath,
              args: [
                resolve(__dirname, "..", "media", "run_model.cjs"),
                directory,
              ],
            },
            signal,
            progress,
            { ELECTRON_RUN_AS_NODE: "1" },
            "onnx-light",
          );
        result = deserialize(
          await readFile(join(directory, "result.bin")),
        ) as RunResult;
      } finally {
        await rm(directory, { recursive: true, force: true });
      }
      signal.throwIfAborted();
      if (
        retainedSession &&
        (!retainedSession.alive || retainedSession.generation !== generation)
      )
        result.lifecycle.retained = false;
      return result;
    }
    const prepared = await model.prepareRun(inputs, path, options);
    signal.throwIfAborted();
    const directory = await mkdtemp(join(tmpdir(), "onnx-light-run-"));
    let result: Omit<RunResult, "outputs"> & {
      outputs: (RunResult["outputs"][number] & { hasData?: boolean })[];
    };
    try {
      await writeFile(join(directory, "model.onnx"), prepared.model);
      for (const [index, bytes] of prepared.inputs.entries())
        await writeFile(join(directory, `input-${index}.pb`), bytes);
      await writeFile(
        join(directory, "settings.json"),
        JSON.stringify({
          logSeverity: logging.logSeverity ?? 2,
          logVerbosity: logging.logVerbosity ?? 0,
          enableProfiling,
          inputs: prepared.inputs.length,
          keepLoaded,
        }),
      );
      signal.throwIfAborted();
      if (retainedSession)
        await retainedSession.run(directory, signal, progress);
      else
        await runProcess(
          { executable: pythonPath, args: ["-u", runner, directory] },
          signal,
          progress,
          {},
          "ONNX Runtime",
          undefined,
          process.platform === "win32" ? "utf16le" : "utf8",
        );
      result = JSON.parse(
        await readFile(join(directory, "result.json"), "utf8"),
      );
      for (const [index, input] of (result.inputs ?? []).entries())
        input.data = prepared.inputs[index];
      for (const [index, output] of result.outputs.entries()) {
        if (output.hasData)
          output.data = await readFile(join(directory, `output-${index}.pb`));
        delete output.hasData;
      }
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
    signal.throwIfAborted();
    if (
      retainedSession &&
      (!retainedSession.alive || retainedSession.generation !== generation)
    )
      result.lifecycle.retained = false;
    return result;
  } catch (error) {
    await release();
    throw error;
  } finally {
    signal.removeEventListener("abort", abort);
  }
}
