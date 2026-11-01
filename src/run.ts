import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { isAbsolute, join } from "node:path";
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
    if (engine === "onnxruntime")
      this.worker = new PromptSession(this.reportError, {
        executable: python,
        args: ["-u", runner, "--worker"],
        stderrEncoding: process.platform === "win32" ? "utf16le" : "utf8",
        label: "ONNX Runtime",
      });
    return this.generation;
  }

  run(
    directory: string,
    signal: AbortSignal,
    progress: (output: string) => void,
  ): Promise<void> {
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
    this.worker = undefined;
    this.model?.releaseRunSession();
    this.model = undefined;
    this.key = undefined;
    this.closing = (worker?.close() ?? Promise.resolve()).finally(() => {
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
  signal.throwIfAborted();
  const engine = request.engine ?? "onnx-light";
  if (engine !== "onnx-light" && engine !== "onnxruntime")
    throw new Error("Choose a supported execution engine.");
  if (!Array.isArray(request.inputs)) throw new Error("Expected model inputs.");
  if (
    request.keepLoaded !== undefined &&
    typeof request.keepLoaded !== "boolean"
  )
    throw new Error("Expected a boolean Keep session loaded setting.");
  const keepLoaded = request.keepLoaded === true;
  const retainedSession = keepLoaded ? session : undefined;
  if (keepLoaded && !retainedSession)
    throw new Error("Retained execution requires a panel-owned Run session.");
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
  let generation = session?.generation;
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
  const release = () =>
    session?.generation === generation ? session?.close() : undefined;
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
    if (engine === "onnx-light") {
      const { log, ...result } = await model.run(inputs, path, {
        ...options,
        keepLoaded,
      });
      signal.throwIfAborted();
      if (log) progress(log);
      return result;
    }
    const prepared = await model.prepareRun(inputs, path, options);
    signal.throwIfAborted();
    const directory = await mkdtemp(join(tmpdir(), "onnx-light-run-"));
    try {
      await writeFile(join(directory, "model.onnx"), prepared.model);
      for (const [index, bytes] of prepared.inputs.entries())
        await writeFile(join(directory, `input-${index}.pb`), bytes);
      await writeFile(
        join(directory, "settings.json"),
        JSON.stringify({
          logSeverity: logging.logSeverity ?? 2,
          logVerbosity: logging.logVerbosity ?? 0,
          inputs: prepared.inputs.length,
          keepLoaded,
        }),
      );
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
      const result = JSON.parse(
        await readFile(join(directory, "result.json"), "utf8"),
      ) as Omit<RunResult, "inputs" | "inputData" | "outputData">;
      return {
        ...result,
        inputs: prepared.inputPreviews,
        inputData: prepared.inputs,
        outputData: await Promise.all(
          result.outputs.map((output, index) =>
            output.canSave
              ? readFile(join(directory, `output-${index}.pb`))
              : null,
          ),
        ),
      };
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  } catch (error) {
    await release();
    throw error;
  } finally {
    signal.removeEventListener("abort", abort);
  }
}
