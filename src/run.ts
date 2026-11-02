import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { isAbsolute, join, resolve } from "node:path";
import { deserialize } from "node:v8";
import { runProcess } from "./export";
import { GraphPath, NativeModel, RunInputSource, RunResult } from "./graph";

export async function runModel(
  model: NativeModel,
  path: GraphPath,
  request: Record<string, unknown>,
  pythonPath: string,
  runner: string,
  progress: (output: string) => void,
  signal: AbortSignal = new AbortController().signal,
): Promise<RunResult> {
  signal.throwIfAborted();
  const engine = request.engine ?? "onnx-light";
  if (engine !== "onnx-light" && engine !== "onnxruntime")
    throw new Error("Choose a supported execution engine.");
  if (!Array.isArray(request.inputs)) throw new Error("Expected model inputs.");
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
  if (engine === "onnx-light") {
    if (options.verbose === 0) {
      const { log, ...result } = await model.run(inputs, path, options);
      if (log) progress(log);
      return result;
    }
    const directory = await mkdtemp(join(tmpdir(), "onnx-light-run-"));
    try {
      const snapshot = await model.prepareRun([], path, options);
      await writeFile(join(directory, "model.onnx"), snapshot.model);
      const sources = [];
      for (const [index, input] of inputs.entries()) {
        if (typeof input !== "string" && input.source === "protobuf") {
          await writeFile(join(directory, `input-${index}.pb`), input.data);
          sources.push({ name: input.name, source: input.source });
        } else sources.push(input);
      }
      await writeFile(
        join(directory, "request.json"),
        JSON.stringify({ inputs: sources, options }),
      );
      await runProcess(
        {
          executable: process.execPath,
          args: [resolve(__dirname, "..", "media", "run_model.cjs"), directory],
        },
        signal,
        progress,
        { ELECTRON_RUN_AS_NODE: "1" },
        "onnx-light",
      );
      return deserialize(
        await readFile(join(directory, "result.bin")),
      ) as RunResult;
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
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
      }),
    );
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
    ) as Omit<RunResult, "outputs"> & {
      outputs: (RunResult["outputs"][number] & { hasData?: boolean })[];
    };
    for (const [index, input] of (result.inputs ?? []).entries())
      input.data = prepared.inputs[index];
    for (const [index, output] of result.outputs.entries()) {
      if (output.hasData)
        output.data = await readFile(join(directory, `output-${index}.pb`));
      delete output.hasData;
    }
    return result;
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
}
