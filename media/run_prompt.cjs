const { readFileSync, writeFileSync, writeSync } = require("node:fs");
const { dirname, isAbsolute, join, resolve } = require("node:path");
const { createInterface } = require("node:readline");
const native = require("../dist/native/onnx_light_viewer.node");

const emit = (event) => writeSync(1, `\x1e${JSON.stringify(event)}\n`);

function run(input, output, sessionKey) {
  const request = JSON.parse(readFileSync(input, "utf8"));
  if (request.engine === "onnxruntime-genai") {
    const directory = dirname(request.modelPath);
    const config = JSON.parse(
      readFileSync(join(directory, "genai_config.json"), "utf8"),
    );
    const decoder = config.model.decoder.filename;
    if (decoder && resolve(directory, decoder) !== resolve(request.modelPath))
      throw new Error(
        "The selected model is not the GenAI configuration's decoder.",
      );
  }
  const libraryPath =
    process.env.ONNX_LIGHT_VIEWER_GENAI_LIBRARY ||
    (process.platform === "win32"
      ? "onnxruntime-genai.dll"
      : process.platform === "darwin"
        ? "libonnxruntime-genai.dylib"
        : "libonnxruntime-genai.so");
  if (process.env.ONNX_LIGHT_VIEWER_GENAI_LIBRARY && !isAbsolute(libraryPath))
    throw new Error(
      "ONNX_LIGHT_VIEWER_GENAI_LIBRARY must be an absolute path.",
    );
  const runtimePath = process.env.ORT_LIB_PATH || "";
  if (runtimePath && !isAbsolute(runtimePath))
    throw new Error("ORT_LIB_PATH must be an absolute path.");
  const settings = request.sessionSettings ?? {};
  const options = {};
  for (const [key, name] of [
    ["intraOpThreads", "intra_op_num_threads"],
    ["interOpThreads", "inter_op_num_threads"],
    ["graphOptimization", "graph_optimization_level"],
    ["cpuArena", "enable_cpu_mem_arena"],
    ["memoryPattern", "enable_mem_pattern"],
    ["logSeverity", "log_severity_level"],
  ])
    if (settings[key] !== undefined) options[name] = settings[key];
  if (settings.parallelExecution !== undefined)
    options["session.execution_mode"] = settings.parallelExecution
      ? "parallel"
      : "sequential";
  if (settings.prepacking !== undefined)
    options["session.disable_prepacking"] = settings.prepacking ? "0" : "1";
  if (settings.profile)
    options.enable_profiling = join(
      request.profileDirectory,
      `genai-${process.pid}`,
    );
  const capture = native.runNativePrompt(
    {
      ...request,
      sessionSettings: settings,
      sessionKey: sessionKey ?? input,
      genaiOverlay: JSON.stringify({
        model: { decoder: { session_options: options } },
      }),
      libraryPath,
      runtimePath,
    },
    (token, index) => emit({ type: "token", token, index }),
    (lifecycle) => emit({ type: "lifecycle", lifecycle }),
  );
  const result = {
    prompt: request.prompt,
    engine: request.engine,
    tokens: capture.tokens,
    lifecycle: capture.lifecycle,
  };
  if (settings.profile) {
    result.profileDirectory = request.profileDirectory;
    if (request.engine === "onnx-light")
      writeFileSync(
        join(
          request.profileDirectory,
          `onnx-light-${process.pid}-${Date.now()}.json`,
        ),
        JSON.stringify(capture.profileEvents),
        { flag: "wx", mode: 0o600 },
      );
  }
  if (request.engine === "onnx-light")
    result.limitations = [
      "onnx-light Generate returns completed token IDs only: per-token streaming, " +
        "timings, raw scores and vocabulary comparisons require an upstream SDK " +
        "generation observer (xadupre/onnx-light#5692).",
    ];
  else if (capture.logits === null)
    result.limitations = [
      "Vocabulary logits exceeded the retention budget; discrepancy metrics " +
        "are unavailable. Selected raw scores and token timings remain available.",
    ];
  writeFileSync(
    request.capturePath,
    JSON.stringify({
      promptIds: capture.promptIds,
      tokens: capture.tokens,
      logits: capture.logits?.map((row) => Array.from(row)) ?? null,
    }),
    { flag: "wx", mode: 0o600 },
  );
  writeFileSync(output, JSON.stringify(result), { flag: "wx", mode: 0o600 });
}

if (process.argv[2] === "--worker") {
  const input = createInterface({ input: process.stdin });
  input.on("line", (line) => {
    try {
      const request = JSON.parse(line);
      run(request.input, request.output, request.key);
      emit({ type: "done" });
    } catch (error) {
      native.releaseNativePrompt();
      emit({ type: "error", error: error.stack ?? String(error) });
    }
  });
  input.on("close", () => native.releaseNativePrompt());
} else {
  run(process.argv[2], process.argv[3]);
}
