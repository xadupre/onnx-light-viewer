const { readFileSync, writeFileSync, writeSync } = require("node:fs");
const { dirname, isAbsolute, join, resolve } = require("node:path");
const native = require("../dist/native/onnx_light_viewer.node");

const request = JSON.parse(readFileSync(process.argv[2], "utf8"));
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
  throw new Error("ONNX_LIGHT_VIEWER_GENAI_LIBRARY must be an absolute path.");
const runtimePath = process.env.ORT_LIB_PATH || "";
if (runtimePath && !isAbsolute(runtimePath))
  throw new Error("ORT_LIB_PATH must be an absolute path.");
const capture = native.runNativePrompt(
  { ...request, libraryPath, runtimePath },
  (token, index) => {
    writeSync(1, `\x1e${JSON.stringify({ type: "token", token, index })}\n`);
  },
);
const result = {
  prompt: request.prompt,
  engine: request.engine,
  tokens: capture.tokens,
};
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
    ...capture,
    logits: capture.logits?.map((row) => Array.from(row)) ?? null,
  }),
  {
    flag: "wx",
    mode: 0o600,
  },
);
writeFileSync(process.argv[3], JSON.stringify(result), {
  flag: "wx",
  mode: 0o600,
});
