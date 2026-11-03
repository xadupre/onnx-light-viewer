const {
  readFileSync,
  readdirSync,
  writeFileSync,
  writeSync,
} = require("node:fs");
const { execFileSync } = require("node:child_process");
const { dirname, isAbsolute, join, resolve } = require("node:path");
const { createInterface } = require("node:readline");
const native = require("../dist/native/onnx_light_viewer.node");
const { genaiSessionOptions } = require("./prompt-settings.cjs");
const { summarizeProfile } = require("./profiling.cjs");

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
  const repeat =
    request.engine === "onnxruntime-genai" ? (request.repeat ?? 1) : 1;
  if (!Number.isSafeInteger(repeat) || repeat < 1 || repeat > 100)
    throw new Error("Repeat must be an integer from 1 to 100.");
  const { options, providers } = genaiSessionOptions(
    settings,
    settings.profile
      ? join(request.profileDirectory, `genai-${process.pid}`)
      : undefined,
  );
  const nativeRequest = {
    ...request,
    sessionSettings: settings,
    sessionKey: sessionKey ?? input,
    genaiOverlay: JSON.stringify({
      model: { decoder: { session_options: options } },
    }),
    ...(providers === undefined ? {} : { genaiProviders: providers }),
    libraryPath,
    runtimePath,
  };
  const retained =
    request.keepLoaded &&
    !(request.engine === "onnxruntime-genai" && settings.profile);
  const generate = () => {
    let capture;
    let firstLifecycle;
    for (let iteration = 0; iteration < repeat; iteration++) {
      const last = iteration === repeat - 1;
      capture = native.runNativePrompt(
        {
          ...nativeRequest,
          keepLoaded: !last || retained,
          captureBudget: last ? request.captureBudget : 0,
        },
        (token, index) => {
          if (last) emit({ type: "token", token, index });
        },
        (lifecycle) => {
          if (iteration === 0) {
            firstLifecycle = { ...lifecycle, retained: Boolean(retained) };
            emit({ type: "lifecycle", lifecycle: firstLifecycle });
          }
        },
      );
    }
    capture.lifecycle = firstLifecycle;
    return capture;
  };
  let capture;
  try {
    capture = generate();
  } catch (error) {
    if (
      process.env.ONNX_LIGHT_VIEWER_GENAI_LIBRARY ||
      !error.message?.startsWith(
        "Could not load the native ONNX Runtime GenAI library:",
      )
    )
      throw error;
    process.stderr.write(
      "Locating native GenAI SDK libraries in the Python environment selected in Settings.\n",
    );
    const libraries = JSON.parse(
      execFileSync(
        request.pythonPath,
        [
          "-I",
          "-c",
          `
import importlib.util
import json
import os
from pathlib import Path
import sys

def package_directory(name):
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None:
        raise SystemExit(
            f"Missing {name} in the Python environment selected in Settings. "
            "Install the runtime there, or set ONNX_LIGHT_VIEWER_GENAI_LIBRARY "
            "to the native C++ SDK library's absolute path."
        )
    return Path(spec.origin).parent

name = "onnxruntime-genai.dll" if sys.platform == "win32" else "libonnxruntime-genai.dylib" if sys.platform == "darwin" else "libonnxruntime-genai.so"
library = package_directory("onnxruntime_genai") / name
if not library.is_file():
    raise SystemExit(f"Native GenAI SDK library not found: {library}")
runtime = os.environ.get("ORT_LIB_PATH", "")
if not runtime:
    pattern = "onnxruntime.dll" if sys.platform == "win32" else "libonnxruntime*.dylib" if sys.platform == "darwin" else "libonnxruntime.so*"
    candidates = sorted((package_directory("onnxruntime") / "capi").glob(pattern))
    if not candidates:
        raise SystemExit("Native ONNX Runtime library not found in the selected Python environment.")
    runtime = str(candidates[0])
print(json.dumps({"libraryPath": str(library), "runtimePath": runtime}))
`,
        ],
        { encoding: "utf8", timeout: 10000, maxBuffer: 65536 },
      ),
    );
    if (
      typeof libraries.libraryPath !== "string" ||
      !isAbsolute(libraries.libraryPath) ||
      typeof libraries.runtimePath !== "string" ||
      !isAbsolute(libraries.runtimePath)
    )
      throw new Error("Native SDK discovery returned invalid library paths.");
    nativeRequest.libraryPath = libraries.libraryPath;
    nativeRequest.runtimePath = libraries.runtimePath;
    capture = generate();
  }
  const result = {
    prompt: request.prompt,
    engine: request.engine,
    tokens: capture.tokens,
    lifecycle: capture.lifecycle,
    repeat,
  };
  if (settings.profile) {
    result.profileDirectory = request.profileDirectory;
    if (request.engine === "onnxruntime-genai") {
      const profiles = readdirSync(request.profileDirectory)
        .filter((name) => name.startsWith("genai-") && name.endsWith(".json"))
        .sort();
      const summaries = profiles.map((name) => {
        const trace = JSON.parse(
          readFileSync(join(request.profileDirectory, name), "utf8"),
        );
        if (!Array.isArray(trace))
          throw new Error("ONNX Runtime returned an invalid profiling trace.");
        const profile = summarizeProfile(trace);
        if (profiles.length > 1)
          for (const node of profile.nodes)
            node.name = `${name} / ${node.name}`;
        return profile;
      });
      result.profiling = {
        trace: summaries.flatMap((profile) => profile.trace),
        nodes: summaries
          .flatMap((profile) => profile.nodes)
          .sort((a, b) => b.durationUs - a.durationUs),
      };
    } else
      writeFileSync(
        join(
          request.profileDirectory,
          `onnx-light-${process.pid}-${Date.now()}.json`,
        ),
        JSON.stringify(capture.profileEvents),
        { flag: "wx", mode: 0o600 },
      );
  }
  if (capture.logits === null)
    result.limitations = [
      "Vocabulary logits exceeded the retention budget; discrepancy metrics " +
        "are unavailable. Token timings remain available. " +
        (request.engine === "onnx-light"
          ? "Raw scores are unavailable for steps without an SDK logit snapshot."
          : "Selected raw scores remain available."),
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
