import { execFile } from "node:child_process";
import { homedir } from "node:os";
import { isAbsolute } from "node:path";
import { promisify } from "node:util";

export interface ApplicationSettings {
  downloadDirectory: string;
  pythonPath: string;
}

export function applicationSettings(value: unknown): ApplicationSettings {
  if (
    !value ||
    typeof value !== "object" ||
    !("downloadDirectory" in value) ||
    typeof value.downloadDirectory !== "string" ||
    !isAbsolute(value.downloadDirectory) ||
    value.downloadDirectory.includes("\0") ||
    !("pythonPath" in value) ||
    typeof value.pythonPath !== "string" ||
    !value.pythonPath.trim() ||
    value.pythonPath.includes("\0") ||
    value.pythonPath.startsWith("-")
  )
    throw new Error(
      "Enter an absolute download directory and a Python executable.",
    );
  return {
    downloadDirectory: value.downloadDirectory,
    pythonPath: value.pythonPath,
  };
}

export const pythonPackages = [
  "onnx",
  "onnxruntime",
  "onnxruntime-gpu",
  "onnxruntime-extensions",
  "onnxruntime-genai",
  "onnx-light",
  "onnx-light-cpu",
  "onnx-light-ort",
  "optimum",
  "optimum-onnx",
  "transformers",
  "diffusers",
  "modelbuilder",
  "yobx",
  "skl2onnx",
  "tf2onnx",
  "onnxmltools",
  "torch",
  "tensorflow",
  "keras",
  "ai-edge-litert",
  "scikit-learn",
  "scikit-survival",
  "category-encoders",
  "imbalanced-learn",
  "statsmodels",
  "xgboost",
  "lightgbm",
  "catboost",
  "coremltools",
  "numpy",
  "pandas",
  "polars",
  "jax",
];

/** Reads distribution metadata without importing model libraries or workspace code. */
export async function inspectPythonPackages(pythonPath: string): Promise<{
  executable: string;
  version: string;
  packages: { name: string; version: string | null }[];
}> {
  const script = `
import importlib.metadata as metadata
import json
import re
import sys
normalize = lambda name: re.sub(r"[-_.]+", "-", name).lower()
installed = {normalize(d.metadata["Name"]): d.version for d in metadata.distributions() if d.metadata["Name"]}
print(json.dumps({
    "executable": sys.executable,
    "version": sys.version.split()[0],
    "packages": [{"name": name, "version": installed.get(normalize(name))} for name in json.loads(sys.argv[1])]
}))
`;
  const { stdout } = await promisify(execFile)(
    pythonPath,
    ["-I", "-c", script, JSON.stringify(pythonPackages)],
    { cwd: homedir(), timeout: 10000, maxBuffer: 1024 * 1024 },
  );
  return JSON.parse(stdout);
}
