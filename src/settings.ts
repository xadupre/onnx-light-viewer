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
  "yet-another-onnx-builder",
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
import os
from pathlib import Path
import re
import site
import sys
prefixes = [sys.prefix, sys.exec_prefix]
executable = os.environ.get("__PYVENV_LAUNCHER__", sys.executable) if sys.platform == "darwin" else sys.executable
executable_directory = Path(executable).absolute().parent
for config in (executable_directory / "pyvenv.cfg", executable_directory.parent / "pyvenv.cfg"):
    if not config.is_file():
        continue
    include_system = True
    for line in config.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip().lower() == "include-system-site-packages":
            include_system = value.strip().lower() == "true"
    sys.prefix = sys.exec_prefix = str(config.parent)
    prefixes = [sys.prefix]
    if include_system:
        prefixes.extend([sys.base_prefix, sys.base_exec_prefix])
    break
paths = []
seen = set()
def add_path(path):
    path = os.path.normcase(os.path.abspath(path))
    if path not in seen and os.path.exists(path):
        seen.add(path)
        paths.append(path)
for directory in site.getsitepackages(prefixes):
    if not os.path.isdir(directory):
        continue
    add_path(directory)
    for entry in sorted(Path(directory).glob("*.pth")):
        if entry.name.startswith(".") or not entry.is_file():
            continue
        try:
            contents = entry.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            contents = entry.read_text(encoding="locale")
        for line in contents.splitlines():
            if not line.strip() or line.startswith(("#", "import ", "import\\t")):
                continue
            add_path(os.path.join(directory, line.rstrip()))
normalize = lambda name: re.sub(r"[-_.]+", "-", name).lower()
installed = {}
# Keep environment paths out of sys.path so only distribution metadata is read.
for distribution in metadata.distributions(path=paths):
    name = distribution.metadata["Name"]
    if name:
        installed.setdefault(normalize(name), distribution.version)
print(json.dumps({
    "executable": sys.executable,
    "version": sys.version.split()[0],
    "packages": [{"name": name, "version": installed.get(normalize(name))} for name in json.loads(sys.argv[1])]
}))
`;
  const { stdout } = await promisify(execFile)(
    pythonPath,
    ["-I", "-S", "-c", script, JSON.stringify(pythonPackages)],
    { cwd: homedir(), timeout: 10000, maxBuffer: 1024 * 1024 },
  );
  return JSON.parse(stdout);
}
