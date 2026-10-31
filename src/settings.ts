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
  "onnxscript",
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
  "mobius-onnx",
  "olive-ai",
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

export type PackageRelease = {
  name: string;
  url: string;
} & ({ version: string; error?: never } | { error: string; version?: never });

export async function latestPackageVersions(
  signal?: AbortSignal,
): Promise<PackageRelease[]> {
  const sources = [
    ...[
      ["onnx-light-viewer", "onnx-light-viewer"],
      ["onnx-light C++", "onnx-light"],
    ].map(([name, repository]) => ({
      name,
      api: `https://api.github.com/repos/xadupre/${repository}/releases/latest`,
      url: `https://github.com/xadupre/${repository}/releases/latest`,
      github: true,
    })),
    ...pythonPackages.map((name) => ({
      name,
      api: `https://pypi.org/pypi/${name}/json`,
      url: `https://pypi.org/project/${name}/`,
      github: false,
    })),
  ];
  const requestSignal = AbortSignal.any([
    AbortSignal.timeout(15000),
    ...(signal ? [signal] : []),
  ]);
  const results = new Array<PackageRelease>(sources.length);
  let next = 0;
  await Promise.all(
    Array.from({ length: 6 }, async () => {
      while (next < sources.length) {
        const index = next++;
        const source = sources[index];
        try {
          const response = await fetch(source.api, {
            signal: requestSignal,
            headers: {
              Accept: "application/json",
              "User-Agent": "onnx-light-viewer",
            },
          });
          if (!response.ok) {
            await response.body?.cancel();
            if (response.status === 404)
              throw new Error(
                source.github
                  ? "No published GitHub release."
                  : "No published PyPI project.",
              );
            throw new Error(`HTTP ${response.status} ${response.statusText}`);
          }
          const metadata: unknown = await response.json();
          const info =
            !source.github &&
            metadata &&
            typeof metadata === "object" &&
            "info" in metadata
              ? metadata.info
              : metadata;
          let version: unknown;
          if (info && typeof info === "object") {
            if (source.github && "tag_name" in info) version = info.tag_name;
            else if (!source.github && "version" in info)
              version = info.version;
          }
          if (typeof version !== "string" || !/^v?\d[\w.!+-]*$/.test(version))
            throw new Error("Invalid published version metadata.");
          results[index] = {
            name: source.name,
            url: source.url,
            version: version.replace(/^v(?=\d)/, ""),
          };
        } catch (error) {
          if (signal?.aborted) throw error;
          results[index] = {
            name: source.name,
            url: source.url,
            error: error instanceof Error ? error.message : String(error),
          };
        }
      }
    }),
  );
  return results;
}

/** Reads distribution metadata without importing model libraries or workspace code. */
export async function inspectPythonPackages(pythonPath: string): Promise<{
  executable: string;
  version: string;
  packages: { name: string; version: string | null; source?: string }[];
}> {
  const script = `
import ast
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
python_paths = []
for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep):
    if entry and os.path.isdir(entry):
        directory = os.path.abspath(entry)
        python_paths.append(directory)
        add_path(directory)
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
modules = {
    "mobius-onnx": "mobius",
    "olive-ai": "olive",
    "yet-another-onnx-builder": "yobx",
    "scikit-learn": "sklearn",
    "scikit-survival": "sksurv",
    "imbalanced-learn": "imblearn",
    "optimum-onnx": "optimum.onnx",
    "onnxruntime-gpu": None,
}
def source_version(package):
    candidates = [package] if package.is_file() else [
        package / "__init__.py", package / "_version.py", package / "version.py"
    ]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            statements = ast.parse(candidate.read_bytes(), filename=str(candidate)).body
        except (OSError, SyntaxError, UnicodeError):
            continue
        for statement in statements:
            if isinstance(statement, ast.Assign):
                targets = statement.targets
            elif isinstance(statement, ast.AnnAssign):
                targets = [statement.target]
            else:
                continue
            if any(isinstance(target, ast.Name) and target.id in ("__version__", "version", "VERSION") for target in targets):
                value = statement.value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    return value.value
    return None
def package_info(name):
    module = modules.get(name, name.replace("-", "_"))
    if module:
        for directory in python_paths:
            package = Path(directory).joinpath(*module.split("."))
            if not package.is_dir():
                package = package.with_suffix(".py")
            if not package.exists():
                continue
            version = source_version(package)
            if version is None:
                version = next((d.version for d in metadata.distributions(path=[directory])
                                if normalize(d.metadata["Name"] or "") == normalize(name)), None)
            return {"name": name, "version": version, "source": str(package)}
    return {"name": name, "version": installed.get(normalize(name))}
print(json.dumps({
    "executable": sys.executable,
    "version": sys.version.split()[0],
    "packages": [package_info(name) for name in json.loads(sys.argv[1])]
}))
`;
  const { stdout } = await promisify(execFile)(
    pythonPath,
    ["-I", "-S", "-c", script, JSON.stringify(pythonPackages)],
    { cwd: homedir(), timeout: 10000, maxBuffer: 1024 * 1024 },
  );
  return JSON.parse(stdout);
}

export async function supportedModelIds(pythonPath: string): Promise<string[]> {
  const { stdout } = await promisify(execFile)(
    pythonPath,
    [
      "-c",
      "import json; from modelbuilder.architectures import list_supported_model_ids; print(json.dumps(list_supported_model_ids()))",
    ],
    { cwd: homedir(), timeout: 10000, maxBuffer: 1024 * 1024 },
  );
  const ids: unknown = JSON.parse(stdout);
  if (
    !Array.isArray(ids) ||
    !ids.every(
      (id): id is string =>
        typeof id === "string" && /^[\w.-]+\/[\w.-]+$/.test(id),
    )
  )
    throw new Error("Invalid mbext model ID suggestions.");
  return [...new Set(ids)].sort();
}
