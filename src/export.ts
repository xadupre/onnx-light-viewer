import { execFile, spawn, type ChildProcess } from "node:child_process";
import {
  lstat,
  mkdir,
  mkdtemp,
  readFile,
  readdir,
  rm,
  writeFile,
} from "node:fs/promises";
import { homedir } from "node:os";
import { basename, dirname, isAbsolute, join, resolve } from "node:path";
import { setTimeout as delay } from "node:timers/promises";
import { stripVTControlCharacters } from "node:util";
import type { Memento } from "vscode";

const exportEnvironment = {
  PYTHONUNBUFFERED: "1",
  HF_HUB_DISABLE_PROGRESS_BARS: "0",
  TQDM_POSITION: "-1",
};

const offlineEnvironment = {
  HF_HUB_OFFLINE: "1",
  TRANSFORMERS_OFFLINE: "1",
};

const downloadSource = `
import json
import pathlib
import sys
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

model, cache, result, include_weights = sys.argv[1:]
info = HfApi().model_info(model)
if not info.sha:
    raise ValueError("The repository did not provide an immutable revision.")
files = {entry.rfilename for entry in info.siblings}
checkpoint = next((name for name in (
    "model.safetensors.index.json", "model.safetensors",
    "pytorch_model.bin.index.json", "pytorch_model.bin",
) if name in files), None)
if include_weights == "0":
    weights = set()
elif checkpoint is None:
    raise ValueError("No supported root checkpoint found; use an absolute local model directory.")
elif checkpoint.endswith(".index.json"):
    index = hf_hub_download(model, checkpoint, revision=info.sha, cache_dir=cache)
    with open(index, encoding="utf-8") as stream:
        weights = set(json.load(stream)["weight_map"].values())
    if not weights or not weights <= files:
        raise ValueError("The checkpoint index references missing weight files.")
else:
    weights = {checkpoint}
metadata = {name for name in files if name.endswith((
    ".json", ".model", ".txt", ".tiktoken", ".jinja", ".vocab", ".merges",
))}
print(f"Downloading {model}@{info.sha} into {cache}", flush=True)
source = snapshot_download(model, revision=info.sha, cache_dir=cache,
                           allow_patterns=sorted(metadata | weights))
with open(result, "w", encoding="utf-8") as stream:
    json.dump(str(pathlib.Path(source).resolve()), stream)
print(f"Source ready: {source}", flush=True)
`;

type Exporter = "mbext" | "modelbuilder" | "mobius";

function isExporter(value: unknown): value is Exporter {
  return value === "mbext" || value === "modelbuilder" || value === "mobius";
}

interface ExportHistorySnapshot {
  models: string[];
  directories: string[];
  exporter?: Exporter;
}

/** Keeps recent export inputs across sessions and serializes edits across panels. */
export class ExportHistory {
  private pending: Promise<void> = Promise.resolve();

  constructor(private readonly storage: Pick<Memento, "get" | "update">) {}

  snapshot(): ExportHistorySnapshot {
    const value = this.storage.get<unknown>("exportHistory");
    if (value === undefined) return { models: [], directories: [] };
    if (
      !value ||
      typeof value !== "object" ||
      !("models" in value) ||
      !("directories" in value) ||
      !Array.isArray(value.models) ||
      !Array.isArray(value.directories) ||
      ![...value.models, ...value.directories].every(
        (entry) =>
          typeof entry === "string" && entry.trim() && !/[\0\r\n]/.test(entry),
      )
    )
      throw new Error("Invalid saved export history.");
    const exporter = "exporter" in value ? value.exporter : undefined;
    if (exporter !== undefined && !isExporter(exporter))
      throw new Error("Invalid saved conversion tool.");
    return {
      models: [...value.models],
      directories: [...value.directories],
      ...(exporter === undefined ? {} : { exporter }),
    };
  }

  remember(plan: ExportPlan): Promise<void> {
    const operation = this.pending.then(async () => {
      const history = this.snapshot();
      const recent = (value: string, entries: string[]) =>
        [value, ...entries.filter((entry) => entry !== value)].slice(0, 50);
      const directories = recent(plan.directory, history.directories);
      await this.storage.update("exportHistory", {
        exporter: plan.exporter,
        models: recent(plan.model, history.models),
        directories: plan.directoryTemplate
          ? recent(plan.directoryTemplate, directories)
          : directories,
      });
    });
    this.pending = operation.then(
      () => undefined,
      () => undefined,
    );
    return operation;
  }
}

export function defaultExportDirectory(root: string): string {
  return join(root, "<model_id>", "<exporter>");
}

async function pathExists(path: string): Promise<boolean> {
  try {
    await lstat(path);
    return true;
  } catch (error) {
    if (error instanceof Error && "code" in error && error.code === "ENOENT")
      return false;
    throw error;
  }
}

interface ExportAdvancedOptions {
  excludeEmbeds?: boolean;
  excludeLmHead?: boolean;
  includeHiddenStates?: boolean;
  pruneLmHead?: boolean;
  useQdq?: boolean;
  blockSize?: number;
  quantizationAlgorithm?: string;
  opset?: number;
  maxShardSize?: string;
  noWeights?: boolean;
  optimize?: string;
  externalData?: string;
  staticCache?: boolean;
  maxSequenceLength?: number;
  runtime?: string;
}

function exportAdvancedOptions(
  value: unknown,
  exporter: Exporter,
): ExportAdvancedOptions {
  if (value === undefined) return {};
  if (!value || typeof value !== "object" || Array.isArray(value))
    throw new Error("Invalid advanced export options.");
  const options: ExportAdvancedOptions = {};
  const mobius = exporter === "mobius";
  for (const [key, entry] of Object.entries(value)) {
    switch (key) {
      case "excludeEmbeds":
      case "excludeLmHead":
      case "includeHiddenStates":
      case "pruneLmHead":
      case "useQdq":
        if (mobius || typeof entry !== "boolean")
          throw new Error(`Invalid ${key} option for ${exporter}.`);
        options[key] = entry;
        break;
      case "noWeights":
      case "staticCache":
        if (!mobius || typeof entry !== "boolean")
          throw new Error(`Invalid ${key} option for ${exporter}.`);
        options[key] = entry;
        break;
      case "opset":
      case "maxSequenceLength":
        if (
          (key === "opset" ? exporter !== "mbext" : !mobius) ||
          typeof entry !== "number" ||
          !Number.isSafeInteger(entry) ||
          entry < 1
        )
          throw new Error(`${key} must be a supported positive integer.`);
        options[key] = entry;
        break;
      case "blockSize":
        if (
          mobius ||
          typeof entry !== "number" ||
          ![16, 32, 64, 128, 256].includes(entry)
        )
          throw new Error("Choose a supported quantization block size.");
        options.blockSize = entry;
        break;
      case "quantizationAlgorithm":
        if (
          mobius ||
          typeof entry !== "string" ||
          ![
            "rtn",
            "rtn_last",
            "k_quant",
            "k_quant_mixed",
            "k_quant_last",
          ].includes(entry)
        )
          throw new Error("Choose a supported quantization algorithm.");
        options.quantizationAlgorithm = entry;
        break;
      case "maxShardSize":
        if (
          exporter === "modelbuilder" ||
          typeof entry !== "string" ||
          !/^[1-9]\d*(?:B|KB|MB|GB)$/i.test(entry)
        )
          throw new Error(
            "Shard size must be positive with a B, KB, MB or GB unit.",
          );
        options.maxShardSize = entry.toUpperCase();
        break;
      case "optimize":
        if (
          !mobius ||
          typeof entry !== "string" ||
          ![
            "all",
            "group_query_attention",
            "packed_attention",
            "skip_norm",
          ].includes(entry)
        )
          throw new Error("Choose a supported Mobius optimization.");
        options.optimize = entry;
        break;
      case "externalData":
        if (!mobius || (entry !== "onnx" && entry !== "safetensors"))
          throw new Error("Choose a supported Mobius external data format.");
        options.externalData = entry;
        break;
      case "runtime":
        if (!mobius || (entry !== "ort-genai" && entry !== "onnx-genai"))
          throw new Error("Choose a supported Mobius runtime.");
        options.runtime = entry;
        break;
      default:
        throw new Error(`Unknown advanced export option: ${key}.`);
    }
  }
  if (
    options.excludeLmHead &&
    (options.includeHiddenStates || options.pruneLmHead)
  )
    throw new Error(
      "Hidden-state outputs and head pruning require the language modeling head.",
    );
  if (options.maxSequenceLength !== undefined && !options.staticCache)
    throw new Error("Maximum sequence length requires static cache.");
  return options;
}

export interface ExportPlan {
  exporter: Exporter;
  layerCount?: number;
  advancedOptions?: ExportAdvancedOptions;
  model: string;
  precision: string;
  provider: "cpu" | "cuda";
  executable: string;
  args: string[];
  directory: string;
  directoryTemplate?: string;
  command: string;
  script: string;
}

/** Builds an argument vector, never an executable shell command. */
export function exportPlan(
  value: unknown,
  pythonPath: string,
  downloadDirectory?: string,
): ExportPlan {
  if (
    !value ||
    typeof value !== "object" ||
    !("exporter" in value) ||
    !isExporter(value.exporter) ||
    !("model" in value) ||
    typeof value.model !== "string" ||
    !value.model.trim() ||
    value.model.trim().startsWith("-") ||
    /[\0\r\n]/.test(value.model) ||
    !("directory" in value) ||
    typeof value.directory !== "string" ||
    /[\0\r\n]/.test(value.directory) ||
    !("precision" in value) ||
    typeof value.precision !== "string" ||
    !["fp32", "fp16", "bf16", "int4", "int2", "fp4"].includes(
      value.precision,
    ) ||
    !("provider" in value) ||
    (value.provider !== "cpu" && value.provider !== "cuda")
  )
    throw new Error(
      "Choose an exporter, model, absolute output directory, precision and device.",
    );
  const model = value.model.trim();
  const layerCount = "layerCount" in value ? value.layerCount : undefined;
  if (
    layerCount !== undefined &&
    (value.exporter === "mobius" ||
      typeof layerCount !== "number" ||
      !Number.isSafeInteger(layerCount) ||
      layerCount < 1)
  )
    throw new Error("The builder layer limit must be a positive integer.");
  const advancedOptions = exportAdvancedOptions(
    "advancedOptions" in value ? value.advancedOptions : undefined,
    value.exporter,
  );
  if (
    value.precision !== "int4" &&
    (advancedOptions.useQdq ||
      advancedOptions.blockSize !== undefined ||
      advancedOptions.quantizationAlgorithm !== undefined)
  )
    throw new Error("Quantization options require INT4 precision.");
  let directory = value.directory;
  if (directory.includes("<model_id>")) {
    const parts = isAbsolute(model) ? [basename(model)] : model.split(/[\\/]/);
    if (
      parts.some(
        (part) =>
          !part ||
          part === "." ||
          part === ".." ||
          /[<>:"|?*\x00-\x1f]/.test(part) ||
          /[. ]$/.test(part),
      )
    )
      throw new Error("The model ID cannot be used in a directory template.");
    directory = directory.replaceAll("<model_id>", join(...parts));
  }
  directory = directory.replaceAll("<exporter>", value.exporter);
  if (/[<>]/.test(directory))
    throw new Error(
      "Directory templates support only <model_id> and <exporter>.",
    );
  if (!isAbsolute(directory)) {
    if (!downloadDirectory || !isAbsolute(downloadDirectory))
      throw new Error("Enter an absolute output directory.");
    directory = resolve(downloadDirectory, directory);
  }
  const precision = value.precision;
  const mobius = value.exporter === "mobius";
  const args = mobius
    ? [
        "-u",
        "-m",
        "mobius",
        "build",
        isAbsolute(model) ? "--config" : "--model",
        model,
        "--output",
        directory,
        "--dtype",
        precision === "fp32" ? "f32" : precision === "fp16" ? "f16" : precision,
        "--ep",
        value.provider,
      ]
    : [
        "-u",
        "-m",
        value.exporter === "mbext"
          ? "modelbuilder.builder"
          : "onnxruntime_genai.models.builder",
        isAbsolute(model) ? "-i" : "-m",
        model,
        "-o",
        directory,
        "-p",
        precision,
        "-e",
        value.provider,
      ];
  if (mobius) {
    if (advancedOptions.noWeights) args.push("--no-weights");
    if (advancedOptions.staticCache) args.push("--features", "static-cache");
    for (const [key, flag] of [
      ["optimize", "--optimize"],
      ["externalData", "--external-data"],
      ["maxSequenceLength", "--max-seq-len"],
      ["runtime", "--runtime"],
    ] as const)
      if (advancedOptions[key] !== undefined)
        args.push(flag, String(advancedOptions[key]));
  } else {
    if (advancedOptions.opset !== undefined)
      args.push("--onnx_opset", String(advancedOptions.opset));
    const extra = value.exporter === "mbext" ? ["hf_remote=false"] : [];
    if (layerCount !== undefined) extra.push(`num_hidden_layers=${layerCount}`);
    for (const [key, flag] of [
      ["excludeEmbeds", "exclude_embeds"],
      ["excludeLmHead", "exclude_lm_head"],
      ["includeHiddenStates", "include_hidden_states"],
      ["pruneLmHead", "prune_lm_head"],
      ["useQdq", "use_qdq"],
    ] as const)
      if (advancedOptions[key]) extra.push(`${flag}=true`);
    if (advancedOptions.blockSize !== undefined)
      extra.push(`int4_block_size=${advancedOptions.blockSize}`);
    if (advancedOptions.quantizationAlgorithm !== undefined)
      extra.push(`int4_algo_config=${advancedOptions.quantizationAlgorithm}`);
    if (extra.length) args.push("--extra_options", ...extra);
  }
  if (advancedOptions.maxShardSize !== undefined)
    args.push("--max-shard-size", advancedOptions.maxShardSize);
  const quote = (argument: string) => {
    const safe =
      process.platform === "win32" ? /^[\w./\\:=-]+$/ : /^[\w@%+=:,./-]+$/;
    if (safe.test(argument)) return argument;
    return process.platform === "win32"
      ? `'${argument.replaceAll("'", "''")}'`
      : `'${argument.replaceAll("'", "'\\''")}'`;
  };
  return {
    exporter: value.exporter,
    ...(layerCount === undefined ? {} : { layerCount }),
    ...(Object.keys(advancedOptions).length ? { advancedOptions } : {}),
    model,
    precision,
    provider: value.provider,
    executable: pythonPath,
    args,
    directory,
    ...(directory === value.directory
      ? {}
      : { directoryTemplate: value.directory }),
    command: `${process.platform === "win32" ? "& " : ""}${[pythonPath, ...args].map(quote).join(" ")}`,
    script: join(directory, "export_model.py"),
  };
}

/** Keeps existing output intact, reserving a fresh subfolder for another attempt. */
export async function prepareExportPlan(
  plan: ExportPlan,
  signal: AbortSignal,
): Promise<ExportPlan> {
  signal.throwIfAborted();
  if (plan.exporter === "mobius") {
    const parent = dirname(plan.directory);
    await mkdir(parent, { recursive: true });
    const occupied = await pathExists(plan.directory);
    signal.throwIfAborted();
    const staging = occupied
      ? await mkdtemp(join(plan.directory, `${plan.exporter}-`))
      : await mkdtemp(join(parent, `${plan.exporter}-`));
    signal.throwIfAborted();
    const directory = occupied
      ? join(staging, basename(plan.directory))
      : plan.directory;
    return {
      ...exportPlan({ ...plan, directory }, plan.executable),
      script: join(staging, "export_model.py"),
      ...(plan.directoryTemplate === undefined
        ? {}
        : { directoryTemplate: plan.directoryTemplate }),
    };
  }
  await mkdir(plan.directory, { recursive: true });
  const entries = await readdir(plan.directory);
  signal.throwIfAborted();
  if (!entries.length) return plan;
  const directory = await mkdtemp(join(plan.directory, `${plan.exporter}-`));
  signal.throwIfAborted();
  return {
    ...exportPlan({ ...plan, directory }, plan.executable),
    ...(plan.directoryTemplate === undefined
      ? {}
      : { directoryTemplate: plan.directoryTemplate }),
  };
}

async function terminateExport(child: ChildProcess): Promise<void> {
  const pid = child.pid;
  if (pid === undefined) return;
  if (process.platform === "win32") {
    if (child.exitCode !== null || child.signalCode !== null) return;
    await new Promise<void>((resolve, reject) => {
      execFile("taskkill", ["/PID", String(pid), "/T", "/F"], (error) => {
        // taskkill returns 128 if the target exited before termination.
        if (
          error &&
          !(
            error.code === 128 &&
            (child.exitCode !== null || child.signalCode !== null)
          )
        ) {
          if (child.exitCode !== null || child.signalCode !== null) resolve();
          else if (child.kill()) resolve();
          else reject(error);
        } else resolve();
      });
    });
    return;
  }
  const killGroup = (signal: NodeJS.Signals) => {
    try {
      process.kill(-pid, signal);
    } catch (error) {
      if (!(
        error instanceof Error &&
        "code" in error &&
        error.code === "ESRCH"
      ))
        throw error;
    }
  };
  killGroup("SIGTERM");
  await delay(2000);
  killGroup("SIGKILL");
}

async function runProcess(
  plan: Pick<ExportPlan, "executable" | "args">,
  signal: AbortSignal,
  progress: (output: string) => void,
  environment: NodeJS.ProcessEnv,
  label: string,
): Promise<void> {
  signal.throwIfAborted();
  await new Promise<void>((resolve, reject) => {
    const child = spawn(plan.executable, plan.args, {
      cwd: homedir(),
      env: { ...process.env, ...environment },
      stdio: ["ignore", "pipe", "pipe"],
      detached: process.platform !== "win32",
    });
    let output = "";
    let timer: ReturnType<typeof setTimeout> | undefined;
    let termination: Promise<void> | undefined;
    let finished = false;
    const readableOutput = () =>
      stripVTControlCharacters(output).replace(/\r\n?/g, "\n");
    const flush = () => {
      timer = undefined;
      progress(readableOutput());
    };
    const finish = (error?: Error) => {
      if (finished) return;
      finished = true;
      signal.removeEventListener("abort", abort);
      if (timer) clearTimeout(timer);
      flush();
      const settle = () => {
        if (error) reject(error);
        else resolve();
      };
      if (termination) void termination.then(settle, reject);
      else settle();
    };
    const abort = () => {
      if (termination || child.pid === undefined) return;
      termination = terminateExport(child);
      void termination.catch(finish);
    };
    for (const stream of [child.stdout, child.stderr]) {
      stream.setEncoding("utf8");
      stream.on("data", (chunk: string) => {
        output = (output + chunk).slice(-65536);
        if (!timer) timer = setTimeout(flush, 100);
      });
    }
    child.once("error", finish);
    child.once("close", (code, termination) =>
      finish(
        signal.aborted
          ? new DOMException("Export cancelled.", "AbortError")
          : code === 0
            ? undefined
            : new Error(
                `${label} exited with ${termination ? `signal ${termination}` : `code ${code}`}.\n${readableOutput()}`,
              ),
      ),
    );
    signal.addEventListener("abort", abort, { once: true });
    if (signal.aborted) abort();
  });
  signal.throwIfAborted();
}

/** Caches pinned source metadata and weights, unless graph-only export omits weights. */
export async function downloadExportSource(
  plan: ExportPlan,
  cacheDirectory: string,
  signal: AbortSignal,
  progress: (output: string) => void,
): Promise<ExportPlan> {
  signal.throwIfAborted();
  if (isAbsolute(plan.model)) return plan;
  await mkdir(cacheDirectory, { recursive: true });
  const request = await mkdtemp(join(cacheDirectory, ".request-"));
  try {
    const result = join(request, "source.json");
    await runProcess(
      {
        executable: plan.executable,
        args: [
          "-u",
          "-c",
          downloadSource,
          plan.model,
          cacheDirectory,
          result,
          plan.advancedOptions?.noWeights ? "0" : "1",
        ],
      },
      signal,
      progress,
      exportEnvironment,
      "Downloader",
    );
    const source: unknown = JSON.parse(await readFile(result, "utf8"));
    if (
      typeof source !== "string" ||
      !isAbsolute(source) ||
      !(await lstat(source)).isDirectory()
    )
      throw new Error("The downloader did not return a local model directory.");
    return {
      ...exportPlan({ ...plan, model: source }, plan.executable),
      script: plan.script,
      ...(plan.directoryTemplate === undefined
        ? {}
        : { directoryTemplate: plan.directoryTemplate }),
    };
  } finally {
    await rm(request, { recursive: true, force: true });
  }
}

/** Runs the selected CLI with bounded, throttled output and explicit cancellation. */
export async function runExport(
  plan: ExportPlan,
  signal: AbortSignal,
  progress: (output: string) => void,
): Promise<string[]> {
  signal.throwIfAborted();
  if (plan.exporter === "mobius") {
    await mkdir(dirname(plan.directory), { recursive: true });
    if (await pathExists(plan.directory))
      throw new Error(
        "The Mobius package path already exists. Retry to create a new subfolder.",
      );
  } else {
    await mkdir(plan.directory, { recursive: true });
    if ((await readdir(plan.directory)).length)
      throw new Error(
        "The conversion directory is no longer empty. Retry to create a new subfolder.",
      );
  }
  const environment = {
    ...exportEnvironment,
    ...(isAbsolute(plan.model) ? offlineEnvironment : {}),
  };
  const script = `import os\nimport subprocess\n\nsubprocess.run(${JSON.stringify([plan.executable, ...plan.args])}, cwd=${JSON.stringify(homedir())}, env={**os.environ, **${JSON.stringify(environment)}}, check=True)\n`;
  await writeFile(plan.script, script, { flag: "wx" });
  await runProcess(plan, signal, progress, environment, "Exporter");
  const models: string[] = [];
  const visit = async (directory: string) => {
    for (const entry of await readdir(directory, { withFileTypes: true })) {
      signal.throwIfAborted();
      const path = join(directory, entry.name);
      if (entry.isDirectory()) await visit(path);
      else if (entry.isFile() && /\.(?:onnx|pipe-onnx)$/i.test(entry.name))
        models.push(path);
    }
  };
  await visit(plan.directory);
  if (!models.length)
    throw new Error(
      "The exporter completed but produced no ONNX model or pipeline.",
    );
  return models.sort();
}
