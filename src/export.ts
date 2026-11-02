import { execFile, spawn, type ChildProcess } from "node:child_process";
import {
  cp,
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
import type { PinnedOliveRecipe } from "./oliveRecipes";

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

model, cache, result, include_weights, include_code, *revision = sys.argv[1:]
cache = cache or None
info = HfApi().model_info(model, **({"revision": revision[0]} if revision and revision[0] else {}))
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
if include_code == "1":
    metadata.update(name for name in files if name.endswith(".py"))
print(f"Downloading {model}@{info.sha} into {cache or 'the Hugging Face cache'}", flush=True)
source = snapshot_download(model, revision=info.sha, cache_dir=cache,
                           allow_patterns=sorted(metadata | weights))
with open(result, "w", encoding="utf-8") as stream:
    json.dump(str(pathlib.Path(source).resolve()), stream)
print(f"Source ready: {source}", flush=True)
`;

type Exporter = "mbext" | "modelbuilder" | "mobius" | "olive" | "olive-recipes";
type OliveWorkflow = "optimize" | "quantize";

function isExporter(value: unknown): value is Exporter {
  return (
    value === "mbext" ||
    value === "modelbuilder" ||
    value === "mobius" ||
    value === "olive" ||
    value === "olive-recipes"
  );
}

interface ExportHistorySnapshot {
  models: string[];
  directories: string[];
  exporter?: Exporter;
}

/** Keeps recent export inputs across sessions and serializes edits across panels. */
export class ExportHistory {
  private pending: Promise<void> = Promise.resolve();
  private listeners = new Set<(history: ExportHistorySnapshot) => void>();

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
    return this.update(() => {
      const history = this.snapshot();
      const recent = (value: string, entries: string[]) =>
        [value, ...entries.filter((entry) => entry !== value)].slice(0, 50);
      const directories = recent(plan.directory, history.directories);
      return {
        exporter: plan.exporter,
        models: recent(plan.model, history.models),
        directories: plan.directoryTemplate
          ? recent(plan.directoryTemplate, directories)
          : directories,
      };
    });
  }

  subscribe(listener: (history: ExportHistorySnapshot) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  remove(target: "models" | "directories", value: string): Promise<void> {
    return this.update(() => {
      const history = this.snapshot();
      return {
        ...history,
        [target]: history[target].filter((entry) => entry !== value),
      };
    });
  }

  clear(): Promise<void> {
    return this.update(() => ({ models: [], directories: [] }));
  }

  prune(
    keep: (entry: string, target: "models" | "directories") => Promise<boolean>,
  ): Promise<void> {
    return this.update(async () => {
      const history = this.snapshot();
      const changes = await Promise.all(
        (["models", "directories"] as const).map(async (target) => {
          const previous = history[target];
          const retained = await Promise.all(
            previous.map((entry) => keep(entry, target)),
          );
          history[target] = previous.filter((_, index) => retained[index]);
          return history[target].length !== previous.length;
        }),
      );
      return changes.some(Boolean) ? history : undefined;
    });
  }

  private update(
    change: () =>
      ExportHistorySnapshot | Promise<ExportHistorySnapshot | undefined>,
  ): Promise<void> {
    const operation = this.pending.then(async () => {
      const history = await change();
      if (history === undefined) return;
      await this.storage.update("exportHistory", history);
      for (const listener of this.listeners) listener(history);
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
  oliveExporter?: string;
  task?: string;
  trustRemoteCode?: boolean;
  numSplits?: number;
  memory?: number;
}

function exportAdvancedOptions(
  value: unknown,
  exporter: Exporter,
  oliveWorkflow?: OliveWorkflow,
): ExportAdvancedOptions {
  if (value === undefined) return {};
  if (!value || typeof value !== "object" || Array.isArray(value))
    throw new Error("Invalid advanced export options.");
  if (exporter === "olive-recipes" && Object.keys(value).length)
    throw new Error(
      "Advanced options are not supported for Olive recipes. Clear advanced options before converting.",
    );
  const options: ExportAdvancedOptions = {};
  const mobius = exporter === "mobius";
  const builder = exporter === "mbext" || exporter === "modelbuilder";
  const oliveOptimize = exporter === "olive" && oliveWorkflow === "optimize";
  for (const [key, entry] of Object.entries(value)) {
    switch (key) {
      case "excludeEmbeds":
      case "excludeLmHead":
      case "includeHiddenStates":
      case "pruneLmHead":
        if (!builder || typeof entry !== "boolean")
          throw new Error(`Invalid ${key} option for ${exporter}.`);
        options[key] = entry;
        break;
      case "useQdq":
        if ((!builder && exporter !== "olive") || typeof entry !== "boolean")
          throw new Error(`Invalid ${key} option for ${exporter}.`);
        options.useQdq = entry;
        break;
      case "trustRemoteCode":
        if (!oliveOptimize || typeof entry !== "boolean")
          throw new Error(`Invalid ${key} option for ${exporter}.`);
        options.trustRemoteCode = entry;
        break;
      case "oliveExporter":
        if (
          !oliveOptimize ||
          typeof entry !== "string" ||
          ![
            "model_builder",
            "dynamo_exporter",
            "torchscript_exporter",
          ].includes(entry)
        )
          throw new Error("Choose a supported Olive ONNX exporter.");
        options.oliveExporter = entry;
        break;
      case "task":
        if (
          !oliveOptimize ||
          typeof entry !== "string" ||
          !/^[a-zA-Z0-9][a-zA-Z0-9_-]*$/.test(entry)
        )
          throw new Error("Enter a Hugging Face task name for Olive.");
        options.task = entry;
        break;
      case "numSplits":
      case "memory":
        if (
          !oliveOptimize ||
          typeof entry !== "number" ||
          !Number.isSafeInteger(entry) ||
          entry < 1
        )
          throw new Error(`${key} must be a positive integer for Olive.`);
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
          (!builder && !oliveOptimize) ||
          typeof entry !== "number" ||
          ![16, 32, 64, 128, 256].includes(entry)
        )
          throw new Error("Choose a supported quantization block size.");
        options.blockSize = entry;
        break;
      case "quantizationAlgorithm":
        if (
          !builder ||
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
          (exporter !== "mbext" && !mobius) ||
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
        if (!mobius || entry !== "ort-genai")
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
  oliveWorkflow?: OliveWorkflow;
  recipe?: PinnedOliveRecipe;
  cwd?: string;
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
  provenance?: ExportProvenance;
}

export interface ExportProvenance {
  version: 1;
  model: string;
  source?: string;
  revision?: string;
  exporter: string;
  packageDirectory?: string;
}

export const provenancePath = (model: string): string =>
  `${model}.provenance.json`;

export async function readProvenance(
  model: string,
): Promise<ExportProvenance | undefined> {
  try {
    const value: unknown = JSON.parse(
      await readFile(provenancePath(model), "utf8"),
    );
    if (
      !value ||
      typeof value !== "object" ||
      !("version" in value) ||
      value.version !== 1 ||
      !("model" in value) ||
      typeof value.model !== "string" ||
      !("exporter" in value) ||
      typeof value.exporter !== "string" ||
      ("source" in value &&
        (typeof value.source !== "string" || !isAbsolute(value.source))) ||
      ("revision" in value && typeof value.revision !== "string") ||
      ("packageDirectory" in value &&
        (typeof value.packageDirectory !== "string" ||
          !isAbsolute(value.packageDirectory)))
    )
      throw new Error("Invalid model provenance.");
    return value as ExportProvenance;
  } catch (error) {
    if (error instanceof Error && "code" in error && error.code === "ENOENT")
      return undefined;
    throw error;
  }
}

/** Builds an argument vector, never an executable shell command. */
export function exportPlan(
  value: unknown,
  pythonPath: string,
  downloadDirectory?: string,
  recipe?: PinnedOliveRecipe,
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
    ![
      "fp32",
      "fp16",
      "bf16",
      "int4",
      "int2",
      "fp4",
      ...(value.exporter === "olive" ? ["int8"] : []),
    ].includes(value.precision) ||
    !("provider" in value) ||
    (value.provider !== "cpu" && value.provider !== "cuda")
  )
    throw new Error(
      "Choose an exporter, model, absolute output directory, precision and device.",
    );
  const model = value.model.trim();
  const oliveWorkflow =
    "oliveWorkflow" in value ? value.oliveWorkflow : "optimize";
  if (oliveWorkflow !== "optimize" && oliveWorkflow !== "quantize")
    throw new Error(
      "Choose an Olive export/optimization or quantization workflow.",
    );
  if (
    value.exporter === "olive" &&
    oliveWorkflow === "optimize" &&
    !["fp32", "fp16", "bf16", "int4", "int8"].includes(value.precision)
  )
    throw new Error("Olive export supports FP32, FP16, BF16, INT4 or INT8.");
  if (
    value.exporter === "olive" &&
    oliveWorkflow === "quantize" &&
    (!isAbsolute(model) ||
      !/\.onnx$/i.test(model) ||
      !["int4", "int8"].includes(value.precision) ||
      value.provider !== "cpu")
  )
    throw new Error(
      "Olive quantization requires an absolute ONNX file, INT4 or INT8, and CPU.",
    );
  if (value.exporter === "olive-recipes" && !recipe)
    throw new Error("Choose a pinned Olive recipe.");
  const layerCount = "layerCount" in value ? value.layerCount : undefined;
  if (
    layerCount !== undefined &&
    (!["mbext", "modelbuilder"].includes(value.exporter) ||
      typeof layerCount !== "number" ||
      !Number.isSafeInteger(layerCount) ||
      layerCount < 1)
  )
    throw new Error("The builder layer limit must be a positive integer.");
  const advancedOptions = exportAdvancedOptions(
    "advancedOptions" in value ? value.advancedOptions : undefined,
    value.exporter,
    value.exporter === "olive" ? oliveWorkflow : undefined,
  );
  if (
    value.precision !== "int4" &&
    ((advancedOptions.useQdq &&
      !(value.exporter === "olive" && value.precision === "int8")) ||
      advancedOptions.blockSize !== undefined ||
      advancedOptions.quantizationAlgorithm !== undefined)
  )
    throw new Error(
      value.exporter === "olive"
        ? "Olive Q/DQ requires INT4 or INT8; block size requires INT4."
        : "Quantization options require INT4 precision.",
    );
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
  const recipeRoot = join(directory, "olive-recipe");
  const cwd = recipe ? join(recipeRoot, dirname(recipe.id)) : undefined;
  let args: string[];
  if (value.exporter === "olive" && oliveWorkflow === "quantize")
    args = [
      "-u",
      "-m",
      "olive",
      "quantize",
      "-m",
      model,
      "--output_path",
      directory,
      "--precision",
      precision,
      "--algorithm",
      "rtn",
      "--implementation",
      precision === "int4" ? "olive" : "ort",
    ];
  else if (value.exporter === "olive")
    args = [
      "-u",
      "-m",
      "olive",
      "optimize",
      "-m",
      model,
      "--output_path",
      directory,
      "--precision",
      precision,
      "--device",
      value.provider === "cuda" ? "gpu" : "cpu",
      "--provider",
      value.provider === "cuda"
        ? "CUDAExecutionProvider"
        : "CPUExecutionProvider",
    ];
  else if (value.exporter === "olive-recipes" && recipe)
    args = [
      "-u",
      "-m",
      "olive",
      "run",
      "--run-config",
      join(recipeRoot, recipe.id),
      "--output_path",
      directory,
    ];
  else
    args = mobius
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
          precision === "fp32"
            ? "f32"
            : precision === "fp16"
              ? "f16"
              : precision,
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
  if (value.exporter === "olive") {
    if (advancedOptions.useQdq)
      args.push(
        oliveWorkflow === "quantize"
          ? "--use_qdq_encoding"
          : "--use_qdq_format",
      );
    if (advancedOptions.trustRemoteCode) args.push("--trust_remote_code");
    for (const [key, flag] of [
      ["oliveExporter", "--exporter"],
      ["task", "--task"],
      ["blockSize", "--block_size"],
      ["numSplits", "--num_split"],
      ["memory", "--memory"],
    ] as const)
      if (advancedOptions[key] !== undefined)
        args.push(flag, String(advancedOptions[key]));
  } else if (mobius) {
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
  } else if (value.exporter === "mbext" || value.exporter === "modelbuilder") {
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
    const quantizationPrefix = value.exporter === "mbext" ? "int4_" : "";
    if (advancedOptions.blockSize !== undefined)
      extra.push(
        `${quantizationPrefix}block_size=${advancedOptions.blockSize}`,
      );
    if (advancedOptions.quantizationAlgorithm !== undefined)
      extra.push(
        `${quantizationPrefix}algo_config=${advancedOptions.quantizationAlgorithm}`,
      );
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
    ...(value.exporter === "olive" ? { oliveWorkflow } : {}),
    ...(recipe ? { recipe, cwd } : {}),
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

async function reserveExportDirectory(
  parent: string,
  exporter: Exporter,
  signal: AbortSignal,
  excludedDirectory?: string,
): Promise<string> {
  const now = new Date();
  const date = [
    now.getFullYear(),
    String(now.getMonth() + 1).padStart(2, "0"),
    String(now.getDate()).padStart(2, "0"),
  ].join("-");
  const prefix = `${exporter}-${date}`;
  for (let attempt = 1; ; attempt++) {
    signal.throwIfAborted();
    const directory = join(
      parent,
      attempt === 1 ? prefix : `${prefix}-${attempt}`,
    );
    if (
      excludedDirectory !== undefined &&
      resolve(directory).toLowerCase() ===
        resolve(excludedDirectory).toLowerCase()
    )
      continue;
    try {
      await mkdir(directory);
      return directory;
    } catch (error) {
      if (!(
        error instanceof Error &&
        "code" in error &&
        error.code === "EEXIST"
      ))
        throw error;
    }
  }
}

/** Keeps existing output intact, reserving a dated subfolder for another attempt. */
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
    const staging = await reserveExportDirectory(
      occupied ? plan.directory : parent,
      plan.exporter,
      signal,
      plan.directory,
    );
    signal.throwIfAborted();
    const directory = occupied
      ? join(staging, basename(plan.directory))
      : plan.directory;
    return {
      ...exportPlan({ ...plan, directory }, plan.executable),
      ...(plan.provenance ? { provenance: plan.provenance } : {}),
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
  const directory = await reserveExportDirectory(
    plan.directory,
    plan.exporter,
    signal,
  );
  signal.throwIfAborted();
  return {
    ...exportPlan(
      { ...plan, directory },
      plan.executable,
      undefined,
      plan.recipe,
    ),
    ...(plan.provenance ? { provenance: plan.provenance } : {}),
    ...(plan.directoryTemplate === undefined
      ? {}
      : { directoryTemplate: plan.directoryTemplate }),
  };
}

export async function terminateExport(child: ChildProcess): Promise<void> {
  const pid = child.pid;
  if (pid === undefined) return;
  if (process.platform === "win32") {
    if (child.exitCode !== null || child.signalCode !== null) return;
    await new Promise<void>((resolve, reject) => {
      execFile("taskkill", ["/PID", String(pid), "/T", "/F"], (error) => {
        if (
          !error ||
          child.exitCode !== null ||
          child.signalCode !== null ||
          child.kill()
        ) {
          resolve();
          return;
        }
        // Windows process exit can precede the ChildProcess exit notification.
        try {
          process.kill(pid, 0);
        } catch (probeError) {
          if (
            probeError instanceof Error &&
            "code" in probeError &&
            probeError.code === "ESRCH"
          ) {
            resolve();
            return;
          }
        }
        reject(error);
      });
    });
    return;
  }
  const liveWorkers = async () => {
    const processes = await new Promise<string>((resolve, reject) => {
      execFile(
        "ps",
        ["-A", "-o", "pid=,pgid=,stat="],
        { timeout: 1000 },
        (error, stdout) => {
          if (error) reject(error);
          else resolve(stdout);
        },
      );
    });
    return processes.split("\n").filter((line) => {
      const [, group, state] = line.trim().split(/\s+/);
      return Number(group) === pid && !state?.startsWith("Z");
    });
  };
  const killGroup = async (signal: NodeJS.Signals | 0) => {
    try {
      process.kill(-pid, signal);
      return true;
    } catch (error) {
      if (error instanceof Error && "code" in error) {
        if (error.code === "ESRCH") return false;
        // Accept macOS EPERM only after verifying that no live workers remain.
        if (
          error.code === "EPERM" &&
          process.platform === "darwin" &&
          !(await liveWorkers()).length
        )
          return false;
      }
      throw error;
    }
  };
  if (!(await killGroup("SIGTERM"))) return;
  await delay(2000);
  if (!(await killGroup("SIGKILL"))) return;
  const deadline = Date.now() + 5000;
  while (await killGroup(0)) {
    const workers = await liveWorkers();
    // Orphan zombies can remain in the group, but can no longer execute or write.
    if (!workers.length) return;
    if (Date.now() >= deadline)
      throw new Error(`Exporter workers did not stop:\n${workers.join("\n")}`);
    await delay(25);
  }
}

export async function runProcess(
  plan: Pick<ExportPlan, "executable" | "args" | "cwd">,
  signal: AbortSignal,
  progress: (output: string) => void,
  environment: NodeJS.ProcessEnv,
  label: string,
  stdout?: (chunk: string) => void,
  stderrEncoding: BufferEncoding = "utf8",
): Promise<void> {
  signal.throwIfAborted();
  await new Promise<void>((resolve, reject) => {
    const child = spawn(plan.executable, plan.args, {
      cwd: plan.cwd ?? homedir(),
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
      stream.setEncoding(stream === child.stderr ? stderrEncoding : "utf8");
      stream.on("data", (chunk: string) => {
        if (finished) return;
        if (stream === child.stdout && stdout) {
          try {
            stdout(chunk);
          } catch (error) {
            abort();
            finish(error instanceof Error ? error : new Error(String(error)));
            return;
          }
        }
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
  if (!exportNeedsDownload(plan)) return plan;
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
          plan.advancedOptions?.trustRemoteCode ? "1" : "0",
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
      provenance: {
        version: 1,
        model: plan.provenance?.model ?? plan.model,
        source,
        ...(basename(dirname(source)) === "snapshots"
          ? { revision: basename(source) }
          : {}),
        exporter: plan.exporter,
      },
      script: plan.script,
      ...(plan.directoryTemplate === undefined
        ? {}
        : { directoryTemplate: plan.directoryTemplate }),
    };
  } finally {
    await rm(request, { recursive: true, force: true });
  }
}

export function exportNeedsDownload(plan: ExportPlan): boolean {
  return plan.exporter !== "olive-recipes" && !isAbsolute(plan.model);
}

async function pinRecipeSource(
  plan: ExportPlan,
  signal: AbortSignal,
  progress: (output: string) => void,
): Promise<ExportProvenance | undefined> {
  if (!plan.recipe) return undefined;
  const recipe = join(plan.directory, "olive-recipe", plan.recipe.id);
  const contents = await readFile(recipe, "utf8");
  const value: unknown = /\.ya?ml$/i.test(recipe)
    ? (await import("yaml")).parse(contents)
    : JSON.parse(contents);
  const record = (value: unknown): value is Record<string, unknown> =>
    !!value && typeof value === "object" && !Array.isArray(value);
  const unknown: ExportProvenance = {
    version: 1,
    model: `olive-recipe:${plan.recipe.id}@${plan.recipe.revision}`,
    exporter: plan.exporter,
  };
  if (!record(value) || !record(value.input_model)) return unknown;
  const input = record(value.input_model.config)
    ? value.input_model.config
    : value.input_model;
  const kwargs = record(input.load_kwargs) ? input.load_kwargs : {};
  const extra = record(kwargs.extra_args) ? kwargs.extra_args : {};
  const hfModel =
    typeof value.input_model.type === "string" &&
    /hf|huggingface/i.test(value.input_model.type);
  if (typeof input.model_path !== "string" || !input.model_path.trim())
    return unknown;
  const original = input.model_path;
  let source = resolve(dirname(recipe), original);
  let model = original;
  let revision: string | undefined;
  const requestedRevision =
    typeof kwargs.revision === "string"
      ? kwargs.revision
      : typeof extra.revision === "string"
        ? extra.revision
        : typeof input.revision === "string"
          ? input.revision
          : "";
  if (!(await pathExists(source))) {
    if (!hfModel || !/^[\w.-]+(?:\/[\w.-]+)?$/.test(original)) return unknown;
    const request = await mkdtemp(join(plan.directory, ".recipe-source-"));
    try {
      const result = join(request, "source.json");
      await runProcess(
        {
          executable: plan.executable,
          args: [
            "-u",
            "-c",
            downloadSource,
            original,
            process.env.HF_HUB_CACHE ?? "",
            result,
            "1",
            kwargs.trust_remote_code === true ||
            extra.trust_remote_code === true ||
            input.trust_remote_code === true
              ? "1"
              : "0",
            requestedRevision,
          ],
        },
        signal,
        progress,
        exportEnvironment,
        "Recipe source downloader",
      );
      const downloaded: unknown = JSON.parse(await readFile(result, "utf8"));
      if (
        typeof downloaded !== "string" ||
        !isAbsolute(downloaded) ||
        !(await lstat(downloaded)).isDirectory()
      )
        throw new Error(
          "The recipe downloader did not return a local checkpoint.",
        );
      source = downloaded;
    } finally {
      await rm(request, { recursive: true, force: true });
    }
  }
  if ((await lstat(source)).isDirectory()) {
    const configPath = join(source, "config.json");
    if (await pathExists(configPath)) {
      const config: unknown = JSON.parse(await readFile(configPath, "utf8"));
      if (record(config)) {
        if (
          isAbsolute(original) &&
          typeof config._name_or_path === "string" &&
          /^[\w.-]+(?:\/[\w.-]+)?$/.test(config._name_or_path)
        )
          model = config._name_or_path;
        if (
          typeof config._commit_hash === "string" &&
          /^[a-f0-9]{40}$/i.test(config._commit_hash)
        )
          revision = config._commit_hash;
      }
    }
    if (
      basename(dirname(source)) === "snapshots" &&
      /^[a-f0-9]{40}$/i.test(basename(source))
    )
      revision = basename(source);
  } else {
    const inherited = await readProvenance(source);
    if (inherited) {
      input.model_path = source;
      await writeFile(recipe, `${JSON.stringify(value, null, 2)}\n`);
      return { ...inherited, exporter: plan.exporter };
    }
  }
  input.model_path = source;
  if (record(input.load_kwargs))
    input.load_kwargs = {
      ...kwargs,
      ...(revision && "revision" in kwargs ? { revision } : {}),
      ...(record(kwargs.extra_args)
        ? {
            extra_args: {
              ...extra,
              ...(revision && "revision" in extra ? { revision } : {}),
            },
          }
        : {}),
    };
  if (revision && "revision" in input) input.revision = revision;
  await writeFile(recipe, `${JSON.stringify(value, null, 2)}\n`);
  return {
    version: 1,
    model,
    source,
    ...(revision ? { revision } : {}),
    exporter: plan.exporter,
  };
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
    ...(isAbsolute(plan.model) &&
    !plan.recipe &&
    (plan.exporter !== "olive" || plan.oliveWorkflow === "quantize")
      ? offlineEnvironment
      : {}),
  };
  if (plan.recipe)
    await cp(plan.recipe.snapshot, join(plan.directory, "olive-recipe"), {
      recursive: true,
      filter: async (source) => {
        signal.throwIfAborted();
        if (basename(source) === ".git") return false;
        if ((await lstat(source)).isSymbolicLink())
          throw new Error(
            `Olive recipe snapshots cannot contain symbolic links: ${source}`,
          );
        return true;
      },
    });
  const recipeProvenance = await pinRecipeSource(plan, signal, progress);
  signal.throwIfAborted();
  const script = `${plan.recipe ? `# Olive recipes commit: ${plan.recipe.revision}\n` : ""}import os\nimport subprocess\n\nsubprocess.run(${JSON.stringify([plan.executable, ...plan.args])}, cwd=${JSON.stringify(plan.cwd ?? homedir())}, env={**os.environ, **${JSON.stringify(environment)}}, check=True)\n`;
  await writeFile(plan.script, script, { flag: "wx" });
  await runProcess(plan, signal, progress, environment, "Exporter");
  const models: string[] = [];
  const visit = async (directory: string) => {
    for (const entry of await readdir(directory, { withFileTypes: true })) {
      signal.throwIfAborted();
      const path = join(directory, entry.name);
      if (plan.recipe && path === join(plan.directory, "olive-recipe"))
        continue;
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
  const inherited =
    plan.provenance ??
    recipeProvenance ??
    (plan.oliveWorkflow === "quantize"
      ? await readProvenance(plan.model)
      : undefined);
  const provenance: ExportProvenance = {
    version: 1,
    model: inherited?.model ?? plan.model,
    ...(inherited?.source
      ? { source: inherited.source }
      : isAbsolute(plan.model)
        ? { source: plan.model }
        : {}),
    ...(inherited?.revision
      ? { revision: inherited.revision }
      : isAbsolute(plan.model) && basename(dirname(plan.model)) === "snapshots"
        ? { revision: basename(plan.model) }
        : {}),
    exporter: plan.exporter,
    ...(plan.exporter === "mobius" ? { packageDirectory: plan.directory } : {}),
  };
  for (const model of models) {
    signal.throwIfAborted();
    await writeFile(
      provenancePath(model),
      `${JSON.stringify(provenance, null, 2)}\n`,
      { flag: "wx" },
    );
  }
  return models.sort();
}
