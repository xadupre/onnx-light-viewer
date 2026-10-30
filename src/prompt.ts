import { createHmac, randomBytes } from "node:crypto";
import {
  lstat,
  mkdir,
  mkdtemp,
  readFile,
  readdir,
  rm,
  stat,
  writeFile,
} from "node:fs/promises";
import { basename, dirname, isAbsolute, join, resolve } from "node:path";
import { readProvenance, runProcess } from "./export";
import { loadModelFile } from "./graph";
import { PromptSession } from "./promptSession";

export interface PromptToken {
  text: string;
  id: number;
  /** Raw logit, or null when the engine cannot expose it. */
  score: number | null;
  /** Native generation step latency; the first token includes prefill. */
  timeMs: number | null;
}

export interface PromptDiscrepancy {
  maxAbsolute: number;
  relative: number;
  l2: number;
  below001: number;
  below01: number;
  below1: number;
  total: number;
}

export interface PromptResult {
  prompt: string;
  engine: string;
  tokens: PromptToken[];
  /** HF argmax predictions at each exported-token prefix, not a free-running sequence. */
  reference?: { tokens: PromptToken[]; loadMs?: number; warmupMs?: number };
  lifecycle?: PromptLifecycle;
  profileDirectory?: string;
  discrepancies?: PromptDiscrepancy[];
  limitations?: string[];
  signature: string;
}

export interface PromptLifecycle {
  loadMs: number;
  warmupMs?: number;
  reused: boolean;
  retained: boolean;
}

interface PromptOptions {
  engine: "onnx-light" | "onnxruntime-genai";
  prompt: string;
  maxTokens: number;
  validate: boolean;
  keepLoaded: boolean;
  referenceSource?: string;
  sessionSettings: Record<string, string | number | boolean>;
}

const signingKey = randomBytes(32);
const retainedLogits = new Map<string, Buffer>();
const retentionBudget = 256 * 1024 * 1024;
let retainedBytes = 0;

function retain(signature: string, capture: Buffer): void {
  const existing = retainedLogits.get(signature);
  if (existing) retainedBytes -= existing.length;
  retainedLogits.delete(signature);
  if (capture.length > retentionBudget) return;
  retainedLogits.set(signature, capture);
  retainedBytes += capture.length;
  while (retainedLogits.size > 8 || retainedBytes > retentionBudget) {
    const oldest = retainedLogits.keys().next().value!;
    retainedBytes -= retainedLogits.get(oldest)!.length;
    retainedLogits.delete(oldest);
  }
}

function optionsFrom(value: unknown): PromptOptions {
  if (!value || typeof value !== "object" || Array.isArray(value))
    throw new Error("Invalid prompt options.");
  const options = value as Record<string, unknown>;
  if (
    !["onnx-light", "onnxruntime-genai"].includes(String(options.engine)) ||
    typeof options.prompt !== "string" ||
    !options.prompt.trim() ||
    options.prompt.includes("\0") ||
    options.prompt.length > 1_000_000 ||
    !Number.isSafeInteger(options.maxTokens) ||
    (options.maxTokens as number) < 1 ||
    (options.maxTokens as number) > 4096 ||
    (options.validate !== undefined && typeof options.validate !== "boolean") ||
    (options.keepLoaded !== undefined &&
      typeof options.keepLoaded !== "boolean") ||
    Object.keys(options).some(
      (key) =>
        ![
          "engine",
          "prompt",
          "maxTokens",
          "validate",
          "keepLoaded",
          "referenceSource",
          "sessionSettings",
        ].includes(key),
    )
  )
    throw new Error(
      "Choose a supported engine, prompt and 1–4096 output tokens.",
    );
  if (
    options.referenceSource !== undefined &&
    (typeof options.referenceSource !== "string" ||
      !options.referenceSource.trim() ||
      options.referenceSource.includes("\0") ||
      (!isAbsolute(options.referenceSource) &&
        !/^[\w.-]+(?:\/[\w.-]+)?$/.test(options.referenceSource)))
  )
    throw new Error(
      "Enter a Hugging Face model ID or an absolute local checkpoint directory.",
    );
  return {
    engine: options.engine as PromptOptions["engine"],
    prompt: options.prompt,
    maxTokens: options.maxTokens as number,
    validate: options.validate === true,
    keepLoaded: options.keepLoaded === true,
    ...(typeof options.referenceSource === "string"
      ? { referenceSource: options.referenceSource }
      : {}),
    sessionSettings: sessionSettingsFrom(
      String(options.engine),
      options.sessionSettings,
    ),
  };
}

function sessionSettingsFrom(
  engine: string,
  value: unknown,
): PromptOptions["sessionSettings"] {
  if (value === undefined) return {};
  if (!value || typeof value !== "object" || Array.isArray(value))
    throw new Error("Invalid session settings.");
  const numeric: Record<string, number> =
    engine === "onnx-light"
      ? {
          threads: 4096,
          verbose: 10,
          initialCapacity: 1048576,
          loadMemoryMiB: 1048576,
        }
      : {
          intraOpThreads: 4096,
          interOpThreads: 4096,
          deviceId: 65535,
          gpuMemoryMiB: 1048576,
          logSeverity: 4,
        };
  const boolean =
    engine === "onnx-light"
      ? [
          "checkShapes",
          "releaseIntermediates",
          "profile",
          "loadOverlap",
          "sdkLoader",
        ]
      : [
          "cpuArena",
          "memoryPattern",
          "profile",
          "parallelExecution",
          "prepacking",
        ];
  const enums: Record<string, string[]> =
    engine === "onnx-light"
      ? {}
      : {
          provider: ["cpu", "cuda", "dml"],
          graphOptimization: [
            "ORT_DISABLE_ALL",
            "ORT_ENABLE_BASIC",
            "ORT_ENABLE_EXTENDED",
            "ORT_ENABLE_ALL",
          ],
        };
  const result: PromptOptions["sessionSettings"] = {};
  for (const [key, setting] of Object.entries(value)) {
    if (Object.hasOwn(numeric, key)) {
      if (
        typeof setting !== "number" ||
        !Number.isSafeInteger(setting) ||
        setting < (key === "gpuMemoryMiB" ? 1 : 0) ||
        setting > numeric[key]
      )
        throw new Error(`Invalid session setting: ${key}.`);
    } else if (boolean.includes(key)) {
      if (typeof setting !== "boolean")
        throw new Error(`Invalid session setting: ${key}.`);
    } else if (Object.hasOwn(enums, key)) {
      if (typeof setting !== "string" || !enums[key].includes(setting))
        throw new Error(`Invalid session setting: ${key}.`);
    } else throw new Error(`Unsupported ${engine} session setting: ${key}.`);
    result[key] = setting;
  }
  if (
    "deviceId" in result &&
    !["cuda", "dml"].includes(String(result.provider))
  )
    throw new Error("A GPU device requires an explicit GPU provider.");
  if ("gpuMemoryMiB" in result && result.provider !== "cuda")
    throw new Error("The GPU memory limit requires the CUDA provider.");
  if (
    ("loadOverlap" in result || "loadMemoryMiB" in result) &&
    result.sdkLoader !== true
  )
    throw new Error("Select the SDK loader to configure parallel loading.");
  return result;
}

function checkLifecycle(value: unknown): asserts value is PromptLifecycle {
  if (
    !value ||
    typeof value !== "object" ||
    !("loadMs" in value) ||
    typeof value.loadMs !== "number" ||
    !Number.isFinite(value.loadMs) ||
    value.loadMs < 0 ||
    ("warmupMs" in value &&
      (typeof value.warmupMs !== "number" ||
        !Number.isFinite(value.warmupMs) ||
        value.warmupMs < 0)) ||
    !("reused" in value) ||
    typeof value.reused !== "boolean" ||
    !("retained" in value) ||
    typeof value.retained !== "boolean"
  )
    throw new Error("The native runner returned invalid lifecycle timings.");
}

async function exists(file: string): Promise<boolean> {
  try {
    return (await stat(file)).isFile();
  } catch (error) {
    if (
      error instanceof Error &&
      "code" in error &&
      (error.code === "ENOENT" || error.code === "ENOTDIR")
    )
      return false;
    throw error;
  }
}

async function hasTokenizerPackage(directory: string): Promise<boolean> {
  return (
    (await exists(join(directory, "genai_config.json"))) &&
    ((await exists(join(directory, "tokenizer.json"))) ||
      (await exists(join(directory, "tokenizer_config.json"))))
  );
}

async function lightSource(
  directories: string[],
  genai: boolean,
): Promise<string | undefined> {
  for (const directory of directories) {
    const configPath = join(directory, "config.json");
    if (!(await exists(configPath)) || !(await hasTokenizerPackage(directory)))
      continue;
    const contents = await readFile(configPath, "utf8");
    let config: Record<string, unknown>;
    try {
      config = JSON.parse(contents);
    } catch (error) {
      if (!(error instanceof SyntaxError)) throw error;
      console.warn(
        `Skipping invalid model configuration at ${configPath}: ${error.message}`,
      );
      continue;
    }
    if (!config || typeof config !== "object" || Array.isArray(config)) {
      console.warn(
        `Skipping invalid model configuration at ${configPath}: expected a JSON object.`,
      );
      continue;
    }
    const causal =
      genai ||
      config.is_decoder === true ||
      (Array.isArray(config.architectures) &&
        config.architectures.some(
          (architecture) =>
            typeof architecture === "string" &&
            /(?:CausalLM|LMHeadModel|ForConditionalGeneration)$/.test(
              architecture,
            ),
        ));
    if (causal && config.is_encoder_decoder !== true) return directory;
  }
  return undefined;
}

/** Availability describes model assets; missing native runtime errors surface at run time. */
export async function promptInfo(
  modelPath: string,
  packageDirectory?: string,
): Promise<{
  engines: string[];
  available: boolean;
  reason?: string;
  pipeline?: boolean;
  validationRequiresReference?: boolean;
}> {
  if (!isAbsolute(modelPath) || !(await exists(modelPath)))
    return {
      engines: [],
      available: false,
      reason: "Save a local model first.",
    };
  const engines: string[] = [];
  const genai =
    !/\.pipe-onnx$/i.test(modelPath) &&
    (await exists(join(dirname(modelPath), "genai_config.json")));
  if (genai && (await hasTokenizerPackage(dirname(modelPath))))
    engines.push("onnxruntime-genai");
  const provenance = await readProvenance(modelPath);
  const directories = [
    ...(provenance?.source ? [provenance.source] : []),
    dirname(modelPath),
    ...(packageDirectory ? [packageDirectory] : []),
  ];
  if (await lightSource(directories, genai)) engines.unshift("onnx-light");
  const recordedReference = !!(
    provenance?.source ||
    (provenance?.model && !isAbsolute(provenance.model))
  );
  const localReference =
    !recordedReference &&
    (await exists(join(dirname(modelPath), "config.json"))) &&
    ((await exists(join(dirname(modelPath), "tokenizer.json"))) ||
      (await exists(join(dirname(modelPath), "tokenizer_config.json")))) &&
    (
      await Promise.all(
        [
          "model.safetensors",
          "model.safetensors.index.json",
          "pytorch_model.bin",
          "pytorch_model.bin.index.json",
        ].map((name) => exists(join(dirname(modelPath), name))),
      )
    ).some(Boolean);
  return {
    engines,
    validationRequiresReference: !recordedReference && !localReference,
    ...(/\.pipe-onnx$/i.test(modelPath) ? { pipeline: true } : {}),
    available: engines.length > 0,
    ...(engines.length
      ? {}
      : {
          reason:
            "Prompt is disabled because no compatible local generation package was found. ONNX Runtime GenAI requires genai_config.json and tokenizer assets beside the model; onnx-light requires causal-model config.json, genai_config.json and tokenizer assets together in a local source or export directory. Integrated embeddings are supported. For Mobius, export with --runtime ort-genai to include the GenAI configuration and tokenizer.",
        }),
  };
}

function checkTokens(tokens: unknown): asserts tokens is PromptToken[] {
  if (
    !Array.isArray(tokens) ||
    tokens.length > 4096 ||
    tokens.some(
      (token) =>
        !token ||
        typeof token.text !== "string" ||
        !Number.isSafeInteger(token.id) ||
        token.id < 0 ||
        (token.score !== null && !Number.isFinite(token.score)) ||
        (token.timeMs !== null &&
          (!Number.isFinite(token.timeMs) || token.timeMs < 0)),
    )
  )
    throw new Error("The prompt runner returned invalid tokens or logits.");
}

export async function runPrompt(
  modelPath: string,
  value: unknown,
  pythonPath: string,
  signal: AbortSignal,
  progress: (output: string) => void,
  previous?: PromptResult,
  token?: (token: PromptToken, index: number) => void,
  session?: PromptSession,
  lifecycle?: (value: PromptLifecycle) => void,
): Promise<PromptResult> {
  signal.throwIfAborted();
  const options = optionsFrom(value);
  if (options.keepLoaded && !session)
    throw new Error("Keeping a model loaded requires a panel session owner.");
  if (!options.validate) previous = undefined;
  const info = await promptInfo(modelPath);
  if (
    info.pipeline &&
    (options.sessionSettings.sdkLoader === true ||
      "loadOverlap" in options.sessionSettings ||
      "loadMemoryMiB" in options.sessionSettings)
  )
    throw new Error(
      "The viewer does not yet expose SDK owning-loader options for pipelines.",
    );
  if (!info.engines.includes(options.engine))
    throw new Error(
      info.reason ??
        "This model does not support that engine. Both engines require genai_config.json and tokenizer assets in the tokenizer directory; onnx-light also requires causal-model config.json.",
    );
  const provenance = await readProvenance(modelPath);
  const directories = [
    ...(provenance?.source ? [provenance.source] : []),
    dirname(modelPath),
  ];
  const source =
    options.engine === "onnxruntime-genai"
      ? dirname(modelPath)
      : await lightSource(
          directories,
          !info.pipeline &&
            (await exists(join(dirname(modelPath), "genai_config.json"))),
        );
  if (!source)
    throw new Error(
      "No local tokenizer package with config.json, genai_config.json and tokenizer assets is available for this model.",
    );
  let eosTokenIds: number[] | undefined;
  if (options.engine === "onnx-light") {
    for (const name of ["generation_config.json", "config.json"]) {
      const path = join(source, name);
      if (!(await exists(path))) continue;
      const config = JSON.parse(await readFile(path, "utf8")) as {
        eos_token_id?: unknown;
      };
      if (config.eos_token_id === undefined || config.eos_token_id === null)
        continue;
      const values: unknown[] = Array.isArray(config.eos_token_id)
        ? config.eos_token_id
        : [config.eos_token_id];
      eosTokenIds = values.map((value) => {
        if (
          typeof value !== "number" ||
          !Number.isSafeInteger(value) ||
          value < 0
        )
          throw new Error(
            "eos_token_id must be a non-negative integer or a list of non-negative integers.",
          );
        return value;
      });
      break;
    }
  }
  const identity = await lstat(modelPath);
  const saved = await loadModelFile(modelPath, basename(modelPath));
  signal.throwIfAborted();
  const paths = new Set([
    ...(await readdir(dirname(modelPath), { withFileTypes: true }))
      .filter(
        (entry) =>
          (entry.isFile() || entry.isSymbolicLink()) &&
          !entry.name.startsWith("."),
      )
      .map((entry) => join(dirname(modelPath), entry.name)),
    ...saved
      .externalDataLocations()
      .map((location) => resolve(dirname(modelPath), location)),
  ]);
  if (source !== dirname(modelPath))
    for (const entry of await readdir(source, { withFileTypes: true }))
      if (
        (entry.isFile() || entry.isSymbolicLink()) &&
        !entry.name.startsWith(".") &&
        (/\.(?:json|txt|model|tiktoken)$/.test(entry.name) ||
          entry.name.startsWith("tokenizer.model."))
      )
        paths.add(join(source, entry.name));
  const assets = await Promise.all(
    [...paths].sort().map(async (path) => {
      const details = await stat(path);
      return [path, details.size, details.mtimeMs, details.ctimeMs];
    }),
  );
  const sessionContext = {
    modelPath,
    size: identity.size,
    modified: identity.mtimeMs,
    changed: identity.ctimeMs,
    assets,
    provenance,
    source,
    eosTokenIds,
    engine: options.engine,
    sessionSettings: options.sessionSettings,
  };
  const context = JSON.stringify({
    ...sessionContext,
    prompt: options.prompt,
    maxTokens: options.maxTokens,
  });
  const signature = (tokens: PromptToken[], capture: Buffer) =>
    createHmac("sha256", signingKey)
      .update(context)
      .update(JSON.stringify(tokens))
      .update(capture)
      .digest("hex");
  const previousCapture = previous
    ? retainedLogits.get(previous.signature)
    : undefined;
  if (previous !== undefined) {
    checkTokens(previous.tokens);
    if (
      !options.validate ||
      previous.prompt !== options.prompt ||
      previous.engine !== options.engine ||
      previous.tokens.length > options.maxTokens ||
      !previousCapture ||
      previous.signature !== signature(previous.tokens, previousCapture)
    )
      throw new Error(
        "The previous prompt result does not match this model and options, or its vocabulary logits are no longer retained. Run with validation enabled to compare a fresh result.",
      );
  }
  // Keep requests separate, beside the saved model, never in a shared global temp directory.
  const request = await mkdtemp(join(dirname(modelPath), ".prompt-"));
  try {
    const input = join(request, "input.json");
    const output = join(request, "output.json");
    const capture = join(request, "logits.json");
    const profileDirectory = join(dirname(modelPath), ".prompt-profiles");
    const referenceEnvironment = {
      PYTHONUNBUFFERED: "1",
      CUDA_VISIBLE_DEVICES: "",
      HF_HUB_OFFLINE: "1",
      TRANSFORMERS_OFFLINE: "1",
    };
    if (options.sessionSettings.profile)
      await mkdir(profileDirectory, { recursive: true });
    if (previousCapture)
      await writeFile(capture, previousCapture, {
        mode: 0o600,
        flag: "wx",
      });
    if (previous)
      await writeFile(
        output,
        JSON.stringify({
          prompt: previous.prompt,
          engine: previous.engine,
          tokens: previous.tokens,
          lifecycle: previous.lifecycle,
          limitations: previous.limitations,
          profileDirectory: previous.profileDirectory,
        }),
        { mode: 0o600, flag: "wx" },
      );
    await writeFile(
      input,
      JSON.stringify({
        modelPath,
        pythonPath,
        ...options,
        provenance,
        source,
        eosTokenIds,
        capturePath: capture,
        captureBudget: retentionBudget,
        profileDirectory,
        ...(previous
          ? {
              tokenIds: previous.tokens.map((token) => token.id),
              reuseCapture: true,
            }
          : {}),
      }),
      { mode: 0o600, flag: "wx" },
    );
    if (options.validate)
      await runProcess(
        {
          executable: pythonPath,
          args: [
            "-u",
            resolve(__dirname, "..", "media", "run_prompt.py"),
            "--check-reference",
            input,
          ],
        },
        signal,
        progress,
        referenceEnvironment,
        "Hugging Face reference preflight",
      );
    if (!previousCapture) {
      let next = 0;
      const worker = session ?? new PromptSession();
      await worker.run(
        input,
        output,
        createHmac("sha256", signingKey)
          .update(JSON.stringify(sessionContext))
          .digest("hex"),
        signal,
        progress,
        (value) => {
          signal.throwIfAborted();
          if (!value || typeof value !== "object" || !("type" in value))
            throw new Error("The native runner returned an invalid event.");
          if (value.type === "lifecycle" && "lifecycle" in value) {
            checkLifecycle(value.lifecycle);
            lifecycle?.(value.lifecycle);
            return;
          }
          const event = value as {
            type: string;
            token: PromptToken;
            index: number;
          };
          checkTokens([event.token]);
          if (
            event.type !== "token" ||
            event.index !== next ||
            next >= options.maxTokens
          )
            throw new Error(
              "The native runner returned an invalid token stream.",
            );
          token?.(event.token, next);
          next++;
        },
      );
      if (!session || !options.keepLoaded) await worker.close();
    }
    if (options.validate) {
      await runProcess(
        {
          executable: pythonPath,
          args: [
            "-u",
            resolve(__dirname, "..", "media", "run_prompt.py"),
            input,
            output,
          ],
        },
        signal,
        progress,
        referenceEnvironment,
        "Hugging Face validation",
      );
    }
    signal.throwIfAborted();
    const result = JSON.parse(await readFile(output, "utf8")) as PromptResult;
    if (result.prompt !== options.prompt || result.engine !== options.engine)
      throw new Error("The prompt runner returned a mismatched result.");
    checkTokens(result.tokens);
    if (!previousCapture) {
      checkLifecycle(result.lifecycle);
      if (result.lifecycle.warmupMs === undefined)
        throw new Error("The native runner did not complete warmup.");
    }
    if (
      result.limitations !== undefined &&
      (!Array.isArray(result.limitations) ||
        !result.limitations.every(
          (limitation) => typeof limitation === "string" && limitation.trim(),
        ))
    )
      throw new Error("The prompt runner returned invalid limitations.");
    if (result.tokens.length > options.maxTokens)
      throw new Error("The prompt runner exceeded the output token limit.");
    if (options.validate) {
      checkTokens(result.reference?.tokens);
      if (
        result.reference.tokens.length !== result.tokens.length ||
        (result.discrepancies === undefined
          ? !result.limitations?.length
          : !Array.isArray(result.discrepancies) ||
            result.discrepancies.length !== result.tokens.length ||
            result.discrepancies.some((metric) =>
              [
                "maxAbsolute",
                "relative",
                "l2",
                "below001",
                "below01",
                "below1",
                "total",
              ].some(
                (key) =>
                  !Number.isFinite(metric[key as keyof PromptDiscrepancy]) ||
                  metric[key as keyof PromptDiscrepancy] < 0,
              ),
            ))
      )
        throw new Error(
          "The prompt runner returned invalid reference comparisons.",
        );
    }
    if ((await lstat(capture)).size > retentionBudget)
      throw new Error(
        "The prompt logits capture exceeds the retention budget.",
      );
    const captured = previousCapture ?? (await readFile(capture));
    const signed = signature(result.tokens, captured);
    retain(signed, captured);
    return { ...result, signature: signed };
  } finally {
    await rm(request, { recursive: true, force: true });
  }
}
