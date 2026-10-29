import { createHmac, randomBytes } from "node:crypto";
import {
  lstat,
  mkdtemp,
  readFile,
  readdir,
  rm,
  writeFile,
} from "node:fs/promises";
import { dirname, isAbsolute, join, resolve } from "node:path";
import { readProvenance, runProcess } from "./export";

export interface PromptToken {
  text: string;
  id: number;
  /** Raw logit, or null when the engine cannot expose it. */
  score: number | null;
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
  reference?: { tokens: PromptToken[] };
  discrepancies?: PromptDiscrepancy[];
  limitations?: string[];
  signature: string;
}

interface PromptOptions {
  engine: "onnx-light" | "onnxruntime-genai";
  prompt: string;
  maxTokens: number;
  validate: boolean;
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
    Object.keys(options).some(
      (key) => !["engine", "prompt", "maxTokens", "validate"].includes(key),
    )
  )
    throw new Error(
      "Choose a supported engine, prompt and 1–4096 output tokens.",
    );
  return {
    engine: options.engine as PromptOptions["engine"],
    prompt: options.prompt,
    maxTokens: options.maxTokens as number,
    validate: options.validate === true,
  };
}

async function exists(file: string): Promise<boolean> {
  try {
    return (await lstat(file)).isFile();
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

/** Availability describes model assets; Python dependency errors remain explicit at run time. */
export async function promptInfo(modelPath: string): Promise<{
  engines: string[];
  available: boolean;
  reason?: string;
}> {
  if (!isAbsolute(modelPath) || !(await exists(modelPath)))
    return {
      engines: [],
      available: false,
      reason: "Save a local model first.",
    };
  if (/\.pipe-onnx$/i.test(modelPath))
    return {
      engines: [],
      available: false,
      reason:
        "The onnx-light Python API does not directly execute PipelineProto containers. Pipeline execution requires an upstream public SDK capability.",
    };
  const engines: string[] = [];
  const genai = await exists(join(dirname(modelPath), "genai_config.json"));
  if (genai) engines.push("onnxruntime-genai");
  const provenance = await readProvenance(modelPath);
  const directories = [
    dirname(modelPath),
    ...(provenance?.source ? [provenance.source] : []),
  ];
  for (const directory of directories) {
    const configPath = join(directory, "config.json");
    if (
      !(await exists(configPath)) ||
      !(
        (await exists(join(directory, "tokenizer.json"))) ||
        (await exists(join(directory, "tokenizer_config.json")))
      )
    )
      continue;
    const config = JSON.parse(await readFile(configPath, "utf8")) as {
      architectures?: unknown;
      is_decoder?: unknown;
      is_encoder_decoder?: unknown;
    };
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
    if (causal && config.is_encoder_decoder !== true) {
      engines.unshift("onnx-light");
      break;
    }
  }
  return {
    engines,
    available: engines.length > 0,
    ...(engines.length
      ? {}
      : {
          reason:
            "Prompt execution requires local causal-model/tokenizer metadata for onnx-light, or an ONNX Runtime GenAI package (genai_config.json).",
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
        (token.score !== null && !Number.isFinite(token.score)),
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
): Promise<PromptResult> {
  signal.throwIfAborted();
  const options = optionsFrom(value);
  if (!options.validate) previous = undefined;
  const info = await promptInfo(modelPath);
  if (!info.engines.includes(options.engine))
    throw new Error(info.reason ?? "This model does not support that engine.");
  const provenance = await readProvenance(modelPath);
  const identity = await lstat(modelPath);
  const assets = await Promise.all(
    (await readdir(dirname(modelPath), { withFileTypes: true }))
      .filter((entry) => entry.isFile() && !entry.name.startsWith("."))
      .sort((left, right) => left.name.localeCompare(right.name))
      .map(async (entry) => {
        const stat = await lstat(join(dirname(modelPath), entry.name));
        return [entry.name, stat.size, stat.mtimeMs, stat.ctimeMs];
      }),
  );
  const context = JSON.stringify({
    modelPath,
    size: identity.size,
    modified: identity.mtimeMs,
    changed: identity.ctimeMs,
    assets,
    provenance,
    engine: options.engine,
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
    if (previousCapture)
      await writeFile(capture, previousCapture, {
        mode: 0o600,
        flag: "wx",
      });
    await writeFile(
      input,
      JSON.stringify({
        modelPath,
        ...options,
        provenance,
        capturePath: capture,
        captureBudget: retentionBudget,
        ...(previous
          ? {
              tokenIds: previous.tokens.map((token) => token.id),
              reuseCapture: true,
            }
          : {}),
      }),
      { mode: 0o600, flag: "wx" },
    );
    const script = resolve(__dirname, "..", "media", "run_prompt.py");
    await runProcess(
      { executable: pythonPath, args: ["-u", script, input, output] },
      signal,
      progress,
      { PYTHONUNBUFFERED: "1", HF_HUB_OFFLINE: "1", TRANSFORMERS_OFFLINE: "1" },
      "Prompt",
    );
    signal.throwIfAborted();
    const result = JSON.parse(await readFile(output, "utf8")) as PromptResult;
    if (result.prompt !== options.prompt || result.engine !== options.engine)
      throw new Error("The prompt runner returned a mismatched result.");
    checkTokens(result.tokens);
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
