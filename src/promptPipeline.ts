import { createHash } from "node:crypto";
import { open, readFile, rm, stat, writeFile } from "node:fs/promises";
import { basename, dirname, isAbsolute, join } from "node:path";
import { preparePromptPipeline } from "./graph";
import {
  readProvenance,
  provenancePath,
  type ExportProvenance,
} from "./export";
import { promptInfo } from "./prompt";

export interface PromptPipelineInfo {
  path: string;
  available: boolean;
  label: string;
}

interface Sources {
  root: string;
  prefill: string;
  embedding: string;
  provenance: ExportProvenance;
}

export type PromptPipelineChecks = Map<
  string,
  ReturnType<typeof preparePromptPipeline>
>;

async function file(path: string): Promise<boolean> {
  try {
    return (await stat(path)).isFile();
  } catch (error) {
    if (error instanceof Error && "code" in error && error.code === "ENOENT")
      return false;
    throw error;
  }
}

async function sources(path: string): Promise<Sources | undefined> {
  const provenance = await readProvenance(path);
  if (provenance?.exporter !== "mobius") return;
  let root = provenance.packageDirectory ?? dirname(path);
  if (
    !provenance.packageDirectory &&
    ["decoder", "embedding", "model"].includes(basename(root)) &&
    !(await file(join(root, "config.json"))) &&
    !(await file(join(root, "genai_config.json")))
  )
    root = dirname(root);
  let prefill = path;
  const configPath = join(root, "genai_config.json");
  if (await file(configPath)) {
    const config = JSON.parse(await readFile(configPath, "utf8"));
    const decoder: unknown = config?.model?.decoder?.filename;
    if (
      typeof decoder !== "string" ||
      !decoder ||
      isAbsolute(decoder) ||
      decoder.split(/[\\/]/).includes("..")
    )
      throw new Error("Invalid Mobius decoder filename in genai_config.json.");
    prefill = join(root, decoder);
  } else if (await file(join(root, "decoder", "model.onnx"))) {
    prefill = join(root, "decoder", "model.onnx");
  } else if (await file(join(root, "model.onnx"))) {
    prefill = join(root, "model.onnx");
  }
  const embedding = join(root, "embedding", "model.onnx");
  return {
    root,
    prefill,
    embedding: (await file(embedding)) ? embedding : "",
    provenance,
  };
}

/** Inspects saved package contracts/graph ports, never runs inference or loads weights. */
export async function promptPipelineInfo(
  path: string,
  checks: PromptPipelineChecks = new Map(),
): Promise<PromptPipelineInfo> {
  if (!isAbsolute(path) || !(await file(path)))
    return { path, available: false, label: "File no longer exists" };
  if (/\.pipe-onnx$/i.test(path)) {
    const available = (await promptInfo(path)).engines.includes("onnx-light");
    return {
      path,
      available,
      label: available
        ? "PipelineProto: generation assets available"
        : "PipelineProto: causal-model/tokenizer assets are missing",
    };
  }
  const candidate = await sources(path);
  if (!candidate)
    return { path, available: false, label: "No Mobius pipeline identified" };
  if (!(await file(candidate.prefill)))
    return { path, available: false, label: "Mobius: decoder file is missing" };
  const info = await promptInfo(candidate.prefill);
  if (!info.engines.includes("onnx-light"))
    return {
      path,
      available: false,
      label: "Mobius: causal-model/tokenizer assets are missing",
    };
  const key = JSON.stringify([
    candidate.prefill,
    candidate.embedding,
    candidate.root,
  ]);
  let check = checks.get(key);
  if (!check) {
    check = preparePromptPipeline(
      candidate.prefill,
      candidate.embedding,
      candidate.root,
      null,
    );
    checks.set(key, check);
  }
  const result = await check;
  return {
    path,
    available: result.available,
    label: result.available
      ? "Mobius: generation pipeline can be assembled"
      : `Mobius: ${result.reason}`,
  };
}

/** Writes a descriptor beside the export, reusing its external weights unchanged. */
export async function loadPromptPipeline(path: string): Promise<string> {
  if (/\.pipe-onnx$/i.test(path)) {
    if (!(await file(path)))
      throw new Error("The pipeline file no longer exists.");
    return path;
  }
  const candidate = await sources(path);
  if (!candidate) throw new Error("This history entry is not a Mobius export.");
  const info = await promptPipelineInfo(path);
  if (!info.available) throw new Error(info.label);
  const identities = await Promise.all(
    [candidate.prefill, candidate.embedding]
      .filter(Boolean)
      .map(async (source) => {
        const identity = await stat(source);
        return [source, identity.size, identity.mtimeMs, identity.ctimeMs];
      }),
  );
  const key = createHash("sha256")
    .update(JSON.stringify(["viewer-pipeline-v1", identities]))
    .digest("hex")
    .slice(0, 20);
  const output = join(candidate.root, `viewer-${key}.pipe-onnx`);
  let handle;
  try {
    handle = await open(output, "wx", 0o600);
  } catch (error) {
    if (!(error instanceof Error && "code" in error && error.code === "EEXIST"))
      throw error;
    const existing = await readProvenance(output);
    if (
      existing?.exporter !== "mobius" ||
      existing.packageDirectory !== candidate.root
    )
      throw new Error(
        `Pipeline destination already exists and is not owned by the viewer: ${output}`,
      );
    return output;
  }
  try {
    const result = await preparePromptPipeline(
      candidate.prefill,
      candidate.embedding,
      candidate.root,
      handle.fd,
    );
    if (!result.available) throw new Error(result.reason);
    await writeFile(
      provenancePath(output),
      JSON.stringify({
        ...candidate.provenance,
        packageDirectory: candidate.root,
      }),
      { flag: "wx", mode: 0o600 },
    );
  } catch (error) {
    await handle.close();
    await rm(output);
    throw error;
  }
  await handle.close();
  return output;
}
