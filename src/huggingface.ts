import { createHash, randomUUID } from "node:crypto";
import { createWriteStream } from "node:fs";
import { mkdir, rename, rm, stat } from "node:fs/promises";
import { join } from "node:path";
import { Readable, Transform } from "node:stream";
import { pipeline } from "node:stream/promises";
import { huggingFaceTasks } from "./huggingFaceTasks";

export interface HuggingFaceSource {
  url: string;
  fileName: string;
  size?: number;
}

export interface HuggingFaceRepository {
  url: string;
}

export interface DownloadProgress {
  received: number;
  total?: number;
}

export interface HuggingFaceModelFile extends HuggingFaceSource {
  path: string;
}

export interface HuggingFaceSearchResult extends HuggingFaceRepository {
  id: string;
  downloads: number;
  task?: string;
}

export async function searchHuggingFaceModels(
  query: string,
  task = "",
  signal?: AbortSignal,
): Promise<HuggingFaceSearchResult[]> {
  if (!huggingFaceTasks.some(({ value }) => value === task))
    throw new Error("Choose a supported Hugging Face task.");
  const url = new URL("https://huggingface.co/api/models");
  url.searchParams.set("filter", "onnx");
  url.searchParams.set("sort", "downloads");
  url.searchParams.set("direction", "-1");
  url.searchParams.set("limit", "12");
  for (const field of ["downloads", "pipeline_tag", "gated", "private"])
    url.searchParams.append("expand[]", field);
  if (query.trim()) url.searchParams.set("search", query.trim());
  if (task) url.searchParams.set("pipeline_tag", task);
  const response = await fetch(url, { signal });
  if (!response.ok) {
    await response.body?.cancel();
    throw new Error(
      `Hugging Face search failed: HTTP ${response.status} ${response.statusText}.`,
    );
  }
  const metadata: unknown = await response.json();
  if (!Array.isArray(metadata))
    throw new Error("Invalid Hugging Face search response.");
  const models: HuggingFaceSearchResult[] = [];
  const entries: unknown[] = metadata;
  for (const entry of entries) {
    if (
      !entry ||
      typeof entry !== "object" ||
      !("id" in entry) ||
      typeof entry.id !== "string" ||
      !/^[\w.-]+\/[\w.-]+$/.test(entry.id) ||
      entry.id.split("/").some((part) => /^\.+$/.test(part)) ||
      !("downloads" in entry) ||
      typeof entry.downloads !== "number" ||
      !Number.isSafeInteger(entry.downloads) ||
      entry.downloads < 0 ||
      ("pipeline_tag" in entry &&
        entry.pipeline_tag !== null &&
        typeof entry.pipeline_tag !== "string")
    )
      throw new Error("Invalid Hugging Face model search result.");
    if (
      ("private" in entry && entry.private) ||
      ("gated" in entry && entry.gated)
    )
      continue;
    models.push({
      id: entry.id,
      url: `https://huggingface.co/${entry.id}`,
      downloads: entry.downloads,
      ...("pipeline_tag" in entry && typeof entry.pipeline_tag === "string"
        ? { task: entry.pipeline_tag }
        : {}),
    });
  }
  return models;
}

export function parseHuggingFaceInput(
  input: string,
): HuggingFaceSource | HuggingFaceRepository {
  const value = input.trim();
  const url = new URL(
    /^[\w.-]+\/[\w.-]+$/.test(value)
      ? `https://huggingface.co/${value}`
      : value,
  );
  const parts = url.pathname.replace(/\/$/, "").slice(1).split("/");
  if (
    url.origin === "https://huggingface.co" &&
    !url.username &&
    !url.password &&
    !url.search &&
    !url.hash &&
    parts.length === 2 &&
    parts.every((part) => /^[\w.-]+$/.test(part) && !/^\.+$/.test(part))
  )
    return { url: `${url.origin}/${parts.join("/")}` };
  return parseHuggingFaceUrl(value);
}

export async function listHuggingFaceModels(
  repository: HuggingFaceRepository,
  signal?: AbortSignal,
): Promise<HuggingFaceModelFile[]> {
  const url = new URL(repository.url);
  url.pathname = `/api/models${url.pathname}`;
  url.searchParams.set("blobs", "true");
  url.searchParams.append("expand[]", "siblings");
  const response = await fetch(url, { signal });
  if (!response.ok) {
    await response.body?.cancel();
    throw new Error(
      `Cannot list Hugging Face repository ${repository.url}: HTTP ${response.status} ${response.statusText}. Only public, ungated repositories are supported.`,
    );
  }
  const metadata: unknown = await response.json();
  if (
    typeof metadata !== "object" ||
    metadata === null ||
    !("siblings" in metadata) ||
    !Array.isArray(metadata.siblings)
  )
    throw new Error(`Invalid Hugging Face file listing: ${repository.url}`);
  const files: HuggingFaceModelFile[] = [];
  const entries: unknown[] = metadata.siblings;
  for (const entry of entries) {
    if (
      typeof entry !== "object" ||
      entry === null ||
      !("rfilename" in entry) ||
      typeof entry.rfilename !== "string"
    )
      throw new Error(`Invalid Hugging Face file listing: ${repository.url}`);
    const path = entry.rfilename;
    if (!/\.onnx$/i.test(path)) continue;
    const parts = path.split("/");
    if (
      parts.some(
        (part) => !part || part === "." || part === ".." || /[\\\0]/.test(part),
      )
    )
      throw new Error(`Invalid Hugging Face model path: ${path}`);
    const size =
      "size" in entry && entry.size !== null
        ? entry.size
        : "lfs" in entry &&
            entry.lfs !== null &&
            typeof entry.lfs === "object" &&
            "size" in entry.lfs
          ? entry.lfs.size
          : undefined;
    if (
      size !== undefined &&
      (typeof size !== "number" || !Number.isSafeInteger(size) || size < 0)
    )
      throw new Error(`Invalid Hugging Face file size: ${path}`);
    files.push({
      path,
      fileName: parts[parts.length - 1],
      url: `${repository.url}/resolve/main/${parts.map(encodeURIComponent).join("/")}`,
      ...(size === undefined ? {} : { size }),
    });
  }
  if (!files.length)
    throw new Error(`No ONNX files found in ${repository.url}.`);
  return files.sort((a, b) => (a.path < b.path ? -1 : a.path > b.path ? 1 : 0));
}

/** Normalizes a public Hub file link without retaining query strings or fragments. */
export function parseHuggingFaceUrl(input: string): HuggingFaceSource {
  const url = new URL(input.trim());
  if (
    url.origin !== "https://huggingface.co" ||
    url.username ||
    url.password ||
    [...url.searchParams.keys()].some((key) => key !== "download")
  ) {
    throw new Error(
      "Use a public HTTPS huggingface.co file URL without credentials.",
    );
  }
  const parts = url.pathname.slice(1).split("/");
  if (
    parts.length < 5 ||
    !parts[0] ||
    !parts[1] ||
    !["blob", "resolve"].includes(parts[2]) ||
    !parts[3]
  ) {
    throw new Error(
      "Expected https://huggingface.co/owner/repository/blob/revision/path/model.onnx.",
    );
  }
  const fileName = decodeURIComponent(parts[parts.length - 1]);
  if (!/\.onnx$/i.test(fileName) || /[/\\\0]/.test(fileName)) {
    throw new Error("The Hugging Face URL must point to an .onnx model file.");
  }
  parts[2] = "resolve";
  url.pathname = `/${parts.join("/")}`;
  url.search = "";
  url.hash = "";
  return { url: url.href, fileName };
}

/**
 * Streams a file into an extension-owned cache and publishes it atomically only
 * after the caller can load it. Failed/cancelled transfers never become cache hits.
 * @param load Validates/projects the file and retains any bytes needed for recovery.
 */
export async function cachedHuggingFaceModel<T>(
  source: HuggingFaceSource,
  cacheRoot: string,
  load: (filePath: string) => Promise<T>,
  signal?: AbortSignal,
  onProgress?: (progress: DownloadProgress) => void,
): Promise<{ value: T; cached: boolean; filePath: string }> {
  signal?.throwIfAborted();
  const key = createHash("sha256").update(source.url).digest("hex");
  const directory = join(cacheRoot, key);
  const filePath = join(directory, "model.onnx");
  const existing = await stat(filePath).catch((error: unknown) => {
    if (error instanceof Error && "code" in error && error.code === "ENOENT")
      return undefined;
    throw error;
  });
  if (existing) {
    if (!existing.isFile() || existing.size === 0)
      throw new Error(`Invalid cached model: ${filePath}`);
    const value = await load(filePath);
    signal?.throwIfAborted();
    return { value, cached: true, filePath };
  }

  await mkdir(directory, { recursive: true });
  const temporary = join(directory, `${randomUUID()}.partial`);
  try {
    const response = await fetch(source.url, { signal });
    if (!response.ok) {
      await response.body?.cancel();
      throw new Error(
        `Hugging Face download failed for ${source.url}: HTTP ${response.status} ${response.statusText}. Only public, ungated model files are supported.`,
      );
    }
    if (!response.body) throw new Error("Hugging Face returned no model data.");
    const length = Number(response.headers.get("content-length"));
    const encoding = response.headers.get("content-encoding");
    const total =
      Number.isSafeInteger(length) &&
      length > 0 &&
      (!encoding || encoding === "identity")
        ? length
        : undefined;
    let received = 0;
    let reportedAt = Date.now();
    onProgress?.({ received, total });
    const progress = new Transform({
      transform(chunk: Buffer, _encoding, callback) {
        received += chunk.length;
        const now = Date.now();
        if (now - reportedAt >= 100 || received === total) {
          onProgress?.({ received, total });
          reportedAt = now;
        }
        callback(null, chunk);
      },
    });
    await pipeline(
      Readable.fromWeb(response.body),
      progress,
      createWriteStream(temporary, { flags: "wx" }),
      { signal },
    );
    onProgress?.({ received, total });
    const value = await load(temporary);
    signal?.throwIfAborted();
    await rename(temporary, filePath);
    return { value, cached: false, filePath };
  } finally {
    await rm(temporary, { force: true });
  }
}
