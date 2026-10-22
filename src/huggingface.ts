import { createHash, randomUUID } from "node:crypto";
import { createWriteStream } from "node:fs";
import { mkdir, rename, rm, stat } from "node:fs/promises";
import { join } from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";

export interface HuggingFaceSource {
  url: string;
  fileName: string;
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
    await pipeline(
      Readable.fromWeb(response.body),
      createWriteStream(temporary, { flags: "wx" }),
      { signal },
    );
    const value = await load(temporary);
    signal?.throwIfAborted();
    await rename(temporary, filePath);
    return { value, cached: false, filePath };
  } finally {
    await rm(temporary, { force: true });
  }
}
