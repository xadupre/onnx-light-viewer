import { createHash } from "node:crypto";
import { execFile } from "node:child_process";
import { createWriteStream } from "node:fs";
import {
  lstat,
  mkdir,
  mkdtemp,
  readFile,
  rename,
  rm,
  writeFile,
} from "node:fs/promises";
import { dirname, join } from "node:path";
import { totalmem } from "node:os";
import { promisify } from "node:util";
import { Readable, Transform } from "node:stream";
import { pipeline } from "node:stream/promises";
import { XMLParser, XMLValidator } from "fast-xml-parser";
import type { DownloadProgress } from "./huggingface";
import { inspectRuntimePackage } from "./runtimePackage";

const catalogUrl = "https://api.catalog.azureml.ms/asset-gallery/v1.0/models";
const registryUrl =
  "https://centralus.api.azureml.ms/modelregistry/v1.0/registry/models/nonazureaccount";
const headers = {
  "User-Agent": "AzureAiStudio",
  "Content-Type": "application/json",
  "x-ms-use-full-service-contracts": "true",
};

export interface FoundryModel {
  id: string;
  name: string;
  device: string;
  provider: string;
  license: string;
  size: number;
  memory?: number;
}

export interface FoundryMemory {
  ram: number;
  nvidia?: number;
  warning?: string;
}

function usesNvidiaMemory(model: FoundryModel): boolean {
  return (
    model.device === "gpu" &&
    /^(?:CUDA|Tensorrt|NvTensorRTRTX)ExecutionProvider$/i.test(model.provider)
  );
}

export async function inspectFoundryMemory(
  probeNvidia: boolean,
  signal?: AbortSignal,
): Promise<FoundryMemory> {
  signal?.throwIfAborted();
  const constrained = process.constrainedMemory();
  const memory: FoundryMemory = {
    ram: constrained > 0 ? Math.min(totalmem(), constrained) : totalmem(),
  };
  if (!probeNvidia) return memory;
  try {
    const { stdout } = await promisify(execFile)(
      "nvidia-smi",
      ["--query-gpu=memory.total", "--format=csv,noheader,nounits"],
      { signal, timeout: 5000, maxBuffer: 65536, windowsHide: true },
    );
    const sizes = stdout
      .trim()
      .split(/\r?\n/)
      .map((value) => Number(value.trim()) * 1048576);
    if (
      !sizes.length ||
      sizes.some((size) => !Number.isSafeInteger(size) || size <= 0)
    )
      throw new Error("NVIDIA returned no valid dedicated-memory capacity.");
    // A model must fit on one adapter; adding the capacities would overstate it.
    memory.nvidia = Math.max(...sizes);
  } catch (error) {
    signal?.throwIfAborted();
    memory.warning = `NVIDIA VRAM could not be checked: ${error instanceof Error ? error.message : String(error)}`;
  }
  return memory;
}

export function filterFoundryModels(
  models: FoundryModel[],
  memory: FoundryMemory,
) {
  const visible: (FoundryModel & { memoryLimit: number | undefined })[] = [];
  let hidden = 0;
  let unchecked = 0;
  for (const model of models) {
    const memoryLimit =
      model.device === "cpu" || model.device === "npu"
        ? memory.ram
        : usesNvidiaMemory(model)
          ? memory.nvidia
          : undefined;
    if (
      model.memory !== undefined &&
      memoryLimit !== undefined &&
      model.memory > memoryLimit
    ) {
      hidden++;
      continue;
    }
    if (model.memory === undefined || memoryLimit === undefined) unchecked++;
    visible.push({ ...model, memoryLimit });
  }
  return { models: visible, memory, hidden, unchecked };
}

export async function localFoundryModels(device: string, signal?: AbortSignal) {
  const models = await listFoundryModels(device, signal);
  const memory = await inspectFoundryMemory(
    models.some(usesNvidiaMemory),
    signal,
  );
  return filterFoundryModels(models, memory);
}

function object(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function assetId(value: unknown): string {
  if (
    typeof value !== "string" ||
    !/^azureml:\/\/registries\/[\w.-]+\/models\/[\w.-]+\/versions\/\d+$/.test(
      value,
    )
  )
    throw new Error("Invalid Foundry Local model ID.");
  return value;
}

async function request(
  url: string | URL,
  options: RequestInit,
): Promise<Response> {
  const response = await fetch(url, { ...options, redirect: "error" });
  if (!response.ok) {
    await response.body?.cancel();
    // Signed blob URLs are credentials and must never appear in error messages.
    throw new Error(`Foundry Local request failed: HTTP ${response.status}.`);
  }
  return response;
}

export async function listFoundryModels(
  device: string,
  signal?: AbortSignal,
): Promise<FoundryModel[]> {
  if (!["cpu", "gpu", "npu", ""].includes(device))
    throw new Error("Choose a supported Foundry Local device.");
  const filters = [
    { field: "type", operator: "eq", values: ["models"] },
    { field: "kind", operator: "eq", values: ["Versioned"] },
    { field: "labels", operator: "eq", values: ["latest"] },
    {
      field: "annotations/systemCatalogData/deploymentOptions",
      operator: "eq",
      values: ["Foundry Local on Devices"],
    },
    { field: "annotations/archived", operator: "NotEquals", values: ["true"] },
    ...(device
      ? [
          {
            field: "properties/variantInfo/variantMetadata/device",
            operator: "eq",
            values: [device],
          },
        ]
      : []),
  ];
  const models = new Map<string, FoundryModel>();
  const seen = new Set<string>();
  let continuationToken: string | undefined;
  do {
    const response = await request(catalogUrl, {
      method: "POST",
      headers,
      signal,
      body: JSON.stringify({ filters, pageSize: 50, continuationToken }),
    });
    const data: unknown = await response.json();
    if (!object(data) || !Array.isArray(data.value))
      throw new Error("Invalid Foundry Local catalog response.");
    for (const entry of data.value) {
      if (!object(entry) || !object(entry.properties)) continue;
      const properties = entry.properties;
      const variant =
        object(properties.variantInfo) &&
        properties.variantInfo.variantMetadata;
      if (!object(variant) || variant.modelType !== "ONNX") continue;
      const annotations = object(entry.annotations) ? entry.annotations : {};
      const tags = object(annotations.tags) ? annotations.tags : {};
      if (tags.foundryLocal === "hide" || tags.foundryLocal === "test")
        continue;
      const system = object(annotations.systemCatalogData)
        ? annotations.systemCatalogData
        : {};
      if (
        typeof properties.name !== "string" ||
        typeof variant.device !== "string" ||
        typeof variant.executionProvider !== "string" ||
        !Number.isSafeInteger(variant.fileSizeBytes) ||
        typeof variant.fileSizeBytes !== "number" ||
        variant.fileSizeBytes < 0 ||
        (variant.vRamFootprintBytes !== undefined &&
          variant.vRamFootprintBytes !== null &&
          (typeof variant.vRamFootprintBytes !== "number" ||
            !Number.isSafeInteger(variant.vRamFootprintBytes) ||
            variant.vRamFootprintBytes < 0))
      )
        throw new Error("Invalid Foundry Local model metadata.");
      const id = assetId(entry.assetId);
      models.set(id, {
        id,
        name: `${properties.name}:${properties.version}`,
        device: variant.device,
        provider: variant.executionProvider,
        license:
          typeof system.license === "string"
            ? system.license
            : "See model license",
        size: variant.fileSizeBytes,
        ...(typeof variant.vRamFootprintBytes === "number" &&
        variant.vRamFootprintBytes > 0
          ? { memory: variant.vRamFootprintBytes }
          : {}),
      });
    }
    if (data.continuationToken) {
      if (
        typeof data.continuationToken !== "string" ||
        seen.has(data.continuationToken)
      )
        throw new Error("Invalid Foundry Local catalog continuation token.");
      seen.add(data.continuationToken);
      continuationToken = data.continuationToken;
    } else continuationToken = undefined;
  } while (continuationToken);
  return [...models.values()].sort((a, b) => a.name.localeCompare(b.name));
}

interface ModelFile {
  path: string;
  size: number;
}

function modelPath(value: unknown): string {
  if (
    typeof value !== "string" ||
    !value ||
    value
      .split("/")
      .some(
        (part) =>
          !part ||
          part === "." ||
          part === ".." ||
          /[\x00-\x1f<>:"\\|?*]/.test(part) ||
          /[ .]$/.test(part) ||
          /^(?:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\.|$)/i.test(part),
      )
  )
    throw new Error("Unsafe Foundry Local model file path.");
  return value;
}

function blobUrl(value: unknown): URL {
  if (typeof value !== "string")
    throw new Error("Missing Foundry Local download URL.");
  const url = new URL(value);
  if (
    url.protocol !== "https:" ||
    !/^[a-z0-9]+\.blob\.core\.windows\.net$/.test(url.hostname) ||
    url.username ||
    url.password ||
    url.port ||
    url.hash
  )
    throw new Error("Invalid Foundry Local blob host.");
  return url;
}

async function listFiles(
  container: URL,
  signal?: AbortSignal,
): Promise<ModelFile[]> {
  const files: ModelFile[] = [];
  const names = new Set<string>();
  const markers = new Set<string>();
  const parser = new XMLParser({
    ignoreAttributes: true,
    parseTagValue: false,
    trimValues: false,
    isArray: (name) => name === "Blob",
  });
  let marker = "";
  do {
    const url = new URL(container);
    url.searchParams.set("restype", "container");
    url.searchParams.set("comp", "list");
    if (marker) url.searchParams.set("marker", marker);
    const response = await request(url, { signal });
    const xml = await response.text();
    if (/<!DOCTYPE/i.test(xml) || XMLValidator.validate(xml) !== true)
      throw new Error("Invalid Foundry Local blob listing.");
    const data: unknown = parser.parse(xml);
    const listing = object(data) && data.EnumerationResults;
    if (!object(listing))
      throw new Error("Invalid Foundry Local blob listing.");
    const blobs = object(listing.Blobs) ? listing.Blobs.Blob : [];
    if (!Array.isArray(blobs))
      throw new Error("Invalid Foundry Local blob entries.");
    for (const blob of blobs) {
      if (!object(blob) || !object(blob.Properties))
        throw new Error("Invalid Foundry Local blob entry.");
      const path = modelPath(blob.Name);
      const key = path.toLowerCase();
      const size = Number(blob.Properties["Content-Length"]);
      if (names.has(key) || !Number.isSafeInteger(size) || size < 0)
        throw new Error("Invalid or duplicate Foundry Local file.");
      names.add(key);
      files.push({ path, size });
    }
    const next = listing.NextMarker;
    if (typeof next !== "string" || (next && markers.has(next)))
      throw new Error("Invalid Foundry Local blob continuation marker.");
    marker = next;
    markers.add(marker);
  } while (marker);
  if (!files.some((file) => /\.(?:onnx|ort)$/i.test(file.path)))
    throw new Error("The Foundry Local package contains no ONNX or ORT model.");
  return files;
}

async function cacheExists(directory: string): Promise<boolean> {
  try {
    const info = await lstat(directory);
    if (!info.isDirectory() || info.isSymbolicLink())
      throw new Error("Invalid Foundry Local cache directory.");
    return true;
  } catch (error) {
    if (error instanceof Error && "code" in error && error.code === "ENOENT")
      return false;
    throw error;
  }
}

async function validateCache(directory: string, id: string): Promise<void> {
  const manifest: unknown = JSON.parse(
    await readFile(join(directory, ".viewer-download.json"), "utf8"),
  );
  if (!object(manifest) || manifest.id !== id || !Array.isArray(manifest.files))
    throw new Error("Invalid Foundry Local cache manifest.");
  for (const file of manifest.files) {
    if (!object(file)) throw new Error("Invalid Foundry Local cached file.");
    const info = await lstat(join(directory, modelPath(file.path)));
    if (!info.isFile() || info.size !== file.size)
      throw new Error(
        "Incomplete Foundry Local cache. Remove this model's cache directory and retry.",
      );
  }
  await inspectRuntimePackage(directory);
}

export async function downloadFoundryModel(
  input: string,
  cacheRoot: string,
  signal?: AbortSignal,
  onProgress?: (progress: DownloadProgress) => void,
): Promise<string> {
  const id = assetId(input);
  signal?.throwIfAborted();
  const root = join(cacheRoot, "foundry");
  const directory = join(root, createHash("sha256").update(id).digest("hex"));
  if (await cacheExists(directory)) {
    await validateCache(directory, id);
    signal?.throwIfAborted();
    return directory;
  }
  const registry = new URL(registryUrl);
  registry.searchParams.set("assetId", id);
  const response = await request(registry, { headers, signal });
  const data: unknown = await response.json();
  if (!object(data))
    throw new Error("Invalid Foundry Local registry response.");
  const container = blobUrl(data.blobSasUri);
  const files = await listFiles(container, signal);
  const total = files.reduce((sum, file) => sum + file.size, 0);
  if (!Number.isSafeInteger(total))
    throw new Error("Foundry Local package is too large.");
  await mkdir(root, { recursive: true });
  const temporary = await mkdtemp(join(root, ".download-"));
  let received = 0;
  let reportedAt = 0;
  try {
    for (const file of files) {
      signal?.throwIfAborted();
      const url = new URL(container);
      url.pathname = `${url.pathname.replace(/\/$/, "")}/${file.path.split("/").map(encodeURIComponent).join("/")}`;
      const response = await request(url, { signal });
      if (!response.body)
        throw new Error("Foundry Local returned no file data.");
      const destination = join(temporary, file.path);
      await mkdir(dirname(destination), { recursive: true });
      let bytes = 0;
      const progress = new Transform({
        transform(chunk: Buffer, _encoding, callback) {
          bytes += chunk.length;
          received += chunk.length;
          const now = Date.now();
          if (now - reportedAt >= 100) {
            onProgress?.({ received, total });
            reportedAt = now;
          }
          callback(null, chunk);
        },
      });
      await pipeline(
        Readable.fromWeb(response.body),
        progress,
        createWriteStream(destination, { flags: "wx" }),
        { signal },
      );
      if (bytes !== file.size)
        throw new Error("Incomplete Foundry Local file download.");
    }
    await inspectRuntimePackage(temporary);
    await writeFile(
      join(temporary, ".viewer-download.json"),
      JSON.stringify({ id, files }),
      { flag: "wx" },
    );
    signal?.throwIfAborted();
    try {
      await rename(temporary, directory);
    } catch (error) {
      if (
        !(
          error instanceof Error &&
          "code" in error &&
          (error.code === "EEXIST" ||
            error.code === "ENOTEMPTY" ||
            (process.platform === "win32" &&
              (error.code === "EPERM" || error.code === "EACCES")))
        ) ||
        !(await cacheExists(directory))
      )
        throw error;
      await validateCache(directory, id);
    }
    signal?.throwIfAborted();
    onProgress?.({ received, total });
    return directory;
  } finally {
    await rm(temporary, { recursive: true, force: true });
  }
}
