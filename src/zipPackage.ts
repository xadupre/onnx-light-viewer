import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { unzip, zip, type UnzipFileInfo } from "fflate";

const maxEntries = 10_000;
const maxUncompressedSize = 8 * 1024 ** 3;
const zip32Limit = 0xffffffff;
const modelExtension = /(?:\.onnx(?:\.enc)?|\.pipe-onnx)$/i;

export interface ModelPackage {
  directory: string;
  models: string[];
  dispose(): Promise<void>;
}

function validateName(name: string, directory: boolean): string[] {
  const path = directory ? name.slice(0, -1) : name;
  const parts = path.split("/");
  if (
    !path ||
    /[\\\0:]/.test(name) ||
    parts.some(
      (part) =>
        !part ||
        part === "." ||
        part === ".." ||
        /[. ]$/.test(part) ||
        /^(?:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\.|$)/i.test(part),
    ) ||
    path === "__proto__"
  ) {
    throw new Error(`Unsafe ZIP entry name: ${JSON.stringify(name)}`);
  }
  return parts;
}

function nameValidator() {
  const explicit = new Set<string>();
  const paths = new Map<string, { name: string; directory: boolean }>();
  return (name: string, directory: boolean) => {
    const parts = validateName(name, directory);
    const path = parts.join("/");
    const key = path.toLowerCase();
    if (explicit.has(key)) {
      throw new Error(`Duplicate ZIP entry: ${name}`);
    }
    explicit.add(key);
    for (let i = 1; i <= parts.length; i++) {
      const current = parts.slice(0, i).join("/");
      const currentKey = current.toLowerCase();
      const isDirectory = i < parts.length || directory;
      const previous = paths.get(currentKey);
      if (
        previous &&
        (previous.name !== current || !previous.directory || !isDirectory)
      ) {
        throw new Error(`Conflicting ZIP entry: ${name}`);
      }
      paths.set(currentKey, { name: current, directory: isDirectory });
    }
  };
}

interface Entry extends UnzipFileInfo {
  crc: number;
}

// fflate does not expose Unix modes or CRCs. Inspect only that metadata here;
// names, ZIP64 sizes and decompression remain fflate's responsibility.
function centralMetadata(bytes: Uint8Array): { crc: number; mode: number }[] {
  if (bytes.length < 22) throw new Error("Invalid ZIP archive");
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let end = bytes.length - 22;
  const earliest = Math.max(0, end - 65_535);
  while (
    end >= earliest &&
    (view.getUint32(end, true) !== 0x06054b50 ||
      end + 22 + view.getUint16(end + 20, true) !== bytes.length)
  ) {
    end--;
  }
  if (end < earliest) throw new Error("Invalid ZIP archive");
  if (
    view.getUint16(end + 4, true) !== 0 ||
    view.getUint16(end + 6, true) !== 0
  ) {
    throw new Error("Multi-disk ZIP archives are unsupported");
  }
  let count = view.getUint16(end + 10, true);
  let offset = view.getUint32(end + 16, true);
  let size = view.getUint32(end + 12, true);
  let centralEnd = end;
  if (end >= 20 && view.getUint32(end - 20, true) === 0x07064b50) {
    const zip64Offset = Number(view.getBigUint64(end - 12, true));
    if (
      !Number.isSafeInteger(zip64Offset) ||
      zip64Offset < 0 ||
      zip64Offset + 56 > end - 20 ||
      view.getUint32(zip64Offset, true) !== 0x06064b50 ||
      view.getUint32(zip64Offset + 16, true) !== 0 ||
      view.getUint32(zip64Offset + 20, true) !== 0
    ) {
      throw new Error("Invalid ZIP64 archive");
    }
    count = Number(view.getBigUint64(zip64Offset + 32, true));
    size = Number(view.getBigUint64(zip64Offset + 40, true));
    offset = Number(view.getBigUint64(zip64Offset + 48, true));
    centralEnd = zip64Offset;
  }
  if (
    count > maxEntries ||
    !Number.isSafeInteger(offset) ||
    !Number.isSafeInteger(size) ||
    offset < 0 ||
    size < 0 ||
    offset + size !== centralEnd
  ) {
    throw new Error("Invalid ZIP directory or too many ZIP entries");
  }
  const result: { crc: number; mode: number }[] = [];
  for (let i = 0; i < count; i++) {
    if (
      offset + 46 > centralEnd ||
      view.getUint32(offset, true) !== 0x02014b50
    ) {
      throw new Error("Invalid ZIP directory");
    }
    if (view.getUint16(offset + 8, true) & 1) {
      throw new Error("Encrypted ZIP entries are unsupported");
    }
    result.push({
      crc: view.getUint32(offset + 16, true),
      mode: view.getUint32(offset + 38, true) >>> 16,
    });
    offset +=
      46 +
      view.getUint16(offset + 28, true) +
      view.getUint16(offset + 30, true) +
      view.getUint16(offset + 32, true);
    if (offset > centralEnd) throw new Error("Invalid ZIP directory");
  }
  if (offset !== centralEnd) throw new Error("Invalid ZIP directory");
  return result;
}

const crcTable = Uint32Array.from({ length: 256 }, (_, value) => {
  for (let bit = 0; bit < 8; bit++) {
    value = (value >>> 1) ^ (value & 1 ? 0xedb88320 : 0);
  }
  return value >>> 0;
});

function crc32(bytes: Uint8Array): number {
  let crc = 0xffffffff;
  for (const byte of bytes) crc = (crc >>> 8) ^ crcTable[(crc ^ byte) & 255];
  return (crc ^ 0xffffffff) >>> 0;
}

function unzipAsync(
  bytes: Uint8Array,
  filter: (entry: UnzipFileInfo) => boolean,
): Promise<Record<string, Uint8Array>> {
  return new Promise((resolve, reject) => {
    try {
      unzip(bytes, { filter }, (error, files) =>
        error ? reject(error) : resolve(files),
      );
    } catch (error) {
      reject(error);
    }
  });
}

export async function extractModelPackage(
  bytes: Uint8Array,
): Promise<ModelPackage> {
  // Snapshot before the asynchronous validation pass so caller mutations cannot
  // change names or sizes between validation and extraction.
  bytes = bytes.slice();
  const metadata = centralMetadata(bytes);
  const validate = nameValidator();
  const entries: Entry[] = [];
  let total = 0;
  await unzipAsync(bytes, (entry) => {
    const directory = entry.name.endsWith("/");
    validate(entry.name, directory);
    const info = metadata[entries.length];
    if (!info) throw new Error("Invalid ZIP entry count");
    const type = info.mode & 0xf000;
    if (type !== 0 && type !== (directory ? 0x4000 : 0x8000)) {
      throw new Error(`Unsupported ZIP entry type: ${entry.name}`);
    }
    total += entry.originalSize;
    if (
      entries.length >= maxEntries ||
      !Number.isSafeInteger(entry.originalSize) ||
      entry.originalSize < 0 ||
      total > maxUncompressedSize ||
      (directory && entry.originalSize !== 0) ||
      (entry.compression !== 0 && entry.compression !== 8) ||
      (entry.compression === 0 && entry.size !== entry.originalSize)
    ) {
      throw new Error(
        "ZIP entry size, compression or package limit is invalid",
      );
    }
    entries.push({ ...entry, crc: info.crc });
    return false;
  });
  if (entries.length !== metadata.length) {
    throw new Error("Invalid ZIP entry count");
  }
  const models = entries
    .filter(
      (entry) => !entry.name.endsWith("/") && modelExtension.test(entry.name),
    )
    .map((entry) => entry.name)
    .sort();
  if (!models.length) throw new Error("ZIP package contains no ONNX models");

  const files = await unzipAsync(bytes, () => true);
  for (const entry of entries) {
    const data = Object.hasOwn(files, entry.name)
      ? files[entry.name]
      : undefined;
    if (
      !data ||
      data.length !== entry.originalSize ||
      crc32(data) !== entry.crc
    ) {
      throw new Error(`Corrupt ZIP entry: ${entry.name}`);
    }
  }
  const directory = await mkdtemp(
    join(resolve(tmpdir()), "onnx-light-viewer-package-"),
  );
  const dispose = () => rm(directory, { recursive: true, force: true });
  try {
    for (const entry of entries) {
      const target = join(directory, entry.name);
      if (entry.name.endsWith("/")) {
        await mkdir(target, { recursive: true, mode: 0o700 });
      } else {
        await mkdir(dirname(target), { recursive: true, mode: 0o700 });
        await writeFile(target, files[entry.name], { flag: "wx", mode: 0o600 });
      }
    }
    return {
      directory,
      models: models.map((name) => join(directory, name)),
      dispose,
    };
  } catch (error) {
    await dispose();
    throw error;
  }
}

export async function createModelPackage(
  modelName: string,
  model: Uint8Array,
  weights: { name: string; bytes: Uint8Array }[],
): Promise<Uint8Array> {
  const validate = nameValidator();
  const files: Record<string, Uint8Array> = Object.create(null);
  const entries = [{ name: modelName, bytes: model }, ...weights];
  if (entries.length >= 0xffff) {
    throw new Error("Creating ZIP64 packages is unsupported");
  }
  if (!modelExtension.test(modelName)) {
    throw new Error(
      "The model filename must end in .onnx, .onnx.enc or .pipe-onnx",
    );
  }
  let total = 0;
  let outputBound = 22;
  for (const entry of entries) {
    validate(entry.name, false);
    total += entry.bytes.length;
    const nameSize = Buffer.byteLength(entry.name, "utf8");
    // fflate's non-ZIP64 writer needs 32-bit sizes and offsets. Account for
    // both headers and its worst-case DEFLATE buffer, not just input bytes.
    const compressedBound =
      entry.bytes.length + 5 * (1 + Math.ceil(entry.bytes.length / 7000));
    outputBound += compressedBound + 76 + 2 * nameSize;
    if (
      entry.bytes.length >= zip32Limit ||
      compressedBound >= zip32Limit ||
      outputBound >= zip32Limit
    ) {
      throw new Error("Creating ZIP64 packages is unsupported");
    }
    if (entries.length > maxEntries || total > maxUncompressedSize) {
      throw new Error("ZIP package limit exceeded");
    }
    files[entry.name] = entry.bytes;
  }
  return new Promise((resolve, reject) => {
    zip(files, (error, result) => (error ? reject(error) : resolve(result)));
  });
}
