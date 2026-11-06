import { createWriteStream } from "node:fs";
import { lstat, mkdir, mkdtemp, readdir, realpath, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { Transform } from "node:stream";
import { pipeline } from "node:stream/promises";
import crc32 from "buffer-crc32";
import { openPromise } from "yauzl";
import { ZipFile } from "yazl";

function archiveMember(name: string): string {
  const member = name.replace(/\/$/, "");
  if (
    !member ||
    member.includes("\\") ||
    member.includes("\0") ||
    member
      .split("/")
      .some(
        (part) =>
          !part ||
          part === "." ||
          part === ".." ||
          part.includes(":") ||
          /[. ]$/.test(part) ||
          /^(?:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\.|$)/i.test(part),
      )
  )
    throw new Error(`Unsafe ZIP package member: ${name}`);
  return member;
}

async function extractArchive(
  source: string,
  directory: string,
): Promise<void> {
  const zip = await openPromise(source, {
    lazyEntries: true,
    autoClose: false,
    strictFileNames: true,
    validateEntrySizes: true,
  });
  try {
    const members = new Set<string>();
    for await (const entry of zip.eachEntry()) {
      const name = archiveMember(entry.fileName);
      if (members.has(name))
        throw new Error(`Duplicate ZIP package member: ${name}`);
      members.add(name);
      const directoryEntry = entry.fileName.endsWith("/");
      const type = (entry.externalFileAttributes >>> 16) & 0xf000;
      if (type && type !== (directoryEntry ? 0x4000 : 0x8000))
        throw new Error(
          `ZIP package links and special files are not supported: ${name}`,
        );
      if (entry.isEncrypted())
        throw new Error(
          `Encrypted ZIP package entries are not supported: ${name}`,
        );
      const target = join(directory, name);
      if (directoryEntry) {
        await mkdir(target, { recursive: true });
        continue;
      }
      await mkdir(dirname(target), { recursive: true });
      const input = await zip.openReadStreamPromise(entry);
      let checksum = 0;
      const verify = new Transform({
        transform(chunk: Buffer, _encoding, callback) {
          checksum = crc32.unsigned(chunk, checksum);
          callback(null, chunk);
        },
        flush(callback) {
          callback(
            checksum === entry.crc32
              ? undefined
              : new Error(`ZIP package checksum mismatch: ${name}`),
          );
        },
      });
      await pipeline(
        input,
        verify,
        createWriteStream(target, { flags: "wx", mode: 0o600 }),
      );
    }
  } finally {
    zip.close();
  }
}

export async function writePackageArchive(
  directory: string,
  destination: string,
): Promise<void> {
  const zip = new ZipFile();
  const output = createWriteStream(destination, { flags: "wx" });
  zip.on("error", (error: Error) => output.destroy(error));
  const writing = pipeline(zip.outputStream, output);
  // Observe stream failures even while the directory walk is still pending.
  void writing.catch(() => {});
  try {
    const add = async (prefix = ""): Promise<void> => {
      for (const entry of await readdir(join(directory, prefix), {
        withFileTypes: true,
      })) {
        const name = archiveMember(
          prefix ? `${prefix}/${entry.name}` : entry.name,
        );
        if (entry.isDirectory()) {
          zip.addEmptyDirectory(name);
          await add(name);
        } else if (entry.isFile()) {
          zip.addFile(join(directory, name), name, { compress: false });
        } else {
          throw new Error(`Package assets must be regular files: ${name}`);
        }
      }
    };
    await add();
    zip.end({ forceZip64Format: true, comment: "" });
    await writing;
  } catch (error) {
    output.destroy(error instanceof Error ? error : new Error(String(error)));
    await writing.catch(() => {});
    throw error;
  }
}

interface PackageLocation {
  directory: string;
  archivePath?: string;
}

/** Extracted assets outlive model recovery and all in-flight panel actions. */
export class PackageArchiveStore {
  private closed = false;
  private pending = new Set<Promise<void>>();
  private roots = new Set<string>();
  private archives = new Map<string, Promise<PackageLocation>>();

  constructor(private readonly temporaryDirectory = tmpdir()) {}

  async use<T>(action: () => Promise<T>): Promise<T> {
    if (this.closed) throw new Error("The package owner has been disposed.");
    let complete!: () => void;
    const operation = new Promise<void>((resolve) => {
      complete = resolve;
    });
    this.pending.add(operation);
    try {
      return await action();
    } finally {
      this.pending.delete(operation);
      complete();
    }
  }

  resolve(path: string): Promise<PackageLocation> {
    return this.use(async () => {
      path = await realpath(path);
      const before = await lstat(path);
      if (before.isDirectory()) return { directory: path };
      if (!before.isFile())
        throw new Error("Choose a package archive or directory.");
      const key = JSON.stringify([
        path,
        before.dev,
        before.ino,
        before.size,
        before.mtimeMs,
      ]);
      let extraction = this.archives.get(key);
      if (!extraction) {
        extraction = this.extract(path, before);
        this.archives.set(key, extraction);
        void extraction.catch(() => this.archives.delete(key));
      }
      return extraction;
    });
  }

  private async extract(
    source: string,
    before: Awaited<ReturnType<typeof lstat>>,
  ): Promise<PackageLocation> {
    const root = await mkdtemp(
      join(this.temporaryDirectory, "onnx-light-viewer-package-"),
    );
    this.roots.add(root);
    try {
      await extractArchive(source, root);
      const after = await lstat(source);
      if (
        !after.isFile() ||
        before.dev !== after.dev ||
        before.ino !== after.ino ||
        before.size !== after.size ||
        before.mtimeMs !== after.mtimeMs
      )
        throw new Error(
          "The package archive changed while loading. Reopen it.",
        );
      let directory = root;
      for (;;) {
        const entries = await readdir(directory, { withFileTypes: true });
        if (entries.length !== 1 || !entries[0].isDirectory()) break;
        directory = join(directory, entries[0].name);
      }
      return { directory: resolve(directory), archivePath: source };
    } catch (error) {
      await rm(root, { recursive: true, force: true });
      this.roots.delete(root);
      throw error;
    }
  }

  async close(sessions: Promise<unknown> = Promise.resolve()): Promise<void> {
    this.closed = true;
    await Promise.allSettled([...this.pending, sessions]);
    await Promise.all(
      [...this.roots].map((root) => rm(root, { recursive: true, force: true })),
    );
    this.roots.clear();
    this.archives.clear();
  }
}
