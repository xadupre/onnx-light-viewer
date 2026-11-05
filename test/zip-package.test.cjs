const assert = require("node:assert/strict");
const fs = require("node:fs/promises");
const { join, isAbsolute, relative } = require("node:path");
const { test } = require("node:test");
const { strToU8, unzipSync, zipSync } = require("fflate");
const {
  createModelPackage,
  extractModelPackage,
} = require("../dist/zipPackage");

const bytes = (text) => strToU8(text);
const model = bytes("model bytes");

function archive(entries, options = {}) {
  return zipSync(Object.assign(Object.create(null), entries), options);
}

function centralOffsets(data) {
  const buffer = Buffer.from(data.buffer, data.byteOffset, data.byteLength);
  const end = buffer.length - 22;
  const count = buffer.readUInt16LE(end + 10);
  let offset = buffer.readUInt32LE(end + 16);
  const result = [];
  for (let i = 0; i < count; i++) {
    result.push(offset);
    offset +=
      46 +
      buffer.readUInt16LE(offset + 28) +
      buffer.readUInt16LE(offset + 30) +
      buffer.readUInt16LE(offset + 32);
  }
  return result;
}

test("extracts every asset, empty directory, and nested model into a private disposable directory", async (t) => {
  const input = archive({
    "models/model.onnx": model,
    "models/model.onnx.data": bytes("external weights"),
    "encrypted.onnx.enc": bytes("encrypted model"),
    "pipeline.pipe-onnx": bytes("pipeline"),
    "tokenizer/config.json": bytes('{"tokenizer":true}'),
    "empty/": new Uint8Array(),
    "ignored.onnx.txt": bytes("not a model"),
  });
  const result = await extractModelPackage(input);
  t.after(() => result.dispose());
  assert.ok(isAbsolute(result.directory));
  assert.deepEqual(
    result.models.map((path) => relative(result.directory, path)),
    ["encrypted.onnx.enc", "models/model.onnx", "pipeline.pipe-onnx"],
  );
  assert.ok(result.models.every(isAbsolute));
  assert.deepEqual(await fs.readFile(result.models[1]), Buffer.from(model));
  assert.equal(
    await fs.readFile(join(result.directory, "models/model.onnx.data"), "utf8"),
    "external weights",
  );
  assert.equal(
    await fs.readFile(join(result.directory, "tokenizer/config.json"), "utf8"),
    '{"tokenizer":true}',
  );
  assert.ok((await fs.stat(join(result.directory, "empty"))).isDirectory());
  if (process.platform !== "win32") {
    assert.equal((await fs.stat(result.directory)).mode & 0o777, 0o700);
    assert.equal((await fs.stat(result.models[1])).mode & 0o777, 0o600);
  }
  await result.dispose();
  await result.dispose();
  await assert.rejects(fs.stat(result.directory), { code: "ENOENT" });
});

test("round-trips model and external weights without flattening their paths", async (t) => {
  const weights = [
    { name: "weights/data.bin", bytes: bytes("weights") },
    { name: "config/tokenizer.json", bytes: bytes("{}") },
  ];
  const zipped = await createModelPackage("models/main.onnx", model, weights);
  const contents = unzipSync(zipped);
  assert.deepEqual(contents["models/main.onnx"], model);
  for (const weight of weights) {
    assert.deepEqual(contents[weight.name], weight.bytes);
  }
  const result = await extractModelPackage(zipped);
  t.after(() => result.dispose());
  assert.deepEqual(result.models, [join(result.directory, "models/main.onnx")]);
  assert.equal(
    await fs.readFile(join(result.directory, "weights/data.bin"), "utf8"),
    "weights",
  );
});

test("round-trips assets large enough to use asynchronous ZIP workers", async (t) => {
  const weights = new Uint8Array(1024 * 1024).fill(42);
  const zipped = await createModelPackage("model.onnx", model, [
    { name: "weights/data", bytes: weights },
  ]);
  const result = await extractModelPackage(zipped);
  t.after(() => result.dispose());
  assert.deepEqual(
    await fs.readFile(join(result.directory, "weights/data")),
    Buffer.from(weights),
  );
});

function declaredBytes(length) {
  return Object.defineProperty(new Uint8Array(), "length", { value: length });
}

test("rejects output entries whose sizes require ZIP64 before compression", async () => {
  for (const size of [0xffffffff, 0x100000000]) {
    await assert.rejects(
      createModelPackage("model.onnx", declaredBytes(size), []),
      /ZIP64 packages is unsupported/,
    );
  }
});

test("rejects combined output requiring ZIP64 even with individually valid sizes", async () => {
  await assert.rejects(
    createModelPackage("model.onnx", declaredBytes(2 ** 31), [
      { name: "weights/data", bytes: declaredBytes(2 ** 31) },
    ]),
    /ZIP64 packages is unsupported/,
  );
});

test("accounts for ZIP headers, central directory, and worst-case compression expansion", async () => {
  const nameSize = Buffer.byteLength("model.onnx");
  const sizeBeforeCentralDirectory = Math.floor(
    ((0xffffffff - 22 - 30 - nameSize - 11) * 7000) / 7005,
  );
  const compressedBound =
    sizeBeforeCentralDirectory +
    5 * (1 + Math.ceil(sizeBeforeCentralDirectory / 7000));
  assert.ok(compressedBound + 22 + 30 + nameSize < 0xffffffff);
  assert.ok(compressedBound + 22 + 76 + 2 * nameSize >= 0xffffffff);
  for (const size of [sizeBeforeCentralDirectory, 0xffffffff - 1024]) {
    await assert.rejects(
      createModelPackage("model.onnx", declaredBytes(size), []),
      /ZIP64 packages is unsupported/,
    );
  }
});

test("rejects output entry counts requiring ZIP64 before processing contents", async () => {
  const weights = Array.from({ length: 0xffff - 1 }, (_, index) => ({
    name: `weights/${index}`,
    bytes: new Uint8Array(),
  }));
  await assert.rejects(
    createModelPackage("model.onnx", model, weights),
    /ZIP64 packages is unsupported/,
  );
});

const unsafeNames = [
  "../escape.onnx",
  "/absolute.onnx",
  "C:/drive.onnx",
  "C:drive.onnx",
  "\\\\server\\model.onnx",
  "folder\\model.onnx",
  "a/../model.onnx",
  "a/./model.onnx",
  "a//model.onnx",
  "bad\0.onnx",
  "a./model.onnx",
  "a /model.onnx",
  "NUL.onnx",
  "weights:stream",
];

for (const name of unsafeNames) {
  test(`rejects unsafe archive and output name ${JSON.stringify(name)} before creating a directory`, async (t) => {
    const makeDirectory = t.mock.method(fs, "mkdtemp", () => {
      throw new Error("Should not create a directory");
    });
    await assert.rejects(
      extractModelPackage(archive({ "model.onnx": model, [name]: model })),
      /Unsafe ZIP entry name/,
    );
    await assert.rejects(
      createModelPackage("model.onnx", model, [{ name, bytes: model }]),
      /Unsafe ZIP entry name/,
    );
    assert.equal(makeDirectory.mock.callCount(), 0);
  });
}

test("rejects empty names, directory-valued output names and missing model extensions", async () => {
  await assert.rejects(
    createModelPackage("model.onnx", model, [{ name: "", bytes: model }]),
    /Unsafe ZIP entry name/,
  );
  await assert.rejects(
    createModelPackage("model.onnx", model, [
      { name: "assets/", bytes: model },
    ]),
    /Unsafe ZIP entry name/,
  );
  await assert.rejects(
    createModelPackage("model.bin", model, []),
    /model filename/,
  );
});

test("rejects duplicate central-directory paths instead of silently overwriting", async () => {
  const zipped = Buffer.from(
    archive({ "model.onnx": model, "other.onnx": model }),
  );
  let offset = zipped.indexOf("other.onnx");
  while (offset !== -1) {
    zipped.write("model.onnx", offset);
    offset = zipped.indexOf("other.onnx", offset + 10);
  }
  await assert.rejects(extractModelPackage(zipped), /Duplicate ZIP entry/);
  await assert.rejects(
    createModelPackage("model.onnx", model, [
      { name: "model.onnx", bytes: model },
    ]),
    /Duplicate ZIP entry/,
  );
});

for (const entries of [
  { "model.onnx": model, asset: model, "asset/data": model },
  { "model.onnx": model, "asset/data": model, asset: model },
  { "model.onnx": model, "asset/": new Uint8Array(), asset: model },
  { "model.onnx": model, "MODEL.ONNX": model },
  { "model.onnx": model, "Assets/a": model, "assets/b": model },
]) {
  test(`rejects file/directory and portable case conflicts: ${Object.keys(entries).join(", ")}`, async () => {
    await assert.rejects(
      extractModelPackage(archive(entries)),
      /Duplicate ZIP entry|Conflicting ZIP entry/,
    );
    const weights = Object.entries(entries)
      .filter(([name]) => name !== "model.onnx")
      .map(([name, data]) => ({ name, bytes: data }));
    await assert.rejects(
      createModelPackage("model.onnx", model, weights),
      /Duplicate ZIP entry|Conflicting ZIP entry|Unsafe ZIP entry/,
    );
  });
}

test("rejects Unix symlinks and special files without writing anything", async (t) => {
  const makeDirectory = t.mock.method(fs, "mkdtemp", () => {
    throw new Error("Should not create a directory");
  });
  for (const mode of [0o120777, 0o020600, 0o010600]) {
    const zipped = archive({
      "model.onnx": model,
      weights: [bytes("../outside"), { os: 3, attrs: mode << 16 }],
    });
    await assert.rejects(
      extractModelPackage(zipped),
      /Unsupported ZIP entry type/,
    );
  }
  assert.equal(makeDirectory.mock.callCount(), 0);
});

test("rejects empty archives and packages without models", async () => {
  await assert.rejects(extractModelPackage(archive({})), /no ONNX models/);
  await assert.rejects(
    extractModelPackage(
      archive({ "weights.bin": model, "fake.onnx/": new Uint8Array() }),
    ),
    /no ONNX models/,
  );
});

test("rejects malformed, truncated and corrupt ZIP data", async () => {
  const valid = archive({ "model.onnx": model }, { level: 0 });
  for (const input of [
    new Uint8Array(),
    bytes("not a ZIP archive"),
    valid.subarray(0, valid.length - 1),
  ]) {
    await assert.rejects(extractModelPackage(input), /Invalid ZIP/);
  }
  const corrupt = Buffer.from(valid);
  corrupt[30 + Buffer.byteLength("model.onnx")] ^= 0xff;
  await assert.rejects(extractModelPackage(corrupt), /Corrupt ZIP entry/);
  const invalidDirectory = Buffer.from(valid);
  invalidDirectory[centralOffsets(invalidDirectory)[0]] ^= 0xff;
  await assert.rejects(extractModelPackage(invalidDirectory), /Invalid ZIP/);
});

test("rejects unsupported compression, encrypted entries and inconsistent stored sizes", async () => {
  for (const [field, value, message] of [
    [10, 99, /compression/],
    [8, 1, /Encrypted ZIP/],
    [24, model.length + 1, /size/],
  ]) {
    const zipped = Buffer.from(archive({ "model.onnx": model }, { level: 0 }));
    const offset = centralOffsets(zipped)[0];
    if (field === 24) zipped.writeUInt32LE(value, offset + field);
    else zipped.writeUInt16LE(value, offset + field);
    await assert.rejects(extractModelPackage(zipped), message);
  }
});

test("checks total declared uncompressed sizes before decompressing any entry", async (t) => {
  const makeDirectory = t.mock.method(fs, "mkdtemp", () => {
    throw new Error("Should not create a directory");
  });
  const zipped = Buffer.from(
    archive({ "model.onnx": model, "one.bin": model, "two.bin": model }),
  );
  for (const offset of centralOffsets(zipped)) {
    zipped.writeUInt16LE(8, offset + 10);
    zipped.writeUInt32LE(3 * 1024 ** 3, offset + 24);
  }
  await assert.rejects(extractModelPackage(zipped), /package limit/);
  assert.equal(makeDirectory.mock.callCount(), 0);
});

test("bounds entry count before decompression", async () => {
  const entries = Object.create(null);
  entries["model.onnx"] = model;
  for (let i = 0; i < 10_000; i++) entries[`asset-${i}`] = new Uint8Array();
  await assert.rejects(
    extractModelPackage(archive(entries, { level: 0 })),
    /too many ZIP entries/,
  );
});

test("cleans up the entire private directory when a write fails", async (t) => {
  const originalMkdtemp = fs.mkdtemp;
  let created;
  t.mock.method(fs, "mkdtemp", async (...args) => {
    created = await originalMkdtemp(...args);
    return created;
  });
  t.mock.method(fs, "writeFile", async () => {
    throw new Error("simulated write failure");
  });
  await assert.rejects(
    extractModelPackage(archive({ "nested/model.onnx": model })),
    /simulated write failure/,
  );
  assert.ok(created);
  await assert.rejects(fs.stat(created), { code: "ENOENT" });
});

test("snapshots caller-owned archive bytes across asynchronous extraction", async (t) => {
  const input = archive({ "model.onnx": model });
  const pending = extractModelPackage(input);
  input.fill(0);
  const result = await pending;
  t.after(() => result.dispose());
  assert.deepEqual(await fs.readFile(result.models[0]), Buffer.from(model));
});
