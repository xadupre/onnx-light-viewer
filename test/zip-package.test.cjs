const assert = require("node:assert/strict");
const fs = require("node:fs/promises");
const { tmpdir } = require("node:os");
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

test("native writer uses real typed-array lengths rather than JavaScript properties", async () => {
  const input = Object.defineProperty(model.slice(), "length", {
    value: 0xffffffff,
  });
  const contents = unzipSync(await createModelPackage("model.onnx", input, []));
  assert.deepEqual(contents["model.onnx"], model);
});

test("rejects output entry counts over the package limit before processing contents", async () => {
  const weights = Array.from({ length: 0xffff - 1 }, (_, index) => ({
    name: `weights/${index}`,
    bytes: new Uint8Array(),
  }));
  await assert.rejects(
    createModelPackage("model.onnx", model, weights),
    /package limit/,
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
  "COM¹.onnx",
  "CONOUT$.onnx",
  "weights:stream",
  "bad\nname.onnx",
  "invalid?.onnx",
];

for (const name of unsafeNames) {
  test(`rejects unsafe archive and output name ${JSON.stringify(name)}`, async () => {
    await assert.rejects(
      extractModelPackage(archive({ "model.onnx": model, [name]: model })),
      /Unsafe ZIP entry name/,
    );
    await assert.rejects(
      createModelPackage("model.onnx", model, [{ name, bytes: model }]),
      /Unsafe ZIP entry name/,
    );
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
  { "model.onnx": model, "Ünicode/a": model, "ünicode/b": model },
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

test("rejects Unix symlinks and special files", async () => {
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

test("checks total declared uncompressed sizes before decompressing any entry", async () => {
  const zipped = Buffer.from(
    archive({ "model.onnx": model, "one.bin": model, "two.bin": model }),
  );
  for (const offset of centralOffsets(zipped)) {
    zipped.writeUInt16LE(8, offset + 10);
    zipped.writeUInt32LE(3 * 1024 ** 3, offset + 24);
    const local = zipped.readUInt32LE(offset + 42);
    zipped.writeUInt16LE(8, local + 8);
    zipped.writeUInt32LE(3 * 1024 ** 3, local + 22);
  }
  await assert.rejects(extractModelPackage(zipped), /package limit/);
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

test("snapshots caller-owned archive bytes across asynchronous extraction", async (t) => {
  const input = archive({ "model.onnx": model });
  const pending = extractModelPackage(input);
  input.fill(0);
  const result = await pending;
  t.after(() => result.dispose());
  assert.deepEqual(await fs.readFile(result.models[0]), Buffer.from(model));
});

function manifest(variants, extra = {}) {
  return bytes(
    JSON.stringify({
      schema_version: "1.0",
      components: { decoder: { variants } },
      ...extra,
    }),
  );
}

test("uses declared component variants, external component/executor JSON, and a wrapper root", async (t) => {
  const input = archive({
    "bundle/": new Uint8Array(),
    "bundle/manifest.json": bytes(
      JSON.stringify({
        schema_version: "1.0",
        components: {
          decoder: {
            variants: {
              cpu: {
                variant_directory: "models/cpu",
                executor_info: { ort: { model_file: "main.onnx" } },
              },
              gpu: {
                variant_directory: "models/gpu",
                executor_info: { ort: "ort.json" },
              },
              other: {
                executor_info: { other: { model_file: "ignored.onnx" } },
              },
            },
          },
          encoder: "encoder/component.json",
        },
      }),
    ),
    "bundle/models/cpu/main.onnx": model,
    "bundle/models/gpu/main.onnx": model,
    "bundle/models/gpu/ort.json": bytes('{"model_file":"main.onnx"}'),
    "bundle/encoder/component.json": bytes(
      '{"variants":{"cpu":{"variant_directory":".","executor_info":{"ort":{"model_file":"encoder.onnx"}}}}}',
    ),
    "bundle/encoder/encoder.onnx": model,
    "bundle/undeclared.onnx": model,
  });
  const result = await extractModelPackage(input);
  t.after(() => result.dispose());
  assert.deepEqual(
    result.models.map((path) => relative(result.directory, path)),
    [
      join("bundle", "encoder", "encoder.onnx"),
      join("bundle", "models", "cpu", "main.onnx"),
      join("bundle", "models", "gpu", "main.onnx"),
    ],
  );
});

test("resolves shared asset model references through the model_package public API", async (t) => {
  const uri = `sha256:${"a".repeat(64)}`;
  const input = archive({
    "manifest.json": manifest({
      cpu: { executor_info: { ort: { model_file: `${uri}/模型.onnx` } } },
    }),
    [`shared_assets/sha256-${"a".repeat(64)}/模型.onnx`]: model,
    [`shared_assets/sha256-${"a".repeat(64)}/weights.data`]: bytes("shared"),
    "undeclared.onnx": model,
  });
  const result = await extractModelPackage(input);
  t.after(() => result.dispose());
  assert.deepEqual(result.models, [
    join(
      result.directory,
      "shared_assets",
      `sha256-${"a".repeat(64)}`,
      "模型.onnx",
    ),
  ]);
  assert.equal(
    await fs.readFile(join(result.models[0], "..", "weights.data"), "utf8"),
    "shared",
  );
});

test("round-trips Unicode ZIP filenames and resolves Unicode variant directories", async (t) => {
  const input = await createModelPackage("模型/主模型.onnx", model, [
    {
      name: "manifest.json",
      bytes: manifest({
        cpu: {
          variant_directory: "模型",
          executor_info: { ort: { model_file: "主模型.onnx" } },
        },
      }),
    },
    { name: "模型/权重.data", bytes: bytes("weights") },
  ]);
  const result = await extractModelPackage(input);
  t.after(() => result.dispose());
  assert.deepEqual(result.models, [
    join(result.directory, "模型", "主模型.onnx"),
  ]);
});

for (const [description, contents] of [
  ["invalid JSON", bytes("{")],
  ["invalid schema", bytes('{"schema_version":99,"components":{}}')],
  ["missing components", bytes('{"schema_version":"1.0"}')],
  ["invalid components", bytes('{"schema_version":"1.0","components":[]}')],
  ["unknown field", manifest({}, { unexpected: true })],
  ["missing ORT executor", manifest({ cpu: {} })],
  ["missing model_file", manifest({ cpu: { executor_info: { ort: {} } } })],
  [
    "non-string model_file",
    manifest({ cpu: { executor_info: { ort: { model_file: 7 } } } }),
  ],
  [
    "incorrect model_path field",
    manifest({
      cpu: { executor_info: { ort: { model_path: "model.onnx" } } },
    }),
  ],
  [
    "malformed session_options",
    manifest({
      cpu: {
        executor_info: {
          ort: { model_file: "model.onnx", session_options: [] },
        },
      },
    }),
  ],
  [
    "missing declared file",
    manifest({
      cpu: { executor_info: { ort: { model_file: "missing.onnx" } } },
    }),
  ],
  [
    "missing external executor",
    manifest({ cpu: { executor_info: { ort: "missing.json" } } }),
  ],
  ["installed layout", manifest({}, { layout: "installed" })],
  [
    "parent path",
    manifest({
      cpu: { executor_info: { ort: { model_file: "../model.onnx" } } },
    }),
  ],
  [
    "absolute path",
    manifest({
      cpu: { executor_info: { ort: { model_file: "/model.onnx" } } },
    }),
  ],
  [
    "NUL path",
    manifest({
      cpu: { executor_info: { ort: { model_file: "model.onnx\0other" } } },
    }),
  ],
]) {
  test(`manifest rejects ${description} instead of falling back to ZIP discovery`, async () => {
    await assert.rejects(
      extractModelPackage(
        archive({ "manifest.json": contents, "model.onnx": model }),
      ),
      /model package|executor_info|model_file|portable layout/i,
    );
  });
}

test("rejects malformed external executor JSON and traversal within a shared asset", async () => {
  await assert.rejects(
    extractModelPackage(
      archive({
        "manifest.json": manifest({
          cpu: { variant_directory: ".", executor_info: { ort: "ort.json" } },
        }),
        "ort.json": bytes("{bad"),
        "model.onnx": model,
      }),
    ),
    /Invalid model package/,
  );
  const uri = `sha256:${"a".repeat(64)}`;
  await assert.rejects(
    extractModelPackage(
      archive({
        "manifest.json": manifest({
          cpu: {
            executor_info: { ort: { model_file: `${uri}/../model.onnx` } },
          },
        }),
        [`shared_assets/sha256-${"a".repeat(64)}/model.onnx`]: model,
      }),
    ),
    /Invalid model package/,
  );
});

test("cleans up real private directories after extraction and manifest failures", async () => {
  const directory = await fs.mkdtemp(join(tmpdir(), "zip-cleanup-test-"));
  const variables = ["TMPDIR", "TMP", "TEMP"];
  const previous = variables.map((name) => process.env[name]);
  try {
    for (const name of variables) process.env[name] = directory;
    const corrupt = Buffer.from(
      archive({ "nested/model.onnx": model }, { level: 0 }),
    );
    corrupt[30 + Buffer.byteLength("nested/model.onnx")] ^= 0xff;
    await assert.rejects(extractModelPackage(corrupt), /Corrupt ZIP entry/);
    assert.deepEqual(await fs.readdir(directory), []);
    await assert.rejects(
      extractModelPackage(
        archive({ "manifest.json": bytes("{"), "model.onnx": model }),
      ),
      /Invalid model package/,
    );
    assert.deepEqual(await fs.readdir(directory), []);
  } finally {
    variables.forEach((name, index) => {
      if (previous[index] === undefined) delete process.env[name];
      else process.env[name] = previous[index];
    });
    await fs.rm(directory, { recursive: true, force: true });
  }
});

test("explicitly rejects external initializer folder overrides rather than misloading weights", async () => {
  const key = "session.model_external_initializers_file_folder_path";
  for (const reference of ["weights", `sha256:${"a".repeat(64)}`, ""]) {
    for (const external of [false, true]) {
      const executor = {
        model_file: "model.onnx",
        session_options: { [key]: reference },
      };
      await assert.rejects(
        extractModelPackage(
          archive({
            "manifest.json": manifest({
              cpu: {
                variant_directory: ".",
                executor_info: { ort: external ? "ort.json" : executor },
              },
            }),
            "ort.json": bytes(JSON.stringify(executor)),
            "model.onnx": model,
            "weights/data.bin": bytes("weights"),
          }),
        ),
        /Unsupported model package session option: session\.model_external_initializers_file_folder_path/,
      );
    }
  }
});

test("validates empty-entry local headers and CRCs that miniz skips", async () => {
  for (const directory of [false, true]) {
    const name = directory ? "empty/" : "empty";
    const data = Buffer.from(
      archive({ "model.onnx": model, [name]: new Uint8Array() }),
    );
    const offset = centralOffsets(data)[1];
    const local = data.readUInt32LE(offset + 42);
    const invalidHeader = Buffer.from(data);
    invalidHeader[local] ^= 0xff;
    await assert.rejects(extractModelPackage(invalidHeader), /Invalid ZIP/);
    const invalidSize = Buffer.from(data);
    invalidSize.writeUInt32LE(1, local + 22);
    await assert.rejects(extractModelPackage(invalidSize), /Invalid ZIP/);
    data.writeUInt32LE(1, offset + 16);
    data.writeUInt32LE(1, local + 14);
    await assert.rejects(extractModelPackage(data), /Corrupt ZIP/);
  }
});

test("native writer snapshots caller-owned model and weights before queuing work", async () => {
  const input = model.slice();
  const weights = bytes("weights");
  const pending = createModelPackage("model.onnx", input, [
    { name: "weights.data", bytes: weights },
  ]);
  input.fill(0);
  weights.fill(0);
  const contents = unzipSync(await pending);
  assert.deepEqual(contents["model.onnx"], model);
  assert.deepEqual(contents["weights.data"], bytes("weights"));
});

test("rejects wrapper-root ambiguity rather than discovering undeclared files", async () => {
  await assert.rejects(
    extractModelPackage(
      archive({
        "bundle/manifest.json": manifest({
          cpu: { executor_info: { ort: { model_file: "model.onnx" } } },
        }),
        "bundle/model.onnx": model,
        "outside.onnx": model,
      }),
    ),
    /Ambiguous model package root/,
  );
});
