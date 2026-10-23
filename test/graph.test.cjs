const assert = require("node:assert/strict");
const { execFileSync } = require("node:child_process");
const { createCipheriv, createDecipheriv, pbkdf2Sync } = require("node:crypto");
const {
  readFileSync,
  mkdtempSync,
  openSync,
  writeSync,
  ftruncateSync,
  closeSync,
  rmSync,
  writeFileSync,
  renameSync,
} = require("node:fs");
const { tmpdir } = require("node:os");
const path = require("node:path");
const { test } = require("node:test");
const { withCommaLocale } = require("../build/native/Release/test_locale.node");
const {
  decodeGraph,
  demoGraph,
  loadModel,
  loadModelFile,
  listBackendTests,
  loadBackendTest,
  isEncryptedModel,
} = require("../dist/graph");

function modelBytes(name) {
  return readFileSync(path.join(__dirname, "models", name));
}

function varint(value) {
  const bytes = [];
  do {
    bytes.push((value % 128) | (value >= 128 ? 128 : 0));
    value = Math.floor(value / 128);
  } while (value);
  return Buffer.from(bytes);
}

function varintField(number, value) {
  return Buffer.concat([varint(number * 8), varint(value)]);
}

function bytesField(number, value) {
  const bytes = Buffer.isBuffer(value) ? value : Buffer.from(value);
  return Buffer.concat([varint(number * 8 + 2), varint(bytes.length), bytes]);
}

function matmulWithWeight(dataField, data = Buffer.alloc(256)) {
  const dimension = (size) =>
    bytesField(
      1,
      typeof size === "string" ? bytesField(2, size) : varintField(1, size),
    );
  const valueInfo = (name) =>
    Buffer.concat([
      bytesField(1, name),
      bytesField(
        2,
        bytesField(
          1,
          Buffer.concat([
            varintField(1, 1),
            bytesField(2, Buffer.concat([dimension("N"), dimension(8)])),
          ]),
        ),
      ),
    ]);
  const weight = Buffer.concat([
    varintField(1, 8),
    varintField(1, 8),
    varintField(2, 1),
    bytesField(8, "weight"),
    bytesField(dataField, data),
  ]);
  const node = Buffer.concat([
    bytesField(1, "x"),
    bytesField(1, "weight"),
    bytesField(2, "y"),
    bytesField(4, "MatMul"),
  ]);
  const graph = Buffer.concat([
    bytesField(1, node),
    bytesField(2, "analysis after loading"),
    bytesField(5, weight),
    bytesField(11, valueInfo("x")),
    bytesField(12, valueInfo("y")),
  ]);
  return Buffer.concat([
    varintField(1, 8),
    bytesField(8, varintField(2, 13)),
    bytesField(7, graph),
  ]);
}

function encryptModelBytes(bytes, password, version) {
  const magic = Buffer.from(`ONNXCRY${version}`);
  const salt = Buffer.alloc(16, 17);
  const nonce = Buffer.alloc(version === 1 ? 16 : 12, 23);
  const key = pbkdf2Sync(password, salt, 100000, 32, "sha256");
  const algorithm = ["aes-256-cbc", "chacha20-poly1305", "aes-256-gcm"][
    version - 1
  ];
  const cipher = createCipheriv(algorithm, key, nonce, { authTagLength: 16 });
  const header = Buffer.concat([magic, salt, nonce]);
  if (version === 3) cipher.setAAD(header);
  const payload = Buffer.concat([cipher.update(bytes), cipher.final()]);
  return Buffer.concat([
    header,
    ...(version === 1 ? [] : [cipher.getAuthTag()]),
    payload,
  ]);
}

test("loads all three encrypted ModelProto formats from buffers and files", async (t) => {
  const directory = mkdtempSync(
    path.join(tmpdir(), "onnx-light-viewer-crypt-"),
  );
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  const password = "model round-trip passphrase";
  const bytes = modelBytes("inference.onnx");
  const expected = loadModel(bytes, "inference.onnx").graph(true);
  for (const version of [1, 2, 3]) {
    const encrypted = encryptModelBytes(bytes, password, version);
    assert.equal(isEncryptedModel(encrypted), true);
    const file = path.join(directory, `模型-${version}.onnx.enc`);
    writeFileSync(file, encrypted);
    for (const model of [
      loadModel(encrypted, "inference.onnx", password),
      await loadModelFile(file, "inference.onnx", password),
    ]) {
      assert.deepEqual(model.graph(true), expected);
      assert.deepEqual(
        loadModel(await model.serialize(), "inference.onnx").graph(),
        expected,
      );
    }
    assert.throws(() => loadModel(encrypted, "encrypted", "wrong password"));
    await assert.rejects(loadModelFile(file, "encrypted", "wrong password"));
    if (version !== 1) {
      const corrupted = Buffer.from(encrypted);
      corrupted[corrupted.length - 1] ^= 1;
      assert.throws(() => loadModel(corrupted, "corrupted", password));
    }
  }
});

test("saves complete streamed weights, inferred shapes and authenticated encrypted copies", async (t) => {
  const directory = mkdtempSync(path.join(tmpdir(), "onnx-light-viewer-save-"));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  for (const field of [4, 9]) {
    const data = Buffer.alloc(256, 63);
    const bytes = matmulWithWeight(field, data);
    const file = path.join(directory, `weights-${field}.onnx`);
    writeFileSync(file, bytes);
    const streamed = await loadModelFile(file, "weights.onnx");
    const complete = loadModel(bytes, "weights.onnx");
    streamed.graph(true);
    complete.graph(true);
    const expected = Buffer.from(await complete.serialize());
    assert.equal(expected.includes(data), true);
    assert.deepEqual(Buffer.from(await streamed.serialize()), expected);
    const password = "export passphrase";
    const encrypted = Buffer.from(await streamed.serialize(password));
    assert.equal(encrypted.subarray(0, 8).toString(), "ONNXCRY3");
    const decipher = createDecipheriv(
      "aes-256-gcm",
      pbkdf2Sync(password, encrypted.subarray(8, 24), 100000, 32, "sha256"),
      encrypted.subarray(24, 36),
    );
    decipher.setAAD(encrypted.subarray(0, 36));
    decipher.setAuthTag(encrypted.subarray(36, 52));
    assert.deepEqual(
      Buffer.concat([
        decipher.update(encrypted.subarray(52)),
        decipher.final(),
      ]),
      expected,
    );
    assert.deepEqual(
      loadModel(encrypted, "weights.onnx", password).graph(),
      complete.graph(),
    );
    assert.notDeepEqual(
      Buffer.from(await streamed.serialize(password)),
      encrypted,
    );
    assert.deepEqual(readFileSync(file), bytes);
    const moved = path.join(directory, `moved-${field}.onnx`);
    renameSync(file, moved);
    assert.deepEqual(
      Buffer.from(await streamed.serialize(undefined, moved)),
      expected,
    );
    writeFileSync(moved, Buffer.concat([bytes, Buffer.from([0])]));
    await assert.rejects(
      streamed.serialize(undefined, moved),
      /source model changed/,
    );
  }
  const external = loadModel(
    modelBytes("initializers.onnx"),
    "initializers.onnx",
  );
  await assert.rejects(
    external.serialize(),
    /External weights require a local source model path/,
  );
});

test("embeds external weights with offsets and repartitions plain external copies without modifying sources", async (t) => {
  const directory = mkdtempSync(
    path.join(tmpdir(), "onnx-light-viewer-external-"),
  );
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  const data = [Buffer.alloc(128, 37), Buffer.alloc(128, 53)];
  const entry = (key, value) =>
    bytesField(13, Buffer.concat([bytesField(1, key), bytesField(2, value)]));
  const modelBytes = (external) =>
    Buffer.concat([
      varintField(1, 8),
      bytesField(8, varintField(2, 13)),
      bytesField(
        7,
        Buffer.concat([
          bytesField(2, "external weights"),
          ...data.map((bytes, index) =>
            bytesField(
              5,
              Buffer.concat([
                varintField(1, 32),
                varintField(2, 1),
                bytesField(8, `weight${index}`),
                ...(external
                  ? [
                      entry("location", "source.data"),
                      entry("offset", String(17 + index * bytes.length)),
                      entry("length", String(bytes.length)),
                      varintField(14, 1),
                    ]
                  : [bytesField(9, bytes)]),
              ]),
            ),
          ),
        ]),
      ),
    ]);
  const source = path.join(directory, "source.onnx");
  const weights = path.join(directory, "source.data");
  const originalWeights = Buffer.concat([Buffer.alloc(17), ...data]);
  writeFileSync(source, modelBytes(true));
  writeFileSync(weights, originalWeights);
  const model = await loadModelFile(source, "external");
  const expected = Buffer.from(
    await loadModel(modelBytes(false), "external").serialize(),
  );
  assert.deepEqual(Buffer.from(await model.serialize()), expected);
  const encrypted = await model.serialize("external passphrase");
  assert.deepEqual(
    Buffer.from(
      await loadModel(encrypted, "external", "external passphrase").serialize(),
    ),
    expected,
  );
  const files = await model.serializeFiles(128, "copy.data");
  assert.equal(files.weights.length, 2);
  for (const { name, bytes } of files.weights) {
    assert.equal(name.startsWith("copy.data"), true);
    assert.ok(bytes.byteLength <= 128);
    writeFileSync(path.join(directory, name), bytes);
  }
  const copy = path.join(directory, "copy.onnx");
  writeFileSync(copy, files.model);
  assert.deepEqual(
    Buffer.from(await (await loadModelFile(copy, "external")).serialize()),
    expected,
  );
  await assert.rejects(
    model.serializeFiles(127, "too-small.data"),
    /size|tensor|external/i,
  );
  assert.deepEqual(readFileSync(source), modelBytes(true));
  assert.deepEqual(readFileSync(weights), originalWeights);
  renameSync(weights, `${weights}.missing`);
  await assert.rejects(model.serialize());
});

test("external export validates chunk sizes and filenames and preserves the retained inline model", async () => {
  for (const field of [4, 9]) {
    const model = loadModel(
      matmulWithWeight(field, Buffer.alloc(256, 37)),
      "weights",
    );
    const expected = Buffer.from(await model.serialize());
    const files = await model.serializeFiles(256, "weights.data");
    assert.equal(files.weights.length, 1);
    assert.equal(files.weights[0].bytes.byteLength, 256);
    assert.deepEqual(
      Buffer.from(files.weights[0].bytes),
      Buffer.alloc(256, 37),
    );
    assert.deepEqual(Buffer.from(await model.serialize()), expected);
    for (const size of [0, -1, 1.5, NaN, Infinity, Number.MAX_SAFE_INTEGER + 1])
      assert.throws(
        () => model.serializeFiles(size, "weights"),
        /integer|size/,
      );
    for (const prefix of [
      "",
      ".",
      "..",
      "../weights",
      "dir\\weights",
      "weights\0",
    ])
      assert.throws(() => model.serializeFiles(256, prefix), /prefix/);
  }
});

test("embeds external sparse values and indices before encrypting the whole model", async (t) => {
  const directory = mkdtempSync(
    path.join(tmpdir(), "onnx-light-viewer-sparse-save-"),
  );
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  const values = Buffer.alloc(64, 37);
  const indices = Buffer.alloc(128);
  for (let index = 0; index < 16; index++)
    indices.writeBigInt64LE(BigInt(index), index * 8);
  const tensor = (name, type, bytes, offset, external) =>
    Buffer.concat([
      varintField(1, 16),
      varintField(2, type),
      bytesField(8, name),
      ...(external
        ? [
            ...Object.entries({
              location: "sparse.data",
              offset: String(offset),
              length: String(bytes.length),
            }).map(([key, value]) =>
              bytesField(
                13,
                Buffer.concat([bytesField(1, key), bytesField(2, value)]),
              ),
            ),
            varintField(14, 1),
          ]
        : [bytesField(9, bytes)]),
    ]);
  const modelBytes = (external) =>
    Buffer.concat([
      varintField(1, 8),
      bytesField(
        7,
        Buffer.concat([
          bytesField(2, "sparse weights"),
          bytesField(
            15,
            Buffer.concat([
              bytesField(1, tensor("sparse", 1, values, 0, external)),
              bytesField(
                2,
                tensor("indices", 7, indices, values.length, external),
              ),
              varintField(3, 128),
            ]),
          ),
        ]),
      ),
    ]);
  const source = path.join(directory, "sparse.onnx");
  writeFileSync(source, modelBytes(true));
  writeFileSync(
    path.join(directory, "sparse.data"),
    Buffer.concat([values, indices]),
  );
  const model = await loadModelFile(source, "sparse");
  const expected = Buffer.from(
    await loadModel(modelBytes(false), "sparse").serialize(),
  );
  assert.deepEqual(Buffer.from(await model.serialize()), expected);
  const encrypted = await model.serialize("sparse password");
  assert.deepEqual(
    Buffer.from(
      await loadModel(encrypted, "sparse", "sparse password").serialize(),
    ),
    expected,
  );
});

test("streams local models, retains small shape constants and never opens external weights", async (t) => {
  for (const name of [
    "inference.onnx",
    "initializers.onnx",
    "subgraphs.onnx",
  ]) {
    const file = path.join(__dirname, "models", name);
    const model = await loadModelFile(file, name);
    assert.deepEqual(model.graph(), loadModel(modelBytes(name), name).graph());
    if (name === "inference.onnx") {
      const inferred = model.graph(true);
      assert.deepEqual(
        inferred.tensors.find(({ name }) => name === "reshaped").shape,
        ["N", "3", "1"],
      );
      assert.deepEqual(model.graph(), inferred);
    }
  }
  await assert.rejects(
    loadModelFile(path.join(__dirname, "models", "absent.onnx"), "absent"),
  );
  await assert.rejects(
    loadModelFile(path.join(__dirname, "models", "missing-ir.onnx"), "invalid"),
    /ONNX model graph/,
  );
  assert.throws(() => loadModelFile(null, "invalid"), TypeError);
  assert.throws(
    () => loadModelFile("invalid\0path", "invalid"),
    /Invalid model file path/,
  );
  const directory = mkdtempSync(path.join(tmpdir(), "onnx-light-viewer-模型-"));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  const unicodePath = path.join(directory, "модель.onnx");
  writeFileSync(unicodePath, modelBytes("relu.onnx"));
  assert.equal(
    (await loadModelFile(unicodePath, "unicode.onnx")).graph().nodes[0].opType,
    "Relu",
  );
});

test("skips a large embedded weight without allocating its payload or blocking the host", async (t) => {
  const directory = mkdtempSync(path.join(tmpdir(), "onnx-light-viewer-"));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  const file = path.join(directory, "large.onnx");
  const varint = (value) => {
    const bytes = [];
    do {
      bytes.push((value % 128) | (value >= 128 ? 128 : 0));
      value = Math.floor(value / 128);
    } while (value);
    return Buffer.from(bytes);
  };
  // Model -> graph -> initializer -> raw_data, ending in a sparse file hole.
  const size = 256 * 1024 * 1024;
  const tensorHeader = Buffer.concat([
    Buffer.from([0x08]),
    varint(size), // dims
    Buffer.from([0x10, 0x02, 0x42, 0x01, 0x77, 0x4a]),
    varint(size),
  ]);
  const graphHeader = Buffer.concat([
    Buffer.from([0x12, 0x05]),
    Buffer.from("large"),
    Buffer.from([0x2a]),
    varint(tensorHeader.length + size),
    tensorHeader,
  ]);
  const header = Buffer.concat([
    Buffer.from([0x08, 0x08, 0x3a]),
    varint(graphHeader.length + size),
    graphHeader,
  ]);
  const fd = openSync(file, "w");
  try {
    writeSync(fd, header);
    ftruncateSync(fd, header.length + size);
  } finally {
    closeSync(fd);
  }
  const rss = process.memoryUsage().rss;
  let settled = false;
  const pending = loadModelFile(file, "large.onnx").then((model) => {
    settled = true;
    return model;
  });
  await Promise.resolve();
  assert.equal(settled, false);
  const model = await pending;
  assert.equal(model.graph().name, "large");
  assert.equal(model.graph().initializerCount, 1);
  assert.deepEqual(model.graph().tensors, [
    { name: "w", type: "UINT8", shape: [String(size)] },
  ]);
  assert.ok(process.memoryUsage().rss - rss < 64 * 1024 * 1024);
});

test("skips a large packed float initializer without allocating its payload", async (t) => {
  const directory = mkdtempSync(path.join(tmpdir(), "onnx-light-viewer-"));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  const file = path.join(directory, "large-packed.onnx");
  const size = 256 * 1024 * 1024;
  const tensorHeader = Buffer.concat([
    Buffer.from([0x08]),
    varint(size / 4),
    Buffer.from([0x10, 0x01, 0x42, 0x01, 0x77, 0x22]),
    varint(size),
  ]);
  const graphHeader = Buffer.concat([
    Buffer.from([0x12, 0x05]),
    Buffer.from("large"),
    Buffer.from([0x2a]),
    varint(tensorHeader.length + size),
    tensorHeader,
  ]);
  const header = Buffer.concat([
    Buffer.from([0x08, 0x08, 0x3a]),
    varint(graphHeader.length + size),
    graphHeader,
  ]);
  const fd = openSync(file, "w");
  try {
    writeSync(fd, header);
    ftruncateSync(fd, header.length + size);
  } finally {
    closeSync(fd);
  }
  const rss = process.memoryUsage().rss;
  const model = await loadModelFile(file, "large-packed.onnx");
  assert.equal(model.graph().initializerCount, 1);
  assert.deepEqual(model.graph().tensors, [
    { name: "w", type: "FLOAT", shape: [String(size / 4)] },
  ]);
  assert.ok(process.memoryUsage().rss - rss < 64 * 1024 * 1024);
});

test("rejects duplicate graphs before applying skipped tensor paths", async (t) => {
  const directory = mkdtempSync(path.join(tmpdir(), "onnx-light-viewer-"));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  const file = path.join(directory, "duplicate-graph.onnx");
  const size = 256 * 1024 * 1024;
  const tensorHeader = Buffer.concat([
    Buffer.from([0x08]),
    varint(size),
    Buffer.from([0x10, 0x02, 0x42, 0x01, 0x77, 0x4a]),
    varint(size),
  ]);
  const graphHeader = Buffer.concat([
    Buffer.from([0x2a]),
    varint(tensorHeader.length + size),
    tensorHeader,
  ]);
  const prefix = Buffer.concat([
    varintField(1, 8),
    bytesField(8, varintField(2, 13)),
    Buffer.from([0x3a]),
    varint(graphHeader.length + size),
    graphHeader,
  ]);
  const duplicate = bytesField(7, Buffer.alloc(0));
  const fd = openSync(file, "w");
  try {
    writeSync(fd, prefix);
    ftruncateSync(fd, prefix.length + size + duplicate.length);
    writeSync(fd, duplicate, 0, duplicate.length, prefix.length + size);
  } finally {
    closeSync(fd);
  }
  await assert.rejects(
    loadModelFile(file, "duplicate-graph.onnx"),
    /Duplicate singular protobuf message field/,
  );
});

test("isolates untracked messages before a Constant with an omitted tensor payload", async (t) => {
  const directory = mkdtempSync(path.join(tmpdir(), "onnx-light-viewer-"));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  const file = path.join(directory, "constant.onnx");
  const size = 256 * 1024 * 1024;
  const valueInfo = (name) =>
    Buffer.concat([
      bytesField(1, name),
      bytesField(
        2,
        bytesField(
          1,
          Buffer.concat([
            varintField(1, 1),
            bytesField(2, bytesField(1, varintField(1, 1))),
          ]),
        ),
      ),
    ]);
  const tensorHeader = Buffer.concat([
    varintField(1, size / 4),
    varintField(2, 1),
    bytesField(8, "constant"),
    varint(9 * 8 + 2),
    varint(size),
  ]);
  const attributeHeader = Buffer.concat([
    bytesField(1, "value"),
    varintField(20, 4),
    varint(5 * 8 + 2),
    varint(tensorHeader.length + size),
    tensorHeader,
  ]);
  const nodeHeader = Buffer.concat([
    bytesField(2, "constant"),
    bytesField(4, "Constant"),
    varint(5 * 8 + 2),
    varint(attributeHeader.length + size),
    attributeHeader,
  ]);
  const graphHeader = Buffer.concat([
    bytesField(11, valueInfo("input")),
    bytesField(2, "input before node"),
    bytesField(12, valueInfo("constant")),
    varint(1 * 8 + 2),
    varint(nodeHeader.length + size),
    nodeHeader,
  ]);
  const header = Buffer.concat([
    varintField(1, 8),
    bytesField(8, varintField(2, 13)),
    varint(7 * 8 + 2),
    varint(graphHeader.length + size),
    graphHeader,
  ]);
  const fd = openSync(file, "w");
  try {
    writeSync(fd, header);
    ftruncateSync(fd, header.length + size);
  } finally {
    closeSync(fd);
  }

  const graph = (await loadModelFile(file, "constant.onnx")).graph();
  assert.equal(graph.nodes.length, 1);
  assert.equal(graph.nodes[0].opType, "Constant");
});

test("allows shape and metadata analysis after omitting raw or typed initializers", async (t) => {
  const directory = mkdtempSync(path.join(tmpdir(), "onnx-light-viewer-"));
  t.after(() => rmSync(directory, { recursive: true, force: true }));

  for (const [name, dataField] of [
    ["raw", 9],
    ["typed", 4],
  ]) {
    const file = path.join(directory, `matmul-${name}.onnx`);
    writeFileSync(file, matmulWithWeight(dataField));
    const model = await loadModelFile(file, `matmul-${name}.onnx`);
    assert.deepEqual(
      model.graph(true).tensors.find(({ name }) => name === "y").shape,
      ["N", "8"],
    );
    assert.equal(model.metadata().nodes.length, 1);
  }
});

test("verifies invalid graph connectivity before inference and metadata without changing the model", () => {
  const bytes = modelBytes("invalid-graph.onnx");
  const model = loadModel(bytes, "invalid-graph.onnx");
  const before = model.graph();
  for (const action of [
    () => model.graph(true),
    () => model.metadata(),
    () => decodeGraph(bytes, "invalid-graph.onnx", true),
  ]) {
    assert.throws(action, /VerifyModel:/);
  }
  assert.deepEqual(model.graph(), before);
  assert.deepEqual(modelBytes("invalid-graph.onnx"), bytes);
});

test("lists the released native backend tests and opens node and model cases on demand", () => {
  const tests = listBackendTests();
  assert.equal(new Set(tests.map(({ id }) => id)).size, tests.length);
  const reluCase = tests.find(({ name }) => name === "test_cc_relu");
  assert.deepEqual(
    tests.find(({ name }) => name === "test_cc_relu"),
    {
      id: reluCase.id,
      name: "test_cc_relu",
      modelName: "test_cc_relu",
      kind: "node",
      tag: "",
    },
  );
  assert.equal(
    tests.find(({ name }) => name === "test_cc_constant_add_chain").kind,
    "model",
  );
  const relu = loadBackendTest(reluCase.id);
  assert.equal(relu.graph().nodes[0].opType, "Relu");
  assert.deepEqual(
    relu.graph(true).tensors.find(({ name }) => name === "Y"),
    {
      name: "Y",
      type: "FLOAT",
      shape: ["2", "3"],
    },
  );
  const model = loadBackendTest(
    tests.find(({ name }) => name === "test_cc_constant_add_chain").id,
  );
  assert.deepEqual(
    model.graph().nodes.map(({ opType }) => opType),
    ["Add", "Add"],
  );
  assert.equal(model.graph(true).initializerCount, 1);
  assert.throws(
    () => loadBackendTest(tests.length),
    /Unknown onnx-light backend test ID/,
  );
  assert.throws(() => loadBackendTest(null), TypeError);
  for (const invalid of [-1, NaN, Infinity, 0.5]) {
    assert.throws(() => loadBackendTest(invalid), RangeError);
  }
  for (const op of ["greater_or_equal", "less_or_equal"]) {
    for (const type of ["float16", "bfloat16"]) {
      for (const suffix of ["", "_2d"]) {
        const name = `test_cc_${op}_${type}${suffix}`;
        const matches = tests.filter((test) => test.name === name);
        assert.equal(matches.length, 1, `Expected one backend case: ${name}`);
        assert.equal(
          loadBackendTest(matches[0].id).graph().nodes[0].opType,
          op === "greater_or_equal" ? "GreaterOrEqual" : "LessOrEqual",
        );
      }
    }
  }
});

test("opens all released Qwen variants, including their symbolic input and output metadata", () => {
  const cases = listBackendTests();
  for (const suffix of ["", "_fused", "_persistent", "_persistent_paged"]) {
    const name = `test_cc_shape_inference_big_qwen3_4_layers_like${suffix}`;
    const entry = cases.find((test) => test.name === name);
    assert.ok(entry, `Missing Qwen backend case: ${name}`);
    const graph = loadBackendTest(entry.id).graph();
    assert.equal(graph.name, name);
    assert.ok(graph.nodes.some((node) => node.opType === "MatMul"));
    assert.deepEqual(
      graph.tensors.find((tensor) => tensor.name === "input_ids").shape,
      ["batch_size", "sequence_length"],
    );
    assert.deepEqual(
      graph.tensors.find((tensor) => tensor.name === "output_0"),
      {
        name: "output_0",
        type: "FLOAT16",
        shape: ["batch_size", "sequence_length", "32000"],
      },
    );
  }
});

test("loads a real one-node ONNX model file matching the demo", () => {
  const bytes = modelBytes("relu.onnx");
  assert.deepEqual(decodeGraph(bytes, "relu.onnx"), demoGraph);
});

test("navigates local function definitions, overloads and their nested graphs without caller captures", async () => {
  const bytes = modelBytes("functions.onnx");
  for (const model of [
    loadModel(bytes, "functions.onnx"),
    await loadModelFile(
      path.join(__dirname, "models", "functions.onnx"),
      "functions.onnx",
    ),
  ]) {
    const root = model.graph();
    const call = root.nodes[0].function;
    assert.deepEqual(call.path, [{ node: 0, function: 0 }]);
    assert.equal(call.name, "example::Transform");
    assert.equal(call.nodeCount, 3);
    assert.equal(root.nodes[0].schema, null);
    const body = model.graph(false, call.path);
    assert.deepEqual(body.inputs, ["x", "cond"]);
    assert.deepEqual(body.outputs, ["y"]);
    assert.equal(body.initializerCount, 0);
    assert.equal(body.tensors.find(({ name }) => name === "x").shape, null);
    assert.equal(body.nodes[1].opset, "11");
    assert.equal(
      body.nodes[1].schema.defaults.find(({ name }) => name === "axis").value,
      "1",
    );
    assert.equal(
      model.metadata(call.path).values.find(({ name }) => name === "x")
        .constant,
      false,
    );
    const branch = body.nodes[0].attributes.find(
      ({ name }) => name === "then_branch",
    ).graphs[0];
    const child = model.graph(false, branch.path);
    assert.equal(child.nodes[0].opType, "Relu");
    assert.equal(child.breadcrumbs.length, 3);
    assert.deepEqual(child.inputs, ["x"]);
    assert.equal(child.nodes[0].opset, "11");
    const nested = model.graph(false, body.nodes[2].function.path);
    assert.equal(nested.nodes[0].opType, "Sigmoid");
    assert.equal(nested.name, "example::Identity [fast]");
    assert.equal(nested.breadcrumbs.length, 3);
    assert.equal(
      model.graph(false, root.nodes[1].function.path).nodes[0].opType,
      "Neg",
    );
    for (const invalid of [
      [{ node: 0, function: 1 }],
      [{ node: 99, function: 0 }],
      [{ node: 0, function: 99 }],
      [{ node: 0, function: -1 }],
      [{ node: 0, function: 0.5 }],
    ]) {
      assert.throws(() => model.graph(false, invalid), /subgraph path/);
      assert.throws(() => model.metadata(invalid), /subgraph path/);
    }
    assert.equal(model.graph(true, call.path).name, call.name);
    assert.equal(model.metadata(branch.path).nodes.length, 1);
    assert.deepEqual(
      model.graph().nodes.map(({ function: link }) => link),
      root.nodes.map(({ function: link }) => link),
    );
  }
  assert.deepEqual(modelBytes("functions.onnx"), bytes);
});

test("opens nested GRAPH and GRAPHS attributes with scoped captures, parameters and schemas", () => {
  const bytes = modelBytes("subgraphs.onnx");
  const model = loadModel(bytes, "subgraphs.onnx");
  const root = model.graph();
  const branch = (graph, node, attribute) =>
    graph.nodes[node].attributes.find(({ name }) => name === attribute)
      .graphs[0];
  const then = branch(root, 0, "then_branch");
  const otherwise = branch(root, 0, "else_branch");
  assert.equal(then.name, otherwise.name);
  assert.notDeepEqual(then.path, otherwise.path);
  assert.deepEqual(new Set(root.nodes[0].captures), new Set(["x", "bias"]));
  const child = model.graph(false, then.path);
  assert.deepEqual(
    child.nodes.map(({ opType }) => opType),
    ["Add", "If"],
  );
  assert.deepEqual(new Set(child.inputs), new Set(["x", "condition"]));
  assert.deepEqual(child.tensors.find(({ name }) => name === "x").shape, [
    "N",
    "3",
  ]);
  assert.deepEqual(child.nodes[0].parameters[0].shape, ["3"]);
  assert.equal(child.nodes[0].opset, "13");
  assert.ok(child.nodes[0].schema.doc);
  assert.equal(child.initializerCount, 1);
  assert.ok(!child.tensors.some(({ name }) => name === "z"));
  const inner = branch(child, 1, "then_branch");
  const leaf = model.graph(false, inner.path);
  assert.equal(leaf.breadcrumbs.length, 3);
  assert.deepEqual(leaf.inputs, ["sum"]);
  assert.equal(model.graph(false, otherwise.path).nodes[0].opType, "Identity");
  const graphs = root.nodes[1].attributes[0].graphs;
  assert.equal(graphs.length, 2);
  assert.equal(model.graph(false, graphs[0].path).nodes.length, 0);
  const shadow = model.graph(false, graphs[1].path);
  assert.deepEqual(shadow.tensors.find(({ name }) => name === "x").shape, [
    "2",
  ]);
  assert.deepEqual(shadow.nodes[0].parameters[0].shape, ["2"]);
  // Shadowed names remain inspectable, but violate ONNX SSA for analysis.
  assert.throws(
    () => model.metadata(then.path),
    /VerifyModel:[\s\S]*SSA violation/,
  );
  assert.deepEqual(model.graph(), root);
  assert.deepEqual(modelBytes("subgraphs.onnx"), bytes);
  for (const path of [
    [null],
    [{ node: -1, attribute: 0, graph: 0 }],
    [{ node: 0.5, attribute: 0, graph: 0 }],
    [{ node: 99, attribute: 0, graph: 0 }],
    [{ node: 0, attribute: 99, graph: 0 }],
    [{ node: 0, attribute: 0, graph: 99 }],
  ]) {
    assert.throws(() => model.graph(false, path), /subgraph path/);
    assert.throws(() => model.metadata(path), /subgraph path/);
  }
});

test("refreshes shapes inside a nested graph using the retained enclosing model", () => {
  const model = loadModel(modelBytes("nested-if.onnx"), "nested-if.onnx");
  const then = model
    .graph()
    .nodes[0].attributes.find(({ name }) => name === "then_branch").graphs[0];
  const child = model.graph(false, then.path);
  assert.equal(
    model.metadata(then.path).values.find(({ name }) => name === "bias")
      .constant,
    true,
  );
  assert.equal(model.metadata(then.path).nodes.length, 2);
  const leafPath = child.nodes[1].attributes.find(
    ({ name }) => name === "then_branch",
  ).graphs[0].path;
  const leaf = model.graph(true, leafPath);
  assert.deepEqual(leaf.tensors.find(({ name }) => name === "sum").shape, [
    "N",
    "3",
  ]);
  assert.deepEqual(model.graph(false, leafPath), leaf);
});

test("opens the released Loop and Scan bodies with their inferred output shapes", () => {
  const cases = listBackendTests();
  for (const [name, output, shape] of [
    ["test_cc_loop_basic_trip_count", "scan_out", ["1"]],
    ["test_cc_scan_basic_trip_count", "y_elt", ["2"]],
  ]) {
    const model = loadBackendTest(cases.find((test) => test.name === name).id);
    const body = model
      .graph()
      .nodes.flatMap((node) =>
        node.attributes.flatMap((attribute) => attribute.graphs ?? []),
      )[0];
    const graph = model.graph(true, body.path);
    assert.equal(graph.breadcrumbs.length, 2);
    assert.deepEqual(
      graph.tensors.find(({ name }) => name === output).shape,
      shape,
    );
  }
});

test("analyzes constants, tags and last-use releases without reading weights or changing stored metadata", () => {
  const bytes = modelBytes("analysis.onnx");
  const model = loadModel(bytes, "analysis.onnx");
  const graph = model.graph();
  const metadata = model.metadata();
  assert.deepEqual(metadata.model, [
    { key: "purpose", value: "metadata <inspection>" },
  ]);
  assert.deepEqual(metadata.graph, [
    { key: "graph_note", value: "Keep all entries" },
  ]);
  assert.deepEqual(
    metadata.nodes.map((node) => node.constant),
    [true, true, false, false, false, false],
  );
  assert.equal(metadata.nodes[3].tag, "shape");
  assert.deepEqual(metadata.nodes[1].releaseAfter, ["constant"]);
  assert.deepEqual(metadata.nodes[1].lastUse, ["bias"]);
  assert.deepEqual(metadata.nodes[2].lastUse, ["input"]);
  assert.deepEqual(metadata.nodes[3].releaseAfter, []);
  assert.deepEqual(metadata.nodes[4].releaseAfter, []);
  assert.deepEqual(metadata.nodes[5].releaseAfter, [
    "shared;activation",
    "constant_sum",
  ]);
  assert.deepEqual(metadata.nodes[3].unusedOutputs, ["dimensions"]);
  assert.deepEqual(metadata.nodes[4].unusedOutputs, ["unused"]);
  assert.deepEqual(metadata.nodes[5].unusedOutputs, []);
  assert.deepEqual(metadata.nodes[5].entries, [
    { key: "node_note", value: "<script>not executable</script>" },
  ]);
  assert.deepEqual(
    metadata.values.find((value) => value.name === "bias"),
    {
      name: "bias",
      constant: true,
      tag: "weight",
      entries: [{ key: "weight_note", value: "External payload is absent" }],
    },
  );
  assert.equal(
    metadata.values.find((value) => value.name === "constant_sum").constant,
    true,
  );
  assert.equal(
    metadata.values.find((value) => value.name === "dimensions").tag,
    "shape",
  );
  assert.deepEqual(model.graph(), graph);
  assert.deepEqual(modelBytes("analysis.onnx"), bytes);
  assert.throws(() => model.metadata(true), TypeError);

  const kept = loadModel(
    modelBytes("analysis-kept-output.onnx"),
    "kept.onnx",
  ).metadata();
  assert.deepEqual(kept.nodes[5].releaseAfter, ["constant_sum"]);
  assert.ok(
    kept.nodes.every(
      (node) => !node.unusedOutputs.includes("shared;activation"),
    ),
  );
  assert.deepEqual(
    loadModel(modelBytes("passthrough.onnx"), "pass.onnx").metadata().nodes,
    [],
  );
  assert.equal(
    loadModel(modelBytes("custom.onnx"), "custom.onnx").metadata().nodes[1].tag,
    "weight",
  );
});

test("preserves unnamed and custom operators, omits optional ports and initializer inputs", () => {
  const graph = decodeGraph(modelBytes("custom.onnx"), "model.onnx");
  assert.equal(graph.name, "Custom operators");
  assert.deepEqual(graph.inputs, ["x"]);
  assert.deepEqual(graph.outputs, ["y"]);
  assert.equal(graph.initializerCount, 2);
  assert.equal(graph.nodes[0].name, "Add_1");
  assert.deepEqual(graph.nodes[0].inputs, ["x", "weight"]);
  assert.deepEqual(graph.nodes[0].outputs, ["intermediate"]);
  assert.equal(graph.nodes[1].domain, "example");
  assert.deepEqual(graph.nodes[1].inputs, ["intermediate"]);
  assert.deepEqual(graph.nodes[1].outputs, ["y"]);
  assert.notEqual(graph.nodes[0].id, graph.nodes[1].id);
});

test("rejects malformed model bytes, missing model metadata and missing operators", () => {
  assert.throws(() => decodeGraph(Uint8Array.from([255]), "bad.onnx"));
  assert.throws(
    () => decodeGraph(new Uint8Array(), "empty.onnx"),
    /ONNX model graph/,
  );
  assert.throws(
    () => decodeGraph(modelBytes("missing-ir.onnx"), "missing-ir.onnx"),
    /ONNX model graph/,
  );
  assert.throws(
    () => decodeGraph(modelBytes("missing-graph.onnx"), "missing-graph.onnx"),
    /ONNX model graph/,
  );
  assert.throws(
    () =>
      decodeGraph(modelBytes("missing-operator.onnx"), "missing-operator.onnx"),
    /operator type/,
  );
});

test("accepts a valid graph with no operators", () => {
  const graph = decodeGraph(modelBytes("passthrough.onnx"), "identity.onnx");
  assert.deepEqual(graph.nodes, []);
  assert.deepEqual(graph.inputs, graph.outputs);
  assert.equal(
    decodeGraph(modelBytes("unnamed.onnx"), "unnamed.onnx").name,
    "unnamed.onnx",
  );
});

test("the native decoder respects Uint8Array offsets and does not retain its input buffer", () => {
  const bytes = modelBytes("relu.onnx");
  const padded = new Uint8Array(bytes.length + 16);
  padded.fill(255);
  padded.set(bytes, 8);
  const graph = decodeGraph(padded.subarray(8, 8 + bytes.length), "relu.onnx");
  padded.fill(0);
  assert.deepEqual(graph, demoGraph);
});

test("native input errors are JavaScript exceptions, not process crashes", () => {
  assert.throws(() => decodeGraph("not bytes", "model.onnx"), TypeError);
  assert.throws(
    () => decodeGraph(new Int32Array([1]), "model.onnx"),
    TypeError,
  );
  assert.throws(() => decodeGraph(modelBytes("relu.onnx"), null), TypeError);
});

test("loads typed attributes and dense/sparse initializer parameters without losing int64 precision", () => {
  const graph = decodeGraph(modelBytes("parameters.onnx"), "parameters.onnx");
  const [conv, activation, custom] = graph.nodes;
  assert.deepEqual(
    conv.attributes.map(({ name, type }) => ({ name, type })),
    [
      { name: "kernel_shape", type: "INTS" },
      { name: "strides", type: "INTS" },
    ],
  );
  assert.equal(conv.attributes[1].value, "[2, 2]");
  assert.equal(activation.attributes[0].value, "0.125");
  assert.equal(conv.parameters[0].name, "kernel");
  assert.equal(conv.parameters[0].type, "FLOAT");
  assert.deepEqual(conv.parameters[0].shape, ["1", "1", "3", "3"]);
  assert.deepEqual(conv.parameters[0].storage, {
    external: false,
    location: null,
  });
  assert.equal(
    conv.parameters[0].value,
    `[${Array(9).fill("0.5").join(", ")}]`,
  );
  assert.equal(custom.parameters[0].name, "sparse");
  assert.equal(custom.parameters[0].type, "SPARSE FLOAT");
  const attributes = Object.fromEntries(
    custom.attributes.map((a) => [a.name, a]),
  );
  assert.equal(attributes.threshold.value, "1152921504606846977");
  assert.equal(attributes.ratios.value, "[0.5, 1.5]");
  assert.equal(attributes.labels.value, "['left', 'right']");
  assert.equal(attributes.text.value, "<b>model text</b>");
  assert.match(attributes.value.value, /int64_data: \[3, 7\]/);
  assert.equal(
    attributes.long_list.value,
    "[0, 1, 2, 3, 4, 5, 6, 7] ... (2000 elements)",
  );
  for (const name of ["long_floats", "long_strings", "sparse_value"]) {
    assert.match(attributes[name].value, /\.\.\. \(12 elements\)/);
    assert.ok(attributes[name].value.length < 512);
  }
  assert.match(attributes.long_floats.value, /\[0, 1, 2, 3, 4, 5, 6, 7\]/);
  assert.doesNotMatch(attributes.long_strings.value, /label8/);
  assert.match(
    attributes.sparse_value.value,
    /int64_data: \[0, 1, 2, 3, 4, 5, 6, 7\]/,
  );
});

test("exposes connected initializer metadata and tiny values without external reads", () => {
  const graph = decodeGraph(
    modelBytes("initializers.onnx"),
    "initializers.onnx",
  );
  const parameters = graph.nodes[0].parameters;
  assert.equal(graph.initializerCount, 12);
  assert.equal(parameters.length, 12);
  assert.deepEqual(parameters[9], {
    name: "weight_9",
    type: "FLOAT",
    shape: ["1"],
    storage: { external: false, location: null },
    value: "[9.25]",
  });
  assert.deepEqual(parameters[10].storage, {
    external: true,
    location: "weights/decoder.bin",
  });
  assert.deepEqual(parameters[11], {
    name: "external_sparse",
    type: "SPARSE FLOAT",
    shape: ["1"],
    storage: { external: true, location: "weights/values.bin" },
    indicesStorage: { external: true, location: "weights/indices.bin" },
  });
  assert.ok(parameters.slice(0, 10).every((parameter) => "value" in parameter));
  assert.equal("value" in parameters[10], false);
  assert.deepEqual(
    loadModel(modelBytes("initializers.onnx"), "initializers.onnx").graph(),
    graph,
  );
});

test("previews fewer than 16 initializer elements with locale-independent decimals and exact integer precision", async () => {
  const bytes = modelBytes("tiny-initializers.onnx");
  const graph = decodeGraph(bytes, "tiny-initializers.onnx");
  const parameters = Object.fromEntries(
    graph.nodes[0].parameters.map((p) => [p.name, p]),
  );
  const expected = {
    typed_floats: "[0.5, -2, 3]",
    precision_float: "[0.100000001]",
    precision_double: "[0.10000000000000001]",
    precision_complex64: "[(0.100000001, -0.200000003)]",
    precision_complex128: "[(0.10000000000000001, -0.20000000000000001)]",
    raw_fifteen: `[${Array.from({ length: 15 }, (_, index) => index).join(", ")}]`,
    raw_int64: "[1152921504606846977, -1152921504606846979]",
    int64_fifteen: `[${Array.from({ length: 15 }, (_, index) => index).join(", ")}]`,
    uint64_max: "[18446744073709551615]",
    scalar: "[0.125]",
    booleans: "[false, true]",
    strings: '["<img src=x onerror=alert(1)>", "hello"]',
    empty: "[]",
    float16: "[1, -2]",
    bfloat16: "[1, -2]",
    uint4: "[1, 15, 2]",
    int4: "[-8, -1, 7]",
    uint2: "[0, 1, 2, 3, 3]",
    int2: "[0, 1, -2, -1, -1]",
    float4: "[1, -1]",
    float8: "[1, -2]",
    float6: "[1, 2, 3]",
    complex_fifteen: `[${Array(15).fill("(0.5, -1)").join(", ")}]`,
    tiny_sparse: "values: [0.5, 2]; indices: [0, 3]",
    large_sparse: "values: [0.5, 2]; indices: [0, 999999]",
  };
  const streamed = await loadModelFile(
    path.join(__dirname, "models", "tiny-initializers.onnx"),
    "tiny-initializers.onnx",
  );
  for (const snapshot of [
    graph,
    loadModel(bytes, "tiny-initializers.onnx").graph(),
    streamed.graph(),
    ...withCommaLocale((number) => {
      assert.equal(number, "1.234,5");
      return [
        decodeGraph(bytes, "tiny-initializers.onnx"),
        loadModel(bytes, "tiny-initializers.onnx").graph(),
        streamed.graph(),
      ];
    }),
  ]) {
    const values = Object.fromEntries(
      snapshot.nodes[0].parameters.map((p) => [p.name, p.value]),
    );
    for (const [name, value] of Object.entries(expected))
      assert.equal(values[name], value, name);
    assert.equal(values.sixteen, undefined);
    assert.equal(values.external, undefined);
    assert.equal(values.wide_sparse_indices, undefined);
    assert.equal(values.sparse_sixteen, undefined);
  }
  assert.deepEqual(parameters.large_sparse.shape, ["1000000"]);
  assert.equal(parameters.long_string.value, `["${"x".repeat(128)}"...]`);
});

test("identifies malformed string initializer counts in native preview errors", () => {
  const tensor = Buffer.concat([
    varintField(1, 2),
    varintField(2, 8),
    bytesField(8, "broken_labels"),
    bytesField(6, "only one value"),
  ]);
  const bytes = Buffer.concat([
    varintField(1, 9),
    bytesField(7, bytesField(5, tensor)),
  ]);
  for (const graph of [
    () => decodeGraph(bytes, "malformed-strings.onnx"),
    () => loadModel(bytes, "malformed-strings.onnx").graph(),
  ]) {
    assert.throws(
      graph,
      /Initializer 'broken_labels'.*expected 2 elements, got 1/,
    );
  }
});

test("the bundled tiny language model preserves symbolic batch/sequence dimensions through attention inference", () => {
  const bytes = readFileSync(require.resolve("../media/tiny-llm.onnx"));
  const model = loadModel(bytes, "tiny-llm.onnx");
  const before = model.graph();
  assert.deepEqual(before.inputs, [
    "input_ids",
    "position_ids",
    "attention_mask",
  ]);
  assert.deepEqual(
    before.tensors.find(({ name }) => name === "token_vectors").shape,
    ["batch_size", "sequence_length", "16"],
  );
  assert.equal(
    before.tensors.find(({ name }) => name === "feed_forward_expanded").type,
    null,
  );
  const graph = model.graph(true);
  for (const [name, shape] of [
    ["query.heads", ["batch_size", "2", "sequence_length", "8"]],
    [
      "attention_scores",
      ["batch_size", "2", "sequence_length", "sequence_length"],
    ],
    ["attention_context", ["batch_size", "2", "sequence_length", "8"]],
    ["context_merged", ["batch_size", "sequence_length", "16"]],
    ["feed_forward_expanded", ["batch_size", "sequence_length", "32"]],
    ["logits", ["batch_size", "sequence_length", "32"]],
  ]) {
    const tensor = graph.tensors.find((tensor) => tensor.name === name);
    assert.equal(tensor.type, "FLOAT");
    assert.deepEqual(tensor.shape, shape);
  }
  assert.deepEqual(model.graph(), graph);
});

test("uses the model domain and opset to select LightOpSchema documentation and defaults", () => {
  for (const [version, axis] of [
    [11, "1"],
    [13, "-1"],
  ]) {
    const node = decodeGraph(
      modelBytes(`softmax-${version}.onnx`),
      "softmax.onnx",
    ).nodes[0];
    assert.equal(node.opset, String(version));
    assert.equal(node.schema.sinceVersion, version);
    assert.match(node.schema.doc, /normalized exponential/);
    assert.match(node.schema.doc, /Inputs[\s\S]*Outputs[\s\S]*Attributes/);
    assert.deepEqual(
      node.schema.defaults.find((a) => a.name === "axis"),
      {
        name: "axis",
        type: "INT",
        value: axis,
        default: true,
      },
    );
  }
  const custom = decodeGraph(modelBytes("parameters.onnx"), "parameters.onnx")
    .nodes[2];
  assert.equal(custom.opType, "Softmax");
  assert.equal(custom.domain, "example");
  assert.equal(custom.schema, null);
});

test("exposes declared tensor types, scalar/unknown ranks, symbolic/zero dimensions and initializer shapes", () => {
  const tensors = Object.fromEntries(
    decodeGraph(modelBytes("metadata.onnx"), "metadata.onnx").tensors.map(
      (tensor) => [tensor.name, tensor],
    ),
  );
  assert.deepEqual(tensors.scalar, {
    name: "scalar",
    type: "FLOAT",
    shape: [],
  });
  assert.deepEqual(tensors.unknown_rank, {
    name: "unknown_rank",
    type: "FLOAT",
    shape: null,
  });
  assert.deepEqual(tensors.zero.shape, ["0", "N", "?"]);
  assert.match(tensors.sequence.type, /sequence_type/);
  assert.equal(tensors.sequence.shape, undefined);
  const graph = decodeGraph(modelBytes("custom.onnx"), "custom.onnx");
  assert.deepEqual(
    graph.tensors.find((tensor) => tensor.name === "sparse"),
    {
      name: "sparse",
      type: "SPARSE FLOAT",
      shape: ["1"],
    },
  );
});

test("refreshes intermediate shapes and types with onnx-light, including symbolic Reshape with raw constants", () => {
  const bytes = modelBytes("inference.onnx");
  const original = Buffer.from(bytes);
  const before = decodeGraph(bytes, "inference.onnx");
  assert.deepEqual(
    before.tensors.find((tensor) => tensor.name === "hidden"),
    {
      name: "hidden",
      type: null,
      shape: null,
    },
  );
  const inferred = decodeGraph(bytes, "inference.onnx", true);
  const tensors = Object.fromEntries(
    inferred.tensors.map((tensor) => [tensor.name, tensor]),
  );
  assert.deepEqual(tensors.hidden, {
    name: "hidden",
    type: "FLOAT",
    shape: ["N", "3"],
  });
  assert.deepEqual(tensors.reshaped.shape, ["N", "3", "1"]);
  assert.deepEqual(tensors.transposed.shape, ["N", "1", "3"]);
  assert.equal(tensors.transposed.type, "FLOAT");
  assert.equal(tensors.y.type, "INT64");
  assert.deepEqual(bytes, original);
  bytes.fill(0);
  assert.deepEqual(tensors.y.shape, ["N", "1", "3"]);
});

test("surfaces unsupported shape inference as a JavaScript error", () => {
  assert.throws(
    () => decodeGraph(modelBytes("custom.onnx"), "custom.onnx", true),
    /unsupported domain.*example/,
  );
});

test("persists inferred intermediate value_info in the native model without modifying its source", () => {
  const bytes = modelBytes("inference.onnx");
  const original = Buffer.from(bytes);
  const model = loadModel(bytes, "inference.onnx");
  assert.deepEqual(model.graph(undefined), model.graph());
  assert.throws(() => model.graph("yes"), TypeError);
  assert.equal(
    model.graph().tensors.find(({ name }) => name === "hidden").type,
    null,
  );
  const inferred = model.graph(true);
  assert.deepEqual(
    inferred.tensors.find(({ name }) => name === "hidden"),
    { name: "hidden", type: "FLOAT", shape: ["N", "3"] },
  );
  assert.deepEqual(model.graph(), inferred);
  assert.deepEqual(model.graph(true), inferred);
  inferred.tensors.find(({ name }) => name === "hidden").shape[0] =
    "changed in JS";
  assert.deepEqual(
    model.graph().tensors.find(({ name }) => name === "hidden").shape,
    ["N", "3"],
  );
  assert.deepEqual(bytes, original);
  const unsupported = loadModel(modelBytes("custom.onnx"), "custom.onnx");
  const before = unsupported.graph();
  assert.throws(() => unsupported.graph(true), /unsupported domain.*example/);
  assert.deepEqual(unsupported.graph(), before);
});

test("keeps borrowed raw tensor data alive when only the native graph method is retained", () => {
  execFileSync(process.execPath, [
    "--expose-gc",
    "-e",
    `
    const assert = require("node:assert/strict");
    const { readFileSync } = require("node:fs");
    const { loadModel } = require(${JSON.stringify(require.resolve("../dist/graph"))});
    const graph = (() => {
      const bytes = readFileSync(${JSON.stringify(require.resolve("./models/inference.onnx"))});
      const padded = new Uint8Array(bytes.length + 16);
      padded.set(bytes, 8);
      return loadModel(padded.subarray(8, 8 + bytes.length), "inference.onnx").graph;
    })();
    global.gc();
    assert.deepEqual(
      graph(true).tensors.find(({ name }) => name === "reshaped").shape,
      ["N", "3", "1"],
    );
  `,
  ]);
});

test("keeps native backend case resources alive through garbage collection and later catalog loads", () => {
  execFileSync(process.execPath, [
    "--expose-gc",
    "-e",
    `
    const assert = require("node:assert/strict");
    const { loadBackendTest, listBackendTests } = require(${JSON.stringify(require.resolve("../dist/graph"))});
    const cases = listBackendTests();
    const graph = loadBackendTest(cases.find(test => test.name === "test_cc_constant_node_source").id).graph;
    const before = graph();
    for (let index = 0; index < 3; index++) {
      listBackendTests();
      loadBackendTest(cases.find(test => test.name === "test_cc_relu").id).graph();
      global.gc();
    }
    assert.deepEqual(graph(), before);
    assert.doesNotThrow(() => graph(true));
  `,
  ]);
});
