const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { test } = require("node:test");
const {
  loadModel,
  loadModelFile,
  listBackendTests,
  loadBackendTest,
  tensorStatisticsFromProto,
} = require("../dist/graph");

test("regular native execution runs ReLU without altering the retained graph", async () => {
  const model = loadModel(
    readFileSync(require.resolve("./models/relu.onnx")),
    "relu.onnx",
  );
  const before = model.graph();
  const inputs = model.runInputs();
  assert.equal(inputs.length, 1);
  const result = await model.run([inputs[0].text.replace("{}", "{-1, 0, 2}")]);
  assert.ok(result.timeMs >= 0);
  assert.equal(result.outputs.length, 1);
  assert.match(result.outputs[0].text, /Values: \[0, 0, 2\]/);
  assert.equal(result.inputs[0].name, "input");
  assert.match(result.inputs[0].text, /Values: \[-1, 0, 2\]/);
  const prepared = await model.prepareRun(["float[1,3] input = {-1, 0, 2}"]);
  assert.deepEqual(result.inputs[0].data, prepared.inputs[0]);
  const reexecuted = await loadModel(prepared.model, "snapshot.onnx").run([
    { name: "input", source: "protobuf", data: result.inputs[0].data },
  ]);
  assert.deepEqual(reexecuted.inputs, result.inputs);
  assert.deepEqual(reexecuted.outputs, result.outputs);
  const outputInput = await model.prepareRun([
    { name: "input", source: "protobuf", data: result.outputs[0].data },
  ]);
  assert.deepEqual(
    (
      await model.run([
        { name: "input", source: "protobuf", data: outputInput.inputs[0] },
      ])
    ).outputs,
    result.outputs,
  );
  assert.deepEqual(model.graph(), before);
  await assert.rejects(model.run([]), /input/i);
  await assert.rejects(model.run(["invalid tensor"]), /./);
  assert.throws(() => model.run([1]), /tensor text/);
  assert.throws(
    () => model.runInputs([{ node: 0, attribute: 0, graph: 0 }]),
    /whole model/,
  );
  const tensor = inputs[0].text.replace("{}", "{1, 2, 3}");
  await assert.rejects(model.run([tensor, tensor]), /unique/);
  await assert.rejects(
    model.run(["float[1] unknown = {1}"]),
    /Unknown model input/,
  );
});

test("regular execution works with asynchronously loaded models", async () => {
  const model = await loadModelFile(
    require.resolve("./models/relu.onnx"),
    "relu.onnx",
  );
  const result = await model.run([
    model.runInputs()[0].text.replace("{}", "{1, 2, 3}"),
  ]);
  assert.match(result.outputs[0].text, /Values: \[1, 2, 3\]/);
});

test("native retained sessions serialize predictions, use fresh inputs and release independently", async () => {
  const model = loadModel(
    readFileSync(require.resolve("./models/relu.onnx")),
    "relu.onnx",
  );
  const texts = [
    "float[1,3] input = {-1, 0, 2}",
    "float[1,3] input = {3, -4, 5}",
    "float[1,3] input = {6, 7, -8}",
  ];
  const results = await Promise.all(
    texts.map((text) => model.run([text], [], { keepLoaded: true })),
  );
  assert.equal(results.filter((result) => !result.lifecycle.reused).length, 1);
  assert.ok(results.every((result) => result.lifecycle.retained));
  for (const [index, expected] of [
    "[0, 0, 2]",
    "[3, 0, 5]",
    "[6, 7, 0]",
  ].entries())
    assert.ok(results[index].outputs[0].text.includes(expected));
  const clone = model.clone();
  assert.equal(
    (await clone.run([texts[0]], [], { keepLoaded: true })).lifecycle.reused,
    false,
  );
  clone.releaseRunSession();
  assert.equal(
    (await model.run([texts[0]], [], { keepLoaded: true })).lifecycle.reused,
    true,
  );
  model.releaseRunSession();
  const pending = model.run([texts[0]], [], { keepLoaded: true });
  model.releaseRunSession();
  assert.equal((await pending).lifecycle.retained, false);
  const afterRelease = await model.run([texts[0]], [], { keepLoaded: true });
  assert.equal(afterRelease.lifecycle.reused, false);
  const oneShot = await model.run([texts[0]]);
  assert.equal(oneShot.lifecycle.reused, false);
  assert.equal(oneShot.lifecycle.retained, false);
});

test("native Run failures release retained state before the next direct call", async (t) => {
  const model = loadModel(
    readFileSync(require.resolve("./models/relu.onnx")),
    "relu.onnx",
  );
  t.after(() => model.releaseRunSession());
  const inputs = ["float[1,3] input = {-1, 0, 2}"];
  const predict = () => model.run(inputs, [], { keepLoaded: true });
  for (const fail of [
    () => assert.rejects(model.run(["not a tensor"], [], { keepLoaded: true })),
    () =>
      assert.throws(
        () => model.run(inputs, [], { keepLoaded: "yes" }),
        /boolean/,
      ),
    () =>
      assert.throws(
        () => model.run(inputs, [], { keepLoaded: true, verbose: 11 }),
        /Verbosity/,
      ),
  ]) {
    assert.equal((await predict()).lifecycle.retained, true);
    assert.equal((await predict()).lifecycle.reused, true);
    await fail();
    assert.equal((await predict()).lifecycle.reused, false);
  }
});

test("a failing native worker does not release its replacement retained state", async (t) => {
  const model = loadModel(
    readFileSync(require.resolve("./models/relu.onnx")),
    "relu.onnx",
  );
  t.after(() => model.releaseRunSession());
  const inputs = ["float[1,3] input = {-1, 0, 2}"];
  const failure = assert.rejects(
    model.run(["not a tensor"], [], { keepLoaded: true }),
  );
  model.releaseRunSession();
  const replacement = model.run(inputs, [], { keepLoaded: true });
  await failure;
  assert.equal((await replacement).lifecycle.retained, true);
  const reused = await model.run(inputs, [], { keepLoaded: true });
  assert.equal(reused.lifecycle.reused, true);
  assert.equal(reused.lifecycle.retained, true);
});

test("regular execution supports no-input models and scalar integer/string outputs", async () => {
  const cases = listBackendTests();
  for (const [name, expected] of [
    ["test_cc_constant_value_int", "Values: [42]"],
    ["test_cc_constant_value_string", 'Values: ["hello"]'],
  ]) {
    const model = loadBackendTest(cases.find((item) => item.name === name).id);
    assert.deepEqual(model.runInputs(), []);
    const result = await model.run([]);
    assert.ok(result.outputs[0].text.includes(expected));
  }
});

test("regular input forms omit initializer defaults and preserve scalar types", () => {
  const model = loadModel(
    readFileSync(require.resolve("./models/functions.onnx")),
    "functions.onnx",
  );
  assert.deepEqual(model.runInputs(), [
    { name: "cond", text: "bool cond = {}", shape: [] },
  ]);
});

test("random inputs use SDK generation and prepared TensorProto inputs preserve exact values", async () => {
  const model = loadModel(
    readFileSync(require.resolve("./models/relu.onnx")),
    "relu.onnx",
  );
  assert.deepEqual(model.runInputs()[0].shape, ["1", "3"]);
  const result = await model.run([{ name: "input", source: "random" }], [], {
    verbose: 0,
  });
  assert.match(result.outputs[0].text, /Values: \[/);
  assert.ok(result.inputs[0].data instanceof Uint8Array);
  const replay = await model.prepareRun([
    { name: "input", source: "protobuf", data: result.inputs[0].data },
  ]);
  assert.deepEqual(replay.inputs[0], result.inputs[0].data);
  assert.deepEqual(
    (
      await loadModel(replay.model, "random-snapshot.onnx").run([
        { name: "input", source: "protobuf", data: replay.inputs[0] },
      ])
    ).outputs,
    result.outputs,
  );
  const prepared = await model.prepareRun(["float[1,3] input = {-1, 0, 2}"]);
  const copy = loadModel(prepared.model, "snapshot.onnx");
  const fromFile = await copy.run([
    { name: "input", source: "protobuf", data: prepared.inputs[0] },
  ]);
  assert.match(fromFile.outputs[0].text, /Values: \[0, 0, 2\]/);
  const raw = await model.prepareRun([{ name: "input", source: "random" }]);
  const source = {
    name: "input",
    source: "protobuf",
    data: new Uint8Array(raw.inputs[0]),
  };
  const expected = await model.run([source]);
  for (let index = 0; index < 10; ++index)
    assert.deepEqual((await model.run([source])).outputs, expected.outputs);
  await assert.rejects(
    model.run([
      { name: "input", source: "protobuf", data: Buffer.from("invalid") },
    ]),
    /parse/,
  );
  assert.throws(() => model.run([], [], { verbose: 11 }), /Verbosity/);
  assert.throws(
    () => model.run([], [], { dimensions: { batch: -1 } }),
    /dimension/,
  );
});

test("dynamic random inputs require concrete dimensions without changing model metadata", async () => {
  const model = loadModel(
    readFileSync(require.resolve("./models/inference.onnx")),
    "inference.onnx",
  );
  const before = model.graph();
  assert.deepEqual(model.runInputs()[0].shape, ["N", "3"]);
  assert.deepEqual(model.runInputs()[0].dynamicAxes, [0]);
  await assert.rejects(
    model.run([{ name: "x", source: "random" }]),
    /Specify dynamic dimension: symbol:N/,
  );
  const result = await model.run([{ name: "x", source: "random" }], [], {
    dimensions: { "symbol:N": 2 },
  });
  assert.match(result.outputs[0].text, /Values: \[/);
  await assert.rejects(
    model.run(["float[1,3] x = {1, 2, 3}"], [], {
      dimensions: { "symbol:N": 2 },
    }),
    /does not match/,
  );
  assert.deepEqual(model.graph(), before);
});

test("regular execution preserves int64 input and output values beyond JavaScript precision", async () => {
  const entry = listBackendTests().find(
    (item) => item.name === "test_cc_identity_int64",
  );
  const model = loadBackendTest(entry.id);
  const result = await model.run([
    model
      .runInputs()[0]
      .text.replace("{}", "{9007199254740993, -9007199254740993, 0, 1, 2, 3}"),
  ]);
  assert.match(
    result.outputs[0].text,
    /Values: \[9007199254740993, -9007199254740993, 0, 1, 2, 3\]/,
  );
  const prepared = await model.prepareRun([
    {
      name: model.runInputs()[0].name,
      source: "protobuf",
      data: result.outputs[0].data,
    },
  ]);
  assert.deepEqual(
    (
      await loadModel(prepared.model, "int64-snapshot.onnx").run([
        {
          name: model.runInputs()[0].name,
          source: "protobuf",
          data: prepared.inputs[0],
        },
      ])
    ).outputs,
    result.outputs,
  );
});

test("standalone TensorProto statistics compute full numeric summaries even for small tensors", async () => {
  const model = loadModel(
    readFileSync(require.resolve("./models/relu.onnx")),
    "relu.onnx",
  );
  const result = await model.run(["float[1,3] input = {-1, 0, 2}"]);
  const data = result.inputs[0].data;
  const padded = Buffer.concat([Buffer.from([255]), data, Buffer.from([255])]);
  const slice = new Uint8Array(
    padded.buffer,
    padded.byteOffset + 1,
    data.byteLength,
  );
  const pending = tensorStatisticsFromProto(slice);
  slice.fill(255);
  const statistics = await pending;
  assert.equal(statistics.name, "input");
  assert.equal(statistics.dtype, "FLOAT");
  assert.deepEqual(statistics.shape, ["1", "3"]);
  assert.equal(statistics.count, "3");
  assert.equal(statistics.sizeBytes, "12");
  assert.equal(statistics.min, -1);
  assert.equal(statistics.max, 2);
  assert.equal(statistics.mean, 1 / 3);
  assert.equal(statistics.median, 0);
  assert.deepEqual(statistics.quantiles, [-0.5, 0, 1]);
  assert.deepEqual(statistics.rectangleShape, [32, 32]);
  assert.equal(statistics.rectangleMap.length, 1024);
  assert.equal("values" in statistics, false);
  assert.equal(
    (await tensorStatisticsFromProto(result.outputs[0].data)).mean,
    2 / 3,
  );
});

test("run previews include 32 numeric elements but complete data survives above the limit", async () => {
  for (const count of [32, 33]) {
    const model = identityModel(1, count);
    const text = `float[${count}] input = {${Array.from({ length: count }, (_, index) => index).join(", ")}}`;
    const result = await model.run([text]);
    for (const tensor of [result.inputs[0], result.outputs[0]]) {
      if (count === 32) assert.match(tensor.text, /Values: \[0, 1, 2/);
      else assert.match(tensor.text, /Omitted \(preview limit: 32 elements\)/);
      const statistics = await tensorStatisticsFromProto(tensor.data);
      assert.equal(statistics.count, String(count));
      assert.equal(statistics.mean, (count - 1) / 2);
      assert.equal(statistics.max, count - 1);
    }
    const prepared = await model.prepareRun([
      { name: "input", source: "protobuf", data: result.outputs[0].data },
    ]);
    const repeated = await loadModel(prepared.model, "numeric.onnx").run([
      { name: "input", source: "protobuf", data: prepared.inputs[0] },
    ]);
    assert.deepEqual(repeated.inputs[0].data, prepared.inputs[0]);
    assert.deepEqual(repeated.outputs, result.outputs);
  }
});

test("non-tensor run outputs do not expose TensorProto data", async () => {
  const entry = listBackendTests().find(
    (item) => item.name === "test_cc_sequence_construct",
  );
  const model = loadBackendTest(entry.id);
  const result = await model.run(
    model
      .runInputs()
      .map((input) => input.text.replace("{}", "{1, 2, 3, 4, 5, 6}")),
  );
  assert.equal(result.inputs.length, 3);
  assert.ok(result.inputs.every((input) => input.data instanceof Uint8Array));
  assert.deepEqual(result.outputs, [
    {
      name: "output_sequence",
      text: "Non-tensor output (no tensor preview).",
    },
  ]);
});

test("run TensorProto data preserves all strings while previews remain bounded", async () => {
  for (const count of [1, 32, 33]) {
    const model = identityModel(8, count);
    const long = "saved-in-full-".repeat(30);
    const strings = Array.from({ length: count }, (_, index) =>
      index === 0 ? long : `string-${index}`,
    );
    const text = `string[${count}] input = {${strings.map((value) => JSON.stringify(value)).join(", ")}}`;
    const prepared = await model.prepareRun([text]);
    const result = await model.run([text]);
    assert.deepEqual(result.inputs[0].data, prepared.inputs[0]);
    for (const tensor of [result.inputs[0], result.outputs[0]]) {
      assert.ok(tensor.data instanceof Uint8Array);
      assert.ok(Buffer.from(tensor.data).includes(Buffer.from(long)));
      assert.ok(
        Buffer.from(tensor.data).includes(Buffer.from(strings[count - 1])),
      );
      if (count <= 32) {
        assert.match(tensor.text, /Values: \[/);
        assert.ok(!tensor.text.includes(long));
      } else {
        assert.match(tensor.text, /Omitted \(preview limit: 32 elements\)/);
      }
    }
    const saved = await model.prepareRun([
      { name: "input", source: "protobuf", data: result.outputs[0].data },
    ]);
    assert.deepEqual(saved.inputs[0], prepared.inputs[0]);
    const repeated = await loadModel(saved.model, "strings.onnx").run([
      { name: "input", source: "protobuf", data: saved.inputs[0] },
    ]);
    assert.deepEqual(repeated.outputs, result.outputs);
    await assert.rejects(
      tensorStatisticsFromProto(result.outputs[0].data),
      /requires a real numeric tensor/,
    );
  }
});

test("standalone statistics reject invalid transport, external payloads and SDK-invalid tensors", async () => {
  const native = require("../dist/native/onnx_light_viewer.node");
  for (const value of [
    undefined,
    "tensor",
    [],
    new Uint16Array([1]),
    new DataView(new ArrayBuffer(1)),
  ]) {
    assert.throws(() => tensorStatisticsFromProto(value), /Uint8Array/);
  }
  assert.throws(() => tensorStatisticsFromProto(new Uint8Array()), /nonempty/);
  assert.throws(() => native.tensorStatisticsFromProto(), /Uint8Array/);
  assert.throws(
    () => native.tensorStatisticsFromProto(new Uint8Array([1]), false),
    /Uint8Array/,
  );
  await assert.rejects(
    tensorStatisticsFromProto(Buffer.from("invalid")),
    /parse TensorProto/,
  );
  await assert.rejects(
    tensorStatisticsFromProto(
      Buffer.concat([varintField(2, 1), varintField(14, 1)]),
    ),
    /inline values/,
  );
  await assert.rejects(
    tensorStatisticsFromProto(
      Buffer.concat([
        varintField(2, 1),
        bytesField(
          13,
          Buffer.concat([
            bytesField(1, "location"),
            bytesField(2, "weights.bin"),
          ]),
        ),
      ]),
    ),
    /inline values/,
  );
  await assert.rejects(
    tensorStatisticsFromProto(
      Buffer.concat([
        varintField(1, 3),
        varintField(2, 1),
        bytesField(9, Buffer.alloc(4)),
      ]),
    ),
    /raw_data size does not match the shape and data_type/,
  );
});

function varint(value) {
  const bytes = [];
  do {
    bytes.push((value & 127) | (value > 127 ? 128 : 0));
    value = Math.floor(value / 128);
  } while (value);
  return Buffer.from(bytes);
}

function varintField(field, value) {
  return Buffer.concat([varint(field * 8), varint(value)]);
}

function bytesField(field, value) {
  const bytes = Buffer.from(value);
  return Buffer.concat([varint(field * 8 + 2), varint(bytes.length), bytes]);
}

function identityModel(dtype, count) {
  const type = bytesField(
    1,
    Buffer.concat([
      varintField(1, dtype),
      bytesField(2, bytesField(1, varintField(1, count))),
    ]),
  );
  const value = (name) =>
    Buffer.concat([bytesField(1, name), bytesField(2, type)]);
  return loadModel(
    Buffer.concat([
      varintField(1, 9),
      bytesField(8, varintField(2, 18)),
      bytesField(
        7,
        Buffer.concat([
          bytesField(2, "identity"),
          bytesField(11, value("input")),
          bytesField(12, value("output")),
          bytesField(
            1,
            Buffer.concat([
              bytesField(1, "input"),
              bytesField(2, "output"),
              bytesField(4, "Identity"),
            ]),
          ),
        ]),
      ),
    ]),
    "identity.onnx",
  );
}
