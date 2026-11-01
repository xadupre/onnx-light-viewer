const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { test } = require("node:test");
const {
  loadModel,
  loadModelFile,
  listBackendTests,
  loadBackendTest,
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
  await assert.rejects(
    model.run([{ name: "x", source: "random" }]),
    /Specify dynamic dimension: N/,
  );
  const result = await model.run([{ name: "x", source: "random" }], [], {
    dimensions: { N: 2 },
  });
  assert.match(result.outputs[0].text, /Values: \[/);
  await assert.rejects(
    model.run(["float[1,3] x = {1, 2, 3}"], [], { dimensions: { N: 2 } }),
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
});
