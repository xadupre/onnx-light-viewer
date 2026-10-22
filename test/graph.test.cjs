const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const path = require("node:path");
const { test } = require("node:test");
const { decodeGraph, demoGraph } = require("../dist/graph");

function modelBytes(name) {
  return readFileSync(path.join(__dirname, "models", name));
}

test("loads a real one-node ONNX model file matching the demo", () => {
  const bytes = modelBytes("relu.onnx");
  assert.deepEqual(decodeGraph(bytes, "relu.onnx"), demoGraph);
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
  assert.match(conv.attributes[1].value, /\[2, 2\]/);
  assert.match(activation.attributes[0].value, /0\.125/);
  assert.equal(conv.parameters[0].name, "kernel");
  assert.equal(conv.parameters[0].type, "FLOAT");
  assert.match(conv.parameters[0].value, /dims: \[1, 1, 3, 3\]/);
  assert.match(conv.parameters[0].value, /0\.5/);
  assert.equal(custom.parameters[0].name, "sparse");
  assert.equal(custom.parameters[0].type, "SPARSE FLOAT");
  const attributes = Object.fromEntries(
    custom.attributes.map((a) => [a.name, a]),
  );
  assert.match(attributes.threshold.value, /1152921504606846977/);
  assert.match(attributes.ratios.value, /\[0\.5, 1\.5\]/);
  assert.match(attributes.labels.value, /"left", "right"/);
  assert.match(attributes.text.value, /<b>model text<\/b>/);
  assert.match(attributes.value.value, /int64_data: \[3, 7\]/);
  assert.ok(attributes.long_list.value.length <= 4096);
  assert.match(attributes.long_list.value, /\.\.\.$/);
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
