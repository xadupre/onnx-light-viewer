const assert = require("node:assert/strict");
const { execFileSync } = require("node:child_process");
const { readFileSync } = require("node:fs");
const path = require("node:path");
const { test } = require("node:test");
const { decodeGraph, demoGraph, loadModel } = require("../dist/graph");

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
  assert.deepEqual(conv.parameters[0].shape, ["1", "1", "3", "3"]);
  assert.deepEqual(conv.parameters[0].storage, {
    external: false,
    location: null,
  });
  assert.equal("value" in conv.parameters[0], false);
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
  assert.equal(
    attributes.long_list.value,
    "{ long_list: [0, 1, 2, 3, 4, 5, 6, 7] } ... (2000 elements)",
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

test("exposes all connected initializer shapes/types and external file locations, without values or external reads", () => {
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
  assert.ok(parameters.every((parameter) => !("value" in parameter)));
  assert.deepEqual(
    loadModel(modelBytes("initializers.onnx"), "initializers.onnx").graph(),
    graph,
  );
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
