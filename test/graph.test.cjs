const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const path = require("node:path");
const { test } = require("node:test");
const { onnx } = require("onnx-proto");
const { decodeGraph, demoGraph } = require("../dist/graph");

function encode(graph) {
  return onnx.ModelProto.encode(
    onnx.ModelProto.create({ irVersion: 9, graph }),
  ).finish();
}

test("loads a real one-node ONNX model file matching the demo", () => {
  const bytes = readFileSync(path.join(__dirname, "models", "relu.onnx"));
  assert.deepEqual(decodeGraph(bytes, "relu.onnx"), demoGraph);
});

test("preserves unnamed and custom operators, omits optional ports and initializer inputs", () => {
  const graph = decodeGraph(
    encode({
      input: [{ name: "x" }, { name: "weight" }, { name: "sparse" }],
      output: [{ name: "y" }],
      initializer: [{ name: "weight" }],
      sparseInitializer: [{ values: { name: "sparse" } }],
      node: [
        {
          opType: "Add",
          input: ["x", "weight", ""],
          output: ["intermediate", ""],
        },
        {
          opType: "Custom",
          domain: "example",
          input: ["intermediate"],
          output: ["y"],
        },
      ],
    }),
    "model.onnx",
  );
  assert.equal(graph.name, "model.onnx");
  assert.deepEqual(graph.inputs, ["x"]);
  assert.deepEqual(graph.outputs, ["y"]);
  assert.equal(graph.initializerCount, 2);
  assert.equal(graph.nodes[0].name, "Add_1");
  assert.deepEqual(graph.nodes[0].inputs, ["x", "weight"]);
  assert.deepEqual(graph.nodes[0].outputs, ["intermediate"]);
  assert.equal(graph.nodes[1].domain, "example");
  assert.notEqual(graph.nodes[0].id, graph.nodes[1].id);
});

test("rejects malformed protobuf, missing model metadata and missing operators", () => {
  assert.throws(() => decodeGraph(Uint8Array.from([255]), "bad.onnx"));
  assert.throws(
    () => decodeGraph(new Uint8Array(), "empty.onnx"),
    /ONNX model graph/,
  );
  assert.throws(
    () =>
      decodeGraph(onnx.ModelProto.encode({ graph: {} }).finish(), "no-ir.onnx"),
    /ONNX model graph/,
  );
  assert.throws(
    () => decodeGraph(encode({ node: [{}] }), "bad-node.onnx"),
    /operator type/,
  );
});

test("accepts a valid graph with no operators", () => {
  const graph = decodeGraph(
    encode({
      input: [{ name: "x" }],
      output: [{ name: "x" }],
    }),
    "identity.onnx",
  );
  assert.deepEqual(graph.nodes, []);
  assert.deepEqual(graph.inputs, graph.outputs);
});
