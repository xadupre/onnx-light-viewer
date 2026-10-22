const assert = require("node:assert/strict");
const { test } = require("node:test");
const { onnx } = require("onnx-proto");
const { decodeGraph, demoGraph } = require("../dist/graph");

function encode(graph) {
  return onnx.ModelProto.encode(
    onnx.ModelProto.create({ irVersion: 9, graph }),
  ).finish();
}

test("the one-node demo has the same structure as a decoded ONNX graph", () => {
  const bytes = encode({
    name: demoGraph.name,
    input: [{ name: "input" }],
    output: [{ name: "output" }],
    node: [
      {
        name: "activation",
        opType: "Relu",
        input: ["input"],
        output: ["output"],
      },
    ],
  });
  assert.deepEqual(decodeGraph(bytes, "demo.onnx"), demoGraph);
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
