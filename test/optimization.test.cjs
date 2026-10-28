const assert = require("node:assert/strict");
const { execFileSync, spawnSync } = require("node:child_process");
const {
  mkdtemp,
  readFile,
  readdir,
  rm,
  writeFile,
} = require("node:fs/promises");
const path = require("node:path");
const { test } = require("node:test");
const { optimizeWithOnnxScript } = require("../dist/optimization");

const python =
  process.env.ONNX_SCRIPT_PYTHON ??
  (process.platform === "win32" ? "python" : "python3");
const available =
  spawnSync(python, ["-P", "-c", "import onnx, onnxscript"]).status === 0;

test(
  "ONNX Script runs real default optimization and fusion without changing input",
  {
    skip:
      !available && "Install onnx and onnxscript or set ONNX_SCRIPT_PYTHON.",
  },
  async (t) => {
    const directory = await mkdtemp(path.join(__dirname, "onnxscript-"));
    t.after(() => rm(directory, { recursive: true, force: true }));
    const source = path.join(directory, "source.onnx");
    execFileSync(python, [
      "-P",
      "-c",
      `
import sys
import onnx
import onnx.helper as oh
from onnx import TensorProto
graph = oh.make_graph([
    oh.make_node("Identity", ["x"], ["identity"]),
    oh.make_node("Relu", ["identity"], ["y"]),
], "optimizer", [oh.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
   [oh.make_tensor_value_info("y", TensorProto.FLOAT, [2])])
onnx.save(oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)]), sys.argv[1])
`,
      source,
    ]);
    const original = await readFile(source);
    const result = await optimizeWithOnnxScript(original, python, directory);
    assert.equal(result.changed, true);
    const output = path.join(directory, "result.onnx");
    await writeFile(output, result.bytes);
    const nodes = JSON.parse(
      execFileSync(
        python,
        [
          "-P",
          "-c",
          `
import json, sys, onnx
model = onnx.load(sys.argv[1])
onnx.checker.check_model(model)
print(json.dumps([node.op_type for node in model.graph.node]))
`,
          output,
        ],
        { encoding: "utf8" },
      ),
    );
    assert.deepEqual(nodes, ["Relu"]);
    const unchanged = await optimizeWithOnnxScript(
      result.bytes,
      python,
      path.relative(process.cwd(), directory),
    );
    assert.equal(unchanged.changed, false);
    assert.deepEqual(await readFile(source), original);
    assert.deepEqual((await readdir(directory)).sort(), [
      "result.onnx",
      "source.onnx",
    ]);
  },
);

test("ONNX Script removes its private working directory after process failure", async (t) => {
  const directory = await mkdtemp(path.join(__dirname, "onnxscript-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  await assert.rejects(
    optimizeWithOnnxScript(
      new Uint8Array([0]),
      path.join(directory, "missing-python"),
      directory,
    ),
    /ONNX Script optimization failed.*configured Python environment/s,
  );
  assert.deepEqual(await readdir(directory), []);
});

test(
  "ONNX Script fuses Conv and BatchNormalization with equivalent outputs",
  {
    skip:
      !available && "Install onnx and onnxscript or set ONNX_SCRIPT_PYTHON.",
  },
  async (t) => {
    const directory = await mkdtemp(path.join(__dirname, "onnxscript-"));
    t.after(() => rm(directory, { recursive: true, force: true }));
    const source = path.join(directory, "source.onnx");
    execFileSync(python, [
      "-P",
      "-c",
      `
import sys
import onnx
import onnx.helper as oh
from onnx import TensorProto
from onnxscript import rewriter

weights = [oh.make_tensor("w", TensorProto.FLOAT, [1, 1, 1, 1], [2.0])]
weights.extend(oh.make_tensor(name, TensorProto.FLOAT, [1], [value])
               for name, value in (("scale", 3.0), ("bias", 4.0),
                                   ("mean", 1.0), ("variance", 2.0)))
graph = oh.make_graph([
    oh.make_node("Conv", ["x", "w"], ["conv"]),
    oh.make_node("BatchNormalization",
                 ["conv", "scale", "bias", "mean", "variance"], ["y"]),
], "fusion", [oh.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 3, 3])],
   [oh.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 3, 3])], weights)
model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)])
assert [node.op_type for node in rewriter.rewrite(model).graph.node] == ["Conv"]
onnx.save(model, sys.argv[1])
`,
      source,
    ]);
    const original = await readFile(source);
    const result = await optimizeWithOnnxScript(original, python, directory);
    assert.equal(result.changed, true);
    const output = path.join(directory, "result.onnx");
    await writeFile(output, result.bytes);
    execFileSync(python, [
      "-P",
      "-c",
      `
import sys
import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator

source, output = (onnx.load(name) for name in sys.argv[1:])
onnx.checker.check_model(output)
assert [node.op_type for node in source.graph.node] == ["Conv", "BatchNormalization"]
assert [node.op_type for node in output.graph.node] == ["Conv"]
feeds = {"x": np.arange(9, dtype=np.float32).reshape(1, 1, 3, 3)}
expected = ReferenceEvaluator(source).run(None, feeds)[0]
actual = ReferenceEvaluator(output).run(None, feeds)[0]
np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
`,
      source,
      output,
    ]);
    assert.deepEqual(await readFile(source), original);
  },
);

test(
  "ONNX Script rejects unsupported extended ONNX data at the Python boundary",
  {
    skip:
      !available && "Install onnx and onnxscript or set ONNX_SCRIPT_PYTHON.",
  },
  async (t) => {
    const directory = await mkdtemp(path.join(__dirname, "onnxscript-"));
    t.after(() => rm(directory, { recursive: true, force: true }));
    const standard = await readFile(require.resolve("./models/relu.onnx"));
    // Unknown protobuf field 19000, encoded as a varint, is otherwise ignored.
    const extended = Buffer.concat([
      standard,
      Buffer.from([0xc0, 0xa3, 0x09, 0x01]),
    ]);
    for (const bytes of [
      await readFile(require.resolve("../examples/quantized-2bit.onnx")),
      extended,
    ]) {
      await assert.rejects(
        optimizeWithOnnxScript(bytes, python, directory),
        /unsupported ONNX protobuf extensions/,
      );
    }
    assert.deepEqual(await readdir(directory), []);
  },
);

test(
  "ONNX Script honors configured source packages on PYTHONPATH",
  {
    skip:
      !available && "Install onnx and onnxscript or set ONNX_SCRIPT_PYTHON.",
  },
  async (t) => {
    const directory = await mkdtemp(path.join(__dirname, "onnxscript-"));
    t.after(() => rm(directory, { recursive: true, force: true }));
    const environment = JSON.parse(
      execFileSync(
        python,
        [
          "-P",
          "-c",
          `
import json, os, sys
print(json.dumps({
    "executable": sys._base_executable,
    "paths": [path for path in sys.path if os.path.isabs(path)],
}))
`,
        ],
        { encoding: "utf8" },
      ),
    );
    const previous = process.env.PYTHONPATH;
    try {
      process.env.PYTHONPATH = environment.paths.join(path.delimiter);
      const result = await optimizeWithOnnxScript(
        await readFile(require.resolve("./models/relu.onnx")),
        environment.executable,
        directory,
      );
      assert.equal(result.changed, false);
    } finally {
      if (previous === undefined) delete process.env.PYTHONPATH;
      else process.env.PYTHONPATH = previous;
    }
    assert.deepEqual(await readdir(directory), []);
  },
);
