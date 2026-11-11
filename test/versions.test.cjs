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
const { changeModelVersions } = require("../dist/versions");

const python =
  process.env.ONNX_SCRIPT_PYTHON ??
  (process.platform === "win32" ? "python" : "python3");
const available = spawnSync(python, ["-P", "-c", "import onnx"]).status === 0;
const optional = {
  skip: !available && "Install onnx or set ONNX_SCRIPT_PYTHON.",
};

async function workspace(t) {
  const directory = await mkdtemp(path.join(__dirname, "versions-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  return directory;
}

test("ONNX versions reject invalid input before executing Python", async (t) => {
  const directory = await workspace(t);
  for (const versions of [
    {},
    { opset: 0 },
    { irVersion: -1 },
    { opset: 1.5 },
    { opset: "18" },
    { irVersion: null },
    { irVersion: Number.MAX_SAFE_INTEGER + 1 },
  ]) {
    await assert.rejects(
      changeModelVersions(
        new Uint8Array(),
        versions,
        "missing-python",
        directory,
      ),
      /positive safe integer|Specify/,
    );
  }
  await assert.rejects(
    changeModelVersions(
      new Uint8Array(),
      { opset: 18 },
      path.join(directory, "missing-python"),
      directory,
    ),
    /configured Python environment/,
  );
  assert.deepEqual(await readdir(directory), []);
});

test(
  "ONNX versions really convert upgrade/downgrade semantics and preserve unspecified versions",
  optional,
  async (t) => {
    const directory = await workspace(t);
    const source = path.join(directory, "source.onnx");
    execFileSync(python, [
      "-P",
      "-c",
      `
import sys, onnx
import onnx.helper as oh
from onnx import TensorProto
graph = oh.make_graph([oh.make_node("ReduceSum", ["x"], ["y"], axes=[1])],
    "versions", [oh.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])],
    [oh.make_tensor_value_info("y", TensorProto.FLOAT, [2, 1])])
model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 12), oh.make_opsetid("custom", 7)])
model.ir_version = 8
onnx.save(model, sys.argv[1])
`,
      source,
    ]);
    const original = await readFile(source);
    let bytes = original;
    for (const [versions, expectedOpset, expectedIR, changed] of [
      [{ opset: 13 }, 13, 8, true],
      [{ opset: 12 }, 12, 8, true],
      [{ irVersion: 9 }, 12, 9, true],
      [{ irVersion: 8 }, 12, 8, true],
      [{ opset: 12, irVersion: 8 }, 12, 8, false],
    ]) {
      const result = await changeModelVersions(
        bytes,
        versions,
        python,
        directory,
      );
      assert.equal(result.changed, changed);
      assert.equal(result.opset, expectedOpset);
      assert.equal(result.irVersion, expectedIR);
      const output = path.join(directory, "output.onnx");
      await writeFile(output, result.bytes);
      execFileSync(python, [
        "-P",
        "-c",
        `
import sys, onnx
import numpy as np
from onnx.reference import ReferenceEvaluator
model = onnx.load(sys.argv[1])
onnx.checker.check_model(model)
assert next(item.version for item in model.opset_import if item.domain == "custom") == 7
node = next(node for node in model.graph.node if node.op_type == "ReduceSum")
if int(sys.argv[2]) == 13:
    assert len(node.input) == 2 and not any(a.name == "axes" for a in node.attribute)
else:
    assert len(node.input) == 1 and any(a.name == "axes" for a in node.attribute)
np.testing.assert_equal(ReferenceEvaluator(model).run(None, {"x": np.arange(6, dtype=np.float32).reshape(2, 3)})[0],
                       np.array([[3], [12]], dtype=np.float32))
`,
        output,
        String(expectedOpset),
      ]);
      bytes = result.bytes;
    }
    assert.deepEqual(await readFile(source), original);
    assert.deepEqual((await readdir(directory)).sort(), [
      "output.onnx",
      "source.onnx",
    ]);
  },
);

test(
  "ONNX versions reject unknown extensions and checker/converter failures without artifacts",
  optional,
  async (t) => {
    const directory = await workspace(t);
    const standard = await readFile(require.resolve("./models/relu.onnx"));
    const extended = Buffer.concat([
      standard,
      Buffer.from([0xc0, 0xa3, 0x09, 0x01]),
    ]);
    await assert.rejects(
      changeModelVersions(extended, { opset: 18 }, python, directory),
      /unsupported ONNX protobuf extensions/,
    );
    await assert.rejects(
      changeModelVersions(standard, { opset: 9999 }, python, directory),
      /ONNX version conversion failed/,
    );
    await assert.rejects(
      changeModelVersions(standard, { irVersion: 1 }, python, directory),
      /ONNX version conversion failed/,
    );
    await assert.rejects(
      changeModelVersions(
        new Uint8Array([0]),
        { opset: 18 },
        python,
        directory,
      ),
      /ONNX version conversion failed/,
    );
    assert.deepEqual(await readdir(directory), []);
  },
);
