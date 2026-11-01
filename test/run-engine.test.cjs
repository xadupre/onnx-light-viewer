const assert = require("node:assert/strict");
const { execFileSync } = require("node:child_process");
const { readFileSync } = require("node:fs");
const { mkdtemp, writeFile, rm } = require("node:fs/promises");
const { tmpdir } = require("node:os");
const path = require("node:path");
const { test } = require("node:test");
const {
  loadModel,
  loadModelFile,
  listBackendTests,
  loadBackendTest,
} = require("../dist/graph");
const { runModel } = require("../dist/run");
const { runProcess } = require("../dist/export");

const python = process.platform === "win32" ? "python" : "python3";
const runner = path.resolve(__dirname, "../media/run_model.py");
const run = (model, request, progress = () => {}) =>
  runModel(model, [], request, python, runner, progress);

test("numeric-looking symbolic dimensions remain required for both engines", async () => {
  const bytes = execFileSync(python, [
    "-c",
    `
import onnx
import sys
model = onnx.load(sys.argv[1])
model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = "3"
sys.stdout.buffer.write(model.SerializeToString())
`,
    require.resolve("./models/inference.onnx"),
  ]);
  const model = loadModel(bytes, "numeric-dimension.onnx");
  assert.deepEqual(model.runInputs()[0].shape, ["3", "3"]);
  assert.deepEqual(model.runInputs()[0].dynamicAxes, [0]);
  for (const engine of ["onnx-light", "onnxruntime"]) {
    const request = { engine, inputs: [{ name: "x", source: "random" }] };
    await assert.rejects(
      run(model, request),
      /Specify dynamic dimension: symbol:3/,
    );
    assert.equal(
      (
        await run(model, {
          ...request,
          dimensions: { "symbol:3": 2 },
        })
      ).outputs.length,
      1,
    );
  }
});

test("symbolic and unnamed dimensions remain independent for every input source", async (t) => {
  const bytes = execFileSync(python, [
    "-c",
    `
import onnx
import sys
from onnx import TensorProto
import onnx.helper as oh
input = oh.make_tensor_value_info("X", TensorProto.FLOAT, ["X:1", None])
output = oh.make_tensor_value_info("Y", TensorProto.FLOAT, ["X:1", None])
graph = oh.make_graph([oh.make_node("Identity", ["X"], ["Y"])], "graph", [input], [output])
model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
model.ir_version = 10
sys.stdout.buffer.write(model.SerializeToString())
`,
  ]);
  const model = loadModel(bytes, "colliding-dimensions.onnx");
  assert.deepEqual(model.runInputs()[0].shape, ["X:1", null]);
  assert.deepEqual(model.runInputs()[0].dynamicAxes, [0, 1]);
  const dimensions = { "symbol:X:1": 2, "axis:X:1": 3 };
  const text = "float[2,3] X = {1, 2, 3, 4, 5, 6}";
  const prepared = await model.prepareRun([text]);
  const directory = await mkdtemp(path.join(tmpdir(), "viewer-run-test-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const file = path.join(directory, "input.pb");
  await writeFile(file, prepared.inputs[0]);

  for (const engine of ["onnx-light", "onnxruntime"]) {
    const result = await run(model, {
      engine,
      dimensions,
      inputs: [text],
    });
    assert.match(
      result.outputs[0].text,
      /Values: \[1(?:\.0)?, 2(?:\.0)?, 3(?:\.0)?, 4(?:\.0)?, 5(?:\.0)?, 6(?:\.0)?\]/,
    );
    const fromFile = await run(model, {
      engine,
      dimensions,
      inputs: [{ name: "X", source: "protobuf", path: file }],
    });
    assert.match(
      fromFile.outputs[0].text,
      /Values: \[1(?:\.0)?, 2(?:\.0)?, 3(?:\.0)?, 4(?:\.0)?, 5(?:\.0)?, 6(?:\.0)?\]/,
    );
    const random = await run(model, {
      engine,
      dimensions,
      inputs: [{ name: "X", source: "random" }],
    });
    assert.equal(random.outputs.length, 1);
  }
});

test("Run honors panel-owned cancellation before starting execution", async () => {
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(
    runModel(null, [], {}, python, runner, () => {}, controller.signal),
    { name: "AbortError" },
  );
});

test("ONNX Runtime bounds long string previews but preserves full tensors", async (t) => {
  const directory = await mkdtemp(
    path.join(tmpdir(), "viewer-run-string-preview-"),
  );
  t.after(() => rm(directory, { recursive: true, force: true }));
  const long = "x".repeat(100_000);
  execFileSync(python, [
    "-c",
    `
import json
import onnx
import onnx.helper as oh
from onnx import TensorProto
import pathlib
import sys

directory = pathlib.Path(sys.argv[1])
input_info = oh.make_tensor_value_info("X", TensorProto.STRING, [1])
output_info = oh.make_tensor_value_info("Y", TensorProto.STRING, [1])
graph = oh.make_graph([oh.make_node("Identity", ["X"], ["Y"])], "strings", [input_info], [output_info])
model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
model.ir_version = 10
onnx.save(model, directory / "model.onnx")
tensor = oh.make_tensor("X", TensorProto.STRING, [1], ["x" * ${long.length}])
(directory / "input-0.pb").write_bytes(tensor.SerializeToString())
(directory / "settings.json").write_text(json.dumps({"logSeverity": 3, "logVerbosity": 0, "inputs": 1}))
`,
    directory,
  ]);
  execFileSync(python, [runner, directory]);
  const result = JSON.parse(
    readFileSync(path.join(directory, "result.json"), "utf8"),
  );
  for (const tensor of [result.inputs[0], result.outputs[0]]) {
    assert.ok(tensor.text.length < 500);
    assert.match(tensor.text, /…|b'\.\.\.'/);
    assert.doesNotMatch(tensor.text, /x{1000}/);
  }
  const size = execFileSync(python, [
    "-c",
    "import onnx, sys; print(len(onnx.TensorProto.FromString(open(sys.argv[1], 'rb').read()).string_data[0]))",
    path.join(directory, "output-0.pb"),
  ])
    .toString()
    .trim();
  assert.equal(size, "100000");
});

test("Run validates engine logging, dimensions and input-source transport", async () => {
  const model = loadModel(
    readFileSync(require.resolve("./models/inference.onnx")),
    "inference.onnx",
  );
  await assert.rejects(run(model, { inputs: [], engine: "unknown" }), /engine/);
  await assert.rejects(
    run(model, { inputs: [], settings: { verbose: 11 } }),
    /verbose/,
  );
  await assert.rejects(
    run(model, {
      inputs: [],
      engine: "onnxruntime",
      settings: { logSeverity: 5 },
    }),
    /logSeverity/,
  );
  await assert.rejects(
    run(model, { inputs: [], dimensions: { N: 1.5 } }),
    /safe integers/,
  );
  await assert.rejects(
    run(model, { inputs: [] }),
    /Specify dynamic dimension: symbol:N/,
  );
  await assert.rejects(
    run(
      { runInputs: () => [{ name: "x", shape: ["3"], dynamicAxes: [0] }] },
      { inputs: [] },
    ),
    /Specify dynamic dimension: symbol:3/,
  );
  await assert.rejects(
    run(model, {
      inputs: [{ name: "x", source: "protobuf", path: "relative.pb" }],
      dimensions: { "symbol:N": 2 },
    }),
    /absolute/,
  );
});

test("both engines execute protobuf and random inputs from independent current-model snapshots", async (t) => {
  const directory = await mkdtemp(path.join(tmpdir(), "viewer-run-test-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const model = loadModel(
    readFileSync(require.resolve("./models/relu.onnx")),
    "relu.onnx",
  );
  const before = model.graph();
  const prepared = await model.prepareRun(["float[1,3] input = {-1, 0, 2}"]);
  const file = path.join(directory, "input.pb");
  await writeFile(file, prepared.inputs[0]);
  for (const engine of ["onnx-light", "onnxruntime"]) {
    const result = await run(model, {
      engine,
      inputs: [{ name: "input", source: "protobuf", path: file }],
    });
    assert.match(
      result.outputs[0].text,
      /Values: \[0(?:\.0)?, 0(?:\.0)?, 2(?:\.0)?\]/,
    );
    const random = await run(model, {
      engine,
      inputs: [{ name: "input", source: "random" }],
    });
    assert.equal(random.outputs[0].name, "output");
    assert.deepEqual(model.graph(), before);
  }
});

test("both engines execute unsaved function edits without mutating the retained model or original file", async () => {
  const file = require.resolve("./models/relu.onnx");
  const originalBytes = readFileSync(file);
  const source = await loadModelFile(file, "relu.onnx");
  source.createFunctionPattern([0], "EditableRun");
  const preview = await source.previewFusion("EditableRun");
  const fused = await preview.model.applyFusion("EditableRun", [
    preview.matches[0].anchor,
  ]);
  const definition = fused.modelFunction(0);
  const edited = fused.updateModelFunction(
    0,
    definition.text.replace("Relu", "Neg"),
  );
  const graph = edited.graph();
  const bytes = Buffer.from(await edited.serialize());
  for (const engine of ["onnx-light", "onnxruntime"]) {
    const result = await run(edited, {
      engine,
      inputs: ["float[1,3] input = {-1, 0, 2}"],
    });
    assert.match(
      result.outputs[0].text,
      /Values: \[1(?:\.0)?, -?0(?:\.0)?, -2(?:\.0)?\]/,
    );
    assert.deepEqual(edited.graph(), graph);
    assert.deepEqual(Buffer.from(await edited.serialize()), bytes);
    assert.deepEqual(readFileSync(file), originalBytes);
  }
});

test("onnx-light forwards SDK event diagnostics only when verbosity is enabled", async () => {
  const model = loadModel(
    readFileSync(require.resolve("./models/relu.onnx")),
    "relu.onnx",
  );
  const request = {
    engine: "onnx-light",
    inputs: ["float[1,3] input = {-1, 0, 2}"],
  };
  let quietLogs = "";
  const quiet = await run(model, request, (output) => {
    quietLogs = output;
  });
  assert.equal(quietLogs, "");
  let logs = "";
  const verbose = await run(
    model,
    { ...request, settings: { verbose: 1 } },
    (output) => {
      logs = output;
    },
  );
  assert.match(logs, /Relu/);
  assert.match(logs, /run.?node/i);
  assert.deepEqual(verbose.outputs, quiet.outputs);
});

test("ONNX Runtime transport decodes UTF-16 stderr while preserving UTF-8 stdout and failure details", async () => {
  let logs = "";
  const stdout = "stdout caf\u00e9";
  const stderr = "stderr caf\u00e9 \u6f22 \ud83d\ude00";
  await assert.rejects(
    runProcess(
      {
        executable: python,
        args: [
          "-u",
          "-c",
          `
import sys
import time
sys.stdout.buffer.write("stdout caf\\u00e9\\n".encode("utf-8"))
sys.stdout.flush()
sys.stderr.reconfigure(encoding="utf-16-le", errors="backslashreplace")
message = "stderr caf\\u00e9 \\u6f22 \\U0001f600\\n".encode(sys.stderr.encoding)
sys.stderr.buffer.write(message[:1])
sys.stderr.flush()
time.sleep(0.02)
sys.stderr.buffer.write(message[1:])
sys.stderr.flush()
raise RuntimeError("Python failure: caf\\u00e9 \\U0001f600")
`,
        ],
      },
      new AbortController().signal,
      (output) => {
        logs = output;
      },
      {},
      "ONNX Runtime",
      undefined,
      "utf16le",
    ),
    (error) => {
      assert.match(error.message, /ONNX Runtime exited with code 1/);
      assert.ok(error.message.includes(stdout));
      assert.ok(error.message.includes(stderr));
      assert.match(error.message, /Traceback/);
      assert.ok(
        error.message.includes("Python failure: caf\u00e9 \ud83d\ude00"),
      );
      return true;
    },
  );
  assert.ok(logs.includes(stdout));
  assert.ok(logs.includes(stderr));
  assert.equal(logs.includes("\0"), false);
});

test("ONNX Runtime preserves int64 precision and exposes severity/verbosity logs", async () => {
  const model = loadBackendTest(
    listBackendTests().find((item) => item.name === "test_cc_identity_int64")
      .id,
  );
  const input = model
    .runInputs()[0]
    .text.replace("{}", "{9007199254740993, -9007199254740993, 0, 1, 2, 3}");
  let logs = "";
  const result = await run(
    model,
    {
      engine: "onnxruntime",
      settings: { logSeverity: 0, logVerbosity: 1 },
      inputs: [input],
    },
    (output) => {
      logs = output;
    },
  );
  assert.match(result.outputs[0].text, /9007199254740993, -9007199254740993/);
  assert.match(logs, /onnxruntime|Session Options/i);
});
