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
const { runModel, RunSession } = require("../dist/run");
const { runProcess } = require("../dist/export");

const python = process.platform === "win32" ? "python" : "python3";
const runner = path.resolve(__dirname, "../media/run_model.py");
const run = (model, request, progress = () => {}) =>
  runModel(model, [], request, python, runner, progress);

for (const engine of ["onnx-light", "onnxruntime"])
  test(`${engine} Run retains real sessions with fresh inputs and invalidates ownership/settings`, async (t) => {
    const session = new RunSession();
    t.after(() => session.close());
    const model = loadModel(
      readFileSync(require.resolve("./models/inference.onnx")),
      "inference.onnx",
    );
    const request = {
      engine,
      keepLoaded: true,
      dimensions: { "symbol:N": 1 },
      inputs: ["float[1,3] x = {-1, 0, 2}"],
    };
    const predict = (
      owner = model,
      options = request,
      signal = new AbortController().signal,
    ) =>
      runModel(owner, [], options, python, runner, () => {}, signal, session);
    const first = await predict();
    assert.equal(first.lifecycle.reused, false);
    assert.equal(first.lifecycle.retained, true);
    const fresh = await predict(model, {
      ...request,
      dimensions: { "symbol:N": 2 },
      inputs: ["float[2,3] x = {3, 4, 5, 6, 7, 8}"],
    });
    assert.equal(fresh.lifecycle.reused, true);
    assert.equal(fresh.lifecycle.retained, true);
    assert.notDeepEqual(fresh.outputs, first.outputs);
    assert.equal(fresh.lifecycle.loadMs, first.lifecycle.loadMs);
    await session.close();
    assert.equal((await predict()).lifecycle.reused, false);
    const settings =
      engine === "onnx-light" ? { verbose: 1 } : { logSeverity: 3 };
    const configured = { ...request, settings };
    assert.equal((await predict(model, configured)).lifecycle.reused, false);
    assert.equal((await predict(model, configured)).lifecycle.reused, true);
    const changed = model.clone();
    assert.equal((await predict(changed, configured)).lifecycle.reused, false);
    assert.equal((await predict(changed, configured)).lifecycle.reused, true);
    const oneShot = await predict(changed, {
      ...configured,
      keepLoaded: false,
    });
    assert.equal(oneShot.lifecycle.retained, false);
    assert.equal(oneShot.lifecycle.reused, false);
    assert.equal((await predict(changed, configured)).lifecycle.reused, false);
    await assert.rejects(
      predict(changed, { ...configured, inputs: ["float[1,3] x = {1}"] }),
      /./,
    );
    assert.equal((await predict(changed, configured)).lifecycle.reused, false);
    const controller = new AbortController();
    const pending = predict(changed, configured, controller.signal);
    queueMicrotask(() => controller.abort());
    await assert.rejects(pending, { name: "AbortError" });
    assert.equal((await predict(changed, configured)).lifecycle.reused, false);
  });

for (const engine of ["onnx-light", "onnxruntime"])
  test(`${engine} retained pipeline sessions invalidate when switching stages`, async (t) => {
    const session = new RunSession();
    t.after(() => session.close());
    const model = await loadModelFile(
      require.resolve("./models/tiny.pipe-onnx"),
      "tiny.pipe-onnx",
    );
    for (const stage of ["embedding", "prefill", "embedding"]) {
      const request = {
        engine,
        keepLoaded: true,
        dimensions: { "symbol:batch": 1, "symbol:sequence": 1 },
        inputs: [
          stage === "embedding"
            ? "int64[1,1] input_ids = {0}"
            : "float[1,1,3] inputs_embeds = {1, 2, 3}",
        ],
      };
      for (const reused of [false, true]) {
        const result = await runModel(
          model,
          [{ pipeline: stage }],
          request,
          python,
          runner,
          () => {},
          new AbortController().signal,
          session,
        );
        assert.equal(result.lifecycle.reused, reused);
        assert.equal(result.lifecycle.retained, true);
      }
    }
  });

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
    assert.equal(random.inputs[0].name, "input");
    assert.match(random.inputs[0].text, /Values: \[/);
    assert.equal(random.lifecycle.reused, false);
    assert.equal(random.lifecycle.retained, false);
    assert.ok(random.lifecycle.loadMs >= 0);
    await writeFile(file, random.inputData[0]);
    const replay = await run(model, {
      engine,
      inputs: [{ name: "input", source: "protobuf", path: file }],
    });
    assert.deepEqual(replay.inputs, random.inputs);
    assert.deepEqual(replay.outputs, random.outputs);
    assert.deepEqual(model.graph(), before);
    await writeFile(file, prepared.inputs[0]);
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

test("saved inputs replay exact int64 values beyond JavaScript precision in both engines", async (t) => {
  const model = loadBackendTest(
    listBackendTests().find((item) => item.name === "test_cc_identity_int64")
      .id,
  );
  const directory = await mkdtemp(path.join(tmpdir(), "viewer-exact-input-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const file = path.join(directory, "input.pb");
  const text = model
    .runInputs()[0]
    .text.replace("{}", "{9007199254740993, -9007199254740993, 0, 1, 2, 3}");
  for (const engine of ["onnx-light", "onnxruntime"]) {
    const result = await run(model, { engine, inputs: [text] });
    assert.match(result.inputs[0].text, /9007199254740993, -9007199254740993/);
    await writeFile(file, result.inputData[0]);
    const replay = await run(model, {
      engine,
      inputs: [{ name: result.inputs[0].name, source: "protobuf", path: file }],
    });
    assert.deepEqual(replay.inputs, result.inputs);
    assert.deepEqual(replay.outputs, result.outputs);
    assert.deepEqual(
      Buffer.from(replay.inputData[0]),
      Buffer.from(result.inputData[0]),
    );
    assert.equal(result.outputs[0].canSave, true);
    await writeFile(file, result.outputData[0]);
    const fromOutput = await run(model, {
      engine,
      inputs: [{ name: result.inputs[0].name, source: "protobuf", path: file }],
    });
    assert.deepEqual(fromOutput.outputs, result.outputs);
  }
});

test("output export retains complete tensors beyond the preview limit in both engines", async () => {
  const model = loadModel(
    readFileSync(require.resolve("./models/inference.onnx")),
    "inference.onnx",
  );
  for (const engine of ["onnx-light", "onnxruntime"]) {
    const result = await run(model, {
      engine,
      dimensions: { "symbol:N": 64 },
      inputs: [{ name: "x", source: "random" }],
    });
    assert.match(result.outputs[0].text, /preview limit: 32/);
    const size = Number(
      execFileSync(
        python,
        [
          "-c",
          "import sys, onnx; from onnx import numpy_helper; t = onnx.TensorProto(); t.ParseFromString(sys.stdin.buffer.read()); print(numpy_helper.to_array(t).size)",
        ],
        { input: result.outputData[0], encoding: "utf8" },
      ),
    );
    assert.ok(size > 32);
  }
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
