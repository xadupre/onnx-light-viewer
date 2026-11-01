const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { mkdtemp, writeFile, rm } = require("node:fs/promises");
const { tmpdir } = require("node:os");
const path = require("node:path");
const { test } = require("node:test");
const {
  loadModel,
  listBackendTests,
  loadBackendTest,
} = require("../dist/graph");
const { runModel } = require("../dist/run");

const python = process.platform === "win32" ? "python" : "python3";
const runner = path.resolve(__dirname, "../media/run_model.py");
const run = (model, request, progress = () => {}) =>
  runModel(model, [], request, python, runner, progress);

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
    /Specify dynamic dimension: N/,
  );
  await assert.rejects(
    run(model, {
      inputs: [{ name: "x", source: "protobuf", path: "relative.pb" }],
      dimensions: { N: 2 },
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
