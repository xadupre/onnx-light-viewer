const assert = require("node:assert/strict");
const { execFileSync } = require("node:child_process");
const { once } = require("node:events");
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
const { runInNewContext } = require("node:vm");

const python = process.platform === "win32" ? "python" : "python3";
const runner = path.resolve(__dirname, "../media/run_model.py");
const run = (model, request, progress = () => {}) =>
  runModel(model, [], request, python, runner, progress);

function runModule(overrides) {
  const loaded = {};
  runInNewContext(readFileSync(require.resolve("../dist/run"), "utf8"), {
    exports: loaded,
    process,
    AbortController,
    require(name) {
      return (
        overrides[name] ??
        require(
          name.startsWith(".")
            ? path.resolve(__dirname, "../dist", name)
            : name,
        )
      );
    },
  });
  return loaded;
}

for (const engine of ["onnx-light", "onnxruntime"])
  test(`${engine} Run retains real sessions with fresh inputs and invalidates ownership/settings`, async (t) => {
    const session = new RunSession();
    t.after(() => session.close());
    const directory = await mkdtemp(
      path.join(tmpdir(), "viewer-run-preflight-"),
    );
    t.after(() => rm(directory, { recursive: true, force: true }));
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
    for (const invalid of [
      { ...configured, settings: { verbose: -1, logSeverity: -1 } },
      {
        ...configured,
        inputs: [
          {
            name: "x",
            source: "protobuf",
            path: path.join(directory, "missing.pb"),
          },
        ],
      },
    ]) {
      await assert.rejects(predict(changed, invalid));
      assert.equal(
        (await predict(changed, configured)).lifecycle.reused,
        false,
      );
      assert.equal((await predict(changed, configured)).lifecycle.reused, true);
    }
    const controller = new AbortController();
    const pending = predict(changed, configured, controller.signal);
    queueMicrotask(() => controller.abort());
    await assert.rejects(pending, { name: "AbortError" });
    assert.equal((await predict(changed, configured)).lifecycle.reused, false);
  });

test(
  "a Python worker exiting during result cleanup cannot restore a retained session",
  { timeout: 10000 },
  async (t) => {
    const directory = await mkdtemp(path.join(tmpdir(), "viewer-run-exit-"));
    const stop = path.join(directory, "stop");
    const worker = path.join(directory, "worker.py");
    await writeFile(
      worker,
      `
import json
import pathlib
import sys
import time

request = json.loads(sys.stdin.readline())
directory = pathlib.Path(request["input"])
(directory / "output-0.pb").write_bytes((directory / "input-0.pb").read_bytes())
(directory / "result.json").write_text(json.dumps({
    "timeMs": 1,
    "inputs": [{"name": "input", "text": "Input"}],
    "outputs": [{"name": "output", "text": "Output", "hasData": True}],
    "lifecycle": {"loadMs": 1, "reused": False, "retained": True}
}))
print("\\x1e" + json.dumps({"type": "done"}), flush=True)
while not pathlib.Path(${JSON.stringify(stop)}).exists():
    time.sleep(0.01)
`,
    );
    let exited;
    const lost = new Promise((resolve) => (exited = resolve));
    const errors = [];
    const session = new RunSession((error) => {
      errors.push(error);
      exited();
    });
    t.after(async () => {
      await writeFile(stop, "");
      await session.close();
      await rm(directory, { recursive: true, force: true });
    });
    // The real worker exits after output reads, while temporary-file cleanup awaits it.
    const delayed = {};
    runInNewContext(readFileSync(require.resolve("../dist/run"), "utf8"), {
      exports: delayed,
      require(name) {
        if (name === "node:fs/promises")
          return {
            ...require(name),
            async rm(directory, options) {
              assert.equal(session.alive, true);
              await writeFile(stop, "");
              await lost;
              await rm(directory, options);
            },
          };
        return require(
          name.startsWith(".")
            ? path.resolve(__dirname, "../dist", name)
            : name,
        );
      },
    });
    const model = loadModel(
      readFileSync(require.resolve("./models/relu.onnx")),
      "relu.onnx",
    );
    const result = await delayed.runModel(
      model,
      [],
      {
        engine: "onnxruntime",
        keepLoaded: true,
        inputs: ["float[1,3] input = {-1, 0, 2}"],
      },
      python,
      worker,
      () => {},
      new AbortController().signal,
      session,
    );
    assert.equal(errors.length, 1);
    assert.match(errors[0].message, /ONNX Runtime exited/);
    assert.equal(result.lifecycle.retained, false);
    assert.equal(session.alive, false);
    assert.deepEqual(result.outputs[0].data, result.inputs[0].data);
  },
);

test("a failed pending input read does not unload a newer Run session", async (t) => {
  const session = new RunSession();
  t.after(() => session.close());
  const directory = await mkdtemp(path.join(tmpdir(), "viewer-run-preflight-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const model = loadModel(
    readFileSync(require.resolve("./models/relu.onnx")),
    "relu.onnx",
  );
  const request = {
    keepLoaded: true,
    inputs: ["float[1,3] input = {-1, 0, 2}"],
  };
  const predict = (owner, options = request) =>
    runModel(
      owner,
      [],
      options,
      python,
      runner,
      () => {},
      new AbortController().signal,
      session,
    );
  await predict(model);
  const failure = assert.rejects(
    predict(model, {
      ...request,
      inputs: [
        {
          name: "input",
          source: "protobuf",
          path: path.join(directory, "missing.pb"),
        },
      ],
    }),
    { code: "ENOENT" },
  );
  const replacement = model.clone();
  const generation = await session.ensure(
    replacement,
    [],
    "onnx-light",
    {},
    python,
    runner,
    new AbortController().signal,
  );
  await failure;
  assert.equal(session.generation, generation);
  assert.equal((await predict(replacement)).lifecycle.reused, false);
  assert.equal((await predict(replacement)).lifecycle.reused, true);
});

for (const phase of ["write", "read", "cleanup"])
  for (const keepLoaded of [false, true])
    test(
      `ORT cancellation during ${phase} returns AbortError (retained: ${keepLoaded})`,
      { timeout: 15000 },
      async (t) => {
        const session = new RunSession();
        t.after(() => session.close());
        const controller = new AbortController();
        const fs = require("node:fs/promises");
        let cancelled = false;
        const delayed = runModule({
          "node:fs/promises": {
            ...fs,
            async writeFile(file, ...args) {
              await fs.writeFile(file, ...args);
              if (
                phase === "write" &&
                path.basename(file) === "settings.json"
              ) {
                cancelled = true;
                controller.abort();
              }
            },
            async readFile(file, ...args) {
              const value = await fs.readFile(file, ...args);
              if (phase === "read" && path.basename(file) === "result.json") {
                cancelled = true;
                controller.abort();
              }
              return value;
            },
            async rm(...args) {
              await fs.rm(...args);
              if (phase === "cleanup") {
                cancelled = true;
                controller.abort();
              }
            },
          },
        });
        const model = loadModel(
          readFileSync(require.resolve("./models/relu.onnx")),
          "relu.onnx",
        );
        await assert.rejects(
          delayed.runModel(
            model,
            [],
            {
              engine: "onnxruntime",
              keepLoaded,
              inputs: ["float[1,3] input = {-1, 0, 2}"],
            },
            python,
            runner,
            () => {},
            controller.signal,
            session,
          ),
          { name: "AbortError" },
        );
        assert.equal(cancelled, true);
        await session.close();
        assert.equal(session.alive, false);
      },
    );

for (const mode of ["idle", "active"])
  test(
    `failed ${mode} worker cleanup retains both owners and blocks replacement until retry succeeds`,
    { timeout: 15000 },
    async (t) => {
      const directory = await mkdtemp(path.join(tmpdir(), "viewer-run-close-"));
      const workerFile = path.join(directory, "worker.py");
      const stop = path.join(directory, "stop");
      await writeFile(
        workerFile,
        `
import json
import pathlib
import sys
import time

while True:
    line = sys.stdin.readline()
    if line:
        request = json.loads(line)
        event = "ready" if pathlib.Path(request["input"]).name == "active" else "done"
        print("\\x1e" + json.dumps({"type": event}), flush=True)
    elif pathlib.Path(${JSON.stringify(stop)}).exists():
        break
    else:
        time.sleep(0.01)
`,
      );
      const { terminateExport } = require("../dist/export");
      const permissionError = Object.assign(new Error("kill EPERM"), {
        code: "EPERM",
      });
      let permitTermination = false;
      const prompt = {};
      runInNewContext(
        readFileSync(require.resolve("../dist/promptSession"), "utf8"),
        {
          exports: prompt,
          process,
          Error,
          DOMException,
          setTimeout,
          clearTimeout,
          require(name) {
            return name === "./export"
              ? {
                  async terminateExport(child) {
                    if (!permitTermination) throw permissionError;
                    await terminateExport(child);
                  },
                }
              : require(name);
          },
        },
      );
      const loaded = runModule({ "./promptSession": prompt });
      const errors = [];
      const session = new loaded.RunSession((error) => errors.push(error));
      let child;
      t.after(async () => {
        permitTermination = true;
        await writeFile(stop, "");
        try {
          await session.close();
          if (child?.exitCode === null && child.signalCode === null)
            await terminateExport(child);
        } finally {
          await rm(directory, { recursive: true, force: true });
        }
      });
      const model = loadModel(
        readFileSync(require.resolve("./models/relu.onnx")),
        "relu.onnx",
      );
      const signal = new AbortController().signal;
      const ensure = () =>
        session.ensure(
          model,
          [],
          "onnxruntime",
          {},
          python,
          workerFile,
          signal,
        );
      await ensure();
      const operation = session.run(
        path.join(directory, mode),
        signal,
        () => {},
      );
      const outcome =
        mode === "active"
          ? assert.rejects(operation, (error) => error === permissionError)
          : operation;
      const worker = session.worker;
      child = worker.child;
      if (mode === "active") await once(child.stdout, "data");
      else await outcome;
      await assert.rejects(
        session.close(),
        (error) => error === permissionError,
      );
      if (mode === "active") await outcome;
      assert.equal(session.worker, worker);
      assert.equal(worker.child, child);
      assert.equal(session.alive, true);
      await assert.rejects(ensure(), (error) => error === permissionError);
      await assert.rejects(
        worker.run(
          directory,
          "unused",
          "new-key",
          signal,
          () => {},
          () => {},
        ),
        (error) => error === permissionError,
      );
      assert.equal(worker.child, child);
      permitTermination = true;
      await session.close();
      assert.equal(session.worker, undefined);
      assert.equal(worker.child, undefined);
      assert.equal(child.exitCode !== null || child.signalCode !== null, true);
      await ensure();
      await session.run(directory, signal, () => {});
      assert.notEqual(session.worker.child.pid, child.pid);
      assert.deepEqual(errors, []);
    },
  );

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
    assert.equal(random.inputs[0].name, "input");
    assert.match(random.inputs[0].text, /Values: \[/);
    assert.equal(random.lifecycle.reused, false);
    assert.equal(random.lifecycle.retained, false);
    assert.ok(random.lifecycle.loadMs >= 0);
    await writeFile(file, random.inputs[0].data);
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
    await writeFile(file, result.inputs[0].data);
    const replay = await run(model, {
      engine,
      inputs: [{ name: result.inputs[0].name, source: "protobuf", path: file }],
    });
    assert.deepEqual(replay.inputs, result.inputs);
    assert.deepEqual(replay.outputs, result.outputs);
    assert.deepEqual(
      Buffer.from(replay.inputs[0].data),
      Buffer.from(result.inputs[0].data),
    );
    assert.equal(!!result.outputs[0].data, true);
    await writeFile(file, result.outputs[0].data);
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
        { input: result.outputs[0].data, encoding: "utf8" },
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
