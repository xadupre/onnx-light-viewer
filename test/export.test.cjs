const assert = require("node:assert/strict");
const { execFile } = require("node:child_process");
const {
  mkdtemp,
  mkdir,
  readFile,
  readdir,
  rm,
  writeFile,
} = require("node:fs/promises");
const { tmpdir } = require("node:os");
const path = require("node:path");
const { promisify } = require("node:util");
const { test } = require("node:test");
const {
  defaultExportDirectory,
  ExportHistory,
  exportPlan,
  prepareExportPlan,
  runExport,
} = require("../dist/export");

const python = process.platform === "win32" ? "python" : "python3";
const options = (directory, exporter = "mbext") => ({
  exporter,
  model: "owner/model",
  directory,
  precision: "fp32",
  provider: "cpu",
});
async function directory(t) {
  const root = await mkdtemp(path.join(tmpdir(), "viewer-export-' spaced-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  return root;
}

test("persists bounded, deduplicated export suggestions", async () => {
  const root = path.join(tmpdir(), "downloads");
  const preferences = new Map();
  const storage = {
    get: (key) => structuredClone(preferences.get(key)),
    update: async (key, value) => {
      await new Promise((resolve) => setImmediate(resolve));
      preferences.set(key, structuredClone(value));
    },
  };
  const history = new ExportHistory(storage);
  await Promise.all(
    Array.from({ length: 52 }, (_, index) =>
      history.remember(
        exportPlan(
          {
            ...options(path.join(root, String(index))),
            model: `owner/model-${index}`,
          },
          python,
        ),
      ),
    ),
  );
  const recent = exportPlan(
    { ...options(path.join(root, "3"), "mobius"), model: "owner/model-3" },
    python,
  );
  await history.remember(recent);
  const restored = new ExportHistory(storage).snapshot();
  assert.equal(restored.exporter, "mobius");
  assert.equal(restored.models.length, 50);
  assert.equal(restored.directories.length, 50);
  assert.equal(restored.models[0], recent.model);
  assert.equal(restored.directories[0], recent.directory);
  assert.equal(new Set(restored.models).size, 50);
  assert.ok(restored.models.includes("owner/model-51"));
  assert.ok(!restored.models.includes("owner/model-0"));
  preferences.set("exportHistory", { models: [null], directories: [] });
  assert.throws(() => history.snapshot(), /Invalid saved export history/);
  await assert.rejects(
    history.remember(recent),
    /Invalid saved export history/,
  );
  preferences.set("exportHistory", { models: [], directories: [] });
  assert.deepEqual(history.snapshot(), { models: [], directories: [] });
  preferences.set("exportHistory", {
    models: [],
    directories: [],
    exporter: "invalid",
  });
  assert.throws(() => history.snapshot(), /Invalid saved conversion tool/);
  preferences.delete("exportHistory");
  await history.remember(recent);
  assert.deepEqual(history.snapshot(), {
    exporter: "mobius",
    models: [recent.model],
    directories: [recent.directory],
  });
});

test("defaults to model/exporter folders and passes every listed precision/device combination to the CLI", () => {
  const root = path.join(tmpdir(), "downloads");
  for (const exporter of ["mbext", "modelbuilder", "mobius"])
    for (const precision of ["fp32", "fp16", "bf16", "int4", "int2", "fp4"])
      for (const provider of ["cpu", "cuda"]) {
        const plan = exportPlan(
          {
            ...options(defaultExportDirectory(root), exporter),
            precision,
            provider,
          },
          python,
        );
        assert.equal(
          plan.directory,
          path.join(root, "owner", "model", exporter),
        );
        const flag = exporter === "mobius" ? "--dtype" : "-p";
        const dtype =
          exporter === "mobius" && precision === "fp32"
            ? "f32"
            : exporter === "mobius" && precision === "fp16"
              ? "f16"
              : precision;
        assert.equal(plan.args[plan.args.indexOf(flag) + 1], dtype);
        assert.equal(
          plan.args[
            plan.args.indexOf(exporter === "mobius" ? "--ep" : "-e") + 1
          ],
          provider,
        );
        assert.equal(
          plan.args.includes("--extra_options"),
          exporter === "mbext",
        );
        if (exporter === "mbext")
          assert.deepEqual(plan.args.slice(-2), [
            "--extra_options",
            "hf_remote=false",
          ]);
      }
});

test("expands model and exporter directory templates without allowing model path traversal", async (t) => {
  const root = await directory(t);
  const template = path.join("base", "<model_id>", "<exporter>");
  for (const exporter of ["mbext", "modelbuilder", "mobius"]) {
    const plan = exportPlan(options(template, exporter), python, root);
    const expected = path.join(root, "base", "owner", "model", exporter);
    assert.equal(plan.directory, expected);
    assert.equal(plan.directoryTemplate, template);
    assert.ok(plan.args.includes(expected));
    assert.ok(!plan.command.includes("<model_id>"));
    assert.ok(!plan.command.includes("<exporter>"));
    const prepared = await prepareExportPlan(
      plan,
      new AbortController().signal,
    );
    if (exporter === "mobius") {
      assert.notEqual(prepared.script, path.join(expected, "export_model.py"));
      assert.equal(
        path.dirname(path.dirname(prepared.script)),
        path.dirname(prepared.directory),
      );
      assert.equal(prepared.directory, expected);
      await assert.rejects(readdir(prepared.directory), { code: "ENOENT" });
      await mkdir(prepared.directory);
    } else {
      assert.equal(prepared.script, path.join(expected, "export_model.py"));
    }
    await writeFile(prepared.script, "Previous conversion");
    const retry = await prepareExportPlan(plan, new AbortController().signal);
    assert.equal(retry.directoryTemplate, template);
    if (exporter === "mobius") {
      assert.equal(path.dirname(path.dirname(retry.directory)), expected);
      assert.equal(path.dirname(retry.script), path.dirname(retry.directory));
      assert.equal(path.basename(retry.directory), path.basename(expected));
    } else {
      assert.equal(path.dirname(retry.directory), expected);
      assert.equal(retry.script, path.join(retry.directory, "export_model.py"));
    }
    const values = new Map();
    const history = new ExportHistory({
      get: (key) => values.get(key),
      update: async (key, value) => values.set(key, value),
    });
    await history.remember(retry);
    assert.deepEqual(history.snapshot().directories, [
      template,
      retry.directory,
    ]);
  }
  const absolute = path.join(root, "<exporter>", "<model_id>", "<exporter>");
  const local = exportPlan(
    {
      ...options(absolute, "modelbuilder"),
      model: path.join(root, "local checkpoint"),
    },
    python,
  );
  assert.equal(
    local.directory,
    path.join(root, "modelbuilder", "local checkpoint", "modelbuilder"),
  );
  assert.ok(local.args.includes("-i"));
  for (const model of [
    "../escape",
    "owner/../../escape",
    "owner\\..\\escape",
    "owner//model",
  ])
    assert.throws(
      () => exportPlan({ ...options(template), model }, python, root),
      /directory template/,
    );
  for (const directory of ["base/<unknown>", "base/<model_id"])
    assert.throws(
      () => exportPlan(options(directory), python, root),
      /support only/,
    );
});

test("shows plain command arguments without quotes and quotes only shell-sensitive values", () => {
  const destination =
    process.platform === "win32" ? "C:\\models\\output" : "/tmp/output";
  const plan = exportPlan(options(destination), python);
  assert.equal(
    plan.command,
    `${process.platform === "win32" ? "& " : ""}${[python, ...plan.args].join(" ")}`,
  );
  const spaced = exportPlan(
    options(path.join(destination, "space folder")),
    python,
  );
  assert.ok(
    spaced.command.includes(`-o '${path.join(destination, "space folder")}'`),
  );
  const quoted = exportPlan(
    { ...options(destination), model: "owner/model's" },
    python,
  );
  assert.ok(
    quoted.command.includes(
      process.platform === "win32" ? "'owner/model''s'" : "'owner/model'\\''s'",
    ),
  );
  const expansion = exportPlan(
    { ...options(destination), model: "$HOME/model" },
    python,
  );
  assert.ok(expansion.command.includes("'$HOME/model'"));
});

test("builds the three documented exporter CLIs without interpreting model IDs as shell code", () => {
  const destination = path.join(tmpdir(), "export folder");
  for (const [exporter, module] of [
    ["mbext", "modelbuilder.builder"],
    ["modelbuilder", "onnxruntime_genai.models.builder"],
    ["mobius", "mobius"],
  ]) {
    const model = "owner/model'; echo not-a-command";
    const plan = exportPlan(
      { ...options(destination, exporter), model },
      python,
    );
    assert.deepEqual(plan.args.slice(0, 3), ["-u", "-m", module]);
    assert.equal(plan.args.filter((argument) => argument === model).length, 1);
    assert.ok(plan.args.includes(destination));
    assert.ok(plan.command.includes(module));
    assert.equal(plan.script, path.join(destination, "export_model.py"));
    if (exporter === "mbext")
      assert.deepEqual(plan.args.slice(-2), [
        "--extra_options",
        "hf_remote=false",
      ]);
    const local = exportPlan(
      {
        ...options(destination, exporter),
        model: path.join(tmpdir(), "checkpoint"),
      },
      python,
    );
    assert.ok(local.args.includes(exporter === "mobius" ? "--config" : "-i"));
  }
  for (const change of [
    { exporter: "unknown" },
    { exporter: ["mobius"] },
    { directory: "relative" },
    { model: "--trust-remote-code" },
    { model: "model\nanother-command" },
    { precision: null },
  ])
    assert.throws(() =>
      exportPlan({ ...options(destination), ...change }, python),
    );
});

test("retries in fresh subfolders and preserves prior conversion files for all three exporters", async (t) => {
  const root = await directory(t);
  for (const exporter of ["mbext", "modelbuilder", "mobius"]) {
    const signal = new AbortController().signal;
    const original = exportPlan(
      options(path.join(root, exporter), exporter),
      python,
    );
    const empty = await prepareExportPlan(original, signal);
    assert.equal(empty.directory, original.directory);
    await writeFile(empty.script, "Previous conversion script");
    if (exporter === "mobius") await mkdir(original.directory);
    await writeFile(
      path.join(original.directory, "weights.bin"),
      "Previous model weights",
    );
    const [first, second] = await Promise.all([
      prepareExportPlan(original, signal),
      prepareExportPlan(original, signal),
    ]);
    assert.notEqual(first.directory, second.directory);
    for (const prepared of [first, second]) {
      if (exporter === "mobius") {
        assert.ok(
          path
            .basename(path.dirname(prepared.directory))
            .startsWith(`${exporter}-`),
        );
        assert.equal(
          path.dirname(path.dirname(prepared.directory)),
          original.directory,
        );
        assert.equal(
          path.dirname(prepared.script),
          path.dirname(prepared.directory),
        );
        await assert.rejects(readdir(prepared.directory), { code: "ENOENT" });
      } else {
        assert.ok(path.basename(prepared.directory).startsWith(`${exporter}-`));
        assert.equal(path.dirname(prepared.directory), original.directory);
        assert.deepEqual(await readdir(prepared.directory), []);
      }
      const flag = exporter === "mobius" ? "--output" : "-o";
      assert.equal(
        prepared.args[prepared.args.indexOf(flag) + 1],
        prepared.directory,
      );
      if (exporter !== "mobius") {
        assert.equal(
          prepared.script,
          path.join(prepared.directory, "export_model.py"),
        );
      }
      assert.equal(
        prepared.command,
        exportPlan({ ...options(prepared.directory, exporter) }, python)
          .command,
      );
    }
    assert.equal(
      await readFile(empty.script, "utf8"),
      "Previous conversion script",
    );
    assert.equal(
      await readFile(path.join(original.directory, "weights.bin"), "utf8"),
      "Previous model weights",
    );
  }
});

test("streams a real exporter process, preserves model bytes and writes an executable reproduction script", async (t) => {
  const root = await directory(t);
  const destination = path.join(root, "output");
  const source = path.join(__dirname, "models", "relu.onnx");
  const plan = exportPlan(options(destination), python);
  plan.args = [
    "-u",
    "-c",
    `
import pathlib
import shutil
import sys
import time
print("Starting export", flush=True)
time.sleep(0.2)
print("x" * 100000)
shutil.copyfile(sys.argv[2], pathlib.Path(sys.argv[1]) / "model.onnx")
print("Saved ONNX", flush=True)
`,
    destination,
    source,
  ];
  const output = [];
  const models = await runExport(plan, new AbortController().signal, (text) =>
    output.push(text),
  );
  assert.deepEqual(models, [path.join(destination, "model.onnx")]);
  assert.deepEqual(await readFile(models[0]), await readFile(source));
  assert.ok(output.some((text) => text.includes("Starting export")));
  assert.match(output.at(-1), /Saved ONNX/);
  assert.ok(output.every((text) => text.length <= 65536));
  const script = await readFile(plan.script, "utf8");
  assert.ok(script.includes(JSON.stringify(source)));
  await promisify(execFile)(python, [plan.script]);
  assert.deepEqual(await readFile(models[0]), await readFile(source));
  await assert.rejects(
    runExport(plan, new AbortController().signal, () => {}),
    /no longer empty/,
  );
});

test("streams piped download progress readably and preserves progress settings in the reproduction script", async (t) => {
  const root = await directory(t);
  const destination = path.join(root, "download");
  const plan = exportPlan(options(destination), python);
  plan.args = [
    "-u",
    "-c",
    `
import os
import pathlib
import shutil
import sys
import time
print("Downloading checkpoint", flush=True)
if os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS") == "0" and os.environ.get("TQDM_POSITION") == "-1":
    for percent in (0, 50, 100):
        sys.stderr.write("\\r\\x1b[32mmodel.safetensors: %d%%\\x1b[0m\\x1b[A" % percent)
        sys.stderr.flush()
        time.sleep(0.3)
shutil.copyfile(sys.argv[2], pathlib.Path(sys.argv[1]) / "model.onnx")
print("\\nConverted", flush=True)
`,
    destination,
    path.join(__dirname, "models", "relu.onnx"),
  ];
  const updates = [];
  await runExport(plan, new AbortController().signal, (text) =>
    updates.push(text),
  );
  assert.ok(
    updates.some((text) => text.includes("50%") && !text.includes("100%")),
  );
  assert.match(updates.at(-1), /model\.safetensors: 100%/);
  assert.ok(updates.every((text) => !/[\r\x1b]/.test(text)));
  assert.match(updates.at(-1), /Downloading checkpoint/);
  const replay = await promisify(execFile)(python, [plan.script], {
    env: {
      ...process.env,
      HF_HUB_DISABLE_PROGRESS_BARS: "1",
      TQDM_POSITION: "0",
    },
  });
  assert.match(replay.stderr, /model\.safetensors: 100%/);
});

test("reports failed, missing and output-less exporter processes instead of loading partial models", async (t) => {
  const root = await directory(t);
  for (const [name, code, reason] of [
    [
      "failed",
      "raise RuntimeError('export failed explicitly')",
      /code 1.*export failed explicitly/s,
    ],
    ["empty", "print('no files')", /produced no ONNX model/],
    ["missing", "", /ENOENT/],
  ]) {
    const plan = exportPlan(options(path.join(root, name)), python);
    plan.args = ["-u", "-c", code];
    if (name === "missing") plan.executable = path.join(root, "missing-python");
    await assert.rejects(
      runExport(plan, new AbortController().signal, () => {}),
      reason,
    );
  }
});

test("cancelling terminates the actual exporter before returning and keeps its reproduction script", async (t) => {
  const root = await directory(t);
  const plan = exportPlan(options(path.join(root, "cancelled")), python);
  plan.args = [
    "-u",
    "-c",
    "import os, signal, time\nif os.name != 'nt': signal.signal(signal.SIGTERM, signal.SIG_IGN)\nprint(os.getpid(), flush=True)\ntime.sleep(30)",
  ];
  const controller = new AbortController();
  let pid;
  await assert.rejects(
    runExport(plan, controller.signal, (output) => {
      if (controller.signal.aborted) return;
      pid = Number(output.trim());
      controller.abort();
    }),
    { name: "AbortError" },
  );
  assert.ok(pid > 0);
  assert.throws(() => process.kill(pid, 0), { code: "ESRCH" });
  assert.match(await readFile(plan.script, "utf8"), /subprocess.run/);
});
