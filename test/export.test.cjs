const assert = require("node:assert/strict");
const { execFile } = require("node:child_process");
const {
  mkdtemp,
  mkdir,
  readFile,
  readdir,
  realpath,
  rm,
  writeFile,
} = require("node:fs/promises");
const { tmpdir } = require("node:os");
const path = require("node:path");
const { promisify } = require("node:util");
const { test } = require("node:test");
const {
  defaultExportDirectory,
  downloadExportSource,
  ExportHistory,
  exportPlan,
  prepareExportPlan,
  runExport,
} = require("../dist/export");
const { exportHub } = require("./helpers/export-hub.cjs");

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

test("downloads a pinned source into the application cache once and converts it offline for every exporter", async (t) => {
  const root = await directory(t);
  const hub = await exportHub(t);
  const previous = process.env.HF_ENDPOINT;
  process.env.HF_ENDPOINT = hub.endpoint;
  t.after(() => {
    if (previous === undefined) delete process.env.HF_ENDPOINT;
    else process.env.HF_ENDPOINT = previous;
  });
  const cache = path.join(root, "downloads", "huggingface");
  for (const exporter of ["mbext", "modelbuilder", "mobius"]) {
    const plan = await prepareExportPlan(
      exportPlan(options(path.join(root, exporter), exporter), python),
      new AbortController().signal,
    );
    const output = [];
    const local = await downloadExportSource(
      plan,
      cache,
      new AbortController().signal,
      (text) => output.push(text),
    );
    assert.equal(
      await realpath(local.model),
      await realpath(
        path.join(cache, "models--owner--model", "snapshots", hub.revision),
      ),
    );
    assert.ok(local.args.includes(exporter === "mobius" ? "--config" : "-i"));
    assert.equal(local.script, plan.script);
    assert.match(output.at(-1), /Source ready/);
    assert.deepEqual(await readdir(local.model), [
      "config.json",
      "model.safetensors",
    ]);
    local.args = [
      "-u",
      "-c",
      `
import os
import pathlib
import shutil
import sys
assert os.environ["HF_HUB_OFFLINE"] == "1"
assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
assert (pathlib.Path(sys.argv[1]) / "model.safetensors").read_bytes() == b"source weights"
pathlib.Path(sys.argv[2]).mkdir(parents=True, exist_ok=True)
shutil.copyfile(sys.argv[3], pathlib.Path(sys.argv[2]) / "model.onnx")
`,
      local.model,
      local.directory,
      path.join(__dirname, "models", "relu.onnx"),
    ];
    await runExport(local, new AbortController().signal, () => {});
    await promisify(execFile)(python, [local.script]);
    assert.match(await readFile(local.script, "utf8"), /HF_HUB_OFFLINE/);
  }
  assert.equal(
    hub.requests.filter(
      ({ method, pathname }) =>
        method === "GET" && pathname.includes("/resolve/"),
    ).length,
    2,
  );
  assert.ok(
    hub.requests
      .filter(({ pathname }) => pathname.includes("/resolve/"))
      .every(({ pathname }) => pathname.includes(hub.revision)),
  );
  assert.ok(
    (await readdir(cache)).every((name) => !name.startsWith(".request-")),
  );
  const sharded = await downloadExportSource(
    exportPlan(
      { ...options(path.join(root, "sharded")), model: "owner/sharded" },
      python,
    ),
    cache,
    new AbortController().signal,
    () => {},
  );
  assert.deepEqual(await readdir(sharded.model), [
    "config.json",
    "model-00001.safetensors",
    "model-00002.safetensors",
    "model.safetensors.index.json",
  ]);
  const failure = exportPlan(
    { ...options(path.join(root, "failure")), model: "download-fail" },
    python,
  );
  await assert.rejects(
    downloadExportSource(
      failure,
      cache,
      new AbortController().signal,
      () => {},
    ),
    /404|Repository not found/i,
  );
  const local = exportPlan(
    { ...options(path.join(root, "local")), model: root },
    python,
  );
  const count = hub.requests.length;
  assert.equal(
    await downloadExportSource(
      local,
      cache,
      new AbortController().signal,
      () => {},
    ),
    local,
  );
  assert.equal(hub.requests.length, count);
});

test("cancels an active source download and cleans only its request metadata", async (t) => {
  const root = await directory(t);
  const controller = new AbortController();
  let downloading = false;
  const hub = await exportHub(t, (method, pathname) => {
    if (method === "GET" && pathname.endsWith("/model.safetensors")) {
      downloading = true;
      controller.abort();
    }
  });
  const previous = process.env.HF_ENDPOINT;
  process.env.HF_ENDPOINT = hub.endpoint;
  t.after(() => {
    if (previous === undefined) delete process.env.HF_ENDPOINT;
    else process.env.HF_ENDPOINT = previous;
  });
  const cache = path.join(root, "cache");
  await assert.rejects(
    downloadExportSource(
      exportPlan(options(path.join(root, "output")), python),
      cache,
      controller.signal,
      () => {},
    ),
    { name: "AbortError" },
  );
  assert.equal(downloading, true);
  assert.ok((await readdir(cache)).includes("models--owner--model"));
  assert.ok(
    (await readdir(cache)).every((name) => !name.startsWith(".request-")),
  );
});

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

test("preserves the mbext layer limit through download, output retries and reproduction scripts", async (t) => {
  const root = await directory(t);
  const hub = await exportHub(t);
  const previous = process.env.HF_ENDPOINT;
  process.env.HF_ENDPOINT = hub.endpoint;
  t.after(() => {
    if (previous === undefined) delete process.env.HF_ENDPOINT;
    else process.env.HF_ENDPOINT = previous;
  });
  const plan = exportPlan({ ...options(root), layerCount: 2 }, python);
  assert.deepEqual(plan.args.slice(-3), [
    "--extra_options",
    "hf_remote=false",
    "num_hidden_layers=2",
  ]);
  assert.match(plan.command, /num_hidden_layers=2/);
  await writeFile(path.join(root, "existing.txt"), "Keep this output");
  const retry = await prepareExportPlan(plan, new AbortController().signal);
  assert.notEqual(retry.directory, root);
  const local = await downloadExportSource(
    retry,
    path.join(root, "cache"),
    new AbortController().signal,
    () => {},
  );
  assert.equal(local.layerCount, 2);
  assert.deepEqual(local.args.slice(-3), plan.args.slice(-3));
  assert.equal(local.script, retry.script);
  local.args = [
    "-u",
    "-c",
    `import pathlib, shutil, sys
assert sys.argv[-3:] == ["--extra_options", "hf_remote=false", "num_hidden_layers=2"]
shutil.copyfile(sys.argv[1], pathlib.Path(sys.argv[2]) / "model.onnx")
`,
    path.join(__dirname, "models", "relu.onnx"),
    local.directory,
    ...local.args,
  ];
  assert.deepEqual(
    await runExport(local, new AbortController().signal, () => {}),
    [path.join(local.directory, "model.onnx")],
  );
  assert.match(await readFile(local.script, "utf8"), /num_hidden_layers=2/);
});

test("rejects invalid or unsupported exporter layer limits", () => {
  const destination = path.join(tmpdir(), "layer-limit");
  for (const layerCount of [0, -1, 1.5, NaN, Infinity, 2 ** 53, "2", null])
    assert.throws(
      () => exportPlan({ ...options(destination), layerCount }, python),
      /layer limit must be a positive integer/,
    );
  assert.throws(
    () =>
      exportPlan({ ...options(destination, "mobius"), layerCount: 2 }, python),
    /builder layer limit/,
  );
  assert.ok(
    !exportPlan(options(destination), python).args.some((arg) =>
      arg.startsWith("num_hidden_layers="),
    ),
  );
});

test("builds advanced arguments using each exporter's supported CLI", () => {
  const destination = path.join(tmpdir(), "advanced-export");
  for (const exporter of ["mbext", "modelbuilder"]) {
    const plan = exportPlan(
      {
        ...options(destination, exporter),
        precision: "int4",
        layerCount: 2,
        advancedOptions: {
          excludeEmbeds: true,
          includeHiddenStates: true,
          pruneLmHead: true,
          useQdq: true,
          blockSize: 64,
          quantizationAlgorithm: "k_quant",
          ...(exporter === "mbext" ? { opset: 20, maxShardSize: "2gb" } : {}),
        },
      },
      python,
    );
    assert.equal(
      plan.args.filter((arg) => arg === "--extra_options").length,
      1,
    );
    const extra = plan.args.slice(plan.args.indexOf("--extra_options") + 1);
    assert.deepEqual(extra, [
      ...(exporter === "mbext" ? ["hf_remote=false"] : []),
      "num_hidden_layers=2",
      "exclude_embeds=true",
      "include_hidden_states=true",
      "prune_lm_head=true",
      "use_qdq=true",
      ...(exporter === "mbext"
        ? ["int4_block_size=64", "int4_algo_config=k_quant"]
        : ["block_size=64", "algo_config=k_quant"]),
      ...(exporter === "mbext" ? ["--max-shard-size", "2GB"] : []),
    ]);
    assert.equal(plan.args.includes("--onnx_opset"), exporter === "mbext");
    if (exporter === "mbext")
      assert.equal(plan.args[plan.args.indexOf("--onnx_opset") + 1], "20");
    const headless = exportPlan(
      {
        ...options(destination, exporter),
        advancedOptions: { excludeLmHead: true, useQdq: false },
      },
      python,
    );
    assert.ok(headless.args.includes("exclude_lm_head=true"));
    assert.ok(!headless.args.includes("use_qdq=true"));
  }
  const plan = exportPlan(
    {
      ...options(destination, "mobius"),
      advancedOptions: {
        noWeights: true,
        optimize: "all",
        externalData: "safetensors",
        maxShardSize: "2GB",
        staticCache: true,
        maxSequenceLength: 256,
        runtime: "ort-genai",
      },
    },
    python,
  );
  assert.deepEqual(plan.args.slice(12), [
    "--no-weights",
    "--features",
    "static-cache",
    "--optimize",
    "all",
    "--external-data",
    "safetensors",
    "--max-seq-len",
    "256",
    "--runtime",
    "ort-genai",
    "--max-shard-size",
    "2GB",
  ]);
});

test("validates advanced option types, exporter compatibility and dependencies", () => {
  const destination = path.join(tmpdir(), "advanced-options");
  for (const [exporter, advancedOptions, precision = "int4"] of [
    ["mbext", null],
    ["mbext", []],
    ["mbext", "exclude_embeds=true"],
    ["mbext", { arbitrary: true }],
    ["mbext", { excludeEmbeds: "true" }],
    ["mbext", { blockSize: 3 }],
    ["mbext", { blockSize: "32" }],
    ["mbext", { quantizationAlgorithm: "--injected" }],
    ["mbext", { opset: 0 }],
    ["mbext", { opset: 1.5 }],
    ["mbext", { maxShardSize: "0GB" }],
    ["mbext", { maxShardSize: "2" }],
    ["mobius", { maxShardSize: "2" }],
    ["mbext", { maxShardSize: "2GB\n--reuse-weights" }],
    ["mbext", { excludeLmHead: true, includeHiddenStates: true }],
    ["mbext", { excludeLmHead: true, pruneLmHead: true }],
    ["mbext", { useQdq: true }, "fp32"],
    ["mbext", { blockSize: 32 }, "fp16"],
    ["mbext", { quantizationAlgorithm: "rtn" }, "bf16"],
    ["mbext", { noWeights: true }],
    ["modelbuilder", { opset: 20 }],
    ["modelbuilder", { maxShardSize: "2GB" }],
    ["mobius", { excludeEmbeds: true }],
    ["mobius", { blockSize: 32 }],
    ["mobius", { noWeights: "true" }],
    ["mobius", { staticCache: 1 }],
    ["mobius", { optimize: "--model" }],
    ["mobius", { externalData: "unknown" }],
    ["mobius", { runtime: "unknown" }],
    ["mobius", { maxSequenceLength: 128 }],
    ["mobius", { staticCache: true, maxSequenceLength: -1 }],
  ])
    assert.throws(
      () =>
        exportPlan(
          { ...options(destination, exporter), precision, advancedOptions },
          python,
        ),
      Error,
      `${exporter}: ${JSON.stringify(advancedOptions)}`,
    );
});

test("restricts builder quantization options to INT4", () => {
  const destination = path.join(tmpdir(), "quantization-options");
  for (const exporter of ["mbext", "modelbuilder"]) {
    for (const advancedOptions of [
      { useQdq: true },
      { blockSize: 32 },
      { quantizationAlgorithm: "rtn" },
    ]) {
      assert.throws(
        () =>
          exportPlan(
            {
              ...options(destination, exporter),
              precision: "int2",
              advancedOptions,
            },
            python,
          ),
        /Quantization options require INT4 precision/,
      );
    }
    assert.equal(
      exportPlan(
        { ...options(destination, exporter), precision: "int2" },
        python,
      ).precision,
      "int2",
    );
  }
});

test("accepts explicit shard-size units", () => {
  const destination = path.join(tmpdir(), "shard-size-options");
  for (const exporter of ["mbext", "mobius"]) {
    for (const size of ["2B", "2kb", "2MB", "2gb"]) {
      const plan = exportPlan(
        {
          ...options(destination, exporter),
          advancedOptions: { maxShardSize: size },
        },
        python,
      );
      assert.equal(
        plan.args[plan.args.indexOf("--max-shard-size") + 1],
        size.toUpperCase(),
      );
    }
  }
});

test("preserves advanced options through cached downloads, retries and reproduction", async (t) => {
  const root = await directory(t);
  const hub = await exportHub(t);
  const previous = process.env.HF_ENDPOINT;
  process.env.HF_ENDPOINT = hub.endpoint;
  t.after(() => {
    if (previous === undefined) delete process.env.HF_ENDPOINT;
    else process.env.HF_ENDPOINT = previous;
  });
  for (const exporter of ["mbext", "modelbuilder", "mobius"]) {
    const advancedOptions =
      exporter === "mobius"
        ? { noWeights: true, optimize: "skip_norm", maxShardSize: "1GB" }
        : {
            excludeEmbeds: true,
            excludeLmHead: true,
            blockSize: 64,
            quantizationAlgorithm: "k_quant",
          };
    const plan = exportPlan(
      {
        ...options(path.join(root, exporter), exporter),
        precision: "int4",
        advancedOptions,
      },
      python,
    );
    await mkdir(plan.directory);
    await writeFile(path.join(plan.directory, "existing.txt"), "Keep output");
    const retry = await prepareExportPlan(plan, new AbortController().signal);
    assert.notEqual(retry.directory, plan.directory);
    const local = await downloadExportSource(
      retry,
      path.join(root, "cache"),
      new AbortController().signal,
      () => {},
    );
    assert.deepEqual(local.advancedOptions, advancedOptions);
    assert.deepEqual(local.args.slice(8), retry.args.slice(8));
    const expectedOptions =
      exporter === "mobius"
        ? ["--no-weights"]
        : exporter === "mbext"
          ? [
              "exclude_lm_head=true",
              "int4_block_size=64",
              "int4_algo_config=k_quant",
            ]
          : ["exclude_lm_head=true", "block_size=64", "algo_config=k_quant"];
    local.args = [
      "-u",
      "-c",
      `import pathlib, shutil, sys
assert all(option in sys.argv[3:] for option in ${JSON.stringify(expectedOptions)})
pathlib.Path(sys.argv[2]).mkdir(parents=True, exist_ok=True)
shutil.copyfile(sys.argv[1], pathlib.Path(sys.argv[2]) / "model.onnx")
`,
      path.join(__dirname, "models", "relu.onnx"),
      local.directory,
      ...local.args,
    ];
    await runExport(local, new AbortController().signal, () => {});
    const script = await readFile(local.script, "utf8");
    for (const option of expectedOptions) assert.ok(script.includes(option));
    await promisify(execFile)(python, [local.script]);
  }
});

test("Mobius graph-only conversion caches pinned metadata without downloading weights", async (t) => {
  const root = await directory(t);
  const hub = await exportHub(t);
  const previous = process.env.HF_ENDPOINT;
  process.env.HF_ENDPOINT = hub.endpoint;
  t.after(() => {
    if (previous === undefined) delete process.env.HF_ENDPOINT;
    else process.env.HF_ENDPOINT = previous;
  });
  const plan = exportPlan(
    { ...options(root, "mobius"), advancedOptions: { noWeights: true } },
    python,
  );
  const cache = path.join(root, "cache");
  const local = await downloadExportSource(
    plan,
    cache,
    new AbortController().signal,
    () => {},
  );
  assert.deepEqual(await readdir(local.model), ["config.json"]);
  assert.ok(local.args.includes("--no-weights"));
  assert.ok(
    hub.requests
      .filter(({ pathname }) => pathname.includes("/resolve/"))
      .every(
        ({ pathname }) =>
          pathname.endsWith("/config.json") && pathname.includes(hub.revision),
      ),
  );
  await downloadExportSource(
    exportPlan(options(root, "mobius"), python),
    cache,
    new AbortController().signal,
    () => {},
  );
  assert.deepEqual(await readdir(local.model), [
    "config.json",
    "model.safetensors",
  ]);
  const configOnly = { ...options(root, "mobius"), model: "owner/config-only" };
  const metadata = await downloadExportSource(
    exportPlan({ ...configOnly, advancedOptions: { noWeights: true } }, python),
    cache,
    new AbortController().signal,
    () => {},
  );
  assert.deepEqual(await readdir(metadata.model), ["config.json"]);
  await assert.rejects(
    downloadExportSource(
      exportPlan(configOnly, python),
      cache,
      new AbortController().signal,
      () => {},
    ),
    /No supported root checkpoint found/,
  );
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

for (const inheritedOutput of [true, false]) {
  test(
    `cancelling stops exporter workers with ${inheritedOutput ? "inherited" : "closed"} output before returning`,
    { timeout: 10000 },
    async (t) => {
      const root = await directory(t);
      const heartbeat = path.join(root, "worker-heartbeat");
      const plan = exportPlan(options(path.join(root, "cancelled")), python);
      const worker = `
import pathlib
import signal
import sys
import time
if sys.platform != "win32":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
for index in range(600):
    pathlib.Path(sys.argv[1]).write_text(str(index))
    time.sleep(0.05)
`;
      plan.args = [
        "-u",
        "-c",
        `
import os
import pathlib
import subprocess
import sys
import time
output = ${inheritedOutput ? "None" : "subprocess.DEVNULL"}
worker = subprocess.Popen([sys.executable, "-u", "-c", sys.argv[1], sys.argv[2]], stdout=output, stderr=output)
while not pathlib.Path(sys.argv[2]).exists():
    time.sleep(0.01)
print(os.getpid(), worker.pid, flush=True)
time.sleep(30)
`,
        worker,
        heartbeat,
      ];
      const controller = new AbortController();
      let pids = [];
      t.after(() => {
        for (const pid of pids) {
          try {
            process.kill(pid, "SIGKILL");
          } catch (error) {
            if (error.code !== "ESRCH") throw error;
          }
        }
      });
      await assert.rejects(
        runExport(plan, controller.signal, (output) => {
          if (controller.signal.aborted) return;
          pids = output.trim().split(/\s+/).map(Number);
          controller.abort();
        }),
        { name: "AbortError" },
      );
      assert.equal(pids.length, 2);
      assert.ok(pids.every((pid) => Number.isSafeInteger(pid) && pid > 0));
      const stopped = await readFile(heartbeat, "utf8");
      await new Promise((resolve) => setTimeout(resolve, 250));
      assert.equal(await readFile(heartbeat, "utf8"), stopped);
    },
  );
}
