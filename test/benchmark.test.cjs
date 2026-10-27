const assert = require("node:assert/strict");
const { execFile } = require("node:child_process");
const { mkdtemp, mkdir, readFile, rm, writeFile } = require("node:fs/promises");
const { tmpdir } = require("node:os");
const path = require("node:path");
const { promisify } = require("node:util");
const { test } = require("node:test");
const { exportHub } = require("./helpers/export-hub.cjs");

const execute = promisify(execFile);
const python = process.platform === "win32" ? "python" : "python3";
const benchmark = path.join(
  __dirname,
  "../scripts/benchmark_qwen35_exporters.py",
);

test("benchmark selects the default or requested checkpoint for all five exporters and reports the selected model", async (t) => {
  const root = await mkdtemp(path.join(tmpdir(), "viewer-benchmark-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const hub = await exportHub(t);
  const cache = path.join(root, "cache");
  const recipeRevision = "a".repeat(40);
  const recipeDirectory = path.join(
    cache,
    "benchmark-olive-recipes",
    recipeRevision,
    "cpu_and_mobile",
  );
  await mkdir(recipeDirectory, { recursive: true });
  await writeFile(
    path.join(recipeDirectory, "text.json"),
    JSON.stringify({
      input_model: { type: "HfModel", model_path: "Qwen/Qwen3.5-4B" },
      passes: {
        m: {
          type: "ModelBuilder",
          precision: "int4",
          extra_options: { prune_lm_head: true },
        },
        t: {
          type: "GraphSurgeries",
          surgeries: [{ surgeon: "TieWordEmbeddings" }],
        },
      },
    }),
  );
  const entryPoint = `
import json
import os
from pathlib import Path
import sys

args = sys.argv[1:]
assert os.environ["HF_HUB_OFFLINE"] == "1"
if args[0] == "run":
    config = json.loads(Path(args[args.index("--run-config") + 1]).read_text())
    source = Path(config["input_model"]["model_path"])
    assert config["passes"]["m"]["precision"] == "fp32"
    assert "prune_lm_head" not in config["passes"]["m"]["extra_options"]
else:
    flag = "--config" if args[0] == "build" else "-m" if args[0] == "capture-onnx-graph" else "-i"
    source = Path(args[args.index(flag) + 1])
assert source.name == ${JSON.stringify(hub.revision)}
model = json.loads((source / "config.json").read_text())["modelId"]
print("source-model=" + model, flush=True)
print("Fixture stops before model creation and ONNX validation.", flush=True)
sys.exit(23)
`;
  for (const module of [
    "modelbuilder",
    "onnxruntime_genai",
    "onnxruntime_genai/models",
    "mobius",
    "olive",
  ]) {
    await mkdir(path.join(root, module), { recursive: true });
    await writeFile(path.join(root, module, "__init__.py"), "");
  }
  for (const filename of [
    "modelbuilder/builder.py",
    "onnxruntime_genai/models/builder.py",
    "mobius/__main__.py",
    "olive/__main__.py",
  ])
    await writeFile(path.join(root, filename), entryPoint);

  const env = {
    ...process.env,
    PYTHONPATH: root,
    HF_ENDPOINT: hub.endpoint,
  };
  const common = [
    benchmark,
    "--cache-dir",
    cache,
    "--recipe-revision",
    recipeRevision,
    "--profiles",
    "cpu-fp32",
  ];
  for (const model of ["Qwen/Qwen3.5-4B", "owner/sharded"]) {
    const output = path.join(root, model.replace("/", "-") + ".json");
    const args = [...common, "--output", output];
    if (model !== "Qwen/Qwen3.5-4B") args.push("--model", model);
    await assert.rejects(execute(python, args, { env }), { code: 1 });
    const report = JSON.parse(await readFile(output, "utf8"));
    assert.equal(report.model, model);
    assert.equal(report.arguments.model, model);
    assert.equal(report.status, "failed");
    assert.equal(report.model_download.revision, hub.revision);
    assert.equal(report.model_download.cache_hit, false);
    assert.equal(
      path.basename(path.dirname(path.dirname(report.model_download.source))),
      "models--" + model.replace("/", "--"),
    );
    assert.deepEqual(
      report.results.map(({ exporter }) => exporter),
      ["mbext", "modelbuilder", "mobius", "olive", "olive-recipes"],
    );
    for (const row of report.results) {
      assert.equal(row.status, "failed");
      assert.match(row.error, /Exporter exited with status 23/);
      assert.ok(row.log.startsWith(`source-model=${model}\n`), row.log);
      if (row.exporter === "olive-recipes")
        assert.equal(
          row.recipe_config.input_model.model_path,
          report.model_download.source,
        );
      else assert.ok(row.command.includes(report.model_download.source));
    }
  }
  const downloads = hub.requests.filter(
    ({ method, pathname }) =>
      method === "GET" && pathname.includes("/resolve/"),
  );
  assert.equal(
    downloads.filter(({ pathname }) => pathname.startsWith("/Qwen/Qwen3.5-4B/"))
      .length,
    2,
  );
  assert.equal(
    downloads.filter(({ pathname }) => pathname.startsWith("/owner/sharded/"))
      .length,
    4,
  );
  assert.ok(downloads.every(({ pathname }) => pathname.includes(hub.revision)));

  const requestCount = hub.requests.length;
  const invalidOutput = path.join(root, "invalid.json");
  await assert.rejects(
    execute(
      python,
      [...common, "--model", "../invalid", "--output", invalidOutput],
      { env },
    ),
    { code: 1 },
  );
  const invalid = JSON.parse(await readFile(invalidOutput, "utf8"));
  assert.equal(invalid.status, "failed");
  assert.match(invalid.error, /Repo id/);
  assert.deepEqual(invalid.results, []);
  assert.equal(hub.requests.length, requestCount);
});
