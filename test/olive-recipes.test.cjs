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
const { OliveRecipes } = require("../dist/oliveRecipes");
const {
  exportPlan,
  exportNeedsDownload,
  prepareExportPlan,
  runExport,
} = require("../dist/export");

const execute = promisify(execFile);
const python = process.platform === "win32" ? "python" : "python3";
async function fixture(t) {
  const root = await mkdtemp(path.join(tmpdir(), "viewer-olive-' spaced-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const remote = path.join(root, "remote");
  await mkdir(path.join(remote, "model", "cpu"), { recursive: true });
  const git = (...args) => execute("git", ["-C", remote, ...args]);
  const commit = async () => {
    await git("add", ".");
    await git(
      "-c",
      "user.name=Recipe test",
      "-c",
      "user.email=recipe@example.invalid",
      "-c",
      "commit.gpgsign=false",
      "commit",
      "-qm",
      "Update recipe",
    );
    return (await git("rev-parse", "HEAD")).stdout.trim();
  };
  await git("init", "-q");
  await writeFile(path.join(remote, "requirements.txt"), "olive-ai\n");
  await writeFile(
    path.join(remote, "model", "cpu", "requirements.txt"),
    "olive-ai==0.9.3\nonnxruntime\n",
  );
  await writeFile(path.join(remote, "model", "cpu", "config.json"), "{}");
  await writeFile(path.join(remote, "model", "cpu", "second.json"), "{}");
  await writeFile(
    path.join(remote, "model", "cpu", "optimize.py"),
    "# manual recipe\n",
  );
  await writeFile(
    path.join(remote, "model", "cpu", "info.yaml"),
    `recipes:
  - name: Pinned model
    file: config.json
    devices: [cpu, gpu]
    eps: CPUExecutionProvider
  - file: second.json
    device: cpu
    ep: CPUExecutionProvider
  - name: Manual pinned model
    file: optimize.py
`,
  );
  const revision = await commit();
  const cache = path.join(root, "cache");
  return {
    root,
    remote,
    cache,
    revision,
    commit,
    signal: new AbortController().signal,
  };
}

test("caches every recipe including version pins and manual workflows by immutable Git commit", async (t) => {
  const { root, remote, cache, revision, commit, signal } = await fixture(t);
  const recipes = new OliveRecipes(cache, remote);
  const [first, concurrent] = await Promise.all([
    recipes.refresh(signal),
    recipes.refresh(signal),
  ]);
  assert.equal(first, concurrent);
  assert.equal(first.revision, revision);
  assert.equal(first.recipes.length, 3);
  const pinned = recipes.resolve("model/cpu/config.json", revision);
  assert.deepEqual(pinned.requirements, ["olive-ai", "olive-ai==0.9.3"]);
  assert.match(pinned.target, /cpu, gpu, CPUExecutionProvider/);
  assert.throws(
    () =>
      recipes.resolve(
        first.recipes.find((entry) => entry.manualReason).id,
        revision,
      ),
    /source instructions/,
  );
  assert.equal(await recipes.refresh(signal), first);
  const persistent = await new OliveRecipes(cache, remote).refresh(signal);
  assert.deepEqual(persistent, first);
  await writeFile(
    path.join(remote, "model", "cpu", "requirements.txt"),
    "git+https://github.com/microsoft/Olive.git@v0.10.0\n",
  );
  const updatedRevision = await commit();
  const updated = await recipes.refresh(signal);
  assert.equal(updated.revision, updatedRevision);
  assert.deepEqual(updated.recipes[0].requirements, [
    "olive-ai",
    "git+https://github.com/microsoft/Olive.git@v0.10.0",
  ]);
  assert.match(
    await readFile(
      path.join(pinned.snapshot, "model", "cpu", "requirements.txt"),
      "utf8",
    ),
    /==0.9.3/,
  );
  assert.throws(() => recipes.resolve(pinned.id, revision), /Refresh/);
  assert.throws(
    () => recipes.resolve(path.join(root, "untrusted.json"), updatedRevision),
    /Choose a recipe/,
  );
  assert.deepEqual(
    (await readdir(cache)).sort(),
    [revision, updatedRevision].sort(),
  );
});

test("reuses the persisted model list rather than reparsing an unchanged checkout, including concurrent panels", async (t) => {
  const { remote, cache, revision, signal } = await fixture(t);
  const [first, otherPanel] = await Promise.all([
    new OliveRecipes(cache, remote).refresh(signal),
    new OliveRecipes(cache, remote).refresh(signal),
  ]);
  assert.deepEqual(first, otherPanel);
  await writeFile(
    path.join(cache, revision, "source", "model", "cpu", "info.yaml"),
    "not: [valid YAML",
  );
  assert.deepEqual(
    await new OliveRecipes(cache, remote).refresh(signal),
    first,
  );
});

test("reports invalid or escaping metadata and cancellation without publishing a partial catalog", async (t) => {
  const { root, remote, cache, commit, signal } = await fixture(t);
  const info = path.join(remote, "model", "cpu", "info.yaml");
  await writeFile(
    info,
    "recipes:\n  - name: escape\n    file: ../../../outside.json\n",
  );
  await commit();
  await assert.rejects(
    new OliveRecipes(cache, remote).refresh(signal),
    /inside its snapshot/,
  );
  assert.deepEqual(await readdir(cache), []);
  await writeFile(info, "recipes: invalid\n");
  await commit();
  await assert.rejects(
    new OliveRecipes(cache, remote).refresh(signal),
    /Invalid Olive recipe metadata/,
  );
  const cancelled = new AbortController();
  cancelled.abort();
  await assert.rejects(
    new OliveRecipes(path.join(root, "cancelled"), remote).refresh(
      cancelled.signal,
    ),
    /abort/i,
  );
});

test("runs and reproduces a pinned recipe in an isolated checkout with its own relative resources and no forced offline mode", async (t) => {
  const { root, remote, cache, revision, signal } = await fixture(t);
  const recipes = new OliveRecipes(cache, remote);
  await recipes.refresh(signal);
  const recipe = recipes.resolve("model/cpu/config.json", revision);
  const destination = path.join(root, "output");
  await mkdir(destination);
  await writeFile(path.join(destination, "keep.txt"), "previous output");
  const options = {
    exporter: "olive-recipes",
    model: recipe.name,
    directory: destination,
    precision: "fp32",
    provider: "cpu",
  };
  const plan = await prepareExportPlan(
    exportPlan(options, python, undefined, recipe),
    signal,
  );
  assert.notEqual(plan.directory, destination);
  assert.equal(exportNeedsDownload(plan), false);
  assert.deepEqual(plan.args, [
    "-u",
    "-m",
    "olive",
    "run",
    "--run-config",
    path.join(plan.directory, "olive-recipe", recipe.id),
    "--output_path",
    plan.directory,
  ]);
  const packageRoot = path.join(root, "python");
  await mkdir(path.join(packageRoot, "olive"), { recursive: true });
  await writeFile(path.join(packageRoot, "olive", "__init__.py"), "");
  await writeFile(
    path.join(packageRoot, "olive", "__main__.py"),
    `
import os
import pathlib
import shutil
import sys
assert sys.argv[1] == "run"
assert os.environ.get("HF_HUB_OFFLINE", "") == ${JSON.stringify(process.env.HF_HUB_OFFLINE ?? "")}
assert os.environ.get("TRANSFORMERS_OFFLINE", "") == ${JSON.stringify(process.env.TRANSFORMERS_OFFLINE ?? "")}
assert pathlib.Path("config.json").read_text() == "{}"
assert pathlib.Path("../../requirements.txt").read_text() == "olive-ai\\n"
pathlib.Path("generated-cache").write_text("private to this conversion")
output = pathlib.Path(sys.argv[sys.argv.index("--output_path") + 1])
shutil.copyfile(${JSON.stringify(path.join(__dirname, "models", "relu.onnx"))}, output / "model.onnx")
`,
  );
  const previous = process.env.PYTHONPATH;
  process.env.PYTHONPATH = packageRoot;
  t.after(() => {
    if (previous === undefined) delete process.env.PYTHONPATH;
    else process.env.PYTHONPATH = previous;
  });
  assert.deepEqual(await runExport(plan, signal, () => {}), [
    path.join(plan.directory, "model.onnx"),
  ]);
  await execute(python, [plan.script]);
  assert.match(await readFile(plan.script, "utf8"), new RegExp(revision));
  assert.equal(
    await readFile(path.join(destination, "keep.txt"), "utf8"),
    "previous output",
  );
  assert.equal(
    (await readdir(path.join(recipe.snapshot, "model", "cpu"))).includes(
      "generated-cache",
    ),
    false,
  );
  assert.equal(
    (await readdir(path.join(plan.directory, "olive-recipe"))).includes(".git"),
    false,
  );
});

test("runs Olive quantize in a Python process with the correct local ONNX arguments for INT4 and INT8", async (t) => {
  const { root, signal } = await fixture(t);
  const source = path.join(__dirname, "models", "relu.onnx");
  const packageRoot = path.join(root, "python");
  await mkdir(path.join(packageRoot, "olive"), { recursive: true });
  await writeFile(path.join(packageRoot, "olive", "__init__.py"), "");
  await writeFile(
    path.join(packageRoot, "olive", "__main__.py"),
    `
import os
import pathlib
import shutil
import sys
args = sys.argv[1:]
assert args[0] == "quantize"
precision = args[args.index("--precision") + 1]
assert args[args.index("--algorithm") + 1] == "rtn"
assert args[args.index("--implementation") + 1] == ("olive" if precision == "int4" else "ort")
assert os.environ["HF_HUB_OFFLINE"] == "1"
assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
source = pathlib.Path(args[args.index("-m") + 1])
output = pathlib.Path(args[args.index("--output_path") + 1])
shutil.copyfile(source, output / "quantized.onnx")
`,
  );
  const previous = process.env.PYTHONPATH;
  process.env.PYTHONPATH = packageRoot;
  t.after(() => {
    if (previous === undefined) delete process.env.PYTHONPATH;
    else process.env.PYTHONPATH = previous;
  });
  const options = {
    exporter: "olive",
    model: source,
    directory: path.join(root, "output"),
    precision: "int4",
    provider: "cpu",
  };
  for (const precision of ["int4", "int8"]) {
    const plan = await prepareExportPlan(
      exportPlan({ ...options, precision }, python),
      signal,
    );
    assert.equal(exportNeedsDownload(plan), false);
    assert.deepEqual(await runExport(plan, signal, () => {}), [
      path.join(plan.directory, "quantized.onnx"),
    ]);
    await execute(python, [plan.script]);
  }
  for (const invalid of [
    { model: "owner/model" },
    { precision: "fp32" },
    { provider: "cuda" },
    { layerCount: 2 },
    { advancedOptions: { blockSize: 32 } },
  ])
    assert.throws(() => exportPlan({ ...options, ...invalid }, python));
  assert.throws(
    () => exportPlan({ ...options, exporter: "olive-recipes" }, python),
    /pinned Olive recipe/,
  );
});
