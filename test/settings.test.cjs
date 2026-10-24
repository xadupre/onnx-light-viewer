const assert = require("node:assert/strict");
const { execFileSync } = require("node:child_process");
const {
  existsSync,
  mkdirSync,
  mkdtempSync,
  rmSync,
  writeFileSync,
} = require("node:fs");
const { tmpdir } = require("node:os");
const path = require("node:path");
const { test } = require("node:test");
const { inspectPythonPackages, pythonPackages } = require("../dist/settings");

function setPythonPath(t, value) {
  const previous = process.env.PYTHONPATH;
  process.env.PYTHONPATH = value;
  t.after(() => {
    if (previous === undefined) delete process.env.PYTHONPATH;
    else process.env.PYTHONPATH = previous;
  });
}

test("inspects real Python metadata without importing optional model libraries", async (t) => {
  setPythonPath(t, "");
  const executable = process.platform === "win32" ? "python" : "python3";
  const info = await inspectPythonPackages(executable);
  assert.ok(info.executable);
  assert.match(info.version, /^\d+\.\d+\.\d+/);
  assert.deepEqual(
    info.packages.map(({ name }) => name),
    pythonPackages,
  );
  const expected = JSON.parse(
    execFileSync(
      executable,
      [
        "-I",
        "-c",
        'import importlib.metadata as m, json, re; print(json.dumps([[re.sub(r"[-_.]+", "-", d.metadata["Name"]).lower(), d.version] for d in m.distributions() if d.metadata["Name"]]))',
      ],
      { encoding: "utf8" },
    ),
  );
  for (const pkg of info.packages)
    assert.equal(
      pkg.version,
      expected.find(([name]) => name === pkg.name)?.[1] ?? null,
    );
});

function pythonEnvironment(t, includeSystem = false) {
  const directory = mkdtempSync(path.join(tmpdir(), "viewer-python-metadata-"));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  const environment = path.join(directory, "venv");
  execFileSync(process.platform === "win32" ? "python" : "python3", [
    "-m",
    "venv",
    "--without-pip",
    ...(includeSystem ? ["--system-site-packages"] : []),
    environment,
  ]);
  const python = path.join(
    environment,
    process.platform === "win32" ? "Scripts/python.exe" : "bin/python",
  );
  const site = execFileSync(
    python,
    ["-I", "-c", 'import sysconfig; print(sysconfig.get_path("purelib"))'],
    { encoding: "utf8" },
  ).trim();
  return { directory, python, site };
}

test("detects the yobx distribution and preserves the selected environment's version", async (t) => {
  setPythonPath(t, "");
  const { directory, python, site } = pythonEnvironment(t, true);
  const fallback = path.join(directory, "fallback");
  for (const [root, name, version] of [
    [site, "onnx", "99.0"],
    [fallback, "onnx", "1.0"],
    [site, "yet-another-onnx-builder", "0.2.3"],
  ]) {
    const metadata = path.join(
      root,
      `${name.replaceAll("-", "_")}-${version}.dist-info`,
    );
    mkdirSync(metadata, { recursive: true });
    writeFileSync(
      path.join(metadata, "METADATA"),
      `Metadata-Version: 2.1\nName: ${name}\nVersion: ${version}\n`,
    );
  }
  writeFileSync(path.join(site, "fallback.pth"), `${fallback}\n`);
  const expected = execFileSync(
    python,
    ["-I", "-c", 'import importlib.metadata as m; print(m.version("onnx"))'],
    { encoding: "utf8" },
  ).trim();
  assert.equal(expected, "99.0");
  const info = await inspectPythonPackages(python);
  assert.equal(
    info.packages.find(({ name }) => name === "onnx").version,
    expected,
  );
  assert.equal(
    info.packages.find(({ name }) => name === "yet-another-onnx-builder")
      ?.version,
    "0.2.3",
  );
});

test("reads virtualenv and path-only metadata without executing Python startup hooks", async (t) => {
  setPythonPath(t, "");
  const { directory, python, site } = pythonEnvironment(t);
  const extra = path.join(directory, "extra");
  const metadata = path.join(extra, "onnx-99.0.dist-info");
  mkdirSync(metadata, { recursive: true });
  writeFileSync(
    path.join(metadata, "METADATA"),
    "Metadata-Version: 2.1\nName: onnx\nVersion: 99.0\n",
  );
  const local = path.join(site, "torch-98.0.dist-info");
  mkdirSync(local, { recursive: true });
  writeFileSync(
    path.join(local, "METADATA"),
    "Metadata-Version: 2.1\nName: torch\nVersion: 98.0\n",
  );
  const pthMarker = path.join(directory, "pth-executed");
  const siteMarker = path.join(directory, "sitecustomize-executed");
  writeFileSync(
    path.join(site, "inspection.pth"),
    [
      "# Metadata path, not executable code",
      path.relative(site, extra),
      `import pathlib; pathlib.Path(${JSON.stringify(pthMarker)}).touch(); print("pth startup output")`,
      `import\tpathlib; pathlib.Path(${JSON.stringify(pthMarker)}).touch()`,
      "",
    ].join("\n"),
  );
  writeFileSync(
    path.join(site, "sitecustomize.py"),
    `from pathlib import Path\nPath(${JSON.stringify(siteMarker)}).touch()\nprint("sitecustomize startup output")\n`,
  );
  const info = await inspectPythonPackages(python);
  assert.equal(
    info.packages.find(({ name }) => name === "onnx").version,
    "99.0",
  );
  assert.equal(
    info.packages.find(({ name }) => name === "torch").version,
    "98.0",
  );
  assert.equal(
    info.packages.find(({ name }) => name === "yet-another-onnx-builder")
      .version,
    null,
  );
  assert.equal(existsSync(pthMarker), false);
  assert.equal(existsSync(siteMarker), false);
});

test("inspects PYTHONPATH metadata and source versions without importing packages", async (t) => {
  const { directory, python, site } = pythonEnvironment(t);
  const first = path.join(directory, "first");
  const second = path.join(directory, "second");
  const marker = path.join(directory, "source-imported");
  for (const [root, name, value] of [
    [first, "onnx_light", '"99.2"'],
    [second, "onnx_light", '"1.0"'],
    [first, "yobx", '"98.4"'],
    [first, "torch", "calculate_version()"],
  ]) {
    const packageDirectory = path.join(root, name);
    mkdirSync(packageDirectory, { recursive: true });
    writeFileSync(
      path.join(packageDirectory, "__init__.py"),
      `from pathlib import Path\nPath(${JSON.stringify(marker)}).touch()\n__version__ = ${value}\n`,
    );
  }
  for (const [root, name, version] of [
    [first, "skl2onnx", "97.3"],
    [site, "onnx-light", "0.1"],
  ]) {
    const metadata = path.join(root, `${name}-${version}.dist-info`);
    mkdirSync(metadata, { recursive: true });
    writeFileSync(
      path.join(metadata, "METADATA"),
      `Metadata-Version: 2.1\nName: ${name}\nVersion: ${version}\n`,
    );
  }
  setPythonPath(t, [first, second].join(path.delimiter));
  const info = await inspectPythonPackages(python);
  const packages = new Map(info.packages.map((pkg) => [pkg.name, pkg]));
  assert.deepEqual(packages.get("onnx-light"), {
    name: "onnx-light",
    version: "99.2",
    source: path.join(first, "onnx_light"),
  });
  assert.equal(packages.get("yet-another-onnx-builder").version, "98.4");
  assert.equal(packages.get("skl2onnx").version, "97.3");
  assert.equal(packages.get("torch").version, null);
  assert.equal(packages.get("torch").source, path.join(first, "torch"));
  assert.equal(existsSync(marker), false);
});
