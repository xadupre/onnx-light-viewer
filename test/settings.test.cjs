const assert = require("node:assert/strict");
const { execFileSync } = require("node:child_process");
const { mkdirSync, mkdtempSync, rmSync, writeFileSync } = require("node:fs");
const { tmpdir } = require("node:os");
const path = require("node:path");
const { test } = require("node:test");
const { inspectPythonPackages, pythonPackages } = require("../dist/settings");

test("inspects real Python metadata without importing optional model libraries", async () => {
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

test("detects the yobx distribution and preserves the selected environment's version", async (t) => {
  const directory = mkdtempSync(path.join(tmpdir(), "viewer-python-metadata-"));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  const environment = path.join(directory, "venv");
  execFileSync(process.platform === "win32" ? "python" : "python3", [
    "-m",
    "venv",
    "--without-pip",
    "--system-site-packages",
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
