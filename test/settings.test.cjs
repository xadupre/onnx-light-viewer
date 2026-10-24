const assert = require("node:assert/strict");
const { execFileSync } = require("node:child_process");
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
        'import importlib.metadata as m, json; print(json.dumps({d.metadata["Name"].lower().replace("_", "-"): d.version for d in m.distributions() if d.metadata["Name"]}))',
      ],
      { encoding: "utf8" },
    ),
  );
  for (const pkg of info.packages)
    assert.equal(pkg.version, expected[pkg.name] ?? null);
});
