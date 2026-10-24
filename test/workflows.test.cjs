const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const path = require("node:path");
const { test } = require("node:test");

function workflow(name) {
  return readFileSync(
    path.resolve(__dirname, "..", ".github", "workflows", name),
    "utf8",
  );
}

test("CI runs unit tests on Linux, macOS, and Windows without building documentation", () => {
  const source = workflow("tests.yml");
  assert.match(source, /runs-on: \$\{\{ matrix\.os \}\}/);
  assert.match(source, /os: \[ubuntu-latest, macos-latest, windows-latest\]/);
  assert.match(source, /run: npm test/);
  assert.match(source, /if: runner\.os == 'Linux'\s+run: clang-format/);
  assert.doesNotMatch(source, /sphinx|setup-python/);
});

test("documentation builds in a separate Linux-only workflow", () => {
  const source = workflow("documentation.yml");
  assert.match(source, /push:\s+branches: \[main\]/);
  assert.match(source, /pull_request:/);
  assert.match(source, /workflow_dispatch:/);
  assert.match(source, /runs-on: ubuntu-latest/);
  assert.match(source, /python -m sphinx -n -W --keep-going -b html/);
  assert.doesNotMatch(source, /matrix:|macos-|windows-/);
});

test("releases retain native packages for all supported platforms", () => {
  const source = workflow("release.yml");
  assert.match(source, /runs-on: \$\{\{ matrix\.os \}\}/);
  for (const target of [
    "linux-x64",
    "linux-arm64",
    "darwin-x64",
    "darwin-arm64",
    "win32-x64",
    "win32-arm64",
  ]) {
    assert.ok(source.includes(`target: ${target}`));
  }
  assert.doesNotMatch(source, /sphinx|setup-python/);
});

test("Windows checkouts preserve LF line endings without changing ONNX binaries", () => {
  const source = readFileSync(
    path.resolve(__dirname, "..", ".gitattributes"),
    "utf8",
  );
  assert.match(source, /^\* text=auto eol=lf$/m);
  assert.match(source, /^\*\.onnx binary$/m);
});
