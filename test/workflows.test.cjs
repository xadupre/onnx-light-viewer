const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const path = require("node:path");
const { test } = require("node:test");
const { skipSdkReplay } = require("./helpers/sdk-replay.cjs");

test("SDK replay skips expire above onnx-light 0.2.3", () => {
  for (const version of [
    "0.1.10",
    "0.2.2",
    "0.2.3",
    "0.2.3-rc.1",
    "0.2.3+build",
  ])
    assert.match(skipSdkReplay(version), /xadupre\/onnx-light#5756/);
  for (const version of [
    "0.2.4",
    "0.2.4.1",
    "0.2.4-rc.1",
    "0.2.5",
    "0.2.10",
    "0.3.0",
    "1.0.0",
  ])
    assert.equal(skipSdkReplay(version), false);
  assert.throws(() => skipSdkReplay("invalid"), /Invalid onnx-light version/);
});

function workflow(name) {
  return readFileSync(
    path.resolve(__dirname, "..", ".github", "workflows", name),
    "utf8",
  );
}

const documentation = workflow("documentation.yml");

test("native builds use onnx-light 0.2.5 with pinned checksums", () => {
  const cmake = readFileSync(
    path.resolve(__dirname, "..", "CMakeLists.txt"),
    "utf8",
  );
  const version = cmake.match(/set\(ONNX_LIGHT_VERSION "([^"]+)"\)/)[1];
  assert.equal(version, "0.2.5");
  assert.equal([...cmake.matchAll(/set\(_sha256 "[a-f0-9]{64}"\)/g)].length, 5);
  for (const [asset, sha256] of [
    [
      "onnx-light-cpp-linux-x86_64.tar.gz",
      "96de7f7799f218acec77ddf5bb14771028122253bcf98ca896d36e72bac83da1",
    ],
    [
      "onnx-light-cpp-linux-aarch64.tar.gz",
      "ab1dcab710427a2366276f5dc26f4d3f46a8af1297fd9f762c8d25caa30fd04c",
    ],
    [
      "onnx-light-cpp-macos-universal2.tar.gz",
      "cc43feb08c5e962b08995ee85bea638374ac440b67601c5e0814e72020af6dd6",
    ],
    [
      "onnx-light-cpp-windows-AMD64.zip",
      "c1f8af8d402d3d1b0b95272a7816b3a64aa616655f3452d51be05eb9c1955f63",
    ],
    [
      "onnx-light-cpp-windows-ARM64.zip",
      "d475332e8fb5dc30fa611c410d5eb7d41b119f13cde364b07ab734b51f45c38a",
    ],
  ]) {
    assert.ok(
      cmake.includes(`set(_asset "${asset}")\n  set(_sha256 "${sha256}")`),
    );
  }
  assert.match(cmake, /URL_HASH "SHA256=\$\{_sha256\}"/);
  assert.match(cmake, /set\(ONNX_LIGHT_CPU_VERSION "0\.2\.5"\)/);
  assert.match(
    cmake,
    /https:\/\/github\.com\/xadupre\/onnx-light-cpu\/archive\/refs\/tags\/\$\{ONNX_LIGHT_CPU_VERSION\}\.tar\.gz/,
  );
  assert.match(
    cmake,
    /URL_HASH "SHA256=b30dc73b499b5c2c1309a74e170228dc681afd92f8eda29a0688ce6017d68f89"/,
  );
  assert.match(
    cmake,
    /target_link_libraries\(onnx_light_viewer PRIVATE[\s\S]*?\n  lib_onnx_light_cpu_kernels/,
  );
});

test("Python tooling pins both SDKs to 0.2.5 without conflicting direct URLs", () => {
  const coverage = workflow("coverage.yml");
  assert.match(coverage, /-r test\/requirements\.txt/);
  assert.doesNotMatch(
    coverage,
    /https:\/\/github\.com\/xadupre\/onnx-light\/releases\/download\//,
  );
  const requirements = readFileSync(
    path.resolve(__dirname, "requirements.txt"),
    "utf8",
  );
  assert.match(requirements, /^onnx-light==0\.2\.5$/m);
  assert.doesNotMatch(requirements, /^onnx-light\s*@/m);
  assert.match(
    requirements,
    /^--find-links https:\/\/github\.com\/xadupre\/onnx-light\/releases\/expanded_assets\/0\.2\.5$/m,
  );
  assert.match(
    requirements,
    /^--find-links https:\/\/github\.com\/xadupre\/onnx-light-cpu\/releases\/expanded_assets\/0\.2\.5$/m,
  );
  assert.match(requirements, /^onnx-light-cpu==0\.2\.5$/m);
});

test("CI runs unit tests on Linux, Windows and macOS ARM64 without building documentation", () => {
  const source = workflow("tests.yml");
  assert.match(source, /runs-on: \$\{\{ matrix\.os \}\}/);
  assert.match(source, /os: \[ubuntu-latest, windows-latest, macos-latest\]/);
  assert.match(source, /run: npm test/);
  assert.match(source, /if: runner\.os == 'Linux'\s+run: clang-format/);
  assert.doesNotMatch(source, /sphinx/);
  assert.match(source, /uses: actions\/setup-python@/);
  assert.match(source, /python -m pip install -r test\/requirements\.txt/);
});

test("documentation runs weekly or manually in a Linux-only workflow", () => {
  assert.match(documentation, /cron: "0 3 \* \* 1"/);
  assert.match(documentation, /^  workflow_dispatch:/m);
  assert.doesNotMatch(documentation, /^  (push|pull_request):/m);
  assert.match(documentation, /runs-on: ubuntu-latest/);
  assert.match(
    documentation,
    /python -m sphinx -n -W --keep-going -b html docs build\/docs\/html/,
  );
  assert.doesNotMatch(documentation, /matrix:|macos-|windows-/);
});

test("documentation uses Python 3.13 for Sphinx 9.2 compatibility", () => {
  assert.match(documentation, /python-version: "3\.13"/);
});

test("documentation landing page displays the package version", () => {
  const config = readFileSync(
    path.resolve(__dirname, "..", "docs", "conf.py"),
    "utf8",
  );
  const index = readFileSync(
    path.resolve(__dirname, "..", "docs", "index.rst"),
    "utf8",
  );
  assert.match(config, /release = json\.loads\(/);
  assert.match(
    config,
    /Path\(__file__\)\.resolve\(\)\.parents\[1\] \/ "package\.json"/,
  );
  assert.match(config, /\)\["version"\]/);
  assert.match(index, /^Version: \|release\|$/m);
});

test("releases retain Linux and Windows packages while macOS builds are paused", () => {
  const source = workflow("release.yml");
  assert.match(source, /runs-on: \$\{\{ matrix\.os \}\}/);
  for (const target of [
    "linux-x64",
    "linux-arm64",
    "win32-x64",
    "win32-arm64",
  ]) {
    assert.ok(source.includes(`target: ${target}`));
  }
  assert.doesNotMatch(source, /os: macos-|target: darwin-/);
  assert.doesNotMatch(source, /sphinx|setup-python/);
  assert.doesNotMatch(source, /vsce package[^\n]*--no-dependencies/);
});

test("README uses a PNG ONNX badge for VSIX packaging and retains the SVG source", () => {
  const readme = readFileSync(
    path.resolve(__dirname, "..", "README.md"),
    "utf8",
  );
  assert.equal(
    readme.split("[![Uses ONNX](media/onnx-badge.png)](https://onnx.ai/)")
      .length - 1,
    2,
    "Both the displayed badge and the reuse example must use the PNG",
  );
  assert.doesNotMatch(readme, /!\[Uses ONNX\]\([^)]*\.svg\)/);
  assert.match(readme, /\[the SVG badge\]\(media\/onnx-badge\.svg\)/);
  const badge = readFileSync(
    path.resolve(__dirname, "..", "media", "onnx-badge.png"),
  );
  assert.deepEqual(
    badge.subarray(0, 8),
    Buffer.from("89504e470d0a1a0a", "hex"),
  );
  assert.equal(badge.readUInt32BE(16), 88);
  assert.equal(badge.readUInt32BE(20), 22);
});

test("Windows checkouts preserve LF line endings without changing ONNX binaries", () => {
  const source = readFileSync(
    path.resolve(__dirname, "..", ".gitattributes"),
    "utf8",
  );
  assert.match(source, /^\* text=auto eol=lf$/m);
  assert.match(source, /^\*\.onnx binary$/m);
});

test("documentation failures open an issue with a link to the failed run", () => {
  assert.match(documentation, /needs: docs/);
  assert.match(
    documentation,
    /if: \$\{\{ always\(\) && needs\.docs\.result == 'failure' \}\}/,
  );
  assert.match(documentation, /permissions:\n      issues: write/);
  assert.match(documentation, /GH_TOKEN: \$\{\{ github\.token \}\}/);
  assert.match(documentation, /GH_REPO: \$\{\{ github\.repository \}\}/);
  assert.match(documentation, /\/actions\/runs\/\$\{\{ github\.run_id \}\}/);
  assert.match(
    documentation,
    /gh issue create --title "Documentation build failed"/,
  );
  assert.match(documentation, /--body ".*\$RUN_URL"/);
});
