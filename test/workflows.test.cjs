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

test("extension release versions stay consistent", () => {
  const manifest = JSON.parse(
    readFileSync(path.resolve(__dirname, "..", "package.json"), "utf8"),
  );
  const lock = JSON.parse(
    readFileSync(path.resolve(__dirname, "..", "package-lock.json"), "utf8"),
  );
  const changelog = readFileSync(
    path.resolve(__dirname, "..", "CHANGELOGS.md"),
    "utf8",
  );
  assert.equal(manifest.version, "0.2.7");
  assert.equal(lock.version, manifest.version);
  assert.equal(lock.packages[""].version, manifest.version);
  assert.equal(changelog.match(/^(\d+\.\d+\.\d+) \(/m)[1], manifest.version);
});

test("native builds use onnx-light 0.2.7 with pinned checksums", () => {
  const cmake = readFileSync(
    path.resolve(__dirname, "..", "CMakeLists.txt"),
    "utf8",
  );
  const version = cmake.match(/set\(ONNX_LIGHT_VERSION "([^"]+)"\)/)[1];
  assert.equal(version, "0.2.7");
  assert.equal([...cmake.matchAll(/set\(_sha256 "[a-f0-9]{64}"\)/g)].length, 5);
  for (const [asset, sha256] of [
    [
      "onnx-light-cpp-linux-x86_64.tar.gz",
      "d05574ce5009a613981daa5130996838e5e0389321ed2de8d4d78b1fdfd490c6",
    ],
    [
      "onnx-light-cpp-linux-aarch64.tar.gz",
      "46a9d31f3104f826ffd794d817999d85b5880f7c27ee35ccbb15b90a5b6b6a0a",
    ],
    [
      "onnx-light-cpp-macos-universal2.tar.gz",
      "9dd2aafa9d2e9c7227f23afff46d95422840bbc18b82b331ac17a0d074b91193",
    ],
    [
      "onnx-light-cpp-windows-AMD64.zip",
      "c67dbf8a6c0554a7fa28ce0b98e3a6d4b96c902730ab810fcba3a2c06fc4339e",
    ],
    [
      "onnx-light-cpp-windows-ARM64.zip",
      "c2d9b8e6be4d5d91d4ae1ec7f20dd1f43b8874508e1388741e37e8db8050c4ea",
    ],
  ]) {
    assert.ok(
      cmake.includes(`set(_asset "${asset}")\n  set(_sha256 "${sha256}")`),
    );
  }
  assert.match(cmake, /URL_HASH "SHA256=\$\{_sha256\}"/);
  assert.match(cmake, /set\(ONNX_LIGHT_CPU_VERSION "0\.2\.7"\)/);
  assert.match(
    cmake,
    /https:\/\/github\.com\/xadupre\/onnx-light-cpu\/archive\/refs\/tags\/\$\{ONNX_LIGHT_CPU_VERSION\}\.tar\.gz/,
  );
  assert.match(
    cmake,
    /URL_HASH "SHA256=51a33bfef3f032521762ee1f7f753614c451233112786d15c85a417f8b986130"/,
  );
  assert.match(
    cmake,
    /target_link_libraries\(onnx_light_viewer PRIVATE[\s\S]*?\n  lib_onnx_light_cpu_kernels/,
  );
});

test("Python tooling pins both SDKs to 0.2.7 and selects full Linux SDK wheels", () => {
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
  assert.match(requirements, /^onnx-light==0\.2\.7$/m);
  for (const [python, abi] of [
    ["3.12", "cp312"],
    ["3.13", "cp313"],
    ["3.14", "cp314"],
  ])
    assert.ok(
      requirements.includes(
        `onnx-light @ https://github.com/xadupre/onnx-light/releases/download/0.2.7/onnx_light-0.2.7-${abi}-${abi}-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl ; sys_platform == "linux" and platform_machine == "x86_64" and python_version == "${python}"`,
      ),
    );
  assert.match(
    requirements,
    /^--find-links https:\/\/github\.com\/xadupre\/onnx-light\/releases\/expanded_assets\/0\.2\.7$/m,
  );
  assert.match(
    requirements,
    /^--find-links https:\/\/github\.com\/xadupre\/onnx-light-cpu\/releases\/expanded_assets\/0\.2\.7$/m,
  );
  assert.match(requirements, /^onnx-light-cpu==0\.2\.7$/m);
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

test("documentation uses relative paths for its own pages and CLI output examples", () => {
  const links = readFileSync(
    path.resolve(__dirname, "..", "docs", "links.rst"),
    "utf8",
  );
  assert.match(links, /:doc:`Documentation <index>`/);
  assert.doesNotMatch(
    links,
    /https:\/\/xadupre\.github\.io\/docs\/onnx-light-viewer\//,
  );
  const exporting = readFileSync(
    path.resolve(__dirname, "..", "docs", "features", "exporting_models.rst"),
    "utf8",
  );
  assert.match(exporting, /--output_path \.\/new-export/);
  assert.doesNotMatch(exporting, /\/path\/to\//);
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
