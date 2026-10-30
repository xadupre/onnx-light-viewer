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

const documentation = workflow("documentation.yml");

test("native builds use onnx-light 0.2.3 with pinned checksums", () => {
  const cmake = readFileSync(
    path.resolve(__dirname, "..", "CMakeLists.txt"),
    "utf8",
  );
  const version = cmake.match(/set\(ONNX_LIGHT_VERSION "([^"]+)"\)/)[1];
  assert.equal(version, "0.2.3");
  assert.equal([...cmake.matchAll(/set\(_sha256 "[a-f0-9]{64}"\)/g)].length, 5);
  for (const [asset, sha256] of [
    [
      "onnx-light-cpp-linux-x86_64.tar.gz",
      "9fa0c3df5e7d12a0ccb138823a2c26267f94df97deb66c5401fffbf697f0fabb",
    ],
    [
      "onnx-light-cpp-linux-aarch64.tar.gz",
      "8d172f87a957bb5e95e90db19cca345198c738e6e97f01ae832545a71ab51e97",
    ],
    [
      "onnx-light-cpp-macos-universal2.tar.gz",
      "c2bdaf16ef9e1918f98ce0bbf53ef6422fe2e4ec7be621f08aeb64337e554e6b",
    ],
    [
      "onnx-light-cpp-windows-AMD64.zip",
      "b1e6e5bcf4f15364715a8ca2d6097cca15fc3770daa13ced636946eeaa272967",
    ],
    [
      "onnx-light-cpp-windows-ARM64.zip",
      "8ce0f194ce7c28f3f904df69ddcb9991d44f776789632581492bab18bcc5b82f",
    ],
  ]) {
    assert.ok(
      cmake.includes(`set(_asset "${asset}")\n  set(_sha256 "${sha256}")`),
    );
  }
  assert.match(cmake, /URL_HASH "SHA256=\$\{_sha256\}"/);
  assert.match(cmake, /set\(ONNX_LIGHT_CPU_VERSION "0\.2\.1\.1"\)/);
  assert.match(
    cmake,
    /https:\/\/github\.com\/xadupre\/onnx-light-cpu\/archive\/refs\/tags\/\$\{ONNX_LIGHT_CPU_VERSION\}\.tar\.gz/,
  );
  assert.match(
    cmake,
    /URL_HASH "SHA256=ec9ad7a8d58fd24716f4c61d9cf3fdb8bf5d791e7fbea7cbbdb78b8f1e78ac0c"/,
  );
  assert.match(
    cmake,
    /target_link_libraries\(onnx_light_viewer PRIVATE[\s\S]*?\n  lib_onnx_light_cpu_kernels/,
  );
});

test("Python tooling uses full onnx-light 0.2.3 wheels", () => {
  const version = "0.2.3";
  assert.ok(
    workflow("coverage.yml").includes(
      `https://github.com/xadupre/onnx-light/releases/download/${version}/onnx_light-${version}-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl#sha256=a3030b93b8359960e629beab064c1798468ae84a83d8fd1d0f295dcf5ff69407`,
    ),
  );
  const requirements = readFileSync(
    path.resolve(__dirname, "requirements.txt"),
    "utf8",
  );
  assert.match(requirements, /^onnx-light==0\.2\.3$/m);
  assert.ok(
    requirements.includes(
      `https://github.com/xadupre/onnx-light/releases/download/${version}/onnx_light-${version}-cp312-cp312-win_amd64.whl`,
    ),
  );
  assert.match(requirements, /^onnx-light-cpu==0\.2\.1\.1$/m);
});

test("CI runs unit tests on Linux and Windows without building documentation", () => {
  const source = workflow("tests.yml");
  assert.match(source, /runs-on: \$\{\{ matrix\.os \}\}/);
  assert.match(source, /os: \[ubuntu-latest, windows-latest\]/);
  assert.doesNotMatch(source, /macos-/);
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
