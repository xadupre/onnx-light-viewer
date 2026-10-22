const assert = require("node:assert/strict");
const { execFileSync } = require("node:child_process");
const { mkdtempSync, writeFileSync, readFileSync, rmSync } = require("node:fs");
const { tmpdir } = require("node:os");
const path = require("node:path");
const { test } = require("node:test");
const { decodeGraph, demoGraph } = require("../dist/graph");

test("the launcher compiles and opens the demo in a new VS Code window from any directory", (t) => {
  const temporary = mkdtempSync(path.join(tmpdir(), "onnx viewer launch "));
  t.after(() => rmSync(temporary, { recursive: true, force: true }));
  const argumentsFile = path.join(temporary, "arguments.json");
  writeFileSync(
    path.join(temporary, "code"),
    `#!${process.execPath}
require("node:fs").writeFileSync(process.env.LAUNCH_ARGUMENTS_FILE, JSON.stringify(process.argv.slice(2)));
`,
    { mode: 0o755 },
  );
  const root = path.resolve(__dirname, "..");
  execFileSync("bash", [path.join(root, "start.sh")], {
    cwd: temporary,
    env: {
      ...process.env,
      PATH: `${temporary}${path.delimiter}${process.env.PATH}`,
      LAUNCH_ARGUMENTS_FILE: argumentsFile,
    },
    stdio: "pipe",
  });
  assert.deepEqual(JSON.parse(readFileSync(argumentsFile, "utf8")), [
    "--new-window",
    "--disable-extensions",
    `--extensionDevelopmentPath=${root}`,
    root,
    path.join(root, "dist", "demo.onnx"),
  ]);
  assert.deepEqual(
    decodeGraph(
      readFileSync(path.join(root, "dist", "demo.onnx")),
      "demo.onnx",
    ),
    demoGraph,
  );
});
