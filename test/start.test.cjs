const assert = require("node:assert/strict");
const { execFileSync } = require("node:child_process");
const {
  mkdirSync,
  mkdtempSync,
  writeFileSync,
  readFileSync,
  rmSync,
} = require("node:fs");
const { tmpdir } = require("node:os");
const path = require("node:path");
const { test } = require("node:test");
const { format } = require("prettier");

test("the launcher bypasses the remote CLI and forces the developed ONNX editor", async (t) => {
  if (process.platform === "win32") {
    t.skip("The shell launcher requires a POSIX environment");
    return;
  }
  const temporary = mkdtempSync(path.join(tmpdir(), "onnx viewer launch "));
  t.after(() => rmSync(temporary, { recursive: true, force: true }));
  const argumentsFile = path.join(temporary, "arguments.json");
  const remoteBin = path.join(temporary, "remote-cli");
  const desktopBin = path.join(temporary, "desktop bin");
  mkdirSync(remoteBin);
  mkdirSync(desktopBin);
  writeFileSync(
    path.join(remoteBin, "code"),
    "#!/bin/sh\nprintf 'The remote CLI must not be used\\n' >&2\nexit 1\n",
    { mode: 0o755 },
  );
  writeFileSync(
    path.join(desktopBin, "code"),
    `#!${process.execPath}
require("node:fs").writeFileSync(process.env.LAUNCH_ARGUMENTS_FILE, JSON.stringify({
  args: process.argv.slice(2),
  ipcHook: process.env.VSCODE_IPC_HOOK_CLI ?? null,
}));
`,
    { mode: 0o755 },
  );
  const root = path.resolve(__dirname, "..");
  execFileSync("bash", [path.join(root, "start.sh")], {
    cwd: temporary,
    env: {
      ...process.env,
      PATH: [remoteBin, desktopBin, process.env.PATH].join(path.delimiter),
      VSCODE_IPC_HOOK_CLI: "/unused/test-vscode.sock",
      LAUNCH_ARGUMENTS_FILE: argumentsFile,
    },
    stdio: "pipe",
  });
  const launch = JSON.parse(readFileSync(argumentsFile, "utf8"));
  assert.equal(launch.ipcHook, null);
  const workspacePath = path.join(
    root,
    ".vscode",
    "development.code-workspace",
  );
  assert.deepEqual(launch.args, [
    "--new-window",
    "--disable-extensions",
    `--extensionDevelopmentPath=${root}`,
    workspacePath,
    path.join(root, "test", "models", "relu.onnx"),
  ]);
  const workspace = JSON.parse(
    await format(readFileSync(workspacePath, "utf8"), {
      parser: "json-stringify",
    }),
  );
  const manifest = JSON.parse(
    readFileSync(path.join(root, "package.json"), "utf8"),
  );
  assert.equal(
    workspace.settings["workbench.editorAssociations"]["*.onnx"],
    manifest.contributes.customEditors[0].viewType,
  );
});
