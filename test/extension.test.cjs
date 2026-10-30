const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const {
  copyFile,
  mkdir,
  mkdtemp,
  readFile,
  rm,
  writeFile,
  rename,
  readdir,
  stat,
  symlink,
  link,
} = require("node:fs/promises");
const { createCipheriv, createHash, pbkdf2Sync } = require("node:crypto");
const { execFileSync, spawnSync } = require("node:child_process");
const { createServer } = require("node:http");
const { homedir, tmpdir } = require("node:os");
const { createRequire } = require("node:module");
const path = require("node:path");
const vm = require("node:vm");
const { test } = require("node:test");
const { JSDOM } = require("jsdom");
const { exportHub } = require("./helpers/export-hub.cjs");

test("prompt host requires workspace trust and a locally saved converted model", async () => {
  const extension = await openExtension();
  extension.setTrusted(false);
  await extension.receive({
    type: "run-prompt",
    requestId: 1,
    options: { engine: "onnx-light", prompt: "Hello", maxTokens: 32 },
  });
  assert.equal(extension.messages.at(-1).type, "prompt-error");
  assert.match(extension.messages.at(-1).error, /Trust this workspace/);
  extension.setTrusted(true);
  await extension.receive({ type: "prompt-info", requestId: 2 });
  assert.match(extension.messages.at(-1).error, /locally saved converted/);
  await extension.receive({ type: "run-prompt", requestId: -1 });
  assert.match(extension.messages.at(-1).error, /Invalid prompt request/);
});

test("a fresh validated prompt does not reuse a previous generation's captured prefixes", async (t) => {
  const {
    promptModel,
    nativeEnvironment,
  } = require("./helpers/prompt-model.cjs");
  const root = await mkdtemp(
    path.join(tmpdir(), "onnx-viewer-fresh-validation-"),
  );
  t.after(() => rm(root, { force: true, recursive: true }));
  const model = await promptModel(root);
  for (const [name, value] of Object.entries(nativeEnvironment())) {
    const previous = process.env[name];
    process.env[name] = value;
    t.after(() => {
      if (previous === undefined) delete process.env[name];
      else process.env[name] = previous;
    });
  }
  const extension = await openExtension(undefined, undefined, true);
  t.after(() => extension.dispose());
  await extension.resolveCustomEditor(model);
  const options = { engine: "onnxruntime-genai", prompt: "H", maxTokens: 1 };
  await extension.receive({
    type: "run-prompt",
    requestId: 1,
    reuseResult: false,
    options,
  });
  assert.equal(extension.messages.at(-1).type, "prompt-result");
  const nativeEvents = extension.messages.filter(({ type }) =>
    ["prompt-lifecycle", "prompt-token"].includes(type),
  ).length;
  await extension.receive({
    type: "run-prompt",
    requestId: 2,
    reuseResult: false,
    options: {
      ...options,
      prompt: "HH",
      validate: true,
      referenceSource: root,
    },
  });
  assert.equal(extension.messages.at(-1).type, "prompt-error");
  assert.match(
    extension.messages.at(-1).error,
    /checkpoint weights are missing/,
  );
  assert.equal(
    extension.messages.filter(({ type }) =>
      ["prompt-lifecycle", "prompt-token"].includes(type),
    ).length,
    nativeEvents,
  );
});

class FileSystemError extends Error {
  constructor(code) {
    super(code);
    this.code = code;
  }
}

async function openExtension(
  customModel,
  command = "onnxLightViewer.open",
  localFiles = false,
  storage,
  downloadUrl,
  preferences = new Map(),
) {
  const messages = [];
  const errors = [];
  const reads = [];
  const picks = [];
  const logs = [];
  const logViews = [];
  const events = [];
  const passwordPrompts = [];
  const saveDialogs = [];
  const openDialogs = [];
  const writes = [];
  const nativeModels = [];
  const commands = new Map();
  const receivers = new Set();
  const disposals = new Set();
  const pendingReads = new Map();
  const virtualFiles = new Map();
  let webviewPanelOptions;
  let customEditorOptions;
  const receive = (message) =>
    Promise.all([...receivers].map((callback) => callback(message)));
  let selection;
  let openDialogFailure;
  let clipboardText = "";
  let clipboardFailure;
  const clipboardWrites = [];
  let editor;
  let backendSelection;
  let client;
  const fileLoadGates = [];
  let huggingFaceInput;
  let cancelDownload = false;
  let modelSource = "local";
  let repositoryFiles;
  let repositoryFileSelection;
  let exportedFileSelection;
  let passwordValues = [];
  let saveDestination;
  let failSaveWrite = false;
  let failSaveRename = 0;
  let saveDialogGate;
  let saveWriteGate;
  let historyUpdateGate;
  let errorMessageGate;
  const listedRepositories = [];
  const hubSearches = [];
  const hubSearchGates = [];
  let hubSearchResults = [];
  const disposable = () => ({ dispose() {} });
  const root = path.resolve(__dirname, "..");
  const uri = (filePath, scheme = localFiles ? "file" : "virtual") => ({
    path: filePath.replaceAll("\\", "/"),
    fsPath:
      process.platform === "win32"
        ? path.normalize(filePath.replace(/^\/(?=[a-z]:\/)/i, ""))
        : filePath,
    scheme,
    toString: () => {
      const url = new URL(`${scheme}:///`);
      url.pathname = filePath.replaceAll("\\", "/");
      return url.href;
    },
  });
  const webview = {
    postMessage: async (message) => {
      events.push(message);
      client?.(message);
      if (message.type === "log-entry") {
        const { entry } = message;
        logs.push(
          `[${entry.timestamp}] [${entry.status}] ${entry.action}${entry.details ? `\n${entry.details}` : ""}`,
        );
      } else if (message.type === "show-logs") {
        logViews.push(true);
      } else if (
        ![
          "log-history",
          "application-status",
          "path-history",
          "model-progress",
        ].includes(message.type)
      ) {
        messages.push(message);
      }
      return true;
    },
    onDidReceiveMessage: (callback) => {
      receivers.add(callback);
      return { dispose: () => receivers.delete(callback) };
    },
    asWebviewUri: (uri) => ({ toString: () => `file://${uri.path}` }),
  };
  const panel = {
    webview,
    onDidDispose: (callback) => {
      disposals.add(callback);
      return { dispose: () => disposals.delete(callback) };
    },
    onDidChangeViewState: disposable,
    reveal() {},
  };
  const vscode = {
    env: {
      clipboard: {
        readText: async () => {
          if (clipboardFailure) throw clipboardFailure;
          return clipboardText;
        },
        writeText: async (text) => {
          if (clipboardFailure) throw clipboardFailure;
          clipboardText = text;
          clipboardWrites.push(text);
        },
      },
    },
    FileSystemError,
    FileType: { File: 1, Directory: 2, SymbolicLink: 64 },
    Uri: {
      file: (filePath) => uri(filePath, "file"),
      parse: (value) => {
        const url = new URL(value);
        return uri(decodeURIComponent(url.pathname), url.protocol.slice(0, -1));
      },
      joinPath: (base, ...parts) =>
        uri(path.posix.join(base.path, ...parts), base.scheme),
    },
    ViewColumn: { One: 1 },
    ProgressLocation: { Notification: 15 },
    commands: {
      executeCommand: async (command, resource, editorViewType) => {
        assert.equal(command, "vscode.openWith");
        assert.equal(editorViewType, "onnxLightViewer.graph");
        await editor.resolveCustomEditor({ uri: resource }, panel);
      },
      registerCommand: (name, callback) => {
        commands.set(name, callback);
        return disposable();
      },
    },
    workspace: {
      isTrusted: true,
      fs: {
        createDirectory: async (uri) => {
          await mkdir(uri.path, { recursive: true });
        },
        stat: async (uri) => {
          let details;
          try {
            details = await stat(uri.fsPath);
          } catch (error) {
            if (error.code === "ENOTDIR")
              throw new FileSystemError("FileNotADirectory");
            throw error;
          }
          return {
            type: details.isDirectory() ? 2 : 1,
            ctime: details.ctimeMs,
            mtime: details.mtimeMs,
            size: details.size,
          };
        },
        readDirectory: async (directory) =>
          (await readdir(directory.path, { withFileTypes: true })).map(
            (entry) => [entry.name, entry.isDirectory() ? 2 : 1],
          ),
        readFile: async (uri) => {
          reads.push(uri.path);
          const key = path.basename(uri.path);
          const pending = pendingReads.get(key);
          await (pending ?? Promise.resolve());
          pendingReads.delete(key);
          return virtualFiles.get(uri.path) ?? readFile(uri.path);
        },
        writeFile: async (uri, bytes) => {
          writes.push(uri.path);
          await writeFile(
            uri.path,
            failSaveWrite ? bytes.subarray(0, 8) : bytes,
          );
          if (failSaveWrite) {
            failSaveWrite = false;
            throw new Error("Test filesystem refused the complete write.");
          }
          if (saveWriteGate && --saveWriteGate.after === 0) {
            const gate = saveWriteGate;
            saveWriteGate = undefined;
            gate.arrived();
            await gate.wait;
          }
        },
        rename: async (source, destination, options) => {
          if (failSaveRename > 0 && --failSaveRename === 0) {
            throw new Error("Test filesystem refused the rename.");
          }
          if (!options.overwrite) {
            await link(source.path, destination.path);
            await rm(source.path);
          } else await rename(source.path, destination.path);
        },
        delete: async (uri) => {
          try {
            await rm(uri.path);
          } catch (error) {
            if (error.code === "ENOENT")
              throw new FileSystemError("FileNotFound");
            throw error;
          }
        },
      },
    },
    window: {
      createWebviewPanel: (_viewType, _title, _column, options) => {
        webviewPanelOptions = options;
        return panel;
      },
      registerCustomEditorProvider: (_viewType, provider, options) => {
        editor = provider;
        customEditorOptions = options;
        return disposable();
      },
      showOpenDialog: async (options) => {
        openDialogs.push(options);
        if (openDialogFailure) {
          const error = openDialogFailure;
          openDialogFailure = undefined;
          throw error;
        }
        return selection;
      },
      showInputBox: async (options) => {
        if (!options.password) return huggingFaceInput;
        passwordPrompts.push(options);
        return passwordValues.shift();
      },
      showSaveDialog: async (options) => {
        saveDialogs.push(options);
        const gate = saveDialogGate;
        saveDialogGate = undefined;
        if (gate) {
          gate.arrived();
          await gate.wait;
        }
        return saveDestination;
      },
      withProgress: async (_options, work) =>
        work(
          { report() {} },
          {
            isCancellationRequested: cancelDownload,
            onCancellationRequested: disposable,
          },
        ),
      showQuickPick: async (items, options) => {
        picks.push({ items, options });
        if (options.title === "Open an ONNX model")
          return items.find(({ id }) => id === modelSource);
        if (options.title === "Choose an ONNX file")
          return items.find(({ label }) => label === repositoryFileSelection);
        if (options.title === "Choose exported ONNX model")
          return items.find(({ label }) => label === exportedFileSelection);
        return backendSelection === undefined
          ? undefined
          : {
              label: backendSelection,
              id:
                items.find((item) => item.label === backendSelection)?.id ?? -1,
            };
      },
      showErrorMessage: async (message) => {
        errors.push(message);
        const gate = errorMessageGate;
        errorMessageGate = undefined;
        if (gate) {
          gate.arrived();
          await gate.wait;
        }
      },
    },
  };
  const filename = path.join(root, "dist", "extension.js");
  const requireExtension = createRequire(filename);
  const graph = { ...requireExtension("./graph") };
  const huggingface = {
    ...requireExtension("./huggingface"),
    searchHuggingFaceModels: async (query, task, signal) => {
      hubSearches.push({ query, task, signal });
      const results = hubSearchResults;
      const gate = hubSearchGates.shift();
      if (gate) await gate;
      signal.throwIfAborted();
      return results;
    },
    cachedHuggingFaceModel: (source, ...args) =>
      requireExtension("./huggingface").cachedHuggingFaceModel(
        downloadUrl ? { ...source, url: downloadUrl } : source,
        ...args,
      ),
    listHuggingFaceModels: async (repository) => {
      listedRepositories.push(repository);
      if (!repositoryFiles)
        throw new Error("Repository files were not provided.");
      return repositoryFiles;
    },
  };
  for (const name of ["loadModel", "loadBackendTest"]) {
    const load = graph[name];
    graph[name] = (...args) => {
      const native = load(...args);
      nativeModels.push(native);
      return native;
    };
  }
  const loadFile = graph.loadModelFile;
  graph.loadModelFile = async (...args) => {
    const gate = fileLoadGates.shift();
    if (gate) await gate;
    const native = await loadFile(...args);
    nativeModels.push(native);
    return native;
  };
  const module = { exports: {} };
  const execute = vm.runInThisContext(
    `(function(require, module, exports) { ${readFileSync(filename, "utf8")}\n})`,
    { filename },
  );
  execute(
    (name) =>
      name === "vscode"
        ? vscode
        : name === "./graph"
          ? graph
          : name === "./huggingface"
            ? huggingface
            : requireExtension(name),
    module,
    module.exports,
  );
  module.exports.activate({
    extensionUri: { path: root },
    extension: { packageJSON: require("../package.json") },
    globalStorageUri: { fsPath: storage ?? tmpdir() },
    globalState: {
      get: (key) => structuredClone(preferences.get(key)),
      update: async (key, value) => {
        if (key === "pathHistory" && historyUpdateGate) {
          const gate = historyUpdateGate;
          historyUpdateGate = undefined;
          gate.arrived();
          await gate.wait;
        }
        preferences.set(key, structuredClone(value));
      },
    },
    subscriptions: [],
  });
  if (customModel) {
    await editor.resolveCustomEditor(
      { uri: uri(path.join(__dirname, "models", customModel)) },
      panel,
    );
  } else {
    await commands.get(command)();
  }
  return {
    messages,
    errors,
    reads,
    picks,
    logs,
    logViews,
    events,
    preferences,
    webviewPanelOptions,
    customEditorOptions,
    setTrusted: (trusted) => {
      vscode.workspace.isTrusted = trusted;
    },
    setWorkspaceFolders: (paths) => {
      vscode.workspace.workspaceFolders = paths.map((folder) => ({
        uri: uri(folder),
        name: path.basename(folder),
      }));
    },
    selectExportedFile: (name) => {
      exportedFileSelection = name;
    },
    pauseNextHistoryUpdate: () => {
      let release, arrived;
      const wait = new Promise((resolve) => {
        release = resolve;
      });
      const started = new Promise((resolve) => {
        arrived = resolve;
      });
      historyUpdateGate = { wait, arrived };
      return { started, release };
    },
    pauseNextErrorMessage: () => {
      let release, arrived;
      const wait = new Promise((resolve) => {
        release = resolve;
      });
      const opened = new Promise((resolve) => {
        arrived = resolve;
      });
      errorMessageGate = { wait, arrived };
      return { opened, release };
    },
    passwordPrompts,
    saveDialogs,
    openDialogs,
    setOpenSelection: (filePath) => {
      selection = filePath ? [vscode.Uri.file(filePath)] : undefined;
    },
    failNextOpenDialog: (message) => {
      openDialogFailure = new Error(message);
    },
    setClipboard: (text, error) => {
      clipboardText = text;
      clipboardFailure = error ? new Error(error) : undefined;
    },
    clipboardWrites,
    writes,
    webview,
    openFromExplorer: (name) =>
      commands.get("onnxLightViewer.openFile")(
        uri(path.resolve(__dirname, "models", name)),
      ),
    resolveCustomEditor: (name) =>
      editor.resolveCustomEditor(
        { uri: uri(path.resolve(__dirname, "models", name)) },
        panel,
      ),
    deferRead: (name) => {
      let release;
      pendingReads.set(
        name,
        new Promise((resolve) => {
          release = resolve;
        }),
      );
      return release;
    },
    dispose: () => {
      // Ensure idempotency: once disposed, subsequent calls do nothing.
      const callbacks = [...disposals];
      disposals.clear();
      for (const callback of callbacks) callback();
    },
    connect: (callback) => {
      client = callback;
    },
    nativeModels,
    pauseNextFileLoad: (error) => {
      let resume;
      const gate = new Promise((resolve, reject) => {
        resume = () => (error ? reject(error) : resolve());
      });
      fileLoadGates.push(gate);
      return resume;
    },
    listedRepositories,
    setRepositoryFiles: (files) => {
      repositoryFiles = files;
    },
    hubSearches,
    setHubSearchResults: (models) => {
      hubSearchResults = models;
    },
    pauseNextHubSearch: (error) => {
      let resume;
      hubSearchGates.push(
        new Promise((resolve, reject) => {
          resume = () => (error ? reject(error) : resolve());
        }),
      );
      return resume;
    },
    showLogs: () => commands.get("onnxLightViewer.showLogs")(),
    receive: (message) => receive(message),
    setPasswords: (...values) => {
      passwordValues = values;
    },
    openFile: async (filePath) => {
      modelSource = "local";
      selection = [uri(filePath)];
      await receive({ type: "open" });
    },
    openBytes: async (fileName, bytes) => {
      modelSource = "local";
      const location = uri(path.join(__dirname, "models", fileName), "virtual");
      virtualFiles.set(location.path, bytes);
      selection = [location];
      await receive({ type: "open" });
    },
    save: async (
      filePath,
      encrypted = false,
      passwords = [],
      externalSize,
      reuse = false,
    ) => {
      saveDestination = filePath ? vscode.Uri.file(filePath) : undefined;
      passwordValues = passwords;
      await receive({
        type: "save",
        encrypted,
        storage: reuse
          ? "reuse"
          : externalSize === undefined
            ? "embedded"
            : "external",
        ...(externalSize === undefined
          ? {}
          : { maxExternalFileSize: externalSize }),
      });
    },
    failNextSaveWrite: () => {
      failSaveWrite = true;
    },
    failNextSaveRename: (after = 1) => {
      failSaveRename = after;
    },
    pauseNextSaveDialog: () => {
      let release, arrived;
      const wait = new Promise((resolve) => {
        release = resolve;
      });
      const opened = new Promise((resolve) => {
        arrived = resolve;
      });
      saveDialogGate = { wait, arrived };
      return { opened, release };
    },
    pauseNextSaveWrite: (after = 1) => {
      let release, arrived;
      const wait = new Promise((resolve) => {
        release = resolve;
      });
      const written = new Promise((resolve) => {
        arrived = resolve;
      });
      saveWriteGate = { after, wait, arrived };
      return { written, release };
    },
    open: async (name) => {
      modelSource = "local";
      selection = [uri(path.join(__dirname, "models", name))];
      await receive({ type: "open" });
    },
    chooseBackend: async (name, type = "open") => {
      modelSource = "backend";
      backendSelection = name;
      await receive({ type });
    },
    openHuggingFace: async (input, cancel = false, repository) => {
      modelSource = "huggingface";
      huggingFaceInput = input;
      cancelDownload = cancel;
      repositoryFiles = repository?.files;
      repositoryFileSelection = repository?.selectedFile;
      await receive({ type: "open" });
    },
    cancelOpen: async () => {
      modelSource = undefined;
      await receive({ type: "open" });
    },
  };
}

test("retains webview context while hidden for both editor creation paths", async () => {
  const extension = await openExtension();
  assert.equal(extension.webviewPanelOptions.retainContextWhenHidden, true);
  assert.equal(
    extension.customEditorOptions.webviewOptions.retainContextWhenHidden,
    true,
  );
  extension.dispose();
});

test("the Explorer context menu offers ONNX Light Viewer only for supported model filenames", () => {
  const { contributes } = require("../package.json");
  const menu = contributes.menus["explorer/context"].find(
    ({ command }) => command === "onnxLightViewer.openFile",
  );
  assert.ok(menu);
  assert.match(menu.when, /!explorerResourceIsFolder/);
  const [, expression, flags] = menu.when.match(
    /resourceFilename =~ \/(.+)\/([a-z]*)/,
  );
  const filename = new RegExp(expression, flags);
  for (const name of [
    "model.onnx",
    "model.onnx.enc",
    "model.pipe-onnx",
    "MODEL.ONNX",
    "MODEL.ONNX.ENC",
    "MODEL.PIPE-ONNX",
  ])
    assert.ok(filename.test(name), name);
  for (const name of [
    "model.txt",
    "model.onnx.data",
    "model.onnx.backup",
    "model.onnx.enc.data",
    "not-onnx",
    "model.enc",
  ])
    assert.equal(filename.test(name), false, name);
});

test("opens Explorer files with the custom editor for local and virtual resources", async (t) => {
  for (const localFiles of [false, true]) {
    const extension = await openExtension(undefined, undefined, localFiles);
    t.after(() => extension.dispose());
    await extension.openFromExplorer("inference.onnx");
    await extension.receive({ type: "ready" });
    assert.equal(extension.messages.at(-1).type, "graph");
    assert.equal(extension.messages.at(-1).graph.name, "Shape inference");
    await extension.receive({ type: "refresh-shapes" });
    assert.equal(extension.messages.at(-1).type, "shape-info");
    assert.deepEqual(extension.errors, []);
  }
});

test("defaults model saves to a workspace cache subdirectory without moving the open directory", async (t) => {
  const extension = await openExtension(undefined, undefined, true);
  t.after(() => extension.dispose());
  const workspace = await mkdtemp(
    path.join(tmpdir(), "viewer-save-workspace-"),
  );
  t.after(() => rm(workspace, { recursive: true, force: true }));
  const saveDirectory = path.join(workspace, ".cache", ".onnx-light-viewer");
  extension.setWorkspaceFolders([
    workspace,
    path.join(__dirname, "other-workspace"),
  ]);
  await extension.receive({ type: "ready" });
  const latestFile = () =>
    extension.messages.filter(({ type }) => type === "graph").at(-1).file;
  assert.equal(latestFile().saveDirectory, saveDirectory);
  assert.equal(path.dirname(latestFile().savePath), saveDirectory);
  await assert.rejects(stat(saveDirectory), { code: "ENOENT" });
  await extension.open("relu.onnx");
  const sourceDirectory = path.join(__dirname, "models");
  assert.equal(latestFile().directory, sourceDirectory);
  assert.equal(latestFile().saveDirectory, saveDirectory);
  assert.equal(
    latestFile().savePath,
    path.join(saveDirectory, "relu-copy.onnx"),
  );
  await extension.receive({
    type: "browse-files",
    target: "save",
    source: "local",
    requestId: 1,
  });
  assert.ok((await stat(saveDirectory)).isDirectory());
  const listing = extension.messages
    .filter(({ type }) => type === "file-list")
    .at(-1);
  assert.equal(listing.location, saveDirectory);
  await rm(saveDirectory, { recursive: true });
  await extension.save();
  assert.ok((await stat(saveDirectory)).isDirectory());
  assert.equal(
    extension.saveDialogs.at(-1).defaultUri.fsPath,
    latestFile().savePath,
  );
  await rm(saveDirectory, { recursive: true });
  const destination = latestFile().savePath;
  await extension.save(destination);
  const { loadModel } = require("../dist/graph");
  const saved = loadModel(await readFile(destination), "relu-copy.onnx");
  assert.equal(saved.graph().nodes[0].opType, "Relu");
  extension.setWorkspaceFolders([]);
  await extension.open("relu.onnx");
  assert.equal(latestFile().saveDirectory, sourceDirectory);
  assert.equal(
    latestFile().savePath,
    path.join(sourceDirectory, "relu-copy.onnx"),
  );
  await extension.save();
  assert.equal(
    extension.saveDialogs.at(-1).defaultUri.fsPath,
    latestFile().savePath,
  );
});

test("opens the save dialog when the suggested directory cannot be created but rejects an unwritable selected destination", async (t) => {
  const directory = await mkdtemp(
    path.join(tmpdir(), "viewer-save-directory-error-"),
  );
  t.after(() => rm(directory, { recursive: true, force: true }));
  const workspace = path.join(directory, "workspace");
  await mkdir(workspace);
  await writeFile(path.join(workspace, ".cache"), "Not a directory");
  const extension = await openExtension("relu.onnx", undefined, true);
  t.after(() => extension.dispose());
  extension.setWorkspaceFolders([workspace]);
  const destination = path.join(directory, "saved.onnx");
  await extension.save(destination);
  assert.equal(extension.saveDialogs.length, 1);
  assert.equal(extension.messages.at(-1).type, "save-info");
  assert.deepEqual(extension.errors, []);
  const saved = await readFile(destination);
  const { loadModel } = require("../dist/graph");
  assert.equal(loadModel(saved, "saved").graph().nodes[0].opType, "Relu");
  assert.match(
    extension.logs.join("\n"),
    /\[FAILED\] Prepare suggested save directory/,
  );
  const writes = extension.writes.length;
  await extension.save(
    path.join(workspace, ".cache", ".onnx-light-viewer", "saved.onnx"),
  );
  assert.equal(extension.saveDialogs.length, 2);
  assert.equal(extension.messages.at(-1).type, "save-error");
  assert.equal(extension.errors.length, 1);
  assert.equal(extension.writes.length, writes);
  assert.deepEqual(await readFile(destination), saved);
});

test("persists only viewer preferences across closed extension panels", async () => {
  const preferences = new Map();
  const first = await openExtension(
    undefined,
    undefined,
    false,
    undefined,
    undefined,
    preferences,
  );
  const settings = {
    width: 480,
    panels: { operator: false, find: true, metadata: true },
  };
  await first.receive({
    type: "viewer-settings",
    settings: { ...settings, password: "not-a-preference" },
  });
  assert.deepEqual(preferences.get("viewerSettings"), settings);
  first.dispose();
  const second = await openExtension(
    undefined,
    undefined,
    false,
    undefined,
    undefined,
    preferences,
  );
  await second.receive({ type: "ready" });
  assert.deepEqual(
    second.messages.find(({ type }) => type === "viewer-settings"),
    {
      type: "viewer-settings",
      settings,
    },
  );
  second.dispose();
});

test("Settings removes stored histories and resets only viewer preferences without deleting files", async (t) => {
  const directory = await mkdtemp(
    path.join(tmpdir(), "viewer-settings-history-"),
  );
  t.after(() => rm(directory, { recursive: true, force: true }));
  const saved = path.join(directory, "saved.onnx");
  await writeFile(saved, "keep this file");
  const entry = { source: "local", kind: "file", path: saved };
  const application = { downloadDirectory: directory, pythonPath: "python3" };
  const viewer = {
    width: 420,
    panels: { operator: false, find: false, metadata: true },
  };
  const preferences = new Map([
    ["applicationSettings", application],
    ["viewerSettings", viewer],
    ["pathHistory", { open: [entry], save: [entry] }],
    [
      "exportHistory",
      {
        models: ["owner/model", "owner/other"],
        directories: [directory],
        exporter: "mobius",
      },
    ],
  ]);
  const extension = await openExtension(
    undefined,
    undefined,
    true,
    undefined,
    undefined,
    preferences,
  );
  t.after(() => extension.dispose());
  await extension.receive({ type: "ready" });
  await extension.receive({ type: "get-settings" });
  const settings = () =>
    extension.messages.filter(({ type }) => type === "settings-info").at(-1);
  const conversionHistory = () =>
    extension.messages.filter(({ type }) => type === "export-history").at(-1)
      .history;
  assert.deepEqual(settings().pathHistory, { open: [entry], save: [entry] });
  assert.equal(settings().application.downloadDirectory, directory);
  assert.equal(typeof settings().saveDirectory, "string");
  await extension.receive({
    type: "path-history-edit",
    target: "open",
    historyAction: "remove",
    historyEntry: entry,
  });
  assert.deepEqual(preferences.get("pathHistory"), {
    open: [],
    save: [entry],
  });
  await extension.receive({
    type: "path-history-edit",
    target: "save",
    historyAction: "clear",
  });
  assert.deepEqual(preferences.get("pathHistory"), { open: [], save: [] });
  await extension.receive({
    type: "export-history-edit",
    target: "models",
    historyAction: "remove",
    historyEntry: "owner/model",
  });
  assert.deepEqual(conversionHistory(), {
    models: ["owner/other"],
    directories: [directory],
    exporter: "mobius",
  });
  await extension.receive({
    type: "export-history-edit",
    target: "directories",
    historyAction: "remove",
    historyEntry: directory,
  });
  assert.deepEqual(conversionHistory().directories, []);
  await extension.receive({
    type: "export-history-edit",
    historyAction: "clear",
  });
  assert.deepEqual(conversionHistory(), { models: [], directories: [] });
  await extension.receive({
    type: "export-history-edit",
    target: "unexpected",
    historyAction: "remove",
    historyEntry: "owner/other",
  });
  assert.match(
    extension.messages.at(-1).error,
    /Invalid conversion history edit/,
  );
  await extension.receive({ type: "reset-viewer-settings" });
  assert.deepEqual(settings().viewer, {
    panels: { operator: true, find: true, metadata: false },
  });
  assert.equal(preferences.get("viewerSettings"), undefined);
  assert.deepEqual(preferences.get("applicationSettings"), application);
  assert.equal(await readFile(saved, "utf8"), "keep this file");
});

test("Olive host enforces workspace trust, quantizes without a checkpoint download and rejects unpinned recipe paths", async (t) => {
  const root = await mkdtemp(path.join(tmpdir(), "viewer-olive-host-"));
  const previous = process.env.PYTHONPATH;
  process.env.PYTHONPATH = root;
  t.after(async () => {
    if (previous === undefined) delete process.env.PYTHONPATH;
    else process.env.PYTHONPATH = previous;
    await rm(root, { recursive: true, force: true });
  });
  await mkdir(path.join(root, "olive"));
  await writeFile(path.join(root, "olive", "__init__.py"), "");
  await writeFile(
    path.join(root, "olive", "__main__.py"),
    `
import os
import pathlib
import shutil
import sys
args = sys.argv[1:]
assert args[0] == "quantize"
assert args[args.index("--precision") + 1] == "int4"
assert args[args.index("--implementation") + 1] == "olive"
assert os.environ["HF_HUB_OFFLINE"] == "1"
source = pathlib.Path(args[args.index("-m") + 1])
output = pathlib.Path(args[args.index("--output_path") + 1])
shutil.copyfile(source, output / "quantized.onnx")
`,
  );
  const extension = await openExtension(
    "relu.onnx",
    undefined,
    true,
    undefined,
    undefined,
    new Map([
      [
        "applicationSettings",
        {
          downloadDirectory: root,
          pythonPath: process.platform === "win32" ? "python" : "python3",
        },
      ],
    ]),
  );
  t.after(() => extension.dispose());
  const exportOptions = {
    exporter: "olive",
    oliveWorkflow: "quantize",
    model: path.join(__dirname, "models", "relu.onnx"),
    directory: path.join(root, "output"),
    precision: "int4",
    provider: "cpu",
  };
  extension.setTrusted(false);
  await extension.receive({ type: "olive-recipes", requestId: 1 });
  assert.match(
    extension.messages.find((message) => message.type === "olive-recipes")
      .error,
    /Trust this workspace/,
  );
  await extension.receive({
    type: "export",
    actionId: "untrusted",
    exportOptions,
  });
  assert.match(
    extension.messages.find((message) => message.type === "export-error").error,
    /Trust this workspace/,
  );
  assert.equal((await readdir(root)).includes("output"), false);
  extension.setTrusted(true);
  await extension.receive({
    type: "export",
    actionId: "quantize",
    exportOptions,
  });
  const result = extension.messages.find(
    (message) =>
      message.type === "export-info" && message.actionId === "quantize",
  );
  assert.equal(result.loaded, true);
  assert.equal(result.path, path.join(root, "output", "quantized.onnx"));
  const progress = extension.messages.filter(
    (message) =>
      message.type === "export-progress" && message.actionId === "quantize",
  );
  assert.equal(
    progress.some((message) => message.phase === "download"),
    false,
  );
  assert.equal(
    progress.some((message) => message.downloadSkipped === true),
    true,
  );
  await extension.receive({
    type: "export-preview",
    requestId: 2,
    exportOptions: {
      ...exportOptions,
      exporter: "olive-recipes",
      recipeId: "../untrusted.json",
      recipeRevision: "a".repeat(40),
      recipe: { id: "../untrusted.json", snapshot: root },
    },
  });
  assert.match(
    extension.messages.find(
      (message) => message.type === "export-preview" && message.requestId === 2,
    ).error,
    /Refresh the Olive recipe list/,
  );
});

test("runs all exporter entry points in Python processes and preserves the model on failure or cancellation", async (t) => {
  const root = await mkdtemp(path.join(tmpdir(), "viewer-export-host-"));
  const hub = await exportHub(t);
  const oldEndpoint = process.env.HF_ENDPOINT;
  process.env.HF_ENDPOINT = hub.endpoint;
  const oldPythonPath = process.env.PYTHONPATH;
  process.env.PYTHONPATH = [root, oldPythonPath]
    .filter(Boolean)
    .join(path.delimiter);
  t.after(async () => {
    if (oldPythonPath === undefined) delete process.env.PYTHONPATH;
    else process.env.PYTHONPATH = oldPythonPath;
    if (oldEndpoint === undefined) delete process.env.HF_ENDPOINT;
    else process.env.HF_ENDPOINT = oldEndpoint;
    await rm(root, { recursive: true, force: true });
  });
  for (const directory of [
    "modelbuilder",
    "onnxruntime_genai",
    "onnxruntime_genai/models",
    "mobius",
    "olive",
  ]) {
    await mkdir(path.join(root, directory), { recursive: true });
    await writeFile(path.join(root, directory, "__init__.py"), "");
  }
  const script = `
import pathlib
import json
import os
import shutil
import sys
import time
args = sys.argv[1:]
olive = args[0] == "optimize"
if olive:
    assert args[args.index("--exporter") + 1] == "dynamo_exporter"
    assert args[args.index("--task") + 1] == "text-generation"
    assert args[args.index("--device") + 1] == "cpu"
    assert args[args.index("--provider") + 1] == "CPUExecutionProvider"
    assert os.environ.get("HF_HUB_OFFLINE", "") == ${JSON.stringify(process.env.HF_HUB_OFFLINE ?? "")}
    assert os.environ.get("TRANSFORMERS_OFFLINE", "") == ${JSON.stringify(process.env.TRANSFORMERS_OFFLINE ?? "")}
else:
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
source_flag = "-m" if olive else "--config" if "--config" in args else "-i"
source = pathlib.Path(args[args.index(source_flag) + 1])
model = json.loads((source / "config.json").read_text())["modelId"]
assert (source / "model.safetensors").read_bytes() == b"source weights"
print("Working on ONNX export", flush=True)
print(json.dumps(args), flush=True)
if model == "wait":
    time.sleep(30)
if model == "fail":
    raise RuntimeError("Failed export leaves the displayed model intact")
flag = "--output_path" if olive else "-o" if "-o" in args else "--output"
output = pathlib.Path(args[args.index(flag) + 1])
if args[0] == "build":
    if output.exists():
        raise RuntimeError("Mobius package path already exists")
    output.mkdir(parents=True)
else:
    output.mkdir(parents=True, exist_ok=True)
shutil.copyfile(${JSON.stringify(path.join(__dirname, "models", "relu.onnx"))}, output / "model.onnx")
if model == "multiple":
    shutil.copyfile(output / "model.onnx", output / "second.onnx")
print("Saved model.onnx", flush=True)
`;
  for (const filename of [
    "modelbuilder/builder.py",
    "onnxruntime_genai/models/builder.py",
    "mobius/__main__.py",
    "olive/__main__.py",
  ])
    await writeFile(path.join(root, filename), script);
  const preferences = new Map([
    [
      "applicationSettings",
      {
        downloadDirectory: path.join(root, "downloads"),
        pythonPath: process.platform === "win32" ? "python" : "python3",
      },
    ],
  ]);
  const extension = await openExtension(
    "relu.onnx",
    undefined,
    true,
    undefined,
    undefined,
    preferences,
  );
  t.after(() => extension.dispose());
  const options = (exporter, name) => ({
    exporter,
    model: "owner/model",
    directory: path.join(root, `output-${name}`),
    precision: "fp32",
    provider: "cpu",
  });
  for (const exporter of ["mbext", "modelbuilder", "mobius", "olive"]) {
    const exportOptions = {
      ...options(exporter, exporter),
      advancedOptions:
        exporter === "olive"
          ? { oliveExporter: "dynamo_exporter", task: "text-generation" }
          : exporter === "mobius"
            ? { optimize: "all", staticCache: true, maxSequenceLength: 128 }
            : { excludeEmbeds: true, includeHiddenStates: true },
    };
    const advancedArgument =
      exporter === "olive"
        ? "--exporter"
        : exporter === "mobius"
          ? "--max-seq-len"
          : "include_hidden_states=true";
    await extension.receive({
      type: "export-preview",
      requestId: 1,
      exportOptions,
    });
    assert.equal(extension.messages.at(-1).type, "export-preview");
    assert.ok(
      extension.messages
        .at(-1)
        .command.includes(
          exporter === "mbext" ? "modelbuilder.builder" : exporter,
        ),
    );
    assert.ok(extension.messages.at(-1).command.includes(advancedArgument));
    await extension.receive({
      type: "export",
      actionId: `ui:export:${exporter}`,
      exportOptions,
    });
    const result = extension.messages.at(-1);
    assert.equal(result.type, "export-info", result.error);
    assert.equal(result.loaded, true);
    assert.ok(Number.isFinite(result.durationMs) && result.durationMs > 0);
    assert.ok(result.downloadDurationMs > 0);
    assert.equal(result.downloadSkipped, false);
    assert.equal(preferences.get("exportHistory").exporter, exporter);
    assert.equal(result.path, path.join(exportOptions.directory, "model.onnx"));
    assert.equal(
      extension.messages.filter((message) => message.type === "graph").at(-1)
        .file.directory,
      exportOptions.directory,
    );
    assert.match(await readFile(result.script, "utf8"), /subprocess.run/);
    assert.ok(
      (await readFile(result.script, "utf8")).includes(advancedArgument),
    );
    assert.ok(
      extension.events.some(
        (event) =>
          event.type === "export-progress" &&
          event.actionId === `ui:export:${exporter}` &&
          event.output?.includes(JSON.stringify(advancedArgument)),
      ),
    );
    assert.ok(
      extension.events.some(
        (event) =>
          event.type === "export-progress" &&
          event.output?.includes("Saved model.onnx"),
      ),
    );
    if (exporter === "olive")
      assert.ok(
        extension.events.some(
          (event) =>
            event.type === "export-progress" &&
            event.actionId === `ui:export:${exporter}` &&
            event.phase === "conversion" &&
            event.stage ===
              "Exporting with Olive (calibration downloads may be required)",
        ),
      );
  }
  assert.equal(
    hub.requests.filter(
      ({ method, pathname }) =>
        method === "GET" && pathname.includes("/owner/model/resolve/"),
    ).length,
    2,
  );
  assert.ok(
    extension.events.some(
      (event) =>
        event.phase === "download" &&
        event.cacheDirectory === path.join(root, "downloads", "huggingface"),
    ),
  );
  const occupied = options("modelbuilder", "mbext");
  const previousScript = await readFile(
    path.join(occupied.directory, "export_model.py"),
    "utf8",
  );
  const previousModel = await readFile(
    path.join(occupied.directory, "model.onnx"),
  );
  await extension.receive({
    type: "export",
    actionId: "ui:export:retry",
    exportOptions: occupied,
  });
  const retry = extension.messages.at(-1);
  assert.equal(retry.type, "export-info", retry.error);
  assert.equal(retry.loaded, true);
  const retryDirectory = path.dirname(retry.path);
  assert.equal(path.dirname(retryDirectory), occupied.directory);
  assert.ok(path.basename(retryDirectory).startsWith("modelbuilder-"));
  assert.equal(retry.script, path.join(retryDirectory, "export_model.py"));
  assert.equal(
    await readFile(path.join(occupied.directory, "export_model.py"), "utf8"),
    previousScript,
  );
  assert.deepEqual(
    await readFile(path.join(occupied.directory, "model.onnx")),
    previousModel,
  );
  const command = extension.events.find(
    (event) =>
      event.type === "export-progress" &&
      event.actionId === "ui:export:retry" &&
      event.command,
  );
  assert.equal(command.directory, retryDirectory);
  assert.equal(command.script, retry.script);
  assert.ok(command.command.includes(retryDirectory));
  assert.ok(
    (await readFile(retry.script, "utf8")).includes(
      JSON.stringify(retryDirectory),
    ),
  );
  const defaultOptions = {
    ...options("mbext", "default"),
    directory: "",
    model: "owner/default",
  };
  const defaultDirectory = path.join(
    root,
    "downloads",
    "owner",
    "default",
    "mbext",
  );
  const defaultTemplate = path.join(
    root,
    "downloads",
    "<model_id>",
    "<exporter>",
  );
  await extension.receive({
    type: "export-preview",
    requestId: 2,
    exportOptions: defaultOptions,
  });
  assert.equal(extension.messages.at(-1).directory, defaultDirectory);
  assert.equal(extension.messages.at(-1).directoryTemplate, defaultTemplate);
  assert.equal(preferences.has("exportHistory"), true);
  assert.ok(!preferences.get("exportHistory").models.includes("owner/default"));
  await extension.receive({ type: "export", exportOptions: defaultOptions });
  assert.equal(
    extension.messages.at(-1).path,
    path.join(defaultDirectory, "model.onnx"),
  );
  const restored = await openExtension(
    "relu.onnx",
    undefined,
    true,
    undefined,
    undefined,
    preferences,
  );
  await restored.receive({ type: "get-settings" });
  assert.deepEqual(restored.messages.at(-1).exportHistory, {
    exporter: "mbext",
    models: ["owner/default", "owner/model"],
    directories: [
      defaultTemplate,
      defaultDirectory,
      retryDirectory,
      ...["olive", "mobius", "modelbuilder", "mbext"].map(
        (name) => options(name, name).directory,
      ),
    ],
  });
  assert.equal(restored.messages.at(-1).exportDirectory, defaultTemplate);
  restored.dispose();
  const templateOptions = {
    ...options("modelbuilder", "template"),
    directory: path.join("base", "<model_id>", "<exporter>"),
  };
  const expandedDirectory = path.join(
    root,
    "downloads",
    "base",
    "owner",
    "model",
    "modelbuilder",
  );
  await extension.receive({
    type: "export-preview",
    requestId: 3,
    exportOptions: templateOptions,
  });
  assert.equal(extension.messages.at(-1).directory, expandedDirectory);
  assert.ok(!extension.messages.at(-1).command.includes("<model_id>"));
  await extension.receive({ type: "export", exportOptions: templateOptions });
  assert.equal(
    extension.messages.at(-1).path,
    path.join(expandedDirectory, "model.onnx"),
  );
  await extension.receive({ type: "export", exportOptions: templateOptions });
  const templateRetry = extension.messages.at(-1);
  assert.equal(templateRetry.type, "export-info", templateRetry.error);
  assert.equal(
    path.dirname(path.dirname(templateRetry.path)),
    expandedDirectory,
  );
  assert.deepEqual(preferences.get("exportHistory").directories.slice(0, 2), [
    templateOptions.directory,
    path.dirname(templateRetry.path),
  ]);
  await extension.receive({
    type: "export",
    exportOptions: {
      ...options("mobius", "multiple-cancelled"),
      model: "multiple",
    },
  });
  assert.equal(extension.messages.at(-1).type, "export-info");
  assert.equal(extension.messages.at(-1).loaded, false);
  const cancelledDirectory = path.join(root, "output-multiple-cancelled");
  assert.deepEqual(
    preferences
      .get("pathHistory")
      .open.filter(
        (entry) =>
          entry.kind === "file" &&
          path.dirname(entry.path) === cancelledDirectory,
      )
      .map((entry) => path.basename(entry.path))
      .sort(),
    ["model.onnx", "second.onnx"],
  );
  assert.equal(
    preferences
      .get("pathHistory")
      .open.filter(
        (entry) =>
          entry.kind === "directory" && entry.path === cancelledDirectory,
      ).length,
    1,
  );
  assert.ok(
    Number.isFinite(extension.messages.at(-1).durationMs) &&
      extension.messages.at(-1).durationMs > 0,
  );
  extension.selectExportedFile("second.onnx");
  await extension.receive({
    type: "export",
    exportOptions: {
      ...options("mobius", "multiple-selected"),
      model: "multiple",
    },
  });
  assert.equal(extension.messages.at(-1).loaded, true);
  assert.equal(
    extension.messages.at(-1).path,
    path.join(root, "output-multiple-selected", "second.onnx"),
  );
  assert.equal(
    preferences.get("pathHistory").open[0].path,
    extension.messages.at(-1).path,
  );
  const graphs = extension.messages.filter(
    (message) => message.type === "graph",
  ).length;
  extension.setTrusted(false);
  await extension.receive({
    type: "export",
    exportOptions: options("mbext", "blocked"),
  });
  assert.match(extension.messages.at(-1).error, /Trust this workspace/);
  assert.equal((await readdir(root)).includes("output-blocked"), false);
  extension.setTrusted(true);
  const failureNotification = extension.pauseNextErrorMessage();
  t.after(() => failureNotification.release());
  const failed = extension.receive({
    type: "export",
    exportOptions: {
      ...options("mbext", "failed"),
      model: "fail",
      precision: "fp4",
    },
  });
  await failureNotification.opened;
  assert.match(extension.messages.at(-1).error, /Failed export leaves/);
  let started;
  const progress = new Promise((resolve) => {
    started = resolve;
  });
  extension.connect((event) => {
    if (event.type === "export-progress" && event.output?.includes("Working"))
      started();
  });
  const cancelled = extension.receive({
    type: "export",
    actionId: "ui:export:cancel",
    exportOptions: { ...options("mbext", "cancelled"), model: "wait" },
  });
  await Promise.race([
    progress,
    cancelled.then(() =>
      assert.fail("The retry ended before the exporter started."),
    ),
  ]);
  failureNotification.release();
  await failed;
  await extension.receive({
    type: "export",
    exportOptions: options("mbext", "duplicate"),
  });
  assert.match(extension.messages.at(-1).error, /already running/);
  await extension.receive({ type: "cancel-export" });
  await cancelled;
  assert.ok(
    extension.logs.some((log) =>
      log.includes("[CANCELLED] Convert ONNX model"),
    ),
  );
  assert.equal(
    extension.messages.filter((message) => message.type === "graph").length,
    graphs,
  );
  const starting = new Promise((resolve) => {
    started = resolve;
  });
  const disposed = extension.receive({
    type: "export",
    exportOptions: { ...options("mbext", "disposed"), model: "wait" },
  });
  await starting;
  extension.dispose();
  const events = extension.events.length;
  await disposed;
  assert.equal(extension.events.length, events);
});

test("Settings reports defaults, versions and persisted paths without running Python", async (t) => {
  const preferences = new Map();
  const storage = await mkdtemp(path.join(tmpdir(), "viewer-settings-"));
  t.after(() => rm(storage, { recursive: true, force: true }));
  const open = () =>
    openExtension(undefined, undefined, false, storage, undefined, preferences);
  const first = await open();
  t.after(() => first.dispose());
  await first.receive({ type: "get-settings" });
  const info = first.messages.at(-1);
  assert.equal(info.type, "settings-info");
  assert.equal(
    info.application.downloadDirectory,
    path.join(homedir(), ".cache", "onnx-light-viewer"),
  );
  assert.equal(info.appVersion, require("../package.json").version);
  assert.equal(info.nativeVersion, require("../dist/graph").onnxLightVersion);
  await first.receive({ type: "get-status-info" });
  const status = first.messages.at(-1);
  assert.equal(status.type, "status-info");
  assert.equal(status.appVersion, info.appVersion);
  assert.equal(status.nativeVersion, info.nativeVersion);
  assert.deepEqual(status.application, info.application);
  assert.deepEqual(info.viewer.panels, {
    operator: true,
    find: true,
    metadata: false,
  });
  assert.equal(
    first.messages.some(({ type }) => type === "python-info"),
    false,
  );
  const settings = {
    downloadDirectory: path.join(storage, "custom"),
    pythonPath: path.join(storage, "missing-python"),
  };
  await first.receive({
    type: "application-settings",
    settings: { ...settings, password: "discard" },
  });
  assert.deepEqual(preferences.get("applicationSettings"), settings);
  assert.equal((await stat(settings.downloadDirectory)).isDirectory(), true);
  const blocked = path.join(storage, "not-a-directory");
  await writeFile(
    blocked,
    "Cannot create a download directory over this file.",
  );
  await first.receive({
    type: "application-settings",
    settings: { ...settings, downloadDirectory: blocked },
  });
  assert.equal(first.messages.at(-1).type, "settings-error");
  assert.deepEqual(preferences.get("applicationSettings"), settings);
  const second = await open();
  t.after(() => second.dispose());
  await second.receive({ type: "get-settings" });
  assert.deepEqual(second.messages.at(-1).application, settings);
  await first.receive({
    type: "application-settings",
    settings: {
      ...settings,
      pythonPath: path.join(storage, "different-python"),
    },
  });
  const inspection = {
    type: "inspect-python",
    requestId: 1,
    pythonPath: settings.pythonPath,
  };
  await second.receive(inspection);
  assert.equal(second.messages.at(-1).type, "settings-error");
  assert.match(second.messages.at(-1).error, /ENOENT/);
  assert.ok(second.messages.at(-1).error.includes(settings.pythonPath));
  assert.equal(second.messages.at(-1).requestId, inspection.requestId);
  assert.equal(second.messages.at(-1).pythonPath, inspection.pythonPath);
  second.setTrusted(false);
  await second.receive({ ...inspection, requestId: 2 });
  assert.match(second.messages.at(-1).error, /Trust this workspace/);
  assert.equal(second.messages.at(-1).requestId, 2);
});

test("Settings correlates real Python results with the panel's requested interpreter", async (t) => {
  const extension = await openExtension();
  t.after(() => extension.dispose());
  await extension.receive({ type: "get-settings" });
  const { application } = extension.messages.at(-1);
  await extension.receive({
    type: "application-settings",
    settings: {
      ...application,
      pythonPath: path.join(tmpdir(), "missing-python"),
    },
  });
  await extension.receive({
    type: "inspect-python",
    requestId: 7,
    pythonPath: application.pythonPath,
  });
  const response = extension.messages.at(-1);
  assert.equal(response.type, "python-info");
  assert.equal(response.requestId, 7);
  assert.equal(response.pythonPath, application.pythonPath);
  assert.match(response.python.version, /^\d+\.\d+\.\d+/);
  assert.ok(response.python.packages.some(({ name }) => name === "onnx"));
});

test("Settings returns a correlated busy error when a recreated webview requests inspection", async (t) => {
  const extension = await openExtension();
  t.after(() => extension.dispose());
  const request = {
    type: "inspect-python",
    pythonPath: process.platform === "win32" ? "python" : "python3",
  };
  const pending = extension.receive({ ...request, requestId: 2 });
  await extension.receive({ ...request, requestId: 1 });
  const busy = extension.messages.at(-1);
  assert.equal(busy.type, "settings-error");
  assert.equal(busy.requestId, 1);
  assert.equal(busy.pythonPath, request.pythonPath);
  assert.match(busy.error, /inspection is already running/);
  await pending;
  const original = extension.messages.at(-1);
  assert.equal(original.type, "python-info");
  assert.equal(original.requestId, 2);
  await extension.receive({ ...request, requestId: 3 });
  const retry = extension.messages.at(-1);
  assert.equal(retry.type, "python-info");
  assert.equal(retry.requestId, 3);
});

test("Settings rejects invalid paths without overwriting preferences", async (t) => {
  const extension = await openExtension();
  t.after(() => extension.dispose());
  for (const settings of [
    null,
    { downloadDirectory: "relative", pythonPath: "python3" },
    { downloadDirectory: tmpdir(), pythonPath: "" },
    { downloadDirectory: tmpdir(), pythonPath: "-c" },
    { downloadDirectory: `${tmpdir()}\0`, pythonPath: "python3" },
  ]) {
    await extension.receive({ type: "application-settings", settings });
    assert.equal(extension.messages.at(-1).type, "settings-error");
    assert.equal(extension.preferences.has("applicationSettings"), false);
  }
});

test("uses the configured directory for cached Hugging Face downloads", async (t) => {
  const storage = await mkdtemp(path.join(tmpdir(), "viewer-settings-cache-"));
  t.after(() => rm(storage, { recursive: true }));
  const downloadDirectory = path.join(storage, "custom-downloads");
  const url =
    "https://huggingface.co/viewer-tests/fixtures/resolve/main/inference.onnx";
  const key = createHash("sha256").update(url).digest("hex");
  const cache = path.join(downloadDirectory, key);
  await mkdir(cache, { recursive: true });
  await copyFile(
    path.join(__dirname, "models/inference.onnx"),
    path.join(cache, "model.onnx"),
  );
  const extension = await openExtension(undefined, undefined, false, storage);
  t.after(() => extension.dispose());
  await extension.receive({
    type: "application-settings",
    settings: { downloadDirectory, pythonPath: "python3" },
  });
  await extension.openHuggingFace(url);
  assert.equal(extension.messages.at(-1).type, "graph");
  assert.ok(extension.logs.some((entry) => entry.includes("Cache hit")));
});
test("rejects invalid preferences and reports storage failures", async () => {
  const preferences = new Map();
  const extension = await openExtension(
    undefined,
    undefined,
    false,
    undefined,
    undefined,
    preferences,
  );
  for (const settings of [
    null,
    { width: 199, panels: { operator: true, find: true, metadata: false } },
    {
      width: Infinity,
      panels: { operator: true, find: true, metadata: false },
    },
    { width: 300, panels: { operator: "yes", find: true, metadata: false } },
  ]) {
    await extension.receive({ type: "viewer-settings", settings });
  }
  assert.equal(preferences.size, 0);
  assert.equal(extension.errors.length, 4);
  assert.ok(
    extension.errors.every((message) =>
      message.startsWith("Could not save viewer settings:"),
    ),
  );
  extension.dispose();
  const failing = await openExtension(
    undefined,
    undefined,
    false,
    undefined,
    undefined,
    {
      get: () => undefined,
      set: () => {
        throw new Error("Preference storage is read-only.");
      },
    },
  );
  await failing.receive({
    type: "viewer-settings",
    settings: {
      width: 300,
      panels: { operator: true, find: true, metadata: false },
    },
  });
  assert.match(failing.errors.at(-1), /Preference storage is read-only/);
  assert.ok(
    failing.logs.some((entry) =>
      entry.includes("[FAILED] Save viewer settings"),
    ),
  );
  failing.dispose();
});

test("ignores invalid stored preferences and histories without blocking viewer startup", async (t) => {
  for (const settings of [null, { width: 300 }, { panels: false }]) {
    const preferences = new Map([
      ["viewerSettings", settings],
      ["pathHistory", { open: [null, {}], save: "old-format" }],
    ]);
    const extension = await openExtension(
      undefined,
      undefined,
      false,
      undefined,
      undefined,
      preferences,
    );
    t.after(() => extension.dispose());
    await extension.receive({ type: "ready" });
    assert.equal(
      extension.messages.some(({ type }) => type === "viewer-settings"),
      false,
    );
    assert.equal(
      extension.events.some(({ type }) => type === "path-history"),
      false,
    );
    assert.ok(extension.messages.some(({ type }) => type === "graph"));
    assert.ok(
      extension.errors.some((message) =>
        /invalid saved path history/i.test(message),
      ),
    );
    assert.ok(
      extension.errors.some((message) =>
        message.startsWith("Could not restore viewer settings:"),
      ),
    );
    assert.ok(
      extension.logs.some((entry) =>
        entry.includes("[FAILED] Restore viewer settings"),
      ),
    );
  }
});

test("routes task-filtered Hub searches and drops superseded or disposed search results", async () => {
  const extension = await openExtension();
  const first = {
    id: "author/first",
    url: "https://huggingface.co/author/first",
    downloads: 10,
  };
  const second = {
    id: "author/second",
    url: "https://huggingface.co/author/second",
    downloads: 20,
  };
  extension.setHubSearchResults([first]);
  const release = extension.pauseNextHubSearch();
  const pending = extension.receive({
    type: "search-huggingface",
    query: "first",
    task: "",
    requestId: 1,
  });

  extension.setHubSearchResults([second]);
  await extension.receive({
    type: "search-huggingface",
    query: "second",
    task: "text-generation",
    requestId: 2,
  });
  assert.equal(extension.hubSearches[0].signal.aborted, true);
  release();
  await pending;
  let responses = extension.events.filter(
    ({ type }) => type === "huggingface-search-results",
  );
  assert.equal(responses.length, 1);
  assert.deepEqual(responses[0], {
    type: "huggingface-search-results",
    requestId: 2,
    models: [second],
  });
  assert.equal(extension.hubSearches[1].task, "text-generation");
  const fail = extension.pauseNextHubSearch(
    new Error("Hub search unavailable"),
  );
  const failed = extension.receive({
    type: "search-huggingface",
    query: "",
    task: "",
    requestId: 3,
  });
  fail();
  await failed;
  assert.match(
    extension.events.find(({ type }) => type === "huggingface-search-error")
      .error,
    /unavailable/,
  );
  const close = extension.pauseNextHubSearch();
  const closing = extension.receive({
    type: "search-huggingface",
    query: "",
    task: "",
    requestId: 4,
  });
  extension.dispose();
  assert.equal(extension.hubSearches.at(-1).signal.aborted, true);
  close();
  await closing;
  responses = extension.events.filter(
    ({ type }) => type === "huggingface-search-results",
  );
  assert.equal(responses.length, 1);
});

test("publishes repository file sizes after search results without downloading models", async () => {
  const extension = await openExtension();
  const repository = {
    id: "author/model",
    url: "https://huggingface.co/author/model",
    downloads: 10,
  };
  extension.setHubSearchResults([repository]);
  extension.setRepositoryFiles([
    {
      url: `${repository.url}/resolve/main/model.onnx`,
      path: "model.onnx",
      fileName: "model.onnx",
      size: 1024,
    },
    {
      url: `${repository.url}/resolve/main/unknown.onnx`,
      path: "unknown.onnx",
      fileName: "unknown.onnx",
    },
  ]);
  const nativeCount = extension.nativeModels.length;
  await extension.receive({
    type: "search-huggingface",
    query: "",
    task: "",
    requestId: 21,
  });
  const resultsIndex = extension.events.findIndex(
    ({ type }) => type === "huggingface-search-results",
  );
  const sizesIndex = extension.events.findIndex(
    ({ type }) => type === "huggingface-model-files",
  );
  assert.ok(resultsIndex >= 0 && sizesIndex > resultsIndex);
  assert.deepEqual(extension.events[sizesIndex], {
    type: "huggingface-model-files",
    requestId: 21,
    id: repository.id,
    entries: [
      {
        name: "model.onnx",
        path: `${repository.url}/resolve/main/model.onnx`,
        directory: false,
        size: 1024,
      },
      {
        name: "unknown.onnx",
        path: `${repository.url}/resolve/main/unknown.onnx`,
        directory: false,
      },
    ],
  });
  assert.equal(extension.nativeModels.length, nativeCount);
  extension.dispose();
});

test("history pipeline inspection preserves history and assembles Mobius through the local open action", async (t) => {
  const directory = await mkdtemp(
    path.join(tmpdir(), "onnx-viewer-history-pipeline-"),
  );
  t.after(() => rm(directory, { recursive: true, force: true }));
  const { promptModel } = require("./helpers/prompt-model.cjs");
  const model = await promptModel(directory, true);
  await writeFile(
    `${model}.provenance.json`,
    JSON.stringify({
      version: 1,
      exporter: "mobius",
      model: "owner/model",
      source: directory,
    }),
  );
  const history = {
    open: [{ source: "local", kind: "file", path: model }],
    save: [],
  };
  const preferences = new Map([["pathHistory", history]]);
  const extension = await openExtension(
    undefined,
    undefined,
    true,
    undefined,
    undefined,
    preferences,
  );
  t.after(() => extension.dispose());
  await extension.receive({
    type: "prompt-pipeline-history",
    requestId: 1,
    paths: [model],
  });
  assert.equal(extension.messages.at(-1).entries[0].available, true);
  assert.deepEqual(preferences.get("pathHistory"), history);
  await extension.receive({
    type: "prompt-pipeline-history",
    requestId: 2,
    paths: [path.join(directory, "not-in-history.onnx")],
  });
  assert.match(extension.messages.at(-1).error, /Invalid pipeline history/);
  const open = {
    type: "open",
    source: "local",
    location: model,
    password: "",
    asPipeline: true,
  };
  extension.setTrusted(false);
  await extension.receive(open);
  assert.match(extension.messages.at(-1).error, /Trust this workspace/);
  assert.deepEqual(preferences.get("pathHistory"), history);
  extension.setTrusted(true);
  await extension.receive(open);
  assert.ok(extension.messages.some(({ type }) => type === "show-prompt"));
  const graph = extension.messages.findLast(({ type }) => type === "graph");
  assert.deepEqual(
    graph.graph.nodes.map((node) => node.name),
    ["prefill"],
  );
  const pipeline = preferences.get("pathHistory").open[0].path;
  assert.match(pipeline, /viewer-.*\.pipe-onnx$/);
  assert.ok((await stat(pipeline)).isFile());
});

test("remembers only successful Open and Save paths and supports persistent removal and clearing", async (t) => {
  const directory = await mkdtemp(
    path.join(tmpdir(), "onnx-viewer-path-history-"),
  );
  t.after(() => rm(directory, { recursive: true }));
  const preferences = new Map();
  const extension = await openExtension(
    undefined,
    undefined,
    true,
    undefined,
    undefined,
    preferences,
  );
  const source = require.resolve("./models/relu.onnx");
  await extension.openFile(source);
  const opened = preferences.get("pathHistory");
  assert.deepEqual(
    opened.open.map(({ path: location }) => location),
    [source, path.dirname(source)],
  );
  await extension.openFile(source);
  assert.deepEqual(preferences.get("pathHistory"), opened);
  await extension.openFile(path.join(directory, "missing.onnx"));
  await extension.cancelOpen();
  assert.deepEqual(preferences.get("pathHistory"), opened);
  const destination = path.join(directory, "copy.onnx");
  await extension.save(destination);
  assert.equal(preferences.get("pathHistory").save[0].path, destination);
  const saved = structuredClone(preferences.get("pathHistory"));
  await extension.save(undefined);
  await extension.save(source);
  assert.deepEqual(preferences.get("pathHistory"), saved);
  extension.dispose();
  const reopened = await openExtension(
    undefined,
    undefined,
    true,
    undefined,
    undefined,
    preferences,
  );
  await reopened.receive({ type: "ready" });
  assert.deepEqual(
    reopened.events.find(({ type }) => type === "path-history").history,
    saved,
  );
  await reopened.receive({
    type: "path-history-edit",
    target: "open",
    historyAction: "remove",
    historyEntry: saved.open[0],
  });
  assert.deepEqual(preferences.get("pathHistory").open, saved.open.slice(1));
  await reopened.receive({
    type: "path-history-edit",
    target: "save",
    historyAction: "clear",
  });
  assert.deepEqual(preferences.get("pathHistory").save, []);
  assert.deepEqual(preferences.get("pathHistory").open, saved.open.slice(1));
  reopened.dispose();
});

test("displaying histories removes missing local paths but preserves references, templates and surviving order", async (t) => {
  const { pathToFileURL } = require("node:url");
  const directory = await mkdtemp(path.join(tmpdir(), "viewer-prune-history-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const file = path.join(directory, "keep.onnx");
  await writeFile(file, "keep");
  const missing = path.join(directory, "missing.onnx");
  const missingDirectory = path.join(directory, "missing");
  const relativeDirectory = `${path.basename(directory)}-conversion-output`;
  await mkdir(path.join(directory, relativeDirectory));
  const entry = (location, kind = "file", source = "local") => ({
    source,
    kind,
    path: location,
  });
  const remote = entry(
    "https://huggingface.co/owner/model/resolve/main/model.onnx",
    "file",
    "huggingface",
  );
  const backend = entry("test_cc_relu", "file", "backend");
  const retained = [
    remote,
    entry(file),
    backend,
    entry(directory, "directory"),
    entry(pathToFileURL(file).href),
  ];
  const template = path.join(directory, "<model_id>", "<exporter>");
  const preferences = new Map([
    [
      "applicationSettings",
      { downloadDirectory: directory, pythonPath: "python3" },
    ],
    [
      "pathHistory",
      {
        open: [
          entry(missing),
          ...retained,
          entry(missingDirectory, "directory"),
          entry(path.join(file, "missing.onnx")),
        ],
        save: [entry(missing), entry(file), entry(directory, "directory")],
      },
    ],
    [
      "exportHistory",
      {
        models: [missingDirectory, "Qwen/Qwen3-0.6B", directory],
        directories: [missingDirectory, relativeDirectory, directory, template],
        exporter: "mobius",
      },
    ],
  ]);
  const extension = await openExtension(
    undefined,
    undefined,
    true,
    undefined,
    undefined,
    preferences,
  );
  t.after(() => extension.dispose());
  await extension.receive({ type: "ready" });
  await extension.receive({ type: "get-path-history", target: "open" });
  assert.deepEqual(preferences.get("pathHistory"), {
    open: retained,
    save: [entry(file), entry(directory, "directory")],
  });
  assert.deepEqual(
    extension.events.findLast(({ type }) => type === "path-history").history,
    preferences.get("pathHistory"),
  );
  assert.equal(await readFile(file, "utf8"), "keep");
  await extension.receive({ type: "get-settings" });
  const settings = extension.messages.findLast(
    ({ type }) => type === "settings-info",
  );
  assert.deepEqual(settings.exportHistory, {
    models: ["Qwen/Qwen3-0.6B", directory],
    directories: [relativeDirectory, directory, template],
    exporter: "mobius",
  });
  assert.deepEqual(settings.pathHistory, preferences.get("pathHistory"));
  const paths = preferences.get("pathHistory");
  const exports = preferences.get("exportHistory");
  await extension.receive({ type: "get-settings" });
  assert.equal(preferences.get("pathHistory"), paths);
  assert.equal(preferences.get("exportHistory"), exports);
  await rm(file);
  await extension.receive({ type: "get-path-history", target: "save" });
  assert.deepEqual(preferences.get("pathHistory"), {
    open: [remote, backend, entry(directory, "directory")],
    save: [entry(directory, "directory")],
  });
  const reopened = await openExtension(
    undefined,
    undefined,
    true,
    undefined,
    undefined,
    preferences,
  );
  t.after(() => reopened.dispose());
  await reopened.receive({ type: "ready" });
  assert.deepEqual(
    reopened.events.findLast(({ type }) => type === "path-history").history,
    preferences.get("pathHistory"),
  );
  assert.deepEqual(extension.errors, []);
});

test("loading an existing history file preserves the complete stored history and its order", async (t) => {
  const source = require.resolve("./models/relu.onnx");
  const history = {
    open: [
      { source: "local", kind: "file", path: "/exports/other.onnx" },
      { source: "backend", kind: "file", path: "test_cc_relu" },
      { source: "local", kind: "file", path: source },
      { source: "local", kind: "directory", path: path.dirname(source) },
    ],
    save: [{ source: "local", kind: "file", path: "/copies/other.onnx" }],
  };
  const expected = structuredClone(history);
  const preferences = new Map([["pathHistory", history]]);
  const extension = await openExtension(
    undefined,
    undefined,
    true,
    undefined,
    undefined,
    preferences,
  );
  t.after(() => extension.dispose());
  await extension.receive({ type: "ready" });
  await extension.openFile(source);
  assert.equal(
    extension.messages.findLast(({ type }) => type === "graph").graph.nodes[0]
      .opType,
    "Relu",
  );
  assert.deepEqual(preferences.get("pathHistory"), expected);
  await extension.resolveCustomEditor(source);
  assert.deepEqual(preferences.get("pathHistory"), expected);
  assert.deepEqual(
    extension.events.findLast(({ type }) => type === "path-history").history,
    expected,
  );
  assert.equal(extension.errors.length, 0);
});

test("publishes replacement graphs before delayed Open history persistence", async (t) => {
  for (const source of ["local", "backend"]) {
    const extension = await openExtension();
    t.after(() => extension.dispose());
    await extension.receive({ type: "ready" });
    const initial = extension.messages.findLast(({ type }) => type === "graph");
    const gate = extension.pauseNextHistoryUpdate();
    const opening =
      source === "local"
        ? extension.open("relu.onnx")
        : extension.chooseBackend("test_cc_relu", "open-backend-test");
    await gate.started;
    try {
      const replacement = extension.messages.findLast(
        ({ type }) => type === "graph",
      );
      assert.notEqual(replacement, initial);
      assert.equal(replacement.graph.nodes[0].opType, "Relu");
      assert.equal(extension.preferences.get("pathHistory"), undefined);
      assert.equal(extension.errors.length, 0);
    } finally {
      gate.release();
      await opening;
    }
    assert.equal(
      extension.preferences.get("pathHistory").open[0].source,
      source === "local" ? "local" : "backend",
    );
  }
});

test("records models opened directly by the editor and restores their Open history", async (t) => {
  const directory = await mkdtemp(
    path.join(tmpdir(), "viewer-editor-history-"),
  );
  t.after(() => rm(directory, { recursive: true, force: true }));
  const file = path.join(directory, "model.onnx");
  await copyFile(require.resolve("./models/relu.onnx"), file);
  const preferences = new Map();
  const extension = await openExtension(
    undefined,
    undefined,
    true,
    undefined,
    undefined,
    preferences,
  );
  t.after(() => extension.dispose());
  await extension.resolveCustomEditor(file);
  await extension.receive({ type: "ready" });
  const expected = [
    { source: "local", kind: "file", path: file },
    { source: "local", kind: "directory", path: directory },
  ];
  assert.deepEqual(preferences.get("pathHistory").open, expected);
  assert.deepEqual(
    extension.events.findLast(({ type }) => type === "path-history").history
      .open,
    expected,
  );
  const reopened = await openExtension(
    undefined,
    undefined,
    true,
    undefined,
    undefined,
    preferences,
  );
  t.after(() => reopened.dispose());
  await reopened.receive({ type: "ready" });
  assert.deepEqual(
    reopened.events.findLast(({ type }) => type === "path-history").history
      .open,
    expected,
  );
});

test("publishes initial editor graphs before delayed Open history persistence", async (t) => {
  const extension = await openExtension();
  t.after(() => extension.dispose());
  const releaseRead = extension.deferRead("relu.onnx");
  const gate = extension.pauseNextHistoryUpdate();
  const opening = extension.resolveCustomEditor("relu.onnx");
  await extension.receive({ type: "ready" });
  releaseRead();
  await gate.started;
  try {
    const graph = extension.messages.findLast(({ type }) => type === "graph");
    assert.ok(graph);
    assert.equal(graph.graph.nodes[0].opType, "Relu");
    assert.equal(extension.preferences.get("pathHistory"), undefined);
    assert.equal(extension.errors.length, 0);
  } finally {
    gate.release();
    await opening;
  }
  const history = extension.preferences.get("pathHistory").open;
  assert.equal(history[0].source, "local");
  assert.equal(history[0].kind, "file");
  assert.match(history[0].path, /\/relu\.onnx$/);
});

test("path paste reads the VS Code clipboard without loading a model and reports clipboard errors", async () => {
  const extension = await openExtension("relu.onnx");
  const reads = extension.reads.length;
  const location = "/home/models/qwen3-0.6b-mobius/model.onnx";
  extension.setClipboard(location);
  await extension.receive({ type: "paste-model-path", requestId: 1 });
  assert.deepEqual(extension.messages.at(-1), {
    type: "model-path-selected",
    requestId: 1,
    location,
  });
  extension.setClipboard("");
  await extension.receive({ type: "paste-model-path", requestId: 2 });
  assert.equal(extension.messages.at(-1).error, "The clipboard is empty.");
  extension.setClipboard("", "Clipboard unavailable");
  await extension.receive({ type: "paste-model-path", requestId: 3 });
  assert.equal(extension.messages.at(-1).error, "Clipboard unavailable");
  assert.equal(extension.reads.length, reads);
  assert.equal(extension.openDialogs.length, 0);
});

test("native path selection preserves the picker and returns a path without loading or replacing the model", async () => {
  const extension = await openExtension("relu.onnx");
  const reads = extension.reads.length;
  const graphs = extension.messages.filter(
    ({ type }) => type === "graph",
  ).length;
  const selected = path.join(tmpdir(), "outside-workspace.onnx");
  extension.setOpenSelection(selected);
  await extension.receive({ type: "select-model-path", requestId: 1 });
  assert.deepEqual(extension.messages.at(-1), {
    type: "model-path-selected",
    requestId: 1,
    location: selected,
  });
  assert.deepEqual(extension.openDialogs.at(-1), {
    canSelectMany: false,
    filters: { "ONNX models and pipelines": ["onnx", "enc", "pipe-onnx"] },
    openLabel: "Open ONNX graph",
  });
  extension.setOpenSelection(undefined);
  await extension.receive({ type: "select-model-path", requestId: 2 });
  assert.equal(extension.messages.at(-1).location, null);
  extension.failNextOpenDialog("Dialog unavailable");
  await extension.receive({ type: "select-model-path", requestId: 3 });
  assert.equal(extension.messages.at(-1).error, "Dialog unavailable");
  assert.equal(extension.reads.length, reads);
  assert.equal(
    extension.messages.filter(({ type }) => type === "graph").length,
    graphs,
  );
  assert.deepEqual(extension.errors, []);
});

test("opens outside-workspace models through the path dialog and saves encrypted copies without popup pickers", async (t) => {
  const directory = await mkdtemp(
    path.join(tmpdir(), "onnx-light-inline-files-"),
  );
  t.after(() => rm(directory, { recursive: true, force: true }));
  const source = path.join(directory, "source.onnx");
  await copyFile(require.resolve("./models/fusion.onnx"), source);
  await mkdir(path.join(directory, "nested"));
  await writeFile(path.join(directory, "notes.txt"), "not a model");
  const original = await readFile(source);
  const extension = await openExtension(
    undefined,
    "onnxLightViewer.open",
    true,
  );
  const dom = new JSDOM(extension.webview.html, { runScripts: "outside-only" });
  t.after(() => dom.window.close());
  const { window } = dom;
  window.structuredClone = structuredClone;
  const { document } = window;
  const pathDialog = document.getElementById("open-path-dialog");
  pathDialog.showModal = () => {
    pathDialog.open = true;
  };
  pathDialog.close = () => {
    pathDialog.open = false;
  };
  const requests = [];
  window.HTMLCanvasElement.prototype.getContext = () => ({
    font: "",
    measureText: (text) => ({ width: text.length * 8 }),
  });

  document.getElementById("canvas").getBoundingClientRect = () =>
    new window.DOMRect(0, 0, 1060, 350);
  window.acquireVsCodeApi = () => ({
    postMessage: (message) => requests.push(extension.receive(message)),
  });
  extension.connect((data) =>
    window.dispatchEvent(new window.MessageEvent("message", { data })),
  );
  for (const script of document.querySelectorAll("script[src]"))
    window.eval(readFileSync(new URL(script.src), "utf8"));
  const flush = async () => {
    while (requests.length) await Promise.all(requests.splice(0));
  };
  const submit = (id) =>
    document
      .getElementById(id)
      .dispatchEvent(new window.Event("submit", { cancelable: true }));
  await flush();
  document.getElementById("open-model").click();
  await flush();
  document.getElementById("open-path-browse").click();
  await flush();
  document.getElementById("open-location").value = directory;
  submit("open-browse");
  await flush();
  let files = document.querySelectorAll("#open-files button");
  assert.deepEqual(
    [...files].map((button) => button.textContent),
    ["[Folder] nested", "source.onnx"],
  );
  files[0].click();
  await flush();
  assert.equal(
    document.getElementById("open-location").value,
    path.join(directory, "nested"),
  );
  document.getElementById("open-parent").click();
  await flush();
  files = document.querySelectorAll("#open-files button");
  files[1].click();
  document.getElementById("choose-open-path").click();
  assert.equal(pathDialog.open, false);
  assert.equal(document.getElementById("open-file").value, source);
  assert.equal(extension.openDialogs.length, 0);
  submit("open-form");
  await flush();
  assert.equal(document.querySelectorAll(".graph-node").length, 5);
  document.getElementById("memory-tab").click();
  await flush();
  submit("memory-form");
  await flush();
  assert.equal(document.getElementById("memory-results").hidden, false);
  assert.equal(document.getElementById("memory-peak").textContent, "512 bytes");
  assert.equal(document.querySelectorAll("#memory-events tr").length, 5);
  document.getElementById("model-tab").click();
  document.getElementById("save-model").click();
  await flush();
  assert.equal(document.getElementById("save-options").hidden, false);
  assert.equal(document.getElementById("open-panel").hidden, true);
  assert.equal(document.querySelector("dialog[open]"), null);
  const encrypted = document.querySelector('input[value="embedded-encrypted"]');
  encrypted.click();
  const destination = path.join(directory, "source-copy.onnx.enc");
  assert.equal(document.getElementById("save-file").value, destination);
  document.getElementById("save-password").value = "inline-password";
  document.getElementById("save-confirmation").value = "wrong";
  submit("save-form");
  assert.match(
    document.getElementById("save-options-error").textContent,
    /same non-empty password/,
  );
  document.getElementById("save-confirmation").value = "inline-password";
  submit("save-form");
  assert.equal(document.getElementById("save-submit").disabled, true);
  assert.equal(document.getElementById("save-password").value, "");
  await flush();
  const { loadModelFile } = require("../dist/graph");
  assert.equal(
    (await loadModelFile(destination, "copy", "inline-password")).graph().nodes
      .length,
    5,
  );
  assert.equal(document.getElementById("save-submit").disabled, false);
  assert.match(document.getElementById("save-status").textContent, /Saved/);
  const saved = await readFile(destination);
  for (const overwrite of [false, true]) {
    document.getElementById("save-overwrite").checked = overwrite;
    document.getElementById("save-password").value = "inline-password";
    document.getElementById("save-confirmation").value = "inline-password";
    submit("save-form");
    await flush();
    if (!overwrite) {
      assert.match(
        document.getElementById("save-options-error").textContent,
        /exist/i,
      );
      assert.deepEqual(await readFile(destination), saved);
    } else
      assert.match(document.getElementById("save-status").textContent, /Saved/);
  }
  document.getElementById("open-model").click();
  await flush();
  assert.equal(document.getElementById("save-options").hidden, true);
  document.getElementById("open-file").value = destination;
  document.getElementById("open-password").value = "inline-password";
  submit("open-form");
  await flush();
  assert.equal(document.querySelectorAll(".graph-node").length, 5);
  assert.deepEqual(await readFile(source), original);
  assert.deepEqual(extension.picks, []);
  assert.deepEqual(extension.passwordPrompts, []);
  assert.deepEqual(extension.saveDialogs, []);
  assert.equal(extension.errors.length, 1);
  assert.ok(
    extension.logs.every((entry) => !entry.includes("inline-password")),
  );
});

test("lists and opens backend models in the embedded selector without a quick pick", async () => {
  const extension = await openExtension();
  await extension.receive({
    type: "browse-files",
    target: "open",
    source: "backend",
    requestId: 1,
  });
  const listing = extension.messages.at(-1);
  assert.equal(listing.type, "file-list");
  assert.ok(listing.entries.length > 0);
  await extension.receive({
    type: "open",
    source: "backend",
    location: listing.entries[0].path,
    password: "",
    actionId: "ui:inline-backend",
  });
  assert.equal(extension.messages.at(-1).type, "graph");
  assert.deepEqual(extension.picks, []);
  assert.deepEqual(extension.errors, []);
  const location =
    "https://huggingface.co/example/models/resolve/main/model.onnx";
  await extension.receive({
    type: "browse-files",
    source: "huggingface",
    target: "open",
    location,
    requestId: 2,
  });
  const remote = extension.messages.at(-1);
  assert.equal(remote.type, "file-list");
  assert.equal(remote.entries.length, 1);
  assert.equal(remote.entries[0].name, "model.onnx");
  assert.equal(remote.entries[0].path, location);
  await extension.receive({
    type: "browse-files",
    source: "local",
    target: "open",
    location: "/does-not-exist/onnx-light-viewer",
    requestId: 3,
  });
  assert.equal(extension.messages.at(-1).type, "file-list-error");
});

test("opens pipeline files, navigates stages, applies stage fusions and saves the entire container", async (t) => {
  const directory = await mkdtemp(
    path.join(tmpdir(), "onnx-light-pipeline-host-"),
  );
  t.after(() => rm(directory, { recursive: true, force: true }));
  const field = (number, bytes) => {
    let length = bytes.length;
    const prefix = [number * 8 + 2];
    do {
      prefix.push((length % 128) | (length >= 128 ? 128 : 0));
      length = Math.floor(length / 128);
    } while (length);
    return Buffer.concat([Buffer.from(prefix), bytes]);
  };
  const bytes = Buffer.concat([
    field(1, Buffer.from("Pipeline")),
    field(8, await readFile(require.resolve("./models/fusion.onnx"))),
    field(9, await readFile(require.resolve("./models/relu.onnx"))),
  ]);
  const source = path.join(directory, "source.pipe-onnx");
  await writeFile(source, bytes);
  const extension = await openExtension(undefined, undefined, true);
  t.after(() => extension.dispose());
  await extension.openFile(source);
  const latestGraph = () =>
    extension.messages.findLast(({ type }) => type === "graph");
  assert.equal(latestGraph().graph.pipeline, true);
  assert.equal(latestGraph().statistics, undefined);
  assert.match(latestGraph().file.savePath, /source-copy\.pipe-onnx$/);
  await extension.receive({
    type: "browse-files",
    target: "open",
    source: "local",
    location: directory,
    requestId: 1,
  });
  assert.ok(
    extension.messages
      .at(-1)
      .entries.some(({ name }) => name === "source.pipe-onnx"),
  );
  await extension.receive({ type: "fusion-patterns", requestId: 2 });
  assert.equal(extension.messages.at(-1).type, "fusion-error");
  const stagePath = [{ pipeline: "prefill" }];
  await extension.receive({ type: "navigate", path: stagePath, nodeIndex: 0 });
  assert.equal(latestGraph().graph.breadcrumbs[1].name, "prefill");
  assert.deepEqual(latestGraph().selectedNodes, [0]);
  assert.equal(latestGraph().statistics, undefined);
  await extension.receive({ type: "get-statistics", requestId: 1 });
  const originalStatistics = extension.messages.at(-1);
  assert.equal(originalStatistics.type, "statistics-info");
  assert.equal(originalStatistics.statistics.nodes.length, 2);
  await extension.receive({ type: "refresh-shapes" });
  assert.equal(extension.messages.at(-1).type, "shape-info");
  await extension.receive({ type: "metadata" });
  assert.equal(extension.messages.at(-1).type, "metadata-info");
  await extension.receive({
    type: "preview-fusion",
    pattern: "TransposeTranspose",
    requestId: 3,
  });
  const preview = extension.messages.at(-1);
  assert.equal(preview.type, "fusion-preview");
  await extension.receive({
    type: "apply-fusion",
    previewId: preview.previewId,
    anchors: [0, 2],
    requestId: 4,
  });
  const stage = latestGraph().graph;
  assert.deepEqual(latestGraph().file.fusionHistory, [
    {
      matches: [
        { pattern: "TransposeTranspose", nodes: [0] },
        { pattern: "TransposeTranspose", nodes: [1] },
      ],
      stage: "prefill",
    },
  ]);
  assert.equal(stage.nodes[0].opType, "Identity");
  assert.deepEqual(stage.breadcrumbs.at(-1).path, stagePath);
  await extension.receive({ type: "get-statistics", requestId: 2 });
  const updatedStatistics = extension.messages.at(-1);
  assert.equal(updatedStatistics.type, "statistics-info");
  assert.equal(
    updatedStatistics.statistics.nodes[0].pipelineModel.nodeCount,
    stage.nodes.length,
  );
  assert.ok(
    updatedStatistics.statistics.nodes[0].pipelineModel.nodeCount <
      originalStatistics.statistics.nodes[0].pipelineModel.nodeCount,
  );
  const destination = path.join(directory, "copy.pipe-onnx");
  await extension.save(destination);
  assert.deepEqual(extension.saveDialogs.at(-1).filters, {
    "ONNX pipelines": ["pipe-onnx"],
  });
  const { loadModelFile } = require("../dist/graph");
  const restored = await loadModelFile(destination, "copy.pipe-onnx");
  assert.equal(restored.graph().nodes.length, 2);
  assert.equal(restored.graph(false, stagePath).nodes[0].opType, "Identity");
  assert.equal(
    restored.graph(false, [{ pipeline: "decode" }]).nodes[0].opType,
    "Relu",
  );
  assert.deepEqual(await readFile(source), bytes);
  await extension.receive({ type: "navigate", path: [] });
  assert.equal(latestGraph().graph.canFuse, false);
  assert.equal(latestGraph().statistics, undefined);
  await extension.receive({
    type: "select-fusion",
    operation: 0,
    match: 1,
    pattern: "TransposeTranspose",
  });
  assert.deepEqual(latestGraph().graph.breadcrumbs.at(-1).path, stagePath);
  assert.deepEqual(latestGraph().selectedNodes, [1]);
  assert.equal(latestGraph().file.fusionCount, 1);
  const encrypted = path.join(directory, "copy.pipe-onnx.enc");
  await extension.save(encrypted, true, ["secret", "secret"]);
  assert.match(extension.errors.at(-1), /PipelineProto encryption/);
  await assert.rejects(readFile(encrypted), { code: "ENOENT" });
  await extension.receive({ type: "undo-fusion", all: true });
  assert.equal(latestGraph().file.fusionCount, 0);
  assert.equal(latestGraph().file.pendingRewritings[0].stage, "prefill");
  assert.equal(latestGraph().file.pendingRewritings[0].matches.length, 2);
  assert.deepEqual(latestGraph().graph.breadcrumbs.at(-1).path, stagePath);
  const queuedPath = path.join(directory, "queued.pipe-onnx");
  await extension.receive({
    type: "save",
    location: queuedPath,
    overwrite: false,
    includeRewritings: true,
  });
  const queued = await loadModelFile(queuedPath, "queued.pipe-onnx");
  const scenario = queued.rewritingScenarios(stagePath)[0];
  const reapplied = await queued.applyRewritingScenario(
    scenario.name,
    true,
    undefined,
    stagePath,
  );
  assert.deepEqual(reapplied.graph(false, stagePath), stage);
  await extension.receive({ type: "replay-rewritings", requestId: 5 });
  assert.deepEqual(latestGraph().graph, stage);
  assert.deepEqual(latestGraph().file.pendingRewritings, []);
  assert.equal(latestGraph().file.fusionCount, 1);
});

test("lists saved scenarios and applies one or all remaining steps with host undo", async (t) => {
  const { loadModel } = require("../dist/graph");
  const sourceModel = loadModel(
    readFileSync(require.resolve("./models/fusion.onnx")),
    "scenario",
  );
  const first = await sourceModel.applyFusion("TransposeTranspose", [0]);
  const full = await first.applyFusion("TransposeTranspose", [1]);
  const directory = await mkdtemp(path.join(tmpdir(), "onnx-light-scenarios-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const source = path.join(directory, "scenario.onnx");
  await writeFile(source, await full.serialize(undefined, undefined, true));
  const extension = await openExtension(undefined, undefined, true);
  t.after(() => extension.dispose());
  await extension.openFile(source);
  const latestGraph = () =>
    extension.messages.findLast((message) => message.type === "graph");
  await extension.receive({ type: "fusion-patterns", requestId: 1 });
  const name = extension.messages.at(-1).scenarios[0].name;
  assert.deepEqual(
    extension.messages.at(-1).scenarios[0].steps[0].nodes,
    [0, 1],
  );
  await extension.receive({
    type: "apply-rewriting",
    scenario: name,
    all: false,
    requestId: 2,
  });
  assert.equal(extension.messages.at(-1).type, "rewriting-info");
  assert.equal(extension.messages.at(-1).requestId, 2);
  assert.equal(latestGraph().fusionRequestId, 2);
  assert.equal(extension.messages.at(-1).scenarios[0].applied, 1);
  assert.deepEqual(
    extension.messages.at(-1).scenarios[0].steps[1].nodes,
    [1, 2],
  );
  assert.deepEqual(latestGraph().graph, first.graph());
  assert.equal(latestGraph().file.fusionCount, 1);
  await extension.receive({
    type: "apply-rewriting",
    scenario: name,
    all: true,
    requestId: 3,
  });
  assert.deepEqual(latestGraph().graph, full.graph());
  assert.equal(extension.messages.at(-1).scenarios[0].applied, 2);
  await extension.receive({ type: "undo-fusion" });
  assert.deepEqual(latestGraph().graph, first.graph());
  await extension.receive({ type: "fusion-patterns", requestId: 4 });
  assert.equal(extension.messages.at(-1).scenarios[0].applied, 1);
  await extension.receive({ type: "undo-fusion", all: true });
  assert.deepEqual(latestGraph().graph, sourceModel.graph());
  await extension.receive({
    type: "apply-rewriting",
    scenario: "missing",
    all: true,
    requestId: 5,
  });
  assert.equal(extension.messages.at(-1).type, "fusion-error");
  assert.match(extension.messages.at(-1).error, /Unknown/);
  const completions = extension.messages.filter(
    (message) => message.type === "rewriting-info",
  ).length;
  extension.connect((message) => {
    if (message.type === "graph")
      void extension.receive({ type: "cancel-fusion" });
  });
  await extension.receive({
    type: "apply-rewriting",
    scenario: name,
    all: true,
    requestId: 6,
  });
  assert.equal(
    extension.messages.filter((message) => message.type === "rewriting-info")
      .length,
    completions,
  );
  assert.deepEqual(latestGraph().graph, full.graph());
});

test("validates ONNX Script requests and requires workspace trust without editing", async (t) => {
  const directory = await mkdtemp(path.join(__dirname, "onnxscript-host-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const extension = await openExtension(
    "../../examples/optimization.onnx",
    undefined,
    true,
    directory,
    undefined,
    new Map([
      [
        "applicationSettings",
        {
          downloadDirectory: directory,
          pythonPath: path.join(directory, "missing-python"),
        },
      ],
    ]),
  );
  t.after(() => extension.dispose());
  const latest = () =>
    extension.messages.findLast(({ type }) => type === "graph");
  const original = latest();
  const request = {
    type: "apply-optimization",
    backend: "onnxscript",
    operation: "Optimize",
    patterns: [],
    requestId: 223,
  };
  extension.setTrusted(false);
  await extension.receive(request);
  assert.match(extension.messages.at(-1).error, /Trust this workspace/);
  extension.setTrusted(true);
  for (const invalid of [
    { backend: "unknown" },
    { operation: "Fuse" },
    { patterns: ["IdentityRemoval"] },
  ]) {
    await extension.receive({ ...request, ...invalid });
    assert.equal(extension.messages.at(-1).type, "fusion-error");
    assert.equal(latest(), original);
  }
  await extension.receive(request);
  assert.match(extension.messages.at(-1).error, /missing-python/);
  assert.equal(extension.messages.at(-1).requestId, 223);
  assert.equal(latest(), original);
  assert.deepEqual(await readdir(directory), []);
});

const onnxScriptPython =
  process.env.ONNX_SCRIPT_PYTHON ??
  (process.platform === "win32" ? "python" : "python3");
const hasOnnxScript =
  spawnSync(onnxScriptPython, ["-P", "-c", "import onnx, onnxscript"])
    .status === 0;

test(
  "ONNX Script materializes the current snapshot without retaining undo checkpoints",
  {
    skip:
      !hasOnnxScript &&
      "Install onnx and onnxscript or set ONNX_SCRIPT_PYTHON.",
  },
  async (t) => {
    const directory = await mkdtemp(path.join(__dirname, "onnxscript-host-"));
    t.after(() => rm(directory, { recursive: true, force: true }));
    const extension = await openExtension(
      undefined,
      undefined,
      true,
      directory,
      undefined,
      new Map([
        [
          "applicationSettings",
          {
            downloadDirectory: directory,
            pythonPath: onnxScriptPython,
          },
        ],
      ]),
    );
    t.after(() => extension.dispose());
    const source = path.join(directory, "original.onnx");
    await copyFile(require.resolve("../examples/optimization.onnx"), source);
    await extension.openFile(source);
    const latest = () =>
      extension.messages.findLast(({ type }) => type === "graph");
    await extension.receive({
      type: "apply-optimization",
      operation: "Optimize",
      patterns: ["Identity"],
      requestId: 1,
    });
    assert.equal(
      extension.messages.at(-1).type,
      "optimization-info",
      JSON.stringify(extension.messages.at(-1)),
    );
    assert.equal(latest().file.fusionCount, 1);
    // The host must optimize its current edited handle, not reload this source.
    await writeFile(source, "source replaced after loading");
    await extension.receive({
      type: "apply-optimization",
      backend: "onnxscript",
      operation: "Optimize",
      patterns: [],
      requestId: 2,
    });
    const response = extension.messages.at(-1);
    assert.equal(response.type, "optimization-info", JSON.stringify(response));
    assert.equal(response.backend, "onnxscript");
    assert.equal(response.requestId, 2);
    assert.equal(typeof response.changed, "boolean");
    assert.deepEqual(response.matches, []);
    assert.ok(Array.isArray(response.patterns));
    assert.ok(Array.isArray(response.optimizationPatterns));
    assert.deepEqual(response.scenarios, []);
    assert.equal(response.changed, true);
    assert.equal(response.undoSupported, false);
    assert.equal(response.notice, undefined);
    assert.match(response.logs, /After optimization:.*Identity nodes/);
    assert.match(response.logs, /After fusion:.*Identity nodes/);
    assert.ok(
      extension.logs.some(
        (entry) =>
          entry.includes("[SUCCEEDED] Optimize ONNX model") &&
          entry.includes("After optimization:") &&
          entry.includes("After fusion:"),
      ),
    );
    assert.equal(latest().fusionRequestId, 2);
    assert.equal(latest().file.fusionCount, 0);
    assert.deepEqual(latest().file.fusionHistory, []);
    const optimized = latest().graph;
    assert.equal(
      optimized.nodes.some(({ name }) => name === "constant_sum"),
      false,
    );
    const saved = path.join(directory, "saved.onnx");
    await extension.receive({
      type: "save",
      location: saved,
      overwrite: false,
      storage: "embedded",
      encrypted: false,
      includeRewritings: true,
    });
    assert.equal(extension.messages.at(-1).type, "save-info");
    const loaded = await require("../dist/graph").loadModelFile(
      saved,
      optimized.name,
    );
    assert.deepEqual(loaded.graph(), optimized);
    assert.deepEqual(loaded.rewritingScenarios(), []);
    assert.equal(
      await readFile(source, "utf8"),
      "source replaced after loading",
    );
    assert.equal(
      (await readdir(directory)).some((name) => name.startsWith("onnxscript-")),
      false,
    );
    const optimizeNative = (patterns) =>
      extension.receive({
        type: "apply-optimization",
        operation: "Optimize",
        patterns,
      });
    const optimizePython = () =>
      extension.receive({
        type: "apply-optimization",
        backend: "onnxscript",
        operation: "Optimize",
        patterns: [],
      });
    await optimizeNative(["GemmTranspose"]);
    assert.equal(extension.messages.at(-1).type, "optimization-info");
    assert.equal(extension.messages.at(-1).changed, false);
    assert.deepEqual(extension.messages.at(-1).matches, []);
    assert.deepEqual(latest().graph, optimized);
    assert.equal(latest().file.fusionCount, 0);
    assert.deepEqual(latest().file.fusionHistory, []);
    await extension.receive({ type: "undo-fusion" });
    assert.equal(extension.messages.at(-1).type, "undo-fusion-error");
    assert.match(extension.messages.at(-1).error, /No fusion operation/);
    // Leave MatMulAdd for native undo coverage; GemmTranspose absorbs input transposes only.
    await optimizeNative(["MatMulAdd"]);
    assert.equal(extension.messages.at(-1).type, "optimization-info");
    assert.equal(extension.messages.at(-1).changed, true);
    assert.ok(
      extension.messages
        .at(-1)
        .matches.some(({ pattern }) => pattern === "MatMulAdd"),
    );
    assert.notDeepEqual(latest().graph, optimized);
    assert.equal(latest().file.fusionCount, 1);
    await extension.receive({ type: "undo-fusion" });
    assert.deepEqual(latest().graph, optimized);
    assert.equal(latest().file.fusionCount, 0);
    await extension.receive({ type: "undo-fusion" });
    assert.equal(extension.messages.at(-1).type, "undo-fusion-error");
    assert.match(extension.messages.at(-1).error, /No fusion operation/);
    assert.deepEqual(latest().graph, optimized);
    assert.equal(latest().file.fusionCount, 0);
    await optimizeNative(["MatMulAdd"]);
    assert.equal(extension.messages.at(-1).type, "optimization-info");
    assert.equal(extension.messages.at(-1).changed, true);
    assert.equal(latest().file.fusionCount, 1);
    await extension.receive({ type: "undo-fusion", all: true });
    assert.deepEqual(latest().graph, optimized);
    assert.equal(latest().file.fusionCount, 0);
    await extension.openFile(require.resolve("./models/relu.onnx"));
    const unchanged = latest().graph;
    await optimizePython();
    assert.equal(extension.messages.at(-1).changed, false);
    assert.equal(extension.messages.at(-1).undoSupported, false);
    assert.match(extension.messages.at(-1).notice, /history is retained/);
    assert.deepEqual(latest().graph, unchanged);
    assert.equal(latest().file.fusionCount, 0);
  },
);

test(
  "ONNX Script retains source weight ownership after importing embedded results",
  {
    skip:
      !hasOnnxScript &&
      "Install onnx and onnxscript or set ONNX_SCRIPT_PYTHON.",
  },
  async (t) => {
    const directory = await mkdtemp(path.join(__dirname, "onnxscript-host-"));
    t.after(() => rm(directory, { recursive: true, force: true }));
    const source = path.join(directory, "source.onnx");
    const weights = path.join(directory, "weights.bin");
    execFileSync(onnxScriptPython, [
      "-P",
      "-c",
      `
import sys, onnx
model = onnx.load(sys.argv[1])
onnx.save_model(model, sys.argv[2], save_as_external_data=True,
                all_tensors_to_one_file=True, location="weights.bin", size_threshold=0)
`,
      require.resolve("../examples/optimization.onnx"),
      source,
    ]);
    const sourceBytes = await readFile(source);
    const weightBytes = await readFile(weights);
    const alias = path.join(directory, "weight-alias.bin");
    const extension = await openExtension(
      undefined,
      undefined,
      true,
      directory,
      undefined,
      new Map([
        [
          "applicationSettings",
          {
            downloadDirectory: directory,
            pythonPath: onnxScriptPython,
          },
        ],
      ]),
    );
    t.after(() => extension.dispose());
    await extension.openFile(source);
    await extension.receive({
      type: "apply-optimization",
      backend: "onnxscript",
      operation: "Optimize",
      patterns: [],
      requestId: 223,
    });
    const result = extension.messages.at(-1);
    assert.equal(result.type, "optimization-info", JSON.stringify(result));
    assert.equal(result.changed, true);
    const copy = path.join(directory, "embedded.onnx");
    await extension.save(copy);
    assert.equal(extension.messages.at(-1).type, "save-info");
    const embedded = await require("../dist/graph").loadModelFile(
      copy,
      "embedded",
    );
    assert.deepEqual(embedded.externalDataLocations(), []);
    await link(weights, alias);
    const writeCount = extension.writes.length;
    for (const destination of [weights, alias]) {
      await extension.receive({
        type: "save",
        location: destination,
        overwrite: true,
        storage: "embedded",
        encrypted: false,
      });
      assert.equal(extension.messages.at(-1).type, "save-error");
      assert.match(extension.messages.at(-1).error, /different destination/);
      assert.equal(extension.writes.length, writeCount);
      assert.deepEqual(await readFile(source), sourceBytes);
      assert.deepEqual(await readFile(weights), weightBytes);
      assert.deepEqual(await readFile(alias), weightBytes);
    }
  },
);

test(
  "discards late ONNX Script results and errors after replacing the model",
  {
    skip:
      (!hasOnnxScript || process.platform === "win32") &&
      "Requires ONNX Script and POSIX executable scripts.",
  },
  async (t) => {
    const directory = await mkdtemp(path.join(__dirname, "onnxscript-host-"));
    t.after(() => rm(directory, { recursive: true, force: true }));
    for (const fail of [false, true]) {
      const marker = path.join(directory, `started-${fail}`);
      const release = path.join(directory, `release-${fail}`);
      const executable = path.join(directory, `python-${fail}`);
      await writeFile(
        executable,
        `#!${process.execPath}
const fs = require("node:fs");
fs.writeFileSync(${JSON.stringify(marker)}, "");
while (!fs.existsSync(${JSON.stringify(release)}))
  Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 10);
if (${fail}) process.exit(3);
const result = require("node:child_process").spawnSync(
  ${JSON.stringify(onnxScriptPython)}, process.argv.slice(2), { stdio: "inherit" });
process.exit(result.status ?? 1);
`,
        { mode: 0o700 },
      );
      const extension = await openExtension(
        "../../examples/optimization.onnx",
        undefined,
        true,
        directory,
        undefined,
        new Map([
          [
            "applicationSettings",
            {
              downloadDirectory: directory,
              pythonPath: executable,
            },
          ],
        ]),
      );
      t.after(() => extension.dispose());
      const pending = extension.receive({
        type: "apply-optimization",
        backend: "onnxscript",
        operation: "Optimize",
        patterns: [],
        requestId: 223,
      });
      try {
        for (let attempt = 0; ; attempt++) {
          try {
            await stat(marker);
            break;
          } catch {
            assert.ok(attempt < 500, "Python optimization never started.");
            await new Promise((resolve) => setTimeout(resolve, 10));
          }
        }
        await extension.openFile(
          require.resolve("../examples/optimization.onnx"),
        );
        const count = extension.messages.length;
        const errors = extension.errors.length;
        await writeFile(release, "");
        await pending;
        assert.equal(extension.messages.length, count);
        assert.equal(extension.errors.length, errors);
        assert.equal(
          (await readdir(directory)).some((name) =>
            name.startsWith("onnxscript-"),
          ),
          false,
        );
      } finally {
        await writeFile(release, "");
        await pending;
        extension.dispose();
      }
    }
  },
);

test("previews global passes without editing, rejects stale previews and optimizes with replayable saves", async (t) => {
  const directory = await mkdtemp(
    path.join(tmpdir(), "onnx-light-optimization-"),
  );
  t.after(() => rm(directory, { recursive: true, force: true }));
  const extension = await openExtension(undefined, undefined, true);
  t.after(() => extension.dispose());
  const source = require.resolve("../examples/optimization.onnx");
  await extension.openFile(source);
  const originalBytes = await readFile(source);
  const latest = () =>
    extension.messages.findLast((message) => message.type === "graph");
  const original = latest().graph;
  await extension.receive({ type: "fusion-patterns", requestId: 1 });
  const catalog = extension.messages.at(-1).optimizationPatterns;
  assert.ok(catalog.some((pattern) => pattern.name === "GemmTranspose"));
  await extension.receive({
    type: "preview-fusion",
    pattern: "IdentityRemoval",
    requestId: 2,
  });
  const stale = extension.messages.at(-1);
  assert.equal(stale.type, "fusion-preview");
  assert.equal(stale.matches[0].global, true);
  assert.ok(stale.matches[0].nodes.length);
  assert.deepEqual(latest().graph, original);
  assert.equal(latest().file.fusionCount, 0);
  await extension.receive({ type: "cancel-fusion" });
  await extension.receive({
    type: "apply-fusion",
    previewId: stale.previewId,
    candidates: [0],
    requestId: 3,
  });
  assert.equal(extension.messages.at(-1).type, "fusion-error");
  assert.match(extension.messages.at(-1).error, /stale/);
  await extension.receive({
    type: "preview-fusion",
    pattern: "ConstantFolding",
    requestId: 4,
  });
  const preview = extension.messages.at(-1);
  await extension.receive({
    type: "apply-fusion",
    previewId: preview.previewId,
    candidates: [0],
    requestId: 5,
  });
  assert.equal(extension.messages.at(-1).type, "fusion-info");
  assert.equal(extension.messages.at(-1).requestId, 5);
  assert.equal(latest().fusionRequestId, 5);
  assert.equal(latest().file.fusionCount, 1);
  assert.equal(
    latest().graph.nodes.some((node) => node.name === "constant_sum"),
    false,
  );
  await extension.receive({ type: "undo-fusion" });
  assert.deepEqual(latest().graph, original);
  await extension.receive({
    type: "apply-optimization",
    operation: "Optimize",
    patterns: ["unknown"],
    requestId: 6,
  });
  assert.equal(extension.messages.at(-1).type, "fusion-error");
  assert.equal(latest().file.fusionCount, 0);
  await extension.receive({
    type: "apply-optimization",
    operation: "Optimize",
    patterns: ["TransposeTranspose", "MatMulAdd", "GemmTranspose"],
    requestId: 7,
  });
  assert.equal(extension.messages.at(-1).type, "optimization-info");
  const report = extension.messages.at(-1).optimizationReport;
  assert.ok(report.wallTimeNs > 0);
  assert.equal(
    report.patterns.some(
      (pattern) => pattern.name === "GemmTranspose" && pattern.matches > 0,
    ),
    false,
  );
  assert.ok(
    report.patterns.some(
      (pattern) => pattern.name === "MatMulAdd" && pattern.matches > 0,
    ),
  );
  assert.equal(
    extension.messages
      .at(-1)
      .matches.some((match) => match.pattern === "GemmTranspose"),
    false,
  );
  const optimized = latest().graph;
  await extension.receive({ type: "fusion-patterns", requestId: 8 });
  assert.deepEqual(extension.messages.at(-1).optimizationReport, report);
  const { loadModelFile } = require("../dist/graph");
  const plain = path.join(directory, "plain.onnx");
  const variants = path.join(directory, "variants.onnx");
  for (const [location, includeRewritings] of [
    [plain, false],
    [variants, true],
  ]) {
    await extension.receive({
      type: "save",
      location,
      overwrite: false,
      storage: "embedded",
      encrypted: false,
      includeRewritings,
    });
    assert.equal(extension.messages.at(-1).type, "save-info");
  }
  assert.deepEqual((await loadModelFile(plain, "plain")).graph(), optimized);
  const saved = await loadModelFile(variants, "variants");
  assert.deepEqual(saved.graph(), original);
  const replayed = await saved.applyRewritingScenario(
    saved.rewritingScenarios()[0].name,
    true,
  );
  assert.deepEqual(replayed.graph(), optimized);
  assert.deepEqual(await readFile(source), originalBytes);
  await extension.receive({ type: "undo-fusion", all: true });
  assert.deepEqual(latest().graph, original);
  await extension.receive({ type: "fusion-patterns", requestId: 9 });
  assert.equal(extension.messages.at(-1).optimizationReport, null);
});

test("creates a native function pattern and retains it through preview, undo, replay and recovery", async (t) => {
  const extension = await openExtension("fusion.onnx", undefined, true);
  t.after(() => extension.dispose());
  await extension.receive({ type: "ready" });
  const latest = () =>
    extension.messages.findLast(({ type }) => type === "graph");
  assert.ok(latest(), extension.errors.join("\n"));
  const original = latest().graph;
  const expect = (type) =>
    assert.equal(
      extension.messages.at(-1).type,
      type,
      extension.messages.at(-1).error,
    );
  await extension.receive({
    type: "preview-fusion",
    pattern: "TransposeTranspose",
    requestId: 0,
  });
  expect("fusion-preview");
  const stale = extension.messages.at(-1);
  await extension.receive({
    type: "create-function-pattern",
    nodes: [0, 1],
    name: "HostTransposePair",
    requestId: 1,
  });
  expect("function-pattern-created");
  const created = extension.messages.at(-1);
  assert.equal(created.requestId, 1);
  assert.equal(created.pattern.name, "HostTransposePair");
  assert.ok(created.pattern.text.length > 0);
  assert.match(created.pattern.text, /Transpose/);
  assert.ok(created.pattern.python.length > 0);
  assert.ok(created.patterns.includes(created.pattern.name));
  assert.deepEqual(
    created.optimizationPatterns.find(
      ({ name }) => name === created.pattern.name,
    ).definition,
    created.pattern,
  );
  assert.ok(Array.isArray(created.scenarios));
  assert.deepEqual(latest().graph, original);
  assert.equal(latest().file.fusionCount, 0);
  await extension.receive({
    type: "apply-fusion",
    previewId: stale.previewId,
    candidates: [0],
    requestId: 101,
  });
  expect("fusion-error");
  await extension.receive({
    type: "preview-fusion",
    pattern: created.pattern.name,
    requestId: 2,
  });
  expect("fusion-preview");
  const preview = extension.messages.at(-1);
  assert.ok(preview.matches.length > 0);
  await extension.receive({
    type: "apply-fusion",
    previewId: preview.previewId,
    candidates: [0],
    requestId: 3,
  });
  expect("fusion-info");
  await extension.receive({ type: "undo-fusion", requestId: 4 });
  expect("graph");
  assert.equal(latest().file.fusionCount, 0);
  assert.ok(latest().file.pendingRewritings.length > 0);
  await extension.receive({
    type: "create-function-pattern",
    nodes: [0, 1],
    name: "HostQueuedTransposePair",
    requestId: 5,
  });
  expect("function-pattern-created");
  await extension.receive({ type: "replay-rewritings", requestId: 6 });
  expect("optimization-info");
  assert.ok(
    extension.messages.at(-1).patterns.includes("HostQueuedTransposePair"),
  );
  await extension.receive({ type: "undo-fusion", all: true, requestId: 7 });
  await extension.receive({ type: "fusion-patterns", requestId: 8 });
  assert.ok(
    extension.messages.at(-1).patterns.includes("HostQueuedTransposePair"),
  );
  await extension.open("fusion.onnx");
  const native = extension.nativeModels.at(-1);
  const graph = native.graph;
  native.graph = (infer, ...args) => {
    if (infer) throw new Error("Host test shape failure");
    return graph(infer, ...args);
  };
  await extension.receive({
    type: "create-function-pattern",
    nodes: [4],
    name: "HostRecoveryMatMul",
    requestId: 9,
  });
  expect("function-pattern-created");
  await extension.receive({ type: "refresh-shapes" });
  expect("shape-error");
  await extension.receive({ type: "fusion-patterns", requestId: 10 });
  assert.ok(extension.messages.at(-1).patterns.includes("HostRecoveryMatMul"));
  await extension.receive({
    type: "apply-optimization",
    operation: "Optimize",
    patterns: ["HostRecoveryMatMul"],
    requestId: 11,
  });
  expect("optimization-info");
  assert.ok(latest().file.fusionCount > 0);
});

test("function pattern copy uses the VS Code clipboard for current edits and reports validation and clipboard errors", async (t) => {
  const extension = await openExtension();
  t.after(() => extension.dispose());
  for (const [format, text] of [
    ["text", "UnsavedMatch (x) => (y) {\n y = Relu(x)\n}"],
    ["replacement", "UnsavedReplacement (x) => (y) {\n y = Identity(x)\n}"],
    ["python", 'print("unsaved Python 🐍")\n'],
    ["text", ""],
  ]) {
    await extension.receive({
      type: "copy-function-pattern",
      name: "UnsavedPattern",
      format,
      text,
    });
    assert.equal(extension.clipboardWrites.at(-1), text);
    assert.deepEqual(extension.messages.at(-1), {
      type: "function-pattern-copied",
      name: "UnsavedPattern",
      format,
    });
  }
  const writes = extension.clipboardWrites.length;
  const request = {
    type: "copy-function-pattern",
    name: "UnsavedPattern",
    format: "text",
    text: "current edits",
  };
  for (const fields of [
    { name: undefined },
    { name: 7 },
    { text: undefined },
    { text: {} },
    { format: "unsupported" },
    { format: undefined },
    { format: ["text"] },
  ]) {
    await extension.receive({ ...request, ...fields });
    const response = extension.messages.at(-1);
    assert.equal(response.type, "function-pattern-copy-error");
    assert.match(response.error, /Expected/);
    assert.equal(extension.clipboardWrites.length, writes);
  }
  extension.setClipboard("", "Clipboard permission denied");
  await extension.receive(request);
  assert.deepEqual(extension.messages.at(-1), {
    type: "function-pattern-copy-error",
    name: request.name,
    format: request.format,
    error: "Clipboard permission denied",
  });
  assert.equal(extension.clipboardWrites.length, writes);
});

test("function pattern updates apply edited SDK bodies and invalidate stale previews", async (t) => {
  const extension = await openExtension("fusion.onnx", undefined, true);
  t.after(() => extension.dispose());
  const expect = (type) =>
    assert.equal(
      extension.messages.at(-1).type,
      type,
      extension.messages.at(-1).error,
    );
  await extension.receive({
    type: "create-function-pattern",
    nodes: [0, 1],
    name: "HostEditablePair",
    requestId: 1,
  });
  expect("function-pattern-created");
  const definition = extension.messages.at(-1).pattern;
  assert.equal(definition.replacement, definition.text);
  await extension.receive({
    type: "preview-fusion",
    pattern: definition.name,
    requestId: 2,
  });
  expect("fusion-preview");
  const stale = extension.messages.at(-1);
  await extension.receive({
    type: "update-function-pattern",
    name: definition.name,
    text: definition.text,
    replacement:
      '<domain: "onnx_light.viewer", opset_import: ["" : 13]> HostEditedReplacement(x) => (a) { a = Identity(x) }',
    requestId: 3,
  });
  expect("function-pattern-created");
  const updated = extension.messages.at(-1);
  assert.equal(updated.requestId, 3);
  assert.match(updated.pattern.replacement, /Identity/);
  assert.deepEqual(
    updated.optimizationPatterns.find(({ name }) => name === definition.name)
      .definition,
    updated.pattern,
  );
  await extension.receive({
    type: "apply-fusion",
    previewId: stale.previewId,
    candidates: [0],
    requestId: 4,
  });
  expect("fusion-error");
  await extension.receive({
    type: "preview-fusion",
    pattern: definition.name,
    requestId: 5,
  });
  expect("fusion-preview");
  const preview = extension.messages.at(-1);
  assert.equal(
    preview.matches[0].replacement[0].opType,
    "HostEditedReplacement",
  );
  await extension.receive({
    type: "apply-fusion",
    previewId: preview.previewId,
    candidates: [0],
    requestId: 6,
  });
  expect("fusion-info");
  await extension.receive({ type: "undo-fusion", requestId: 7 });
  expect("graph");
  await extension.receive({ type: "fusion-patterns", requestId: 8 });
  assert.deepEqual(
    extension.messages
      .at(-1)
      .optimizationPatterns.find(({ name }) => name === definition.name)
      .definition,
    updated.pattern,
  );
});

test("function pattern update transport rejects malformed fields and preserves definitions on SDK errors", async (t) => {
  const extension = await openExtension("fusion.onnx", undefined, true);
  t.after(() => extension.dispose());
  const native = extension.nativeModels.at(-1);
  const definition = native.createFunctionPattern([0, 1], "HostEditablePair");
  const update = native.updateFunctionPattern;
  let calls = 0;
  native.updateFunctionPattern = (...args) => {
    calls++;
    return update(...args);
  };
  const request = {
    type: "update-function-pattern",
    name: definition.name,
    text: definition.text,
    replacement: definition.replacement,
    requestId: 41,
  };
  for (const fields of [
    { name: 7 },
    { text: undefined },
    { text: [] },
    { replacement: null },
    { replacement: {} },
  ]) {
    await extension.receive({ ...request, ...fields });
    assert.equal(extension.messages.at(-1).type, "function-pattern-error");
    assert.equal(extension.messages.at(-1).requestId, 41);
  }
  assert.equal(calls, 0);
  for (const fields of [
    { name: "TransposeTranspose" },
    { text: "invalid function text" },
    { replacement: "invalid replacement text" },
  ]) {
    await extension.receive({ ...request, ...fields });
    assert.equal(extension.messages.at(-1).type, "function-pattern-error");
    assert.deepEqual(
      native.optimizationPatterns().find(({ name }) => name === definition.name)
        .definition,
      definition,
    );
  }
  assert.equal(calls, 3);
});

test("function pattern transport rejects malformed selections and delegates semantic names to native", async (t) => {
  const extension = await openExtension("fusion.onnx", undefined, true);
  t.after(() => extension.dispose());
  const native = extension.nativeModels.at(-1);
  const create = native.createFunctionPattern;
  let calls = 0;
  native.createFunctionPattern = (...args) => {
    calls++;
    return create(...args);
  };
  for (const nodes of [
    undefined,
    [],
    Array(1),
    [0, , 1],
    [0, 0],
    [-1],
    [native.graph().nodes.length],
    [0.5],
    ["0"],
    [NaN],
    [Infinity],
    [Number.MAX_SAFE_INTEGER + 1],
  ]) {
    await extension.receive({
      type: "create-function-pattern",
      nodes,
      name: "HostInvalidSelection",
      requestId: 11,
    });
    assert.equal(extension.messages.at(-1).type, "function-pattern-error");
    assert.equal(extension.messages.at(-1).requestId, 11);
  }
  await extension.receive({
    type: "create-function-pattern",
    nodes: [0, 1],
    name: 7,
    requestId: 12,
  });
  assert.equal(extension.messages.at(-1).type, "function-pattern-error");
  assert.equal(calls, 0);
  await extension.receive({
    type: "create-function-pattern",
    nodes: [0, 1],
    name: "HostCollision",
    requestId: 13,
  });
  assert.equal(
    extension.messages.at(-1).type,
    "function-pattern-created",
    extension.messages.at(-1).error,
  );
  await extension.receive({
    type: "create-function-pattern",
    nodes: [0, 1],
    name: "HostCollision",
    requestId: 14,
  });
  assert.equal(extension.messages.at(-1).type, "function-pattern-error");
  assert.equal(calls, 2);
});

test("function pattern creation and updates recover native handles and suppress stale recovery results and errors", async (t) => {
  for (const updating of [false, true]) {
    for (const invalidate of [
      "cancel",
      "navigate",
      "replace",
      "dispose",
      "none",
    ]) {
      for (const fail of [false, true]) {
        const extension = await openExtension("fusion.onnx", undefined, true);
        t.after(() => extension.dispose());
        const native = extension.nativeModels.at(-1);
        let resumeReload;
        if (updating) {
          const clone = native.clone;
          const gate = new Promise((resolve) => {
            resumeReload = resolve;
          });
          native.clone = () => {
            const retained = clone();
            retained.clone = () =>
              gate.then(() => {
                if (fail) throw new Error("Host delayed recovery failure");
                return clone();
              });
            return retained;
          };
        }
        if (updating)
          await extension.receive({
            type: "create-function-pattern",
            nodes: [0, 1],
            name: "HostDelayedPair",
          });
        const definition = updating
          ? extension.messages.at(-1).pattern
          : undefined;
        const graph = native.graph;
        native.graph = (infer, ...args) => {
          if (infer) throw new Error("Host test invalidation");
          return graph(infer, ...args);
        };
        await extension.receive({ type: "refresh-shapes" });
        const release = updating
          ? resumeReload
          : extension.pauseNextFileLoad(
              fail ? new Error("Host delayed recovery failure") : undefined,
            );
        const pending = extension.receive({
          type: updating
            ? "update-function-pattern"
            : "create-function-pattern",
          nodes: [0, 1],
          name: "HostDelayedPair",
          text: definition?.text,
          replacement: definition?.replacement,
          requestId: 91,
        });
        let navigation;
        if (invalidate === "replace") await extension.open("relu.onnx");
        else if (invalidate === "dispose") extension.dispose();
        else if (invalidate === "navigate")
          navigation = extension.receive({ type: "navigate", path: [] });
        else if (invalidate !== "none")
          await extension.receive({ type: "cancel-fusion" });
        const errors = extension.errors.length;
        release();
        await pending;
        await navigation;
        const result = extension.messages.filter(
          ({ type, requestId }) =>
            requestId === 91 &&
            ["function-pattern-created", "function-pattern-error"].includes(
              type,
            ),
        );
        if (invalidate === "none") {
          assert.equal(result.length, 1);
          assert.equal(
            result[0].type,
            fail ? "function-pattern-error" : "function-pattern-created",
            result[0].error,
          );
        } else {
          assert.deepEqual(result, []);
          assert.equal(
            extension.errors
              .slice(errors)
              .some((error) =>
                error.includes("Could not save function pattern"),
              ),
            false,
          );
        }
      }
    }
  }
});

test("function pattern creation is stage-local and rejects pipeline overviews and nested graphs", async (t) => {
  const field = (number, bytes) => {
    let length = bytes.length;
    const prefix = [number * 8 + 2];
    do {
      prefix.push((length % 128) | (length >= 128 ? 128 : 0));
      length = Math.floor(length / 128);
    } while (length);
    return Buffer.concat([Buffer.from(prefix), bytes]);
  };
  const bytes = Buffer.concat([
    field(1, Buffer.from("HostPatternPipeline")),
    field(8, readFileSync(require.resolve("./models/fusion.onnx"))),
    field(9, readFileSync(require.resolve("./models/fusion.onnx"))),
  ]);
  const extension = await openExtension();
  t.after(() => extension.dispose());
  const expect = (type) =>
    assert.equal(
      extension.messages.at(-1).type,
      type,
      extension.messages.at(-1).error,
    );
  await extension.openBytes("host-pattern.pipe-onnx", bytes);
  const request = {
    type: "create-function-pattern",
    nodes: [0, 1],
    name: "HostStagePair",
    requestId: 21,
  };
  await extension.receive(request);
  expect("function-pattern-error");
  await extension.receive({
    type: "navigate",
    path: [{ pipeline: "prefill" }],
  });
  await extension.receive(request);
  expect("function-pattern-created");
  const initialDefinition = extension.messages.at(-1).pattern;
  const update = {
    type: "update-function-pattern",
    name: request.name,
    text: initialDefinition.text,
    replacement:
      '<domain: "onnx_light.viewer", opset_import: ["" : 13]> HostStageReplacement(x) => (a) { a = Identity(x) }',
    requestId: 31,
  };
  await extension.receive(update);
  expect("function-pattern-created");
  const stageDefinition = extension.messages.at(-1).pattern;
  assert.match(stageDefinition.replacement, /HostStageReplacement/);
  await extension.receive({
    type: "preview-fusion",
    pattern: request.name,
    requestId: 22,
  });
  expect("fusion-preview");
  const preview = extension.messages.at(-1);
  await extension.receive({
    type: "apply-fusion",
    previewId: preview.previewId,
    candidates: [0],
    requestId: 23,
  });
  expect("fusion-info");
  await extension.receive({ type: "undo-fusion", requestId: 24 });
  expect("graph");
  await extension.receive({
    ...request,
    name: "HostStageQueuedPair",
    requestId: 25,
  });
  expect("function-pattern-created");
  await extension.receive({ type: "replay-rewritings", requestId: 26 });
  expect("optimization-info");
  assert.ok(extension.messages.at(-1).patterns.includes("HostStageQueuedPair"));
  await extension.receive({ type: "undo-fusion", all: true, requestId: 27 });
  await extension.receive({ type: "fusion-patterns", requestId: 28 });
  const definition = extension.messages
    .at(-1)
    .optimizationPatterns.find(
      ({ name }) => name === "HostStageQueuedPair",
    ).definition;
  assert.equal(definition.name, "HostStageQueuedPair");
  assert.ok(definition.text.length > 0);
  assert.ok(definition.python.length > 0);
  await extension.receive({ type: "navigate", path: [{ pipeline: "decode" }] });
  await extension.receive(update);
  expect("function-pattern-error");
  await extension.receive({ type: "fusion-patterns", requestId: 29 });
  for (const name of ["HostStagePair", "HostStageQueuedPair"])
    assert.equal(extension.messages.at(-1).patterns.includes(name), false);
  await extension.receive({
    type: "navigate",
    path: [{ pipeline: "prefill" }],
  });
  await extension.receive({ type: "fusion-patterns", requestId: 30 });
  assert.deepEqual(
    extension.messages
      .at(-1)
      .optimizationPatterns.find(({ name }) => name === request.name)
      .definition,
    stageDefinition,
  );
  assert.deepEqual(
    extension.messages
      .at(-1)
      .optimizationPatterns.find(({ name }) => name === "HostStageQueuedPair")
      .definition,
    definition,
  );
  await extension.open("subgraphs.onnx");
  const graph = extension.messages.findLast(
    ({ type }) => type === "graph",
  ).graph;
  const child = graph.nodes.flatMap((node) =>
    node.attributes.flatMap((attribute) => attribute.graphs ?? []),
  )[0];
  assert.ok(child);
  await extension.receive({ type: "navigate", path: child.path });
  await extension.receive(request);
  expect("function-pattern-error");
  await extension.receive(update);
  expect("function-pattern-error");
});

test("explains native pattern matches without invalidating a matching preview", async (t) => {
  const extension = await openExtension("fusion.onnx", undefined, true);
  t.after(() => extension.dispose());
  await extension.receive({
    type: "preview-fusion",
    pattern: "TransposeTranspose",
    requestId: 1,
  });
  const preview = extension.messages.at(-1);
  for (const [pattern, anchor, status] of [
    ["TransposeTranspose", 0, "matched"],
    ["TransposeTranspose", 1, "rejected"],
    ["GemmTranspose", 0, "op-type-mismatch"],
  ]) {
    await extension.receive({
      type: "explain-fusion",
      pattern,
      anchor,
      requestId: 2,
    });
    const result = extension.messages.at(-1);
    assert.equal(result.type, "fusion-diagnostic");
    assert.equal(result.requestId, 2);
    assert.equal(result.diagnostic.pattern, pattern);
    assert.equal(result.diagnostic.anchor, anchor);
    assert.equal(result.diagnostic.status, status);
  }
  await extension.receive({
    type: "explain-fusion",
    pattern: "TransposeTranspose",
    anchor: -1,
    requestId: 3,
  });
  assert.equal(extension.messages.at(-1).type, "fusion-diagnostic-error");
  await extension.receive({
    type: "apply-fusion",
    previewId: preview.previewId,
    candidates: [0],
    requestId: 4,
  });
  assert.equal(extension.messages.at(-1).type, "fusion-info");
});

test("discards diagnosis results and errors after cancellation, navigation or replacement", async (t) => {
  for (const invalidate of ["cancel", "navigate", "replace"]) {
    for (const fail of [false, true]) {
      const extension = await openExtension("fusion.onnx", undefined, true);
      t.after(() => extension.dispose());
      const native = extension.nativeModels.at(-1);
      const explainFusion = native.explainFusion;
      let release, arrived;
      const gate = new Promise((resolve) => {
        release = resolve;
      });
      const started = new Promise((resolve) => {
        arrived = resolve;
      });
      native.explainFusion = async (...args) => {
        const result = await explainFusion(...args);
        arrived();
        await gate;
        if (fail) throw new Error("Late diagnostic failure");
        return result;
      };
      const pending = extension.receive({
        type: "explain-fusion",
        pattern: "TransposeTranspose",
        anchor: 0,
        requestId: 71,
      });
      await started;
      if (invalidate === "replace") await extension.open("relu.onnx");
      else
        await extension.receive({
          type:
            invalidate === "cancel" ? "cancel-fusion-diagnostic" : "navigate",
          path: [],
        });
      const count = extension.messages.length;
      const errors = extension.errors.length;
      release();
      await pending;
      assert.equal(extension.messages.length, count);
      assert.equal(extension.errors.length, errors);
    }
  }
});

test("discards optimization preview results after replacing their owning model", async (t) => {
  const extension = await openExtension(undefined, undefined, true);
  t.after(() => extension.dispose());
  await extension.openFile(require.resolve("../examples/optimization.onnx"));
  const native = extension.nativeModels.at(-1);
  const previewFusion = native.previewFusion;
  let release, arrived;
  const gate = new Promise((resolve) => {
    release = resolve;
  });
  const started = new Promise((resolve) => {
    arrived = resolve;
  });
  native.previewFusion = async (...args) => {
    const result = await previewFusion(...args);
    arrived();
    await gate;
    return result;
  };
  const pending = extension.receive({
    type: "preview-fusion",
    pattern: "ConstantFolding",
    requestId: 1,
  });
  await started;
  await extension.openFile(require.resolve("./models/relu.onnx"));
  release();
  await pending;
  assert.equal(
    extension.messages.some((message) => message.type === "fusion-preview"),
    false,
  );
  assert.equal(
    extension.messages.some((message) => message.type === "fusion-error"),
    false,
  );
  const graph = extension.messages.findLast(
    (message) => message.type === "graph",
  );
  assert.deepEqual(
    graph.graph.nodes.map((node) => node.opType),
    ["Relu"],
  );
  assert.equal(graph.file.fusionCount, 0);
});

test("unfuses the last operation or all successive native fusions atomically", async () => {
  const extension = await openExtension(
    undefined,
    "onnxLightViewer.open",
    true,
  );
  await extension.openFile(require.resolve("./models/fusion.onnx"));
  const latestGraph = () =>
    extension.messages.findLast((message) => message.type === "graph");
  const original = latestGraph().graph.nodes;
  let requestId = 0;
  const fuseNext = async () => {
    await extension.receive({
      type: "preview-fusion",
      pattern: "TransposeTranspose",
      requestId: ++requestId,
    });
    const preview = extension.messages.at(-1);
    assert.equal(preview.type, "fusion-preview");
    await extension.receive({
      type: "apply-fusion",
      previewId: preview.previewId,
      candidates: [0],
      requestId: ++requestId,
    });
    assert.equal(extension.messages.at(-1).type, "fusion-info");
    assert.equal(extension.messages.at(-1).requestId, requestId);
    assert.equal(
      extension.messages.findLast((message) => message.type === "graph")
        .fusionRequestId,
      requestId,
    );
    assert.equal(
      extension.messages.at(-1).patterns.includes("TransposeTranspose"),
      latestGraph().graph.nodes.some((node) => node.opType === "Transpose"),
    );
  };
  await fuseNext();
  await fuseNext();
  assert.equal(latestGraph().file.fusionCount, 2);
  assert.deepEqual(latestGraph().file.fusionHistory, [
    { matches: [{ pattern: "TransposeTranspose", nodes: [0] }] },
    { matches: [{ pattern: "TransposeTranspose", nodes: [1] }] },
  ]);
  await extension.receive({ type: "undo-fusion", all: "all" });
  assert.equal(extension.messages.at(-1).type, "undo-fusion-error");
  assert.match(extension.messages.at(-1).error, /boolean/);
  assert.equal(latestGraph().file.fusionCount, 2);
  await extension.receive({ type: "undo-fusion", all: false });
  assert.equal(latestGraph().file.fusionCount, 1);
  assert.deepEqual(latestGraph().file.fusionHistory, [
    { matches: [{ pattern: "TransposeTranspose", nodes: [0] }] },
  ]);
  assert.equal(latestGraph().graph.nodes.length, original.length - 1);
  const pendingGraph = latestGraph();
  for (const request of [
    { type: "preview-fusion", pattern: "UnknownPattern" },
    {
      type: "apply-optimization",
      operation: "Optimize",
      patterns: ["UnknownPattern"],
    },
  ]) {
    await extension.receive({ ...request, requestId: ++requestId });
    assert.equal(extension.messages.at(-1).type, "fusion-error");
    assert.equal(latestGraph(), pendingGraph);
    assert.equal(latestGraph().file.pendingRewritings.length, 1);
  }
  await extension.receive({
    type: "preview-fusion",
    pattern: "TransposeTranspose",
    requestId: ++requestId,
  });
  assert.equal(extension.messages.at(-1).type, "fusion-preview");
  assert.equal(extension.messages.at(-1).replayed, true);
  assert.deepEqual(extension.messages.at(-1).matches, []);
  assert.equal(latestGraph().file.fusionCount, 2);
  assert.deepEqual(latestGraph().file.pendingRewritings, []);
  const beforeUnfuse = extension.messages.filter(
    (message) => message.type === "graph",
  ).length;
  await extension.receive({ type: "undo-fusion", all: true });
  assert.equal(extension.messages.at(-1).type, "graph");
  assert.equal(latestGraph().file.fusionCount, 0);
  assert.deepEqual(latestGraph().file.fusionHistory, []);
  assert.equal(latestGraph().file.pendingRewritings.length, 2);
  assert.deepEqual(latestGraph().graph.nodes, original);
  assert.equal(
    extension.messages.filter((message) => message.type === "graph").length,
    beforeUnfuse + 1,
  );
});

test("does not reopen fusion controls after cancellation during the updated graph delivery", async () => {
  const extension = await openExtension("fusion.onnx");
  await extension.receive({
    type: "preview-fusion",
    pattern: "TransposeTranspose",
    requestId: 1,
  });
  const preview = extension.messages.at(-1);
  assert.equal(preview.type, "fusion-preview");
  extension.connect((message) => {
    if (message.type === "graph")
      void extension.receive({ type: "cancel-fusion" });
  });
  await extension.receive({
    type: "apply-fusion",
    previewId: preview.previewId,
    candidates: [0],
    requestId: 2,
  });
  assert.equal(
    extension.messages.findLast((message) => message.type === "graph").file
      .fusionCount,
    1,
  );
  assert.equal(
    extension.messages.some((message) => message.type === "fusion-info"),
    false,
  );
  await extension.receive({ type: "undo-fusion", all: true });
  const previews = extension.messages.filter(
    ({ type }) => type === "fusion-preview",
  ).length;
  await extension.receive({
    type: "preview-fusion",
    pattern: "TransposeTranspose",
    requestId: 3,
  });
  assert.equal(
    extension.messages.filter(({ type }) => type === "fusion-preview").length,
    previews,
  );
  assert.equal(
    extension.messages.findLast(({ type }) => type === "graph").file
      .fusionCount,
    1,
  );
  assert.deepEqual(
    extension.messages.findLast(({ type }) => type === "graph").file
      .pendingRewritings,
    [],
  );
});

test("previews and applies selected native fusions, rejects stale previews and saves edited copies", async (t) => {
  const directory = await mkdtemp(
    path.join(tmpdir(), "onnx-light-fusion-host-"),
  );
  t.after(() => rm(directory, { recursive: true, force: true }));
  const source = path.join(directory, "source.onnx");
  await copyFile(require.resolve("./models/fusion.onnx"), source);
  const extension = await openExtension(
    undefined,
    "onnxLightViewer.open",
    true,
  );
  await extension.openFile(source);
  const original = await readFile(source);
  let edited;
  const native = extension.nativeModels.at(-1);
  const previewFusion = native.previewFusion;
  native.previewFusion = async (...args) => {
    const preview = await previewFusion(...args);
    const applyFusion = preview.model.applyFusion;
    preview.model.applyFusion = async (...selected) =>
      (edited = await applyFusion(...selected));
    return preview;
  };
  await extension.receive({ type: "fusion-patterns", requestId: 1 });
  assert.ok(extension.messages.at(-1).patterns.includes("TransposeTranspose"));
  await extension.receive({
    type: "preview-fusion",
    pattern: "TransposeTranspose",
    requestId: 2,
  });
  const preview = extension.messages.at(-1);
  assert.equal(preview.type, "fusion-preview");
  assert.equal(preview.matches.length, 2);
  assert.equal(preview.model, undefined);
  await extension.receive({
    type: "apply-fusion",
    previewId: preview.previewId,
    anchors: [0],
    requestId: 3,
  });
  assert.equal(extension.messages.at(-1).type, "fusion-info");
  assert.deepEqual(extension.messages.at(-1).patterns, edited.fusionPatterns());
  const graph = extension.messages.findLast(
    (message) => message.type === "graph",
  ).graph;
  assert.deepEqual(
    graph.nodes.map((node) => node.opType),
    ["Identity", "Transpose", "Transpose", "MatMul"],
  );
  const destination = path.join(directory, "edited.onnx");
  await extension.save(destination);
  const { loadModelFile } = require("../dist/graph");
  assert.deepEqual(
    (await loadModelFile(destination, "edited"))
      .graph()
      .nodes.map((node) => node.opType),
    graph.nodes.map((node) => node.opType),
  );
  const historyFile = path.join(directory, "history.onnx");
  await extension.receive({
    type: "save",
    location: historyFile,
    overwrite: false,
    storage: "embedded",
    encrypted: false,
    includeRewritings: true,
  });
  assert.equal(extension.messages.at(-1).type, "save-info");
  assert.equal(
    (await loadModelFile(historyFile, "history")).graph().nodes.length,
    5,
  );
  await extension.receive({
    type: "apply-fusion",
    previewId: preview.previewId,
    anchors: [2],
    requestId: 4,
  });
  assert.equal(extension.messages.at(-1).type, "fusion-error");
  assert.deepEqual(await readFile(source), original);
  edited.metadata = () => {
    throw new Error("Native metadata failure after editing");
  };
  await extension.receive({ type: "metadata" });
  assert.equal(extension.messages.at(-1).type, "metadata-error");
  await extension.receive({
    type: "select-fusion",
    operation: 0,
    match: 0,
    pattern: "TransposeTranspose",
  });
  assert.equal(extension.messages.at(-1).type, "graph");
  assert.deepEqual(extension.messages.at(-1).selectedNodes, [0]);
  assert.equal(extension.messages.at(-1).file.fusionCount, 1);
  await extension.save(destination);
  assert.equal(extension.messages.at(-1).type, "save-info");
  assert.deepEqual(
    (await loadModelFile(destination, "edited"))
      .graph()
      .nodes.map((node) => node.opType),
    graph.nodes.map((node) => node.opType),
  );
  await extension.receive({ type: "undo-fusion" });
  assert.equal(extension.messages.at(-1).type, "graph");
  assert.equal(extension.messages.at(-1).graph.nodes.length, 5);
  assert.equal(extension.messages.at(-1).file.fusionCount, 0);
  assert.deepEqual(extension.messages.at(-1).file.fusionHistory, []);
  await extension.receive({ type: "undo-fusion" });
  assert.equal(extension.messages.at(-1).type, "undo-fusion-error");
  await extension.receive({
    type: "preview-fusion",
    pattern: "TransposeTranspose",
    requestId: 5,
  });
  const discarded = extension.messages.at(-1);
  await extension.receive({ type: "cancel-fusion" });
  await extension.receive({
    type: "apply-fusion",
    previewId: discarded.previewId,
    anchors: [1],
    requestId: 6,
  });
  assert.equal(extension.messages.at(-1).type, "fusion-error");
  const pending = extension.receive({
    type: "preview-fusion",
    pattern: "TransposeTranspose",
    requestId: 7,
  });
  await extension.openFile(require.resolve("./models/relu.onnx"));
  await pending;
  assert.notEqual(extension.messages.at(-1).requestId, 7);
});

test("rejects invalid candidate selections and stale multi-pattern matching results", async () => {
  const extension = await openExtension("fusion-scaled.onnx");
  const request = {
    type: "preview-fusion",
    patterns: ["TransposeTranspose", "MatMulScaleFusion"],
    requestId: 1,
  };
  await extension.receive(request);
  const preview = extension.messages.at(-1);
  assert.equal(preview.type, "fusion-preview");
  assert.deepEqual(
    preview.matches.map(({ pattern, anchor }) => [pattern, anchor]),
    [
      ["TransposeTranspose", 0],
      ["TransposeTranspose", 2],
      ["MatMulScaleFusion", 4],
    ],
  );
  for (const candidates of [[], [0, 0], [3], [-1]]) {
    await extension.receive({
      type: "apply-fusion",
      previewId: preview.previewId,
      candidates,
      requestId: 2,
    });
    assert.equal(extension.messages.at(-1).type, "fusion-error");
  }
  await extension.receive({ ...request, requestId: 3 });
  assert.notEqual(extension.messages.at(-1).previewId, preview.previewId);
  await extension.receive({
    type: "apply-fusion",
    previewId: preview.previewId,
    candidates: [0, 2],
    requestId: 4,
  });
  assert.equal(extension.messages.at(-1).type, "fusion-error");
});

test("saves reused external weights only beside their original source", async (t) => {
  const directory = await mkdtemp(
    path.join(tmpdir(), "onnx-light-reuse-host-"),
  );
  t.after(() => rm(directory, { recursive: true, force: true }));
  const { loadModel, loadModelFile } = require("../dist/graph");
  const native = loadModel(
    readFileSync(require.resolve("./models/fusion.onnx")),
    "fusion",
  );
  const files = await native.serializeFiles(1024, "original.data");
  const source = path.join(directory, "source.onnx");
  await writeFile(source, files.model);
  for (const weight of files.weights)
    await writeFile(path.join(directory, weight.name), weight.bytes);
  const extension = await openExtension(
    undefined,
    "onnxLightViewer.open",
    true,
  );
  await extension.openFile(source);
  await extension.receive({
    type: "preview-fusion",
    pattern: "TransposeTranspose",
    requestId: 1,
  });
  await extension.receive({
    type: "apply-fusion",
    previewId: extension.messages.at(-1).previewId,
    anchors: [0, 2],
    requestId: 2,
  });
  const destination = path.join(directory, "edited.onnx");
  const before = await readdir(directory);
  await extension.save(destination, false, [], 1024, true);
  assert.equal(extension.messages.at(-1).type, "save-info");
  assert.deepEqual(
    (await readdir(directory)).sort(),
    [...before, "edited.onnx"].sort(),
  );
  assert.deepEqual(
    (await loadModelFile(destination, "edited")).externalDataLocations(),
    [files.weights[0].name],
  );
  const other = path.join(directory, "other");
  await mkdir(other);
  await extension.save(path.join(other, "edited.onnx"), false, [], 1024, true);
  assert.equal(extension.messages.at(-1).type, "save-error");
  assert.match(extension.errors.at(-1), /source folder/);
  assert.deepEqual(await readdir(other), []);
});

test("saves inferred ModelProto copies as plain or encrypted files and opens encrypted copies", async (t) => {
  const directory = await mkdtemp(
    path.join(tmpdir(), "onnx-light-viewer-save-"),
  );
  t.after(() => rm(directory, { recursive: true, force: true }));
  const { loadModel } = require("../dist/graph");
  for (const localFiles of [false, true]) {
    const extension = await openExtension(
      "inference.onnx",
      undefined,
      localFiles,
    );
    await extension.receive({ type: "ready" });
    await extension.receive({ type: "refresh-shapes" });
    const expected = extension.nativeModels.at(-1).graph();
    const plain = path.join(directory, `plain-${localFiles}.onnx`);
    const encrypted = `${plain}.enc`;
    const password = "private export passphrase";
    await extension.save(plain);
    assert.equal(extension.messages.at(-1).type, "save-info");
    assert.deepEqual(
      loadModel(await readFile(plain), expected.name).graph(),
      expected,
    );
    await extension.save(encrypted, true, [password, password]);
    assert.equal(
      (await readFile(encrypted)).subarray(0, 8).toString(),
      "ONNXCRY3",
    );
    extension.setPasswords(password);
    await extension.openFile(encrypted);
    assert.equal(extension.messages.at(-1).type, "graph");
    await extension.save(plain);
    assert.deepEqual(
      loadModel(await readFile(plain), expected.name).graph(),
      expected,
    );
    assert.equal(extension.passwordPrompts.length, 3);
    assert.equal(
      extension.passwordPrompts.every(({ password }) => password === true),
      true,
    );
    assert.equal(JSON.stringify(extension.events).includes(password), false);
    assert.equal(extension.logs.join("\n").includes(password), false);
    assert.deepEqual(extension.errors, []);
  }
  assert.equal(
    (await readdir(directory)).some((name) => name.endsWith(".tmp")),
    false,
  );
});

test("keeps malformed decrypted plaintext out of errors and webview logs", async (t) => {
  const directory = await mkdtemp(
    path.join(tmpdir(), "onnx-light-viewer-private-logs-"),
  );
  t.after(() => rm(directory, { recursive: true, force: true }));
  const marker = "LEAKME42";
  const graph = Buffer.concat([Buffer.from([0xa7, 0x06]), Buffer.from(marker)]);
  const plaintext = Buffer.concat([
    Buffer.from([0x08, 0x08, 0x3a, graph.length]),
    graph,
  ]);
  const password = "malformed input password";
  const salt = Buffer.alloc(16, 17);
  const nonce = Buffer.alloc(12, 23);
  const header = Buffer.concat([Buffer.from("ONNXCRY3"), salt, nonce]);
  const cipher = createCipheriv(
    "aes-256-gcm",
    pbkdf2Sync(password, salt, 100000, 32, "sha256"),
    nonce,
  );
  cipher.setAAD(header);
  const payload = Buffer.concat([cipher.update(plaintext), cipher.final()]);
  const source = path.join(directory, "invalid.onnx.enc");
  await writeFile(
    source,
    Buffer.concat([header, cipher.getAuthTag(), payload]),
  );
  for (const local of [false, true]) {
    const extension = await openExtension("relu.onnx", undefined, local);
    extension.setPasswords(password);
    await extension.openFile(source);
    assert.equal(extension.messages.at(-1).type, "load-error");
    assert.match(extension.errors.at(-1), /decode the decrypted ONNX model/);
    for (const output of [
      extension.errors,
      extension.logs,
      extension.messages,
      extension.events,
    ])
      assert.equal(JSON.stringify(output).includes(marker), false);
  }
});

test("cancels password and save dialogs and keeps existing files intact after failed writes", async (t) => {
  const directory = await mkdtemp(
    path.join(tmpdir(), "onnx-light-viewer-save-errors-"),
  );
  t.after(() => rm(directory, { recursive: true, force: true }));
  const extension = await openExtension("relu.onnx", undefined, true);
  const destination = path.join(directory, "existing.onnx");
  const original = Buffer.from("existing destination");
  await writeFile(destination, original);
  await extension.save(undefined);
  await extension.save(destination, true);
  await extension.save(destination, true, ["password", "different"]);
  assert.equal(extension.writes.length, 0);
  for (const fail of [
    extension.failNextSaveWrite,
    extension.failNextSaveRename,
  ]) {
    fail();
    await extension.save(destination);
    assert.equal(extension.messages.at(-1).type, "save-error");
    assert.deepEqual(await readFile(destination), original);
    assert.deepEqual(await readdir(directory), ["existing.onnx"]);
  }
  const source = path.join(directory, "source.onnx");
  await copyFile(require.resolve("./models/relu.onnx"), source);
  await extension.openFile(source);
  const writes = extension.writes.length;
  await extension.save(source);
  assert.equal(extension.writes.length, writes);
  assert.match(extension.errors.at(-1), /different destination/);
  const encrypted = path.join(directory, "encrypted.onnx.enc");
  const { loadModel } = require("../dist/graph");
  await writeFile(
    encrypted,
    await loadModel(
      readFileSync(require.resolve("./models/relu.onnx")),
      "relu",
    ).serialize("password"),
  );
  const errors = extension.errors.length;
  extension.setPasswords(undefined);
  await extension.openFile(encrypted);
  assert.equal(extension.errors.length, errors);
  assert.match(extension.logs.join("\n"), /\[CANCELLED\] Open ONNX model/);
  extension.setPasswords("wrong");
  await extension.openFile(encrypted);
  assert.equal(extension.messages.at(-1).type, "load-error");
  await extension.receive({ type: "ready" });
  assert.equal(extension.messages.at(-1).graph.nodes[0].opType, "Relu");
});

test("saves real external weights, embeds them into encrypted copies and rolls back failed multi-file exports", async (t) => {
  const directory = await mkdtemp(
    path.join(tmpdir(), "onnx-light-viewer-external-save-"),
  );
  t.after(() => rm(directory, { recursive: true, force: true }));
  const { loadModel, loadModelFile } = require("../dist/graph");
  const source = path.join(directory, "source.onnx");
  await copyFile(require.resolve("../media/tiny-llm.onnx"), source);
  const original = await readFile(source);
  const extension = await openExtension(
    undefined,
    "onnxLightViewer.open",
    true,
  );
  await extension.openFile(source);
  const expectedGraph = extension.nativeModels.at(-1).graph();
  const copy = path.join(directory, "copy.onnx");
  await extension.save(copy, false, [], 1024 * 1024);
  assert.equal(extension.messages.at(-1).type, "save-info");
  const weights = (await readdir(directory)).filter((name) =>
    name.startsWith("copy.onnx."),
  );
  assert.ok(weights.length > 0);
  for (const name of weights)
    assert.ok(
      (await readFile(path.join(directory, name))).byteLength <= 1024 * 1024,
    );
  const expected = await (await loadModelFile(copy, "copy")).serialize();
  assert.deepEqual(loadModel(expected, "copy").graph(), expectedGraph);
  await extension.openFile(copy);
  const encrypted = path.join(directory, "embedded.onnx.enc");
  await extension.save(encrypted, true, [
    "external password",
    "external password",
  ]);
  assert.equal(extension.messages.at(-1).type, "save-info");
  assert.deepEqual(
    await (
      await loadModelFile(encrypted, "copy", "external password")
    ).serialize(),
    expected,
  );
  const existing = Buffer.from("preserve existing destination");
  const failed = path.join(directory, "failed.onnx");
  await writeFile(failed, existing);
  const before = (await readdir(directory)).sort();
  for (const fail of [
    extension.failNextSaveWrite,
    () => extension.failNextSaveRename(),
    () => extension.failNextSaveRename(weights.length + 1),
  ]) {
    fail();
    await extension.save(failed, false, [], 1024 * 1024);
    assert.equal(extension.messages.at(-1).type, "save-error");
    assert.deepEqual(await readFile(failed), existing);
    assert.deepEqual((await readdir(directory)).sort(), before);
  }
  const lateAlias = path.join(directory, "late.onnx");
  const originalWeight = await readFile(path.join(directory, weights[0]));
  const gate = extension.pauseNextSaveWrite(weights.length + 1);
  const saving = extension.save(lateAlias, false, [], 1024 * 1024);
  await gate.written;
  await link(path.join(directory, weights[0]), lateAlias);
  gate.release();
  await saving;
  assert.equal(extension.messages.at(-1).type, "save-error");
  assert.match(extension.errors.at(-1), /different destination/);
  assert.deepEqual(await readFile(lateAlias), originalWeight);
  assert.deepEqual(
    await readFile(path.join(directory, weights[0])),
    originalWeight,
  );
  await rm(lateAlias);
  assert.deepEqual((await readdir(directory)).sort(), before);
  assert.deepEqual(await readFile(source), original);
  for (const name of weights) await rm(path.join(directory, name));
  const writeCount = extension.writes.length;
  await extension.save(path.join(directory, "missing-weights.onnx"));
  assert.equal(extension.messages.at(-1).type, "save-error");
  assert.equal(extension.writes.length, writeCount);
});

test("preserves dense and sparse external sources when selected as save destinations", async (t) => {
  const directory = await mkdtemp(
    path.join(tmpdir(), "onnx-light-viewer-protect-weights-"),
  );
  t.after(() => rm(directory, { recursive: true, force: true }));
  const source = path.join(directory, "source.onnx");
  await copyFile(require.resolve("./models/initializers.onnx"), source);
  const weightsDirectory = path.join(directory, "weights");
  await mkdir(weightsDirectory);
  const files = [
    [source, await readFile(source)],
    [path.join(weightsDirectory, "decoder.bin"), Buffer.from([0, 0, 128, 63])],
    [path.join(weightsDirectory, "values.bin"), Buffer.from([0, 0, 0, 64])],
    [path.join(weightsDirectory, "indices.bin"), Buffer.alloc(8)],
  ];
  for (const [file, bytes] of files.slice(1)) await writeFile(file, bytes);
  const extension = await openExtension(undefined, undefined, true);
  await extension.openFile(source);
  assert.equal(extension.messages.at(-1).type, "graph");
  const alias = path.join(directory, "alias");
  await symlink(weightsDirectory, alias, "junction");
  for (const destination of [
    ...files.slice(1).map(([file]) => file),
    path.join(alias, "decoder.bin"),
  ]) {
    for (const mode of ["plain", "encrypted", "external"]) {
      await extension.save(
        destination,
        mode === "encrypted",
        ["password", "password"],
        mode === "external" ? 1024 * 1024 : undefined,
      );
      assert.equal(extension.messages.at(-1).type, "save-error");
      assert.match(extension.errors.at(-1), /different destination/);
      assert.deepEqual(extension.writes, []);
      for (const [file, bytes] of files)
        assert.deepEqual(await readFile(file), bytes);
    }
  }
});

test("rejects invalid external file sizes and the unavailable encrypted external mode before writing", async () => {
  const extension = await openExtension();
  for (const size of [
    0,
    -1,
    1.5,
    Infinity,
    Number.MAX_SAFE_INTEGER + 1,
    "1024",
  ]) {
    await extension.receive({
      type: "save",
      storage: "external",
      encrypted: false,
      maxExternalFileSize: size,
    });
    assert.equal(extension.messages.at(-1).type, "save-error");
    assert.match(extension.errors.at(-1), /positive.*integer/);
  }
  await extension.receive({
    type: "save",
    storage: "external",
    encrypted: true,
    maxExternalFileSize: 1024,
  });
  assert.match(extension.errors.at(-1), /onnx-light\/issues\/5533/);
  assert.deepEqual(extension.writes, []);
  assert.deepEqual(extension.passwordPrompts, []);
});

test("preserves source files reached through filesystem aliases and rechecks before publication", async (t) => {
  const directory = await mkdtemp(
    path.join(tmpdir(), "onnx-light-viewer-save-alias-"),
  );
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sourceDirectory = path.join(directory, "models");
  await mkdir(sourceDirectory);
  const source = path.join(sourceDirectory, "Model.onnx");
  await copyFile(require.resolve("./models/relu.onnx"), source);
  const original = await readFile(source);
  const extension = await openExtension(
    undefined,
    "onnxLightViewer.open",
    true,
  );
  await extension.openFile(source);
  const alias = path.join(directory, "alias");
  await symlink(sourceDirectory, alias, "junction");
  const hardlink = path.join(directory, "hardlink.onnx");
  await link(source, hardlink);
  for (const destination of [path.join(alias, "Model.onnx"), hardlink]) {
    await extension.save(destination);
    assert.match(extension.errors.at(-1), /different destination/);
    assert.deepEqual(await readFile(source), original);
    assert.equal(extension.writes.length, 0);
  }
  const caseAlias = path.join(sourceDirectory, "model.onnx");
  let caseInsensitive = true;
  try {
    await stat(caseAlias);
  } catch (error) {
    if (error.code !== "ENOENT") throw error;
    caseInsensitive = false;
  }
  await extension.save(caseAlias);
  if (caseInsensitive) {
    assert.match(extension.errors.at(-1), /different destination/);
    assert.equal(extension.writes.length, 0);
  } else {
    assert.equal(extension.messages.at(-1).type, "save-info");
  }
  assert.deepEqual(await readFile(source), original);
  const lateAlias = path.join(directory, "late.onnx");
  const gate = extension.pauseNextSaveWrite();
  const saving = extension.save(lateAlias);
  await gate.written;
  await link(source, lateAlias);
  gate.release();
  await saving;
  assert.equal(extension.messages.at(-1).type, "save-error");
  assert.match(extension.errors.at(-1), /different destination/);
  assert.deepEqual(await readFile(lateAlias), original);
  assert.deepEqual(await readFile(source), original);
  assert.equal(
    (await readdir(directory)).some((name) => name.endsWith(".tmp")),
    false,
  );
});

test("cancels pending proto writes and rolls back external weights on replacement or disposal", async (t) => {
  const directory = await mkdtemp(
    path.join(tmpdir(), "onnx-light-viewer-save-write-"),
  );
  t.after(() => rm(directory, { recursive: true, force: true }));
  for (const external of [false, true]) {
    for (const dispose of [false, true]) {
      const destinationDirectory = path.join(
        directory,
        `${external}-${dispose}`,
      );
      await mkdir(destinationDirectory);
      const destination = path.join(destinationDirectory, "copy.onnx");
      const original = Buffer.from("existing destination");
      await writeFile(destination, original);
      const extension = await openExtension(
        undefined,
        "onnxLightViewer.openSymbolicExample",
        true,
      );
      const maxFileSize = 1024 * 1024;
      const weightCount = external
        ? (
            await extension.nativeModels
              .at(-1)
              .serializeFiles(maxFileSize, "count")
          ).weights.length
        : 0;
      if (external) assert.ok(weightCount > 0);
      const gate = extension.pauseNextSaveWrite(weightCount + 1);
      const saving = extension.save(
        destination,
        false,
        [],
        external ? maxFileSize : undefined,
      );
      await gate.written;
      if (dispose) extension.dispose();
      else await extension.open("relu.onnx");
      gate.release();
      await saving;
      assert.deepEqual(await readFile(destination), original);
      assert.deepEqual(await readdir(destinationDirectory), ["copy.onnx"]);
      assert.equal(
        extension.messages.some(({ type }) => type === "save-info"),
        false,
      );
      assert.deepEqual(extension.errors, []);
    }
  }
});

test("reuses a decrypted model's host-only password after native recovery", async (t) => {
  const directory = await mkdtemp(
    path.join(tmpdir(), "onnx-light-viewer-save-recovery-"),
  );
  t.after(() => rm(directory, { recursive: true, force: true }));
  const extension = await openExtension("invalid-graph.onnx", undefined, true);
  const encrypted = path.join(directory, "invalid.onnx.enc");
  const password = "recovery passphrase";
  await extension.save(encrypted, true, [password, password]);
  extension.setPasswords(password);
  await extension.openFile(encrypted);
  await extension.receive({ type: "metadata" });
  assert.equal(extension.messages.at(-1).type, "metadata-error");
  const prompts = extension.passwordPrompts.length;
  await extension.save(path.join(directory, "recovered.onnx"));
  assert.equal(extension.messages.at(-1).type, "save-info");
  assert.equal(extension.passwordPrompts.length, prompts);
  assert.equal(JSON.stringify(extension.events).includes(password), false);
});

test("does not write a model after replacement or panel disposal during the save dialog", async (t) => {
  const directory = await mkdtemp(
    path.join(tmpdir(), "onnx-light-viewer-stale-save-"),
  );
  t.after(() => rm(directory, { recursive: true, force: true }));
  for (const dispose of [false, true]) {
    const extension = await openExtension("inference.onnx", undefined, true);
    const gate = extension.pauseNextSaveDialog();
    const save = extension.save(path.join(directory, "copy.onnx"));
    await gate.opened;
    if (dispose) extension.dispose();
    else await extension.open("relu.onnx");
    gate.release();
    await save;
    assert.deepEqual(extension.writes, []);
    assert.deepEqual(await readdir(directory), []);
  }
});

test("streams local custom-editor files and replacements without reading whole files into the host", async () => {
  const extension = await openExtension("inference.onnx", undefined, true);
  assert.deepEqual(extension.reads, []);
  await extension.receive({ type: "ready" });
  assert.equal(extension.messages.at(-1).type, "graph");
  await extension.receive({ type: "refresh-shapes" });
  assert.deepEqual(
    extension.messages.at(-1).tensors.find(({ name }) => name === "reshaped")
      .shape,
    ["N", "3", "1"],
  );
  await extension.open("invalid-graph.onnx");
  await extension.receive({ type: "metadata" });
  const count = extension.nativeModels.length;
  await extension.receive({ type: "metadata" });
  assert.equal(extension.nativeModels.length, count + 1);
  await extension.open("initializers.onnx");
  assert.equal(extension.messages.at(-1).type, "graph");
  assert.deepEqual(extension.reads, []);
});

test("estimates root and pipeline memory on request and reports errors without invalidating the model", async (t) => {
  for (const name of [
    "inference.onnx",
    "nested-if.onnx",
    "functions.onnx",
    "tiny.pipe-onnx",
  ]) {
    const extension = await openExtension(name);
    t.after(() => extension.dispose());
    const native = extension.nativeModels.at(-1);
    if (name === "functions.onnx")
      await extension.receive({
        type: "navigate",
        path: native.graph().localFunctions[0].path,
      });
    if (name === "nested-if.onnx")
      await extension.receive({
        type: "navigate",
        path: native
          .graph()
          .nodes[0].attributes.find(({ name }) => name === "then_branch")
          .graphs[0].path,
      });
    await extension.receive({ type: "memory-inputs", requestId: 1 });
    const inputs = extension.messages.at(-1);
    assert.equal(inputs.type, "memory-inputs");
    assert.equal(inputs.stages.length, name.endsWith("pipe-onnx") ? 3 : 0);
    const dimensions = Object.fromEntries(
      inputs.dimensions.map(({ key, defaultValue }) => [key, defaultValue]),
    );
    await extension.receive({
      type: "memory-profile",
      requestId: 2,
      dimensions,
      path: inputs.path,
    });
    const profile = extension.messages.at(-1);
    assert.equal(profile.type, "memory-profile");
    assert.ok(profile.profile.events.length > 0);
    assert.deepEqual(
      profile.profile.events.map(({ opType }) => opType),
      native.graph(false, inputs.path).nodes.map(({ opType }) => opType),
    );
    await extension.receive({
      type: "memory-profile",
      requestId: 3,
      dimensions: { extra: -1 },
    });
    assert.equal(extension.messages.at(-1).type, "memory-error");
    await extension.receive({ type: "memory-inputs", requestId: 4 });
    assert.equal(extension.messages.at(-1).type, "memory-inputs");
    if (name.endsWith("pipe-onnx")) {
      await extension.receive({
        type: "memory-inputs",
        requestId: 5,
        path: [{ pipeline: "embedding" }, { function: 0 }],
      });
      const error = extension.messages.at(-1);
      assert.equal(error.type, "memory-error");
      assert.equal(error.stages.length, 3);
      await extension.receive({
        type: "memory-inputs",
        requestId: 6,
        path: error.stages[1].path,
      });
      assert.equal(extension.messages.at(-1).type, "memory-inputs");
    }
    assert.equal(extension.nativeModels.length, 1);
  }
});

test("tensor statistics requests validate transport and keep native handles usable", async (t) => {
  const extension = await openExtension("tiny-initializers.onnx");
  t.after(() => extension.dispose());
  const request = {
    type: "tensor-statistics",
    requestId: 1,
    name: "typed_floats",
    path: [],
  };
  await extension.receive(request);
  assert.equal(extension.messages.at(-1).type, "tensor-statistics-info");
  assert.equal(extension.messages.at(-1).statistics.values, "[0.5, -2, 3]");
  assert.equal(extension.messages.at(-1).statistics.mean, undefined);
  for (const invalid of [
    { requestId: "1" },
    { requestId: 0 },
    { name: 7 },
    { name: "" },
    { path: "root" },
    { path: [{ node: -1, attribute: 0, graph: 0 }] },
  ]) {
    await extension.receive({ ...request, ...invalid });
    assert.equal(extension.messages.at(-1).type, "tensor-statistics-error");
    assert.match(
      extension.messages.at(-1).error,
      /Invalid tensor statistics request/,
    );
  }
  await extension.receive({ ...request, name: "absent", requestId: 2 });
  assert.equal(extension.messages.at(-1).type, "tensor-statistics-error");
  assert.match(extension.messages.at(-1).error, /No stored tensor data/);
  await extension.receive({ ...request, requestId: 3 });
  assert.equal(extension.messages.at(-1).type, "tensor-statistics-info");
  assert.equal(extension.nativeModels.length, 1);
});

test("tensor statistics discard delayed computations after navigation, replacement or disposal", async () => {
  for (const action of ["navigate", "replace", "dispose", "new-request"]) {
    for (const fail of [false, true]) {
      const extension = await openExtension("tiny-initializers.onnx");
      try {
        const native = extension.nativeModels.at(-1);
        const original = native.tensorStatistics;
        let resume;
        const gate = new Promise((resolve) => {
          resume = resolve;
        });
        native.tensorStatistics = async (...args) => {
          await gate;
          if (fail) throw new Error("Delayed tensor failure");
          return original(...args);
        };
        const pending = extension.receive({
          type: "tensor-statistics",
          requestId: 1,
          name: "typed_floats",
          path: [],
        });
        await new Promise((resolve) => setImmediate(resolve));
        if (action === "replace") await extension.open("relu.onnx");
        else if (action === "navigate")
          await extension.receive({ type: "navigate", path: [] });
        else if (action === "dispose") extension.dispose();
        else {
          native.tensorStatistics = original;
          await extension.receive({
            type: "tensor-statistics",
            requestId: 2,
            name: "scalar",
            path: [],
          });
        }
        const count = extension.messages.length;
        resume();
        await pending;
        assert.equal(extension.messages.length, count, `${action}: ${fail}`);
      } finally {
        extension.dispose();
      }
    }
  }
});

test("statistics list external dense and sparse files with current physical sizes", async (t) => {
  const directory = await mkdtemp(
    path.join(tmpdir(), "onnx-light-statistics-files-"),
  );
  t.after(() => rm(directory, { recursive: true, force: true }));
  const source = path.join(directory, "source.onnx");
  await copyFile(require.resolve("./models/initializers.onnx"), source);
  await mkdir(path.join(directory, "weights"));
  for (const local of [true, false]) {
    const extension = await openExtension(undefined, undefined, local);
    t.after(() => extension.dispose());
    await extension.openFile(source);
    const reads = [...extension.reads];
    const weights = [
      ["weights/decoder.bin", 4096],
      ["weights/indices.bin", 8],
      ["weights/values.bin", 1536],
    ];
    for (const [name, size] of weights)
      await writeFile(path.join(directory, name), Buffer.alloc(size));
    await extension.receive({ type: "get-statistics", requestId: 1 });
    assert.deepEqual(extension.messages.at(-1).files, [
      { name: "source.onnx", external: false, size: (await stat(source)).size },
      ...weights.map(([name, size]) => ({ name, external: true, size })),
    ]);
    await writeFile(
      path.join(directory, "weights/values.bin"),
      Buffer.alloc(3072),
    );
    await rm(path.join(directory, "weights/indices.bin"));
    await extension.receive({ type: "get-statistics", requestId: 2 });
    const files = extension.messages.at(-1).files;
    assert.equal(
      files.find(({ name }) => name === "weights/values.bin").size,
      3072,
    );
    const missing = files.find(({ name }) => name === "weights/indices.bin");
    assert.equal(missing.size, undefined);
    assert.match(missing.error, /ENOENT|not found/i);
    assert.deepEqual(extension.errors, []);
    assert.deepEqual(extension.reads, reads);
  }
});

test("statistics rediscover external file references after native recovery", async (t) => {
  const directory = await mkdtemp(
    path.join(tmpdir(), "onnx-light-statistics-recovery-"),
  );
  t.after(() => rm(directory, { recursive: true, force: true }));
  const source = path.join(directory, "source.onnx");
  await copyFile(require.resolve("./models/invalid-graph.onnx"), source);
  const extension = await openExtension(undefined, undefined, true);
  t.after(() => extension.dispose());
  await extension.openFile(source);
  await extension.receive({ type: "metadata" });
  assert.equal(extension.messages.at(-1).type, "metadata-error");
  await copyFile(require.resolve("./models/initializers.onnx"), source);
  await mkdir(path.join(directory, "weights"));
  await writeFile(
    path.join(directory, "weights/decoder.bin"),
    Buffer.alloc(2048),
  );
  await extension.receive({ type: "get-statistics", requestId: 1 });
  const result = extension.messages.at(-1);
  assert.equal(result.type, "statistics-info");
  assert.deepEqual(
    result.files.map(({ name }) => name),
    [
      "source.onnx",
      "weights/decoder.bin",
      "weights/indices.bin",
      "weights/values.bin",
    ],
  );
  assert.equal(result.files[1].size, 2048);
  assert.match(result.files[2].error, /ENOENT|not found/i);
});

test("statistics recover complete model totals on demand after nested refresh failures", async (t) => {
  for (const name of ["nested-if.onnx", "functions.onnx", "tiny.pipe-onnx"]) {
    for (const type of ["refresh-shapes", "metadata"]) {
      const extension = await openExtension(name, undefined, true);
      t.after(() => extension.dispose());
      await extension.receive({ type: "ready" });
      const root = extension.messages.at(-1).graph;
      const native = extension.nativeModels.at(-1);
      const expected = root.pipeline ? root : native.summary();
      const graph = native.graph;
      const summary = native.summary;
      let statisticsCalls = 0;
      native.summary = (...args) => {
        statisticsCalls++;
        return summary(...args);
      };
      native.graph = (infer, path = []) => {
        if (infer && type === "refresh-shapes")
          throw new Error("Nested inference failed.");
        if (root.pipeline && !path.length) statisticsCalls++;
        return graph(infer, path);
      };
      if (type === "metadata")
        native.metadata = () => {
          throw new Error("Nested metadata failed.");
        };
      const path = root.pipeline
        ? root.pipelineStages[0].path
        : name === "functions.onnx"
          ? root.localFunctions[0].path
          : root.nodes[0].attributes.find(({ name }) => name === "then_branch")
              .graphs[0].path;
      await extension.receive({ type: "navigate", path });
      assert.equal(extension.messages.at(-1).statistics, undefined);
      assert.equal(statisticsCalls, 0);
      await extension.receive({ type });
      assert.equal(
        extension.messages.at(-1).type,
        type === "metadata" ? "metadata-error" : "shape-error",
      );
      const loaded = extension.nativeModels.length;
      await extension.receive({ type: "ready" });
      assert.equal(extension.messages.at(-1).statistics, undefined);
      assert.equal(extension.nativeModels.length, loaded);
      await extension.receive({ type: "get-statistics", requestId: 1 });
      const result = extension.messages.at(-1);
      assert.equal(result.type, "statistics-info");
      assert.equal(result.requestId, 1);
      assert.deepEqual(result.statistics, expected);
      assert.equal(extension.nativeModels.length, loaded + 1);
      assert.ok(result.files.some((file) => file.name === name));
    }
  }
});

test("statistics failures are explicit and can be retried without using nested counts", async (t) => {
  const extension = await openExtension("functions.onnx");
  t.after(() => extension.dispose());
  const native = extension.nativeModels.at(-1);
  const root = native.graph();
  await extension.receive({
    type: "navigate",
    path: root.localFunctions[0].path,
  });
  const summary = native.summary;
  native.summary = () => {
    throw new Error("Root statistics failed.");
  };
  await extension.receive({ type: "get-statistics", requestId: 1 });
  assert.deepEqual(extension.messages.at(-1), {
    type: "statistics-error",
    requestId: 1,
    error: "Root statistics failed.",
  });
  assert.match(extension.errors.at(-1), /Root statistics failed/);
  native.summary = summary;
  await extension.receive({ type: "get-statistics", requestId: 2 });
  assert.equal(extension.messages.at(-1).type, "statistics-info");
  assert.equal(
    extension.messages.at(-1).statistics.nodeCount,
    root.nodes.length,
  );
});

test("ignores stale recovery results and errors after replacing a local model", async () => {
  for (const type of [
    "navigate",
    "get-statistics",
    "memory-inputs",
    "memory-profile",
    "tensor-statistics",
    "metadata",
    "refresh-shapes",
    "verify",
    "undo-fusion",
    "apply-rewriting",
    "explain-fusion",
  ]) {
    const extension = await openExtension("inference.onnx", undefined, true);
    for (const path of [[], [{ node: 9999, attribute: 0, graph: 0 }]]) {
      await extension.open("invalid-graph.onnx");
      await extension.receive({ type: "metadata" });
      const resume = extension.pauseNextFileLoad();
      const recovery = extension.receive({
        type,
        requestId: 1,
        name: "weight",
        dimensions: {},
        path,
        scenario: "scenario",
        all: false,
        pattern: "TransposeTranspose",
        anchor: 0,
      });
      await extension.open("relu.onnx");
      const count = extension.messages.length;
      const errors = extension.errors.length;
      const nativeCount = extension.nativeModels.length;
      resume();
      await recovery;
      assert.equal(extension.messages.length, count);
      assert.equal(extension.errors.length, errors);
      await extension.receive({ type: "ready" });
      assert.equal(extension.messages.at(-1).graph.nodes[0].opType, "Relu");
      await extension.receive({ type: "refresh-shapes" });
      assert.equal(extension.nativeModels.length, nativeCount + 1);
    }
  }
});

test("navigation selects a requested node index and rejects invalid indices", async (t) => {
  const extension = await openExtension("relu.onnx", undefined, true);
  t.after(() => extension.dispose());
  await extension.receive({ type: "navigate", path: [], nodeIndex: 0 });
  const graph = extension.messages.at(-1);
  assert.equal(graph.type, "graph");
  assert.deepEqual(graph.selectedNodes, [0]);
  for (const nodeIndex of [-1, 0.5, 9999, "0", null]) {
    await extension.receive({ type: "navigate", path: [], nodeIndex });
    assert.equal(extension.messages.at(-1).type, "navigation-error");
    assert.match(
      extension.messages.at(-1).error,
      /Invalid navigation node index/,
    );
  }
  await extension.receive({ type: "navigate", path: [] });
  assert.equal(extension.messages.at(-1).type, "graph");
  assert.equal(extension.messages.at(-1).selectedNodes, undefined);
});

test("shares concurrent native recovery and keeps inferred state for navigation", async () => {
  const extension = await openExtension("inference.onnx", undefined, true);
  const initial = extension.nativeModels.at(-1);
  const graph = initial.graph;
  let failInference = true;
  initial.graph = (infer, ...args) => {
    if (infer && failInference) {
      failInference = false;
      throw new Error("temporary inference failure");
    }
    return graph(infer, ...args);
  };
  await extension.receive({ type: "refresh-shapes" });
  assert.equal(extension.messages.at(-1).type, "shape-error");

  const count = extension.nativeModels.length;
  const resumeShape = extension.pauseNextFileLoad();
  const shapes = extension.receive({ type: "refresh-shapes" });
  const resumeMetadata = extension.pauseNextFileLoad();
  const metadata = extension.receive({ type: "metadata" });
  resumeShape();
  await shapes;
  resumeMetadata();
  await metadata;
  assert.equal(extension.nativeModels.length, count + 1);
  for (const type of ["shape-info", "metadata-info"]) {
    assert.equal(
      extension.messages.filter((message) => message.type === type).length,
      1,
    );
  }
  await extension.receive({ type: "navigate", path: [] });
  assert.deepEqual(
    extension.messages
      .at(-1)
      .graph.tensors.find(({ name }) => name === "reshaped").shape,
    ["N", "3", "1"],
  );
});

test("ignores delayed recovery results and errors after disposing the panel", async () => {
  for (const message of [
    { type: "navigate", path: [] },
    { type: "get-statistics", requestId: 1 },
    { type: "memory-inputs", requestId: 1 },
    { type: "memory-profile", requestId: 1, dimensions: {} },
    { type: "tensor-statistics", requestId: 1, name: "weight", path: [] },
    { type: "navigate", path: [{ node: 9999, attribute: 0, graph: 0 }] },
    { type: "metadata" },
    { type: "refresh-shapes" },
    { type: "verify" },
    { type: "undo-fusion" },
    { type: "apply-rewriting", scenario: "scenario", all: false },
    { type: "explain-fusion", pattern: "TransposeTranspose", anchor: 0 },
  ]) {
    const extension = await openExtension(
      "invalid-graph.onnx",
      undefined,
      true,
    );
    await extension.receive({ type: "metadata" });
    const resume = extension.pauseNextFileLoad();
    const recovery = extension.receive(message);
    extension.dispose();
    const events = extension.events.length;
    const errors = extension.errors.length;
    resume();
    await recovery;
    assert.equal(extension.events.length, events);
    assert.equal(extension.errors.length, errors);
  }
});

test("discards analysis from the previous graph when navigation shares delayed recovery", async () => {
  for (const type of ["refresh-shapes", "metadata", "verify"]) {
    const extension = await openExtension("nested-if.onnx", undefined, true);
    const native = extension.nativeModels.at(-1);
    const graph = native.graph;
    const root = graph(false);
    const path = root.nodes[0].attributes.find(
      ({ name }) => name === "then_branch",
    ).graphs[0].path;
    native.graph = (infer, ...args) => {
      if (infer) throw new Error("temporary inference failure");
      return graph(infer, ...args);
    };
    await extension.receive({ type: "refresh-shapes" });
    assert.equal(extension.messages.at(-1).type, "shape-error");

    const resume = extension.pauseNextFileLoad();
    const navigation = extension.receive({ type: "navigate", path });
    const analysis = extension.receive({ type });
    resume();
    await Promise.all([navigation, analysis]);

    const child = extension.messages
      .filter(({ type }) => type === "graph")
      .at(-1).graph;
    assert.equal(child.breadcrumbs.length, 2);
    assert.equal(child.nodes.length, 2);
    assert.equal(
      extension.messages.some(({ type: messageType }) =>
        type === "refresh-shapes"
          ? messageType === "shape-info"
          : messageType ===
            (type === "verify" ? "verify-info" : "metadata-info"),
      ),
      false,
    );

    await extension.receive({ type });
    if (type === "refresh-shapes") {
      assert.equal(extension.messages.at(-1).type, "shape-info");
      assert.deepEqual(
        extension.messages.at(-1).tensors.find(({ name }) => name === "sum")
          .shape,
        ["N", "3"],
      );
    } else if (type === "verify") {
      assert.equal(extension.messages.at(-1).type, "verify-info");
    } else {
      assert.equal(extension.messages.at(-1).type, "metadata-info");
      assert.equal(extension.messages.at(-1).metadata.nodes.length, 2);
    }
  }
});

for (const [type, buttonId, errorType, infoType] of [
  ["refresh-shapes", "refresh-shapes", "shape-error", "shape-info"],
  ["metadata", "refresh-metadata", "metadata-error", "metadata-info"],
]) {
  test(`settles ${type} when subsequent navigation shares a rejected reload`, async (t) => {
    const extension = await openExtension("nested-if.onnx", undefined, true);
    const native = extension.nativeModels.at(-1);
    const graph = native.graph;
    const root = graph(false);
    const childPath = root.nodes[0].attributes.find(
      ({ name }) => name === "then_branch",
    ).graphs[0].path;
    native.graph = (infer, ...args) => {
      if (infer) throw new Error("temporary inference failure");
      return graph(infer, ...args);
    };
    await extension.receive({ type: "refresh-shapes" });

    const dom = new JSDOM(extension.webview.html, {
      runScripts: "outside-only",
    });
    t.after(() => dom.window.close());
    const { window } = dom;
    window.structuredClone = structuredClone;
    const { document } = window;
    const requests = [];
    window.HTMLCanvasElement.prototype.getContext = () => ({
      font: "",
      measureText: (text) => ({ width: text.length * 8 }),
    });
    document.getElementById("canvas").getBoundingClientRect = () =>
      new window.DOMRect(0, 0, 1060, 350);
    window.acquireVsCodeApi = () => ({
      postMessage: (message) => requests.push(extension.receive(message)),
    });
    extension.connect((data) =>
      window.dispatchEvent(new window.MessageEvent("message", { data })),
    );
    for (const script of document.querySelectorAll("script[src]"))
      window.eval(readFileSync(new URL(script.src), "utf8"));
    const flush = async () => {
      while (requests.length) await Promise.all(requests.splice(0));
    };
    await flush();

    const start = extension.messages.length;
    const reloads = extension.logs.filter((line) =>
      line.includes("[RUNNING] Reload native model:"),
    ).length;
    const resume = extension.pauseNextFileLoad(new Error("reload failed"));
    const button = document.getElementById(buttonId);
    button.click();
    assert.equal(button.disabled, true);
    assert.equal(button.getAttribute("aria-busy"), "true");
    const navigation = extension.receive({
      type: "navigate",
      path: childPath,
    });
    assert.equal(extension.messages.length, start);
    resume();
    await Promise.all([flush(), navigation]);

    const responses = extension.messages.slice(start);
    assert.equal(responses.filter(({ type }) => type === errorType).length, 1);
    assert.equal(
      responses.filter(({ type }) => type === "navigation-error").length,
      1,
    );
    assert.ok(responses.every(({ error }) => error === "reload failed"));
    assert.equal(
      extension.logs.filter((line) =>
        line.includes("[RUNNING] Reload native model:"),
      ).length,
      reloads + 1,
    );
    assert.equal(button.disabled, false);
    assert.equal(button.getAttribute("aria-busy"), "false");
    await extension.receive({ type: "ready" });
    assert.deepEqual(extension.messages.at(-1).graph, root);

    button.click();
    assert.equal(button.disabled, true);
    await flush();
    assert.equal(extension.messages.at(-1).type, infoType);
    assert.equal(button.disabled, false);
    assert.equal(button.getAttribute("aria-busy"), "false");
  });
}

test("opens a cached Hugging Face model and preserves the display on invalid input or cancellation", async (t) => {
  const storage = await mkdtemp(path.join(tmpdir(), "onnx-viewer-hf-host-"));
  t.after(() => rm(storage, { recursive: true }));
  const url =
    "https://huggingface.co/viewer-tests/fixtures/resolve/main/inference.onnx";
  const key = createHash("sha256").update(url).digest("hex");
  const cache = path.join(storage, "huggingface", key);
  await mkdir(cache, { recursive: true });
  await copyFile(
    path.join(__dirname, "models/inference.onnx"),
    path.join(cache, "model.onnx"),
  );
  const extension = await openExtension(
    undefined,
    "onnxLightViewer.open",
    false,
    storage,
    undefined,
    new Map([
      [
        "applicationSettings",
        {
          downloadDirectory: path.join(storage, "huggingface"),
          pythonPath: process.platform === "win32" ? "python" : "python3",
        },
      ],
    ]),
  );
  await extension.openHuggingFace(url.replace("/resolve/", "/blob/"));
  assert.deepEqual(extension.preferences.get("pathHistory").open, [
    { source: "huggingface", kind: "file", path: url },
    {
      source: "huggingface",
      kind: "directory",
      path: "https://huggingface.co/viewer-tests/fixtures",
    },
  ]);
  assert.deepEqual(
    extension.picks.at(-1).items.map(({ label }) => label),
    ["Local file", "Hugging Face", "Backend test"],
  );
  const graph = extension.messages.at(-1).graph;
  const repository = {
    files: [
      { url, fileName: "inference.onnx", path: "inference.onnx" },
      {
        url: url.replace("inference.onnx", "other.onnx"),
        fileName: "other.onnx",
        path: "other.onnx",
      },
    ],
    selectedFile: "inference.onnx",
  };
  await extension.openHuggingFace("viewer-tests/fixtures", false, repository);
  assert.equal(extension.messages.at(-1).graph.name, graph.name);
  assert.equal(extension.picks.at(-1).options.title, "Choose an ONNX file");
  await extension.openHuggingFace(
    "https://huggingface.co/viewer-tests/fixtures",
    false,
    { files: [repository.files[0]] },
  );
  assert.deepEqual(extension.listedRepositories, [
    { url: "https://huggingface.co/viewer-tests/fixtures" },
    { url: "https://huggingface.co/viewer-tests/fixtures" },
  ]);
  assert.ok(graph.nodes.some(({ opType }) => opType === "Reshape"));
  assert.ok(
    extension.logs.some((line) =>
      line.includes(`[SUCCEEDED] Cache hit: ${url}`),
    ),
  );
  await extension.receive({ type: "refresh-shapes" });
  assert.equal(extension.messages.at(-1).type, "shape-info");
  assert.ok(!extension.reads.includes(path.join(cache, "model.onnx")));
  await extension.openHuggingFace("https://example.com/model.onnx");
  assert.equal(extension.messages.at(-1).type, "load-error");
  await extension.receive({ type: "ready" });
  assert.equal(extension.messages.at(-1).graph.name, graph.name);
  const errors = extension.errors.length;
  const history = structuredClone(extension.preferences.get("pathHistory"));
  await extension.openHuggingFace(undefined);
  await extension.openHuggingFace(url, true);
  await extension.openHuggingFace("viewer-tests/fixtures", false, {
    files: repository.files,
  });
  await extension.cancelOpen();
  await extension.receive({ type: "ready" });
  assert.equal(extension.messages.at(-1).graph.name, graph.name);
  assert.equal(extension.errors.length, errors);
  assert.deepEqual(extension.preferences.get("pathHistory"), history);
  assert.equal(
    extension.logs.filter((line) =>
      line.includes("[CANCELLED] Open ONNX model"),
    ).length,
    4,
  );
});
test("recovers a downloaded model from its published cache path after the temporary file is renamed", async (t) => {
  const storage = await mkdtemp(
    path.join(tmpdir(), "onnx-viewer-hf-recovery-"),
  );
  t.after(() => rm(storage, { recursive: true }));
  const bytes = await readFile(
    path.join(__dirname, "models/custom-unannotated.onnx"),
  );
  let downloads = 0;
  const server = createServer((_request, response) => {
    downloads++;
    response.end(bytes);
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  t.after(async () => {
    server.closeAllConnections();
    await new Promise((resolve) => server.close(resolve));
  });
  const extension = await openExtension(
    undefined,
    undefined,
    true,
    storage,
    `http://127.0.0.1:${server.address().port}/custom.onnx`,
    new Map([
      [
        "applicationSettings",
        {
          downloadDirectory: path.join(storage, "huggingface"),
          pythonPath: process.platform === "win32" ? "python" : "python3",
        },
      ],
    ]),
  );
  await extension.openHuggingFace(
    "https://huggingface.co/viewer-tests/fixtures/resolve/main/custom.onnx",
  );
  const progress = extension.events.filter(
    (event) => event.type === "model-progress" && event.received !== undefined,
  );
  assert.equal(progress[0].received, 0);
  assert.equal(progress.at(-1).received, bytes.length);
  assert.equal(progress.at(-1).total, bytes.length);
  assert.equal(
    extension.preferences.get("pathHistory").open[0].path,
    "https://huggingface.co/viewer-tests/fixtures/resolve/main/custom.onnx",
  );
  assert.equal(extension.messages.at(-1).type, "graph");
  await extension.receive({ type: "refresh-shapes" });
  assert.equal(extension.messages.at(-1).type, "shape-error");
  await extension.receive({ type: "metadata" });
  assert.equal(extension.messages.at(-1).type, "metadata-info");
  assert.equal(downloads, 1);
  assert.deepEqual(extension.reads, []);
});

test("selects optimization groups and previews global passes through the real webview bridge", async (t) => {
  const extension = await openExtension("../../examples/optimization.onnx");
  t.after(() => extension.dispose());
  const dom = new JSDOM(extension.webview.html, { runScripts: "outside-only" });
  t.after(() => dom.window.close());
  const { window } = dom;
  window.structuredClone = structuredClone;
  const { document } = window;
  const requests = [];
  const reportDialog = document.getElementById("optimization-report");
  reportDialog.showModal = () => {
    reportDialog.open = true;
  };
  reportDialog.close = () => {
    reportDialog.open = false;
  };
  window.HTMLCanvasElement.prototype.getContext = () => ({
    font: "",
    measureText: (text) => ({ width: text.length * 8 }),
  });
  document.getElementById("canvas").getBoundingClientRect = () =>
    new window.DOMRect(0, 0, 1060, 350);
  window.acquireVsCodeApi = () => ({
    postMessage: (message) => requests.push(extension.receive(message)),
  });
  extension.connect((data) =>
    window.dispatchEvent(new window.MessageEvent("message", { data })),
  );
  for (const script of document.querySelectorAll("script[src]"))
    window.eval(readFileSync(new URL(script.src), "utf8"));
  const flush = async () => {
    while (requests.length) await Promise.all(requests.splice(0));
  };
  await flush();
  const count = () => document.querySelectorAll(".graph-node").length;
  const original = count();
  document.getElementById("optimize-model").click();
  await flush();
  const patterns = document.getElementById("optimization-patterns");
  assert.equal(document.getElementById("optimization-panel").hidden, false);
  assert.equal(document.getElementById("fusion-panel").hidden, true);
  assert.equal(document.getElementById("optimization-picker").tagName, "DIV");
  assert.equal(document.querySelector("#optimization-picker > summary"), null);
  assert.ok(patterns.querySelector('input[value="GemmTranspose"]'));
  const nativeCatalog = extension.messages.at(-1).optimizationPatterns;
  assert.equal(
    patterns.querySelector('input[value="RepeatedLayer"]').checked,
    false,
  );
  const groupInput = (id, name) => {
    const picker = document.getElementById(id);
    assert.equal(picker.disabled, false);
    const input = picker.querySelector(`input[value="${name}"]`);
    assert.ok(input);
    return input;
  };
  const chooseGroup = (id, name) => {
    const input = groupInput(id, name);
    input.parentElement.click();
    return input.checked;
  };
  const checked = (fieldset) =>
    [...fieldset.querySelectorAll("input:checked")]
      .filter((input) => !input.hasAttribute("data-priority"))
      .map((input) => input.value)
      .sort();
  const members = (...groups) =>
    nativeCatalog
      .filter((pattern) =>
        groups.some((name) =>
          name === "Default"
            ? pattern.enabledByDefault
            : pattern.groups.includes(name),
        ),
      )
      .map((pattern) => pattern.name)
      .sort();
  assert.equal(
    groupInput("optimization-pattern-group", "Default").checked,
    true,
  );
  assert.equal(groupInput("optimization-pattern-group", "all").checked, false);
  assert.deepEqual(checked(patterns), members("Default"));
  assert.ok(members("onnx-light-cpu").length > 0);
  for (const name of [
    "Default",
    "Layer",
    "L1",
    "L2",
    "onnx-light-cpu",
    "all",
  ]) {
    document.getElementById("clear-optimization-patterns").click();
    chooseGroup("optimization-pattern-group", name);
    assert.deepEqual(checked(patterns), members(name));
  }
  document.getElementById("clear-optimization-patterns").click();
  for (const name of ["Default", "L1", "L2", "Layer"])
    assert.equal(chooseGroup("optimization-pattern-group", name), true);
  assert.deepEqual(checked(patterns), members("Default", "L1", "L2", "Layer"));
  assert.equal(chooseGroup("optimization-pattern-group", "Default"), false);
  assert.deepEqual(checked(patterns), members("L1", "L2", "Layer"));
  assert.equal(chooseGroup("optimization-pattern-group", "L1"), false);
  assert.deepEqual(checked(patterns), members("L2", "Layer"));
  assert.equal(chooseGroup("optimization-pattern-group", "Layer"), false);
  assert.deepEqual(checked(patterns), members("L2"));
  assert.equal(chooseGroup("optimization-pattern-group", "L2"), false);
  assert.deepEqual(checked(patterns), []);
  assert.equal(count(), original);
  assert.ok(patterns.querySelectorAll("details").length > 1);
  document.getElementById("clear-optimization-patterns").click();
  assert.equal(document.getElementById("run-optimization").disabled, true);
  const group = [...patterns.querySelectorAll("details")].find(
    (group) => group.querySelectorAll("input[data-pattern]").length > 1,
  );
  const toggle = group.querySelector("input[data-priority]");
  toggle.click();
  assert.equal(
    group.querySelectorAll("input[data-pattern]:checked").length,
    group.querySelectorAll("input[data-pattern]").length,
  );
  group.querySelector("input[data-pattern]").click();
  assert.equal(toggle.indeterminate, true);
  document.getElementById("select-optimization-patterns").click();
  assert.equal(groupInput("optimization-pattern-group", "all").checked, true);
  patterns.querySelector('input[value="RepeatedLayer"]').click();
  assert.equal(
    groupInput("optimization-pattern-group", "Default").checked,
    true,
  );
  assert.equal(groupInput("optimization-pattern-group", "all").checked, false);
  const defaultPattern = patterns.querySelector("input[data-pattern]:checked");
  defaultPattern.click();
  assert.equal(
    document.querySelectorAll("#optimization-pattern-group input:checked")
      .length,
    0,
  );
  defaultPattern.click();
  assert.equal(
    groupInput("optimization-pattern-group", "Default").checked,
    true,
  );
  document.getElementById("select-optimization-patterns").click();
  document.getElementById("optimize-model").click();
  document.getElementById("fuse-pattern").click();
  await flush();
  const fusionPatterns = document.getElementById("fusion-pattern");
  const available = [...fusionPatterns.querySelectorAll("input")].map(
    (input) => input.value,
  );
  document.getElementById("clear-patterns").click();
  chooseGroup("fusion-pattern-group", "onnx-light-cpu");
  assert.deepEqual(
    checked(fusionPatterns),
    members("onnx-light-cpu").filter((name) => available.includes(name)),
  );
  assert.ok(checked(fusionPatterns).length > 0);
  chooseGroup("fusion-pattern-group", "onnx-light-cpu");
  chooseGroup("fusion-pattern-group", "Layer");
  assert.deepEqual(checked(fusionPatterns), ["RepeatedLayer"]);
  chooseGroup("fusion-pattern-group", "Default");
  assert.equal(groupInput("fusion-pattern-group", "Layer").checked, true);
  assert.deepEqual(
    checked(fusionPatterns),
    members("Default", "Layer").filter((name) => available.includes(name)),
  );
  chooseGroup("fusion-pattern-group", "Default");
  assert.deepEqual(checked(fusionPatterns), ["RepeatedLayer"]);
  chooseGroup("fusion-pattern-group", "all");
  assert.deepEqual(
    checked(fusionPatterns),
    nativeCatalog
      .filter((pattern) => available.includes(pattern.name))
      .map((pattern) => pattern.name)
      .sort(),
  );
  assert.ok(
    !checked(fusionPatterns).some((name) =>
      ["IdentityRemoval", "ConstantFolding", "RemoveUnusedNodes"].includes(
        name,
      ),
    ),
  );
  chooseGroup("fusion-pattern-group", "all");
  assert.deepEqual(checked(fusionPatterns), ["RepeatedLayer"]);
  document.getElementById("clear-patterns").click();
  const selectPass = (name) =>
    document.querySelector(`#fusion-pattern input[value="${name}"]`).click();
  for (const name of [
    "IdentityRemoval",
    "ConstantFolding",
    "RemoveUnusedNodes",
  ])
    selectPass(name);
  await flush();
  document.getElementById("preview-fusion").click();
  await flush();
  assert.equal(count(), original);
  const candidates = document.querySelectorAll("#fusion-matches input");
  assert.equal(candidates.length, 3);
  candidates[0].click();
  candidates[1].click();
  assert.equal(document.getElementById("apply-fusion").disabled, true);
  assert.match(
    document.getElementById("fusion-feedback").textContent,
    /individually/,
  );
  document.getElementById("select-fusions").click();
  assert.equal(
    document.querySelectorAll("#fusion-matches input:checked").length,
    1,
  );
  document.getElementById("clear-fusions").click();
  [...document.querySelectorAll("#fusion-matches li")]
    .find(
      (item) => item.querySelector("label").textContent === "ConstantFolding",
    )
    .querySelector("input")
    .click();
  document.getElementById("apply-fusion").click();
  await flush();
  assert.equal(count(), original - 1);
  assert.match(
    document.getElementById("fusion-status").textContent,
    /Applied 1 fusion/,
  );
  assert.equal(document.getElementById("fusion-panel").hidden, false);
  selectPass("ConstantFolding");
  await flush();
  document.getElementById("preview-fusion").click();
  await flush();
  assert.equal(document.querySelectorAll("#fusion-matches li").length, 0);
  assert.equal(document.getElementById("apply-fusion").disabled, true);
  document.getElementById("clear-patterns").click();
  selectPass("SameChildren");
  await flush();
  assert.deepEqual(
    [...document.getElementById("diagnostic-pattern").options].map(
      (option) => option.value,
    ),
    ["SameChildren"],
  );
  document.getElementById("fusion-diagnostics").open = false;
  document.getElementById("preview-fusion").click();
  await flush();
  assert.match(
    document.getElementById("fusion-feedback").textContent,
    /No matching candidates.*Why did matching fail/,
  );
  assert.equal(document.getElementById("fusion-diagnostics").open, true);
  document
    .querySelector('[data-id="node-0"]')
    .dispatchEvent(
      new window.KeyboardEvent("keydown", { key: "Enter", bubbles: true }),
    );
  document.getElementById("explain-fusion").click();
  await flush();
  assert.match(
    document.getElementById("diagnostic-feedback").textContent,
    /Pattern does not match: no equivalent sibling nodes found/,
  );
  assert.equal(document.getElementById("diagnostic-source").hidden, false);
  document.getElementById("undo-fusion").click();
  document.getElementById("unfuse-last").click();
  await flush();
  assert.equal(count(), original);
  document.getElementById("optimize-model").click();
  await flush();
  document.getElementById("run-optimization").click();
  assert.equal(document.getElementById("run-optimization").disabled, true);
  await flush();
  assert.ok(count() < original);
  assert.equal(document.getElementById("optimization-panel").hidden, false);
  assert.equal(document.getElementById("fusion-panel").hidden, true);
  const optimized = extension.messages.findLast(
    (message) => message.type === "graph",
  ).graph;
  assert.equal(
    optimized.nodes.some(
      (node) => node.opType === "Identity" || node.name === "unused_relu",
    ),
    false,
  );
  const results = document.getElementById("optimization-results");
  assert.match(results.textContent, /MatMulAdd/);
  assert.doesNotMatch(results.textContent, /GemmTranspose/);
  const matchedPatterns = results.textContent;
  const restoreOriginal = document.getElementById(
    "optimization-restore-original",
  );
  const unfuseOptimization = document.getElementById(
    "optimization-unfuse-last",
  );
  const pendingCount = () =>
    extension.messages
      .findLast(({ type }) => type === "graph")
      .file.pendingRewritings.flatMap((operation) => operation.matches).length;
  assert.equal(
    document.getElementById("rewriting-count").textContent,
    "0 pending rewritings",
  );
  assert.equal(restoreOriginal.disabled, false);
  assert.equal(unfuseOptimization.disabled, false);
  document.getElementById("optimize-model").click();
  document.getElementById("optimize-model").click();
  await flush();
  assert.equal(results.textContent, matchedPatterns);
  document.getElementById("optimizer-onnxscript").click();
  assert.equal(
    document.getElementById("native-optimization-history").hidden,
    true,
  );
  document.getElementById("optimizer-onnx-light").click();
  assert.equal(
    document.getElementById("native-optimization-history").hidden,
    false,
  );
  assert.equal(results.textContent, matchedPatterns);
  document.getElementById("run-optimization").click();
  await flush();
  assert.equal(
    extension.messages.findLast(
      (message) => message.type === "optimization-info",
    ).changed,
    false,
  );
  assert.equal(results.textContent, matchedPatterns);
  assert.equal(reportDialog.open, false);
  document.getElementById("show-optimization-report").click();
  assert.equal(reportDialog.open, true);
  const patternRows = document.querySelectorAll(
    "#optimization-report-patterns tbody tr",
  );
  assert.equal(
    patternRows.length,
    extension.messages.findLast(
      (message) => message.type === "optimization-info",
    ).optimizationReport.patterns.length,
  );
  assert.match(
    document.getElementById("optimization-report-rejections").textContent,
    /not consumed/,
  );
  document.getElementById("close-optimization-report").click();
  assert.equal(reportDialog.open, false);
  [...results.querySelectorAll("button")]
    .find((button) => button.textContent.startsWith("MatMulAdd "))
    .click();
  assert.ok(document.querySelectorAll(".graph-node.selected").length);
  document.getElementById("undo-fusion").click();
  assert.equal(document.querySelectorAll("#unfuse-history > li").length, 2);
  unfuseOptimization.click();
  assert.equal(restoreOriginal.disabled, true);
  await flush();
  assert.equal(count(), original - 1);
  assert.equal(document.getElementById("optimization-panel").hidden, false);
  assert.equal(restoreOriginal.disabled, false);
  assert.equal(unfuseOptimization.disabled, false);
  assert.equal(results.querySelectorAll("button").length, 1);
  assert.ok(pendingCount() > 0);
  assert.equal(
    document.querySelectorAll("#pending-rewriting-steps li").length,
    pendingCount(),
  );
  document.getElementById("run-optimization").click();
  await flush();
  assert.ok(count() < original);
  restoreOriginal.click();
  await flush();
  assert.equal(count(), original);
  assert.equal(document.getElementById("optimization-panel").hidden, false);
  assert.equal(restoreOriginal.disabled, true);
  assert.equal(results.querySelectorAll("button").length, 0);
  assert.ok(pendingCount() > 0);
  assert.equal(document.getElementById("pending-rewritings").hidden, false);
  assert.match(
    document.getElementById("pending-rewriting-steps").textContent,
    /ConstantFolding/,
  );
  assert.match(
    document.getElementById("optimization-status").textContent,
    /kept pending/,
  );
  assert.equal(
    document.getElementById("rewriting-count").textContent,
    `${pendingCount()} pending rewritings`,
  );
  document.getElementById("replay-rewritings").click();
  await flush();
  assert.deepEqual(
    extension.messages.findLast(({ type }) => type === "graph").graph,
    optimized,
  );
  assert.equal(
    document.getElementById("rewriting-count").textContent,
    "0 pending rewritings",
  );
  assert.equal(document.getElementById("pending-rewritings").hidden, true);
  assert.deepEqual(extension.errors, []);
});

test("keeps an unfused rewriting queue visible, replays it before matching and appends new fusions", async (t) => {
  const { loadModel } = require("../dist/graph");
  const extension = await openExtension("fusion.onnx", undefined, true);
  t.after(() => extension.dispose());
  const dom = new JSDOM(extension.webview.html, { runScripts: "outside-only" });
  t.after(() => dom.window.close());
  const { window } = dom;
  const { document } = window;
  window.structuredClone = structuredClone;
  window.HTMLCanvasElement.prototype.getContext = () => ({
    font: "",
    measureText: (text) => ({ width: text.length * 8 }),
  });
  document.getElementById("canvas").getBoundingClientRect = () =>
    new window.DOMRect(0, 0, 1060, 350);
  const requests = [];
  window.acquireVsCodeApi = () => ({
    postMessage: (message) => requests.push(extension.receive(message)),
  });
  extension.connect((data) =>
    window.dispatchEvent(new window.MessageEvent("message", { data })),
  );
  for (const script of document.querySelectorAll("script[src]"))
    window.eval(readFileSync(new URL(script.src), "utf8"));
  const flush = async () => {
    while (requests.length) await Promise.all(requests.splice(0));
  };
  const latest = () =>
    extension.messages.findLast(({ type }) => type === "graph");
  const counter = () => document.getElementById("rewriting-count").textContent;
  const match = async () => {
    document
      .querySelector('#fusion-pattern input[value="TransposeTranspose"]')
      .click();
    await flush();
    document.getElementById("preview-fusion").click();
    await flush();
  };
  const applyFirst = async () => {
    document.querySelector("#fusion-matches input").click();
    document.getElementById("apply-fusion").click();
    await flush();
  };
  const restore = async () => {
    document.getElementById("optimize-model").click();
    await flush();
    document.getElementById("optimization-restore-original").click();
    await flush();
  };
  await flush();
  const original = latest().graph;
  document.getElementById("fuse-pattern").click();
  await flush();
  await match();
  assert.equal(document.querySelectorAll("#fusion-matches input").length, 2);
  await applyFirst();
  const first = latest().graph;
  assert.equal(latest().file.fusionCount, 1);
  assert.equal(counter(), "0 pending rewritings");
  await restore();
  assert.deepEqual(latest().graph, original);
  assert.equal(counter(), "1 pending rewriting");
  assert.match(
    document.getElementById("pending-rewriting-steps").textContent,
    /TransposeTranspose/,
  );
  assert.match(
    document.getElementById("optimization-status").textContent,
    /1 rewriting kept pending/,
  );
  document.getElementById("fuse-pattern").click();
  await flush();
  assert.deepEqual(latest().graph, original);
  assert.equal(counter(), "1 pending rewriting");
  await match();
  assert.deepEqual(latest().graph, first);
  assert.equal(counter(), "0 pending rewritings");
  assert.equal(document.getElementById("fusion-panel").hidden, false);
  assert.equal(document.querySelectorAll("#fusion-matches input").length, 1);
  assert.match(
    document.getElementById("fusion-status").textContent,
    /Pending rewritings reapplied/,
  );
  await applyFirst();
  const second = latest().graph;
  assert.equal(latest().file.fusionCount, 2);
  assert.equal(
    latest().file.fusionHistory.flatMap((operation) => operation.matches)
      .length,
    2,
  );
  await restore();
  assert.deepEqual(latest().graph, original);
  assert.equal(counter(), "2 pending rewritings");
  assert.equal(
    document.querySelectorAll("#pending-rewriting-steps li").length,
    2,
  );
  const directory = await mkdtemp(
    path.join(tmpdir(), "viewer-queued-rewritings-"),
  );
  t.after(() => rm(directory, { recursive: true, force: true }));
  const destination = path.join(directory, "queued.onnx");
  await extension.receive({
    type: "save",
    location: destination,
    overwrite: false,
    includeRewritings: true,
  });
  const saved = loadModel(await readFile(destination), "queued");
  const [scenario] = saved.rewritingScenarios();
  assert.equal(scenario.steps.length, 2);
  assert.deepEqual(saved.graph(), original);
  assert.deepEqual(
    (await saved.applyRewritingScenario(scenario.name, true)).graph(),
    second,
  );
  assert.equal(counter(), "2 pending rewritings");
  const plainPath = path.join(directory, "plain.onnx");
  await extension.receive({
    type: "save",
    location: plainPath,
    overwrite: false,
    includeRewritings: false,
  });
  const plain = loadModel(await readFile(plainPath), "plain");
  assert.deepEqual(plain.graph(), original);
  assert.deepEqual(plain.rewritingScenarios(), []);
  assert.equal(counter(), "2 pending rewritings");
  await extension.open("relu.onnx");
  assert.equal(counter(), "0 pending rewritings");
  assert.equal(document.getElementById("pending-rewritings").hidden, true);
  assert.deepEqual(extension.errors, []);
});

test("matches multiple patterns and fuses checked candidates through the real webview bridge", async (t) => {
  const extension = await openExtension("fusion-scaled.onnx");
  const dom = new JSDOM(extension.webview.html, { runScripts: "outside-only" });
  t.after(() => dom.window.close());
  const { window } = dom;
  window.structuredClone = structuredClone;
  const { document } = window;
  const requests = [];
  window.HTMLCanvasElement.prototype.getContext = () => ({
    font: "",
    measureText: (text) => ({ width: text.length * 8 }),
  });
  document.getElementById("canvas").getBoundingClientRect = () =>
    new window.DOMRect(0, 0, 1060, 350);
  window.acquireVsCodeApi = () => ({
    postMessage: (message) => requests.push(extension.receive(message)),
  });
  extension.connect((data) =>
    window.dispatchEvent(new window.MessageEvent("message", { data })),
  );
  for (const script of document.querySelectorAll("script[src]"))
    window.eval(readFileSync(new URL(script.src), "utf8"));
  const flush = async () => {
    while (requests.length) await Promise.all(requests.splice(0));
  };
  await flush();
  const originalNodeCount = document.querySelectorAll(".graph-node").length;
  assert.equal(document.getElementById("undo-fusion").disabled, true);
  document.getElementById("fuse-pattern").click();
  await flush();
  const list = document.getElementById("fusion-pattern");
  const actual = [...list.querySelectorAll('input[type="checkbox"]')].map(
    (input) => input.value,
  );
  const expected = extension.nativeModels.at(-1).fusionPatterns().sort();
  assert.ok(actual.length > 0);
  assert.deepEqual(actual, expected);
  assert.ok(actual.includes("TransposeTranspose"));
  assert.equal(list.disabled, false);
  assert.equal(list.closest("details").open, true);
  assert.equal(document.getElementById("fusion-panel").hidden, false);
  assert.equal(
    list.closest("aside"),
    document.querySelector(".graph-body aside"),
  );
  for (const input of list.querySelectorAll("input"))
    if (["TransposeTranspose", "MatMulScaleFusion"].includes(input.value))
      input.parentElement.click();
  await flush();
  assert.equal(document.querySelectorAll("#fusion-matches li").length, 0);
  document.getElementById("preview-fusion").click();
  await flush();
  assert.equal(document.querySelectorAll("#fusion-matches li").length, 3);
  document.querySelector("#fusion-matches input").click();
  const previous = extension.messages.findLast(
    (message) => message.type === "fusion-preview",
  );
  document.getElementById("preview-fusion").click();
  await flush();
  const current = extension.messages.findLast(
    (message) => message.type === "fusion-preview",
  );
  assert.notEqual(current.previewId, previous.previewId);
  assert.equal(document.querySelectorAll("#fusion-matches li").length, 3);
  assert.equal(
    document.querySelectorAll("#fusion-matches input:checked").length,
    0,
  );
  assert.equal(document.querySelectorAll(".graph-node.selected").length, 0);
  const candidates = document.querySelectorAll("#fusion-matches input");
  candidates[0].click();
  candidates[1].click();
  assert.equal(document.querySelectorAll(".graph-node.selected").length, 2);
  document.getElementById("apply-fusion").click();
  assert.match(
    document.getElementById("fusion-status").textContent,
    /Applying 2 fusion/,
  );
  await flush();
  assert.equal(document.querySelectorAll(".graph-node").length, 4);
  assert.match(
    document.getElementById("fusion-status").textContent,
    /Applied 2 fusion/,
  );
  assert.equal(document.getElementById("fusion-panel").hidden, false);
  const refreshed = extension.messages.findLast(
    (message) => message.type === "fusion-info",
  );
  assert.deepEqual(
    [...list.querySelectorAll("input")].map((input) => input.value),
    [...refreshed.patterns].sort(),
  );
  assert.equal(document.querySelectorAll("#fusion-matches li").length, 0);
  assert.equal(list.querySelectorAll("input:checked").length, 0);
  list.querySelector('input[value="TransposeTranspose"]').parentElement.click();
  await flush();
  document.getElementById("preview-fusion").click();
  await flush();
  assert.equal(document.querySelectorAll("#fusion-matches li").length, 1);
  const undo = document.getElementById("undo-fusion");
  assert.equal(undo.disabled, false);
  undo.click();
  const history = document.getElementById("unfuse-history");
  assert.equal(history.children.length, 1);
  for (const match of current.matches.slice(0, 2))
    assert.ok(history.textContent.includes(match.pattern));
  const operation = extension.messages.findLast(
    (message) => message.type === "graph",
  ).file.fusionHistory[0];
  const pattern = operation.matches[0].pattern;
  [...history.querySelectorAll("button")]
    .find((button) => button.textContent.startsWith(`${pattern} (`))
    .click();
  await flush();
  assert.deepEqual(
    [...document.querySelectorAll(".graph-node.selected")]
      .map((node) => node.dataset.id)
      .sort(),
    operation.matches[0].nodes.map((node) => `node-${node}`).sort(),
  );
  document.getElementById("unfuse-last").click();
  assert.equal(undo.disabled, true);
  await flush();
  assert.equal(history.children.length, 0);
  assert.equal(
    document.querySelectorAll(".graph-node").length,
    originalNodeCount,
  );
  assert.equal(undo.disabled, true);
  assert.ok(
    extension.logs.some((line) => line.includes("[SUCCEEDED] Undo fusion")),
  );
  assert.deepEqual(extension.errors, []);
});

test("keeps operation logs quiet and exposes real native exception recovery in Status", async (t) => {
  const extension = await openExtension();
  const dom = new JSDOM(extension.webview.html, { runScripts: "outside-only" });
  t.after(() => dom.window.close());
  const { window } = dom;
  window.structuredClone = structuredClone;
  const { document } = window;
  const requests = [];
  window.HTMLCanvasElement.prototype.getContext = () => ({
    font: "",
    measureText(text) {
      return { width: text.length * 8 };
    },
  });
  document.getElementById("canvas").getBoundingClientRect = () =>
    new window.DOMRect(0, 0, 1060, 350);
  window.acquireVsCodeApi = () => ({
    postMessage: (message) => requests.push(extension.receive(message)),
  });
  extension.connect((data) =>
    window.dispatchEvent(new window.MessageEvent("message", { data })),
  );
  for (const script of document.querySelectorAll("script[src]"))
    window.eval(readFileSync(new URL(script.src), "utf8"));
  const flush = async () => {
    while (requests.length) await Promise.all(requests.splice(0));
  };
  await flush();
  const entries = () => [...document.querySelectorAll(".log-entry")];
  assert.match(
    document.getElementById("status-library").textContent,
    /Loaded.*onnx-light/,
  );
  assert.equal(document.getElementById("status-handle").textContent, "Ready");
  assert.equal(document.getElementById("status-running").textContent, "Idle");
  assert.ok(
    entries().some(
      (row) =>
        row.dataset.status === "SUCCEEDED" &&
        /Load model:/.test(row.textContent),
    ),
  );
  assert.equal(
    document.getElementById("logs-view").parentElement,
    document.querySelector(".workspace"),
  );

  for (const id of ["refresh-shapes", "refresh-metadata"]) {
    document.getElementById(id).click();
    const row = entries().at(-1);
    assert.equal(row.dataset.status, "RUNNING");
    document.getElementById("status-tab").click();
    assert.match(
      document.getElementById("status-running").textContent,
      /Refresh/,
    );
    document.getElementById("logs-tab").click();
    assert.equal(document.getElementById("model-view").hidden, true);
    await flush();
    assert.equal(row.dataset.status, "SUCCEEDED");
    assert.equal(row.querySelector(".log-status").textContent, "OK");
    assert.equal(document.getElementById("status-running").textContent, "Idle");
    document.getElementById("model-tab").click();
    await flush();
  }

  document.getElementById("refresh-positions").click();
  const positions = entries().at(-1);
  await flush();
  assert.match(positions.textContent, /OKRefresh positions/);
  const count = entries().length;
  document
    .querySelector(".graph-node")
    .dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
  document.getElementById("logs-tab").click();
  document.getElementById("logs-view").click();
  document.getElementById("log-entries").click();
  for (const id of ["status-tab", "model-tab", "zoom-in"])
    document.getElementById(id).click();
  document
    .getElementById("graph")
    .dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
  extension.showLogs();
  await flush();
  assert.equal(entries().length, count);

  await extension.open("custom-unannotated.onnx");
  const reads = extension.reads.length;
  const failedHandle = extension.nativeModels.at(-1);
  document.getElementById("model-tab").click();
  document.getElementById("refresh-shapes").click();
  const failure = entries().at(-1);
  await flush();
  assert.equal(failure.dataset.status, "FAILED");
  assert.match(
    failure.querySelector("pre").textContent,
    /Error: ComputeShapeModel:/,
  );
  document.getElementById("status-tab").click();
  assert.match(
    document.getElementById("status-library").textContent,
    /Loaded.*onnx-light/,
  );
  assert.match(
    document.getElementById("status-handle").textContent,
    /Discarded/,
  );
  assert.match(
    document.getElementById("status-recovery").textContent,
    /next native action recreates/,
  );
  assert.match(
    document.getElementById("status-failure-details").textContent,
    /ComputeShapeModel:/,
  );
  assert.equal(document.getElementById("status-running").textContent, "Idle");
  document.getElementById("model-tab").click();
  document.getElementById("refresh-metadata").click();
  document.getElementById("status-tab").click();
  await flush();
  assert.notEqual(extension.nativeModels.at(-1), failedHandle);
  assert.equal(extension.reads.length, reads);
  assert.equal(document.getElementById("status-handle").textContent, "Ready");
  assert.equal(
    document.getElementById("status-recovery").textContent,
    "No model recovery pending.",
  );
  assert.match(
    document.getElementById("status-failure-details").textContent,
    /ComputeShapeModel:/,
  );
  await extension.receive({ type: "ready" });
  const replay = extension.events
    .filter(({ type }) => type === "log-history")
    .at(-1).entries;
  assert.equal(entries().length, replay.length);
  assert.equal(new Set(replay.map(({ id }) => id)).size, replay.length);
});

test("shows missing-graph loading errors in the canvas even before the webview is ready", async (t) => {
  const extension = await openExtension("missing-graph.onnx");
  const dom = new JSDOM(extension.webview.html, { runScripts: "outside-only" });
  t.after(() => dom.window.close());
  const { window } = dom;
  window.structuredClone = structuredClone;
  const { document } = window;
  const requests = [];
  window.HTMLCanvasElement.prototype.getContext = () => ({
    font: "",
    measureText: (text) => ({ width: text.length * 8 }),
  });
  document.getElementById("canvas").getBoundingClientRect = () =>
    new window.DOMRect(0, 0, 1060, 350);
  window.acquireVsCodeApi = () => ({
    postMessage: (message) => requests.push(extension.receive(message)),
  });
  extension.connect((data) =>
    window.dispatchEvent(new window.MessageEvent("message", { data })),
  );
  for (const script of document.querySelectorAll("script[src]"))
    window.eval(readFileSync(new URL(script.src), "utf8"));
  while (requests.length) await Promise.all(requests.splice(0));
  const error = document.getElementById("model-error");
  assert.equal(error.hidden, false);
  assert.equal(document.getElementById("model-view").hidden, false);
  assert.match(error.textContent, /does not contain an ONNX model graph/);
  assert.equal(document.getElementById("load-progress").hidden, true);
  assert.equal(document.getElementById("open-model").disabled, false);
  await extension.open("relu.onnx");
  while (requests.length) await Promise.all(requests.splice(0));
  assert.equal(error.hidden, true);
  const graph = document.getElementById("nodes").innerHTML;
  await extension.open("missing-graph.onnx");
  while (requests.length) await Promise.all(requests.splice(0));
  assert.equal(error.hidden, false);
  assert.equal(document.getElementById("nodes").innerHTML, graph);
});

test("reports a file deleted after history display as soon as opening fails and allows retry", async (t) => {
  const directory = await mkdtemp(path.join(tmpdir(), "onnx-viewer-missing-"));
  t.after(() => rm(directory, { recursive: true }));
  const source = require.resolve("./models/relu.onnx");
  const deleted = path.join(directory, "deleted.onnx");
  await copyFile(source, deleted);
  const preferences = new Map();
  const extension = await openExtension(
    undefined,
    undefined,
    true,
    undefined,
    undefined,
    preferences,
  );
  t.after(() => extension.dispose());
  await extension.openFile(deleted);
  const history = structuredClone(preferences.get("pathHistory"));
  const dom = new JSDOM(extension.webview.html, { runScripts: "outside-only" });
  t.after(() => dom.window.close());
  const { window } = dom;
  const { document } = window;
  const byId = (id) => document.getElementById(id);
  window.structuredClone = structuredClone;
  window.HTMLCanvasElement.prototype.getContext = () => ({
    font: "",
    measureText: (text) => ({ width: text.length * 8 }),
  });
  byId("canvas").getBoundingClientRect = () =>
    new window.DOMRect(0, 0, 1060, 350);
  const requests = [];
  window.acquireVsCodeApi = () => ({
    postMessage: (message) => requests.push(extension.receive(message)),
  });
  let failureFeedback;
  extension.connect((data) => {
    window.dispatchEvent(new window.MessageEvent("message", { data }));
    if (
      data.type === "log-entry" &&
      data.entry.action === "Open ONNX model" &&
      data.entry.status === "FAILED"
    )
      failureFeedback = {
        visible: !byId("model-error").hidden,
        message: byId("open-error").textContent,
      };
  });
  for (const script of document.querySelectorAll("script[src]"))
    window.eval(readFileSync(new URL(script.src), "utf8"));
  while (requests.length) await Promise.all(requests.splice(0));
  const graph = byId("nodes").innerHTML;
  byId("open-model").click();
  while (requests.length) await Promise.all(requests.splice(0));
  await rm(deleted);
  [...byId("open-history-entries").querySelectorAll("button")]
    .find((button) => button.textContent === deleted)
    .click();
  byId("open-form").requestSubmit();
  while (requests.length) await Promise.all(requests.splice(0));
  assert.equal(failureFeedback.visible, true);
  assert.match(failureFeedback.message, /ENOENT/);
  assert.ok(failureFeedback.message.includes(deleted));
  assert.equal(byId("nodes").innerHTML, graph);
  assert.equal(byId("open-panel").hidden, false);
  assert.equal(byId("load-progress").hidden, true);
  assert.equal(byId("open-progress").hidden, true);
  assert.equal(byId("model-view").getAttribute("aria-busy"), "false");
  assert.equal(byId("open-submit").disabled, false);
  assert.equal(byId("open-model").disabled, false);
  assert.equal(byId("save-model").disabled, false);
  assert.ok(extension.errors.some((error) => error.includes(deleted)));
  assert.deepEqual(preferences.get("pathHistory"), history);
  byId("open-file").value = source;
  byId("open-form").requestSubmit();
  while (requests.length) await Promise.all(requests.splice(0));
  assert.equal(byId("model-error").hidden, true);
  assert.equal(byId("open-panel").hidden, true);
  assert.equal(byId("open-model").disabled, false);
});

test("keeps failed initial loading inspectable and can open Logs before any model", async () => {
  const failed = await openExtension("missing-ir.onnx");
  await failed.receive({ type: "ready" });
  const history = failed.events.find(
    ({ type }) => type === "log-history",
  ).entries;
  assert.equal(history.length, 1);
  assert.equal(history[0].status, "FAILED");
  assert.match(history[0].details, /ONNX model graph/);
  assert.match(failed.messages.at(-1).error, /ONNX model graph/);
  const failedStatus = failed.events
    .filter(({ type }) => type === "application-status")
    .at(-1);
  assert.equal(failedStatus.library.loaded, true);
  assert.equal(failedStatus.model.state, "none");
  failed.showLogs();
  assert.deepEqual(failed.logViews, [true]);
  await failed.open("inference.onnx");
  assert.equal(failed.messages.at(-1).type, "graph");

  const empty = await openExtension(undefined, "onnxLightViewer.showLogs");
  assert.match(empty.webview.html, /id="logs-tab"[^>]*aria-selected="true"/);
  await empty.receive({ type: "ready" });
  assert.equal(empty.messages.at(-1).type, "load-state");
  assert.equal(empty.messages.at(-1).pending, false);
  assert.equal(
    empty.events.filter(({ type }) => type === "application-status").at(-1)
      .model.state,
    "none",
  );
  assert.deepEqual(empty.logs, []);
  await empty.receive({ type: "open", actionId: "ui:cancelled-file-picker" });
  assert.ok(
    empty.events.some(
      ({ type, entry }) =>
        type === "log-entry" &&
        entry.id === "ui:cancelled-file-picker" &&
        entry.status === "CANCELLED",
    ),
  );
});

test("verifies the current model, reports errors and preserves native and display state", async () => {
  const extension = await openExtension("inference.onnx");
  const native = extension.nativeModels.at(-1);
  const before = native.graph();
  await extension.receive({ type: "verify", actionId: "ui:verify" });
  assert.equal(extension.messages.at(-1).type, "verify-info");
  assert.deepEqual(native.graph(), before);
  assert.ok(
    extension.events.some(
      ({ type, entry }) =>
        type === "log-entry" &&
        entry.id === "ui:verify" &&
        entry.status === "SUCCEEDED",
    ),
  );
  await extension.open("invalid-graph.onnx");
  const invalid = extension.nativeModels.at(-1);
  await extension.receive({ type: "verify" });
  assert.equal(extension.messages.at(-1).type, "verify-error");
  assert.match(extension.messages.at(-1).error, /^VerifyModel:/);
  assert.match(extension.errors.at(-1), /ONNX verification failed/);
  const count = extension.nativeModels.length;
  await extension.receive({ type: "verify" });
  assert.equal(extension.nativeModels.length, count);
  assert.equal(extension.nativeModels.at(-1), invalid);
  await extension.open("tiny.pipe-onnx");
  await extension.receive({ type: "verify" });
  assert.equal(extension.messages.at(-1).type, "verify-info");
});

test("keeps timestamped action history, verifies invalid graphs and reloads before retrying", async () => {
  const extension = await openExtension();
  assert.match(
    extension.logs[0],
    /^\[\d{4}-\d{2}-\d{2}T[\d:.]+Z\] \[RUNNING\] Load model:/,
  );
  assert.match(extension.logs[1], /\[SUCCEEDED\] Load model:/);
  await extension.open("invalid-graph.onnx");
  const before = extension.messages.at(-1).graph;
  const reads = extension.reads.length;
  const failedNative = extension.nativeModels.at(-1);
  const nativeCount = extension.nativeModels.length;
  await extension.receive({ type: "refresh-shapes" });
  assert.equal(extension.messages.at(-1).type, "shape-error");
  assert.match(extension.messages.at(-1).error, /^VerifyModel:/);
  assert.ok(
    extension.logs.some((line) =>
      /\[FAILED\] Refresh shapes[\s\S]*Error: VerifyModel:/.test(line),
    ),
  );
  assert.ok(extension.logs.some((line) => line.includes("[INVALIDATED]")));
  await extension.receive({ type: "ready" });
  assert.deepEqual(extension.messages.at(-1).graph, before);
  await extension.receive({ type: "metadata" });
  assert.equal(extension.messages.at(-1).type, "metadata-error");
  assert.match(extension.messages.at(-1).error, /^VerifyModel:/);
  assert.equal(extension.nativeModels.length, nativeCount + 1);
  assert.notEqual(extension.nativeModels.at(-1), failedNative);
  assert.ok(
    extension.logs.some((line) =>
      line.includes("[SUCCEEDED] Reload native model: Invalid graph"),
    ),
  );
  assert.equal(extension.reads.length, reads);
  await extension.open("inference.onnx");
  await extension.receive({ type: "refresh-shapes" });
  assert.equal(extension.messages.at(-1).type, "shape-info");
  assert.ok(
    extension.logs.some((line) => line.includes("[SUCCEEDED] Refresh shapes")),
  );
  assert.ok(
    extension.logs.some((line) => line.includes("[FAILED] Refresh metadata")),
  );
  await extension.open("missing-ir.onnx");
  assert.ok(
    extension.logs.some((line) =>
      /\[FAILED\] Load model:[\s\S]*Error:.*ONNX model graph/.test(line),
    ),
  );
  extension.showLogs();
  await extension.receive({ type: "show-logs" });
  assert.deepEqual(extension.logViews, [true, true]);
});

test("opens function call paths through the host and returns to the caller", async () => {
  const extension = await openExtension("functions.onnx");
  await extension.receive({ type: "ready" });
  const root = extension.messages.at(-1).graph;
  const path = root.nodes[0].function.path;
  const reads = extension.reads.length;
  await extension.receive({ type: "navigate", path });
  assert.equal(extension.messages.at(-1).graph.name, "example::Transform");
  await extension.receive({ type: "metadata" });
  assert.equal(extension.messages.at(-1).metadata.nodes.length, 3);
  await extension.receive({ type: "refresh-shapes" });
  assert.equal(extension.messages.at(-1).type, "shape-info");
  await extension.receive({
    type: "navigate",
    path: [{ node: 0, function: 1 }],
  });
  assert.equal(extension.messages.at(-1).type, "navigation-error");
  await extension.receive({ type: "ready" });
  assert.equal(extension.messages.at(-1).graph.name, "example::Transform");
  await extension.receive({ type: "navigate", path: [] });
  assert.equal(extension.messages.at(-1).graph.name, root.name);
  const definition = root.localFunctions[0];
  await extension.receive({ type: "navigate", path: definition.path });
  assert.equal(extension.messages.at(-1).graph.name, definition.name);
  await extension.receive({ type: "metadata" });
  assert.equal(
    extension.messages.at(-1).metadata.nodes.length,
    definition.nodeCount,
  );
  await extension.receive({ type: "navigate", path: [{ function: 9999 }] });
  assert.equal(extension.messages.at(-1).type, "navigation-error");
  await extension.receive({ type: "navigate", path: [] });
  assert.equal(extension.messages.at(-1).graph.name, root.name);
  assert.equal(extension.reads.length, reads);
});

test("navigates retained subgraphs, scopes refreshes and reports invalid paths without replacing the view", async () => {
  const extension = await openExtension("nested-if.onnx");
  await extension.receive({ type: "ready" });
  const root = extension.messages.at(-1).graph;
  const path = root.nodes[0].attributes.find(
    ({ name }) => name === "then_branch",
  ).graphs[0].path;
  const reads = extension.reads.length;
  await extension.receive({ type: "navigate", path });
  const child = extension.messages.at(-1).graph;
  assert.equal(extension.messages.at(-1).statistics, undefined);
  await extension.receive({ type: "get-statistics", requestId: 1 });
  const statistics = extension.messages.at(-1).statistics;
  assert.equal(statistics.nodeCount, root.nodes.length);
  assert.equal(statistics.nodes, undefined);
  assert.equal(statistics.initializers, undefined);
  assert.equal(
    statistics.operatorTypes.reduce(
      (count, operator) => count + operator.count,
      0,
    ),
    root.nodes.length,
  );
  assert.equal(child.breadcrumbs.length, 2);
  assert.equal(child.nodes.length, 2);
  await extension.receive({ type: "metadata" });
  assert.equal(extension.messages.at(-1).metadata.nodes.length, 2);
  await extension.receive({ type: "refresh-shapes" });
  assert.equal(extension.messages.at(-1).type, "shape-info");
  assert.deepEqual(
    extension.messages.at(-1).tensors.find(({ name }) => name === "sum").shape,
    ["N", "3"],
  );
  await extension.receive({ type: "ready" });
  assert.equal(extension.messages.at(-1).graph.breadcrumbs.length, 2);
  for (const path of [null, [{ node: 99, attribute: 0, graph: 0 }]]) {
    await extension.receive({ type: "navigate", path });
    assert.equal(extension.messages.at(-1).type, "navigation-error");
  }
  await extension.receive({ type: "ready" });
  assert.equal(extension.messages.at(-1).graph.breadcrumbs.length, 2);
  await extension.receive({ type: "navigate", path: [] });
  assert.equal(extension.messages.at(-1).graph.breadcrumbs.length, 1);
  assert.equal(extension.messages.at(-1).metadata, undefined);
  assert.equal(extension.reads.length, reads);
  await extension.open("relu.onnx");
  assert.equal(extension.messages.at(-1).graph.breadcrumbs.length, 1);
});

test("refreshes the bundled demo and the currently opened model using its retained native model", async () => {
  const extension = await openExtension();
  await extension.receive({ type: "ready" });
  assert.equal(extension.messages.at(-1).graph.nodes[0].opType, "Relu");
  await extension.receive({ type: "refresh-shapes" });
  assert.equal(extension.messages.at(-1).type, "shape-info");
  assert.deepEqual(extension.messages.at(-1).tensors[0].shape, ["1", "3"]);
  await extension.open("inference.onnx");
  const graph = extension.messages.at(-1).graph;
  assert.equal(
    graph.tensors.find((tensor) => tensor.name === "hidden").type,
    null,
  );
  const readCount = extension.reads.length;
  await extension.receive({ type: "refresh-shapes" });
  const result = extension.messages.at(-1);
  assert.equal(result.type, "shape-info");
  assert.deepEqual(
    result.tensors.find((tensor) => tensor.name === "hidden"),
    {
      name: "hidden",
      type: "FLOAT",
      shape: ["N", "3"],
    },
  );
  assert.equal(extension.reads.length, readCount);
  await extension.receive({ type: "refresh-shapes" });
  assert.deepEqual(extension.messages.at(-1).tensors, result.tensors);
  assert.deepEqual(extension.errors, []);
  await extension.receive({ type: "ready" });
  assert.equal(
    extension.messages
      .at(-1)
      .graph.tensors.find((tensor) => tensor.name === "hidden").type,
    "FLOAT",
  );
});

test("opens backend tests from the model source picker, preserves cancellation and supports shape refresh", async () => {
  const extension = await openExtension();
  await extension.receive({ type: "ready" });
  const original = extension.messages.at(-1);
  await extension.chooseBackend();
  assert.equal(extension.messages.at(-1), original);
  assert.deepEqual(
    extension.picks[0].items.map(({ label }) => label),
    ["Local file", "Hugging Face", "Backend test"],
  );
  assert.ok(
    extension.logs.some((entry) =>
      entry.includes("[CANCELLED] Open ONNX model"),
    ),
  );
  const { items, options } = extension.picks[1];
  assert.match(options.placeHolder, /Search/);
  assert.equal(options.matchOnDescription, true);
  assert.equal(options.matchOnDetail, true);
  assert.ok(items.some((item) => item.label === "test_cc_relu"));
  assert.ok(
    items.some(
      (item) =>
        item.label === "test_cc_constant_add_chain" &&
        item.description.includes("model"),
    ),
  );
  const reads = extension.reads.length;
  await extension.chooseBackend("test_cc_constant_add_chain");
  assert.equal(
    extension.messages.at(-1).graph.name,
    "test_cc_constant_add_chain",
  );
  assert.deepEqual(
    extension.messages.at(-1).graph.nodes.map(({ opType }) => opType),
    ["Add", "Add"],
  );
  await extension.receive({ type: "refresh-shapes" });
  assert.equal(extension.messages.at(-1).type, "shape-info");
  assert.equal(extension.reads.length, reads);
  assert.deepEqual(extension.errors, []);
});

test("analyzes retained native metadata on demand, preserves it on ready and resets it for another model", async () => {
  const extension = await openExtension("analysis.onnx");
  await extension.receive({ type: "ready" });
  assert.equal(extension.messages.at(-1).metadata, undefined);
  const reads = extension.reads.length;
  await extension.receive({ type: "metadata" });
  const info = extension.messages.at(-1);
  assert.equal(info.type, "metadata-info");
  assert.deepEqual(info.metadata.nodes[5].releaseAfter, [
    "shared;activation",
    "constant_sum",
  ]);
  assert.equal(extension.reads.length, reads);
  await extension.receive({ type: "metadata" });
  assert.equal(extension.messages.at(-1).type, "metadata-info");
  assert.notEqual(extension.messages.at(-1).metadata, info.metadata);
  assert.deepEqual(extension.messages.at(-1).metadata, info.metadata);
  assert.equal(extension.reads.length, reads);
  await extension.receive({ type: "ready" });
  assert.deepEqual(extension.messages.at(-1).metadata, info.metadata);
  await extension.open("relu.onnx");
  assert.equal(extension.messages.at(-1).metadata, undefined);
  await extension.receive({ type: "metadata" });
  assert.deepEqual(extension.messages.at(-1).metadata.model, []);
  assert.deepEqual(
    extension.messages.at(-1).metadata.nodes[0].releaseAfter,
    [],
  );
  assert.deepEqual(extension.errors, []);
});

test("reports a backend model loading failure without replacing the open graph", async () => {
  const extension = await openExtension();
  await extension.receive({ type: "ready" });
  const before = extension.messages.at(-1).graph;
  await extension.chooseBackend("not-a-backend-test");
  assert.match(
    extension.errors[0],
    /Could not open ONNX model:.*Invalid onnx-light backend test ID/,
  );
  await extension.receive({ type: "ready" });
  assert.deepEqual(extension.messages.at(-1).graph, before);
});
test("opens the bundled symbolic transformer through its command and refreshes its intermediate shapes", async () => {
  const extension = await openExtension(
    undefined,
    "onnxLightViewer.openSymbolicExample",
  );
  await extension.receive({ type: "ready" });
  assert.equal(
    extension.messages.at(-1).graph.name,
    "Tiny symbolic transformer",
  );
  assert.ok(extension.reads[0].endsWith("media/tiny-llm.onnx"));
  await extension.receive({ type: "refresh-shapes" });
  assert.deepEqual(
    extension.messages
      .at(-1)
      .tensors.find(({ name }) => name === "feed_forward_expanded").shape,
    ["batch_size", "sequence_length", "32"],
  );
  assert.deepEqual(extension.errors, []);
});

test("reports native inference errors without publishing partial results or changing the current graph", async () => {
  const extension = await openExtension();
  await extension.open("custom-unannotated.onnx");
  const before = extension.messages.at(-1).graph;
  await extension.receive({ type: "refresh-shapes" });
  assert.equal(extension.messages.at(-1).type, "shape-error");
  assert.match(
    extension.messages.at(-1).error,
    /ComputeShapeModel:[\s\S]*unsupported domain.*example/,
  );
  assert.match(extension.errors[0], /Could not infer ONNX shapes\/types/);
  await extension.receive({ type: "ready" });
  assert.deepEqual(extension.messages.at(-1).graph, before);
  const reads = extension.reads.length;
  await extension.receive({ type: "metadata" });
  assert.equal(extension.messages.at(-1).type, "metadata-info");
  assert.equal(extension.reads.length, reads);
  assert.ok(
    extension.logs.some((line) =>
      line.includes("[SUCCEEDED] Reload native model: Custom operators"),
    ),
  );
  await extension.open("inference.onnx");
  await extension.receive({ type: "refresh-shapes" });
  assert.equal(extension.messages.at(-1).type, "shape-info");
});

test("supports native shape refresh when a model is opened directly as a custom editor", async () => {
  const extension = await openExtension("inference.onnx");
  await extension.receive({ type: "ready" });
  assert.equal(extension.messages.at(-1).graph.name, "Shape inference");
  await extension.receive({ type: "refresh-shapes" });
  assert.equal(extension.messages.at(-1).type, "shape-info");
  assert.equal(
    extension.messages.at(-1).tensors.find((tensor) => tensor.name === "hidden")
      .type,
    "FLOAT",
  );
  assert.deepEqual(extension.errors, []);
});

test("keeps only the latest custom editor model when a panel is reused during loading", async () => {
  const extension = await openExtension();
  const release = extension.deferRead("relu.onnx");
  const first = extension.resolveCustomEditor("relu.onnx");
  await extension.receive({ type: "ready" });
  await extension.resolveCustomEditor("inference.onnx");
  await extension.receive({ type: "ready" });
  assert.equal(extension.messages.at(-1).graph.name, "Shape inference");

  const count = extension.events.length;
  release();
  await first;
  assert.equal(extension.events.length, count);
  const messageCount = extension.messages.length;
  await extension.receive({ type: "ready" });
  assert.deepEqual(
    extension.messages.slice(messageCount).map(({ graph }) => graph.name),
    ["Shape inference"],
  );
  await extension.receive({ type: "refresh-shapes" });
  assert.deepEqual(
    extension.messages.at(-1).tensors.find(({ name }) => name === "hidden"),
    { name: "hidden", type: "FLOAT", shape: ["N", "3"] },
  );
  assert.deepEqual(extension.errors, []);
});

test("discards native responses from a previous custom editor session", async () => {
  for (const type of ["navigate", "metadata", "refresh-shapes"]) {
    const extension = await openExtension("analysis.onnx");
    const pending = extension.receive({ type, path: [] });
    const next = extension.resolveCustomEditor("inference.onnx");
    await extension.receive({ type: "ready" });
    await Promise.all([pending, next]);
    const results = extension.messages.filter(({ type }) =>
      ["graph", "metadata-info", "shape-info"].includes(type),
    );
    assert.deepEqual(
      results.map(({ graph }) => graph?.name),
      ["Shape inference"],
    );
    assert.deepEqual(extension.errors, []);
  }
});

test("ignores initial loading failures after a panel is replaced or disposed", async () => {
  for (const dispose of [false, true]) {
    const extension = await openExtension();
    const release = extension.deferRead("missing.onnx");
    const first = extension.resolveCustomEditor("missing.onnx");
    if (dispose) extension.dispose();
    else {
      await extension.resolveCustomEditor("inference.onnx");
      await extension.receive({ type: "ready" });
    }
    const count = extension.events.length;
    release();
    await first;
    assert.equal(extension.events.length, count);
    assert.deepEqual(extension.errors, []);
  }
});
