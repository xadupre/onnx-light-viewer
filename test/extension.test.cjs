const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { readFile } = require("node:fs/promises");
const { createRequire } = require("node:module");
const path = require("node:path");
const vm = require("node:vm");
const { test } = require("node:test");
const { JSDOM } = require("jsdom");

async function openExtension(
  customModel,
  command = "onnxLightViewer.open",
  localFiles = false,
) {
  const messages = [];
  const errors = [];
  const reads = [];
  const picks = [];
  const logs = [];
  const logViews = [];
  const events = [];
  const nativeModels = [];
  const commands = new Map();
  let receive;
  let selection;
  let editor;
  let backendSelection;
  let client;
  let fileLoadGate;
  const disposable = () => ({ dispose() {} });
  const root = path.resolve(__dirname, "..");
  const uri = (filePath) => ({
    path: filePath,
    fsPath: filePath,
    scheme: localFiles ? "file" : "virtual",
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
        !["log-history", "application-status"].includes(message.type)
      ) {
        messages.push(message);
      }
      return true;
    },
    onDidReceiveMessage: (callback) => {
      receive = callback;
      return disposable();
    },
    asWebviewUri: (uri) => ({ toString: () => `file://${uri.path}` }),
  };
  const panel = {
    webview,
    onDidDispose: disposable,
    onDidChangeViewState: disposable,
    reveal() {},
  };
  const vscode = {
    Uri: {
      joinPath: (base, ...parts) => uri(path.posix.join(base.path, ...parts)),
    },
    ViewColumn: { One: 1 },
    commands: {
      registerCommand: (name, callback) => {
        commands.set(name, callback);
        return disposable();
      },
    },
    workspace: {
      fs: {
        readFile: async (uri) => {
          reads.push(uri.path);
          return readFile(uri.path);
        },
      },
    },
    window: {
      createWebviewPanel: () => panel,
      registerCustomEditorProvider: (_viewType, provider) => {
        editor = provider;
        return disposable();
      },
      showOpenDialog: async () => selection,
      showQuickPick: async (items, options) => {
        picks.push({ items, options });
        return backendSelection === undefined
          ? undefined
          : {
              label: backendSelection,
              id:
                items.find((item) => item.label === backendSelection)?.id ?? -1,
            };
      },
      showErrorMessage: async (message) => errors.push(message),
    },
  };
  const filename = path.join(root, "dist", "extension.js");
  const requireExtension = createRequire(filename);
  const graph = { ...requireExtension("./graph") };
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
    const gate = fileLoadGate;
    fileLoadGate = undefined;
    const native = await loadFile(...args);
    if (gate) await gate;
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
          : requireExtension(name),
    module,
    module.exports,
  );
  module.exports.activate({ extensionUri: { path: root }, subscriptions: [] });
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
    webview,
    connect: (callback) => {
      client = callback;
    },
    nativeModels,
    pauseNextFileLoad: () => {
      let resume;
      fileLoadGate = new Promise((resolve) => {
        resume = resolve;
      });
      return resume;
    },
    showLogs: () => commands.get("onnxLightViewer.showLogs")(),
    receive: (message) => receive(message),
    open: async (name) => {
      selection = [uri(path.join(__dirname, "models", name))];
      await receive({ type: "open" });
    },
    chooseBackend: async (name) => {
      backendSelection = name;
      await receive({ type: "open-backend-test" });
    },
  };
}

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

test("ignores stale recovery results and errors after replacing a local model", async () => {
  for (const type of ["navigate", "metadata", "refresh-shapes"]) {
    const extension = await openExtension("inference.onnx", undefined, true);
    for (const path of [[], [{ node: 9999, attribute: 0, graph: 0 }]]) {
      await extension.open("invalid-graph.onnx");
      await extension.receive({ type: "metadata" });
      const resume = extension.pauseNextFileLoad();
      const recovery = extension.receive({ type, path });
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

test("keeps operation logs quiet and exposes real native exception recovery in Status", async (t) => {
  const extension = await openExtension();
  const dom = new JSDOM(extension.webview.html, { runScripts: "outside-only" });
  t.after(() => dom.window.close());
  const { window } = dom;
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
  window.eval(
    readFileSync(path.resolve(__dirname, "../media/viewer.js"), "utf8"),
  );
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

  await extension.open("custom.onnx");
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

test("opens a searchable native backend selector, preserves the model on cancellation and supports shape refresh", async () => {
  const extension = await openExtension();
  await extension.receive({ type: "ready" });
  const original = extension.messages.at(-1);
  await extension.chooseBackend();
  assert.equal(extension.messages.at(-1), original);
  const { items, options } = extension.picks[0];
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
    /Could not open onnx-light backend test:.*Invalid onnx-light backend test ID/,
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
  await extension.open("custom.onnx");
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
