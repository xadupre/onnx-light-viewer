const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { readFile } = require("node:fs/promises");
const { createRequire } = require("node:module");
const path = require("node:path");
const vm = require("node:vm");
const { test } = require("node:test");

async function openExtension(customModel, command = "onnxLightViewer.open") {
  const messages = [];
  const errors = [];
  const reads = [];
  const picks = [];
  const commands = new Map();
  let receive;
  let selection;
  let editor;
  let backendSelection;
  const disposable = () => ({ dispose() {} });
  const root = path.resolve(__dirname, "..");
  const webview = {
    postMessage: async (message) => {
      messages.push(message);
      return true;
    },
    onDidReceiveMessage: (callback) => {
      receive = callback;
      return disposable();
    },
    asWebviewUri: (uri) => ({ toString: () => `file://${uri.path}` }),
  };
  const panel = { webview, onDidDispose: disposable };
  const vscode = {
    Uri: {
      joinPath: (base, ...parts) => ({
        path: path.posix.join(base.path, ...parts),
      }),
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
  const module = { exports: {} };
  const execute = vm.runInThisContext(
    `(function(require, module, exports) { ${readFileSync(filename, "utf8")}\n})`,
    { filename },
  );
  execute(
    (name) => (name === "vscode" ? vscode : requireExtension(name)),
    module,
    module.exports,
  );
  module.exports.activate({ extensionUri: { path: root }, subscriptions: [] });
  if (customModel) {
    await editor.resolveCustomEditor(
      { uri: { path: path.join(__dirname, "models", customModel) } },
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
    receive: (message) => receive(message),
    open: async (name) => {
      selection = [{ path: path.join(__dirname, "models", name) }];
      await receive({ type: "open" });
    },
    chooseBackend: async (name) => {
      backendSelection = name;
      await receive({ type: "open-backend-test" });
    },
  };
}

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
  assert.match(extension.messages.at(-1).error, /unsupported domain.*example/);
  assert.match(extension.errors[0], /Could not infer ONNX shapes\/types/);
  await extension.receive({ type: "ready" });
  assert.deepEqual(extension.messages.at(-1).graph, before);
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
