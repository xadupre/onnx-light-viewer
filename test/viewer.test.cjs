const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { test } = require("node:test");
const { JSDOM } = require("jsdom");
const { demoGraph } = require("../dist/graph");
const { webviewHtml } = require("../dist/webview");

function openViewer(t) {
  const messages = [];
  const dom = new JSDOM(
    webviewHtml(
      (name) => `https://viewer.invalid/${name}`,
      "https://viewer.invalid",
      "test-nonce",
    ),
    { runScripts: "outside-only" },
  );
  t.after(() => dom.window.close());
  dom.window.acquireVsCodeApi = () => ({
    postMessage: (message) => messages.push(message),
  });
  dom.window.eval(readFileSync(require.resolve("../media/viewer.js"), "utf8"));
  const load = (graph, isDemo = false) => {
    dom.window.dispatchEvent(
      new dom.window.MessageEvent("message", {
        data: { type: "graph", graph, isDemo },
      }),
    );
  };
  return {
    document: dom.window.document,
    window: dom.window,
    messages,
    load,
  };
}

test("renders the demo, inspects its operator and requests a real file", (t) => {
  const { document, messages, load } = openViewer(t);
  assert.equal(messages[0].type, "ready");
  load(demoGraph, true);
  assert.equal(document.querySelectorAll(".graph-node").length, 1);
  assert.equal(document.querySelectorAll(".port").length, 2);
  assert.equal(document.querySelectorAll(".edge").length, 2);
  assert.equal(document.getElementById("node-title").textContent, "Relu");
  assert.equal(document.getElementById("node-inputs").textContent, "input");
  assert.equal(
    document.querySelector(".graph-node").getAttribute("aria-pressed"),
    "true",
  );
  document.getElementById("open-model").click();
  document.getElementById("open-another").click();
  assert.equal(messages.filter((message) => message.type === "open").length, 2);
  const graph = document.getElementById("graph");
  const initialViewBox = graph.getAttribute("viewBox");
  document.getElementById("zoom-in").click();
  assert.notEqual(graph.getAttribute("viewBox"), initialViewBox);
  document.getElementById("zoom-fit").click();
  assert.equal(graph.getAttribute("viewBox"), initialViewBox);
});

test("loads a connected graph, selects with the keyboard and treats model text as text", (t) => {
  const { document, window, load } = openViewer(t);
  const unsafeName = '<img src=x onerror="alert(1)">';
  load({
    ...demoGraph,
    name: unsafeName,
    nodes: [
      { ...demoGraph.nodes[0], outputs: ["hidden"] },
      {
        id: "node-1",
        name: unsafeName,
        opType: "Sigmoid",
        domain: "ai.onnx",
        inputs: ["hidden"],
        outputs: ["output"],
      },
    ],
  });
  assert.equal(document.getElementById("graph-name").textContent, unsafeName);
  assert.equal(document.getElementById("graph-name").children.length, 0);
  assert.equal(document.querySelectorAll(".edge").length, 3);
  assert.equal(
    document.getElementById("model-badge").textContent,
    "ONNX MODEL",
  );
  const secondNode = document.querySelectorAll(".graph-node")[1];
  secondNode.dispatchEvent(
    new window.KeyboardEvent("keydown", { key: "Enter" }),
  );
  assert.equal(document.getElementById("node-title").textContent, "Sigmoid");
  assert.equal(document.getElementById("node-name").textContent, unsafeName);
  assert.equal(document.getElementById("node-name").children.length, 0);
  assert.equal(secondNode.getAttribute("aria-pressed"), "true");
  assert.equal(document.getElementById("node-inputs").textContent, "hidden");
});

test("replaces the demo with a no-operator graph without stale nodes or inspector values", (t) => {
  const { document, load } = openViewer(t);
  load(demoGraph, true);
  load({
    name: "passthrough",
    nodes: [],
    inputs: ["x"],
    outputs: ["x"],
    initializerCount: 0,
  });
  assert.equal(document.querySelectorAll(".graph-node").length, 0);
  assert.equal(document.querySelectorAll(".edge").length, 1);
  assert.equal(
    document.getElementById("node-title").textContent,
    "No operator",
  );
  assert.equal(document.getElementById("node-inputs").children.length, 0);
  assert.equal(document.getElementById("node-name").textContent, "—");
});
