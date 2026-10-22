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

function chainGraph() {
  return {
    ...demoGraph,
    nodes: Array.from({ length: 6 }, (_, index) => ({
      ...demoGraph.nodes[0],
      id: `node-${index}`,
      name: `activation_${index}`,
      inputs: [index === 0 ? "input" : `tensor_${index - 1}`],
      outputs: [index === 5 ? "output" : `tensor_${index}`],
    })),
  };
}

function nodePositions(document) {
  return Array.from(document.querySelectorAll(".graph-node rect"), (rect) => ({
    x: Number(rect.getAttribute("x")) + Number(rect.getAttribute("width")) / 2,
    y: Number(rect.getAttribute("y")) + Number(rect.getAttribute("height")) / 2,
  }));
}

test("assigns random horizontal positions and node-index rows on each model load", (t) => {
  const { document, load } = openViewer(t);
  const model = chainGraph();
  load(model);
  const positions = nodePositions(document);
  positions.forEach((position, index) => {
    assert.equal(position.y, 110 + index * 145);
    assert.ok(position.x >= 300 && position.x < 760);
  });
  assert.ok(new Set(positions.map((position) => position.x)).size > 1);
  assert.equal(document.querySelectorAll(".edge").length, 7);
  const bounds = document
    .getElementById("graph")
    .getAttribute("viewBox")
    .split(" ")
    .map(Number);
  for (const rect of document.querySelectorAll("#nodes rect")) {
    const x = Number(rect.getAttribute("x"));
    const y = Number(rect.getAttribute("y"));
    assert.ok(x >= bounds[0] && y >= bounds[1]);
    assert.ok(x + Number(rect.getAttribute("width")) <= bounds[0] + bounds[2]);
    assert.ok(y + Number(rect.getAttribute("height")) <= bounds[1] + bounds[3]);
  }
  load(model);
  assert.notDeepEqual(
    nodePositions(document).map((position) => position.x),
    positions.map((position) => position.x),
  );
  assert.deepEqual(
    nodePositions(document).map((position) => position.y),
    positions.map((position) => position.y),
  );
});

test("zooms in and out with buttons and the wheel without rearranging the graph", (t) => {
  const { document, window, load } = openViewer(t);
  load(chainGraph());
  const positions = nodePositions(document);
  const graph = document.getElementById("graph");
  const initialViewBox = graph.getAttribute("viewBox");
  const viewWidth = () => Number(graph.getAttribute("viewBox").split(" ")[2]);
  const initialWidth = viewWidth();
  document.getElementById("zoom-in").click();
  assert.ok(viewWidth() < initialWidth);
  document.getElementById("zoom-out").click();
  assert.equal(graph.getAttribute("viewBox"), initialViewBox);
  const wheel = (deltaY) => {
    const event = new window.WheelEvent("wheel", { deltaY, cancelable: true });
    graph.dispatchEvent(event);
    return event;
  };
  assert.equal(wheel(-100).defaultPrevented, true);
  assert.ok(viewWidth() < initialWidth);
  wheel(100);
  assert.equal(graph.getAttribute("viewBox"), initialViewBox);
  wheel(100);
  assert.ok(viewWidth() > initialWidth);
  const zoomedOut = graph.getAttribute("viewBox");
  wheel(0);
  assert.equal(graph.getAttribute("viewBox"), zoomedOut);
  document.getElementById("zoom-fit").click();
  assert.equal(graph.getAttribute("viewBox"), initialViewBox);
  assert.deepEqual(nodePositions(document), positions);
});
