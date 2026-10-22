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
  let viewport = { width: 1060, height: 350 };
  dom.window.document.getElementById("canvas").getBoundingClientRect = () =>
    new dom.window.DOMRect(0, 0, viewport.width, viewport.height);
  const resize = (width, height) => {
    viewport = { width, height };
    dom.window.dispatchEvent(new dom.window.Event("resize"));
  };
  dom.window.eval(readFileSync(require.resolve("../media/viewer.js"), "utf8"));
  const load = (graph) => {
    dom.window.dispatchEvent(
      new dom.window.MessageEvent("message", {
        data: { type: "graph", graph },
      }),
    );
  };
  return {
    document: dom.window.document,
    window: dom.window,
    messages,
    load,
    resize,
  };
}

test("renders the demo, inspects its operator and requests a real file", (t) => {
  const { document, messages, load } = openViewer(t);
  assert.equal(messages[0].type, "ready");
  load(demoGraph);
  assert.doesNotMatch(
    document.body.textContent,
    /A LITTLE CLARITY|Your model, connected|Explore the graph|A first ONNX graph|ONNX MODEL/,
  );
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
  assert.equal(messages.filter((message) => message.type === "open").length, 1);
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
  assert.equal(document.querySelectorAll(".edge").length, 3);
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
  load(demoGraph);
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

test("renders a portrait graph with inputs above operators and outputs below", (t) => {
  const { document, load } = openViewer(t);
  load(demoGraph);
  const box = (selector) => {
    const rect = document.querySelector(selector);
    return Object.fromEntries(
      ["x", "y", "width", "height"].map((key) => [
        key,
        Number(rect.getAttribute(key)),
      ]),
    );
  };
  const input = box(".port:not(.output) rect");
  const operator = box(".graph-node rect");
  const output = box(".port.output rect");
  assert.ok(input.y + input.height < operator.y);
  assert.ok(operator.y + operator.height < output.y);
  const boxes = [input, operator, output];
  const contentWidth =
    Math.max(...boxes.map((rect) => rect.x + rect.width)) -
    Math.min(...boxes.map((rect) => rect.x));
  assert.ok(output.y + output.height - input.y > contentWidth);
  const edges = document.querySelectorAll(".edge");
  for (const [index, from, to] of [
    [0, input, operator],
    [1, operator, output],
  ]) {
    const points = edges[index]
      .getAttribute("d")
      .match(/-?\d+(?:\.\d+)?(?:e[+-]?\d+)?/gi)
      .map(Number);
    assert.deepEqual(points.slice(0, 2), [
      from.x + from.width / 2,
      from.y + from.height,
    ]);
    assert.deepEqual(points.slice(6), [to.x + to.width / 2, to.y]);
    assert.equal(points[2], points[0]);
    assert.equal(points[4], points[6]);
    assert.equal(points[3], points[5]);
  }
});

test("spreads multiple graph ports across top and bottom rows without overlap", (t) => {
  const { document, load, resize } = openViewer(t);
  const inputs = Array.from({ length: 8 }, (_, index) => `input_${index}`);
  load({
    name: "Multiple graph ports",
    nodes: [],
    inputs,
    outputs: inputs.slice(0, 5),
    initializerCount: 0,
  });
  resize(360, 640);
  const [x, y, width, height] = document
    .getElementById("graph")
    .getAttribute("viewBox")
    .split(" ")
    .map(Number);
  for (const selector of [".port:not(.output) rect", ".port.output rect"]) {
    const ports = Array.from(document.querySelectorAll(selector));
    let previousRight = -Infinity;
    for (const port of ports) {
      const left = Number(port.getAttribute("x"));
      const top = Number(port.getAttribute("y"));
      const right = left + Number(port.getAttribute("width"));
      const bottom = top + Number(port.getAttribute("height"));
      assert.equal(top, Number(ports[0].getAttribute("y")));
      assert.ok(left > previousRight);
      assert.ok(left >= x && right <= x + width);
      assert.ok(top >= y && bottom <= y + height);
      previousRight = right;
    }
  }
  assert.equal(document.querySelectorAll(".edge").length, 5);
});

test("assigns random horizontal positions and node-index rows on each model load", (t) => {
  const { document, load } = openViewer(t);
  const model = chainGraph();
  load(model);
  const positions = nodePositions(document);
  positions.forEach((position, index) => {
    assert.equal(position.y, 180 + index * 145);
    assert.ok(position.x >= 100 && position.x < 400);
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

test("fits resized viewports without resetting zoom or rearranging nodes", (t) => {
  const { document, load, resize } = openViewer(t);
  load(chainGraph());
  const positions = nodePositions(document);
  const graph = document.getElementById("graph");
  const viewBox = () => graph.getAttribute("viewBox").split(" ").map(Number);
  document.getElementById("zoom-in").click();
  for (const [width, height] of [
    [1920, 1080],
    [360, 640],
    [1200, 300],
  ]) {
    resize(width, height);
    const zoomed = viewBox();
    assert.ok(Math.abs(zoomed[2] / zoomed[3] - width / height) < 1e-10);
    document.getElementById("zoom-fit").click();
    assert.ok(Math.abs(viewBox()[2] - zoomed[2] * 1.25) < 1e-10);
    document.getElementById("zoom-in").click();
    assert.deepEqual(nodePositions(document), positions);
  }
  const beforeHide = graph.getAttribute("viewBox");
  resize(0, 0);
  assert.equal(graph.getAttribute("viewBox"), beforeHide);
  resize(800, 600);
  assert.ok(viewBox().every(Number.isFinite));
  assert.ok(Math.abs(viewBox()[2] / viewBox()[3] - 800 / 600) < 1e-10);
  assert.deepEqual(nodePositions(document), positions);
});
