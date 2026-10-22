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
  const graph = dom.window.document.getElementById("graph");
  const captures = new Set();
  graph.setPointerCapture = (id) => captures.add(id);
  graph.hasPointerCapture = (id) => captures.has(id);
  graph.releasePointerCapture = (id) => {
    captures.delete(id);
    const event = new dom.window.Event("lostpointercapture");
    Object.defineProperty(event, "pointerId", { value: id });
    graph.dispatchEvent(event);
  };
  const pointer = (target, type, x, y, modifiers = {}) => {
    const [left, top, width, height] = graph
      .getAttribute("viewBox")
      .split(" ")
      .map(Number);
    const event = new dom.window.MouseEvent(type, {
      bubbles: true,
      cancelable: true,
      clientX: ((x - left) * viewport.width) / width,
      clientY: ((y - top) * viewport.height) / height,
      ...modifiers,
    });
    Object.defineProperty(event, "pointerId", {
      value: modifiers.pointerId ?? 1,
    });
    target.dispatchEvent(event);
  };
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
    pointer,
  };
}

test("renders the demo, inspects its operator and requests a real file", (t) => {
  const { document, messages, load, window } = openViewer(t);
  assert.equal(messages[0].type, "ready");
  const errors = [];
  window.addEventListener("error", (event) => errors.push(event.error));
  document
    .getElementById("graph")
    .dispatchEvent(
      new window.MouseEvent("pointerdown", { bubbles: true, button: 0 }),
    );
  assert.deepEqual(errors, []);
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

function nodePositions(document, selector = ".graph-node rect") {
  return Array.from(document.querySelectorAll(selector), (rect) => {
    const [dx, dy] = rect.parentElement
      .getAttribute("transform")
      ?.match(/-?\d+(?:\.\d+)?(?:e[+-]?\d+)?/gi)
      .map(Number) ?? [0, 0];
    return {
      x:
        Number(rect.getAttribute("x")) +
        Number(rect.getAttribute("width")) / 2 +
        dx,
      y:
        Number(rect.getAttribute("y")) +
        Number(rect.getAttribute("height")) / 2 +
        dy,
    };
  });
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

function assertMoved(before, after, indices, dx, dy) {
  before.forEach((position, index) => {
    assert.ok(
      Math.abs(
        after[index].x - position.x - (indices.includes(index) ? dx : 0),
      ) < 1e-8,
    );
    assert.ok(
      Math.abs(
        after[index].y - position.y - (indices.includes(index) ? dy : 0),
      ) < 1e-8,
    );
  });
}

test("drags a zoomed operator and updates its connections without moving other nodes", (t) => {
  const { document, load, resize, pointer } = openViewer(t);
  load(chainGraph());
  resize(1024, 700);
  document.getElementById("zoom-in").click();
  const graph = document.getElementById("graph");
  const before = nodePositions(document);
  const nodes = document.querySelectorAll(".graph-node");
  const edgePaths = () =>
    Array.from(document.querySelectorAll(".edge"), (edge) =>
      edge.getAttribute("d"),
    );
  const edgesBefore = edgePaths();
  const labelsBefore = Array.from(
    document.querySelectorAll(".tensor-label"),
    (text) => text.getAttribute("y"),
  );
  const start = before[2];
  pointer(nodes[2], "pointerdown", start.x, start.y);
  assert.equal(graph.hasPointerCapture(1), true);
  pointer(graph, "pointermove", start.x + 100, start.y, { pointerId: 2 });
  assert.deepEqual(nodePositions(document), before);
  pointer(graph, "pointermove", start.x + 40, start.y + 70);
  assertMoved(before, nodePositions(document), [2], 40, 70);
  pointer(graph, "pointerup", start.x + 40, start.y + 70);
  assert.equal(graph.hasPointerCapture(1), false);
  assert.equal(nodes[2].getAttribute("aria-pressed"), "true");
  assert.equal(nodes[0].getAttribute("aria-pressed"), "false");
  edgePaths().forEach((path, index) => {
    if (index === 2 || index === 3) {
      assert.notEqual(path, edgesBefore[index]);
      assert.notEqual(
        document.querySelectorAll(".tensor-label")[index].getAttribute("y"),
        labelsBefore[index],
      );
    } else {
      assert.equal(path, edgesBefore[index]);
    }
  });
  pointer(graph, "pointermove", start.x + 200, start.y + 200);
  assertMoved(before, nodePositions(document), [2], 40, 70);
});

test("modifier clicks toggle selection and dragging a selected operator moves the whole selection", (t) => {
  const { document, load, pointer, window } = openViewer(t);
  load(chainGraph());
  const graph = document.getElementById("graph");
  const nodes = document.querySelectorAll(".graph-node");
  const before = nodePositions(document);
  for (const [index, modifiers] of [
    [1, { ctrlKey: true }],
    [2, { shiftKey: true }],
    [1, { metaKey: true }],
  ]) {
    const point = before[index];
    pointer(nodes[index], "pointerdown", point.x, point.y, modifiers);
    pointer(graph, "pointerup", point.x, point.y, modifiers);
    nodes[index].dispatchEvent(
      new window.MouseEvent("click", {
        bubbles: true,
        detail: 1,
        ...modifiers,
      }),
    );
  }
  assert.deepEqual(
    Array.from(nodes, (node) => node.getAttribute("aria-pressed")),
    ["true", "false", "true", "false", "false", "false"],
  );
  pointer(nodes[0], "pointerdown", before[0].x, before[0].y);
  pointer(graph, "pointermove", before[0].x - 35, before[0].y + 45);
  pointer(graph, "pointerup", before[0].x - 35, before[0].y + 45);
  assertMoved(before, nodePositions(document), [0, 2], -35, 45);
});

test("rectangle selection supports reverse and additive gestures and moves together", (t) => {
  const { document, load, pointer, window } = openViewer(t);
  load(chainGraph());
  const graph = document.getElementById("graph");
  pointer(graph, "pointerdown", 500, 400);
  pointer(graph, "pointermove", -10, 110);
  assert.equal(document.querySelectorAll(".graph-node.selected").length, 2);
  assert.ok(document.querySelector(".selection-rectangle"));
  pointer(graph, "pointerup", -10, 110);
  assert.equal(document.querySelector(".selection-rectangle"), null);
  pointer(graph, "pointerdown", -10, 700, { ctrlKey: true });
  pointer(graph, "pointermove", 500, 820, { ctrlKey: true });
  pointer(graph, "pointerup", 500, 820, { ctrlKey: true });
  assert.deepEqual(
    Array.from(
      document.querySelectorAll(".graph-node.selected"),
      (node) => node.dataset.id,
    ),
    ["node-0", "node-1", "node-4"],
  );
  const before = nodePositions(document);
  pointer(
    document.querySelector('[data-id="node-1"]'),
    "pointerdown",
    before[1].x,
    before[1].y,
  );
  pointer(graph, "pointerup", before[1].x + 30, before[1].y - 25);
  assertMoved(before, nodePositions(document), [0, 1, 4], 30, -25);
  pointer(graph, "pointerdown", 0, 0);
  pointer(graph, "pointerup", 0, 0);
  assert.equal(document.querySelectorAll(".graph-node.selected").length, 0);
  graph.dispatchEvent(
    new window.KeyboardEvent("keydown", { key: "a", ctrlKey: true }),
  );
  assert.equal(document.querySelectorAll(".graph-node.selected").length, 6);
  graph.dispatchEvent(new window.KeyboardEvent("keydown", { key: "Escape" }));
  assert.equal(document.querySelectorAll(".graph-node.selected").length, 0);
});

test("cancels interrupted drags and rectangles and drops state on model replacement", (t) => {
  const { document, load, pointer, window } = openViewer(t);
  load(chainGraph());
  const graph = document.getElementById("graph");
  const before = nodePositions(document);
  for (const cancel of [
    () => pointer(graph, "pointercancel", 0, 0),
    () =>
      graph.dispatchEvent(
        new window.KeyboardEvent("keydown", { key: "Escape" }),
      ),
    () => window.dispatchEvent(new window.Event("blur")),
  ]) {
    pointer(
      document.querySelector(".graph-node"),
      "pointerdown",
      before[0].x,
      before[0].y,
    );
    pointer(graph, "pointermove", before[0].x + 100, before[0].y + 100);
    cancel();
    assertMoved(before, nodePositions(document), [], 0, 0);
    assert.equal(graph.hasPointerCapture(1), false);
    assert.equal(graph.classList.contains("dragging"), false);
  }
  pointer(graph, "pointerdown", 500, 100);
  pointer(graph, "pointermove", -10, 900);
  graph.dispatchEvent(new window.KeyboardEvent("keydown", { key: "Escape" }));
  assert.equal(document.querySelectorAll(".graph-node.selected").length, 1);
  assert.equal(document.querySelector(".selection-rectangle"), null);
  pointer(graph, "pointerdown", 500, 100);
  pointer(graph, "pointermove", -10, 900);
  load(demoGraph);
  assert.equal(graph.hasPointerCapture(1), false);
  assert.equal(document.querySelector(".selection-rectangle"), null);
  assert.equal(document.querySelectorAll(".graph-node.selected").length, 1);
  assert.equal(document.querySelectorAll(".graph-node").length, 1);
});

test("Fit includes operators dragged beyond the original bounds without snapping during drag", (t) => {
  const { document, load, pointer } = openViewer(t);
  load(demoGraph);
  const graph = document.getElementById("graph");
  const viewBoxBefore = graph.getAttribute("viewBox");
  const start = nodePositions(document)[0];
  pointer(
    document.querySelector(".graph-node"),
    "pointerdown",
    start.x,
    start.y,
  );
  pointer(graph, "pointermove", -800, -900);
  pointer(graph, "pointerup", -800, -900);
  assert.equal(graph.getAttribute("viewBox"), viewBoxBefore);
  document.getElementById("zoom-fit").click();
  const [x, y, width, height] = graph
    .getAttribute("viewBox")
    .split(" ")
    .map(Number);
  const moved = nodePositions(document)[0];
  assert.ok(moved.x - 80 >= x && moved.x + 80 <= x + width);
  assert.ok(moved.y - 47 >= y && moved.y + 47 <= y + height);
  assert.notEqual(graph.getAttribute("viewBox"), viewBoxBefore);
});

test("drags graph inputs and outputs independently at zoom and updates their edges", (t) => {
  const { document, load, pointer } = openViewer(t);
  load(demoGraph);
  document.getElementById("zoom-in").click();
  const graph = document.getElementById("graph");
  for (const [selector, index, title, name, edgeIndex] of [
    [".port:not(.output)", 0, "Graph input", "input", 0],
    [".port.output", 2, "Graph output", "output", 1],
  ]) {
    const before = nodePositions(document, "#nodes rect");
    const paths = Array.from(document.querySelectorAll(".edge"), (edge) =>
      edge.getAttribute("d"),
    );
    const start = before[index];
    pointer(document.querySelector(selector), "pointerdown", start.x, start.y);
    pointer(graph, "pointermove", start.x + 45, start.y - 30);
    assertMoved(
      before,
      nodePositions(document, "#nodes rect"),
      [index],
      45,
      -30,
    );
    assert.notEqual(
      document.querySelectorAll(".edge")[edgeIndex].getAttribute("d"),
      paths[edgeIndex],
    );
    assert.equal(
      document.querySelectorAll(".edge")[1 - edgeIndex].getAttribute("d"),
      paths[1 - edgeIndex],
    );
    pointer(graph, "pointerup", start.x + 45, start.y - 30);
    assert.equal(
      document.querySelector(selector).getAttribute("aria-pressed"),
      "true",
    );
    assert.equal(document.getElementById("node-title").textContent, title);
    assert.equal(document.getElementById("node-name").textContent, name);
    assert.equal(document.getElementById("node-domain").textContent, "—");
  }
});

test("rectangle and modifier selections mix ports with operators and move as one group", (t) => {
  const { document, load, pointer, window } = openViewer(t);
  load(demoGraph);
  const graph = document.getElementById("graph");
  const before = nodePositions(document, "#nodes rect");
  const input = before[0];
  pointer(graph, "pointerdown", input.x + 75, input.y - 5);
  pointer(graph, "pointerup", input.x + 61, input.y + 5);
  assert.equal(document.querySelectorAll("#nodes .selected").length, 0);
  pointer(graph, "pointerdown", input.x + 75, input.y - 5);
  pointer(graph, "pointerup", input.x + 59, input.y + 5);
  assert.equal(
    document.querySelector(".port:not(.output)").getAttribute("aria-pressed"),
    "true",
  );
  for (const [index, selector, key] of [
    [1, ".graph-node", "ctrlKey"],
    [2, ".port.output", "shiftKey"],
  ]) {
    pointer(
      document.querySelector(selector),
      "pointerdown",
      before[index].x,
      before[index].y,
      { [key]: true },
    );
    pointer(graph, "pointerup", before[index].x, before[index].y, {
      [key]: true,
    });
  }
  assert.equal(document.querySelectorAll("#nodes .selected").length, 3);
  pointer(
    document.querySelector(".port.output"),
    "pointerdown",
    before[2].x,
    before[2].y,
  );
  pointer(graph, "pointerup", before[2].x - 40, before[2].y + 25);
  assertMoved(
    before,
    nodePositions(document, "#nodes rect"),
    [0, 1, 2],
    -40,
    25,
  );
  graph.dispatchEvent(new window.KeyboardEvent("keydown", { key: "Escape" }));
  graph.dispatchEvent(
    new window.KeyboardEvent("keydown", { key: "a", ctrlKey: true }),
  );
  assert.equal(document.querySelectorAll("#nodes .selected").length, 3);
  document.querySelector(".port.output").dispatchEvent(
    new window.KeyboardEvent("keydown", {
      key: " ",
      shiftKey: true,
      bubbles: true,
    }),
  );
  assert.equal(document.querySelectorAll("#nodes .selected").length, 2);
});

test("same-name input/output nodes remain distinct and Fit includes moved ports without operators", (t) => {
  const { document, load, pointer, window } = openViewer(t);
  load({
    name: "passthrough",
    nodes: [],
    inputs: ["x"],
    outputs: ["x"],
    initializerCount: 0,
  });
  const graph = document.getElementById("graph");
  const before = nodePositions(document, ".port rect");
  const input = document.querySelector(".port:not(.output)");
  const output = document.querySelector(".port.output");
  assert.notEqual(input.dataset.id, output.dataset.id);
  graph.dispatchEvent(
    new window.KeyboardEvent("keydown", { key: "a", metaKey: true }),
  );
  assert.equal(document.querySelectorAll(".port.selected").length, 2);
  pointer(input, "pointerdown", before[0].x, before[0].y);
  pointer(graph, "pointerup", before[0].x - 1000, before[0].y - 1000);
  const after = nodePositions(document, ".port rect");
  assertMoved(before, after, [0, 1], -1000, -1000);
  const endpoints = document
    .querySelector(".edge")
    .getAttribute("d")
    .match(/-?\d+(?:\.\d+)?(?:e[+-]?\d+)?/gi)
    .map(Number);
  assert.ok(Math.abs(endpoints[0] - after[0].x) < 1e-8);
  assert.ok(Math.abs(endpoints[1] - after[0].y - 26) < 1e-8);
  assert.ok(Math.abs(endpoints[6] - after[1].x) < 1e-8);
  assert.ok(Math.abs(endpoints[7] - after[1].y + 26) < 1e-8);
  document.getElementById("zoom-fit").click();
  const [x, y, width, height] = graph
    .getAttribute("viewBox")
    .split(" ")
    .map(Number);
  for (const port of after) {
    assert.ok(port.x - 60 >= x && port.x + 60 <= x + width);
    assert.ok(port.y - 26 >= y && port.y + 26 <= y + height);
  }
  load(demoGraph);
  assert.equal(document.querySelectorAll(".port.selected").length, 0);
  assert.equal(document.querySelectorAll(".graph-node.selected").length, 1);
});
