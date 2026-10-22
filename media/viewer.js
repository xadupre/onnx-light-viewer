/* global acquireVsCodeApi */
const vscode = acquireVsCodeApi();
const svgNS = "http://www.w3.org/2000/svg";
const byId = (id) => document.getElementById(id);
const svg = byId("graph");
let bounds = { x: 0, y: 0, width: 800, height: 350 };
let layoutBounds = bounds;
let zoom = 1;
let nodeViews = new Map();
let selected = new Set();
let connections = [];
let gesture;

function element(tag, attributes, text) {
  const node = document.createElementNS(svgNS, tag);
  for (const [name, value] of Object.entries(attributes)) {
    node.setAttribute(name, String(value));
  }
  if (text !== undefined) node.textContent = text;
  return node;
}

function label(parent, x, y, text, className, anchor = "middle") {
  const textNode = element(
    "text",
    { x, y, class: className, "text-anchor": anchor },
    text,
  );
  parent.append(textNode);
  return textNode;
}

function shortName(value, length = 22) {
  return value.length > length ? `${value.slice(0, length - 1)}…` : value;
}

function inspect(view) {
  const node = view?.node;
  const port = view?.port;
  const hasOperators = [...nodeViews.values()].some((view) => view.node);
  byId("node-title").textContent =
    node?.opType ??
    (port
      ? port.output
        ? "Graph output"
        : "Graph input"
      : hasOperators
        ? "No selection"
        : "No operator");
  byId("node-description").textContent = node
    ? node.opType === "Relu" && node.domain === "ai.onnx"
      ? "Rectified linear activation: max(0, x)."
      : `${node.opType} operator`
    : port
      ? `Graph ${port.output ? "output" : "input"} tensor.`
      : hasOperators
        ? "Select a node to inspect its inputs and outputs."
        : "This graph contains no operator nodes.";
  byId("node-name").textContent = node?.name ?? port?.name ?? "—";
  byId("node-domain").textContent = node?.domain ?? "—";
  for (const direction of ["inputs", "outputs"]) {
    const names =
      node?.[direction] ??
      (port && direction === (port.output ? "inputs" : "outputs")
        ? [port.name]
        : []);
    byId(`${direction === "inputs" ? "input" : "output"}-count`).textContent =
      names.length;
    byId(`node-${direction}`).replaceChildren(
      ...names.map((name) => {
        const item = document.createElement("li");
        item.textContent = name;
        return item;
      }),
    );
  }
}

function updateSelection(activeId) {
  for (const [id, { group }] of nodeViews) {
    group.classList.toggle("selected", selected.has(id));
    group.setAttribute("aria-pressed", String(selected.has(id)));
  }
  inspect(nodeViews.get(activeId ?? [...selected].at(-1)));
}

function registerNode(id, view) {
  view.initial = { ...view.position };
  nodeViews.set(id, view);
  const { group } = view;
  group.setAttribute("tabindex", "0");
  group.setAttribute("role", "button");
  group.setAttribute("data-id", id);
  group.addEventListener("click", (event) => {
    if (event.detail === 0) {
      selectNode(id, event.ctrlKey || event.metaKey || event.shiftKey);
    }
  });
  group.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      selectNode(id, event.ctrlKey || event.metaKey || event.shiftKey);
    }
  });
}

function selectNode(id, additive) {
  endGesture();
  if (additive && selected.has(id)) {
    selected.delete(id);
  } else {
    if (!additive && !selected.has(id)) selected.clear();
    selected.add(id);
  }
  updateSelection(selected.has(id) ? id : undefined);
}

function updateConnection(connection) {
  const { from, to, fromOperator, toOperator, path, text } = connection;
  const y1 = from.y + (fromOperator ? 47 : 26);
  const y2 = to.y - (toOperator ? 47 : 26);
  const middle = (y1 + y2) / 2;
  path.setAttribute(
    "d",
    `M${from.x},${y1} C${from.x},${middle} ${to.x},${middle} ${to.x},${y2}`,
  );
  text.setAttribute("x", (from.x + to.x) / 2 + 10);
  text.setAttribute("y", middle);
}

function moveNodes(origins, dx, dy) {
  for (const [id, origin] of origins) {
    const view = nodeViews.get(id);
    view.position.x = origin.x + dx;
    view.position.y = origin.y + dy;
    view.group.setAttribute(
      "transform",
      `translate(${view.position.x - view.initial.x} ${view.position.y - view.initial.y})`,
    );
  }
  connections.forEach(updateConnection);
}

function endGesture(cancel = false) {
  if (!gesture) return;
  const previous = gesture;
  gesture = undefined;
  previous.rectangle?.remove();
  svg.classList.remove("dragging", "selecting");
  if (cancel) {
    if (previous.origins) {
      moveNodes(previous.origins, 0, 0);
    } else {
      selected = previous.originalSelected;
      updateSelection();
    }
  }
  if (svg.hasPointerCapture(previous.pointerId)) {
    svg.releasePointerCapture(previous.pointerId);
  }
}

function graphPoint(event) {
  const viewport = byId("canvas").getBoundingClientRect();
  if (!viewport.width || !viewport.height) return;
  const [x, y, width, height] = svg
    .getAttribute("viewBox")
    .split(" ")
    .map(Number);
  return {
    x: x + ((event.clientX - viewport.left) * width) / viewport.width,
    y: y + ((event.clientY - viewport.top) * height) / viewport.height,
  };
}

function moveGesture(event) {
  if (!gesture || gesture.pointerId !== event.pointerId) return;
  const point = graphPoint(event);
  if (!point) {
    endGesture(true);
    return;
  }
  if (
    !gesture.moved &&
    Math.hypot(
      event.clientX - gesture.clientX,
      event.clientY - gesture.clientY,
    ) < 3
  ) {
    return;
  }
  gesture.moved = true;
  const dx = point.x - gesture.start.x;
  const dy = point.y - gesture.start.y;
  if (gesture.origins) {
    moveNodes(gesture.origins, dx, dy);
  } else {
    const x = Math.min(point.x, gesture.start.x);
    const y = Math.min(point.y, gesture.start.y);
    const width = Math.abs(dx);
    const height = Math.abs(dy);
    for (const [key, value] of Object.entries({ x, y, width, height })) {
      gesture.rectangle.setAttribute(key, value);
    }
    selected = new Set(gesture.baseSelected);
    for (const [id, { position, halfWidth, halfHeight }] of nodeViews) {
      if (
        position.x + halfWidth >= x &&
        position.x - halfWidth <= x + width &&
        position.y + halfHeight >= y &&
        position.y - halfHeight <= y + height
      ) {
        selected.add(id);
      }
    }
    updateSelection();
  }
}

function reframe() {
  let x = 0;
  let y = 0;
  let right = layoutBounds.width;
  let bottom = layoutBounds.height;
  for (const { position, halfWidth, halfHeight } of nodeViews.values()) {
    x = Math.min(x, position.x - halfWidth - 20);
    y = Math.min(y, position.y - halfHeight - 20);
    right = Math.max(right, position.x + halfWidth + 20);
    bottom = Math.max(bottom, position.y + halfHeight + 20);
  }
  bounds = { x, y, width: right - x, height: bottom - y };
}

function fit() {
  const viewport = byId("canvas").getBoundingClientRect();
  if (viewport.width === 0 || viewport.height === 0) return;
  const scale =
    Math.max(bounds.width / viewport.width, bounds.height / viewport.height) /
    zoom;
  const width = viewport.width * scale;
  const height = viewport.height * scale;
  svg.setAttribute(
    "viewBox",
    `${bounds.x + (bounds.width - width) / 2} ${bounds.y + (bounds.height - height) / 2} ${width} ${height}`,
  );
}

function changeZoom(factor) {
  endGesture();
  zoom = Math.min(4, Math.max(0.25, zoom * factor));
  fit();
}

function render(graph) {
  endGesture();
  nodeViews = new Map();
  selected = new Set();
  connections = [];
  const tensors = new Set([
    ...graph.inputs,
    ...graph.outputs,
    ...graph.nodes.flatMap((node) => [...node.inputs, ...node.outputs]),
  ]);
  byId("graph-summary").textContent =
    `${graph.nodes.length} node${graph.nodes.length === 1 ? "" : "s"} / ${tensors.size} tensors`;
  byId("initializer-count").textContent =
    `${graph.initializerCount} initializer${graph.initializerCount === 1 ? "" : "s"}`;

  const edges = byId("edges");
  const nodes = byId("nodes");
  edges.replaceChildren();
  nodes.replaceChildren();
  const producers = new Map();
  const positions = new Map();
  for (const node of graph.nodes) {
    for (const output of node.outputs) producers.set(output, node.id);
  }
  bounds = {
    x: 0,
    y: 0,
    width: Math.max(
      500,
      40 + Math.max(graph.inputs.length, graph.outputs.length) * 140,
    ),
    height: Math.max(650, 250 + graph.nodes.length * 145),
  };
  layoutBounds = bounds;
  graph.nodes.forEach((node, index) => {
    positions.set(node.id, {
      x: 100 + Math.random() * (bounds.width - 200),
      y: 180 + index * 145,
    });
  });
  const addPort = (name, index, output) => {
    const count = output ? graph.outputs.length : graph.inputs.length;
    const position = {
      x: ((index + 1) * bounds.width) / (count + 1),
      y: output ? bounds.height - 60 : 60,
    };
    const group = element("g", {
      class: `port${output ? " output" : ""}`,
      "aria-label": `${output ? "Output" : "Input"}: ${name}`,
    });
    group.append(
      element("title", {}, name),
      element("rect", {
        x: position.x - 60,
        y: position.y - 26,
        width: 120,
        height: 52,
        rx: 9,
      }),
    );
    label(
      group,
      position.x,
      position.y - 6,
      output ? "OUTPUT" : "INPUT",
      "port-type",
    );
    label(group, position.x, position.y + 12, shortName(name, 16), "port-name");
    registerNode(`${output ? "output" : "input"}-${index}`, {
      port: { name, output },
      group,
      position,
      halfWidth: 60,
      halfHeight: 26,
    });
    nodes.append(group);
    return position;
  };
  const inputPositions = new Map(
    graph.inputs.map((name, index) => [name, addPort(name, index, false)]),
  );
  const connect = (from, to, name, fromOperator, toOperator) => {
    if (!from || !to) return;
    const path = element("path", { class: "edge" });
    edges.append(path);
    const text = label(
      edges,
      0,
      0,
      shortName(name, 15),
      "tensor-label",
      "start",
    );
    const connection = { from, to, fromOperator, toOperator, path, text };
    connections.push(connection);
    updateConnection(connection);
  };
  for (const node of graph.nodes) {
    const position = positions.get(node.id);
    const group = element("g", {
      class: "graph-node",
      "aria-label": `${node.opType}: ${node.name}`,
    });
    group.append(
      element("title", {}, `${node.name} (${node.opType})`),
      element("rect", {
        x: position.x - 80,
        y: position.y - 47,
        width: 160,
        height: 94,
        rx: 12,
      }),
    );
    label(group, position.x, position.y - 23, "OPERATOR", "operator-tag");
    label(
      group,
      position.x,
      position.y + 3,
      shortName(node.opType, 15),
      "operator-label",
    );
    label(
      group,
      position.x,
      position.y + 25,
      shortName(node.name),
      "operator-name",
    );
    registerNode(node.id, {
      node,
      group,
      position,
      halfWidth: 80,
      halfHeight: 47,
    });
    nodes.append(group);
    for (const input of node.inputs) {
      const producer = producers.get(input);
      connect(
        producer ? positions.get(producer) : inputPositions.get(input),
        position,
        input,
        Boolean(producer),
        true,
      );
    }
  }
  graph.outputs.forEach((name, index) => {
    const producer = producers.get(name);
    connect(
      producer ? positions.get(producer) : inputPositions.get(name),
      addPort(name, index, true),
      name,
      Boolean(producer),
      false,
    );
  });
  zoom = 1;
  fit();
  if (graph.nodes.length) selected.add(graph.nodes[0].id);
  updateSelection();
}

svg.addEventListener("pointerdown", (event) => {
  if (event.button !== 0 || gesture || !svg.hasAttribute("viewBox")) return;
  const start = graphPoint(event);
  if (!start) return;
  const group = event.target.closest(".graph-node, .port");
  const additive = event.ctrlKey || event.metaKey || event.shiftKey;
  const originalSelected = new Set(selected);
  event.preventDefault();
  if (group) {
    selectNode(group.dataset.id, additive);
    group.focus({ preventScroll: true });
    if (!selected.has(group.dataset.id)) return;
    gesture = {
      origins: new Map(
        [...selected].map((id) => [id, { ...nodeViews.get(id).position }]),
      ),
    };
    svg.classList.add("dragging");
  } else {
    selected = additive ? new Set(selected) : new Set();
    updateSelection();
    svg.focus({ preventScroll: true });
    const rectangle = element("rect", { class: "selection-rectangle" });
    svg.append(rectangle);
    gesture = {
      rectangle,
      baseSelected: new Set(selected),
      originalSelected,
    };
    svg.classList.add("selecting");
  }
  Object.assign(gesture, {
    start,
    clientX: event.clientX,
    clientY: event.clientY,
    pointerId: event.pointerId,
  });
  svg.setPointerCapture(event.pointerId);
});
svg.addEventListener("pointermove", moveGesture);
svg.addEventListener("pointerup", (event) => {
  if (gesture?.pointerId !== event.pointerId) return;
  moveGesture(event);
  endGesture();
});
svg.addEventListener("pointercancel", (event) => {
  if (gesture?.pointerId === event.pointerId) endGesture(true);
});
svg.addEventListener("lostpointercapture", (event) => {
  if (gesture?.pointerId === event.pointerId) endGesture();
});
svg.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    event.preventDefault();
    if (gesture) {
      endGesture(true);
    } else {
      selected.clear();
      updateSelection();
    }
  } else if (
    (event.ctrlKey || event.metaKey) &&
    event.key.toLowerCase() === "a"
  ) {
    event.preventDefault();
    endGesture();
    selected = new Set(nodeViews.keys());
    updateSelection();
  }
});
window.addEventListener("blur", () => endGesture(true));
byId("open-model").addEventListener("click", () =>
  vscode.postMessage({ type: "open" }),
);
byId("zoom-in").addEventListener("click", () => {
  changeZoom(1.25);
});
byId("zoom-out").addEventListener("click", () => {
  changeZoom(1 / 1.25);
});
byId("zoom-fit").addEventListener("click", () => {
  endGesture();
  reframe();
  zoom = 1;
  fit();
});
svg.addEventListener(
  "wheel",
  (event) => {
    if (event.deltaY === 0) return;
    event.preventDefault();
    changeZoom(event.deltaY < 0 ? 1.25 : 1 / 1.25);
  },
  { passive: false },
);
window.addEventListener("resize", () => {
  endGesture();
  fit();
});
window.addEventListener("message", (event) => {
  if (event.data?.type === "graph") {
    render(event.data.graph);
  }
});
vscode.postMessage({ type: "ready" });
