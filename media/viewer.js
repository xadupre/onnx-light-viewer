/* global acquireVsCodeApi */
const vscode = acquireVsCodeApi();
const svgNS = "http://www.w3.org/2000/svg";
const byId = (id) => document.getElementById(id);
const svg = byId("graph");
let bounds = { x: 0, y: 0, width: 800, height: 350 };
let layoutBounds = bounds;
let zoom = 1;
let pan = { x: 0, y: 0 };
let nodeViews = new Map();
let selected = new Set();
let connections = [];
let gesture;
let inspectedNode;
let inspectedView;
let tensorInformation = new Map();
let tensorLabels = [];

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

function tensorDescription(name) {
  const info = tensorInformation.get(name);
  const type = info?.type ?? "?";
  if (info && info.shape === undefined) return type;
  return `${type} [${info?.shape === null || !info ? "?" : info.shape.join(", ")}]`;
}

function tensorLabel(parent, x, y, name, length, anchor = "middle") {
  const text = label(parent, x, y, "", "tensor-info", anchor);
  const title = element("title", {});
  const value = element("tspan", {});
  text.append(title, value);
  tensorLabels.push({
    names: Array.isArray(name) ? name : [name],
    text,
    title,
    value,
    length,
  });
  return text;
}

function updateTensorInformation(tensors) {
  tensorInformation = new Map(tensors.map((tensor) => [tensor.name, tensor]));
  for (const { names, text, title, value, length } of tensorLabels) {
    const description = names.map(tensorDescription).join(" | ");
    title.textContent = names
      .map((name) => `${name}: ${tensorDescription(name)}`)
      .join("\n");
    text.setAttribute("aria-label", description);
    value.textContent = shortName(description, length);
  }
}

function refreshPositions() {
  endGesture();
  const positions = new Map();
  for (const [id, view] of nodeViews) {
    positions.set(id, {
      x: view.node
        ? 100 + Math.random() * (layoutBounds.width - 200)
        : view.initial.x,
      y: view.initial.y,
    });
  }
  moveNodes(positions, 0, 0);
  bounds = { ...layoutBounds };
  zoom = 1;
  pan = { x: 0, y: 0 };
  fit();
}

function refreshStatus(pending, message = "") {
  byId("refresh-shapes").disabled = pending;
  byId("refresh-shapes").setAttribute("aria-busy", String(pending));
  byId("shape-status").textContent = message;
  byId("shape-status").title = message;
}

function resetDocumentation() {
  if (byId("operator-doc").open) byId("operator-doc").close();
  byId("doc-title").textContent = "";
  byId("doc-version").textContent = "";
  byId("doc-body").textContent = "";
}

function inspect(view) {
  inspectedView = view;
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
    ? `${node.opType} operator`
    : port
      ? `Graph ${port.output ? "output" : "input"} tensor.`
      : hasOperators
        ? "Select a node to inspect its inputs and outputs."
        : "This graph contains no operator nodes.";
  byId("node-name").textContent = node?.name ?? port?.name ?? "—";
  byId("node-domain").textContent = node?.domain ?? "—";
  byId("node-opset").textContent = node?.opset ?? "—";
  if (inspectedNode !== node) {
    resetDocumentation();
    inspectedNode = node;
  }
  byId("show-doc").disabled = !node?.schema?.doc;
  byId("show-doc").title = node?.schema?.doc
    ? "Documentation from onnx-light LightOpSchema"
    : "No LightOpSchema documentation available for this selection.";
  const attributes = node?.attributes ?? [];
  const names = new Set(attributes.map((attribute) => attribute.name));
  const defaults = (node?.schema?.defaults ?? []).filter(
    (attribute) => !names.has(attribute.name),
  );
  parameterList("attributes", [...attributes, ...defaults], "attribute-count");
  initializerList(node?.parameters ?? []);
  byId("node-parameters").hidden = !node;
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
        const tensorName = document.createElement("span");
        tensorName.className = "tensor-name";
        tensorName.textContent = name;
        const information = document.createElement("span");
        information.className = "tensor-info";
        information.textContent = tensorDescription(name);
        item.append(tensorName, information);
        return item;
      }),
    );
  }
}

function initializerList(parameters) {
  byId("parameter-count").textContent = parameters.length;
  byId("node-initializers").replaceChildren(
    ...parameters.map((parameter) => {
      const item = document.createElement("li");
      const heading = document.createElement("strong");
      heading.textContent = `${parameter.name} (${parameter.type})`;
      const shape = document.createElement("div");
      shape.className = "tensor-info";
      shape.textContent = `Shape: [${parameter.shape.join(", ")}]`;
      item.append(heading, shape);
      for (const [label, storage] of [
        [parameter.indicesStorage ? "Values" : "Storage", parameter.storage],
        ["Indices", parameter.indicesStorage],
      ]) {
        if (!storage) continue;
        const description = document.createElement("div");
        description.className = "initializer-storage";
        description.textContent = `${label}: ${storage.external ? `external file: ${storage.location || "not specified"}` : "embedded"}`;
        item.append(description);
      }
      return item;
    }),
  );
}

function parameterList(id, parameters, countId) {
  byId(countId).textContent = parameters.length;
  byId(`node-${id}`).replaceChildren(
    ...parameters.map((parameter) => {
      const item = document.createElement("li");
      const heading = document.createElement("strong");
      heading.textContent = `${parameter.name} (${parameter.type})${parameter.default ? " [default]" : ""}`;
      const value = document.createElement("pre");
      value.textContent = parameter.value;
      item.append(heading, value);
      return item;
    }),
  );
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
  const { from, to, fromOperator, toOperator, path, text, information } =
    connection;
  const y1 = from.y + (fromOperator ? 47 : 26);
  const y2 = to.y - (toOperator ? 47 : 26);
  const middle = (y1 + y2) / 2;
  path.setAttribute(
    "d",
    `M${from.x},${y1} C${from.x},${middle} ${to.x},${middle} ${to.x},${y2}`,
  );
  text.setAttribute("x", (from.x + to.x) / 2 + 10);
  text.setAttribute("y", middle);
  information.setAttribute("x", (from.x + to.x) / 2 + 10);
  information.setAttribute("y", middle + 14);
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
    `${bounds.x + (bounds.width - width) / 2 + pan.x} ${bounds.y + (bounds.height - height) / 2 + pan.y} ${width} ${height}`,
  );
}

function changeZoom(factor) {
  endGesture();
  zoom = Math.min(4, Math.max(0.25, zoom * factor));
  fit();
}

function render(graph) {
  endGesture();
  resetDocumentation();
  inspectedNode = undefined;
  inspectedView = undefined;
  tensorLabels = [];
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
      position.y - 12,
      output ? "OUTPUT" : "INPUT",
      "port-type",
    );
    label(group, position.x, position.y + 3, shortName(name, 16), "port-name");
    tensorLabel(group, position.x, position.y + 18, name, 18);
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
    const information = tensorLabel(edges, 0, 0, name, 34, "start");
    const connection = {
      from,
      to,
      fromOperator,
      toOperator,
      path,
      text,
      information,
    };
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
      position.y + 21,
      shortName(node.name),
      "operator-name",
    );
    if (node.outputs.length) {
      const information = tensorLabel(
        group,
        position.x,
        position.y + 39,
        node.outputs,
        26,
      );
      information.classList.add("node-tensor-info");
    }
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
  pan = { x: 0, y: 0 };
  fit();
  if (graph.nodes.length) selected.add(graph.nodes[0].id);
  updateTensorInformation(graph.tensors ?? []);
  updateSelection();
  refreshStatus(false);
  byId("refresh-positions").disabled = false;
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
byId("show-doc").addEventListener("click", () => {
  if (!inspectedNode?.schema?.doc) return;
  const { schema, opType, domain, opset } = inspectedNode;
  byId("doc-title").textContent = `${opType} documentation`;
  byId("doc-version").textContent =
    `${domain} · model opset ${opset} · LightOpSchema since version ${schema.sinceVersion}`;
  byId("doc-body").textContent = schema.doc;
  byId("operator-doc").showModal();
});
byId("close-doc").addEventListener("click", () => byId("operator-doc").close());
byId("open-model").addEventListener("click", () =>
  vscode.postMessage({ type: "open" }),
);
byId("open-backend-test").addEventListener("click", () =>
  vscode.postMessage({ type: "open-backend-test" }),
);
byId("refresh-shapes").addEventListener("click", () => {
  refreshStatus(true, "Inferring shapes/types…");
  vscode.postMessage({ type: "refresh-shapes" });
});
byId("refresh-positions").addEventListener("click", refreshPositions);
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
  pan = { x: 0, y: 0 };
  fit();
});
svg.addEventListener(
  "wheel",
  (event) => {
    if (event.ctrlKey) {
      if (event.deltaY === 0) return;
      event.preventDefault();
      changeZoom(event.deltaY < 0 ? 1.25 : 1 / 1.25);
      return;
    }
    if ((!event.deltaX && !event.deltaY) || !svg.hasAttribute("viewBox"))
      return;
    const viewport = byId("canvas").getBoundingClientRect();
    if (!viewport.width || !viewport.height) return;
    event.preventDefault();
    endGesture();
    const [, , width, height] = svg
      .getAttribute("viewBox")
      .split(" ")
      .map(Number);
    const unitX =
      event.deltaMode === 2 ? viewport.width : event.deltaMode === 1 ? 16 : 1;
    const unitY =
      event.deltaMode === 2 ? viewport.height : event.deltaMode === 1 ? 16 : 1;
    pan.x +=
      ((event.shiftKey ? event.deltaX || event.deltaY : event.deltaX) *
        unitX *
        width) /
      viewport.width;
    if (!event.shiftKey) {
      pan.y += (event.deltaY * unitY * height) / viewport.height;
    }
    fit();
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
  } else if (event.data?.type === "shape-info") {
    updateTensorInformation(event.data.tensors);
    inspect(inspectedView);
    refreshStatus(false, "Shapes/types refreshed");
  } else if (event.data?.type === "shape-error") {
    refreshStatus(false, `Shape/type inference failed: ${event.data.error}`);
  }
});
vscode.postMessage({ type: "ready" });
