/* global acquireVsCodeApi */
const vscode = acquireVsCodeApi();
const svgNS = "http://www.w3.org/2000/svg";
const byId = (id) => document.getElementById(id);
const svg = byId("graph");
let bounds = { width: 800, height: 350 };
let zoom = 1;

function element(tag, attributes, text) {
  const node = document.createElementNS(svgNS, tag);
  for (const [name, value] of Object.entries(attributes)) {
    node.setAttribute(name, String(value));
  }
  if (text !== undefined) node.textContent = text;
  return node;
}

function label(parent, x, y, text, className, anchor = "middle") {
  parent.append(
    element("text", { x, y, class: className, "text-anchor": anchor }, text),
  );
}

function shortName(value, length = 22) {
  return value.length > length ? `${value.slice(0, length - 1)}…` : value;
}

function inspect(node) {
  byId("node-title").textContent = node?.opType ?? "No operator";
  byId("node-description").textContent = node
    ? node.opType === "Relu" && node.domain === "ai.onnx"
      ? "Rectified linear activation: max(0, x)."
      : `${node.opType} operator`
    : "This graph contains no operator nodes.";
  byId("node-name").textContent = node?.name ?? "—";
  byId("node-domain").textContent = node?.domain ?? "—";
  for (const direction of ["inputs", "outputs"]) {
    const names = node?.[direction] ?? [];
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
  for (const group of byId("nodes").querySelectorAll(".graph-node")) {
    const selected = group.dataset.id === node?.id;
    group.classList.toggle("selected", selected);
    group.setAttribute("aria-pressed", String(selected));
  }
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
    `${(bounds.width - width) / 2} ${(bounds.height - height) / 2} ${width} ${height}`,
  );
}

function changeZoom(factor) {
  zoom = Math.min(4, Math.max(0.25, zoom * factor));
  fit();
}

function render(graph) {
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
    width: Math.max(
      500,
      40 + Math.max(graph.inputs.length, graph.outputs.length) * 140,
    ),
    height: Math.max(650, 250 + graph.nodes.length * 145),
  };
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
    const group = element("g", { class: `port${output ? " output" : ""}` });
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
    nodes.append(group);
    return position;
  };
  const inputPositions = new Map(
    graph.inputs.map((name, index) => [name, addPort(name, index, false)]),
  );
  const connect = (from, to, name, fromOperator, toOperator) => {
    if (!from || !to) return;
    const y1 = from.y + (fromOperator ? 47 : 26);
    const y2 = to.y - (toOperator ? 47 : 26);
    const middle = (y1 + y2) / 2;
    edges.append(
      element("path", {
        class: "edge",
        d: `M${from.x},${y1} C${from.x},${middle} ${to.x},${middle} ${to.x},${y2}`,
      }),
    );
    label(
      edges,
      (from.x + to.x) / 2 + 10,
      middle,
      shortName(name, 15),
      "tensor-label",
      "start",
    );
  };
  for (const node of graph.nodes) {
    const position = positions.get(node.id);
    const group = element("g", {
      class: "graph-node",
      tabindex: 0,
      role: "button",
      "aria-label": `${node.opType}: ${node.name}`,
      "data-id": node.id,
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
    group.addEventListener("click", () => inspect(node));
    group.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        inspect(node);
      }
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
  inspect(graph.nodes[0]);
}

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
window.addEventListener("resize", fit);
window.addEventListener("message", (event) => {
  if (event.data?.type === "graph") {
    render(event.data.graph);
  }
});
vscode.postMessage({ type: "ready" });
