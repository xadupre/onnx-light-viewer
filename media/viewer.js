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
let renderedSelection = new Set();
let connectionsByPosition = new Map();
let connections = [];
let camera;
let virtualized = false;
let overviewVisible = false;
let overviewDirty = true;
let crossingConnections = new Set();
let readablePending = false;
let hasOperators = false;
let gesture;
let inspectedNode;
let inspectedView;
let tensorInformation = new Map();
let tensorLabels = [];
let metadataVisible = false;
let metadataPending = false;
let graphMetadata;
let nodeMetadata = new Map();
let valueMetadata = new Map();
let connectionsByName = new Map();
let releasedValues = new Set();
let unusedLabels = [];
let hasModel = false;
let replacementPending = false;
let nextAction = 0;
const logSession = crypto.randomUUID();
const logRows = new Map();
const pendingActions = new Map();
const statusLabels = {
  RUNNING: "Running",
  SUCCEEDED: "OK",
  FAILED: "Failed",
  CANCELLED: "Cancelled",
  INVALIDATED: "Reload required",
};
const RefreshStatus = Object.freeze({
  IDLE: "idle",
  PENDING: "pending",
  UPDATED: "updated",
  UNCHANGED: "unchanged",
  FAILED: "failed",
});
const tabs = ["model", "logs", "status", "controls"];

function showTab(name, focus = false) {
  endGesture();
  for (const tab of tabs) {
    const active = tab === name;
    byId(`${tab}-view`).hidden = !active;
    byId(`${tab}-tab`).setAttribute("aria-selected", String(active));
    byId(`${tab}-tab`).tabIndex = active ? 0 : -1;
  }
  if (focus) byId(`${name}-tab`).focus();
  if (name === "model") fit();
  else if (name === "logs")
    byId("logs-view").scrollTop = byId("logs-view").scrollHeight;
}

function displayLog(entry) {
  const panel = byId("logs-view");
  const follow =
    !panel.hidden &&
    panel.scrollHeight - panel.scrollTop - panel.clientHeight < 32;
  let row = logRows.get(entry.id);
  if (entry.status === "RUNNING" && row && row.entry.status !== "RUNNING")
    return;
  if (!row) {
    const item = document.createElement("li");
    item.className = "log-entry";
    const heading = document.createElement("div");
    heading.className = "log-heading";
    const time = document.createElement("time");
    const status = document.createElement("span");
    status.className = "log-status";
    const action = document.createElement("strong");
    const details = document.createElement("pre");
    heading.append(time, status, action);
    item.append(heading, details);
    byId("log-entries").append(item);
    row = { item, time, status, action, details };
    logRows.set(entry.id, row);
  }
  row.entry = entry;
  row.item.dataset.status = entry.status;
  row.time.dateTime = entry.timestamp;
  row.time.textContent = entry.timestamp;
  row.status.textContent = statusLabels[entry.status];
  row.action.textContent = entry.action;
  row.details.textContent = entry.details ?? "";
  row.details.hidden = !entry.details;
  byId("logs-empty").hidden = true;
  if (follow) panel.scrollTop = panel.scrollHeight;
  if (entry.status !== "RUNNING" && pendingActions.has(entry.id)) {
    const type = pendingActions.get(entry.id);
    pendingActions.delete(entry.id);
    if (type === "open" || type === "open-backend-test") {
      replacementPending = [...pendingActions.values()].some(
        (pending) => pending === "open" || pending === "open-backend-test",
      );
      setLoading(replacementPending);
    }
  }
}

function newAction(action) {
  const entry = {
    id: `ui:${logSession}:${++nextAction}`,
    timestamp: new Date().toISOString(),
    action,
    status: "RUNNING",
  };
  displayLog(entry);
  vscode.postMessage({ type: "interaction", entry: { ...entry } });
  return entry;
}

function requestAction(type, action, parameters = {}) {
  const entry = newAction(action);
  pendingActions.set(entry.id, type);
  if (type === "open" || type === "open-backend-test") {
    replacementPending = true;
    setLoading(true);
  }
  vscode.postMessage({ type, ...parameters, actionId: entry.id });
}

function setLoading(pending, message = "") {
  byId("load-progress").hidden = !pending;
  byId("model-view").setAttribute("aria-busy", String(pending));
  for (const id of ["open-model", "open-backend-test"])
    byId(id).disabled = pending;
  for (const id of [
    "refresh-shapes",
    "refresh-metadata",
    "refresh-positions",
    "show-metadata",
  ])
    byId(id).disabled =
      pending || !hasModel || byId(id).getAttribute("aria-busy") === "true";
  byId("load-status").textContent =
    message || (pending ? "Loading model…" : "");
}

window.addEventListener("error", (event) => {
  const entry = newAction("Viewer error");
  entry.status = "FAILED";
  entry.details = event.error?.stack ?? event.message;
  displayLog(entry);
  vscode.postMessage({ type: "interaction", entry: { ...entry } });
});
for (const name of tabs) {
  byId(`${name}-tab`).addEventListener("click", () => showTab(name));
  byId(`${name}-tab`).addEventListener("keydown", (event) => {
    let index = tabs.indexOf(name);
    if (event.key === "Home") index = 0;
    else if (event.key === "End") index = tabs.length - 1;
    else if (event.key === "ArrowLeft")
      index = (index + tabs.length - 1) % tabs.length;
    else if (event.key === "ArrowRight") index = (index + 1) % tabs.length;
    else return;
    event.preventDefault();
    const next = tabs[index];
    byId(`${next}-tab`).click();
    byId(`${next}-tab`).focus();
  });
}

function displayStatus(status) {
  byId("status-library").textContent = status.library.loaded
    ? `Loaded — onnx-light ${status.library.version}`
    : "Not loaded";
  byId("status-model").textContent = status.model.name ?? "No model loaded";
  const states = {
    none: "No native model",
    ready: "Ready",
    invalidated: "Discarded after a failed analysis",
  };
  byId("status-handle").textContent = states[status.model.state];
  byId("status-running").textContent = status.running.length
    ? status.running.join("\n")
    : "Idle";
  const recovery = {
    none: "Open a model to create a native handle.",
    ready: "No model recovery pending.",
    invalidated:
      "The next native action recreates the model from its original in-memory bytes or backend case. The previous display is preserved; shapes and metadata may need refreshing.",
  };
  byId("status-recovery").textContent = recovery[status.model.state];
  byId("status-failure").hidden = !status.lastFailure;
  if (status.lastFailure) {
    byId("status-failure-action").textContent = status.lastFailure.action;
    byId("status-failure-time").textContent = status.lastFailure.timestamp;
    byId("status-failure-details").textContent =
      status.lastFailure.details ?? "";
  }
}

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

function subgraphLabel(attribute, index) {
  return `${attribute.name}${attribute.graphs.length > 1 ? ` [${index}]` : ""}`;
}

function operatorSizes(nodes) {
  const context = document.createElement("canvas").getContext("2d");
  const family = getComputedStyle(svg).fontFamily;
  const widths = new Map();
  context.font = `600 19px ${family}`;
  for (const { opType } of nodes) {
    if (!widths.has(opType)) {
      widths.set(opType, context.measureText(opType).width);
    }
  }
  context.font = `10px ${family}`;
  return new Map(
    nodes.map((node) => {
      const typeWidth = widths.get(node.opType);
      const nameWidth = context.measureText(node.name).width;
      const initializers = (node.parameters ?? []).map((parameter) => {
        const description = `${parameter.name}: ${parameter.type} [${parameter.shape.join(", ")}]`;
        const text = shortName(description, 38);
        return {
          description,
          text,
          textWidth: context.measureText(text).width,
        };
      });
      const subgraphs = (node.attributes ?? []).flatMap((attribute) =>
        (attribute.graphs ?? []).map((graph, index) => {
          const text = subgraphLabel(attribute, index);
          return {
            ...graph,
            label: text,
            textWidth: context.measureText(text).width,
          };
        }),
      );
      const width = Math.min(
        280,
        Math.max(
          160,
          Math.ceil(Math.max(typeWidth, nameWidth) + 32),
          ...initializers.map((parameter) =>
            Math.ceil(parameter.textWidth + 24),
          ),
          ...subgraphs.map((graph) => Math.ceil(graph.textWidth + 48)),
        ),
      );
      return [
        node.id,
        {
          width,
          height: 94 + initializers.length * 18 + subgraphs.length * 22,
          initializers,
          subgraphs,
          typeFontSize: Math.min(19, (19 * (width - 32)) / (typeWidth || 1)),
          nameFontSize: Math.min(10, (10 * (width - 32)) / (nameWidth || 1)),
        },
      ];
    }),
  );
}

function randomNodeX(halfWidth) {
  const margin = halfWidth + 20;
  return margin + Math.random() * (layoutBounds.width - 2 * margin);
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
  const entry = {
    names: Array.isArray(name) ? name : [name],
    text,
    title,
    value,
    length,
  };
  tensorLabels.push(entry);
  updateTensorLabel(entry);
  return text;
}

function updateTensorLabel({ names, text, title, value, length }) {
  const description = names.map(tensorDescription).join(" | ");
  title.textContent = names
    .map((name) => `${name}: ${tensorDescription(name)}`)
    .join("\n");
  text.setAttribute("aria-label", description);
  value.textContent = shortName(description, length);
}

function countUpdates(previous, current, key) {
  const before = new Map(
    previous.map((item) => [item[key], JSON.stringify(item)]),
  );
  const after = new Map(
    current.map((item) => [item[key], JSON.stringify(item)]),
  );
  return [...new Set([...before.keys(), ...after.keys()])].filter(
    (id) => before.get(id) !== after.get(id),
  ).length;
}

function updateTensorInformation(tensors) {
  const updates = countUpdates(
    [...tensorInformation.values()],
    tensors,
    "name",
  );
  tensorInformation = new Map(tensors.map((tensor) => [tensor.name, tensor]));
  for (const entry of tensorLabels) updateTensorLabel(entry);
  return updates;
}

function refreshPositions() {
  endGesture();
  const positions = new Map();
  for (const [id, view] of nodeViews) {
    positions.set(id, {
      x: view.node ? randomNodeX(view.halfWidth) : view.initial.x,
      y: view.initial.y,
    });
  }
  moveNodes(positions, 0, 0);
  bounds = { ...layoutBounds };
  zoom = 1;
  pan = { x: 0, y: 0 };
  fit();
}

function refreshButtonStatus(id, status, message) {
  const button = byId(id);
  button.dataset.defaultTitle ??= button.title;
  button.dataset.refreshStatus = status;
  button.title = message || button.dataset.defaultTitle;
}

function refreshStatus(
  pending,
  message = "",
  status = pending ? RefreshStatus.PENDING : RefreshStatus.IDLE,
) {
  byId("refresh-shapes").disabled = pending;
  byId("refresh-shapes").setAttribute("aria-busy", String(pending));
  refreshButtonStatus("refresh-shapes", status, message);
  byId("shape-status").textContent = message;
  byId("shape-status").title = message;
  // Status changes can wrap the toolbar and resize the canvas.
  if (gesture) gesture.viewport = undefined;
  if (svg.hasAttribute("viewBox")) fit();
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
  const captures = node?.captures ?? [];
  byId("node-captures").hidden = captures.length === 0;
  byId("captured-inputs").replaceChildren(
    ...captures.map((name) => {
      const item = document.createElement("li");
      item.textContent = `${name}: ${tensorDescription(name)}`;
      return item;
    }),
  );
  inspectMetadata();
}

function metadataEntries(parent, entries) {
  parent.replaceChildren();
  if (parent.parentElement.tagName === "DETAILS")
    parent.parentElement.hidden = entries.length === 0;
  for (const { key, value } of entries) {
    const term = document.createElement("dt");
    const description = document.createElement("dd");
    term.textContent = key;
    description.textContent = value;
    parent.append(term, description);
  }
}

function metadataNames(id, names) {
  byId(`${id}-section`).hidden = names.length === 0;
  byId(id).replaceChildren(
    ...names.map((name) => {
      const item = document.createElement("li");
      item.textContent = name;
      return item;
    }),
  );
}

function updateReleases() {
  const next = new Set(
    metadataVisible
      ? (nodeMetadata.get(inspectedView?.node?.id)?.releaseAfter ?? [])
      : [],
  );
  for (const name of new Set([...releasedValues, ...next])) {
    const released = next.has(name);
    if (released === releasedValues.has(name)) continue;
    for (const connection of connectionsByName.get(name) ?? [])
      markReleased(connection, released);
  }
  releasedValues = next;
  if (virtualized) drawCrossingConnections();
}

function markReleased({ path, text, name }, released) {
  if (!path) return;
  path.classList.toggle("released", released);
  text.classList.toggle("released", released);
  path.setAttribute(
    "aria-label",
    released ? `${name}: releasable after ${inspectedView.node.name}` : name,
  );
}

function inspectMetadata() {
  updateReleases();
  if (!metadataVisible || !graphMetadata) return;
  const node = inspectedView?.node;
  const port = inspectedView?.port;
  const info = node ? nodeMetadata.get(node.id) : valueMetadata.get(port?.name);
  metadataEntries(byId("metadata-analysis"), [
    { key: "Name", value: node?.name ?? port?.name ?? "No selection" },
    { key: "Tag", value: info?.tag ?? "Unknown" },
    {
      key: "Constant",
      value: info ? (info.constant ? "Yes" : "No") : "Unknown",
    },
  ]);
  metadataNames("metadata-releases", node ? info.releaseAfter : []);
  metadataNames("metadata-last-use", node ? info.lastUse : []);
  metadataNames("metadata-unused", node ? info.unusedOutputs : []);
  metadataEntries(byId("metadata-node"), node ? info.entries : []);
  const names = new Set(
    node
      ? [...node.inputs, ...(node.captures ?? []), ...node.outputs]
      : port
        ? [port.name]
        : [],
  );
  byId("metadata-value-count").textContent = names.size;
  byId("metadata-values").parentElement.hidden = names.size === 0;
  byId("metadata-values").replaceChildren(
    ...[...names].map((name) => {
      const value = valueMetadata.get(name);
      const item = document.createElement("li");
      const heading = document.createElement("strong");
      heading.className = "metadata-value-name";
      heading.textContent = name;
      const badges = document.createElement("div");
      badges.className = "metadata-badges";
      for (const text of [value?.tag, value?.constant ? "constant" : null]) {
        if (!text) continue;
        const badge = document.createElement("span");
        badge.textContent = text;
        badges.append(badge);
      }
      const description = document.createElement("div");
      description.className = "metadata-value-type";
      description.textContent = tensorDescription(name);
      item.append(heading, badges, description);
      if (value?.entries.length) {
        const details = document.createElement("details");
        const summary = document.createElement("summary");
        summary.textContent = "Stored metadata";
        const entries = document.createElement("dl");
        details.append(summary, entries);
        metadataEntries(entries, value.entries);
        item.append(details);
      }
      return item;
    }),
  );
}

function metadataStatus(
  pending,
  message = "",
  status = pending ? RefreshStatus.PENDING : RefreshStatus.IDLE,
) {
  metadataPending = pending;
  byId("show-metadata").setAttribute("aria-busy", String(pending));
  byId("refresh-metadata").disabled = pending;
  byId("refresh-metadata").setAttribute("aria-busy", String(pending));
  refreshButtonStatus("show-metadata", status, message);
  refreshButtonStatus("refresh-metadata", status, message);
  byId("metadata-status").textContent = message;
  byId("metadata-status").title = message;
  if (gesture) gesture.viewport = undefined;
  if (svg.hasAttribute("viewBox")) fit();
}

function acceptMetadata(metadata) {
  const updates =
    countUpdates(graphMetadata?.model ?? [], metadata.model, "key") +
    countUpdates(graphMetadata?.graph ?? [], metadata.graph, "key") +
    countUpdates(graphMetadata?.nodes ?? [], metadata.nodes, "id") +
    countUpdates(graphMetadata?.values ?? [], metadata.values, "name");
  graphMetadata = metadata;
  nodeMetadata = new Map(metadata.nodes.map((node) => [node.id, node]));
  valueMetadata = new Map(metadata.values.map((value) => [value.name, value]));
  metadataStatus(
    false,
    `Metadata refreshed: ${updates} update${updates === 1 ? "" : "s"}`,
    updates ? RefreshStatus.UPDATED : RefreshStatus.UNCHANGED,
  );
  byId("metadata-content").hidden = false;
  metadataEntries(byId("metadata-model"), metadata.model);
  metadataEntries(byId("metadata-graph"), metadata.graph);
  for (const label of unusedLabels) label.remove();
  unusedLabels = [];
  for (const { id, unusedOutputs } of metadata.nodes) {
    addUnusedLabel(nodeViews.get(id), unusedOutputs);
  }
  inspectMetadata();
}

function addUnusedLabel(view, unusedOutputs = []) {
  if (!view?.group || !unusedOutputs.length) return;
  const text = label(
    view.group,
    view.initial.x,
    view.initial.y - view.halfHeight + 17,
    `Unused: ${unusedOutputs.length}`,
    "unused-label",
  );
  text.append(
    element("title", {}, `Unused outputs: ${unusedOutputs.join(", ")}`),
  );
  text.setAttribute("visibility", metadataVisible ? "visible" : "hidden");
  unusedLabels.push(text);
}

function requestMetadata() {
  if (metadataPending) return;
  metadataStatus(true, "Analyzing metadata…");
  byId("metadata-content").hidden = !graphMetadata;
  requestAction("metadata", "Refresh metadata");
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
      for (const [index, graph] of (parameter.graphs ?? []).entries()) {
        const button = document.createElement("button");
        button.className = "icon-button";
        button.textContent = `Open ${subgraphLabel(parameter, index)}: ${graph.name} (${graph.nodeCount} nodes)`;
        button.addEventListener("click", () => navigate(graph.path));
        item.append(button);
      }
      return item;
    }),
  );
}

function updateSelection(activeId, forceInspect = false) {
  for (const id of new Set([...renderedSelection, ...selected])) {
    const isSelected = selected.has(id);
    if (isSelected === renderedSelection.has(id)) continue;
    updateNodeSelection(nodeViews.get(id), isSelected);
  }
  renderedSelection = new Set(selected);
  const view = nodeViews.get(activeId ?? [...selected].at(-1));
  if (forceInspect || view !== inspectedView) inspect(view);
}

function updateNodeSelection({ group, node, subgraphs }, isSelected) {
  if (!group) return;
  group.classList.toggle("selected", isSelected);
  if (subgraphs?.length)
    group.setAttribute(
      "aria-label",
      `${isSelected ? "Selected " : ""}${node.opType}: ${node.name}`,
    );
  else group.setAttribute("aria-pressed", String(isSelected));
}

function registerNode(id, view) {
  view.id = id;
  view.initial = { ...view.position };
  nodeViews.set(id, view);
  if (!virtualized) mountNode(view);
}

function mountNode(view) {
  const { id } = view;
  view.group = view.create(view.initial);
  const { group } = view;
  group.setAttribute("tabindex", "0");
  group.setAttribute("role", view.subgraphs?.length ? "group" : "button");
  if (!view.subgraphs?.length) group.setAttribute("aria-pressed", "false");
  group.setAttribute("data-id", id);
  updateNodeSelection(view, selected.has(id));
  group.setAttribute(
    "transform",
    `translate(${view.position.x - view.initial.x} ${view.position.y - view.initial.y})`,
  );
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
  addUnusedLabel(view, nodeMetadata.get(id)?.unusedOutputs);
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
  if (!connection.group) return;
  const { from, to, fromHeight, toHeight, path, text, information } =
    connection;
  const y1 = from.y + fromHeight;
  const y2 = to.y - toHeight;
  const middle = (y1 + y2) / 2;
  path.setAttribute("d", connectionPath(connection));
  text.setAttribute("x", (from.x + to.x) / 2 + 10);
  text.setAttribute("y", middle);
  information.setAttribute("x", (from.x + to.x) / 2 + 10);
  information.setAttribute("y", middle + 14);
}

function setVisible(view, group, visible) {
  if (virtualized) {
    visible &&= !overviewVisible;
    if (visible) {
      if (!view.group) {
        if (view.node || view.port) mountNode(view);
        else view.create();
      }
      if (!view.group.parentNode)
        byId(view.node || view.port ? "nodes" : "edges").append(view.group);
    } else {
      view.group?.remove();
    }
    view.visible = visible;
    return;
  }
  if (view.visible === visible) return;
  view.visible = visible;
  group.setAttribute("display", visible ? "inline" : "none");
}

function intersectsCamera(left, top, right, bottom) {
  return (
    !camera ||
    (right >= camera.x - 240 &&
      left <= camera.x + camera.width + 240 &&
      bottom >= camera.y - 80 &&
      top <= camera.y + camera.height + 80)
  );
}

function updateNodeVisibility(view) {
  const { position, halfWidth, halfHeight } = view;
  setVisible(
    view,
    view.group,
    intersectsCamera(
      position.x - halfWidth,
      position.y - halfHeight,
      position.x + halfWidth,
      position.y + halfHeight,
    ),
  );
}

function updateConnectionVisibility(connection) {
  const { from, to, fromHeight, toHeight } = connection;
  const visible = intersectsCamera(
    Math.min(from.x, to.x),
    Math.min(from.y + fromHeight, to.y - toHeight),
    Math.max(from.x, to.x) + 240,
    Math.max(from.y + fromHeight, to.y - toHeight) + 28,
  );
  const middle = (from.y + fromHeight + to.y - toHeight) / 2;
  const labelsVisible =
    !virtualized ||
    intersectsCamera(
      (from.x + to.x) / 2,
      middle,
      (from.x + to.x) / 2 + 240,
      middle + 28,
    );
  if (virtualized && !overviewVisible && visible && !labelsVisible)
    crossingConnections.add(connection);
  else crossingConnections.delete(connection);
  setVisible(connection, connection.group, visible && labelsVisible);
}

function drawCrossingConnections() {
  const normal = [];
  const released = [];
  for (const connection of crossingConnections)
    (releasedValues.has(connection.name) ? released : normal).push(
      connectionPath(connection),
    );
  for (const [id, segments] of [
    ["crossing-edges", normal],
    ["released-crossing-edges", released],
  ]) {
    const path = byId(id);
    const data = segments.join(" ");
    if (path.getAttribute("d") !== data) path.setAttribute("d", data);
  }
}

function moveNodes(origins, dx, dy) {
  const affected = new Set();
  for (const [id, origin] of origins) {
    const view = nodeViews.get(id);
    if (
      view.position.x === origin.x + dx &&
      view.position.y === origin.y + dy
    ) {
      continue;
    }
    view.position.x = origin.x + dx;
    view.position.y = origin.y + dy;
    view.group?.setAttribute(
      "transform",
      `translate(${view.position.x - view.initial.x} ${view.position.y - view.initial.y})`,
    );
    updateNodeVisibility(view);
    for (const connection of connectionsByPosition.get(view.position) ?? []) {
      affected.add(connection);
    }
  }
  for (const connection of affected) {
    updateConnection(connection);
    updateConnectionVisibility(connection);
  }
  if (origins.size) overviewDirty = true;
  if (virtualized) drawCrossingConnections();
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

function graphPoint(event, viewport) {
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
  gesture.viewport ??= byId("canvas").getBoundingClientRect();
  const point = graphPoint(event, gesture.viewport);
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
  if (readablePending) {
    readableZoom();
    return;
  }
  const scale =
    Math.max(bounds.width / viewport.width, bounds.height / viewport.height) /
    zoom;
  const width = viewport.width * scale;
  const height = viewport.height * scale;
  camera = {
    x: bounds.x + (bounds.width - width) / 2 + pan.x,
    y: bounds.y + (bounds.height - height) / 2 + pan.y,
    width,
    height,
  };
  svg.classList.toggle("overview", scale > 2);
  overviewVisible = virtualized && scale > 2;
  byId("graph-overview").setAttribute(
    "display",
    overviewVisible ? "inline" : "none",
  );
  if (overviewVisible && overviewDirty) drawOverview();
  for (const view of nodeViews.values()) updateNodeVisibility(view);
  for (const connection of connections) updateConnectionVisibility(connection);
  drawCrossingConnections();
  svg.setAttribute("viewBox", `${camera.x} ${camera.y} ${width} ${height}`);
}

function connectionPath({ from, to, fromHeight, toHeight }) {
  const y1 = from.y + fromHeight;
  const y2 = to.y - toHeight;
  const middle = (y1 + y2) / 2;
  return `M${from.x},${y1} C${from.x},${middle} ${to.x},${middle} ${to.x},${y2}`;
}

function drawOverview() {
  byId("overview-edges").setAttribute(
    "d",
    connections.map(connectionPath).join(" "),
  );
  byId("overview-nodes").setAttribute(
    "d",
    [...nodeViews.values()]
      .map(
        ({ position, halfWidth, halfHeight }) =>
          `M${position.x - halfWidth},${position.y - halfHeight}h${halfWidth * 2}v${halfHeight * 2}h${-halfWidth * 2}Z`,
      )
      .join(" "),
  );
  overviewDirty = false;
}

function changeZoom(factor) {
  endGesture();
  zoom = Math.max(0.25, zoom * factor);
  fit();
}

function readableZoom() {
  if (!hasModel) return;
  const viewport = byId("canvas").getBoundingClientRect();
  if (!viewport.width || !viewport.height) {
    readablePending = true;
    return;
  }
  readablePending = false;
  endGesture();
  zoom = Math.max(
    bounds.width / viewport.width,
    bounds.height / viewport.height,
  );
  const position = inspectedView?.position ?? {
    x: bounds.x + bounds.width / 2 + pan.x,
    y: bounds.y + bounds.height / 2 + pan.y,
  };
  pan = {
    x: position.x - bounds.x - bounds.width / 2,
    y: position.y - bounds.y - bounds.height / 2,
  };
  fit();
}

function findNode(event) {
  event.preventDefault();
  const query = byId("node-search").value.trim().toLowerCase();
  const matches = query
    ? [...nodeViews.values()].filter(
        ({ node }) =>
          node &&
          (node.name.toLowerCase().includes(query) ||
            node.opType.toLowerCase().includes(query)),
      )
    : [];
  if (!matches.length) {
    byId("node-search-result").textContent = query
      ? "No matching node"
      : "Enter a name or operator";
    return;
  }
  const index = (matches.indexOf(inspectedView) + 1) % matches.length;
  const view = matches[index];
  selected = new Set([view.id]);
  updateSelection(view.id);
  readableZoom();
  byId("node-search-result").textContent = `${index + 1} / ${matches.length}`;
}

function navigate(path) {
  endGesture();
  byId("navigation-status").textContent = "Opening graph…";
  svg.setAttribute("aria-busy", "true");
  fit();
  requestAction("navigate", "Navigate graph", { path });
}

function render(graph, metadata) {
  hasModel = true;
  byId("node-search-result").textContent = "";
  if (!replacementPending) setLoading(false);
  endGesture();
  refreshStatus(false);
  svg.setAttribute("aria-busy", "false");
  byId("navigation-status").textContent = "";
  const breadcrumbs = graph.breadcrumbs ?? [];
  byId("graph-navigation").hidden = breadcrumbs.length < 2;
  byId("graph-navigation").replaceChildren(
    ...(breadcrumbs.length > 1 ? breadcrumbs : []).map((crumb, index) => {
      const button = document.createElement("button");
      button.className = "icon-button";
      button.textContent = crumb.name;
      if (index === breadcrumbs.length - 1) {
        button.disabled = true;
        button.setAttribute("aria-current", "page");
      } else {
        button.addEventListener("click", () => navigate(crumb.path));
      }
      return button;
    }),
  );
  resetDocumentation();
  inspectedNode = undefined;
  inspectedView = undefined;
  tensorLabels = [];
  nodeViews = new Map();
  selected = new Set();
  renderedSelection = new Set();
  connectionsByPosition = new Map();
  connections = [];
  camera = undefined;
  virtualized =
    graph.nodes.length + graph.inputs.length + graph.outputs.length > 1000;
  overviewVisible = false;
  overviewDirty = true;
  crossingConnections = new Set();
  readablePending = false;
  byId("graph-overview").setAttribute("display", "none");
  byId("overview-edges").removeAttribute("d");
  byId("overview-nodes").removeAttribute("d");
  hasOperators = graph.nodes.length > 0;
  graphMetadata = undefined;
  nodeMetadata = new Map();
  valueMetadata = new Map();
  connectionsByName = new Map();
  releasedValues = new Set();
  unusedLabels = [];
  byId("metadata-content").hidden = true;
  metadataStatus(false);
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
  const sizes = operatorSizes(graph.nodes);
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
    height: Math.max(
      650,
      250 +
        [...sizes.values()].reduce((sum, size) => sum + size.height + 51, 0),
    ),
  };
  layoutBounds = bounds;
  let rowTop = 133;
  graph.nodes.forEach((node) => {
    const size = sizes.get(node.id);
    positions.set(node.id, {
      x: randomNodeX(size.width / 2),
      y: rowTop + size.height / 2,
    });
    rowTop += size.height + 51;
  });
  const addPort = (name, index, output) => {
    const count = output ? graph.outputs.length : graph.inputs.length;
    const position = {
      x: ((index + 1) * bounds.width) / (count + 1),
      y: output ? bounds.height - 60 : 60,
    };
    const create = (position) => {
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
        position.y - 3,
        shortName(name, 16),
        "port-name",
      );
      tensorLabel(group, position.x, position.y + 15, name, 18);
      return group;
    };
    const id = `${output ? "output" : "input"}-${index}`;
    registerNode(id, {
      port: { name, output },
      create,
      position,
      halfWidth: 60,
      halfHeight: 26,
    });
    if (!virtualized) nodes.append(nodeViews.get(id).group);
    return position;
  };
  const inputPositions = new Map(
    graph.inputs.map((name, index) => [name, addPort(name, index, false)]),
  );
  const connect = (from, to, name, fromHeight, toHeight) => {
    if (!from || !to) return;
    const connection = {
      from,
      to,
      name,
      fromHeight,
      toHeight,
      create() {
        const group = element("g", {});
        const path = element("path", { class: "edge" });
        group.append(path);
        const text = label(
          group,
          0,
          0,
          shortName(name, 15),
          "tensor-label",
          "start",
        );
        const information = tensorLabel(group, 0, 0, name, 34, "start");
        Object.assign(connection, {
          group,
          from,
          to,
          fromHeight,
          toHeight,
          path,
          text,
          information,
        });
        updateConnection(connection);
        markReleased(connection, releasedValues.has(name));
      },
    };
    if (!virtualized) {
      connection.create();
      edges.append(connection.group);
    }
    connections.push(connection);
    if (!connectionsByName.has(name)) connectionsByName.set(name, []);
    connectionsByName.get(name).push(connection);
    for (const position of new Set([from, to])) {
      if (!connectionsByPosition.has(position)) {
        connectionsByPosition.set(position, new Set());
      }
      connectionsByPosition.get(position).add(connection);
    }
  };
  for (const node of graph.nodes) {
    const position = positions.get(node.id);
    const size = sizes.get(node.id);
    const create = (position) => {
      const top = position.y - size.height / 2;
      const group = element("g", {
        class: "graph-node",
        "aria-label": `${node.opType}: ${node.name}`,
      });
      group.append(
        element("title", {}, `${node.name} (${node.opType})`),
        element("rect", {
          x: position.x - size.width / 2,
          y: top,
          width: size.width,
          height: size.height,
          rx: 12,
        }),
      );
      const typeLabel = label(
        group,
        position.x,
        top + 41,
        node.opType,
        "operator-label",
      );
      typeLabel.style.fontSize = `${size.typeFontSize}px`;
      const nameLabel = label(
        group,
        position.x,
        top + 61,
        node.name,
        "operator-name",
      );
      nameLabel.style.fontSize = `${size.nameFontSize}px`;
      if (node.outputs.length) {
        const information = tensorLabel(
          group,
          position.x,
          top + 81,
          node.outputs,
          26,
        );
        information.classList.add("node-tensor-info");
      }
      size.initializers.forEach((parameter, index) => {
        const text = label(
          group,
          position.x - size.width / 2 + 12,
          top + 99 + index * 18,
          parameter.text,
          "tensor-info node-initializer",
          "start",
        );
        text.style.fontSize = `${Math.min(9, (10 * (size.width - 24)) / (parameter.textWidth || 1))}px`;
        text.setAttribute("aria-label", parameter.description);
        text.append(element("title", {}, parameter.description));
      });
      size.subgraphs.forEach((graph, index) => {
        const y = top + 94 + size.initializers.length * 18 + index * 22;
        const link = element("g", {
          class: "subgraph-link",
          role: "button",
          tabindex: "0",
          "aria-label": `Open ${graph.label}: ${graph.name}`,
        });
        link.append(
          element("title", {}, `${graph.name} (${graph.nodeCount} nodes)`),
          element("rect", {
            x: position.x - size.width / 2 + 3,
            y,
            width: size.width - 6,
            height: 21,
            rx: 4,
          }),
        );
        const text = label(
          link,
          position.x - size.width / 2 + 12,
          y + 15,
          graph.label,
          "subgraph-label",
          "start",
        );
        text.style.fontSize = `${Math.min(10, (10 * (size.width - 48)) / (graph.textWidth || 1))}px`;
        label(
          link,
          position.x + size.width / 2 - 12,
          y + 15,
          ">",
          "subgraph-label",
          "end",
        );
        link.addEventListener("pointerdown", (event) =>
          event.stopPropagation(),
        );
        link.addEventListener("click", (event) => {
          event.stopPropagation();
          if (svg.getAttribute("aria-busy") !== "true") navigate(graph.path);
        });
        link.addEventListener("keydown", (event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            event.stopPropagation();
            if (svg.getAttribute("aria-busy") !== "true") navigate(graph.path);
          }
        });
        group.append(link);
      });
      return group;
    };
    registerNode(node.id, {
      node,
      create,
      position,
      halfWidth: size.width / 2,
      halfHeight: size.height / 2,
      subgraphs: size.subgraphs,
    });
    if (!virtualized) nodes.append(nodeViews.get(node.id).group);
    for (const input of [...node.inputs, ...(node.captures ?? [])]) {
      const producer = producers.get(input);
      connect(
        producer ? positions.get(producer) : inputPositions.get(input),
        position,
        input,
        producer ? sizes.get(producer).height / 2 : 26,
        size.height / 2,
      );
    }
  }
  graph.outputs.forEach((name, index) => {
    const producer = producers.get(name);
    connect(
      producer ? positions.get(producer) : inputPositions.get(name),
      addPort(name, index, true),
      name,
      producer ? sizes.get(producer).height / 2 : 26,
      26,
    );
  });
  overviewDirty = true;
  zoom = 1;
  pan = { x: 0, y: 0 };
  if (graph.nodes.length) selected.add(graph.nodes[0].id);
  updateTensorInformation(graph.tensors ?? []);
  updateSelection(undefined, true);
  if (virtualized) readableZoom();
  else fit();
  byId("refresh-positions").disabled = false;
  if (metadata) acceptMetadata(metadata);
  else if (metadataVisible) requestMetadata();
}

svg.addEventListener("pointerdown", (event) => {
  if (
    event.button !== 0 ||
    gesture ||
    !svg.hasAttribute("viewBox") ||
    svg.getAttribute("aria-busy") === "true"
  )
    return;
  const viewport = byId("canvas").getBoundingClientRect();
  const start = graphPoint(event, viewport);
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
    viewport,
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
  requestAction("open", "Open ONNX model"),
);
byId("open-backend-test").addEventListener("click", () =>
  requestAction("open-backend-test", "Open backend test"),
);
byId("show-metadata").addEventListener("click", () => {
  metadataVisible = !metadataVisible;
  byId("show-metadata").setAttribute("aria-expanded", String(metadataVisible));
  byId("node-details").hidden = metadataVisible;
  byId("metadata-panel").hidden = !metadataVisible;
  byId("metadata-panel").parentElement.scrollTop = 0;
  for (const label of unusedLabels)
    label.setAttribute("visibility", metadataVisible ? "visible" : "hidden");
  inspectMetadata();
  if (metadataVisible && !graphMetadata) requestMetadata();
});
byId("refresh-metadata").addEventListener("click", requestMetadata);
byId("refresh-shapes").addEventListener("click", () => {
  refreshStatus(true, "Inferring shapes/types…");
  requestAction("refresh-shapes", "Refresh shapes");
});
byId("refresh-positions").addEventListener("click", () => {
  const entry = newAction("Refresh positions");
  try {
    refreshPositions();
    entry.status = "SUCCEEDED";
  } catch (error) {
    entry.status = "FAILED";
    entry.details =
      error instanceof Error ? (error.stack ?? error.message) : String(error);
  }
  displayLog(entry);
  vscode.postMessage({ type: "interaction", entry: { ...entry } });
});
byId("zoom-in").addEventListener("click", () => {
  changeZoom(1.25);
});
byId("zoom-out").addEventListener("click", () => {
  changeZoom(1 / 1.25);
});
byId("zoom-readable").addEventListener("click", readableZoom);
byId("find-node").addEventListener("submit", findNode);
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
  if (event.data?.type === "application-status") {
    displayStatus(event.data);
  } else if (event.data?.type === "log-entry") {
    displayLog(event.data.entry);
  } else if (event.data?.type === "log-history") {
    for (const entry of event.data.entries) displayLog(entry);
    const panel = byId("logs-view");
    if (!panel.hidden) panel.scrollTop = panel.scrollHeight;
  } else if (event.data?.type === "show-logs") {
    showTab("logs", true);
  } else if (event.data?.type === "load-state") {
    setLoading(
      event.data.pending,
      event.data.error ? `Could not open model: ${event.data.error}` : "",
    );
  } else if (event.data?.type === "load-error") {
    replacementPending = false;
    for (const [id, type] of pendingActions) {
      if (type === "open" || type === "open-backend-test")
        pendingActions.delete(id);
    }
    setLoading(false, `Could not open model: ${event.data.error}`);
  } else if (event.data?.type === "graph") {
    render(event.data.graph, event.data.metadata);
  } else if (event.data?.type === "navigation-error") {
    svg.setAttribute("aria-busy", "false");
    byId("navigation-status").textContent =
      `Could not open graph: ${event.data.error}`;
    fit();
  } else if (event.data?.type === "metadata-info") {
    acceptMetadata(event.data.metadata);
  } else if (event.data?.type === "metadata-error") {
    metadataStatus(
      false,
      `Metadata analysis failed: ${event.data.error}`,
      RefreshStatus.FAILED,
    );
  } else if (event.data?.type === "shape-info") {
    const updates = updateTensorInformation(event.data.tensors);
    inspect(inspectedView);
    refreshStatus(
      false,
      `Shapes/types refreshed: ${updates} tensor${updates === 1 ? "" : "s"} updated`,
      updates ? RefreshStatus.UPDATED : RefreshStatus.UNCHANGED,
    );
  } else if (event.data?.type === "shape-error") {
    refreshStatus(
      false,
      `Shape/type inference failed: ${event.data.error}`,
      RefreshStatus.FAILED,
    );
  }
});
setLoading(true);
vscode.postMessage({ type: "ready" });
