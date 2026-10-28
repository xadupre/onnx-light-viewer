/* global acquireVsCodeApi, createSidebar, createFilePanels, createInspector,
          createMetadataPanel, createFusionPanel, createSettingsPanel, createExportPanel,
          createMemoryPanel, createTensorListPanel, createTensorStatisticsPanel, sugiyamaLayout, graphPathSegments,
          flattenGraphPath, graphPointDistance */
const vscode = acquireVsCodeApi();
const svgNS = "http://www.w3.org/2000/svg";
const byId = (id) => document.getElementById(id);
const svg = byId("graph");
let bounds = { x: 0, y: 0, width: 800, height: 350 };
let layoutBounds = bounds;
let zoom = 1;
let pan = { x: 0, y: 0 };
let nodeViews = new Map();
let viewsByPosition = new WeakMap();
let searchQuery;
let searchMatches = [];
let selected = new Set();
let renderedSelection = new Set();
let connectionsByPosition = new Map();
let connections = [];
let selectedConnection;
let camera;
let panFrame;
let virtualized = false;
let overviewVisible = false;
let overviewDirty = true;
let crossingConnections = new Set();
let readablePending = false;
let gesture;
let zoomAreaMode = false;
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
let fusionUndoCount = 0;
let displayedGraph;
let graphConstants = [];
let constantsByOutput = new Map();
let statisticsRequest = 0;
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
const tabs = [
  "model",
  "logs",
  "statistics",
  "memory",
  "tensors",
  "status",
  "controls",
  "repositories",
  "settings",
];

const sidebar = createSidebar({
  byId,
  fit,
  endGesture,
  onMetadataToggle: setMetadataVisible,
  onSettingsChange: (settings) => {
    settingsPanel.restoreViewer(settings);
    vscode.postMessage({ type: "viewer-settings", settings });
  },
});
const settingsPanel = createSettingsPanel({
  byId,
  postMessage: (message) => vscode.postMessage(message),
  onViewerSettings: (settings) => {
    sidebar.restoreSettings(settings);
    vscode.postMessage({ type: "viewer-settings", settings });
  },
});
const inspector = createInspector({
  byId,
  tensorDescription,
  subgraphLabel,
  navigate,
  openTensorStatistics: (name) => openTensorStatistics([name]),
  hasStoredTensor: (name) => tensorListPanel.hasStoredTensor(name),
  nodeInitializers: (node) => tensorListPanel.nodeInitializers(node),
});
const tensorStatisticsPanel = createTensorStatisticsPanel({
  byId,
  postMessage: (message) => vscode.postMessage(message),
});
const tensorListPanel = createTensorListPanel({
  byId,
  openStatistics: openTensorStatistics,
});

function openTensorStatistics(names) {
  if (
    !hasModel ||
    replacementPending ||
    byId("model-view").getAttribute("aria-busy") === "true" ||
    svg.getAttribute("aria-busy") === "true"
  )
    return;
  tensorStatisticsPanel.open(
    names,
    displayedGraph?.breadcrumbs?.at(-1)?.path ?? [],
  );
}

const metadataPanel = createMetadataPanel({ byId, tensorDescription });
const memoryPanel = createMemoryPanel({
  byId,
  postMessage: (message) => vscode.postMessage(message),
  formatFileSize,
  selectNode: (path, index) => {
    if (replacementPending || svg.getAttribute("aria-busy") === "true") return;
    showTab("model", true);
    const currentPath = displayedGraph.breadcrumbs?.at(-1)?.path ?? [];
    if (JSON.stringify(currentPath) === JSON.stringify(path ?? []))
      focusNodes([index]);
    else navigate(path ?? [], index);
  },
});
const filePanels = createFilePanels({
  byId,
  postMessage: (message) => vscode.postMessage(message),
  requestAction,
  revealPanel: sidebar.revealPanel,
  isReplacing: () => replacementPending,
});
const exportPanel = createExportPanel({
  byId,
  postMessage: (message) => vscode.postMessage(message),
  requestAction,
  revealPanel: sidebar.revealPanel,
});
const fusionPanel = createFusionPanel({
  byId,
  postMessage: (message) => vscode.postMessage(message),
  requestAction,
  revealPanel: sidebar.revealPanel,
  fit,
  canFuse: () =>
    !replacementPending &&
    hasModel &&
    (displayedGraph?.canFuse ?? displayedGraph?.breadcrumbs?.length === 1),
  clearSelection: () => {
    selected = new Set();
    updateSelection();
  },
  selectedNode: () =>
    selected.size === 1 ? nodeViews.get([...selected][0])?.node : undefined,
  focusNodes,
});

function focusNodes(nodes) {
  const visible = nodes
    .map((index) => `node-${index}`)
    .filter((id) => nodeViews.has(id));
  selected = new Set(visible);
  updateSelection(visible[0]);
  if (visible.length) readableZoom();
}

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
  else if (name === "statistics") requestStatistics();
  else if (name === "memory") memoryPanel.open();
  else if (name === "tensors") {
    const names = [...selected].flatMap((id) => {
      const view = nodeViews.get(id);
      return view?.node
        ? [
            ...view.node.inputs,
            ...(view.node.captures ?? []),
            ...view.node.outputs,
          ]
        : view?.port
          ? [view.port.name]
          : [];
    });
    if (selectedConnection) names.push(selectedConnection.name);
    tensorListPanel.open(names);
  } else if (name === "settings") vscode.postMessage({ type: "get-settings" });
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
    if (type === "open" || type === "export") {
      replacementPending = [...pendingActions.values()].some(
        (pending) => pending === "open" || pending === "export",
      );
      setLoading(replacementPending);
      if (type === "open") {
        filePanels.finishOpening(entry.status);
        if (entry.status === "FAILED")
          showLoadError(
            entry.details || "Opening the model failed. See Logs for details.",
          );
      } else exportPanel.finish(entry.status);
    } else if (type === "save") {
      filePanels.finishSaving(entry.status, replacementPending || !hasModel);
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
  if (
    ["open", "export", "navigate", "undo-fusion", "select-fusion"].includes(
      type,
    )
  )
    fusionPanel.invalidate();
  const entry = newAction(action);
  pendingActions.set(entry.id, type);
  if (type === "open" || type === "export") {
    if (type === "export") exportPanel.start(entry.id);
    replacementPending = true;
    setLoading(true);
  } else if (type === "save") {
    filePanels.startSaving();
  }
  vscode.postMessage({ type, ...parameters, actionId: entry.id });
}

function setLoading(pending, message = "") {
  if (pending) tensorStatisticsPanel.reset();
  tensorListPanel.setBusy(pending);
  if (pending) byId("model-error").hidden = true;
  byId("load-progress").hidden = !pending;
  byId("model-view").setAttribute("aria-busy", String(pending));
  filePanels.setLoading(pending, hasModel);
  exportPanel.setLoading(pending);
  for (const id of [
    "refresh-shapes",
    "refresh-metadata",
    "refresh-positions",
    "verify-model",
  ])
    byId(id).disabled =
      pending || !hasModel || byId(id).getAttribute("aria-busy") === "true";
  for (const id of ["undo-fusion", "unfuse-last", "unfuse-all"])
    byId(id).disabled =
      pending ||
      !hasModel ||
      !fusionUndoCount ||
      byId("undo-fusion").getAttribute("aria-busy") === "true";
  for (const button of byId("unfuse-history").querySelectorAll("button"))
    button.disabled =
      pending ||
      byId("undo-fusion").getAttribute("aria-busy") === "true" ||
      button.dataset.nodeCount === "0";
  byId("load-status").textContent =
    message || (pending ? "Loading model…" : "");
  fusionPanel.setEnabled(
    !pending &&
      hasModel &&
      (displayedGraph?.canFuse ?? displayedGraph?.breadcrumbs?.length === 1),
  );
}

function showLoadError(error) {
  replacementPending = false;
  for (const [id, type] of pendingActions) {
    if (type === "open") pendingActions.delete(id);
  }
  setLoading(false, `Could not open model: ${error}`);
  byId("model-error-details").textContent = error;
  byId("model-error").hidden = false;
  filePanels.showOpenError(error);
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

function statisticsStatus(message, pending = false) {
  byId("statistics-view").setAttribute("aria-busy", String(pending));
  byId("statistics-empty").textContent = message;
  byId("statistics-empty").hidden = false;
  byId("statistics-content").hidden = true;
}

function requestStatistics() {
  const requestId = ++statisticsRequest;
  statisticsStatus(
    hasModel
      ? "Updating model statistics…"
      : "Open a model to see its statistics.",
    hasModel,
  );
  if (hasModel) vscode.postMessage({ type: "get-statistics", requestId });
}

function statisticsTable(id, columns, rows) {
  const table = byId(id);
  table.replaceChildren();
  const head = table.createTHead().insertRow();
  for (const column of columns) {
    const cell = document.createElement("th");
    cell.scope = "col";
    cell.textContent = column;
    head.append(cell);
  }
  const body = table.createTBody();
  for (const values of rows) {
    const row = body.insertRow();
    for (const [index, value] of values.entries()) {
      const cell = document.createElement(index === 0 ? "th" : "td");
      if (index === 0) cell.scope = "row";
      cell.textContent = value;
      row.append(cell);
    }
  }
}

function formatFileSize(size) {
  if (!Number.isSafeInteger(size) || size < 0) return "Unavailable";
  if (size < 1024) return `${size.toLocaleString()} bytes`;
  const units = ["KiB", "MiB", "GiB", "TiB"];
  let value = size;
  let unit = -1;
  do {
    value /= 1024;
    unit++;
  } while (value >= 1024 && unit < units.length - 1);
  return `${value.toFixed(1)} ${units[unit]} (${size.toLocaleString()} bytes)`;
}

function renderStatistics(graph, files) {
  byId("statistics-empty").hidden = Boolean(graph);
  byId("statistics-content").hidden = !graph;
  if (!graph) return;

  const stages = graph.pipeline
    ? graph.nodes.flatMap((node) =>
        node.pipelineModel ? [{ ...node.pipelineModel, name: node.name }] : [],
      )
    : [];
  const pipeline = stages.length > 0;
  const availableStages = stages.filter(
    (stage) => stage.statisticsAvailable !== false,
  );
  const total = (field, fallback) =>
    pipeline
      ? stages.reduce((sum, stage) => sum + (stage[field] ?? 0), 0)
      : fallback;
  const detailedTotal = (field, fallback) => {
    if (!pipeline) return fallback;
    const count = availableStages.reduce(
      (sum, stage) => sum + (stage[field] ?? 0),
      0,
    );
    if (availableStages.length === stages.length) return count;
    return availableStages.length
      ? `Partial: ${count.toLocaleString()}`
      : "Unavailable";
  };
  const summary = byId("statistics-summary");
  summary.replaceChildren();
  const entries = [
    ["Nodes", total("nodeCount", graph.nodeCount ?? graph.nodes.length)],
    ["Initializers", total("initializerCount", graph.initializerCount)],
    [
      "Dense initializers",
      detailedTotal(
        "denseInitializerCount",
        graph.denseInitializerCount ?? graph.initializers?.length ?? 0,
      ),
    ],
    [
      "Sparse initializers",
      detailedTotal(
        "sparseInitializerCount",
        graph.sparseInitializerCount ?? graph.sparseInitializers?.length ?? 0,
      ),
    ],
    [
      "Encoded initializers",
      detailedTotal(
        "encodedInitializerCount",
        graph.encodedInitializerCount ?? graph.encodedInitializers?.length ?? 0,
      ),
    ],
    [
      "Local functions",
      detailedTotal(
        "functionCount",
        graph.functionCount ?? graph.localFunctions?.length ?? 0,
      ),
    ],
    ["Inputs", graph.inputs.length],
    ["Outputs", graph.outputs.length],
  ];
  for (const [key, value] of entries) {
    const term = document.createElement("dt");
    const description = document.createElement("dd");
    term.textContent = key;
    description.textContent =
      typeof value === "number" ? value.toLocaleString() : value;
    summary.append(term, description);
  }
  byId("statistics-model-name").textContent = graph.name || "Unnamed model";

  byId("statistics-stages").hidden = !pipeline;
  if (pipeline) {
    statisticsTable(
      "statistics-stage-table",
      ["Stage", "Nodes", "Initializers", "Functions", "Inputs", "Outputs"],
      stages.map((stage) => [
        stage.name,
        Number(stage.nodeCount ?? 0).toLocaleString(),
        Number(stage.initializerCount ?? 0).toLocaleString(),
        stage.statisticsAvailable === false
          ? "Unavailable"
          : Number(stage.functionCount ?? 0).toLocaleString(),
        Number(stage.inputs?.length ?? 0).toLocaleString(),
        Number(stage.outputs?.length ?? 0).toLocaleString(),
      ]),
    );
  }

  const operatorCounts = new Map();
  const addOperator = (domain, opType, count = 1) => {
    const key = JSON.stringify([domain, opType]);
    const current = operatorCounts.get(key);
    if (current) current.count += count;
    else operatorCounts.set(key, { domain, opType, count });
  };
  if (pipeline) {
    for (const stage of availableStages)
      for (const operator of stage.operatorTypes ?? [])
        addOperator(operator.domain, operator.opType, operator.count);
  } else if (graph.operatorTypes) {
    for (const operator of graph.operatorTypes)
      addOperator(operator.domain, operator.opType, operator.count);
  } else {
    for (const node of graph.nodes) addOperator(node.domain, node.opType);
  }
  const operators = [...operatorCounts.values()].sort(
    (left, right) =>
      right.count - left.count ||
      left.domain.localeCompare(right.domain) ||
      left.opType.localeCompare(right.opType),
  );
  statisticsTable(
    "statistics-operators",
    ["Domain", "Operator", "Count"],
    operators.map(({ domain, opType, count }) => [
      domain || "(default)",
      opType,
      count.toLocaleString(),
    ]),
  );
  const partial = availableStages.length < stages.length;
  if (!operators.length || partial) {
    const row = byId("statistics-operators").tBodies[0].insertRow();
    const cell = row.insertCell();
    cell.colSpan = 3;
    cell.textContent = partial
      ? availableStages.length
        ? "Partial operator statistics: some stages are unavailable."
        : "Operator statistics unavailable."
      : "No operators.";
  }

  const fileRows = (files ?? []).map(({ name, size, external, error }) => [
    external ? "External data" : "Model",
    name,
    error ? `Unavailable: ${error}` : formatFileSize(size),
  ]);
  statisticsTable(
    "statistics-files",
    ["File", "Name", "Size"],
    fileRows.length
      ? fileRows
      : [["Source", "No source file available", "Unavailable"]],
  );
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

function shortName(value, length = 22, middle = false) {
  if (value.length <= length) return value;
  if (!middle) return `${value.slice(0, length - 1)}…`;
  const start = Math.ceil((length - 1) / 2);
  const end = length - 1 - start;
  return `${value.slice(0, start)}…${end ? value.slice(-end) : ""}`;
}

function subgraphLabel(attribute, index) {
  return `${attribute.name}${attribute.graphs.length > 1 ? ` [${index}]` : ""}`;
}

function bindGraphLink(link, activate) {
  let suppressPointerClick = false;
  link.addEventListener("pointerdown", (event) => {
    suppressPointerClick = zoomAreaMode || event.altKey;
    if (!suppressPointerClick) event.stopPropagation();
  });
  link.addEventListener("click", (event) => {
    event.stopPropagation();
    if (suppressPointerClick) {
      suppressPointerClick = false;
      if (event.detail > 0) return;
    }
    if (svg.getAttribute("aria-busy") !== "true") activate();
  });
  link.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      event.stopPropagation();
      if (svg.getAttribute("aria-busy") !== "true") activate();
    }
  });
}

function operatorSizes(nodes) {
  const context = document.createElement("canvas").getContext("2d");
  const family = getComputedStyle(svg).fontFamily;
  const widths = new Map();
  context.font = `600 16px ${family}`;
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
      const parameters = [
        ...(node.attributes ?? [])
          .filter((attribute) => !attribute.graphs?.length)
          .map((attribute) => ({
            description: `${attribute.name} = ${attribute.value}`,
            className: "node-attribute",
          })),
        ...(node.parameters ?? []).map((parameter) => ({
          description: `${parameter.name}: ${parameter.type}${parameter.shape === undefined ? "" : ` [${parameter.shape?.join(", ") ?? "?"}]`}${parameter.value === undefined ? "" : ` = ${parameter.value}`}`,
          className: "node-initializer",
        })),
      ];
      const compact =
        parameters.length <= 2 &&
        parameters.every(
          ({ description }) =>
            description.length <= 38 && !/[\r\n]/.test(description),
        );
      const parameterRows = (
        compact
          ? parameters
          : [
              {
                description: `${parameters.length} parameter${parameters.length === 1 ? "" : "s"} (see inspector)`,
                className: "node-parameter-summary",
              },
            ]
      ).map((parameter) => ({
        ...parameter,
        textWidth: context.measureText(parameter.description).width,
      }));
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
      if (node.function) {
        const text = "Function";
        subgraphs.push({
          ...node.function,
          label: text,
          textWidth: context.measureText(text).width,
        });
      }
      const width = Math.min(
        280,
        Math.max(
          140,
          Math.ceil(Math.max(typeWidth, nameWidth) + 24),
          ...parameterRows.map((parameter) =>
            Math.ceil(parameter.textWidth + 24),
          ),
          ...subgraphs.map((graph) => Math.ceil(graph.textWidth + 68)),
        ),
      );
      let nameLength = node.name.length;
      if (nameWidth > width - 24) {
        let low = 1;
        let high = nameLength;
        while (low < high) {
          const length = Math.ceil((low + high) / 2);
          const name = shortName(node.name, length, true);
          if (context.measureText(name).width <= width - 24) low = length;
          else high = length - 1;
        }
        nameLength = low;
      }
      return [
        node.id,
        {
          width,
          height: 88 + parameterRows.length * 16 + subgraphs.length * 20,
          parameterRows,
          subgraphs,
          displayName: shortName(node.name, nameLength, true),
          typeFontSize: Math.min(16, (16 * (width - 24)) / (typeWidth || 1)),
        },
      ];
    }),
  );
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
  tensorListPanel.updateTypes(tensors);
  for (const entry of tensorLabels) updateTensorLabel(entry);
  return updates;
}

function refreshPositions() {
  endGesture();
  const positions = new Map();
  for (const [id, view] of nodeViews) {
    positions.set(id, { ...view.initial });
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

function verificationStatus(
  pending,
  message = "",
  status = pending ? RefreshStatus.PENDING : RefreshStatus.IDLE,
) {
  byId("verify-model").setAttribute("aria-busy", String(pending));
  byId("verify-model").disabled = pending || replacementPending || !hasModel;
  refreshButtonStatus("verify-model", status, message);
  byId("verify-status").textContent = message;
  if (gesture) gesture.viewport = undefined;
  if (svg.hasAttribute("viewBox")) fit();
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
  metadataPanel.render(inspectedView, {
    visible: metadataVisible,
    metadata: graphMetadata,
    nodeMetadata,
    valueMetadata,
    scope: inspectionScope(),
  });
}

function metadataStatus(
  pending,
  message = "",
  status = pending ? RefreshStatus.PENDING : RefreshStatus.IDLE,
) {
  metadataPending = pending;
  metadataPanel.setPending(pending);
  byId("refresh-metadata").disabled = pending;
  byId("refresh-metadata").setAttribute("aria-busy", String(pending));
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
  metadataPanel.accept(metadata);
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
    view.initial.y - view.halfHeight + 29,
    `Unused: ${unusedOutputs.length}`,
    "unused-label",
  );
  text.append(
    element("title", {}, `Unused outputs: ${unusedOutputs.join(", ")}`),
  );
  text.setAttribute("visibility", metadataVisible ? "visible" : "hidden");
  unusedLabels.push(text);
}

function setMetadataVisible(visible) {
  metadataVisible = visible;
  for (const label of unusedLabels)
    label.setAttribute("visibility", metadataVisible ? "visible" : "hidden");
  inspectMetadata();
  if (metadataVisible && !graphMetadata && hasModel && !replacementPending)
    requestMetadata();
}

function requestMetadata() {
  if (metadataPending) return;
  metadataStatus(true, "Analyzing metadata…");
  byId("metadata-content").hidden = !graphMetadata;
  requestAction("metadata", "Refresh metadata");
}

function inspectionScope(view) {
  if (!displayedGraph) return;
  const step = displayedGraph.breadcrumbs?.at(-1)?.path.at(-1);
  const kind = step
    ? "function" in step
      ? "Function"
      : "pipeline" in step
        ? "Model"
        : "Graph"
    : displayedGraph.pipeline
      ? "Pipeline"
      : "Model";
  return {
    graph: displayedGraph,
    kind,
    name: displayedGraph.name,
    opsets: displayedGraph.opsets,
    inputs: displayedGraph.inputs,
    outputs: displayedGraph.outputs,
    initializers: displayedGraph.initializers,
    sparseInitializers: displayedGraph.sparseInitializers,
    encodedInitializers: displayedGraph.encodedInitializers,
    localFunctions: displayedGraph.localFunctions,
    constants: view
      ? [
          ...new Set(
            (view.node
              ? [...view.node.inputs, ...(view.node.captures ?? [])]
              : [view.port.name]
            )
              .map((name) => constantsByOutput.get(name))
              .filter(Boolean),
          ),
        ]
      : graphConstants,
  };
}

function inspect(view) {
  inspectedView = view;
  inspector.inspect(view, inspectionScope(view));
  inspectMetadata();
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
  fusionPanel.selectionChanged();
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
  view.searchTerms = view.node
    ? [view.node.name.toLowerCase(), view.node.opType.toLowerCase()]
    : [];
  nodeViews.set(id, view);
  viewsByPosition.set(view.position, view);
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
  selectConnection(undefined);
  if (additive && selected.has(id)) {
    selected.delete(id);
  } else {
    if (!additive && !selected.has(id)) selected.clear();
    selected.add(id);
  }
  updateSelection(selected.has(id) ? id : undefined);
}

function updateConnection(connection) {
  if (selectedConnection === connection) updateConnectionHighlight();
  if (!connection.group) return;
  const { path, hit, text, information } = connection;
  const { x, y } = connectionLabelPosition(connection);
  const data = connectionPath(connection);
  path.setAttribute("d", data);
  hit.setAttribute("d", data);
  text.setAttribute("x", x + 10);
  text.setAttribute("y", y);
  information.setAttribute("x", x + 10);
  information.setAttribute("y", y + 14);
}

function selectConnection(connection) {
  if (selectedConnection === connection) return;
  selectedConnection?.group?.setAttribute("aria-pressed", "false");
  selectedConnection = connection;
  connection?.group?.setAttribute("aria-pressed", "true");
  updateConnectionHighlight();
}

function updateConnectionHighlight() {
  let highlight = byId("connection-highlight");
  if (!selectedConnection) {
    highlight?.remove();
    return;
  }
  if (!highlight) {
    highlight = element("g", {
      id: "connection-highlight",
      "aria-hidden": "true",
    });
    svg.append(highlight);
  }
  const { from, to } = selectedConnection;
  highlight.replaceChildren(
    element("path", {
      class: "selected-edge",
      d: connectionPath(selectedConnection),
    }),
    ...[...new Set([from, to])].map((position) => {
      const { halfWidth, halfHeight } = viewsByPosition.get(position);
      return element("rect", {
        class: "connected-node",
        x: position.x - halfWidth - 3,
        y: position.y - halfHeight - 3,
        width: halfWidth * 2 + 6,
        height: halfHeight * 2 + 6,
        rx: 12,
      });
    }),
  );
}

function findConnectionAt(point, candidates, viewport) {
  const [, , width] = svg.getAttribute("viewBox").split(" ").map(Number);
  const tolerance = (10 * width) / viewport.width;
  let closest;
  let distance = tolerance;
  for (const connection of candidates) {
    if (
      point.x < connection.left - tolerance ||
      point.x > connection.right + tolerance ||
      point.y < connection.top - tolerance ||
      point.y > connection.bottom + tolerance
    )
      continue;
    const points = flattenGraphPath(
      connectionPoints(connection),
      !!connection.route,
      tolerance / 8,
    );
    for (let index = 1; index < points.length; index++) {
      const next = graphPointDistance(point, points[index - 1], points[index]);
      if (next < distance) {
        closest = connection;
        distance = next;
      }
    }
  }
  return closest;
}

function setVisible(view, group, visible) {
  if (virtualized) visible &&= !overviewVisible;
  if (view.visible === visible) return;
  view.visible = visible;
  if (virtualized) {
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
    return;
  }
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
  const points = connectionPoints(connection);
  let left = Infinity,
    top = Infinity,
    right = -Infinity,
    bottom = -Infinity;
  for (const { x, y } of points) {
    left = Math.min(left, x);
    top = Math.min(top, y);
    right = Math.max(right, x);
    bottom = Math.max(bottom, y);
  }
  Object.assign(connection, { left, top, right, bottom });
  const label = connectionLabelPosition(connection);
  const visible = intersectsCamera(
    Math.min(left, label.x),
    Math.min(top, label.y),
    Math.max(right, label.x + 240),
    Math.max(bottom, label.y + 28),
  );
  const labelsVisible =
    !virtualized ||
    intersectsCamera(label.x, label.y, label.x + 240, label.y + 28);
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
  setZoomAreaMode(false);
  if (!gesture) return;
  const previous = gesture;
  gesture = undefined;
  previous.rectangle?.remove();
  svg.classList.remove("dragging", "selecting");
  if (cancel) {
    if (previous.origins) {
      moveNodes(previous.origins, 0, 0);
    } else if (!previous.zoomArea) {
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
  if (panFrame !== undefined) fit();
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
  gesture.end = point;
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
    if (gesture.zoomArea) return;
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
  for (const connection of connections) {
    if (!connection.route) continue;
    for (const point of connectionPoints(connection)) {
      x = Math.min(x, point.x - 20);
      y = Math.min(y, point.y - 20);
      right = Math.max(right, point.x + 20);
      bottom = Math.max(bottom, point.y + 20);
    }
  }
  bounds = { x, y, width: right - x, height: bottom - y };
}

function fit() {
  if (panFrame !== undefined) {
    window.cancelAnimationFrame(panFrame);
    panFrame = undefined;
  }
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

function schedulePan() {
  if (panFrame !== undefined) return;
  panFrame = window.requestAnimationFrame(() => {
    panFrame = undefined;
    fit();
  });
}

function setZoomAreaMode(active) {
  zoomAreaMode = active;
  byId("zoom-area").setAttribute("aria-pressed", String(active));
  svg.classList.toggle("zoom-area", active);
}

function connectionPoints({ from, to, fromHeight, toHeight, route }) {
  const start = { x: from.x, y: from.y + fromHeight };
  const end = { x: to.x, y: to.y - toHeight };
  if (!route) return [start, end];
  const span = route.to.y - toHeight - route.from.y - fromHeight;
  return [
    start,
    ...route.points.map((point, index) => {
      // Translate the whole route for group drags, blend offsets for one endpoint.
      const fraction = span
        ? (point.y - route.from.y - fromHeight) / span
        : (index + 1) / (route.points.length + 1);
      return {
        x:
          point.x +
          (from.x - route.from.x) * (1 - fraction) +
          (to.x - route.to.x) * fraction,
        y:
          point.y +
          (from.y - route.from.y) * (1 - fraction) +
          (to.y - route.to.y) * fraction,
      };
    }),
    end,
  ];
}

function connectionLabelPosition(connection) {
  const points = connectionPoints(connection);
  let length = 0;
  for (let i = 1; i < points.length; i++)
    length += Math.hypot(
      points[i].x - points[i - 1].x,
      points[i].y - points[i - 1].y,
    );
  let remaining = length / 2;
  for (let i = 1; i < points.length; i++) {
    const start = points[i - 1],
      end = points[i];
    const segment = Math.hypot(end.x - start.x, end.y - start.y);
    if (remaining <= segment && segment) {
      const fraction = remaining / segment;
      return {
        x: start.x + (end.x - start.x) * fraction,
        y: start.y + (end.y - start.y) * fraction,
      };
    }
    remaining -= segment;
  }
  return points[0];
}

function connectionPath(connection) {
  if (connection.route) {
    const points = connectionPoints(connection);
    let path = `M${points[0].x},${points[0].y}`;
    for (const segment of graphPathSegments(points, true)) {
      const end = segment.at(-1);
      if (segment.length === 2) path += ` L${end.x},${end.y}`;
      else {
        const control = segment[1];
        path += ` Q${control.x},${control.y} ${end.x},${end.y}`;
      }
    }
    return path;
  }
  const { from, to, fromHeight, toHeight } = connection;
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
  const labels = byId("overview-labels");
  labels.replaceChildren();
  for (const {
    node,
    position,
    halfHeight,
    typeFontSize,
  } of nodeViews.values()) {
    if (!node) continue;
    const text = label(
      labels,
      position.x,
      position.y - halfHeight + 52,
      node.opType,
      "operator-label",
    );
    text.style.fontSize = `${typeFontSize}px`;
  }
  overviewDirty = false;
}

function resetZoom() {
  endGesture();
  reframe();
  zoom = 1;
  pan = { x: 0, y: 0 };
  fit();
}

function zoomToArea({ start, end, viewport, moved }) {
  if (!moved || !end) return;
  const width = Math.abs(end.x - start.x);
  const height = Math.abs(end.y - start.y);
  if (
    (width * viewport.width) / camera.width < 3 ||
    (height * viewport.height) / camera.height < 3
  )
    return;
  zoom = Math.max(
    0.25,
    Math.max(bounds.width / viewport.width, bounds.height / viewport.height) /
      Math.max(width / viewport.width, height / viewport.height),
  );
  pan = {
    x: (start.x + end.x) / 2 - bounds.x - bounds.width / 2,
    y: (start.y + end.y) / 2 - bounds.y - bounds.height / 2,
  };
  fit();
}

function changeZoom(factor, anchor) {
  endGesture();
  const previousZoom = zoom;
  zoom = Math.max(0.25, zoom * factor);
  if (anchor && camera) {
    const ratio = 1 - previousZoom / zoom;
    pan.x += (anchor.x - camera.x - camera.width / 2) * ratio;
    pan.y += (anchor.y - camera.y - camera.height / 2) * ratio;
  }
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

function goToBoundary(bottom) {
  if (!hasModel) return;
  const viewport = byId("canvas").getBoundingClientRect();
  if (!viewport.width || !viewport.height) return;
  endGesture();
  const scale = readablePending
    ? 1
    : Math.max(bounds.width / viewport.width, bounds.height / viewport.height) /
      zoom;
  readablePending = false;
  const width = viewport.width * scale;
  const height = viewport.height * scale;
  reframe();
  zoom = Math.max(bounds.width / width, bounds.height / height);
  pan = {
    x: 0,
    y: (bottom ? 1 : -1) * Math.max(0, (bounds.height - height) / 2),
  };
  fit();
}

function matchingNodes() {
  const query = byId("node-search").value.trim().toLowerCase();
  if (query !== searchQuery) {
    searchQuery = query;
    searchMatches = query
      ? [...nodeViews.values()].filter(({ searchTerms }) =>
          searchTerms.some((term) => term.includes(query)),
        )
      : [];
  }
  return searchMatches;
}

function selectSearchMatch(view, index, count) {
  byId("node-search-suggestions").hidden = true;
  selected = new Set([view.id]);
  updateSelection(view.id);
  readableZoom();
  byId("node-search-result").textContent = `${index + 1} / ${count}`;
}

function suggestNodes() {
  const matches = matchingNodes();
  const suggestions = byId("node-search-suggestions");
  suggestions.replaceChildren();
  suggestions.hidden = !matches.length;
  byId("node-search-result").textContent = !byId("node-search").value.trim()
    ? ""
    : matches.length
      ? `${matches.length} matching nodes${matches.length > 20 ? " (first 20 shown)" : ""}`
      : "No matching node";
  matches.slice(0, 20).forEach((view, index) => {
    const item = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = `${view.node.name || "(unnamed)"} — ${view.node.opType}`;
    button.addEventListener("click", () =>
      selectSearchMatch(view, index, matches.length),
    );
    item.append(button);
    suggestions.append(item);
  });
}

function findNode(event) {
  event.preventDefault();
  const matches = matchingNodes();
  if (!matches.length) {
    byId("node-search-result").textContent = byId("node-search").value.trim()
      ? "No matching node"
      : "Enter a name or operator";
    return;
  }
  const index = (matches.indexOf(inspectedView) + 1) % matches.length;
  selectSearchMatch(matches[index], index, matches.length);
}

function navigate(path, nodeIndex) {
  tensorStatisticsPanel.reset();
  tensorListPanel.setBusy(true);
  endGesture();
  byId("navigation-status").textContent = "Opening graph…";
  svg.setAttribute("aria-busy", "true");
  fit();
  requestAction("navigate", "Navigate graph", {
    path,
    ...(nodeIndex === undefined ? {} : { nodeIndex }),
  });
}

function render(graph, metadata) {
  graphConstants = [];
  constantsByOutput = new Map();
  const operators = [];
  for (const node of graph.nodes) {
    if (
      node.opType === "Constant" &&
      (!node.domain || node.domain === "ai.onnx") &&
      !node.function &&
      !node.pipelineModel
    ) {
      graphConstants.push(node);
      for (const output of node.outputs) constantsByOutput.set(output, node);
    } else operators.push(node);
  }
  graph = { ...graph, nodes: operators };
  byId("model-error").hidden = true;
  hasModel = true;
  byId("go-top").disabled = false;
  byId("go-bottom").disabled = false;
  searchQuery = undefined;
  searchMatches = [];
  byId("node-search-result").textContent = "";
  byId("node-search-suggestions").replaceChildren();
  byId("node-search-suggestions").hidden = true;
  if (!replacementPending) setLoading(false);
  endGesture();
  refreshStatus(false);
  verificationStatus(false);
  svg.setAttribute("aria-busy", "false");
  byId("navigation-status").textContent = "";
  const breadcrumbs = graph.breadcrumbs ?? [];
  const navigation = graph.pipelineStages?.length
    ? [breadcrumbs[0], ...graph.pipelineStages, ...breadcrumbs.slice(2)]
    : breadcrumbs;
  const currentPath = JSON.stringify(breadcrumbs.at(-1)?.path);
  byId("graph-navigation").hidden = false;
  byId("graph-navigation").replaceChildren(
    ...navigation.map((crumb) => {
      const button = document.createElement("button");
      button.className = "icon-button";
      button.textContent = crumb.name;
      if (JSON.stringify(crumb.path) === currentPath) {
        button.disabled = true;
        button.setAttribute("aria-current", "page");
      } else {
        button.addEventListener("click", () => navigate(crumb.path));
      }
      return button;
    }),
  );
  inspector.reset();
  inspectedView = undefined;
  tensorLabels = [];
  nodeViews = new Map();
  viewsByPosition = new WeakMap();
  selectConnection(undefined);
  selected = new Set();
  renderedSelection = new Set();
  connectionsByPosition = new Map();
  connections = [];
  camera = undefined;
  virtualized =
    graph.nodes.length + graph.inputs.length + graph.outputs.length >= 250;
  overviewVisible = false;
  overviewDirty = true;
  crossingConnections = new Set();
  readablePending = false;
  byId("graph-overview").setAttribute("display", "none");
  byId("overview-edges").removeAttribute("d");
  byId("overview-nodes").removeAttribute("d");
  byId("overview-labels").replaceChildren();
  graphMetadata = undefined;
  nodeMetadata = new Map();
  valueMetadata = new Map();
  connectionsByName = new Map();
  releasedValues = new Set();
  unusedLabels = [];
  metadataPanel.reset();
  metadataStatus(false);
  const tensors = new Set([
    ...graph.inputs,
    ...graph.outputs,
    ...graph.nodes.flatMap((node) => [...node.inputs, ...node.outputs]),
  ]);
  const unit =
    graph.pipeline && graph.breadcrumbs.length === 1 ? "model" : "node";
  byId("graph-summary").textContent =
    `${graph.nodes.length} ${unit}${graph.nodes.length === 1 ? "" : "s"} / ${tensors.size} tensors`;
  byId("initializer-count").textContent =
    `${graph.initializerCount} initializer${graph.initializerCount === 1 ? "" : "s"}`;

  const edges = byId("edges");
  const nodes = byId("nodes");
  edges.replaceChildren();
  nodes.replaceChildren();
  const sizes = operatorSizes(graph.nodes);
  const layout = sugiyamaLayout(graph, sizes);
  const { producers, positions } = layout;
  bounds = layout.bounds;
  layoutBounds = bounds;
  const addPort = (name, index, output) => {
    const position = (
      output ? layout.outputPositions : layout.inputPositions
    ).get(name);
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
  const connect = (from, to, name, fromHeight, toHeight, points) => {
    if (!from || !to) return;
    const connection = {
      from,
      to,
      name,
      fromHeight,
      toHeight,
      route: points?.length
        ? { points, from: { ...from }, to: { ...to } }
        : undefined,
      create() {
        const group = element("g", {
          class: "connection",
          role: "button",
          tabindex: "0",
          "aria-label": `Connection: ${name}`,
          "aria-pressed": String(selectedConnection === connection),
        });
        const hit = element("path", { class: "edge-hit" });
        const path = element("path", { class: "edge" });
        group.append(hit, path);
        group.addEventListener("pointerdown", (event) => {
          if (
            event.button !== 0 ||
            zoomAreaMode ||
            event.altKey ||
            svg.getAttribute("aria-busy") === "true"
          )
            return;
          event.preventDefault();
          event.stopPropagation();
          endGesture();
          selectConnection(connection);
          group.focus({ preventScroll: true });
        });
        group.addEventListener("click", (event) => {
          if (event.detail === 0 && !zoomAreaMode && !event.altKey)
            selectConnection(connection);
        });
        group.addEventListener("keydown", (event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            selectConnection(connection);
          }
        });
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
          hit,
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
    const storedInputs = node.pipelineModel
      ? []
      : tensorListPanel.storedInputs(node);
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
        top + 52,
        node.opType,
        "operator-label",
      );
      typeLabel.style.fontSize = `${size.typeFontSize}px`;
      const nameLabel = label(
        group,
        position.x + size.width / 2 - 12,
        top + 15,
        size.displayName,
        "operator-name",
        "end",
      );
      nameLabel.style.fontSize = "10px";
      nameLabel.setAttribute("aria-label", node.name);
      if (storedInputs.length) {
        const link = element("g", {
          class: "node-statistics-link",
          role: "button",
          tabindex: "0",
          "aria-label": `Inspect stored tensors for ${node.opType}: ${node.name}`,
        });
        const x = position.x - size.width / 2 + 8;
        const y = top + 18;
        link.append(
          element("title", {}, "Browse tensors and weight statistics"),
          element("rect", { x, y, width: 20, height: 20, rx: 4 }),
        );
        const icon = byId("tensors-tab").querySelector("svg").cloneNode(true);
        icon.setAttribute("class", "node-statistics-icon");
        icon.setAttribute("x", String(x + 1));
        icon.setAttribute("y", String(y + 1));
        link.append(icon);
        link.setAttribute("aria-controls", "tensor-statistics-dialog");
        link.setAttribute("aria-haspopup", "dialog");
        bindGraphLink(link, () => openTensorStatistics(storedInputs));
        group.append(link);
      }
      if (node.outputs.length) {
        const information = tensorLabel(
          group,
          position.x,
          top + 78 + size.parameterRows.length * 16,
          node.outputs,
          26,
        );
        information.classList.add("node-tensor-info");
      }
      size.parameterRows.forEach((parameter, index) => {
        const text = label(
          group,
          position.x,
          top + 68 + index * 16,
          parameter.description,
          `tensor-info node-parameter ${parameter.className}`,
        );
        text.style.fontSize = `${Math.min(9, (10 * (size.width - 24)) / (parameter.textWidth || 1))}px`;
        text.setAttribute("aria-label", parameter.description);
        text.append(element("title", {}, parameter.description));
      });
      size.subgraphs.forEach((graph, index) => {
        const y = top + 88 + size.parameterRows.length * 16 + index * 20;
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
            height: 19,
            rx: 4,
          }),
        );
        const icon = element("svg", {
          class: "subgraph-icon",
          x: position.x - size.width / 2 + 10,
          y: y + 2,
          width: 16,
          height: 16,
          viewBox: "0 0 24 24",
          "aria-hidden": "true",
        });
        icon.append(
          element("path", {
            d: "M7 12h3M10 5v14M10 5h5M10 19h5M1 9h6v6H1ZM15 2h7v6h-7ZM15 16h7v6h-7Z",
          }),
        );
        link.append(icon);
        const text = label(
          link,
          position.x - size.width / 2 + 32,
          y + 14,
          graph.label,
          "subgraph-label",
          "start",
        );
        text.style.fontSize = `${Math.min(10, (10 * (size.width - 68)) / (graph.textWidth || 1))}px`;
        label(
          link,
          position.x + size.width / 2 - 12,
          y + 14,
          ">",
          "subgraph-label",
          "end",
        );
        bindGraphLink(link, () => navigate(graph.path));
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
      typeFontSize: size.typeFontSize,
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
        producer
          ? layout.routes.get(node.id)?.get(producer)
          : layout.inputRoutes.get(node.id)?.get(input),
      );
    }
  }
  graph.outputs.forEach((name, index) => {
    const producer = producers.get(name);
    const from = producer ? positions.get(producer) : inputPositions.get(name);
    const to = addPort(name, index, true);
    const fromHeight = producer ? sizes.get(producer).height / 2 : 26;
    connect(from, to, name, fromHeight, 26, layout.outputRoutes.get(name));
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
  if (!zoomAreaMode && !event.altKey && !group && !additive) {
    const connection = findConnectionAt(start, connections, viewport);
    if (connection) {
      event.preventDefault();
      selectConnection(connection);
      svg.focus({ preventScroll: true });
      return;
    }
  }
  const originalSelected = new Set(selected);
  event.preventDefault();
  if (zoomAreaMode || event.altKey) {
    const rectangle = element("rect", { class: "selection-rectangle" });
    svg.append(rectangle);
    gesture = { rectangle, zoomArea: true };
    svg.classList.add("selecting");
    svg.focus({ preventScroll: true });
  } else if (group) {
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
    selectConnection(undefined);
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
  const previous = gesture;
  endGesture();
  if (previous?.zoomArea) zoomToArea(previous);
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
    if (gesture || zoomAreaMode) {
      endGesture(true);
    } else {
      selectConnection(undefined);
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
byId("canvas").addEventListener("keydown", (event) => {
  const editable = event.target.closest("[contenteditable]");
  if (
    event.defaultPrevented ||
    event.altKey ||
    event.target.closest("#find-node, input, textarea, select") ||
    (editable &&
      editable.getAttribute("contenteditable").toLowerCase() !== "false") ||
    !hasModel
  )
    return;
  if (event.ctrlKey || event.metaKey) {
    if (event.key === "Home" || event.key === "End") {
      event.preventDefault();
      goToBoundary(event.key === "End");
    }
    return;
  }
  switch (event.key.toLowerCase()) {
    case "f":
      resetZoom();
      break;
    case "r":
      readableZoom();
      break;
    case "+":
    case "=":
      changeZoom(1.25);
      break;
    case "-":
      changeZoom(1 / 1.25);
      break;
    default:
      return;
  }
  event.preventDefault();
});
window.addEventListener("blur", () => endGesture(true));
byId("refresh-metadata").addEventListener("click", requestMetadata);
byId("undo-fusion").addEventListener("click", () => {
  const panel = byId("unfuse-panel");
  panel.hidden = !panel.hidden;
  byId("undo-fusion").setAttribute("aria-expanded", String(!panel.hidden));
  if (!panel.hidden) sidebar.revealPanel(panel);
});
byId("close-unfuse").addEventListener("click", () => {
  byId("unfuse-panel").hidden = true;
  byId("undo-fusion").setAttribute("aria-expanded", "false");
});
for (const [id, all] of [
  ["unfuse-last", false],
  ["unfuse-all", true],
])
  byId(id).addEventListener("click", () => {
    byId("undo-fusion").setAttribute("aria-busy", "true");
    setLoading(replacementPending);
    byId("fusion-status").textContent = all
      ? "Unfusing all operations…"
      : "Unfusing the last operation…";
    requestAction("undo-fusion", "Undo fusion", { all });
  });
byId("verify-model").addEventListener("click", () => {
  verificationStatus(true, "Verifying model…");
  requestAction("verify", "Verify model");
});
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
byId("go-top").addEventListener("click", () => goToBoundary(false));
byId("go-bottom").addEventListener("click", () => goToBoundary(true));
byId("find-node").addEventListener("submit", findNode);
byId("node-search").addEventListener("input", suggestNodes);
byId("node-search").addEventListener("click", suggestNodes);
byId("zoom-fit").addEventListener("click", resetZoom);
byId("zoom-area").addEventListener("click", () => {
  const active = !zoomAreaMode;
  endGesture();
  setZoomAreaMode(active);
  svg.focus({ preventScroll: true });
});
svg.addEventListener(
  "wheel",
  (event) => {
    if (event.ctrlKey) {
      if (event.deltaY === 0) return;
      event.preventDefault();
      const anchor = svg.hasAttribute("viewBox")
        ? graphPoint(event, byId("canvas").getBoundingClientRect())
        : undefined;
      changeZoom(event.deltaY < 0 ? 1.25 : 1 / 1.25, anchor);
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
    if (virtualized) schedulePan();
    else fit();
  },
  { passive: false },
);
window.addEventListener("message", (event) => {
  settingsPanel.receive(event.data);
  if (event.data?.type === "model-progress") {
    if (
      event.data.actionId !== undefined &&
      pendingActions.get(event.data.actionId) !== event.data.target
    )
      return;
    filePanels.setProgress(event.data);
    return;
  }
  if (
    tensorStatisticsPanel.handleMessage(event.data) ||
    memoryPanel.handleMessage(event.data) ||
    exportPanel.handleMessage(event.data) ||
    filePanels.handleMessage(event.data) ||
    fusionPanel.handleMessage(event.data)
  )
    return;
  if (event.data?.type === "application-status") {
    displayStatus(event.data);
  } else if (event.data?.type === "viewer-settings") {
    sidebar.restoreSettings(event.data.settings);
    settingsPanel.restoreViewer(event.data.settings);
  } else if (
    ["settings-info", "python-info", "settings-error"].includes(
      event.data?.type,
    )
  ) {
    if (event.data.type === "settings-info")
      sidebar.restoreSettings(event.data.viewer);
  } else if (event.data?.type === "log-entry") {
    displayLog(event.data.entry);
  } else if (event.data?.type === "log-history") {
    for (const entry of event.data.entries) displayLog(entry);
    const panel = byId("logs-view");
    if (!panel.hidden) panel.scrollTop = panel.scrollHeight;
  } else if (event.data?.type === "show-logs") {
    showTab("logs", true);
  } else if (event.data?.type === "load-state") {
    if (event.data.error) showLoadError(event.data.error);
    else setLoading(event.data.pending);
  } else if (event.data?.type === "load-error") {
    showLoadError(event.data.error);
  } else if (
    event.data?.type === "statistics-info" ||
    event.data?.type === "statistics-error"
  ) {
    if (
      event.data.requestId !== statisticsRequest ||
      byId("statistics-view").hidden
    )
      return;
    if (event.data.type === "statistics-error") {
      statisticsStatus(`Statistics unavailable: ${event.data.error}`);
    } else {
      byId("statistics-view").setAttribute("aria-busy", "false");
      renderStatistics(event.data.statistics, event.data.files);
    }
  } else if (event.data?.type === "graph") {
    fusionUndoCount = event.data.file?.fusionCount ?? 0;
    byId("undo-fusion").setAttribute("aria-busy", "false");
    byId("undo-fusion").title =
      `Choose last or all (${fusionUndoCount} fusion operation${fusionUndoCount === 1 ? "" : "s"} available)`;
    byId("unfuse-count").textContent =
      `${fusionUndoCount} fusion operation${fusionUndoCount === 1 ? "" : "s"} available to unfuse.`;
    const history = event.data.file?.fusionHistory ?? [];
    const historyList = byId("unfuse-history");
    historyList.replaceChildren();
    historyList.start = history.length;
    for (const [index, operation] of [...history].reverse().entries()) {
      const item = document.createElement("li");
      const heading = document.createElement("strong");
      heading.textContent = `${index === 0 ? "Last operation" : "Earlier operation"}${operation.stage ? ` (${operation.stage})` : ""}`;
      const list = document.createElement("ul");
      for (const [matchIndex, match] of operation.matches.entries()) {
        const entry = document.createElement("li");
        const button = document.createElement("button");
        button.textContent = `${match.pattern} (match ${matchIndex + 1})`;
        button.disabled = !match.nodes.length;
        button.dataset.nodeCount = String(match.nodes.length);
        button.title = match.nodes.length
          ? "Select the remaining nodes produced by this match."
          : "No nodes remain selectable in this main graph.";
        button.addEventListener("click", () => {
          const path = displayedGraph.breadcrumbs.at(-1)?.path ?? [];
          const sameGraph = operation.stage
            ? path.length === 1 && path[0].pipeline === operation.stage
            : path.length === 0;
          if (sameGraph) {
            focusNodes(match.nodes);
            return;
          }
          requestAction(
            "select-fusion",
            `Select fused pattern: ${match.pattern}`,
            {
              operation: history.length - 1 - index,
              match: matchIndex,
              pattern: match.pattern,
            },
          );
        });
        entry.append(button);
        list.append(entry);
      }
      item.append(heading, list);
      historyList.append(item);
    }
    if (!fusionUndoCount) {
      byId("unfuse-panel").hidden = true;
      byId("undo-fusion").setAttribute("aria-expanded", "false");
    }
    filePanels.reset();
    displayedGraph = event.data.graph;
    tensorListPanel.setGraph(displayedGraph);
    tensorStatisticsPanel.reset();
    memoryPanel.setGraph(displayedGraph);
    fusionPanel.reset(
      displayedGraph,
      event.data.fusionRequestId,
      event.data.file,
    );
    render(event.data.graph, event.data.metadata);
    if (!byId("statistics-view").hidden) requestStatistics();
    else {
      statisticsRequest++;
      statisticsStatus("Open Statistics to update model statistics.");
    }
    filePanels.setFile(event.data.file);
    fusionPanel.refresh();
    if (event.data.selectedNodes) focusNodes(event.data.selectedNodes);
  } else if (event.data?.type === "undo-fusion-error") {
    fusionPanel.undoFailed(event.data);
    byId("undo-fusion").setAttribute("aria-busy", "false");
    setLoading(replacementPending);
    byId("fusion-status").textContent =
      `Could not undo fusion: ${event.data.error}`;
  } else if (event.data?.type === "navigation-error") {
    svg.setAttribute("aria-busy", "false");
    tensorListPanel.setBusy(replacementPending);
    byId("navigation-status").textContent =
      `Could not open graph: ${event.data.error}`;
    fit();
  } else if (event.data?.type === "verify-info") {
    verificationStatus(
      false,
      displayedGraph.pipeline ? "Pipeline is valid" : "Model is valid",
      RefreshStatus.UNCHANGED,
    );
  } else if (event.data?.type === "verify-error") {
    verificationStatus(
      false,
      `Verification failed: ${event.data.error}`,
      RefreshStatus.FAILED,
    );
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
