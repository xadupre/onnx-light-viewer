const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { test } = require("node:test");
const { JSDOM } = require("jsdom");
const {
  decodeGraph,
  demoGraph,
  loadModel,
  onnxLightVersion,
} = require("../dist/graph");
const { webviewHtml } = require("../dist/webview");

function openViewer(t, initialTab = "model", setup = () => {}) {
  const messages = [];
  const events = [];
  const dom = new JSDOM(
    webviewHtml(
      (name) => `https://viewer.invalid/${name}`,
      "https://viewer.invalid",
      "test-nonce",
      onnxLightVersion,
      initialTab,
    ),
    { runScripts: "outside-only", pretendToBeVisual: true },
  );
  t.after(() => dom.window.close());
  dom.window.HTMLCanvasElement.prototype.getContext = () => ({
    font: "",
    measureText(text) {
      return {
        width: text.length * Number(this.font.match(/([\d.]+)px/)[1]) * 0.6,
      };
    },
  });
  dom.window.acquireVsCodeApi = () => ({
    postMessage: (message) => {
      events.push(message);
      if (message.type !== "interaction") messages.push(message);
    },
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
  for (const dialog of dom.window.document.querySelectorAll("dialog")) {
    dialog.showModal = () => {
      dialog.open = true;
    };
    dialog.close = () => {
      dialog.open = false;
    };
  }
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
  const stylesheet = dom.window.document.createElement("style");
  stylesheet.textContent = readFileSync(
    require.resolve("../media/viewer.css"),
    "utf8",
  );
  dom.window.document.head.append(stylesheet);
  setup(dom.window);
  for (const script of dom.window.document.querySelectorAll("script[src]")) {
    const name = new URL(script.src).pathname.slice(1);
    dom.window.eval(readFileSync(require.resolve(`../media/${name}`), "utf8"));
  }
  const load = (graph, fusionRequestId) => {
    dom.window.dispatchEvent(
      new dom.window.MessageEvent("message", {
        data: { type: "graph", graph, fusionRequestId },
      }),
    );
  };
  return {
    document: dom.window.document,
    window: dom.window,
    messages,
    events,
    load,
    resize,
    pointer,
  };
}

function submitOpen(document, window) {
  document.getElementById("open-file").value = "/models/model.onnx";
  document
    .getElementById("open-form")
    .dispatchEvent(new window.Event("submit", { cancelable: true }));
}

function showStatistics(
  { document, window, messages },
  statistics,
  files = [],
) {
  document.getElementById("statistics-tab").click();
  const { requestId } = messages.findLast(
    ({ type }) => type === "get-statistics",
  );
  window.dispatchEvent(
    new window.MessageEvent("message", {
      data: { type: "statistics-info", requestId, statistics, files },
    }),
  );
}

test("Verify reports success and errors, prevents duplicate requests and resets on navigation", (t) => {
  const { document, window, messages, load } = openViewer(t);
  const button = document.getElementById("verify-model");
  const status = document.getElementById("verify-status");
  const send = (data) =>
    window.dispatchEvent(new window.MessageEvent("message", { data }));
  assert.equal(button.disabled, true);
  assert.ok(button.closest(".graph-toolbar"));
  load(demoGraph);
  const graph = document.getElementById("nodes").innerHTML;
  button.click();
  button.click();
  assert.equal(messages.filter(({ type }) => type === "verify").length, 1);
  assert.equal(button.disabled, true);
  assert.equal(button.getAttribute("aria-busy"), "true");
  assert.equal(button.dataset.refreshStatus, "pending");
  send({ type: "verify-info" });
  assert.equal(status.textContent, "Model is valid");
  assert.equal(button.disabled, false);
  assert.equal(button.getAttribute("aria-busy"), "false");
  button.click();
  send({ type: "verify-error", error: "VerifyModel: invalid graph" });
  assert.match(
    status.textContent,
    /Verification failed: VerifyModel: invalid graph/,
  );
  assert.equal(button.dataset.refreshStatus, "failed");
  assert.equal(button.disabled, false);
  assert.equal(document.getElementById("nodes").innerHTML, graph);
  load({ ...demoGraph, pipeline: true });
  assert.equal(status.textContent, "");
  assert.equal(button.dataset.refreshStatus, "idle");
  button.click();
  send({ type: "verify-info" });
  assert.equal(status.textContent, "Pipeline is valid");
  send({ type: "load-state", pending: true });
  assert.equal(button.disabled, true);
  send({ type: "load-state", pending: false });
  assert.equal(button.disabled, false);
});

test("memory estimates are lazy, accept dimensions, and discard stale results and errors", (t) => {
  const { document, window, messages, load } = openViewer(t);
  const send = (data) =>
    window.dispatchEvent(new window.MessageEvent("message", { data }));
  const memoryRequests = () =>
    messages.filter(({ type }) => type.startsWith("memory-"));
  document.getElementById("memory-tab").click();
  assert.equal(memoryRequests().length, 0);
  document.getElementById("model-tab").click();
  load(demoGraph);
  assert.equal(memoryRequests().length, 0);
  document.getElementById("memory-tab").click();
  const setup = memoryRequests().at(-1);
  send({
    type: "memory-inputs",
    requestId: setup.requestId,
    dimensions: [{ key: "symbol:N", label: "N", defaultValue: 1 }],
    stages: [],
    path: [],
  });
  const dimension = document.querySelector("#memory-dimensions input");
  assert.equal(dimension.value, "1");
  dimension.value = "3";
  const submit = () =>
    document
      .getElementById("memory-form")
      .dispatchEvent(new window.Event("submit", { cancelable: true }));
  submit();
  const request = memoryRequests().at(-1);
  assert.equal(request.type, "memory-profile");
  assert.equal(request.dimensions["symbol:N"], 3);
  assert.equal(document.getElementById("memory-run").disabled, true);
  send({ type: "memory-profile", requestId: setup.requestId, profile: {} });
  assert.equal(document.getElementById("memory-results").hidden, true);
  send({
    type: "memory-error",
    requestId: request.requestId,
    error: "Unresolved shape",
  });
  assert.match(
    document.getElementById("memory-status").textContent,
    /Unresolved shape/,
  );
  assert.equal(document.getElementById("memory-run").disabled, false);
  submit();
  const retry = memoryRequests().at(-1);
  send({
    type: "memory-error",
    requestId: request.requestId,
    error: "Old failure",
  });
  assert.match(
    document.getElementById("memory-status").textContent,
    /Estimating memory/,
  );
  send({
    type: "memory-profile",
    requestId: retry.requestId,
    profile: {
      name: "Model",
      device: "CPU",
      peakBytes: 24,
      events: [
        {
          index: 0,
          node: "<img src=x>",
          opType: "Relu",
          existingBytes: 12,
          outputBytes: 12,
          scratchBytes: 0,
          deviceBytes: 24,
        },
      ],
    },
  });
  assert.equal(document.getElementById("memory-results").hidden, false);
  assert.equal(document.getElementById("memory-peak").textContent, "24 bytes");
  assert.equal(document.querySelector("#memory-events img"), null);
  assert.match(
    document.querySelector("#memory-events").textContent,
    /<img src=x>/,
  );
  load(demoGraph);
  assert.equal(document.getElementById("memory-results").hidden, true);
  send({
    type: "memory-error",
    requestId: retry.requestId,
    error: "Old model",
  });
  assert.doesNotMatch(
    document.getElementById("memory-status").textContent,
    /Old model/,
  );
});

test("memory setup failures still allow choosing another pipeline stage", (t) => {
  const { document, window, messages, load } = openViewer(t);
  load({ ...demoGraph, pipeline: true });
  document.getElementById("memory-tab").click();
  const stages = ["embedding", "prefill"].map((pipeline) => ({
    name: pipeline,
    path: [{ pipeline }],
  }));
  window.dispatchEvent(
    new window.MessageEvent("message", {
      data: {
        type: "memory-error",
        requestId: messages.at(-1).requestId,
        error: "Unknown tensor rank",
        stages,
        path: stages[0].path,
      },
    }),
  );
  assert.equal(document.getElementById("memory-stage-field").hidden, false);
  assert.equal(document.getElementById("memory-run").disabled, true);
  const stage = document.getElementById("memory-stage");
  assert.equal(stage.disabled, false);
  stage.selectedIndex = 1;
  stage.dispatchEvent(new window.Event("change"));
  assert.equal(messages.at(-1).type, "memory-inputs");
  assert.equal(messages.at(-1).path[0].pipeline, "prefill");
});

function showMemoryProfile({ document, window, messages }, bytes, path = []) {
  const send = (data) =>
    window.dispatchEvent(new window.MessageEvent("message", { data }));
  document.getElementById("memory-tab").click();
  send({
    type: "memory-inputs",
    requestId: messages.at(-1).requestId,
    dimensions: [],
    stages: [],
    path,
  });
  document
    .getElementById("memory-form")
    .dispatchEvent(new window.Event("submit", { cancelable: true }));
  send({
    type: "memory-profile",
    requestId: messages.at(-1).requestId,
    profile: {
      name: "Peaks",
      device: "CPU",
      peakBytes: bytes.length ? Math.max(...bytes) : null,
      events: bytes.map((deviceBytes, index) => ({
        index,
        node: index ? "same name" : "",
        opType: "Relu",
        existingBytes: deviceBytes,
        outputBytes: 0,
        scratchBytes: 0,
        deviceBytes,
      })),
    },
  });
}

test("memory local peaks handle empty, constant, endpoint and plateau traces", (t) => {
  for (const [bytes, expected] of [
    [[], []],
    [[0], [1]],
    [[4, 4, 4], [1]],
    [[1, 2, 3], [3]],
    [[3, 2, 1], [1]],
    [
      [1, 3, 3, 1, 4, 2],
      [2, 5],
    ],
    [
      [3, 3, 1, 2, 2],
      [1, 4],
    ],
    [[0, 0, 2, 2, 1, 1], [3]],
  ]) {
    const viewer = openViewer(t);
    const { document, load } = viewer;
    load(demoGraph);
    showMemoryProfile(viewer, bytes);
    assert.match(
      document.getElementById("memory-peaks-count").textContent,
      new RegExp(`^${expected.length} local peak`),
    );
    document.getElementById("memory-peaks-only").click();
    assert.deepEqual(
      [...document.querySelectorAll("#memory-events th")].map((cell) =>
        Number(cell.textContent),
      ),
      expected,
      JSON.stringify(bytes),
    );
    for (const row of document.querySelectorAll("#memory-events tr"))
      assert.match(row.lastElementChild.textContent, /\(local peak\)$/);
    assert.equal(document.getElementById("memory-previous").disabled, true);
    assert.equal(document.getElementById("memory-next").disabled, true);
  }
});

test("memory local peaks paginate without losing execution indices", (t) => {
  const viewer = openViewer(t);
  const { document, load } = viewer;
  load(demoGraph);
  showMemoryProfile(
    viewer,
    Array.from({ length: 203 }, (_, index) => index % 2),
  );
  document.getElementById("memory-peaks-only").click();
  assert.equal(document.querySelectorAll("#memory-events tr").length, 100);
  assert.equal(document.querySelector("#memory-events th").textContent, "2");
  document.getElementById("memory-next").click();
  assert.equal(document.querySelectorAll("#memory-events tr").length, 1);
  assert.equal(document.querySelector("#memory-events th").textContent, "202");
  assert.equal(document.getElementById("memory-next").disabled, true);
  document.getElementById("memory-previous").click();
  assert.equal(document.querySelector("#memory-events th").textContent, "2");
});

test("memory peak links focus node indices in the estimated graph or pipeline stage", (t) => {
  for (const [currentPath, estimatedPath] of [
    [[], []],
    [[{ node: 0, attribute: 0, graph: 0 }], []],
    [[], [{ pipeline: "prefill" }]],
  ]) {
    const viewer = openViewer(t);
    const { document, window, messages, load } = viewer;
    const graph = {
      ...demoGraph,
      breadcrumbs: [{ name: "Current", path: currentPath }],
    };
    load(graph);
    showMemoryProfile(viewer, [4], estimatedPath);
    document.getElementById("memory-peaks-only").click();
    const button = document.querySelector("#memory-events button");
    assert.equal(button.textContent, "node-0");
    document.getElementById("graph").setAttribute("aria-busy", "true");
    button.click();
    assert.equal(document.getElementById("memory-view").hidden, false);
    document.getElementById("graph").setAttribute("aria-busy", "false");
    button.click();
    assert.equal(document.getElementById("model-view").hidden, false);
    if (currentPath.length || estimatedPath.length) {
      assert.equal(messages.at(-1).type, "navigate");
      assert.equal(messages.at(-1).nodeIndex, 0);
      assert.deepEqual(
        JSON.parse(JSON.stringify(messages.at(-1).path)),
        estimatedPath,
      );
      window.dispatchEvent(
        new window.MessageEvent("message", {
          data: {
            type: "graph",
            graph: {
              ...demoGraph,
              breadcrumbs: [{ name: "Estimated", path: estimatedPath }],
            },
            selectedNodes: [0],
          },
        }),
      );
    } else {
      assert.equal(
        messages.some(({ type }) => type === "navigate"),
        false,
      );
    }
    assert.equal(
      document.querySelector(".graph-node.selected").dataset.id,
      "node-0",
    );
    assert.equal(
      document.getElementById("memory-results").hidden,
      currentPath.length > 0 || estimatedPath.length > 0,
    );
  }
});

test("memory chart batches 20,000 events and paginates every row", (t) => {
  const { document, window, messages, load } = openViewer(t);
  load(demoGraph);
  document.getElementById("memory-tab").click();
  const send = (data) =>
    window.dispatchEvent(new window.MessageEvent("message", { data }));
  send({
    type: "memory-inputs",
    requestId: messages.at(-1).requestId,
    dimensions: [],
    stages: [],
    path: [],
  });
  document
    .getElementById("memory-form")
    .dispatchEvent(new window.Event("submit", { cancelable: true }));
  const events = Array.from({ length: 20_000 }, (_, index) => ({
    index,
    node: `node${index}`,
    opType: "Relu",
    existingBytes: 16,
    outputBytes: 0,
    scratchBytes: 0,
    deviceBytes: index === 12345 ? 32 : 16,
  }));
  send({
    type: "memory-profile",
    requestId: messages.at(-1).requestId,
    profile: { name: "Large", device: "CPU", peakBytes: 32, events },
  });
  assert.equal(document.querySelectorAll("#memory-events tr").length, 100);
  assert.equal(document.querySelectorAll("#memory-chart path").length, 2);
  assert.equal(
    document.getElementById("memory-curve").getAttribute("d").match(/[ML]/g)
      .length,
    20_000,
  );
  document.getElementById("memory-next").click();
  assert.match(document.getElementById("memory-range").textContent, /101–200/);
  assert.equal(document.querySelector("#memory-events th").textContent, "101");
  document.getElementById("memory-previous").click();
  assert.equal(document.querySelector("#memory-events th").textContent, "1");
  document.getElementById("memory-next").click();
  document.getElementById("memory-peaks-only").click();
  assert.equal(document.querySelectorAll("#memory-events tr").length, 1);
  assert.equal(
    document.querySelector("#memory-events th").textContent,
    "12346",
  );
  assert.equal(document.getElementById("memory-previous").disabled, true);
  assert.equal(document.getElementById("memory-next").disabled, true);
  document.getElementById("memory-peaks-only").click();
  assert.equal(document.querySelector("#memory-events th").textContent, "1");
});

test("statistics tab shows model counts, operator distribution and file sizes", (t) => {
  const viewer = openViewer(t);
  const { document, window } = viewer;
  const graph = structuredClone(demoGraph);
  graph.nodes.push({
    ...graph.nodes[0],
    id: "node-1",
    name: "addition",
    opType: "Add",
  });
  graph.initializerCount = 3;
  graph.initializers = [{ name: "dense" }, { name: "weights" }];
  graph.sparseInitializers = [{ name: "sparse" }];
  graph.encodedInitializers = [];
  graph.localFunctions = [{ name: "local-function" }];
  window.dispatchEvent(
    new window.MessageEvent("message", {
      data: {
        type: "graph",
        graph,
        file: {
          files: [
            { name: "model.onnx", size: 2048, external: false },
            { name: "weights.bin", size: 1024, external: true },
          ],
        },
      },
    }),
  );
  showStatistics(viewer, graph, [
    { name: "model.onnx", size: 2048, external: false },
    { name: "weights.bin", size: 1024, external: true },
  ]);
  assert.equal(document.getElementById("statistics-view").hidden, false);
  assert.equal(document.getElementById("status-view").hidden, true);
  const summary = new Map(
    [...document.querySelectorAll("#statistics-summary dt")].map((term) => [
      term.textContent,
      term.nextElementSibling.textContent,
    ]),
  );
  assert.equal(summary.get("Nodes"), "2");
  assert.equal(summary.get("Initializers"), "3");
  assert.equal(summary.get("Dense initializers"), "2");
  assert.equal(summary.get("Sparse initializers"), "1");
  assert.equal(summary.get("Local functions"), "1");
  assert.equal(
    [...document.querySelectorAll("#statistics-operators tbody tr")]
      .map((row) => row.textContent)
      .join(" "),
    "ai.onnxAdd1 ai.onnxRelu1",
  );
  const files = document.getElementById("statistics-files").textContent;
  assert.match(files, /model\.onnx/);
  assert.match(files, /2\.0 KiB/);
  assert.match(files, /weights\.bin/);
  assert.match(files, /1\.0 KiB/);
  document.getElementById("statistics-tab").focus();
  document
    .getElementById("statistics-tab")
    .dispatchEvent(new window.KeyboardEvent("keydown", { key: "ArrowRight" }));
  assert.equal(document.getElementById("memory-view").hidden, false);
  assert.equal(document.getElementById("statistics-view").hidden, true);
});

test("statistics retain missing external files with an explicit literal size error", (t) => {
  const viewer = openViewer(t);
  viewer.load(demoGraph);
  showStatistics(viewer, demoGraph, [
    {
      name: "weights/<missing>.bin",
      external: true,
      error: "File not found <img src=x>",
    },
  ]);
  const table = viewer.document.getElementById("statistics-files");
  assert.match(table.textContent, /External data/);
  assert.match(table.textContent, /weights\/<missing>\.bin/);
  assert.match(table.textContent, /Unavailable: File not found <img src=x>/);
  assert.equal(table.querySelector("img"), null);
});

test("statistics tab aggregates operator and node counts for each pipeline stage", (t) => {
  const viewer = openViewer(t);
  const { document, window } = viewer;
  const model = loadModel(
    readFileSync(require.resolve("./models/tiny.pipe-onnx")),
    "tiny.pipe-onnx",
  );
  const graph = model.graph();
  graph.nodes.forEach((node) => {
    node.pipelineModel.name = "main_graph";
  });
  graph.initializerCount = 0;
  window.dispatchEvent(
    new window.MessageEvent("message", {
      data: { type: "graph", graph },
    }),
  );
  showStatistics(viewer, graph);
  assert.equal(
    document.querySelectorAll("#statistics-stage-table tbody tr").length,
    3,
  );
  assert.deepEqual(
    [...document.querySelectorAll("#statistics-stage-table tbody th")].map(
      (cell) => cell.textContent,
    ),
    ["embedding", "prefill", "decode"],
  );
  const stages = graph.nodes.map((node) => node.pipelineModel);
  const expectedNodeCount = stages.reduce(
    (count, stage) => count + stage.nodeCount,
    0,
  );
  const summary = new Map(
    [...document.querySelectorAll("#statistics-summary dt")].map((term) => [
      term.textContent,
      term.nextElementSibling.textContent,
    ]),
  );
  assert.equal(
    summary.get("Initializers"),
    String(stages.reduce((count, stage) => count + stage.initializerCount, 0)),
  );
  assert.equal(summary.get("Nodes"), String(expectedNodeCount));
  assert.equal(
    [...document.querySelectorAll("#statistics-operators tbody tr")].length,
    new Set(
      stages.flatMap((stage) =>
        stage.operatorTypes.map(({ domain, opType }) =>
          JSON.stringify([domain, opType]),
        ),
      ),
    ).size,
  );
  assert.match(
    document.getElementById("statistics-files").textContent,
    /No source file available/,
  );
});

test("statistics identify unavailable pipeline projections instead of showing zero counts", (t) => {
  const viewer = openViewer(t);
  const { document, window } = viewer;
  const graph = loadModel(
    readFileSync(require.resolve("./models/tiny.pipe-onnx")),
    "tiny.pipe-onnx",
  ).graph();
  graph.nodes[0].pipelineModel.statisticsAvailable = false;
  graph.nodes[0].pipelineModel.operatorTypes = [
    { domain: "test", opType: "Unavailable", count: 42 },
  ];
  window.dispatchEvent(
    new window.MessageEvent("message", {
      data: { type: "graph", graph, file: { files: [] } },
    }),
  );
  showStatistics(viewer, graph);

  const summary = new Map(
    [...document.querySelectorAll("#statistics-summary dt")].map((term) => [
      term.textContent,
      term.nextElementSibling.textContent,
    ]),
  );
  assert.match(summary.get("Dense initializers"), /^Partial:/);
  assert.match(summary.get("Local functions"), /^Partial:/);
  assert.equal(
    document.querySelector("#statistics-stage-table tbody tr th").textContent,
    "embedding",
  );
  assert.equal(
    document
      .querySelector("#statistics-stage-table tbody tr")
      .textContent.includes("Unavailable"),
    true,
  );
  assert.match(
    document.getElementById("statistics-operators").textContent,
    /Partial operator statistics/,
  );
  assert.doesNotMatch(
    document.getElementById("statistics-operators").textContent,
    /Unavailable/,
  );
});

test("statistics accept compact model summaries while viewing a function", (t) => {
  const viewer = openViewer(t);
  const { document, window } = viewer;
  const model = loadModel(
    readFileSync(require.resolve("./models/functions.onnx")),
    "functions.onnx",
  );
  const root = model.graph();
  const graph = model.graph(false, root.localFunctions[0].path);
  const statistics = model.summary();
  window.dispatchEvent(
    new window.MessageEvent("message", {
      data: { type: "graph", graph },
    }),
  );
  showStatistics(viewer, statistics);
  const summary = new Map(
    [...document.querySelectorAll("#statistics-summary dt")].map((term) => [
      term.textContent,
      term.nextElementSibling.textContent,
    ]),
  );
  assert.equal(summary.get("Nodes"), String(root.nodes.length));
  assert.equal(
    summary.get("Local functions"),
    String(root.localFunctions.length),
  );
  assert.equal(
    summary.get("Dense initializers"),
    String(root.initializers.length),
  );
  assert.equal(
    [
      ...document.querySelectorAll("#statistics-operators tbody td:last-child"),
    ].reduce((sum, cell) => sum + Number(cell.textContent), 0),
    root.nodes.length,
  );
});

test("statistics distinguish empty operator distributions from partial and unavailable data", (t) => {
  const viewer = openViewer(t);
  const { document, window } = viewer;
  const empty = { ...demoGraph, nodes: [] };
  const pipeline = loadModel(
    readFileSync(require.resolve("./models/tiny.pipe-onnx")),
    "tiny.pipe-onnx",
  ).graph();
  const partial = structuredClone(pipeline);
  for (const [index, node] of partial.nodes.entries()) {
    node.pipelineModel.statisticsAvailable = index === 0;
    node.pipelineModel.operatorTypes = [];
    if (index === 0) node.pipelineModel.nodeCount = 0;
  }
  const unavailable = structuredClone(partial);
  unavailable.nodes[0].pipelineModel.statisticsAvailable = false;
  const emptyPipeline = structuredClone(partial);
  for (const node of emptyPipeline.nodes) {
    node.pipelineModel.statisticsAvailable = true;
    node.pipelineModel.nodeCount = 0;
  }
  for (const [graph, message] of [
    [empty, "No operators."],
    [emptyPipeline, "No operators."],
    [partial, "Partial operator statistics: some stages are unavailable."],
    [unavailable, "Operator statistics unavailable."],
  ]) {
    document.getElementById("model-tab").click();
    window.dispatchEvent(
      new window.MessageEvent("message", {
        data: { type: "graph", graph, file: { files: [] } },
      }),
    );
    showStatistics(viewer, graph);
    assert.equal(
      document.querySelector("#statistics-operators tbody td").textContent,
      message,
    );
    assert.equal(
      document.querySelectorAll("#statistics-operators tbody").length,
      1,
    );
  }
});

test("statistics refresh on opening the tab and reject stale results and errors", (t) => {
  const viewer = openViewer(t);
  const { document, window, messages, load } = viewer;
  const requests = () =>
    messages.filter(({ type }) => type === "get-statistics");
  const send = (data) =>
    window.dispatchEvent(new window.MessageEvent("message", { data }));
  const content = document.getElementById("statistics-content");
  const status = document.getElementById("statistics-empty");
  document.getElementById("statistics-tab").click();
  assert.equal(requests().length, 0);
  document.getElementById("model-tab").click();
  load(demoGraph);
  assert.equal(requests().length, 0);
  showStatistics(viewer, demoGraph);
  assert.equal(requests().length, 1);
  assert.equal(content.hidden, false);

  document.getElementById("model-tab").click();
  document.getElementById("statistics-tab").click();
  const { requestId } = requests().at(-1);
  assert.equal(requests().length, 2);
  assert.equal(content.hidden, true);
  assert.match(status.textContent, /Updating model statistics/);
  send({
    type: "statistics-info",
    requestId: requests()[0].requestId,
    statistics: demoGraph,
  });
  assert.equal(content.hidden, true);
  send({ type: "statistics-error", requestId, error: "Root analysis failed." });
  assert.equal(content.hidden, true);
  assert.match(
    status.textContent,
    /Statistics unavailable: Root analysis failed/,
  );

  document.getElementById("model-tab").click();
  document.getElementById("statistics-tab").click();
  const retry = requests().at(-1);
  send({ type: "statistics-error", requestId, error: "Stale failure." });
  assert.match(status.textContent, /Updating model statistics/);
  send({
    type: "statistics-info",
    requestId: retry.requestId,
    statistics: demoGraph,
  });
  assert.equal(content.hidden, false);

  load({ ...demoGraph, name: "Replacement" });
  const replacement = requests().at(-1);
  assert.equal(requests().length, 4);
  assert.equal(content.hidden, true);
  send({
    type: "statistics-info",
    requestId: retry.requestId,
    statistics: demoGraph,
  });
  assert.equal(content.hidden, true);
  send({
    type: "statistics-info",
    requestId: replacement.requestId,
    statistics: { ...demoGraph, name: "Replacement" },
  });
  assert.equal(
    document.getElementById("statistics-model-name").textContent,
    "Replacement",
  );
});

test("pipeline navigation enables stage fusions and disables unsupported encrypted saves", (t) => {
  const { document, window, messages } = openViewer(t);
  const send = (graph, file) =>
    window.dispatchEvent(
      new window.MessageEvent("message", {
        data: { type: "graph", graph, file },
      }),
    );
  send(demoGraph, {
    directory: "/models",
    savePath: "/models/model-copy.onnx",
  });
  const encrypted = document.querySelector(
    'input[name="save-format"][value="embedded-encrypted"]',
  );
  encrypted.checked = true;
  encrypted.dispatchEvent(new window.Event("change", { bubbles: true }));
  document.getElementById("save-password").value = "discard";
  const pipeline = {
    ...structuredClone(demoGraph),
    pipeline: true,
    canFuse: false,
    nodes: [
      {
        ...structuredClone(demoGraph.nodes[0]),
        attributes: [
          {
            name: "prefill",
            type: "MODEL",
            value: "Relu",
            default: false,
            graphs: [
              {
                name: "prefill",
                path: [{ pipeline: "prefill" }],
                nodeCount: 1,
              },
            ],
          },
        ],
      },
    ],
  };
  send(pipeline, {
    directory: "/models",
    savePath: "/models/copy.pipe-onnx",
    pipeline: true,
  });
  assert.equal(document.getElementById("fuse-pattern").disabled, true);
  assert.equal(encrypted.disabled, true);
  assert.equal(encrypted.checked, false);
  assert.equal(
    document.getElementById("save-file").value,
    "/models/copy.pipe-onnx",
  );
  assert.equal(document.getElementById("save-password").value, "");
  document
    .querySelector(".subgraph-link")
    .dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
  assert.equal(messages.at(-1).type, "navigate");
  assert.deepEqual(JSON.parse(JSON.stringify(messages.at(-1).path)), [
    { pipeline: "prefill" },
  ]);
  send(
    {
      ...structuredClone(demoGraph),
      pipeline: true,
      canFuse: true,
      breadcrumbs: [
        { name: "Pipeline", path: [] },
        { name: "prefill", path: [{ pipeline: "prefill" }] },
      ],
    },
    {
      directory: "/models",
      savePath: "/models/copy.pipe-onnx",
      pipeline: true,
    },
  );
  assert.equal(document.getElementById("fuse-pattern").disabled, false);
  document.getElementById("fuse-pattern").click();
  assert.equal(messages.at(-1).type, "fusion-patterns");
  send(demoGraph, {
    directory: "/models",
    savePath: "/models/model-copy.onnx",
  });
  assert.equal(encrypted.disabled, false);
});

test("keeps all pipeline models accessible from the permanent navigation bar", (t) => {
  const { document, messages, load } = openViewer(t);
  const model = loadModel(
    readFileSync(require.resolve("./models/tiny.pipe-onnx")),
    "tiny.pipe-onnx",
  );
  load(model.graph());
  const buttons = () => [
    ...document.querySelectorAll("#graph-navigation button"),
  ];
  assert.deepEqual(
    buttons().map((button) => button.textContent),
    ["Three-token cycle", "embedding", "prefill", "decode"],
  );
  for (const role of ["prefill", "decode", "embedding"]) {
    buttons()
      .find((button) => button.textContent === role)
      .click();
    const request = messages.at(-1);
    assert.equal(request.type, "navigate");
    assert.deepEqual(JSON.parse(JSON.stringify(request.path)), [
      { pipeline: role },
    ]);
    load(model.graph(false, request.path));
    assert.equal(buttons().length, 4);
    assert.equal(
      document.querySelector("#graph-navigation [aria-current]").textContent,
      role,
    );
    assert.equal(
      buttons().find((button) => button.textContent === role).disabled,
      true,
    );
  }
  buttons()[0].click();
  load(model.graph(false, messages.at(-1).path));
  assert.equal(
    document.querySelector("#graph-navigation [aria-current]").textContent,
    "Three-token cycle",
  );
});

test("resets sidebar controllers together when the viewer replaces the graph", (t) => {
  const { document, window, messages, load } = openViewer(t);
  const send = (data) =>
    window.dispatchEvent(new window.MessageEvent("message", { data }));
  const first = structuredClone(demoGraph);
  first.nodes[0].schema.doc = "First model documentation";
  load(first);
  document.getElementById("show-doc").click();
  assert.equal(document.getElementById("operator-doc").open, true);
  document.getElementById("open-model").click();
  const browse = messages.at(-1);
  assert.equal(browse.type, "browse-files");
  document.getElementById("fuse-pattern").click();
  const patterns = messages.at(-1);
  assert.equal(patterns.type, "fusion-patterns");
  send({
    type: "fusion-patterns",
    requestId: patterns.requestId,
    patterns: ["TransposeTranspose"],
  });
  document.querySelector("#fusion-pattern input").click();
  document.getElementById("preview-fusion").click();
  const preview = messages.at(-1);
  assert.equal(preview.type, "preview-fusion");

  const replacement = structuredClone(demoGraph);
  replacement.nodes[0].name = "Replacement node";
  replacement.nodes[0].schema.doc = "Replacement documentation";
  load(replacement);
  send({
    type: "file-list",
    target: "open",
    requestId: browse.requestId,
    location: "/old",
    parent: "/",
    entries: [{ name: "old.onnx", path: "/old/old.onnx", directory: false }],
  });
  send({
    type: "fusion-preview",
    requestId: preview.requestId,
    previewId: 1,
    matches: [{ nodes: [0], replacement: [] }],
  });
  send({
    type: "fusion-error",
    requestId: preview.requestId,
    error: "Old model fusion failed",
  });
  assert.equal(document.getElementById("open-files").children.length, 0);
  assert.equal(
    document.getElementById("open-files").getAttribute("aria-busy"),
    "false",
  );
  assert.equal(document.getElementById("fusion-panel").hidden, true);
  assert.equal(document.getElementById("fusion-matches").children.length, 0);
  assert.equal(document.getElementById("fusion-status").textContent, "");
  assert.equal(
    document.getElementById("node-name").textContent,
    "Replacement node",
  );
  assert.equal(document.getElementById("operator-doc").open, false);
  assert.equal(document.getElementById("doc-body").textContent, "");
  document.getElementById("show-doc").click();
  assert.match(
    document.getElementById("doc-body").textContent,
    /Replacement documentation/,
  );
});

test("resizes the sidebar with pointer and keyboard controls without rearranging nodes", (t) => {
  const { document, window, load, messages } = openViewer(t);
  const sidebar = document.getElementById("graph-panels");
  const resizer = document.getElementById("sidebar-resizer");
  const body = sidebar.parentElement;
  const style = window.getComputedStyle(body);
  const minimumWidth = parseFloat(
    style.getPropertyValue("--sidebar-min-width"),
  );
  const graphMinimum = parseFloat(style.getPropertyValue("--graph-min-width"));
  const resizerWidth = parseFloat(
    style.getPropertyValue("--sidebar-resizer-width"),
  );
  const frames = [];
  window.requestAnimationFrame = (callback) => frames.push(callback);
  const flushFrame = () => frames.splice(0).forEach((callback) => callback());
  let bodyWidth = 1200;
  const width = () =>
    Math.min(
      bodyWidth - graphMinimum - resizerWidth,
      parseFloat(body.style.getPropertyValue("--sidebar-width")) || 300,
    );
  body.getBoundingClientRect = () => new window.DOMRect(0, 0, bodyWidth, 600);
  sidebar.getBoundingClientRect = () =>
    new window.DOMRect(bodyWidth - width(), 0, width(), 600);
  document.getElementById("canvas").getBoundingClientRect = () =>
    new window.DOMRect(0, 0, bodyWidth - width() - resizerWidth, 600);
  let captured;
  resizer.setPointerCapture = (id) => (captured = id);
  resizer.hasPointerCapture = (id) => captured === id;
  resizer.releasePointerCapture = () => {
    captured = undefined;
    pointer("lostpointercapture", 0);
  };
  const pointer = (type, x, pointerId = 1, button = 0) => {
    const event = new window.MouseEvent(type, {
      clientX: x,
      button,
      bubbles: true,
      cancelable: true,
    });
    Object.defineProperty(event, "pointerId", { value: pointerId });
    resizer.dispatchEvent(event);
  };
  const key = (key) =>
    resizer.dispatchEvent(
      new window.KeyboardEvent("keydown", { key, cancelable: true }),
    );
  load(demoGraph);
  const nodes = document.getElementById("nodes").innerHTML;
  const graph = document.getElementById("graph");
  const viewBox = graph.getAttribute("viewBox");
  assert.equal(resizer.getAttribute("role"), "separator");
  assert.equal(resizer.getAttribute("aria-orientation"), "vertical");
  assert.equal(resizer.getAttribute("aria-controls"), sidebar.id);
  pointer("pointerdown", 955, 1, 2);
  pointer("pointermove", 800);
  assert.equal(body.style.getPropertyValue("--sidebar-width"), "");
  pointer("pointerdown", 900);
  assert.equal(captured, 1);
  assert.equal(document.activeElement, resizer);
  assert.equal(document.body.classList.contains("resizing-sidebar"), true);
  pointer("pointermove", 700, 2);
  assert.equal(width(), 300);
  pointer("pointermove", 750);
  pointer("pointermove", 700);
  assert.equal(width(), 500);
  assert.equal(resizer.getAttribute("aria-valuemin"), String(minimumWidth));
  assert.equal(resizer.getAttribute("aria-valuemax"), "1034");
  assert.equal(resizer.getAttribute("aria-valuenow"), "500");
  assert.equal(resizer.getAttribute("aria-valuetext"), "500 pixels");
  assert.equal(graph.getAttribute("viewBox"), viewBox);
  assert.equal(frames.length, 1);
  flushFrame();
  assert.notEqual(graph.getAttribute("viewBox"), viewBox);
  pointer("pointerup", 700);
  assert.deepEqual(JSON.parse(JSON.stringify(messages.at(-1))), {
    type: "viewer-settings",
    settings: {
      width: 500,
      panels: { operator: true, find: true, metadata: false },
    },
  });
  assert.equal(captured, undefined);
  assert.equal(document.body.classList.contains("resizing-sidebar"), false);
  pointer("pointermove", 600);
  assert.equal(width(), 500);
  key("ArrowLeft");
  assert.equal(width(), 520);
  key("ArrowRight");
  assert.equal(width(), 500);
  key("Home");
  assert.equal(width(), 200);
  key("ArrowRight");
  assert.equal(width(), 200);
  key("End");
  assert.equal(width(), 1034);
  key("ArrowLeft");
  assert.equal(width(), 1034);
  key("Home");
  pointer("pointerdown", 1000);
  pointer("pointermove", -1000);
  assert.equal(width(), 1034);
  pointer("pointermove", 2000);
  assert.equal(width(), 200);
  pointer("pointercancel", 2000);
  assert.equal(captured, undefined);
  assert.equal(document.body.classList.contains("resizing-sidebar"), false);
  key("ArrowLeft");
  document.getElementById("open-model").click();
  assert.equal(width(), 220);
  pointer("pointerdown", 980);
  pointer("lostpointercapture", 980);
  pointer("pointermove", 500);
  assert.equal(width(), 220);
  pointer("pointerdown", 980);
  bodyWidth = 800;
  window.dispatchEvent(new window.Event("resize"));
  assert.equal(captured, undefined);
  pointer("pointermove", 500);
  assert.equal(width(), 220);
  key("End");
  assert.equal(width(), 634);
  assert.equal(resizer.getAttribute("aria-valuemax"), "634");
  flushFrame();
  assert.equal(document.getElementById("nodes").innerHTML, nodes);
});

test("restores sidebar width and panel visibility without saving transient form fields", (t) => {
  const first = openViewer(t);
  const initial = {
    width: 480,
    panels: { operator: false, find: true, metadata: false },
  };
  const restore = (viewer, settings) =>
    viewer.window.dispatchEvent(
      new viewer.window.MessageEvent("message", {
        data: { type: "viewer-settings", settings },
      }),
    );
  restore(first, initial);
  const body = first.document.getElementById("graph-panels").parentElement;
  assert.equal(body.style.getPropertyValue("--sidebar-width"), "480px");
  const operator = first.document.getElementById("show-operator");
  assert.equal(operator.getAttribute("aria-expanded"), "false");
  assert.equal(
    first.document.getElementById(operator.getAttribute("aria-controls"))
      .hidden,
    true,
  );
  assert.equal(
    first.messages.filter(({ type }) => type === "viewer-settings").length,
    0,
  );
  first.document.getElementById("open-password").value = "not-a-preference";
  first.document.getElementById("show-metadata").click();
  const saved = first.messages.findLast(
    ({ type }) => type === "viewer-settings",
  ).settings;
  assert.deepEqual(JSON.parse(JSON.stringify(saved)), {
    ...initial,
    panels: { ...initial.panels, metadata: true },
  });
  const second = openViewer(t);
  restore(second, saved);
  assert.equal(
    second.document
      .getElementById("graph-panels")
      .parentElement.style.getPropertyValue("--sidebar-width"),
    "480px",
  );
  assert.equal(
    second.document
      .getElementById("show-operator")
      .getAttribute("aria-expanded"),
    "false",
  );
  assert.equal(
    second.document
      .getElementById("show-metadata")
      .getAttribute("aria-expanded"),
    "true",
  );
  assert.equal(
    second.document.getElementById("show-find").getAttribute("aria-expanded"),
    "true",
  );
  second.load(demoGraph);
  assert.ok(second.messages.some(({ type }) => type === "metadata"));
});

test("initializes sidebar announcements from layout and batches observer redraws", (t) => {
  let notifyResize;
  let sidebarWidth = 360;
  const frames = [];
  const { document, window, load } = openViewer(t, "model", (window) => {
    window.document.querySelector(".graph-body").getBoundingClientRect = () =>
      new window.DOMRect(0, 0, 1200, 600);
    window.document.getElementById("graph-panels").getBoundingClientRect = () =>
      new window.DOMRect(0, 0, sidebarWidth, 600);
    window.requestAnimationFrame = (callback) => frames.push(callback);
    window.ResizeObserver = class {
      constructor(callback) {
        notifyResize = callback;
      }
      observe(element) {
        assert.equal(element.id, "graph-panels");
      }
    };
  });
  const resizer = document.getElementById("sidebar-resizer");
  assert.equal(resizer.getAttribute("aria-valuenow"), "360");
  assert.equal(resizer.getAttribute("aria-valuetext"), "360 pixels");
  load(demoGraph);
  const graph = document.getElementById("graph");
  const observer = new window.MutationObserver(() => {});
  observer.observe(graph, { attributes: true, attributeFilter: ["viewBox"] });
  t.after(() => observer.disconnect());
  for (sidebarWidth of [400, 450, 500]) notifyResize();
  assert.equal(resizer.getAttribute("aria-valuenow"), "500");
  assert.equal(frames.length, 1);
  assert.equal(observer.takeRecords().length, 0);
  frames.shift()();
  assert.equal(observer.takeRecords().length, 1);
  notifyResize();
  assert.equal(frames.length, 1);
  frames.shift()();
});

test("discovers popular ONNX repositories by task and searches before browsing model files", (t) => {
  const { document, window, messages, load } = openViewer(t);
  load(demoGraph);
  document.getElementById("open-model").click();
  document.getElementById("open-source-huggingface").click();
  assert.equal(document.querySelector("select#open-source"), null);
  assert.equal(document.querySelectorAll("#open-source button").length, 3);
  assert.equal(
    document
      .getElementById("open-source-huggingface")
      .getAttribute("aria-pressed"),
    "true",
  );
  assert.equal(
    document.getElementById("open-source-local").getAttribute("aria-pressed"),
    "false",
  );
  assert.equal(document.getElementById("hf-discovery").hidden, false);
  assert.equal(document.getElementById("hf-task").closest("[hidden]"), null);
  assert.equal(document.getElementById("open-browser").open, false);
  assert.equal(document.getElementById("open-file-list").hidden, true);
  assert.match(
    document.getElementById("hf-search-help").textContent,
    /click Search \(or press Enter\)/,
  );
  assert.ok(document.querySelector("#hf-search-form label[for='hf-task']"));
  assert.ok(
    document
      .getElementById("hf-task")
      .compareDocumentPosition(document.getElementById("hf-query")) &
      window.Node.DOCUMENT_POSITION_FOLLOWING,
  );
  const popular = messages.at(-1);
  assert.equal(popular.type, "search-huggingface");
  assert.equal(popular.query, "");
  assert.equal(popular.task, "");
  const send = (data) =>
    window.dispatchEvent(new window.MessageEvent("message", { data }));
  const repository = {
    id: "author/popular",
    url: "https://huggingface.co/author/popular",
    downloads: 12500,
    task: "feature-extraction",
  };
  send({
    type: "huggingface-search-results",
    requestId: popular.requestId,
    models: [repository],
  });
  assert.match(
    document.getElementById("hf-results-title").textContent,
    /Popular/,
  );
  assert.match(
    document.getElementById("hf-search-results").textContent,
    /author\/popular/,
  );
  assert.match(
    document.getElementById("hf-search-results").textContent,
    /12,500 downloads/,
  );
  send({
    type: "huggingface-model-files",
    requestId: popular.requestId,
    id: repository.id,
    entries: [{ size: 22 * 1024 * 1024 }, { size: 90 * 1024 * 1024 }, {}],
  });
  assert.equal(
    document.querySelector(".hub-model-size").textContent,
    "ONNX files: 22 MiB - 90 MiB (2/3 sizes known)",
  );
  assert.equal(messages.filter(({ type }) => type === "open").length, 0);
  document.getElementById("hf-task").value = "feature-extraction";
  document.getElementById("hf-task").dispatchEvent(new window.Event("change"));
  const task = messages.at(-1);
  assert.equal(task.task, "feature-extraction");
  document.getElementById("hf-query").value = "bert";
  document
    .getElementById("hf-search-form")
    .dispatchEvent(new window.Event("submit", { cancelable: true }));
  const search = messages.at(-1);
  assert.equal(search.query, "bert");
  assert.equal(search.task, "feature-extraction");
  send({
    type: "huggingface-search-results",
    requestId: task.requestId,
    models: [repository],
  });
  assert.equal(
    document.querySelectorAll("#hf-search-results button").length,
    0,
  );
  send({
    type: "huggingface-search-results",
    requestId: search.requestId,
    models: [{ ...repository, id: "<img src=x onerror=alert(1)>" }],
  });
  assert.equal(document.querySelector("#hf-search-results img"), null);
  document.getElementById("open-file").value = "/old/model.onnx";
  document.querySelector("#hf-search-results button").click();
  const browse = messages.at(-1);
  assert.equal(browse.type, "browse-files");
  assert.equal(browse.source, "huggingface");
  assert.equal(browse.location, repository.url);
  assert.equal(document.getElementById("open-file").value, "");
  assert.equal(messages.filter(({ type }) => type === "open").length, 0);
  const file = `${repository.url}/resolve/main/onnx/model.onnx`;
  send({
    type: "file-list",
    target: "open",
    requestId: browse.requestId,
    location: repository.url,
    entries: [
      {
        name: "onnx/model.onnx",
        path: file,
        directory: false,
        size: 90 * 1024 * 1024,
      },
      { name: "onnx/unknown.onnx", path: `${file}-unknown`, directory: false },
    ],
  });
  assert.equal(document.getElementById("open-file-list").hidden, false);
  assert.equal(
    document.querySelector("#open-files .file-size").textContent,
    "File size: 90 MiB",
  );
  assert.equal(document.querySelectorAll("#open-files .file-size").length, 2);
  assert.match(
    document.querySelectorAll("#open-files .file-size")[1].textContent,
    /unavailable/,
  );
  assert.doesNotMatch(
    document.getElementById("hf-files-help").textContent,
    /click Open/,
  );
  document.querySelector("#open-files button").click();
  assert.equal(messages.filter(({ type }) => type === "open").length, 0);
  document
    .querySelector("#open-files button")
    .dispatchEvent(new window.MouseEvent("dblclick", { bubbles: true }));
  assert.equal(messages.at(-1).type, "open");
  assert.equal(messages.at(-1).location, file);
  assert.equal(document.getElementById("open-progress").hidden, false);
  document
    .querySelector("#open-files button")
    .dispatchEvent(new window.MouseEvent("dblclick", { bubbles: true }));
  assert.equal(messages.filter(({ type }) => type === "open").length, 1);
});

test("reports Hub search errors and ignores responses after closing or switching sources", (t) => {
  const { document, window, messages, load } = openViewer(t);
  load(demoGraph);
  document.getElementById("open-source-huggingface").click();
  const send = (data) =>
    window.dispatchEvent(new window.MessageEvent("message", { data }));
  let request = messages.at(-1).requestId;
  send({
    type: "huggingface-search-error",
    requestId: request,
    error: "Search unavailable: HTTP 503",
  });

  assert.match(
    document.getElementById("hf-search-status").textContent,
    /HTTP 503/,
  );
  assert.equal(
    document.getElementById("hf-search-results").getAttribute("aria-busy"),
    "false",
  );
  document.getElementById("hf-search-form").requestSubmit();
  request = messages.at(-1).requestId;
  send({ type: "huggingface-search-results", requestId: request, models: [] });
  assert.match(
    document.getElementById("hf-search-status").textContent,
    /No public ONNX models/,
  );
  document.getElementById("hf-search-form").requestSubmit();
  request = messages.at(-1).requestId;
  document.getElementById("cancel-open").click();
  send({
    type: "huggingface-search-error",
    requestId: request,
    error: "Late error",
  });
  assert.equal(document.getElementById("hf-search-status").textContent, "");
  document.getElementById("open-model").click();
  request = messages.at(-1).requestId;
  document.getElementById("open-source-local").click();
  send({
    type: "huggingface-search-results",
    requestId: request,
    models: [{ id: "old/result" }],
  });
  assert.equal(document.getElementById("hf-discovery").hidden, true);
  assert.equal(
    document.querySelectorAll("#hf-search-results button").length,
    0,
  );
});

test("reports missing or unavailable repository sizes without accepting stale metadata", (t) => {
  const { document, window, messages, load } = openViewer(t);
  load(demoGraph);
  document.getElementById("open-source-huggingface").click();
  const requestId = messages.at(-1).requestId;
  const send = (data) =>
    window.dispatchEvent(new window.MessageEvent("message", { data }));
  send({
    type: "huggingface-search-results",
    requestId,
    models: [
      {
        id: "author/model",
        url: "https://huggingface.co/author/model",
        downloads: 10,
      },
    ],
  });
  const label = document.querySelector(".hub-model-size");
  send({
    type: "huggingface-model-files",
    requestId: requestId - 1,
    id: "author/model",
    entries: [{ size: 999 }],
  });
  assert.match(label.textContent, /loading/);
  send({
    type: "huggingface-model-files",
    requestId,
    id: "author/model",
    entries: [{}],
  });
  assert.match(label.textContent, /not supplied/);
  send({
    type: "huggingface-model-files",
    requestId,
    id: "author/model",
    error: "HTTP 429",
  });
  assert.match(label.textContent, /unavailable: HTTP 429/);
});

test("opens backend tests on double click while keeping single clicks and local files selection-only", (t) => {
  const { document, window, messages, load } = openViewer(t);
  load(demoGraph);
  document.getElementById("open-model").click();
  function list(value) {
    document.getElementById(`open-source-${value}`).click();
    const request = messages.at(-1);
    window.dispatchEvent(
      new window.MessageEvent("message", {
        data: {
          type: "file-list",
          target: "open",
          requestId: request.requestId,
          location: "",
          entries: [{ name: "Relu", path: "42", directory: false }],
        },
      }),
    );
    return document.querySelector("#open-files button");
  }
  const local = list("local");
  local.click();
  local.dispatchEvent(new window.MouseEvent("dblclick", { bubbles: true }));
  assert.equal(messages.filter(({ type }) => type === "open").length, 0);
  const backend = list("backend");
  backend.click();
  assert.equal(document.getElementById("open-file").value, "42");
  assert.equal(messages.filter(({ type }) => type === "open").length, 0);
  backend.dispatchEvent(new window.MouseEvent("dblclick", { bubbles: true }));
  const requests = messages.filter(({ type }) => type === "open");
  assert.equal(requests.length, 1);
  assert.equal(requests[0].source, "backend");
  assert.equal(requests[0].location, "42");
  assert.equal(document.getElementById("open-progress").hidden, false);
  backend.dispatchEvent(new window.MouseEvent("dblclick", { bubbles: true }));
  assert.equal(messages.filter(({ type }) => type === "open").length, 1);
});

test("suggests remembered paths without submitting forms and supports per-entry removal and clearing", (t) => {
  const { document, window, messages, load } = openViewer(t);
  load(demoGraph);
  const local = {
    source: "local",
    kind: "file",
    path: "/models/<b>recent</b>.onnx",
  };
  const directory = { source: "local", kind: "directory", path: "/models" };
  const remote = {
    source: "huggingface",
    kind: "file",
    path: "https://huggingface.co/owner/repo/resolve/main/model.onnx",
  };
  const backend = { source: "backend", kind: "file", path: "42" };
  const saved = { source: "local", kind: "file", path: "/copies/model.onnx" };
  const history = {
    open: [local, directory, remote, backend],
    save: [saved, { ...directory, path: "/copies" }],
  };
  const send = () =>
    window.dispatchEvent(
      new window.MessageEvent("message", {
        data: { type: "path-history", history },
      }),
    );
  send();
  for (const target of ["open", "save"])
    for (const field of ["file", "location"])
      assert.equal(
        document.getElementById(`${target}-${field}`).list.id,
        `${target}-${field}-history`,
      );
  assert.deepEqual(
    [...document.querySelectorAll("#open-file-history option")].map(
      (option) => option.value,
    ),
    [local.path],
  );
  assert.deepEqual(
    [...document.querySelectorAll("#save-file-history option")].map(
      (option) => option.value,
    ),
    [saved.path],
  );
  assert.equal(document.querySelector("#open-history b"), null);
  document.getElementById("open-model").click();
  document.getElementById("open-history").open = true;
  const count = messages.length;
  document.querySelector("#open-history-entries li button").click();
  assert.equal(document.getElementById("open-file").value, local.path);
  assert.equal(document.activeElement.id, "open-file");
  document
    .querySelectorAll("#open-history-entries li")[1]
    .querySelector("button")
    .click();
  assert.equal(document.getElementById("open-location").value, "/models");
  document.querySelector("#save-history-entries li button").click();
  assert.equal(document.getElementById("save-file").value, saved.path);
  assert.equal(messages.length, count);
  const input = document.getElementById("open-file");
  input.value = "/new/model.onnx";
  for (const key of [
    "ArrowLeft",
    "ArrowRight",
    "Home",
    "End",
    "ArrowDown",
    "Escape",
  ]) {
    const event = new window.KeyboardEvent("keydown", {
      key,
      cancelable: true,
    });
    input.dispatchEvent(event);
    assert.equal(event.defaultPrevented, false);
    assert.equal(input.value, "/new/model.onnx");
  }
  document.querySelector("#open-history-entries li button:last-child").click();
  assert.equal(messages.at(-1).type, "path-history-edit");
  assert.equal(messages.at(-1).historyAction, "remove");
  assert.equal(messages.at(-1).historyEntry.path, local.path);
  history.open.shift();
  send();
  assert.equal(
    document.querySelectorAll("#open-file-history option").length,
    0,
  );
  for (const [value, path] of [
    ["huggingface", remote.path],
    ["backend", backend.path],
  ]) {
    document.getElementById(`open-source-${value}`).click();
    assert.equal(
      document.querySelector("#open-file-history option").value,
      path,
    );
  }
  history.open = [remote];
  send();
  assert.equal(document.getElementById("open-history").hidden, true);
  document.getElementById("open-source-huggingface").click();
  assert.equal(document.getElementById("open-history").hidden, false);
  document.getElementById("open-source-local").click();
  assert.equal(document.getElementById("open-history").hidden, true);
  assert.equal(document.getElementById("save-history").hidden, false);
  document.getElementById("save-clear-history").click();
  assert.deepEqual(JSON.parse(JSON.stringify(messages.at(-1))), {
    type: "path-history-edit",
    target: "save",
    historyAction: "clear",
  });
  history.save = [];
  send();
  assert.equal(document.getElementById("save-history").hidden, true);
  assert.equal(
    document.querySelectorAll("#save-file-history option").length,
    0,
  );
});

test("shows real download progress, handles unknown totals and ignores stale operation updates", (t) => {
  const { document, window, messages, load } = openViewer(t);
  load(demoGraph);
  submitOpen(document, window);
  const actionId = messages.at(-1).actionId;
  const progress = document.getElementById("open-progress");
  const global = document.getElementById("load-progress");
  assert.equal(progress.hidden, false);
  assert.equal(progress.hasAttribute("value"), false);
  const send = (data) =>
    window.dispatchEvent(new window.MessageEvent("message", { data }));
  send({
    type: "model-progress",
    target: "open",
    actionId,
    stage: "Downloading Hugging Face model",
    received: 50,
    total: 100,
  });
  for (const bar of [progress, global]) {
    assert.equal(bar.value, 50);
    assert.equal(bar.max, 100);
  }
  send({
    type: "model-progress",
    target: "open",
    actionId: "obsolete",
    stage: "Wrong download",
    received: 99,
    total: 100,
  });
  assert.equal(progress.value, 50);
  send({
    type: "model-progress",
    target: "open",
    actionId,
    stage: "Downloading Hugging Face model",
    received: 75,
  });
  assert.equal(progress.hasAttribute("value"), false);
  assert.match(
    document.getElementById("open-progress-status").textContent,
    /75 bytes/,
  );
  send({
    type: "model-progress",
    target: "open",
    actionId,
    stage: "Opening downloaded model",
  });
  assert.equal(progress.hasAttribute("value"), false);
  send({
    type: "log-entry",
    entry: {
      id: actionId,
      timestamp: new Date().toISOString(),
      action: "Open ONNX model",
      status: "CANCELLED",
    },
  });
  assert.equal(progress.hidden, true);
  send({
    type: "model-progress",
    target: "open",
    actionId,
    stage: "Late download",
    received: 100,
    total: 100,
  });
  assert.equal(progress.hidden, true);
  assert.equal(progress.hasAttribute("value"), false);
});

test("saves without replacing the graph and reports success, cancellation and errors", (t) => {
  const { document, window, messages, load } = openViewer(t);
  const button = document.getElementById("save-model");
  const status = document.getElementById("save-status");
  const send = (data) =>
    window.dispatchEvent(new window.MessageEvent("message", { data }));
  assert.equal(button.disabled, true);
  load(demoGraph);
  const graph = document.getElementById("nodes").innerHTML;
  assert.equal(button.disabled, false);
  for (const result of ["SUCCEEDED", "CANCELLED", "FAILED"]) {
    button.click();
    document
      .getElementById("save-form")
      .dispatchEvent(new window.Event("submit", { cancelable: true }));
    const request = messages.at(-1);
    assert.equal(request.type, "save");
    assert.equal(button.disabled, true);
    assert.equal(button.getAttribute("aria-busy"), "true");
    assert.equal(document.getElementById("save-progress").hidden, false);
    assert.equal(
      document.getElementById("save-progress").hasAttribute("value"),
      false,
    );
    window.dispatchEvent(
      new window.MessageEvent("message", {
        data: {
          type: "model-progress",
          target: "save",
          actionId: request.actionId,
          stage: "Writing model file",
        },
      }),
    );
    assert.match(
      document.getElementById("save-progress-status").textContent,
      /Writing model file/,
    );
    const count = messages.length;
    button.click();
    assert.equal(messages.length, count);
    if (result === "SUCCEEDED")
      send({ type: "save-info", name: "copy.onnx.enc" });
    if (result === "FAILED") send({ type: "save-error", error: "Disk full" });
    send({
      type: "log-entry",
      entry: {
        id: request.actionId,
        action: "Save ONNX model",
        timestamp: new Date().toISOString(),
        status: result,
      },
    });
    assert.equal(button.disabled, false);
    assert.equal(button.getAttribute("aria-busy"), "false");
    assert.equal(document.getElementById("save-progress").hidden, true);
    assert.match(
      status.textContent,
      result === "SUCCEEDED"
        ? /Saved copy.onnx.enc/
        : result === "CANCELLED"
          ? /cancelled/
          : /Disk full/,
    );
    assert.equal(document.getElementById("nodes").innerHTML, graph);
  }
});

test("shows matching adjacent Open/Save sidebar buttons and an explicit storage/encryption grid", (t) => {
  const { document, window, messages, load } = openViewer(t);
  load(demoGraph);
  const open = document.getElementById("open-model");
  const save = document.getElementById("save-model");
  assert.equal(open.textContent, "Open");
  assert.equal(save.textContent, "Save");
  assert.equal(open.nextElementSibling, save);
  assert.equal(open.className, save.className);
  assert.equal(open.parentElement, document.getElementById("sidebar-controls"));
  assert.equal(
    open.className,
    document.getElementById("show-operator").className,
  );
  const dialog = document.getElementById("save-options");
  const form = document.getElementById("save-form");
  const size = document.getElementById("weight-file-size");
  const select = (value) => {
    const input = document.querySelector(`input[value="${value}"]`);
    input.checked = true;
    input.dispatchEvent(new window.Event("change", { bubbles: true }));
  };
  const submit = () =>
    form.dispatchEvent(new window.Event("submit", { cancelable: true }));
  save.click();
  assert.equal(dialog.hidden, false);
  assert.equal(dialog.tagName, "SECTION");
  assert.equal(dialog.closest("aside"), open.closest("aside"));
  assert.equal(form.querySelectorAll('input[type="radio"]').length, 5);
  assert.equal(
    form.querySelector('input[value="external-encrypted"]').disabled,
    true,
  );
  assert.equal(
    form.querySelector("a").href,
    "https://github.com/xadupre/onnx-light/issues/5533",
  );
  let before = messages.length;
  document.getElementById("cancel-save").click();
  assert.equal(dialog.hidden, true);
  assert.equal(messages.length, before);
  save.click();
  before = messages.length;
  select("external-plain");
  assert.equal(size.disabled, false);
  assert.equal(document.getElementById("external-size").hidden, false);
  for (const value of ["", "0", "-1", "0.5", "8589934592"]) {
    size.value = value;
    submit();
    assert.equal(dialog.hidden, false);
    assert.equal(messages.length, before);
    assert.match(
      document.getElementById("save-options-error").textContent,
      /positive/,
    );
  }
  size.value = "2";
  submit();
  assert.equal(dialog.hidden, false);
  assert.deepEqual(
    { ...messages.at(-1), actionId: undefined },
    {
      type: "save",
      storage: "external",
      encrypted: false,
      maxExternalFileSize: 2 * 1024 * 1024,
      location: "model-copy.onnx",
      password: "",
      confirmation: "",
      overwrite: false,
      includeRewritings: false,
      actionId: undefined,
    },
  );
  open.click();
  submitOpen(document, window);
  assert.equal(messages.at(-1).type, "open");
});

test("chooses saved rewriting scenarios, applies next or remaining steps and recovers from errors", (t) => {
  const { document, window, messages, load } = openViewer(t);
  load(demoGraph);
  const send = (data) =>
    window.dispatchEvent(new window.MessageEvent("message", { data }));
  const scenarios = [
    {
      name: "z-last",
      applied: 0,
      available: true,
      steps: [{ pattern: "Identity", path: [], nodes: [] }],
    },
    {
      name: "a-first",
      applied: 0,
      available: true,
      steps: [
        { pattern: "TransposeTranspose", path: [], nodes: [0] },
        { pattern: "SameChildren", path: [], nodes: [] },
      ],
    },
  ];
  const select = document.getElementById("rewriting-scenario");
  const next = document.getElementById("rewriting-next");
  const all = document.getElementById("rewriting-all");
  const section = document.getElementById("rewriting-scenarios");
  assert.equal(section.open, false);
  document.getElementById("fuse-pattern").click();
  send({
    type: "fusion-patterns",
    requestId: messages.at(-1).requestId,
    patterns: [],
    scenarios,
  });
  assert.equal(section.open, true);
  assert.deepEqual(
    [...select.options].map((option) => option.value),
    ["a-first", "z-last"],
  );
  assert.equal(next.disabled, false);
  next.click();
  next.click();
  assert.equal(messages.at(-1).type, "apply-rewriting");
  assert.equal(messages.at(-1).scenario, "a-first");
  assert.equal(messages.at(-1).all, false);
  assert.equal(
    messages.filter((message) => message.type === "apply-rewriting").length,
    1,
  );
  assert.equal(all.disabled, true);
  send({
    type: "fusion-error",
    requestId: messages.at(-1).requestId,
    error: "Invalid stored step",
  });
  assert.equal(next.disabled, false);
  assert.match(
    document.getElementById("fusion-feedback").textContent,
    /Invalid stored step/,
  );
  next.click();
  const nextRequest = messages.at(-1).requestId;
  load(demoGraph, nextRequest);
  scenarios[1].applied = 1;
  scenarios[1].steps[1].nodes = [0];
  scenarios[0].available = false;
  send({
    type: "rewriting-info",
    requestId: nextRequest,
    name: "a-first",
    patterns: [],
    scenarios,
  });
  assert.equal(document.getElementById("fusion-panel").hidden, false);
  assert.match(
    document.getElementById("rewriting-progress").textContent,
    /1 \/ 2/,
  );
  assert.match(
    document.getElementById("rewriting-steps").children[0].textContent,
    /^Applied:/,
  );
  select.value = "z-last";
  select.dispatchEvent(new window.Event("change"));
  assert.equal(all.disabled, true);
  select.value = "a-first";
  select.dispatchEvent(new window.Event("change"));
  all.click();
  assert.equal(messages.at(-1).all, true);
  const allRequest = messages.at(-1).requestId;
  scenarios[1].applied = 2;
  scenarios[1].available = false;
  load(demoGraph, allRequest);
  send({
    type: "rewriting-info",
    requestId: allRequest,
    name: "a-first",
    patterns: [],
    scenarios,
  });
  assert.equal(next.disabled, true);
  assert.equal(all.disabled, true);
  load(demoGraph);
  assert.equal(select.options.length, 0);
  assert.equal(section.open, false);
});

test("selects saved pattern nodes and automatically reveals the next pattern after applying a step", async (t) => {
  const { document, window, messages, load } = openViewer(t);
  const saved = loadModel(
    readFileSync(require.resolve("../examples/rewriting-scenarios.onnx")),
    "Saved rewriting example",
  );
  load(saved.graph());
  const send = (data) =>
    window.dispatchEvent(new window.MessageEvent("message", { data }));
  const selectedNodes = () =>
    [...document.querySelectorAll(".graph-node.selected")].map(
      (node) => node.dataset.id,
    );
  const buttons = () => document.querySelectorAll("#rewriting-steps button");
  document.getElementById("fuse-pattern").click();
  send({
    type: "fusion-patterns",
    requestId: messages.at(-1).requestId,
    patterns: [],
    scenarios: saved.rewritingScenarios(),
  });
  assert.equal(buttons()[1].disabled, true);
  buttons()[0].click();
  assert.deepEqual(selectedNodes(), ["node-0", "node-1"]);
  assert.equal(buttons()[0].getAttribute("aria-current"), "true");
  document.getElementById("rewriting-next").click();
  assert.equal(buttons()[0].disabled, true);
  const request = messages.at(-1);
  const first = await saved.applyRewritingScenario(
    request.scenario,
    request.all,
  );
  load(first.graph(), request.requestId);
  send({
    type: "rewriting-info",
    requestId: request.requestId,
    name: request.scenario,
    patterns: [],
    scenarios: first.rewritingScenarios(),
  });
  assert.deepEqual(selectedNodes(), ["node-1", "node-2"]);
  assert.equal(buttons()[1].getAttribute("aria-current"), "true");
  assert.equal(buttons()[1].disabled, false);
  buttons()[0].click();
  assert.deepEqual(selectedNodes(), ["node-0"]);
  buttons()[1].click();
  assert.deepEqual(selectedNodes(), ["node-1", "node-2"]);
  document.getElementById("rewriting-next").click();
  const remainingRequest = messages.at(-1).requestId;
  const full = await first.applyRewritingScenario(request.scenario, false);
  load(full.graph(), remainingRequest);
  send({
    type: "rewriting-info",
    requestId: remainingRequest,
    name: request.scenario,
    patterns: [],
    scenarios: full.rewritingScenarios(),
  });
  assert.equal(document.getElementById("rewriting-next").disabled, true);
  buttons()[1].click();
  assert.deepEqual(selectedNodes(), ["node-1"]);
  document.getElementById("fuse-pattern").click();
  assert.deepEqual(selectedNodes(), []);
  document.getElementById("fuse-pattern").click();
  send({
    type: "fusion-patterns",
    requestId: messages.at(-1).requestId,
    patterns: [],
    scenarios: full.rewritingScenarios(),
  });
  assert.equal(
    document.querySelector('#rewriting-steps [aria-current="true"]'),
    null,
  );
});

test("rejects queued fusion and scenario completions after local close or navigation", async (t) => {
  const saved = loadModel(
    readFileSync(require.resolve("../examples/rewriting-scenarios.onnx")),
    "Saved rewriting example",
  );
  for (const type of ["fusion-info", "rewriting-info", "optimization-info"]) {
    for (const invalidation of ["close", "navigate", "replace"]) {
      await t.test(`${type}: ${invalidation}`, async (t) => {
        const { document, window, messages, load } = openViewer(t);
        const send = (data) =>
          window.dispatchEvent(new window.MessageEvent("message", { data }));
        const fuse = document.getElementById(
          type === "optimization-info" ? "optimize-model" : "fuse-pattern",
        );
        load(saved.graph());
        fuse.click();
        send({
          type: "fusion-patterns",
          requestId: messages.at(-1).requestId,
          patterns: ["TransposeTranspose"],
          optimizationPatterns: saved.optimizationPatterns(),
          scenarios: saved.rewritingScenarios(),
        });
        if (type === "optimization-info") {
          document.getElementById("run-optimization").click();
        } else if (type === "rewriting-info") {
          document.getElementById("rewriting-next").click();
        } else {
          document.querySelector("#fusion-pattern label").click();
          document.getElementById("preview-fusion").click();
          const preview = await saved.previewFusion("TransposeTranspose");
          send({
            type: "fusion-preview",
            requestId: messages.at(-1).requestId,
            previewId: 1,
            matches: preview.matches,
          });
          document.querySelector("#fusion-matches input").click();
          document.getElementById("apply-fusion").click();
        }
        const requestId = messages.at(-1).requestId;
        const edited =
          type === "optimization-info"
            ? await saved.optimize(["TransposeTranspose"], "Optimize")
            : type === "rewriting-info"
              ? await saved.applyRewritingScenario(
                  "onnx-light-viewer.fusions",
                  false,
                )
              : await saved.applyFusion("TransposeTranspose", [0]);
        if (invalidation === "close") fuse.click();
        load(
          {
            ...edited.graph(),
            canFuse: true,
            breadcrumbs: [
              { name: "Pipeline", path: [] },
              { name: "prefill", path: [{ pipeline: "prefill" }] },
            ],
          },
          requestId,
        );
        if (invalidation === "navigate") {
          document.querySelector("#graph-navigation button").click();
          assert.equal(messages.at(-1).type, "navigate");
        } else if (invalidation === "replace") {
          load(saved.graph());
        }
        send({
          type,
          requestId,
          count: 1,
          operation: "Optimize",
          changed: true,
          matches: edited.fusionHistory().at(-1).matches,
          name: "onnx-light-viewer.fusions",
          patterns: edited.fusionPatterns(),
          scenarios: edited.rewritingScenarios(),
          optimizationReport: edited.optimizationReport(),
        });
        assert.equal(document.getElementById("fusion-panel").hidden, true);
        assert.equal(
          document.getElementById("optimization-panel").hidden,
          true,
        );
        assert.equal(fuse.getAttribute("aria-expanded"), "false");
        assert.equal(
          document.getElementById("rewriting-scenario").options.length,
          0,
        );
        assert.equal(document.getElementById("fusion-status").textContent, "");
        assert.equal(
          document.getElementById("optimization-report").open,
          false,
        );
        assert.equal(
          document.getElementById("show-optimization-report").disabled,
          true,
        );
      });
    }
  }
});

test("opens tabular optimization reports only on request, including no-op runs and reopened Optimize", async (t) => {
  const { document, window, messages, load } = openViewer(t);
  const source = loadModel(
    readFileSync(require.resolve("../examples/optimization.onnx")),
    "optimization",
  );
  const send = (data) =>
    window.dispatchEvent(new window.MessageEvent("message", { data }));
  let model = source;
  load(model.graph());
  const button = document.getElementById("optimize-model");
  const report = document.getElementById("optimization-report");
  const section = document.getElementById("optimization-report-section");
  const showReport = document.getElementById("show-optimization-report");
  assert.equal(report.parentElement, document.body);
  assert.equal(section.contains(report), false);
  for (const changed of [true, false]) {
    button.click();
    send({
      type: "fusion-patterns",
      requestId: messages.at(-1).requestId,
      patterns: model.fusionPatterns(),
      optimizationPatterns: model.optimizationPatterns(),
      optimizationReport: model.optimizationReport(),
    });
    document.getElementById("run-optimization").click();
    const request = messages.at(-1);
    model = await model.optimize(request.patterns, "Optimize");
    load(model.graph(), request.requestId);
    send({
      type: "optimization-info",
      requestId: request.requestId,
      operation: "Optimize",
      changed,
      matches: changed ? model.fusionHistory().at(-1).matches : [],
      patterns: model.fusionPatterns(),
      optimizationPatterns: model.optimizationPatterns(),
      optimizationReport: model.optimizationReport(),
    });
    assert.equal(section.hidden, false);
    assert.equal(showReport.disabled, false);
    assert.equal(report.open, false);
    showReport.click();
    assert.equal(report.open, true);
    const data = model.optimizationReport();
    const rows = [
      ...document.querySelectorAll("#optimization-report-patterns tbody tr"),
    ];
    assert.deepEqual(
      rows.map((row) =>
        [...row.cells].slice(0, 3).map((cell) => cell.textContent),
      ),
      data.patterns.map((pattern) => [
        pattern.name,
        String(pattern.attempts),
        String(pattern.matches),
      ]),
    );
    assert.equal(
      document.querySelector(
        "#optimization-report-summary tbody tr:nth-child(2) td",
      ).textContent,
      String(data.rewrites),
    );
    const elapsed = document.querySelector(
      "#optimization-report-summary tbody tr:nth-child(3)",
    );
    assert.equal(
      elapsed.cells[0].textContent,
      "Full native operation elapsed (ms)",
    );
    const elapsedMs = new Intl.NumberFormat(undefined, {
      maximumFractionDigits: 6,
    }).format(data.wallTimeNs / 1e6);
    assert.equal(elapsed.cells[1].textContent, elapsedMs);
    assert.equal(
      document.querySelectorAll("#optimization-report-phases tbody tr").length,
      5,
    );
    const rejections = data.patterns.flatMap((pattern) => pattern.noMatches);
    if (rejections.length) {
      const text = document.getElementById(
        "optimization-report-rejections",
      ).textContent;
      assert.ok(text.includes(rejections[0].reason));
      const link = document.querySelector("#optimization-report-rejections a");
      assert.ok(
        link.href.startsWith(
          `https://github.com/xadupre/onnx-light/blob/${onnxLightVersion}/onnx_light/`,
        ),
      );
      assert.equal(link.hash, `#L${rejections[0].sourceLine}`);
      assert.equal(
        link.title,
        `${rejections[0].sourceFile}:${rejections[0].sourceLine}`,
      );
      assert.equal(link.textContent, link.href);
    } else
      assert.match(
        document.getElementById("optimization-report-rejections").textContent,
        /No rejection diagnostics/,
      );
    assert.match(
      document.getElementById("optimization-report-subgraphs").textContent,
      /No subgraphs/,
    );
    document.getElementById("close-optimization-report").click();
    assert.equal(report.open, false);
    button.click();
    button.click();
    send({
      type: "fusion-patterns",
      requestId: messages.at(-1).requestId,
      patterns: model.fusionPatterns(),
      optimizationPatterns: model.optimizationPatterns(),
      optimizationReport: model.optimizationReport(),
    });
    showReport.click();
    assert.equal(report.open, true);
    assert.equal(
      document.querySelector(
        "#optimization-report-summary tbody tr:nth-child(3) td",
      ).textContent,
      elapsedMs,
    );
    if (changed) {
      document.getElementById("close-optimization-report").click();
      button.click();
    }
  }
  load(demoGraph);
  assert.equal(section.hidden, true);
  assert.equal(report.open, false);
  assert.equal(
    document.getElementById("optimization-report-body").children.length,
    0,
  );
  assert.equal(showReport.disabled, true);
});

test("renders report details safely and closes the report immediately on navigation", async (t) => {
  const { document, window, messages, load } = openViewer(t);
  const source = loadModel(
    readFileSync(require.resolve("../examples/optimization.onnx")),
    "optimization",
  );
  const optimized = await source.optimize(["TransposeTranspose"], "Optimize");
  const data = optimized.optimizationReport();
  data.subgraphs = [
    {
      path: ["branch <graph>"],
      iterations: 2,
      rewrites: 1,
      elapsedTimeNs: 1500000,
    },
  ];
  data.patterns[0].noMatches = [
    {
      reason: "<img src=x> rejected",
      sourceFile:
        "C:\\build\\onnx_light\\onnx_extensions\\patterns\\transpose\\transpose_pattern.cc",
      sourceLine: 7,
      occurrences: 3,
    },
    {
      reason: "Unknown source",
      sourceFile: "<source>.cc",
      sourceLine: 7,
      occurrences: 1,
    },
  ];
  const send = (value) =>
    window.dispatchEvent(new window.MessageEvent("message", { data: value }));
  load({
    ...optimized.graph(),
    canFuse: true,
    breadcrumbs: [
      { name: "Pipeline", path: [] },
      { name: "prefill", path: [{ pipeline: "prefill" }] },
    ],
  });
  document.getElementById("optimize-model").click();
  const response = {
    type: "fusion-patterns",
    requestId: messages.at(-1).requestId,
    patterns: optimized.fusionPatterns(),
    optimizationPatterns: optimized.optimizationPatterns(),
    optimizationReport: data,
  };
  send(response);
  document.getElementById("show-optimization-report").click();
  const dialog = document.getElementById("optimization-report");
  assert.equal(dialog.open, true);
  assert.equal(
    document.getElementById("optimization-report-scope").textContent,
    "Pipeline / prefill",
  );
  const row = document.querySelector("#optimization-report-subgraphs tbody tr");
  assert.deepEqual(
    [...row.cells].slice(0, 3).map((cell) => cell.textContent),
    ["branch <graph>", "2", "1"],
  );
  assert.match(row.cells[3].textContent, /^1[.,]5$/);
  assert.equal(
    document.querySelector("#optimization-report-rejections tbody td")
      .textContent,
    "<img src=x> rejected",
  );
  assert.equal(dialog.querySelector("img"), null);
  const sourceLink = document.querySelector(
    "#optimization-report-rejections a",
  );
  assert.equal(
    sourceLink.href,
    `https://github.com/xadupre/onnx-light/blob/${onnxLightVersion}/onnx_light/onnx_extensions/patterns/transpose/transpose_pattern.cc#L7`,
  );
  assert.equal(
    document.querySelectorAll("#optimization-report-rejections a").length,
    1,
  );
  assert.match(dialog.textContent, /<source>\.cc:7/);
  document.querySelector("#graph-navigation button").click();
  assert.equal(messages.at(-1).type, "navigate");
  assert.equal(dialog.open, false);
  assert.equal(
    document.getElementById("show-optimization-report").disabled,
    true,
  );
  assert.equal(
    document.getElementById("optimization-report-body").children.length,
    0,
  );
  send(response);
  assert.equal(
    document.getElementById("optimization-report-section").hidden,
    true,
  );
});

test("explains the selected pattern/node and distinguishes rejection, missing diagnostics and matches", async (t) => {
  const { document, window, messages, load } = openViewer(t);
  const source = loadModel(
    readFileSync(require.resolve("./models/fusion.onnx")),
    "fusion",
  );
  const send = (data) =>
    window.dispatchEvent(new window.MessageEvent("message", { data }));
  load(source.graph());
  document.getElementById("fuse-pattern").click();
  send({
    type: "fusion-patterns",
    requestId: messages.at(-1).requestId,
    patterns: source.fusionPatterns(),
    optimizationPatterns: source.optimizationPatterns(),
  });
  const picker = document.getElementById("diagnostic-pattern");
  const explain = document.getElementById("explain-fusion");
  const feedback = document.getElementById("diagnostic-feedback");
  assert.equal(
    document.getElementById("fusion-panel").lastElementChild.id,
    "fusion-diagnostics",
  );
  assert.equal(document.getElementById("diagnostic-source").tagName, "SECTION");
  assert.equal(document.querySelector("#diagnostic-source summary"), null);
  assert.equal(picker.options.length, 0);
  assert.match(
    feedback.textContent,
    /Select one or more native patterns above/,
  );
  document
    .querySelector('#fusion-pattern input[value="TransposeTranspose"]')
    .click();
  assert.deepEqual(
    [...picker.options].map((option) => option.value),
    ["TransposeTranspose"],
  );
  document.querySelector('[data-id="node-0"]').dispatchEvent(
    new window.KeyboardEvent("keydown", {
      key: "Enter",
      bubbles: true,
      ctrlKey: true,
    }),
  );
  assert.equal(explain.disabled, true);
  picker.value = "TransposeTranspose";
  picker.dispatchEvent(new window.Event("change"));
  for (const [anchor, status, label] of [
    [1, "rejected", "Pattern does not match"],
    [4, "op-type-mismatch", "Pattern cannot match this operator type"],
    [0, "matched", "Match found"],
    [0, "unavailable", "No match (reason unavailable)"],
  ]) {
    document
      .querySelector(`[data-id="node-${anchor}"]`)
      .dispatchEvent(
        new window.KeyboardEvent("keydown", { key: "Enter", bubbles: true }),
      );
    assert.equal(explain.disabled, false);
    explain.click();
    const request = messages.at(-1);
    assert.equal(request.type, "explain-fusion");
    assert.equal(request.anchor, anchor);
    assert.equal(request.pattern, "TransposeTranspose");
    assert.equal(explain.disabled, true);
    const diagnostic =
      status === "unavailable"
        ? {
            pattern: request.pattern,
            anchor,
            status,
            reason: "No match. Native diagnostic unavailable.",
          }
        : await source.explainFusion(request.pattern, anchor);
    send({
      type: "fusion-diagnostic",
      requestId: request.requestId,
      diagnostic,
    });
    assert.ok(feedback.textContent.startsWith(`${label}:`));
    assert.equal(explain.disabled, false);
    assert.equal(
      document.getElementById("diagnostic-source").hidden,
      !diagnostic.source,
    );
    if (diagnostic.source) {
      const link = document.querySelector("#diagnostic-location a");
      assert.equal(
        link.href,
        `https://github.com/xadupre/onnx-light/blob/${onnxLightVersion}/onnx_light/onnx_extensions/patterns/transpose/transpose_pattern.cc#L${diagnostic.source.line}`,
      );
      assert.equal(link.textContent, link.href);
      assert.equal(
        link.title,
        `${diagnostic.source.file}:${diagnostic.source.line}`,
      );
    }
  }
  explain.click();
  send({
    type: "fusion-diagnostic-error",
    requestId: messages.at(-1).requestId,
    error: "<unreadable weights>",
  });
  assert.equal(feedback.textContent, "Diagnosis failed: <unreadable weights>");
  assert.equal(feedback.children.length, 0);
  assert.equal(explain.disabled, false);
  document.querySelector('[data-id="node-1"]').dispatchEvent(
    new window.KeyboardEvent("keydown", {
      key: "Enter",
      bubbles: true,
      ctrlKey: true,
    }),
  );
  assert.equal(explain.disabled, true);
  assert.match(
    document.getElementById("diagnostic-node").textContent,
    /Select exactly one/,
  );
  document.getElementById("select-patterns").click();
  assert.deepEqual(
    [...picker.options].map((option) => option.value),
    source
      .fusionPatterns()
      .filter((name) =>
        source.optimizationPatterns().some((pattern) => pattern.name === name),
      )
      .sort(),
  );
  document.getElementById("clear-patterns").click();
  assert.equal(picker.options.length, 0);
  assert.equal(explain.disabled, true);
  assert.match(
    feedback.textContent,
    /Select one or more native patterns above/,
  );
});

test("invalidates pending diagnostics after selection, pattern, close, navigation and model changes", async (t) => {
  for (const invalidate of [
    "selection",
    "pattern",
    "checked-patterns",
    "close",
    "navigate",
    "replace",
  ]) {
    const { document, window, messages, load } = openViewer(t);
    const source = loadModel(
      readFileSync(require.resolve("./models/fusion.onnx")),
      "fusion",
    );
    const send = (data) =>
      window.dispatchEvent(new window.MessageEvent("message", { data }));
    const graph = source.graph();
    load(
      invalidate === "navigate"
        ? {
            ...graph,
            canFuse: true,
            breadcrumbs: [
              { name: "Pipeline", path: [] },
              { name: "prefill", path: [{ pipeline: "prefill" }] },
            ],
          }
        : graph,
    );
    document.getElementById("fuse-pattern").click();
    send({
      type: "fusion-patterns",
      requestId: messages.at(-1).requestId,
      patterns: source.fusionPatterns(),
      optimizationPatterns: source.optimizationPatterns(),
    });
    const picker = document.getElementById("diagnostic-pattern");
    for (const pattern of ["TransposeTranspose", "SameChildren"])
      document
        .querySelector(`#fusion-pattern input[value="${pattern}"]`)
        .click();
    picker.value = "TransposeTranspose";
    picker.dispatchEvent(new window.Event("change"));
    document
      .querySelector('[data-id="node-1"]')
      .dispatchEvent(
        new window.KeyboardEvent("keydown", { key: "Enter", bubbles: true }),
      );
    document.getElementById("explain-fusion").click();
    const request = messages.at(-1);
    assert.equal(request.type, "explain-fusion", invalidate);
    if (invalidate === "selection")
      document
        .querySelector('[data-id="node-0"]')
        .dispatchEvent(
          new window.KeyboardEvent("keydown", { key: "Enter", bubbles: true }),
        );
    else if (invalidate === "pattern") {
      picker.value = "SameChildren";
      picker.dispatchEvent(new window.Event("change"));
    } else if (invalidate === "checked-patterns") {
      document
        .querySelector('#fusion-pattern input[value="TransposeTranspose"]')
        .click();
    } else if (invalidate === "close")
      document.getElementById("fuse-pattern").click();
    else if (invalidate === "navigate")
      document.querySelector("#graph-navigation button").click();
    else load(demoGraph);
    const feedback = document.getElementById("diagnostic-feedback").textContent;
    assert.ok(feedback.length);
    send({
      type: "fusion-diagnostic",
      requestId: request.requestId,
      diagnostic: await source.explainFusion(request.pattern, request.anchor),
    });
    send({
      type: "fusion-diagnostic-error",
      requestId: request.requestId,
      error: "Obsolete failure",
    });
    assert.equal(
      document.getElementById("diagnostic-feedback").textContent,
      feedback,
    );
    assert.equal(document.getElementById("diagnostic-source").hidden, true);
  }
});

test("offers last and all Unfuse actions beside Fuse and preserves repeated undo", (t) => {
  const { document, window, messages } = openViewer(t);
  const selectionGraph = loadModel(
    readFileSync(require.resolve("./models/fusion.onnx")),
    "fusion",
  ).graph();
  const history = [
    {
      matches: [
        { pattern: "TransposeTranspose", nodes: [2] },
        { pattern: "TransposeTranspose", nodes: [0] },
      ],
    },
    {
      matches: [
        { pattern: "SameChildrenFromInput", nodes: [0] },
        { pattern: "MatMulScaleFusion", nodes: [] },
      ],
      stage: "prefill",
    },
  ];
  const send = (data) =>
    window.dispatchEvent(new window.MessageEvent("message", { data }));
  const graph = (fusionCount) =>
    send({
      type: "graph",
      graph: selectionGraph,
      file: {
        fusionCount,
        fusionHistory: history.slice(0, fusionCount),
        directory: "/models",
        savePath: "/models/copy.onnx",
      },
    });
  const undo = document.getElementById("undo-fusion");
  const last = document.getElementById("unfuse-last");
  const all = document.getElementById("unfuse-all");
  const options = document.getElementById("unfuse-panel");
  const historyList = document.getElementById("unfuse-history");
  assert.equal(undo.textContent, "Unfuse");
  assert.equal(
    document.getElementById("fuse-pattern").nextElementSibling,
    undo,
  );
  assert.equal(undo.parentElement.id, "sidebar-controls");
  graph(2);
  assert.equal(undo.disabled, false);
  undo.click();
  assert.equal(options.hidden, false);
  assert.ok(
    document
      .getElementById("unfuse-all")
      .parentElement.compareDocumentPosition(historyList) &
      window.Node.DOCUMENT_POSITION_FOLLOWING,
  );
  assert.equal(historyList.children.length, 2);
  assert.equal(historyList.start, 2);
  assert.equal(
    historyList.firstElementChild.querySelector("strong").textContent,
    "Last operation (prefill)",
  );
  assert.deepEqual(
    [...historyList.firstElementChild.querySelectorAll("li")].map(
      (item) => item.textContent,
    ),
    ["SameChildrenFromInput (match 1)", "MatMulScaleFusion (match 2)"],
  );
  assert.deepEqual(
    [...historyList.lastElementChild.querySelectorAll("button")].map(
      (button) => button.textContent,
    ),
    ["TransposeTranspose (match 1)", "TransposeTranspose (match 2)"],
  );
  assert.equal(undo.getAttribute("aria-expanded"), "true");
  assert.equal(
    messages.some(({ type }) => type === "undo-fusion"),
    false,
  );
  const patternButtons = historyList.querySelectorAll("button");
  patternButtons[0].click();
  assert.equal(messages.at(-1).type, "select-fusion");
  assert.equal(messages.at(-1).operation, 1);
  assert.equal(messages.at(-1).match, 0);
  assert.equal(messages.at(-1).pattern, "SameChildrenFromInput");
  assert.equal(patternButtons[1].disabled, true);
  assert.match(patternButtons[1].title, /No nodes remain/);
  const previous = messages.at(-1);
  const graphMarkup = document
    .querySelector(".graph-node")
    .getAttribute("transform");
  patternButtons[2].click();
  assert.equal(messages.at(-1), previous);
  assert.deepEqual(
    [...document.querySelectorAll(".graph-node.selected")].map(
      (node) => node.dataset.id,
    ),
    ["node-2"],
  );
  patternButtons[3].click();
  assert.equal(messages.at(-1), previous);
  assert.deepEqual(
    [...document.querySelectorAll(".graph-node.selected")].map(
      (node) => node.dataset.id,
    ),
    ["node-0"],
  );
  assert.equal(
    document.querySelector(".graph-node").getAttribute("transform"),
    graphMarkup,
  );
  send({
    type: "graph",
    graph: selectionGraph,
    file: { fusionCount: 2, fusionHistory: history },
    selectedNodes: [0],
  });
  assert.deepEqual(
    [...document.querySelectorAll(".graph-node.selected")].map(
      (node) => node.dataset.id,
    ),
    ["node-0"],
  );
  last.click();
  last.click();
  assert.equal(messages.filter(({ type }) => type === "undo-fusion").length, 1);
  assert.equal(messages.at(-1).all, false);
  assert.equal(undo.getAttribute("aria-busy"), "true");
  assert.equal(all.disabled, true);
  assert.ok(
    [...historyList.querySelectorAll("button")].every(
      (button) => button.disabled,
    ),
  );
  graph(1);
  assert.equal(historyList.children.length, 1);
  assert.equal(historyList.start, 1);
  assert.equal(
    historyList.querySelector("strong").textContent,
    "Last operation",
  );
  assert.match(historyList.textContent, /TransposeTranspose/);
  assert.doesNotMatch(historyList.textContent, /SameChildrenFromInput/);
  assert.equal(undo.disabled, false);
  assert.equal(options.hidden, false);
  assert.match(
    document.getElementById("unfuse-count").textContent,
    /^1 fusion operation /,
  );
  last.click();
  send({ type: "undo-fusion-error", error: "Reload failed" });
  assert.equal(historyList.children.length, 1);
  assert.equal(undo.disabled, false);
  assert.equal(last.disabled, false);
  assert.equal(all.disabled, false);
  assert.match(
    document.getElementById("fusion-status").textContent,
    /Reload failed/,
  );
  document.getElementById("close-unfuse").click();
  assert.equal(options.hidden, true);
  undo.click();
  assert.equal(options.hidden, false);
  graph(2);
  all.click();
  all.click();
  assert.equal(messages.filter(({ type }) => type === "undo-fusion").length, 3);
  assert.equal(messages.at(-1).all, true);
  graph(0);
  assert.equal(historyList.children.length, 0);
  assert.equal(options.hidden, true);
  assert.equal(undo.getAttribute("aria-expanded"), "false");
  assert.equal(last.disabled, true);
  assert.equal(all.disabled, true);
  document.getElementById("save-model").click();
  document.getElementById("save-rewritings").checked = true;
  document
    .getElementById("save-form")
    .dispatchEvent(new window.Event("submit", { cancelable: true }));
  assert.equal(messages.at(-1).includeRewritings, true);
  graph(0);
  assert.equal(undo.disabled, true);
});

test("makes Open and Save mutually exclusive while preserving other sidebar panels", (t) => {
  const { document, window, messages, load } = openViewer(t);
  load(demoGraph);
  for (const [first, second] of [
    ["open", "save"],
    ["save", "open"],
  ]) {
    const firstPanel = document.getElementById(
      first === "open" ? "open-panel" : "save-options",
    );
    const secondPanel = document.getElementById(
      second === "open" ? "open-panel" : "save-options",
    );
    if (firstPanel.hidden) document.getElementById(`${first}-model`).click();
    const previous = messages.at(-1);
    document.getElementById(`${first}-password`).value = "temporary";
    document.getElementById(`${second}-model`).click();
    assert.equal(firstPanel.hidden, true);
    assert.equal(secondPanel.hidden, false);
    assert.equal(
      document.getElementById(`${first}-model`).getAttribute("aria-expanded"),
      "false",
    );
    assert.equal(
      document.getElementById(`${second}-model`).getAttribute("aria-expanded"),
      "true",
    );
    assert.equal(document.getElementById(`${first}-password`).value, "");
    assert.equal(document.getElementById("operator-panel").hidden, false);
    assert.equal(document.getElementById("find-panel").hidden, false);
    window.dispatchEvent(
      new window.MessageEvent("message", {
        data: {
          type: "file-list-error",
          target: first,
          requestId: previous.requestId,
          error: "Old listing",
        },
      }),
    );
    assert.doesNotMatch(firstPanel.textContent, /Old listing/);
  }
  document.getElementById("open-model").click();
  assert.equal(document.getElementById("open-panel").hidden, true);
  assert.equal(document.getElementById("save-options").hidden, true);
});

test("closes Open only after a successful opening and allows reopening the selector", (t) => {
  const { document, window, messages, load } = openViewer(t);
  const open = document.getElementById("open-model");
  const panel = document.getElementById("open-panel");
  const complete = (status) => {
    window.dispatchEvent(
      new window.MessageEvent("message", {
        data: {
          type: "log-entry",
          entry: {
            id: messages.at(-1).actionId,
            action: "Open ONNX model",
            timestamp: new Date().toISOString(),
            status,
          },
        },
      }),
    );
  };
  load(demoGraph);
  open.click();
  for (const status of ["FAILED", "CANCELLED"]) {
    document.getElementById("open-file").focus();
    submitOpen(document, window);
    complete(status);
    assert.equal(panel.hidden, false);
    assert.equal(open.getAttribute("aria-expanded"), "true");
    assert.equal(document.activeElement, document.getElementById("open-file"));
  }
  document.getElementById("open-file").focus();
  submitOpen(document, window);
  assert.equal(panel.hidden, false);
  complete("SUCCEEDED");
  load(demoGraph);
  assert.equal(panel.hidden, true);
  assert.equal(open.getAttribute("aria-expanded"), "false");
  assert.equal(open.disabled, false);
  assert.equal(document.activeElement, open);
  open.click();
  assert.equal(panel.hidden, false);
  assert.equal(open.getAttribute("aria-expanded"), "true");
  document.getElementById("open-file").focus();
  submitOpen(document, window);
  const other = document.getElementById("show-metadata");
  other.focus();
  complete("SUCCEEDED");
  assert.equal(panel.hidden, true);
  assert.equal(document.activeElement, other);
});

test("browses folders inline, selects files without opening and ignores stale listings", (t) => {
  const { document, window, messages, load } = openViewer(t);
  load(demoGraph);
  const send = (data) =>
    window.dispatchEvent(new window.MessageEvent("message", { data }));
  const open = document.getElementById("open-model");
  open.click();
  assert.equal(document.getElementById("open-panel").hidden, false);
  assert.equal(open.getAttribute("aria-expanded"), "true");
  const request = messages.at(-1);
  assert.equal(request.type, "browse-files");
  assert.equal(
    messages.some((message) => message.type === "open"),
    false,
  );
  send({
    type: "file-list",
    target: "open",
    requestId: request.requestId,
    location: "/models",
    parent: "/",
    entries: [
      { name: "nested", path: "/models/nested", directory: true },
      { name: "<model>.onnx", path: "/models/<model>.onnx", directory: false },
    ],
  });
  const list = document.getElementById("open-files");
  assert.equal(list.querySelector("model"), null);
  document.getElementById("open-filter").value = "MODEL";
  document
    .getElementById("open-filter")
    .dispatchEvent(new window.Event("input"));
  assert.deepEqual(
    [...list.children].map((item) => item.hidden),
    [true, false],
  );
  document.getElementById("open-filter").value = "";
  document
    .getElementById("open-filter")
    .dispatchEvent(new window.Event("input"));
  list.querySelectorAll("button")[1].click();
  assert.equal(
    document.getElementById("open-file").value,
    "/models/<model>.onnx",
  );
  assert.equal(
    messages.some((message) => message.type === "open"),
    false,
  );
  list.querySelector("button").click();
  const nested = messages.at(-1);
  assert.equal(nested.location, "/models/nested");
  send({
    type: "file-list-error",
    target: "open",
    requestId: request.requestId,
    error: "Old error",
  });
  assert.equal(document.getElementById("open-error").textContent, "");
  send({
    type: "file-list-error",
    target: "open",
    requestId: nested.requestId,
    error: "Access denied",
  });
  assert.equal(
    document.getElementById("open-error").textContent,
    "Access denied",
  );
  assert.equal(list.children.length, 0);
  document.getElementById("open-password").value = "not-persisted";
  open.click();
  assert.equal(document.getElementById("open-panel").hidden, true);
  assert.equal(document.getElementById("open-password").value, "");
  send({
    type: "file-list",
    target: "open",
    requestId: nested.requestId,
    location: "/old",
    entries: [],
  });
  assert.equal(
    document.getElementById("open-location").value,
    "/models/nested",
  );
});

test("preserves save destination separators while browsing directories inline", (t) => {
  const { document, window, messages, load } = openViewer(t);
  load(demoGraph);
  document.getElementById("save-model").click();
  const requestId = messages.at(-1).requestId;
  for (const [location, expected] of [
    ["/models", "/models/copy.onnx"],
    ["/", "/copy.onnx"],
    ["/models/with\\backslash", "/models/with\\backslash/copy.onnx"],
    ["C:\\models", "C:\\models\\copy.onnx"],
    ["C:\\", "C:\\copy.onnx"],
    ["C:\\models\\", "C:\\models\\copy.onnx"],
    ["C:/models", "C:/models/copy.onnx"],
    ["\\\\server\\share\\models", "\\\\server\\share\\models\\copy.onnx"],
    ["\\\\server\\share\\", "\\\\server\\share\\copy.onnx"],
    ["vscode-remote://host/models", "vscode-remote://host/models/copy.onnx"],
  ]) {
    document.getElementById("save-file").value = "copy.onnx";
    window.dispatchEvent(
      new window.MessageEvent("message", {
        data: {
          type: "file-list",
          target: "save",
          requestId,
          location,
          entries: [],
        },
      }),
    );
    assert.equal(document.getElementById("save-file").value, expected);
  }
});

test("toggles separated, stacked sidebar panels independently from Op Data Find Fuse controls", (t) => {
  const { document, window, load } = openViewer(t);
  const style = document.createElement("style");
  style.textContent = readFileSync(
    require.resolve("../media/viewer.css"),
    "utf8",
  );
  document.head.append(style);
  load(demoGraph);
  const sidebar = document.querySelector("aside");
  const controls = document.getElementById("sidebar-controls");
  assert.equal(sidebar.firstElementChild, controls);
  const allButtons = [...controls.querySelectorAll("button")];
  assert.deepEqual(
    allButtons.map((button) => button.lastChild.textContent.trim()),
    [
      "Open",
      "Save",
      "Convert",
      "Find",
      "Optimize",
      "Fuse",
      "Unfuse",
      "Op",
      "Data",
    ],
  );
  assert.match(
    window.getComputedStyle(controls).gridTemplateColumns,
    /^repeat\(3,/,
  );
  const buttons = [
    "show-find",
    "show-operator",
    "show-metadata",
    "optimize-model",
    "fuse-pattern",
  ].map((id) => document.getElementById(id));
  const panels = buttons.map((button) =>
    document.getElementById(button.getAttribute("aria-controls")),
  );
  assert.deepEqual(
    [...sidebar.querySelectorAll(":scope > section")].filter(
      (panel) =>
        ![
          "open-panel",
          "save-options",
          "export-panel",
          "unfuse-panel",
        ].includes(panel.id),
    ),
    panels,
  );
  assert.deepEqual(
    panels.map((panel) => panel.hidden),
    [false, false, true, true, true],
  );
  const panelStyle = [...style.sheet.cssRules].find(
    (rule) => rule.selectorText === ".inspector-panel",
  ).style;
  assert.match(panelStyle.border, /solid/);
  const inspectorTitle = document.getElementById("operator-title");
  const inspectorFont = window.getComputedStyle(inspectorTitle);
  for (const panel of sidebar.querySelectorAll(":scope > .inspector-panel")) {
    const title = panel.firstElementChild;
    assert.equal(title.tagName, "H2");
    assert.equal(title.id, panel.getAttribute("aria-labelledby"));
    assert.ok(title.classList.contains("panel-title"));
    const titleStyle = window.getComputedStyle(title);
    for (const property of [
      "fontFamily",
      "fontSize",
      "fontWeight",
      "letterSpacing",
    ])
      assert.equal(titleStyle[property], inspectorFont[property]);
    assert.equal(titleStyle.textTransform, "uppercase");
  }
  const titleStyle = [...style.sheet.cssRules].find(
    (rule) => rule.selectorText === ".inspector-panel > .panel-title",
  ).style;
  assert.equal(titleStyle.color, "var(--bg)");
  assert.equal(titleStyle.background, "var(--text)");
  const verify = () => {
    for (const [index, panel] of panels.entries()) {
      assert.equal(
        buttons[index].getAttribute("aria-expanded"),
        String(!panel.hidden),
      );
      assert.ok(parseFloat(window.getComputedStyle(panel).marginBottom) > 0);
    }
  };
  verify();
  for (const [index, button] of buttons.entries()) {
    const before = panels.map((panel) => panel.hidden);
    button.click();
    assert.deepEqual(
      panels.map((panel) => panel.hidden),
      before.map((hidden, position) => (position === index ? !hidden : hidden)),
    );
    verify();
    button.click();
    assert.deepEqual(
      panels.map((panel) => panel.hidden),
      before,
    );
    verify();
  }
  buttons[2].click();
  buttons[3].click();
  buttons[4].click();
  assert.deepEqual(
    panels.map((panel) => panel.hidden),
    [false, false, false, false, false],
  );
  buttons[0].click();
  buttons[1].click();
  const title = document.getElementById("node-title").textContent;
  buttons[0].click();
  buttons[1].click();
  assert.equal(document.getElementById("node-title").textContent, title);
  assert.equal(
    document.getElementById("node-search").closest("section").hidden,
    false,
  );
});

test("runs matching explicitly, highlights candidates and fuses only checked matches", async (t) => {
  const { document, window, messages, load } = openViewer(t);
  const native = loadModel(
    readFileSync(require.resolve("./models/fusion.onnx")),
    "fusion",
  );
  const preview = await native.previewFusion("TransposeTranspose");
  const send = (data) =>
    window.dispatchEvent(new window.MessageEvent("message", { data }));
  const button = document.getElementById("fuse-pattern");
  assert.equal(
    button.parentElement,
    document.getElementById("sidebar-controls"),
  );
  assert.equal(button.textContent, "Fuse");
  assert.equal(
    button.className,
    document.getElementById("show-operator").className,
  );
  assert.equal(button.disabled, true);
  load(native.graph());
  document.querySelector("aside").scrollTop = 500;
  button.click();
  assert.equal(document.querySelector("aside").scrollTop, 0);
  assert.equal(document.getElementById("fusion-panel").hidden, false);
  assert.equal(
    document.getElementById("fusion-panel").parentElement.tagName,
    "ASIDE",
  );
  send({
    type: "fusion-patterns",
    requestId: messages.at(-1).requestId,
    patterns: ["TransposeTranspose"],
  });
  assert.equal(document.getElementById("preview-fusion").disabled, true);
  const pattern = document.getElementById("fusion-pattern");
  pattern.querySelector("label").click();
  assert.equal(messages.at(-1).type, "cancel-fusion");
  document.getElementById("preview-fusion").click();
  const requestId = messages.at(-1).requestId;
  assert.deepEqual(Array.from(messages.at(-1).patterns), [
    "TransposeTranspose",
  ]);
  send({
    type: "fusion-preview",
    requestId,
    previewId: 42,
    matches: preview.matches,
  });
  const matches = document.getElementById("fusion-matches");
  assert.equal(matches.closest("details").open, true);
  assert.equal(
    document.getElementById("apply-fusion").closest("details"),
    matches.closest("details"),
  );
  assert.equal(document.getElementById("close-fusion"), null);
  assert.equal(
    document.getElementById("preview-fusion").closest("details"),
    pattern.closest("details"),
  );
  assert.equal(
    document.getElementById("fusion-feedback").closest("details"),
    pattern.closest("details"),
  );
  assert.equal(matches.querySelector("button"), null);
  assert.equal(matches.children.length, 2);
  assert.match(matches.textContent, /first:[\s\S]*second:/);
  assert.match(matches.textContent, /Identity\(x\) -> a/);
  assert.equal(document.getElementById("apply-fusion").disabled, true);
  assert.equal(document.querySelectorAll(".graph-node.selected").length, 1);
  matches.children[1].querySelector("pre").click();
  assert.deepEqual(
    [...document.querySelectorAll(".graph-node.selected")].map(
      (node) => node.dataset.id,
    ),
    ["node-2", "node-3"],
  );
  assert.equal(matches.querySelectorAll("input:checked").length, 0);
  const first = matches.querySelector("input");
  first.checked = true;
  first.dispatchEvent(new window.Event("change"));
  first.checked = false;
  first.dispatchEvent(new window.Event("change"));
  assert.equal(first.checked, false);
  assert.equal(document.getElementById("apply-fusion").disabled, true);
  assert.equal(
    messages.some((message) => message.type === "apply-fusion"),
    false,
  );
  assert.equal(matches.querySelector(".approve-fusion, .reject-fusion"), null);
  first.click();
  document.getElementById("apply-fusion").click();
  assert.deepEqual(Array.from(messages.at(-1).candidates), [0]);
  assert.equal(messages.at(-1).previewId, 42);
  send({
    type: "fusion-error",
    requestId: messages.at(-1).requestId,
    error: "Please retry.",
  });
  assert.match(
    document.getElementById("fusion-feedback").textContent,
    /Please retry/,
  );
  document.getElementById("select-fusions").click();
  assert.equal(matches.querySelectorAll("input:checked").length, 2);
  document.getElementById("clear-fusions").click();
  assert.equal(matches.querySelectorAll("input:checked").length, 0);
  assert.equal(document.getElementById("apply-fusion").disabled, true);
  document.getElementById("select-fusions").click();
  document.getElementById("apply-fusion").click();
  assert.deepEqual(Array.from(messages.at(-1).candidates), [0, 1]);
  load(native.graph());
  send({
    type: "fusion-preview",
    requestId,
    previewId: 42,
    matches: preview.matches,
  });
  assert.equal(matches.children.length, 0);
  assert.equal(document.getElementById("fusion-panel").hidden, true);
  const graph = native.graph();
  load({
    ...graph,
    breadcrumbs: [...graph.breadcrumbs, { name: "nested", path: [] }],
  });
  assert.equal(button.disabled, true);
});

test("selects multiple patterns, reruns matching from scratch and ignores old previews", async (t) => {
  const { document, window, messages, load } = openViewer(t);
  const native = loadModel(
    readFileSync(require.resolve("./models/fusion.onnx")),
    "fusion",
  );
  load(native.graph());
  const send = (data) =>
    window.dispatchEvent(new window.MessageEvent("message", { data }));
  document.getElementById("fuse-pattern").click();
  send({
    type: "fusion-patterns",
    requestId: messages.at(-1).requestId,
    patterns: ["TransposeTranspose", "Identity"],
  });
  const pattern = document.getElementById("fusion-pattern");
  const transpose = pattern.querySelector('input[value="TransposeTranspose"]');
  const identity = pattern.querySelector('input[value="Identity"]');
  assert.equal(document.getElementById("clear-patterns").disabled, true);
  document.getElementById("select-patterns").click();
  assert.equal(pattern.querySelectorAll("input:checked").length, 2);
  assert.equal(document.getElementById("preview-fusion").disabled, false);
  document.getElementById("clear-patterns").click();
  assert.equal(pattern.querySelectorAll("input:checked").length, 0);
  assert.equal(document.getElementById("preview-fusion").disabled, true);
  transpose.parentElement.click();
  assert.equal(transpose.checked, true);
  transpose.parentElement.click();
  assert.equal(transpose.checked, false);
  assert.equal(document.getElementById("preview-fusion").disabled, true);
  transpose.parentElement.click();
  document.getElementById("preview-fusion").click();
  const previous = messages.at(-1).requestId;
  const { matches } = await native.previewFusion("TransposeTranspose");
  send({ type: "fusion-preview", requestId: previous, previewId: 1, matches });
  const approved = document.querySelector("#fusion-matches input");
  approved.checked = true;
  approved.dispatchEvent(new window.Event("change"));
  assert.equal(document.getElementById("apply-fusion").disabled, false);
  identity.parentElement.click();
  assert.equal(pattern.querySelectorAll("input:checked").length, 2);
  assert.equal(transpose.checked, true);
  assert.equal(messages.at(-1).type, "cancel-fusion");
  document.getElementById("preview-fusion").click();
  const current = messages.at(-1).requestId;
  assert.deepEqual(Array.from(messages.at(-1).patterns), [
    "Identity",
    "TransposeTranspose",
  ]);
  const checkedBefore = [...pattern.querySelectorAll("input")].map(
    (input) => input.checked,
  );
  document.getElementById("clear-patterns").click();
  transpose.parentElement.click();
  assert.deepEqual(
    [...pattern.querySelectorAll("input")].map((input) => input.checked),
    checkedBefore,
  );
  assert.notEqual(current, previous);
  assert.equal(document.querySelectorAll(".graph-node.selected").length, 0);
  assert.equal(document.querySelectorAll("#fusion-matches input").length, 0);
  assert.equal(document.getElementById("apply-fusion").disabled, true);
  send({ type: "fusion-preview", requestId: previous, previewId: 1, matches });
  assert.equal(document.querySelectorAll("#fusion-matches input").length, 0);
  send({
    type: "fusion-preview",
    requestId: current,
    previewId: 2,
    matches: [],
  });
  assert.match(
    document.getElementById("fusion-feedback").textContent,
    /No matching candidates/,
  );
  document.getElementById("preview-fusion").click();
  assert.notEqual(messages.at(-1).requestId, current);
  assert.equal(messages.at(-1).type, "preview-fusion");
  document.getElementById("fuse-pattern").click();
  assert.equal(document.getElementById("fusion-panel").hidden, true);
  assert.equal(
    document.getElementById("fuse-pattern").getAttribute("aria-expanded"),
    "false",
  );
  assert.equal(messages.at(-1).type, "cancel-fusion");
  send({
    type: "fusion-preview",
    requestId: messages.at(-2).requestId,
    previewId: 3,
    matches,
  });
  assert.equal(document.querySelectorAll("#fusion-matches input").length, 0);
  document.getElementById("fuse-pattern").click();
  assert.equal(document.getElementById("fusion-panel").hidden, false);
  document.getElementById("fuse-pattern").click();
  assert.equal(document.getElementById("fusion-panel").hidden, true);
  assert.equal(
    messages.some((message) => message.type === "apply-fusion"),
    false,
  );
});

test("explains parallel transpose matching with explicit before and after operators", async (t) => {
  const { document, window, messages, load } = openViewer(t);
  const native = loadModel(
    readFileSync(require.resolve("./models/fusion.onnx")),
    "fusion",
  );
  load(native.graph());
  const send = (data) =>
    window.dispatchEvent(new window.MessageEvent("message", { data }));
  document.getElementById("fuse-pattern").click();
  send({
    type: "fusion-patterns",
    requestId: messages.at(-1).requestId,
    patterns: ["SameChildrenFromInput"],
  });
  document.querySelector("#fusion-pattern label").click();
  document.getElementById("preview-fusion").click();
  const preview = await native.previewFusion("SameChildrenFromInput");
  send({
    type: "fusion-preview",
    requestId: messages.at(-1).requestId,
    previewId: 1,
    matches: preview.matches.map((match) => ({
      ...match,
      pattern: "SameChildrenFromInput",
    })),
  });
  const item = document.querySelector("#fusion-matches li");
  assert.equal(
    item.querySelector("label").textContent,
    "SameChildrenFromInput",
  );
  assert.deepEqual(
    [...item.querySelectorAll(".fusion-stage strong")].map(
      (heading) => heading.textContent,
    ),
    ["Before", "After"],
  );
  const [before, after] = item.querySelectorAll("pre");
  assert.equal(
    before.textContent,
    "first: Transpose(x, perm=[1, 0]) -> xt\nthird: Transpose(x, perm=[1, 0]) -> yt",
  );
  assert.equal(
    after.textContent,
    "Transpose(x, perm=[1, 0]) -> xt\nIdentity(xt) -> yt",
  );
  after.click();
  assert.deepEqual(
    [...document.querySelectorAll(".graph-node.selected")].map(
      (node) => node.dataset.id,
    ),
    ["node-0", "node-2"],
  );
  assert.equal(item.querySelector("input").checked, false);
});

test("darkens matches intersecting any earlier candidate independently of selection", (t) => {
  const { document, window, messages, load } = openViewer(t);
  const native = loadModel(
    readFileSync(require.resolve("./models/fusion.onnx")),
    "fusion",
  );
  load(native.graph());
  const send = (data) =>
    window.dispatchEvent(new window.MessageEvent("message", { data }));
  document.getElementById("fuse-pattern").click();
  send({
    type: "fusion-patterns",
    requestId: messages.at(-1).requestId,
    patterns: ["TransposeTranspose"],
  });
  document.querySelector("#fusion-pattern label").click();
  const matches = [[0], [0, 1], [1], [2]].map((nodes) => ({
    pattern: "TransposeTranspose",
    nodes,
    replacement: [],
  }));
  const showMatches = (matches) => {
    document.getElementById("preview-fusion").click();
    send({
      type: "fusion-preview",
      requestId: messages.at(-1).requestId,
      previewId: 1,
      matches,
    });
    return [...document.querySelectorAll("#fusion-matches li")];
  };
  const overlaps = (items) =>
    items.map((item) => item.classList.contains("fusion-match-overlap"));
  const graph = document.getElementById("nodes").innerHTML;
  const items = showMatches(matches);
  assert.deepEqual(overlaps(items), [false, true, true, false]);
  assert.equal(document.getElementById("nodes").innerHTML, graph);
  assert.match(items[1].title, /earlier match/);
  assert.equal(items[0].title, "");
  document.getElementById("select-fusions").click();
  assert.deepEqual(
    items.map((item) => item.querySelector("input").checked),
    [true, false, true, true],
  );
  assert.deepEqual(overlaps(items), [false, true, true, false]);
  document.getElementById("clear-fusions").click();
  assert.deepEqual(overlaps(items), [false, true, true, false]);
  assert.deepEqual(overlaps(showMatches([...matches].reverse())), [
    false,
    false,
    true,
    true,
  ]);
  assert.deepEqual(overlaps(showMatches([matches[1]])), [false]);
});

test("selects only non-overlapping matches and sends the external-weight reuse choice", async (t) => {
  const { document, window, messages, load } = openViewer(t);
  const native = loadModel(
    readFileSync(require.resolve("./models/fusion.onnx")),
    "fusion",
  );
  load(native.graph());
  const send = (data) =>
    window.dispatchEvent(new window.MessageEvent("message", { data }));
  document.getElementById("fuse-pattern").click();
  send({
    type: "fusion-patterns",
    requestId: messages.at(-1).requestId,
    patterns: ["TransposeTranspose"],
  });
  const pattern = document.getElementById("fusion-pattern");
  pattern.querySelector("label").click();
  document.getElementById("preview-fusion").click();
  const preview = await native.previewFusion("TransposeTranspose");
  preview.matches[1].nodes = [1, 2];
  send({
    type: "fusion-preview",
    requestId: messages.at(-1).requestId,
    previewId: 1,
    matches: preview.matches,
  });
  const inputs = [...document.querySelectorAll("#fusion-matches input")];
  for (const input of inputs) input.checked = true;
  inputs[0].dispatchEvent(new window.Event("change"));
  assert.equal(document.getElementById("apply-fusion").disabled, true);
  assert.match(
    document.getElementById("fusion-feedback").textContent,
    /overlap/,
  );
  document.getElementById("select-fusions").click();
  assert.deepEqual(
    inputs.map((input) => input.checked),
    [true, false],
  );
  document.getElementById("fuse-pattern").click();
  assert.equal(document.querySelectorAll(".graph-node.selected").length, 0);
  assert.equal(document.getElementById("fusion-panel").hidden, true);
  assert.equal(messages.at(-1).type, "cancel-fusion");
  assert.equal(
    messages.some((message) => message.type === "apply-fusion"),
    false,
  );
  document.getElementById("save-model").click();
  const reuse = document.querySelector('input[value="reuse-plain"]');
  reuse.checked = true;
  reuse.dispatchEvent(new window.Event("change", { bubbles: true }));
  assert.equal(document.getElementById("weight-file-size").disabled, false);
  document
    .getElementById("save-form")
    .dispatchEvent(new window.Event("submit", { cancelable: true }));
  assert.equal(messages.at(-1).storage, "reuse");
  assert.equal(messages.at(-1).maxExternalFileSize, 5120 * 1024 * 1024);
});

test("shows loading progress until success, failure or cancellation, including initial loads", (t) => {
  const { document, window, load, messages } = openViewer(t);
  const progress = document.getElementById("load-progress");
  const state = (data) =>
    window.dispatchEvent(new window.MessageEvent("message", { data }));
  assert.equal(progress.hidden, false);
  assert.equal(progress.hasAttribute("value"), false);
  assert.equal(progress.getAttribute("aria-label"), "Loading ONNX model");
  assert.equal(document.getElementById("open-model").disabled, true);
  load(demoGraph);
  assert.equal(progress.hidden, true);
  document.getElementById("open-model").click();
  submitOpen(document, window);
  assert.equal(progress.hidden, false);
  const request = messages.at(-1);
  state({
    type: "log-entry",
    entry: {
      id: request.actionId,
      action: "Open model",
      timestamp: new Date().toISOString(),
      status: "CANCELLED",
    },
  });
  assert.equal(progress.hidden, true);
  state({ type: "load-state", pending: true });
  assert.equal(progress.hidden, false);
  assert.equal(
    document.getElementById("model-view").getAttribute("aria-busy"),
    "true",
  );
  state({ type: "load-error", error: "Invalid model" });
  assert.equal(progress.hidden, true);
  const error = document.getElementById("model-error");
  assert.equal(error.hidden, false);
  assert.match(error.textContent, /Invalid model/);
  assert.match(
    document.getElementById("load-status").textContent,
    /Invalid model/,
  );
  assert.equal(document.getElementById("open-model").disabled, false);
  state({ type: "load-state", pending: true });
  assert.equal(error.hidden, true);
  load(demoGraph);
  assert.equal(error.hidden, true);
  assert.equal(progress.hidden, true);
  assert.equal(
    document.getElementById("model-view").getAttribute("aria-busy"),
    "false",
  );
});

test("keeps replacement loading visible when an in-flight navigation renders", (t) => {
  const { document, messages, load, window } = openViewer(t);
  const model = loadModel(
    readFileSync(require.resolve("./models/subgraphs.onnx")),
    "subgraphs.onnx",
  );
  const root = model.graph();
  const path = root.nodes[0].attributes.find(({ graphs }) => graphs?.length)
    .graphs[0].path;
  load(model.graph(false, path));
  const state = (data) =>
    window.dispatchEvent(new window.MessageEvent("message", { data }));

  document.getElementById("open-model").click();
  submitOpen(document, window);
  const replacement = messages.at(-1);
  document.querySelector("#graph-navigation button").click();
  assert.equal(messages.at(-1).type, "navigate");
  load(root);
  assert.equal(document.getElementById("load-progress").hidden, false);
  assert.equal(
    document.getElementById("model-view").getAttribute("aria-busy"),
    "true",
  );
  assert.equal(document.getElementById("open-model").disabled, true);

  state({
    type: "log-entry",
    entry: {
      id: replacement.actionId,
      action: "Open model",
      timestamp: new Date().toISOString(),
      status: "SUCCEEDED",
    },
  });
  assert.equal(document.getElementById("load-progress").hidden, true);
  assert.equal(document.getElementById("open-model").disabled, false);
});

test("switches all tabs without logging clicks or changing graph, selection or camera", async (t) => {
  const { document, load, window, events } = openViewer(t);
  load(demoGraph);
  document.getElementById("zoom-in").click();
  const graph = document.getElementById("nodes").innerHTML;
  const inspector = document.querySelector("aside").textContent;
  const camera = document.getElementById("graph").getAttribute("viewBox");
  document.getElementById("logs-tab").click();
  await Promise.resolve();
  assert.equal(document.getElementById("model-view").hidden, true);
  assert.equal(document.getElementById("logs-view").hidden, false);
  document.getElementById("logs-view").click();
  document.getElementById("status-tab").click();
  assert.equal(document.getElementById("status-view").hidden, false);
  assert.equal(document.getElementById("logs-view").hidden, true);
  document
    .getElementById("status-tab")
    .dispatchEvent(
      new window.KeyboardEvent("keydown", { key: "End", bubbles: true }),
    );
  assert.equal(document.activeElement.id, "settings-tab");
  assert.equal(document.getElementById("settings-view").hidden, false);
  assert.equal(document.getElementById("status-view").hidden, true);
  document.getElementById("controls-tab").click();
  assert.equal(document.getElementById("controls-view").hidden, false);
  assert.equal(document.getElementById("repositories-view").hidden, true);
  document.getElementById("logs-tab").click();
  document
    .getElementById("logs-tab")
    .dispatchEvent(
      new window.KeyboardEvent("keydown", { key: "ArrowLeft", bubbles: true }),
    );
  await Promise.resolve();
  assert.equal(document.activeElement.id, "model-tab");
  assert.equal(document.getElementById("logs-view").hidden, true);
  assert.equal(document.getElementById("log-entries").children.length, 0);
  assert.equal(events.filter(({ type }) => type === "interaction").length, 0);
  assert.equal(document.getElementById("nodes").innerHTML, graph);
  assert.equal(document.querySelector("aside").textContent, inspector);
  assert.equal(
    document.getElementById("graph").getAttribute("viewBox"),
    camera,
  );
});

test("Settings edits saved paths and existing sidebar preferences and renders package versions safely", (t) => {
  const { document, window, messages } = openViewer(t);
  const send = (data) =>
    window.dispatchEvent(new window.MessageEvent("message", { data }));
  const byId = (id) => document.getElementById(id);
  byId("settings-tab").click();
  assert.equal(messages.at(-1).type, "get-settings");
  send({
    type: "settings-info",
    application: {
      downloadDirectory: "/downloads",
      pythonPath: "/usr/bin/python3",
    },
    viewer: {
      width: 420,
      panels: { operator: false, find: true, metadata: false },
    },
    appVersion: "1.2.3",
    nativeVersion: "0.2.0",
    platform: "linux / x64",
    trusted: true,
  });
  assert.equal(byId("settings-app-version").textContent, "1.2.3");
  assert.equal(byId("settings-native-version").textContent, "0.2.0");
  assert.equal(byId("settings-width").value, "420");
  assert.equal(byId("operator-panel").hidden, true);
  assert.equal(byId("show-operator").getAttribute("aria-expanded"), "false");
  assert.equal(
    document
      .querySelector(".graph-body")
      .style.getPropertyValue("--sidebar-width"),
    "420px",
  );
  byId("settings-download").value = "/new-downloads";
  byId("application-settings-form").dispatchEvent(
    new window.Event("submit", { cancelable: true }),
  );
  assert.deepEqual(JSON.parse(JSON.stringify(messages.at(-1))), {
    type: "application-settings",
    settings: {
      downloadDirectory: "/new-downloads",
      pythonPath: "/usr/bin/python3",
    },
  });
  byId("settings-operator").click();
  assert.equal(byId("operator-panel").hidden, false);
  assert.equal(messages.at(-1).type, "viewer-settings");
  byId("settings-width").value = "";
  byId("settings-width").dispatchEvent(new window.Event("change"));
  assert.equal(
    document
      .querySelector(".graph-body")
      .style.getPropertyValue("--sidebar-width"),
    "",
  );
  byId("show-find").click();
  assert.equal(byId("settings-find").checked, false);
  byId("settings-inspect-python").click();
  assert.equal(messages.at(-1).type, "inspect-python");
  const inspection = messages.at(-1);
  assert.equal(inspection.pythonPath, "/usr/bin/python3");
  assert.equal(byId("settings-inspect-python").disabled, true);
  send({
    ...inspection,
    type: "python-info",
    python: {
      executable: "/usr/bin/python3",
      version: "3.12.3",
      packages: [
        {
          name: "transformers",
          version: null,
          source: "/sources/transformers",
        },
        { name: "torch", version: "<img src=x onerror=alert(1)>" },
        { name: "onnx", version: null },
      ],
    },
  });
  assert.equal(byId("settings-inspect-python").disabled, false);
  assert.match(byId("settings-packages").textContent, /Not installed/);
  assert.equal(byId("settings-packages").querySelector("img"), null);
  assert.deepEqual(
    [...byId("settings-packages").querySelectorAll("dt")].map(
      (name) => name.textContent,
    ),
    ["onnx", "torch", "transformers"],
  );
  assert.match(
    byId("settings-packages").textContent,
    /Version unavailable \(PYTHONPATH\)/,
  );
  assert.equal(
    byId("settings-packages").querySelector("dd:last-child").title,
    "/sources/transformers",
  );
  byId("settings-inspect-python").click();
  send({
    ...messages.at(-1),
    type: "settings-error",
    error: "Python not found",
  });
  assert.equal(byId("settings-feedback").textContent, "Python not found");
  assert.equal(byId("settings-inspect-python").disabled, false);
  assert.equal(byId("settings-packages").children.length, 0);
  const info = {
    type: "settings-info",
    application: {
      downloadDirectory: "/downloads",
      pythonPath: "/other/python",
    },
    viewer: { panels: { operator: true, find: true, metadata: false } },
    trusted: true,
  };
  byId("settings-inspect-python").click();
  const staleInspection = messages.at(-1);
  send(info);
  send({
    ...staleInspection,
    type: "python-info",
    python: {
      executable: "/usr/bin/python3",
      version: "3.12.3",
      packages: [{ name: "onnx", version: "old" }],
    },
  });
  assert.equal(byId("settings-packages").children.length, 0);
  assert.equal(byId("settings-python-status").textContent, "Not checked");
  send({ ...info, trusted: false });
  assert.equal(byId("settings-inspect-python").disabled, true);
  send({ type: "settings-error", error: "Invalid path" });
  assert.equal(byId("settings-inspect-python").disabled, true);
});

test("Settings applies panel toggles while retaining the last valid sidebar width", (t) => {
  const { document, window, messages } = openViewer(t);
  const byId = (id) => document.getElementById(id);
  window.dispatchEvent(
    new window.MessageEvent("message", {
      data: {
        type: "viewer-settings",
        settings: {
          width: 420,
          panels: { operator: true, find: true, metadata: false },
        },
      },
    }),
  );
  for (const invalid of ["199", "420.5"]) {
    byId("settings-width").value = invalid;
    const before = messages.length;
    byId("settings-width").dispatchEvent(new window.Event("change"));
    assert.equal(messages.length, before);
    const wasHidden = byId("operator-panel").hidden;
    byId("settings-operator").click();
    assert.equal(byId("operator-panel").hidden, !wasHidden);
    assert.equal(messages.at(-1).settings.width, 420);
    assert.equal(messages.at(-1).settings.panels.operator, wasHidden);
  }
  byId("settings-width").value = "";
  byId("settings-width").dispatchEvent(new window.Event("change"));
  byId("settings-width").value = "199";
  byId("settings-find").click();
  assert.equal(byId("find-panel").hidden, true);
  assert.equal("width" in messages.at(-1).settings, false);
  assert.equal(
    document
      .querySelector(".graph-body")
      .style.getPropertyValue("--sidebar-width"),
    "",
  );
});

test("Settings keeps inspections pending after save errors and rejects unrelated responses", (t) => {
  const { document, window, messages } = openViewer(t);
  const send = (data) =>
    window.dispatchEvent(new window.MessageEvent("message", { data }));
  const byId = (id) => document.getElementById(id);
  const info = {
    type: "settings-info",
    application: {
      downloadDirectory: "/downloads",
      pythonPath: "/first/python",
    },
    viewer: { panels: { operator: true, find: true, metadata: false } },
    trusted: true,
  };
  send(info);
  byId("settings-inspect-python").click();
  const first = messages.at(-1);
  send({ type: "settings-error", error: "Invalid download directory" });
  assert.equal(
    byId("settings-feedback").textContent,
    "Invalid download directory",
  );
  assert.equal(byId("settings-inspect-python").disabled, true);
  assert.equal(
    byId("settings-python-status").textContent,
    "Checking Python packages…",
  );
  send({
    ...info,
    application: { ...info.application, pythonPath: "/second/python" },
  });
  byId("settings-inspect-python").click();
  assert.equal(messages.at(-1), first);
  const python = {
    executable: "/first/python",
    version: "3.12",
    packages: [{ name: "onnx", version: "old" }],
  };
  send({ ...first, type: "python-info", python });
  assert.equal(byId("settings-packages").children.length, 0);
  assert.equal(byId("settings-inspect-python").disabled, false);
  byId("settings-inspect-python").click();
  const second = messages.at(-1);
  assert.notEqual(second.requestId, first.requestId);
  assert.equal(second.pythonPath, "/second/python");
  send({ ...first, type: "settings-error", error: "Old inspection failed" });
  send({ ...first, type: "python-info", python });
  send({
    ...second,
    pythonPath: first.pythonPath,
    type: "python-info",
    python,
  });
  assert.equal(byId("settings-inspect-python").disabled, true);
  assert.equal(byId("settings-packages").children.length, 0);
  send({
    ...second,
    type: "python-info",
    python: { ...python, executable: "/second/python", packages: [] },
  });
  assert.equal(byId("settings-inspect-python").disabled, false);
  assert.match(byId("settings-python-status").textContent, /\/second\/python/);
});

for (const initialTab of ["model", "logs"]) {
  test(`opens ONNX without a model from ${initialTab} with exporter cards and package specializations`, (t) => {
    const { document, window, messages } = openViewer(t, initialTab);
    const tab = document.getElementById("repositories-tab");
    const panel = document.getElementById("repositories-view");
    assert.equal(tab.textContent, "ONNX");
    assert.equal(panel.hidden, true);
    assert.equal(tab.getAttribute("aria-selected"), "false");
    assert.equal(tab.tabIndex, -1);
    assert.equal(tab.getAttribute("role"), "tab");
    assert.equal(tab.getAttribute("aria-controls"), panel.id);
    assert.equal(panel.getAttribute("role"), "tabpanel");
    assert.equal(panel.getAttribute("aria-labelledby"), tab.id);
    assert.equal(panel.tabIndex, 0);
    tab.click();
    assert.equal(panel.hidden, false);
    assert.equal(tab.getAttribute("aria-selected"), "true");
    assert.equal(tab.tabIndex, 0);
    for (const name of ["model", "logs", "status", "controls"]) {
      assert.equal(document.getElementById(`${name}-view`).hidden, true);
      assert.equal(
        document.getElementById(`${name}-tab`).getAttribute("aria-selected"),
        "false",
      );
      assert.equal(document.getElementById(`${name}-tab`).tabIndex, -1);
    }
    const repositories = [
      [
        "ONNX",
        "onnx/onnx",
        "repositories/onnx.png",
        ["onnx"],
        "https://onnx.ai/onnx/",
      ],
      [
        "ONNX Runtime",
        "microsoft/onnxruntime",
        "repositories/onnxruntime.png",
        ["onnxruntime"],
        "https://onnxruntime.ai/docs/",
      ],
      [
        "ONNX Runtime Extensions",
        "microsoft/onnxruntime-extensions",
        "repositories/onnxruntime.png",
        ["onnxruntime", "onnxruntime-extensions"],
        "https://onnxruntime.ai/docs/extensions/",
      ],
      [
        "ONNX Runtime GenAI",
        "microsoft/onnxruntime-genai",
        "repositories/onnxruntime.png",
        ["onnxruntime-genai"],
        "https://onnxruntime.ai/docs/genai/",
      ],
      [
        "onnx-light",
        "xadupre/onnx-light",
        "repositories/onnx-light.svg",
        ["onnx-light"],
        "https://sdpython.github.io/doc/onnx-light/dev/index.html",
      ],
      [
        "onnx-light-cpu",
        "xadupre/onnx-light-cpu",
        "repositories/onnx-light-cpu.svg",
        ["onnx-light"],
        "https://xadupre.github.io/docs/onnx-light-cpu/",
      ],
      [
        "onnx-light-ort",
        "xadupre/onnx-light-ort",
        "repositories/onnx-light-ort.svg",
        ["onnx-light", "onnxruntime"],
        "https://xadupre.github.io/onnx-light-ort/",
      ],
      [
        "onnx-light-viewer",
        "xadupre/onnx-light-viewer",
        "logo.svg",
        ["VS Code", "onnx-light"],
        "https://xadupre.github.io/docs/onnx-light-viewer/",
      ],
      [
        "Mobius",
        "onnxruntime/mobius",
        "repositories/onnxruntime.png",
        ["Transformers", "Diffusers", "ONNX Runtime GenAI"],
        "https://onnxruntime.github.io/mobius/",
      ],
      [
        "modelbuilder",
        "microsoft/onnxruntime-genai/tree/main/src/python/py/models",
        "repositories/onnxruntime.png",
        ["Transformers", "onnxruntime-genai"],
        "https://onnxruntime.ai/docs/genai/howto/build-model.html",
      ],
      [
        "mbext",
        "xadupre/mbext",
        "repositories/mbext.svg",
        ["Transformers", "onnxruntime-genai", "modelbuilder", "onnx-light"],
        "https://xadupre.github.io/docs/mbext/index.html",
      ],
      [
        "yet-another-onnx-builder",
        "xadupre/yet-another-onnx-builder",
        "repositories/yet-another-onnx-builder.svg",
        [
          "PyTorch",
          "TensorFlow",
          "Keras",
          "LiteRT",
          "scikit-learn",
          "scikit-survival",
          "category_encoders",
          "imbalanced-learn",
          "statsmodels",
          "XGBoost",
          "LightGBM",
          "NumPy",
          "SQL",
          "pandas",
          "Polars",
          "JAX",
          "yobx",
          "onnx-light",
        ],
        "https://xadupre.github.io/docs/yet-another-onnx-builder/index.html",
      ],
      [
        "sklearn-onnx",
        "onnx/sklearn-onnx",
        "repositories/sklearn-onnx.png",
        ["scikit-learn"],
        "https://onnx.ai/sklearn-onnx/",
      ],
      [
        "tensorflow-onnx",
        "onnx/tensorflow-onnx",
        "repositories/onnx.png",
        ["TensorFlow", "Keras", "TFLite", "TensorFlow.js"],
        "https://github.com/onnx/tensorflow-onnx/blob/main/README.md",
      ],
      [
        "onnxmltools",
        "onnx/onnxmltools",
        "repositories/onnxmltools.png",
        ["XGBoost", "LightGBM", "CatBoost", "Core ML"],
        "https://onnx.ai/onnxmltools/",
      ],
    ];
    assert.deepEqual(
      [...panel.querySelectorAll("h3")].map((heading) => heading.textContent),
      ["Official projects", "onnx-light ecosystem", "Exporters"],
    );
    assert.deepEqual(
      [
        ...panel
          .querySelectorAll(".repository-list")[2]
          .querySelectorAll(".repository-heading > a:first-child"),
      ].map((link) => link.textContent),
      repositories.slice(8).map(([name]) => name),
    );
    const cards = [...panel.querySelectorAll(".repository-list li")];
    assert.equal(cards.length, repositories.length);
    cards.forEach((card, index) => {
      const [name, repository, logo, packages, documentationUrl] =
        repositories[index];
      const link = card.querySelector("a");
      assert.equal(link.textContent, name);
      assert.equal(link.href, `https://github.com/${repository}`);
      const documentationLink = card.querySelector(".repository-documentation");
      assert.ok(documentationLink);
      assert.equal(card.querySelectorAll("a").length, 2);
      assert.equal(documentationLink.href, documentationUrl);
      assert.equal(
        documentationLink.getAttribute("aria-label"),
        `${name} documentation`,
      );
      assert.equal(documentationLink.title, `${name} documentation`);
      assert.equal(
        documentationLink.querySelector("svg").getAttribute("aria-hidden"),
        "true",
      );
      assert.ok(documentationLink.querySelector("svg path"));
      documentationLink.focus();
      assert.equal(document.activeElement, documentationLink);
      assert.ok(card.querySelector("p").textContent.trim().length > 20);
      if (name === "mbext") {
        assert.match(
          card.querySelector("p").textContent,
          /rewrite.*onnx-light.*faster/i,
        );
      }
      const specialization = card.querySelector(".repository-specialization");
      assert.ok(specialization);
      assert.equal(
        specialization.querySelector("strong").textContent,
        "Specialized for:",
      );
      for (const packageName of packages) {
        assert.ok(
          specialization.textContent.includes(packageName),
          `${name} specialization should identify ${packageName}`,
        );
      }
      const icon = link.querySelector("svg, img");
      assert.ok(icon);
      assert.equal(icon.getAttribute("width"), "28");
      assert.equal(icon.getAttribute("height"), "28");
      if (repository === "xadupre/onnx-light-viewer") {
        assert.equal(icon.alt, "");
        assert.equal(icon.src, "https://viewer.invalid/logo.svg");
      } else {
        assert.equal(icon.localName, "svg");
        assert.equal(icon.getAttribute("aria-hidden"), "true");
        assert.ok(icon.querySelector("path"));
      }
      const image = card.querySelector(":scope > .repository-logo");
      assert.ok(image);
      assert.equal(image.alt, "");
      assert.equal(image.src, `https://viewer.invalid/${logo}`);
      assert.equal(image.getAttribute("width"), "96");
      assert.equal(image.getAttribute("height"), "64");
      const asset = readFileSync(require.resolve(`../media/${logo}`));
      if (logo.endsWith(".svg")) {
        const svg = new JSDOM(asset.toString(), {
          contentType: "image/svg+xml",
        });
        assert.equal(svg.window.document.documentElement.localName, "svg");
        assert.equal(
          svg.window.document.querySelector("script, image, foreignObject"),
          null,
        );
        svg.window.close();
      } else {
        assert.deepEqual(
          [...asset.subarray(0, 8)],
          [137, 80, 78, 71, 13, 10, 26, 10],
        );
      }
    });
    tab.focus();
    for (const [key, expected] of [
      ["ArrowRight", "settings"],
      ["ArrowRight", "model"],
      ["ArrowLeft", "settings"],
      ["ArrowLeft", "repositories"],
      ["ArrowLeft", "controls"],
      ["ArrowRight", "repositories"],
      ["Home", "model"],
      ["End", "settings"],
    ]) {
      document.activeElement.dispatchEvent(
        new window.KeyboardEvent("keydown", { key, bubbles: true }),
      );
      assert.equal(document.activeElement.id, `${expected}-tab`);
      assert.equal(document.getElementById(`${expected}-view`).hidden, false);
    }
    assert.deepEqual(
      messages.map(({ type }) => type),
      ["ready", "get-settings", "get-settings", "get-settings"],
    );
  });

  test(`opens Controls without a model from ${initialTab} and supports keyboard tab navigation`, (t) => {
    const { document, window, messages } = openViewer(t, initialTab);
    const tab = document.getElementById("controls-tab");
    const panel = document.getElementById("controls-view");
    assert.equal(panel.hidden, true);
    assert.equal(tab.getAttribute("aria-selected"), "false");
    assert.equal(tab.tabIndex, -1);
    assert.equal(tab.getAttribute("role"), "tab");
    assert.equal(tab.getAttribute("aria-controls"), panel.id);
    assert.equal(panel.getAttribute("role"), "tabpanel");
    assert.equal(panel.getAttribute("aria-labelledby"), tab.id);
    assert.equal(panel.tabIndex, 0);
    tab.click();
    assert.equal(panel.hidden, false);
    assert.equal(tab.getAttribute("aria-selected"), "true");
    assert.equal(tab.tabIndex, 0);
    for (const name of ["model", "logs", "status", "repositories"]) {
      assert.equal(document.getElementById(`${name}-view`).hidden, true);
      assert.equal(
        document.getElementById(`${name}-tab`).getAttribute("aria-selected"),
        "false",
      );
      assert.equal(document.getElementById(`${name}-tab`).tabIndex, -1);
    }
    assert.deepEqual(
      [...panel.querySelectorAll("h3")].map((heading) => heading.textContent),
      [
        "Mouse and trackpad",
        "Graph keyboard shortcuts",
        "Navigation and buttons",
      ],
    );
    const tables = panel.querySelectorAll("table");
    assert.equal(tables.length, 3);
    for (const [index, table] of [...tables].entries()) {
      assert.equal(
        document.getElementById(table.getAttribute("aria-labelledby")),
        panel.querySelectorAll("h3")[index],
      );
      assert.deepEqual(
        [...table.querySelectorAll("thead th")].map((cell) => [
          cell.textContent,
          cell.getAttribute("scope"),
        ]),
        [
          ["Control", "col"],
          ["Action", "col"],
        ],
      );
      for (const row of table.querySelectorAll("tbody tr")) {
        assert.equal(row.cells.length, 2);
        assert.equal(row.cells[0].tagName, "TH");
        assert.equal(row.cells[0].getAttribute("scope"), "row");
        assert.equal(row.cells[1].tagName, "TD");
        assert.ok(row.cells[0].textContent.trim());
        assert.ok(row.cells[1].textContent.trim());
      }
    }
    for (const control of [
      "Ctrl + scroll",
      "Alt + drag",
      "Fit the graph",
      "Jump to readable size",
      "Shift + scroll",
      "Ctrl/Cmd + A",
      "Enter or Space",
      "Escape",
      "Drag the background",
      "Home / End",
      "Ctrl/Cmd + Home",
      "Ctrl/Cmd + End",
      "Top / Bottom",
    ])
      assert.ok(panel.textContent.includes(control), control);
    tab.focus();
    for (const [key, expected] of [
      ["ArrowRight", "repositories"],
      ["ArrowRight", "settings"],
      ["ArrowRight", "model"],
      ["ArrowLeft", "settings"],
      ["ArrowLeft", "repositories"],
      ["ArrowLeft", "controls"],
      ["ArrowLeft", "status"],
      ["ArrowRight", "controls"],
      ["Home", "model"],
      ["End", "settings"],
    ]) {
      document.activeElement.dispatchEvent(
        new window.KeyboardEvent("keydown", { key, bubbles: true }),
      );
      assert.equal(document.activeElement.id, `${expected}-tab`);
      assert.equal(document.getElementById(`${expected}-view`).hidden, false);
    }
    assert.deepEqual(
      messages.map(({ type }) => type),
      ["ready", "get-settings", "get-settings", "get-settings"],
    );
  });
}

test("shows the compiled onnx-light release before loading a graph and after model replacement", (t) => {
  const { document, load } = openViewer(t);
  const version = readFileSync(
    require.resolve("../CMakeLists.txt"),
    "utf8",
  ).match(/set\(ONNX_LIGHT_VERSION\s+"([^"]+)"\)/)[1];
  const release = document.getElementById("onnx-light-version");
  assert.equal(release.textContent, `onnx-light ${version}`);
  load(demoGraph);
  load(
    decodeGraph(
      readFileSync(require.resolve("./models/passthrough.onnx")),
      "passthrough.onnx",
    ),
  );
  assert.equal(release.textContent, `onnx-light ${version}`);
});

test("requests model opening without a separate Hugging Face button and clears busy state after cancellation or failure", (t) => {
  const { document, window, messages, load } = openViewer(t);
  load(demoGraph);
  assert.equal(document.getElementById("open-huggingface"), null);
  const button = document.getElementById("open-model");
  button.click();
  submitOpen(document, window);
  const request = messages.at(-1);
  assert.equal(request.type, "open");
  assert.equal(button.disabled, true);
  document.getElementById("logs-tab").click();
  assert.equal(document.getElementById("logs-tab").textContent, "Logs");
  assert.equal(document.getElementById("logs-view").hidden, false);
  window.dispatchEvent(
    new window.MessageEvent("message", {
      data: {
        type: "log-entry",
        entry: {
          id: request.actionId,
          timestamp: new Date().toISOString(),
          action: "Open ONNX model",
          status: "CANCELLED",
        },
      },
    }),
  );
  assert.equal(button.disabled, false);
  assert.equal(document.getElementById("logs-tab").textContent, "Logs");
  document.getElementById("model-tab").click();
  button.click();
  submitOpen(document, window);
  window.dispatchEvent(
    new window.MessageEvent("message", {
      data: { type: "load-error", error: "HTTP 403" },
    }),
  );
  assert.equal(button.disabled, false);
  assert.match(document.getElementById("load-status").textContent, /HTTP 403/);
  assert.equal(document.querySelector(".graph-node text").textContent, "Relu");
});
test("updates one action row, safely renders errors and retains history across model replacement", async (t) => {
  const { document, window, messages, load } = openViewer(t);
  load(demoGraph);
  document.getElementById("refresh-shapes").click();
  const actionId = messages.at(-1).actionId;
  const row = document.querySelector(".log-entry");
  assert.equal(row.dataset.status, "RUNNING");
  const entry = {
    id: actionId,
    timestamp: new Date().toISOString(),
    action: "Refresh shapes",
    status: "FAILED",
    details: 'VerifyModel: <img src=x onerror="throw 1">',
  };
  window.dispatchEvent(
    new window.MessageEvent("message", {
      data: { type: "log-entry", entry },
    }),
  );
  assert.equal(row.dataset.status, "FAILED");
  assert.equal(row.querySelector("pre").textContent, entry.details);
  assert.equal(row.querySelector("img"), null);
  window.dispatchEvent(
    new window.MessageEvent("message", {
      data: { type: "log-history", entries: [entry] },
    }),
  );
  load(demoGraph);
  assert.equal(document.querySelectorAll(".log-entry").length, 1);
  assert.equal(document.querySelector(".log-entry"), row);
  await Promise.resolve();
  assert.equal(row.dataset.status, "FAILED");
});

for (const initialTab of ["model", "logs"]) {
  test(`opens logs at the latest event with ${initialTab} initially selected`, (t) => {
    const { document, window } = openViewer(t, initialTab);
    const panel = document.getElementById("logs-view");
    Object.defineProperties(panel, {
      clientHeight: { get: () => (panel.hidden ? 0 : 100) },
      scrollHeight: {
        get: () =>
          panel.hidden
            ? 0
            : 160 +
              document.getElementById("log-entries").children.length * 100,
      },
    });
    const entries = Array.from({ length: 20 }, (_, i) => ({
      id: `host:${i}`,
      timestamp: "2026-10-22T20:00:00.000Z",
      action: `Action ${i}`,
      status: "SUCCEEDED",
    }));
    const receive = (data) =>
      window.dispatchEvent(new window.MessageEvent("message", { data }));
    receive({ type: "log-history", entries });
    assert.equal(panel.scrollTop, panel.hidden ? 0 : panel.scrollHeight);

    document.getElementById("logs-tab").click();
    assert.equal(panel.scrollTop, panel.scrollHeight);

    panel.scrollTop = 0;
    receive({
      type: "log-entry",
      entry: { ...entries[0], id: "host:live" },
    });
    assert.equal(panel.scrollTop, 0);

    document.getElementById("model-tab").click();
    document.getElementById("model-tab").dispatchEvent(
      new window.KeyboardEvent("keydown", {
        key: "ArrowRight",
        bubbles: true,
      }),
    );
    assert.equal(panel.hidden, false);
    assert.equal(panel.scrollTop, panel.scrollHeight);

    panel.scrollTop = panel.scrollHeight - panel.clientHeight;
    receive({
      type: "log-entry",
      entry: { ...entries[0], id: "host:follow" },
    });
    assert.equal(panel.scrollTop, panel.scrollHeight);

    panel.scrollTop = 0;
    document.getElementById("status-tab").click();
    receive({ type: "show-logs" });
    assert.equal(panel.hidden, false);
    assert.equal(panel.scrollTop, panel.scrollHeight);
    assert.equal(document.activeElement.id, "logs-tab");
  });
}

test("opens graph attributes and nested breadcrumbs, including empty graphs and failed navigation", (t) => {
  const { document, window, messages, load } = openViewer(t);
  const model = loadModel(
    readFileSync(require.resolve("./models/subgraphs.onnx")),
    "subgraphs.onnx",
  );
  const root = model.graph();
  load(root);
  assert.equal(document.getElementById("graph-navigation").hidden, false);
  assert.equal(document.querySelectorAll("#graph-navigation button").length, 1);
  assert.equal(
    document.querySelector("#graph-navigation [aria-current]").textContent,
    root.name,
  );
  const buttons = document.querySelectorAll("#node-attributes button");
  assert.equal(buttons.length, 2);
  assert.match(
    document.getElementById("captured-inputs").textContent,
    /x: FLOAT \[N, 3\]/,
  );
  buttons[1].click();
  const request = messages.at(-1);
  assert.equal(request.type, "navigate");
  assert.equal(
    document.getElementById("graph").getAttribute("aria-busy"),
    "true",
  );
  load(model.graph(false, request.path));
  assert.equal(document.getElementById("graph-navigation").hidden, false);
  assert.equal(document.querySelectorAll("#graph-navigation button").length, 2);
  assert.equal(
    document.querySelector("#graph-navigation [aria-current]").disabled,
    true,
  );
  assert.equal(document.querySelectorAll(".graph-node").length, 2);
  document
    .querySelector('[data-id="node-1"]')
    .dispatchEvent(
      new window.KeyboardEvent("keydown", { key: "Enter", bubbles: true }),
    );
  document.querySelector("#node-attributes button").click();
  load(model.graph(false, messages.at(-1).path));
  assert.equal(document.querySelectorAll("#graph-navigation button").length, 3);
  assert.equal(document.querySelectorAll(".graph-node").length, 1);
  document.querySelector("#graph-navigation button").click();
  assert.deepEqual(Array.from(messages.at(-1).path), []);
  window.dispatchEvent(
    new window.MessageEvent("message", {
      data: { type: "navigation-error", error: "invalid path" },
    }),
  );
  assert.match(
    document.getElementById("navigation-status").textContent,
    /invalid path/,
  );
  assert.equal(document.querySelectorAll(".graph-node").length, 1);
  load(root);
  document
    .querySelector('[data-id="node-1"]')
    .dispatchEvent(
      new window.KeyboardEvent("keydown", { key: "Enter", bubbles: true }),
    );
  assert.equal(document.querySelectorAll("#node-attributes button").length, 2);
  document.querySelector("#node-attributes button").click();
  load(model.graph(false, messages.at(-1).path));
  assert.equal(document.querySelectorAll(".graph-node").length, 0);
  assert.equal(document.getElementById("node-title").textContent, "Graph");
  load(demoGraph);
  assert.equal(document.getElementById("graph-navigation").hidden, false);
  assert.equal(document.querySelectorAll("#graph-navigation button").length, 1);
  assert.equal(
    document.querySelector("#graph-navigation [aria-current]").textContent,
    demoGraph.name,
  );
  assert.equal(document.querySelectorAll("#node-attributes button").length, 0);
  assert.equal(document.getElementById("navigation-status").textContent, "");
});

test("opens every subgraph from its operator rectangle with pointer and keyboard controls", (t) => {
  const { document, window, messages, load, pointer } = openViewer(t);
  const model = loadModel(
    readFileSync(require.resolve("./models/subgraphs.onnx")),
    "subgraphs.onnx",
  );
  const root = model.graph();
  for (const [nodeIndex, linkIndex, key] of [
    [0, 0, null],
    [0, 1, "Enter"],
    [1, 0, " "],
    [1, 1, "Enter"],
  ]) {
    load(root);
    const group = document.querySelector(`[data-id="node-${nodeIndex}"]`);
    const link = group.querySelectorAll(".subgraph-link")[linkIndex];
    assert.equal(group.getAttribute("role"), "group");
    assert.equal(link.getAttribute("role"), "button");
    assert.equal(link.getAttribute("tabindex"), "0");
    const body = group.querySelector("rect");
    const row = link.querySelector("rect");
    assert.ok(
      Number(row.getAttribute("y")) >=
        Number(group.querySelector(".node-tensor-info").getAttribute("y")) + 8,
    );
    assert.ok(
      Number(row.getAttribute("y")) + Number(row.getAttribute("height")) <=
        Number(body.getAttribute("y")) + Number(body.getAttribute("height")),
    );
    const count = messages.length;
    const positions = nodePositions(document);
    if (key) {
      link.dispatchEvent(
        new window.KeyboardEvent("keydown", { key, bubbles: true }),
      );
    } else {
      const x = Number(row.getAttribute("x")) + 10;
      const y = Number(row.getAttribute("y")) + 10;
      pointer(link.querySelector("text"), "pointerdown", x, y);
      pointer(document.getElementById("graph"), "pointermove", x + 10, y + 10);
      pointer(document.getElementById("graph"), "pointerup", x + 10, y + 10);
      assert.deepEqual(nodePositions(document), positions);
      assert.equal(messages.length, count);
      link
        .querySelector("text")
        .dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
    }
    assert.equal(messages.length, count + 1);
    assert.equal(messages.at(-1).type, "navigate");
    const expected = root.nodes[nodeIndex].attributes.flatMap(
      (attribute) => attribute.graphs ?? [],
    )[linkIndex];
    assert.deepEqual(
      JSON.parse(JSON.stringify(messages.at(-1).path)),
      expected.path,
    );
    load(model.graph(false, messages.at(-1).path));
    assert.equal(document.getElementById("graph-navigation").hidden, false);
  }
  load(root);
  document.getElementById("show-metadata").click();
  document
    .querySelector(".subgraph-link")
    .dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
  assert.equal(messages.at(-1).type, "navigate");
  load(demoGraph);
  assert.equal(document.querySelectorAll(".subgraph-link").length, 0);
});

test("sizes subgraph operators for their links and preserves dragging, selection and connection anchors", (t) => {
  const { document, window, messages, load, pointer } = openViewer(t);
  const model = loadModel(
    readFileSync(require.resolve("./models/subgraphs.onnx")),
    "subgraphs.onnx",
  );
  load(model.graph());
  const graph = document.getElementById("graph");
  const group = document.querySelector('[data-id="node-0"]');
  const body = group.querySelector("rect");
  const next = document.querySelector('[data-id="node-1"] > rect');
  const top = Number(body.getAttribute("y"));
  const bottom = top + Number(body.getAttribute("height"));
  assert.ok(Number(next.getAttribute("y")) > bottom);
  assert.ok(
    document.querySelector(".edge").getAttribute("d").endsWith(`,${top}`),
  );
  const before = nodePositions(document);
  pointer(
    group.querySelector(".operator-label"),
    "pointerdown",
    before[0].x,
    top + 41,
  );
  pointer(graph, "pointermove", before[0].x + 30, top + 61);
  pointer(graph, "pointerup", before[0].x + 30, top + 61);
  assertMoved(before, nodePositions(document), [0], 30, 20);
  assert.ok(
    document
      .querySelector(".edge")
      .getAttribute("d")
      .endsWith(`,${top + 20}`),
  );
  assert.equal(messages.filter(({ type }) => type === "navigate").length, 0);
  const left = Number(body.getAttribute("x")) + 30;
  pointer(graph, "pointerdown", left - 4, bottom + 16);
  pointer(graph, "pointermove", left + 4, bottom + 24);
  pointer(graph, "pointerup", left + 4, bottom + 24);
  assert.equal(group.classList.contains("selected"), true);
  group.dispatchEvent(
    new window.MouseEvent("click", { ctrlKey: true, bubbles: true }),
  );
  assert.equal(group.classList.contains("selected"), false);
});

for (const [kind, panelOpen] of [
  ["shapes", false],
  ["metadata", false],
  ["metadata", true],
]) {
  test(`keeps ${kind} button colors in sync with the latest refresh result with the panel ${panelOpen ? "open" : "closed"}`, (t) => {
    const { document, window, load } = openViewer(t);
    const stylesheet = document.createElement("style");
    stylesheet.textContent = readFileSync(
      require.resolve("../media/viewer.css"),
      "utf8",
    );
    document.head.append(stylesheet);
    const model = loadModel(
      readFileSync(require.resolve("./models/inference.onnx")),
      "inference.onnx",
    );
    load(model.graph());
    const refresh = document.getElementById(`refresh-${kind}`);
    const buttons = [refresh];
    const titles = buttons.map((button) => button.title);
    const idleColor = window.getComputedStyle(refresh).color;
    const colors = new Set([idleColor]);
    const status = document.getElementById(
      kind === "shapes" ? "shape-status" : "metadata-status",
    );
    const check = (expected) => {
      for (const [index, button] of buttons.entries()) {
        assert.equal(button.dataset.refreshStatus, expected);
        assert.equal(button.getAttribute("aria-describedby"), status.id);
        assert.equal(
          button.getAttribute("aria-busy"),
          String(expected === "pending"),
        );
        assert.equal(button.title, status.textContent || titles[index]);
        assert.equal(
          window.getComputedStyle(button).color,
          window.getComputedStyle(refresh).color,
        );
        if (expected !== "idle") {
          assert.equal(
            window.getComputedStyle(button).borderColor,
            window.getComputedStyle(refresh).borderColor,
          );
        }
      }
      assert.equal(refresh.disabled, expected === "pending");
      if (expected === "idle") {
        assert.equal(window.getComputedStyle(refresh).color, idleColor);
      } else {
        assert.notEqual(window.getComputedStyle(refresh).color, idleColor);
        colors.add(window.getComputedStyle(refresh).color);
      }
    };
    const receive = (data) =>
      window.dispatchEvent(new window.MessageEvent("message", { data }));
    const result =
      kind === "shapes"
        ? { type: "shape-info", tensors: model.graph(true).tensors }
        : { type: "metadata-info", metadata: model.metadata() };

    check("idle");
    if (panelOpen) document.getElementById("show-metadata").click();
    refresh.click();
    check("pending");
    receive(result);
    check("updated");
    document.getElementById("show-metadata").click();
    document.getElementById("show-metadata").click();
    check("updated");
    refresh.click();
    check("pending");
    receive(result);
    check("unchanged");
    refresh.click();
    check("pending");
    receive({
      type: kind === "shapes" ? "shape-error" : "metadata-error",
      error: "Refresh failed",
    });
    check("failed");
    refresh.click();
    check("pending");
    receive(result);
    check("unchanged");

    assert.equal(colors.size, 5);

    load(demoGraph);
    check(panelOpen ? "pending" : "idle");
    if (kind === "metadata") {
      if (!panelOpen) document.getElementById("show-metadata").click();
      check("pending");
      receive(result);
      check("updated");
    }
  });
}

test("refreshes whole-model metadata from the global toolbar with the inspector closed", (t) => {
  const { document, window, messages, load } = openViewer(t);
  const refresh = document.getElementById("refresh-metadata");
  assert.equal(refresh.disabled, true);
  const model = loadModel(
    readFileSync(require.resolve("./models/analysis.onnx")),
    "analysis.onnx",
  );
  load(model.graph());
  assert.ok(refresh.closest(".graph-toolbar"));
  assert.ok(
    document.getElementById("metadata-status").closest(".graph-toolbar"),
  );
  assert.equal(refresh.disabled, false);
  refresh.click();
  assert.equal(messages.at(-1).type, "metadata");
  assert.equal(refresh.disabled, true);
  assert.equal(document.getElementById("metadata-panel").hidden, true);
  assert.match(
    document.getElementById("metadata-status").textContent,
    /Analyzing/,
  );
  document.getElementById("canvas").getBoundingClientRect = () =>
    new window.DOMRect(0, 0, 1060, 300);
  const metadata = model.metadata();
  window.dispatchEvent(
    new window.MessageEvent("message", {
      data: { type: "metadata-info", metadata },
    }),
  );
  assert.equal(
    document.getElementById("metadata-status").textContent,
    `Metadata refreshed: ${metadata.model.length + metadata.graph.length + metadata.nodes.length + metadata.values.length} updates`,
  );
  assert.equal(document.getElementById("metadata-panel").hidden, true);
  assert.equal(refresh.disabled, false);
  const [, , width, height] = document
    .getElementById("graph")
    .getAttribute("viewBox")
    .split(" ")
    .map(Number);
  assert.ok(Math.abs(width / height - 1060 / 300) < 1e-9);
  const count = messages.filter(({ type }) => type === "metadata").length;
  document.getElementById("show-metadata").click();
  assert.equal(
    messages.filter(({ type }) => type === "metadata").length,
    count,
  );
  document
    .querySelector('[data-id="node-5"]')
    .dispatchEvent(
      new window.KeyboardEvent("keydown", { key: "Enter", bubbles: true }),
    );
  assert.equal(
    document.getElementById("metadata-releases").textContent,
    "shared;activationconstant_sum",
  );
  assert.equal(document.querySelectorAll(".edge.released").length, 4);
});

test("counts metadata changes and replaces cached analysis and graph markers on every refresh", (t) => {
  const { document, window, load } = openViewer(t);
  const errors = [];
  window.addEventListener("error", (event) => errors.push(event.error));
  const model = loadModel(
    readFileSync(require.resolve("./models/analysis.onnx")),
    "analysis.onnx",
  );
  load(model.graph());
  const send = (metadata) => {
    document.getElementById("refresh-metadata").click();
    window.dispatchEvent(
      new window.MessageEvent("message", {
        data: { type: "metadata-info", metadata },
      }),
    );
  };
  const metadata = model.metadata();
  send(metadata);
  const status = document.getElementById("metadata-status");
  send(structuredClone(metadata));
  assert.equal(status.textContent, "Metadata refreshed: 0 updates");
  assert.equal(status.title, status.textContent);
  const changed = structuredClone(metadata);
  changed.model.push({ key: "new", value: "updated model" });
  changed.graph.push({ key: "new", value: "updated graph" });
  changed.nodes[0].constant = !changed.nodes[0].constant;
  changed.nodes[0].tag = "updated tag";
  changed.values.find(
    (value) => value.name === model.graph().nodes[0].outputs[0],
  ).tag = "updated value";
  const unused = changed.nodes.find((node) => node.unusedOutputs.length);
  unused.unusedOutputs = [];
  send(changed);
  assert.equal(status.textContent, "Metadata refreshed: 5 updates");
  assert.equal(document.querySelectorAll(".unused-label").length, 1);
  document.getElementById("show-metadata").click();
  assert.match(
    document.getElementById("metadata-model").textContent,
    /updated model/,
  );
  assert.match(
    document.getElementById("metadata-graph").textContent,
    /updated graph/,
  );
  assert.match(
    document.getElementById("metadata-analysis").textContent,
    /updated tag/,
  );
  assert.match(
    document.getElementById("metadata-values").textContent,
    /updated value/,
  );
  send(metadata);
  assert.equal(status.textContent, "Metadata refreshed: 5 updates");
  assert.equal(document.querySelectorAll(".unused-label").length, 2);
  const oneChange = structuredClone(metadata);
  oneChange.nodes[0].tag = "one update";
  send(oneChange);
  assert.equal(status.textContent, "Metadata refreshed: 1 update");
  send(oneChange);
  assert.equal(status.textContent, "Metadata refreshed: 0 updates");
  load(model.graph());
  assert.equal(status.textContent, "Analyzing metadata…");
  send(metadata);
  assert.equal(
    status.textContent,
    `Metadata refreshed: ${metadata.model.length + metadata.graph.length + metadata.nodes.length + metadata.values.length} updates`,
  );
  assert.deepEqual(errors, []);
});

test("styles Data like the other panel toggles independently of metadata refresh status", (t) => {
  const { document, window, load } = openViewer(t);
  const style = document.createElement("style");
  style.textContent = readFileSync(
    require.resolve("../media/viewer.css"),
    "utf8",
  );
  document.head.append(style);
  const model = loadModel(
    readFileSync(require.resolve("./models/analysis.onnx")),
    "analysis.onnx",
  );
  load(model.graph());
  const toggle = document.getElementById("show-metadata");
  const title = toggle.title;
  const highlight = () => {
    const style = window.getComputedStyle(toggle);
    return {
      background: style.backgroundColor,
      shadow: style.boxShadow,
      weight: style.fontWeight,
      color: style.color,
    };
  };
  const inactive = highlight();
  assert.equal(toggle.querySelector(".toggle-indicator"), null);
  assert.equal(toggle.getAttribute("aria-expanded"), "false");
  toggle.click();
  const active = highlight();
  assert.notEqual(active.background, inactive.background);
  assert.notEqual(active.shadow, inactive.shadow);
  assert.equal(active.weight, "700");
  assert.equal(toggle.dataset.refreshStatus, undefined);
  assert.equal(toggle.getAttribute("aria-expanded"), "true");
  assert.equal(document.getElementById("metadata-panel").hidden, false);
  window.dispatchEvent(
    new window.MessageEvent("message", {
      data: { type: "metadata-info", metadata: model.metadata() },
    }),
  );
  assert.deepEqual(highlight(), active);
  assert.equal(toggle.title, title);
  load(model.graph());
  assert.deepEqual(highlight(), active);
  toggle.click();
  assert.equal(toggle.getAttribute("aria-expanded"), "false");
  assert.equal(document.getElementById("metadata-panel").hidden, true);
  const closed = highlight();
  assert.notEqual(closed.background, active.background);
  assert.equal(closed.shadow, inactive.shadow);
  assert.equal(closed.weight, inactive.weight);
  assert.equal(closed.color, inactive.color);
  toggle.click();
  assert.deepEqual(highlight(), active);
});

function assertMetadataTable(table, entries) {
  assert.equal(table.tagName, "TABLE");
  assert.ok(table.getAttribute("aria-label"));
  assert.deepEqual(
    Array.from(table.querySelectorAll("thead th"), (cell) => [
      cell.scope,
      cell.textContent,
    ]),
    [
      ["col", "Key"],
      ["col", "Value"],
    ],
  );
  assert.deepEqual(
    Array.from(table.tBodies[0].rows, (row) => {
      assert.equal(row.cells[0].tagName, "TH");
      assert.equal(row.cells[0].scope, "row");
      assert.equal(row.cells[1].tagName, "TD");
      return { key: row.cells[0].textContent, value: row.cells[1].textContent };
    }),
    entries,
  );
}

test("metadata tables preserve duplicate keys, empty and multiline values, and literal markup on refresh", (t) => {
  const { document, window, load } = openViewer(t);
  const model = loadModel(
    readFileSync(require.resolve("./models/analysis.onnx")),
    "analysis.onnx",
  );
  load(model.graph());
  document.getElementById("show-metadata").click();
  const metadata = model.metadata();
  metadata.model = [
    { key: "<img src=x onerror=alert(1)>", value: "<script>alert(1)</script>" },
    { key: "duplicate", value: "" },
    { key: "duplicate", value: "first line\nsecond line" },
    { key: "", value: "x".repeat(500) },
  ];
  const send = () =>
    window.dispatchEvent(
      new window.MessageEvent("message", {
        data: { type: "metadata-info", metadata },
      }),
    );
  send();
  const table = document.getElementById("metadata-model");
  assertMetadataTable(table, metadata.model);
  assert.equal(table.querySelectorAll("img, script").length, 0);
  metadata.model = [];
  send();
  assertMetadataTable(table, []);
  assert.equal(table.parentElement.hidden, true);
  metadata.model = [{ key: "updated", value: "new value" }];
  send();
  assertMetadataTable(table, metadata.model);
  assert.equal(table.parentElement.hidden, false);
});

test("renders PipelineProto fields as metadata table rows instead of a serialized dictionary", (t) => {
  const { document, window, load, pointer } = openViewer(t);
  const model = loadModel(
    readFileSync(require.resolve("./models/tiny.pipe-onnx")),
    "tiny.pipe-onnx",
  );
  load(model.graph());
  const svg = document.getElementById("graph");
  pointer(svg, "pointerdown", 0, 0);
  pointer(svg, "pointerup", 0, 0);
  document.getElementById("show-metadata").click();
  const metadata = model.metadata();
  window.dispatchEvent(
    new window.MessageEvent("message", {
      data: { type: "metadata-info", metadata },
    }),
  );
  const table = document.getElementById("metadata-model");
  assertMetadataTable(table, metadata.model);
  const rows = Object.fromEntries(
    [...table.tBodies[0].rows].map((row) => [
      row.cells[0].textContent,
      row.cells[1].textContent,
    ]),
  );
  assert.equal(rows.name, "Three-token cycle");
  assert.equal(rows.max_new_tokens, "4");
  assert.equal(rows.temperature, "0");
  assert.equal(rows.seed, "42");
  assert.equal(Object.hasOwn(rows, "PipelineProto"), false);
});

test("shows native metadata in the sidebar and marks released and never-used results without changing the layout", (t) => {
  const { document, window, messages, load, pointer } = openViewer(t);
  const model = loadModel(
    readFileSync(require.resolve("./models/analysis.onnx")),
    "analysis.onnx",
  );
  load(model.graph());
  const graph = document.getElementById("graph");
  const positions = nodePositions(document);
  const viewBox = graph.getAttribute("viewBox");
  const toggle = document.getElementById("show-metadata");
  toggle.click();
  assert.equal(messages.filter(({ type }) => type === "metadata").length, 1);
  assert.equal(toggle.getAttribute("aria-expanded"), "true");
  assert.equal(document.getElementById("metadata-panel").hidden, false);
  assert.equal(document.getElementById("node-details").hidden, false);
  assert.equal(document.getElementById("operator-panel").hidden, false);
  window.dispatchEvent(
    new window.MessageEvent("message", {
      data: { type: "metadata-info", metadata: model.metadata() },
    }),
  );
  assert.equal(
    document.getElementById("metadata-panel").getAttribute("aria-busy"),
    "false",
  );
  assert.equal(document.getElementById("metadata-content").hidden, false);
  assert.equal(document.getElementById("metadata-unused-section").hidden, true);
  assert.equal(
    document.getElementById("metadata-releases-section").hidden,
    true,
  );
  assert.equal(
    document.getElementById("metadata-node").parentElement.hidden,
    true,
  );
  assert.equal(
    document.getElementById("metadata-values").parentElement.hidden,
    false,
  );
  assert.doesNotMatch(
    document.querySelector("main").textContent,
    /Dashed, crossed-out connections are no longer needed/,
  );
  assert.match(
    document.getElementById("metadata-analysis").textContent,
    /ConstantYes/,
  );
  assert.match(
    document.getElementById("metadata-model").textContent,
    /metadata <inspection>/,
  );
  const metadata = model.metadata();
  assertMetadataTable(
    document.getElementById("metadata-model"),
    metadata.model,
  );
  assertMetadataTable(
    document.getElementById("metadata-graph"),
    metadata.graph,
  );
  assertMetadataTable(document.getElementById("metadata-analysis"), [
    { key: "Name", value: model.graph().nodes[0].name },
    { key: "Tag", value: metadata.nodes[0].tag },
    { key: "Constant", value: "Yes" },
  ]);
  assert.equal(document.querySelectorAll(".unused-label").length, 2);
  assert.match(
    document.querySelector('[data-id="node-3"] .unused-label title')
      .textContent,
    /dimensions/,
  );
  for (const badge of document.querySelectorAll(".unused-label")) {
    const group = badge.parentElement;
    const name = group.querySelector(".operator-name");
    const type = group.querySelector(".operator-label");
    const top = Number(group.querySelector("rect").getAttribute("y"));
    const badgeY = Number(badge.getAttribute("y"));
    assert.ok(Number(name.getAttribute("y")) >= top + 10);
    assert.ok(badgeY - Number(name.getAttribute("y")) >= 14);
    assert.ok(
      Number(type.getAttribute("y")) -
        parseFloat(type.style.fontSize) -
        badgeY >=
        6,
    );
    assert.equal(badge.getAttribute("visibility"), "visible");
  }
  const select = (index) => {
    const group = document.querySelector(`[data-id="node-${index}"]`);
    const point = positions[index];
    pointer(group, "pointerdown", point.x, point.y);
    pointer(graph, "pointerup", point.x, point.y);
  };
  select(3);
  assert.equal(
    document.getElementById("metadata-unused-section").hidden,
    false,
  );
  assert.equal(
    document.getElementById("metadata-unused").textContent,
    "dimensions",
  );
  assert.match(
    document.getElementById("metadata-analysis").textContent,
    /Tagshape/,
  );
  assert.equal(document.querySelectorAll(".edge.released").length, 0);
  select(5);
  assert.equal(document.getElementById("metadata-unused-section").hidden, true);
  assert.equal(
    document.getElementById("metadata-releases-section").hidden,
    false,
  );
  assert.equal(
    document.getElementById("metadata-value-count").textContent,
    "3",
  );
  assert.equal(
    document.getElementById("metadata-node").parentElement.hidden,
    false,
  );
  assert.ok(document.querySelector("#metadata-values .metadata-badges"));
  assert.equal(
    document.querySelector("#metadata-panel details, #metadata-panel summary"),
    null,
  );
  assert.equal(document.querySelectorAll(".edge.released").length, 4);
  assert.equal(document.querySelectorAll(".tensor-label.released").length, 4);
  assert.deepEqual(
    Array.from(
      document.querySelectorAll("#metadata-releases li"),
      (item) => item.textContent,
    ),
    ["shared;activation", "constant_sum"],
  );
  assert.equal(document.querySelectorAll("#metadata-node script").length, 0);
  assert.match(
    document.getElementById("metadata-node").textContent,
    /<script>not executable<\/script>/,
  );
  assert.match(
    document.getElementById("metadata-values").textContent,
    /A result used by three nodes/,
  );
  assertMetadataTable(
    document.getElementById("metadata-node"),
    metadata.nodes[5].entries,
  );
  for (const table of document.querySelectorAll("#metadata-values table")) {
    const name = table.closest("li").querySelector("strong").textContent;
    assertMetadataTable(
      table,
      metadata.values.find((value) => value.name === name).entries,
    );
  }
  select(1);
  assert.equal(document.querySelectorAll(".edge.released").length, 1);
  assert.equal(
    document.getElementById("metadata-last-use").textContent,
    "bias",
  );
  assert.deepEqual(nodePositions(document), positions);
  assert.equal(graph.getAttribute("viewBox"), viewBox);
  toggle.click();
  assert.equal(document.querySelectorAll(".edge.released").length, 0);
  assert.equal(document.getElementById("node-details").hidden, false);
  assert.ok(
    Array.from(document.querySelectorAll(".unused-label")).every(
      (label) => label.getAttribute("visibility") === "hidden",
    ),
  );
  const count = messages.filter(({ type }) => type === "metadata").length;
  toggle.click();
  assert.equal(
    messages.filter(({ type }) => type === "metadata").length,
    count,
  );
  assert.equal(document.querySelectorAll(".edge.released").length, 1);
});

test("reports metadata errors, allows retry and clears stale metadata and release markers on replacement", (t) => {
  const { document, window, load, messages } = openViewer(t);
  load(demoGraph);
  const toggle = document.getElementById("show-metadata");
  toggle.click();
  window.dispatchEvent(
    new window.MessageEvent("message", {
      data: { type: "metadata-error", error: "analysis failed" },
    }),
  );
  assert.match(
    document.getElementById("metadata-status").textContent,
    /analysis failed/,
  );
  assert.equal(
    document.getElementById("metadata-panel").getAttribute("aria-busy"),
    "false",
  );
  toggle.click();
  toggle.click();
  assert.equal(
    messages.filter((message) => message.type === "metadata").length,
    2,
  );
  const model = loadModel(
    readFileSync(require.resolve("./models/analysis.onnx")),
    "analysis.onnx",
  );
  load(model.graph());
  window.dispatchEvent(
    new window.MessageEvent("message", {
      data: { type: "metadata-info", metadata: model.metadata() },
    }),
  );
  assert.equal(document.querySelectorAll(".unused-label").length, 2);
  const refresh = document.getElementById("refresh-metadata");
  const content =
    document.getElementById("metadata-analysis").firstElementChild;
  const count = messages.length;
  refresh.click();
  assert.equal(messages.length, count + 1);
  assert.equal(messages.at(-1).type, "metadata");
  assert.equal(refresh.disabled, true);
  assert.equal(document.getElementById("metadata-content").hidden, false);
  refresh.click();
  assert.equal(messages.length, count + 1);
  window.dispatchEvent(
    new window.MessageEvent("message", {
      data: { type: "metadata-error", error: "refresh failed" },
    }),
  );
  assert.equal(refresh.disabled, false);
  assert.equal(document.getElementById("metadata-content").hidden, false);
  assert.match(
    document.getElementById("metadata-status").textContent,
    /refresh failed/,
  );
  refresh.click();
  assert.equal(messages.length, count + 2);
  window.dispatchEvent(
    new window.MessageEvent("message", {
      data: { type: "metadata-info", metadata: model.metadata() },
    }),
  );
  assert.equal(refresh.disabled, false);
  assert.equal(
    document.getElementById("metadata-status").textContent,
    "Metadata refreshed: 0 updates",
  );
  assert.notEqual(
    document.getElementById("metadata-analysis").firstElementChild,
    content,
  );
  load(demoGraph);
  assert.equal(messages.at(-1).type, "metadata");
  assert.equal(document.getElementById("metadata-content").hidden, true);
  assert.equal(
    document.querySelectorAll(".unused-label, .edge.released").length,
    0,
  );
});

test("lists local definitions in model inspection and opens them using their buttons", (t) => {
  const { document, load, pointer, messages } = openViewer(t);
  const model = loadModel(
    readFileSync(require.resolve("./models/functions.onnx")),
    "functions.onnx",
  );
  const root = model.graph();
  const inspectModel = () => {
    const svg = document.getElementById("graph");
    pointer(svg, "pointerdown", 0, 0);
    pointer(svg, "pointerup", 0, 0);
  };
  load(root);
  const section = document.getElementById("scope-functions");
  const list = document.getElementById("local-functions");
  assert.equal(section.hidden, true);
  inspectModel();
  assert.equal(section.hidden, false);
  assert.equal(list.children.length, root.localFunctions.length);
  assert.equal(
    document.getElementById("function-count").textContent,
    String(root.localFunctions.length),
  );
  for (const [index, button] of [
    ...list.querySelectorAll("button"),
  ].entries()) {
    assert.ok(button.textContent.includes(root.localFunctions[index].name));
    assert.ok(
      button.textContent.includes(
        `${root.localFunctions[index].nodeCount} nodes`,
      ),
    );
  }
  list.querySelector("button").click();
  assert.equal(messages.at(-1).type, "navigate");
  assert.deepEqual(
    JSON.parse(JSON.stringify(messages.at(-1).path)),
    root.localFunctions[0].path,
  );
  load(model.graph(false, messages.at(-1).path));
  assert.equal(section.hidden, true);
  assert.equal(list.children.length, 0);
  assert.equal(
    document.querySelector('#graph-navigation [aria-current="page"]')
      .textContent,
    root.localFunctions[0].name,
  );
  document.querySelector("#graph-navigation button").click();
  load(model.graph(false, messages.at(-1).path));
  inspectModel();
  assert.equal(list.children.length, root.localFunctions.length);
  load(demoGraph);
  inspectModel();
  assert.equal(list.children.length, 0);
  assert.equal(document.getElementById("function-count").textContent, "0");
});

test("opens local functions from the operator or inspector and returns using breadcrumbs", (t) => {
  const { document, window, messages, load } = openViewer(t);
  const model = loadModel(
    readFileSync(require.resolve("./models/functions.onnx")),
    "functions.onnx",
  );
  const root = model.graph();
  load(root);
  const button = document.getElementById("open-function");
  assert.equal(button.hidden, false);
  assert.match(button.textContent, /example::Transform/);
  const link = document.querySelector(".subgraph-link");
  assert.match(
    link.getAttribute("aria-label"),
    /Open Function: example::Transform/,
  );
  link.dispatchEvent(
    new window.KeyboardEvent("keydown", { key: "Enter", bubbles: true }),
  );
  assert.deepEqual(
    JSON.parse(JSON.stringify(messages.at(-1).path)),
    root.nodes[0].function.path,
  );
  load(model.graph(false, messages.at(-1).path));
  assert.equal(button.hidden, true);
  document
    .querySelector('[data-id="node-2"]')
    .dispatchEvent(
      new window.KeyboardEvent("keydown", { key: "Enter", bubbles: true }),
    );
  assert.equal(button.hidden, false);
  assert.match(button.textContent, /example::Identity \[fast\]/);
  button.click();
  load(model.graph(false, messages.at(-1).path));
  assert.equal(document.getElementById("node-title").textContent, "Sigmoid");
  assert.equal(button.hidden, true);
  const crumbs = document.querySelectorAll("#graph-navigation button");
  assert.equal(crumbs.length, 3);
  crumbs[1].click();
  load(model.graph(false, messages.at(-1).path));
  assert.equal(
    document.querySelector('#graph-navigation [aria-current="page"]')
      .textContent,
    "example::Transform",
  );
  document.querySelector("#graph-navigation button").click();
  load(model.graph(false, messages.at(-1).path));
  assert.equal(button.hidden, false);
  load(demoGraph);
  assert.equal(button.hidden, true);
});

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
    /A LITTLE CLARITY|Your model, connected|Explore the graph|ONNX MODEL|All connected initializers are listed|Attribute arrays show up to 8 elements/,
  );
  assert.equal(document.querySelectorAll(".graph-node").length, 1);
  assert.equal(document.querySelectorAll(".port").length, 2);
  assert.equal(document.querySelectorAll(".edge").length, 2);
  assert.equal(document.getElementById("node-title").textContent, "Relu");
  assert.equal(
    document.querySelector("#node-inputs .tensor-name").textContent,
    "input",
  );
  assert.equal(
    document.querySelector(".graph-node").getAttribute("aria-pressed"),
    "true",
  );
  document.getElementById("open-model").click();
  submitOpen(document, window);
  assert.equal(messages.filter((message) => message.type === "open").length, 1);
  window.dispatchEvent(
    new window.MessageEvent("message", {
      data: {
        type: "log-entry",
        entry: {
          id: messages.at(-1).actionId,
          timestamp: new Date().toISOString(),
          action: "Open ONNX model",
          status: "CANCELLED",
        },
      },
    }),
  );
  assert.equal(document.getElementById("open-backend-test"), null);
  const graph = document.getElementById("graph");
  const initialViewBox = graph.getAttribute("viewBox");
  document.getElementById("zoom-in").click();
  assert.notEqual(graph.getAttribute("viewBox"), initialViewBox);
  document.getElementById("zoom-fit").click();
  assert.equal(graph.getAttribute("viewBox"), initialViewBox);
});

test("export defaults follow the download directory without replacing chosen paths, and histories fill safe suggestions", async (t) => {
  const { document, window, messages } = openViewer(t);
  const byId = (id) => document.getElementById(id);
  const send = (data) =>
    window.dispatchEvent(new window.MessageEvent("message", { data }));
  const settings = {
    type: "settings-info",
    application: { downloadDirectory: "/downloads", pythonPath: "python3" },
    viewer: { panels: { operator: true, find: true, metadata: false } },
    trusted: true,
    exportDirectory: "/downloads/<model_id>/<exporter>",
    exportHistory: {
      exporter: "mobius",
      models: ["owner/previous", "<img src=x>"],
      directories: ["/previous/output"],
    },
  };
  const change = (id, value) => {
    byId(id).value = value;
    byId(id).dispatchEvent(new window.Event("input", { bubbles: true }));
  };
  send({ type: "load-state", pending: false });
  byId("export-model").click();
  assert.deepEqual(
    [...byId("export-tool").options].map((option) => [
      option.value,
      option.textContent,
    ]),
    [
      ["modelbuilder", "modelbuilder (ONNX Runtime GenAI)"],
      ["mobius", "Mobius"],
      ["mbext", "mbext (fast export)"],
    ],
  );
  send(settings);
  assert.equal(byId("export-tool").value, "mobius");
  assert.deepEqual(
    [...byId("export-precision").options].map((option) => option.value),
    ["fp32", "fp16", "bf16", "int4", "int2", "fp4"],
  );
  assert.ok(
    [...byId("export-precision").options].every((option) => !option.disabled),
  );
  assert.equal(
    byId("export-directory").value,
    "/downloads/<model_id>/<exporter>",
  );
  assert.equal(
    byId("export-source").getAttribute("list"),
    "export-source-history",
  );
  assert.deepEqual(
    [...byId("export-source-history").options].map((option) => option.value),
    settings.exportHistory.models,
  );
  assert.equal(byId("export-source-history").querySelector("img"), null);
  change("export-precision", "fp4");
  byId("export-tool").value = "modelbuilder";
  byId("export-tool").dispatchEvent(
    new window.Event("change", { bubbles: true }),
  );
  send(settings);
  assert.equal(byId("export-tool").value, "modelbuilder");
  assert.equal(byId("export-precision").value, "fp4");
  change("export-source", "owner/model");
  await new Promise((resolve) => window.setTimeout(resolve, 180));
  const preview = messages.findLast(
    (message) => message.type === "export-preview",
  );
  assert.equal(preview.exportOptions.directory, "");
  send({
    type: "export-preview",
    requestId: preview.requestId,
    directory: "/downloads/owner/model/modelbuilder",
    directoryTemplate: "/downloads/<model_id>/<exporter>",
  });
  assert.equal(
    byId("export-directory").value,
    "/downloads/<model_id>/<exporter>",
  );
  change("export-directory", "/previous/output");
  send({
    ...settings,
    exportDirectory: "/different/downloads/<model_id>/<exporter>",
  });
  assert.equal(byId("export-directory").value, "/previous/output");
  change("export-source", "owner/another");
  await new Promise((resolve) => window.setTimeout(resolve, 180));
  const custom = messages.findLast(
    (message) => message.type === "export-preview",
  );
  assert.equal(custom.exportOptions.directory, "/previous/output");
  send({
    type: "export-preview",
    requestId: custom.requestId,
    directory: "/downloads/owner_another",
  });
  assert.equal(byId("export-directory").value, "/previous/output");
  send({
    type: "export-history",
    history: {
      models: ["owner/another", "owner/model"],
      directories: ["/previous/output", "/downloads/owner_model"],
    },
  });
  assert.deepEqual(
    [...byId("export-directory-history").options].map((option) => option.value),
    ["/previous/output", "/downloads/owner_model"],
  );
  change("export-directory", "");
  await new Promise((resolve) => window.setTimeout(resolve, 180));
  const reset = messages.findLast(
    (message) => message.type === "export-preview",
  );
  assert.equal(reset.exportOptions.directory, "");
  send({
    type: "export-preview",
    requestId: reset.requestId,
    directory: "/different/downloads/owner_another",
    directoryTemplate: "/different/downloads/<model_id>/<exporter>",
  });
  assert.equal(
    byId("export-directory").value,
    "/different/downloads/<model_id>/<exporter>",
  );
  const template = "base/<model_id>/<exporter>";
  change("export-directory", template);
  byId("export-tool").value = "mbext";
  byId("export-tool").dispatchEvent(
    new window.Event("change", { bubbles: true }),
  );
  await new Promise((resolve) => window.setTimeout(resolve, 180));
  const templated = messages.findLast(
    (message) => message.type === "export-preview",
  );
  assert.equal(templated.exportOptions.directory, template);
  assert.equal(templated.exportOptions.exporter, "mbext");
  send({
    type: "export-preview",
    requestId: templated.requestId,
    directory: "/downloads/base/owner/another/mbext",
    command:
      "python3 -m modelbuilder.builder -o /downloads/base/owner/another/mbext",
  });
  assert.equal(byId("export-directory").value, template);
  assert.match(
    byId("export-command").textContent,
    /base\/owner\/another\/mbext/,
  );
});

test("exports from the right panel without a loaded model and shows commands, progress and cancellation", async (t) => {
  const { document, window, messages, events, load } = openViewer(t);
  const send = (data) =>
    window.dispatchEvent(new window.MessageEvent("message", { data }));
  send({ type: "load-state", pending: false });
  const button = document.getElementById("export-model");
  const output = document.getElementById("export-output");
  Object.defineProperty(output, "scrollHeight", {
    get: () =>
      document.getElementById("export-panel").hidden
        ? 0
        : output.textContent.split("\n").length * 20,
  });
  assert.equal(button.disabled, false);
  assert.equal(document.getElementById("close-export"), null);
  button.click();
  assert.equal(document.getElementById("export-panel").hidden, false);
  const field = (id, value) => {
    document.getElementById(id).value = value;
    document
      .getElementById(id)
      .dispatchEvent(new window.Event("change", { bubbles: true }));
  };
  field("export-source", "owner/model");
  field("export-directory", "/tmp/export-model");
  await new Promise((resolve) => window.setTimeout(resolve, 180));
  const preview = messages.findLast(
    (message) => message.type === "export-preview",
  );
  assert.equal(preview.exportOptions.model, "owner/model");
  send({
    type: "export-preview",
    requestId: preview.requestId - 1,
    command: "stale",
  });
  assert.equal(document.getElementById("export-command").textContent, "");
  send({
    type: "export-preview",
    requestId: preview.requestId,
    command: "python -m modelbuilder.builder",
    script: "/tmp/export-model/export_model.py",
  });
  assert.match(
    document.getElementById("export-command").textContent,
    /modelbuilder.builder/,
  );
  document.getElementById("export-form").requestSubmit();
  const request = messages.at(-1);
  assert.equal(request.type, "export");
  const runningPreviewCount = messages.filter(
    (message) => message.type === "export-preview",
  ).length;
  assert.equal(document.getElementById("export-fields").disabled, true);
  assert.equal(document.getElementById("export-progress").hidden, false);
  assert.equal(document.getElementById("open-model").disabled, true);
  button.click();
  assert.equal(document.getElementById("export-panel").hidden, true);
  assert.equal(button.getAttribute("aria-expanded"), "false");
  assert.equal(
    messages.some((message) => message.type === "cancel-export"),
    false,
  );
  button.click();
  assert.equal(document.getElementById("export-panel").hidden, false);
  assert.equal(button.getAttribute("aria-expanded"), "true");
  assert.equal(document.getElementById("export-progress").hidden, false);
  send({
    type: "export-progress",
    actionId: "old-action",
    stage: "Stale output",
    command: "stale command",
    output: "stale",
  });
  assert.equal(output.textContent, "python -m modelbuilder.builder");
  send({
    type: "export-progress",
    actionId: request.actionId,
    stage: "Exporting",
    command: "python3 -u -m modelbuilder.builder",
    output: "<img src=x> 50%",
  });
  assert.equal(
    document.getElementById("export-status").textContent,
    "Exporting",
  );
  assert.equal(
    output.textContent,
    "python3 -u -m modelbuilder.builder\n\n<img src=x> 50%",
  );
  assert.equal(document.querySelector("#export-output img"), null);
  assert.equal(output.scrollTop, output.scrollHeight);
  output.scrollTop = 0;
  const stdout = [
    ...Array.from({ length: 40 }, (_, index) => `Output line ${index}`),
    "Last stdout line",
  ].join("\n");
  send({
    type: "export-progress",
    actionId: request.actionId,
    stage: "Exporting",
    output: stdout,
  });
  assert.equal(
    output.textContent,
    `python3 -u -m modelbuilder.builder\n\n${stdout}`,
  );
  assert.equal(output.scrollTop, output.scrollHeight);
  button.click();
  send({
    type: "export-progress",
    actionId: request.actionId,
    stage: "Exporting",
    output: `${stdout}\nFinished`,
  });
  assert.equal(output.scrollTop, 0);
  button.click();
  assert.equal(output.scrollTop, output.scrollHeight);
  await new Promise((resolve) => window.setTimeout(resolve, 180));
  assert.equal(
    messages.filter((message) => message.type === "export-preview").length,
    runningPreviewCount,
  );
  assert.equal(document.querySelector("#export-output img"), null);
  document.getElementById("cancel-export").click();
  assert.equal(messages.at(-1).type, "cancel-export");
  const entry = events.find(
    (event) =>
      event.type === "interaction" && event.entry.id === request.actionId,
  ).entry;
  send({ type: "log-entry", entry: { ...entry, status: "CANCELLED" } });
  assert.equal(document.getElementById("export-fields").disabled, false);
  assert.equal(document.getElementById("export-progress").hidden, true);
  assert.match(
    document.getElementById("export-status").textContent,
    /cancelled/,
  );
  assert.equal(document.getElementById("open-model").disabled, false);
  field("export-precision", "int2");
  field("export-tool", "mobius");
  assert.equal(
    [...document.getElementById("export-precision").options].find(
      (option) => option.value === "int4",
    ).disabled,
    false,
  );
  document.getElementById("export-form").requestSubmit();
  const next = messages.at(-1);
  assert.equal(next.exportOptions.exporter, "mobius");
  assert.equal(next.exportOptions.precision, "int2");
  load(demoGraph);
  send({
    type: "export-info",
    actionId: next.actionId,
    path: "/tmp/export-model/model.onnx",
    script: "/tmp/export-model/export_model.py",
    loaded: true,
    durationMs: 1234,
  });
  const running = events.find(
    (event) => event.type === "interaction" && event.entry.id === next.actionId,
  ).entry;
  send({ type: "log-entry", entry: { ...running, status: "SUCCEEDED" } });
  assert.equal(
    document.getElementById("export-status").textContent,
    "Converted and opened /tmp/export-model/model.onnx Conversion time: 1.23 s.",
  );
  assert.match(
    document.getElementById("export-script").textContent,
    /export_model.py/,
  );
  assert.equal(document.getElementById("export-progress").hidden, true);
  assert.equal(document.getElementById("save-model").disabled, false);
  document.getElementById("export-form").requestSubmit();
  const saved = messages.at(-1);
  send({
    type: "export-info",
    actionId: next.actionId,
    loaded: false,
    durationMs: 99999,
  });
  assert.equal(
    document.getElementById("export-status").textContent,
    "Starting conversion…",
  );
  send({
    type: "export-info",
    actionId: saved.actionId,
    loaded: false,
    durationMs: 0,
  });
  const saving = events.find(
    (event) =>
      event.type === "interaction" && event.entry.id === saved.actionId,
  ).entry;
  send({ type: "log-entry", entry: { ...saving, status: "CANCELLED" } });
  assert.equal(
    document.getElementById("export-status").textContent,
    "Conversion saved; opening cancelled. Conversion time: 0.00 s.",
  );
});

test("shows stored encoded field values and folds nested layouts without dequantizing weights", (t) => {
  const { document, load } = openViewer(t);
  const graph = decodeGraph(
    readFileSync(require.resolve("../examples/quantized-3bit.onnx")),
    "quantized-3bit.onnx",
  );
  load(graph);
  const parameters = document.getElementById("node-initializers");
  assert.match(parameters.textContent, /weight \(FLOAT\).*Shape: \[8, 4\]/s);
  assert.match(
    parameters.textContent,
    /Encoding: STRUCT.*quantization.v1\/q3_k/s,
  );
  assert.match(parameters.textContent, /Storage: embedded/);
  assert.doesNotMatch(parameters.textContent, /Values:|raw_data:/);
  assert.equal(document.getElementById("parameter-count").textContent, "1");
  const tables = parameters.querySelectorAll(".initializer-type-table");
  assert.equal(tables.length, 2);
  assert.equal(tables[0].caption.textContent, "Logical type: FLOAT");
  assert.equal(tables[1].caption.textContent, "Encoding: STRUCT");
  const rows = Object.fromEntries(
    [...tables[1].tBodies[0].rows].map((row) => {
      assert.equal(row.cells[0].scope, "row");
      return [row.cells[0].title, row];
    }),
  );
  const code = rows["blocks.run_0.Element [0].codes"];
  assert.equal(code.cells[1].textContent, "UINT8 [12]");
  assert.equal(
    code.cells[2].textContent,
    "[255, 255, 255, 0, 0, 0, 0, 0, 0, 72, 146, 36]",
  );
  const scale = rows["blocks.run_0.Element [0].scale"];
  assert.equal(scale.cells[1].textContent, "DOUBLE []");
  assert.equal(scale.cells[2].textContent, "[1]");
  assert.equal(rows["blocks.run_0.Element [0].codes.Shape"], undefined);
  assert.equal(scale.querySelector("button"), null);
  assert.equal(rows.permutation.cells[1].textContent, "INT64 [0]");
  assert.equal(rows.permutation.cells[2].textContent, "");
  assert.equal(rows.permutation.cells[2].querySelector("pre"), null);
  assert.equal(
    [...tables[1].tBodies[0].rows].filter(
      (row) => row.cells[0].title === "blocks",
    ).length,
    1,
  );
  const disclosure = tables[1].closest("details");
  assert.equal(disclosure.open, false);
  assert.equal(tables[0].closest("details").open, false);
  assert.equal(code.hidden, true);
  for (const toggle of tables[1].querySelectorAll(".initializer-fold"))
    assert.equal(toggle.getAttribute("aria-expanded"), "false");
  disclosure.querySelector("summary").click();
  assert.equal(disclosure.open, true);
  rows.Layout.querySelector("button").click();
  assert.equal(rows.reserved.hidden, false);
  const blocks = rows.blocks.querySelector("button");
  blocks.click();
  assert.equal(scale.hidden, true);
  const run = rows["blocks.run_0"].querySelector("button");
  run.click();
  assert.equal(scale.hidden, true);
  rows["blocks.run_0.Element [0]"].querySelector("button").click();
  assert.equal(scale.hidden, false);
  assert.equal(code.hidden, false);
  run.click();
  assert.equal(run.getAttribute("aria-expanded"), "false");
  assert.equal(scale.hidden, true);
  blocks.click();
  assert.equal(scale.hidden, true);
  assert.equal(code.hidden, true);
  assert.equal(rows.reserved.hidden, false);
  blocks.click();
  assert.equal(scale.hidden, true);
  assert.equal(code.hidden, true);
  assert.equal(run.getAttribute("aria-expanded"), "false");
  run.click();
  assert.equal(scale.hidden, false);
  assert.equal(code.hidden, false);
  disclosure.querySelector("summary").click();
  assert.equal(disclosure.open, false);
  disclosure.querySelector("summary").click();
  assert.equal(disclosure.open, true);
  assert.equal(scale.hidden, false);
  const [parameter] = graph.nodes[0].parameters;
  parameter.encodingDetails.entries.find(({ key }) =>
    key.endsWith(".codes"),
  ).data = '<img src=x onerror="alert(1)">';
  parameter.shape = null;
  load(graph);
  assert.match(parameters.textContent, /Shape: \[\?\]/);
  assert.ok(parameters.textContent.includes('<img src=x onerror="alert(1)">'));
  assert.equal(parameters.querySelector("img"), null);
  delete parameter.shape;
  load(graph);
  assert.doesNotMatch(parameters.textContent, /Shape:/);
});

test("shows actual attributes, connected initializer parameters and versioned defaults", (t) => {
  const { document, load, window } = openViewer(t);
  const graph = decodeGraph(
    readFileSync(require.resolve("./models/parameters.onnx")),
    "parameters.onnx",
  );
  load(graph);
  assert.match(
    document.getElementById("node-attributes").textContent,
    /strides \(INTS\)[\s\S]*\[2, 2\]/,
  );
  const strides = Array.from(
    document.querySelectorAll("#node-attributes li"),
  ).find(
    (item) => item.querySelector("strong").textContent === "strides (INTS)",
  );
  assert.equal(strides.querySelector("pre").textContent, "[2, 2]");
  assert.equal(strides.textContent.match(/strides/g).length, 1);
  assert.match(
    document.getElementById("node-initializers").textContent,
    /kernel \(FLOAT\)[\s\S]*Shape: \[1, 1, 3, 3\][\s\S]*Storage: embedded/,
  );
  assert.equal(
    document.querySelector("#node-initializers pre").textContent,
    `Values: [${Array(9).fill("0.5").join(", ")}]`,
  );
  document
    .querySelector('[data-id="node-1"]')
    .dispatchEvent(new window.KeyboardEvent("keydown", { key: "Enter" }));
  const attributes = document.getElementById("node-attributes");
  assert.equal(attributes.children.length, 1);
  assert.match(attributes.textContent, /alpha \(FLOAT\)[\s\S]*0\.125/);
  assert.doesNotMatch(attributes.textContent, /\[default\]/);
  for (const [version, axis] of [
    [11, "1"],
    [13, "-1"],
  ]) {
    load(
      decodeGraph(
        readFileSync(require.resolve(`./models/softmax-${version}.onnx`)),
        "softmax.onnx",
      ),
    );
    assert.equal(
      document.getElementById("node-opset").textContent,
      String(version),
    );
    assert.match(attributes.textContent, /axis \(INT\) \[default\]/);
    assert.equal(attributes.querySelector("pre").textContent, axis);
  }
});

test("shows at most two short attributes and initializers inside operators", (t) => {
  const { document, load } = openViewer(t);
  const attribute = { name: "axis", type: "INT", value: "-1" };
  const parameter = {
    name: "bias",
    type: "FLOAT",
    shape: [],
    value: "[0.5]",
  };
  for (const [attributes, parameters, expected] of [
    [[attribute], [], ["axis = -1"]],
    [[], [parameter], ["bias: FLOAT [] = [0.5]"]],
    [[attribute], [parameter], ["axis = -1", "bias: FLOAT [] = [0.5]"]],
    [
      [attribute, { ...attribute, name: "other" }],
      [],
      ["axis = -1", "other = -1"],
    ],
    [
      [],
      [parameter, parameter],
      ["bias: FLOAT [] = [0.5]", "bias: FLOAT [] = [0.5]"],
    ],
    [
      [{ ...attribute, value: "x".repeat(31) }],
      [],
      [`axis = ${"x".repeat(31)}`],
    ],
    [
      [{ name: "<img>", type: "STRING", value: "<b>literal</b>" }],
      [],
      ["<img> = <b>literal</b>"],
    ],
  ]) {
    const graph = structuredClone(demoGraph);
    graph.nodes[0].attributes = attributes;
    graph.nodes[0].parameters = parameters;
    load(graph);
    const group = document.querySelector(".graph-node");
    const rows = Array.from(
      group.querySelectorAll(".node-attribute, .node-initializer"),
    );
    assert.deepEqual(
      rows.map((row) => row.firstChild.textContent),
      expected,
    );
    assert.equal(group.querySelector(".node-parameter-summary"), null);
    assert.equal(group.querySelector("img, b"), null);
    const body = group.querySelector("rect");
    assert.equal(
      Number(body.getAttribute("height")),
      88 + expected.length * 16,
    );
    const bottom =
      Number(body.getAttribute("y")) + Number(body.getAttribute("height"));
    const type = group.querySelector(".operator-label");
    const information = group.querySelector(".node-tensor-info");
    for (const [index, row] of rows.entries()) {
      assert.equal(row.getAttribute("aria-label"), expected[index]);
      assert.equal(row.querySelector("title").textContent, expected[index]);
      assert.equal(row.getAttribute("text-anchor"), "middle");
      assert.equal(row.getAttribute("x"), type.getAttribute("x"));
      assert.notEqual(
        document.defaultView.getComputedStyle(row).fill,
        document.defaultView.getComputedStyle(type).fill,
      );
      assert.ok(Number(row.getAttribute("y")) > Number(type.getAttribute("y")));
      assert.ok(
        Number(row.getAttribute("y")) < Number(information.getAttribute("y")),
      );
      assert.ok(Number(row.getAttribute("y")) < bottom);
    }
  }
  load(demoGraph);
  assert.equal(document.querySelector(".node-initializer"), null);
  assert.equal(
    document.querySelector(".graph-node > rect").getAttribute("height"),
    "88",
  );
});

test("summarizes numerous or long parameters while preserving inspector details", (t) => {
  const { document, load } = openViewer(t);
  const parameter = { name: "bias", type: "FLOAT", shape: [], value: "[0.5]" };
  for (const [attributes, parameters, count] of [
    [[{ name: "axis", type: "INT", value: "-1" }], [parameter, parameter], 3],
    [[], Array(12).fill(parameter), 12],
    [[{ name: "text", type: "STRING", value: "x".repeat(32) }], [], 1],
    [[{ name: "text", type: "STRING", value: "line\nbreak" }], [], 1],
    [[], [{ ...parameter, value: "x".repeat(80) }], 1],
    [
      [
        {
          name: "<img>",
          type: "STRING",
          value: "<script>alert(1)</script>".repeat(2),
        },
      ],
      [],
      1,
    ],
  ]) {
    const graph = structuredClone(demoGraph);
    graph.nodes[0].attributes = attributes;
    graph.nodes[0].parameters = parameters;
    load(graph);
    const group = document.querySelector(".graph-node");
    const summary = group.querySelector(".node-parameter-summary");
    assert.equal(
      summary.firstChild.textContent,
      `${count} parameter${count === 1 ? "" : "s"} (see inspector)`,
    );
    assert.equal(
      group.querySelectorAll(".node-attribute, .node-initializer").length,
      0,
    );
    assert.equal(group.querySelector("rect").getAttribute("height"), "104");
    assert.equal(
      document.querySelectorAll("#node-initializers li").length,
      parameters.length,
    );
    assert.equal(
      document.querySelectorAll("#node-attributes li").length,
      attributes.length,
    );
    for (const attribute of attributes) {
      assert.ok(
        document
          .getElementById("node-attributes")
          .textContent.includes(attribute.value),
      );
    }
    assert.equal(group.querySelector("img, script"), null);
  }
});

test("shows tiny initializer values as safe text and omits large and external payloads", (t) => {
  const { document, load } = openViewer(t);
  load(
    decodeGraph(
      readFileSync(require.resolve("./models/tiny-initializers.onnx")),
      "tiny-initializers.onnx",
    ),
  );
  const items = Array.from(document.querySelectorAll("#node-initializers li"));
  const item = (name) =>
    items.find((entry) =>
      entry.querySelector("strong").textContent.startsWith(`${name} (`),
    );
  assert.equal(
    item("raw_int64").querySelector("pre").textContent,
    "Values: [1152921504606846977, -1152921504606846979]",
  );
  assert.equal(
    item("strings").querySelector("pre").textContent,
    'Values: ["<img src=x onerror=alert(1)>", "hello"]',
  );
  assert.equal(item("strings").querySelector("img"), null);
  assert.equal(item("sixteen").querySelector("pre"), null);
  assert.equal(item("external").querySelector("pre"), null);
  assert.equal(document.querySelector(".node-initializer"), null);
  assert.match(
    document.querySelector(".node-parameter-summary").firstChild.textContent,
    /^\d+ parameters \(see inspector\)$/,
  );
});

test("fits initializer previews safely above subgraph links and preserves layout and dragging", (t) => {
  const { document, load, pointer } = openViewer(t);
  const graph = chainGraph(2);
  const description =
    '<img src=x onerror="alert(1)">' + "long_weight".repeat(10);
  graph.nodes[0].parameters = [{ name: description, type: "FLOAT", shape: [] }];
  graph.nodes[0].attributes = [
    {
      name: "body",
      graphs: [{ name: "child", path: [], nodeCount: 0 }],
    },
  ];
  load(graph);
  const group = document.querySelector('[data-id="node-0"]');
  const body = group.querySelector("rect");
  const row = group.querySelector(".node-parameter-summary");
  assert.equal(row.firstChild.textContent, "1 parameter (see inspector)");
  assert.ok(
    document
      .getElementById("node-initializers")
      .textContent.includes(description),
  );
  assert.equal(group.querySelector("img"), null);
  const width = Number(body.getAttribute("width"));
  assert.ok(width >= 140 && width <= 280);
  assert.ok(
    row.firstChild.textContent.length * parseFloat(row.style.fontSize) * 0.6 <=
      width - 24,
  );
  assert.ok(
    Number(row.getAttribute("y")) <
      Number(group.querySelector(".subgraph-link rect").getAttribute("y")),
  );
  const top = Number(body.getAttribute("y"));
  const bottom = top + Number(body.getAttribute("height"));
  assert.ok(
    Number(
      document.querySelector('[data-id="node-1"] > rect').getAttribute("y"),
    ) > bottom,
  );
  const edge = document.querySelector(".edge");
  assert.ok(edge.getAttribute("d").endsWith(`,${top}`));
  const before = nodePositions(document);
  pointer(row, "pointerdown", before[0].x, Number(row.getAttribute("y")));
  const svg = document.getElementById("graph");
  pointer(
    svg,
    "pointermove",
    before[0].x + 30,
    Number(row.getAttribute("y")) + 20,
  );
  pointer(
    svg,
    "pointerup",
    before[0].x + 30,
    Number(row.getAttribute("y")) + 20,
  );
  assertMoved(before, nodePositions(document), [0], 30, 20);
  const endpoint = Number(edge.getAttribute("d").split(",").at(-1));
  assert.ok(Math.abs(endpoint - top - 20) < 1e-8);
});

test("lists more than eight initializer metadata entries, including external dense/sparse filenames, as literal text", (t) => {
  const { document, load } = openViewer(t);
  const graph = decodeGraph(
    readFileSync(require.resolve("./models/initializers.onnx")),
    "initializers.onnx",
  );
  const unsafe = '<img src=x onerror="alert(1)">';
  graph.nodes[0].parameters[10].storage.location = unsafe;
  load(graph);
  const list = document.getElementById("node-initializers");
  assert.equal(list.children.length, 12);
  assert.equal(document.getElementById("parameter-count").textContent, "12");
  assert.match(
    list.children[9].textContent,
    /weight_9 \(FLOAT\).*Shape: \[1\].*Storage: embedded/,
  );
  assert.ok(list.children[10].textContent.includes(`external file: ${unsafe}`));
  assert.match(
    list.children[11].textContent,
    /Values: external file: weights\/values.bin/,
  );
  assert.match(
    list.children[11].textContent,
    /Indices: external file: weights\/indices.bin/,
  );
  assert.equal(list.querySelector("img"), null);
  assert.equal(list.querySelectorAll("pre").length, 10);
  assert.equal(list.children[10].querySelector("pre"), null);
  assert.equal(list.children[11].querySelector("pre"), null);
});

test("model inspection lists dense, sparse and encoded weights and clears them on selection or navigation", (t) => {
  const { document, window, load, pointer } = openViewer(t);
  const inspectModel = () => {
    const svg = document.getElementById("graph");
    pointer(svg, "pointerdown", 0, 0);
    pointer(svg, "pointerup", 0, 0);
  };
  const graph = decodeGraph(
    readFileSync(require.resolve("./models/initializers.onnx")),
    "initializers.onnx",
  );
  const unsafe = '<img src=x onerror="alert(1)">';
  graph.initializers[10].storage.location = unsafe;
  load(graph);
  assert.equal(document.getElementById("scope-initializers").hidden, true);
  inspectModel();
  assert.equal(document.getElementById("scope-initializers").hidden, false);
  assert.equal(document.getElementById("node-parameters").hidden, true);
  assert.equal(
    document.getElementById("dense-initializer-count").textContent,
    "11",
  );
  assert.equal(
    document.getElementById("sparse-initializer-count").textContent,
    "1",
  );
  assert.equal(
    document.getElementById("encoded-initializer-count").textContent,
    "0",
  );
  const dense = document.getElementById("scope-dense-initializers");
  assert.equal(dense.children.length, 11);
  assert.match(
    dense.children[9].textContent,
    /weight_9.*Shape: \[1\].*Values: \[9.25\]/,
  );
  assert.ok(dense.children[10].textContent.includes(unsafe));
  assert.equal(dense.querySelector("img"), null);
  assert.match(
    document.getElementById("scope-sparse-initializers").textContent,
    /external_sparse.*Values: external file: weights\/values.bin.*Indices: external file: weights\/indices.bin/,
  );
  document
    .querySelector(".graph-node")
    .dispatchEvent(new window.KeyboardEvent("keydown", { key: "Enter" }));
  assert.equal(document.getElementById("scope-initializers").hidden, true);
  assert.equal(dense.children.length, 0);
  const encoded = decodeGraph(
    readFileSync(require.resolve("../examples/quantized-2bit.onnx")),
    "quantized-2bit.onnx",
  );
  load(encoded);
  inspectModel();
  const encodedList = document.getElementById("scope-encoded-initializers");
  assert.equal(encodedList.children.length, encoded.encodedInitializers.length);
  assert.ok(encodedList.children.length > 0);
  assert.match(encodedList.textContent, /Encoding: STRUCT/);
  assert.equal(dense.children.length, 0);
  load(demoGraph);
  inspectModel();
  assert.equal(encodedList.children.length, 0);
  assert.equal(
    document.getElementById("encoded-initializer-count").textContent,
    "0",
  );
});

test("opens native LightOpSchema docs and clears stale docs and parameters on selection or model changes", (t) => {
  const { document, load, window } = openViewer(t);
  const docButton = document.getElementById("show-doc");
  const dialog = document.getElementById("operator-doc");
  assert.equal(docButton.disabled, true);
  load(demoGraph);
  assert.equal(docButton.disabled, false);
  docButton.click();
  assert.equal(dialog.open, true);
  assert.match(document.getElementById("doc-body").textContent, /Relu/);
  assert.ok(document.querySelector("#doc-body p"));
  assert.match(
    document.getElementById("doc-version").textContent,
    /model opset 13/,
  );
  document.getElementById("close-doc").click();
  assert.equal(dialog.open, false);
  docButton.click();
  load(demoGraph);
  assert.equal(dialog.open, false);
  assert.equal(document.getElementById("doc-body").textContent, "");
  docButton.click();
  document
    .querySelector(".port")
    .dispatchEvent(new window.KeyboardEvent("keydown", { key: "Enter" }));
  assert.equal(dialog.open, false);
  assert.equal(docButton.disabled, true);
  assert.equal(document.getElementById("node-parameters").hidden, true);
  assert.equal(document.getElementById("node-attributes").children.length, 0);
  load(
    decodeGraph(
      readFileSync(require.resolve("./models/parameters.onnx")),
      "parameters.onnx",
    ),
  );
  docButton.click();
  document
    .querySelector('[data-id="node-2"]')
    .dispatchEvent(new window.KeyboardEvent("keydown", { key: "Enter" }));
  assert.equal(dialog.open, false);
  assert.equal(docButton.disabled, true);
  assert.equal(document.getElementById("doc-body").textContent, "");
  assert.match(
    document.getElementById("node-initializers").textContent,
    /SPARSE FLOAT/,
  );
  load({ ...demoGraph, nodes: [] });
  assert.equal(document.getElementById("node-parameters").hidden, true);
  assert.equal(document.getElementById("node-initializers").children.length, 0);
});

test("separates native input and output entries from their descriptions", (t) => {
  const { document, load } = openViewer(t);
  load(
    decodeGraph(
      readFileSync(require.resolve("./models/custom.onnx")),
      "custom.onnx",
    ),
  );
  document.getElementById("show-doc").click();
  const body = document.getElementById("doc-body");
  assert.deepEqual(
    [...body.querySelectorAll("h2")].map((heading) => heading.textContent),
    ["Inputs", "Outputs", "Attributes", "Type constraints"],
  );
  const parameters = [...body.querySelectorAll("h3")];
  assert.deepEqual(
    parameters.map((heading) => heading.textContent),
    ["A (T)", "B (T)", "C (T)"],
  );
  for (const parameter of parameters) {
    assert.equal(parameter.nextElementSibling.tagName, "P");
    assert.ok(parameter.nextElementSibling.textContent.trim());
  }
});

test("renders Markdown documentation with headings, lists, code, tables and links", (t) => {
  const { document, load } = openViewer(t);
  const graph = structuredClone(demoGraph);
  graph.nodes[0].schema.doc = [
    "# Operator",
    "",
    "**Bold** and *italic* with `axis`.",
    "",
    "- input",
    "- output",
    "",
    "1. First",
    "2. Next",
    "",
    "```python",
    "x = '<tag>'",
    "```",
    "",
    "| Input | Type |",
    "| --- | --- |",
    "| x | FLOAT |",
    "",
    "> Note",
    "",
    "[ONNX](https://onnx.ai/)",
  ].join("\n");
  load(graph);
  document.getElementById("show-doc").click();
  const body = document.getElementById("doc-body");
  assert.equal(body.querySelector("h1").textContent, "Operator");
  assert.equal(body.querySelector("strong").textContent, "Bold");
  assert.equal(body.querySelector("em").textContent, "italic");
  assert.equal(body.querySelector("p code").textContent, "axis");
  assert.equal(body.querySelectorAll("ul li").length, 2);
  assert.equal(body.querySelectorAll("ol li").length, 2);
  assert.equal(
    body.querySelector("pre code.language-python").textContent,
    "x = '<tag>'\n",
  );
  assert.equal(body.querySelector("tag"), null);
  assert.deepEqual(
    [...body.querySelectorAll("th")].map((cell) => cell.textContent),
    ["Input", "Type"],
  );
  assert.equal(body.querySelector("blockquote").textContent.trim(), "Note");
  assert.equal(body.querySelector("a").href, "https://onnx.ai/");
});

test("does not activate HTML, unsafe Markdown links or remote images in documentation", (t) => {
  const { document, load, window } = openViewer(t);
  const graph = structuredClone(demoGraph);
  graph.nodes[0].schema.doc = [
    "<script>window.docExecuted = true</script>",
    '<img src=x onerror="window.docExecuted = true">',
    "",
    "[run](javascript:alert(1))",
    "[encoded](jav&#x61;script:alert(1))",
    "[mixed](JaVaScRiPt:alert(1))",
    "[command](command:workbench.action.openSettings)",
    "[file](file:///etc/passwd)",
    "[data](data:text/html;base64,PHNjcmlwdD4=)",
    "[relative](../model.onnx)",
    "[scheme-relative](//tracking.invalid/pixel)",
    "![image label](https://tracking.invalid/pixel)",
    '![<img src=x onerror="alert(1)">](https://tracking.invalid/pixel)',
  ].join("\n");
  load(graph);
  document.getElementById("show-doc").click();
  const body = document.getElementById("doc-body");
  assert.equal(body.querySelector("a, img, script"), null);
  assert.match(body.textContent, /<script>window.docExecuted = true<\/script>/);
  assert.match(body.textContent, /image label/);
  assert.match(body.textContent, /<img src=x onerror="alert\(1\)">/);
  assert.equal(window.docExecuted, undefined);
});

test("renders attribute names, values and raw HTML in schema documentation as text", (t) => {
  const { document, load } = openViewer(t);
  const text = '<img src=x onerror="alert(1)">';
  load({
    ...demoGraph,
    nodes: [
      {
        ...demoGraph.nodes[0],
        attributes: [
          { name: text, type: "STRING", value: text, default: false },
        ],
        schema: { ...demoGraph.nodes[0].schema, doc: text },
      },
    ],
  });
  document.getElementById("show-doc").click();
  for (const id of ["node-attributes", "doc-body"]) {
    assert.ok(document.getElementById(id).textContent.includes(text));
    assert.equal(document.getElementById(id).querySelector("img"), null);
  }
});

test("links the viewer logo to its repository and requests a shape refresh only while idle", (t) => {
  const { document, load, messages, window } = openViewer(t);
  const link = document.getElementById("repository-link");
  assert.equal(link.tagName, "A");
  assert.equal(link.href, "https://github.com/xadupre/onnx-light-viewer");
  assert.ok(link.querySelector('img[src$="/logo.svg"]'));
  assert.match(link.getAttribute("aria-label"), /GitHub repository/);
  const refresh = document.getElementById("refresh-shapes");
  assert.ok(refresh.querySelector("svg"));
  assert.equal(refresh.textContent.trim(), "Refresh shapes");
  assert.equal(refresh.disabled, true);
  load(demoGraph);
  assert.equal(refresh.disabled, false);
  refresh.click();
  refresh.click();
  assert.equal(refresh.disabled, true);
  assert.equal(refresh.getAttribute("aria-busy"), "true");
  assert.equal(
    messages.filter((message) => message.type === "refresh-shapes").length,
    1,
  );
  window.dispatchEvent(
    new window.MessageEvent("message", {
      data: { type: "shape-error", error: "Unsupported operator" },
    }),
  );
  assert.equal(refresh.disabled, false);
  assert.equal(refresh.getAttribute("aria-busy"), "false");
  assert.match(
    document.getElementById("shape-status").textContent,
    /Unsupported operator/,
  );
  assert.equal(
    document.querySelector("#node-inputs .tensor-info").textContent,
    "FLOAT [1, 3]",
  );
  load(demoGraph);
  assert.equal(document.getElementById("shape-status").textContent, "");
});

test("refreshes real onnx-light metadata on edges, ports and the current inspector without resetting the view", (t) => {
  const { document, load, window, pointer } = openViewer(t);
  const bytes = readFileSync(require.resolve("./models/inference.onnx"));
  load(decodeGraph(bytes, "inference.onnx"));
  const nodes = document.querySelectorAll(".graph-node");
  nodes[1].dispatchEvent(new window.KeyboardEvent("keydown", { key: "Enter" }));
  const position = nodePositions(document)[1];
  pointer(nodes[1], "pointerdown", position.x, position.y);
  pointer(
    document.getElementById("graph"),
    "pointermove",
    position.x + 45,
    position.y + 20,
  );
  pointer(
    document.getElementById("graph"),
    "pointerup",
    position.x + 45,
    position.y + 20,
  );
  nodes[2].dispatchEvent(
    new window.KeyboardEvent("keydown", { key: "Enter", ctrlKey: true }),
  );
  document.getElementById("zoom-in").click();
  const graph = document.getElementById("graph");
  graph.dispatchEvent(
    new window.WheelEvent("wheel", {
      deltaY: 40,
      shiftKey: true,
      cancelable: true,
    }),
  );
  const viewBox = graph.getAttribute("viewBox");
  const positions = nodePositions(document);
  const selected = [...document.querySelectorAll('[aria-pressed="true"]')].map(
    (node) => node.dataset.id,
  );
  const paths = [...document.querySelectorAll(".edge")].map((edge) =>
    edge.getAttribute("d"),
  );
  assert.equal(
    document.querySelector("#node-outputs .tensor-info").textContent,
    "? [?]",
  );
  document.getElementById("refresh-shapes").click();
  window.dispatchEvent(
    new window.MessageEvent("message", {
      data: {
        type: "shape-info",
        tensors: decodeGraph(bytes, "inference.onnx", true).tensors,
      },
    }),
  );
  assert.equal(document.getElementById("refresh-shapes").disabled, false);
  assert.equal(document.getElementById("node-title").textContent, "Transpose");
  assert.equal(
    document.querySelector("#node-outputs .tensor-info").textContent,
    "FLOAT [N, 1, 3]",
  );
  assert.deepEqual(
    [...document.querySelectorAll(".graph-node .node-tensor-info")].map(
      (label) => label.getAttribute("aria-label"),
    ),
    ["FLOAT [N, 3]", "FLOAT [N, 3, 1]", "FLOAT [N, 1, 3]", "INT64 [N, 1, 3]"],
  );
  assert.ok(
    [...document.querySelectorAll("#edges .tensor-info")].some(
      (label) => label.getAttribute("aria-label") === "FLOAT [N, 3, 1]",
    ),
  );
  assert.equal(
    document
      .querySelector(".port.output .tensor-info")
      .getAttribute("aria-label"),
    "INT64 [N, 1, 3]",
  );
  assert.equal(graph.getAttribute("viewBox"), viewBox);
  assert.deepEqual(nodePositions(document), positions);
  assert.deepEqual(
    [...document.querySelectorAll('[aria-pressed="true"]')].map(
      (node) => node.dataset.id,
    ),
    selected,
  );
  assert.deepEqual(
    [...document.querySelectorAll(".edge")].map((edge) =>
      edge.getAttribute("d"),
    ),
    paths,
  );
  document
    .querySelector(".port.output")
    .dispatchEvent(new window.KeyboardEvent("keydown", { key: "Enter" }));
  assert.equal(
    document.querySelector("#node-inputs .tensor-info").textContent,
    "INT64 [N, 1, 3]",
  );
  assert.match(
    document.getElementById("shape-status").textContent,
    /Shapes\/types refreshed: [1-9]\d* tensors? updated/,
  );
  document.getElementById("refresh-shapes").click();
  window.dispatchEvent(
    new window.MessageEvent("message", {
      data: {
        type: "shape-info",
        tensors: decodeGraph(bytes, "inference.onnx", true).tensors,
      },
    }),
  );
  assert.equal(
    document.getElementById("shape-status").textContent,
    "Shapes/types refreshed: 0 tensors updated",
  );
});

test("counts each changed tensor once and resets the baseline when another graph is loaded", (t) => {
  const { document, window, load } = openViewer(t);
  const tensors = [
    { name: "input", type: null, shape: null },
    { name: "output", type: null, shape: null },
  ];
  load({ ...demoGraph, tensors });
  const send = (tensors, count) => {
    document.getElementById("refresh-shapes").click();
    window.dispatchEvent(
      new window.MessageEvent("message", {
        data: { type: "shape-info", tensors },
      }),
    );
    const status = document.getElementById("shape-status");
    assert.equal(
      status.textContent,
      `Shapes/types refreshed: ${count} tensor${count === 1 ? "" : "s"} updated`,
    );
    assert.equal(status.title, status.textContent);
  };
  send(structuredClone(tensors).reverse(), 0);
  const changed = [
    { name: "input", type: "FLOAT", shape: [] },
    { name: "output", type: "FLOAT", shape: ["N", "3"] },
  ];
  send(changed, 2);
  assert.equal(
    document.querySelector("#node-inputs .tensor-info").textContent,
    "FLOAT []",
  );
  assert.equal(
    document.querySelector("#node-outputs .tensor-info").textContent,
    "FLOAT [N, 3]",
  );
  send([{ ...changed[0], type: "INT64" }, changed[1]], 1);
  send([{ ...changed[0], shape: null }, changed[1]], 1);
  send([{ name: "input", type: "sequence" }, changed[1]], 1);
  send([changed[1]], 1);
  send(changed, 1);
  send(changed, 0);
  load({ ...demoGraph, tensors });
  assert.equal(document.getElementById("shape-status").textContent, "");
  send(changed, 2);
});

test("shows every operator output's type and shape even when no connection consumes it", (t) => {
  const { document, load } = openViewer(t);
  load({
    ...demoGraph,
    nodes: [{ ...demoGraph.nodes[0], outputs: ["hidden", "unused"] }],
    outputs: [],
    tensors: [
      { name: "hidden", type: "FLOAT", shape: ["N", "3"] },
      { name: "unused", type: "BOOL", shape: [] },
    ],
  });
  const information = document.querySelector(".graph-node .node-tensor-info");
  assert.equal(
    information.getAttribute("aria-label"),
    "FLOAT [N, 3] | BOOL []",
  );
  assert.equal(
    information.querySelector("tspan").textContent,
    "FLOAT [N, 3] | BOOL []",
  );
  assert.equal(
    information.querySelector("title").textContent,
    "hidden: FLOAT [N, 3]\nunused: BOOL []",
  );
});

test("distinguishes scalar and unknown-rank tensors and renders symbolic dimensions literally", (t) => {
  const { document, load, window } = openViewer(t);
  load(
    decodeGraph(
      readFileSync(require.resolve("./models/metadata.onnx")),
      "metadata.onnx",
    ),
  );
  const ports = document.querySelectorAll(".port");
  const labels = [...ports].map((port) =>
    port.querySelector(".tensor-info").getAttribute("aria-label"),
  );
  assert.ok(labels.includes("FLOAT []"));
  assert.ok(labels.includes("FLOAT [?]"));
  assert.ok(labels.includes("INT64 [0, N, ?]"));
  const text = '<img src=x onerror="alert(1)">';
  load({
    ...demoGraph,
    tensors: [{ name: "input", type: "FLOAT", shape: [text] }],
  });
  assert.match(
    document.querySelector("#node-inputs .tensor-info").textContent,
    /<img src=x/,
  );
  assert.equal(document.querySelector("#node-inputs img"), null);
  assert.equal(
    document.querySelector("#nodes .tensor-info title").textContent,
    `input: FLOAT [${text}]`,
  );
  assert.equal(document.querySelector("#nodes .tensor-info img"), null);
  window.dispatchEvent(
    new window.MessageEvent("message", {
      data: { type: "shape-info", tensors: [] },
    }),
  );
  assert.equal(
    document.querySelector("#node-inputs .tensor-info").textContent,
    "? [?]",
  );
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
  assert.equal(
    document.querySelector("#node-inputs .tensor-name").textContent,
    "hidden",
  );
});

test("replaces the demo with a no-operator graph and inspects its scope", (t) => {
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
  assert.equal(document.getElementById("node-title").textContent, "Model");
  assert.equal(document.getElementById("node-inputs").children.length, 1);
  assert.equal(document.getElementById("node-name").textContent, "passthrough");
});

test("background selection inspects the current model, subgraph, function or pipeline", (t) => {
  const { document, window, load, pointer } = openViewer(t);
  const nested = loadModel(
    readFileSync(require.resolve("./models/subgraphs.onnx")),
    "subgraphs.onnx",
  );
  const nestedPath = nested
    .graph()
    .nodes[0].attributes.find((attribute) => attribute.graphs?.length)
    .graphs[0].path;
  const functions = loadModel(
    readFileSync(require.resolve("./models/functions.onnx")),
    "functions.onnx",
  );
  const functionPath = functions.graph().nodes.find((node) => node.function)
    .function.path;
  const pipeline = loadModel(
    readFileSync(require.resolve("./models/tiny.pipe-onnx")),
    "tiny.pipe-onnx",
  );
  for (const [graph, kind] of [
    [demoGraph, "Model"],
    [functions.graph(), "Model"],
    [nested.graph(false, nestedPath), "Graph"],
    [functions.graph(false, functionPath), "Function"],
    [pipeline.graph(), "Pipeline"],
    [pipeline.graph(false, [{ pipeline: "prefill" }]), "Model"],
  ]) {
    load(graph);
    const svg = document.getElementById("graph");
    pointer(svg, "pointerdown", 0, 0);
    pointer(svg, "pointerup", 0, 0);
    assert.equal(document.querySelectorAll(".graph-node.selected").length, 0);
    assert.equal(document.getElementById("node-title").textContent, kind);
    assert.equal(document.getElementById("node-name").textContent, graph.name);
    assert.equal(
      document.getElementById("node-domain").textContent,
      [...new Set(graph.opsets.map(({ domain }) => domain))].join(", "),
    );
    assert.deepEqual(
      [...document.getElementById("node-opset").children].map(
        (entry) => entry.textContent,
      ),
      graph.opsets.map(({ domain, version }) => `${domain}: ${version}`),
    );
    assert.equal(
      document.getElementById("node-name-label").textContent,
      `${kind.toUpperCase()} NAME`,
    );
    assert.deepEqual(
      [...document.querySelectorAll("#node-inputs .tensor-name")].map(
        (value) => value.textContent,
      ),
      graph.inputs,
    );
    if (document.getElementById("metadata-panel").hidden)
      document.getElementById("show-metadata").click();
    window.dispatchEvent(
      new window.MessageEvent("message", {
        data: {
          type: "metadata-info",
          metadata: {
            model: [{ key: "description", value: "Visible metadata" }],
            graph: [],
            nodes: [],
            values: [],
          },
        },
      }),
    );
    assertMetadataTable(document.getElementById("metadata-analysis"), [
      { key: kind, value: graph.name },
    ]);
    assert.equal(
      document.querySelector(
        "#metadata-panel details, #metadata-panel summary",
      ),
      null,
    );
    assert.equal(
      document.getElementById("metadata-model").parentElement.hidden,
      false,
    );
  }
});

test("pipeline model boxes show the stage identity and imports rather than operator placeholders", (t) => {
  const { document, load, pointer, messages } = openViewer(t);
  const pipeline = loadModel(
    readFileSync(require.resolve("./models/tiny.pipe-onnx")),
    "tiny.pipe-onnx",
  );
  const overview = pipeline.graph();
  load(overview);
  assert.match(
    document.getElementById("graph-summary").textContent,
    /^3 models /,
  );
  const boxes = [...document.querySelectorAll(".graph-node")];
  const positions = nodePositions(document);
  for (const [index, node] of overview.nodes.entries()) {
    const { x, y } = positions[index];
    pointer(boxes[index], "pointerdown", x, y);
    pointer(boxes[index], "pointerup", x, y);
    const model = node.pipelineModel;
    assert.equal(document.getElementById("node-title").textContent, "Model");
    assert.equal(
      document.getElementById("node-name-label").textContent,
      "MODEL NAME",
    );
    assert.equal(document.getElementById("node-name").textContent, model.name);
    assert.ok(
      document
        .getElementById("node-description")
        .textContent.includes(
          `${node.name} model in pipeline ${overview.name}`,
        ),
    );
    assert.ok(
      document
        .getElementById("node-description")
        .textContent.includes(`Contains ${model.nodeCount} operator`),
    );
    assert.equal(
      document.getElementById("node-domain").textContent,
      [...new Set(model.opsets.map(({ domain }) => domain))].join(", "),
    );
    assert.deepEqual(
      [...document.getElementById("node-opset").children].map(
        (entry) => entry.textContent,
      ),
      model.opsets.map(({ domain, version }) => `${domain}: ${version}`),
    );
  }
  const svg = document.getElementById("graph");
  pointer(svg, "pointerdown", 0, 0);
  pointer(svg, "pointerup", 0, 0);
  assert.equal(document.getElementById("node-title").textContent, "Pipeline");
  assert.match(
    document.getElementById("node-description").textContent,
    /each box is a contained model, not an operator/,
  );
  pointer(boxes[1], "pointerdown", positions[1].x, positions[1].y);
  pointer(boxes[1], "pointerup", positions[1].x, positions[1].y);
  document.querySelector("#node-attributes button").click();
  const path = [{ pipeline: "prefill" }];
  assert.deepEqual(JSON.parse(JSON.stringify(messages.at(-1).path)), path);
  const stage = pipeline.graph(false, path);
  load(stage);
  assert.match(
    document.getElementById("graph-summary").textContent,
    /^\d+ nodes? /,
  );
  assert.equal(
    document.getElementById("node-name-label").textContent,
    "NODE NAME",
  );
  assert.equal(
    document.getElementById("node-opset").textContent,
    stage.nodes[0].opset,
  );
});

function chainGraph(length = 6) {
  return {
    ...demoGraph,
    nodes: Array.from({ length }, (_, index) => ({
      ...demoGraph.nodes[0],
      id: `node-${index}`,
      name: `activation_${index}`,
      inputs: [index === 0 ? "input" : `tensor_${index - 1}`],
      outputs: [index === length - 1 ? "output" : `tensor_${index}`],
    })),
  };
}

function translationOffsets(transform) {
  return (
    transform?.match(/-?\d+(?:\.\d+)?(?:e[+-]?\d+)?/gi).map(Number) ?? [0, 0]
  );
}

test("extracts SVG translation offsets with decimal and scientific notation", () => {
  for (const [transform, expected] of [
    ["translate(80 50)", [80, 50]],
    ["translate(-80.25 50.5)", [-80.25, 50.5]],
    ["translate(8e+1 -5E-2)", [80, -0.05]],
    ["translate(-8.25e-7, 5.5E+2)", [-8.25e-7, 550]],
    [null, [0, 0]],
  ]) {
    assert.deepEqual(translationOffsets(transform), expected);
  }
});

function nodePositions(document, selector = ".graph-node > rect") {
  return Array.from(document.querySelectorAll(selector), (rect) => {
    const [dx, dy] = translationOffsets(
      rect.parentElement.getAttribute("transform"),
    );
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

test("fits complete operator types and names without redundant node or port badges", (t) => {
  const { document, load, pointer } = openViewer(t);
  const nodes = [
    {
      ...demoGraph.nodes[0],
      id: "normalization",
      opType: "LayerNormalization",
    },
    {
      ...demoGraph.nodes[0],
      id: "long-type",
      opType: "VeryLongCustomNormalizationOperatorNameThatCannotFitAtFullSize",
    },
    { ...demoGraph.nodes[0], id: "long-name", name: "model_layer_".repeat(10) },
  ];
  load({ ...demoGraph, nodes });
  assert.equal(document.querySelector(".operator-tag, .port-type"), null);
  const groups = document.querySelectorAll(".graph-node");
  for (const [index, group] of Array.from(groups).entries()) {
    assert.equal(
      group.querySelector(".operator-label").textContent,
      nodes[index].opType,
    );
    assert.equal(
      group.querySelector(".operator-name").textContent,
      nodes[index].name,
    );
    const width = Number(group.querySelector("rect").getAttribute("width"));
    assert.ok(width > 140 && width <= 280);
    const body = group.querySelector("rect");
    const name = group.querySelector(".operator-name");
    assert.equal(name.getAttribute("text-anchor"), "end");
    assert.equal(
      Number(name.getAttribute("x")),
      Number(body.getAttribute("x")) + width - 12,
    );
    assert.equal(
      Number(name.getAttribute("y")),
      Number(body.getAttribute("y")) + 15,
    );
    assert.ok(
      Number(name.getAttribute("y")) <
        Number(group.querySelector(".operator-label").getAttribute("y")),
    );
  }
  assert.equal(
    groups[0].querySelector(".operator-label").style.fontSize,
    "16px",
  );
  assert.ok(
    parseFloat(groups[1].querySelector(".operator-label").style.fontSize) < 16,
  );
  assert.ok(
    parseFloat(groups[2].querySelector(".operator-name").style.fontSize) < 10,
  );
  assert.match(
    document.querySelector(".port").getAttribute("aria-label"),
    /^Input:/,
  );
  assert.match(
    document.querySelector(".port.output").getAttribute("aria-label"),
    /^Output:/,
  );

  const graph = document.getElementById("graph");
  const rect = groups[1].querySelector("rect");
  const right =
    Number(rect.getAttribute("x")) + Number(rect.getAttribute("width"));
  const { y } = nodePositions(document)[1];
  pointer(graph, "pointerdown", right + 10, y - 10);
  pointer(graph, "pointermove", right - 2, y + 10);
  pointer(graph, "pointerup", right - 2, y + 10);
  assert.equal(groups[1].getAttribute("aria-pressed"), "true");
});

test("changes only affected selections and edges, and preserves an unchanged inspector during dragging", (t) => {
  const { document, window, load, pointer } = openViewer(t);
  load(chainGraph(200));
  const graph = document.getElementById("graph");
  const nodes = document.querySelectorAll(".graph-node");
  const edges = document.querySelectorAll(".edge");
  const before = nodePositions(document);
  const inspectorInput =
    document.getElementById("node-inputs").firstElementChild;
  const canvas = document.getElementById("canvas");
  const viewport = canvas.getBoundingClientRect;
  let viewportReads = 0;
  canvas.getBoundingClientRect = () => {
    viewportReads++;
    return viewport();
  };
  const observer = new window.MutationObserver(() => {});
  observer.observe(document.querySelector("main"), {
    attributes: true,
    childList: true,
    subtree: true,
  });
  t.after(() => observer.disconnect());
  pointer(nodes[0], "pointerdown", before[0].x, before[0].y);
  assert.equal(
    document.getElementById("node-inputs").firstElementChild,
    inspectorInput,
  );
  assert.equal(
    observer
      .takeRecords()
      .filter(({ target }) => target.closest(".graph-node, .port")).length,
    0,
  );
  pointer(graph, "pointermove", before[0].x + 1000, before[0].y + 1000);
  assert.deepEqual(
    observer
      .takeRecords()
      .filter(({ target }) => target.matches(".edge"))
      .map(({ target }) => target),
    [edges[0], edges[1]],
  );
  pointer(graph, "pointerup", before[0].x + 1000, before[0].y + 1000);
  assert.equal(viewportReads, 1);
  assert.equal(
    observer.takeRecords().filter(({ target }) => target.matches(".edge"))
      .length,
    0,
  );

  pointer(nodes[1], "pointerdown", before[1].x, before[1].y, { ctrlKey: true });
  const changed = observer
    .takeRecords()
    .filter(({ target }) => target.matches(".graph-node, .port"));
  assert.deepEqual(
    new Set(changed.map(({ target }) => target)),
    new Set([nodes[1]]),
  );
  assert.notEqual(
    document.getElementById("node-inputs").firstElementChild,
    inspectorInput,
  );
  pointer(graph, "pointermove", before[1].x + 1000, before[1].y + 1000);
  const moved = observer
    .takeRecords()
    .filter(({ target }) => target.matches(".edge"))
    .map(({ target }) => target);
  assert.equal(moved.length, 3);
  assert.deepEqual(new Set(moved), new Set([edges[0], edges[1], edges[2]]));
  assert.equal(viewportReads, 2);
  window.dispatchEvent(
    new window.MessageEvent("message", {
      data: { type: "shape-info", tensors: demoGraph.tensors },
    }),
  );
  assert.equal(viewportReads, 3);
  pointer(graph, "pointermove", before[1].x + 1100, before[1].y + 1100);
  assert.equal(viewportReads, 4);
  pointer(graph, "pointercancel", before[1].x, before[1].y);
  assertMoved(before, nodePositions(document), [0], 1000, 1000);
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

test("aligns a dependency chain deterministically on each model load", (t) => {
  const { document, load } = openViewer(t);
  const model = chainGraph();
  load(model);
  const positions = nodePositions(document);
  positions.forEach((position, index) => {
    assert.equal(position.y, 177 + index * 124);
    const margin =
      Number(
        document
          .querySelector(`[data-id="node-${index}"] > rect`)
          .getAttribute("width"),
      ) /
        2 +
      20;
    assert.ok(position.x >= margin && position.x <= 500 - margin);
  });
  assert.equal(new Set(positions.map((position) => position.x)).size, 1);
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
  assert.deepEqual(nodePositions(document), positions);
});

test("places parallel branches together and connects captures in dependency order", (t) => {
  const { document, load } = openViewer(t);
  const model = chainGraph(4);
  const [source, left, right, join] = model.nodes;
  source.outputs = ["source"];
  left.inputs = ["source"];
  left.outputs = ["left"];
  left.parameters = [{ name: "weight", type: "float32", value: "[1]" }];
  right.inputs = [];
  right.captures = ["source"];
  right.outputs = ["right"];
  join.inputs = ["left", "right"];
  model.nodes = [join, right, source, left];
  load(model);
  const boxes = new Map(
    [...document.querySelectorAll(".graph-node")].map((group) => [
      group.dataset.id,
      group.querySelector("rect"),
    ]),
  );
  const position = (node) =>
    nodePositions(document, `[data-id="${node.id}"] > rect`)[0];
  assert.equal(position(left).y, position(right).y);
  assert.ok(position(source).y < position(left).y);
  assert.ok(position(join).y > position(left).y);
  assert.ok(
    Math.abs(position(left).x - position(right).x) >
      (Number(boxes.get(left.id).getAttribute("width")) +
        Number(boxes.get(right.id).getAttribute("width"))) /
        2,
  );
  assert.equal(document.querySelectorAll(".edge").length, 6);
  const positions = nodePositions(document);
  load(model);
  assert.deepEqual(nodePositions(document), positions);
});

test("refreshes operator and port positions independently while retaining shapes, selection and inspector", (t) => {
  const { document, window, load, pointer, messages } = openViewer(t);
  const refresh = document.getElementById("refresh-positions");
  assert.equal(refresh.disabled, true);
  load(
    decodeGraph(
      readFileSync(require.resolve("./models/inference.onnx")),
      "inference.onnx",
      true,
    ),
  );
  assert.equal(refresh.disabled, false);
  const graph = document.getElementById("graph");
  const initialViewBox = graph.getAttribute("viewBox");
  const ports = nodePositions(document, ".port rect");
  const initialNodeY = nodePositions(document).map(({ y }) => y);
  const nodes = document.querySelectorAll(".graph-node");
  nodes[1].dispatchEvent(new window.KeyboardEvent("keydown", { key: "Enter" }));
  for (const port of document.querySelectorAll(".port")) {
    port.dispatchEvent(
      new window.KeyboardEvent("keydown", { key: "Enter", ctrlKey: true }),
    );
  }
  const position = nodePositions(document)[1];
  pointer(nodes[1], "pointerdown", position.x, position.y);
  pointer(graph, "pointermove", position.x + 600, position.y + 400);
  pointer(graph, "pointerup", position.x + 600, position.y + 400);
  document.getElementById("zoom-in").click();
  graph.dispatchEvent(
    new window.WheelEvent("wheel", { deltaY: 100, shiftKey: true }),
  );
  const moved = nodePositions(document);
  assert.notDeepEqual(nodePositions(document, ".port rect"), ports);
  const selected = [...document.querySelectorAll('[aria-pressed="true"]')].map(
    (node) => node.dataset.id,
  );
  const inspector = document.querySelector("aside").textContent;
  const labels = [...document.querySelectorAll(".tensor-info")].map(
    (label) => label.textContent,
  );
  const messageCount = messages.length;
  refresh.click();
  assert.deepEqual(nodePositions(document, ".port rect"), ports);
  assert.deepEqual(
    nodePositions(document).map(({ y }) => y),
    initialNodeY,
  );
  assert.notDeepEqual(nodePositions(document), moved);
  assert.equal(graph.getAttribute("viewBox"), initialViewBox);
  assert.deepEqual(
    [...document.querySelectorAll('[aria-pressed="true"]')].map(
      (node) => node.dataset.id,
    ),
    selected,
  );
  assert.equal(document.querySelector("aside").textContent, inspector);
  assert.deepEqual(
    [...document.querySelectorAll(".tensor-info")].map(
      (label) => label.textContent,
    ),
    labels,
  );
  assert.equal(messages.length, messageCount);
  const firstEdge = document
    .querySelector(".edge")
    .getAttribute("d")
    .match(/-?\d+(?:\.\d+)?(?:e[+-]?\d+)?/gi)
    .map(Number);
  assert.deepEqual(firstEdge.slice(0, 2), [ports[0].x, ports[0].y + 26]);
});

test("zooms in and out with buttons and Ctrl+wheel without rearranging the graph", (t) => {
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
    const event = new window.WheelEvent("wheel", {
      deltaY,
      ctrlKey: true,
      cancelable: true,
    });
    graph.dispatchEvent(event);
    return event;
  };
  assert.equal(wheel(-100).defaultPrevented, true);
  assert.ok(viewWidth() < initialWidth);
  wheel(100);
  graph
    .getAttribute("viewBox")
    .split(" ")
    .map(Number)
    .forEach((value, index) =>
      assert.ok(
        Math.abs(value - Number(initialViewBox.split(" ")[index])) < 1e-9,
      ),
    );
  wheel(100);
  assert.ok(viewWidth() > initialWidth);
  const zoomedOut = graph.getAttribute("viewBox");
  wheel(0);
  assert.equal(graph.getAttribute("viewBox"), zoomedOut);
  document.getElementById("zoom-fit").click();
  assert.equal(graph.getAttribute("viewBox"), initialViewBox);
  assert.deepEqual(nodePositions(document), positions);
});

test("magnifies a rectangle in either direction without moving or selecting nodes", (t) => {
  const { document, window, load, pointer } = openViewer(t);
  const style = document.createElement("style");
  style.textContent = readFileSync(
    require.resolve("../media/viewer.css"),
    "utf8",
  );
  document.head.append(style);
  load(chainGraph());
  const graph = document.getElementById("graph");
  const button = document.getElementById("zoom-area");
  const inactiveBackground = window.getComputedStyle(button).backgroundColor;
  let activeBackground;
  const indicator = button.querySelector(".toggle-indicator");
  assert.equal(window.getComputedStyle(indicator).visibility, "hidden");
  const initial = graph.getAttribute("viewBox");
  const positions = nodePositions(document);
  const nodes = [...document.querySelectorAll(".graph-node")];
  const selection = () =>
    nodes.map((node) => node.classList.contains("selected"));
  const selected = selection();
  const inspector = document.querySelector("aside").textContent;
  for (const reverse of [false, true]) {
    const start = reverse ? { x: 300, y: 300 } : { x: 100, y: 150 };
    const end = reverse ? { x: 100, y: 150 } : { x: 300, y: 300 };
    if (!reverse) {
      button.click();
      assert.equal(button.getAttribute("aria-pressed"), "true");
      assert.equal(document.activeElement, graph);
      activeBackground = window.getComputedStyle(button).backgroundColor;
      assert.notEqual(activeBackground, inactiveBackground);
      assert.equal(window.getComputedStyle(button).fontWeight, "700");
      assert.equal(window.getComputedStyle(indicator).visibility, "visible");
    }
    pointer(nodes[0], "pointerdown", start.x, start.y, { altKey: reverse });
    pointer(graph, "pointermove", end.x, end.y);
    assert.ok(document.querySelector(".selection-rectangle"));
    assert.deepEqual(selection(), selected);
    pointer(graph, "pointerup", end.x, end.y);
    const [x, y, width, height] = graph
      .getAttribute("viewBox")
      .split(" ")
      .map(Number);
    assert.ok(Math.abs(x + width / 2 - 200) < 1e-9);
    assert.ok(Math.abs(y + height / 2 - 225) < 1e-9);
    assert.ok(Math.abs(height - 150) < 1e-9);
    assert.ok(Math.abs(width / height - 1060 / 350) < 1e-9);
    assert.equal(document.querySelector(".selection-rectangle"), null);
    assert.equal(button.getAttribute("aria-pressed"), "false");
    assert.notEqual(
      window.getComputedStyle(button).backgroundColor,
      activeBackground,
    );
    assert.equal(window.getComputedStyle(button).fontWeight, "600");
    assert.equal(window.getComputedStyle(indicator).visibility, "hidden");
    assert.equal(graph.hasPointerCapture(1), false);
    assert.deepEqual(selection(), selected);
    assert.deepEqual(nodePositions(document), positions);
    assert.equal(document.querySelector("aside").textContent, inspector);
    document.getElementById("zoom-out").click();
    assert.ok(Number(graph.getAttribute("viewBox").split(" ")[3]) > height);
    document.getElementById("zoom-fit").click();
    assert.equal(graph.getAttribute("viewBox"), initial);
  }
});

test("clamps area zoom outside the captured viewport so zooming out never zooms in", (t) => {
  const { document, load, pointer } = openViewer(t);
  load(chainGraph());
  const graph = document.getElementById("graph");
  const box = () => graph.getAttribute("viewBox").split(" ").map(Number);
  for (let index = 0; index < 10; index++)
    document.getElementById("zoom-out").click();
  const [, , minimumWidth, minimumHeight] = box();
  for (const altKey of [false, true]) {
    const [x, y, width, height] = box();
    if (!altKey) document.getElementById("zoom-area").click();
    pointer(graph, "pointerdown", x + width * 0.05, y + height * 0.05, {
      altKey,
    });
    assert.equal(graph.hasPointerCapture(1), true);
    pointer(graph, "pointerup", x + width * 1.15, y + height * 1.15);
    const area = box();
    assert.ok(Math.abs(area[2] - minimumWidth) < 1e-8);
    assert.ok(Math.abs(area[3] - minimumHeight) < 1e-8);
    assert.ok(Math.abs(area[0] + area[2] / 2 - (x + width * 0.6)) < 1e-8);
    assert.ok(Math.abs(area[1] + area[3] / 2 - (y + height * 0.6)) < 1e-8);
    document.getElementById("zoom-out").click();
    assert.deepEqual(box(), area);
  }
});

test("cancels area zoom and ignores clicks and degenerate rectangles", (t) => {
  const { document, window, load, pointer, resize } = openViewer(t);
  load(chainGraph());
  const graph = document.getElementById("graph");
  const button = document.getElementById("zoom-area");
  const initial = graph.getAttribute("viewBox");
  const selected = document.querySelector(".graph-node.selected");
  for (const cancel of [
    () =>
      graph.dispatchEvent(
        new window.KeyboardEvent("keydown", { key: "Escape" }),
      ),
    () => pointer(graph, "pointercancel", 300, 300),
    () => graph.releasePointerCapture(1),
    () => window.dispatchEvent(new window.Event("blur")),
    () => resize(1060, 350),
    () => button.click(),
  ]) {
    button.click();
    pointer(graph, "pointerdown", 100, 100);
    pointer(graph, "pointermove", 300, 300);
    cancel();
    pointer(graph, "pointerup", 300, 300);
    assert.equal(graph.getAttribute("viewBox"), initial);
    assert.equal(button.getAttribute("aria-pressed"), "false");
    assert.equal(document.querySelector(".selection-rectangle"), null);
    assert.equal(document.querySelector(".graph-node.selected"), selected);
  }
  for (const [x, y] of [
    [100, 100],
    [100, 300],
    [300, 100],
    [100.1, 100.1],
  ]) {
    button.click();
    pointer(graph, "pointerdown", 100, 100);
    pointer(graph, "pointerup", x, y);
    assert.equal(graph.getAttribute("viewBox"), initial);
    assert.equal(button.getAttribute("aria-pressed"), "false");
  }
  button.click();
  graph.dispatchEvent(new window.KeyboardEvent("keydown", { key: "Escape" }));
  assert.equal(button.getAttribute("aria-pressed"), "false");
  assert.equal(document.querySelector(".graph-node.selected"), selected);
  button.click();
  pointer(graph, "pointerdown", 100, 100);
  load(demoGraph);
  assert.equal(button.getAttribute("aria-pressed"), "false");
  assert.equal(graph.hasPointerCapture(1), false);
  assert.equal(document.querySelector(".selection-rectangle"), null);
});

test("prioritizes area zoom on subgraph rows without breaking normal activation", (t) => {
  const { document, window, load, pointer, messages } = openViewer(t);
  const model = loadModel(
    readFileSync(require.resolve("./models/subgraphs.onnx")),
    "subgraphs.onnx",
  );
  const graph = document.getElementById("graph");
  for (const altKey of [false, true]) {
    for (const outcome of ["zoom", "cancel", "click"]) {
      load(model.graph());
      const link = document.querySelector(".subgraph-link");
      const target = link.querySelector("text");
      const row = link.querySelector("rect");
      const x = Number(row.getAttribute("x")) + 10;
      const y = Number(row.getAttribute("y")) + 10;
      const initial = graph.getAttribute("viewBox");
      const positions = nodePositions(document);
      const selected = document.querySelector(".graph-node.selected");
      const count = messages.length;
      if (!altKey) document.getElementById("zoom-area").click();
      pointer(target, "pointerdown", x, y, { altKey });
      assert.equal(graph.hasPointerCapture(1), true);
      assert.ok(document.querySelector(".selection-rectangle"));
      if (outcome !== "click") {
        pointer(graph, "pointermove", x + 40, y + 40);
      }
      if (outcome === "cancel") {
        graph.dispatchEvent(
          new window.KeyboardEvent("keydown", { key: "Escape" }),
        );
      }
      pointer(
        graph,
        "pointerup",
        x + (outcome === "click" ? 0 : 40),
        y + (outcome === "click" ? 0 : 40),
      );
      target.dispatchEvent(
        new window.MouseEvent("click", { bubbles: true, detail: 1 }),
      );
      assert.equal(messages.length, count);
      assert.equal(graph.getAttribute("aria-busy"), "false");
      assert.equal(document.querySelector(".selection-rectangle"), null);
      assert.equal(graph.hasPointerCapture(1), false);
      assert.equal(
        document.getElementById("zoom-area").getAttribute("aria-pressed"),
        "false",
      );
      assert.equal(document.querySelector(".graph-node.selected"), selected);
      assert.deepEqual(nodePositions(document), positions);
      if (outcome === "zoom") {
        assert.ok(
          Number(graph.getAttribute("viewBox").split(" ")[3]) <
            Number(initial.split(" ")[3]),
        );
        pointer(target, "pointerdown", x, y);
        pointer(graph, "pointerup", x, y);
        target.dispatchEvent(
          new window.MouseEvent("click", { bubbles: true, detail: 1 }),
        );
      } else {
        assert.equal(graph.getAttribute("viewBox"), initial);
        if (outcome === "cancel") {
          link.dispatchEvent(
            new window.KeyboardEvent("keydown", {
              key: "Enter",
              bubbles: true,
            }),
          );
        } else {
          target.dispatchEvent(
            new window.MouseEvent("click", { bubbles: true }),
          );
        }
      }
      assert.equal(messages.length, count + 1);
      assert.equal(messages.at(-1).type, "navigate");
    }
  }
});

test("keyboard zoom shortcuts match controls and stay scoped to the graph canvas", (t) => {
  const { document, window, load } = openViewer(t);
  const graph = document.getElementById("graph");
  const key = (target, value, modifiers = {}) => {
    const event = new window.KeyboardEvent("keydown", {
      key: value,
      bubbles: true,
      cancelable: true,
      ...modifiers,
    });
    target.dispatchEvent(event);
    return event;
  };
  assert.equal(key(graph, "r").defaultPrevented, false);
  load(chainGraph());
  const initial = graph.getAttribute("viewBox");
  const node = document.querySelector(".graph-node");
  assert.equal(key(node, "+").defaultPrevented, true);
  const zoomed = graph.getAttribute("viewBox");
  assert.notEqual(zoomed, initial);
  key(graph, "-");
  assert.equal(graph.getAttribute("viewBox"), initial);
  key(graph, "=");
  assert.equal(graph.getAttribute("viewBox"), zoomed);
  key(document.getElementById("zoom-fit"), "F");
  assert.equal(graph.getAttribute("viewBox"), initial);
  key(node, "R");
  const readable = graph.getAttribute("viewBox");
  assert.equal(Number(readable.split(" ")[2]), 1060);
  document.getElementById("zoom-fit").click();
  document.getElementById("zoom-readable").click();
  assert.equal(graph.getAttribute("viewBox"), readable);
  for (const target of [
    document.getElementById("model-tab"),
    document.querySelector("aside"),
    document.getElementById("operator-doc"),
    document.getElementById("node-search"),
    document.querySelector("#find-node button"),
  ]) {
    for (const shortcut of ["f", "r", "+", "-"]) {
      assert.equal(key(target, shortcut).defaultPrevented, false);
      assert.equal(graph.getAttribute("viewBox"), readable);
    }
  }
  for (const modifiers of [
    { ctrlKey: true },
    { metaKey: true },
    { altKey: true },
  ]) {
    assert.equal(key(graph, "f", modifiers).defaultPrevented, false);
    assert.equal(graph.getAttribute("viewBox"), readable);
  }
});

test("Top/Bottom buttons align model boundaries without changing zoom or node positions", (t) => {
  const { document, window, load, resize } = openViewer(t);
  const controls = document.querySelector(".zoom-controls");
  const style = window.getComputedStyle(controls);
  assert.equal(style.left, "18px");
  assert.equal(style.bottom, "18px");
  assert.equal(style.display, "grid");
  assert.deepEqual(
    [...controls.children].map((button) => [
      button.id,
      window.getComputedStyle(button).gridColumn,
    ]),
    [
      ["go-top", "1"],
      ["go-bottom", "2"],
      ["zoom-in", "1"],
      ["zoom-out", "2"],
      ["zoom-fit", "1"],
      ["zoom-readable", "2"],
      ["zoom-area", "1 / -1"],
    ],
  );
  const graph = document.getElementById("graph");
  const top = document.getElementById("go-top");
  const bottom = document.getElementById("go-bottom");
  assert.equal(top.disabled, true);
  assert.equal(bottom.disabled, true);
  top.click();
  bottom.click();
  assert.equal(graph.hasAttribute("viewBox"), false);
  load(chainGraph(40));
  assert.equal(top.disabled, false);
  assert.equal(bottom.disabled, false);
  assert.equal(top.getAttribute("aria-keyshortcuts"), "Control+Home Meta+Home");
  assert.equal(
    bottom.getAttribute("aria-keyshortcuts"),
    "Control+End Meta+End",
  );
  const view = () => graph.getAttribute("viewBox").split(" ").map(Number);
  document.getElementById("zoom-fit").click();
  const [, modelTop, , modelHeight] = view();
  const positions = nodePositions(document);
  document.getElementById("zoom-readable").click();
  const size = view().slice(2);
  const observer = new window.MutationObserver(() => {});
  observer.observe(graph, { attributes: true, attributeFilter: ["viewBox"] });
  t.after(() => observer.disconnect());
  for (const width of [1060, 600]) {
    resize(width, 350);
    observer.takeRecords();
    const currentSize = view().slice(2);
    top.click();
    assert.equal(observer.takeRecords().length, 1);
    assert.ok(Math.abs(view()[1] - modelTop) < 1e-8);
    assert.deepEqual(view().slice(2), currentSize);
    bottom.click();
    assert.equal(observer.takeRecords().length, 1);
    assert.ok(Math.abs(view()[1] + view()[3] - modelTop - modelHeight) < 1e-8);
    assert.deepEqual(view().slice(2), currentSize);
  }
  assert.equal(view()[3], size[1]);
  assert.deepEqual(nodePositions(document), positions);
  document.getElementById("zoom-fit").click();
  const fitted = graph.getAttribute("viewBox");
  top.click();
  assert.equal(graph.getAttribute("viewBox"), fitted);
  bottom.click();
  assert.equal(graph.getAttribute("viewBox"), fitted);
});

test("model boundary shortcuts match buttons and do not intercept editing or other tabs", (t) => {
  const { document, window, load } = openViewer(t);
  const graph = document.getElementById("graph");
  const key = (target, value, modifiers = {}) => {
    const event = new window.KeyboardEvent("keydown", {
      key: value,
      bubbles: true,
      cancelable: true,
      ...modifiers,
    });
    target.dispatchEvent(event);
    return event;
  };
  assert.equal(key(graph, "Home", { ctrlKey: true }).defaultPrevented, false);
  load(chainGraph(40));
  document.getElementById("zoom-readable").click();
  document.getElementById("go-top").click();
  const top = graph.getAttribute("viewBox");
  document.getElementById("go-bottom").click();
  const bottom = graph.getAttribute("viewBox");
  assert.notEqual(top, bottom);
  const editable = document.createElement("div");
  const child = document.createElement("span");
  editable.append(child);
  document.getElementById("canvas").append(editable);
  for (const value of ["true", "", "plaintext-only"]) {
    editable.setAttribute("contenteditable", value);
    assert.equal(key(child, "Home", { ctrlKey: true }).defaultPrevented, false);
    assert.equal(graph.getAttribute("viewBox"), bottom);
  }
  editable.setAttribute("contenteditable", "false");
  assert.equal(key(child, "Home", { ctrlKey: true }).defaultPrevented, true);
  assert.equal(graph.getAttribute("viewBox"), top);
  document.getElementById("go-bottom").click();
  for (const modifiers of [{ ctrlKey: true }, { metaKey: true }]) {
    for (const target of [
      graph,
      document.querySelector(".graph-node"),
      document.getElementById("zoom-fit"),
    ]) {
      assert.equal(key(target, "Home", modifiers).defaultPrevented, true);
      assert.equal(graph.getAttribute("viewBox"), top);
      assert.equal(key(target, "End", modifiers).defaultPrevented, true);
      assert.equal(graph.getAttribute("viewBox"), bottom);
    }
    for (const target of [
      document.getElementById("node-search"),
      document.getElementById("open-file"),
      document.getElementById("operator-doc"),
      document.getElementById("controls-view"),
    ]) {
      for (const shortcut of ["Home", "End"]) {
        assert.equal(key(target, shortcut, modifiers).defaultPrevented, false);
        assert.equal(graph.getAttribute("viewBox"), bottom);
      }
    }
  }
  for (const modifiers of [
    {},
    { altKey: true },
    { ctrlKey: true, altKey: true },
  ]) {
    for (const shortcut of ["Home", "End"]) {
      assert.equal(key(graph, shortcut, modifiers).defaultPrevented, false);
      assert.equal(graph.getAttribute("viewBox"), bottom);
    }
  }
});

test("model boundary navigation includes nodes moved outside the layout at the same zoom", (t) => {
  const { document, load, pointer } = openViewer(t);
  load(chainGraph());
  const graph = document.getElementById("graph");
  document.getElementById("zoom-readable").click();
  const view = () => graph.getAttribute("viewBox").split(" ").map(Number);
  const size = view().slice(2);
  const node = document.querySelector(".graph-node");
  const start = nodePositions(document)[0];
  for (const y of [-900, 3000]) {
    const current = nodePositions(document)[0];
    pointer(node, "pointerdown", current.x, current.y);
    pointer(graph, "pointermove", start.x, y);
    pointer(graph, "pointerup", start.x, y);
    document.getElementById(y < 0 ? "go-top" : "go-bottom").click();
    const [, top, , height] = view();
    const rect = node.querySelector("rect");
    const halfHeight = Number(rect.getAttribute("height")) / 2;
    const edge = y + (y < 0 ? -halfHeight - 20 : halfHeight + 20);
    assert.ok(Math.abs((y < 0 ? top : top + height) - edge) < 1e-8);
    view()
      .slice(2)
      .forEach((value, index) =>
        assert.ok(Math.abs(value - size[index]) < 1e-8),
      );
  }
});

test("Ctrl+wheel keeps the graph point under the pointer fixed while zooming in and out", (t) => {
  const { document, window, load } = openViewer(t);
  load(chainGraph());
  const graph = document.getElementById("graph");
  const view = () => graph.getAttribute("viewBox").split(" ").map(Number);
  const point = () => {
    const [x, y, width, height] = view();
    return [x + width * 0.75, y + height * 0.25];
  };
  const initialWidth = view()[2];
  const anchor = point();
  for (const deltaY of [-100, 100, 100, -100]) {
    graph.dispatchEvent(
      new window.WheelEvent("wheel", {
        ctrlKey: true,
        deltaY,
        clientX: 1060 * 0.75,
        clientY: 350 * 0.25,
      }),
    );
    point().forEach((value, index) =>
      assert.ok(Math.abs(value - anchor[index]) < 1e-9),
    );
  }
  assert.ok(Math.abs(view()[2] - initialWidth) < 1e-9);
});

test("zooms large graphs to a readable scale beyond four times Fit with buttons and Ctrl+wheel", (t) => {
  const { document, window, load } = openViewer(t);
  load(chainGraph(200));
  const graph = document.getElementById("graph");
  const initial = graph.getAttribute("viewBox");
  const positions = nodePositions(document);
  for (const zoomIn of [
    () => document.getElementById("zoom-in").click(),
    () =>
      graph.dispatchEvent(
        new window.WheelEvent("wheel", { deltaY: -100, ctrlKey: true }),
      ),
  ]) {
    for (let step = 0; step < 30; step++) zoomIn();
    const viewBox = graph.getAttribute("viewBox").split(" ").map(Number);
    assert.ok(viewBox.every(Number.isFinite));
    assert.ok((19 * 350) / viewBox[3] >= 19);
    document.getElementById("zoom-out").click();
    assert.ok(
      Math.abs(
        Number(graph.getAttribute("viewBox").split(" ")[3]) / viewBox[3] - 1.25,
      ) < 1e-9,
    );
    document.getElementById("zoom-fit").click();
    assert.equal(graph.getAttribute("viewBox"), initial);
    assert.deepEqual(nodePositions(document), positions);
  }
});

test("keeps operator types visible in regular and virtualized overviews", (t) => {
  const { document, window, load, pointer } = openViewer(t);
  const graph = document.getElementById("graph");
  const overview = document.getElementById("graph-overview");
  const labels = document.getElementById("overview-labels");
  for (const length of [6, 250]) {
    const model = chainGraph(length);
    model.nodes[0].opType = "<CustomOperator>";
    load(model);
    document.getElementById("zoom-readable").click();
    const group = document.querySelector('[data-id="node-0"]');
    const type = group.querySelector(".operator-label");
    const name = group.querySelector(".operator-name");
    const position = {
      x: Number(type.getAttribute("x")),
      y: Number(type.getAttribute("y")),
    };
    pointer(type, "pointerdown", position.x, position.y);
    pointer(graph, "pointermove", position.x + 20, position.y + 10);
    pointer(graph, "pointerup", position.x + 20, position.y + 10);
    assert.notEqual(window.getComputedStyle(name).display, "none");
    document.getElementById("zoom-fit").click();
    assert.equal(graph.classList.contains("overview"), true);
    const types =
      length === 6
        ? document.querySelectorAll(".graph-node .operator-label")
        : labels.querySelectorAll(".operator-label");
    assert.equal(types.length, length);
    assert.equal(types[0].textContent, "<CustomOperator>");
    assert.equal(types[0].children.length, 0);
    const dx = length === 6 ? 0 : 20;
    const dy = length === 6 ? 0 : 10;
    assert.ok(
      Math.abs(Number(types[0].getAttribute("x")) - position.x - dx) < 1e-9,
    );
    assert.ok(
      Math.abs(Number(types[0].getAttribute("y")) - position.y - dy) < 1e-9,
    );
    assert.equal(types[0].style.fontSize, type.style.fontSize);
    for (const text of types)
      assert.notEqual(window.getComputedStyle(text).display, "none");
    if (length === 250) {
      assert.equal(overview.getAttribute("display"), "inline");
      assert.equal(document.querySelectorAll(".graph-node").length, 0);
      assert.equal(name.isConnected, false);
    } else assert.equal(window.getComputedStyle(name).display, "none");
    document.getElementById("zoom-in").click();
    assert.equal(graph.classList.contains("overview"), true);
    if (length === 250) assert.equal(labels.firstChild, types[0]);
    document.getElementById("zoom-readable").click();
    assert.equal(graph.classList.contains("overview"), false);
    assert.equal(overview.getAttribute("display"), "none");
    assert.notEqual(window.getComputedStyle(name).display, "none");
  }
  load(demoGraph);
  assert.equal(labels.children.length, 0);
});

test("keeps graph outlines visible when zooming out in regular and virtualized overviews", (t) => {
  const { document, window, load } = openViewer(t);
  const style = document.createElement("style");
  style.textContent = readFileSync(
    require.resolve("../media/viewer.css"),
    "utf8",
  );
  document.head.append(style);
  const graph = document.getElementById("graph");
  for (const length of [6, 2000]) {
    load(chainGraph(length));
    document.getElementById("zoom-readable").click();
    const node = document.querySelector(".graph-node:not(.selected) > rect");
    assert.notEqual(
      window.getComputedStyle(node).getPropertyValue("vector-effect"),
      "non-scaling-stroke",
    );
    for (const zoomOut of [
      () => document.getElementById("zoom-out").click(),
      () =>
        graph.dispatchEvent(
          new window.WheelEvent("wheel", { deltaY: 100, ctrlKey: true }),
        ),
    ]) {
      for (let step = 0; step < 50; step++) zoomOut();
      assert.equal(graph.classList.contains("overview"), true);
      const outlines =
        length === 6
          ? document.querySelectorAll(".graph-node rect, .port rect, .edge")
          : document.querySelectorAll("#graph-overview path");
      assert.ok(outlines.length > 0);
      for (const outline of outlines) {
        assert.equal(
          window.getComputedStyle(outline).getPropertyValue("vector-effect"),
          "non-scaling-stroke",
        );
      }
      assert.ok(
        graph
          .getAttribute("viewBox")
          .split(" ")
          .map(Number)
          .every(Number.isFinite),
      );
      document.getElementById("zoom-readable").click();
    }
  }
});

test("jumps to readable zoom and culls offscreen SVG content without rebuilding a large graph", (t) => {
  const { document, window, load } = openViewer(t);
  const model = chainGraph(200);
  model.nodes.at(-1).inputs.push("tensor_0");
  load(model);
  const graph = document.getElementById("graph");
  const initial = graph.getAttribute("viewBox");
  const positions = nodePositions(document);
  const nodes = [...document.querySelectorAll(".graph-node")];
  const edges = [...document.querySelectorAll(".edge")];
  const crossingEdge = edges.at(-2).parentElement;
  assert.equal(graph.classList.contains("overview"), true);
  document.getElementById("zoom-readable").click();
  const [x, y, width, height] = graph
    .getAttribute("viewBox")
    .split(" ")
    .map(Number);
  assert.equal(width, 1060);
  assert.equal(height, 350);
  assert.ok(Math.abs(x + width / 2 - positions[0].x) < 1e-9);
  assert.ok(Math.abs(y + height / 2 - positions[0].y) < 1e-9);
  assert.equal(graph.classList.contains("overview"), false);
  assert.equal(nodes[0].getAttribute("display"), "inline");
  assert.equal(nodes.at(-1).getAttribute("display"), "none");
  assert.ok(
    nodes.filter((node) => node.getAttribute("display") !== "none").length < 10,
  );
  assert.ok(
    edges.filter(
      (edge) => edge.parentElement.getAttribute("display") !== "none",
    ).length < 10,
  );
  const observer = new window.MutationObserver(() => {});
  observer.observe(graph, { subtree: true, attributes: true });
  document.getElementById("zoom-in").click();
  const changes = observer.takeRecords();
  observer.disconnect();
  assert.ok(changes.length < 30);
  assert.ok(
    changes.every(
      ({ target, attributeName }) =>
        target === graph || attributeName === "display",
    ),
  );
  graph.dispatchEvent(
    new window.WheelEvent("wheel", { deltaY: 10000, cancelable: true }),
  );
  assert.equal(nodes[0].getAttribute("display"), "none");
  assert.equal(nodes.at(-1).getAttribute("display"), "none");
  assert.equal(crossingEdge.getAttribute("display"), "inline");
  assert.ok(nodes.some((node) => node.getAttribute("display") === "inline"));
  document.getElementById("zoom-fit").click();
  assert.equal(graph.getAttribute("viewBox"), initial);
  assert.equal(graph.classList.contains("overview"), true);
  assert.ok(nodes.every((node) => node.getAttribute("display") === "inline"));
  assert.deepEqual([...document.querySelectorAll(".graph-node")], nodes);
  assert.deepEqual(nodePositions(document), positions);
});

test("suggests matching nodes in the sidebar and selects individual results", (t) => {
  const { document, window, load } = openViewer(t);
  const input = document.getElementById("node-search");
  const suggestions = document.getElementById("node-search-suggestions");
  const status = document.getElementById("node-search-result");
  const form = document.getElementById("find-node");
  const search = (value) => {
    input.value = value;
    input.dispatchEvent(new window.Event("input", { bubbles: true }));
  };
  assert.ok(form.closest("aside"));
  assert.equal(form.closest("#canvas"), null);
  assert.equal(suggestions.hidden, true);
  search("relu");
  assert.equal(status.textContent, "No matching node");
  const model = chainGraph(3);
  model.nodes[1].name = "<img src=x onerror=alert(1)>";
  model.nodes[2].name = model.nodes[0].name;
  load(model);
  search(" ReLU ");
  assert.equal(suggestions.hidden, false);
  assert.equal(suggestions.children.length, 3);
  assert.equal(status.textContent, "3 matching nodes");
  assert.equal(
    suggestions.children[1].textContent,
    "<img src=x onerror=alert(1)> — Relu",
  );
  assert.equal(suggestions.querySelector("img"), null);
  assert.ok(
    [...suggestions.querySelectorAll("button")].every(
      (button) => button.type === "button" && button.tabIndex === 0,
    ),
  );
  suggestions.children[2].querySelector("button").click();
  assert.equal(
    document.querySelector(".graph-node.selected").getAttribute("data-id"),
    "node-2",
  );
  assert.equal(status.textContent, "3 / 3");
  assert.equal(suggestions.hidden, true);
  assert.equal(input.value, " ReLU ");
  input.click();
  assert.equal(suggestions.hidden, false);
  assert.equal(suggestions.children.length, 3);
  assert.equal(
    Number(
      document.getElementById("graph").getAttribute("viewBox").split(" ")[2],
    ),
    1060,
  );
  form.dispatchEvent(new window.Event("submit", { cancelable: true }));
  assert.equal(
    document.querySelector(".graph-node.selected").getAttribute("data-id"),
    "node-0",
  );
  assert.equal(status.textContent, "1 / 3");
  assert.equal(suggestions.hidden, true);
  input.focus();
  assert.equal(suggestions.hidden, true);
  input.click();
  assert.equal(suggestions.hidden, false);
  assert.equal(suggestions.children.length, 3);
  form.dispatchEvent(new window.Event("submit", { cancelable: true }));
  assert.equal(status.textContent, "2 / 3");
  assert.equal(suggestions.hidden, true);
  search("ACTIVATION_0");
  assert.equal(suggestions.hidden, false);
  assert.equal(suggestions.children.length, 2);
  search("does-not-exist");
  assert.equal(suggestions.hidden, true);
  assert.equal(suggestions.children.length, 0);
  assert.equal(status.textContent, "No matching node");
  search("   ");
  assert.equal(suggestions.hidden, true);
  assert.equal(status.textContent, "");
  form.dispatchEvent(new window.Event("submit", { cancelable: true }));
  assert.equal(status.textContent, "Enter a name or operator");
});

test("rebuilds cached search labels and matches when the graph changes", (t) => {
  const { document, window, load } = openViewer(t);
  const input = document.getElementById("node-search");
  const suggestions = document.getElementById("node-search-suggestions");
  const search = () => {
    input.value = "ALPHA";
    input.dispatchEvent(new window.Event("input", { bubbles: true }));
  };
  const initial = chainGraph(2);
  initial.nodes[0].name = "Alpha original";
  load(initial);
  search();
  assert.equal(suggestions.children.length, 1);
  assert.equal(suggestions.textContent, "Alpha original — Relu");
  suggestions.querySelector("button").click();
  assert.equal(
    document.querySelector(".graph-node.selected").dataset.id,
    "node-0",
  );
  const replacement = chainGraph(2);
  replacement.nodes[1].name = "alpha replacement";
  load(replacement);
  search();
  assert.equal(suggestions.children.length, 1);
  assert.equal(suggestions.textContent, "alpha replacement — Relu");
  suggestions.querySelector("button").click();
  assert.equal(
    document.querySelector(".graph-node.selected").dataset.id,
    "node-1",
  );
});

test("bounds node suggestions, finds offscreen matches and clears stale results on graph replacement", (t) => {
  const { document, window, load } = openViewer(t);
  const input = document.getElementById("node-search");
  const suggestions = document.getElementById("node-search-suggestions");
  const search = (value) => {
    input.value = value;
    input.dispatchEvent(new window.Event("input", { bubbles: true }));
  };
  load(chainGraph(1200));
  search("relu");
  assert.equal(suggestions.children.length, 20);
  assert.equal(
    document.getElementById("node-search-result").textContent,
    "1200 matching nodes (first 20 shown)",
  );
  search("activation_1199");
  assert.equal(suggestions.children.length, 1);
  suggestions.querySelector("button").click();
  assert.equal(
    document.getElementById("node-name").textContent,
    "activation_1199",
  );
  assert.equal(
    document.querySelector(".graph-node.selected").getAttribute("data-id"),
    "node-1199",
  );
  load({ ...demoGraph, nodes: [] });
  assert.equal(suggestions.children.length, 0);
  assert.equal(suggestions.hidden, true);
  assert.equal(document.getElementById("node-search-result").textContent, "");
  search("relu");
  assert.equal(suggestions.hidden, true);
  assert.equal(
    document.getElementById("node-search-result").textContent,
    "No matching node",
  );
});

test("virtualizes 20,000 nodes, finds distant operators and uses a batched overview", (t) => {
  const { document, window, load } = openViewer(t);
  const model = chainGraph(20_000);
  model.nodes[19_998].parameters = [
    { name: "weight", type: "FLOAT", shape: ["3", "4"] },
  ];
  model.nodes[19_998].attributes = [
    { name: "body", graphs: [{ name: "child", path: [], nodeCount: 0 }] },
  ];
  load(model);
  const graph = document.getElementById("graph");
  const mounted = () => document.querySelectorAll(".graph-node").length;
  assert.ok(mounted() > 0 && mounted() < 20);
  assert.equal(Number(graph.getAttribute("viewBox").split(" ")[3]), 350);
  const first = document.querySelector(".graph-node");
  window.dispatchEvent(
    new window.MessageEvent("message", {
      data: {
        type: "metadata-info",
        metadata: {
          model: [],
          graph: [],
          values: [],
          nodes: [
            {
              id: "node-19998",
              unusedOutputs: ["tensor_19998"],
              releaseAfter: [],
              lastUse: [],
              entries: [],
            },
          ],
        },
      },
    }),
  );
  const search = (value) => {
    document.getElementById("node-search").value = value;
    document
      .getElementById("find-node")
      .dispatchEvent(new window.Event("submit", { cancelable: true }));
  };
  search("activation_19998");
  assert.equal(
    document.getElementById("node-name").textContent,
    "activation_19998",
  );
  assert.equal(
    document.querySelector(".graph-node.selected").getAttribute("data-id"),
    "node-19998",
  );
  assert.match(
    document.querySelector(".graph-node.selected .unused-label").textContent,
    /Unused: 1/,
  );
  const selected = document.querySelector(".graph-node.selected");
  const initializer = selected.querySelector(".node-initializer");
  assert.equal(initializer.getAttribute("aria-label"), "weight: FLOAT [3, 4]");
  assert.ok(
    Number(initializer.getAttribute("y")) <
      Number(selected.querySelector(".subgraph-link rect").getAttribute("y")),
  );
  assert.equal(first.isConnected, false);
  assert.ok(mounted() < 20);
  search("RELU");
  assert.equal(
    document.getElementById("node-name").textContent,
    "activation_19999",
  );
  const current = graph.getAttribute("viewBox");
  search("does-not-exist");
  assert.equal(
    document.getElementById("node-search-result").textContent,
    "No matching node",
  );
  assert.equal(graph.getAttribute("viewBox"), current);
  document.getElementById("zoom-fit").click();
  assert.equal(mounted(), 0);
  assert.equal(document.querySelectorAll(".edge").length, 0);
  assert.equal(
    document.getElementById("graph-overview").getAttribute("display"),
    "inline",
  );
  assert.equal(
    document.getElementById("overview-nodes").getAttribute("d").match(/M/g)
      .length,
    20_002,
  );
  document.getElementById("zoom-readable").click();
  assert.equal(
    document.getElementById("graph-overview").getAttribute("display"),
    "none",
  );
  assert.ok(mounted() > 0 && mounted() < 20);
  assert.equal(
    document.querySelector(".graph-node.selected").getAttribute("data-id"),
    "node-19999",
  );
  window.dispatchEvent(
    new window.MessageEvent("message", {
      data: {
        type: "shape-info",
        tensors: [{ name: "tensor_19998", type: "float32", shape: ["42"] }],
      },
    }),
  );
  search("activation_19998");
  assert.match(
    document.querySelector(".graph-node.selected .node-tensor-info")
      .textContent,
    /float32 \[42\]/,
  );
  document.getElementById("zoom-fit").click();
  load(chainGraph(1500));
  assert.ok(mounted() < 20);
  document.getElementById("zoom-fit").click();
  assert.equal(
    document.getElementById("overview-nodes").getAttribute("d").match(/M/g)
      .length,
    1502,
  );
});

test("batches high-fanout connections without creating thousands of offscreen labels", async (t) => {
  const { document, window, load } = openViewer(t);
  const model = chainGraph(2000);
  for (const node of model.nodes.slice(1)) node.inputs.push("input");
  load(model);
  const crossing = document.getElementById("crossing-edges");
  assert.ok(crossing.getAttribute("d").match(/M/g).length > 1900);
  assert.ok(document.querySelectorAll(".edge").length < 30);
  assert.ok(document.querySelectorAll(".tensor-info").length < 60);
  document
    .getElementById("graph")
    .dispatchEvent(
      new window.WheelEvent("wheel", { deltaY: 40000, cancelable: true }),
    );
  await new Promise(window.requestAnimationFrame);
  assert.ok(document.querySelectorAll(".edge").length < 40);
  assert.ok(crossing.getAttribute("d").length > 0);
  document.getElementById("zoom-fit").click();
  assert.equal(crossing.getAttribute("d"), "");
  document.getElementById("zoom-readable").click();
  assert.ok(document.querySelectorAll(".graph-node").length < 20);
});

test("keeps positions correct when selected offscreen nodes move before their SVG is created", (t) => {
  const { document, window, load, pointer } = openViewer(t);
  load(chainGraph(2000));
  const graph = document.getElementById("graph");
  const positions = nodePositions(document);
  const step = positions[1].y - positions[0].y;
  graph.dispatchEvent(
    new window.KeyboardEvent("keydown", {
      key: "a",
      ctrlKey: true,
      bubbles: true,
      cancelable: true,
    }),
  );
  const first = document.querySelector(".graph-node");
  pointer(first, "pointerdown", positions[0].x, positions[0].y);
  pointer(graph, "pointermove", positions[0].x + 80, positions[0].y + 50);
  pointer(graph, "pointerup", positions[0].x + 80, positions[0].y + 50);
  document.getElementById("node-search").value = "activation_700";
  document
    .getElementById("find-node")
    .dispatchEvent(new window.Event("submit", { cancelable: true }));
  const target = document.querySelector(".graph-node.selected");
  const [x, y] = translationOffsets(target.getAttribute("transform"));
  assert.ok(Math.abs(x - 80) < 1e-9);
  assert.ok(Math.abs(y - 50) < 1e-9);
  const actual = nodePositions(document, ".graph-node.selected > rect")[0];
  assert.equal(actual.y, positions[0].y + 700 * step + 50);
});

test("defers readable framing until a large model loaded in a hidden tab has a viewport", (t) => {
  const { document, load, resize } = openViewer(t);
  resize(0, 0);
  load(chainGraph(2000));
  assert.equal(document.querySelectorAll(".graph-node").length, 0);
  resize(1060, 350);
  assert.equal(
    Number(
      document.getElementById("graph").getAttribute("viewBox").split(" ")[3],
    ),
    350,
  );
  assert.ok(document.querySelectorAll(".graph-node").length < 20);
  assert.equal(
    document.querySelector(".graph-node.selected").getAttribute("data-id"),
    "node-0",
  );
});

test("virtualizes models from 250 total nodes while preserving navigation and cached elements", (t) => {
  const { document, load } = openViewer(t);
  const graph = document.getElementById("graph");
  const mounted = () => document.querySelectorAll(".graph-node").length;
  load(chainGraph(247));
  assert.equal(mounted(), 247);
  for (const length of [248, 249, 600]) {
    load(chainGraph(length));
    assert.ok(mounted() > 0 && mounted() < 20);
    assert.equal(Number(graph.getAttribute("viewBox").split(" ")[3]), 350);
    const first = document.querySelector(".graph-node");
    document.getElementById("go-bottom").click();
    assert.equal(first.isConnected, false);
    assert.ok(mounted() > 0 && mounted() < 20);
    document.getElementById("go-top").click();
    assert.equal(first.isConnected, true);
    document.getElementById("zoom-fit").click();
    assert.equal(mounted(), 0);
    assert.equal(
      document.getElementById("overview-nodes").getAttribute("d").match(/M/g)
        .length,
      length + 2,
    );
    document.getElementById("zoom-readable").click();
    assert.ok(mounted() > 0 && mounted() < 20);
  }
});

test("coalesces large-model wheel events without losing motion or leaving stale frames", (t) => {
  const frames = new Map();
  let nextFrame = 0;
  const { document, window, load } = openViewer(t, "model", (window) => {
    window.requestAnimationFrame = (callback) => {
      const id = nextFrame++;
      frames.set(id, callback);
      return id;
    };
    window.cancelAnimationFrame = (id) => frames.delete(id);
  });
  const flushFrame = () => {
    const callbacks = [...frames.values()];
    frames.clear();
    for (const callback of callbacks) callback();
  };
  load(chainGraph(2000));
  const graph = document.getElementById("graph");
  const view = () => graph.getAttribute("viewBox").split(" ").map(Number);
  const wheel = (options) =>
    graph.dispatchEvent(
      new window.WheelEvent("wheel", { cancelable: true, ...options }),
    );
  const before = view();
  const observer = new window.MutationObserver(() => {});
  observer.observe(graph, { attributes: true, attributeFilter: ["viewBox"] });
  for (const options of [
    { deltaY: 20 },
    { deltaY: 2, deltaMode: 1 },
    { deltaY: 5, shiftKey: true },
    { deltaX: 7 },
  ]) {
    wheel(options);
  }
  assert.equal(frames.size, 1);
  assert.deepEqual(view(), before);
  flushFrame();
  assert.equal(observer.takeRecords().length, 1);
  observer.disconnect();
  const after = view();
  assert.ok(Math.abs(after[0] - before[0] - 12) < 1e-9);
  assert.ok(Math.abs(after[1] - before[1] - 52) < 1e-9);
  assert.deepEqual(after.slice(2), before.slice(2));
  assert.equal(
    document.getElementById("node-name").textContent,
    "activation_0",
  );

  wheel({ deltaY: 40 });
  assert.equal(frames.size, 1);
  wheel({ ctrlKey: true, deltaY: -100, clientX: 530, clientY: 175 });
  assert.equal(frames.size, 0);
  const zoomed = view();
  assert.ok(Math.abs(zoomed[2] - after[2] / 1.25) < 1e-9);
  assert.ok(
    Math.abs(zoomed[0] + zoomed[2] / 2 - after[0] - after[2] / 2) < 1e-9,
  );
  assert.ok(
    Math.abs(zoomed[1] + zoomed[3] / 2 - after[1] - after[3] / 2 - 40) < 1e-9,
  );
  wheel({ deltaY: 30 });
  document.getElementById("zoom-fit").click();
  assert.equal(frames.size, 0);
  assert.equal(document.querySelectorAll(".graph-node").length, 0);
  wheel({ deltaY: 30 });
  assert.equal(frames.size, 1);
  load(demoGraph);
  assert.equal(frames.size, 0);
  const replacement = view();
  flushFrame();
  assert.deepEqual(view(), replacement);
  assert.equal(
    document.querySelectorAll(".graph-node").length,
    demoGraph.nodes.length,
  );
});

test("wheel scrolling pans vertically and Shift+wheel pans horizontally without zooming", (t) => {
  const { document, window, load, resize } = openViewer(t);
  load(chainGraph());
  resize(900, 600);
  const graph = document.getElementById("graph");
  const positions = nodePositions(document);
  const viewBox = () => graph.getAttribute("viewBox").split(" ").map(Number);
  for (const [options, dx, dy] of [
    [{ deltaY: 80 }, 0, 80],
    [{ deltaY: -80, shiftKey: true }, -80, 0],
    [{ deltaX: -40 }, -40, 0],
    [{ deltaX: 25, deltaY: 100, shiftKey: true }, 25, 0],
    [{ deltaY: 2, deltaMode: 1 }, 0, 32],
    [{ deltaY: 1, deltaMode: 2 }, 0, 600],
    [{ deltaY: 1, deltaMode: 2, shiftKey: true }, 900, 0],
  ]) {
    document.getElementById("zoom-fit").click();
    document.getElementById("zoom-in").click();
    const before = viewBox();
    const event = new window.WheelEvent("wheel", {
      ...options,
      cancelable: true,
    });
    graph.dispatchEvent(event);
    const after = viewBox();
    assert.equal(event.defaultPrevented, true);
    assert.deepEqual(after.slice(2), before.slice(2));
    assert.ok(Math.abs(after[0] - before[0] - (dx * before[2]) / 900) < 1e-9);
    assert.ok(Math.abs(after[1] - before[1] - (dy * before[3]) / 600) < 1e-9);
  }
  const before = graph.getAttribute("viewBox");
  const empty = new window.WheelEvent("wheel", { cancelable: true });
  graph.dispatchEvent(empty);
  assert.equal(empty.defaultPrevented, false);
  assert.equal(graph.getAttribute("viewBox"), before);
  assert.deepEqual(nodePositions(document), positions);
});

test("panning survives zoom and resize, supports dragging, and resets on Fit or model load", (t) => {
  const { document, window, load, resize, pointer } = openViewer(t);
  load(chainGraph());
  resize(900, 600);
  const graph = document.getElementById("graph");
  const viewBox = () => graph.getAttribute("viewBox").split(" ").map(Number);
  const center = () => {
    const [x, y, width, height] = viewBox();
    return [x + width / 2, y + height / 2];
  };
  const initialCenter = center();
  document.getElementById("zoom-in").click();
  graph.dispatchEvent(
    new window.WheelEvent("wheel", { deltaY: 80, shiftKey: true }),
  );
  graph.dispatchEvent(new window.WheelEvent("wheel", { deltaY: 40 }));
  const pannedCenter = center();
  assert.notDeepEqual(pannedCenter, initialCenter);
  document.getElementById("zoom-in").click();
  resize(600, 900);
  center().forEach((value, index) =>
    assert.ok(Math.abs(value - pannedCenter[index]) < 1e-9),
  );
  const before = nodePositions(document);
  pointer(
    document.querySelector(".graph-node"),
    "pointerdown",
    before[0].x,
    before[0].y,
  );
  pointer(graph, "pointerup", 250, before[0].y + 20);
  assertMoved(before, nodePositions(document), [0], 250 - before[0].x, 20);
  document.getElementById("zoom-fit").click();
  center().forEach((value, index) =>
    assert.ok(Math.abs(value - initialCenter[index]) < 1e-9),
  );
  const fitted = graph.getAttribute("viewBox");
  graph.dispatchEvent(
    new window.WheelEvent("wheel", {
      deltaY: -100,
      ctrlKey: true,
      shiftKey: true,
    }),
  );
  assert.ok(viewBox()[2] < Number(fitted.split(" ")[2]));
  graph.dispatchEvent(
    new window.WheelEvent("wheel", { deltaY: 100, shiftKey: true }),
  );
  load(chainGraph());
  assert.equal(graph.getAttribute("viewBox"), fitted);
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

for (const modifier of ["ctrlKey", "metaKey"]) {
  test(`${modifier} clicks add and remove operators and ports with persistent high-contrast selection`, (t) => {
    const { document, load, pointer, window } = openViewer(t);
    load(chainGraph());
    const graph = document.getElementById("graph");
    graph.dispatchEvent(new window.KeyboardEvent("keydown", { key: "Escape" }));
    const targets = [
      document.querySelector(".graph-node"),
      document.querySelector(".port:not(.output)"),
      document.querySelector(".port.output"),
    ];
    const initial = targets.map((target) => {
      const style = window.getComputedStyle(
        target.querySelector(":scope > rect"),
      );
      return { fill: style.fill, stroke: style.stroke };
    });
    const click = (target) => {
      pointer(target, "pointerdown", 0, 0, { [modifier]: true });
      pointer(graph, "pointerup", 0, 0, { [modifier]: true });
      target.dispatchEvent(
        new window.MouseEvent("click", {
          bubbles: true,
          detail: 1,
          [modifier]: true,
        }),
      );
    };
    targets.forEach((target, index) => {
      click(target);
      assert.equal(target.getAttribute("aria-pressed"), "true");
      const style = window.getComputedStyle(
        target.querySelector(":scope > rect"),
      );
      assert.notEqual(style.fill, initial[index].fill);
      assert.notEqual(style.stroke, initial[index].stroke);
      assert.ok(parseFloat(style.strokeWidth) >= 3);
      assert.equal(
        style.getPropertyValue("vector-effect"),
        "non-scaling-stroke",
      );
      assert.equal(document.querySelectorAll(".selected").length, index + 1);
    });
    const selectionStyle = window.getComputedStyle(
      targets[0].querySelector(":scope > rect"),
    );
    const selectionFill = selectionStyle.fill;
    const selectionStroke = selectionStyle.stroke;
    let remaining = targets.length;
    for (const index of [1, 0, 2]) {
      click(targets[index]);
      assert.equal(targets[index].getAttribute("aria-pressed"), "false");
      const style = window.getComputedStyle(
        targets[index].querySelector(":scope > rect"),
      );
      assert.notEqual(style.fill, selectionFill);
      assert.notEqual(style.stroke, selectionStroke);
      assert.equal(document.querySelectorAll(".selected").length, --remaining);
    }
    assert.equal(document.querySelectorAll(".selected").length, 0);
  });
}

test("rectangle selection supports reverse and additive gestures and moves together", (t) => {
  const { document, load, pointer, window } = openViewer(t);
  load(chainGraph());
  const graph = document.getElementById("graph");
  const positions = nodePositions(document);
  const selectionBottom = (positions[1].y + positions[2].y) / 2;
  pointer(graph, "pointerdown", 500, selectionBottom);
  pointer(graph, "pointermove", -10, 110);
  assert.equal(document.querySelectorAll(".graph-node.selected").length, 2);
  assert.ok(document.querySelector(".selection-rectangle"));
  pointer(graph, "pointerup", -10, 110);
  assert.equal(document.querySelector(".selection-rectangle"), null);
  const targetY = nodePositions(document)[4].y;
  pointer(graph, "pointerdown", -10, targetY - 20, { ctrlKey: true });
  pointer(graph, "pointermove", 500, targetY + 20, { ctrlKey: true });
  pointer(graph, "pointerup", 500, targetY + 20, { ctrlKey: true });
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
