/**
 * Builds the static shell; model-derived text arrives later through messages.
 * The DOM IDs are the binding points used by media/viewer.js.
 * Inputs are trusted host-generated values interpolated without HTML escaping;
 * never supply model text here. The CSP blocks other content by default.
 * @param resource Resolves bundled filenames (viewer.css, viewer.js, logo.svg)
 * to webview URI strings within the panel's allowed media resource root.
 * @param cspSource The panel webview's cspSource, allowing local images/styles.
 * @param nonce Fresh per-panel nonce authorizing only the bundled scripts.
 * @param onnxLightVersion C++ release embedded in the compiled native addon.
 * @param initialTab Initially selected viewer tab.
 * @returns Complete HTML document; graph and log data require the ready handshake.
 */
export function webviewHtml(
  resource: (name: string) => string,
  cspSource: string,
  nonce: string,
  onnxLightVersion: string,
  initialTab: "model" | "logs" = "model",
): string {
  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src ${cspSource}; style-src ${cspSource}; script-src 'nonce-${nonce}';">
  <link rel="stylesheet" href="${resource("viewer.css")}">
  <title>ONNX Light Viewer</title>
</head>
<body>
  <main class="workspace" aria-label="ONNX graph explorer">
    <nav class="viewer-tabs" role="tablist" aria-label="Viewer">
      <button id="model-tab" role="tab" aria-controls="model-view" aria-selected="${initialTab === "model"}" tabindex="${initialTab === "model" ? 0 : -1}">Model</button>
      <button id="logs-tab" role="tab" aria-controls="logs-view" aria-selected="${initialTab === "logs"}" tabindex="${initialTab === "logs" ? 0 : -1}">Logs</button>
      <button id="status-tab" role="tab" aria-controls="status-view" aria-selected="false" tabindex="-1">Status</button>
      <button id="controls-tab" role="tab" aria-controls="controls-view" aria-selected="false" tabindex="-1">Controls</button>
      <button id="repositories-tab" role="tab" aria-controls="repositories-view" aria-selected="false" tabindex="-1">Repositories</button>
    </nav>
    <section id="model-view" role="tabpanel" aria-labelledby="model-tab"${initialTab === "logs" ? " hidden" : ""}>
      <header class="graph-toolbar">
        <div><a id="repository-link" href="https://github.com/xadupre/onnx-light-viewer" title="ONNX Light Viewer on GitHub" aria-label="Open the ONNX Light Viewer GitHub repository"><img src="${resource("logo.svg")}" alt="" width="24" height="24"></a><button id="open-model" class="primary">Open ONNX model</button><button id="refresh-shapes" class="icon-button" disabled title="Refresh shape and type information with onnx-light" aria-label="Refresh shape and type information" aria-describedby="shape-status"><svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><path d="M20 11a8 8 0 1 0-2 6M20 4v7h-7"/></svg>Refresh shapes</button><button id="refresh-metadata" class="icon-button" disabled title="Recompute constants, tags and lifetimes for the displayed graph with onnx-light" aria-label="Refresh metadata for the displayed graph" aria-describedby="metadata-status"><svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><path d="M20 11a8 8 0 1 0-2 6M20 4v7h-7"/></svg>Refresh metadata</button><button id="refresh-positions" class="icon-button" disabled title="Reset the portrait layout with new random horizontal positions" aria-label="Refresh node positions"><svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><path d="M20 11a8 8 0 1 0-2 6M20 4v7h-7"/></svg>Refresh positions</button></div>
        <div><button id="show-metadata" class="icon-button" aria-expanded="false" aria-controls="metadata-panel" title="Show native constants, tags, lifetimes and stored metadata" aria-describedby="metadata-status"><span class="toggle-indicator" aria-hidden="true">&#10003;</span>Metadata</button></div>
        <div class="graph-stats" aria-live="polite"><span id="graph-summary">No model loaded</span><span id="initializer-count">0 initializers</span><span id="onnx-light-version" title="onnx-light C++ release used by this viewer">onnx-light ${onnxLightVersion}</span><span id="shape-status" role="status"></span><span id="metadata-status" role="status"></span><span id="navigation-status" role="status"></span><span id="load-status" role="status"></span></div>
        <nav id="graph-navigation" aria-label="Graph breadcrumbs" hidden></nav>
        <progress id="load-progress" aria-label="Loading ONNX model" hidden></progress>
      </header>
      <div class="graph-body">
        <div class="canvas" id="canvas">
          <form id="find-node" role="search"><input id="node-search" type="search" placeholder="Find node or operator" aria-label="Find a node by name or operator" autocomplete="off"><button type="submit" title="Go to the next matching node" aria-label="Go to the next matching node">Go</button><span id="node-search-result" role="status"></span></form>
          <svg id="graph" role="group" aria-label="Interactive ONNX graph" tabindex="0"><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z"/></marker></defs><g id="graph-overview" aria-hidden="true" display="none"><path id="overview-edges"/><path id="overview-nodes"/></g><path id="crossing-edges" class="crossing-edge"/><path id="released-crossing-edges" class="crossing-edge released"/><g id="edges"></g><g id="nodes"></g></svg>
          <div class="zoom-controls" aria-label="Graph zoom"><button id="zoom-out" title="Zoom out (-)" aria-label="Zoom out" aria-keyshortcuts="-">−</button><button id="zoom-fit" title="Fit graph (F)" aria-keyshortcuts="F">Fit</button><button id="zoom-readable" title="Zoom to readable size (100%), centered on the inspected node (R)" aria-label="Zoom to readable size" aria-keyshortcuts="R"><svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><path d="m8 17 4-10 4 10M10 13h4M4 8V4h4M16 4h4v4M20 16v4h-4M8 20H4v-4"/></svg></button><button id="zoom-area" title="Drag a rectangle to zoom into an area; Alt+drag also works. Escape cancels." aria-pressed="false"><span class="toggle-indicator" aria-hidden="true">&#10003;</span> Zoom area</button><button id="zoom-in" title="Zoom in (+)" aria-label="Zoom in" aria-keyshortcuts="+">+</button></div>
        </div>
        <aside aria-label="Node inspector">
          <div class="eyebrow">INSPECTOR</div><div class="inspector-heading"><h2 id="node-title">No model loaded</h2><button id="show-doc" disabled aria-haspopup="dialog">Doc</button></div><p id="node-description">Open an ONNX model to explore its connections.</p>
          <button id="open-function" class="icon-button" hidden>Open function</button>
          <div id="node-details"><dl><dt>NODE NAME</dt><dd id="node-name">—</dd><dt>DOMAIN</dt><dd id="node-domain">—</dd><dt>OPSET</dt><dd id="node-opset">—</dd></dl>
          <h3>Inputs <span id="input-count">0</span></h3><ul id="node-inputs"></ul>
          <h3>Outputs <span id="output-count">0</span></h3><ul id="node-outputs"></ul>
          <section id="node-captures" hidden><h3>Captured inputs</h3><ul id="captured-inputs"></ul></section>
          <section id="node-parameters" hidden>
            <h3>Attributes <span id="attribute-count">0</span></h3><ul id="node-attributes"></ul>
            <h3>Initializers <span id="parameter-count">0</span></h3><ul id="node-initializers"></ul>
          </section></div>
          <section id="metadata-panel" hidden>
            <div id="metadata-content" hidden>
              <dl id="metadata-analysis"></dl>
              <section id="metadata-releases-section" hidden><h3>Releasable after this node</h3><ul id="metadata-releases"></ul></section>
              <section id="metadata-last-use-section" hidden><h3 title="Last reads of caller-owned values; these are not released">Last use: inputs / weights</h3><ul id="metadata-last-use"></ul></section>
              <section id="metadata-unused-section" hidden><h3>Unused outputs</h3><ul id="metadata-unused"></ul></section>
              <details><summary>Stored node metadata</summary><dl id="metadata-node"></dl></details>
              <details><summary>Connected tensors (<span id="metadata-value-count">0</span>)</summary><ul id="metadata-values"></ul></details>
              <details><summary>Model metadata</summary><dl id="metadata-model"></dl></details>
              <details><summary>Graph metadata</summary><dl id="metadata-graph"></dl></details>
            </div>
          </section>
        </aside>
      </div>
    </section>
    <section id="logs-view" role="tabpanel" aria-labelledby="logs-tab" tabindex="0"${initialTab === "model" ? " hidden" : ""}>
      <h2>Action history</h2>
      <p>Model operations and errors only. Operations show Running, OK, Failed or Cancelled.</p>
      <p id="logs-empty">No actions yet.</p>
      <ol id="log-entries" aria-label="Action history"></ol>
    </section>
    <section id="status-view" role="tabpanel" aria-labelledby="status-tab" tabindex="0" hidden>
      <h2>Application status</h2>
      <dl class="status-details" aria-live="polite">
        <dt>Native library</dt><dd id="status-library">Waiting for extension host</dd>
        <dt>Current graph</dt><dd id="status-model">Waiting for extension host</dd>
        <dt>Native model handle</dt><dd id="status-handle">Waiting for extension host</dd>
        <dt>Current operations</dt><dd id="status-running">Waiting for extension host</dd>
        <dt>Recovery</dt><dd id="status-recovery">Waiting for extension host</dd>
      </dl>
      <h3>What happens after an exception?</h3>
      <p>A caught C++ exception becomes a JavaScript error and is recorded in Logs. It does not unload onnx-light. A failed shape or metadata refresh discards only the native model handle; the next native action recreates it. The displayed graph is kept.</p>
      <section id="status-failure" hidden>
        <h3>Last failed action</h3>
        <strong id="status-failure-action"></strong>
        <p id="status-failure-time"></p>
        <pre id="status-failure-details"></pre>
      </section>
    </section>
    <section id="controls-view" role="tabpanel" aria-labelledby="controls-tab" tabindex="0" hidden>
      <h2>Keyboard and mouse controls</h2>
      <p>Graph shortcuts apply when the graph, one of its nodes or its zoom controls has focus. Use Cmd on macOS where Ctrl/Cmd is shown.</p>
      <h3 id="mouse-controls-heading">Mouse and trackpad</h3>
      <table class="controls-table" aria-labelledby="mouse-controls-heading">
        <thead><tr><th scope="col">Control</th><th scope="col">Action</th></tr></thead>
        <tbody>
          <tr><th scope="row">Click a node, input or output</th><td>Select it and show its details in the inspector.</td></tr>
          <tr><th scope="row">Click the background</th><td>Clear selection.</td></tr>
          <tr><th scope="row"><kbd>Ctrl/Cmd</kbd> or <kbd>Shift</kbd> + click</th><td>Toggle a node in the selection.</td></tr>
          <tr><th scope="row">Drag a node</th><td>Move it. Drag a selected node to move the whole selection.</td></tr>
          <tr><th scope="row">Drag the background</th><td>Select nodes in a rectangle. Hold <kbd>Ctrl/Cmd</kbd> or <kbd>Shift</kbd> to add to the selection.</td></tr>
          <tr><th scope="row">Scroll</th><td>Pan vertically.</td></tr>
          <tr><th scope="row"><kbd>Shift</kbd> + scroll</th><td>Pan horizontally. Horizontal trackpad scrolling also pans horizontally.</td></tr>
          <tr><th scope="row"><kbd>Ctrl</kbd> + scroll</th><td>Zoom in or out around the pointer.</td></tr>
          <tr><th scope="row">Zoom area, then drag</th><td>Magnify the rectangle without changing selection or node positions. The mode ends after one drag.</td></tr>
          <tr><th scope="row"><kbd>Alt</kbd> + drag</th><td>Magnify a rectangle, including over nodes and subgraph links, without opening the subgraph.</td></tr>
        </tbody>
      </table>
      <h3 id="keyboard-controls-heading">Graph keyboard shortcuts</h3>
      <table class="controls-table" aria-labelledby="keyboard-controls-heading">
        <thead><tr><th scope="col">Control</th><th scope="col">Action</th></tr></thead>
        <tbody>
          <tr><th scope="row"><kbd>Enter</kbd> or <kbd>Space</kbd></th><td>Select the focused node, input or output. Hold <kbd>Ctrl/Cmd</kbd> or <kbd>Shift</kbd> to toggle it in the selection.</td></tr>
          <tr><th scope="row"><kbd>Ctrl/Cmd + A</kbd></th><td>Select all nodes, inputs and outputs.</td></tr>
          <tr><th scope="row"><kbd>F</kbd></th><td>Fit the graph.</td></tr>
          <tr><th scope="row"><kbd>R</kbd></th><td>Jump to readable size (100%), centered on the inspected node.</td></tr>
          <tr><th scope="row"><kbd>+</kbd> / <kbd>−</kbd></th><td>Zoom in / out.</td></tr>
          <tr><th scope="row"><kbd>Escape</kbd></th><td>Cancel the current drag, selection rectangle or Zoom area mode; otherwise clear selection.</td></tr>
        </tbody>
      </table>
      <h3 id="navigation-controls-heading">Navigation and buttons</h3>
      <table class="controls-table" aria-labelledby="navigation-controls-heading">
        <thead><tr><th scope="col">Control</th><th scope="col">Action</th></tr></thead>
        <tbody>
          <tr><th scope="row"><kbd>Tab</kbd> / <kbd>Shift + Tab</kbd></th><td>Move focus between controls and graph nodes.</td></tr>
          <tr><th scope="row"><kbd>Enter</kbd> or <kbd>Space</kbd></th><td>Activate a focused button or subgraph link.</td></tr>
          <tr><th scope="row"><kbd>Left</kbd> / <kbd>Right</kbd> arrow</th><td>Switch tabs when a tab header has focus.</td></tr>
          <tr><th scope="row"><kbd>Home</kbd> / <kbd>End</kbd></th><td>Select the first / last tab when a tab header has focus.</td></tr>
          <tr><th scope="row">+ / − buttons</th><td>Zoom in / out.</td></tr>
          <tr><th scope="row">Fit</th><td>Fit the whole graph in the view.</td></tr>
          <tr><th scope="row">Zoom to readable size</th><td>Zoom to 100%, centered on the inspected node.</td></tr>
          <tr><th scope="row">Refresh positions</th><td>Reset the node layout.</td></tr>
          <tr><th scope="row">Subgraph link / breadcrumbs</th><td>Open a subgraph / return to a parent graph.</td></tr>
          <tr><th scope="row">Doc</th><td>Open operator documentation from the inspector. Use Close or <kbd>Escape</kbd> to dismiss it.</td></tr>
        </tbody>
      </table>
    </section>
    <section id="repositories-view" role="tabpanel" aria-labelledby="repositories-tab" tabindex="0" hidden>
      <h2>ONNX repositories</h2>
      <p>Explore the official ONNX projects and the onnx-light ecosystem on GitHub.</p>
      <h3>Official projects</h3>
      <ul class="repository-list">
        <li>
          <a href="https://github.com/onnx/onnx"><svg viewBox="0 0 24 24" width="28" height="28" aria-hidden="true"><circle cx="5" cy="12" r="3"/><circle cx="18" cy="5" r="3"/><circle cx="18" cy="19" r="3"/><path d="m8 11 7-5M8 13l7 5"/></svg>ONNX</a>
          <img class="repository-logo" src="${resource("repositories/onnx.png")}" alt="" width="96" height="64">
          <p>An open standard for representing machine learning models, with a common model format and operator definitions.</p>
        </li>
        <li>
          <a href="https://github.com/microsoft/onnxruntime"><svg viewBox="0 0 24 24" width="28" height="28" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="m10 7 7 5-7 5Z"/></svg>ONNX Runtime</a>
          <img class="repository-logo" src="${resource("repositories/onnxruntime.png")}" alt="" width="96" height="64">
          <p>A cross-platform engine for accelerating machine learning inference and training across hardware backends.</p>
        </li>
        <li>
          <a href="https://github.com/microsoft/onnxruntime-extensions"><svg viewBox="0 0 24 24" width="28" height="28" aria-hidden="true"><path d="M8 4H4v16h16v-7h-4a3 3 0 1 1 0-6h4V4h-7V3a3 3 0 0 0-6 0v1"/></svg>ONNX Runtime Extensions</a>
          <img class="repository-logo" src="${resource("repositories/onnxruntime.png")}" alt="" width="96" height="64">
          <p>Custom operators and tools for adding preprocessing and postprocessing to ONNX models.</p>
        </li>
        <li>
          <a href="https://github.com/microsoft/onnxruntime-genai"><svg viewBox="0 0 24 24" width="28" height="28" aria-hidden="true"><path d="m12 3 3 6 6 3-6 3-3 6-3-6-6-3 6-3ZM19 2v4M17 4h4"/></svg>ONNX Runtime GenAI</a>
          <img class="repository-logo" src="${resource("repositories/onnxruntime.png")}" alt="" width="96" height="64">
          <p>Generative AI APIs for ONNX Runtime, including tokenization, generation loops, sampling and beam search.</p>
        </li>
      </ul>
      <h3>onnx-light ecosystem</h3>
      <ul class="repository-list">
        <li>
          <a href="https://github.com/xadupre/onnx-light"><svg viewBox="0 0 24 24" width="28" height="28" aria-hidden="true"><path d="m14 2-9 12h6l-1 8 9-13h-6Z"/></svg>onnx-light</a>
          <img class="repository-logo" src="${resource("repositories/onnx-light.svg")}" alt="" width="96" height="64">
          <p>A lightweight C++ library with Python bindings for manipulating ONNX models, shape inference and graph optimization.</p>
        </li>
        <li>
          <a href="https://github.com/xadupre/onnx-light-cpu"><svg viewBox="0 0 24 24" width="28" height="28" aria-hidden="true"><rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 2v4m6-4v4M9 18v4m6-4v4M2 9h4m-4 6h4m12-6h4m-4 6h4M10 10h4v4h-4Z"/></svg>onnx-light-cpu</a>
          <img class="repository-logo" src="${resource("repositories/onnx-light-cpu.svg")}" alt="" width="96" height="64">
          <p>CPU execution kernels for running ONNX models with onnx-light.</p>
        </li>
        <li>
          <a href="https://github.com/xadupre/onnx-light-ort"><svg viewBox="0 0 24 24" width="28" height="28" aria-hidden="true"><path d="M8 4v5m8-5v5M5 9h14v3a7 7 0 0 1-14 0ZM12 19v3"/></svg>onnx-light-ort</a>
          <img class="repository-logo" src="${resource("repositories/onnx-light-ort.svg")}" alt="" width="96" height="64">
          <p>Custom-operator interoperability between onnx-light and ONNX Runtime, in both directions.</p>
        </li>
        <li>
          <a href="https://github.com/xadupre/onnx-light-viewer"><img src="${resource("logo.svg")}" alt="" width="28" height="28">onnx-light-viewer</a>
          <img class="repository-logo" src="${resource("logo.svg")}" alt="" width="96" height="64">
          <p>An interactive ONNX graph viewer for VS Code, powered by onnx-light.</p>
        </li>
      </ul>
    </section>
  </main>
  <dialog id="operator-doc" aria-labelledby="doc-title">
    <header class="inspector-heading"><h2 id="doc-title"></h2><button id="close-doc" autofocus>Close</button></header>
    <p id="doc-version"></p><div id="doc-body"></div>
  </dialog>
  <script nonce="${nonce}" src="${resource("vendor/markdown-it.min.js")}"></script>
  <script nonce="${nonce}" src="${resource("viewer.js")}"></script>
</body>
</html>`;
}
