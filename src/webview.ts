/**
 * Builds the static shell; model-derived text arrives later through messages.
 * The DOM IDs are the binding points used by media/viewer.js.
 * Inputs are trusted host-generated values interpolated without HTML escaping;
 * never supply model text here. The CSP blocks other content by default.
 * @param resource Resolves bundled filenames (viewer.css, viewer.js, logo.svg)
 * to webview URI strings within the panel's allowed media resource root.
 * @param cspSource The panel webview's cspSource, allowing local images/styles.
 * @param nonce Fresh per-panel nonce authorizing only the bundled viewer script.
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
        <div><button id="show-metadata" class="icon-button" aria-expanded="false" aria-controls="metadata-panel" title="Show native constants, tags, lifetimes and stored metadata" aria-describedby="metadata-status">Metadata</button><button id="open-backend-test" class="icon-button" title="Search and open an onnx-light backend test">Backend test</button></div>
        <div class="graph-stats" aria-live="polite"><span id="graph-summary">No model loaded</span><span id="initializer-count">0 initializers</span><span id="onnx-light-version" title="onnx-light C++ release used by this viewer">onnx-light ${onnxLightVersion}</span><span id="shape-status" role="status"></span><span id="metadata-status" role="status"></span><span id="navigation-status" role="status"></span><span id="load-status" role="status"></span></div>
        <nav id="graph-navigation" aria-label="Graph breadcrumbs" hidden></nav>
      </header>
      <div class="graph-body">
        <div class="canvas" id="canvas">
          <svg id="graph" role="group" aria-label="Interactive ONNX graph" tabindex="0"><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z"/></marker></defs><g id="edges"></g><g id="nodes"></g></svg>
          <div class="zoom-controls" aria-label="Graph zoom"><button id="zoom-out" aria-label="Zoom out">−</button><button id="zoom-fit" title="Fit graph">Fit</button><button id="zoom-readable" title="Zoom to readable size (100%), centered on the inspected node" aria-label="Zoom to readable size"><svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><path d="m8 17 4-10 4 10M10 13h4M4 8V4h4M16 4h4v4M20 16v4h-4M8 20H4v-4"/></svg></button><button id="zoom-in" aria-label="Zoom in">+</button></div>
        </div>
        <aside aria-label="Node inspector">
          <div class="eyebrow">INSPECTOR</div><div class="inspector-heading"><h2 id="node-title">No model loaded</h2><button id="show-doc" disabled aria-haspopup="dialog">Doc</button></div><p id="node-description">Open an ONNX model to explore its connections.</p>
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
      <p>Graph shortcuts apply when the graph or one of its nodes has focus. Use Cmd on macOS where Ctrl/Cmd is shown.</p>
      <h3>Mouse and trackpad</h3>
      <ul>
        <li>Click a node, input or output to select it and show its details in the inspector. Click the background to clear selection.</li>
        <li>Ctrl/Cmd or Shift + click toggles a node in the selection.</li>
        <li>Drag a node to move it; drag a selected node to move the whole selection.</li>
        <li>Drag the background to select nodes in a rectangle. Hold Ctrl/Cmd or Shift to add to the selection.</li>
        <li>Scroll to pan vertically; Shift + scroll pans horizontally. Horizontal trackpad scrolling also pans horizontally.</li>
        <li>Ctrl + scroll zooms in or out.</li>
      </ul>
      <h3>Graph keyboard shortcuts</h3>
      <ul>
        <li>Enter or Space selects the focused node, input or output. Hold Ctrl/Cmd or Shift to toggle it in the selection.</li>
        <li>Ctrl/Cmd + A selects all nodes, inputs and outputs.</li>
        <li>Escape cancels the current drag or selection rectangle; otherwise it clears selection.</li>
      </ul>
      <h3>Navigation and buttons</h3>
      <ul>
        <li>Tab / Shift + Tab moves focus between controls and graph nodes. Enter or Space activates a focused button or subgraph link.</li>
        <li>Left / Right arrow switches tabs when a tab header has focus. Home / End selects the first / last tab.</li>
        <li>Use the + and − buttons to zoom, Fit to fit the graph, or Zoom to readable size to zoom to 100% centered on the inspected node.</li>
        <li>Use Refresh positions to reset the node layout.</li>
        <li>Activate a subgraph link to open it; use the graph breadcrumbs to return to a parent graph.</li>
        <li>Use Doc in the inspector to open operator documentation, and Close or Escape to dismiss it.</li>
      </ul>
    </section>
    <section id="repositories-view" role="tabpanel" aria-labelledby="repositories-tab" tabindex="0" hidden>
      <h2>ONNX repositories</h2>
      <p>Explore the official ONNX projects and the onnx-light ecosystem on GitHub.</p>
      <h3>Official projects</h3>
      <ul class="repository-list">
        <li>
          <a href="https://github.com/onnx/onnx"><svg viewBox="0 0 24 24" width="28" height="28" aria-hidden="true"><circle cx="5" cy="12" r="3"/><circle cx="18" cy="5" r="3"/><circle cx="18" cy="19" r="3"/><path d="m8 11 7-5M8 13l7 5"/></svg>ONNX</a>
          <p>An open standard for representing machine learning models, with a common model format and operator definitions.</p>
        </li>
        <li>
          <a href="https://github.com/microsoft/onnxruntime"><svg viewBox="0 0 24 24" width="28" height="28" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="m10 7 7 5-7 5Z"/></svg>ONNX Runtime</a>
          <p>A cross-platform engine for accelerating machine learning inference and training across hardware backends.</p>
        </li>
        <li>
          <a href="https://github.com/microsoft/onnxruntime-extensions"><svg viewBox="0 0 24 24" width="28" height="28" aria-hidden="true"><path d="M8 4H4v16h16v-7h-4a3 3 0 1 1 0-6h4V4h-7V3a3 3 0 0 0-6 0v1"/></svg>ONNX Runtime Extensions</a>
          <p>Custom operators and tools for adding preprocessing and postprocessing to ONNX models.</p>
        </li>
        <li>
          <a href="https://github.com/microsoft/onnxruntime-genai"><svg viewBox="0 0 24 24" width="28" height="28" aria-hidden="true"><path d="m12 3 3 6 6 3-6 3-3 6-3-6-6-3 6-3ZM19 2v4M17 4h4"/></svg>ONNX Runtime GenAI</a>
          <p>Generative AI APIs for ONNX Runtime, including tokenization, generation loops, sampling and beam search.</p>
        </li>
      </ul>
      <h3>onnx-light ecosystem</h3>
      <ul class="repository-list">
        <li>
          <a href="https://github.com/xadupre/onnx-light"><svg viewBox="0 0 24 24" width="28" height="28" aria-hidden="true"><path d="m14 2-9 12h6l-1 8 9-13h-6Z"/></svg>onnx-light</a>
          <p>A lightweight C++ library with Python bindings for manipulating ONNX models, shape inference and graph optimization.</p>
        </li>
        <li>
          <a href="https://github.com/xadupre/onnx-light-cpu"><svg viewBox="0 0 24 24" width="28" height="28" aria-hidden="true"><rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 2v4m6-4v4M9 18v4m6-4v4M2 9h4m-4 6h4m12-6h4m-4 6h4M10 10h4v4h-4Z"/></svg>onnx-light-cpu</a>
          <p>CPU execution kernels for running ONNX models with onnx-light.</p>
        </li>
        <li>
          <a href="https://github.com/xadupre/onnx-light-ort"><svg viewBox="0 0 24 24" width="28" height="28" aria-hidden="true"><path d="M8 4v5m8-5v5M5 9h14v3a7 7 0 0 1-14 0ZM12 19v3"/></svg>onnx-light-ort</a>
          <p>Custom-operator interoperability between onnx-light and ONNX Runtime, in both directions.</p>
        </li>
        <li>
          <a href="https://github.com/xadupre/onnx-light-viewer"><img src="${resource("logo.svg")}" alt="" width="28" height="28">onnx-light-viewer</a>
          <p>An interactive ONNX graph viewer for VS Code, powered by onnx-light.</p>
        </li>
      </ul>
    </section>
  </main>
  <dialog id="operator-doc" aria-labelledby="doc-title">
    <header class="inspector-heading"><h2 id="doc-title"></h2><button id="close-doc" autofocus>Close</button></header>
    <p id="doc-version"></p><div id="doc-body"></div>
  </dialog>
  <script nonce="${nonce}" src="${resource("viewer.js")}"></script>
</body>
</html>`;
}
