export function webviewHtml(
  resource: (name: string) => string,
  cspSource: string,
  nonce: string,
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
      <header class="graph-toolbar">
        <div><a id="repository-link" href="https://github.com/xadupre/onnx-light-viewer" title="ONNX Light Viewer on GitHub" aria-label="Open the ONNX Light Viewer GitHub repository"><img src="${resource("logo.svg")}" alt="" width="24" height="24"></a><button id="open-model" class="primary">Open ONNX model</button><button id="refresh-shapes" class="icon-button" disabled title="Refresh shape and type information with onnx-light" aria-label="Refresh shape and type information"><svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><path d="M20 11a8 8 0 1 0-2 6M20 4v7h-7"/></svg>Refresh shapes</button><button id="refresh-metadata" class="icon-button" disabled title="Recompute constants, tags and lifetimes for the displayed graph with onnx-light" aria-label="Refresh metadata for the displayed graph"><svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><path d="M20 11a8 8 0 1 0-2 6M20 4v7h-7"/></svg>Refresh metadata</button><button id="refresh-positions" class="icon-button" disabled title="Reset the portrait layout with new random horizontal positions" aria-label="Refresh node positions"><svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><path d="M20 11a8 8 0 1 0-2 6M20 4v7h-7"/></svg>Refresh positions</button></div>
        <div><button id="show-metadata" class="icon-button" aria-expanded="false" aria-controls="metadata-panel" title="Show native constants, tags, lifetimes and stored metadata">Metadata</button><button id="open-backend-test" class="icon-button" title="Search and open an onnx-light backend test">Backend test</button></div>
        <div class="graph-stats" aria-live="polite"><span id="graph-summary">1 node / 2 tensors</span><span id="initializer-count">0 initializers</span><span id="shape-status" role="status"></span><span id="metadata-status" role="status"></span><span id="navigation-status" role="status"></span></div>
        <nav id="graph-navigation" aria-label="Graph breadcrumbs" hidden></nav>
      </header>
      <div class="graph-body">
        <div class="canvas" id="canvas">
          <svg id="graph" role="group" aria-label="Interactive ONNX graph" tabindex="0"><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z"/></marker></defs><g id="edges"></g><g id="nodes"></g></svg>
          <div class="zoom-controls" aria-label="Graph zoom"><button id="zoom-out" aria-label="Zoom out">−</button><button id="zoom-fit" title="Fit graph">Fit</button><button id="zoom-in" aria-label="Zoom in">+</button></div>
        </div>
        <aside aria-label="Node inspector">
          <div class="eyebrow">INSPECTOR</div><div class="inspector-heading"><h2 id="node-title">Relu</h2><button id="show-doc" disabled aria-haspopup="dialog">Doc</button></div><p id="node-description">Select a node to explore its connections.</p>
          <div id="node-details"><dl><dt>NODE NAME</dt><dd id="node-name">activation</dd><dt>DOMAIN</dt><dd id="node-domain">ai.onnx</dd><dt>OPSET</dt><dd id="node-opset">—</dd></dl>
          <h3>Inputs <span id="input-count">1</span></h3><ul id="node-inputs"></ul>
          <h3>Outputs <span id="output-count">1</span></h3><ul id="node-outputs"></ul>
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
  </main>
  <dialog id="operator-doc" aria-labelledby="doc-title">
    <header class="inspector-heading"><h2 id="doc-title"></h2><button id="close-doc" autofocus>Close</button></header>
    <p id="doc-version"></p><div id="doc-body"></div>
  </dialog>
  <script nonce="${nonce}" src="${resource("viewer.js")}"></script>
</body>
</html>`;
}
