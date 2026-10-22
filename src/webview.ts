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
        <div><img src="${resource("logo.svg")}" alt="" width="24" height="24"><button id="open-model" class="primary">Open ONNX model</button></div>
        <div class="graph-stats" aria-live="polite"><span id="graph-summary">1 node / 2 tensors</span><span id="initializer-count">0 initializers</span></div>
      </header>
      <div class="graph-body">
        <div class="canvas" id="canvas">
          <svg id="graph" role="group" aria-label="Interactive ONNX graph" tabindex="0"><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z"/></marker></defs><g id="edges"></g><g id="nodes"></g></svg>
          <div class="zoom-controls" aria-label="Graph zoom"><button id="zoom-out" aria-label="Zoom out">−</button><button id="zoom-fit" title="Fit graph">Fit</button><button id="zoom-in" aria-label="Zoom in">+</button></div>
        </div>
        <aside aria-label="Node inspector">
          <div class="eyebrow">INSPECTOR</div><div class="inspector-heading"><h2 id="node-title">Relu</h2><button id="show-doc" disabled aria-haspopup="dialog">Doc</button></div><p id="node-description">Select a node to explore its connections.</p>
          <dl><dt>NODE NAME</dt><dd id="node-name">activation</dd><dt>DOMAIN</dt><dd id="node-domain">ai.onnx</dd><dt>OPSET</dt><dd id="node-opset">—</dd></dl>
          <h3>Inputs <span id="input-count">1</span></h3><ul id="node-inputs"></ul>
          <h3>Outputs <span id="output-count">1</span></h3><ul id="node-outputs"></ul>
          <section id="node-parameters" hidden>
            <h3>Attributes <span id="attribute-count">0</span></h3><ul id="node-attributes"></ul>
            <h3>Initializers <span id="parameter-count">0</span></h3><ul id="node-initializers"></ul>
            <p>Tensor raw_data and external weights are not loaded. Long previews are truncated.</p>
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
