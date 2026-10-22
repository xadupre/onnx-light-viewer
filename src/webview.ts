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
  <header class="topbar">
    <div class="brand"><img src="${resource("logo.svg")}" alt="" width="36" height="36"><div><strong>ONNX <span>Light Viewer</span></strong><small>MODEL EXPLORER</small></div></div>
    <button id="open-model" class="primary"><span aria-hidden="true">+</span> Open ONNX model</button>
  </header>
  <main>
    <section class="intro">
      <div><div class="eyebrow">A LITTLE CLARITY FOR YOUR MODELS</div><h1>Your model, connected.</h1><p>Explore the graph. Follow the tensors. Start with one node.</p></div>
      <span id="model-badge" class="badge">DEMO MODEL</span>
    </section>
    <section class="workspace" aria-label="ONNX graph explorer">
      <div class="graph-toolbar"><div><span class="status-dot"></span><strong id="graph-name">A first ONNX graph</strong></div><span id="graph-summary">1 node / 2 tensors</span></div>
      <div class="graph-body">
        <div class="canvas" id="canvas">
          <svg id="graph" role="group" aria-label="Interactive ONNX graph" tabindex="0"><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z"/></marker></defs><g id="edges"></g><g id="nodes"></g></svg>
          <div class="canvas-caption">ONNX GRAPH <span>Select a node to inspect it</span></div>
          <div class="zoom-controls" aria-label="Graph zoom"><button id="zoom-out" aria-label="Zoom out">−</button><button id="zoom-fit" title="Fit graph">Fit</button><button id="zoom-in" aria-label="Zoom in">+</button></div>
        </div>
        <aside aria-label="Node inspector">
          <div class="eyebrow">INSPECTOR</div><h2 id="node-title">Relu</h2><p id="node-description">Select a node to explore its connections.</p>
          <dl><dt>NODE NAME</dt><dd id="node-name">activation</dd><dt>DOMAIN</dt><dd id="node-domain">ai.onnx</dd></dl>
          <h3>Inputs <span id="input-count">1</span></h3><ul id="node-inputs"></ul>
          <h3>Outputs <span id="output-count">1</span></h3><ul id="node-outputs"></ul>
        </aside>
      </div>
      <footer><span><i class="legend-input"></i> Input <i class="legend-operator"></i> Operator <i class="legend-output"></i> Output</span><span id="initializer-count">0 initializers</span></footer>
    </section>
    <section class="welcome-note"><span class="note-icon" aria-hidden="true">↗</span><div><strong id="welcome-title">One node. A good place to start.</strong><p id="welcome-description">This demo connects an input to a Relu operator and an output. Open an .onnx file to explore your own model.</p></div><button id="open-another">Browse files <span aria-hidden="true">→</span></button></section>
  </main>
  <script nonce="${nonce}" src="${resource("viewer.js")}"></script>
</body>
</html>`;
}
