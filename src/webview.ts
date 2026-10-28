import { huggingFaceTasks } from "./huggingFaceTasks";

/**
 * Builds the static shell; model-derived text arrives later through messages.
 * The DOM IDs are the binding points used by media/viewer.js.
 * Inputs are trusted host-generated values interpolated without HTML escaping;
 * never supply model text here. The CSP blocks other content by default.
 * @param resource Resolves bundled styles, scripts and images
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
  const historyControls = (target: "open" | "save"): string =>
    `<details id="${target}-history" class="path-history" hidden><summary>Recent paths</summary><p>Suggestions are history, not a guarantee that a path still exists or is writable. Selecting one only fills the field.</p><button id="${target}-clear-history" type="button">Clear ${target === "open" ? "Open" : "Save"} history</button><ul id="${target}-history-entries" aria-label="Recent ${target} paths"></ul></details>`;
  const documentationLink = (name: string, url: string): string =>
    `<a class="repository-documentation" href="${url}" aria-label="${name} documentation" title="${name} documentation"><svg viewBox="0 0 24 24" width="24" height="24" aria-hidden="true"><path d="M12 5v16M12 5C9 3 5 3 2 4v15c3-1 7-1 10 2 3-3 7-3 10-2V4c-3-1-7-1-10 1Z"/></svg></a>`;
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
      <button id="statistics-tab" role="tab" aria-controls="statistics-view" aria-selected="false" tabindex="-1">Statistics</button>
      <button id="memory-tab" role="tab" aria-controls="memory-view" aria-selected="false" tabindex="-1">Memory</button>
      <button id="tensors-tab" role="tab" aria-controls="tensors-view" aria-selected="false" tabindex="-1"><svg class="tensor-statistics-icon" viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><path d="M4 3v17h17M8 16v-5m5 5V6m5 10V9"/></svg>Tensors</button>
      <button id="status-tab" role="tab" aria-controls="status-view" aria-selected="false" tabindex="-1">Status</button>
      <button id="controls-tab" role="tab" aria-controls="controls-view" aria-selected="false" tabindex="-1">Controls</button>
      <button id="repositories-tab" role="tab" aria-controls="repositories-view" aria-selected="false" tabindex="-1">ONNX</button>
      <button id="settings-tab" role="tab" aria-controls="settings-view" aria-selected="false" tabindex="-1">Settings</button>
    </nav>
    <section id="model-view" role="tabpanel" aria-labelledby="model-tab"${initialTab === "logs" ? " hidden" : ""}>
      <header class="graph-toolbar">
        <div><a id="repository-link" href="https://github.com/xadupre/onnx-light-viewer" title="ONNX Light Viewer on GitHub" aria-label="Open the ONNX Light Viewer GitHub repository"><img src="${resource("logo.svg")}" alt="" width="24" height="24"></a><button id="refresh-shapes" class="icon-button" disabled title="Refresh shape and type information with onnx-light" aria-label="Refresh shape and type information" aria-describedby="shape-status"><svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><path d="M20 11a8 8 0 1 0-2 6M20 4v7h-7"/></svg>Refresh shapes</button><button id="refresh-metadata" class="icon-button" disabled title="Recompute constants, tags and lifetimes for the displayed graph with onnx-light" aria-label="Refresh metadata for the displayed graph" aria-describedby="metadata-status"><svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><path d="M20 11a8 8 0 1 0-2 6M20 4v7h-7"/></svg>Refresh metadata</button><button id="refresh-positions" class="icon-button" disabled title="Restore the hierarchical layout and reset manually moved positions" aria-label="Refresh node positions"><svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><path d="M20 11a8 8 0 1 0-2 6M20 4v7h-7"/></svg>Refresh positions</button><button id="verify-model" class="icon-button" disabled title="Verify the entire model or all pipeline stages with onnx-light, without changing them" aria-label="Verify model" aria-describedby="verify-status"><svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><path d="m5 12 4 4L19 6"/></svg>Verify</button></div>
        <div class="graph-stats" aria-live="polite"><span id="graph-summary">No model loaded</span><span id="initializer-count">0 initializers</span><span id="onnx-light-version" data-version="${onnxLightVersion}" title="onnx-light C++ release used by this viewer">onnx-light ${onnxLightVersion}</span><span id="shape-status" role="status"></span><span id="metadata-status" role="status"></span><span id="verify-status" role="status"></span><span id="navigation-status" role="status"></span><span id="load-status" role="status"></span></div>
        <span id="save-status" role="status"></span>
        <span id="fusion-status" role="status"></span>
        <nav id="graph-navigation" aria-label="Graph breadcrumbs" hidden></nav>
        <progress id="load-progress" aria-label="Loading ONNX model" hidden></progress>
      </header>
      <div class="graph-body">
        <div class="canvas" id="canvas">
          <section id="model-error" role="alert" hidden><h2>Could not open model</h2><p id="model-error-details"></p><p>Use Open to choose another model. Details are available in Logs.</p></section>
          <svg id="graph" role="group" aria-label="Interactive ONNX graph" tabindex="0"><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z"/></marker></defs><g id="graph-overview" aria-hidden="true" display="none"><path id="overview-edges"/><path id="overview-nodes"/><g id="overview-labels"/></g><path id="crossing-edges" class="crossing-edge"/><path id="released-crossing-edges" class="crossing-edge released"/><g id="edges"></g><g id="nodes"></g></svg>
          <div class="zoom-controls" aria-label="Graph zoom and navigation"><button id="go-top" disabled title="Go to the top of the model (Ctrl/Cmd+Home)" aria-keyshortcuts="Control+Home Meta+Home">Top</button><button id="go-bottom" disabled title="Go to the bottom of the model (Ctrl/Cmd+End)" aria-keyshortcuts="Control+End Meta+End">Bottom</button><button id="zoom-in" title="Zoom in (+)" aria-label="Zoom in" aria-keyshortcuts="+">+</button><button id="zoom-out" title="Zoom out (-)" aria-label="Zoom out" aria-keyshortcuts="-">−</button><button id="zoom-fit" title="Fit graph (F)" aria-keyshortcuts="F">Fit</button><button id="zoom-readable" title="Zoom to readable size (100%), centered on the inspected node (R)" aria-label="Zoom to readable size" aria-keyshortcuts="R"><svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><path d="m8 17 4-10 4 10M10 13h4M4 8V4h4M16 4h4v4M20 16v4h-4M8 20H4v-4"/></svg></button><button id="zoom-area" title="Drag a rectangle to zoom into an area; Alt+drag also works. Escape cancels." aria-pressed="false"><span class="toggle-indicator" aria-hidden="true">&#10003;</span> Zoom area</button></div>
        </div>
        <div id="sidebar-resizer" role="separator" tabindex="0" aria-label="Resize side panel" aria-orientation="vertical" aria-controls="graph-panels" title="Drag to resize the side panel, or use Left and Right arrow keys"></div>
        <aside id="graph-panels" aria-label="Graph panels">
          <nav id="sidebar-controls" aria-label="Sidebar panels and model actions">
            <button id="open-model" class="icon-button" title="Show or hide the model file selector" aria-expanded="false" aria-controls="open-panel">Open</button>
            <button id="save-model" class="icon-button" disabled title="Show or hide model save options" aria-expanded="false" aria-controls="save-options" aria-describedby="save-status">Save</button>
            <button id="export-model" class="icon-button" title="Convert a model to ONNX with Python" aria-expanded="false" aria-controls="export-panel">Convert</button>
            <button id="show-find" class="icon-button" aria-expanded="true" aria-controls="find-panel" title="Show or hide node search">Find</button>
            <button id="optimize-model" class="icon-button" disabled title="Show or hide full native model optimization" aria-expanded="false" aria-controls="optimization-panel">Optimize</button>
            <button id="fuse-pattern" class="icon-button" disabled title="Show or hide fusion patterns and select candidates to fuse on the main graph" aria-expanded="false" aria-controls="fusion-panel">Fuse</button>
            <button id="undo-fusion" class="icon-button" disabled title="Choose whether to unfuse the last operation or all operations" aria-expanded="false" aria-controls="unfuse-panel" aria-describedby="fusion-status">Unfuse</button>
            <button id="show-operator" class="icon-button" aria-expanded="true" aria-controls="operator-panel" title="Show or hide operator and tensor details">Op</button>
            <button id="show-metadata" class="icon-button" aria-expanded="false" aria-controls="metadata-panel" title="Show or hide native constants, tags, lifetimes and stored metadata">Data</button>
          </nav>
          <section id="export-panel" class="inspector-panel file-panel" aria-labelledby="export-title" hidden>
            <h2 id="export-title" class="eyebrow panel-title">Convert to ONNX</h2>
            <table id="export-timings" aria-label="Download and conversion times" hidden>
              <thead><tr><th scope="col">Step</th><th scope="col">Elapsed time</th></tr></thead>
              <tbody>
                <tr><th scope="row">Download</th><td id="export-download-time">Not started</td></tr>
                <tr><th scope="row">Conversion</th><td id="export-conversion-time">Not started</td></tr>
              </tbody>
            </table>
            <p id="export-cache"></p>
            <p id="export-trust" hidden>Trust this workspace before running an exporter.</p>
            <form id="export-form">
              <fieldset id="export-fields">
                <div class="export-field">
                  <label for="export-tool">Exporter</label>
                  <select id="export-tool"><option value="modelbuilder">modelbuilder (ONNX Runtime GenAI)</option><option value="mobius">Mobius</option><option value="olive">Olive</option><option value="olive-recipes">Olive recipes (model export workflows)</option><option value="mbext">mbext (fast export)</option></select>
                </div>
                <div id="export-olive-options" class="export-option" hidden>
                  <div class="export-field">
                    <label for="export-olive-workflow">Olive workflow</label>
                    <select id="export-olive-workflow" disabled><option value="optimize">Export checkpoint / optimize ONNX</option><option value="quantize">Quantize existing ONNX (CPU, offline)</option></select>
                  </div>
                  <p>Export accepts a Hugging Face ID or local checkpoint and produces ONNX. Olive may download calibration data; install its dependencies in Settings' Python environment first.</p>
                </div>
                <div id="export-source-options" class="export-option">
                <label id="export-source-label" for="export-source">Hugging Face model ID or absolute local model directory</label>
                <input id="export-source" required placeholder="Qwen/Qwen3-0.6B" autocomplete="off" list="export-source-history"><datalist id="export-source-history"></datalist>
                </div>
                <div id="export-olive-quantization" class="export-option" hidden>
                  <div class="export-field">
                    <label for="export-olive-precision">Quantization precision (CPU, RTN)</label>
                    <select id="export-olive-precision" disabled><option value="int4">INT4 (Olive)</option><option value="int8">INT8 (ONNX Runtime dynamic)</option></select>
                  </div>
                </div>
                <div id="export-recipe-options" hidden>
                  <div class="export-field">
                    <label for="export-recipe">Olive recipe</label>
                    <select id="export-recipe" disabled required><option value="">Load the recipe list</option></select>
                  </div>
                  <button id="export-recipe-refresh" type="button">Refresh recipes</button>
                  <p id="export-recipe-status" role="status"></p>
                  <p id="export-recipe-details"></p>
                  <p>Recipes keep their own model, precision and target. Install the recipe's requirements in the configured Python environment, including any pinned Olive version, before running.</p>
                  <a id="export-recipe-link" href="https://github.com/microsoft/olive-recipes">Recipe source and requirements</a>
                </div>
                <label for="export-directory">Output directory or template</label>
                <input id="export-directory" required placeholder="/path/to/new-export" autocomplete="off" list="export-directory-history"><datalist id="export-directory-history"></datalist>
                <div id="export-target-options" class="export-option">
                  <div class="export-field">
                    <label for="export-precision">Precision</label>
                    <select id="export-precision"><option value="fp32">FP32</option><option value="fp16">FP16</option><option value="bf16">BF16</option><option value="int4">INT4</option><option value="int8" hidden disabled>INT8</option><option value="int2">INT2</option><option value="fp4">FP4</option></select>
                  </div>
                  <div class="export-field">
                    <label for="export-provider">Target device</label>
                    <select id="export-provider"><option value="cpu">CPU</option><option value="cuda">CUDA</option></select>
                  </div>
                </div>
                <details id="export-advanced" class="export-advanced">
                  <summary>Advanced options</summary>
                  <div id="export-layer-options" class="export-option" data-exporters="mbext modelbuilder" hidden>
                    <div class="export-field">
                      <label for="export-layer-count">Layer limit (test model)</label>
                      <input id="export-layer-count" type="number" min="1" step="1" placeholder="All layers" disabled>
                    </div>
                  </div>
                  <div class="export-option" data-exporters="mbext modelbuilder" hidden>
                    <label><input id="export-exclude-embeds" type="checkbox" data-export-option="excludeEmbeds" disabled> Exclude embeddings</label>
                    <label><input id="export-exclude-lm-head" type="checkbox" data-export-option="excludeLmHead" disabled> Exclude language modeling head</label>
                  </div>
                  <div class="export-option" data-exporters="mbext modelbuilder" data-excludes="export-exclude-lm-head" hidden>
                    <label><input id="export-include-hidden-states" type="checkbox" data-export-option="includeHiddenStates" disabled> Include hidden states</label>
                    <label><input id="export-prune-lm-head" type="checkbox" data-export-option="pruneLmHead" disabled> Last-token logits only</label>
                  </div>
                  <div class="export-option" data-exporters="olive" data-olive-workflows="optimize" hidden>
                    <div class="export-field">
                      <label for="export-olive-exporter">ONNX exporter</label>
                      <select id="export-olive-exporter" data-export-option="oliveExporter" disabled><option value="">Olive default (model builder)</option><option value="model_builder">Model builder</option><option value="dynamo_exporter">PyTorch Dynamo</option><option value="torchscript_exporter">PyTorch TorchScript</option></select>
                    </div>
                    <div class="export-field">
                      <label for="export-olive-task">Hugging Face task</label>
                      <input id="export-olive-task" placeholder="Olive default" list="export-olive-tasks" data-export-option="task" disabled>
                    </div>
                    <datalist id="export-olive-tasks"><option value="text-generation-with-past"><option value="text-generation"><option value="text-classification"><option value="feature-extraction"></datalist>
                    <label><input id="export-olive-trust-code" type="checkbox" data-export-option="trustRemoteCode" disabled> Trust model repository code (executes Python)</label>
                    <div class="export-field">
                      <label for="export-olive-splits">Model splits</label>
                      <input id="export-olive-splits" type="number" min="1" step="1" placeholder="Olive default" data-export-option="numSplits" disabled>
                    </div>
                    <div class="export-field">
                      <label for="export-olive-memory">Device memory (MB)</label>
                      <input id="export-olive-memory" type="number" min="1" step="1" placeholder="Olive default" data-export-option="memory" disabled>
                    </div>
                  </div>
                  <div class="export-option" data-exporters="mbext modelbuilder olive" data-precisions="int4 int8" hidden>
                    <label><input id="export-use-qdq" type="checkbox" data-export-option="useQdq" disabled> Q/DQ quantization</label>
                  </div>
                  <div class="export-option" data-exporters="mbext modelbuilder olive" data-precisions="int4" data-olive-workflows="optimize" hidden>
                    <div class="export-field">
                      <label for="export-block-size">Quantization block size</label>
                      <select id="export-block-size" data-export-option="blockSize" data-number disabled><option value="">Exporter default</option><option>16</option><option>32</option><option>64</option><option>128</option><option>256</option></select>
                    </div>
                  </div>
                  <div class="export-option" data-exporters="mbext modelbuilder" data-precisions="int4" hidden>
                    <div class="export-field">
                      <label for="export-quantization-algorithm">Quantization algorithm</label>
                      <select id="export-quantization-algorithm" data-export-option="quantizationAlgorithm" disabled><option value="">Exporter default</option><option value="rtn">RTN</option><option value="rtn_last">RTN, INT8 last MatMul</option><option value="k_quant">K-quant</option><option value="k_quant_mixed">K-quant mixed</option><option value="k_quant_last">K-quant, INT8 last MatMul</option></select>
                    </div>
                  </div>
                  <div class="export-option" data-exporters="mbext" hidden>
                    <div class="export-field">
                      <label for="export-opset">ONNX opset</label>
                      <input id="export-opset" type="number" min="1" step="1" placeholder="Exporter default" data-export-option="opset" disabled>
                    </div>
                  </div>
                  <div class="export-option" data-exporters="mbext mobius" hidden>
                    <div class="export-field">
                      <label for="export-max-shard-size">Maximum weight file size</label>
                      <input id="export-max-shard-size" placeholder="2GB" pattern="[1-9][0-9]*([KkMmGg]?[Bb])" data-export-option="maxShardSize" disabled>
                    </div>
                  </div>
                  <div class="export-option" data-exporters="mobius" hidden>
                    <label><input id="export-no-weights" type="checkbox" data-export-option="noWeights" disabled> Graph without weights</label>
                    <div class="export-field">
                      <label for="export-optimize">Graph optimizations</label>
                      <select id="export-optimize" data-export-option="optimize" disabled><option value="">Exporter default</option><option value="all">All</option><option value="group_query_attention">Group Query Attention</option><option value="packed_attention">Packed Attention</option><option value="skip_norm">Skip normalization</option></select>
                    </div>
                    <div class="export-field">
                      <label for="export-external-data">Weight file format</label>
                      <select id="export-external-data" data-export-option="externalData" disabled><option value="">Exporter default</option><option value="onnx">ONNX external data</option><option value="safetensors">Safetensors</option></select>
                    </div>
                    <div class="export-field">
                      <label for="export-runtime">Runtime configuration</label>
                      <select id="export-runtime" data-export-option="runtime" disabled><option value="">None</option><option value="ort-genai">ONNX Runtime GenAI</option><option value="onnx-genai">ONNX GenAI</option></select>
                    </div>
                    <label><input id="export-static-cache" type="checkbox" data-export-option="staticCache" disabled> Static KV cache</label>
                  </div>
                  <div class="export-option" data-exporters="mobius" data-requires="export-static-cache" hidden>
                    <div class="export-field">
                      <label for="export-max-sequence-length">Maximum sequence length</label>
                      <input id="export-max-sequence-length" type="number" min="1" step="1" placeholder="Model default" data-export-option="maxSequenceLength" disabled>
                    </div>
                  </div>
                </details>
              </fieldset>
              <p id="export-script"></p>
              <progress id="export-progress" aria-label="Converting ONNX model" hidden></progress>
              <p id="export-status" role="status"></p>
              <h3>Command and output</h3>
              <pre id="export-output" class="export-output" aria-label="Exporter command and output"><code id="export-command" aria-label="Exporter command"></code><span id="export-stdout"></span></pre>
              <p id="export-error" role="alert"></p>
              <footer><button id="cancel-export" type="button" disabled>Cancel conversion</button><button id="export-submit" class="primary" type="submit">Convert and open</button></footer>
            </form>
          </section>
          <section id="unfuse-panel" class="inspector-panel" aria-labelledby="unfuse-title" hidden>
            <h2 id="unfuse-title" class="eyebrow panel-title">Unfuse</h2>
            <p id="unfuse-count" role="status"></p>
            <p>Unfuse last reverses one Fuse selected operation. Repeat to go back further, or unfuse all operations in this session.</p>
            <div class="fusion-selection-controls"><button id="unfuse-last" disabled>Unfuse last</button><button id="unfuse-all" disabled>Unfuse all</button><button id="close-unfuse">Close</button></div>
            <ol id="unfuse-history" reversed aria-label="Applied fusion operations, most recent first"></ol>
          </section>
          <section id="find-panel" class="inspector-panel" aria-labelledby="find-title">
            <h2 id="find-title" class="eyebrow panel-title">Find</h2>
            <form id="find-node" role="search"><label for="node-search">Find node or operator</label><div class="node-search-controls"><input id="node-search" type="search" placeholder="Name or operator" aria-controls="node-search-suggestions" autocomplete="off"><button type="submit" title="Go to the next matching node" aria-label="Go to the next matching node">Go</button></div><span id="node-search-result" role="status"></span><ul id="node-search-suggestions" aria-label="Matching nodes" hidden></ul></form>
          </section>
          <section id="operator-panel" class="inspector-panel" aria-labelledby="operator-title">
            <h2 id="operator-title" class="eyebrow panel-title">INSPECTOR</h2>
            <div class="inspector-heading"><h2 id="node-title">No model loaded</h2><button id="show-doc" disabled aria-haspopup="dialog">Doc</button></div><p id="node-description">Open an ONNX model to explore its connections.</p>
            <button id="open-function" class="icon-button" hidden>Open function</button>
            <div id="node-details"><dl><dt id="node-name-label">NODE NAME</dt><dd id="node-name">—</dd><dt>DOMAIN</dt><dd id="node-domain">—</dd><dt>OPSET</dt><dd id="node-opset">—</dd></dl>
            <details class="inspector-section" open><summary>Inputs <span id="input-count">0</span></summary><ul id="node-inputs"></ul></details>
            <details class="inspector-section" open><summary>Outputs <span id="output-count">0</span></summary><ul id="node-outputs"></ul></details>
            <details id="node-captures" class="inspector-section" open hidden><summary>Captured inputs</summary><ul id="captured-inputs"></ul></details>
            <section id="node-parameters" hidden>
              <details class="inspector-section" open><summary>Attributes <span id="attribute-count">0</span></summary><ul id="node-attributes"></ul></details>
              <details class="inspector-section" open><summary>Initializers <span id="parameter-count">0</span></summary><ul id="node-initializers"></ul></details>
            </section>
            <details id="node-constants" class="inspector-section" open hidden><summary>Constants <span id="constant-count">0</span></summary><ul id="constant-nodes"></ul></details>
            <section id="scope-initializers" hidden>
              <h3 class="eyebrow">Graph initializers</h3>
              <details class="inspector-section" open><summary>Initializers <span id="dense-initializer-count">0</span></summary><ul id="scope-dense-initializers"></ul></details>
              <details class="inspector-section" open><summary>Sparse initializers <span id="sparse-initializer-count">0</span></summary><ul id="scope-sparse-initializers"></ul></details>
              <details class="inspector-section" open><summary>Encoded initializers <span id="encoded-initializer-count">0</span></summary><ul id="scope-encoded-initializers"></ul></details>
            </section>
            <details id="scope-functions" class="inspector-section" open hidden>
              <summary>Local functions <span id="function-count">0</span></summary><ul id="local-functions"></ul>
            </details></div>
          </section>
          <section id="metadata-panel" class="inspector-panel" aria-labelledby="metadata-title" hidden>
            <h2 id="metadata-title" class="eyebrow panel-title">Data</h2>
            <div id="metadata-content" hidden>
              <table id="metadata-analysis" aria-label="Selection metadata"></table>
              <section id="metadata-releases-section" hidden><h3>Releasable after this node</h3><ul id="metadata-releases"></ul></section>
              <section id="metadata-last-use-section" hidden><h3 title="Last reads of caller-owned values; these are not released">Last use: inputs / weights</h3><ul id="metadata-last-use"></ul></section>
              <section id="metadata-unused-section" hidden><h3>Unused outputs</h3><ul id="metadata-unused"></ul></section>
              <section><h3>Stored node metadata</h3><table id="metadata-node" aria-label="Stored node metadata"></table></section>
              <section><h3>Connected tensors (<span id="metadata-value-count">0</span>)</h3><ul id="metadata-values"></ul></section>
              <section><h3>Model metadata</h3><table id="metadata-model" aria-label="Model metadata"></table></section>
              <section><h3>Graph metadata</h3><table id="metadata-graph" aria-label="Graph metadata"></table></section>
            </div>
          </section>
          <section id="optimization-panel" class="inspector-panel" aria-labelledby="optimization-title" hidden>
            <h2 id="optimization-title" class="eyebrow panel-title">Global optimization</h2>
            <div class="fusion-selection-controls optimizer-backends" role="group" aria-label="Optimizer backend">
              <button id="optimizer-onnx-light" aria-pressed="true">onnx-light</button>
              <button id="optimizer-onnxscript" aria-pressed="false">onnxscript</button>
            </div>
            <p id="onnxscript-optimization-note" hidden>Runs ONNX Script with Python.</p>
            <div id="optimization-picker" class="fusion-picker">
              <div id="native-optimization-patterns">
              <fieldset id="optimization-pattern-group" disabled aria-label="Native pattern groups"><legend>Select pattern groups</legend></fieldset>
              <div class="fusion-selection-controls"><button id="select-optimization-patterns" disabled>Select all</button><button id="clear-optimization-patterns" disabled>Clear selection</button></div>
              <fieldset id="optimization-patterns" disabled aria-label="Optimization patterns"></fieldset>
              <p id="optimization-feedback" role="status"></p>
              </div>
              <button id="run-optimization" disabled>Run global optimization</button>
              <pre id="onnxscript-optimization-output" class="native-report" aria-label="ONNX Script output" hidden></pre>
              <p id="optimization-status" role="status"></p>
              <section id="native-optimization-history">
                <div class="fusion-selection-controls"><button id="optimization-unfuse-last" disabled>Unfuse last</button><button id="optimization-restore-original" disabled>Back to original model</button></div>
                <p>Unfuse last reverses one complete native operation. Back to original model undoes all available native operations, including other pipeline stages. ONNX Script changes cannot be undone.</p>
                <h3>Matched patterns</h3>
                <ol id="optimization-results" aria-label="Applied optimization patterns"></ol>
              </section>
            </div>
            <div id="optimization-report-section" hidden>
              <button id="show-optimization-report" disabled aria-haspopup="dialog">Show optimization report</button>
            </div>
          </section>
          <section id="fusion-panel" class="inspector-panel" aria-labelledby="fusion-title" hidden>
            <h2 id="fusion-title" class="eyebrow panel-title">Fusion patterns</h2>
            <p>Select patterns, run matching, then fuse selected candidates. Run global passes one at a time.</p>
            <details id="rewriting-scenarios" class="fusion-picker">
              <summary>Saved rewriting scenarios</summary>
              <label for="rewriting-scenario">Scenario</label><select id="rewriting-scenario" disabled></select>
              <p id="rewriting-progress" role="status">No saved rewriting scenarios in this model.</p>
              <ol id="rewriting-steps" aria-label="Stored rewriting steps"></ol>
              <div class="fusion-selection-controls"><button id="rewriting-next" disabled>Apply next pattern</button><button id="rewriting-all" disabled>Apply remaining scenario</button></div>
              <p>Scenarios are alternatives against the original graph. Unfuse current edits before switching scenarios. Each click can be undone with Unfuse.</p>
            </details>
            <details class="fusion-picker" open>
              <summary>Native patterns</summary>
              <fieldset id="fusion-pattern-group" disabled aria-label="Native pattern groups"><legend>Select pattern groups</legend></fieldset>
              <div class="fusion-selection-controls"><button id="select-patterns" disabled>Select all</button><button id="clear-patterns" disabled>Clear selection</button></div>
              <fieldset id="fusion-pattern" disabled aria-label="Native patterns" aria-describedby="fusion-feedback"></fieldset>
              <button id="preview-fusion" disabled>Run Matching</button>
              <p id="fusion-feedback" role="status"></p>
            </details>
            <details class="fusion-picker" open>
              <summary>Fusion candidates</summary>
              <div class="fusion-selection-controls"><button id="select-fusions" disabled>Select all compatible</button><button id="clear-fusions" disabled>Clear selection</button></div>
              <ul id="fusion-matches" aria-label="Fusion candidates"></ul>
              <footer><button id="apply-fusion" class="primary" disabled>Fuse selected</button></footer>
            </details>
            <details id="fusion-diagnostics" class="fusion-picker" open>
              <summary>Why did matching fail?</summary>
              <p>Expected a fusion candidate but found none? Select the operator where you expected a match, choose one of the native patterns checked above, then click Explain failed match to see the rejection reason. This does not apply a rewrite. Global passes use whole-graph previews instead.</p>
              <label for="diagnostic-pattern">Pattern</label><select id="diagnostic-pattern" disabled></select>
              <p id="diagnostic-node">Select exactly one operator in the graph.</p>
              <button id="explain-fusion" disabled>Explain failed match</button>
              <p id="diagnostic-feedback" role="status"></p>
              <section id="diagnostic-source" hidden><h3>Native source location</h3><p id="diagnostic-location" class="native-report"></p></section>
            </details>
          </section>
          <section id="open-panel" class="inspector-panel file-panel" aria-labelledby="open-title" hidden>
            <h2 id="open-title" class="eyebrow panel-title">Open model</h2>
            <div id="open-source" class="source-picker" role="group" aria-label="Model source">
              <button id="open-source-local" type="button" data-source="local" aria-pressed="true">Local file</button>
              <button id="open-source-huggingface" type="button" data-source="huggingface" aria-pressed="false">Hugging Face</button>
              <button id="open-source-backend" type="button" data-source="backend" aria-pressed="false">Backend tests</button>
            </div>
            <section id="hf-discovery" aria-labelledby="hf-search-title" hidden>
              <h3 id="hf-search-title">1. Find an ONNX model</h3>
              <p id="hf-search-help">Choose a task, then enter a model name or author and click Search (or press Enter). Leave the name blank for popular models.</p>
              <form id="hf-search-form">
                <label for="hf-task">Task</label>
                <select id="hf-task" aria-describedby="hf-search-help">${huggingFaceTasks.map(({ value, label }) => `<option value="${value}">${label}</option>`).join("")}</select>
                <label for="hf-query">Model name or author</label>
                <div class="hub-search-row"><input id="hf-query" type="search" placeholder="e.g. MiniLM, bert, whisper" autocomplete="off" aria-describedby="hf-search-help"><button class="primary" type="submit">Search</button></div>
              </form>
              <h3 id="hf-results-title">Popular ONNX models</h3>
              <p id="hf-search-status" role="status"></p>
              <ul id="hf-search-results" class="file-list" aria-label="Hugging Face ONNX repositories"></ul>
            </section>
            <details id="open-browser" hidden><summary id="open-browser-title">Or enter a repository / URL</summary></details>
            <form id="open-form">
              <label for="open-file">Model path, URL or backend test ID</label>
              <input id="open-file" type="text" required autocomplete="off" list="open-file-history"><datalist id="open-file-history"></datalist>
              <div id="open-path-options" class="path-picker-actions" role="group" aria-label="Model path selection">
                <button id="open-path-browse" type="button" aria-haspopup="dialog" aria-controls="open-path-dialog">Browse folders...</button>
                <button id="open-native-picker" type="button">VS Code file picker...</button>
              </div>
              <label for="open-password">Password (encrypted models only)</label>
              <input id="open-password" type="password" autocomplete="off">
              <progress id="open-progress" aria-label="Opening ONNX model" aria-describedby="open-progress-status" hidden></progress><p id="open-progress-status" role="status" hidden></p>
              <p id="open-error" role="alert"></p>
              <footer><button id="cancel-open" type="button">Close</button><button id="open-submit" class="primary" type="submit">Open model</button></footer>
            </form>
            ${historyControls("open")}
          </section>
          <dialog id="open-path-dialog" class="file-panel" aria-labelledby="open-path-title" aria-describedby="open-path-help">
            <h2 id="open-path-title">Choose a model path</h2>
            <p id="open-path-help">Enter any accessible directory, including outside the VS Code workspace. Select a model to fill Model path without opening it.</p>
            <div id="open-path-content">
            <form id="open-browse" class="file-browse">
              <label id="open-location-label" for="open-location">Directory</label>
              <input id="open-location" type="text" autocomplete="off" list="open-location-history"><datalist id="open-location-history"></datalist>
              <div><button type="submit">Browse</button><button id="open-parent" type="button" disabled>Parent folder</button></div>
            </form>
            <div id="open-file-list">
            <h3 id="hf-files-title" tabindex="-1" hidden>2. Choose an ONNX file</h3>
            <p id="hf-files-help" hidden>Double-click an ONNX file to download and open it. Sizes exclude external weights.</p>
            <label for="open-filter">Filter models and folders</label>
            <input id="open-filter" type="search" aria-controls="open-files">
            <ul id="open-files" class="file-list" aria-label="Models and folders"></ul>
            </div>
            </div>
            <p id="open-path-error" role="alert"></p>
            <p id="open-path-selection" role="status"></p>
            <footer><button id="cancel-open-path" type="button">Cancel</button><button id="choose-open-path" class="primary" type="button" disabled>Choose path</button></footer>
          </dialog>
          <section id="save-options" class="inspector-panel file-panel" aria-labelledby="save-title" hidden>
            <h2 id="save-title" class="eyebrow panel-title">Save model copy</h2>
            <form id="save-browse" class="file-browse">
              <label for="save-location">Destination directory</label>
              <input id="save-location" type="text" autocomplete="off" list="save-location-history"><datalist id="save-location-history"></datalist>
              <div><button type="submit">Browse</button><button id="save-parent" type="button" disabled>Parent folder</button></div>
            </form>
            <ul id="save-files" class="file-list" aria-label="Destination folders and models"></ul>
            <form id="save-form">
              <label for="save-file">Destination path</label><input id="save-file" type="text" value="model-copy.onnx" required autocomplete="off" list="save-file-history"><datalist id="save-file-history"></datalist>
              <fieldset class="save-grid">
                <legend>Weight storage and encryption</legend>
                <label><input type="radio" name="save-format" value="embedded-plain" checked>Single model or pipeline file</label>
                <label><input type="radio" name="save-format" value="embedded-encrypted">Encrypted .onnx.enc file, including all weights</label>
                <label><input type="radio" name="save-format" value="external-plain">Model or pipeline and separate weight files</label>
                <label class="unavailable"><input type="radio" name="save-format" value="external-encrypted" disabled aria-describedby="external-encryption-note">Separate encrypted weight files (unavailable)</label>
                <p id="external-encryption-note">Unavailable in onnx-light. <a href="https://github.com/xadupre/onnx-light/issues/5533">Track native support</a>.</p>
                <label><input type="radio" name="save-format" value="reuse-plain">Reuse unchanged external files</label>
                <p>For reuse, save in the source folder. Embedded weights still need saving.</p>
              </fieldset>
              <div id="external-size" hidden>
                <label for="weight-file-size">Maximum weight file size (MiB)</label>
                <input id="weight-file-size" type="number" min="1" max="8589934591" step="1" value="5120" required disabled aria-describedby="weight-size-note">
                <p id="weight-size-note">1 MiB = 1,048,576 bytes. A tensor cannot span files; choose a size large enough for the largest tensor.</p>
              </div>
              <div id="save-encryption" hidden>
                <label for="save-password">Password</label><input id="save-password" type="password" autocomplete="new-password">
                <label for="save-confirmation">Confirm password</label><input id="save-confirmation" type="password" autocomplete="new-password">
              </div>
              <label><input id="save-rewritings" type="checkbox" aria-describedby="save-rewritings-note">Include fusion rewritings</label>
              <p id="save-rewritings-note">Unchecked: save the fused graph without rewriting plans. Checked: save the original graph and one concatenated RewritingProto plan per modified model. Undo history is session-local.</p>
              <label><input id="save-overwrite" type="checkbox">Replace an existing destination copy</label>
              <progress id="save-progress" aria-label="Saving ONNX model" aria-describedby="save-progress-status" hidden></progress><p id="save-progress-status" role="status" hidden></p>
              <p id="save-options-error" role="alert"></p>
              <footer><button id="cancel-save" type="button">Close</button><button id="save-submit" class="primary" type="submit">Save copy</button></footer>
            </form>
            ${historyControls("save")}
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
    <section id="statistics-view" role="tabpanel" aria-labelledby="statistics-tab" tabindex="0" hidden>
      <h2>Model statistics</h2>
      <p>Statistics cover the complete model, regardless of the displayed subgraph, and are updated whenever this tab is opened.</p>
      <p id="statistics-empty" role="status">Open a model to see its statistics.</p>
      <div id="statistics-content" hidden>
        <h3 id="statistics-model-name"></h3>
        <dl id="statistics-summary" class="status-details"></dl>
        <section id="statistics-stages" hidden>
          <h3>Pipeline stages</h3>
          <table id="statistics-stage-table" aria-label="Pipeline stage statistics"></table>
        </section>
        <section>
          <h3>Model and external files</h3>
          <p>Physical file sizes are refreshed when this tab opens. Unavailable files remain listed with a reason.</p>
          <table id="statistics-files" aria-label="Model and external files"></table>
        </section>
        <section>
          <h3>Operator type distribution</h3>
          <table id="statistics-operators" aria-label="Operator type distribution"></table>
        </section>
      </div>
    </section>
    <section id="memory-view" role="tabpanel" aria-labelledby="memory-tab" tabindex="0" hidden>
      <h2>Memory profiling</h2>
      <p>Estimate CPU tensor memory without executing the model. Profiling covers the complete root graph, or one selected pipeline stage, independently of the displayed subgraph.</p>
      <p>One event per node in execution order, accounting for native lifetime and potential in-place reuse decisions. Function/control-flow internals and allocator overhead are not included. Scratch estimates depend on registered estimators; zero can mean unavailable.</p>
      <form id="memory-form">
        <div id="memory-stage-field" hidden><label for="memory-stage">Pipeline stage</label><select id="memory-stage"></select></div>
        <fieldset><legend>Dynamic dimensions (default: 1)</legend><div id="memory-dimensions"></div><p id="memory-no-dimensions" hidden>No dynamic dimensions to fill.</p></fieldset>
        <button id="memory-run" type="submit" disabled>Estimate memory</button>
      </form>
      <p id="memory-status" role="status">Open a model to estimate its memory.</p>
      <div id="memory-results" hidden>
        <h3 id="memory-model-name"></h3>
        <dl class="status-details"><dt>Estimated peak</dt><dd id="memory-peak"></dd><dt>Device</dt><dd id="memory-device"></dd></dl>
        <svg id="memory-chart" viewBox="0 0 1000 330" role="img" aria-labelledby="memory-chart-title memory-chart-description">
          <title id="memory-chart-title">Estimated device memory by execution event</title>
          <desc id="memory-chart-description">Step plot: one interval between nodes, two intervals during each node. Outputs and scratch are allocated before execution; releases happen after execution. Rectangles span each node's two-interval execution window, with a gap between nodes. Highlighted rectangles identify common compute-heavy operations, not measured timings. Exact values and SDK decisions are in the table below.</desc>
          <path class="memory-axis" d="M60,20V240H980"></path>
          <text id="memory-chart-peak" x="60" y="16"></text><text x="10" y="244">0</text>
          <path id="memory-node-intervals" class="memory-node-interval"></path>
          <path id="memory-operator-intervals" class="memory-operator-interval"></path>
          <g id="memory-node-numbers"></g>
          <g id="memory-operator-labels"></g>
          <path id="memory-curve"></path>
        </svg>
        <p id="memory-final-status" role="status" hidden>Final post-release memory is unavailable in onnx-light. The curve stops before that unknown value.</p>
        <h3>Estimated execution events</h3>
        <p id="memory-peaks-count"></p>
        <label><input id="memory-peaks-only" type="checkbox"> Local peaks only</label>
        <div class="memory-table"><table><thead><tr><th scope="col">Event</th><th scope="col">Node</th><th scope="col">Operator</th><th scope="col">Before allocation</th><th scope="col">New outputs</th><th scope="col">Scratch</th><th scope="col">During execution</th><th scope="col">After release</th><th scope="col">In-place</th><th scope="col">Release</th><th scope="col">Release shape</th><th scope="col">Tensor shapes</th></tr></thead><tbody id="memory-events"></tbody></table></div>
        <nav aria-label="Memory events"><button id="memory-previous" type="button" disabled>Previous</button><span id="memory-range" role="status"></span><button id="memory-next" type="button" disabled>Next</button></nav>
      </div>
    </section>
    <section id="tensors-view" role="tabpanel" aria-labelledby="tensors-tab" tabindex="0" hidden>
      <h2>Tensors</h2>
      <p>Dense, sparse and encoded initializers and constants in the displayed graph. Inputs and intermediate tensors without stored data are not listed. Highlighted rows belong to the selected operator or tensor. The model is not executed.</p>
      <details id="tensor-distributions" open>
        <summary>Stored tensor distributions</summary>
        <p>All stored tensors in this graph, grouped by type and shape. Logical sizes are dense-equivalent data sizes, not file sizes or execution peaks. Sparse and encoded tensors are counted separately. Unknown sizes remain explicit. Filters below affect only the tensor list.</p>
        <div id="tensor-distribution-groups"></div>
      </details>
      <label for="tensor-list-search">Filter by name, category or consumer</label>
      <input id="tensor-list-search" type="search" autocomplete="off">
      <label for="tensor-list-type">Filter by tensor type</label>
      <select id="tensor-list-type"><option value="">All types</option></select>
      <p id="tensor-list-status" role="status">Open a model to browse tensors.</p>
      <div class="tensor-list-table"><table aria-label="Graph tensors"><thead><tr><th scope="col">Name</th><th scope="col">Category</th><th scope="col">Type</th><th scope="col">Shape</th><th scope="col">Consumers (operator and name)</th><th scope="col">Values / statistics</th></tr></thead><tbody id="tensor-list-rows"></tbody></table></div>
      <nav aria-label="Tensor pages"><button id="tensor-list-previous" type="button" disabled>Previous</button><button id="tensor-list-next" type="button" disabled>Next</button></nav>
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
    <section id="settings-view" role="tabpanel" aria-labelledby="settings-tab" tabindex="0" hidden>
      <h2>Settings</h2>
      <p>Settings are saved for this VS Code user. Existing downloads are not moved when the directory changes.</p>
      <form id="application-settings-form">
        <label for="settings-download">Default download directory (Hugging Face cache)</label>
        <input id="settings-download" type="text" required autocomplete="off">
        <label for="settings-python">Python executable</label>
        <input id="settings-python" type="text" required autocomplete="off">
        <p>Python is not needed to view models. Save an executable name or absolute path to inspect optional packages on the extension host.</p>
        <button type="submit">Save paths</button>
      </form>
      <p id="settings-feedback" role="status"></p>
      <section id="settings-stored-data" aria-labelledby="settings-stored-title">
        <h3 id="settings-stored-title">Stored data and histories</h3>
        <p>Histories and preferences use VS Code globalState, shared across workspaces for this VS Code user. Removing entries only forgets suggestions: downloaded files and saved models are never deleted.</p>
        <dl class="status-details">
          <dt>Download cache</dt><dd id="settings-cache-download">Waiting for extension host</dd>
          <dt>Default model saves</dt><dd id="settings-cache-saves">Waiting for extension host</dd>
        </dl>
        <h4>Open history (<span id="settings-open-count">0</span>)</h4>
        <button id="settings-clear-open" type="button" disabled>Clear Open history</button>
        <ul id="settings-open-history" aria-label="Open history"></ul>
        <h4>Save history (<span id="settings-save-count">0</span>)</h4>
        <button id="settings-clear-save" type="button" disabled>Clear Save history</button>
        <ul id="settings-save-history" aria-label="Save history"></ul>
        <h4>Conversion history (<span id="settings-export-count">0</span>)</h4>
        <p>Last conversion tool: <span id="settings-exporter">None</span></p>
        <button id="settings-clear-export" type="button" disabled>Clear conversion history</button>
        <ul id="settings-export-history" aria-label="Conversion history"></ul>
        <p id="settings-history-feedback" role="status"></p>
      </section>
      <h3>Viewer preferences</h3>
      <button id="settings-reset-viewer" type="button">Reset viewer preferences</button>
      <p>Restores the default sidebar width and panels. Download and Python paths are unchanged.</p>
      <label for="settings-width">Sidebar width (pixels; blank uses the default)</label>
      <input id="settings-width" type="number" min="200" step="1">
      <div id="settings-panels">
        <label><input id="settings-operator" type="checkbox" checked> Operator inspector</label>
        <label><input id="settings-find" type="checkbox" checked> Find panel</label>
        <label><input id="settings-metadata" type="checkbox"> Data panel</label>
      </div>
      <h3>Application and native build</h3>
      <dl class="status-details">
        <dt>ONNX Light Viewer</dt><dd id="settings-app-version">Waiting for extension host</dd>
        <dt>Compiled with onnx-light C++</dt><dd id="settings-native-version">Waiting for extension host</dd>
        <dt>Extension host platform</dt><dd id="settings-platform">Waiting for extension host</dd>
      </dl>
      <p>The bundled native addon uses the onnx-light C++ release above, independently of any Python installation.</p>
      <h3>Optional Python packages</h3>
      <p>Checks distribution metadata and source versions on PYTHONPATH without importing model libraries or running startup hooks. Nothing is installed. Requires a trusted workspace.</p>
      <button id="settings-inspect-python" type="button" disabled>Refresh package versions</button>
      <p id="settings-python-status" role="status">Not checked</p>
      <dl id="settings-packages" class="status-details"></dl>
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
          <tr><th scope="row"><kbd>Ctrl/Cmd + Home</kbd></th><td>Go to the top of the model without changing zoom.</td></tr>
          <tr><th scope="row"><kbd>Ctrl/Cmd + End</kbd></th><td>Go to the bottom of the model without changing zoom.</td></tr>
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
          <tr><th scope="row">Save</th><td>Choose embedded or external weights and plain or encrypted output. Embedded encrypted output protects all weights. External plain output asks for the maximum weight file size; external encrypted output awaits native support.</td></tr>
          <tr><th scope="row">Op / Data / Find / Fuse</th><td>Show or hide each sidebar panel independently. Visible panels stack vertically.</td></tr>
          <tr><th scope="row"><kbd>Enter</kbd> or <kbd>Space</kbd></th><td>Activate a focused button or subgraph link.</td></tr>
          <tr><th scope="row"><kbd>Left</kbd> / <kbd>Right</kbd> arrow</th><td>Switch tabs when a tab header has focus.</td></tr>
          <tr><th scope="row"><kbd>Home</kbd> / <kbd>End</kbd></th><td>Select the first / last tab when a tab header has focus.</td></tr>
          <tr><th scope="row">+ / − buttons</th><td>Zoom in / out.</td></tr>
          <tr><th scope="row">Fit</th><td>Fit the whole graph in the view.</td></tr>
          <tr><th scope="row">Top / Bottom</th><td>Go to the top / bottom of the model without changing zoom. Keyboard shortcuts work while focus is in the graph canvas.</td></tr>
          <tr><th scope="row">Zoom to readable size</th><td>Zoom to 100%, centered on the inspected node.</td></tr>
          <tr><th scope="row">Refresh positions</th><td>Reset the node layout.</td></tr>
          <tr><th scope="row">Subgraph link / breadcrumbs</th><td>Open a subgraph / return to a parent graph.</td></tr>
          <tr><th scope="row">Doc</th><td>Open operator documentation from the inspector. Use Close or <kbd>Escape</kbd> to dismiss it.</td></tr>
        </tbody>
      </table>
    </section>
    <section id="repositories-view" role="tabpanel" aria-labelledby="repositories-tab" tabindex="0" hidden>
      <h2>ONNX ecosystem</h2>
      <p>Explore ONNX projects and exporters, with the packages and ecosystems they specialize in.</p>
      <h3>Official projects</h3>
      <ul class="repository-list">
        <li>
          <div class="repository-heading">
            <a href="https://github.com/microsoft/Olive"><svg viewBox="0 0 24 24" width="28" height="28" aria-hidden="true"><path d="M5 21 19 3M9 16C2 15 2 9 2 9s8-1 7 7ZM14 10c-1-7 7-7 7-7s1 6-7 7Z"/></svg>Olive</a>
            ${documentationLink("Olive", "https://microsoft.github.io/Olive/")}
          </div>
          <img class="repository-logo" src="${resource("repositories/olive.svg")}" alt="" width="96" height="64">
          <p>Export Hugging Face checkpoints to ONNX, optimize models or quantize local ONNX files with olive-ai in Convert.</p>
          <p class="repository-specialization"><strong>Specialized for:</strong> ONNX optimization and quantization with olive-ai.</p>
        </li>
        <li>
          <div class="repository-heading">
            <a href="https://github.com/microsoft/olive-recipes"><svg viewBox="0 0 24 24" width="28" height="28" aria-hidden="true"><path d="M4 3h16v18H4ZM8 7h8M8 12h8M8 17h5"/></svg>Olive recipes</a>
            ${documentationLink("Olive recipes", "https://github.com/microsoft/olive-recipes#readme")}
          </div>
          <img class="repository-logo" src="${resource("repositories/olive.svg")}" alt="" width="96" height="64">
          <p>Model-specific Olive export and optimization workflows. Convert caches the recipe list by Git commit and shows recipe-specific Olive requirements.</p>
          <p class="repository-specialization"><strong>Specialized for:</strong> Model-specific export workflows with olive-ai.</p>
        </li>
        <li>
          <div class="repository-heading">
          <a href="https://github.com/onnx/onnx"><svg viewBox="0 0 24 24" width="28" height="28" aria-hidden="true"><circle cx="5" cy="12" r="3"/><circle cx="18" cy="5" r="3"/><circle cx="18" cy="19" r="3"/><path d="m8 11 7-5M8 13l7 5"/></svg>ONNX</a>
          ${documentationLink("ONNX", "https://onnx.ai/onnx/")}
          </div>
          <img class="repository-logo" src="${resource("repositories/onnx.png")}" alt="" width="96" height="64">
          <p>An open standard for representing machine learning models, with a common model format and operator definitions.</p>
          <p class="repository-specialization"><strong>Specialized for:</strong> Framework-independent model interchange with the onnx package.</p>
        </li>
        <li>
          <div class="repository-heading">
          <a href="https://github.com/microsoft/onnxruntime"><svg viewBox="0 0 24 24" width="28" height="28" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="m10 7 7 5-7 5Z"/></svg>ONNX Runtime</a>
          ${documentationLink("ONNX Runtime", "https://onnxruntime.ai/docs/")}
          </div>
          <img class="repository-logo" src="${resource("repositories/onnxruntime.png")}" alt="" width="96" height="64">
          <p>A cross-platform engine for accelerating machine learning inference and training across hardware backends.</p>
          <p class="repository-specialization"><strong>Specialized for:</strong> ONNX model execution with onnxruntime.</p>
        </li>
        <li>
          <div class="repository-heading">
          <a href="https://github.com/microsoft/onnxruntime-extensions"><svg viewBox="0 0 24 24" width="28" height="28" aria-hidden="true"><path d="M8 4H4v16h16v-7h-4a3 3 0 1 1 0-6h4V4h-7V3a3 3 0 0 0-6 0v1"/></svg>ONNX Runtime Extensions</a>
          ${documentationLink("ONNX Runtime Extensions", "https://onnxruntime.ai/docs/extensions/")}
          </div>
          <img class="repository-logo" src="${resource("repositories/onnxruntime.png")}" alt="" width="96" height="64">
          <p>Custom operators and tools for adding preprocessing and postprocessing to ONNX models.</p>
          <p class="repository-specialization"><strong>Specialized for:</strong> onnxruntime preprocessing and postprocessing with onnxruntime-extensions.</p>
        </li>
        <li>
          <div class="repository-heading">
          <a href="https://github.com/microsoft/onnxruntime-genai"><svg viewBox="0 0 24 24" width="28" height="28" aria-hidden="true"><path d="m12 3 3 6 6 3-6 3-3 6-3-6-6-3 6-3ZM19 2v4M17 4h4"/></svg>ONNX Runtime GenAI</a>
          ${documentationLink("ONNX Runtime GenAI", "https://onnxruntime.ai/docs/genai/")}
          </div>
          <img class="repository-logo" src="${resource("repositories/onnxruntime.png")}" alt="" width="96" height="64">
          <p>Generative AI APIs for ONNX Runtime, including tokenization, generation loops, sampling and beam search.</p>
          <p class="repository-specialization"><strong>Specialized for:</strong> Generative model execution with onnxruntime-genai.</p>
        </li>
      </ul>
      <h3>onnx-light ecosystem</h3>
      <ul class="repository-list">
        <li>
          <div class="repository-heading">
          <a href="https://github.com/xadupre/onnx-light"><svg viewBox="0 0 24 24" width="28" height="28" aria-hidden="true"><path d="m14 2-9 12h6l-1 8 9-13h-6Z"/></svg>onnx-light</a>
          ${documentationLink("onnx-light", "https://sdpython.github.io/doc/onnx-light/dev/index.html")}
          </div>
          <img class="repository-logo" src="${resource("repositories/onnx-light.svg")}" alt="" width="96" height="64">
          <p>A lightweight C++ library with Python bindings for manipulating ONNX models, shape inference and graph optimization.</p>
          <p class="repository-specialization"><strong>Specialized for:</strong> ONNX model I/O and graph analysis with onnx-light.</p>
        </li>
        <li>
          <div class="repository-heading">
          <a href="https://github.com/xadupre/onnx-light-cpu"><svg viewBox="0 0 24 24" width="28" height="28" aria-hidden="true"><rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 2v4m6-4v4M9 18v4m6-4v4M2 9h4m-4 6h4m12-6h4m-4 6h4M10 10h4v4h-4Z"/></svg>onnx-light-cpu</a>
          ${documentationLink("onnx-light-cpu", "https://xadupre.github.io/docs/onnx-light-cpu/")}
          </div>
          <img class="repository-logo" src="${resource("repositories/onnx-light-cpu.svg")}" alt="" width="96" height="64">
          <p>CPU execution kernels for running ONNX models with onnx-light.</p>
          <p class="repository-specialization"><strong>Specialized for:</strong> onnx-light CPU execution.</p>
        </li>
        <li>
          <div class="repository-heading">
          <a href="https://github.com/xadupre/onnx-light-ort"><svg viewBox="0 0 24 24" width="28" height="28" aria-hidden="true"><path d="M8 4v5m8-5v5M5 9h14v3a7 7 0 0 1-14 0ZM12 19v3"/></svg>onnx-light-ort</a>
          ${documentationLink("onnx-light-ort", "https://xadupre.github.io/onnx-light-ort/")}
          </div>
          <img class="repository-logo" src="${resource("repositories/onnx-light-ort.svg")}" alt="" width="96" height="64">
          <p>Custom-operator interoperability between onnx-light and ONNX Runtime, in both directions.</p>
          <p class="repository-specialization"><strong>Specialized for:</strong> onnx-light and onnxruntime interoperability.</p>
        </li>
        <li>
          <div class="repository-heading">
          <a href="https://github.com/xadupre/onnx-light-viewer"><img src="${resource("logo.svg")}" alt="" width="28" height="28">onnx-light-viewer</a>
          ${documentationLink("onnx-light-viewer", "https://xadupre.github.io/docs/onnx-light-viewer/")}
          </div>
          <img class="repository-logo" src="${resource("logo.svg")}" alt="" width="96" height="64">
          <p>An interactive ONNX graph viewer for VS Code, powered by onnx-light.</p>
          <p class="repository-specialization"><strong>Specialized for:</strong> VS Code model inspection with onnx-light.</p>
        </li>
      </ul>
      <h3>Exporters</h3>
      <p>Use Convert in the right panel to run an installed mbext, modelbuilder or Mobius exporter and open its ONNX output. Supported models and target runtimes vary by exporter.</p>
      <ul class="repository-list">
        <li>
          <div class="repository-heading">
          <a href="https://github.com/onnxruntime/mobius"><svg viewBox="0 0 24 24" width="28" height="28" aria-hidden="true"><path d="M14 3h7v7M21 3l-9 9M10 3H3v18h18v-7"/></svg>Mobius</a>
          ${documentationLink("Mobius", "https://onnxruntime.github.io/mobius/")}
          </div>
          <img class="repository-logo" src="${resource("repositories/onnxruntime.png")}" alt="" width="96" height="64">
          <p>Builds ONNX graphs with ONNX Script and applies pretrained weights rather than tracing a PyTorch model.</p>
          <p class="repository-specialization"><strong>Specialized for:</strong> Hugging Face Transformers and Diffusers models, including ONNX Runtime GenAI workflows.</p>
        </li>
        <li>
          <div class="repository-heading">
          <a href="https://github.com/microsoft/onnxruntime-genai/tree/main/src/python/py/models"><svg viewBox="0 0 24 24" width="28" height="28" aria-hidden="true"><path d="M14 3h7v7M21 3l-9 9M10 3H3v18h18v-7"/></svg>modelbuilder</a>
          ${documentationLink("modelbuilder", "https://onnxruntime.ai/docs/genai/howto/build-model.html")}
          </div>
          <img class="repository-logo" src="${resource("repositories/onnxruntime.png")}" alt="" width="96" height="64">
          <p>The ONNX Runtime GenAI model builder constructs optimized and quantized ONNX graphs for supported generative architectures.</p>
          <p class="repository-specialization"><strong>Specialized for:</strong> Hugging Face Transformers models targeting onnxruntime-genai.</p>
        </li>
        <li>
          <div class="repository-heading">
          <a href="https://github.com/xadupre/mbext"><svg viewBox="0 0 24 24" width="28" height="28" aria-hidden="true"><path d="M14 3h7v7M21 3l-9 9M10 3H3v18h18v-7"/></svg>mbext</a>
          ${documentationLink("mbext", "https://xadupre.github.io/docs/mbext/index.html")}
          </div>
          <img class="repository-logo" src="${resource("repositories/mbext.svg")}" alt="" width="96" height="64">
          <p>A rewrite using onnx-light for faster ONNX model construction and export.</p>
          <p class="repository-specialization"><strong>Specialized for:</strong> Hugging Face Transformers models targeting onnxruntime-genai, using onnx-light through the modelbuilder package.</p>
        </li>
        <li>
          <div class="repository-heading">
          <a href="https://github.com/xadupre/yet-another-onnx-builder"><svg viewBox="0 0 24 24" width="28" height="28" aria-hidden="true"><path d="M14 3h7v7M21 3l-9 9M10 3H3v18h18v-7"/></svg>yet-another-onnx-builder</a>
          ${documentationLink("yet-another-onnx-builder", "https://xadupre.github.io/docs/yet-another-onnx-builder/index.html")}
          </div>
          <img class="repository-logo" src="${resource("repositories/yet-another-onnx-builder.svg")}" alt="" width="96" height="64">
          <p>Uses onnx-light for ONNX graph construction, optimization and shape inference, with the yobx.to_onnx conversion API across multiple source libraries.</p>
          <p class="repository-specialization"><strong>Specialized for:</strong> PyTorch, TensorFlow/Keras/LiteRT, scikit-learn, scikit-survival, category_encoders, imbalanced-learn, statsmodels, XGBoost and LightGBM, plus NumPy function tracing, using yobx and onnx-light. SQL queries, pandas/Polars-style DataFrame transformations and JAX support are in progress.</p>
        </li>
        <li>
          <div class="repository-heading">
          <a href="https://github.com/onnx/sklearn-onnx"><svg viewBox="0 0 24 24" width="28" height="28" aria-hidden="true"><path d="M14 3h7v7M21 3l-9 9M10 3H3v18h18v-7"/></svg>sklearn-onnx</a>
          ${documentationLink("sklearn-onnx", "https://onnx.ai/sklearn-onnx/")}
          </div>
          <img class="repository-logo" src="${resource("repositories/sklearn-onnx.png")}" alt="" width="96" height="64">
          <p>Converts supported scikit-learn estimators and pipelines to ONNX through the skl2onnx package.</p>
          <p class="repository-specialization"><strong>Specialized for:</strong> scikit-learn models and pipelines.</p>
        </li>
        <li>
          <div class="repository-heading">
          <a href="https://github.com/onnx/tensorflow-onnx"><svg viewBox="0 0 24 24" width="28" height="28" aria-hidden="true"><path d="M14 3h7v7M21 3l-9 9M10 3H3v18h18v-7"/></svg>tensorflow-onnx</a>
          ${documentationLink("tensorflow-onnx", "https://github.com/onnx/tensorflow-onnx/blob/main/README.md")}
          </div>
          <img class="repository-logo" src="${resource("repositories/onnx.png")}" alt="" width="96" height="64">
          <p>Converts TensorFlow, Keras and TFLite models to ONNX through tf2onnx, with experimental TensorFlow.js support.</p>
          <p class="repository-specialization"><strong>Specialized for:</strong> TensorFlow, Keras, TFLite and TensorFlow.js.</p>
        </li>
        <li>
          <div class="repository-heading">
          <a href="https://github.com/onnx/onnxmltools"><svg viewBox="0 0 24 24" width="28" height="28" aria-hidden="true"><path d="M14 3h7v7M21 3l-9 9M10 3H3v18h18v-7"/></svg>onnxmltools</a>
          ${documentationLink("onnxmltools", "https://onnx.ai/onnxmltools/")}
          </div>
          <img class="repository-logo" src="${resource("repositories/onnxmltools.png")}" alt="" width="96" height="64">
          <p>Converts models from multiple machine-learning toolkits to ONNX, including boosted-tree libraries.</p>
          <p class="repository-specialization"><strong>Specialized for:</strong> XGBoost, LightGBM, CatBoost, Core ML and other ML toolkits.</p>
        </li>
      </ul>
    </section>
  </main>
  <dialog id="operator-doc" aria-labelledby="doc-title">
    <header class="inspector-heading"><h2 id="doc-title"></h2><button id="close-doc" autofocus>Close</button></header>
    <p id="doc-version"></p><div id="doc-body"></div>
  </dialog>
  <dialog id="tensor-statistics-dialog" aria-labelledby="tensor-statistics-title">
    <header class="inspector-heading"><h2 id="tensor-statistics-title">Tensor statistics</h2><button id="close-tensor-statistics" autofocus>Close</button></header>
    <p>Stored tensor data only. The model is not executed; intermediate values may be unavailable.</p>
    <div id="tensor-statistics-picker">
      <label for="tensor-statistics-name">Tensor name</label>
      <input id="tensor-statistics-name" list="tensor-statistics-names" autocomplete="off"><datalist id="tensor-statistics-names"></datalist>
      <div><button id="tensor-statistics-previous">Previous tensor</button><span id="tensor-statistics-position"></span><button id="tensor-statistics-next">Next tensor</button></div>
    </div>
    <p id="tensor-statistics-status" role="status"></p>
    <progress id="tensor-statistics-progress" aria-label="Loading tensor data" aria-describedby="tensor-statistics-status" hidden></progress>
    <div id="tensor-statistics-content"></div>
  </dialog>
  <dialog id="optimization-report" aria-labelledby="optimization-report-title">
    <header class="inspector-heading"><h2 id="optimization-report-title">Global optimization report</h2><button id="close-optimization-report" autofocus>Close</button></header>
    <p id="optimization-report-scope"></p>
    <p>Counters include nested graphs. Phase timings marked root exclude them. All timings are in milliseconds. Full native operation elapsed is measured independently, including native preparation, queueing and final canonicalization, but excluding graph transfer and rendering. Native phase timings cover only the instrumented optimizer phases.</p>
    <div id="optimization-report-body"></div>
    <p>Each subgraph's counters and elapsed time include its descendants; subgraph rows are not additive.</p>
  </dialog>
  <script nonce="${nonce}" src="${resource("vendor/markdown-it.min.js")}"></script>
  <script nonce="${nonce}" src="${resource("sidebar.js")}"></script>
  <script nonce="${nonce}" src="${resource("file-panels.js")}"></script>
  <script nonce="${nonce}" src="${resource("export-panel.js")}"></script>
  <script nonce="${nonce}" src="${resource("inspector.js")}"></script>
  <script nonce="${nonce}" src="${resource("metadata-panel.js")}"></script>
  <script nonce="${nonce}" src="${resource("fusion-panel.js")}"></script>
  <script nonce="${nonce}" src="${resource("settings-panel.js")}"></script>
  <script nonce="${nonce}" src="${resource("memory-panel.js")}"></script>
  <script nonce="${nonce}" src="${resource("tensor-list-panel.js")}"></script>
  <script nonce="${nonce}" src="${resource("tensor-statistics-panel.js")}"></script>
  <script nonce="${nonce}" src="${resource("vendor/dagre.min.js")}"></script>
  <script nonce="${nonce}" src="${resource("graph-layout.js")}"></script>
  <script nonce="${nonce}" src="${resource("viewer.js")}"></script>
</body>
</html>`;
}
