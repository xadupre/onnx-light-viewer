function createFusionPanel({
  byId,
  postMessage,
  requestAction,
  revealPanel,
  fit,
  canFuse,
  canSelectRewriting,
  clearSelection,
  selectedNode,
  selectedNodes,
  focusNodes,
}) {
  let fusionGraph;
  let fusionRequest = 0;
  let fusionCompletion;
  let fusionPreview;
  let fusionBusy = false;
  let undoBusy = false;
  let optimizationSelection;
  let optimizationReport;
  let optimizationBackend = "onnx-light";
  let selectedOptimizationResult = false;
  let optimizationHistory = [];
  let pendingRewritings = [];
  let undoCount = 0;
  let restoreRequest;
  let restoreNotice = "";
  let scenarios = [];
  let rewritingState = { names: [] };
  let selectedStep;
  let diagnosticRequest = 0;
  let diagnosticBusy = false;
  let diagnosticNode;
  let diagnosticResult;
  let nativePatterns = new Set();
  let optimizationFinished = false;
  let recordingFromRewritings = false;
  let appendOne = false;
  let creatingPattern = false;
  let selectionCatalogLoaded = false;
  let functionBodyRequest = 0;
  let restoreFunctionEditor = false;
  let libraryRevision = 0;
  const selectionPatterns = new Map();
  const selectionPatternDrafts = new Map();
  const partitionPatterns = ["RepeatedLayer", "PartitionGroup"];
  function canOpenSelectionPattern() {
    return canFuse() && !fusionBusy && selectedNodes().length > 0;
  }
  function openSelectionPattern() {
    if (!canOpenSelectionPattern()) return false;
    selectionChanged();
    byId("selection-pattern-dialog").showModal();
    if (!selectionCatalogLoaded) {
      fusionControls(true);
      byId("selection-pattern-status").textContent =
        "Loading created patterns…";
      requestAction("fusion-patterns", "List selection patterns", {
        requestId: ++fusionRequest,
      });
    } else {
      const pattern = selectionPatterns.get(
        byId("selection-pattern-picker").value,
      );
      if (pattern?.functionIndex !== undefined && pattern.text === undefined)
        showSelectionPattern();
    }
    return true;
  }
  function selectionPatternControls() {
    const count = selectedNodes().length;
    const name = byId("selection-pattern-name").value.trim();
    const modelFunction = fusionGraph?.localFunctions?.some(
      (fn) => fn.functionName === name,
    );
    const nameExists =
      selectionPatterns.has(name) || nativePatterns.has(name) || modelFunction;
    byId("selection-pattern-name").setCustomValidity(
      nameExists ? `Pattern name already exists: ${name}.` : "",
    );
    byId("selection-pattern-name").setAttribute(
      "aria-invalid",
      String(nameExists),
    );
    const feedback = byId("selection-pattern-name-feedback");
    feedback.hidden = !nameExists;
    feedback.textContent = nameExists
      ? `Pattern name already exists: ${name}.${modelFunction ? " The loaded ONNX model already contains this function." : ""} Choose another name to create a new pattern.${selectionPatterns.has(name) || modelFunction ? " To edit it, select it under Patterns / model functions." : ""}`
      : "";
    byId("open-selection-pattern").disabled = !canOpenSelectionPattern();
    byId("create-function-pattern").disabled =
      fusionBusy || !canFuse() || !count || !name || nameExists;
    byId("selection-pattern-name").disabled = fusionBusy || !canFuse();
    const pattern = selectionPatterns.get(
      byId("selection-pattern-picker").value,
    );
    const bodyUnloaded =
      pattern?.functionIndex !== undefined && pattern.text === undefined;
    for (const id of [
      "selection-pattern-text",
      "selection-pattern-replacement",
    ])
      byId(id).disabled = fusionBusy || !canFuse() || bodyUnloaded;
    for (const format of ["text", "replacement"])
      byId(`copy-selection-pattern-${format}`).disabled = bodyUnloaded;
    byId("selection-pattern-picker").disabled = fusionBusy;
    byId("update-function-pattern").disabled =
      fusionBusy ||
      !canFuse() ||
      !pattern ||
      bodyUnloaded ||
      (byId("selection-pattern-text").value === pattern.text &&
        byId("selection-pattern-replacement").value === pattern.replacement);
    byId("revert-function-pattern").disabled =
      fusionBusy ||
      !canFuse() ||
      !pattern ||
      bodyUnloaded ||
      !selectionPatternDrafts.has(byId("selection-pattern-picker").value);
  }
  function showSelectionPattern() {
    const requestId = ++functionBodyRequest;
    const pattern = selectionPatterns.get(
      byId("selection-pattern-picker").value,
    );
    const draft = selectionPatternDrafts.get(
      byId("selection-pattern-picker").value,
    );
    byId("selection-pattern-result").hidden = !pattern;
    byId("selection-pattern-text").value = draft?.text ?? pattern?.text ?? "";
    byId("selection-pattern-replacement").value =
      draft?.replacement ?? pattern?.replacement ?? "";
    const modelFunction = pattern?.functionIndex !== undefined;
    byId("selection-pattern-match").hidden = modelFunction;
    byId("selection-pattern-replacement-label").textContent = modelFunction
      ? "Function body"
      : "Replacement";
    byId("selection-pattern-edit-help").textContent = modelFunction
      ? "Edit this ONNX function's body without changing its name, domain, overload or input/output/attribute interface. Apply edits updates all its calls in memory; Save keeps the change."
      : "Edit the match and replacement functions using onnx-light textparser syntax, then apply the definition before matching or optimizing.";
    selectionPatternControls();
    if (
      modelFunction &&
      pattern.text === undefined &&
      byId("selection-pattern-dialog").open
    ) {
      byId("selection-pattern-status").textContent = "Loading function body…";
      requestAction("load-model-function", "Load ONNX function", {
        requestId,
        functionIndex: pattern.functionIndex,
      });
    }
  }
  function diagnosticControls() {
    const picker = byId("diagnostic-pattern");
    picker.disabled = fusionBusy || !canFuse() || !picker.options.length;
    const diagnostics = byId("fusion-diagnostics");
    diagnostics.setAttribute("aria-disabled", String(picker.disabled));
    const summary = diagnostics.querySelector("summary");
    summary.tabIndex = picker.disabled ? -1 : 0;
    if (picker.disabled) diagnostics.open = false;
    byId("explain-fusion").disabled =
      picker.disabled || diagnosticBusy || !diagnosticNode;
    byId("fusion-diagnostics").setAttribute(
      "aria-busy",
      String(diagnosticBusy),
    );
    byId("diagnostic-node").textContent = diagnosticNode
      ? `Node #${diagnosticNode.id.slice(5)}: ${diagnosticNode.name || diagnosticNode.opType} (${diagnosticNode.opType})`
      : "Select exactly one operator in the graph.";
    byId("diagnostic-feedback").textContent = diagnosticBusy
      ? "Checking why this pattern did not match the selected node…"
      : (diagnosticResult ??
        (!picker.options.length
          ? "Select one or more native patterns above to investigate a failed match."
          : !diagnosticNode
            ? "Select exactly one operator in the graph where you expected a match."
            : `Click Explain failed match to find why ${picker.value} did not match this node.`));
  }
  function clearDiagnostic() {
    diagnosticRequest++;
    if (diagnosticBusy) postMessage({ type: "cancel-fusion-diagnostic" });
    diagnosticBusy = false;
    diagnosticResult = undefined;
    byId("diagnostic-source").hidden = true;
    byId("diagnostic-location").textContent = "";
    diagnosticControls();
  }
  function selectionChanged() {
    selectionPatternControls();
    const count = selectedNodes().length;
    if (!fusionBusy)
      byId("selection-pattern-status").textContent = count
        ? `${count} operator${count === 1 ? "" : "s"} selected. Create a pattern without changing the model.`
        : "Select one or more operators in the graph.";
    const node = selectedNode();
    if (node?.id === diagnosticNode?.id) return;
    diagnosticNode = node;
    clearDiagnostic();
  }
  function updateDiagnosticPatterns() {
    const picker = byId("diagnostic-pattern");
    const previous = picker.value;
    picker.replaceChildren();
    const names = [
      ...byId("fusion-pattern").querySelectorAll("input[data-pattern]:checked"),
    ]
      .map((input) => input.value)
      .filter((name) => nativePatterns.has(name))
      .sort();
    for (const name of names) {
      const option = document.createElement("option");
      option.value = option.textContent = name;
      picker.append(option);
    }
    if ([...picker.options].some((option) => option.value === previous))
      picker.value = previous;
    clearDiagnostic();
  }
  function showOptimizationReport(report) {
    optimizationReport = report;
    if (byId("optimization-report").open) byId("optimization-report").close();
    byId("optimization-report-body").replaceChildren();
    byId("optimization-report-scope").textContent = "";
    byId("optimization-report-section").hidden = !report;
    byId("show-optimization-report").disabled = fusionBusy || !report;
  }
  function nativeSourceLocation(file, line) {
    const location = `${file}${line ? `:${line}` : ""}`;
    const normalized = file.replaceAll("\\", "/");
    const marker = normalized.lastIndexOf("/onnx_light/");
    const path = marker >= 0 ? normalized.slice(marker + 1) : normalized;
    const parts = path.split("/");
    if (
      !path.startsWith("onnx_light/") ||
      parts.some((part) => !part || part === "." || part === "..")
    )
      return document.createTextNode(location);
    const version = document.body.dataset.onnxLightVersion;
    const link = document.createElement("a");
    link.className = "native-source-link";
    link.href = `https://github.com/xadupre/onnx-light/blob/${encodeURIComponent(version)}/${parts.map(encodeURIComponent).join("/")}${line ? `#L${line}` : ""}`;
    link.textContent = link.href;
    link.title = location;
    return link;
  }
  function reportTable(id, title, headers, rows, emptyMessage) {
    const table = document.createElement("table");
    table.id = id;
    const caption = table.createCaption();
    caption.textContent = title;
    const heading = table.createTHead().insertRow();
    const renderedRows = [];
    for (const [index, name] of headers.entries()) {
      const cell = document.createElement("th");
      cell.scope = "col";
      cell.setAttribute("aria-sort", "none");
      const button = document.createElement("button");
      button.textContent = name;
      button.disabled = !rows.length;
      button.addEventListener("click", () => {
        const descending = cell.getAttribute("aria-sort") === "ascending";
        for (const header of heading.cells)
          header.setAttribute("aria-sort", "none");
        cell.setAttribute("aria-sort", descending ? "descending" : "ascending");
        renderedRows.sort((a, b) => {
          const left = a.values[index];
          const right = b.values[index];
          const comparison =
            typeof left === "number" && typeof right === "number"
              ? left - right
              : String(left).localeCompare(String(right));
          return descending ? -comparison : comparison;
        });
        body.append(...renderedRows.map((item) => item.row));
      });
      cell.append(button);
      heading.append(cell);
    }
    const body = table.createTBody();
    for (const values of rows) {
      const row = body.insertRow();
      for (const [index, value] of values.entries()) {
        const cell = document.createElement(index ? "td" : "th");
        if (!index) cell.scope = "row";
        if (value instanceof Node) cell.append(value);
        else cell.textContent = String(value?.text ?? value);
        row.append(cell);
      }
      renderedRows.push({
        row,
        values: values.map((value) =>
          value instanceof Node
            ? value.textContent
            : (value?.sortValue ?? value),
        ),
      });
    }
    if (!rows.length) {
      const cell = body.insertRow().insertCell();
      cell.colSpan = headers.length;
      cell.textContent = emptyMessage;
    }
    const item = document.createElement("li");
    const link = document.createElement("a");
    link.href = `#${id}`;
    link.textContent = title;
    link.addEventListener("click", (event) => {
      event.preventDefault();
      table.scrollIntoView({ block: "start" });
    });
    item.append(link);
    byId("optimization-report-toc").append(item);
    byId("optimization-report-body").append(table);
  }
  function openOptimizationReport() {
    if (byId("show-optimization-report").disabled) return;
    const report = optimizationReport;
    const formatter = new Intl.NumberFormat(undefined, {
      maximumFractionDigits: 6,
    });
    const milliseconds = (ns) => ({
      text: formatter.format(ns / 1e6),
      sortValue: ns,
    });
    byId("optimization-report-body").replaceChildren();
    const contents = document.createElement("nav");
    contents.setAttribute("aria-label", "Table of contents");
    const title = document.createElement("h3");
    title.textContent = "Table of contents";
    const list = document.createElement("ul");
    list.id = "optimization-report-toc";
    contents.append(title, list);
    byId("optimization-report-body").append(contents);
    byId("optimization-report-scope").textContent = fusionGraph.breadcrumbs
      .map((item) => item.name)
      .join(" / ");
    reportTable(
      "optimization-report-summary",
      "Optimization counters",
      ["Metric", "Value"],
      [
        ["Iterations (including subgraphs)", report.iterations],
        ["Rewrites (including subgraphs)", report.rewrites],
        ["Full native operation elapsed (ms)", milliseconds(report.wallTimeNs)],
        ["Total native phase time (ms)", milliseconds(report.totalTimeNs)],
      ],
    );
    reportTable(
      "optimization-report-phases",
      "Native phase timings",
      ["Phase", "Time (ms)"],
      [
        ["Matching (root)", milliseconds(report.matchingTimeNs)],
        ["Rewriting (root)", milliseconds(report.rewritingTimeNs)],
        ["Cleanup (root)", milliseconds(report.cleanupTimeNs)],
        ["Shape/Size relocation", milliseconds(report.moveShapeAndSizeTimeNs)],
        ["Constant folding (root)", milliseconds(report.constantFoldingTimeNs)],
        [
          "Subgraph optimization",
          milliseconds(report.subgraphOptimizationTimeNs),
        ],
      ],
    );
    reportTable(
      "optimization-report-patterns",
      "Pattern matches (root and subgraphs)",
      [
        "Pattern",
        "Attempts",
        "Accepted matches",
        "Matching (ms)",
        "Applying (ms)",
      ],
      report.patterns.map((pattern) => [
        pattern.name,
        pattern.attempts,
        pattern.matches,
        milliseconds(pattern.matchTimeNs),
        milliseconds(pattern.applyTimeNs),
      ]),
      "No pattern statistics.",
    );
    reportTable(
      "optimization-report-rejections",
      "Rejected candidates",
      ["Pattern", "Reason", "Occurrences", "Native source"],
      report.patterns.flatMap((pattern) =>
        pattern.noMatches.map((reason) => [
          pattern.name,
          reason.reason,
          reason.occurrences,
          reason.sourceFile
            ? nativeSourceLocation(reason.sourceFile, reason.sourceLine)
            : "Not supplied",
        ]),
      ),
      "No rejection diagnostics were reported.",
    );
    reportTable(
      "optimization-report-subgraphs",
      "Optimized subgraphs",
      [
        "Graph path",
        "Iterations (including descendants)",
        "Rewrites (including descendants)",
        "Elapsed (ms)",
      ],
      report.subgraphs.map((subgraph) => [
        subgraph.path.join(" / "),
        subgraph.iterations,
        subgraph.rewrites,
        milliseconds(subgraph.elapsedTimeNs),
      ]),
      "No subgraphs were optimized.",
    );
    byId("optimization-report").showModal();
  }
  function currentScenario() {
    return scenarios.find(
      (item) => item.name === byId("rewriting-scenario").value,
    );
  }
  function focusScenarioStep(index) {
    const scenario = currentScenario();
    const step = scenario?.steps[index];
    if (fusionBusy || !canFuse() || !step?.nodes.length) return;
    selectedStep = { name: scenario.name, index };
    for (const [stepIndex, button] of [
      ...byId("rewriting-steps").querySelectorAll(".rewriting-step"),
    ].entries())
      button.setAttribute("aria-current", String(stepIndex === index));
    focusNodes(step.nodes);
  }
  function updateScenario() {
    const scenario = currentScenario();
    byId("rewriting-scenario").disabled =
      fusionBusy ||
      !canSelectRewriting() ||
      !(rewritingState.names.length || undoCount);
    for (const id of ["rewriting-next", "rewriting-all"])
      byId(id).disabled = fusionBusy || !scenario?.available || !canFuse();
    byId("rewriting-restart").disabled = fusionBusy || !scenario || !canFuse();
    byId("rewriting-steps").replaceChildren();
    const path = fusionGraph?.breadcrumbs?.at(-1)?.path ?? [];
    const stage = path[0]?.pipeline;
    const rootVisible =
      (!path.length && !fusionGraph?.pipeline) ||
      (path.length === 1 && stage !== undefined);
    let nodesByKey;
    const pendingMatches = pendingRewritings
      .filter((operation) => operation.stage === stage)
      .flatMap((operation) => operation.matches);
    const workingSteps = [
      ...optimizationHistory
        .filter((operation) => operation.stage === stage)
        .flatMap((operation) => operation.matches)
        .map((match) => ({
          ...match,
          path: match.impact?.scope ?? [],
          applied: true,
        })),
      ...pendingMatches.map((match) => ({
        ...match,
        path: match.impact?.scope ?? [],
        nodes: nodesBeforeRewriting(match),
        applied: false,
      })),
    ];
    const steps = workingSteps.length
      ? workingSteps
      : (scenario?.steps ?? []).map((step, index) => ({
          ...step,
          applied: index < scenario.applied,
        }));
    if (workingSteps.length && scenario?.available)
      steps.push(
        ...scenario.steps
          .slice(scenario.applied + pendingMatches.length)
          .map((step) => ({
            ...step,
            applied: false,
          })),
      );
    for (const [index, step] of steps.entries()) {
      const item = document.createElement("li");
      item.className = step.applied ? "rewriting-applied" : "rewriting-pending";
      const label = document.createElement("button");
      label.className = "rewriting-step";
      label.disabled = fusionBusy;
      label.textContent = `${step.pattern}${step.path.length ? ` [${step.path.join(" / ")}]` : ""}`;
      label.setAttribute(
        "aria-current",
        String(
          selectedStep?.name === scenario?.name &&
            selectedStep?.index === index,
        ),
      );
      item.append(label);
      let content;
      label.addEventListener("click", () => {
        if (fusionBusy) return;
        selectedStep = { name: scenario?.name, index };
        for (const [otherIndex, other] of [
          ...byId("rewriting-steps").querySelectorAll(".rewriting-step"),
        ].entries())
          other.setAttribute("aria-current", String(otherIndex === index));
        content ??= appendImpactDetails(item, step, true);
        if (content) content.open = true;
        const nodes =
          !rootVisible || step.path.length
            ? []
            : !step.applied && !step.nodes.length
              ? nodesBeforeRewriting(step)
              : step.nodes;
        focusNodes(nodes);
        byId("fusion-feedback").textContent = step.inspectionError
          ? `Could not inspect this rewriting: ${step.inspectionError}`
          : nodes.length
            ? ""
            : "These nodes are not present in the current graph.";
      });
      byId("rewriting-steps").append(item);
    }
    function nodesBeforeRewriting(match) {
      if (!rootVisible || match.impact?.scope?.length) return [];
      const key = (node) =>
        JSON.stringify([
          node.domain || "ai.onnx",
          node.opType,
          node.inputs,
          node.outputs,
        ]);
      const before = new Set((match.impact?.before ?? []).map(key));
      if (!before.size) return [];
      if (!nodesByKey) {
        nodesByKey = new Map();
        for (const [index, node] of (fusionGraph?.nodes ?? []).entries()) {
          const signature = key(node);
          const indices = nodesByKey.get(signature);
          if (indices) indices.push(index);
          else nodesByKey.set(signature, [index]);
        }
      }
      return [...before]
        .flatMap((signature) => nodesByKey.get(signature) ?? [])
        .sort((left, right) => left - right);
    }
    byId("rewriting-progress").textContent = workingSteps.length
      ? `${steps.filter((step) => step.applied).length} / ${steps.length} steps applied.`
      : scenario
        ? `${scenario.applied} / ${scenario.steps.length} steps applied.${!scenario.available && scenario.applied < scenario.steps.length ? " Unfuse current edits before applying this scenario." : ""}`
        : "None selected: the model is shown without a saved rewriting.";
  }
  function showRewritingState(state) {
    rewritingState = state;
    byId("rewriting-scenario").replaceChildren(
      new Option("None", ""),
      ...[...state.names]
        .sort((a, b) => a.localeCompare(b))
        .map((name) => new Option(name, name)),
    );
    byId("rewriting-scenario").value = state.selected ?? "";
  }
  function showScenarios(values, name) {
    const previous = name ?? byId("rewriting-scenario").value;
    const hadScenarios = scenarios.length > 0;
    scenarios = values ?? [];
    if (!scenarios.length || !hadScenarios)
      byId("rewriting-scenarios").open = true;
    if (values) {
      showRewritingState({
        names: scenarios.map((scenario) => scenario.name),
        selected: previous,
      });
    }
    updateScenario();
  }
  function fusionControls(busy, message = "") {
    busy ||= undoBusy;
    if (busy && diagnosticBusy) clearDiagnostic();
    fusionBusy = busy;
    for (const id of ["unfuse-last", "unfuse-all"])
      byId(id).disabled = busy || !undoCount;
    const path = fusionGraph?.breadcrumbs?.at(-1)?.path ?? [];
    const stage = path.length === 1 ? path[0].pipeline : undefined;
    byId("refuse-next").disabled =
      busy ||
      !canFuse() ||
      !(pendingRewritings.length
        ? pendingRewritings[0].stage === stage
        : currentScenario()?.available);
    byId("append-new").disabled =
      busy ||
      !canFuse() ||
      !byId("fusion-pattern").querySelector("input[data-pattern]:checked");
    for (const id of [
      "global-rewriting",
      "global-rewriting-operation",
      "apply-global-rewriting",
    ])
      byId(id).disabled = busy || !canFuse();
    for (const button of byId("unfuse-history").querySelectorAll("button"))
      button.disabled = busy || button.dataset.nodeCount === "0";
    selectionPatternControls();
    diagnosticControls();
    byId("show-optimization-report").disabled = busy || !optimizationReport;
    for (const backend of ["onnx-light", "onnxscript"]) {
      byId(`optimizer-${backend}`).disabled = busy;
      byId(`optimizer-${backend}`).setAttribute(
        "aria-pressed",
        String(optimizationBackend === backend),
      );
    }
    byId("native-optimization-patterns").hidden =
      optimizationBackend !== "onnx-light";
    byId("onnxscript-optimization-note").hidden =
      optimizationBackend !== "onnxscript";
    const output = byId("onnxscript-optimization-output");
    output.hidden = optimizationBackend !== "onnxscript" || !output.textContent;
    byId("native-optimization-history").hidden =
      optimizationBackend !== "onnx-light" || optimizationFinished;
    byId("optimization-rewriting").hidden =
      !optimizationFinished || optimizationBackend !== "onnx-light";
    const rewritingName = byId("optimization-rewriting-name").value.trim();
    const duplicateName = rewritingState.names.includes(rewritingName);
    byId("optimization-rewriting-name").setCustomValidity(
      duplicateName ? "This rewriting name already exists." : "",
    );
    const hasPending = pendingRewritings.length > 0;
    const hasAppliedEdits = optimizationHistory.some(
      (operation) => operation.stage === stage,
    );
    const canRecord =
      optimizationFinished &&
      canFuse() &&
      hasAppliedEdits &&
      !busy &&
      !hasPending;
    byId("optimization-rewriting-name").disabled = !canRecord;
    byId("create-optimization-rewriting").disabled =
      !canRecord || !rewritingName || duplicateName;
    const selectedRewriting = rewritingState.selected;
    const workingName = byId("working-rewriting-name").value.trim();
    const workingDuplicate = rewritingState.names.includes(workingName);
    byId("working-rewriting-name-field").hidden =
      !!selectedRewriting || !hasAppliedEdits;
    byId("working-rewriting-name").disabled =
      busy || !canFuse() || !hasAppliedEdits;
    byId("working-rewriting-name").setCustomValidity(
      workingDuplicate ? "This rewriting name already exists." : "",
    );
    byId("record-working-rewriting").disabled =
      busy ||
      !canFuse() ||
      !hasAppliedEdits ||
      hasPending ||
      (!selectedRewriting && (!workingName || workingDuplicate));
    byId("record-working-rewriting").textContent = selectedRewriting
      ? `Update ${selectedRewriting}`
      : "Create rewriting";
    byId("append-optimization-rewriting").disabled =
      !canRecord || !selectedRewriting;
    byId("append-optimization-rewriting").textContent = selectedRewriting
      ? `Add to ${selectedRewriting}`
      : "Add to selected rewriting";
    for (const id of [
      "record-working-rewriting",
      "create-optimization-rewriting",
      "append-optimization-rewriting",
    ])
      byId(id).title = hasPending
        ? "Reapply or discard pending rewritings before recording."
        : "";
    for (const id of [
      "optimization-unfuse-last",
      "optimization-restore-original",
    ])
      byId(id).disabled = busy || !canFuse() || !undoCount;
    byId("replay-rewritings").disabled =
      busy || !canFuse() || !pendingRewritings.length;
    for (const control of byId("fusion-matches").querySelectorAll(
      "input, button",
    ))
      control.disabled = busy;
    for (const id of [
      "fusion-panel",
      "optimization-panel",
      "selection-pattern-dialog",
      "fuse-pattern",
      "optimize-model",
    ])
      byId(id).setAttribute("aria-busy", String(busy));
    const patterns = byId("fusion-pattern");
    patterns.disabled = busy || !patterns.querySelector("input");
    byId("fusion-pattern-group").disabled =
      patterns.disabled || !byId("fusion-pattern-group").querySelector("input");
    byId("select-patterns").disabled = patterns.disabled;
    byId("clear-patterns").disabled =
      busy || !patterns.querySelector("input[data-pattern]:checked");
    byId("preview-fusion").disabled =
      busy || !patterns.querySelector("input[data-pattern]:checked");
    byId("select-fusions").disabled = busy || !fusionPreview?.matches.length;
    byId("clear-fusions").disabled = busy || !fusionSelection().length;
    byId("apply-fusion").disabled = busy || !fusionSelection().length;
    byId("fuse-pattern").disabled =
      undoBusy || creatingPattern || !(canFuse() || undoCount);
    byId("optimize-model").disabled = creatingPattern || !canFuse();
    const optimization = byId("optimization-patterns");
    optimization.disabled =
      busy || !canFuse() || !optimization.querySelector("input");
    byId("optimization-pattern-group").disabled =
      optimization.disabled ||
      !byId("optimization-pattern-group").querySelector("input");
    byId("select-optimization-patterns").disabled = optimization.disabled;
    byId("clear-optimization-patterns").disabled =
      busy ||
      !canFuse() ||
      !optimization.querySelector("input[data-pattern]:checked");
    byId("run-optimization").disabled =
      busy ||
      !canFuse() ||
      (optimizationBackend === "onnx-light" &&
        !optimization.querySelector("input[data-pattern]:checked"));
    for (const button of byId("optimization-results").querySelectorAll(
      "button",
    ))
      button.disabled = busy || !canFuse() || button.dataset.empty === "true";
    updateScenario();
    if (message) byId("fusion-feedback").textContent = message;
    else if (fusionPreview) updateFusionSelection();
  }

  function fusionSelection() {
    return [...byId("fusion-matches").querySelectorAll("input:checked")].map(
      (input) => Number(input.value),
    );
  }

  function updateFusionSelection() {
    const candidates = new Set(fusionSelection());
    const nodes = new Set();
    let overlap = false;
    let global = false;
    for (const match of fusionPreview?.matches ?? []) {
      if (!candidates.has(match.id)) continue;
      global ||= !!match.global;
      for (const node of match.nodes) {
        if (nodes.has(node)) overlap = true;
        nodes.add(node);
      }
      if (global && candidates.size > 1) overlap = true;
    }
    byId("apply-fusion").disabled = fusionBusy || !candidates.size || overlap;
    byId("clear-fusions").disabled = fusionBusy || !candidates.size;
    byId("fusion-feedback").textContent =
      global && candidates.size > 1
        ? "Apply a global pass individually."
        : overlap
          ? "Selected matches overlap. Select disjoint matches."
          : !fusionPreview?.matches.length
            ? byId("diagnostic-pattern").options.length
              ? "No matching candidates. Select the operator where you expected a match, then use Why did matching fail? below."
              : "No matching candidates. The selected patterns or passes do not produce a rewrite."
            : `${candidates.size} selected / ${fusionPreview.matches.length} matches. Check candidates, then click Add selected patterns.`;
  }

  function clearFusionPreview() {
    appendOne = false;
    if (fusionPreview) {
      clearSelection();
    }
    fusionPreview = undefined;
    if (selectedOptimizationResult) clearSelection();
    selectedOptimizationResult = false;
    byId("fusion-matches").replaceChildren();
    fusionControls(false);
  }

  function filterPatterns(fieldsetId, filterId) {
    const query = byId(filterId).value.trim().toLocaleLowerCase();
    for (const group of byId(fieldsetId).querySelectorAll("details")) {
      const labels = [...group.querySelectorAll("label")];
      for (const label of labels)
        label.hidden = !label
          .querySelector("input")
          .value.toLocaleLowerCase()
          .includes(query);
      group.hidden = labels.every((label) => label.hidden);
      if (query) {
        group.dataset.filterOpen ??= String(group.open);
        group.open = !group.hidden;
      } else if (group.dataset.filterOpen !== undefined) {
        group.open = group.dataset.filterOpen === "true";
        delete group.dataset.filterOpen;
      }
    }
  }
  byId("optimization-filter").addEventListener("input", () =>
    filterPatterns("optimization-patterns", "optimization-filter"),
  );
  byId("fusion-filter").addEventListener("input", () =>
    filterPatterns("fusion-pattern", "fusion-filter"),
  );

  function showOptimizationPatterns(patterns) {
    if (!patterns) return;
    nativePatterns = new Set(patterns.map((pattern) => pattern.name));
    patterns = patterns.filter(
      (pattern) => !partitionPatterns.includes(pattern.name),
    );
    showPatternPicker(
      "optimization-patterns",
      "optimization-filter",
      patterns,
      optimizationSelection,
    );
    updateOptimizationSelection();
  }

  function showPatternPicker(fieldsetId, filterId, patterns, selection) {
    const fieldset = byId(fieldsetId);
    fieldset.replaceChildren();
    const fromSelection = (pattern) =>
      pattern.groups?.includes("Selection patterns");
    const priorities = [
      ...new Set(
        patterns
          .filter((pattern) => !fromSelection(pattern))
          .map((pattern) => pattern.priority),
      ),
    ].sort((a, b) => a - b);
    const buckets = priorities.map((priority) => ({
      priority,
      title: `Priority ${priority}`,
      members: patterns.filter(
        (pattern) => !fromSelection(pattern) && pattern.priority === priority,
      ),
    }));
    const custom = patterns.filter(fromSelection);
    if (custom.length)
      buckets.push({
        priority: "selection",
        title: "Selection patterns",
        members: custom,
      });
    for (const { priority, title, members: unsorted } of buckets) {
      const group = document.createElement("details");
      const summary = document.createElement("summary");
      const toggle = document.createElement("input");
      toggle.type = "checkbox";
      toggle.dataset.priority = String(priority);
      toggle.setAttribute(
        "aria-label",
        priority === "selection"
          ? "Select selection patterns"
          : `Select ${title.toLowerCase()} patterns`,
      );
      const members = unsorted.sort((a, b) => a.name.localeCompare(b.name));
      summary.append(
        toggle,
        document.createTextNode(` ${title} (${members.length} patterns)`),
      );
      group.append(summary);
      for (const pattern of members) {
        const label = document.createElement("label");
        const input = document.createElement("input");
        input.type = "checkbox";
        input.value = pattern.name;
        input.dataset.pattern = pattern.name;
        input.checked = selection
          ? selection.has(pattern.name)
          : pattern.enabledByDefault !== false;
        label.title = `${pattern.device}; native priority ${pattern.priority}`;
        label.append(input, document.createTextNode(pattern.name));
        group.append(label);
      }
      toggle.addEventListener("change", () => {
        for (const input of group.querySelectorAll("input[data-pattern]"))
          input.checked = toggle.checked;
      });
      fieldset.append(group);
    }
    filterPatterns(fieldsetId, filterId);
    syncPatternPriorities(fieldsetId);
  }

  function showPatternGroups(id, fieldsetId, patterns, changed) {
    const picker = byId(id);
    const legend = document.createElement("legend");
    legend.textContent = "Select pattern groups";
    picker.replaceChildren(legend);
    const fieldset = byId(fieldsetId);
    const available = new Set(
      [...fieldset.querySelectorAll("input")].map((input) => input.value),
    );
    const groups = new Map([["Default", new Set()]]);
    for (const pattern of patterns ?? []) {
      if (!available.has(pattern.name)) continue;
      if (pattern.enabledByDefault) groups.get("Default").add(pattern.name);
      for (const name of pattern.groups ?? []) {
        if (!groups.has(name)) groups.set(name, new Set());
        groups.get(name).add(pattern.name);
      }
    }
    const names = [...groups.keys()]
      .filter((name) => name !== "Default" && name !== "all")
      .sort();
    for (const name of ["Default", ...names, "all"]) {
      const members = groups.get(name);
      if (!members?.size) continue;
      const label = document.createElement("label");
      const input = document.createElement("input");
      input.type = "checkbox";
      input.value = name;
      label.append(
        input,
        document.createTextNode(
          `${name === "all" ? "All" : name} (${members.size} patterns)`,
        ),
      );
      picker.append(label);
    }
    const sync = () => {
      const selected = new Set(
        [...fieldset.querySelectorAll("input:checked")]
          .filter((input) => !input.hasAttribute("data-priority"))
          .map((input) => input.value),
      );
      let matched = false;
      for (const input of picker.querySelectorAll("input")) {
        const members = groups.get(input.value);
        input.checked =
          !matched &&
          members.size === selected.size &&
          [...members].every((name) => selected.has(name));
        matched ||= input.checked;
      }
    };
    fieldset.onchange = sync;
    sync();
    picker.onchange = () => {
      const members = new Set(
        [...picker.querySelectorAll("input:checked")].flatMap((input) => [
          ...groups.get(input.value),
        ]),
      );
      for (const input of fieldset.querySelectorAll("input"))
        if (!input.hasAttribute("data-priority"))
          input.checked = members.has(input.value);
      changed();
    };
  }

  function updateOptimizationSelection() {
    const fieldset = byId("optimization-patterns");
    const selected = [
      ...fieldset.querySelectorAll("input[data-pattern]:checked"),
    ];
    optimizationSelection = new Set(selected.map((input) => input.value));
    syncPatternPriorities("optimization-patterns");
    byId("optimization-feedback").textContent =
      `${selected.length} / ${fieldset.querySelectorAll("input[data-pattern]").length} patterns selected.`;
    fusionControls(fusionBusy);
  }
  function syncPatternPriorities(fieldsetId) {
    for (const group of byId(fieldsetId).querySelectorAll("details")) {
      const toggle = group.querySelector("input[data-priority]");
      const count = group.querySelectorAll("input[data-pattern]").length;
      const checked = group.querySelectorAll(
        "input[data-pattern]:checked",
      ).length;
      toggle.checked = count === checked;
      toggle.indeterminate = checked > 0 && checked < count;
    }
  }

  function impactSummary(match) {
    const impact = match.impact;
    if (!impact)
      return `${match.nodes.length} remaining node${match.nodes.length === 1 ? "" : "s"}`;
    const count = impact.before.length;
    const parts = [`${count} impacted node${count === 1 ? "" : "s"}`];
    if (impact.after.length)
      parts.push(
        `${impact.after.length} replacement node${impact.after.length === 1 ? "" : "s"}`,
      );
    for (const [count, action] of [
      [impact.removedInitializers, "removed"],
      [impact.addedInitializers, "added"],
    ])
      if (count)
        parts.push(`${count} initializer${count === 1 ? "" : "s"} ${action}`);
    return parts.join("; ");
  }
  function appendImpactDetails(container, match, expanded = false) {
    if (!match.impact && !match.inspectionError) return;
    const details = document.createElement("details");
    details.className = "fusion-impact";
    const summary = document.createElement("summary");
    summary.textContent = "Before / after this operation";
    details.append(summary);
    const populate = () => {
      if (!details.open || details.childElementCount > 1) return;
      if (match.inspectionError) {
        const error = document.createElement("p");
        error.textContent = `Could not inspect this rewriting: ${match.inspectionError}`;
        details.append(error);
      }
      if (!match.impact) return;
      if (match.impact.scope.length) {
        const scope = document.createElement("p");
        scope.textContent = `Scope: ${match.impact.scope.join(" / ")}`;
        details.append(scope);
      }
      for (const [title, nodes] of [
        ["Before", match.impact.before],
        ["After", match.impact.after],
      ]) {
        const heading = document.createElement("h4");
        heading.textContent = title;
        const text = document.createElement("pre");
        text.textContent = nodes.length
          ? nodes
              .map((node) => {
                const index = node.index === undefined ? "" : `#${node.index} `;
                const name = node.name ? `${node.name}: ` : "";
                const domain = node.domain ? `${node.domain}::` : "";
                return node.opType
                  ? `${index}${name}${domain}${node.opType}(${node.inputs.join(", ")}) -> ${node.outputs.join(", ")}`
                  : `${index}Node`;
              })
              .join("\n")
          : "No nodes.";
        details.append(heading, text);
      }
    };
    details.addEventListener("toggle", populate);
    if (expanded) {
      details.open = true;
      populate();
    }
    container.append(details);
    return details;
  }
  function showOptimizationResults() {
    const queued = pendingRewritings.flatMap((operation) =>
      operation.matches.map((match) => ({
        pattern: match.pattern,
        stage: operation.stage,
      })),
    );
    byId("rewriting-count").hidden = !queued.length;
    byId("rewriting-count").textContent = queued.length
      ? `${queued.length} pending rewriting${queued.length === 1 ? "" : "s"}`
      : "";
    byId("pending-rewritings").hidden = !queued.length;
    byId("pending-rewriting-steps").replaceChildren(
      ...queued.map((match) => {
        const item = document.createElement("li");
        item.textContent = `${match.stage ? `[${match.stage}] ` : ""}${match.pattern}`;
        return item;
      }),
    );
    byId("optimization-results").replaceChildren();
    if (optimizationFinished) return;
    const path = fusionGraph?.breadcrumbs?.at(-1)?.path ?? [];
    const stage = path.length === 1 ? path[0].pipeline : undefined;
    const matches = optimizationHistory
      .filter((operation) => operation.stage === stage)
      .flatMap((operation) => operation.matches);
    for (const match of matches) {
      const item = document.createElement("li");
      const button = document.createElement("button");
      button.textContent = `${match.pattern} (${impactSummary(match)})`;
      button.dataset.empty = String(!match.nodes.length);
      button.disabled = !match.nodes.length;
      button.title = match.nodes.length
        ? "Highlight surviving result nodes."
        : "No surviving result nodes in the main graph.";
      button.addEventListener("click", () => {
        selectedOptimizationResult = true;
        focusNodes(match.nodes);
      });
      item.append(button);
      appendImpactDetails(item, match);
      byId("optimization-results").append(item);
    }
    if (!matches.length) {
      const item = document.createElement("li");
      item.textContent = "No applied pattern matches in this graph.";
      byId("optimization-results").append(item);
    }
  }

  function previewFusion(single = false) {
    clearFusionPreview();
    appendOne = single;
    byId("fusion-status").textContent = "";
    fusionControls(true, "Matching patterns and building replacements…");
    requestAction("preview-fusion", "Preview fusion pattern", {
      requestId: ++fusionRequest,
      patterns: [
        ...byId("fusion-pattern").querySelectorAll(
          "input[data-pattern]:checked",
        ),
      ]
        .map((input) => input.value)
        .sort(),
      discardPending: true,
    });
  }

  function applyFusion(candidates) {
    const message = `Applying ${candidates.length} fusion match${candidates.length === 1 ? "" : "es"}…`;
    fusionControls(true, message);
    byId("fusion-status").textContent = message;
    requestAction("apply-fusion", "Apply fusion pattern", {
      requestId: ++fusionRequest,
      previewId: fusionPreview.id,
      candidates,
    });
  }

  function focusFusionMatch(match) {
    for (const item of byId("fusion-matches").children)
      item.setAttribute(
        "aria-current",
        String(Number(item.dataset.candidate) === match.id),
      );
    focusNodes(match.nodes);
  }

  function formatFusionNode(node, showName = false) {
    const args = [
      ...node.inputs,
      ...node.attributes.map(
        (attribute) => `${attribute.name}=${attribute.value}`,
      ),
    ];
    const domain =
      node.domain && node.domain !== "ai.onnx" ? `${node.domain}::` : "";
    return `${showName ? `${node.name || node.id}: ` : ""}${domain}${node.opType}(${args.join(", ")}) -> ${node.outputs.join(", ")}`;
  }

  function showFusionMatches(data) {
    fusionPreview = {
      id: data.previewId,
      matches: data.matches.map((match, id) => ({ ...match, id })),
    };
    byId("fusion-matches").replaceChildren();
    const precedingNodes = new Set();
    for (const match of fusionPreview.matches) {
      const item = document.createElement("li");
      item.className = "fusion-match";
      if (match.nodes.some((node) => precedingNodes.has(node))) {
        item.classList.add("fusion-match-overlap");
        item.title = "Shares nodes with an earlier match in this list.";
      }
      for (const node of match.nodes) precedingNodes.add(node);
      item.dataset.candidate = String(match.id);
      const label = document.createElement("label");
      if (match.global)
        label.title = "Whole-scope pass: apply this candidate individually.";
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.value = String(match.id);
      checkbox.addEventListener("change", () => {
        focusFusionMatch(match);
        updateFusionSelection();
      });
      label.append(
        checkbox,
        document.createTextNode(match.pattern || "Fusion candidate"),
      );
      item.append(label);
      for (const [title, text] of [
        [
          "Before",
          match.nodes.length
            ? match.nodes
                .map((index) =>
                  formatFusionNode(fusionGraph.nodes[index], true),
                )
                .join("\n")
            : "Initializer or nested-graph changes; no affected root operators.",
        ],
        [
          "After",
          match.replacement.length
            ? match.replacement.map((node) => formatFusionNode(node)).join("\n")
            : "Remove matched nodes (no replacement operators).",
        ],
      ]) {
        const section = document.createElement("div");
        section.className = "fusion-stage";
        const heading = document.createElement("strong");
        heading.textContent = title;
        const operators = document.createElement("pre");
        operators.textContent = text;
        section.append(heading, operators);
        item.append(section);
      }
      item.addEventListener("click", () => {
        if (!fusionBusy) focusFusionMatch(match);
      });
      byId("fusion-matches").append(item);
    }
    fusionControls(false);
    updateFusionSelection();
    if (!data.matches.length && byId("diagnostic-pattern").options.length)
      byId("fusion-diagnostics").open = true;
  }
  byId("fuse-pattern").addEventListener("click", () => {
    if (!byId("fusion-panel").hidden) {
      clearDiagnostic();
      fusionRequest++;
      byId("fusion-panel").hidden = true;
      byId("fuse-pattern").setAttribute("aria-expanded", "false");
      if (selectedStep) clearSelection();
      selectedStep = undefined;
      clearFusionPreview();
      fit();
      postMessage({ type: "cancel-fusion" });
      return;
    }
    byId("fusion-panel").hidden = false;
    revealPanel(byId("fusion-panel"));
    byId("fuse-pattern").setAttribute("aria-expanded", "true");
    if (!canFuse()) {
      fusionControls(false);
      fit();
      return;
    }
    clearFusionPreview();
    byId("fusion-pattern").replaceChildren();
    updateDiagnosticPatterns();
    fusionControls(true, "Listing applicable pattern anchors…");
    fit();
    requestAction("fusion-patterns", "List fusion patterns", {
      requestId: ++fusionRequest,
    });
  });
  byId("optimize-model").addEventListener("click", () => {
    const panel = byId("optimization-panel");
    panel.hidden = !panel.hidden;
    byId("optimize-model").setAttribute("aria-expanded", String(!panel.hidden));
    clearFusionPreview();
    if (panel.hidden) {
      restoreRequest = undefined;
      fusionRequest++;
      postMessage({ type: "cancel-fusion" });
    } else {
      revealPanel(panel);
      fusionControls(true);
      byId("optimization-status").textContent =
        "Listing native optimization patterns…";
      requestAction("fusion-patterns", "List optimization patterns", {
        requestId: ++fusionRequest,
      });
    }
    fit();
  });
  byId("optimization-patterns").addEventListener(
    "change",
    updateOptimizationSelection,
  );
  for (const backend of ["onnx-light", "onnxscript"])
    byId(`optimizer-${backend}`).addEventListener("click", () => {
      optimizationBackend = backend;
      byId("optimization-status").textContent = "";
      showOptimizationResults();
      fusionControls(fusionBusy);
    });
  byId("optimization-rewriting-name").addEventListener("input", () =>
    fusionControls(fusionBusy),
  );
  for (const [id, append] of [
    ["create-optimization-rewriting", false],
    ["append-optimization-rewriting", true],
  ])
    byId(id).addEventListener("click", () => {
      recordingFromRewritings = false;
      const name = append
        ? rewritingState.selected
        : byId("optimization-rewriting-name").value.trim();
      if (byId(id).disabled || !name) return;
      byId("optimization-rewriting-status").textContent =
        "Recording rewriting…";
      fusionControls(true);
      requestAction("record-rewriting", "Record optimization rewriting", {
        requestId: ++fusionRequest,
        name,
        append,
      });
    });
  byId("working-rewriting-name").addEventListener("input", () =>
    fusionControls(fusionBusy),
  );
  byId("record-working-rewriting").addEventListener("click", () => {
    if (byId("record-working-rewriting").disabled) return;
    const append = !!rewritingState.selected;
    const name = append
      ? rewritingState.selected
      : byId("working-rewriting-name").value.trim();
    recordingFromRewritings = true;
    byId("working-rewriting-status").textContent = "Recording rewriting…";
    fusionControls(true);
    requestAction("record-rewriting", "Record working rewriting", {
      requestId: ++fusionRequest,
      name,
      append,
    });
  });
  for (const [id, all] of [
    ["optimization-unfuse-last", false],
    ["optimization-restore-original", true],
  ])
    byId(id).addEventListener("click", () => {
      const requestId = ++fusionRequest;
      requestAction("undo-fusion", "Undo optimization", { all, requestId });
      restoreRequest = requestId;
      fusionControls(true);
      byId("optimization-status").textContent = all
        ? "Restoring the original model…"
        : "Unfusing the last operation…";
    });
  byId("replay-rewritings").addEventListener("click", () => {
    fusionControls(true);
    byId("optimization-status").textContent = "Applying pending rewritings…";
    requestAction("replay-rewritings", "Replay pending rewritings", {
      requestId: ++fusionRequest,
    });
  });
  for (const [id, checked] of [
    ["select-optimization-patterns", true],
    ["clear-optimization-patterns", false],
  ])
    byId(id).addEventListener("click", () => {
      for (const input of byId("optimization-patterns").querySelectorAll(
        "input[data-pattern]",
      ))
        input.checked = checked;
      byId("optimization-patterns").dispatchEvent(new Event("change"));
    });
  byId("run-optimization").addEventListener("click", () => {
    clearFusionPreview();
    byId("onnxscript-optimization-output").textContent = "";
    fusionControls(true);
    byId("optimization-status").textContent =
      optimizationBackend === "onnxscript"
        ? "Running ONNX Script optimization and fusion…"
        : "Running the complete native optimizer…";
    requestAction("apply-optimization", "Optimize ONNX model", {
      requestId: ++fusionRequest,
      operation: "Optimize",
      ...(optimizationBackend === "onnxscript"
        ? { backend: "onnxscript" }
        : {}),
      patterns:
        optimizationBackend === "onnxscript" ? [] : [...optimizationSelection],
    });
  });
  byId("diagnostic-pattern").addEventListener("change", clearDiagnostic);
  byId("fusion-diagnostics")
    .querySelector("summary")
    .addEventListener("click", (event) => {
      if (byId("fusion-diagnostics").getAttribute("aria-disabled") === "true")
        event.preventDefault();
    });
  byId("show-optimization-report").addEventListener(
    "click",
    openOptimizationReport,
  );
  byId("close-optimization-report").addEventListener("click", () =>
    byId("optimization-report").close(),
  );
  byId("explain-fusion").addEventListener("click", () => {
    if (byId("explain-fusion").disabled) return;
    clearDiagnostic();
    diagnosticBusy = true;
    diagnosticControls();
    requestAction("explain-fusion", "Explain fusion match", {
      requestId: ++diagnosticRequest,
      pattern: byId("diagnostic-pattern").value,
      anchor: Number(diagnosticNode.id.slice(5)),
    });
  });
  function changeFusionPatterns() {
    syncPatternPriorities("fusion-pattern");
    fusionRequest++;
    clearFusionPreview();
    updateDiagnosticPatterns();
    byId("fusion-feedback").textContent =
      "Click Append New to apply one match.";
    postMessage({ type: "cancel-fusion" });
  }
  byId("selection-pattern-name").addEventListener(
    "input",
    selectionPatternControls,
  );
  byId("close-selection-pattern").addEventListener("click", () => {
    restoreFunctionEditor = false;
    functionBodyRequest++;
    byId("selection-pattern-dialog").close();
  });
  byId("selection-pattern-dialog").addEventListener("cancel", () => {
    restoreFunctionEditor = false;
    functionBodyRequest++;
  });
  byId("selection-pattern-picker").addEventListener(
    "change",
    showSelectionPattern,
  );
  for (const field of ["text", "replacement"]) {
    byId(`selection-pattern-${field}`).addEventListener("input", () => {
      const key = byId("selection-pattern-picker").value;
      selectionPatternDrafts.set(key, {
        text: byId("selection-pattern-text").value,
        replacement: byId("selection-pattern-replacement").value,
        libraryRevision:
          selectionPatternDrafts.get(key)?.libraryRevision ?? libraryRevision,
      });
      selectionPatternControls();
    });
  }
  byId("revert-function-pattern").addEventListener("click", () => {
    if (byId("revert-function-pattern").disabled) return;
    selectionPatternDrafts.delete(byId("selection-pattern-picker").value);
    showSelectionPattern();
    byId("selection-pattern-status").textContent = "Edits reverted.";
  });
  for (const format of ["text", "replacement"]) {
    byId(`copy-selection-pattern-${format}`).addEventListener("click", () => {
      postMessage({
        type: "copy-function-pattern",
        name: byId("selection-pattern-picker").value,
        format,
        text: byId(`selection-pattern-${format}`).value,
      });
    });
  }
  byId("update-function-pattern").addEventListener("click", () => {
    if (byId("update-function-pattern").disabled) return;
    creatingPattern = true;
    fusionPreview = undefined;
    byId("fusion-matches").replaceChildren();
    fusionControls(true);
    byId("selection-pattern-status").textContent = "Applying function edits…";
    const pattern = selectionPatterns.get(
      byId("selection-pattern-picker").value,
    );
    if (pattern.functionIndex !== undefined) {
      restoreFunctionEditor = byId("selection-pattern-dialog").open;
      requestAction("update-model-function", "Update ONNX function", {
        requestId: ++fusionRequest,
        functionIndex: pattern.functionIndex,
        text: byId("selection-pattern-replacement").value,
      });
    } else
      requestAction("update-function-pattern", "Update FunctionPattern", {
        requestId: ++fusionRequest,
        name: byId("selection-pattern-picker").value,
        text: byId("selection-pattern-text").value,
        replacement: byId("selection-pattern-replacement").value,
        libraryRevision:
          selectionPatternDrafts.get(byId("selection-pattern-picker").value)
            ?.libraryRevision ?? libraryRevision,
      });
  });
  byId("create-function-pattern").addEventListener("click", () => {
    if (byId("create-function-pattern").disabled) return;
    const nodes = selectedNodes();
    creatingPattern = true;
    fusionPreview = undefined;
    byId("fusion-matches").replaceChildren();
    fusionControls(true);
    byId("selection-pattern-status").textContent = "Creating FunctionPattern…";
    requestAction("create-function-pattern", "Create FunctionPattern", {
      requestId: ++fusionRequest,
      nodes,
      name: byId("selection-pattern-name").value.trim(),
    });
  });
  byId("fusion-pattern").addEventListener("change", changeFusionPatterns);
  for (const [id, checked] of [
    ["select-patterns", true],
    ["clear-patterns", false],
  ])
    byId(id).addEventListener("click", () => {
      for (const input of byId("fusion-pattern").querySelectorAll("input"))
        input.checked = checked;
      byId("fusion-pattern").dispatchEvent(new Event("change"));
    });
  byId("preview-fusion").addEventListener("click", () => previewFusion());
  byId("append-new").addEventListener("click", () => previewFusion(true));
  byId("global-rewriting").addEventListener("click", () => {
    if (byId("global-rewriting").disabled) return;
    const panel = byId("global-rewriting-panel");
    panel.hidden = !panel.hidden;
    byId("global-rewriting").setAttribute(
      "aria-expanded",
      String(!panel.hidden),
    );
  });
  byId("apply-global-rewriting").addEventListener("click", () => {
    if (byId("apply-global-rewriting").disabled) return;
    const operation = byId("global-rewriting-operation").value;
    const identity = operation === "IdentityRemoval";
    clearFusionPreview();
    fusionControls(
      true,
      identity ? "Removing Identity nodes…" : "Partitioning the graph…",
    );
    requestAction(
      identity ? "remove-identities" : "partition-model",
      identity ? "Remove Identity nodes" : "Partition ONNX model",
      {
        requestId: ++fusionRequest,
        ...(identity ? {} : { pattern: operation }),
      },
    );
  });
  byId("refuse-next").addEventListener("click", () => {
    clearFusionPreview();
    fusionControls(true);
    requestAction("refuse-next", "Reapply next rewriting", {
      requestId: ++fusionRequest,
    });
  });
  byId("select-fusions").addEventListener("click", () => {
    const used = new Set();
    let global = false;
    let selected = 0;
    for (const input of byId("fusion-matches").querySelectorAll("input")) {
      const match = fusionPreview.matches.find(
        (candidate) => candidate.id === Number(input.value),
      );
      input.checked =
        !global &&
        !(match.global && selected) &&
        !match.nodes.some((node) => used.has(node));
      if (input.checked) {
        global ||= !!match.global;
        selected++;
        for (const node of match.nodes) used.add(node);
      }
    }
    updateFusionSelection();
  });
  byId("clear-fusions").addEventListener("click", () => {
    for (const input of byId("fusion-matches").querySelectorAll("input"))
      input.checked = false;
    updateFusionSelection();
  });
  byId("apply-fusion").addEventListener("click", () => {
    applyFusion(fusionSelection());
  });
  function selectRewriting(all = true) {
    if (fusionBusy || !canSelectRewriting()) return;
    selectedStep = undefined;
    clearSelection();
    clearFusionPreview();
    fusionControls(true, "Selecting rewriting…");
    requestAction("select-rewriting", "Select rewriting", {
      requestId: ++fusionRequest,
      scenario: byId("rewriting-scenario").value || null,
      all,
    });
  }
  byId("rewriting-scenario").addEventListener("change", () =>
    selectRewriting(),
  );
  byId("rewriting-restart").addEventListener("click", () =>
    selectRewriting(false),
  );
  for (const [id, all] of [
    ["rewriting-next", false],
    ["rewriting-all", true],
  ])
    byId(id).addEventListener("click", () => {
      clearFusionPreview();
      fusionControls(true, "Applying saved rewriting scenario…");
      requestAction("apply-rewriting", "Apply rewriting scenario", {
        requestId: ++fusionRequest,
        scenario: byId("rewriting-scenario").value,
        all,
      });
    });

  function showFusionCatalog(data) {
    selectionCatalogLoaded = true;
    for (const pattern of data.optimizationPatterns ?? [])
      if (pattern.definition)
        selectionPatterns.set(pattern.name, pattern.definition);
    const available = new Set(
      (data.optimizationPatterns ?? []).map((pattern) => pattern.name),
    );
    for (const fn of data.functions ?? []) {
      const key = `model-function:${fn.functionIndex}`;
      selectionPatterns.set(key, { ...selectionPatterns.get(key), ...fn });
      available.add(key);
    }
    if (data.definition?.functionIndex !== undefined) {
      selectionPatternDrafts.delete(
        `model-function:${data.definition.functionIndex}`,
      );
      selectionPatterns.set(
        `model-function:${data.definition.functionIndex}`,
        data.definition,
      );
    }
    for (const name of selectionPatterns.keys())
      if (!available.has(name)) {
        selectionPatterns.delete(name);
        selectionPatternDrafts.delete(name);
      }
    const previous = byId("selection-pattern-picker").value;
    byId("selection-pattern-picker").replaceChildren(
      ...[...selectionPatterns.entries()].map(([key, definition]) => {
        const option = document.createElement("option");
        option.value = key;
        option.textContent = definition.name;
        return option;
      }),
    );
    if (selectionPatterns.has(previous))
      byId("selection-pattern-picker").value = previous;
    else {
      const functionIndex = selectedNode()?.function?.path.at(-1)?.function;
      const key = `model-function:${functionIndex}`;
      if (selectionPatterns.has(key))
        byId("selection-pattern-picker").value = key;
    }
    showSelectionPattern();
    showScenarios(data.scenarios, data.name);
    showOptimizationPatterns(data.optimizationPatterns);
    const catalog = new Map(
      (data.optimizationPatterns ?? []).map((pattern) => [
        pattern.name,
        pattern,
      ]),
    );
    showPatternPicker(
      "fusion-pattern",
      "fusion-filter",
      data.patterns
        .filter((name) => !partitionPatterns.includes(name))
        .map(
          (name) =>
            catalog.get(name) ?? {
              name,
              priority: 1,
              device: "cpu",
              enabledByDefault: false,
              groups: [],
            },
        ),
      new Set(),
    );
    showPatternGroups(
      "fusion-pattern-group",
      "fusion-pattern",
      data.optimizationPatterns,
      changeFusionPatterns,
    );
    showPatternGroups(
      "optimization-pattern-group",
      "optimization-patterns",
      data.optimizationPatterns,
      updateOptimizationSelection,
    );
    updateDiagnosticPatterns();
  }

  function handleMessage(data) {
    if (
      data?.type === "model-function-loaded" ||
      data?.type === "model-function-error"
    ) {
      const key = `model-function:${data.functionIndex}`;
      if (
        data.requestId !== functionBodyRequest ||
        !byId("selection-pattern-dialog").open ||
        byId("selection-pattern-picker").value !== key
      )
        return true;
      if (data.type === "model-function-error") {
        byId("selection-pattern-status").textContent =
          `Could not load function: ${data.error}`;
        return true;
      }
      selectionPatterns.set(key, data.definition);
      showSelectionPattern();
      byId("selection-pattern-status").textContent =
        "Function body loaded. Apply edits updates all its calls; Save keeps the change.";
      return true;
    }
    if (data?.type === "model-function-updated") {
      if (
        fusionCompletion !== data.requestId ||
        fusionRequest !== data.requestId + 1
      )
        return true;
      fusionCompletion = undefined;
      creatingPattern = false;
      showFusionCatalog(data);
      byId("selection-pattern-picker").value =
        `model-function:${data.functionIndex}`;
      showSelectionPattern();
      fusionControls(false);
      if (restoreFunctionEditor) byId("selection-pattern-dialog").showModal();
      restoreFunctionEditor = false;
      byId("selection-pattern-status").textContent =
        "Function updated for all its calls. Save the model to keep the change.";
      return true;
    }
    if (
      data?.type === "function-pattern-copied" ||
      data?.type === "function-pattern-copy-error"
    ) {
      if (data.name === byId("selection-pattern-picker").value)
        byId("selection-pattern-status").textContent =
          data.type === "function-pattern-copied"
            ? `Copied ${data.format === "text" ? "pattern" : data.format}.`
            : `Copy failed: ${data.error}`;
      return true;
    }
    if (
      data?.type === "function-pattern-created" ||
      data?.type === "function-pattern-error"
    ) {
      if (data.requestId !== fusionRequest) return true;
      creatingPattern = false;
      if (data.type === "function-pattern-error") {
        fusionControls(false);
        byId("selection-pattern-status").textContent = data.error;
        return true;
      }
      selectionPatternDrafts.delete(data.pattern.name);
      selectionPatterns.set(data.pattern.name, data.pattern);
      showFusionCatalog(data);
      byId("selection-pattern-picker").value = data.pattern.name;
      showSelectionPattern();
      const checkbox = [
        ...byId("fusion-pattern").querySelectorAll("input"),
      ].find((input) => input.value === data.pattern.name);
      if (checkbox) checkbox.checked = true;
      updateDiagnosticPatterns();
      fusionControls(
        false,
        "Run Matching to find this pattern throughout the model.",
      );
      byId("selection-pattern-status").textContent =
        `Ready: ${data.pattern.name}. Close this window, open Rewritings and select this pattern, or enable Selection patterns in Global optimization.`;
      return true;
    }
    if (
      data?.type === "fusion-diagnostic" ||
      data?.type === "fusion-diagnostic-error"
    ) {
      if (!diagnosticBusy || data.requestId !== diagnosticRequest) return true;
      if (data.type === "fusion-diagnostic-error") {
        diagnosticBusy = false;
        diagnosticResult = `Diagnosis failed: ${data.error}`;
        diagnosticControls();
        return true;
      }
      const result = data.diagnostic;
      if (
        result.pattern !== byId("diagnostic-pattern").value ||
        result.anchor !== Number(diagnosticNode?.id.slice(5))
      )
        return true;
      const labels = {
        matched: "Match found",
        rejected: "Pattern does not match",
        unavailable: "No match (reason unavailable)",
        "op-type-mismatch": "Pattern cannot match this operator type",
      };
      const candidate =
        result.candidate !== undefined && result.candidate !== result.anchor
          ? ` (native candidate #${result.candidate})`
          : "";
      diagnosticBusy = false;
      diagnosticResult = `${labels[result.status]}${candidate}: ${result.reason}`;
      diagnosticControls();
      if (result.source) {
        byId("diagnostic-source").hidden = false;
        byId("diagnostic-location").replaceChildren(
          nativeSourceLocation(result.source.file, result.source.line),
        );
      }
      return true;
    }
    if (
      data?.type === "fusion-patterns" ||
      data?.type === "optimization-info" ||
      data?.type === "rewriting-info" ||
      data?.type === "rewriting-recorded" ||
      data?.type === "fusion-info"
    ) {
      if (data.type === "fusion-patterns" && data.requestId !== fusionRequest)
        return true;
      if (data.type !== "fusion-patterns") {
        if (
          fusionCompletion === undefined ||
          data.requestId !== fusionCompletion ||
          fusionRequest !== fusionCompletion + 1
        )
          return true;
        fusionCompletion = undefined;
        const optimize =
          data.type === "optimization-info" ||
          (data.type === "rewriting-recorded" && !recordingFromRewritings);
        byId(optimize ? "optimization-panel" : "fusion-panel").hidden = false;
        byId(optimize ? "optimize-model" : "fuse-pattern").setAttribute(
          "aria-expanded",
          "true",
        );
      }
      showFusionCatalog(data);
      showOptimizationReport(data.optimizationReport);
      optimizationFinished =
        data.type === "optimization-info" ||
        data.type === "rewriting-recorded" ||
        !!data.optimizationReport;
      if (optimizationFinished) showOptimizationResults();
      if (data.type === "optimization-info")
        byId("onnxscript-optimization-output").textContent =
          data.backend === "onnxscript" ? (data.logs ?? "") : "";
      if (data.type === "fusion-patterns")
        byId("optimization-status").textContent = restoreNotice;
      restoreNotice = "";
      fusionControls(
        false,
        data.patterns.length
          ? "Select patterns, then click Append New."
          : "No native patterns have compatible anchors in this graph.",
      );
      if (
        data.type === "fusion-patterns" &&
        byId("selection-pattern-dialog").open
      )
        selectionChanged();
      if (data.type !== "fusion-patterns") {
        byId("fusion-status").textContent =
          data.type === "optimization-info"
            ? data.backend === "onnxscript"
              ? data.changed
                ? ""
                : "ONNX Script: no changes."
              : data.changed
                ? `${data.operation}: applied ${data.matches.length} rewriting steps. The model remains optimized; record a named rewriting or save a copy.`
                : `${data.operation}: no changes.`
            : data.type === "rewriting-recorded"
              ? `${data.append ? "Updated" : "Created"} rewriting ${data.name}. The model remains optimized.`
              : data.type === "rewriting-info"
                ? data.name === null
                  ? ""
                  : `Applied rewriting scenario ${data.name}. Use Unfuse to undo.`
                : data.operation === "IdentityRemoval"
                  ? data.changed
                    ? `Removed Identity nodes: added ${data.count} rewriting steps. Use Unfuse to undo; save a copy to keep changes.`
                    : "IdentityRemoval: no changes."
                  : partitionPatterns.includes(data.operation)
                    ? data.changed
                      ? `${data.operation}: applied ${data.count} rewriting steps. Use Unfuse to undo; save a copy to keep changes.`
                      : `${data.operation}: no changes.`
                    : `Applied ${data.count} fusion matches. Save a copy to keep changes.`;
        if (data.notice) byId("fusion-status").textContent += ` ${data.notice}`;
        updateScenario();
        revealPanel(
          byId(
            data.type === "optimization-info" ||
              (data.type === "rewriting-recorded" && !recordingFromRewritings)
              ? "optimization-panel"
              : "fusion-panel",
          ),
        );
        fit();
        if (data.type === "rewriting-info")
          focusScenarioStep(currentScenario()?.applied);
        if (data.type === "rewriting-recorded") {
          byId("working-rewriting-status").textContent =
            byId("fusion-status").textContent;
          recordingFromRewritings = false;
        }
        if (
          data.type === "optimization-info" ||
          data.type === "rewriting-recorded"
        ) {
          byId("optimization-status").textContent =
            byId("fusion-status").textContent;
          byId("optimization-rewriting-status").textContent =
            data.type === "rewriting-recorded"
              ? byId("fusion-status").textContent
              : "";
          showOptimizationResults();
        }
      }
    } else if (data?.type === "fusion-preview") {
      if (data.replayed) {
        if (
          fusionCompletion === undefined ||
          data.requestId !== fusionCompletion ||
          fusionRequest !== fusionCompletion + 1
        )
          return true;
        fusionCompletion = undefined;
        showFusionCatalog(data);
        byId("fusion-panel").hidden = false;
        byId("fuse-pattern").setAttribute("aria-expanded", "true");
        byId("fusion-status").textContent =
          "Pending rewritings reapplied. New matches use the optimized graph.";
        showFusionMatches(data);
        revealPanel(byId("fusion-panel"));
        fit();
      } else if (data.requestId === fusionRequest) {
        const single = appendOne;
        appendOne = false;
        showFusionMatches(data);
        if (single && data.matches.length) applyFusion([0]);
        else if (single)
          byId("fusion-feedback").textContent =
            "No match for the selected patterns.";
      }
    } else if (data?.type === "fusion-error") {
      if (data.requestId !== fusionRequest) return true;
      byId("rewriting-scenario").value = rewritingState.selected ?? "";
      if (
        optimizationBackend === "onnxscript" &&
        !byId("optimization-panel").hidden
      )
        byId("onnxscript-optimization-output").textContent = data.error;
      fusionControls(false, data.error);
      if (byId("selection-pattern-dialog").open)
        byId("selection-pattern-status").textContent =
          `Could not load patterns: ${data.error}`;
      byId("fusion-status").textContent = `Fusion failed: ${data.error}`;
      byId("optimization-rewriting-status").textContent = data.error;
      byId("working-rewriting-status").textContent = data.error;
      if (!byId("optimization-panel").hidden)
        byId("optimization-status").textContent =
          `Optimization failed: ${data.error}`;
    } else {
      return false;
    }
    return true;
  }
  function reset(graph, operationRequestId, file) {
    const keepFusion = undoBusy && !byId("fusion-panel").hidden;
    undoBusy = false;
    if (operationRequestId !== fusionRequest) restoreFunctionEditor = false;
    functionBodyRequest++;
    if (byId("selection-pattern-dialog").open)
      byId("selection-pattern-dialog").close();
    creatingPattern = false;
    selectionCatalogLoaded = false;
    selectionPatterns.clear();
    selectionPatternDrafts.clear();
    byId("selection-pattern-picker").replaceChildren();
    showSelectionPattern();
    byId("selection-pattern-status").textContent =
      "Select one or more operators in the graph.";
    const restoring =
      restoreRequest !== undefined && restoreRequest === operationRequestId;
    restoreRequest = undefined;
    restoreNotice = "";
    fusionGraph = graph;
    optimizationFinished = false;
    byId("optimization-rewriting-status").textContent = "";
    byId("optimization-rewriting-name").value = "";
    byId("working-rewriting-name").value = "";
    byId("working-rewriting-status").textContent = "";
    optimizationHistory = file?.fusionHistory ?? [];
    pendingRewritings = file?.pendingRewritings ?? [];
    undoCount = file?.fusionCount ?? 0;
    selectedStep = undefined;
    fusionCompletion =
      operationRequestId === fusionRequest ? operationRequestId : undefined;
    fusionRequest++;
    byId("fusion-panel").hidden = true;
    byId("fuse-pattern").setAttribute("aria-expanded", "false");
    byId("optimization-panel").hidden = true;
    byId("optimize-model").setAttribute("aria-expanded", "false");
    clearFusionPreview();
    clearDiagnostic();
    diagnosticNode = undefined;
    nativePatterns = new Set();
    byId("diagnostic-pattern").replaceChildren();
    diagnosticControls();
    showOptimizationReport(null);
    byId("onnxscript-optimization-output").textContent = "";
    byId("onnxscript-optimization-output").hidden = true;
    byId("fusion-status").textContent = "";
    showRewritingState(file?.rewriting ?? { names: [] });
    showScenarios();
    byId("optimization-patterns").replaceChildren();
    showOptimizationResults();
    byId("optimization-feedback").textContent = "";
    byId("optimization-status").textContent = "";
    if (restoring) {
      const count = pendingRewritings.reduce(
        (total, operation) => total + operation.matches.length,
        0,
      );
      restoreNotice = `Model restored. ${count} rewriting${count === 1 ? "" : "s"} kept pending; the next matching or optimization reapplies them.`;
      byId("optimization-panel").hidden = false;
      byId("optimize-model").setAttribute("aria-expanded", "true");
      fusionControls(true);
      requestAction("fusion-patterns", "List optimization patterns", {
        requestId: ++fusionRequest,
      });
    } else if (keepFusion) {
      byId("fusion-panel").hidden = false;
      byId("fuse-pattern").setAttribute("aria-expanded", "true");
      if (canFuse()) {
        fusionControls(true, "Updating fusion patterns…");
        requestAction("fusion-patterns", "List fusion patterns", {
          requestId: ++fusionRequest,
        });
      }
    }
  }
  function setEnabled(enabled) {
    byId("fuse-pattern").disabled =
      undoBusy || creatingPattern || !(enabled || undoCount);
    byId("optimize-model").disabled = creatingPattern || !enabled;
    for (const id of [
      "optimization-unfuse-last",
      "optimization-restore-original",
    ])
      byId(id).disabled = !enabled || fusionBusy || !undoCount;
    byId("replay-rewritings").disabled =
      !enabled || fusionBusy || !pendingRewritings.length;
    diagnosticControls();
    selectionPatternControls();
    updateScenario();
  }

  return {
    impactSummary,
    appendImpactDetails,
    setUndoBusy: (busy) => {
      undoBusy = busy;
      fusionControls(busy);
    },
    handleMessage,
    reset,
    updateGraph: (graph) => {
      fusionGraph = graph;
      fusionControls(false);
    },
    setEnabled,
    selectionChanged,
    libraryChanged: (data) => {
      libraryRevision = data.revision;
      if (!data.invalidateFusion) return;
      restoreFunctionEditor = false;
      creatingPattern = false;
      selectionCatalogLoaded = false;
      fusionRequest++;
      clearFusionPreview();
      clearDiagnostic();
      fusionControls(false);
      if (byId("selection-pattern-dialog").open) {
        byId("selection-pattern-dialog").close();
        byId("selection-pattern-status").textContent =
          "The pattern library changed. Reopen the editor to load its definitions.";
      }
      if (
        canFuse() &&
        (!byId("fusion-panel").hidden || !byId("optimization-panel").hidden)
      ) {
        fusionControls(true);
        requestAction("fusion-patterns", "Refresh pattern library", {
          requestId: ++fusionRequest,
        });
      }
    },
    canOpenSelectionPattern,
    openSelectionPattern,
    refresh: () => fusionControls(fusionBusy),
    invalidate: () => {
      restoreFunctionEditor = false;
      if (byId("selection-pattern-dialog").open)
        byId("selection-pattern-dialog").close();
      creatingPattern = false;
      restoreRequest = undefined;
      fusionRequest++;
      fusionCompletion = undefined;
      clearFusionPreview();
      clearDiagnostic();
      showOptimizationReport(null);
    },
    undoFailed: (data) => {
      if (restoreRequest === undefined || data.requestId !== restoreRequest)
        return;
      restoreRequest = undefined;
      fusionControls(false);
      byId("optimization-status").textContent =
        `Could not undo optimization: ${data.error}`;
    },
  };
}
