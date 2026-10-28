function createFusionPanel({
  byId,
  postMessage,
  requestAction,
  revealPanel,
  fit,
  canFuse,
  clearSelection,
  selectedNode,
  focusNodes,
}) {
  let fusionGraph;
  let fusionRequest = 0;
  let fusionCompletion;
  let fusionPreview;
  let fusionBusy = false;
  let optimizationSelection;
  let optimizationReport;
  let optimizationBackend = "onnx-light";
  let selectedOptimizationResult = false;
  let optimizationHistory = [];
  let undoCount = 0;
  let restoreRequest;
  let scenarios = [];
  let selectedStep;
  let diagnosticRequest = 0;
  let diagnosticBusy = false;
  let diagnosticNode;
  let diagnosticResult;
  let nativePatterns = new Set();
  function diagnosticControls() {
    const picker = byId("diagnostic-pattern");
    picker.disabled = fusionBusy || !canFuse() || !picker.options.length;
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
    const node = selectedNode();
    if (node?.id === diagnosticNode?.id) return;
    diagnosticNode = node;
    clearDiagnostic();
  }
  function updateDiagnosticPatterns() {
    const picker = byId("diagnostic-pattern");
    const previous = picker.value;
    picker.replaceChildren();
    for (const input of byId("fusion-pattern").querySelectorAll(
      "input:checked",
    )) {
      if (!nativePatterns.has(input.value)) continue;
      const option = document.createElement("option");
      option.value = option.textContent = input.value;
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
    const version = byId("onnx-light-version").dataset.version;
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
    for (const name of headers) {
      const cell = document.createElement("th");
      cell.scope = "col";
      cell.textContent = name;
      heading.append(cell);
    }
    const body = table.createTBody();
    for (const values of rows) {
      const row = body.insertRow();
      for (const [index, value] of values.entries()) {
        const cell = document.createElement(index ? "td" : "th");
        if (!index) cell.scope = "row";
        if (value instanceof Node) cell.append(value);
        else cell.textContent = String(value);
        row.append(cell);
      }
    }
    if (!rows.length) {
      const cell = body.insertRow().insertCell();
      cell.colSpan = headers.length;
      cell.textContent = emptyMessage;
    }
    byId("optimization-report-body").append(table);
  }
  function openOptimizationReport() {
    if (byId("show-optimization-report").disabled) return;
    const report = optimizationReport;
    const formatter = new Intl.NumberFormat(undefined, {
      maximumFractionDigits: 6,
    });
    const milliseconds = (ns) => formatter.format(ns / 1e6);
    byId("optimization-report-body").replaceChildren();
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
      ...byId("rewriting-steps").querySelectorAll("button"),
    ].entries())
      button.setAttribute("aria-current", String(stepIndex === index));
    focusNodes(step.nodes);
  }
  function updateScenario() {
    const scenario = currentScenario();
    byId("rewriting-scenario").disabled = fusionBusy || !scenarios.length;
    for (const id of ["rewriting-next", "rewriting-all"])
      byId(id).disabled = fusionBusy || !scenario?.available || !canFuse();
    byId("rewriting-steps").replaceChildren();
    for (const [index, step] of (scenario?.steps ?? []).entries()) {
      const item = document.createElement("li");
      const button = document.createElement("button");
      button.textContent = `${index < scenario.applied ? "Applied: " : ""}${step.pattern}${step.path.length ? ` [${step.path.join(" / ")}]` : ""}`;
      button.disabled = fusionBusy || !canFuse() || !step.nodes.length;
      button.title = step.path.length
        ? "This step edits a subgraph, not nodes in the main graph."
        : index > scenario.applied
          ? "Apply preceding steps to inspect these nodes."
          : step.nodes.length
            ? "Select this step's nodes in the graph."
            : "No nodes from this step are present in the main graph.";
      button.setAttribute(
        "aria-current",
        String(
          selectedStep?.name === scenario.name && selectedStep.index === index,
        ),
      );
      button.addEventListener("click", () => focusScenarioStep(index));
      item.append(button);
      byId("rewriting-steps").append(item);
    }
    byId("rewriting-progress").textContent = scenario
      ? `${scenario.applied} / ${scenario.steps.length} steps applied.${!scenario.available && scenario.applied < scenario.steps.length ? " Unfuse current edits before applying this scenario." : ""}`
      : "No saved rewriting scenarios in this model.";
  }
  function showScenarios(values, name) {
    const previous = name ?? byId("rewriting-scenario").value;
    const hadScenarios = scenarios.length > 0;
    scenarios = values ?? [];
    if (!scenarios.length || !hadScenarios)
      byId("rewriting-scenarios").open = scenarios.length > 0;
    byId("rewriting-scenario").replaceChildren();
    for (const scenario of [...scenarios].sort((a, b) =>
      a.name.localeCompare(b.name),
    )) {
      const option = document.createElement("option");
      option.value = option.textContent = scenario.name;
      byId("rewriting-scenario").append(option);
    }
    const active = scenarios.find((scenario) => scenario.applied)?.name;
    if (scenarios.some((scenario) => scenario.name === (active ?? previous)))
      byId("rewriting-scenario").value = active ?? previous;
    updateScenario();
  }
  function fusionControls(busy, message = "") {
    if (busy && diagnosticBusy) clearDiagnostic();
    fusionBusy = busy;
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
      optimizationBackend !== "onnx-light";
    for (const id of [
      "optimization-unfuse-last",
      "optimization-restore-original",
    ])
      byId(id).disabled = busy || !canFuse() || !undoCount;
    for (const control of byId("fusion-matches").querySelectorAll(
      "input, button",
    ))
      control.disabled = busy;
    byId("fusion-panel").setAttribute("aria-busy", String(busy));
    byId("optimization-panel").setAttribute("aria-busy", String(busy));
    const patterns = byId("fusion-pattern");
    patterns.disabled = busy || !patterns.querySelector("input");
    byId("fusion-pattern-group").disabled =
      patterns.disabled || !byId("fusion-pattern-group").querySelector("input");
    byId("select-patterns").disabled = patterns.disabled;
    byId("clear-patterns").disabled =
      busy || !patterns.querySelector("input:checked");
    byId("preview-fusion").disabled =
      busy || !patterns.querySelector("input:checked");
    byId("select-fusions").disabled = busy || !fusionPreview?.matches.length;
    byId("clear-fusions").disabled = busy || !fusionSelection().length;
    byId("apply-fusion").disabled = busy || !fusionSelection().length;
    byId("fuse-pattern").disabled = !canFuse();
    byId("optimize-model").disabled = !canFuse();
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
            : `${candidates.size} selected / ${fusionPreview.matches.length} matches. Check candidates, then click Fuse selected.`;
  }

  function clearFusionPreview() {
    if (fusionPreview) {
      clearSelection();
    }
    fusionPreview = undefined;
    if (selectedOptimizationResult) clearSelection();
    selectedOptimizationResult = false;
    byId("fusion-matches").replaceChildren();
    fusionControls(false);
  }

  function showOptimizationPatterns(patterns) {
    if (!patterns) return;
    nativePatterns = new Set(patterns.map((pattern) => pattern.name));
    const fieldset = byId("optimization-patterns");
    fieldset.replaceChildren();
    const priorities = [
      ...new Set(patterns.map((pattern) => pattern.priority)),
    ].sort((a, b) => a - b);
    for (const priority of priorities) {
      const group = document.createElement("details");
      const summary = document.createElement("summary");
      const toggle = document.createElement("input");
      toggle.type = "checkbox";
      toggle.dataset.priority = String(priority);
      toggle.setAttribute("aria-label", `Select priority ${priority} patterns`);
      const members = patterns
        .filter((pattern) => pattern.priority === priority)
        .sort((a, b) => a.name.localeCompare(b.name));
      summary.append(
        toggle,
        document.createTextNode(
          ` Priority ${priority} (${members.length} patterns)`,
        ),
      );
      group.append(summary);
      for (const pattern of members) {
        const label = document.createElement("label");
        const input = document.createElement("input");
        input.type = "checkbox";
        input.value = pattern.name;
        input.dataset.pattern = pattern.name;
        input.checked = optimizationSelection
          ? optimizationSelection.has(pattern.name)
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
    updateOptimizationSelection();
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
    for (const group of fieldset.querySelectorAll("details")) {
      const toggle = group.querySelector("input[data-priority]");
      const count = group.querySelectorAll("input[data-pattern]").length;
      const checked = group.querySelectorAll(
        "input[data-pattern]:checked",
      ).length;
      toggle.checked = count === checked;
      toggle.indeterminate = checked > 0 && checked < count;
    }
    byId("optimization-feedback").textContent =
      `${selected.length} / ${fieldset.querySelectorAll("input[data-pattern]").length} patterns selected.`;
    fusionControls(fusionBusy);
  }

  function showOptimizationResults() {
    byId("optimization-results").replaceChildren();
    const path = fusionGraph?.breadcrumbs?.at(-1)?.path ?? [];
    const stage = path.length === 1 ? path[0].pipeline : undefined;
    const matches = optimizationHistory
      .filter((operation) => operation.stage === stage)
      .flatMap((operation) => operation.matches);
    for (const match of matches) {
      const item = document.createElement("li");
      const button = document.createElement("button");
      button.textContent = `${match.pattern} (${match.nodes.length} nodes)`;
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
      byId("optimization-results").append(item);
    }
    if (!matches.length) {
      const item = document.createElement("li");
      item.textContent = "No applied pattern matches in this graph.";
      byId("optimization-results").append(item);
    }
  }

  function previewFusion() {
    clearFusionPreview();
    byId("fusion-status").textContent = "";
    fusionControls(true, "Matching patterns and building replacements…");
    requestAction("preview-fusion", "Preview fusion pattern", {
      requestId: ++fusionRequest,
      patterns: [
        ...byId("fusion-pattern").querySelectorAll("input:checked"),
      ].map((input) => input.value),
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
      showOptimizationReport(null);
      byId("optimization-status").textContent = "";
      showOptimizationResults();
      fusionControls(fusionBusy);
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
    fusionRequest++;
    clearFusionPreview();
    updateDiagnosticPatterns();
    byId("fusion-feedback").textContent =
      "Click Run Matching to find candidates for the selected patterns.";
    postMessage({ type: "cancel-fusion" });
  }
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
  byId("preview-fusion").addEventListener("click", previewFusion);
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
  byId("rewriting-scenario").addEventListener("change", () => {
    selectedStep = undefined;
    clearSelection();
    updateScenario();
  });
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

  function handleMessage(data) {
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
        const optimize = data.type === "optimization-info";
        byId(optimize ? "optimization-panel" : "fusion-panel").hidden = false;
        byId(optimize ? "optimize-model" : "fuse-pattern").setAttribute(
          "aria-expanded",
          "true",
        );
      }
      showScenarios(data.scenarios, data.name);
      showOptimizationPatterns(data.optimizationPatterns);
      showOptimizationReport(data.optimizationReport);
      if (data.type === "optimization-info")
        byId("onnxscript-optimization-output").textContent =
          data.backend === "onnxscript" ? (data.logs ?? "") : "";
      if (data.type === "fusion-patterns")
        byId("optimization-status").textContent = "";
      byId("fusion-pattern").replaceChildren();
      for (const pattern of [...data.patterns].sort()) {
        const label = document.createElement("label");
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.value = pattern;
        label.append(checkbox, document.createTextNode(pattern));
        byId("fusion-pattern").append(label);
      }
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
      fusionControls(
        false,
        data.patterns.length
          ? `${data.patterns.length} native patterns available. Select patterns, then click Run Matching.`
          : "No native patterns have compatible anchors in this graph.",
      );
      if (data.type !== "fusion-patterns") {
        byId("fusion-status").textContent =
          data.type === "optimization-info"
            ? data.backend === "onnxscript"
              ? data.changed
                ? ""
                : "ONNX Script: no changes."
              : data.changed
                ? `${data.operation}: applied ${data.matches.length} rewriting steps. Use Unfuse to undo, or Save with or without rewritings.`
                : `${data.operation}: no changes.`
            : data.type === "rewriting-info"
              ? `Applied rewriting scenario ${data.name}. Use Unfuse to undo.`
              : `Applied ${data.count} fusion matches. Save a copy to keep changes.`;
        if (data.notice) byId("fusion-status").textContent += ` ${data.notice}`;
        updateScenario();
        revealPanel(
          byId(
            data.type === "optimization-info"
              ? "optimization-panel"
              : "fusion-panel",
          ),
        );
        fit();
        if (data.type === "rewriting-info")
          focusScenarioStep(currentScenario()?.applied);
        if (data.type === "optimization-info") {
          byId("optimization-status").textContent =
            byId("fusion-status").textContent;
          showOptimizationResults();
        }
      }
    } else if (data?.type === "fusion-preview") {
      if (data.requestId === fusionRequest) showFusionMatches(data);
    } else if (data?.type === "fusion-error") {
      if (data.requestId !== fusionRequest) return true;
      if (
        optimizationBackend === "onnxscript" &&
        !byId("optimization-panel").hidden
      )
        byId("onnxscript-optimization-output").textContent = data.error;
      fusionControls(false, data.error);
      byId("fusion-status").textContent = `Fusion failed: ${data.error}`;
      if (!byId("optimization-panel").hidden)
        byId("optimization-status").textContent =
          `Optimization failed: ${data.error}`;
    } else {
      return false;
    }
    return true;
  }
  function reset(graph, operationRequestId, file) {
    const restoring =
      restoreRequest !== undefined && restoreRequest === operationRequestId;
    restoreRequest = undefined;
    fusionGraph = graph;
    optimizationHistory = file?.fusionHistory ?? [];
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
    showScenarios([]);
    byId("optimization-patterns").replaceChildren();
    showOptimizationResults();
    byId("optimization-feedback").textContent = "";
    byId("optimization-status").textContent = "";
    if (restoring) {
      byId("optimization-panel").hidden = false;
      byId("optimize-model").setAttribute("aria-expanded", "true");
      fusionControls(true);
      requestAction("fusion-patterns", "List optimization patterns", {
        requestId: ++fusionRequest,
      });
    }
  }
  function setEnabled(enabled) {
    byId("fuse-pattern").disabled = !enabled;
    byId("optimize-model").disabled = !enabled;
    for (const id of [
      "optimization-unfuse-last",
      "optimization-restore-original",
    ])
      byId(id).disabled = !enabled || fusionBusy || !undoCount;
    diagnosticControls();
  }

  return {
    handleMessage,
    reset,
    setEnabled,
    selectionChanged,
    refresh: () => fusionControls(fusionBusy),
    invalidate: () => {
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
