function createFusionPanel({
  byId,
  postMessage,
  requestAction,
  revealPanel,
  fit,
  canFuse,
  clearSelection,
  focusNodes,
}) {
  let fusionGraph;
  let fusionRequest = 0;
  let fusionCompletion;
  let fusionPreview;
  let fusionBusy = false;
  let optimizationSelection;
  let selectedOptimizationResult = false;
  let scenarios = [];
  let selectedStep;
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
    scenarios = values ?? [];
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
    fusionBusy = busy;
    for (const control of byId("fusion-matches").querySelectorAll(
      "input, button",
    ))
      control.disabled = busy;
    byId("fusion-panel").setAttribute("aria-busy", String(busy));
    byId("optimization-panel").setAttribute("aria-busy", String(busy));
    const patterns = byId("fusion-pattern");
    patterns.disabled = busy || !patterns.querySelector("input");
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
    byId("select-optimization-patterns").disabled = optimization.disabled;
    for (const id of ["clear-optimization-patterns", "run-optimization"])
      byId(id).disabled =
        busy ||
        !canFuse() ||
        !optimization.querySelector("input[data-pattern]:checked");
    for (const button of byId("optimization-results").querySelectorAll(
      "button",
    ))
      button.disabled = busy || button.dataset.empty === "true";
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
          : `${candidates.size} selected / ${fusionPreview?.matches.length ?? 0} matches. Check candidates, then click Fuse selected.`;
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
          : true;
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

  function showOptimizationResults(data) {
    byId("optimization-results").replaceChildren();
    for (const match of data.matches) {
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
    byId("optimization-picker").open = true;
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
  }
  byId("fuse-pattern").addEventListener("click", () => {
    if (!byId("fusion-panel").hidden) {
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
  for (const [id, checked] of [
    ["select-optimization-patterns", true],
    ["clear-optimization-patterns", false],
  ])
    byId(id).addEventListener("click", () => {
      for (const input of byId("optimization-patterns").querySelectorAll(
        "input[data-pattern]",
      ))
        input.checked = checked;
      updateOptimizationSelection();
    });
  byId("run-optimization").addEventListener("click", () => {
    clearFusionPreview();
    fusionControls(true, "Running native iterative optimization…");
    byId("optimization-status").textContent =
      "Running the complete native optimizer…";
    requestAction("apply-optimization", "Optimize ONNX model", {
      requestId: ++fusionRequest,
      operation: "Optimize",
      patterns: [...optimizationSelection],
    });
  });
  function changeFusionPatterns() {
    fusionRequest++;
    clearFusionPreview();
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
      changeFusionPatterns();
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
      fusionControls(
        false,
        data.patterns.length
          ? `${data.patterns.length} native patterns available. Select patterns, then click Run Matching.`
          : "No native patterns have compatible anchors in this graph.",
      );
      if (data.type !== "fusion-patterns") {
        byId("fusion-status").textContent =
          data.type === "optimization-info"
            ? data.changed
              ? `${data.operation}: applied ${data.matches.length} rewriting steps. Use Unfuse to undo, or Save with or without rewritings.`
              : `${data.operation}: no changes.`
            : data.type === "rewriting-info"
              ? `Applied rewriting scenario ${data.name}. Use Unfuse to undo.`
              : `Applied ${data.count} fusion matches. Save a copy to keep changes.`;
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
          showOptimizationResults(data);
        }
      }
    } else if (data?.type === "fusion-preview") {
      if (data.requestId === fusionRequest) showFusionMatches(data);
    } else if (data?.type === "fusion-error") {
      if (data.requestId !== fusionRequest) return true;
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
  function reset(graph, operationRequestId) {
    fusionGraph = graph;
    selectedStep = undefined;
    fusionCompletion =
      operationRequestId === fusionRequest ? operationRequestId : undefined;
    fusionRequest++;
    byId("fusion-panel").hidden = true;
    byId("fuse-pattern").setAttribute("aria-expanded", "false");
    byId("optimization-panel").hidden = true;
    byId("optimize-model").setAttribute("aria-expanded", "false");
    clearFusionPreview();
    byId("fusion-status").textContent = "";
    showScenarios([]);
    byId("optimization-patterns").replaceChildren();
    byId("optimization-results").replaceChildren();
    byId("optimization-feedback").textContent = "";
    byId("optimization-status").textContent = "";
  }
  function setEnabled(enabled) {
    byId("fuse-pattern").disabled = !enabled;
    byId("optimize-model").disabled = !enabled;
  }

  return {
    handleMessage,
    reset,
    setEnabled,
    refresh: () => fusionControls(false),
    invalidate: () => {
      fusionRequest++;
      fusionCompletion = undefined;
      clearFusionPreview();
    },
  };
}
