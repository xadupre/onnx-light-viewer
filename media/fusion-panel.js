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
  let fusionPreview;
  let fusionBusy = false;
  function fusionControls(busy, message = "") {
    fusionBusy = busy;
    for (const control of byId("fusion-matches").querySelectorAll(
      "input, button",
    ))
      control.disabled = busy;
    byId("fusion-panel").setAttribute("aria-busy", String(busy));
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
    for (const match of fusionPreview?.matches ?? []) {
      if (!candidates.has(match.id)) continue;
      for (const node of match.nodes) {
        if (nodes.has(node)) overlap = true;
        nodes.add(node);
      }
    }
    byId("apply-fusion").disabled = fusionBusy || !candidates.size || overlap;
    byId("clear-fusions").disabled = fusionBusy || !candidates.size;
    byId("fusion-feedback").textContent = overlap
      ? "Selected matches overlap. Select disjoint matches."
      : `${candidates.size} selected / ${fusionPreview?.matches.length ?? 0} matches. Check candidates, then click Fuse selected.`;
  }

  function clearFusionPreview() {
    if (fusionPreview) {
      clearSelection();
    }
    fusionPreview = undefined;
    byId("fusion-matches").replaceChildren();
    fusionControls(false);
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

  function showFusionMatches(data) {
    fusionPreview = {
      id: data.previewId,
      matches: data.matches.map((match, id) => ({ ...match, id })),
    };
    byId("fusion-matches").replaceChildren();
    for (const match of fusionPreview.matches) {
      const item = document.createElement("li");
      item.className = "fusion-match";
      item.dataset.candidate = String(match.id);
      const label = document.createElement("label");
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.value = String(match.id);
      checkbox.addEventListener("change", () => {
        focusFusionMatch(match);
        updateFusionSelection();
      });
      const names = match.nodes.map((index) => {
        const node = fusionGraph.nodes[index];
        return `${node.name || node.id} (${node.opType})`;
      });
      label.append(
        checkbox,
        document.createTextNode(
          `${match.pattern ? `${match.pattern}: ` : ""}${names.join(" + ")}`,
        ),
      );
      const replacement = document.createElement("pre");
      replacement.textContent = match.replacement.length
        ? match.replacement
            .map(
              (node) =>
                `${node.domain ? `${node.domain}::` : ""}${node.opType}(${node.inputs.join(", ")}) -> ${node.outputs.join(", ")}${node.attributes.length ? `\n  ${node.attributes.map((attribute) => `${attribute.name}=${attribute.value}`).join(", ")}` : ""}`,
            )
            .join("\n")
        : "Remove matched nodes (no replacement operators).";
      item.append(label, replacement);
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
      byId("close-fusion").click();
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
  byId("close-fusion").addEventListener("click", () => {
    fusionRequest++;
    byId("fusion-panel").hidden = true;
    byId("fuse-pattern").setAttribute("aria-expanded", "false");
    clearFusionPreview();
    fit();
    postMessage({ type: "cancel-fusion" });
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
    for (const input of byId("fusion-matches").querySelectorAll("input")) {
      const match = fusionPreview.matches.find(
        (candidate) => candidate.id === Number(input.value),
      );
      input.checked = !match.nodes.some((node) => used.has(node));
      if (input.checked) for (const node of match.nodes) used.add(node);
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

  function handleMessage(data) {
    if (data?.type === "fusion-patterns") {
      if (data.requestId !== fusionRequest) return true;
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
    } else if (data?.type === "fusion-preview") {
      if (data.requestId === fusionRequest) showFusionMatches(data);
    } else if (data?.type === "fusion-error") {
      if (data.requestId !== fusionRequest) return true;
      fusionControls(false, data.error);
      byId("fusion-status").textContent = `Fusion failed: ${data.error}`;
    } else if (data?.type === "fusion-info") {
      byId("fusion-status").textContent =
        `Applied ${data.count} fusion matches. Save a copy to keep changes.`;
    } else {
      return false;
    }
    return true;
  }
  function reset(graph) {
    fusionGraph = graph;
    fusionRequest++;
    byId("fusion-panel").hidden = true;
    byId("fuse-pattern").setAttribute("aria-expanded", "false");
    clearFusionPreview();
    byId("fusion-status").textContent = "";
  }
  function setEnabled(enabled) {
    byId("fuse-pattern").disabled = !enabled;
  }

  return {
    handleMessage,
    reset,
    setEnabled,
    refresh: () => fusionControls(false),
  };
}
