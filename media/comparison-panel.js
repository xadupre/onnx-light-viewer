function createComparisonPanel({
  byId,
  postMessage,
  requestAction,
  revealPanel,
  impactSummary,
  appendImpactDetails,
}) {
  const pageSize = 100;
  let request = 0;
  let available = false;
  let pipeline = false;
  let busy = false;
  let diff;
  let rendered = 0;

  function controls() {
    byId("show-comparison").disabled = !available;
    byId("compare-choose").disabled = !available || pipeline || busy;
    byId("comparison-panel").setAttribute("aria-busy", String(busy));
  }

  function invalidate() {
    if (busy) postMessage({ type: "cancel-comparison" });
    request++;
    busy = false;
    diff = undefined;
    rendered = 0;
    byId("compare-result").hidden = true;
    byId("compare-steps").replaceChildren();
    byId("compare-declarations").replaceChildren();
    byId("compare-more").hidden = true;
    byId("compare-status").textContent = pipeline
      ? "Pipeline containers cannot be compared."
      : "Choose another ONNX model to compare.";
    controls();
  }

  function setGraph(graph, isPipeline) {
    pipeline = isPipeline === true || graph?.pipeline === true;
    invalidate();
  }

  function setAvailable(enabled) {
    available = enabled;
    controls();
  }

  function appendLines(container, title, values) {
    if (!values.length) return;
    const text = document.createElement("p");
    text.textContent = `${title}: ${values.join(", ")}`;
    container.append(text);
  }

  function showSteps() {
    if (!diff) return;
    const list = byId("compare-steps");
    const end = Math.min(rendered + pageSize, diff.steps.length);
    for (; rendered < end; rendered++) {
      const step = diff.steps[rendered];
      const item = document.createElement("li");
      const heading = document.createElement("strong");
      heading.textContent = step.pattern || "Rewriting";
      const summary = document.createElement("p");
      summary.textContent = impactSummary(step);
      item.append(heading, summary);
      appendLines(
        item,
        "Removed initializer positions before this step",
        step.removedInitializers.map((index) => `#${index}`),
      );
      appendLines(
        item,
        "Removed extended initializer positions before this step",
        step.removedExtendedInitializers.map((index) => `#${index}`),
      );
      appendLines(item, "Added initializers", step.addedInitializers);
      appendLines(item, "Prepared initializers", step.preparedInitializers);
      appendImpactDetails(item, step);
      list.append(item);
    }
    const more = byId("compare-more");
    more.hidden = rendered === diff.steps.length;
    more.textContent = `Show more steps (${diff.steps.length - rendered} remaining)`;
  }

  function declarations(title, values, format) {
    if (!values.length) return;
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = `${title} (${values.length})`;
    details.append(summary);
    details.addEventListener("toggle", () => {
      if (!details.open || details.childElementCount > 1) return;
      const list = document.createElement("ul");
      for (const value of values) {
        const item = document.createElement("li");
        item.textContent = format(value);
        list.append(item);
      }
      details.append(list);
    });
    byId("compare-declarations").append(details);
  }

  function showResult(message) {
    diff = message.diff;
    rendered = 0;
    byId("compare-result").hidden = false;
    byId("compare-models").textContent =
      `Current: ${message.source} → Target: ${message.target}`;
    const count =
      diff.steps.length +
      diff.moves.length +
      diff.opsets.length +
      diff.functions.length +
      diff.replacedFunctions.length;
    byId("compare-summary").textContent = count
      ? `${diff.steps.length} rewriting step${diff.steps.length === 1 ? "" : "s"}, ${diff.moves.length} node move${diff.moves.length === 1 ? "" : "s"}.`
      : "No rewriting differences reported. This is not a byte-for-byte equality check.";
    showSteps();
    declarations("Node moves", diff.moves, (move) => {
      const scope = move.scope.length ? ` in ${move.scope.join(" / ")}` : "";
      const fn = move.function ? ` in function ${move.function}` : "";
      return `#${move.from} → #${move.to}${scope}${fn}`;
    });
    declarations(
      "Required opset imports",
      diff.opsets,
      (opset) => `${opset.domain || "ai.onnx"}: ${opset.version}`,
    );
    declarations("Added functions", diff.functions, (name) => name);
    declarations("Replaced functions", diff.replacedFunctions, (name) => name);
  }

  byId("show-comparison").addEventListener("click", () => {
    const panel = byId("comparison-panel");
    panel.hidden = !panel.hidden;
    byId("show-comparison").setAttribute(
      "aria-expanded",
      String(!panel.hidden),
    );
    if (!panel.hidden) revealPanel(byId("fusion-panel"));
    else if (busy) invalidate();
  });
  byId("fuse-pattern").addEventListener("click", () => {
    if (byId("fusion-panel").hidden && busy) invalidate();
  });
  byId("compare-choose").addEventListener("click", () => {
    if (!available || pipeline || busy) return;
    invalidate();
    busy = true;
    byId("compare-status").textContent =
      "Choosing and comparing the target model…";
    controls();
    requestAction("compare-model", "Compare ONNX models", {
      requestId: ++request,
    });
  });
  byId("compare-more").addEventListener("click", showSteps);

  function handleMessage(message) {
    if (!["compare-result", "compare-error"].includes(message?.type))
      return false;
    if (message.requestId !== request || !busy) return true;
    busy = false;
    if (message.type === "compare-error")
      byId("compare-status").textContent =
        `Comparison failed: ${message.error}`;
    else if (message.cancelled)
      byId("compare-status").textContent = "Comparison cancelled.";
    else {
      byId("compare-status").textContent = "Comparison complete.";
      showResult(message);
    }
    controls();
    return true;
  }

  return { setGraph, setAvailable, invalidate, handleMessage };
}
