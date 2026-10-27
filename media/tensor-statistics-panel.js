function createTensorStatisticsPanel({ byId, postMessage }) {
  const dialog = byId("tensor-statistics-dialog");
  const selector = byId("tensor-statistics-name");
  const status = byId("tensor-statistics-status");
  const content = byId("tensor-statistics-content");
  const cache = new Map();
  let names = [];
  let index = 0;
  let path = [];
  let requestId = 0;
  let pending;
  let opener;
  const ns = "http://www.w3.org/2000/svg";

  function invalidate() {
    pending = undefined;
  }

  function close() {
    invalidate();
    if (dialog.open) {
      dialog.close();
      opener?.focus();
    }
  }

  function reset() {
    close();
    cache.clear();
    names = [];
    selector.value = "";
    byId("tensor-statistics-names").replaceChildren();
    content.replaceChildren();
  }

  function svgElement(tag, attributes = {}, title) {
    const element = document.createElementNS(ns, tag);
    for (const [key, value] of Object.entries(attributes))
      element.setAttribute(key, value);
    if (title !== undefined) {
      const description = document.createElementNS(ns, "title");
      description.textContent = title;
      element.append(description);
    }
    return element;
  }

  function plot(title, viewBox) {
    return svgElement(
      "svg",
      { viewBox, role: "img", "aria-label": title },
      title,
    );
  }

  function render(statistics) {
    const summary = document.createElement("dl");
    summary.className = "status-details";
    for (const [label, value] of [
      ["Name", statistics.name],
      ["Dtype", statistics.dtype],
      ["Shape", `[${statistics.shape.join(", ")}]`],
      ["Count", statistics.count],
      ["Size (bytes)", statistics.sizeBytes],
      ["Average", statistics.mean],
      ["Median", statistics.median],
      ["Minimum", statistics.min],
      ["Maximum", statistics.max],
      ["Q1 (25%)", statistics.quantiles[0]],
      ["Q2 (50%)", statistics.quantiles[1]],
      ["Q3 (75%)", statistics.quantiles[2]],
      ...(statistics.distinctCount === undefined
        ? []
        : [["Distinct count", statistics.distinctCount]]),
    ]) {
      const term = document.createElement("dt");
      const description = document.createElement("dd");
      term.textContent = label;
      description.textContent = String(value ?? "Unavailable");
      summary.append(term, description);
    }
    const charts = document.createElement("div");
    charts.className = "tensor-statistics-charts";
    const histogram = plot(`Histogram for ${statistics.name}`, "0 0 320 180");
    const counts = statistics.distribution.slice(0, 256).map(Number);
    const maximum = Math.max(0, ...counts.filter(Number.isFinite));
    for (const [i, count] of counts.entries()) {
      if (!Number.isFinite(count) || count < 0) continue;
      const height = maximum > 0 ? (count / maximum) * 150 : 0;
      histogram.append(
        svgElement(
          "rect",
          {
            x: (i * 320) / counts.length,
            y: 170 - height,
            width: 320 / counts.length,
            height,
            fill: "currentColor",
          },
          `${statistics.binEdges[i]} to ${statistics.binEdges[i + 1]}: ${statistics.distribution[i]} values`,
        ),
      );
    }
    const heatmap = plot(
      `32 by 32 tensor heatmap for ${statistics.name}`,
      "0 0 32 32",
    );
    const values = statistics.rectangleMap.slice(0, 1024).map(Number);
    const finite = values.filter(Number.isFinite);
    const low = finite.length ? Math.min(...finite) : 0;
    const high = finite.length ? Math.max(...finite) : 0;
    const scale = Math.max(Math.abs(low), Math.abs(high), 1);
    for (const [i, value] of values.entries()) {
      const ratio =
        high === low
          ? 0.5
          : (value / scale - low / scale) / (high / scale - low / scale);
      heatmap.append(
        svgElement(
          "rect",
          {
            x: i % 32,
            y: Math.floor(i / 32),
            width: 1,
            height: 1,
            fill: Number.isFinite(value)
              ? `hsl(${240 * (1 - ratio)}, 80%, 50%)`
              : "var(--muted)",
          },
          `Row ${Math.floor(i / 32) + 1}, column ${(i % 32) + 1}: ${statistics.rectangleMap[i]}`,
        ),
      );
    }
    const legend = document.createElement("p");
    legend.className = "tensor-statistics-legend";
    legend.textContent = finite.length
      ? `Heatmap: blue (low) ${low} → red (high) ${high}. Gray: non-finite. SDK 32 × 32 summary, not model execution.`
      : "Heatmap unavailable: no finite values. Gray: non-finite.";
    charts.append(histogram, heatmap);
    content.replaceChildren(summary, charts, legend);
  }

  function show() {
    invalidate();
    const name = names[index];
    selector.value = name;
    byId("tensor-statistics-position").textContent =
      `${index + 1} of ${names.length}`;
    byId("tensor-statistics-previous").disabled = index === 0;
    byId("tensor-statistics-next").disabled = index === names.length - 1;
    // Keep the name picker bounded even for graphs with millions of values.
    byId("tensor-statistics-names").replaceChildren(
      ...names.slice(Math.max(0, index - 25), index + 75).map((value) => {
        const option = document.createElement("option");
        option.value = value;
        return option;
      }),
    );
    content.replaceChildren();
    if (cache.has(name)) {
      status.textContent = "";
      dialog.setAttribute("aria-busy", "false");
      render(cache.get(name));
      return;
    }
    status.textContent = `Loading tensor statistics for ${name}…`;
    dialog.setAttribute("aria-busy", "true");
    pending = { requestId: ++requestId, name };
    postMessage({ type: "tensor-statistics", ...pending, path });
  }

  selector.addEventListener("input", () => {
    invalidate();
    content.replaceChildren();
    dialog.setAttribute("aria-busy", "false");
    status.textContent = "Choose a tensor name, then confirm the selection.";
  });
  selector.addEventListener("change", () => {
    const next = names.indexOf(selector.value);
    if (next < 0) {
      invalidate();
      content.replaceChildren();
      dialog.setAttribute("aria-busy", "false");
      status.textContent = "Choose a tensor name in this batch.";
      return;
    }
    index = next;
    show();
  });
  for (const [id, delta] of [
    ["previous", -1],
    ["next", 1],
  ])
    byId(`tensor-statistics-${id}`).addEventListener("click", () => {
      index = Math.max(0, Math.min(names.length - 1, index + delta));
      show();
    });
  byId("close-tensor-statistics").addEventListener("click", close);
  dialog.addEventListener("cancel", invalidate);
  dialog.addEventListener("close", invalidate);

  return {
    reset,
    open(batch, graphPath) {
      names = [
        ...new Set(batch.filter((name) => typeof name === "string" && name)),
      ];
      if (!names.length) return;
      path = graphPath;
      index = 0;
      opener = document.activeElement;
      if (!dialog.open) dialog.showModal();
      show();
    },
    handleMessage(message) {
      if (
        !["tensor-statistics-info", "tensor-statistics-error"].includes(
          message?.type,
        )
      )
        return false;
      if (!dialog.open || !pending || message.requestId !== pending.requestId)
        return true;
      const name = pending.name;
      invalidate();
      dialog.setAttribute("aria-busy", "false");
      if (message.type === "tensor-statistics-error") {
        status.textContent = `Tensor data unavailable: ${message.error}. The model is not executed; intermediate values may have no stored data. Choose another tensor to continue.`;
      } else {
        cache.delete(name);
        cache.set(name, message.statistics);
        if (cache.size > 32) cache.delete(cache.keys().next().value);
        status.textContent = "";
        render(message.statistics);
      }
      return true;
    },
  };
}
