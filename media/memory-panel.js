function createMemoryPanel({ byId, postMessage, formatFileSize }) {
  let hasModel = false;
  let requestId = 0;
  let path;
  let configured = false;
  let profile;
  let page = 0;
  const pageSize = 100;
  const view = byId("memory-view");
  const fields = byId("memory-dimensions");
  const status = byId("memory-status");
  const results = byId("memory-results");

  function busy(pending) {
    view.setAttribute("aria-busy", String(pending));
    byId("memory-run").disabled = pending || !configured;
    byId("memory-stage").disabled = pending;
    for (const input of fields.querySelectorAll("input"))
      input.disabled = pending;
  }

  function open() {
    results.hidden = true;
    configured = false;
    status.textContent = hasModel
      ? "Reading model dimensions…"
      : "Open a model to estimate its memory.";
    busy(hasModel);
    const id = ++requestId;
    if (hasModel)
      postMessage({
        type: "memory-inputs",
        requestId: id,
        ...(path ? { path } : {}),
      });
  }

  function table() {
    const events = profile.events;
    const start = page * pageSize;
    const body = byId("memory-events");
    body.replaceChildren();
    for (const event of events.slice(start, start + pageSize)) {
      const row = document.createElement("tr");
      for (const [index, value] of [
        event.index + 1,
        event.node || `node-${event.index}`,
        event.opType,
        formatFileSize(event.existingBytes),
        formatFileSize(event.outputBytes),
        formatFileSize(event.scratchBytes),
        formatFileSize(event.deviceBytes),
      ].entries()) {
        const cell = document.createElement(index === 0 ? "th" : "td");
        if (index === 0) cell.scope = "row";
        cell.textContent = value;
        row.append(cell);
      }
      body.append(row);
    }
    byId("memory-range").textContent = events.length
      ? `Events ${start + 1}–${Math.min(start + pageSize, events.length)} of ${events.length.toLocaleString()}`
      : "No executable nodes; no per-node peak is available.";
    byId("memory-previous").disabled = page === 0;
    byId("memory-next").disabled = start + pageSize >= events.length;
  }

  function chart() {
    const events = profile.events;
    const peak = profile.peakBytes ?? 0;
    const points = events.map((event, index) => {
      const x = 60 + (920 * index) / Math.max(1, events.length - 1);
      const y = 240 - (210 * event.deviceBytes) / Math.max(1, peak);
      return `${index ? "L" : "M"}${x.toFixed(2)},${y.toFixed(2)}`;
    });
    if (events.length === 1)
      points.push(
        `L980,${(240 - (210 * events[0].deviceBytes) / Math.max(1, peak)).toFixed(2)}`,
      );
    byId("memory-curve").setAttribute("d", points.join(" "));
    byId("memory-chart").toggleAttribute("hidden", events.length === 0);
    byId("memory-chart-peak").textContent = formatFileSize(peak);
    byId("memory-chart-last").textContent = events.length.toLocaleString();
  }

  byId("memory-form").addEventListener("submit", (event) => {
    event.preventDefault();
    if (!configured || byId("memory-run").disabled) return;
    const dimensions = {};
    for (const input of fields.querySelectorAll("input")) {
      const value = Number(input.value);
      if (!Number.isSafeInteger(value) || value <= 0) {
        status.textContent =
          "Enter positive safe integers for every dynamic dimension.";
        return;
      }
      dimensions[input.name] = value;
    }
    results.hidden = true;
    status.textContent = "Estimating memory without executing the model…";
    busy(true);
    postMessage({
      type: "memory-profile",
      requestId: ++requestId,
      dimensions,
      path,
    });
  });
  fields.addEventListener("input", () => {
    results.hidden = true;
    status.textContent =
      "Dimensions changed. Estimate again to update the profile.";
  });
  byId("memory-stage").addEventListener("change", () => {
    path = JSON.parse(byId("memory-stage").value);
    fields.replaceChildren();
    open();
  });
  for (const [id, delta] of [
    ["memory-previous", -1],
    ["memory-next", 1],
  ])
    byId(id).addEventListener("click", () => {
      page += delta;
      table();
    });

  function handleMessage(message) {
    if (
      !["memory-inputs", "memory-profile", "memory-error"].includes(
        message?.type,
      )
    )
      return false;
    if (message.requestId !== requestId || view.hidden) return true;
    if (message.stages) {
      const stage = byId("memory-stage");
      stage.replaceChildren();
      for (const item of message.stages) {
        const option = document.createElement("option");
        option.value = JSON.stringify(item.path);
        option.textContent = item.name;
        stage.append(option);
      }
      path = message.path;
      stage.value = JSON.stringify(path);
      byId("memory-stage-field").hidden = message.stages.length === 0;
    }
    if (message.type === "memory-error") {
      status.textContent = `Memory estimate unavailable: ${message.error}`;
      results.hidden = true;
    } else if (message.type === "memory-inputs") {
      const previous = new Map(
        [...fields.querySelectorAll("input")].map((input) => [
          input.name,
          input.value,
        ]),
      );
      fields.replaceChildren();
      for (const [index, dimension] of message.dimensions.entries()) {
        const label = document.createElement("label");
        label.textContent = dimension.label;
        const input = document.createElement("input");
        input.id = `memory-dimension-${index}`;
        label.htmlFor = input.id;
        input.name = dimension.key;
        input.type = "number";
        input.min = "1";
        input.max = String(Number.MAX_SAFE_INTEGER);
        input.step = "1";
        input.required = true;
        input.value =
          previous.get(dimension.key) ?? String(dimension.defaultValue);
        fields.append(label, input);
      }
      byId("memory-no-dimensions").hidden = message.dimensions.length !== 0;
      configured = true;
      status.textContent =
        "Set dynamic dimensions, then choose Estimate memory.";
    } else {
      profile = message.profile;
      page = 0;
      byId("memory-peak").textContent =
        profile.peakBytes === null
          ? "Unavailable (no nodes)"
          : formatFileSize(profile.peakBytes);
      byId("memory-device").textContent = profile.device;
      byId("memory-model-name").textContent = profile.name || "Unnamed model";
      chart();
      table();
      results.hidden = false;
      status.textContent = `${profile.events.length.toLocaleString()} estimated node events. No model execution was performed.`;
    }
    busy(false);
    return true;
  }

  return {
    open,
    handleMessage,
    setGraph(graph) {
      hasModel = true;
      configured = false;
      profile = undefined;
      path = graph.pipeline
        ? graph.breadcrumbs?.[1]?.path?.slice(0, 1)
        : undefined;
      fields.replaceChildren();
      results.hidden = true;
      requestId++;
      busy(false);
      status.textContent = "Open Memory to configure a fresh estimate.";
      if (!view.hidden) open();
    },
  };
}
