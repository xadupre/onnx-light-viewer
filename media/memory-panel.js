function createMemoryPanel({ byId, postMessage, formatFileSize, selectNode }) {
  let hasModel = false;
  let requestId = 0;
  let path;
  let configured = false;
  let profile;
  let peaks = [];
  let peakIndices = new Set();
  let page = 0;
  let zoom = 1;
  let inspectedIndex = 0;
  const pageSize = 100;
  const view = byId("memory-view");
  const fields = byId("memory-dimensions");
  const status = byId("memory-status");
  const results = byId("memory-results");
  const svg = byId("memory-chart");
  const scroll = byId("memory-chart-scroll");
  const inspection = byId("memory-inspection");

  function busy(pending) {
    view.setAttribute("aria-busy", String(pending));
    byId("memory-run").disabled = pending || !configured;
    byId("memory-stage").disabled = pending;
    for (const input of fields.querySelectorAll("input"))
      input.disabled = pending;
  }

  function open() {
    results.hidden = true;
    inspection.hidden = true;
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
    const peaksOnly = byId("memory-peaks-only").checked;
    const events = peaksOnly ? peaks : profile.events;
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
        `${formatFileSize(event.deviceBytes)}${peakIndices.has(event.index) ? " (local peak)" : ""}`,
        event.afterBytes == null
          ? "Unavailable"
          : formatFileSize(event.afterBytes),
        (event.decisions ?? [])
          .filter(({ action }) => action === "inplace")
          .map(({ input, output, kind }) => `${input} → ${output} (${kind})`)
          .join("\n") || "—",
        (event.decisions ?? [])
          .filter(({ action }) => action === "release")
          .map(({ name }) => name)
          .join("\n") || "—",
        (event.decisions ?? [])
          .filter(({ action }) => action === "release_shape_tag")
          .map(({ name }) => name)
          .join("\n") || "—",
        [
          ...(event.inputs ?? []).map((tensor) => ({
            ...tensor,
            direction: "in",
          })),
          ...(event.outputs ?? []).map((tensor) => ({
            ...tensor,
            direction: "out",
          })),
        ]
          .map(
            ({ name, type, shape, tag, direction }) =>
              `${direction}: ${name}: ${type ?? "Unknown"} ${shape ?? "Unknown shape"}${tag ? ` (${tag})` : ""}`,
          )
          .join("\n") || "—",
      ].entries()) {
        const cell = document.createElement(index === 0 ? "th" : "td");
        if (index === 0) cell.scope = "row";
        if (index === 1) {
          const button = document.createElement("button");
          button.type = "button";
          button.textContent = value;
          button.title = "Show this node in Model";
          button.addEventListener("click", () => selectNode(path, event.index));
          cell.append(button);
        } else cell.textContent = value;
        row.append(cell);
      }
      body.append(row);
    }
    byId("memory-range").textContent = events.length
      ? `Events ${start + 1}–${Math.min(start + pageSize, events.length)} of ${events.length.toLocaleString()}`
      : peaksOnly
        ? "No local peaks."
        : "No executable nodes; no per-node peak is available.";
    byId("memory-previous").disabled = page === 0;
    byId("memory-next").disabled = start + pageSize >= events.length;
  }

  function chart() {
    const events = profile.events;
    const peak = profile.peakBytes ?? 0;
    const width = Math.max(1000, scroll.clientWidth) * zoom;
    const plotWidth = width - 80;
    svg.setAttribute("viewBox", `0 0 ${width} 330`);
    svg.style.width = `${width}px`;
    svg
      .querySelector(".memory-axis")
      .setAttribute("d", `M60,20V240H${width - 20}`);
    byId("memory-zoom-level").textContent = `${Math.round(zoom * 100)}%`;
    byId("memory-zoom-out").disabled = zoom === 1;
    const x = (step) =>
      (60 + (plotWidth * step) / (events.length * 3 + 1)).toFixed(5);
    const y = (bytes) => (240 - (210 * bytes) / Math.max(1, peak)).toFixed(5);
    const points = events.length
      ? [
          `M${x(0)},${y(events[0].memory?.before_bytes ?? events[0].existingBytes)}`,
        ]
      : [];
    const interval = (index) =>
      `M${x(index * 3 + 1)},242 H${x(index * 3 + 3)} V252 H${x(index * 3 + 1)}Z`;
    const intervals = [];
    for (const [index, event] of events.entries()) {
      const allocate = index * 3 + 1;
      const release = allocate + 2;
      points.push(`H${x(allocate)} V${y(event.deviceBytes)} H${x(release)}`);
      if (event.afterBytes != null)
        points.push(`V${y(event.afterBytes)} H${x(release + 1)}`);
      intervals.push(interval(index));
    }
    byId("memory-curve").setAttribute("d", points.join(" "));
    byId("memory-node-intervals").setAttribute("d", intervals.join(" "));
    byId("memory-chart").toggleAttribute("hidden", events.length === 0);
    byId("memory-chart-peak").textContent = formatFileSize(peak);
    const append = (parent, tag, attributes, text) => {
      const element = document.createElementNS(
        "http://www.w3.org/2000/svg",
        tag,
      );
      for (const [name, value] of Object.entries(attributes))
        element.setAttribute(name, value);
      if (text !== undefined) element.textContent = text;
      parent.append(element);
      return element;
    };
    const numbers = byId("memory-node-numbers");
    const labels = byId("memory-operator-labels");
    numbers.replaceChildren();
    labels.replaceChildren();
    const center = (index) => Number(x(index * 3 + 2));
    const indices = [];
    if (events.length) {
      indices.push(0);
      const target = Math.max(1, (events.length - 1) / (plotWidth / 100));
      const power = 10 ** Math.floor(Math.log10(target));
      const step =
        [1, 2, 5, 10].find((multiple) => multiple * power >= target) * power;
      for (let index = step - 1; index < events.length - 1; index += step)
        if (center(index) - center(indices.at(-1)) >= 72) indices.push(index);
      const last = events.length - 1;
      if (last) {
        if (indices.length > 1 && center(last) - center(indices.at(-1)) < 72)
          indices.pop();
        indices.push(last);
      }
    }
    for (const index of indices) {
      const position = center(index);
      const group = append(numbers, "g", {
        class: "memory-node-number",
        "data-node-index": events[index].index,
      });
      append(
        group,
        "text",
        { x: position, y: 272, "text-anchor": "middle" },
        String(events[index].index + 1),
      );
    }
    const markers = [];
    const rows = [[], []];
    const priority = (type) =>
      /Attention/i.test(type)
        ? 0
        : /MatMul/i.test(type)
          ? 1
          : /Normalization/i.test(type)
            ? 2
            : 3;
    const candidates = events.map((event, index) => ({ event, index }));
    candidates.sort(
      (a, b) =>
        priority(a.event.opType) - priority(b.event.opType) ||
        a.index - b.index,
    );
    for (const [index, event] of events.entries()) {
      if (
        priority(event.opType) < 3 ||
        /^(?:Gemm|Conv|ConvInteger|QLinearConv|ConvTranspose|Einsum|LSTM|GRU|RNN)$/i.test(
          event.opType,
        )
      )
        markers.push(interval(index));
    }
    const dense = plotWidth / events.length < 72;
    const stride = plotWidth / events.length < 14 ? 10 : 5;
    const placed = [];
    for (const { event, index } of candidates) {
      if (
        dense &&
        priority(event.opType) === 3 &&
        index !== 0 &&
        (index + 1) % stride !== 0
      )
        continue;
      const position = center(index);
      const text = `#${event.index + 1} ${event.opType}`;
      const width = text.length * 8 + 12;
      if (width > plotWidth) continue;
      const left = Math.max(
        60,
        Math.min(position - width / 2, plotWidth + 60 - width),
      );
      const row = rows.findIndex((ranges) =>
        ranges.every(
          ([start, end]) => left >= end + 12 || left + width + 12 <= start,
        ),
      );
      if (row < 0) continue;
      rows[row].push([left, left + width]);
      placed.push({ event, index, left, width, row, text });
    }
    placed.sort((a, b) => a.index - b.index);
    for (const { event, left, width, row, text } of placed) {
      const group = append(labels, "g", { "data-node-index": event.index });
      append(
        group,
        "text",
        { x: left + width / 2, y: 288 + row * 24, "text-anchor": "middle" },
        text,
      );
      append(
        group,
        "title",
        {},
        `${text}${event.node ? `: ${event.node}` : ""}`,
      );
    }
    byId("memory-operator-intervals").setAttribute("d", markers.join(" "));
    byId("memory-final-status").hidden =
      !events.length || events.at(-1).afterBytes != null;
  }

  function inspect(index, phase = "during") {
    const event = profile?.events[index];
    if (!event) return;
    inspectedIndex = index;
    const bytes =
      phase === "before"
        ? (event.memory?.before_bytes ?? event.existingBytes)
        : phase === "after"
          ? event.afterBytes
          : event.deviceBytes;
    const amount = (value) =>
      value == null
        ? "Unavailable"
        : `${formatFileSize(value)} (${value} bytes)`;
    const lines = [
      `#${event.index + 1} ${event.opType}${event.node ? `: ${event.node}` : ""} — ${phase} execution`,
      `Memory: ${amount(bytes)}`,
      `Before allocation: ${amount(event.existingBytes)}`,
      `New outputs: ${amount(event.outputBytes)}`,
      `Scratch: ${amount(event.scratchBytes)}`,
      `After release: ${amount(event.afterBytes)}`,
      "Allocation breakdown during execution (SDK categories):",
    ];
    for (const [key, label] of [
      ["initializers", "Weights (initializers)"],
      ["inputs", "Inputs"],
      ["intermediates", "Intermediates"],
      ["outputs", "New output tensors"],
    ]) {
      const numeric = event.memory?.[key];
      const symbolic = event.symbolicMemory?.[key];
      const tags = new Set([
        ...Object.keys(numeric ?? {}),
        ...Object.keys(symbolic ?? {}),
      ]);
      lines.push(`${label}:`);
      if (!tags.size) lines.push("  —");
      for (const tag of tags)
        lines.push(
          `  ${tag || "tensor"}: ${amount(numeric?.[tag])}; symbolic: ${symbolic?.[tag] ?? "Unavailable"} bytes`,
        );
    }
    const key =
      phase === "before"
        ? "before_bytes"
        : phase === "after"
          ? "after_bytes"
          : "during_bytes";
    lines.push(
      `Symbolic tensor memory (${phase}, scratch excluded): ${event.symbolicMemory?.[key] ?? "Unavailable"} bytes`,
    );
    for (const direction of ["inputs", "outputs"])
      for (const tensor of event[direction] ?? [])
        lines.push(
          `${direction === "inputs" ? "in" : "out"}: ${tensor.name}: ${tensor.type ?? "Unknown"} ${tensor.shape ?? "Unknown shape"}${tensor.tag ? ` (${tensor.tag})` : ""}`,
        );
    inspection.textContent = lines.join("\n");
    inspection.hidden = false;
  }

  function changeZoom(next, anchor = 0.5) {
    if (!profile?.events.length) return;
    const previous = zoom;
    zoom = Math.max(1, Math.min(Math.max(32, profile.events.length / 4), next));
    const position = scroll.scrollLeft + scroll.clientWidth * anchor;
    chart();
    scroll.scrollLeft =
      zoom === 1
        ? 0
        : position * (zoom / previous) - scroll.clientWidth * anchor;
    inspection.hidden = true;
  }
  byId("memory-zoom-in").addEventListener("click", () => changeZoom(zoom * 2));
  byId("memory-zoom-out").addEventListener("click", () => changeZoom(zoom / 2));
  byId("memory-zoom-reset").addEventListener("click", () => changeZoom(1));
  svg.addEventListener(
    "wheel",
    (event) => {
      if (!event.ctrlKey || !profile?.events.length) return;
      event.preventDefault();
      const rect = scroll.getBoundingClientRect();
      changeZoom(
        zoom * (event.deltaY < 0 ? 2 : 0.5),
        rect.width ? (event.clientX - rect.left) / rect.width : 0.5,
      );
    },
    { passive: false },
  );
  svg.addEventListener("pointermove", (event) => {
    if (!profile?.events.length) return;
    const rect = svg.getBoundingClientRect();
    if (!rect.width) return;
    const width = Number(svg.getAttribute("viewBox").split(" ")[2]);
    const matrix = svg.getScreenCTM?.()?.inverse();
    const position = matrix
      ? matrix.a * event.clientX + matrix.c * event.clientY + matrix.e
      : ((event.clientX - rect.left) * width) / rect.width;
    if (position < 60 || position > width - 20) {
      inspection.hidden = true;
      return;
    }
    const step =
      ((position - 60) * (profile.events.length * 3 + 1)) / (width - 80);
    const index = Math.min(
      profile.events.length - 1,
      Math.max(0, Math.floor((step - 1) / 3)),
    );
    inspect(
      index,
      step < 1 ? "before" : step >= index * 3 + 3 ? "after" : "during",
    );
  });
  svg.addEventListener("pointerleave", () => {
    inspection.hidden = true;
  });
  window.addEventListener("resize", () => {
    if (profile && !results.hidden) chart();
  });
  svg.addEventListener("keydown", (event) => {
    if (
      !["ArrowLeft", "ArrowRight"].includes(event.key) ||
      !profile?.events.length
    )
      return;
    event.preventDefault();
    inspect(
      Math.max(
        0,
        Math.min(
          profile.events.length - 1,
          inspection.hidden
            ? inspectedIndex
            : inspectedIndex + (event.key === "ArrowRight" ? 1 : -1),
        ),
      ),
    );
  });

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
    inspection.hidden = true;
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
  byId("memory-peaks-only").addEventListener("change", () => {
    page = 0;
    table();
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
      zoom = 1;
      inspectedIndex = 0;
      scroll.scrollLeft = 0;
      inspection.hidden = true;
      peaks = [];
      for (let start = 0; start < profile.events.length;) {
        let end = start;
        const bytes = profile.events[start].deviceBytes;
        while (
          end + 1 < profile.events.length &&
          profile.events[end + 1].deviceBytes === bytes
        )
          end++;
        if (
          (start === 0 || bytes > profile.events[start - 1].deviceBytes) &&
          (end === profile.events.length - 1 ||
            bytes > profile.events[end + 1].deviceBytes)
        )
          peaks.push(profile.events[start]);
        start = end + 1;
      }
      peakIndices = new Set(peaks.map((event) => event.index));
      byId("memory-peaks-count").textContent =
        `${peaks.length.toLocaleString()} local peak${peaks.length === 1 ? "" : "s"}. Select a node to show it in Model.`;
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
