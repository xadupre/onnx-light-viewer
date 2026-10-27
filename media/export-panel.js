function createExportPanel({ byId, postMessage, requestAction, revealPanel }) {
  let request = 0;
  let timer;
  let actionId;
  let running = false;
  let converted = false;
  let loading = false;
  let defaultDirectory = true;
  let exporterChosen = false;
  let clock;
  let phase;
  let phaseStarted;
  let times = {};
  let downloadSkipped = false;
  function renderTimes() {
    for (const [key, id] of [
      ["download", "export-download-time"],
      ["conversion", "export-conversion-time"],
    ]) {
      const elapsed =
        phase === key ? performance.now() - phaseStarted : times[key];
      byId(id).textContent =
        key === "download" && downloadSkipped
          ? "Not needed (local source)"
          : elapsed === undefined
            ? "Not started"
            : `${(elapsed / 1000).toFixed(2)} s${phase === key ? " (running)" : ""}`;
    }
  }
  function stopClock() {
    if (phase && times[phase] === undefined)
      times[phase] = performance.now() - phaseStarted;
    phase = undefined;
    clearInterval(clock);
    renderTimes();
  }
  const advancedFields = [
    ...byId("export-advanced").querySelectorAll(".export-option"),
  ];
  const advancedControls = [
    ...byId("export-advanced").querySelectorAll("[data-export-option]"),
  ];
  const options = () => {
    const advancedOptions = {};
    for (const control of advancedControls) {
      if (control.disabled) continue;
      if (control.type === "checkbox") {
        if (control.checked)
          advancedOptions[control.dataset.exportOption] = true;
      } else if (control.value !== "") {
        advancedOptions[control.dataset.exportOption] =
          control.type === "number" || control.hasAttribute("data-number")
            ? Number(control.value)
            : control.value;
      }
    }
    return {
      exporter: byId("export-tool").value,
      model: byId("export-source").value,
      directory: defaultDirectory ? "" : byId("export-directory").value,
      precision: byId("export-precision").value,
      provider: byId("export-provider").value,
      ...(!byId("export-layer-count").disabled &&
      byId("export-layer-count").value !== ""
        ? { layerCount: Number(byId("export-layer-count").value) }
        : {}),
      ...(Object.keys(advancedOptions).length ? { advancedOptions } : {}),
    };
  };
  function updateAdvancedOptions() {
    const exporter = byId("export-tool").value;
    const precision = byId("export-precision").value;
    for (const field of advancedFields) {
      const { exporters, precisions, requires, excludes } = field.dataset;
      field.hidden =
        !exporters.split(" ").includes(exporter) ||
        (precisions !== undefined &&
          !precisions.split(" ").includes(precision)) ||
        (requires !== undefined && !byId(requires).checked) ||
        (excludes !== undefined && byId(excludes).checked);
      for (const control of field.querySelectorAll("input, select"))
        control.disabled = field.hidden;
    }
  }
  function historySuggestions(history) {
    for (const [id, entries] of [
      ["export-source-history", history.models],
      ["export-directory-history", history.directories],
    ]) {
      byId(id).replaceChildren(
        ...entries.map((value) => {
          const option = byId(id).ownerDocument.createElement("option");
          option.value = value;
          return option;
        }),
      );
    }
  }
  const directoryChanged = () => {
    defaultDirectory = !byId("export-directory").value;
  };
  byId("export-directory").addEventListener("input", directoryChanged);
  byId("export-directory").addEventListener("change", directoryChanged);
  byId("export-tool").addEventListener("change", () => {
    exporterChosen = true;
    updateAdvancedOptions();
  });
  const enabled = () => {
    byId("export-submit").disabled = running || loading;
    byId("export-fields").disabled = running;
    byId("cancel-export").disabled = !running;
    byId("export-model").disabled = loading && !running;
  };
  const followOutput = () => {
    const output = byId("export-output");
    output.scrollTop = output.scrollHeight;
  };
  function preview() {
    clearTimeout(timer);
    if (running) return;
    const requestId = ++request;
    timer = setTimeout(
      () =>
        postMessage({
          type: "export-preview",
          requestId,
          exportOptions: options(),
        }),
      150,
    );
  }
  const fieldsChanged = () => {
    updateAdvancedOptions();
    preview();
  };
  byId("export-fields").addEventListener("input", fieldsChanged);
  byId("export-fields").addEventListener("change", fieldsChanged);
  byId("export-model").addEventListener("click", () => {
    const panel = byId("export-panel");
    panel.hidden = !panel.hidden;
    byId("export-model").setAttribute("aria-expanded", String(!panel.hidden));
    if (!panel.hidden) {
      revealPanel(panel);
      followOutput();
      postMessage({ type: "get-settings" });
      preview();
    }
  });
  byId("cancel-export").addEventListener("click", () => {
    byId("cancel-export").disabled = true;
    byId("export-status").textContent = "Stopping exporter…";
    postMessage({ type: "cancel-export" });
  });
  byId("export-form").addEventListener("submit", (event) => {
    event.preventDefault();
    if (running || loading) return;
    byId("export-error").textContent = "";
    requestAction("export", "Convert ONNX model", { exportOptions: options() });
  });
  updateAdvancedOptions();
  return {
    start(id) {
      stopClock();
      actionId = id;
      running = true;
      converted = false;
      exporterChosen = true;
      clearTimeout(timer);
      request++;
      times = {};
      downloadSkipped = false;
      byId("export-timings").hidden = false;
      byId("export-cache").textContent = "";
      renderTimes();
      clock = setInterval(renderTimes, 100);
      byId("export-stdout").textContent = "";
      byId("export-progress").hidden = false;
      byId("export-status").textContent = "Starting conversion…";
      revealPanel(byId("export-panel"));
      enabled();
    },
    finish(status) {
      stopClock();
      running = false;
      byId("export-progress").hidden = true;
      if (status === "CANCELLED" && !converted)
        byId("export-status").textContent =
          "Conversion cancelled; any generated files are kept.";
      if (status === "FAILED")
        byId("export-status").textContent = "Conversion failed.";
      enabled();
    },
    setLoading(pending) {
      loading = pending;
      enabled();
    },
    handleMessage(data) {
      if (data?.type === "settings-info") {
        byId("export-trust").hidden = data.trusted;
        if (data.exportHistory) {
          historySuggestions(data.exportHistory);
          if (!exporterChosen && !running && data.exportHistory.exporter) {
            byId("export-tool").value = data.exportHistory.exporter;
            updateAdvancedOptions();
          }
        }
        if (defaultDirectory && data.exportDirectory && !running) {
          byId("export-directory").value = data.exportDirectory;
          preview();
        }
        return false;
      }
      if (data?.type === "export-history") {
        historySuggestions(data.history);
        return true;
      }
      if (data?.type === "export-preview") {
        if (data.requestId !== request) return true;
        byId("export-command").textContent = data.command ?? "";
        byId("export-script").textContent = data.script
          ? `Reproduction script: ${data.script}`
          : "";
        byId("export-error").textContent = data.error ?? "";
        if (defaultDirectory && data.directoryTemplate)
          byId("export-directory").value = data.directoryTemplate;
        return true;
      }
      if (
        !["export-progress", "export-info", "export-error"].includes(data?.type)
      )
        return false;
      if (data.actionId !== actionId) return true;
      if (data.phase !== undefined && data.phase !== phase) {
        if (phase) times[phase] = performance.now() - phaseStarted;
        phase = ["download", "conversion"].includes(data.phase)
          ? data.phase
          : undefined;
        phaseStarted = performance.now();
      }
      if (data.downloadSkipped !== undefined)
        downloadSkipped = data.downloadSkipped;
      if (data.downloadDurationMs !== undefined)
        times.download = data.downloadDurationMs;
      if (data.durationMs !== undefined) times.conversion = data.durationMs;
      if (data.cacheDirectory !== undefined)
        byId("export-cache").textContent =
          `Source cache: ${data.cacheDirectory}`;
      renderTimes();
      if (defaultDirectory && data.directoryTemplate)
        byId("export-directory").value = data.directoryTemplate;
      if (data.command !== undefined)
        byId("export-command").textContent = data.command;
      if (data.script !== undefined)
        byId("export-script").textContent =
          `Reproduction script: ${data.script}`;
      if (data.type === "export-progress") {
        byId("export-status").textContent = data.stage;
        if (data.output !== undefined) {
          byId("export-stdout").textContent = data.output
            ? `\n\n${data.output}`
            : "";
          followOutput();
        }
      } else if (data.type === "export-error") {
        byId("export-error").textContent = data.error;
        byId("export-status").textContent =
          "Conversion failed. See the output and Logs.";
      } else {
        stopClock();
        converted = true;
        byId("export-status").textContent = data.loaded
          ? `Converted and opened ${data.path}`
          : "Conversion saved; opening cancelled.";
        byId("export-status").textContent +=
          ` Conversion time: ${(data.durationMs / 1000).toFixed(2)} s.`;
        byId("export-panel").hidden = false;
        byId("export-model").setAttribute("aria-expanded", "true");
        revealPanel(byId("export-panel"));
      }
      return true;
    },
  };
}
