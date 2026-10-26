function createExportPanel({ byId, postMessage, requestAction, revealPanel }) {
  let request = 0;
  let timer;
  let actionId;
  let running = false;
  let loading = false;
  let defaultDirectory = true;
  let exporterChosen = false;
  const options = () => ({
    exporter: byId("export-tool").value,
    model: byId("export-source").value,
    directory: defaultDirectory ? "" : byId("export-directory").value,
    precision: byId("export-precision").value,
    provider: byId("export-provider").value,
  });
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
  byId("export-fields").addEventListener("input", preview);
  byId("export-fields").addEventListener("change", preview);
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
  return {
    start(id) {
      actionId = id;
      running = true;
      exporterChosen = true;
      clearTimeout(timer);
      request++;
      byId("export-stdout").textContent = "";
      byId("export-progress").hidden = false;
      byId("export-status").textContent = "Starting conversion…";
      enabled();
    },
    finish(status) {
      running = false;
      byId("export-progress").hidden = true;
      if (status === "CANCELLED")
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
        byId("export-python").textContent =
          `Python: ${data.application.pythonPath}. Change it in Settings.`;
        byId("export-trust").hidden = data.trusted;
        if (data.exportHistory) {
          historySuggestions(data.exportHistory);
          if (!exporterChosen && !running && data.exportHistory.exporter) {
            byId("export-tool").value = data.exportHistory.exporter;
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
        byId("export-status").textContent = data.loaded
          ? `Converted and opened ${data.path}`
          : "Conversion saved; opening cancelled.";
      }
      return true;
    },
  };
}
