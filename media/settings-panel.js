function createSettingsPanel({ byId, postMessage, onViewerSettings }) {
  const panelNames = ["operator", "find", "metadata"];
  let inspecting;
  let nextRequestId = 0;
  let trusted = false;
  let pythonPath;
  let sidebarWidth;
  let pathHistory = { open: [], save: [] };
  let exportHistory = { models: [], directories: [] };

  function editStoredData(message) {
    byId("settings-history-feedback").textContent = "Updating stored data…";
    postMessage(message);
  }

  function renderHistory(target, entries) {
    byId(`settings-${target}-count`).textContent = String(entries.length);
    byId(`settings-clear-${target}`).disabled =
      !entries.length && !(target === "export" && exportHistory.exporter);
    byId(`settings-${target}-history`).replaceChildren(
      ...entries.map(({ label, message }) => {
        const row = document.createElement("li");
        const text = document.createElement("span");
        text.textContent = label;
        const remove = document.createElement("button");
        remove.type = "button";
        remove.textContent = "Remove";
        remove.setAttribute("aria-label", `Remove ${label} from history`);
        remove.addEventListener("click", () => editStoredData(message));
        row.append(text, remove);
        return row;
      }),
    );
  }

  function renderHistories() {
    for (const target of ["open", "save"])
      renderHistory(
        target,
        pathHistory[target].map((entry) => ({
          label: `${entry.source} / ${entry.kind}: ${entry.path}`,
          message: {
            type: "path-history-edit",
            target,
            historyAction: "remove",
            historyEntry: entry,
          },
        })),
      );
    renderHistory(
      "export",
      ["models", "directories"].flatMap((target) =>
        exportHistory[target].map((value) => ({
          label: `${target === "models" ? "Model" : "Directory"}: ${value}`,
          message: {
            type: "export-history-edit",
            target,
            historyAction: "remove",
            historyEntry: value,
          },
        })),
      ),
    );
    byId("settings-exporter").textContent = exportHistory.exporter ?? "None";
  }

  for (const target of ["open", "save", "export"])
    byId(`settings-clear-${target}`).addEventListener("click", () =>
      editStoredData({
        type: target === "export" ? "export-history-edit" : "path-history-edit",
        ...(target === "export" ? {} : { target }),
        historyAction: "clear",
      }),
    );
  byId("settings-reset-viewer").addEventListener("click", () => {
    byId("settings-reset-viewer").disabled = true;
    byId("settings-feedback").textContent = "Resetting viewer preferences…";
    postMessage({ type: "reset-viewer-settings" });
  });

  function restoreViewer(settings) {
    sidebarWidth = settings.width;
    byId("settings-width").value = settings.width ?? "";
    for (const name of panelNames)
      byId(`settings-${name}`).checked = settings.panels[name];
  }

  byId("application-settings-form").addEventListener("submit", (event) => {
    event.preventDefault();
    byId("settings-feedback").textContent = "Saving…";
    postMessage({
      type: "application-settings",
      settings: {
        downloadDirectory: byId("settings-download").value,
        pythonPath: byId("settings-python").value,
      },
    });
  });
  for (const id of ["width", ...panelNames])
    byId(`settings-${id}`).addEventListener("change", () => {
      if (id === "width") {
        if (!byId("settings-width").checkValidity()) return;
        const width = byId("settings-width").value;
        sidebarWidth = width === "" ? undefined : Number(width);
      }
      onViewerSettings({
        ...(sidebarWidth === undefined ? {} : { width: sidebarWidth }),
        panels: Object.fromEntries(
          panelNames.map((name) => [name, byId(`settings-${name}`).checked]),
        ),
      });
    });
  byId("settings-inspect-python").addEventListener("click", () => {
    if (inspecting) return;
    inspecting = { requestId: ++nextRequestId, pythonPath };
    byId("settings-inspect-python").disabled = true;
    byId("settings-python-status").textContent = "Checking Python packages…";
    postMessage({ type: "inspect-python", ...inspecting });
  });

  function receive(message) {
    if (!message || typeof message !== "object") return;
    if (message.type === "settings-info") {
      byId("settings-download").value = message.application.downloadDirectory;
      byId("settings-python").value = message.application.pythonPath;
      byId("settings-cache-download").textContent =
        message.application.downloadDirectory;
      byId("settings-cache-saves").textContent =
        message.saveDirectory ?? "Unavailable";
      if (message.pathHistory) pathHistory = message.pathHistory;
      if (message.exportHistory) exportHistory = message.exportHistory;
      renderHistories();
      byId("settings-reset-viewer").disabled = false;
      byId("settings-history-feedback").textContent = "";
      if (pythonPath !== message.application.pythonPath) {
        byId("settings-packages").replaceChildren();
        byId("settings-python-status").textContent = "Not checked";
      }
      pythonPath = message.application.pythonPath;
      trusted = message.trusted;
      byId("settings-app-version").textContent = message.appVersion;
      byId("settings-native-version").textContent = message.nativeVersion;
      byId("settings-platform").textContent = message.platform;
      byId("settings-inspect-python").disabled =
        Boolean(inspecting) || !trusted;
      byId("settings-feedback").textContent = trusted
        ? "Settings loaded."
        : "Trust this workspace to inspect Python packages.";
      restoreViewer(message.viewer);
    } else if (
      message.type === "path-history" ||
      message.type === "export-history"
    ) {
      if (message.type === "path-history") pathHistory = message.history;
      else exportHistory = message.history;
      renderHistories();
      byId("settings-history-feedback").textContent =
        "Stored history updated. No files were deleted.";
    } else if (
      message.type === "python-info" ||
      (message.type === "settings-error" && message.requestId !== undefined)
    ) {
      if (
        !inspecting ||
        message.requestId !== inspecting.requestId ||
        message.pythonPath !== inspecting.pythonPath
      )
        return;
      inspecting = undefined;
      byId("settings-inspect-python").disabled = !trusted;
      if (message.pythonPath !== pythonPath) return;
      if (message.type === "settings-error") {
        byId("settings-feedback").textContent = message.error;
        byId("settings-python-status").textContent = "Not checked";
        byId("settings-packages").replaceChildren();
        return;
      }
      byId("settings-python-status").textContent =
        `Python ${message.python.version} — ${message.python.executable}`;
      byId("settings-packages").replaceChildren();
      for (const pkg of [...message.python.packages].sort((a, b) =>
        a.name.localeCompare(b.name),
      )) {
        const name = document.createElement("dt");
        name.textContent = pkg.name;
        const version = document.createElement("dd");
        version.textContent = pkg.source
          ? `${pkg.version ?? "Version unavailable"} (PYTHONPATH)`
          : (pkg.version ?? "Not installed");
        if (pkg.source) version.title = pkg.source;
        byId("settings-packages").append(name, version);
      }
    } else if (message.type === "settings-error") {
      byId("settings-feedback").textContent = message.error;
      byId("settings-history-feedback").textContent = message.error;
      byId("settings-reset-viewer").disabled = false;
    }
  }

  return { receive, restoreViewer };
}
