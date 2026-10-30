function createSettingsPanel({
  byId,
  postMessage,
  onViewerSettings,
  onPythonPackages,
}) {
  const panelNames = ["operator", "find", "metadata"];
  let inspecting;
  let nextRequestId = 0;
  let trusted = false;
  let pythonPath;
  let packages;
  let appVersion;
  let nativeVersion;
  let checkingReleases;
  let releases = new Map();
  let sidebarWidth;
  let pathHistory = { open: [], save: [] };
  let exportHistory = { models: [], directories: [] };

  function sanitizeReleaseUrl(url) {
    if (typeof url !== "string" || !url) return undefined;
    try {
      const parsed = new URL(url, window.location.href);
      if (parsed.protocol === "https:" || parsed.protocol === "http:")
        return parsed.href;
    } catch {
      return undefined;
    }
    return undefined;
  }

  function renderRelease(cell, name, installed) {
    cell.replaceChildren();
    const release = releases.get(name);
    if (!release) {
      cell.textContent = "Not checked";
      return;
    }
    if (release.error !== undefined) {
      cell.textContent = `Unavailable: ${release.error}`;
    } else {
      const safeUrl = sanitizeReleaseUrl(release.url);
      if (safeUrl) {
        const link = document.createElement("a");
        link.href = safeUrl;
        link.textContent = release.version;
        cell.append(link);
      } else {
        const text = document.createElement("span");
        text.textContent = release.version;
        cell.append(text);
      }
      if (installed) {
        const comparison = document.createElement("span");
        comparison.className = "package-release-comparison";
        comparison.textContent =
          installed.replace(/^v(?=\d)/, "") === release.version
            ? "Matches latest release"
            : "Different from latest release";
        cell.append(comparison);
      }
    }
  }

  function renderVersions() {
    for (const [id, latestId, name, installed] of [
      [
        "settings-app-version",
        "status-app-latest",
        "onnx-light-viewer",
        appVersion,
      ],
      [
        "settings-native-version",
        "status-native-latest",
        "onnx-light C++",
        nativeVersion,
      ],
    ]) {
      const version = byId(id);
      version.textContent = installed ?? "Unavailable";
      renderRelease(byId(latestId), name, installed);
    }
    byId("settings-packages").replaceChildren();
    const entries =
      packages ??
      [...releases.values()]
        .filter(
          ({ name }) =>
            name !== "onnx-light-viewer" && name !== "onnx-light C++",
        )
        .map(({ name }) => ({ name }));
    for (const pkg of entries
      .filter((pkg) => pkg.name !== "modelbuilder" || pkg.version || pkg.source)
      .sort((a, b) => a.name.localeCompare(b.name))) {
      const row = document.createElement("tr");
      const name = document.createElement("th");
      name.scope = "row";
      name.textContent = pkg.name;
      const version = document.createElement("td");
      version.textContent = pkg.source
        ? `${pkg.version ?? "Version unavailable"} (PYTHONPATH)`
        : pkg.version === undefined
          ? "Installed version not checked"
          : (pkg.version ?? "Not installed");
      if (pkg.source) version.title = pkg.source;
      const latest = document.createElement("td");
      renderRelease(latest, pkg.name, pkg.version);
      row.append(name, version, latest);
      byId("settings-packages").append(row);
    }
  }

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
  byId("status-check-releases").addEventListener("click", () => {
    if (checkingReleases) return;
    if (!packages && trusted && !inspecting)
      byId("settings-inspect-python").click();
    checkingReleases = { requestId: ++nextRequestId };
    releases = new Map();
    renderVersions();
    byId("status-check-releases").disabled = true;
    byId("status-release-status").textContent =
      "Checking GitHub and PyPI releases…";
    postMessage({ type: "check-latest-versions", ...checkingReleases });
  });

  function receive(message) {
    if (!message || typeof message !== "object") return;
    if (
      message.type === "latest-versions-info" ||
      message.type === "latest-versions-error"
    ) {
      if (!checkingReleases || message.requestId !== checkingReleases.requestId)
        return;
      checkingReleases = undefined;
      byId("status-check-releases").disabled = false;
      if (message.type === "latest-versions-error") {
        byId("status-release-status").textContent = message.error;
        return;
      }
      releases = new Map(
        message.releases.map((release) => [release.name, release]),
      );
      const failures = message.releases.filter(
        (release) => release.error !== undefined,
      ).length;
      byId("status-release-status").textContent = failures
        ? `Release check completed; ${failures} lookup${failures === 1 ? "" : "s"} failed. See Logs for details.`
        : "Latest published versions checked.";
      renderVersions();
    } else if (
      message.type === "settings-info" ||
      message.type === "status-info"
    ) {
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
        packages = undefined;
        onPythonPackages(packages);
        byId("settings-python-status").textContent = "Not checked";
      }
      pythonPath = message.application.pythonPath;
      trusted = message.trusted;
      appVersion = message.appVersion;
      nativeVersion = message.nativeVersion;
      renderVersions();
      byId("settings-platform").textContent = message.platform;
      byId("status-check-releases").disabled = Boolean(checkingReleases);
      byId("settings-inspect-python").disabled =
        Boolean(inspecting) || !trusted;
      byId("settings-feedback").textContent = trusted
        ? "Settings loaded."
        : "Trust this workspace to inspect Python packages.";
      restoreViewer(message.viewer);
      if (trusted && !packages && !inspecting)
        byId("settings-inspect-python").click();
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
      if (message.pythonPath !== pythonPath) {
        if (trusted) byId("settings-inspect-python").click();
        return;
      }
      if (message.type === "settings-error") {
        byId("settings-feedback").textContent = message.error;
        byId("settings-python-status").textContent = message.error;
        packages = undefined;
        onPythonPackages(packages);
        renderVersions();
        return;
      }
      byId("settings-python-status").textContent =
        `Python ${message.python.version} — ${message.python.executable}`;
      packages = message.python.packages;
      onPythonPackages(packages);
      renderVersions();
    } else if (message.type === "settings-error") {
      byId("settings-feedback").textContent = message.error;
      byId("settings-history-feedback").textContent = message.error;
      byId("settings-reset-viewer").disabled = false;
    }
  }

  return { receive, restoreViewer };
}
