function createSettingsPanel({ byId, postMessage, onViewerSettings }) {
  const panelNames = ["operator", "find", "metadata"];
  let inspecting;
  let nextRequestId = 0;
  let trusted = false;
  let pythonPath;

  function restoreViewer(settings) {
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
      if (!byId("settings-width").checkValidity()) return;
      const width = byId("settings-width").value;
      onViewerSettings({
        ...(width === "" ? {} : { width: Number(width) }),
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
    if (message.type === "settings-info") {
      byId("settings-download").value = message.application.downloadDirectory;
      byId("settings-python").value = message.application.pythonPath;
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
      for (const pkg of message.python.packages) {
        const name = document.createElement("dt");
        name.textContent = pkg.name;
        const version = document.createElement("dd");
        version.textContent = pkg.version ?? "Not installed";
        byId("settings-packages").append(name, version);
      }
    } else if (message.type === "settings-error") {
      byId("settings-feedback").textContent = message.error;
    }
  }

  return { receive, restoreViewer };
}
