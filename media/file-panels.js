function createFilePanels({
  byId,
  postMessage,
  requestAction,
  revealPanel,
  isReplacing,
}) {
  const fileRequests = { open: 0, save: 0 };
  const fileParents = {};
  function browseFiles(target) {
    byId(target === "open" ? "open-error" : "save-options-error").textContent =
      "";
    byId(`${target}-files`).setAttribute("aria-busy", "true");
    postMessage({
      type: "browse-files",
      target,
      source: target === "open" ? byId("open-source").value : "local",
      location: byId(`${target}-location`).value,
      requestId: ++fileRequests[target],
    });
  }
  function toggleFilePanel(target, visible) {
    if (visible) toggleFilePanel(target === "open" ? "save" : "open", false);
    const panel = byId(target === "open" ? "open-panel" : "save-options");
    panel.hidden = !visible;
    byId(`${target}-model`).setAttribute("aria-expanded", String(visible));
    if (visible) {
      revealPanel(panel);
      if (target === "save" || byId("open-source").value !== "huggingface")
        browseFiles(target);
    } else {
      fileRequests[target]++;
      byId(`${target}-files`).setAttribute("aria-busy", "false");
      for (const input of panel.querySelectorAll('input[type="password"]'))
        input.value = "";
    }
  }
  for (const target of ["open", "save"]) {
    byId(`${target}-model`).addEventListener("click", () =>
      toggleFilePanel(
        target,
        byId(target === "open" ? "open-panel" : "save-options").hidden,
      ),
    );
    byId(`cancel-${target}`).addEventListener("click", () =>
      toggleFilePanel(target, false),
    );
    byId(`${target}-browse`).addEventListener("submit", (event) => {
      event.preventDefault();
      browseFiles(target);
    });
    byId(`${target}-parent`).addEventListener("click", () => {
      byId(`${target}-location`).value = fileParents[target];
      browseFiles(target);
    });
  }
  byId("open-source").addEventListener("change", () => {
    fileRequests.open++;
    byId("open-error").textContent = "";
    byId("open-location").value = "";
    byId("open-file").value = "";
    byId("open-filter").value = "";
    byId("open-password").value = "";
    byId("open-files").replaceChildren();
    byId("open-files").setAttribute("aria-busy", "false");
    byId("open-parent").disabled = true;
    byId("open-browse").hidden = byId("open-source").value === "backend";
    if (byId("open-source").value !== "huggingface") browseFiles("open");
  });
  function filterModelFiles() {
    const filter = byId("open-filter").value.toLocaleLowerCase();
    for (const item of byId("open-files").children)
      item.hidden = !item.dataset.search.includes(filter);
  }
  byId("open-filter").addEventListener("input", filterModelFiles);
  byId("open-form").addEventListener("submit", (event) => {
    event.preventDefault();
    if (isReplacing()) return;
    if (!byId("open-file").value.trim()) {
      byId("open-error").textContent = "Select a model or enter its path.";
      return;
    }
    byId("open-error").textContent = "";
    requestAction("open", "Open ONNX model", {
      source: byId("open-source").value,
      location: byId("open-file").value,
      password: byId("open-password").value,
    });
    byId("open-password").value = "";
  });
  byId("save-form").addEventListener("change", (event) => {
    const external = ["external-plain", "reuse-plain"].includes(
      document.querySelector('input[name="save-format"]:checked').value,
    );
    byId("external-size").hidden = !external;
    byId("weight-file-size").disabled = !external;
    const encrypted =
      document.querySelector('input[name="save-format"]:checked').value ===
      "embedded-encrypted";
    byId("save-encryption").hidden = !encrypted;
    byId("save-password").required = encrypted;
    byId("save-confirmation").required = encrypted;
    if (event.target.name === "save-format")
      byId("save-file").value = byId("save-file").value.replace(
        /\.onnx(?:\.enc)?$/i,
        `.onnx${encrypted ? ".enc" : ""}`,
      );
    byId("save-options-error").textContent = "";
  });
  byId("save-form").addEventListener("submit", (event) => {
    event.preventDefault();
    if (byId("save-submit").disabled) return;
    const format = document.querySelector(
      'input[name="save-format"]:checked',
    ).value;
    const external = format === "external-plain" || format === "reuse-plain";
    const maxExternalFileSize =
      Number(byId("weight-file-size").value) * 1024 * 1024;
    if (
      ![
        "embedded-plain",
        "embedded-encrypted",
        "external-plain",
        "reuse-plain",
      ].includes(format) ||
      (external &&
        (!Number.isSafeInteger(maxExternalFileSize) ||
          maxExternalFileSize < 1024 * 1024 ||
          !byId("weight-file-size").checkValidity()))
    ) {
      byId("save-options-error").textContent =
        "Choose a supported format and a positive whole-number file size in MiB.";
      return;
    }
    if (!byId("save-file").value.trim()) {
      byId("save-options-error").textContent = "Enter a destination path.";
      return;
    }
    if (
      format === "embedded-encrypted" &&
      (!byId("save-password").value ||
        byId("save-password").value !== byId("save-confirmation").value)
    ) {
      byId("save-options-error").textContent =
        "Enter and confirm the same non-empty password.";
      return;
    }
    byId("save-options-error").textContent = "";
    requestAction("save", "Save ONNX model", {
      location: byId("save-file").value,
      password: byId("save-password").value,
      confirmation: byId("save-confirmation").value,
      overwrite: byId("save-overwrite").checked,
      storage:
        format === "reuse-plain" ? "reuse" : external ? "external" : "embedded",
      encrypted: format === "embedded-encrypted",
      ...(external ? { maxExternalFileSize } : {}),
    });
    byId("save-password").value = "";
    byId("save-confirmation").value = "";
  });

  function handleMessage(data) {
    if (data?.type === "file-list" || data?.type === "file-list-error") {
      if (
        !["open", "save"].includes(data.target) ||
        data.requestId !== fileRequests[data.target]
      )
        return true;
      const target = data.target;
      const list = byId(`${target}-files`);
      list.setAttribute("aria-busy", "false");
      if (data.type === "file-list-error") {
        list.replaceChildren();
        byId(`${target}-parent`).disabled = true;
        byId(
          target === "open" ? "open-error" : "save-options-error",
        ).textContent = data.error;
        return true;
      }
      byId(`${target}-location`).value = data.location;
      fileParents[target] = data.parent;
      byId(`${target}-parent`).disabled =
        !data.parent || data.parent === data.location;
      if (target === "save" && data.location) {
        const name =
          byId("save-file").value.split(/[\\/]/).pop() || "model-copy.onnx";
        const separator = /^(?:[a-z]:\\|\\\\)/i.test(data.location)
          ? "\\"
          : "/";
        byId("save-file").value =
          `${data.location.replace(/[\\/]$/, "")}${separator}${name}`;
      }
      list.replaceChildren(
        ...data.entries.map((entry) => {
          const item = document.createElement("li");
          item.dataset.search =
            `${entry.name} ${entry.description ?? ""}`.toLocaleLowerCase();
          const button = document.createElement("button");
          button.type = "button";
          button.textContent = `${entry.directory ? "[Folder] " : ""}${entry.name}`;
          button.title = entry.path;
          button.addEventListener("click", () => {
            if (entry.directory) {
              byId(`${target}-location`).value = entry.path;
              browseFiles(target);
            } else {
              byId(`${target}-file`).value = entry.path;
              for (const row of list.children)
                row.removeAttribute("aria-current");
              item.setAttribute("aria-current", "true");
            }
          });
          item.append(button);
          return item;
        }),
      );
      if (target === "open") filterModelFiles();
    } else if (data?.type === "save-info") {
      byId("save-status").textContent = `Saved ${data.name}`;
    } else if (data?.type === "save-error") {
      byId("save-options-error").textContent = data.error;
      byId("save-status").textContent = `Could not save model: ${data.error}`;
    } else {
      return false;
    }
    return true;
  }
  function reset() {
    for (const target of ["open", "save"]) {
      fileRequests[target]++;
      byId(`${target}-files`).setAttribute("aria-busy", "false");
    }
    byId("save-status").textContent = "";
  }
  function setFile(file) {
    if (file) {
      if (byId("open-source").value === "local")
        byId("open-location").value = file.directory;
      byId("save-location").value = file.directory;
      byId("save-file").value =
        file.savePath +
        (document.querySelector('input[name="save-format"]:checked').value ===
        "embedded-encrypted"
          ? ".enc"
          : "");
    }
  }
  function startSaving() {
    byId("save-model").disabled = true;
    byId("save-model").setAttribute("aria-busy", "true");
    byId("save-status").textContent = "Saving model…";
    byId("save-submit").disabled = true;
  }
  function finishSaving(status, disabled) {
    byId("save-model").setAttribute("aria-busy", "false");
    byId("save-model").disabled = disabled;
    byId("save-submit").disabled = false;
    if (status === "CANCELLED")
      byId("save-status").textContent = "Save cancelled.";
  }
  function setLoading(pending, hasModel) {
    byId("open-model").disabled = pending;
    byId("open-submit").disabled = pending;
    byId("save-model").disabled =
      pending ||
      !hasModel ||
      byId("save-model").getAttribute("aria-busy") === "true";
  }
  function showOpenError(message) {
    byId("open-error").textContent = message;
  }

  return {
    handleMessage,
    reset,
    setFile,
    startSaving,
    finishSaving,
    setLoading,
    showOpenError,
  };
}
