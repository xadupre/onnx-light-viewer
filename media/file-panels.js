function createFilePanels({
  byId,
  postMessage,
  requestAction,
  revealPanel,
  isReplacing,
}) {
  const fileRequests = { open: 0, save: 0 };
  const fileParents = {};
  let openSource = "local";
  let pipelineFile = false;
  function fileSize(bytes) {
    const units = ["B", "KiB", "MiB", "GiB", "TiB"];
    const unit =
      bytes > 0
        ? Math.min(
            units.length - 1,
            Math.floor(Math.log(bytes) / Math.log(1024)),
          )
        : 0;
    return `${(bytes / 1024 ** unit).toLocaleString(undefined, { maximumFractionDigits: 1 })} ${units[unit]}`;
  }
  let hubRequest = 0;
  const hubModelSizes = new Map();
  function searchHub() {
    hubModelSizes.clear();
    byId("hf-search-results").replaceChildren();
    byId("hf-search-results").setAttribute("aria-busy", "true");
    byId("hf-search-status").textContent = "Searching Hugging Face…";
    byId("hf-results-title").textContent = byId("hf-query").value.trim()
      ? "ONNX search results"
      : "Popular ONNX models";
    postMessage({
      type: "search-huggingface",
      query: byId("hf-query").value,
      task: byId("hf-task").value,
      requestId: ++hubRequest,
    });
  }
  function invalidateHubSearch() {
    hubRequest++;
    byId("hf-search-results").setAttribute("aria-busy", "false");
    byId("hf-search-status").textContent = "";
  }
  byId("hf-search-form").addEventListener("submit", (event) => {
    event.preventDefault();
    searchHub();
  });
  byId("hf-task").addEventListener("change", searchHub);
  let pathHistory = { open: [], save: [] };
  function renderHistory() {
    for (const target of ["open", "save"]) {
      const source = target === "open" ? openSource : "local";
      const entries = pathHistory[target].filter(
        (entry) => entry.source === source,
      );
      for (const [field, kind] of [
        ["file", "file"],
        ["location", "directory"],
      ]) {
        byId(`${target}-${field}-history`).replaceChildren(
          ...entries
            .filter((entry) => entry.kind === kind)
            .map((entry) => {
              const option = document.createElement("option");
              option.value = entry.path;
              return option;
            }),
        );
      }
      byId(`${target}-history`).hidden = entries.length === 0;
      byId(`${target}-history-entries`).replaceChildren(
        ...entries.map((entry) => {
          const item = document.createElement("li");
          const choose = document.createElement("button");
          choose.type = "button";
          choose.textContent = `${entry.kind === "directory" ? "[Folder] " : ""}${entry.path}`;
          choose.addEventListener("click", () => {
            const field = entry.kind === "directory" ? "location" : "file";
            byId(`${target}-${field}`).value = entry.path;
            byId(`${target}-${field}`).focus();
          });
          const remove = document.createElement("button");
          remove.type = "button";
          remove.textContent = "Remove";
          remove.setAttribute(
            "aria-label",
            `Remove ${entry.path} from ${target} history`,
          );
          remove.addEventListener("click", () =>
            postMessage({
              type: "path-history-edit",
              target,
              historyAction: "remove",
              historyEntry: entry,
            }),
          );
          item.append(choose, remove);
          return item;
        }),
      );
    }
  }
  function browseFiles(target) {
    if (target === "open") byId("open-file-list").hidden = false;
    byId(target === "open" ? "open-error" : "save-options-error").textContent =
      "";
    byId(`${target}-files`).setAttribute("aria-busy", "true");
    postMessage({
      type: "browse-files",
      target,
      source: target === "open" ? openSource : "local",
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
      if (target === "save" || openSource !== "huggingface")
        browseFiles(target);
      else searchHub();
    } else {
      if (target === "open") invalidateHubSearch();
      fileRequests[target]++;
      byId(`${target}-files`).setAttribute("aria-busy", "false");
      for (const input of panel.querySelectorAll('input[type="password"]'))
        input.value = "";
    }
  }
  for (const target of ["open", "save"]) {
    byId(`${target}-clear-history`).addEventListener("click", () =>
      postMessage({
        type: "path-history-edit",
        target,
        historyAction: "clear",
      }),
    );
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
  function selectSource(source) {
    if (openSource === source) return;
    openSource = source;
    for (const button of byId("open-source").querySelectorAll("button"))
      button.setAttribute(
        "aria-pressed",
        String(button.dataset.source === source),
      );
    invalidateHubSearch();
    byId("hf-search-results").replaceChildren();
    const hub = source === "huggingface";
    byId("hf-discovery").hidden = !hub;
    byId("hf-files-title").hidden = !hub;
    byId("hf-files-help").hidden = !hub;
    byId("open-file-list").hidden = hub;
    byId("open-browser").hidden = source === "backend";
    byId("open-browser").open = !hub;
    byId("open-browser-title").textContent = hub
      ? "Or enter a repository / URL"
      : "Browse folders";
    fileRequests.open++;
    byId("open-error").textContent = "";
    byId("open-location").value = "";
    byId("open-file").value = "";
    byId("open-filter").value = "";
    byId("open-password").value = "";
    byId("open-files").replaceChildren();
    byId("open-files").setAttribute("aria-busy", "false");
    byId("open-parent").disabled = true;
    renderHistory();
    byId("open-browse").hidden = source === "backend";
    if (!hub) browseFiles("open");
    else searchHub();
  }
  for (const button of byId("open-source").querySelectorAll("button"))
    button.addEventListener("click", () => selectSource(button.dataset.source));
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
      source: openSource,
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
    if (event.target.name === "save-format" && !pipelineFile)
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
      includeRewritings: byId("save-rewritings").checked,
      storage:
        format === "reuse-plain" ? "reuse" : external ? "external" : "embedded",
      encrypted: format === "embedded-encrypted",
      ...(external ? { maxExternalFileSize } : {}),
    });
    byId("save-password").value = "";
    byId("save-confirmation").value = "";
  });

  function handleMessage(data) {
    if (data?.type === "huggingface-model-files") {
      if (data.requestId !== hubRequest || openSource !== "huggingface")
        return true;
      const label = hubModelSizes.get(data.id);
      if (!label) return true;
      if (data.error) {
        label.textContent = `ONNX size unavailable: ${data.error}`;
        return true;
      }
      const sizes = data.entries
        .filter((entry) => entry.size !== undefined)
        .map((entry) => entry.size);
      if (!sizes.length) {
        label.textContent = "ONNX size not supplied by the Hub";
        return true;
      }
      const minimum = Math.min(...sizes);
      const maximum = Math.max(...sizes);
      label.textContent = `ONNX files: ${fileSize(minimum)}${minimum === maximum ? "" : ` - ${fileSize(maximum)}`} (${sizes.length}/${data.entries.length} sizes known)`;
    } else if (
      data?.type === "huggingface-search-results" ||
      data?.type === "huggingface-search-error"
    ) {
      if (data.requestId !== hubRequest || openSource !== "huggingface")
        return true;
      byId("hf-search-results").setAttribute("aria-busy", "false");
      if (data.type === "huggingface-search-error") {
        byId("hf-search-status").textContent = data.error;
        return true;
      }
      byId("hf-search-status").textContent = data.models.length
        ? `${data.models.length} repositories, most downloaded first. Select one to browse its ONNX files.`
        : "No public ONNX models found. Try another search or task.";
      byId("hf-search-results").replaceChildren(
        ...data.models.map((model) => {
          const item = document.createElement("li");
          const button = document.createElement("button");
          button.type = "button";
          button.append(document.createTextNode(model.id));
          const detail = document.createElement("span");
          detail.className = "hub-model-details";
          const task = Array.from(byId("hf-task").options).find(
            (option) => option.value === model.task,
          );
          detail.textContent = `Task: ${task?.textContent ?? model.task ?? "Unspecified"} · ${model.downloads.toLocaleString()} downloads`;
          button.append(detail);
          const size = document.createElement("span");
          size.className = "hub-model-size";
          size.textContent = "ONNX file sizes: loading…";
          hubModelSizes.set(model.id, size);
          button.append(size);
          button.addEventListener("click", () => {
            byId("hf-search-status").textContent =
              `Loading ONNX files from ${model.id}…`;
            byId("open-location").value = model.url;
            byId("open-file").value = "";
            byId("open-filter").value = "";
            byId("open-files").replaceChildren();
            browseFiles("open");
          });
          item.append(button);
          return item;
        }),
      );
    } else if (data?.type === "path-history") {
      pathHistory = data.history;
      renderHistory();
    } else if (data?.type === "file-list" || data?.type === "file-list-error") {
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
          if (
            entry.size !== undefined ||
            (target === "open" && openSource === "huggingface")
          ) {
            const size = document.createElement("span");
            size.className = "file-size";
            size.textContent =
              entry.size === undefined
                ? "File size unavailable"
                : `File size: ${fileSize(entry.size)}`;
            button.append(size);
          }
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
          if (target === "open" && !entry.directory)
            button.addEventListener("dblclick", () => {
              if (openSource !== "backend" && openSource !== "huggingface")
                return;
              byId("open-file").value = entry.path;
              byId("open-form").requestSubmit();
            });
          item.append(button);
          return item;
        }),
      );
      if (target === "open") filterModelFiles();
      if (target === "open" && openSource === "huggingface") {
        byId("hf-search-status").textContent =
          "Double-click an ONNX file to download and open it.";
        byId("hf-files-title").focus();
      }
    } else if (data?.type === "save-info") {
      byId("save-status").textContent = `Saved ${data.name}`;
      showProgress("save", false);
    } else if (data?.type === "save-error") {
      byId("save-options-error").textContent = data.error;
      byId("save-status").textContent = `Could not save model: ${data.error}`;
      showProgress("save", false);
    } else {
      return false;
    }
    return true;
  }
  function reset() {
    invalidateHubSearch();
    for (const target of ["open", "save"]) {
      fileRequests[target]++;
      byId(`${target}-files`).setAttribute("aria-busy", "false");
    }
    byId("save-status").textContent = "";
  }
  function setFile(file) {
    if (file) {
      pipelineFile = file.pipeline === true;
      const encrypted = document.querySelector(
        'input[name="save-format"][value="embedded-encrypted"]',
      );
      encrypted.disabled = pipelineFile;
      encrypted.title = pipelineFile
        ? "PipelineProto encryption is not supported by onnx-light."
        : "";
      if (pipelineFile && encrypted.checked) {
        const plain = document.querySelector(
          'input[name="save-format"][value="embedded-plain"]',
        );
        plain.checked = true;
        plain.dispatchEvent(new Event("change", { bubbles: true }));
        byId("save-password").value = "";
        byId("save-confirmation").value = "";
      }
      if (openSource === "local") byId("open-location").value = file.directory;
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
    showProgress("save", true);
    byId("save-model").disabled = true;
    byId("save-model").setAttribute("aria-busy", "true");
    byId("save-status").textContent = "Saving model…";
    byId("save-submit").disabled = true;
  }
  function finishOpening(status) {
    if (status === "SUCCEEDED") {
      const restoreFocus = byId("open-panel").contains(document.activeElement);
      toggleFilePanel("open", false);
      if (restoreFocus) byId("open-model").focus();
    }
  }
  function finishSaving(status, disabled) {
    showProgress("save", false);
    byId("save-model").setAttribute("aria-busy", "false");
    byId("save-model").disabled = disabled;
    byId("save-submit").disabled = false;
    if (status === "CANCELLED")
      byId("save-status").textContent = "Save cancelled.";
  }
  function setLoading(pending, hasModel) {
    showProgress("open", pending);
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

  function showProgress(target, pending) {
    byId(`${target}-progress`).hidden = !pending;
    byId(`${target}-progress-status`).hidden = !pending;
    if (pending)
      setProgress({
        target,
        stage: target === "open" ? "Opening model" : "Saving model",
      });
  }

  function setProgress({ target, stage, received, total }) {
    const text = `${stage}${received === undefined ? "" : `: ${received.toLocaleString()}${total ? ` / ${total.toLocaleString()}` : ""} bytes`}`;
    const bars = [byId(`${target}-progress`)];
    if (target === "open") bars.push(byId("load-progress"));
    for (const bar of bars) {
      if (total > 0 && Number.isFinite(total) && Number.isFinite(received)) {
        bar.max = total;
        bar.value = received;
      } else {
        bar.removeAttribute("value");
        bar.removeAttribute("max");
      }
      bar.setAttribute("aria-valuetext", text);
    }
    byId(`${target}-progress-status`).textContent = text;
  }

  return {
    handleMessage,
    reset,
    setFile,
    startSaving,
    finishOpening,
    finishSaving,
    setLoading,
    showOpenError,
    setProgress,
  };
}
