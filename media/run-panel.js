function createRunPanel({
  byId,
  postMessage,
  activatePrompt,
  canActivate = () => true,
}) {
  let requestId = 0;
  let requested = false;
  let available = false;
  let busy = false;
  let retained = false;
  let releaseId = 0;
  let inputs = [];
  const fields = byId("run-fields");
  const status = byId("run-status");
  const results = byId("run-results");
  const stage = byId("run-stage");
  const engine = byId("run-engine");
  const keepLoaded = byId("run-keep-loaded");
  byId("run-settings-open").addEventListener("click", () => {
    byId("settings-tab").click();
    byId("settings-engine-tab").click();
    byId(
      `settings-${engine.value === "onnx-light" ? "onnx-light" : "onnxruntime"}-tab`,
    ).click();
  });
  const showLifecycle = (value) => {
    if (value) retained = value.retained;
    byId("run-unload").disabled = busy || !retained;
    const name = engine.value === "onnxruntime" ? "ONNX Runtime" : "onnx-light";
    byId("run-lifecycle").textContent = value
      ? `${name} session: ${value.reused ? "reused from the previous run" : "created for this run"}, ${value.retained ? "still loaded and available for the next run" : "released after execution"}. Initialization: ${value.loadMs.toFixed(3)} ms.${value.retained ? "" : " No session is kept alive between runs."}`
      : busy
        ? `${name}: run in progress. ${keepLoaded.checked ? "The session will stay loaded for the next run." : "A new session is created for this run and released when it finishes."}`
        : retained
          ? `${name}: session still loaded and available for the next run.`
          : `${name}: no retained session. ${keepLoaded.checked ? "The next Run model creates a session and keeps it loaded." : "A new session is created on every Run model; it is never reused between runs."}`;
  };
  const release = () => {
    if (retained)
      postMessage({ type: "run-release", requestId, releaseId: ++releaseId });
    retained = false;
    showLifecycle();
  };
  byId("run-unload").addEventListener("click", release);
  keepLoaded.addEventListener("change", () => {
    if (!keepLoaded.checked) release();
    else showLifecycle();
  });
  for (const id of ["run-verbose", "run-log-severity", "run-log-verbosity"])
    byId(id).addEventListener("change", release);
  const clearInputPreviews = () => {
    byId("run-input-previews").replaceChildren();
    byId("run-used-inputs").hidden = true;
  };
  const selectEngine = () => {
    const light = engine.value === "onnx-light";
    byId("run-logging-light").hidden = !light;
    byId("run-verbose").disabled = !light;
    byId("run-logging-ort").hidden = light;
    byId("run-log-severity").disabled = light;
    byId("run-log-verbosity").disabled = light;
    showLifecycle();
  };
  engine.addEventListener("change", () => {
    release();
    selectEngine();
  });
  selectEngine();
  const modes = ["prompt", "run"];
  const selectMode = (mode, focus = false) => {
    for (const name of modes) {
      const selected = name === mode;
      const tab = byId(`execution-${name}-tab`);
      tab.setAttribute("aria-selected", String(selected));
      tab.tabIndex = selected ? 0 : -1;
      byId(`execution-${name}-view`).hidden = !selected;
    }
    if (focus) byId(`execution-${mode}-tab`).focus();
    if (!canActivate()) return;
    if (mode === "prompt") activatePrompt();
    else api.activate();
  };
  for (const mode of modes) {
    const tab = byId(`execution-${mode}-tab`);
    tab.addEventListener("click", () => selectMode(mode));
    tab.addEventListener("keydown", (event) => {
      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key))
        return;
      event.preventDefault();
      selectMode(
        event.key === "Home"
          ? "prompt"
          : event.key === "End"
            ? "run"
            : mode === "prompt"
              ? "run"
              : "prompt",
        true,
      );
    });
  }
  const load = () => {
    release();
    requested = true;
    available = false;
    fields.disabled = true;
    results.replaceChildren();
    clearInputPreviews();
    byId("run-log").textContent = "";
    status.textContent = "Loading model inputs…";
    postMessage({
      type: "run-inputs",
      requestId: ++requestId,
      path: JSON.parse(stage.value),
    });
  };
  stage.addEventListener("change", load);
  byId("run-form").addEventListener("submit", (event) => {
    event.preventDefault();
    if (busy || !available) return;
    if (!byId("run-form").reportValidity()) return;
    const dimensions = Object.fromEntries(
      [...byId("run-dimensions").querySelectorAll("input")].map((field) => [
        field.dataset.dimension,
        Number(field.value),
      ]),
    );
    if (
      Object.values(dimensions).some((value) => !Number.isSafeInteger(value))
    ) {
      status.textContent = "Dynamic dimensions must be safe integers.";
      return;
    }
    busy = true;
    showLifecycle();
    fields.disabled = true;
    results.replaceChildren();
    clearInputPreviews();
    byId("run-log").textContent = "";
    status.textContent = "Running model…";
    postMessage({
      type: "run-model",
      requestId: ++requestId,
      path: JSON.parse(stage.value),
      engine: engine.value,
      keepLoaded: keepLoaded.checked,
      settings:
        engine.value === "onnx-light"
          ? { verbose: Number(byId("run-verbose").value) }
          : {
              logSeverity: Number(byId("run-log-severity").value),
              logVerbosity: Number(byId("run-log-verbosity").value),
            },
      dimensions,
      inputs: inputs.map((input, index) => {
        const source = byId(`run-source-${index}`).value;
        if (source === "text") return byId(`run-input-${index}`).value;
        return {
          name: input.name,
          source,
          ...(source === "protobuf"
            ? { path: byId(`run-file-${index}`).value }
            : {}),
        };
      }),
    });
  });
  const api = {
    reset() {
      ++requestId;
      requested = available = busy = false;
      retained = false;
      inputs = [];
      showLifecycle();
      clearInputPreviews();
      fields.disabled = true;
      stage.replaceChildren(new Option("Root graph", "[]"));
      byId("run-inputs").replaceChildren();
      byId("run-dimensions").replaceChildren();
      byId("run-log").textContent = "";
      results.replaceChildren();
      status.textContent = "Open a model to run it.";
    },
    activate() {
      if (!requested) load();
    },
    activateSelected() {
      if (byId("execution-run-view").hidden) activatePrompt();
      else api.activate();
    },
    showPrompt() {
      selectMode("prompt");
    },
    handleMessage(message) {
      if (message?.type === "run-session-lost") {
        retained = false;
        showLifecycle();
        status.textContent = `Session stopped: ${message.error}`;
        return true;
      }
      if (["metadata-error", "shape-error"].includes(message?.type)) {
        if (busy || (requested && !available)) {
          ++requestId;
          requested = available;
          busy = false;
          retained = false;
          showLifecycle();
          clearInputPreviews();
          fields.disabled = !available;
          status.textContent = `Run cancelled: ${message.error}`;
        }
        return false;
      }
      if (
        ![
          "run-inputs",
          "run-result",
          "run-error",
          "run-log",
          "run-file",
          "run-input-saved",
          "run-input-error",
          "run-output-saved",
          "run-output-error",
          "run-session-released",
          "run-session-error",
        ].includes(message?.type)
      )
        return false;
      if (message.requestId !== requestId) return true;
      if (
        ["run-session-released", "run-session-error"].includes(message.type)
      ) {
        if (message.releaseId !== releaseId) return true;
        retained = false;
        showLifecycle();
        if (message.type === "run-session-error")
          status.textContent = `Could not unload session: ${message.error}`;
        return true;
      }
      if (message.type === "run-log") {
        byId("run-log").textContent = message.output;
        return true;
      }
      if (message.type === "run-file") {
        const field = byId(`run-file-${message.index}`);
        if (field && typeof message.path === "string") {
          field.value = message.path;
          const source = byId(`run-source-${message.index}`);
          source.value = "protobuf";
          source.dispatchEvent(new Event("change"));
        }
        return true;
      }
      if (message.type === "run-input-saved") {
        const index = inputs.findIndex((input) => input.name === message.name);
        const field = byId(`run-file-${index}`);
        if (field) field.value = message.path;
        status.textContent = `Saved input ${message.name} to ${message.path}. Use Load input (.pb) to reload it.`;
        return true;
      }
      if (message.type === "run-output-saved") {
        status.textContent = `Saved output ${message.name} to ${message.path}.`;
        return true;
      }
      if (["run-input-error", "run-output-error"].includes(message.type)) {
        status.textContent = `Could not save ${message.type === "run-output-error" ? "output" : "input"}: ${message.error}`;
        return true;
      }
      busy = false;
      if (message.type === "run-inputs") {
        available = true;
        stage.replaceChildren(
          ...(message.stages.length
            ? message.stages.map(
                (item) => new Option(item.name, JSON.stringify(item.path)),
              )
            : [new Option("Root graph", "[]")]),
        );
        stage.value = JSON.stringify(message.path);
        const container = byId("run-inputs");
        container.replaceChildren();
        inputs = message.inputs;
        const dimensionFields = byId("run-dimensions");
        dimensionFields.replaceChildren();
        const dimensions = new Set();
        inputs.forEach((input) =>
          input.shape?.forEach((dimension, axis) => {
            if (
              input.dynamicAxes
                ? !input.dynamicAxes.includes(axis)
                : dimension !== null && /^\d+$/.test(dimension)
            )
              return;
            dimensions.add(
              dimension === null
                ? `axis:${input.name}:${axis}`
                : `symbol:${dimension}`,
            );
          }),
        );
        for (const dimension of dimensions) {
          const label = document.createElement("label");
          const field = document.createElement("input");
          field.id = `run-dimension-${dimensionFields.childElementCount / 2}`;
          label.htmlFor = field.id;
          label.textContent = `Dimension ${dimension}`;
          field.dataset.dimension = dimension;
          field.type = "number";
          field.min = "0";
          field.max = "9007199254740991";
          field.step = "1";
          field.required = true;
          dimensionFields.append(label, field);
        }
        message.inputs.forEach((input, index) => {
          const metadataLabel = document.createElement("span");
          const metadata = document.createElement("pre");
          metadata.id = `run-metadata-${index}`;
          metadataLabel.id = `run-metadata-label-${index}`;
          metadata.setAttribute("aria-labelledby", metadataLabel.id);
          metadataLabel.textContent = `${input.name} type / shape`;
          metadata.textContent = input.text;
          const label = document.createElement("label");
          label.htmlFor = `run-input-${index}`;
          label.textContent = input.name;
          const field = document.createElement("textarea");
          field.id = label.htmlFor;
          field.rows = 6;
          field.required = true;
          field.value = input.text;
          const sourceLabel = document.createElement("label");
          const source = document.createElement("select");
          source.id = `run-source-${index}`;
          sourceLabel.htmlFor = source.id;
          sourceLabel.textContent = `${input.name} source`;
          source.append(
            new Option("Random", "random"),
            new Option("TensorProto file (.pb)", "protobuf"),
            new Option("Tensor text", "text"),
          );
          source.value = input.shape ? "random" : "text";
          const sourceControls = document.createElement("div");
          sourceControls.className = "run-file-controls";
          const loadInput = document.createElement("button");
          loadInput.id = `run-load-${index}`;
          loadInput.type = "button";
          loadInput.textContent = "Load input (.pb)…";
          loadInput.addEventListener("click", () =>
            postMessage({ type: "run-file", requestId, index }),
          );
          sourceControls.append(source, loadInput);
          const fileLabel = document.createElement("label");
          const fileControls = document.createElement("div");
          fileControls.className = "run-file-controls";
          const file = document.createElement("input");
          file.id = `run-file-${index}`;
          file.type = "text";
          file.placeholder = "Absolute path to an inline TensorProto";
          file.required = true;
          fileLabel.htmlFor = file.id;
          fileLabel.textContent = `${input.name} file`;
          const browse = document.createElement("button");
          browse.type = "button";
          browse.textContent = "Browse…";
          browse.addEventListener("click", () =>
            postMessage({ type: "run-file", requestId, index }),
          );
          fileControls.append(file, browse);
          const selectSource = () => {
            label.hidden = field.hidden = source.value !== "text";
            fileLabel.hidden = fileControls.hidden =
              source.value !== "protobuf";
            field.disabled = source.value !== "text";
            file.disabled = source.value !== "protobuf";
          };
          source.addEventListener("change", selectSource);
          selectSource();
          container.append(
            metadataLabel,
            metadata,
            sourceLabel,
            sourceControls,
            fileLabel,
            fileControls,
            label,
            field,
          );
        });
        status.textContent =
          "Ready. Specify dynamic dimensions and input sources, then Run model.";
      } else if (message.type === "run-error") {
        status.textContent = message.error;
        retained = false;
        showLifecycle();
      } else {
        status.textContent = `Completed in ${message.result.timeMs.toFixed(3)} ms (execution only).`;
        showLifecycle(message.result.lifecycle);
        const previews = byId("run-input-previews");
        for (const [index, input] of (message.result.inputs ?? []).entries()) {
          const heading = document.createElement("h3");
          heading.textContent = input.name;
          const values = document.createElement("pre");
          values.textContent = input.text;
          const save = document.createElement("button");
          save.type = "button";
          save.textContent = "Save input (.pb)…";
          save.addEventListener("click", () =>
            postMessage({ type: "run-save-input", requestId, index }),
          );
          previews.append(heading, values, save);
        }
        byId("run-used-inputs").hidden = !previews.childElementCount;
        for (const [index, output] of message.result.outputs.entries()) {
          const heading = document.createElement("h3");
          heading.textContent = output.name;
          const values = document.createElement("pre");
          values.textContent = output.text;
          results.append(heading, values);
          if (output.canSave) {
            const save = document.createElement("button");
            save.type = "button";
            save.textContent = "Save output (.pb)…";
            save.addEventListener("click", () =>
              postMessage({ type: "run-save-output", requestId, index }),
            );
            results.append(save);
          }
        }
      }
      fields.disabled = !available;
      return true;
    },
  };
  return api;
}
