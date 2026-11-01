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
  let inputs = [];
  const fields = byId("run-fields");
  const status = byId("run-status");
  const results = byId("run-results");
  const stage = byId("run-stage");
  const engine = byId("run-engine");
  const selectEngine = () => {
    const light = engine.value === "onnx-light";
    byId("run-logging-light").hidden = !light;
    byId("run-verbose").disabled = !light;
    byId("run-logging-ort").hidden = light;
    byId("run-log-severity").disabled = light;
    byId("run-log-verbosity").disabled = light;
  };
  engine.addEventListener("change", selectEngine);
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
    requested = true;
    available = false;
    fields.disabled = true;
    results.replaceChildren();
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
    fields.disabled = true;
    results.replaceChildren();
    byId("run-log").textContent = "";
    status.textContent = "Running model…";
    postMessage({
      type: "run-model",
      requestId: ++requestId,
      path: JSON.parse(stage.value),
      engine: engine.value,
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
      inputs = [];
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
      if (["metadata-error", "shape-error"].includes(message?.type)) {
        if (busy || (requested && !available)) {
          ++requestId;
          requested = available;
          busy = false;
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
        ].includes(message?.type)
      )
        return false;
      if (message.requestId !== requestId) return true;
      if (message.type === "run-log") {
        byId("run-log").textContent = message.output;
        return true;
      }
      if (message.type === "run-file") {
        const field = byId(`run-file-${message.index}`);
        if (field && typeof message.path === "string")
          field.value = message.path;
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
            sourceLabel,
            source,
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
      } else {
        status.textContent = `Completed in ${message.result.timeMs.toFixed(3)} ms (execution only).`;
        for (const output of message.result.outputs) {
          const heading = document.createElement("h3");
          heading.textContent = output.name;
          const values = document.createElement("pre");
          values.textContent = output.text;
          results.append(heading, values);
        }
      }
      fields.disabled = !available;
      return true;
    },
  };
  return api;
}
