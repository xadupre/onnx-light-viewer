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
  const fields = byId("run-fields");
  const status = byId("run-status");
  const results = byId("run-results");
  const stage = byId("run-stage");
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
    busy = true;
    fields.disabled = true;
    results.replaceChildren();
    status.textContent = "Running model…";
    postMessage({
      type: "run-model",
      requestId: ++requestId,
      path: JSON.parse(stage.value),
      inputs: [...byId("run-inputs").querySelectorAll("textarea")].map(
        (field) => field.value,
      ),
    });
  });
  const api = {
    reset() {
      ++requestId;
      requested = available = busy = false;
      fields.disabled = true;
      stage.replaceChildren(new Option("Root graph", "[]"));
      byId("run-inputs").replaceChildren();
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
      if (!["run-inputs", "run-result", "run-error"].includes(message?.type))
        return false;
      if (message.requestId !== requestId) return true;
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
        message.inputs.forEach((input, index) => {
          const label = document.createElement("label");
          label.htmlFor = `run-input-${index}`;
          label.textContent = input.name;
          const field = document.createElement("textarea");
          field.id = label.htmlFor;
          field.rows = 6;
          field.required = true;
          field.value = input.text;
          container.append(label, field);
        });
        status.textContent =
          "Ready. Enter input tensor values, then Run model.";
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
