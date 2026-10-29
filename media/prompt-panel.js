function createPromptPanel({ byId, postMessage }) {
  let requestId = 0;
  let available = false;
  let busy = false;
  let requested = false;
  let result;
  let options;
  const fields = byId("prompt-fields");
  const status = byId("prompt-status");
  const content = byId("prompt-results");
  const validate = byId("prompt-validate");
  const cancel = byId("prompt-cancel");
  const progress = byId("prompt-progress");
  const setBusy = (value) => {
    busy = value;
    fields.disabled = value || !available;
    cancel.hidden = !value;
    validate.disabled = value;
  };
  const reset = () => {
    if (busy) postMessage({ type: "prompt-cancel" });
    ++requestId;
    available = false;
    requested = false;
    result = undefined;
    options = undefined;
    content.replaceChildren();
    progress.hidden = true;
    validate.hidden = true;
    setBusy(false);
    status.textContent = "Open a converted language model to run a prompt.";
  };
  const render = () => {
    const table = document.createElement("table");
    table.className = "prompt-token-table";
    table.setAttribute("aria-label", "Token-aligned generation and validation");
    const row = (label, values) => {
      const tr = document.createElement("tr");
      const th = document.createElement("th");
      th.scope = "row";
      th.textContent = label;
      tr.append(th);
      for (const value of values) {
        const td = document.createElement("td");
        td.textContent = String(value);
        tr.append(td);
      }
      table.append(tr);
    };
    const tokens = result.tokens;
    row(
      "Output token",
      tokens.map((_, index) => index + 1),
    );
    row(
      "Output",
      tokens.map((token) => token.text),
    );
    row(
      "Token ID",
      tokens.map((token) => token.id),
    );
    row(
      "Score (logit)",
      tokens.map((token) => token.score ?? "Unavailable (SDK)"),
    );
    if (result.reference) {
      row(
        "Hugging Face output",
        result.reference.tokens.map((token) => token.text),
      );
      row(
        "Hugging Face token ID",
        result.reference.tokens.map((token) => token.id),
      );
      row(
        "Hugging Face score (logit)",
        result.reference.tokens.map((token) => token.score),
      );
    }
    if (result.discrepancies) {
      for (const [label, key] of [
        ["Maximum absolute distance", "maxAbsolute"],
        ["Relative error", "relative"],
        ["L2 distance", "l2"],
        ["Coefficients Δ < 0.01", "below001"],
        ["Coefficients Δ < 0.1", "below01"],
        ["Coefficients Δ < 1", "below1"],
        ["Coefficients compared", "total"],
      ])
        row(
          label,
          result.discrepancies.map((value) => value[key]),
        );
    }
    const prompt = document.createElement("p");
    prompt.textContent = `Prompt: ${result.prompt}`;
    const limitations = document.createElement("p");
    limitations.textContent = (result.limitations ?? []).join(" ");
    limitations.hidden = !limitations.textContent;
    content.replaceChildren(prompt, limitations, table);
  };
  const run = (validation) => {
    if (busy || !available) return;
    if (!validation) {
      options = {
        engine: byId("prompt-engine").value,
        prompt: byId("prompt-text").value,
        maxTokens: Number(byId("prompt-max-tokens").value),
      };
      result = undefined;
      content.replaceChildren();
      validate.hidden = true;
    }
    setBusy(true);
    progress.textContent = "";
    progress.hidden = true;
    status.textContent = validation
      ? "Validating with Hugging Face…"
      : "Running prompt…";
    postMessage({
      type: "run-prompt",
      requestId: ++requestId,
      options: { ...options, validate: validation },
    });
  };
  byId("prompt-form").addEventListener("submit", (event) => {
    event.preventDefault();
    run(false);
  });
  validate.addEventListener("click", () => run(true));
  cancel.addEventListener("click", () => {
    postMessage({ type: "prompt-cancel" });
    ++requestId;
    setBusy(false);
    status.textContent = "Cancelled.";
  });
  return {
    reset,
    load() {
      reset();
      requested = true;
      postMessage({ type: "prompt-info", requestId });
    },
    activate() {
      if (!requested) this.load();
    },
    handleMessage(message) {
      if (
        ![
          "prompt-info",
          "prompt-result",
          "prompt-error",
          "prompt-progress",
        ].includes(message?.type)
      )
        return false;
      if (message.requestId !== requestId) return true;
      if (message.type === "prompt-info") {
        available = message.info.available;
        const engine = byId("prompt-engine");
        for (const option of engine.options)
          option.disabled = !message.info.engines.includes(option.value);
        engine.value = message.info.engines[0] ?? "onnx-light";
        setBusy(false);
        status.textContent = available
          ? "Ready. Runtime and tokenizer packages must be installed in Settings' Python environment."
          : message.info.reason;
      } else if (message.type === "prompt-progress") {
        progress.textContent = message.output;
        progress.hidden = !message.output;
      } else if (message.type === "prompt-error") {
        setBusy(false);
        status.textContent = message.error;
      } else {
        result = message.result;
        setBusy(false);
        render();
        validate.hidden = false;
        status.textContent = result.reference
          ? "Validation completed."
          : "Generation completed.";
      }
      return true;
    },
  };
}
