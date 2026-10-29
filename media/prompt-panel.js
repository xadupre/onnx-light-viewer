function createPromptPanel({ byId, postMessage }) {
  let requestId = 0;
  let available = false;
  let busy = false;
  let requested = false;
  let result;
  let options;
  let streaming = false;
  let liveRows = [];
  let timingSummary;
  let prefill;
  let decodeSum = 0;
  let decodeCount = 0;
  let missingDecode = false;
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
    streaming = false;
    liveRows = [];
    content.replaceChildren();
    progress.hidden = true;
    validate.hidden = true;
    setBusy(false);
    status.textContent = "Open a converted language model to run a prompt.";
  };
  const milliseconds = (value) =>
    value == null ? "Unavailable (SDK)" : value.toFixed(3);
  const timingText = (label, first, sum, count, missing) =>
    `${label}: prefill ${milliseconds(first)} ms; decode mean ${
      !count
        ? "N/A (no decode tokens)"
        : milliseconds(missing ? null : sum / count) + " ms"
    } (${count} decode tokens).`;
  const recordTiming = (token, index) => {
    if (index === 0) prefill = token.timeMs;
    else {
      decodeCount++;
      if (token.timeMs == null) missingDecode = true;
      else decodeSum += token.timeMs;
    }
  };
  const updateTiming = () => {
    timingSummary.textContent = timingText(
      result.engine,
      prefill,
      decodeSum,
      decodeCount,
      missingDecode,
    );
  };
  const render = () => {
    liveRows = [];
    prefill = undefined;
    decodeSum = 0;
    decodeCount = 0;
    missingDecode = false;
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
      return tr;
    };
    const tokens = result.tokens;
    for (const [label, value] of [
      ["Output token", (_, index) => index + 1],
      ["Phase", (_, index) => (index === 0 ? "Prefill" : "Decode")],
      ["Output", (token) => token.text],
      ["Token ID", (token) => token.id],
      ["Score (logit)", (token) => token.score ?? "Unavailable (SDK)"],
      ["Time (ms)", (token) => milliseconds(token.timeMs)],
    ])
      liveRows.push({ row: row(label, tokens.map(value)), value });
    tokens.forEach(recordTiming);
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
      row(
        "Hugging Face time (ms)",
        result.reference.tokens.map((token) => milliseconds(token.timeMs)),
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
    timingSummary = document.createElement("p");
    updateTiming();
    const referenceTiming = document.createElement("p");
    referenceTiming.hidden = !result.reference;
    if (result.reference) {
      const decode = result.reference.tokens.slice(1);
      referenceTiming.textContent = timingText(
        "Hugging Face",
        result.reference.tokens[0]?.timeMs,
        decode.reduce((sum, token) => sum + (token.timeMs ?? 0), 0),
        decode.length,
        decode.some((token) => token.timeMs == null),
      );
    }
    content.replaceChildren(
      prompt,
      limitations,
      timingSummary,
      referenceTiming,
      table,
    );
  };
  const run = (validation) => {
    if (busy || !available) return;
    if (!validation) {
      options = {
        engine: byId("prompt-engine").value,
        prompt: byId("prompt-text").value,
        maxTokens: Number(byId("prompt-max-tokens").value),
      };
      result = { ...options, tokens: [] };
      render();
      validate.hidden = true;
    }
    streaming = !validation;
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
    streaming = false;
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
          "prompt-token",
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
          ? "Ready. ONNX engines and tokenization use native C++ libraries; Python is only needed for Hugging Face validation."
          : message.info.reason;
      } else if (message.type === "prompt-token") {
        if (!busy || !streaming || message.index !== result.tokens.length)
          return true;
        result.tokens.push(message.token);
        for (const { row, value } of liveRows) {
          const cell = document.createElement("td");
          cell.textContent = String(value(message.token, message.index));
          row.append(cell);
        }
        recordTiming(message.token, message.index);
        updateTiming();
        status.textContent = `Generated ${result.tokens.length}/${options.maxTokens} tokens…`;
      } else if (message.type === "prompt-progress") {
        progress.textContent = message.output;
        progress.hidden = !message.output;
      } else if (message.type === "prompt-error") {
        streaming = false;
        setBusy(false);
        status.textContent = message.error;
      } else {
        streaming = false;
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
