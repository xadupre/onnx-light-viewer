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
  let validationRequiresReference = false;
  const fields = byId("prompt-fields");
  const status = byId("prompt-status");
  const content = byId("prompt-results");
  const validate = byId("prompt-validate");
  const referenceSource = byId("prompt-reference-source");
  const validateRun = byId("prompt-validate-run");
  const updateReference = () => {
    const missing =
      validationRequiresReference && !referenceSource.value.trim();
    referenceSource.required =
      validateRun.checked && validationRequiresReference;
    validate.disabled = busy || missing;
    byId("prompt-validation-status").textContent = !available
      ? "Open a model to check whether its Hugging Face reference is identified."
      : missing
        ? "HF validation is unavailable: this export does not identify its original checkpoint. Enter a model ID or local checkpoint above before requesting validation. Generation alone is still available."
        : referenceSource.value.trim()
          ? "HF validation will use the model ID or checkpoint above. Local reference files are checked before a validated generation starts; no weights are downloaded."
          : "HF validation will use the recorded export source or local checkpoint. Reference files are checked before a validated generation starts.";
  };
  referenceSource.addEventListener("input", updateReference);
  validateRun.addEventListener("change", updateReference);
  const cancel = byId("prompt-cancel");
  const progress = byId("prompt-progress");
  const showProgress = (output) => {
    progress.textContent = output;
    progress.hidden = !output;
    byId("prompt-output").hidden = !output;
  };
  const lifecycleStatus = byId("prompt-lifecycle");
  const sessionFields = byId("prompt-session-fields");
  const showSettingsEngine = () => {
    const light = byId("prompt-engine").value === "onnx-light";
    byId("prompt-settings-light").hidden = !light;
    byId("prompt-settings-genai").hidden = light;
    byId("prompt-settings-title").textContent = light
      ? "onnx-light session settings"
      : "ONNX Runtime GenAI session settings";
  };
  byId("prompt-settings-open").addEventListener("click", () => {
    showSettingsEngine();
    byId("prompt-run-page").hidden = true;
    byId("prompt-session-page").hidden = false;
  });
  byId("prompt-settings-back").addEventListener("click", () => {
    byId("prompt-run-page").hidden = false;
    byId("prompt-session-page").hidden = true;
  });
  const release = () => {
    postMessage({ type: "prompt-release", requestId });
    if (result?.lifecycle) result.lifecycle.retained = false;
  };
  byId("prompt-unload").addEventListener("click", release);
  byId("prompt-engine").addEventListener("change", release);
  byId("prompt-keep-loaded").addEventListener("change", () => {
    if (!byId("prompt-keep-loaded").checked) release();
  });
  sessionFields.addEventListener("change", () => {
    const light = byId("prompt-settings-light");
    const sdkLoader =
      light.querySelector('[data-setting="sdkLoader"]').value === "true";
    for (const field of light.querySelectorAll("[data-sdk-loader]"))
      field.disabled = !sdkLoader;
    const pane = byId("prompt-settings-genai");
    const provider = pane.querySelector('[data-setting="provider"]').value;
    for (const field of pane.querySelectorAll("[data-gpu]"))
      field.disabled = !["cuda", "dml"].includes(provider);
    for (const field of pane.querySelectorAll("[data-cuda]"))
      field.disabled = provider !== "cuda";
    release();
  });
  const readSettings = () => {
    const pane = byId(
      byId("prompt-engine").value === "onnx-light"
        ? "prompt-settings-light"
        : "prompt-settings-genai",
    );
    const settings = {};
    for (const field of pane.querySelectorAll("[data-setting]")) {
      if (field.disabled || (field.value === "" && field.type !== "checkbox"))
        continue;
      if (!field.checkValidity()) {
        showSettingsEngine();
        byId("prompt-run-page").hidden = true;
        byId("prompt-session-page").hidden = false;
        field.reportValidity();
        return;
      }
      settings[field.dataset.setting] =
        field.type === "checkbox"
          ? field.checked
          : field.type === "number"
            ? Number(field.value)
            : field.hasAttribute("data-boolean")
              ? field.value === "true"
              : field.value;
    }
    return settings;
  };
  const showLifecycle = (value) => {
    lifecycleStatus.textContent = value
      ? `${value.reused ? "Reused model; initial" : "Model"} loading: ${formatDecimal(value.loadMs)} ms; warmup (1 input token, 1 generation step): ${value.warmupMs == null ? "running…" : formatDecimal(value.warmupMs) + " ms"}. ${value.retained ? "Kept loaded." : busy ? "Unload after run." : "Unloaded."}`
      : "";
  };
  const setBusy = (value) => {
    busy = value;
    fields.disabled = value || !available;
    cancel.hidden = !value;
    updateReference();
    byId("prompt-reference-source").disabled = value;
    sessionFields.disabled = value;
    byId("prompt-settings-open").disabled = value;
    byId("prompt-unload").disabled = value;
  };
  const reset = () => {
    if (busy) postMessage({ type: "prompt-cancel" });
    else if (result?.lifecycle?.retained)
      postMessage({ type: "prompt-release" });
    ++requestId;
    available = false;
    validationRequiresReference = false;
    requested = false;
    result = undefined;
    options = undefined;
    byId("prompt-reference-source").value = "";
    showLifecycle();
    streaming = false;
    liveRows = [];
    content.replaceChildren();
    showProgress("");
    validate.hidden = true;
    setBusy(false);
    status.textContent = "Open a converted language model to run a prompt.";
  };
  const formatDecimal = (value) =>
    value == null ? "Unavailable (SDK)" : value.toFixed(3);
  const timingText = (label, first, sum, count, missing) =>
    `${label}: prefill ${formatDecimal(first)} ms; decode mean ${
      !count
        ? "N/A (no decode tokens)"
        : formatDecimal(missing ? null : sum / count) + " ms"
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
  const updateSectionHeading = (row, label, columnCount) => {
    row.replaceChildren();
    let remaining = columnCount;
    const heading = document.createElement("th");
    heading.colSpan = Math.min(remaining, 1000);
    heading.scope = "rowgroup";
    heading.textContent = label;
    row.append(heading);
    remaining -= heading.colSpan;
    while (remaining > 0) {
      const cell = document.createElement("td");
      cell.colSpan = Math.min(remaining, 1000);
      cell.setAttribute("aria-hidden", "true");
      row.append(cell);
      remaining -= cell.colSpan;
    }
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
    const tokens = result.tokens;
    const referenceTokens = result.reference?.tokens;
    const columnCount = Math.max(tokens.length, referenceTokens?.length ?? 0);
    let group;
    const section = (label) => {
      group = document.createElement("tbody");
      const tr = document.createElement("tr");
      tr.className = "prompt-token-section";
      updateSectionHeading(tr, label, columnCount + 1);
      group.append(tr);
      table.append(group);
    };
    if (referenceTokens) {
      const caption = document.createElement("caption");
      caption.textContent =
        "Red columns indicate token ID disagreement or a missing token.";
      table.append(caption);
    }
    const row = (label, values) => {
      const tr = document.createElement("tr");
      const th = document.createElement("th");
      th.scope = "row";
      th.textContent = label;
      tr.append(th);
      for (let index = 0; index < columnCount; index++) {
        const td = document.createElement("td");
        td.textContent = index < values.length ? String(values[index]) : "—";
        if (
          referenceTokens &&
          tokens[index]?.id !== referenceTokens[index]?.id
        ) {
          td.className = "prompt-token-mismatch";
          td.title = `Token ${index + 1}: ONNX and Transformers token IDs disagree or a token is missing.`;
        }
        tr.append(td);
      }
      group.append(tr);
      return tr;
    };
    section(result.engine ? `ONNX · ${result.engine}` : "ONNX");
    for (const [label, value] of [
      ["Output token", (_, index) => index + 1],
      ["Phase", (_, index) => (index === 0 ? "Prefill" : "Decode")],
      ["Output", (token) => token.text],
      ["Token ID", (token) => token.id],
      ["Score (logit)", (token) => formatDecimal(token.score)],
      ["Time (ms)", (token) => formatDecimal(token.timeMs)],
    ])
      liveRows.push({ row: row(label, tokens.map(value)), value });
    tokens.forEach(recordTiming);
    if (result.reference) {
      section("Hugging Face · Transformers");
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
        result.reference.tokens.map((token) => formatDecimal(token.score)),
      );
      row(
        "Hugging Face time (ms)",
        result.reference.tokens.map((token) => formatDecimal(token.timeMs)),
      );
    }
    if (result.discrepancies) {
      section("ONNX / Transformers logit comparison");
      for (const [label, key, format = String] of [
        ["Maximum absolute distance", "maxAbsolute", formatDecimal],
        ["Relative error", "relative", formatDecimal],
        ["L2 distance", "l2", formatDecimal],
        ["Coefficients Δ < 0.01", "below001"],
        ["Coefficients Δ < 0.1", "below01"],
        ["Coefficients Δ < 1", "below1"],
        ["Coefficients compared", "total"],
      ])
        row(
          label,
          result.discrepancies.map((value) => format(value[key])),
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
      if (result.reference.loadMs != null)
        referenceTiming.textContent += ` Loading: ${formatDecimal(result.reference.loadMs)} ms; warmup (1 input token): ${formatDecimal(result.reference.warmupMs)} ms.`;
    }
    const profile = document.createElement("p");
    profile.hidden = !result.profileDirectory;
    profile.textContent = result.profileDirectory
      ? `Profiles: ${result.profileDirectory}. Unload GenAI to finalize its profile.`
      : "";
    content.replaceChildren(
      prompt,
      limitations,
      timingSummary,
      referenceTiming,
      profile,
      table,
    );
  };
  const run = (validation) => {
    if (busy || !available) return;
    const validateRequest = validation || validateRun.checked;
    if (
      validateRequest &&
      validationRequiresReference &&
      !referenceSource.value.trim()
    ) {
      status.textContent =
        "Enter the original Hugging Face model ID or local checkpoint before validation. No generation has started.";
      referenceSource.focus();
      referenceSource.reportValidity();
      return;
    }
    if (!validation) {
      const sessionSettings = readSettings();
      if (!sessionSettings) return;
      options = {
        engine: byId("prompt-engine").value,
        prompt: byId("prompt-text").value,
        maxTokens: Number(byId("prompt-max-tokens").value),
        keepLoaded: byId("prompt-keep-loaded").checked,
        sessionSettings,
      };
      showLifecycle();
      result = { ...options, tokens: [] };
      render();
      validate.hidden = true;
    }
    streaming = !validation;
    setBusy(true);
    showProgress("");
    status.textContent = validation
      ? "Validating with Hugging Face…"
      : validateRequest
        ? "Checking the Hugging Face reference before generation…"
        : "Running prompt…";
    postMessage({
      type: "run-prompt",
      requestId: ++requestId,
      reuseResult: validation,
      options: {
        ...options,
        validate: validateRequest,
        ...(validateRequest && referenceSource.value.trim()
          ? { referenceSource: referenceSource.value.trim() }
          : {}),
      },
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
    if (result?.lifecycle) result.lifecycle.retained = false;
    showLifecycle(result?.lifecycle);
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
          "prompt-lifecycle",
          "prompt-released",
        ].includes(message?.type)
      )
        return false;
      if (message.requestId !== requestId) return true;
      if (message.type === "prompt-info") {
        available = message.info.available;
        validationRequiresReference =
          !!message.info.validationRequiresReference;
        const engine = byId("prompt-engine");
        for (const option of engine.options)
          option.disabled = !message.info.engines.includes(option.value);
        engine.value = message.info.engines.includes("onnxruntime-genai")
          ? "onnxruntime-genai"
          : (message.info.engines[0] ?? "onnx-light");
        showSettingsEngine();
        const light = byId("prompt-settings-light");
        const sdkLoader =
          light.querySelector('[data-setting="sdkLoader"]').value === "true";
        for (const field of light.querySelectorAll("[data-loading]"))
          field.disabled = field.hasAttribute("data-sdk-loader") && !sdkLoader;
        setBusy(false);
        status.textContent = available
          ? "Ready. ONNX engines and tokenization use native C++ libraries; Python can locate SDK libraries and run Hugging Face validation."
          : message.info.reason;
      } else if (message.type === "prompt-lifecycle") {
        if (busy && streaming) {
          result.lifecycle = message.lifecycle;
          showLifecycle(message.lifecycle);
        }
      } else if (message.type === "prompt-released") {
        if (result?.lifecycle) {
          result.lifecycle.retained = false;
          showLifecycle(result.lifecycle);
        }
        status.textContent = "Model unloaded.";
      } else if (message.type === "prompt-token") {
        if (!busy || !streaming || message.index !== result.tokens.length)
          return true;
        result.tokens.push(message.token);
        const section = content.querySelector(".prompt-token-section");
        updateSectionHeading(
          section,
          section.cells[0].textContent,
          result.tokens.length + 1,
        );
        for (const { row, value } of liveRows) {
          const cell = document.createElement("td");
          cell.textContent = String(value(message.token, message.index));
          row.append(cell);
        }
        recordTiming(message.token, message.index);
        updateTiming();
        status.textContent = `Generated ${result.tokens.length}/${options.maxTokens} tokens…`;
      } else if (message.type === "prompt-progress") {
        showProgress(message.output);
      } else if (message.type === "prompt-error") {
        streaming = false;
        setBusy(false);
        if (result?.lifecycle) result.lifecycle.retained = false;
        showLifecycle(result?.lifecycle);
        status.textContent = message.error.split("\n")[0];
        showProgress(message.error);
      } else {
        streaming = false;
        result = message.result;
        setBusy(false);
        showLifecycle(result.lifecycle);
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
