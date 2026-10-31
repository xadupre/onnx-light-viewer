/** Reusable templates are independent of the displayed model and graph. */
function createPatternPanel({ byId, requestAction, postMessage }) {
  let patterns = new Map();
  let revision = 0;
  let editRevision = 0;
  let request = 0;
  let pending;
  const picker = byId("pattern-library-picker");
  const text = byId("pattern-library-text");
  const replacement = byId("pattern-library-replacement");
  const status = byId("pattern-library-status");

  function dirty() {
    const pattern = patterns.get(picker.value);
    return (
      !!pattern &&
      (text.value !== pattern.text || replacement.value !== pattern.replacement)
    );
  }

  function controls() {
    const selected = patterns.has(picker.value);
    picker.disabled = !!pending || dirty() || !patterns.size;
    text.disabled = replacement.disabled = !!pending || !selected;
    byId("update-library-pattern").disabled = !!pending || !dirty();
    byId("revert-library-pattern").disabled = !!pending || !dirty();
    byId("save-pattern-library").disabled =
      !!pending || dirty() || !patterns.size;
    byId("restore-pattern-library").disabled = !!pending;
    byId("copy-pattern-library-text").disabled =
      !!pending || !selected || !text.value;
    byId("copy-pattern-library-replacement").disabled =
      !!pending || !selected || !replacement.value;
  }

  function select() {
    const pattern = patterns.get(picker.value);
    text.value = pattern?.text ?? "";
    replacement.value = pattern?.replacement ?? "";
    editRevision = revision;
    controls();
  }

  function showLibrary(data, preserveEdits = true) {
    const previous = picker.value;
    const keepDraft = preserveEdits && dirty();
    patterns = new Map(data.patterns.map((pattern) => [pattern.name, pattern]));
    revision = data.revision;
    picker.replaceChildren();
    for (const name of patterns.keys()) {
      const option = document.createElement("option");
      option.value = option.textContent = name;
      picker.append(option);
    }
    picker.value = patterns.has(previous)
      ? previous
      : (patterns.keys().next().value ?? "");
    byId("pattern-library-empty").hidden = !!patterns.size;
    byId("pattern-library-editor").hidden = !patterns.size;
    if (!keepDraft || !dirty()) select();
    else controls();
  }

  function run(type, action, extra = {}) {
    pending = { type, requestId: ++request };
    controls();
    status.textContent =
      type === "update-library-pattern" ? "Applying edits..." : `${action}...`;
    requestAction(type, action, { requestId: request, ...extra });
  }

  picker.addEventListener("change", () => {
    select();
    status.textContent = "";
  });
  for (const field of [text, replacement])
    field.addEventListener("input", () => {
      controls();
      status.textContent = dirty()
        ? "Unapplied edits. Apply or revert before saving or selecting another pattern."
        : "";
    });
  byId("revert-library-pattern").addEventListener("click", () => {
    select();
    status.textContent = "Edits reverted.";
  });
  byId("update-library-pattern").addEventListener("click", () => {
    if (pending || !dirty()) return;
    run("update-library-pattern", "Edit library pattern", {
      name: picker.value,
      text: text.value,
      replacement: replacement.value,
      libraryRevision: editRevision,
    });
  });
  byId("save-pattern-library").addEventListener("click", () => {
    if (pending || dirty() || !patterns.size) return;
    run("save-pattern-library", "Save pattern library");
  });
  byId("restore-pattern-library").addEventListener("click", () => {
    if (pending) return;
    run("restore-pattern-library", "Restore pattern library", {
      dirty: dirty(),
    });
  });
  for (const format of ["text", "replacement"])
    byId(`copy-pattern-library-${format}`).addEventListener("click", () => {
      if (pending || !patterns.has(picker.value)) return;
      pending = { type: "copy-function-pattern", format };
      controls();
      status.textContent = "Copying...";
      postMessage({
        type: "copy-function-pattern",
        name: picker.value,
        format,
        text: format === "text" ? text.value : replacement.value,
      });
    });

  controls();
  return {
    handleMessage(data) {
      if (data?.type === "pattern-library") {
        showLibrary(data);
        if (data.invalidateFusion && dirty())
          status.textContent =
            "The library changed. Revert edits before editing again.";
        return true;
      }
      if (
        data?.type === "function-pattern-copied" ||
        data?.type === "function-pattern-copy-error"
      ) {
        if (
          pending?.type !== "copy-function-pattern" ||
          data.name !== picker.value ||
          data.format !== pending.format
        )
          return false;
        pending = undefined;
        status.textContent = data.error
          ? `Could not copy pattern: ${data.error}`
          : "Copied.";
        controls();
        return true;
      }
      if (data?.type !== "pattern-library-result") return false;
      if (
        !pending ||
        data.requestId !== pending.requestId ||
        data.operation !== pending.type
      )
        return true;
      pending = undefined;
      if (data.error)
        status.textContent = `Could not complete action: ${data.error}`;
      else if (data.cancelled) status.textContent = "Cancelled.";
      else {
        showLibrary(data, false);
        status.textContent =
          data.operation === "update-library-pattern"
            ? "Pattern updated for all models."
            : data.operation === "save-pattern-library"
              ? "Pattern library saved."
              : "Patterns restored and available in all models.";
      }
      controls();
      return true;
    },
  };
}
