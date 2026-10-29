function createTensorListPanel({ byId, openStatistics }) {
  const body = byId("tensor-list-rows");
  const search = byId("tensor-list-search");
  const typeFilter = byId("tensor-list-type");
  const table = body.closest("table");
  const headers = [...table.querySelectorAll("th")];
  const columns = document.createElement("colgroup");
  const minimumColumnWidth = 64;
  let columnWidths;
  let columnDrag;
  for (const header of headers) columns.append(document.createElement("col"));
  table.prepend(columns);

  function applyColumnWidths() {
    const total = columnWidths.reduce((sum, width) => sum + width, 0);
    table.style.tableLayout = "fixed";
    table.style.width = `${total}px`;
    for (const [index, width] of columnWidths.entries()) {
      columns.children[index].style.width = `${width}px`;
      const handle = headers[index].querySelector(".tensor-column-resizer");
      handle.setAttribute("aria-valuenow", width);
      handle.setAttribute("aria-valuemax", total);
    }
  }

  function initializeColumnWidths() {
    if (columnWidths) return;
    columnWidths = headers.map((header) =>
      Math.max(minimumColumnWidth, header.getBoundingClientRect().width),
    );
    applyColumnWidths();
  }

  function endColumnDrag(event) {
    if (!columnDrag || (event && event.pointerId !== columnDrag.pointerId))
      return;
    const { handle, pointerId } = columnDrag;
    columnDrag = undefined;
    document.body.classList.remove("resizing-tensor-column");
    if (handle.hasPointerCapture(pointerId))
      handle.releasePointerCapture(pointerId);
  }

  for (const [index, header] of headers.entries()) {
    const handle = document.createElement("div");
    handle.className = "tensor-column-resizer";
    handle.tabIndex = 0;
    handle.setAttribute("role", "separator");
    handle.setAttribute("aria-orientation", "vertical");
    handle.setAttribute("aria-label", `Resize ${header.textContent} column`);
    handle.setAttribute("aria-valuemin", minimumColumnWidth);
    handle.title = "Drag to resize, or use Left and Right arrow keys";
    header.append(handle);
    handle.addEventListener("focus", initializeColumnWidths);
    handle.addEventListener("pointerdown", (event) => {
      if (event.button !== 0 || columnDrag) return;
      event.preventDefault();
      initializeColumnWidths();
      columnDrag = {
        handle,
        pointerId: event.pointerId,
        index,
        x: event.clientX,
        width: columnWidths[index],
      };
      handle.setPointerCapture(event.pointerId);
      handle.focus();
      document.body.classList.add("resizing-tensor-column");
    });
    handle.addEventListener("pointermove", (event) => {
      if (columnDrag?.pointerId !== event.pointerId) return;
      columnWidths[columnDrag.index] = Math.max(
        minimumColumnWidth,
        columnDrag.width + event.clientX - columnDrag.x,
      );
      applyColumnWidths();
    });
    for (const type of ["pointerup", "pointercancel", "lostpointercapture"])
      handle.addEventListener(type, endColumnDrag);
    handle.addEventListener("keydown", (event) => {
      if (!["ArrowLeft", "ArrowRight", "Home"].includes(event.key)) return;
      event.preventDefault();
      initializeColumnWidths();
      endColumnDrag();
      columnWidths[index] =
        event.key === "Home"
          ? minimumColumnWidth
          : Math.max(
              minimumColumnWidth,
              columnWidths[index] + (event.key === "ArrowRight" ? 16 : -16),
            );
      applyColumnWidths();
    });
  }
  window.addEventListener("blur", () => endColumnDrag());
  const pageSize = 100;
  let graph;
  let records;
  let storedNames = new Set();
  let initializers = new Map();
  let tensors = new Map();
  let highlighted = new Set();
  let page = 0;
  let busy = false;
  let distributionsDirty = true;

  function tensorType(entry) {
    return entry.information?.type ?? tensors.get(entry.name)?.type ?? "?";
  }

  function renderDistributions() {
    if (!distributionsDirty) return;
    distributionsDirty = false;
    const container = byId("tensor-distribution-groups");
    container.replaceChildren();
    for (const [kind, title] of [
      ["dense", "Dense tensors"],
      ["sparse", "Sparse tensors"],
      ["encoded", "Encoded values"],
    ]) {
      const entries = records.filter((entry) => {
        const category =
          entry.information?.distribution?.kind ??
          (entry.categories.has("Sparse initializer")
            ? "sparse"
            : entry.categories.has("Encoded initializer")
              ? "encoded"
              : "dense");
        return category === kind;
      });
      const section = document.createElement("section");
      section.dataset.kind = kind;
      const heading = document.createElement("h3");
      heading.textContent = `${title} (${entries.length.toLocaleString("en-US")})`;
      section.append(heading);
      container.append(section);
      if (!entries.length) continue;
      for (const groupBy of ["type", "shape"]) {
        const groups = new Map();
        for (const entry of entries) {
          const information = entry.information ?? tensors.get(entry.name);
          const shape = information?.shape;
          const key =
            groupBy === "type"
              ? tensorType(entry)
              : JSON.stringify(shape ?? null);
          if (!groups.has(key))
            groups.set(key, {
              label:
                groupBy === "type"
                  ? key
                  : Array.isArray(shape)
                    ? `[${shape.join(", ")}]`
                    : "Unknown",
              count: 0,
              elementCount: 0n,
              logicalSizeBytes: 0n,
              unknownElements: 0,
              unknownBytes: 0,
            });
          const group = groups.get(key);
          group.count++;
          const distribution = entry.information?.distribution;
          for (const [field, unknown] of [
            ["elementCount", "unknownElements"],
            ["logicalSizeBytes", "unknownBytes"],
          ]) {
            if (distribution?.[field] == null) group[unknown]++;
            else group[field] += BigInt(distribution[field]);
          }
        }
        const rows = [...groups.values()].sort(
          (left, right) =>
            right.count - left.count || left.label.localeCompare(right.label),
        );
        const details = document.createElement("details");
        details.open = true;
        const summary = document.createElement("summary");
        summary.textContent = `By ${groupBy} (${rows.length.toLocaleString("en-US")} ${rows.length === 1 ? "group" : "groups"})`;
        const wrapper = document.createElement("div");
        wrapper.className = "tensor-list-table";
        const table = document.createElement("table");
        table.dataset.groupBy = groupBy;
        table.setAttribute("aria-label", `${title} by ${groupBy}`);
        const header = table.createTHead().insertRow();
        for (const label of [
          groupBy === "type" ? "Type" : "Shape",
          "Tensors",
          "Elements",
          "Logical size (bytes)",
        ]) {
          const cell = document.createElement("th");
          cell.scope = "col";
          cell.textContent = label;
          header.append(cell);
        }
        const body = table.createTBody();
        const previous = document.createElement("button");
        previous.type = "button";
        previous.textContent = "Previous";
        const next = document.createElement("button");
        next.type = "button";
        next.textContent = "Next";
        const range = document.createElement("span");
        range.setAttribute("role", "status");
        const navigation = document.createElement("nav");
        navigation.className = "tensor-distribution-pages";
        navigation.setAttribute("aria-label", `${title} by ${groupBy} pages`);
        navigation.append(previous, range, next);
        let groupPage = 0;
        const value = (known, unknown) =>
          unknown
            ? known
              ? `${known.toLocaleString("en-US")} + ${unknown} unknown`
              : `Unknown (${unknown} tensor${unknown === 1 ? "" : "s"})`
            : known.toLocaleString("en-US");
        const renderPage = () => {
          body.replaceChildren();
          const start = groupPage * pageSize;
          for (const row of rows.slice(start, start + pageSize)) {
            const element = body.insertRow();
            for (const text of [
              row.label,
              row.count.toLocaleString("en-US"),
              value(row.elementCount, row.unknownElements),
              value(row.logicalSizeBytes, row.unknownBytes),
            ])
              element.insertCell().textContent = text;
          }
          previous.disabled = groupPage === 0;
          next.disabled = start + pageSize >= rows.length;
          range.textContent = `${start + 1}–${Math.min(start + pageSize, rows.length)} of ${rows.length}`;
        };
        previous.addEventListener("click", () => {
          groupPage--;
          renderPage();
        });
        next.addEventListener("click", () => {
          groupPage++;
          renderPage();
        });
        renderPage();
        wrapper.append(table);
        details.append(summary, wrapper, navigation);
        section.append(details);
      }
    }
  }

  function updateTypeFilter() {
    const selected = typeFilter.value;
    const types = [...new Set((records ?? []).map(tensorType))].sort();
    typeFilter.replaceChildren(
      ...["", ...types].map((type) => {
        const option = document.createElement("option");
        option.value = type;
        option.textContent = type || "All types";
        return option;
      }),
    );
    typeFilter.value = types.includes(selected) ? selected : "";
  }

  function build() {
    const entries = new Map();
    function add(name, category, information) {
      if (!name) return;
      if (!entries.has(name))
        entries.set(name, {
          name,
          categories: new Set(),
          consumers: new Map(),
        });
      const entry = entries.get(name);
      if (category) entry.categories.add(category);
      if (information) entry.information = information;
      return entry;
    }
    for (const [field, category] of [
      ["initializers", "Initializer"],
      ["sparseInitializers", "Sparse initializer"],
      ["encodedInitializers", "Encoded initializer"],
    ])
      for (const tensor of graph[field] ?? [])
        add(tensor.name, category, tensor);
    for (const node of graph.nodes) {
      const constant =
        node.opType === "Constant" &&
        (!node.domain || node.domain === "ai.onnx") &&
        !node.function &&
        !node.pipelineModel;
      if (constant)
        for (const name of node.outputs ?? []) {
          const information = node.storedOutputs?.find(
            (tensor) => tensor.name === name,
          );
          add(name, "Constant", information);
        }
      for (const tensor of node.parameters ?? []) {
        const entry = add(tensor.name, undefined, tensor);
        if (!entry.categories.size)
          entry.categories.add("Captured initializer");
      }
    }
    for (const node of graph.nodes) {
      for (const name of new Set([
        ...(node.inputs ?? []),
        ...(node.captures ?? []),
      ])) {
        const entry = entries.get(name);
        if (entry)
          entry.consumers.set(
            node.id,
            `${node.domain && node.domain !== "ai.onnx" ? `${node.domain}::` : ""}${node.opType} (${node.name || node.id})`,
          );
      }
    }
    records = [...entries.values()];
    storedNames = new Set(entries.keys());
    initializers = new Map(
      records
        .filter((entry) => entry.information)
        .map((entry) => [entry.name, entry.information]),
    );
    updateTypeFilter();
  }

  function render(focusHighlighted = false) {
    body.replaceChildren();
    if (!graph) {
      byId("tensor-list-status").textContent =
        "Open a model to browse tensors.";
      byId("tensor-list-previous").disabled = true;
      byId("tensor-list-next").disabled = true;
      return;
    }
    if (!records) build();
    renderDistributions();
    const query = search.value.trim().toLowerCase();
    const matching = records.filter(
      (entry) =>
        (!typeFilter.value || tensorType(entry) === typeFilter.value) &&
        [entry.name, ...entry.categories, ...entry.consumers.values()].some(
          (value) => value.toLowerCase().includes(query),
        ),
    );
    if (focusHighlighted) {
      const first = matching.findIndex((entry) => highlighted.has(entry.name));
      if (first >= 0) page = Math.floor(first / pageSize);
    }
    page = Math.min(
      page,
      Math.max(0, Math.ceil(matching.length / pageSize) - 1),
    );
    const start = page * pageSize;
    for (const entry of matching.slice(start, start + pageSize)) {
      const row = document.createElement("tr");
      row.dataset.name = entry.name;
      row.classList.toggle("tensor-highlighted", highlighted.has(entry.name));
      const information = entry.information ?? tensors.get(entry.name);
      const shape = information
        ? information.shape === undefined
          ? information.type
            ? "—"
            : "?"
          : information.shape === null
            ? "?"
            : `[${information.shape.join(", ")}]`
        : "?";
      for (const value of [
        entry.name,
        [...entry.categories].join(", "),
        tensorType(entry),
        shape,
        [...entry.consumers.values()].join("\n") || "—",
      ]) {
        const cell = document.createElement("td");
        cell.textContent = value;
        row.append(cell);
      }
      const action = document.createElement("td");
      const button = document.createElement("button");
      button.type = "button";
      button.className = "icon-button tensor-list-statistics";
      button.title = `Inspect values or statistics for ${entry.name}`;
      button.setAttribute("aria-label", button.title);
      button.setAttribute("aria-haspopup", "dialog");
      button.disabled = busy;
      button.append(byId("tensors-tab").querySelector("svg").cloneNode(true));
      button.addEventListener("click", () =>
        openStatistics(
          matching.map((entry) => entry.name),
          entry.name,
        ),
      );
      action.append(button);
      row.append(action);
      body.append(row);
    }
    const count = records.filter((entry) => highlighted.has(entry.name)).length;
    byId("tensor-list-status").textContent =
      `${matching.length ? start + 1 : 0}–${Math.min(start + pageSize, matching.length)} of ${matching.length} tensors in ${graph.name}.${count ? ` ${count} highlighted for the selected operator or tensor.` : ""}`;
    byId("tensor-list-previous").disabled = page === 0;
    byId("tensor-list-next").disabled = start + pageSize >= matching.length;
  }

  search.addEventListener("input", () => {
    page = 0;
    render();
  });
  typeFilter.addEventListener("change", () => {
    page = 0;
    render();
  });
  for (const [name, delta] of [
    ["previous", -1],
    ["next", 1],
  ])
    byId(`tensor-list-${name}`).addEventListener("click", () => {
      page += delta;
      render();
    });

  function storedInputs(node) {
    if (!records) build();
    return [
      ...new Set([...(node.inputs ?? []), ...(node.captures ?? [])]),
    ].filter((name) => storedNames.has(name));
  }

  return {
    storedInputs,
    storedTensorNames() {
      if (!records) build();
      return [...storedNames];
    },
    hasStoredTensor(name) {
      if (!records) build();
      return storedNames.has(name);
    },
    nodeInitializers(node) {
      const parameters = [...(node.parameters ?? [])];
      const names = new Set(parameters.map((parameter) => parameter.name));
      for (const name of storedInputs(node))
        if (!names.has(name) && initializers.has(name))
          parameters.push(initializers.get(name));
      return parameters;
    },
    setGraph(value) {
      graph = value;
      records = undefined;
      tensors = new Map(
        (graph?.tensors ?? []).map((tensor) => [tensor.name, tensor]),
      );
      highlighted.clear();
      distributionsDirty = true;
      byId("tensor-distribution-groups").replaceChildren();
      search.value = "";
      typeFilter.value = "";
      updateTypeFilter();
      page = 0;
      if (!byId("tensors-view").hidden) render();
      else body.replaceChildren();
    },
    updateTypes(values) {
      tensors = new Map(values.map((tensor) => [tensor.name, tensor]));
      distributionsDirty = true;
      updateTypeFilter();
      if (!byId("tensors-view").hidden) render();
    },
    setBusy(value) {
      busy = value;
      for (const button of body.querySelectorAll("button"))
        button.disabled = busy;
    },
    open(names = []) {
      highlighted = new Set(names);
      render(true);
    },
  };
}
