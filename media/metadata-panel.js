function createMetadataPanel({ byId, tensorDescription }) {
  function metadataEntries(parent, entries) {
    parent.replaceChildren();
    if (parent.parentElement.tagName === "SECTION")
      parent.parentElement.hidden = entries.length === 0;
    const head = parent.createTHead();
    const header = head.insertRow();
    for (const label of ["Key", "Value"]) {
      const cell = document.createElement("th");
      cell.scope = "col";
      cell.textContent = label;
      header.append(cell);
    }
    const body = parent.createTBody();
    for (const { key, value } of entries) {
      const row = body.insertRow();
      const term = document.createElement("th");
      term.scope = "row";
      const description = document.createElement("td");
      term.textContent = key;
      description.textContent = value;
      row.append(term, description);
    }
  }

  function metadataNames(id, names) {
    byId(`${id}-section`).hidden = names.length === 0;
    byId(id).replaceChildren(
      ...names.map((name) => {
        const item = document.createElement("li");
        item.textContent = name;
        return item;
      }),
    );
  }

  function render(
    view,
    { visible, metadata, nodeMetadata, valueMetadata, scope },
  ) {
    if (!visible || !metadata) return;
    const node = view?.node;
    const port = view?.port;
    const info = node
      ? nodeMetadata.get(node.id)
      : valueMetadata.get(port?.name);
    metadataEntries(
      byId("metadata-analysis"),
      node || port
        ? [
            { key: "Name", value: node?.name ?? port?.name },
            { key: "Tag", value: info?.tag ?? "Unknown" },
            {
              key: "Constant",
              value: info ? (info.constant ? "Yes" : "No") : "Unknown",
            },
          ]
        : [{ key: scope.kind, value: scope.name }],
    );
    metadataNames("metadata-releases", node ? info.releaseAfter : []);
    metadataNames("metadata-last-use", node ? info.lastUse : []);
    metadataNames("metadata-unused", node ? info.unusedOutputs : []);
    metadataEntries(byId("metadata-node"), node ? info.entries : []);
    const names = new Set(
      node
        ? [...node.inputs, ...(node.captures ?? []), ...node.outputs]
        : port
          ? [port.name]
          : [...scope.inputs, ...scope.outputs],
    );
    byId("metadata-value-count").textContent = names.size;
    byId("metadata-values").parentElement.hidden = names.size === 0;
    byId("metadata-values").replaceChildren(
      ...[...names].map((name) => {
        const value = valueMetadata.get(name);
        const item = document.createElement("li");
        const heading = document.createElement("strong");
        heading.className = "metadata-value-name";
        heading.textContent = name;
        const badges = document.createElement("div");
        badges.className = "metadata-badges";
        for (const text of [value?.tag, value?.constant ? "constant" : null]) {
          if (!text) continue;
          const badge = document.createElement("span");
          badge.textContent = text;
          badges.append(badge);
        }
        const description = document.createElement("div");
        description.className = "metadata-value-type";
        description.textContent = tensorDescription(name);
        item.append(heading, badges, description);
        if (value?.entries.length) {
          const heading = document.createElement("strong");
          heading.textContent = "Stored metadata";
          const entries = document.createElement("table");
          entries.setAttribute("aria-label", "Stored tensor metadata");
          item.append(heading, entries);
          metadataEntries(entries, value.entries);
        }
        return item;
      }),
    );
  }

  function accept(metadata) {
    byId("metadata-content").hidden = false;
    metadataEntries(byId("metadata-model"), metadata.model);
    metadataEntries(byId("metadata-graph"), metadata.graph);
  }
  function reset() {
    byId("metadata-content").hidden = true;
  }
  function setPending(pending) {
    byId("metadata-panel").setAttribute("aria-busy", String(pending));
  }

  return { render, accept, reset, setPending };
}
