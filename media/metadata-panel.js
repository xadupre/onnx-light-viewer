function createMetadataPanel({ byId, tensorDescription }) {
  function metadataEntries(parent, entries) {
    parent.replaceChildren();
    if (parent.parentElement.tagName === "DETAILS")
      parent.parentElement.hidden = entries.length === 0;
    for (const { key, value } of entries) {
      const term = document.createElement("dt");
      const description = document.createElement("dd");
      term.textContent = key;
      description.textContent = value;
      parent.append(term, description);
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

  function render(view, { visible, metadata, nodeMetadata, valueMetadata }) {
    if (!visible || !metadata) return;
    const node = view?.node;
    const port = view?.port;
    const info = node
      ? nodeMetadata.get(node.id)
      : valueMetadata.get(port?.name);
    metadataEntries(byId("metadata-analysis"), [
      { key: "Name", value: node?.name ?? port?.name ?? "No selection" },
      { key: "Tag", value: info?.tag ?? "Unknown" },
      {
        key: "Constant",
        value: info ? (info.constant ? "Yes" : "No") : "Unknown",
      },
    ]);
    metadataNames("metadata-releases", node ? info.releaseAfter : []);
    metadataNames("metadata-last-use", node ? info.lastUse : []);
    metadataNames("metadata-unused", node ? info.unusedOutputs : []);
    metadataEntries(byId("metadata-node"), node ? info.entries : []);
    const names = new Set(
      node
        ? [...node.inputs, ...(node.captures ?? []), ...node.outputs]
        : port
          ? [port.name]
          : [],
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
          const details = document.createElement("details");
          const summary = document.createElement("summary");
          summary.textContent = "Stored metadata";
          const entries = document.createElement("dl");
          details.append(summary, entries);
          metadataEntries(entries, value.entries);
          item.append(details);
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
