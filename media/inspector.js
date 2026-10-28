/* global markdownit */
function createInspector({
  byId,
  tensorDescription,
  subgraphLabel,
  navigate,
  openTensorStatistics,
  hasStoredTensor,
  nodeInitializers,
}) {
  function statisticsButton(name) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "icon-button tensor-statistics-button";
    button.append(byId("tensors-tab").querySelector("svg").cloneNode(true));
    button.title = `Inspect values or statistics for ${name}`;
    button.setAttribute("aria-label", button.title);
    button.setAttribute("aria-haspopup", "dialog");
    button.setAttribute("aria-controls", "tensor-statistics-dialog");
    button.addEventListener("click", () => openTensorStatistics(name));
    button.disabled = !name;
    return button;
  }
  function statisticsHeading(name, label) {
    const heading = document.createElement("div");
    heading.className = "tensor-statistics-heading";
    heading.append(label);
    if (hasStoredTensor(name)) heading.append(statisticsButton(name));
    return heading;
  }
  const documentationMarkdown = markdownit({ html: false });
  documentationMarkdown.validateLink = (url) => /^https?:\/\//i.test(url);
  documentationMarkdown.renderer.rules.image = (tokens, index) =>
    documentationMarkdown.utils.escapeHtml(tokens[index].content);
  let inspectedNode;
  let initializerGraph;
  function resetDocumentation() {
    if (byId("operator-doc").open) byId("operator-doc").close();
    byId("doc-title").textContent = "";
    byId("doc-version").textContent = "";
    byId("doc-body").textContent = "";
  }

  function inspect(view, scope) {
    const node = view?.node;
    const port = view?.port;
    const model = node?.pipelineModel;
    byId("node-title").textContent =
      node?.opType ??
      (port
        ? port.output
          ? "Graph output"
          : "Graph input"
        : (scope?.kind ?? "No model loaded"));
    byId("node-description").textContent = model
      ? `${node.name} model in pipeline ${scope.name}. Contains ${model.nodeCount} operator${model.nodeCount === 1 ? "" : "s"}. Use the model link below to open its graph.`
      : node
        ? `${node.opType} ${node.function ? "local function" : "operator"}`
        : port
          ? `Graph ${port.output ? "output" : "input"} tensor.`
          : scope?.kind === "Pipeline"
            ? "Pipeline overview: each box is a contained model, not an operator. Select a model to inspect its imports and ports; use its link to open its graph."
            : scope
              ? `${scope.kind} scope. Select a node or tensor to inspect its details.`
              : "Open an ONNX model to explore its connections.";
    byId("open-function").hidden = !node?.function;
    byId("open-function").textContent = node?.function
      ? `Open function: ${node.function.name}`
      : "Open function";
    byId("node-name-label").textContent = model
      ? "MODEL NAME"
      : node
        ? "NODE NAME"
        : port
          ? "TENSOR NAME"
          : `${scope?.kind.toUpperCase() ?? "MODEL"} NAME`;
    byId("node-name").textContent =
      model?.name ?? node?.name ?? port?.name ?? scope?.name ?? "—";
    const opsets =
      model?.opsets ?? (!node && !port ? (scope?.opsets ?? []) : []);
    const domains = [...new Set(opsets.map((opset) => opset.domain))].join(
      ", ",
    );
    byId("node-domain").textContent =
      (model ? domains : (node?.domain ?? domains)) || "—";
    byId("node-opset").replaceChildren();
    if (opsets.length) {
      for (const { domain, version } of opsets) {
        const entry = document.createElement("div");
        entry.textContent = `${domain}: ${version}`;
        byId("node-opset").append(entry);
      }
    } else {
      byId("node-opset").textContent = node?.opset ?? "—";
    }
    if (inspectedNode !== node) {
      resetDocumentation();
      inspectedNode = node;
    }
    byId("show-doc").disabled = !node?.schema?.doc;
    byId("show-doc").title = node?.schema?.doc
      ? "Documentation from onnx-light LightOpSchema"
      : "No LightOpSchema documentation available for this selection.";
    const attributes = node?.attributes ?? [];
    const names = new Set(attributes.map((attribute) => attribute.name));
    const defaults = (node?.schema?.defaults ?? []).filter(
      (attribute) => !names.has(attribute.name),
    );
    parameterList(
      "attributes",
      [...attributes, ...defaults],
      "attribute-count",
    );
    initializerList(node ? nodeInitializers(node) : []);
    byId("node-parameters").hidden = !node;
    const inspectFunctions = !node && !port && scope?.kind === "Model";
    const functions = inspectFunctions ? (scope.localFunctions ?? []) : [];
    byId("scope-functions").hidden = !inspectFunctions;
    byId("function-count").textContent = functions.length;
    byId("local-functions").replaceChildren(
      ...functions.map((definition) => {
        const item = document.createElement("li");
        const button = document.createElement("button");
        button.className = "icon-button";
        button.type = "button";
        button.textContent = `${definition.name} (${definition.nodeCount} nodes)`;
        button.addEventListener("click", () => navigate(definition.path));
        item.append(button);
        return item;
      }),
    );
    const inspectInitializers = scope && scope.kind !== "Pipeline";
    byId("scope-initializers").hidden = !inspectInitializers;
    if (initializerGraph !== scope?.graph) {
      initializerGraph = scope?.graph;
      for (const [field, id] of [
        ["initializers", "dense"],
        ["sparseInitializers", "sparse"],
        ["encodedInitializers", "encoded"],
      ]) {
        initializerList(
          inspectInitializers ? (scope[field] ?? []) : [],
          `scope-${id}-initializers`,
          `${id}-initializer-count`,
        );
      }
    }
    const constants = scope?.constants ?? [];
    byId("node-constants").hidden = constants.length === 0;
    byId("constant-count").textContent = constants.length;
    byId("constant-nodes").replaceChildren(
      ...constants.map((constant) => {
        const item = document.createElement("li");
        const details = document.createElement("details");
        const summary = document.createElement("summary");
        summary.textContent = constant.name || constant.outputs.join(", ");
        const outputs = document.createElement("div");
        outputs.className = "tensor-info";
        for (const name of constant.outputs) {
          const label = document.createElement("span");
          label.textContent = `${name}: ${tensorDescription(name)}`;
          outputs.append(statisticsHeading(name, label));
        }
        const attributes = document.createElement("ul");
        attributes.append(...(constant.attributes ?? []).map(attributeItem));
        details.append(summary, outputs, attributes);
        item.append(details);
        return item;
      }),
    );
    for (const direction of ["inputs", "outputs"]) {
      const names =
        node?.[direction] ??
        (port && direction === (port.output ? "inputs" : "outputs")
          ? [port.name]
          : !node && !port
            ? (scope?.[direction] ?? [])
            : []);
      byId(`${direction === "inputs" ? "input" : "output"}-count`).textContent =
        names.length;
      byId(`node-${direction}`).replaceChildren(
        ...names.map((name) => {
          const item = document.createElement("li");
          const tensorName = document.createElement("span");
          tensorName.className = "tensor-name";
          tensorName.textContent = name;
          const information = document.createElement("span");
          information.className = "tensor-info";
          information.textContent = tensorDescription(name);
          item.append(tensorName, information);
          return item;
        }),
      );
    }
    const captures = node?.captures ?? [];
    byId("node-captures").hidden = captures.length === 0;
    byId("captured-inputs").replaceChildren(
      ...captures.map((name) => {
        const item = document.createElement("li");
        const label = document.createElement("span");
        label.textContent = `${name}: ${tensorDescription(name)}`;
        item.append(statisticsHeading(name, label));
        return item;
      }),
    );
  }

  function initializerList(
    parameters,
    id = "node-initializers",
    countId = "parameter-count",
  ) {
    byId(countId).textContent = parameters.length;
    byId(id).replaceChildren(
      ...parameters.map((parameter) => {
        const item = document.createElement("li");
        const heading = document.createElement("strong");
        heading.textContent = `${parameter.name} (${parameter.typeDetails?.kind ?? parameter.type})`;
        const shape = document.createElement("div");
        shape.className = "tensor-info";
        shape.textContent = `Shape: [${parameter.shape?.join(", ") ?? "?"}]`;
        item.append(statisticsHeading(parameter.name, heading));
        if (parameter.shape !== undefined) item.append(shape);
        if (parameter.typeDetails) {
          item.append(typeTable("Logical type", parameter.typeDetails));
        }
        if (parameter.encodingDetails) {
          item.append(typeTable("Encoding", parameter.encodingDetails));
        } else if (parameter.encoding !== undefined) {
          const encoding = document.createElement("pre");
          encoding.className = "initializer-values";
          encoding.textContent = `Encoding: ${parameter.encoding}`;
          item.append(encoding);
        }
        if (parameter.value !== undefined) {
          const values = document.createElement("pre");
          values.className = "initializer-values";
          values.textContent = `Values: ${parameter.value}`;
          item.append(values);
        }
        for (const [label, storage] of [
          [parameter.indicesStorage ? "Values" : "Storage", parameter.storage],
          ["Indices", parameter.indicesStorage],
        ]) {
          if (!storage) continue;
          const description = document.createElement("div");
          description.className = "initializer-storage";
          description.textContent = `${label}: ${storage.external ? `external file: ${storage.location || "not specified"}` : "embedded"}`;
          item.append(description);
        }
        return item;
      }),
    );
  }

  function typeTable(title, details) {
    const disclosure = document.createElement("details");
    disclosure.className = "initializer-structure";
    const summary = document.createElement("summary");
    summary.textContent = `${title}: ${details.kind}`;
    disclosure.append(summary);
    const table = document.createElement("table");
    table.className = "initializer-type-table";
    table.setAttribute("aria-label", `${title}: ${details.kind}`);
    table.createCaption().textContent = `${title}: ${details.kind}`;
    table.caption.hidden = true;
    const heading = table.createTHead().insertRow();
    for (const label of ["Field", "Type / description", "Value"]) {
      const cell = document.createElement("th");
      cell.scope = "col";
      cell.textContent = label;
      heading.append(cell);
    }
    const body = table.createTBody();
    const parents = new Set(details.entries.map((entry) => entry.parent));
    const expanded = new Set();
    const rows = [];
    const refresh = () => {
      for (const [index, row] of rows.entries()) {
        let parent = details.entries[index].parent;
        row.hidden = false;
        while (parent !== undefined) {
          if (!expanded.has(parent)) {
            row.hidden = true;
            break;
          }
          parent = details.entries[parent].parent;
        }
      }
    };
    for (const [
      index,
      { key, value, shape, data, parent },
    ] of details.entries.entries()) {
      const row = body.insertRow();
      rows.push(row);
      const name = document.createElement("th");
      name.scope = "row";
      const prefix =
        parent === undefined ? "" : `${details.entries[parent].key}.`;
      const label =
        prefix && key.startsWith(prefix) ? key.slice(prefix.length) : key;
      if (parents.has(index)) {
        const toggle = document.createElement("button");
        toggle.type = "button";
        toggle.className = "initializer-fold";
        toggle.textContent = label;
        toggle.setAttribute("aria-expanded", "false");
        toggle.addEventListener("click", () => {
          if (expanded.has(index)) expanded.delete(index);
          else expanded.add(index);
          toggle.setAttribute("aria-expanded", String(expanded.has(index)));
          refresh();
        });
        name.append(toggle);
      } else {
        name.textContent = label;
      }
      name.title = key;
      const description = document.createElement("td");
      description.textContent = `${value}${shape === undefined ? "" : ` ${shape}`}`;
      const preview = document.createElement("td");
      if (data !== undefined) {
        const values = document.createElement("pre");
        values.textContent = data;
        preview.append(values);
      }
      row.append(name, description, preview);
    }
    refresh();
    disclosure.append(table);
    return disclosure;
  }

  function parameterList(id, parameters, countId) {
    byId(countId).textContent = parameters.length;
    byId(`node-${id}`).replaceChildren(...parameters.map(attributeItem));
  }

  function attributeItem(parameter) {
    const item = document.createElement("li");
    const heading = document.createElement("strong");
    heading.textContent = `${parameter.name} (${parameter.type})${parameter.default ? " [default]" : ""}`;
    const value = document.createElement("pre");
    value.textContent = parameter.value;
    item.append(heading, value);
    for (const [index, graph] of (parameter.graphs ?? []).entries()) {
      const button = document.createElement("button");
      button.className = "icon-button";
      button.textContent = `Open ${subgraphLabel(parameter, index)}: ${graph.name} (${graph.nodeCount} nodes)`;
      button.addEventListener("click", () => navigate(graph.path));
      item.append(button);
    }
    return item;
  }
  byId("show-doc").addEventListener("click", () => {
    if (!inspectedNode?.schema?.doc) return;
    const { schema, opType, domain, opset } = inspectedNode;
    byId("doc-title").textContent = `${opType} documentation`;
    byId("doc-version").textContent =
      `${domain} · model opset ${opset} · LightOpSchema since version ${schema.sinceVersion}`;
    byId("doc-body").innerHTML = documentationMarkdown.render(schema.doc);
    byId("operator-doc").showModal();
  });
  byId("close-doc").addEventListener("click", () =>
    byId("operator-doc").close(),
  );
  byId("open-function").addEventListener("click", () => {
    if (inspectedNode?.function) navigate(inspectedNode.function.path);
  });

  function reset() {
    resetDocumentation();
    inspectedNode = undefined;
  }

  return { inspect, reset };
}
