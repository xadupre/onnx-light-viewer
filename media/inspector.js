/* global markdownit */
function createInspector({ byId, tensorDescription, subgraphLabel, navigate }) {
  const documentationMarkdown = markdownit({ html: false });
  documentationMarkdown.validateLink = (url) => /^https?:\/\//i.test(url);
  documentationMarkdown.renderer.rules.image = (tokens, index) =>
    documentationMarkdown.utils.escapeHtml(tokens[index].content);
  let inspectedNode;
  function resetDocumentation() {
    if (byId("operator-doc").open) byId("operator-doc").close();
    byId("doc-title").textContent = "";
    byId("doc-version").textContent = "";
    byId("doc-body").textContent = "";
  }

  function inspect(view, hasOperators) {
    const node = view?.node;
    const port = view?.port;
    byId("node-title").textContent =
      node?.opType ??
      (port
        ? port.output
          ? "Graph output"
          : "Graph input"
        : hasOperators
          ? "No selection"
          : "No operator");
    byId("node-description").textContent = node
      ? `${node.opType} ${node.function ? "local function" : "operator"}`
      : port
        ? `Graph ${port.output ? "output" : "input"} tensor.`
        : hasOperators
          ? "Select a node to inspect its inputs and outputs."
          : "This graph contains no operator nodes.";
    byId("open-function").hidden = !node?.function;
    byId("open-function").textContent = node?.function
      ? `Open function: ${node.function.name}`
      : "Open function";
    byId("node-name").textContent = node?.name ?? port?.name ?? "—";
    byId("node-domain").textContent = node?.domain ?? "—";
    byId("node-opset").textContent = node?.opset ?? "—";
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
    initializerList(node?.parameters ?? []);
    byId("node-parameters").hidden = !node;
    for (const direction of ["inputs", "outputs"]) {
      const names =
        node?.[direction] ??
        (port && direction === (port.output ? "inputs" : "outputs")
          ? [port.name]
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
        item.textContent = `${name}: ${tensorDescription(name)}`;
        return item;
      }),
    );
  }

  function initializerList(parameters) {
    byId("parameter-count").textContent = parameters.length;
    byId("node-initializers").replaceChildren(
      ...parameters.map((parameter) => {
        const item = document.createElement("li");
        const heading = document.createElement("strong");
        heading.textContent = `${parameter.name} (${parameter.type})`;
        const shape = document.createElement("div");
        shape.className = "tensor-info";
        shape.textContent = `Shape: [${parameter.shape?.join(", ") ?? "?"}]`;
        item.append(heading);
        if (parameter.shape !== undefined) item.append(shape);
        if (parameter.encoding !== undefined) {
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

  function parameterList(id, parameters, countId) {
    byId(countId).textContent = parameters.length;
    byId(`node-${id}`).replaceChildren(
      ...parameters.map((parameter) => {
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
      }),
    );
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
