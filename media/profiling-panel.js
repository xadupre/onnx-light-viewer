function renderProfilingPlots(profiling) {
  const container = document.createElement("section");
  container.className = "profiling-plots";
  container.setAttribute("aria-label", "Execution profiling plots");
  const nodes = profiling.nodes.filter(
    (node) => Number.isFinite(node.durationUs) && node.durationUs >= 0,
  );
  const operators = new Map();
  for (const node of nodes)
    operators.set(node.op, (operators.get(node.op) ?? 0) + node.durationUs);
  const plot = (title, values) => {
    const heading = document.createElement("h3");
    heading.textContent = title;
    container.append(heading);
    if (!values.length) {
      const empty = document.createElement("p");
      empty.textContent = "No kernel timing events were recorded.";
      container.append(empty);
      return;
    }
    values.sort((a, b) => b.durationUs - a.durationUs);
    const max = values[0].durationUs || 1;
    const ns = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(ns, "svg");
    svg.setAttribute("viewBox", `0 0 640 ${values.length * 32 + 28}`);
    svg.setAttribute("role", "img");
    svg.setAttribute(
      "aria-label",
      `${title}; total kernel duration in milliseconds, slowest first`,
    );
    const description = document.createElementNS(ns, "desc");
    description.textContent = values
      .map(
        (value) => `${value.name}: ${(value.durationUs / 1000).toFixed(3)} ms`,
      )
      .join("; ");
    svg.append(description);
    values.forEach((value, index) => {
      const group = document.createElementNS(ns, "g");
      const label = `${value.name}: ${(value.durationUs / 1000).toFixed(3)} ms`;
      const tooltip = document.createElementNS(ns, "title");
      tooltip.textContent = label;
      const text = document.createElementNS(ns, "text");
      text.setAttribute("x", "0");
      text.setAttribute("y", String(index * 32 + 18));
      text.textContent =
        value.name.length > 25 ? `${value.name.slice(0, 24)}…` : value.name;
      const bar = document.createElementNS(ns, "rect");
      bar.setAttribute("x", "230");
      bar.setAttribute("y", String(index * 32 + 3));
      bar.setAttribute("width", String((value.durationUs / max) * 310));
      bar.setAttribute("height", "22");
      const time = document.createElementNS(ns, "text");
      time.setAttribute("x", "550");
      time.setAttribute("y", String(index * 32 + 18));
      time.textContent = `${(value.durationUs / 1000).toFixed(3)} ms`;
      group.append(tooltip, text, bar, time);
      svg.append(group);
    });
    const axis = document.createElementNS(ns, "text");
    axis.setAttribute("x", "230");
    axis.setAttribute("y", String(values.length * 32 + 18));
    axis.textContent = `0 — ${(max / 1000).toFixed(3)} ms (total kernel duration)`;
    svg.append(axis);
    container.append(svg);
  };
  plot(
    "Node totals",
    nodes.map((node) => ({ name: node.name, durationUs: node.durationUs })),
  );
  plot(
    "Operator totals",
    [...operators].map(([name, durationUs]) => ({ name, durationUs })),
  );
  return container;
}
