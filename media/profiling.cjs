function summarizeProfile(trace) {
  const events = trace
    .filter(
      (event) =>
        event?.cat === "Node" &&
        typeof event.name === "string" &&
        event.name.endsWith("_kernel_time") &&
        Number.isFinite(event.dur) &&
        event.dur >= 0,
    )
    .sort(
      (a, b) =>
        (a.pid ?? 0) - (b.pid ?? 0) ||
        (a.tid ?? 0) - (b.tid ?? 0) ||
        (a.ts ?? 0) - (b.ts ?? 0) ||
        b.dur - a.dur,
    );
  const nodes = new Map();
  let scopes = [];
  let thread;
  for (const event of events) {
    const currentThread = JSON.stringify([event.pid, event.tid]);
    if (thread !== currentThread) scopes = [];
    thread = currentThread;
    const start = event.ts ?? 0;
    const end = start + event.dur;
    while (
      scopes.length &&
      (start >= scopes.at(-1).end || end > scopes.at(-1).end)
    )
      scopes.pop();
    const args = event.args ?? {};
    const name = event.name.slice(0, -"_kernel_time".length);
    const op = args.op_name ?? "";
    const provider = args.provider ?? "";
    const parent = scopes.at(-1);
    const key = JSON.stringify([
      parent?.key,
      name,
      op,
      provider,
      args.node_index,
    ]);
    let node = nodes.get(key);
    if (!node) {
      const label =
        args.node_index == null ? name : `${name} [node ${args.node_index}]`;
      node = {
        name: parent ? `${parent.name} / ${label}` : label,
        op,
        provider,
        calls: 0,
        durationUs: 0,
      };
      nodes.set(key, node);
    }
    node.calls++;
    node.durationUs += event.dur;
    if (
      ["If", "Loop", "Scan"].includes(op) &&
      event.ts != null &&
      event.dur > 0
    )
      scopes.push({ end, key, name: node.name });
  }
  return {
    trace,
    nodes: [...nodes.values()].sort((a, b) => b.durationUs - a.durationUs),
  };
}

module.exports = { summarizeProfile };
