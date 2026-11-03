const assert = require("node:assert/strict");
const { test } = require("node:test");
const { summarizeProfile } = require("../media/profiling.cjs");

test("GenAI profile summaries preserve graph-local identities and repeated calls", () => {
  const event = (name, index, ts, dur, op = "Relu", tid = 1) => ({
    cat: "Node",
    name: `${name}_kernel_time`,
    pid: 1,
    tid,
    ts,
    dur,
    args: { node_index: index, op_name: op, provider: "CPUExecutionProvider" },
  });
  const trace = [
    event("shared", 1, 10, 3),
    event("first", 0, 0, 50, "If"),
    event("shared", 1, 70, 6),
    event("second", 1, 60, 30, "If"),
    event("shared", 1, 100, 5),
    event("shared", 1, 120, 7),
    event("shared", 2, 140, 4),
    event("shared", 1, 15, 2, "Relu", 2),
    { cat: "Session", name: "model_run", dur: 100 },
    { cat: "Node", name: "shared_fence_before", dur: 1 },
  ];
  const result = summarizeProfile(trace);
  assert.equal(result.trace, trace);
  assert.deepEqual(
    result.nodes
      .filter((node) => node.op === "Relu")
      .map(({ name, calls, durationUs }) => ({ name, calls, durationUs })),
    [
      { name: "shared [node 1]", calls: 3, durationUs: 14 },
      { name: "second [node 1] / shared [node 1]", calls: 1, durationUs: 6 },
      { name: "shared [node 2]", calls: 1, durationUs: 4 },
      { name: "first [node 0] / shared [node 1]", calls: 1, durationUs: 3 },
    ],
  );
});

test("empty or incomplete traces produce no invalid timing bars", () => {
  assert.deepEqual(summarizeProfile([]), { trace: [], nodes: [] });
  const { nodes } = summarizeProfile([
    null,
    {},
    { cat: "Node", name: "invalid_kernel_time", dur: -1 },
    { cat: "Node", name: "invalid_kernel_time", dur: Infinity },
    { cat: "Node", name: "empty_kernel_time", dur: 0 },
  ]);
  assert.deepEqual(nodes, [
    { name: "empty", op: "", provider: "", calls: 1, durationUs: 0 },
  ]);
});
