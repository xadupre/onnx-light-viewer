const assert = require("node:assert/strict");
const { execFileSync } = require("node:child_process");
const { test } = require("node:test");
const { summarizeProfile } = require("../media/profiling.cjs");

test("deeply nested profiles aggregate repeated scopes within a bounded heap", () => {
  const nodes = JSON.parse(
    execFileSync(
      process.execPath,
      [
        "--max-old-space-size=64",
        "-e",
        `
const { summarizeProfile } = require(process.argv[1]);
const trace = [];
for (let run = 0; run < 2; run++)
  for (let branch = 0; branch < 2; branch++)
    for (let depth = 0; depth < 64; depth++)
      trace.push({
        cat: "Node",
        name: (depth === 0 ? "branch" + branch : "nested") + "_kernel_time",
        ts: run * 1000 + branch * 300 + depth,
        dur: 256 - depth * 2,
        args: { op_name: depth === 63 ? "Relu" : "If", node_index: 0 },
      });
console.log(JSON.stringify(summarizeProfile(trace).nodes));
`,
        require.resolve("../media/profiling.cjs"),
      ],
      { encoding: "utf8", timeout: 10000 },
    ),
  );
  assert.equal(nodes.length, 128);
  assert.ok(nodes.every((node) => node.calls === 2));
  const leaves = nodes.filter((node) => node.op === "Relu");
  assert.equal(leaves.length, 2);
  assert.ok(leaves[0].name.startsWith("branch0 [node 0] / "));
  assert.ok(leaves[1].name.startsWith("branch1 [node 0] / "));
});

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
