const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { test } = require("node:test");
const { runInNewContext } = require("node:vm");

const sugiyamaLayout = runInNewContext(
  `${readFileSync(require.resolve("../media/graph-layout.js"), "utf8")}\nsugiyamaLayout;`,
);

function layout(nodes, inputs = [], outputs = []) {
  return sugiyamaLayout(
    { nodes, inputs, outputs },
    new Map(
      nodes.map((node) => [
        node.id,
        { width: node.width ?? 140, height: node.height ?? 88 },
      ]),
    ),
  );
}

function node(id, inputs = [], extra = {}) {
  return { id, inputs, outputs: [id], ...extra };
}

function assertContained(result, nodes) {
  assert.equal(result.positions.size, nodes.length);
  for (const item of nodes) {
    const { x, y } = result.positions.get(item.id);
    const width = item.width ?? 140;
    const height = item.height ?? 88;
    assert.ok(Number.isFinite(x) && Number.isFinite(y));
    assert.ok(x - width / 2 >= 0);
    assert.ok(x + width / 2 <= result.bounds.width);
    assert.ok(y - height / 2 > 86);
    assert.ok(y + height / 2 < result.bounds.height - 86);
  }
}

test("layers unsorted dependencies, captures and duplicate inputs with differently sized blocks", () => {
  const nodes = [
    node("join", ["left", "right", "left"], { height: 160 }),
    node("right", ["weight"], { captures: ["source"], width: 280 }),
    node("left", ["source"], { width: 180, height: 120 }),
    node("source", ["input"]),
  ];
  const result = layout(nodes, ["input"], ["join"]);
  const at = (id) => result.positions.get(id);
  assert.ok(at("source").y + 44 < at("left").y - 60);
  assert.equal(at("left").y, at("right").y);
  assert.ok(at("left").y + 60 < at("join").y - 80);
  assert.ok(Math.abs(at("left").x - at("right").x) >= 230);
  assertContained(result, nodes);
  assert.deepEqual([...layout(nodes).positions], [...result.positions]);
});

test("barycentric sweeps uncross independent branches and align their chains", () => {
  const nodes = [node("a"), node("b"), node("c", ["b"]), node("d", ["a"])];
  const { positions } = layout(nodes);
  assert.ok(positions.get("a").x < positions.get("b").x);
  assert.ok(positions.get("d").x < positions.get("c").x);
  assert.equal(positions.get("a").x, positions.get("d").x);
  assert.equal(positions.get("b").x, positions.get("c").x);
});

test("handles empty graphs, disconnected nodes, optional tensors and cyclic dependencies deterministically", () => {
  const empty = layout(
    [],
    Array.from({ length: 12 }, (_, i) => `input${i}`),
  );
  assert.equal(empty.positions.size, 0);
  assert.ok(empty.bounds.width >= 12 * 120);
  const nodes = [
    node("a", ["b"]),
    node("b", ["a"]),
    node("tail", ["b"]),
    node("disconnected", ["", "initializer"], { outputs: ["", "external"] }),
  ];
  const first = layout(nodes);
  assertContained(first, nodes);
  assert.equal(first.producers.has(""), false);
  assert.deepEqual([...layout(nodes).positions], [...first.positions]);
});

test("lays out 20,000-node deep and wide graphs without expanding long edges", () => {
  const count = 20_000;
  const chain = Array.from({ length: count }, (_, i) =>
    node(`node${i}`, i === 0 ? [] : [`node${i - 1}`, "node0"]),
  );
  const deep = layout(chain);
  assertContained(deep, chain);
  for (let i = 1; i < count; i++) {
    assert.ok(
      deep.positions.get(`node${i}`).y >
        deep.positions.get(`node${i - 1}`).y + 88,
    );
  }

  const leaves = Array.from({ length: count - 2 }, (_, i) =>
    node(`leaf${i}`, ["source"], { width: i % 2 ? 280 : 140 }),
  );
  const wide = [
    node("source"),
    ...leaves,
    node(
      "join",
      leaves.map(({ id }) => id),
    ),
  ];
  const result = layout(wide);
  assertContained(result, wide);
  let right = -Infinity;
  for (const item of leaves) {
    const position = result.positions.get(item.id);
    assert.equal(position.y, result.positions.get("leaf0").y);
    assert.ok(position.x - item.width / 2 > right);
    right = position.x + item.width / 2;
  }
  assert.ok(result.bounds.height < deep.bounds.height);
});
