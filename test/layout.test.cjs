const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { test } = require("node:test");
const { runInNewContext } = require("node:vm");

const sugiyamaLayout = runInNewContext(
  `${readFileSync(require.resolve("../media/graph-layout.js"), "utf8")}\nsugiyamaLayout;`,
  { dagre: require("@dagrejs/dagre") },
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

function segmentEntersBox(start, end, center, width, height) {
  let low = 0;
  let high = 1;
  for (const [axis, size] of [
    ["x", width],
    ["y", height],
  ]) {
    const minimum = center[axis] - size / 2 + 0.001;
    const maximum = center[axis] + size / 2 - 0.001;
    const delta = end[axis] - start[axis];
    if (!delta) {
      if (start[axis] <= minimum || start[axis] >= maximum) return false;
    } else {
      const first = (minimum - start[axis]) / delta;
      const last = (maximum - start[axis]) / delta;
      low = Math.max(low, Math.min(first, last));
      high = Math.min(high, Math.max(first, last));
    }
  }
  return low < high;
}

function assertRouteAvoidsNodes(result, nodes, source, target) {
  const from = nodes.find(({ id }) => id === source);
  const to = nodes.find(({ id }) => id === target);
  const start = result.positions.get(source);
  const end = result.positions.get(target);
  const route = result.routes.get(target)?.get(source);
  assert.ok(route?.length, `Missing route for ${source} -> ${target}`);
  const points = [
    { x: start.x, y: start.y + (from.height ?? 88) / 2 },
    ...route,
    { x: end.x, y: end.y - (to.height ?? 88) / 2 },
  ];
  for (const point of points) {
    assert.ok(point.x >= 0 && point.x <= result.bounds.width);
    assert.ok(point.y >= 0 && point.y <= result.bounds.height);
  }
  for (const item of nodes) {
    if (item.id === source || item.id === target) continue;
    for (let index = 1; index < points.length; index++)
      assert.equal(
        segmentEntersBox(
          points[index - 1],
          points[index],
          result.positions.get(item.id),
          item.width ?? 140,
          item.height ?? 88,
        ),
        false,
        `${source} -> ${target} crosses ${item.id}`,
      );
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

test("routes nested residual connections around intermediate operators", () => {
  const nodes = [
    node("source"),
    node("projection", ["source"], { width: 280 }),
    node("normalization", ["projection"], { height: 160 }),
    node("residual", ["normalization", "source"]),
    node("activation", ["residual"]),
    node("output", ["activation", "projection"]),
  ];
  const result = layout(nodes);
  assertContained(result, nodes);
  assertRouteAvoidsNodes(result, nodes, "source", "residual");
  assertRouteAvoidsNodes(result, nodes, "projection", "output");
  assert.deepEqual([...layout(nodes).routes], [...result.routes]);
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
  assertRouteAvoidsNodes(deep, chain, "node0", `node${count - 1}`);
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
