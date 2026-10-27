const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { test } = require("node:test");
const { runInNewContext } = require("node:vm");

const {
  sugiyamaLayout,
  shortenGraphRoutes,
  graphPathSegments,
  flattenGraphPath,
} = runInNewContext(
  `${readFileSync(require.resolve("../media/graph-layout.js"), "utf8")}\n({ sugiyamaLayout, shortenGraphRoutes, graphPathSegments, flattenGraphPath });`,
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
  const points = flattenGraphPath(
    [
      { x: start.x, y: start.y + (from.height ?? 88) / 2 },
      ...route,
      { x: end.x, y: end.y - (to.height ?? 88) / 2 },
    ],
    true,
  );
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

test("smooths detours over their full span instead of rounding square corners", () => {
  const points = [
    { x: 0, y: 0 },
    { x: 0, y: 100 },
    { x: 300, y: 100 },
    { x: 300, y: 500 },
    { x: 0, y: 500 },
    { x: 0, y: 600 },
  ];
  const segments = graphPathSegments(points, true);
  assert.ok(segments.every((segment) => segment.length === 3));
  assert.deepEqual(segments[0][0], points[0]);
  assert.deepEqual(segments.at(-1).at(-1), points.at(-1));
  for (let index = 1; index < segments.length - 1; index++) {
    const previous = segments[index - 1];
    const current = segments[index];
    assert.deepEqual(previous.at(-1), current[0]);
    for (const axis of ["x", "y"])
      assert.equal(
        previous[2][axis] - previous[1][axis],
        current[1][axis] - current[0][axis],
      );
  }
  const curve = flattenGraphPath(points, true);
  assert.ok(curve.some(({ x, y }) => x > 50 && y < 90));
});

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
  assert.ok(Math.abs(at("left").x - at("right").x) - (180 + 280) / 2 >= 120);
  assert.ok(at("left").y - 60 - (at("source").y + 44) >= 108);
  assertContained(result, nodes);
  assert.deepEqual(
    [...layout(nodes, ["input"], ["join"]).positions],
    [...result.positions],
  );
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

test("omits routes with no bend points after shortcutting", () => {
  const result = layout([
    node("a"),
    node("b"),
    node("c", ["b"]),
    node("d", ["a", "c"]),
  ]);
  for (const routes of [result.routes.values(), result.inputRoutes.values()]) {
    for (const targets of routes)
      for (const points of targets.values()) assert.ok(points.length);
  }
  for (const points of result.outputRoutes.values()) assert.ok(points.length);
  assert.equal(result.routes.get("d")?.has("a"), undefined);
});

test("places graph ports beside their consumers and producers in both layout engines", () => {
  for (const count of [8, 600, 2100]) {
    const nodes = Array.from({ length: count }, (_, index) =>
      node(`node${index}`, index ? [`node${index - 1}`] : ["start"]),
    );
    nodes[count - 1].inputs.push("late");
    nodes[count - 1].captures = ["captured"];
    const inputs = ["late", "start", "captured", "unused"];
    const outputs = ["node0", `node${count - 1}`, "start"];
    const result = layout(nodes, inputs, outputs);
    assertContained(result, nodes);
    const first = result.positions.get("node0");
    const last = result.positions.get(`node${count - 1}`);
    for (const name of ["late", "captured"]) {
      const port = result.inputPositions.get(name);
      assert.ok(port.y > first.y);
      const gap = last.y - 44 - (port.y + 26);
      assert.ok(gap >= 108 && gap < 180, `${count}: ${name} gap ${gap}`);
    }
    const early = result.outputPositions.get("node0");
    assert.ok(early.y > first.y + 44 + 26);
    assert.ok(early.y < last.y);
    assert.ok(result.inputPositions.get("start").y < first.y);
    assert.ok(result.outputPositions.get("start").y < last.y);
    for (const [name, position] of [
      ...result.inputPositions,
      ...result.outputPositions,
    ]) {
      assert.ok(position.x >= 60, name);
      assert.ok(position.x + 60 <= result.bounds.width, name);
      assert.ok(position.y >= 26, name);
      assert.ok(position.y + 26 <= result.bounds.height, name);
      for (const item of nodes) {
        const operator = result.positions.get(item.id);
        assert.ok(
          Math.abs(position.x - operator.x) >= 130 ||
            Math.abs(position.y - operator.y) >= 70,
          `${name} overlaps ${item.id}`,
        );
      }
    }
    const repeated = layout(nodes, inputs, outputs);
    assert.deepEqual([...repeated.inputPositions], [...result.inputPositions]);
    assert.deepEqual(
      [...repeated.outputPositions],
      [...result.outputPositions],
    );
  }
});

test("shortens exterior routes through free interior corridors after placement", () => {
  for (const blocked of [false, true]) {
    const nodes = [node("source"), node("target", ["source"])];
    if (blocked) nodes.push(node("obstacle"));
    const sizes = new Map(
      nodes.map(({ id }) => [id, { width: 140, height: 88 }]),
    );
    const route = [
      { x: 100, y: 280 },
      { x: 500, y: 280 },
      { x: 500, y: 720 },
      { x: 100, y: 720 },
    ];
    const result = {
      positions: new Map([
        ["source", { x: 100, y: 200 }],
        ["target", { x: 100, y: 800 }],
        ...(blocked ? [["obstacle", { x: 100, y: 500 }]] : []),
      ]),
      producers: new Map(nodes.map(({ id }) => [id, id])),
      routes: new Map([["target", new Map([["source", route]])]]),
      bounds: { width: 600, height: 1000 },
    };
    shortenGraphRoutes({ nodes }, sizes, result);
    const shortened = result.routes.get("target").get("source");
    assert.notDeepEqual(shortened, route);
    if (blocked) {
      assert.ok(shortened.every(({ x }) => x < 250));
      assertRouteAvoidsNodes(result, nodes, "source", "target");
    } else assert.equal(shortened.length, 0);
  }
});

test("keeps a detour when the direct shortcut would cross another arc", () => {
  const nodes = [
    node("source"),
    node("target", ["source"]),
    node("left"),
    node("right", ["left"]),
  ];
  const sizes = new Map(nodes.map(({ id }) => [id, { width: 80, height: 88 }]));
  const route = [
    { x: 200, y: 280 },
    { x: 500, y: 280 },
    { x: 500, y: 720 },
    { x: 200, y: 720 },
  ];
  const result = {
    positions: new Map([
      ["source", { x: 200, y: 200 }],
      ["target", { x: 200, y: 800 }],
      ["left", { x: 50, y: 500 }],
      ["right", { x: 350, y: 500 }],
    ]),
    producers: new Map(nodes.map(({ id }) => [id, id])),
    routes: new Map([["target", new Map([["source", route]])]]),
    bounds: { width: 600, height: 1000 },
  };
  shortenGraphRoutes({ nodes }, sizes, result);
  assert.deepEqual(result.routes.get("target").get("source"), route);
});

test("routes a shared graph input around intervening operators", () => {
  const nodes = [
    node("first", ["shared"]),
    node("middle", ["first"]),
    node("last", ["middle", "shared"]),
  ];
  const result = layout(nodes, ["shared"], ["first", "last"]);
  const port = { id: "input-port", width: 120, height: 52 };
  result.positions.set(port.id, result.inputPositions.get("shared"));
  if (!result.routes.has("last")) result.routes.set("last", new Map());
  result.routes
    .get("last")
    .set(port.id, result.inputRoutes.get("last").get("shared"));
  assertRouteAvoidsNodes(result, [...nodes, port], port.id, "last");
});

test("keeps short fanout branches local instead of expanding them over the whole chain", () => {
  const nodes = Array.from({ length: 400 }, (_, index) =>
    node(`chain${index}`, index ? [`chain${index - 1}`] : []),
  );
  for (let index = 0; index < 600; index++)
    nodes.push(node(`leaf${index}`, ["chain0"]));
  const result = layout(nodes);
  assertContained(result, nodes);
  for (let index = 0; index < 600; index++) {
    assert.equal(
      result.positions.get(`leaf${index}`).y,
      result.positions.get("chain1").y,
    );
    assert.equal(result.routes.get(`leaf${index}`)?.has("chain0"), undefined);
  }
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
