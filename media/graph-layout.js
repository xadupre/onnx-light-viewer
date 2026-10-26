/* global dagre */
function sugiyamaLayout(graph, sizes) {
  const count = graph.nodes.length;
  const indices = new Map();
  const producers = new Map();
  graph.nodes.forEach((node, index) => {
    indices.set(node.id, index);
    for (const output of node.outputs) {
      if (output) producers.set(output, node.id);
    }
  });
  const parents = Array.from({ length: count }, () => []);
  const children = Array.from({ length: count }, () => []);
  const remaining = new Uint32Array(count);
  graph.nodes.forEach((node, index) => {
    const dependencies = new Set();
    for (const input of [...node.inputs, ...(node.captures ?? [])]) {
      if (!input) continue;
      const parent = indices.get(producers.get(input));
      if (parent !== undefined) dependencies.add(parent);
    }
    for (const parent of dependencies) {
      parents[index].push(parent);
      children[parent].push(index);
    }
    remaining[index] = dependencies.size;
  });

  const ranks = new Uint32Array(count);
  const queue = [];
  for (let index = 0; index < count; index++) {
    if (!remaining[index]) queue.push(index);
  }
  let cycleStart = 0;
  for (let head = 0; head < count; head++) {
    if (head === queue.length) {
      // Break a cyclic dependency deterministically without recursive traversal.
      while (!remaining[cycleStart]) cycleStart++;
      remaining[cycleStart] = 0;
      queue.push(cycleStart);
    }
    const index = queue[head];
    for (const child of children[index]) {
      if (!remaining[child]) continue;
      ranks[child] = Math.max(ranks[child], ranks[index] + 1);
      if (--remaining[child] === 0) queue.push(child);
    }
  }

  // Proper crossing reduction needs intermediate ranks for skip connections.
  // Bound expansion and recursive engine depth for very large models.
  let expanded = count;
  let depth = 0;
  for (let index = 0; index < count; index++) {
    depth = Math.max(depth, ranks[index]);
    for (const parent of parents[index]) {
      expanded += 2 * Math.max(1, ranks[index] - ranks[parent]);
    }
  }
  if (count > 0 && count <= 2000 && depth < 500 && expanded <= 20_000) {
    const layered = new dagre.graphlib.Graph();
    layered.setGraph({
      rankdir: "TB",
      ranksep: 36,
      nodesep: 60,
      edgesep: 24,
      marginx: 20,
      ranker: count > 500 ? "longest-path" : "network-simplex",
    });
    layered.setDefaultEdgeLabel(() => ({}));
    graph.nodes.forEach((node, index) => {
      const { width, height } = sizes.get(node.id);
      layered.setNode(String(index), { width, height });
      for (const parent of parents[index])
        layered.setEdge(String(parent), String(index));
    });
    dagre.layout(layered);
    const width = Math.max(
      500,
      layered.graph().width,
      40 + Math.max(graph.inputs.length, graph.outputs.length) * 140,
    );
    const offset = (width - layered.graph().width) / 2;
    const positions = new Map(
      graph.nodes.map((node, index) => {
        const position = layered.node(String(index));
        return [node.id, { x: position.x + offset, y: position.y + 133 }];
      }),
    );
    const routes = new Map();
    for (const edge of layered.edges()) {
      const points = layered.edge(edge).points;
      if (points.length <= 3) continue;
      const target = graph.nodes[Number(edge.w)].id;
      if (!routes.has(target)) routes.set(target, new Map());
      routes.get(target).set(
        graph.nodes[Number(edge.v)].id,
        points.slice(1, -1).map(({ x, y }) => ({ x: x + offset, y: y + 133 })),
      );
    }
    return {
      positions,
      producers,
      routes,
      bounds: {
        x: 0,
        y: 0,
        width,
        height: Math.max(650, layered.graph().height + 286),
      },
    };
  }

  const layers = [];
  const dimensions = graph.nodes.map((node) => sizes.get(node.id));
  const widths = [];
  const heights = [];
  for (let index = 0; index < count; index++) {
    const rank = ranks[index];
    (layers[rank] ??= []).push(index);
    widths[rank] = (widths[rank] ?? -40) + dimensions[index].width + 40;
    heights[rank] = Math.max(heights[rank] ?? 0, dimensions[index].height);
  }
  let width = Math.max(
    500,
    40 + Math.max(graph.inputs.length, graph.outputs.length) * 140,
  );
  for (const layerWidth of widths) width = Math.max(width, layerWidth + 40);
  const centers = new Float64Array(count);
  const pack = (rank) => {
    let left = (width - widths[rank]) / 2;
    for (const index of layers[rank]) {
      centers[index] = left + dimensions[index].width / 2;
      left += dimensions[index].width + 40;
    }
  };
  layers.forEach((_, rank) => pack(rank));

  const scores = new Float64Array(count);
  // Bound crossing reduction and retain long edges directly, without dummy nodes.
  for (let pass = 0; pass < 4; pass++) {
    const downward = pass % 2 === 0;
    const neighbors = downward ? parents : children;
    for (
      let rank = downward ? 1 : layers.length - 2;
      downward ? rank < layers.length : rank >= 0;
      rank += downward ? 1 : -1
    ) {
      const layer = layers[rank];
      if (layer.length < 2) continue;
      for (const index of layer) {
        let total = 0;
        for (const neighbor of neighbors[index]) total += centers[neighbor];
        scores[index] = neighbors[index].length
          ? total / neighbors[index].length
          : centers[index];
      }
      layer.sort(
        (left, right) =>
          scores[left] - scores[right] || centers[left] - centers[right],
      );
      pack(rank);
    }
  }

  const positions = new Map();
  let top = 133;
  layers.forEach((layer, rank) => {
    for (const index of layer) {
      positions.set(graph.nodes[index].id, {
        x: centers[index],
        y: top + heights[rank] / 2,
      });
    }
    top += heights[rank] + 36;
  });
  const routes = new Map();
  const lanes = new Map();
  for (let index = 0; index < count; index++) {
    for (const parent of parents[index]) {
      if (ranks[index] <= ranks[parent] + 1) continue;
      if (!lanes.has(parent)) lanes.set(parent, lanes.size % 16);
      const from = positions.get(graph.nodes[parent].id);
      const to = positions.get(graph.nodes[index].id);
      const lane = lanes.get(parent) * 12;
      const x = from.x + to.x < width ? -20 - lane : width + 20 + lane;
      const y1 = from.y + heights[ranks[parent]] / 2 + 18;
      const y2 = to.y - heights[ranks[index]] / 2 - 18;
      const target = graph.nodes[index].id;
      if (!routes.has(target)) routes.set(target, new Map());
      routes.get(target).set(graph.nodes[parent].id, [
        { x: from.x, y: y1 },
        { x, y: y1 },
        { x, y: y2 },
        { x: to.x, y: y2 },
      ]);
    }
  }
  if (routes.size) {
    const margin = 40 + Math.min(lanes.size - 1, 15) * 12;
    for (const position of positions.values()) position.x += margin;
    for (const targets of routes.values())
      for (const points of targets.values())
        for (const point of points) point.x += margin;
    width += margin * 2;
  }
  return {
    positions,
    producers,
    routes,
    bounds: { x: 0, y: 0, width, height: Math.max(650, top + 117) },
  };
}
