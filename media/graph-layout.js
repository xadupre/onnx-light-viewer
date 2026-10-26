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
  return {
    positions,
    producers,
    bounds: { x: 0, y: 0, width, height: Math.max(650, top + 117) },
  };
}
