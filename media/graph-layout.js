/* global dagre */
const graphRouteRadius = 16;

function sugiyamaLayout(graph, sizes) {
  const dimensions = new Map(sizes);
  const ports = (names, output) =>
    names.map((name) => {
      const id = Symbol(name);
      dimensions.set(id, { width: 120, height: 52 });
      return {
        id,
        name,
        port: output ? "output" : "input",
        inputs: output ? [name] : [],
        outputs: output ? [] : [name],
      };
    });
  const inputs = ports(graph.inputs, false);
  const outputs = ports(graph.outputs, true);
  const layoutGraph = {
    ...graph,
    nodes: [...inputs, ...graph.nodes, ...outputs],
  };
  const result = layeredGraphLayout(layoutGraph, dimensions);
  shortenGraphRoutes(layoutGraph, dimensions, result);
  result.inputPositions = new Map();
  result.outputPositions = new Map();
  result.inputRoutes = new Map();
  result.outputRoutes = new Map();
  const inputNames = new Map(inputs.map(({ id, name }) => [id, name]));
  for (const port of [...inputs, ...outputs]) {
    const positions =
      port.port === "input" ? result.inputPositions : result.outputPositions;
    positions.set(port.name, result.positions.get(port.id));
    result.positions.delete(port.id);
    if (result.producers.get(port.name) === port.id)
      result.producers.delete(port.name);
    if (port.port === "output") {
      const route = result.routes.get(port.id)?.values().next().value;
      if (route?.length) result.outputRoutes.set(port.name, route);
      result.routes.delete(port.id);
    }
  }
  for (const [target, routes] of result.routes) {
    for (const [source, points] of routes) {
      if (!points.length) {
        routes.delete(source);
        continue;
      }
      if (!inputNames.has(source)) continue;
      if (!result.inputRoutes.has(target))
        result.inputRoutes.set(target, new Map());
      result.inputRoutes.get(target).set(inputNames.get(source), points);
      routes.delete(source);
    }
    if (!routes.size) result.routes.delete(target);
  }
  return result;
}

function graphPathSegments(points, routed) {
  if (!routed) {
    const [start, end] = points;
    const middle = (start.y + end.y) / 2;
    return [[start, { x: start.x, y: middle }, { x: end.x, y: middle }, end]];
  }
  const segments = [];
  let start = points[0];
  for (let index = 1; index < points.length - 1; index++) {
    const point = points[index];
    const next = points[index + 1];
    const end = {
      x: (point.x + next.x) / 2,
      y: (point.y + next.y) / 2,
    };
    segments.push([start, point, end]);
    start = end;
  }
  segments.push([start, points.at(-1), points.at(-1)]);
  return segments;
}

function graphPointDistance(point, start, end) {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const length = dx * dx + dy * dy;
  const fraction = length
    ? Math.max(
        0,
        Math.min(
          1,
          ((point.x - start.x) * dx + (point.y - start.y) * dy) / length,
        ),
      )
    : 0;
  return Math.hypot(
    point.x - start.x - fraction * dx,
    point.y - start.y - fraction * dy,
  );
}

function flattenGraphPath(points, routed, tolerance = 0.5) {
  const flattened = [points[0]];
  const flatten = (curve) => {
    const start = curve[0];
    const end = curve.at(-1);
    if (
      curve
        .slice(1, -1)
        .every((point) => graphPointDistance(point, start, end) <= tolerance)
    ) {
      flattened.push(end);
      return;
    }
    const left = [start];
    const right = [end];
    let row = curve;
    while (row.length > 1) {
      row = row.slice(1).map((point, index) => ({
        x: (row[index].x + point.x) / 2,
        y: (row[index].y + point.y) / 2,
      }));
      left.push(row[0]);
      right.push(row.at(-1));
    }
    flatten(left);
    flatten(right.reverse());
  };
  for (const curve of graphPathSegments(points, routed)) flatten(curve);
  return flattened;
}

function graphSpatialIndex(items) {
  const boundsOf = (boxes) => {
    const bounds = {
      left: Infinity,
      top: Infinity,
      right: -Infinity,
      bottom: -Infinity,
    };
    for (const box of boxes) {
      bounds.left = Math.min(bounds.left, box.left);
      bounds.top = Math.min(bounds.top, box.top);
      bounds.right = Math.max(bounds.right, box.right);
      bounds.bottom = Math.max(bounds.bottom, box.bottom);
    }
    return bounds;
  };
  const overlaps = (a, b) =>
    a.left <= b.right &&
    a.right >= b.left &&
    a.top <= b.bottom &&
    a.bottom >= b.top;
  const bounds = boundsOf(items.map((entry) => entry.bounds));
  const horizontal = bounds.right - bounds.left > bounds.bottom - bounds.top;
  const entries = [...items].sort((a, b) =>
    horizontal
      ? a.bounds.left + a.bounds.right - b.bounds.left - b.bounds.right ||
        a.bounds.top + a.bounds.bottom - b.bounds.top - b.bounds.bottom
      : a.bounds.top + a.bounds.bottom - b.bounds.top - b.bounds.bottom ||
        a.bounds.left + a.bounds.right - b.bounds.left - b.bounds.right,
  );
  const build = (start, end, parent) => {
    if (start === end) return;
    const node = { parent };
    if (end - start === 1) {
      node.item = entries[start];
      node.bounds = node.item.bounds;
      node.item.indexNode = node;
    } else {
      const middle = Math.floor((start + end) / 2);
      node.children = [build(start, middle, node), build(middle, end, node)];
      node.bounds = boundsOf(node.children.map((child) => child.bounds));
    }
    return node;
  };
  const root = build(0, entries.length);
  return {
    find(bounds, predicate, budget) {
      const visit = (node) => {
        if (!node || !overlaps(bounds, node.bounds)) return;
        if (budget && --budget.remaining < 0) return { limited: true };
        if (node.item) return predicate(node.item) ? node.item : undefined;
        return visit(node.children[0]) ?? visit(node.children[1]);
      };
      return visit(root);
    },
    update(item) {
      let node = item.indexNode;
      node.bounds = item.bounds;
      while ((node = node.parent))
        node.bounds = boundsOf(node.children.map((child) => child.bounds));
    },
  };
}

function graphSegmentBounds(start, end, margin = 0) {
  return {
    left: Math.min(start.x, end.x) - margin,
    top: Math.min(start.y, end.y) - margin,
    right: Math.max(start.x, end.x) + margin,
    bottom: Math.max(start.y, end.y) + margin,
  };
}

function graphSegmentHitsBox(start, end, box, margin) {
  let low = 0;
  let high = 1;
  for (const [axis, minimum, maximum] of [
    ["x", box.left - margin, box.right + margin],
    ["y", box.top - margin, box.bottom + margin],
  ]) {
    const delta = end[axis] - start[axis];
    if (!delta) {
      if (start[axis] < minimum || start[axis] > maximum) return false;
    } else {
      const first = (minimum - start[axis]) / delta;
      const last = (maximum - start[axis]) / delta;
      low = Math.max(low, Math.min(first, last));
      high = Math.min(high, Math.max(first, last));
    }
  }
  return low <= high;
}

function graphSegmentsCross(a, b, c, d) {
  const side = (p, q, r) =>
    (q.x - p.x) * (r.y - p.y) - (q.y - p.y) * (r.x - p.x);
  return side(a, b, c) * side(a, b, d) < 0 && side(c, d, a) * side(c, d, b) < 0;
}

function shortenGraphRoutes(graph, sizes, result) {
  if (!result.routes.size) return;
  const boxes = graph.nodes.map(({ id }) => {
    const position = result.positions.get(id);
    const size = sizes.get(id);
    return {
      id,
      bounds: {
        left: position.x - size.width / 2,
        top: position.y - size.height / 2,
        right: position.x + size.width / 2,
        bottom: position.y + size.height / 2,
      },
    };
  });
  const nodes = graphSpatialIndex(boxes);
  const edges = [];
  const setGeometry = (edge, points, routed) => {
    edge.rawPoints = points;
    edge.routed = routed;
    edge.points = routed ? points : flattenGraphPath(points, false);
    edge.curvePoints = routed ? undefined : edge.points;
    edge.bounds = {
      left: Infinity,
      top: Infinity,
      right: -Infinity,
      bottom: -Infinity,
    };
    edge.length = 0;
    for (let index = 0; index < edge.points.length; index++) {
      const point = edge.points[index];
      edge.bounds.left = Math.min(edge.bounds.left, point.x);
      edge.bounds.top = Math.min(edge.bounds.top, point.y);
      edge.bounds.right = Math.max(edge.bounds.right, point.x);
      edge.bounds.bottom = Math.max(edge.bounds.bottom, point.y);
      if (index)
        edge.length += Math.hypot(
          point.x - edge.points[index - 1].x,
          point.y - edge.points[index - 1].y,
        );
    }
  };
  for (const node of graph.nodes) {
    const dependencies = new Set();
    for (const input of [...node.inputs, ...(node.captures ?? [])]) {
      const source = result.producers.get(input);
      if (source === undefined || dependencies.has(source)) continue;
      dependencies.add(source);
      const from = result.positions.get(source);
      const to = result.positions.get(node.id);
      const start = { x: from.x, y: from.y + sizes.get(source).height / 2 };
      const end = { x: to.x, y: to.y - sizes.get(node.id).height / 2 };
      const route = result.routes.get(node.id)?.get(source);
      const edge = { source, target: node.id, start, end, route };
      setGeometry(edge, [start, ...(route ?? []), end], !!route);
      edges.push(edge);
    }
  }
  const arcs = graphSpatialIndex(edges);
  const margin = graphRouteRadius + 8;
  const arcMargin = graphRouteRadius * 2 + 1;
  const nodeHit = (edge, start, end, budget) =>
    nodes.find(
      graphSegmentBounds(start, end, margin),
      (node) =>
        node.id !== edge.source &&
        node.id !== edge.target &&
        graphSegmentHitsBox(start, end, node.bounds, margin),
      budget,
    );
  const clear = (edge, points, budget) => {
    for (let index = 1; index < points.length; index++) {
      const start = points[index - 1];
      const end = points[index];
      if (nodeHit(edge, start, end, budget)) return false;
      const hit = arcs.find(
        graphSegmentBounds(start, end, arcMargin),
        (other) => {
          if (other === edge) return false;
          const shared =
            other.source === edge.source || other.target === edge.target;
          const points = (other.curvePoints ??= flattenGraphPath(
            other.rawPoints,
            other.routed,
          ));
          for (let index = 1; index < points.length; index++) {
            if (--budget.remaining < 0) return true;
            const a = points[index - 1];
            const b = points[index];
            if (graphSegmentsCross(start, end, a, b)) return true;
            if (
              !shared &&
              Math.min(
                graphPointDistance(start, a, b),
                graphPointDistance(end, a, b),
                graphPointDistance(a, start, end),
                graphPointDistance(b, start, end),
              ) <= arcMargin
            )
              return true;
          }
          return false;
        },
        budget,
      );
      if (hit) return false;
    }
    return true;
  };
  // Shorter branches first let later branches reuse the same unobstructed trunk.
  edges.sort((a, b) => a.end.y - a.start.y - (b.end.y - b.start.y));
  for (const edge of edges) {
    if (!edge.route || edge.end.y - edge.start.y <= 144) continue;
    const exit = { x: edge.start.x, y: edge.start.y + 72 };
    const entry = { x: edge.end.x, y: edge.end.y - 72 };
    // Dense overlap must not turn the post-pass into an all-pairs traversal.
    const budget = { remaining: 256 };
    const blocker = nodeHit(edge, exit, entry, budget);
    const lanes = new Set([exit.x, entry.x, (exit.x + entry.x) / 2]);
    if (blocker?.bounds) {
      lanes.add(blocker.bounds.left - margin * 2 - 1);
      lanes.add(blocker.bounds.right + margin * 2 + 1);
    }
    const candidates = [
      [edge.start, edge.end],
      [edge.start, exit, entry, edge.end],
      ...[...lanes].map((x) => [
        edge.start,
        exit,
        { x, y: exit.y },
        { x, y: entry.y },
        entry,
        edge.end,
      ]),
    ]
      .filter((points) =>
        points.every(
          ({ x, y }) =>
            x >= 0 &&
            x <= result.bounds.width &&
            y >= 0 &&
            y <= result.bounds.height,
        ),
      )
      .map((routePoints) => {
        let length = 0;
        for (let index = 1; index < routePoints.length; index++)
          length += Math.hypot(
            routePoints[index].x - routePoints[index - 1].x,
            routePoints[index].y - routePoints[index - 1].y,
          );
        return { routePoints, length };
      })
      .sort((a, b) => a.length - b.length);
    for (const candidate of candidates) {
      if (budget.remaining <= 0) break;
      if (
        candidate.length >= edge.length + graphRouteRadius * 4 ||
        !clear(edge, candidate.routePoints, budget)
      )
        continue;
      candidate.points = flattenGraphPath(candidate.routePoints, true);
      candidate.length = 0;
      for (let index = 1; index < candidate.points.length; index++)
        candidate.length += Math.hypot(
          candidate.points[index].x - candidate.points[index - 1].x,
          candidate.points[index].y - candidate.points[index - 1].y,
        );
      if (edge.curveLength === undefined) {
        const points = (edge.curvePoints ??= flattenGraphPath(
          edge.rawPoints,
          edge.routed,
        ));
        edge.curveLength = 0;
        for (let index = 1; index < points.length; index++)
          edge.curveLength += Math.hypot(
            points[index].x - points[index - 1].x,
            points[index].y - points[index - 1].y,
          );
      }
      if (
        candidate.length >= edge.curveLength - 1 ||
        !clear(edge, candidate.points, budget)
      )
        continue;
      const route = candidate.routePoints.slice(1, -1);
      result.routes.get(edge.target).set(edge.source, route);
      setGeometry(edge, candidate.routePoints, true);
      arcs.update(edge);
      break;
    }
  }
}

function layeredGraphLayout(graph, sizes) {
  const rankSeparation = 108;
  const nodeSeparation = 120;
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
  let cyclic = false;
  for (let head = 0; head < count; head++) {
    if (head === queue.length) {
      cyclic = true;
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

  // Inputs share the rank immediately before their earliest consumer.
  for (let index = 0; index < count; index++) {
    if (graph.nodes[index].port !== "input" || !children[index].length)
      continue;
    let rank = Infinity;
    for (const child of children[index]) rank = Math.min(rank, ranks[child]);
    ranks[index] = Math.max(0, rank - 1);
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
  if (
    count > 0 &&
    count <= 2000 &&
    depth < 500 &&
    expanded <= 20_000 &&
    !cyclic
  ) {
    const layered = new dagre.graphlib.Graph();
    layered.setGraph({
      rankdir: "TB",
      ranksep: rankSeparation,
      nodesep: nodeSeparation,
      edgesep: 24,
      marginx: 20,
      // Reuse these ranks so ports stay local and expansion stays within budget.
      ranker: (ranked) => {
        for (const id of ranked.nodes()) {
          const rank = ranks[Number(id)];
          ranked.node(id).rank = rank === undefined ? -1 : 2 * rank;
        }
      },
    });
    layered.setDefaultEdgeLabel(() => ({}));
    graph.nodes.forEach((node, index) => {
      const { width, height } = sizes.get(node.id);
      layered.setNode(String(index), { width, height });
      for (const parent of parents[index])
        layered.setEdge(String(parent), String(index));
    });
    dagre.layout(layered);
    const width = Math.max(500, layered.graph().width);
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
    let left = 0;
    let right = width;
    for (const targets of routes.values())
      for (const points of targets.values())
        for (const point of points) {
          left = Math.min(left, point.x - 20);
          right = Math.max(right, point.x + 20);
        }
    if (left < 0) {
      for (const position of positions.values()) position.x -= left;
      for (const targets of routes.values())
        for (const points of targets.values())
          for (const point of points) point.x -= left;
    }
    return {
      positions,
      producers,
      routes,
      bounds: {
        x: 0,
        y: 0,
        width: right - left,
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
    widths[rank] =
      (widths[rank] ?? -nodeSeparation) +
      dimensions[index].width +
      nodeSeparation;
    heights[rank] = Math.max(heights[rank] ?? 0, dimensions[index].height);
  }
  let width = 500;
  for (const layerWidth of widths) width = Math.max(width, layerWidth + 40);
  const centers = new Float64Array(count);
  const pack = (rank) => {
    let left = (width - widths[rank]) / 2;
    for (const index of layers[rank]) {
      centers[index] = left + dimensions[index].width / 2;
      left += dimensions[index].width + nodeSeparation;
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
    top += heights[rank] + rankSeparation;
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
      const y1 = from.y + heights[ranks[parent]] / 2 + rankSeparation / 2;
      const y2 = to.y - heights[ranks[index]] / 2 - rankSeparation / 2;
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
