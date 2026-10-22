import { onnx } from "onnx-proto";

export interface GraphNode {
  id: string;
  name: string;
  opType: string;
  domain: string;
  inputs: string[];
  outputs: string[];
}

export interface Graph {
  name: string;
  nodes: GraphNode[];
  inputs: string[];
  outputs: string[];
  initializerCount: number;
}

export const demoGraph: Graph = {
  name: "A first ONNX graph",
  nodes: [
    {
      id: "node-0",
      name: "activation",
      opType: "Relu",
      domain: "ai.onnx",
      inputs: ["input"],
      outputs: ["output"],
    },
  ],
  inputs: ["input"],
  outputs: ["output"],
  initializerCount: 0,
};

export function decodeGraph(bytes: Uint8Array, fileName: string): Graph {
  const model = onnx.ModelProto.decode(bytes);
  if (!model.graph || !model.irVersion?.toString().match(/^[1-9]\d*$/)) {
    throw new Error("This file does not contain an ONNX model graph.");
  }

  const graph = model.graph;
  const initializers = new Set(
    (graph.initializer ?? []).map((tensor) => tensor.name),
  );
  for (const tensor of graph.sparseInitializer ?? []) {
    initializers.add(tensor.values?.name);
  }

  return {
    name: graph.name || fileName,
    nodes: (graph.node ?? []).map((node, index) => {
      if (!node.opType) {
        throw new Error(`Node ${index + 1} has no ONNX operator type.`);
      }
      return {
        id: `node-${index}`,
        name: node.name || `${node.opType}_${index + 1}`,
        opType: node.opType,
        domain: node.domain || "ai.onnx",
        inputs: (node.input ?? []).filter(Boolean),
        outputs: (node.output ?? []).filter(Boolean),
      };
    }),
    inputs: (graph.input ?? [])
      .map((input) => input.name || "")
      .filter((name) => name && !initializers.has(name)),
    outputs: (graph.output ?? [])
      .map((output) => output.name || "")
      .filter(Boolean),
    initializerCount: initializers.size,
  };
}
