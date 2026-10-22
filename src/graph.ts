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

const native: {
  decodeGraph(bytes: Uint8Array, fileName: string): Graph;
} = require("./native/onnx_light_viewer.node");

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
  return native.decodeGraph(bytes, fileName);
}
