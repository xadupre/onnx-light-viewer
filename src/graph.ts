export interface GraphParameter {
  name: string;
  type: string;
  value: string;
}

export interface GraphAttribute extends GraphParameter {
  default: boolean;
}

export interface OperatorSchema {
  sinceVersion: number;
  doc: string;
  defaults: GraphAttribute[];
}

export interface GraphNode {
  id: string;
  name: string;
  opType: string;
  domain: string;
  inputs: string[];
  outputs: string[];
  opset: string | null;
  attributes: GraphAttribute[];
  parameters: GraphParameter[];
  schema: OperatorSchema | null;
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
  getSchema(
    opType: string,
    domain: string,
    opset: number,
  ): OperatorSchema | null;
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
      opset: "13",
      attributes: [],
      parameters: [],
      schema: native.getSchema("Relu", "ai.onnx", 13),
    },
  ],
  inputs: ["input"],
  outputs: ["output"],
  initializerCount: 0,
};

export function decodeGraph(bytes: Uint8Array, fileName: string): Graph {
  return native.decodeGraph(bytes, fileName);
}
