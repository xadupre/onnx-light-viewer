export interface GraphParameter {
  name: string;
  type: string;
  value: string;
}

export interface GraphAttribute extends GraphParameter {
  default: boolean;
}

export interface InitializerStorage {
  external: boolean;
  location: string | null;
}

export interface GraphInitializer {
  name: string;
  type: string;
  shape: string[];
  storage: InitializerStorage;
  indicesStorage?: InitializerStorage;
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
  parameters: GraphInitializer[];
  schema: OperatorSchema | null;
}

export interface TensorInformation {
  name: string;
  type: string | null;
  shape?: string[] | null;
}

export interface Graph {
  name: string;
  nodes: GraphNode[];
  inputs: string[];
  outputs: string[];
  initializerCount: number;
  tensors: TensorInformation[];
}

export interface NativeModel {
  graph(inferShapes?: boolean): Graph;
  metadata(): GraphMetadata;
}

export interface MetadataEntry {
  key: string;
  value: string;
}

export interface ValueMetadata {
  name: string;
  entries: MetadataEntry[];
  constant: boolean;
  tag: string | null;
}

export interface NodeMetadata {
  id: string;
  entries: MetadataEntry[];
  constant: boolean;
  tag: string | null;
  releaseAfter: string[];
  lastUse: string[];
  unusedOutputs: string[];
}

export interface GraphMetadata {
  model: MetadataEntry[];
  graph: MetadataEntry[];
  nodes: NodeMetadata[];
  values: ValueMetadata[];
}

export interface BackendTest {
  id: number;
  name: string;
  modelName: string;
  kind: string;
  tag: string;
}

const native: {
  decodeGraph(bytes: Uint8Array, fileName: string, inferShapes: boolean): Graph;
  getSchema(
    opType: string,
    domain: string,
    opset: number,
  ): OperatorSchema | null;
  loadModel(bytes: Uint8Array, fileName: string): NativeModel;
  listBackendTests(): BackendTest[];
  loadBackendTest(id: number): NativeModel;
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
  tensors: [
    { name: "input", type: "FLOAT", shape: ["1", "3"] },
    { name: "output", type: "FLOAT", shape: ["1", "3"] },
  ],
};

export function decodeGraph(
  bytes: Uint8Array,
  fileName: string,
  inferShapes = false,
): Graph {
  return native.decodeGraph(bytes, fileName, inferShapes);
}

export function loadModel(bytes: Uint8Array, fileName: string): NativeModel {
  return native.loadModel(bytes, fileName);
}

export function listBackendTests(): BackendTest[] {
  return native
    .listBackendTests()
    .sort((left, right) => left.name.localeCompare(right.name));
}

export function loadBackendTest(id: number): NativeModel {
  return native.loadBackendTest(id);
}
