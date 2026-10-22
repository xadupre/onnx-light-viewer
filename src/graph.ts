/**
 * Snapshot contracts shared by the native addon, extension host and webview.
 * These describe the view, not the model's tensor payloads.
 */
export interface GraphParameter {
  name: string;
  type: string;
  value: string;
}

export interface GraphAttribute extends GraphParameter {
  default: boolean;
  graphs?: SubgraphLink[];
}

/** Native node/attribute indices plus the graph index within that attribute. */
export interface GraphStep {
  node: number;
  attribute: number;
  graph: number;
}

/** Absolute path from the model's root graph; an empty path selects the root. */
export type GraphPath = GraphStep[];

export interface GraphBreadcrumb {
  name: string;
  path: GraphPath;
}

export interface SubgraphLink extends GraphBreadcrumb {
  nodeCount: number;
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
  captures: string[];
  outputs: string[];
  opset: string | null;
  attributes: GraphAttribute[];
  parameters: GraphInitializer[];
  schema: OperatorSchema | null;
}

export interface TensorInformation {
  name: string;
  type: string | null;
  /** Empty for a scalar, null for unknown rank, omitted for non-tensor types. */
  shape?: string[] | null;
}

export interface Graph {
  name: string;
  breadcrumbs: GraphBreadcrumb[];
  nodes: GraphNode[];
  inputs: string[];
  outputs: string[];
  initializerCount: number;
  tensors: TensorInformation[];
}

export interface NativeModel {
  /**
   * Projects one graph. When requested, inference first updates the entire
   * retained model in memory, including nested graph value_info.
   */
  graph(inferShapes?: boolean, path?: GraphPath): Graph;
  /** Recomputes analysis for one graph without inferring shapes or editing it. */
  metadata(path?: GraphPath): GraphMetadata;
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
  /** Original native collection index, independent of name sorting. */
  id: number;
  name: string;
  modelName: string;
  kind: string;
  tag: string;
}

const native: {
  onnxLightVersion: string;
  decodeGraph(bytes: Uint8Array, fileName: string, inferShapes: boolean): Graph;
  getSchema(
    opType: string,
    domain: string,
    opset: number,
  ): OperatorSchema | null;
  loadModel(bytes: Uint8Array, fileName: string): NativeModel;
  loadModelFile(filePath: string, fileName: string): Promise<NativeModel>;
  listBackendTests(): BackendTest[];
  loadBackendTest(id: number): NativeModel;
} = require("./native/onnx_light_viewer.node");

export const onnxLightVersion = native.onnxLightVersion;

export const demoGraph: Graph = {
  name: "A first ONNX graph",
  breadcrumbs: [{ name: "A first ONNX graph", path: [] }],
  nodes: [
    {
      id: "node-0",
      name: "activation",
      opType: "Relu",
      domain: "ai.onnx",
      inputs: ["input"],
      captures: [],
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

/**
 * One-shot projection, skipping raw tensor bytes unless inference is requested.
 * Use loadModel instead when inferred value_info must survive subsequent calls.
 */
export function decodeGraph(
  bytes: Uint8Array,
  fileName: string,
  inferShapes = false,
): Graph {
  return native.decodeGraph(bytes, fileName, inferShapes);
}

/**
 * Retains bytes in the native handle without copying their tensor payloads.
 * Do not mutate the buffer while the handle is in use. fileName is a display
 * label, not a path to read or save.
 */
export function loadModel(bytes: Uint8Array, fileName: string): NativeModel {
  return native.loadModel(bytes, fileName);
}

/** Streams a local file off the host thread, skipping large tensor payloads. */
export function loadModelFile(
  filePath: string,
  fileName: string,
): Promise<NativeModel> {
  return native.loadModelFile(filePath, fileName);
}

export function listBackendTests(): BackendTest[] {
  return native
    .listBackendTests()
    .sort((left, right) => left.name.localeCompare(right.name));
}

export function loadBackendTest(id: number): NativeModel {
  return native.loadBackendTest(id);
}
