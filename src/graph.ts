/**
 * Snapshot contracts shared by the native addon, extension host and webview.
 * These describe the view, not the model's tensor payloads.
 * Native calls are synchronous and report parsing/analysis failures as errors.
 */

/** Display-ready attribute preview, rather than an editable ONNX value. */
export interface GraphParameter {
  name: string;
  type: string;
  value: string;
}

/** Explicit node attribute or schema default, optionally linking child graphs. */
export interface GraphAttribute extends GraphParameter {
  /** True for a schema-supplied default, false for a stored node attribute. */
  default: boolean;
  graphs?: SubgraphLink[];
}

/**
 * Zero-based native node/attribute indices within the current graph scope.
 * graph is zero for a GRAPH attribute or the item index in a GRAPHS attribute.
 */
export interface SubgraphStep {
  node: number;
  attribute: number;
  graph: number;
}

/** Call-site node index and the index of its model-local function definition. */
export interface FunctionStep {
  node: number;
  function: number;
}

export type GraphStep = SubgraphStep | FunctionStep;

/**
 * Absolute path from the model's root graph; an empty path selects the root.
 * Steps enter graph attributes or the local function called by a node.
 * Indices, rather than names, distinguish potentially duplicate nodes/graphs.
 */
export type GraphPath = GraphStep[];

/** Named ancestor (including the current graph) with an absolute navigation path. */
export interface GraphBreadcrumb {
  name: string;
  path: GraphPath;
}

/** Lightweight child-graph reference; does not recursively include its nodes. */
export interface SubgraphLink extends GraphBreadcrumb {
  nodeCount: number;
}

/** Storage description only; the viewer does not open external weight files. */
export interface InitializerStorage {
  external: boolean;
  /** External-data location when present, otherwise null. */
  location: string | null;
}

/** Dense or sparse weight metadata with bounded previews of tiny inline tensors. */
export interface GraphInitializer {
  name: string;
  type: string;
  shape: string[];
  storage: InitializerStorage;
  /** Present for sparse initializers, describing their indices tensor. */
  indicesStorage?: InitializerStorage;
  /** Inline values (<16 elements per dense tensor or sparse component), with exact integers. */
  value?: string;
}

/** Newest matching operator schema not newer than the model's imported opset. */
export interface OperatorSchema {
  sinceVersion: number;
  doc: string;
  defaults: GraphAttribute[];
}

/** One operator in the selected graph, with resolved schema and weight metadata. */
export interface GraphNode {
  /** Graph-local node index encoded as node-N; not the optional ONNX node name. */
  id: string;
  name: string;
  opType: string;
  domain: string;
  inputs: string[];
  /** Implicit dependencies from nested graph attributes, excluding explicit inputs. */
  captures: string[];
  outputs: string[];
  /** Imported domain opset as text, or null when the domain has no import. */
  opset: string | null;
  attributes: GraphAttribute[];
  parameters: GraphInitializer[];
  /** Null for local function calls or when no native operator schema is available. */
  schema: OperatorSchema | null;
  /** Local function called by this node, matched by domain, name and overload. */
  function?: SubgraphLink;
}

/** Tensor labels for ports and intermediate values, without tensor data. */
export interface TensorInformation {
  name: string;
  type: string | null;
  /**
   * Dimensions stay strings to preserve symbolic names. Empty for a scalar,
   * null for unknown rank, omitted for non-tensor types.
   */
  shape?: string[] | null;
}

/** Display snapshot of one graph, not a recursive serialization of the model. */
export interface Graph {
  name: string;
  breadcrumbs: GraphBreadcrumb[];
  nodes: GraphNode[];
  inputs: string[];
  outputs: string[];
  initializerCount: number;
  tensors: TensorInformation[];
}

/**
 * Retained native model whose inferred information survives graph projections.
 * Methods neither save the model nor execute its operators.
 */
export interface NativeModel {
  /**
   * Projects one graph. When requested, inference first updates the entire
   * retained model in memory, including nested graph value_info.
   * @param inferShapes Defaults to false; true runs model-wide shape inference.
   * @param path Absolute graph path; omitted or empty selects the root.
   * @returns A fresh display snapshot of the selected graph.
   */
  graph(inferShapes?: boolean, path?: GraphPath): Graph;
  /**
   * Recomputes constants, tags and lifetimes without inferring shapes or editing
   * the model. This is separate from the graph snapshot.
   * @param path Absolute graph path; omitted or empty selects the root.
   */
  metadata(path?: GraphPath): GraphMetadata;
}

/** Stored ONNX metadata property, preserved as a key/value pair. */
export interface MetadataEntry {
  key: string;
  value: string;
}

/** Stored properties and native constant/tag analysis for a named value. */
export interface ValueMetadata {
  name: string;
  entries: MetadataEntry[];
  constant: boolean;
  tag: string | null;
}

/** Analysis for a node identified by the same graph-local ID as GraphNode. */
export interface NodeMetadata {
  id: string;
  entries: MetadataEntry[];
  constant: boolean;
  tag: string | null;
  /** Intermediate values releasable after this node. */
  releaseAfter: string[];
  /** Caller-owned inputs/weights reaching their last read, not to be released. */
  lastUse: string[];
  /** Unused results that are not graph outputs. */
  unusedOutputs: string[];
}

/** Read-only analysis snapshot for the selected graph, including model properties. */
export interface GraphMetadata {
  model: MetadataEntry[];
  graph: MetadataEntry[];
  nodes: NodeMetadata[];
  values: ValueMetadata[];
}

/** Catalog entry for a bundled native backend case; listing does not run it. */
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

/** C++ library release embedded in the addon, independent of the loaded model. */
export const onnxLightVersion = native.onnxLightVersion;

/** Minimal Relu snapshot with native schema documentation, requiring no model file. */
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
 * @param bytes Serialized ONNX ModelProto bytes.
 * @param fileName Display label only; no file is opened or saved.
 * @param inferShapes Whether to infer shapes before projecting the root graph.
 * @returns Root graph snapshot without a retained native handle.
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
 * @param bytes Serialized ONNX ModelProto bytes kept alive by the native handle.
 * @param fileName Display label used for the model.
 * @returns A handle for repeated graph projection and analysis.
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

/** Lists cases sorted by name, preserving native IDs even for duplicate names. */
export function listBackendTests(): BackendTest[] {
  return native
    .listBackendTests()
    .sort((left, right) => left.name.localeCompare(right.name));
}

/**
 * Loads a catalog case without executing it or generating expected outputs.
 * @param id Native collection ID from listBackendTests, not its sorted position.
 * @returns A retained model backed by the native test case.
 */
export function loadBackendTest(id: number): NativeModel {
  return native.loadBackendTest(id);
}
