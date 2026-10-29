/**
 * Snapshot contracts shared by the native addon, extension host and webview.
 * These describe the view, not the model's tensor payloads.
 * Native calls are synchronous and report parsing/analysis failures as errors.
 */
import { PipelineModel, PipelineData } from "./pipeline";

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

/** Model-local function index, optionally tied to a call-site node. */
export interface FunctionStep {
  node?: number;
  function: number;
}

export type PipelineRole = "embedding" | "prefill" | "decode";

export interface PipelineStep {
  pipeline: PipelineRole;
}

export type GraphStep = SubgraphStep | FunctionStep | PipelineStep;

/**
 * Absolute path from the model's root graph; an empty path selects the root.
 * Steps enter a pipeline stage, graph attribute or model-local function.
 * Stage roles identify contained models; indices distinguish duplicate nodes/graphs.
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

/** Storage description; inspection does not open external weight files. */
export interface InitializerStorage {
  external: boolean;
  /** External-data location when present, otherwise null. */
  location: string | null;
}

/** Bounded, display-ready rows for a logical type or physical encoding layout. */
export interface GraphTypeDetails {
  kind: string;
  /** Parent indices precede their children; data previews never dequantize weights. */
  entries: (MetadataEntry & {
    parent?: number;
    shape?: string;
    data?: string;
  })[];
}

/** Metadata-only logical tensor size, serialized as exact decimal integers. */
export interface TensorDistribution {
  kind: "dense" | "sparse" | "encoded";
  elementCount: string | null;
  /** Dense-equivalent size; null for unknown shapes or variable-width types. */
  logicalSizeBytes: string | null;
}

/** Dense, sparse or encoded weight metadata with bounded previews. */
export interface GraphInitializer {
  name: string;
  type: string;
  shape?: string[] | null;
  /** EncodedValueProto layout description, resolving model-scoped struct references. */
  encoding?: string;
  typeDetails?: GraphTypeDetails;
  encodingDetails?: GraphTypeDetails;
  storage: InitializerStorage;
  /** Present for sparse initializers, describing their indices tensor. */
  indicesStorage?: InitializerStorage;
  distribution?: TensorDistribution;
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
  /** Stored Constant outputs, without reading or decoding their payloads. */
  storedOutputs?: (TensorInformation & { distribution: TensorDistribution })[];
  /** Null for local function calls or when no native operator schema is available. */
  schema: OperatorSchema | null;
  /** Local function called by this node, matched by domain, name and overload. */
  function?: SubgraphLink;
  /** Summary of a contained model represented by a pipeline overview box, not an operator. */
  pipelineModel?: GraphSummary;
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
  /** Declared imports, including unused domains; functions have their own imports. */
  opsets: { domain: string; version: string }[];
  /** Pipeline container or one of its stage graphs. */
  pipeline?: boolean;
  /** Persistent links to every model present in a pipeline. */
  pipelineStages?: GraphBreadcrumb[];
  /** Overrides root-only fusion availability for pipeline stage roots. */
  canFuse?: boolean;
  breadcrumbs: GraphBreadcrumb[];
  nodes: GraphNode[];
  /** Definitions listed at the model root, including functions with no call sites. */
  localFunctions: SubgraphLink[];
  inputs: string[];
  outputs: string[];
  initializerCount: number;
  /** Initializers declared in this graph, including unused ones, excluding outer captures. */
  initializers: GraphInitializer[];
  sparseInitializers: GraphInitializer[];
  encodedInitializers: GraphInitializer[];
  tensors: TensorInformation[];
}

/** Root ports and counts without projecting individual operators, attributes or schemas. */
export type GraphSummary = Pick<
  Graph,
  "name" | "inputs" | "outputs" | "initializerCount" | "tensors" | "opsets"
> & {
  nodeCount: number;
  functionCount?: number;
  denseInitializerCount?: number;
  sparseInitializerCount?: number;
  encodedInitializerCount?: number;
  operatorTypes?: { domain: string; opType: string; count: number }[];
  statisticsAvailable?: boolean;
};

export interface FusionMatch {
  /** Global passes transform the entire scope and cannot be combined with other candidates. */
  global?: boolean;
  anchor: number;
  nodes: number[];
  replacement: {
    opType: string;
    domain: string;
    name: string;
    inputs: string[];
    outputs: string[];
    attributes: { name: string; value: string }[];
  }[];
}

export interface FusionOperation {
  matches: { pattern: string; nodes: number[] }[];
  stage?: PipelineRole;
}

export interface OptimizationPattern {
  name: string;
  priority: number;
  device: string;
  enabledByDefault: boolean;
  groups: string[];
}

export interface FusionDiagnostic {
  pattern: string;
  anchor: number;
  candidate?: number;
  status: "matched" | "rejected" | "unavailable" | "op-type-mismatch";
  reason: string;
  source?: { file: string; line: number };
}

export interface OptimizationReport {
  iterations: number;
  rewrites: number;
  totalTimeNs: number;
  wallTimeNs: number;
  matchingTimeNs: number;
  rewritingTimeNs: number;
  cleanupTimeNs: number;
  constantFoldingTimeNs: number;
  subgraphOptimizationTimeNs: number;
  patterns: {
    name: string;
    attempts: number;
    matches: number;
    matchTimeNs: number;
    applyTimeNs: number;
    noMatches: {
      reason: string;
      sourceFile: string;
      sourceLine: number;
      occurrences: number;
    }[];
  }[];
  subgraphs: {
    path: string[];
    iterations: number;
    rewrites: number;
    elapsedTimeNs: number;
  }[];
}

export type OptimizationOperation =
  "Optimize" | "IdentityRemoval" | "ConstantFolding" | "RemoveUnusedNodes";

export interface MemoryDimension {
  key: string;
  label: string;
  defaultValue: number;
}

export interface MemoryBreakdown<T extends number | string> {
  total_bytes: T;
  already_allocated_bytes: T;
  output_allocation_bytes: T;
  before_bytes: T;
  during_bytes: T;
  after_bytes: T;
  released_bytes: T;
  initializers: Record<string, T>;
  inputs: Record<string, T>;
  intermediates: Record<string, T>;
  outputs: Record<string, T>;
}

export interface MemoryProfile {
  name: string;
  device: string;
  peakBytes: number | null;
  events: {
    index: number;
    node: string;
    opType: string;
    existingBytes: number;
    outputBytes: number;
    scratchBytes: number;
    deviceBytes: number;
    /** SDK post-release snapshot, including the final node. */
    afterBytes: number | null;
    memory: MemoryBreakdown<number>;
    /** SDK expressions on the original symbolic shapes; scratch is not symbolic. */
    symbolicMemory: MemoryBreakdown<string>;
    decisions: {
      action: "inplace" | "release" | "release_shape_tag";
      name: string;
      input?: string;
      output?: string;
      kind?: "equal" | "greater";
    }[];
    inputs: {
      name: string;
      type: string | null;
      shape: string | null;
      tag: string | null;
    }[];
    outputs: {
      name: string;
      type: string | null;
      shape: string | null;
      tag: string | null;
    }[];
  }[];
}

export interface RewritingScenario {
  name: string;
  steps: { pattern: string; path: string[]; nodes: number[] }[];
  applied: number;
  available: boolean;
}

export type StatisticsNumber = number | "NaN" | "Infinity" | "-Infinity";

export interface TensorValues {
  name: string;
  dtype: string;
  shape: string[];
  count: string;
  sizeBytes: string;
  values: string;
}

export type TensorInspection = TensorStatistics | TensorValues;

/** SDK statistics; integer metadata and nonfinite numbers survive webview transport exactly. */
export interface TensorStatistics {
  name: string;
  dtype: string;
  shape: string[];
  count: string;
  sizeBytes: string;
  min: StatisticsNumber;
  max: StatisticsNumber;
  mean: StatisticsNumber;
  median: StatisticsNumber;
  /** Quartiles at 0.25, 0.5 and 0.75, using linear interpolation. */
  quantiles: StatisticsNumber[];
  distinctCount?: string;
  distribution: string[];
  binEdges: StatisticsNumber[];
  rectangleMap: StatisticsNumber[];
  rectangleShape: number[];
}

/**
 * Retained model or pipeline whose inferred information survives graph projections.
 * Methods do not execute the model or write files.
 */
export interface NativeModel {
  /** Inspects stored data without execution; optionally returns values instead of statistics below 32 elements. */
  tensorStatistics(
    name: string,
    path?: GraphPath,
    valuesIfSmall?: boolean,
  ): Promise<TensorInspection>;
  memoryProfileInputs(path?: GraphPath): MemoryDimension[];
  /** Estimates CPU memory on an independent snapshot, without executing kernels. */
  memoryProfile(
    dimensions: Record<string, number>,
    path?: GraphPath,
  ): Promise<MemoryProfile>;
  clone(): NativeModel;
  /** Number of applied Fuse operations available for session-local undo. */
  fusionCount(): number;
  /** Applied matches and their surviving descendants in the current root graph. */
  fusionHistory(): FusionOperation[];
  /** Returns an independent snapshot before the last Fuse operation. */
  undoFusion(): NativeModel;
  rewritingScenarios(path?: GraphPath): RewritingScenario[];
  applyRewritingScenario(
    name: string,
    all: boolean,
    sourceFilePath?: string,
    path?: GraphPath,
  ): Promise<NativeModel>;
  summary(inferShapes?: boolean): GraphSummary;
  /** Pipeline fusions require the path of a stage root; ordinary models use []. */
  fusionPatterns(path?: GraphPath): string[];
  optimizationPatterns(path?: GraphPath): OptimizationPattern[];
  /** Native report for the last optimization of this snapshot, including no-op runs. */
  optimizationReport(path?: GraphPath): OptimizationReport | null;
  /** Inspects one native pattern/node without applying it or collecting other failures. */
  explainFusion(
    pattern: string,
    anchor: number,
    sourceFilePath?: string,
    path?: GraphPath,
  ): Promise<FusionDiagnostic>;
  /** Runs a global pass or the native iterative optimizer on an independent snapshot. */
  optimize(
    patterns: string[],
    operation: OptimizationOperation,
    sourceFilePath?: string,
    path?: GraphPath,
  ): Promise<NativeModel>;
  previewFusion(
    pattern: string,
    sourceFilePath?: string,
    path?: GraphPath,
  ): Promise<{ model: NativeModel; matches: FusionMatch[] }>;
  /** A pattern array assigns one pattern to each corresponding anchor. */
  applyFusion(
    pattern: string | string[],
    anchors: number[],
    path?: GraphPath,
  ): Promise<NativeModel>;
  /** Whole-model external-data references, excluding omitted inline payload placeholders. */
  externalDataLocations(): string[];
  /**
   * Serializes the whole model or pipeline, including inferred information and weights.
   * A password produces authenticated ONNXCRY3 (AES-256-GCM), otherwise plain ONNX.
   * External weights are read relative to the source model and embedded before saving.
   * sourceFilePath supplies the new location after a source file is moved into a cache.
   * includeRewritings saves the canonical pre-fusion model and a concatenated plan.
   * Otherwise the materialized current graph is saved without rewriting variants.
   */
  serialize(
    password?: string,
    sourceFilePath?: string,
    includeRewritings?: boolean,
  ): Promise<Uint8Array>;
  /** Partitions a plain copy; optionally retains unchanged source external references. */
  serializeFiles(
    maxFileSize: number,
    prefix: string,
    sourceFilePath?: string,
    reuseExternalWeights?: boolean,
    includeRewritings?: boolean,
  ): Promise<{
    model: Uint8Array;
    weights: { name: string; bytes: Uint8Array }[];
  }>;
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
  /** Checks the entire retained model, or every pipeline stage, without editing it. */
  verify(): void;
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
  loadModel(
    bytes: Uint8Array,
    fileName: string,
    password?: string,
  ): NativeModel;
  loadModelFile(
    filePath: string,
    fileName: string,
    password?: string,
  ): Promise<NativeModel>;
  loadPipeline(bytes: Uint8Array, fileName: string): PipelineData;
  loadPipelineFile(filePath: string, fileName: string): Promise<PipelineData>;
  listBackendTests(): BackendTest[];
  loadBackendTest(id: number): NativeModel;
} = require("./native/onnx_light_viewer.node");

/** C++ library release embedded in the addon, independent of the loaded model. */
export const onnxLightVersion = native.onnxLightVersion;

/** Minimal Relu snapshot with native schema documentation, requiring no model file. */
export const demoGraph: Graph = {
  name: "A first ONNX graph",
  opsets: [{ domain: "ai.onnx", version: "13" }],
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
  localFunctions: [],
  initializers: [],
  sparseInitializers: [],
  encodedInitializers: [],
  tensors: [
    { name: "input", type: "FLOAT", shape: ["1", "3"] },
    { name: "output", type: "FLOAT", shape: ["1", "3"] },
  ],
};

/**
 * One-shot projection, skipping raw tensor bytes unless inference is requested.
 * Use loadModel instead when inferred value_info must survive subsequent calls.
 * @param bytes Serialized ONNX ModelProto or PipelineProto bytes.
 * @param fileName Display label; a .pipe-onnx suffix selects PipelineProto.
 * @param inferShapes Whether to infer shapes before projecting the root graph.
 * @returns Root graph snapshot without a retained native handle.
 */
export function decodeGraph(
  bytes: Uint8Array,
  fileName: string,
  inferShapes = false,
): Graph {
  if (/\.pipe-onnx$/i.test(fileName))
    return loadModel(bytes, fileName).graph(inferShapes);
  return native.decodeGraph(bytes, fileName, inferShapes);
}

/**
 * ModelProto borrows tensor payloads; PipelineProto currently owns parsed copies.
 * Do not mutate borrowed buffers while the handle is in use.
 * @param bytes Serialized ONNX ModelProto or PipelineProto bytes.
 * @param fileName Display label; a .pipe-onnx suffix selects PipelineProto, not a file to open.
 * @returns A handle for repeated graph projection and analysis.
 */
export function loadModel(
  bytes: Uint8Array,
  fileName: string,
  password?: string,
): NativeModel {
  if (/\.pipe-onnx$/i.test(fileName)) {
    if (password !== undefined)
      throw new Error(
        "PipelineProto encryption is not supported by onnx-light.",
      );
    return new PipelineModel(native.loadPipeline(bytes, fileName), fileName);
  }
  return native.loadModel(bytes, fileName, password);
}

/** Streams plain files with bounded payloads, or decrypts a complete encrypted file off-thread. */
export function loadModelFile(
  filePath: string,
  fileName: string,
  password?: string,
): Promise<NativeModel> {
  if (/\.pipe-onnx$/i.test(fileName)) {
    if (password !== undefined)
      return Promise.reject(
        new Error("PipelineProto encryption is not supported by onnx-light."),
      );
    return native
      .loadPipelineFile(filePath, fileName)
      .then((pipeline) => new PipelineModel(pipeline, fileName));
  }
  return native.loadModelFile(filePath, fileName, password);
}

/** Recognizes onnx-light's three encrypted ModelProto envelopes from an eight-byte header. */
export function isEncryptedModel(bytes: Uint8Array): boolean {
  return (
    bytes.length >= 8 &&
    /^ONNXCRY[123]$/.test(String.fromCharCode(...bytes.subarray(0, 8)))
  );
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
