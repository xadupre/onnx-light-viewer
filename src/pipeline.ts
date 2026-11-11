import type {
  DimensionEquality,
  DimensionMerge,
  DimensionRename,
  FusionDiagnostic,
  FusionMatch,
  FusionOperation,
  FunctionPatternDefinition,
  Graph,
  GraphMetadata,
  GraphPath,
  GraphSummary,
  MetadataEntry,
  MemoryDimension,
  MemoryProfile,
  ModelFunctionDescriptor,
  NativeModel,
  OptimizationOperation,
  OptimizationPattern,
  OptimizationReport,
  PipelineRole,
  RewritingScenario,
  RunResult,
  TensorInspection,
} from "./graph";
import { updateModelFunctionHistory } from "./graph";

export interface PipelineData {
  name: string;
  metadata: MetadataEntry[];
  models: Partial<Record<PipelineRole, NativeModel>>;
  assemble(
    models: Partial<Record<PipelineRole, Uint8Array>>,
  ): Promise<Uint8Array>;
}

const roles: PipelineRole[] = ["embedding", "prefill", "decode"];

/** Keeps stage models independent while saving the complete container. */
export class PipelineModel implements NativeModel {
  releaseRunSession = () => {
    for (const [, model] of this.stages()) model.releaseRunSession();
  };
  runInputs = (path: GraphPath = []) => {
    const stage = this.select(path);
    if (stage.path.length) throw new Error("Choose a pipeline stage root.");
    return stage.model.runInputs();
  };

  run = (
    inputs: Parameters<NativeModel["run"]>[0],
    path: GraphPath = [],
    options?: Parameters<NativeModel["run"]>[2],
  ): Promise<RunResult> => {
    const stage = this.select(path);
    if (stage.path.length) throw new Error("Choose a pipeline stage root.");
    return stage.model.run(inputs, [], options);
  };

  prepareRun = (
    inputs: Parameters<NativeModel["run"]>[0],
    path: GraphPath = [],
    options?: Parameters<NativeModel["run"]>[2],
  ) => {
    const stage = this.select(path);
    if (stage.path.length) throw new Error("Choose a pipeline stage root.");
    return stage.model.prepareRun(inputs, [], options);
  };

  tensorStatistics = (
    name: string,
    path: GraphPath = [],
    valuesIfSmall = false,
  ): Promise<TensorInspection> => {
    const stage = this.select(path);
    return stage.model.tensorStatistics(name, stage.path, valuesIfSmall);
  };

  memoryProfileInputs = (path: GraphPath = []): MemoryDimension[] => {
    const stage = this.select(path);
    if (stage.path.length) throw new Error("Choose a pipeline stage root.");
    return stage.model.memoryProfileInputs();
  };

  memoryProfile = (
    dimensions: Record<string, number>,
    path: GraphPath = [],
  ): Promise<MemoryProfile> => {
    const stage = this.select(path);
    if (stage.path.length) throw new Error("Choose a pipeline stage root.");
    return stage.model.memoryProfile(dimensions);
  };
  constructor(
    private readonly data: PipelineData,
    private readonly fileName: string,
    private readonly models = data.models,
    private readonly previous?: PipelineModel,
    private readonly fusionOrder: PipelineRole[] = [],
    private readonly undoDimensionEdits: readonly (
      string | readonly DimensionEquality[]
    )[] = [],
  ) {}

  private stages(): [PipelineRole, NativeModel][] {
    return roles.flatMap((role): [PipelineRole, NativeModel][] => {
      const model = this.models[role];
      return model ? [[role, model]] : [];
    });
  }

  private select(path: GraphPath): {
    role: PipelineRole;
    model: NativeModel;
    path: GraphPath;
  } {
    const first = path[0];
    if (!first || !("pipeline" in first) || !roles.includes(first.pipeline))
      throw new Error("Choose a pipeline stage.");
    const model = this.models[first.pipeline];
    if (!model || path.slice(1).some((step) => "pipeline" in step))
      throw new Error("Invalid pipeline stage path.");
    return { role: first.pipeline, model, path: path.slice(1) };
  }

  private replace(role: PipelineRole, model: NativeModel): PipelineModel {
    return new PipelineModel(
      this.data,
      this.fileName,
      Object.fromEntries(
        this.stages().map(([name, current]) => [
          name,
          name === role ? model : current.clone(),
        ]),
      ),
      this.previous,
      this.fusionOrder,
      this.undoDimensionEdits,
    );
  }

  clone = (): PipelineModel => {
    return new PipelineModel(
      this.data,
      this.fileName,
      Object.fromEntries(
        this.stages().map(([role, model]) => [role, model.clone()]),
      ),
      this.previous,
      this.fusionOrder,
      this.undoDimensionEdits,
    );
  };

  dimensionPrefix = (): string | null => {
    const prefixes = this.stages()
      .map(([, model]) => model.dimensionPrefix())
      .filter((prefix): prefix is string => prefix !== null);
    if (!prefixes.length) return null;
    let prefix = prefixes[0];
    for (const candidate of prefixes) {
      let length = 0;
      while (
        length < prefix.length &&
        length < candidate.length &&
        prefix[length] === candidate[length]
      )
        length++;
      prefix = prefix.slice(0, length);
    }
    const boundary = Math.max(
      ...[".", "/", ":", "_"].map((separator) => prefix.lastIndexOf(separator)),
    );
    return prefix.slice(0, boundary + 1);
  };

  symbolicDimensions = (): string[] =>
    [
      ...new Set(
        this.stages().flatMap(([, model]) => model.symbolicDimensions()),
      ),
    ].sort();

  mergeDimensions: (
    source: string,
    target: string,
  ) => DimensionMerge<PipelineModel> = (source, target) => {
    return this.mergeDimensionsBatch([{ source, target }]);
  };

  mergeDimensionsBatch: (
    equalities: readonly DimensionEquality[],
  ) => DimensionMerge<PipelineModel> = (equalities) => {
    const edits = equalities.map(({ source, target }) => ({ source, target }));
    let count = 0;
    const models: Partial<Record<PipelineRole, NativeModel>> = {};
    for (const [role, model] of this.stages()) {
      const edited = model.mergeDimensionsBatch(edits);
      models[role] = edited.model;
      count += edited.count;
    }
    return {
      model: new PipelineModel(
        this.data,
        this.fileName,
        models,
        this.previous,
        this.fusionOrder,
        count ? [...this.undoDimensionEdits, edits] : this.undoDimensionEdits,
      ),
      count,
    };
  };

  shortenDimensionNames: (prefix?: string) => DimensionRename<PipelineModel> = (
    prefix = this.dimensionPrefix() ?? "",
  ) => {
    let count = 0;
    const models: Partial<Record<PipelineRole, NativeModel>> = {};
    for (const [role, model] of this.stages()) {
      const edited = model.shortenDimensionNames(prefix);
      models[role] = edited.model;
      count += edited.count;
    }
    return {
      model: count
        ? new PipelineModel(
            this.data,
            this.fileName,
            models,
            this.previous,
            this.fusionOrder,
            [...this.undoDimensionEdits, prefix],
          )
        : this.clone(),
      prefix,
      count,
    };
  };

  fusionCount = (): number =>
    this.stages().reduce((count, [, model]) => count + model.fusionCount(), 0);

  fusionHistory = (includeImpact = false): FusionOperation[] => {
    const stages = new Map(
      this.stages().map(([role, model]) => [
        role,
        model.fusionHistory(includeImpact),
      ]),
    );
    const counts = new Map<PipelineRole, number>();
    return this.fusionOrder.map((stage) => {
      const index = counts.get(stage) ?? 0;
      counts.set(stage, index + 1);
      const operation = stages.get(stage)?.[index];
      if (!operation) throw new Error("Invalid pipeline fusion history.");
      return { ...operation, stage };
    });
  };

  undoFusion = (): PipelineModel => {
    if (!this.previous) throw new Error("No fusion operation to undo.");
    let restored = this.previous.clone();
    // Independent pipeline snapshots are renamed only when restored.
    for (const edit of this.undoDimensionEdits)
      restored =
        typeof edit === "string"
          ? restored.shortenDimensionNames(edit).model
          : restored.mergeDimensionsBatch(edit).model;
    return restored;
  };

  fusionPrefix = async (
    steps: number,
    _sourceFilePath?: string,
  ): Promise<NativeModel> => {
    let current: PipelineModel = this;
    let history = current.fusionHistory();
    let count = history.reduce(
      (sum, operation) => sum + operation.matches.length,
      0,
    );
    if (!Number.isSafeInteger(steps) || steps < 0 || steps > count)
      throw new Error("Invalid rewriting step count.");
    while (count > steps) {
      const last = history.at(-1)!;
      const before = count - last.matches.length;
      if (steps > before) {
        const role = current.fusionOrder.at(-1)!;
        const stage = current.models[role]!;
        const stageCount = stage
          .fusionHistory()
          .reduce((sum, operation) => sum + operation.matches.length, 0);
        // Stages retain their own source paths, not the container's path.
        const edited = await stage.fusionPrefix(stageCount - (count - steps));
        return new PipelineModel(
          current.data,
          current.fileName,
          { ...current.models, [role]: edited },
          current.previous,
          current.fusionOrder,
          current.undoDimensionEdits,
        );
      }
      current = current.undoFusion();
      history = current.fusionHistory();
      count = history.reduce(
        (sum, operation) => sum + operation.matches.length,
        0,
      );
    }
    return current.clone();
  };

  summary = (inferShapes = false): GraphSummary => {
    const {
      name,
      inputs,
      outputs,
      initializerCount,
      tensors,
      nodes,
      irVersion,
      opsets,
    } = this.graph(inferShapes);
    const stages = nodes.flatMap((node) =>
      node.pipelineModel ? [node.pipelineModel] : [],
    );
    return {
      name,
      inputs,
      outputs,
      initializerCount,
      tensors,
      irVersion,
      opsets,
      nodeCount: stages.length
        ? stages.reduce((count, stage) => count + stage.nodeCount, 0)
        : nodes.length,
      ...(stages.length
        ? {
            functionCount: stages.reduce(
              (count, stage) => count + (stage.functionCount ?? 0),
              0,
            ),
          }
        : {}),
    };
  };

  graph = (inferShapes = false, path: GraphPath = []): Graph => {
    const name = this.data.name || this.fileName;
    const pipelineStages = this.stages().map(([role]) => ({
      name: role,
      path: [{ pipeline: role }],
    }));
    if (path.length) {
      const stage = this.select(path);
      const graph = stage.model.graph(inferShapes, stage.path);
      const prefix: GraphPath = [{ pipeline: stage.role }];
      const link = <T extends { path: GraphPath }>(value: T): T => ({
        ...value,
        path: [...prefix, ...value.path],
      });
      return {
        ...graph,
        pipeline: true,
        pipelineStages,
        canFuse: stage.path.length === 0,
        localFunctions: graph.localFunctions.map(link),
        breadcrumbs: [
          { name, path: [] },
          ...graph.breadcrumbs.map((crumb, index) =>
            link({ ...crumb, name: index === 0 ? stage.role : crumb.name }),
          ),
        ],
        nodes: graph.nodes.map((node) => ({
          ...node,
          ...(node.function ? { function: link(node.function) } : {}),
          attributes: node.attributes.map((attribute) => ({
            ...attribute,
            ...(attribute.graphs ? { graphs: attribute.graphs.map(link) } : {}),
          })),
        })),
      };
    }
    const stages = this.stages().map(([role, model]) => ({
      role,
      graph: model.summary(inferShapes),
    }));
    const port = (role: string, direction: string, value: string) =>
      `${role}:${direction}:${value}`;
    const opsets = new Map<string, Graph["opsets"][number]>();
    for (const { graph } of stages)
      for (const opset of graph.opsets)
        opsets.set(JSON.stringify([opset.domain, opset.version]), opset);
    return {
      name,
      irVersion: null,
      opsets: [...opsets.values()],
      pipeline: true,
      pipelineStages,
      canFuse: false,
      localFunctions: [],
      breadcrumbs: [{ name, path: [] }],
      nodes: stages.map(({ role, graph }, index) => ({
        id: `node-${index}`,
        name: role,
        opType: "Model",
        pipelineModel: graph,
        domain: "onnx-light.pipeline",
        inputs: graph.inputs.map((name) => port(role, "input", name)),
        outputs: graph.outputs.map((name) => port(role, "output", name)),
        captures: [],
        opset: null,
        schema: null,
        parameters: [],
        attributes: [
          {
            name: role,
            type: "MODEL",
            value: graph.name,
            default: false,
            graphs: [
              {
                name: role,
                path: [{ pipeline: role }],
                nodeCount: graph.nodeCount,
              },
            ],
          },
        ],
      })),
      inputs: stages.flatMap(({ role, graph }) =>
        graph.inputs.map((name) => port(role, "input", name)),
      ),
      outputs: stages.flatMap(({ role, graph }) =>
        graph.outputs.map((name) => port(role, "output", name)),
      ),
      initializerCount: stages.reduce(
        (count, { graph }) => count + graph.initializerCount,
        0,
      ),
      initializers: [],
      sparseInitializers: [],
      encodedInitializers: [],
      tensors: stages.flatMap(({ role, graph }) =>
        ["input", "output"].flatMap((direction) =>
          (direction === "input" ? graph.inputs : graph.outputs).map((name) => {
            const tensor = graph.tensors.find((tensor) => tensor.name === name);
            return {
              ...tensor,
              name: port(role, direction, name),
              type: tensor?.type ?? null,
            };
          }),
        ),
      ),
    };
  };

  verify = (): void => {
    for (const [, model] of this.stages()) model.verify();
  };

  metadata = (path: GraphPath = []): GraphMetadata => {
    if (path.length) {
      const stage = this.select(path);
      return stage.model.metadata(stage.path);
    }
    return {
      model: this.data.metadata.map((entry) => ({ ...entry })),
      graph: [],
      nodes: [],
      values: [],
    };
  };

  fusionPatterns = (path: GraphPath = []): string[] => {
    const stage = this.select(path);
    if (stage.path.length)
      throw new Error(
        "Return to a pipeline stage root to apply fusion patterns.",
      );
    return stage.model.fusionPatterns();
  };

  createFunctionPattern = (
    nodes: number[],
    name: string,
    path: GraphPath = [],
  ): FunctionPatternDefinition => {
    const stage = this.select(path);
    this.fusionPatterns(path);
    return stage.model.createFunctionPattern(nodes, name);
  };

  setFunctionPatterns = (patterns: FunctionPatternDefinition[]): void => {
    for (const [, model] of this.stages()) model.setFunctionPatterns(patterns);
  };

  updateFunctionPattern = (
    name: string,
    text: string,
    replacement: string,
    path: GraphPath = [],
  ): FunctionPatternDefinition => {
    const stage = this.select(path);
    this.fusionPatterns(path);
    return stage.model.updateFunctionPattern(name, text, replacement);
  };

  optimizationPatterns = (path: GraphPath = []): OptimizationPattern[] => {
    const stage = this.select(path);
    this.fusionPatterns(path);
    return stage.model.optimizationPatterns();
  };

  modelFunctions = (path: GraphPath = []): FunctionPatternDefinition[] => {
    const stage = this.select(path);
    this.fusionPatterns(path);
    return stage.model.modelFunctions();
  };

  modelFunctionDescriptors = (
    path: GraphPath = [],
  ): ModelFunctionDescriptor[] => {
    const stage = this.select(path);
    this.fusionPatterns(path);
    return stage.model.modelFunctionDescriptors();
  };

  modelFunction = (
    functionIndex: number,
    path: GraphPath = [],
  ): FunctionPatternDefinition => {
    const stage = this.select(path);
    this.fusionPatterns(path);
    return stage.model.modelFunction(functionIndex);
  };

  updateModelFunction = (
    functionIndex: number,
    text: string,
    path: GraphPath = [],
  ): NativeModel => {
    const stage = this.select(path);
    this.fusionPatterns(path);
    const snapshots: PipelineModel[] = [];
    for (
      let snapshot: PipelineModel | undefined = this;
      snapshot;
      snapshot = snapshot.previous
    )
      snapshots.push(snapshot);
    const models = snapshots.map((snapshot) => {
      const current = snapshot.models[stage.role];
      if (!current) throw new Error("Invalid pipeline stage.");
      return current;
    });
    const edited = updateModelFunctionHistory(models, functionIndex, text);
    const replace = (
      snapshot: PipelineModel,
      index: number,
      previous?: PipelineModel,
    ) =>
      new PipelineModel(
        snapshot.data,
        snapshot.fileName,
        snapshot.replace(stage.role, edited[index]).models,
        previous,
        snapshot.fusionOrder,
        snapshot.undoDimensionEdits,
      );
    let previous: PipelineModel | undefined;
    for (let i = snapshots.length - 1; i > 0; --i)
      previous = replace(snapshots[i], i, previous);
    return replace(this, 0, previous);
  };

  optimizationReport = (path: GraphPath = []): OptimizationReport | null => {
    const stage = this.select(path);
    this.fusionPatterns(path);
    return stage.model.optimizationReport();
  };

  explainFusion = async (
    pattern: string,
    anchor: number,
    sourceFilePath?: string,
    path: GraphPath = [],
  ): Promise<FusionDiagnostic> => {
    const stage = this.select(path);
    this.fusionPatterns(path);
    return stage.model.explainFusion(pattern, anchor, sourceFilePath);
  };

  optimize = async (
    patterns: string[],
    operation: OptimizationOperation,
    sourceFilePath?: string,
    path: GraphPath = [],
  ): Promise<NativeModel> => {
    const stage = this.select(path);
    this.optimizationPatterns(path);
    const edited = await stage.model.optimize(
      patterns,
      operation,
      sourceFilePath,
    );
    const replacement = this.replace(stage.role, edited);
    if (edited.fusionCount() === stage.model.fusionCount()) return replacement;
    return new PipelineModel(
      this.data,
      this.fileName,
      replacement.models,
      this,
      [...this.fusionOrder, stage.role],
    );
  };

  withoutFusions = (): PipelineModel =>
    new PipelineModel(
      this.data,
      this.fileName,
      Object.fromEntries(
        this.stages().map(([role, model]) => [role, model.withoutFusions()]),
      ),
    );

  rewritingState = (path: GraphPath = []) =>
    path.length ? this.select(path).model.rewritingState() : { names: [] };

  rewritingScenarios = (path: GraphPath = []): RewritingScenario[] => {
    const stage = this.select(path);
    if (stage.path.length)
      throw new Error("Return to a pipeline stage root to choose a scenario.");
    return stage.model.rewritingScenarios();
  };

  applyRewritingScenario = async (
    name: string,
    all: boolean,
    sourceFilePath?: string,
    path: GraphPath = [],
  ): Promise<NativeModel> => {
    const stage = this.select(path);
    this.rewritingScenarios(path);
    const edited = await stage.model.applyRewritingScenario(
      name,
      all,
      sourceFilePath,
    );
    const replacement = this.replace(stage.role, edited);
    return new PipelineModel(
      this.data,
      this.fileName,
      replacement.models,
      this,
      [...this.fusionOrder, stage.role],
    );
  };

  recordRewriting = async (
    name: string,
    append: boolean,
    sourceFilePath?: string,
    path: GraphPath = [],
  ): Promise<NativeModel> => {
    const stage = this.select(path);
    this.rewritingScenarios(path);
    const edited = await stage.model.recordRewriting(
      name,
      append,
      sourceFilePath,
    );
    const replacement = this.replace(stage.role, edited);
    const order = [
      ...this.fusionOrder.filter((role) => role !== stage.role),
      stage.role,
    ];
    const snapshots = [replacement.models];
    let models = replacement.models;
    for (const role of [...order].reverse()) {
      const model = models[role];
      if (!model) throw new Error("Invalid pipeline stage.");
      models = { ...models, [role]: model.undoFusion() };
      snapshots.push(models);
    }
    let rebuilt = new PipelineModel(
      this.data,
      this.fileName,
      snapshots[order.length],
      undefined,
      [],
    );
    for (let i = order.length - 1; i >= 0; --i)
      rebuilt = new PipelineModel(
        this.data,
        this.fileName,
        snapshots[i],
        rebuilt,
        order.slice(0, order.length - i),
      );
    return rebuilt;
  };

  previewFusion = async (
    pattern: string,
    sourceFilePath?: string,
    path: GraphPath = [],
  ): Promise<{ model: NativeModel; matches: FusionMatch[] }> => {
    const stage = this.select(path);
    this.fusionPatterns(path);
    const preview = await stage.model.previewFusion(pattern, sourceFilePath);
    return { ...preview, model: this.replace(stage.role, preview.model) };
  };

  applyFusion = async (
    pattern: string | string[],
    anchors: number[],
    path: GraphPath = [],
  ): Promise<NativeModel> => {
    const stage = this.select(path);
    this.fusionPatterns(path);
    const edited = await stage.model.applyFusion(pattern, anchors);
    const replacement = this.replace(stage.role, edited);
    return new PipelineModel(
      this.data,
      this.fileName,
      replacement.models,
      this,
      [...this.fusionOrder, stage.role],
    );
  };

  externalDataLocations = (): string[] => {
    return [
      ...new Set(
        this.stages().flatMap(([, model]) => model.externalDataLocations()),
      ),
    ];
  };

  serialize = async (
    password?: string,
    sourceFilePath?: string,
    includeRewritings = false,
    format: "onnx" | "ort" = "onnx",
  ): Promise<Uint8Array> => {
    if (format !== "onnx")
      throw new Error(
        "ORT format saves a single model, not a PipelineProto container.",
      );
    if (password !== undefined)
      throw new Error(
        "PipelineProto encryption is not supported by onnx-light.",
      );
    const models: Partial<Record<PipelineRole, Uint8Array>> = {};
    for (const [role, model] of this.stages())
      models[role] = await model.serialize(
        undefined,
        sourceFilePath,
        includeRewritings,
      );
    return this.data.assemble(models);
  };

  serializeFiles = async (
    maxFileSize: number,
    prefix: string,
    sourceFilePath?: string,
    reuseExternalWeights = false,
    includeRewritings = false,
  ): Promise<{
    model: Uint8Array;
    weights: { name: string; bytes: Uint8Array }[];
  }> => {
    const models: Partial<Record<PipelineRole, Uint8Array>> = {};
    const weights: { name: string; bytes: Uint8Array }[] = [];
    for (const [role, model] of this.stages()) {
      const saved = await model.serializeFiles(
        maxFileSize,
        `${prefix}.${role}`,
        sourceFilePath,
        reuseExternalWeights,
        includeRewritings,
      );
      models[role] = saved.model;
      weights.push(...saved.weights);
    }
    return { model: await this.data.assemble(models), weights };
  };
}
