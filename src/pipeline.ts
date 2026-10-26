import type {
  FusionDiagnostic,
  FusionMatch,
  FusionOperation,
  Graph,
  GraphMetadata,
  GraphPath,
  GraphSummary,
  MetadataEntry,
  NativeModel,
  OptimizationOperation,
  OptimizationPattern,
  OptimizationReport,
  PipelineRole,
  RewritingScenario,
} from "./graph";

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
  constructor(
    private readonly data: PipelineData,
    private readonly fileName: string,
    private readonly models = data.models,
    private readonly previous?: PipelineModel,
    private readonly fusionOrder: PipelineRole[] = [],
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
    );
  };

  fusionCount = (): number =>
    this.stages().reduce((count, [, model]) => count + model.fusionCount(), 0);

  fusionHistory = (): FusionOperation[] => {
    const stages = new Map(
      this.stages().map(([role, model]) => [role, model.fusionHistory()]),
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
    return this.previous.clone();
  };

  summary = (inferShapes = false): GraphSummary => {
    const { name, inputs, outputs, initializerCount, tensors, nodes, opsets } =
      this.graph(inferShapes);
    const stages = nodes.flatMap((node) =>
      node.pipelineModel ? [node.pipelineModel] : [],
    );
    return {
      name,
      inputs,
      outputs,
      initializerCount,
      tensors,
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

  optimizationPatterns = (path: GraphPath = []): OptimizationPattern[] => {
    const stage = this.select(path);
    this.fusionPatterns(path);
    return stage.model.optimizationPatterns();
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
  ): Promise<Uint8Array> => {
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
