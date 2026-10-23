/**
 * Owns panel-local native models and routes webview requests.
 * Display snapshots and action logs cross into the webview; model bytes stay in the host.
 */
import * as vscode from "vscode";
import { randomBytes } from "node:crypto";
import { join, posix } from "node:path";
import {
  listBackendTests,
  loadBackendTest,
  loadModel,
  loadModelFile,
  onnxLightVersion,
  Graph,
  NativeModel,
  GraphMetadata,
  GraphPath,
} from "./graph";
import { webviewHtml } from "./webview";
import { ActionLog, LogEntry } from "./log";
import {
  cachedHuggingFaceModel,
  listHuggingFaceModels,
  parseHuggingFaceInput,
} from "./huggingface";

const viewType = "onnxLightViewer.graph";
const panelSessions = new WeakMap<vscode.WebviewPanel, () => void>();

/**
 * Panel-local display state, independent of the recoverable native handle.
 * Reload uses a local path, original virtual-file bytes or a backend case ID.
 */
interface LoadedModel {
  graph: Graph;
  /** Absent after a failed shape/metadata refresh until the next native action. */
  native?: NativeModel;
  nativeReload?: Promise<NativeModel>;
  reload: () => NativeModel | Promise<NativeModel>;
  /** Cached for the current graph; cleared on navigation, not shape refresh. */
  metadata?: GraphMetadata;
}

/**
 * Returns the retained handle, logging a reload if a previous refresh invalidated
 * it. The changed callback publishes status only after successful recreation.
 */
async function nativeModel(
  model: LoadedModel,
  log: ActionLog,
  changed: () => void,
): Promise<NativeModel> {
  if (!model.native) {
    const reload =
      model.nativeReload ??
      (model.nativeReload = log.run(
        `Reload native model: ${model.graph.name}`,
        model.reload,
      ));
    try {
      const native = await reload;
      if (model.nativeReload === reload) {
        model.native = native;
        model.nativeReload = undefined;
        changed();
      }
      return native;
    } catch (error) {
      if (model.nativeReload === reload) model.nativeReload = undefined;
      throw error;
    }
  }
  return model.native;
}

/** Discards potentially partial native state while retaining displayed snapshots. */
function invalidateModel(model: LoadedModel, log: ActionLog): void {
  model.native = undefined;
  log.record(
    "INVALIDATED",
    `${model.graph.name}: the next native action will reload the original model; inferred information may need refreshing.`,
  );
}

/**
 * VS Code entry point registering viewer commands and the .onnx custom editor.
 * Each panel owns its model/history; Show Logs targets the last activated panel.
 * @param context Extension resources and subscription lifetime supplied by VS Code.
 */
export function activate(context: vscode.ExtensionContext): void {
  let current: { panel: vscode.WebviewPanel; showLogs: () => void } | undefined;
  const openViewer = async (
    uri?: vscode.Uri,
    initialTab: "model" | "logs" = "model",
  ) => {
    const panel = vscode.window.createWebviewPanel(
      viewType,
      "ONNX Light Viewer",
      vscode.ViewColumn.One,
      { enableScripts: true },
    );
    return configurePanel(context, panel, uri, initialTab, (showLogs) => {
      if (showLogs) current = { panel, showLogs };
      else if (current?.panel === panel) current = undefined;
    });
  };
  context.subscriptions.push(
    vscode.commands.registerCommand("onnxLightViewer.showLogs", () => {
      if (current) {
        current.panel.reveal();
        current.showLogs();
      } else {
        return openViewer(undefined, "logs");
      }
    }),
  );
  for (const [command, fileName] of [
    ["onnxLightViewer.open", "relu.onnx"],
    ["onnxLightViewer.openSymbolicExample", "tiny-llm.onnx"],
  ]) {
    context.subscriptions.push(
      vscode.commands.registerCommand(command, async () => {
        await openViewer(
          vscode.Uri.joinPath(context.extensionUri, "media", fileName),
        );
      }),
    );
  }
  context.subscriptions.push(
    vscode.window.registerCustomEditorProvider(viewType, {
      openCustomDocument: (uri) => ({ uri, dispose() {} }),
      resolveCustomEditor: async (document, panel) => {
        await configurePanel(
          context,
          panel,
          document.uri,
          "model",
          (showLogs) => {
            if (showLogs) current = { panel, showLogs };
            else if (current?.panel === panel) current = undefined;
          },
        );
      },
    }),
  );
}

/**
 * Streams local files or reads virtual files through VS Code's filesystem.
 * The reload closure retains a local path or virtual-file bytes for recovery.
 * @param fileName Original basename when the URI points to a temporary/cache file.
 * @returns Root snapshot, retained handle and its recovery factory.
 */
async function readGraph(
  uri: vscode.Uri,
  log: ActionLog,
  fileName = posix.basename(uri.path),
): Promise<LoadedModel> {
  return log.run(`Load model: ${uri.path}`, async () => {
    let reload: LoadedModel["reload"];
    if (uri.scheme === "file") {
      reload = () => loadModelFile(uri.fsPath, fileName);
    } else {
      const bytes = await vscode.workspace.fs.readFile(uri);
      reload = () => loadModel(bytes, fileName);
    }
    const native = await reload();
    return { graph: native.graph(), native, reload };
  });
}

/** Checks message structure; the native layer also checks graph index bounds. */
function isGraphPath(path: unknown): path is GraphPath {
  return (
    Array.isArray(path) &&
    path.every(
      (step: unknown) =>
        typeof step === "object" &&
        step !== null &&
        (Reflect.has(step, "function")
          ? ["node", "function"]
          : ["node", "attribute", "graph"]
        ).every((key) => {
          const index: unknown = Reflect.get(step, key);
          return (
            typeof index === "number" && Number.isInteger(index) && index >= 0
          );
        }),
    )
  );
}

/**
 * Retains the model, current graph and optional metadata snapshot for this panel.
 * Installs the message router before loading so initial errors remain inspectable.
 *
 * ready replays snapshots/history; local, backend and Hugging Face opens replace
 * the model only on success. navigate replaces the graph and clears cached metadata.
 * refresh-shapes returns tensor labels in shape-info; metadata returns a separate
 * metadata-info snapshot. Both refresh failures discard the native handle, not
 * the last display. actionId correlates requests with their UI log entries.
 *
 * @param context Supplies bundled media resources and persistent download storage.
 * @param panel Panel whose message and view-state subscriptions are disposed on close.
 * @param uri Optional model to load initially; omitted leaves the viewer empty.
 * @param initialTab Tab selected in the initial HTML shell.
 * @param activatePanel Registers this panel's Show Logs callback, or clears it on disposal.
 */
async function configurePanel(
  context: vscode.ExtensionContext,
  panel: vscode.WebviewPanel,
  uri: vscode.Uri | undefined,
  initialTab: "model" | "logs",
  activatePanel: (showLogs: (() => void) | undefined) => void,
): Promise<void> {
  panelSessions.get(panel)?.();
  let active = true;
  const postMessage = (message: unknown) =>
    active ? panel.webview.postMessage(message) : Promise.resolve(false);
  const mediaRoot = vscode.Uri.joinPath(context.extensionUri, "media");
  panel.webview.options = {
    enableScripts: true,
    localResourceRoots: [mediaRoot],
  };
  panel.iconPath = vscode.Uri.joinPath(mediaRoot, "logo.svg");
  let model: LoadedModel | undefined;
  let graphRevision = 0;
  let ready = false;
  let showLogsRequested = false;
  let initialLoading = !!uri;
  let initialError: string | undefined;
  let lastFailure: LogEntry | undefined;
  const sendStatus = () =>
    postMessage({
      type: "application-status",
      library: { loaded: true, version: onnxLightVersion },
      model: {
        name: model?.graph.name,
        state: !model ? "none" : model.native ? "ready" : "invalidated",
      },
      running: log
        .snapshot()
        .filter((entry) => entry.status === "RUNNING")
        .map((entry) => entry.action),
      lastFailure,
    });
  const log = new ActionLog((entry) => {
    if (entry.status === "FAILED") lastFailure = entry;
    void postMessage({ type: "log-entry", entry });
    void sendStatus();
  });
  const showLogs = () => {
    showLogsRequested = true;
    if (ready) {
      void postMessage({ type: "show-logs" });
      showLogsRequested = false;
    }
  };
  activatePanel(showLogs);
  const viewState = panel.onDidChangeViewState(() => {
    if (panel.active) activatePanel(showLogs);
  });
  const requireModel = () => {
    if (!model) throw new Error("No ONNX model is loaded.");
    return model;
  };
  const sendGraph = () => {
    const current = requireModel();
    void sendStatus();
    return postMessage({
      type: "graph",
      graph: current.graph,
      ...(current.metadata ? { metadata: current.metadata } : {}),
    });
  };
  const chooseBackendTest = async (
    id: string,
    action = "Open backend test",
  ) => {
    const tests = await log.run("List backend tests", listBackendTests);
    const selection = await vscode.window.showQuickPick(
      tests.map((test) => ({
        id: test.id,
        label: test.name,
        description: [test.kind, test.tag].filter(Boolean).join(" · "),
        detail: test.modelName !== test.name ? test.modelName : undefined,
      })),
      {
        title: "onnx-light backend tests",
        placeHolder: "Search by test name, model or category",
        matchOnDescription: true,
        matchOnDetail: true,
      },
    );
    if (!selection) {
      log.record("CANCELLED", action, id);
      return;
    }
    const name = selection.label;
    const loadedModel = await log.run(`Load backend test: ${name}`, () => {
      const reload = () => loadBackendTest(selection.id);
      const native = reload();
      return { graph: native.graph(), native, reload };
    });
    return { model: loadedModel, name };
  };
  const messages = panel.webview.onDidReceiveMessage(
    async (
      message: {
        type?: string;
        path?: unknown;
        actionId?: string;
        entry?: unknown;
      } | null,
    ) => {
      if (!active) return;
      if (message?.type === "show-logs") {
        showLogs();
      } else if (message?.type === "interaction") {
        log.interaction(message.entry);
      } else if (message?.type === "ready") {
        ready = true;
        await postMessage({
          type: "log-history",
          entries: log.snapshot(),
        });
        await sendStatus();
        if (model) await sendGraph();
        else
          await postMessage({
            type: "load-state",
            pending: initialLoading,
            error: initialError,
          });
        if (showLogsRequested) {
          await postMessage({ type: "show-logs" });
          showLogsRequested = false;
        }
      } else if (message?.type === "open") {
        try {
          const loaded = await log.run(
            "Open ONNX model",
            async (id) => {
              const location = await vscode.window.showQuickPick(
                [
                  { id: "local", label: "Local file" },
                  { id: "huggingface", label: "Hugging Face" },
                  { id: "backend", label: "Backend test" },
                ],
                {
                  title: "Open an ONNX model",
                  placeHolder: "Choose where to open the model from",
                },
              );
              if (!location) {
                log.record("CANCELLED", "Open ONNX model", id);
                return;
              }
              if (location.id === "backend")
                return chooseBackendTest(id, "Open ONNX model");
              if (location.id === "local") {
                const selection = await vscode.window.showOpenDialog({
                  canSelectMany: false,
                  filters: { "ONNX models": ["onnx"] },
                  openLabel: "Open ONNX graph",
                });
                if (!selection?.[0]) {
                  log.record("CANCELLED", "Open ONNX model", id);
                  return;
                }
                return {
                  model: await readGraph(selection[0], log),
                  name: posix.basename(selection[0].path),
                };
              }
              const input = await vscode.window.showInputBox({
                title: "Open a Hugging Face ONNX model",
                prompt:
                  "Enter a public repository ID, repository URL or ONNX file URL. External weight files are not downloaded.",
                placeHolder: "onnx-community/Kokoro-82M-v1.0-ONNX",
                ignoreFocusOut: true,
                validateInput: (value) => {
                  try {
                    parseHuggingFaceInput(value);
                    return undefined;
                  } catch (error) {
                    return error instanceof Error
                      ? error.message
                      : String(error);
                  }
                },
              });
              if (input === undefined) {
                log.record("CANCELLED", "Open ONNX model", id);
                return;
              }
              const target = parseHuggingFaceInput(input);
              return vscode.window.withProgress(
                {
                  location: vscode.ProgressLocation.Notification,
                  title: `Hugging Face: ${"fileName" in target ? target.fileName : target.url}`,
                  cancellable: true,
                },
                async (_progress, token) => {
                  const controller = new AbortController();
                  const cancellation = token.onCancellationRequested(() =>
                    controller.abort(),
                  );
                  if (token.isCancellationRequested) controller.abort();
                  try {
                    let source;
                    if ("fileName" in target) source = target;
                    else {
                      const files = await listHuggingFaceModels(
                        target,
                        controller.signal,
                      );
                      const selection =
                        files.length === 1
                          ? { source: files[0] }
                          : await vscode.window.showQuickPick(
                              files.map((file) => ({
                                label: file.path,
                                source: file,
                              })),
                              {
                                title: "Choose an ONNX file",
                                placeHolder:
                                  "Select a model variant to download",
                              },
                              token,
                            );
                      if (!selection) {
                        log.record("CANCELLED", "Open ONNX model", id);
                        return;
                      }
                      source = selection.source;
                    }
                    controller.signal.throwIfAborted();
                    const result = await cachedHuggingFaceModel(
                      source,
                      join(context.globalStorageUri.fsPath, "huggingface"),
                      (path) =>
                        readGraph(vscode.Uri.file(path), log, source.fileName),
                      controller.signal,
                    );
                    result.value.reload = () =>
                      loadModelFile(result.filePath, source.fileName);
                    log.record(
                      "SUCCEEDED",
                      `${result.cached ? "Cache hit" : "Downloaded and cached"}: ${source.url}`,
                      undefined,
                      result.filePath,
                    );
                    return { model: result.value, name: source.fileName };
                  } catch (error) {
                    if (
                      controller.signal.aborted &&
                      error instanceof Error &&
                      error.name === "AbortError"
                    ) {
                      log.record("CANCELLED", "Open ONNX model", id);
                      return;
                    }
                    throw error;
                  } finally {
                    cancellation.dispose();
                  }
                },
              );
            },
            message.actionId,
          );
          if (!active || !loaded) return;
          model = loaded.model;
          panel.title = `ONNX: ${loaded.name}`;
          await sendGraph();
        } catch (error) {
          if (!active) return;
          const detail = error instanceof Error ? error.message : String(error);
          await postMessage({
            type: "load-error",
            error: detail,
          });
          await vscode.window.showErrorMessage(
            `Could not open ONNX model: ${detail}`,
          );
        }
      } else if (message?.type === "open-backend-test") {
        let loaded: { model: LoadedModel; name: string } | undefined;
        try {
          loaded = await log.run(
            "Open backend test",
            chooseBackendTest,
            message.actionId,
          );
        } catch (error) {
          if (!active) return;
          await postMessage({
            type: "load-error",
            error: error instanceof Error ? error.message : String(error),
          });
          await vscode.window.showErrorMessage(
            `Could not open onnx-light backend test: ${error instanceof Error ? error.message : String(error)}`,
          );
          return;
        }
        if (!active || !loaded) return;
        model = loaded.model;
        panel.title = `ONNX: ${loaded.name}`;
        await sendGraph();
      } else if (message?.type === "navigate") {
        const requestedModel = model;
        const requestedGraphRevision = ++graphRevision;
        let graph: Graph;
        try {
          graph = await log.run(
            `Navigate graph: ${model?.graph.name ?? "no model loaded"}`,
            async () => {
              if (!isGraphPath(message.path))
                throw new Error("Invalid subgraph path.");
              const current = requireModel();
              const native = await nativeModel(current, log, () => {
                void sendStatus();
              });
              return native.graph(false, message.path);
            },
            message.actionId,
          );
        } catch (error) {
          if (
            !active ||
            model !== requestedModel ||
            graphRevision !== requestedGraphRevision
          )
            return;
          const detail = error instanceof Error ? error.message : String(error);
          await postMessage({
            type: "navigation-error",
            error: detail,
          });
          await vscode.window.showErrorMessage(
            `Could not open ONNX subgraph: ${detail}`,
          );
          return;
        }
        if (
          !active ||
          model !== requestedModel ||
          graphRevision !== requestedGraphRevision
        )
          return;
        requireModel().graph = graph;
        requireModel().metadata = undefined;
        graphRevision++;
        await sendGraph();
      } else if (message?.type === "metadata") {
        const requestedModel = model;
        const requestedGraphRevision = graphRevision;
        let metadata: GraphMetadata;
        try {
          metadata = await log.run(
            `Refresh metadata: ${model?.graph.name ?? "no model loaded"}`,
            async () => {
              const current = requireModel();
              const native = await nativeModel(current, log, () => {
                void sendStatus();
              });
              return native.metadata(current.graph.breadcrumbs.at(-1)?.path);
            },
            message.actionId,
          );
        } catch (error) {
          if (
            !active ||
            model !== requestedModel ||
            graphRevision !== requestedGraphRevision
          )
            return;
          if (model) invalidateModel(model, log);
          const detail = error instanceof Error ? error.message : String(error);
          await postMessage({
            type: "metadata-error",
            error: detail,
          });
          await vscode.window.showErrorMessage(
            `Could not analyze ONNX metadata: ${detail}`,
          );
          return;
        }
        if (
          !active ||
          model !== requestedModel ||
          graphRevision !== requestedGraphRevision
        )
          return;
        requireModel().metadata = metadata;
        await postMessage({ type: "metadata-info", metadata });
      } else if (message?.type === "refresh-shapes") {
        const requestedModel = model;
        const requestedGraphRevision = graphRevision;
        let refreshed: Graph;
        try {
          refreshed = await log.run(
            `Refresh shapes: ${model?.graph.name ?? "no model loaded"}`,
            async () => {
              const current = requireModel();
              const native = await nativeModel(current, log, () => {
                void sendStatus();
              });
              return native.graph(true, current.graph.breadcrumbs.at(-1)?.path);
            },
            message.actionId,
          );
        } catch (error) {
          if (
            !active ||
            model !== requestedModel ||
            graphRevision !== requestedGraphRevision
          )
            return;
          if (model) invalidateModel(model, log);
          const detail = error instanceof Error ? error.message : String(error);
          await postMessage({
            type: "shape-error",
            error: detail,
          });
          await vscode.window.showErrorMessage(
            `Could not infer ONNX shapes/types: ${detail}`,
          );
          return;
        }
        if (
          !active ||
          model !== requestedModel ||
          graphRevision !== requestedGraphRevision
        )
          return;
        requireModel().graph = refreshed;
        await postMessage({
          type: "shape-info",
          tensors: refreshed.tensors,
        });
      }
    },
  );
  const dispose = () => {
    active = false;
    messages.dispose();
    viewState.dispose();
    disposal.dispose();
    panelSessions.delete(panel);
    activatePanel(undefined);
  };
  const disposal = panel.onDidDispose(dispose);
  panelSessions.set(panel, dispose);
  const resource = (name: string) =>
    panel.webview.asWebviewUri(vscode.Uri.joinPath(mediaRoot, name)).toString();
  const nonce = randomBytes(16).toString("hex");
  panel.webview.html = webviewHtml(
    resource,
    panel.webview.cspSource,
    nonce,
    onnxLightVersion,
    initialTab,
  );
  if (uri) {
    try {
      model = await readGraph(uri, log);
      if (!active) return;
      initialLoading = false;
      if (ready) await sendGraph();
      else void sendStatus();
    } catch (error) {
      if (!active) return;
      initialLoading = false;
      initialError = error instanceof Error ? error.message : String(error);
      await postMessage({
        type: "load-error",
        error: error instanceof Error ? error.message : String(error),
      });
      await vscode.window.showErrorMessage(
        `Could not open ONNX model: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  }
}
