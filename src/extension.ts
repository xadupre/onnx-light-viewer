/**
 * Owns panel-local native models and routes webview requests.
 * Display snapshots and action logs cross into the webview; model bytes stay in the host.
 */
import * as vscode from "vscode";
import { randomBytes } from "node:crypto";
import { mkdir, open, stat } from "node:fs/promises";
import { dirname, isAbsolute, join, posix, relative } from "node:path";
import { homedir } from "node:os";
import {
  listBackendTests,
  loadBackendTest,
  loadModel,
  loadModelFile,
  isEncryptedModel,
  onnxLightVersion,
  Graph,
  NativeModel,
  GraphMetadata,
  GraphPath,
  FusionMatch,
  FusionOperation,
} from "./graph";
import { webviewHtml } from "./webview";
import { applicationSettings, inspectPythonPackages } from "./settings";
import {
  defaultExportDirectory,
  ExportHistory,
  downloadExportSource,
  exportNeedsDownload,
  exportPlan,
  prepareExportPlan,
  runExport,
} from "./export";
import { ActionLog, LogEntry } from "./log";
import {
  PathHistory,
  historyEntries,
  isHistoryEntry,
  HistoryTarget,
} from "./pathHistory";
import {
  cachedHuggingFaceModel,
  listHuggingFaceModels,
  parseHuggingFaceInput,
  searchHuggingFaceModels,
  HuggingFaceSource,
} from "./huggingface";
import { OliveRecipes } from "./oliveRecipes";
import { optimizeWithOnnxScript } from "./optimization";

const viewType = "onnxLightViewer.graph";
const panelSessions = new WeakMap<vscode.WebviewPanel, () => void>();

function pickLocalModel() {
  return vscode.window.showOpenDialog({
    canSelectMany: false,
    filters: {
      "ONNX models and pipelines": ["onnx", "enc", "pipe-onnx"],
    },
    openLabel: "Open ONNX graph",
  });
}

interface ViewerSettings {
  width?: number;
  panels: { operator: boolean; find: boolean; metadata: boolean };
}

function viewerSettings(value: unknown): ViewerSettings {
  if (!value || typeof value !== "object")
    throw new Error("Invalid viewer settings.");
  const width = "width" in value ? value.width : undefined;
  if (
    width !== undefined &&
    (typeof width !== "number" || !Number.isInteger(width) || width < 200)
  )
    throw new Error("Invalid sidebar width.");
  const panels = "panels" in value ? value.panels : undefined;
  if (
    !panels ||
    typeof panels !== "object" ||
    !("operator" in panels) ||
    typeof panels.operator !== "boolean" ||
    !("find" in panels) ||
    typeof panels.find !== "boolean" ||
    !("metadata" in panels) ||
    typeof panels.metadata !== "boolean"
  )
    throw new Error("Invalid sidebar panel settings.");
  return {
    ...(width === undefined ? {} : { width }),
    panels: {
      operator: panels.operator,
      find: panels.find,
      metadata: panels.metadata,
    },
  };
}

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
  fusionCount?: number;
  fusionHistory?: FusionOperation[];
  /** Materialized Python edits remain modified after undoing later native edits. */
  pythonOptimized?: boolean;
  source?: { uri: vscode.Uri };
  /** Original source ownership survives materializing or removing external data. */
  sourceWeightLocations?: readonly string[];
  fileName?: string;
}

async function modelFiles(model: LoadedModel, native: NativeModel) {
  const source = model.source?.uri;
  const fileDetails = async (
    uri: vscode.Uri | undefined,
    name: string,
    external: boolean,
  ) => {
    if (!uri) return { name, external, error: "Source directory unavailable." };
    try {
      const details = await vscode.workspace.fs.stat(uri);
      if (!(details.type & vscode.FileType.File))
        throw new Error("Not a regular file.");
      return { name, external, size: details.size };
    } catch (error) {
      return {
        name,
        external,
        error: error instanceof Error ? error.message : String(error),
      };
    }
  };
  return Promise.all([
    ...(source
      ? [
          fileDetails(
            source,
            model.fileName ?? posix.basename(source.path),
            false,
          ),
        ]
      : []),
    ...[...new Set(native.externalDataLocations())].map((location) =>
      fileDetails(
        source ? vscode.Uri.joinPath(source, "..", location) : undefined,
        location,
        true,
      ),
    ),
  ]);
}

function copyFileName(model: LoadedModel, encrypted = false): string {
  const extension = model.graph.pipeline ? "pipe-onnx" : "onnx";
  return `${(model.fileName ?? `model.${extension}`).replace(/\.(?:onnx(?:\.enc)?|pipe-onnx)$/i, "")}-copy.${extension}${encrypted ? ".enc" : ""}`;
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
  const pathHistory = new PathHistory(context.globalState, (message) => {
    void vscode.window.showErrorMessage(message);
  });
  const exportHistory = new ExportHistory(context.globalState);
  let current: { panel: vscode.WebviewPanel; showLogs: () => void } | undefined;
  const openViewer = async (
    uri?: vscode.Uri,
    initialTab: "model" | "logs" = "model",
  ) => {
    const panel = vscode.window.createWebviewPanel(
      viewType,
      "ONNX Light Viewer",
      vscode.ViewColumn.One,
      { enableScripts: true, retainContextWhenHidden: true },
    );
    return configurePanel(
      context,
      panel,
      uri,
      initialTab,
      pathHistory,
      exportHistory,
      (showLogs) => {
        if (showLogs) current = { panel, showLogs };
        else if (current?.panel === panel) current = undefined;
      },
    );
  };
  context.subscriptions.push(
    vscode.commands.registerCommand(
      "onnxLightViewer.openFile",
      (uri: vscode.Uri) =>
        vscode.commands.executeCommand("vscode.openWith", uri, viewType),
    ),
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
    vscode.window.registerCustomEditorProvider(
      viewType,
      {
        openCustomDocument: (uri) => ({ uri, dispose() {} }),
        resolveCustomEditor: async (document, panel) => {
          await configurePanel(
            context,
            panel,
            document.uri,
            "model",
            pathHistory,
            exportHistory,
            (showLogs) => {
              if (showLogs) current = { panel, showLogs };
              else if (current?.panel === panel) current = undefined;
            },
          );
        },
      },
      {
        webviewOptions: { retainContextWhenHidden: true },
      },
    ),
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
  suppliedPassword?: string,
): Promise<LoadedModel | undefined> {
  return log.run(`Load model: ${uri.path}`, async (id) => {
    const source = { uri };
    let bytes: Uint8Array | undefined;
    let header: Uint8Array;
    if (uri.scheme === "file") {
      const file = await open(uri.fsPath, "r");
      try {
        const buffer = Buffer.alloc(8);
        const { bytesRead } = await file.read(buffer, 0, 8, 0);
        header = buffer.subarray(0, bytesRead);
      } finally {
        await file.close();
      }
    } else {
      bytes = await vscode.workspace.fs.readFile(uri);
      header = bytes;
    }
    const encrypted = isEncryptedModel(header);
    const password = encrypted
      ? (suppliedPassword ??
        (await vscode.window.showInputBox({
          title: `Decrypt ONNX model: ${fileName}`,
          prompt:
            "Enter the model password. It stays in the extension host's memory for recovery.",
          password: true,
          ignoreFocusOut: true,
          validateInput: (value) =>
            value.length ? undefined : "A password is required.",
        })))
      : undefined;
    if (encrypted && password === undefined) {
      log.record("CANCELLED", `Load model: ${uri.path}`, id);
      return;
    }
    const retainedBytes = bytes;
    const reload = retainedBytes
      ? () => loadModel(retainedBytes, fileName, password)
      : () => loadModelFile(source.uri.fsPath, fileName, password);
    const native = await reload();
    return {
      graph: native.graph(),
      native,
      reload,
      source,
      sourceWeightLocations: native.externalDataLocations(),
      fileName,
    };
  });
}

async function assertSaveDestination(
  sources: readonly vscode.Uri[],
  destination: vscode.Uri,
): Promise<void> {
  const reject = () => {
    throw new Error(
      "Choose a different destination to preserve the source model and its external weights.",
    );
  };
  for (const source of sources) {
    if (source.scheme !== destination.scheme) continue;
    const samePath =
      process.platform === "win32"
        ? source.path.toLowerCase() === destination.path.toLowerCase()
        : source.path === destination.path;
    if (samePath) reject();
  }
  if (destination.scheme !== "file") return;
  const existingFile = async (uri: vscode.Uri) => {
    try {
      return await stat(uri.fsPath, { bigint: true });
    } catch (error) {
      if (error instanceof Error && "code" in error && error.code === "ENOENT")
        return undefined;
      throw error;
    }
  };
  const target = await existingFile(destination);
  if (!target) return;
  for (const source of sources) {
    if (source.scheme !== "file") continue;
    const original = await existingFile(source);
    if (original && original.dev === target.dev && original.ino === target.ino)
      reject();
  }
}

/** Publishes a complete file only while the originating model is still current. */
async function writeModelFile(
  uri: vscode.Uri,
  bytes: Uint8Array,
  overwrite: boolean,
  canPublish: () => boolean | Promise<boolean>,
): Promise<boolean> {
  const temporary = vscode.Uri.joinPath(
    uri,
    "..",
    `.${posix.basename(uri.path)}.${randomBytes(12).toString("hex")}.tmp`,
  );
  let renamed = false;
  try {
    await vscode.workspace.fs.writeFile(temporary, bytes);
    if (!(await canPublish())) return false;
    await vscode.workspace.fs.rename(temporary, uri, { overwrite });
    renamed = true;
    return true;
  } finally {
    if (!renamed)
      try {
        await vscode.workspace.fs.delete(temporary);
      } catch (error) {
        if (!(
          error instanceof vscode.FileSystemError &&
          error.code === "FileNotFound"
        ))
          throw error;
      }
  }
}
/** Checks message structure; the native layer also checks graph index bounds. */
function isGraphPath(path: unknown): path is GraphPath {
  return (
    Array.isArray(path) &&
    path.every((step: unknown) => {
      if (typeof step !== "object" || step === null) return false;
      if ("pipeline" in step)
        return (
          typeof step.pipeline === "string" &&
          ["embedding", "prefill", "decode"].includes(step.pipeline) &&
          Object.keys(step).length === 1
        );
      return (
        Reflect.has(step, "function")
          ? Reflect.has(step, "node")
            ? ["node", "function"]
            : ["function"]
          : ["node", "attribute", "graph"]
      ).every((key) => {
        const index: unknown = Reflect.get(step, key);
        return (
          typeof index === "number" && Number.isInteger(index) && index >= 0
        );
      });
    })
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
 * @param pathHistory Shared persistent history of successful Open/Save paths.
 * @param exportHistory Shared persistent history of validated export inputs.
 * @param activatePanel Registers this panel's Show Logs callback, or clears it on disposal.
 */
async function configurePanel(
  context: vscode.ExtensionContext,
  panel: vscode.WebviewPanel,
  uri: vscode.Uri | undefined,
  initialTab: "model" | "logs",
  pathHistory: PathHistory,
  exportHistory: ExportHistory,
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
  let navigationRequest = 0;
  let statisticsRequest = 0;
  let tensorStatisticsRequest = 0;
  let memoryRequest = 0;
  let fusionRequest = 0;
  let diagnosticRequest = 0;
  let undoRequest = 0;
  let fusionPreview:
    | {
        id: number;
        owner: LoadedModel;
        native: NativeModel;
        snapshot: NativeModel;
        revision: number;
        pattern: string;
        anchors: Set<number>;
        matches: (FusionMatch & { pattern: string })[];
      }
    | undefined;
  let ready = false;
  let hubSearch: AbortController | undefined;
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
  const stopHistory = pathHistory.subscribe((history) => {
    void postMessage({ type: "path-history", history });
  });
  const stopExportHistory = exportHistory.subscribe((history) => {
    void postMessage({ type: "export-history", history });
  });
  const updateHistory = async (
    target: HistoryTarget,
    change: Parameters<PathHistory["update"]>[1],
  ) => {
    try {
      await pathHistory.update(target, change);
    } catch (error) {
      const detail = error instanceof Error ? error.message : String(error);
      log.record("FAILED", `Update path history: ${detail}`);
      await postMessage({ type: "settings-error", error: detail });
      await vscode.window.showErrorMessage(
        `Could not update path history: ${detail}`,
      );
    }
  };
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
  const directoryUri = () =>
    model?.source
      ? vscode.Uri.joinPath(model.source.uri, "..")
      : (vscode.workspace.workspaceFolders?.[0]?.uri ??
        vscode.Uri.file(homedir()));
  const saveDirectoryUri = () => {
    const root = vscode.workspace.workspaceFolders?.[0]?.uri;
    return root
      ? vscode.Uri.joinPath(root, ".cache", ".onnx-light-viewer")
      : directoryUri();
  };
  const prepareSaveDirectory = async (directory: vscode.Uri) => {
    if (
      vscode.workspace.workspaceFolders?.length &&
      directory.toString() === saveDirectoryUri().toString()
    )
      await vscode.workspace.fs.createDirectory(directory);
  };
  const locationText = (uri: vscode.Uri) =>
    uri.scheme === "file" ? uri.fsPath : uri.toString();
  const getApplicationSettings = () =>
    applicationSettings(
      context.globalState.get<unknown>("applicationSettings") ?? {
        downloadDirectory: join(homedir(), ".cache", "onnx-light-viewer"),
        pythonPath: process.platform === "win32" ? "python" : "python3",
      },
    );
  const sendSettings = async () => {
    const application = getApplicationSettings();
    await mkdir(application.downloadDirectory, { recursive: true });
    return postMessage({
      type: "settings-info",
      application,
      exportHistory: exportHistory.snapshot(),
      pathHistory: pathHistory.snapshot(),
      saveDirectory: locationText(saveDirectoryUri()),
      exportDirectory: defaultExportDirectory(application.downloadDirectory),
      viewer: viewerSettings(
        context.globalState.get<unknown>("viewerSettings") ?? {
          panels: { operator: true, find: true, metadata: false },
        },
      ),
      appVersion: context.extension.packageJSON.version,
      nativeVersion: onnxLightVersion,
      platform: `${process.platform} / ${process.arch}`,
      trusted: vscode.workspace.isTrusted,
    });
  };
  let recipeCache: { directory: string; recipes: OliveRecipes } | undefined;
  const getOliveRecipes = () => {
    const directory = join(
      getApplicationSettings().downloadDirectory,
      "olive-recipes",
    );
    if (recipeCache?.directory !== directory)
      recipeCache = { directory, recipes: new OliveRecipes(directory) };
    return recipeCache.recipes;
  };
  const recipeController = new AbortController();
  const getExportPlan = (value: unknown) => {
    const settings = getApplicationSettings();
    if (
      value &&
      typeof value === "object" &&
      "directory" in value &&
      value.directory === ""
    )
      value = {
        ...value,
        directory: defaultExportDirectory(settings.downloadDirectory),
      };
    let recipe;
    if (
      value &&
      typeof value === "object" &&
      "exporter" in value &&
      value.exporter === "olive-recipes"
    ) {
      recipe = getOliveRecipes().resolve(
        "recipeId" in value ? value.recipeId : undefined,
        "recipeRevision" in value ? value.recipeRevision : undefined,
      );
      value = { ...value, model: recipe.name };
    }
    return exportPlan(
      value,
      settings.pythonPath,
      settings.downloadDirectory,
      recipe,
    );
  };
  let inspectingPackages = false;
  let exporting: AbortController | undefined;
  const locationUri = (value: unknown) => {
    if (typeof value !== "string" || !value.trim() || value.includes("\0"))
      throw new Error("Enter a file or directory path.");
    return /^[a-z][a-z\d+.-]*:\/\//i.test(value)
      ? vscode.Uri.parse(value)
      : isAbsolute(value) || /^[a-z]:[\\/]/i.test(value)
        ? vscode.Uri.file(value)
        : vscode.Uri.joinPath(directoryUri(), value);
  };
  const hubFileEntries = (files: (HuggingFaceSource & { path?: string })[]) =>
    files.map((file) => ({
      name: file.path ?? file.fileName,
      path: file.url,
      directory: false,
      ...(file.size === undefined ? {} : { size: file.size }),
    }));
  const sendGraph = (selectedNodes?: number[], fusionRequestId?: unknown) => {
    fusionPreview = undefined;
    fusionRequest++;
    diagnosticRequest++;
    const current = requireModel();
    void sendStatus();
    return postMessage({
      type: "graph",
      graph: current.graph,
      ...(fusionRequestId === undefined ? {} : { fusionRequestId }),
      ...(selectedNodes ? { selectedNodes } : {}),
      file: {
        fusionCount: current.fusionCount ?? 0,
        fusionHistory: current.fusionHistory ?? [],
        ...(current.graph.pipeline ? { pipeline: true } : {}),
        directory: locationText(directoryUri()),
        saveDirectory: locationText(saveDirectoryUri()),
        savePath: locationText(
          vscode.Uri.joinPath(saveDirectoryUri(), copyFileName(current)),
        ),
      },
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
    return {
      model: loadedModel,
      name,
      history: historyEntries("backend", String(selection.id)),
    };
  };
  const messages = panel.webview.onDidReceiveMessage(
    async (
      message: {
        type?: string;
        path?: unknown;
        nodeIndex?: unknown;
        actionId?: string;
        entry?: unknown;
        storage?: unknown;
        encrypted?: unknown;
        maxExternalFileSize?: unknown;
        includeRewritings?: unknown;
        requestId?: unknown;
        name?: unknown;
        dimensions?: unknown;
        previewId?: unknown;
        pattern?: unknown;
        scenario?: unknown;
        patterns?: unknown;
        anchor?: unknown;
        anchors?: unknown;
        candidates?: unknown;
        operation?: unknown;
        backend?: unknown;
        match?: unknown;
        all?: unknown;
        location?: unknown;
        source?: unknown;
        target?: unknown;
        password?: unknown;
        confirmation?: unknown;
        overwrite?: unknown;
        settings?: unknown;
        pythonPath?: unknown;
        historyAction?: unknown;
        historyEntry?: unknown;
        query?: unknown;
        task?: unknown;
        exportOptions?: unknown;
      } | null,
    ) => {
      if (!active) return;
      if (message?.type === "show-logs") {
        showLogs();
      } else if (
        message?.type === "get-settings" ||
        message?.type === "application-settings" ||
        message?.type === "export-history-edit" ||
        message?.type === "reset-viewer-settings" ||
        message?.type === "inspect-python"
      ) {
        try {
          if (message.type === "reset-viewer-settings") {
            await context.globalState.update("viewerSettings", undefined);
          } else if (message.type === "export-history-edit") {
            if (message.historyAction === "clear") await exportHistory.clear();
            else if (
              message.historyAction === "remove" &&
              (message.target === "models" ||
                message.target === "directories") &&
              typeof message.historyEntry === "string" &&
              message.historyEntry.trim() &&
              !/[\0\r\n]/.test(message.historyEntry)
            ) {
              await exportHistory.remove(message.target, message.historyEntry);
            } else throw new Error("Invalid conversion history edit.");
          }
          if (message.type === "application-settings") {
            const settings = applicationSettings(message.settings);
            await mkdir(settings.downloadDirectory, { recursive: true });
            await context.globalState.update("applicationSettings", settings);
          }
          if (message.type === "inspect-python") {
            if (!vscode.workspace.isTrusted)
              throw new Error("Trust this workspace before running Python.");
            if (
              typeof message.requestId !== "number" ||
              !Number.isSafeInteger(message.requestId) ||
              message.requestId <= 0
            )
              throw new Error("Invalid Python inspection request.");
            const { pythonPath } = applicationSettings({
              ...getApplicationSettings(),
              pythonPath: message.pythonPath,
            });
            if (inspectingPackages)
              throw new Error(
                "A Python package inspection is already running. Try again when it finishes.",
              );
            inspectingPackages = true;
            try {
              const python = await inspectPythonPackages(pythonPath);
              await postMessage({
                type: "python-info",
                requestId: message.requestId,
                pythonPath,
                python,
              });
            } finally {
              inspectingPackages = false;
            }
          } else if (message.type !== "export-history-edit")
            await sendSettings();
        } catch (error) {
          const detail = error instanceof Error ? error.message : String(error);
          log.record("FAILED", `Settings: ${detail}`);
          await postMessage({
            type: "settings-error",
            ...(message.type === "inspect-python"
              ? { requestId: message.requestId, pythonPath: message.pythonPath }
              : {}),
            error: detail,
          });
        }
      } else if (message?.type === "interaction") {
        log.interaction(message.entry);
      } else if (
        message?.type === "memory-inputs" ||
        message?.type === "memory-profile"
      ) {
        const requestedModel = model;
        const revision = graphRevision;
        const generation = ++memoryRequest;
        let setup:
          | { stages: NonNullable<Graph["pipelineStages"]>; path: GraphPath }
          | undefined;
        const isCurrent = () =>
          active &&
          model === requestedModel &&
          graphRevision === revision &&
          memoryRequest === generation;
        try {
          const result = await log.run("Estimate model memory", async () => {
            if (
              typeof message.requestId !== "number" ||
              !Number.isSafeInteger(message.requestId) ||
              message.requestId <= 0
            )
              throw new Error("Invalid memory profiling request.");
            const current = requireModel();
            const native = await nativeModel(current, log, () => {
              void sendStatus();
            });
            if (!isCurrent()) return;
            const stages = current.graph.pipeline
              ? (native.graph().pipelineStages ?? [])
              : [];
            const path = message.path ?? stages[0]?.path ?? [];
            if (!isGraphPath(path))
              throw new Error("Invalid memory profiling path.");
            setup = { stages, path };
            if (message.type === "memory-inputs")
              return {
                dimensions: native.memoryProfileInputs(path),
                stages,
                path,
              };
            const values = message.dimensions;
            if (!values || typeof values !== "object" || Array.isArray(values))
              throw new Error("Expected dynamic dimension values.");
            const dimensions = Object.fromEntries(
              Object.entries(values).map(([key, value]): [string, number] => {
                if (
                  typeof value !== "number" ||
                  !Number.isSafeInteger(value) ||
                  value <= 0
                )
                  throw new Error(
                    `Expected a positive safe integer for ${key}.`,
                  );
                return [key, value];
              }),
            );
            return { profile: await native.memoryProfile(dimensions, path) };
          });
          if (isCurrent() && result)
            await postMessage({
              type: message.type,
              requestId: message.requestId,
              ...result,
            });
        } catch (error) {
          if (!isCurrent()) return;
          const detail = error instanceof Error ? error.message : String(error);
          await postMessage({
            type: "memory-error",
            requestId: message.requestId,
            error: detail,
            ...setup,
          });
          await vscode.window.showErrorMessage(
            `Could not estimate ONNX memory: ${detail}`,
          );
        }
      } else if (message?.type === "tensor-statistics") {
        const requestedModel = model;
        const revision = graphRevision;
        const generation = ++tensorStatisticsRequest;
        const isCurrent = () =>
          active &&
          model === requestedModel &&
          graphRevision === revision &&
          tensorStatisticsRequest === generation;
        try {
          const statistics = await log.run("Inspect tensor data", async () => {
            if (
              typeof message.requestId !== "number" ||
              !Number.isSafeInteger(message.requestId) ||
              message.requestId <= 0 ||
              typeof message.name !== "string" ||
              !message.name.length ||
              !isGraphPath(message.path)
            )
              throw new Error("Invalid tensor statistics request.");
            const current = requireModel();
            const native = await nativeModel(current, log, () => {
              void sendStatus();
            });
            if (!isCurrent()) return;
            return native.tensorStatistics(message.name, message.path, true);
          });
          if (isCurrent() && statistics)
            await postMessage({
              type: "tensor-statistics-info",
              requestId: message.requestId,
              statistics,
            });
        } catch (error) {
          if (!isCurrent()) return;
          await postMessage({
            type: "tensor-statistics-error",
            requestId: message.requestId,
            error: error instanceof Error ? error.message : String(error),
          });
        }
      } else if (message?.type === "get-statistics") {
        const requestedModel = model;
        const revision = graphRevision;
        const generation = ++statisticsRequest;
        const isCurrent = () =>
          active &&
          model === requestedModel &&
          graphRevision === revision &&
          statisticsRequest === generation;
        try {
          const result = await log.run("Refresh model statistics", async () => {
            if (
              typeof message.requestId !== "number" ||
              !Number.isSafeInteger(message.requestId) ||
              message.requestId <= 0
            )
              throw new Error("Invalid statistics request.");
            const current = requireModel();
            const native = await nativeModel(current, log, () => {
              void sendStatus();
            });
            if (!isCurrent()) return;
            const statistics = current.graph.pipeline
              ? native.graph()
              : native.summary();
            return { statistics, files: await modelFiles(current, native) };
          });
          if (!isCurrent() || !result) return;
          await postMessage({
            type: "statistics-info",
            requestId: message.requestId,
            ...result,
          });
        } catch (error) {
          if (!isCurrent()) return;
          const detail = error instanceof Error ? error.message : String(error);
          await postMessage({
            type: "statistics-error",
            requestId: message.requestId,
            error: detail,
          });
          await vscode.window.showErrorMessage(
            `Could not refresh ONNX statistics: ${detail}`,
          );
        }
      } else if (message?.type === "olive-recipes") {
        try {
          if (!vscode.workspace.isTrusted)
            throw new Error(
              "Trust this workspace before downloading Olive recipes.",
            );
          const catalog = await getOliveRecipes().refresh(
            recipeController.signal,
          );
          if (!active) return;
          await postMessage({
            type: "olive-recipes",
            requestId: message.requestId,
            ...catalog,
          });
        } catch (error) {
          if (!active) return;
          await postMessage({
            type: "olive-recipes",
            requestId: message.requestId,
            error: error instanceof Error ? error.message : String(error),
          });
        }
      } else if (message?.type === "export-preview") {
        try {
          const plan = getExportPlan(message.exportOptions);
          await postMessage({
            type: "export-preview",
            requestId: message.requestId,
            command: plan.command,
            script: plan.script,
            directory: plan.directory,
            directoryTemplate: plan.directoryTemplate,
          });
        } catch (error) {
          await postMessage({
            type: "export-preview",
            requestId: message.requestId,
            error: error instanceof Error ? error.message : String(error),
          });
        }
      } else if (message?.type === "cancel-export") {
        exporting?.abort();
      } else if (message?.type === "export") {
        try {
          await log.run(
            "Convert ONNX model",
            async (id) => {
              if (!vscode.workspace.isTrusted)
                throw new Error(
                  "Trust this workspace before running an exporter.",
                );
              if (exporting)
                throw new Error(
                  "A conversion is already running in this panel.",
                );
              let plan = getExportPlan(message.exportOptions);
              const controller = new AbortController();
              exporting = controller;
              const originalModel = model;
              const revision = graphRevision;
              let cancellation: string | undefined;
              let exportedFiles: string[] = [];
              let selectedFile: string | undefined;
              const needsDownload = exportNeedsDownload(plan);
              const conversionStage = plan.recipe
                ? "Running pinned Olive recipe (network may be required)"
                : plan.exporter === "olive" && plan.oliveWorkflow === "optimize"
                  ? "Exporting with Olive (calibration downloads may be required)"
                  : "Converting local model (offline)";
              const timings: {
                downloadSkipped: boolean;
                downloadDurationMs?: number;
                durationMs?: number;
              } = { downloadSkipped: !needsDownload };
              let phase: "download" | "conversion" | undefined;
              let phaseStarted = 0;
              const finishPhase = () => {
                if (phase)
                  timings[
                    phase === "download" ? "downloadDurationMs" : "durationMs"
                  ] = performance.now() - phaseStarted;
                phase = undefined;
              };
              try {
                plan = await prepareExportPlan(plan, controller.signal);
                await postMessage({
                  type: "export-progress",
                  actionId: message.actionId,
                  stage: "Starting conversion",
                  command: plan.command,
                  script: plan.script,
                  directory: plan.directory,
                  directoryTemplate: plan.directoryTemplate,
                  output: "",
                });
                await exportHistory.remember(plan);
                if (needsDownload) {
                  const cacheDirectory = join(
                    getApplicationSettings().downloadDirectory,
                    "huggingface",
                  );
                  phase = "download";
                  phaseStarted = performance.now();
                  await postMessage({
                    type: "export-progress",
                    actionId: message.actionId,
                    stage: "Downloading source model",
                    phase,
                    cacheDirectory,
                    ...timings,
                  });
                  plan = await downloadExportSource(
                    plan,
                    cacheDirectory,
                    controller.signal,
                    (output) => {
                      if (!active || controller.signal.aborted) return;
                      void postMessage({
                        type: "export-progress",
                        actionId: message.actionId,
                        stage: "Downloading source model",
                        phase: "download",
                        output,
                      });
                    },
                  );
                  finishPhase();
                }
                controller.signal.throwIfAborted();
                phase = "conversion";
                phaseStarted = performance.now();
                await postMessage({
                  type: "export-progress",
                  actionId: message.actionId,
                  stage: conversionStage,
                  phase,
                  command: plan.command,
                  output: "",
                  ...timings,
                });
                log.record(
                  "SUCCEEDED",
                  "Conversion command",
                  undefined,
                  `${plan.command}\nScript: ${plan.script}`,
                );
                const files = await runExport(
                  plan,
                  controller.signal,
                  (output) => {
                    if (!active || controller.signal.aborted) return;
                    void postMessage({
                      type: "export-progress",
                      actionId: message.actionId,
                      stage: conversionStage,
                      phase: "conversion",
                      output,
                    });
                  },
                );
                exportedFiles = files;
                finishPhase();
                if (!active) return;
                await postMessage({
                  type: "export-progress",
                  actionId: message.actionId,
                  stage: "Conversion completed",
                  phase: "finished",
                  ...timings,
                });
                log.record(
                  "SUCCEEDED",
                  "Download and conversion timings",
                  undefined,
                  JSON.stringify(timings),
                );
                controller.signal.throwIfAborted();
                const selected =
                  files.length === 1
                    ? files[0]
                    : (
                        await vscode.window.showQuickPick(
                          files.map((path) => ({
                            label: relative(plan.directory, path),
                            path,
                          })),
                          {
                            title: "Choose exported ONNX model",
                            placeHolder:
                              "Choose the exported model or pipeline to open",
                          },
                        )
                      )?.path;
                selectedFile = selected;
                if (!selected) {
                  cancellation =
                    "Files saved; opening the exported model was cancelled.";
                  await postMessage({
                    type: "export-info",
                    actionId: message.actionId,
                    script: plan.script,
                    loaded: false,
                    ...timings,
                  });
                  return;
                }
                controller.signal.throwIfAborted();
                await postMessage({
                  type: "export-progress",
                  actionId: message.actionId,
                  stage: "Opening exported model",
                });
                const loaded = await readGraph(vscode.Uri.file(selected), log);
                controller.signal.throwIfAborted();
                if (!active) return;
                if (model !== originalModel || graphRevision !== revision)
                  throw new Error(
                    "The displayed model changed during conversion. Open the saved model manually.",
                  );
                if (!loaded)
                  throw new Error("Opening the exported model was cancelled.");
                model = loaded;
                panel.title = `ONNX: ${posix.basename(selected.replaceAll("\\", "/"))}`;
                await sendGraph();
                await postMessage({
                  type: "export-info",
                  actionId: message.actionId,
                  path: selected,
                  script: plan.script,
                  loaded: true,
                  ...timings,
                });
              } catch (error) {
                const interrupted = phase;
                finishPhase();
                if (active)
                  await postMessage({
                    type: "export-progress",
                    actionId: message.actionId,
                    stage: `${interrupted === "download" ? "Download" : "Conversion"} ${controller.signal.aborted ? "cancelled" : "failed"}.`,
                    phase: "finished",
                    ...timings,
                  });
                if (
                  controller.signal.aborted &&
                  error instanceof Error &&
                  error.name === "AbortError"
                ) {
                  cancellation = "";
                  return;
                }
                throw error;
              } finally {
                // Release before ActionLog publishes a terminal status.
                if (exporting === controller) exporting = undefined;
                if (exportedFiles.length) {
                  const files = selectedFile
                    ? [
                        selectedFile,
                        ...exportedFiles.filter(
                          (file) => file !== selectedFile,
                        ),
                      ]
                    : exportedFiles;
                  const entries = new Map(
                    files.flatMap((file) =>
                      historyEntries("local", file, dirname(file)).map(
                        (entry) =>
                          [
                            JSON.stringify([entry.kind, entry.path]),
                            entry,
                          ] as const,
                      ),
                    ),
                  );
                  await updateHistory("open", {
                    action: "remember",
                    entries: [...entries.values()],
                  });
                }
                if (cancellation !== undefined)
                  log.record(
                    "CANCELLED",
                    "Convert ONNX model",
                    id,
                    cancellation,
                  );
              }
            },
            message.actionId,
          );
        } catch (error) {
          if (!active) return;
          const detail = error instanceof Error ? error.message : String(error);
          await postMessage({
            type: "export-error",
            actionId: message.actionId,
            error: detail,
          });
          await vscode.window.showErrorMessage(
            `Could not convert ONNX model: ${detail}`,
          );
        }
      } else if (message?.type === "path-history-edit") {
        if (
          (message.target === "open" || message.target === "save") &&
          message.historyAction === "clear"
        ) {
          await updateHistory(message.target, { action: "clear" });
        } else if (
          (message.target === "open" || message.target === "save") &&
          message.historyAction === "remove" &&
          isHistoryEntry(message.historyEntry)
        ) {
          await updateHistory(message.target, {
            action: "remove",
            entry: message.historyEntry,
          });
        } else {
          log.record("FAILED", "Invalid path history edit.");
          await vscode.window.showErrorMessage("Invalid path history edit.");
        }
      } else if (message?.type === "viewer-settings") {
        try {
          await context.globalState.update(
            "viewerSettings",
            viewerSettings(message.settings),
          );
        } catch (error) {
          const description =
            error instanceof Error ? error.message : String(error);
          log.record("FAILED", `Save viewer settings: ${description}`);
          await vscode.window.showErrorMessage(
            `Could not save viewer settings: ${description}`,
          );
        }
      } else if (message?.type === "ready") {
        ready = true;
        const history = pathHistory.snapshot();
        if (history.open.length || history.save.length)
          await postMessage({ type: "path-history", history });
        const storedSettings =
          context.globalState.get<unknown>("viewerSettings");
        if (storedSettings !== undefined) {
          try {
            const settings = viewerSettings(storedSettings);
            await postMessage({ type: "viewer-settings", settings });
          } catch (error) {
            const detail =
              error instanceof Error ? error.message : String(error);
            log.record("FAILED", `Restore viewer settings: ${detail}`);
            void vscode.window.showErrorMessage(
              `Could not restore viewer settings: ${detail}`,
            );
          }
        }
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
      } else if (message?.type === "search-huggingface") {
        hubSearch?.abort();
        const controller = new AbortController();
        hubSearch = controller;
        try {
          if (
            typeof message.query !== "string" ||
            typeof message.task !== "string"
          )
            throw new Error("Enter a model search and choose a task.");
          const models = await searchHuggingFaceModels(
            message.query,
            message.task,
            controller.signal,
          );
          if (!active || hubSearch !== controller) return;
          await postMessage({
            type: "huggingface-search-results",
            requestId: message.requestId,
            models,
          });
          let next = 0;
          await Promise.all(
            Array.from({ length: Math.min(3, models.length) }, async () => {
              while (
                active &&
                hubSearch === controller &&
                !controller.signal.aborted
              ) {
                const model = models[next++];
                if (!model) return;
                try {
                  const files = await listHuggingFaceModels(
                    model,
                    controller.signal,
                  );
                  if (
                    !active ||
                    hubSearch !== controller ||
                    controller.signal.aborted
                  )
                    return;
                  await postMessage({
                    type: "huggingface-model-files",
                    requestId: message.requestId,
                    id: model.id,
                    entries: hubFileEntries(files),
                  });
                } catch (error) {
                  if (
                    !active ||
                    hubSearch !== controller ||
                    controller.signal.aborted
                  )
                    return;
                  await postMessage({
                    type: "huggingface-model-files",
                    requestId: message.requestId,
                    id: model.id,
                    error:
                      error instanceof Error ? error.message : String(error),
                  });
                }
              }
            }),
          );
        } catch (error) {
          if (!active || controller.signal.aborted || hubSearch !== controller)
            return;
          await postMessage({
            type: "huggingface-search-error",
            requestId: message.requestId,
            error: error instanceof Error ? error.message : String(error),
          });
        } finally {
          if (hubSearch === controller) hubSearch = undefined;
        }
      } else if (message?.type === "select-model-path") {
        try {
          const selection = await pickLocalModel();
          await postMessage({
            type: "model-path-selected",
            requestId: message.requestId,
            location: selection?.[0] ? locationText(selection[0]) : null,
          });
        } catch (error) {
          await postMessage({
            type: "model-path-selected",
            requestId: message.requestId,
            error: error instanceof Error ? error.message : String(error),
          });
        }
      } else if (message?.type === "browse-files") {
        const target = message.target;
        try {
          if (target !== "open" && target !== "save")
            throw new Error("Invalid file selector.");
          const source = target === "save" ? "local" : message.source;
          let location = "";
          let parent: string | undefined;
          let entries: {
            name: string;
            path: string;
            directory: boolean;
            description?: string;
            size?: number;
          }[];
          if (source === "backend") {
            entries = listBackendTests().map((test) => ({
              name: test.name,
              path: String(test.id),
              directory: false,
              description: [test.modelName, test.kind, test.tag]
                .filter(Boolean)
                .join(" "),
            }));
          } else if (source === "huggingface") {
            if (typeof message.location !== "string")
              throw new Error("Enter a Hugging Face repository or model URL.");
            const parsed = parseHuggingFaceInput(message.location);
            const files =
              "fileName" in parsed
                ? [parsed]
                : await listHuggingFaceModels(parsed);
            location = message.location;
            entries = hubFileEntries(files);
          } else if (source === "local") {
            const uri = message.location
              ? locationUri(message.location)
              : target === "save"
                ? saveDirectoryUri()
                : directoryUri();
            if (target === "save") await prepareSaveDirectory(uri);
            location = locationText(uri);
            parent = locationText(vscode.Uri.joinPath(uri, ".."));
            entries = (await vscode.workspace.fs.readDirectory(uri))
              .filter(
                ([name, type]) =>
                  type & vscode.FileType.Directory ||
                  /\.(?:onnx|enc|pipe-onnx)$/i.test(name),
              )
              .map(([name, type]) => ({
                name,
                path: locationText(vscode.Uri.joinPath(uri, name)),
                directory: !!(type & vscode.FileType.Directory),
              }));
          } else throw new Error("Choose a model source.");
          entries.sort(
            (a, b) =>
              Number(b.directory) - Number(a.directory) ||
              a.name.localeCompare(b.name),
          );
          await postMessage({
            type: "file-list",
            target,
            requestId: message.requestId,
            location,
            parent,
            entries,
          });
        } catch (error) {
          await postMessage({
            type: "file-list-error",
            target,
            requestId: message.requestId,
            error: error instanceof Error ? error.message : String(error),
          });
        }
      } else if (message?.type === "open") {
        try {
          const loaded = await log.run(
            "Open ONNX model",
            async (id) => {
              const inline = message.source !== undefined;
              if (
                inline &&
                (typeof message.password !== "string" ||
                  typeof message.source !== "string" ||
                  !["local", "huggingface", "backend"].includes(message.source))
              )
                throw new Error("Invalid inline model selection.");
              const location = inline
                ? { id: String(message.source) }
                : await vscode.window.showQuickPick(
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
              if (location.id === "backend") {
                if (!inline) return chooseBackendTest(id, "Open ONNX model");
                const test = listBackendTests().find(
                  (test) => String(test.id) === message.location,
                );
                if (!test) throw new Error("Choose a backend test.");
                const reload = () => loadBackendTest(test.id);
                const native = reload();
                return {
                  model: { graph: native.graph(), native, reload },
                  name: test.name,
                  history: historyEntries("backend", String(test.id)),
                };
              }
              if (location.id === "local") {
                const selection = inline
                  ? [locationUri(message.location)]
                  : await pickLocalModel();
                if (!selection?.[0]) {
                  log.record("CANCELLED", "Open ONNX model", id);
                  return;
                }
                const loadedModel = await readGraph(
                  selection[0],
                  log,
                  undefined,
                  inline ? String(message.password) : undefined,
                );
                if (!loadedModel) {
                  log.record("CANCELLED", "Open ONNX model", id);
                  return;
                }
                return {
                  model: loadedModel,
                  name: posix.basename(selection[0].path),
                  history: historyEntries(
                    "local",
                    locationText(selection[0]),
                    locationText(vscode.Uri.joinPath(selection[0], "..")),
                  ),
                };
              }
              const input = inline
                ? message.location
                : await vscode.window.showInputBox({
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
              if (typeof input !== "string")
                throw new Error("Choose a Hugging Face model.");
              const target = parseHuggingFaceInput(input);
              if (inline && !("fileName" in target))
                throw new Error(
                  "Select a model file from the repository first.",
                );
              return vscode.window.withProgress(
                {
                  location: vscode.ProgressLocation.Notification,
                  title: `Hugging Face: ${"fileName" in target ? target.fileName : target.url}`,
                  cancellable: true,
                },
                async (progress, token) => {
                  const controller = new AbortController();
                  const cancellation = token.onCancellationRequested(() =>
                    controller.abort(),
                  );
                  if (token.isCancellationRequested) controller.abort();
                  try {
                    let reported = 0;
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
                      getApplicationSettings().downloadDirectory,
                      async (path) => {
                        await postMessage({
                          type: "model-progress",
                          target: "open",
                          actionId: message.actionId,
                          stage: "Opening model",
                        });
                        const loaded = await readGraph(
                          vscode.Uri.file(path),
                          log,
                          source.fileName,
                          inline ? String(message.password) : undefined,
                        );
                        if (!loaded) {
                          controller.abort();
                          throw new DOMException(
                            "Model loading was cancelled.",
                            "AbortError",
                          );
                        }
                        return loaded;
                      },
                      controller.signal,
                      ({ received, total }) => {
                        if (controller.signal.aborted || !active) return;
                        const percent = total
                          ? Math.min(100, (received / total) * 100)
                          : undefined;
                        progress.report({
                          message: `Downloaded ${received.toLocaleString()}${total ? ` / ${total.toLocaleString()}` : ""} bytes`,
                          ...(percent === undefined
                            ? {}
                            : { increment: percent - reported }),
                        });
                        reported = percent ?? reported;
                        void postMessage({
                          type: "model-progress",
                          target: "open",
                          actionId: message.actionId,
                          stage: "Downloading Hugging Face model",
                          received,
                          total,
                        });
                      },
                    );
                    if (result.value.source)
                      result.value.source.uri = vscode.Uri.file(
                        result.filePath,
                      );
                    log.record(
                      "SUCCEEDED",
                      `${result.cached ? "Cache hit" : "Downloaded and cached"}: ${source.url}`,
                      undefined,
                      result.filePath,
                    );
                    const url = new URL(source.url);
                    return {
                      model: result.value,
                      name: source.fileName,
                      history: historyEntries(
                        "huggingface",
                        source.url,
                        `${url.origin}/${url.pathname.split("/").slice(1, 3).join("/")}`,
                      ),
                    };
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
          await updateHistory("open", {
            action: "remember",
            entries: loaded.history,
          });
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
        let loaded: Awaited<ReturnType<typeof chooseBackendTest>>;
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
        await updateHistory("open", {
          action: "remember",
          entries: loaded.history,
        });
      } else if (message?.type === "cancel-fusion-diagnostic") {
        diagnosticRequest++;
      } else if (message?.type === "explain-fusion") {
        const owner = model;
        const revision = graphRevision;
        const generation = ++diagnosticRequest;
        const isCurrent = () =>
          active &&
          model === owner &&
          graphRevision === revision &&
          diagnosticRequest === generation;
        try {
          await log.run(
            "Explain fusion match",
            async () => {
              const current = requireModel();
              if (!(
                current.graph.canFuse ?? current.graph.breadcrumbs.length === 1
              ))
                throw new Error(
                  "Return to a model's main graph to inspect patterns.",
                );
              const native = await nativeModel(
                current,
                log,
                () => void sendStatus(),
              );
              if (!isCurrent()) return;
              const path = current.graph.breadcrumbs.at(-1)?.path ?? [];
              if (
                typeof message.pattern !== "string" ||
                !native
                  .optimizationPatterns(path)
                  .some((pattern) => pattern.name === message.pattern) ||
                typeof message.anchor !== "number" ||
                !Number.isSafeInteger(message.anchor) ||
                message.anchor < 0 ||
                message.anchor >= current.graph.nodes.length
              )
                throw new Error(
                  "Choose a native pattern and one operator node.",
                );
              const diagnostic = await native.explainFusion(
                message.pattern,
                message.anchor,
                current.source?.uri.scheme === "file"
                  ? current.source.uri.fsPath
                  : undefined,
                path,
              );
              if (!isCurrent()) return;
              await postMessage({
                type: "fusion-diagnostic",
                requestId: message.requestId,
                diagnostic,
              });
            },
            message.actionId,
          );
        } catch (error) {
          if (!isCurrent()) return;
          const detail = error instanceof Error ? error.message : String(error);
          await postMessage({
            type: "fusion-diagnostic-error",
            requestId: message.requestId,
            error: detail,
          });
          await vscode.window.showErrorMessage(
            `Could not explain fusion match: ${detail}`,
          );
        }
      } else if (message?.type === "cancel-fusion") {
        fusionPreview = undefined;
        fusionRequest++;
      } else if (
        message?.type === "undo-fusion" ||
        message?.type === "apply-rewriting" ||
        message?.type === "fusion-patterns" ||
        message?.type === "preview-fusion" ||
        message?.type === "apply-optimization" ||
        message?.type === "apply-fusion"
      ) {
        const owner = model;
        const revision = graphRevision;
        const generation = ++fusionRequest;
        const undoGeneration =
          message.type === "undo-fusion" ? ++undoRequest : undoRequest;
        const isCurrent = () =>
          active &&
          model === owner &&
          graphRevision === revision &&
          (message.type === "undo-fusion"
            ? undoRequest === undoGeneration
            : fusionRequest === generation);
        try {
          await log.run(
            message.type === "undo-fusion"
              ? "Undo fusion"
              : message.type === "apply-optimization"
                ? "Optimize ONNX model"
                : message.type === "apply-rewriting"
                  ? "Apply rewriting scenario"
                  : message.type === "apply-fusion"
                    ? "Apply fusion pattern"
                    : message.type === "preview-fusion"
                      ? "Preview fusion pattern"
                      : "List fusion patterns",
            async (id) => {
              const current = requireModel();
              if (message.type === "undo-fusion") {
                if (
                  message.all !== undefined &&
                  typeof message.all !== "boolean"
                )
                  throw new Error("Expected a boolean unfuse-all option.");
                let edited = await nativeModel(
                  current,
                  log,
                  () => void sendStatus(),
                );
                if (!isCurrent()) return;
                if (message.all === true) {
                  while (edited.fusionCount()) edited = edited.undoFusion();
                } else {
                  edited = edited.undoFusion();
                }
                const currentPath =
                  current.graph.breadcrumbs.at(-1)?.path ?? [];
                const path =
                  currentPath[0] && "pipeline" in currentPath[0]
                    ? [currentPath[0]]
                    : [];
                const graph = edited.graph(false, path);
                const retained = edited.clone();
                model = {
                  ...current,
                  native: edited,
                  nativeReload: undefined,
                  graph,
                  metadata: undefined,
                  fusionCount: edited.fusionCount(),
                  fusionHistory: edited.fusionHistory(),
                  reload: () => retained.clone(),
                };
                graphRevision++;
                navigationRequest++;
                panel.title = `ONNX: ${current.fileName ?? graph.name}${model.pythonOptimized || model.fusionCount ? " (modified)" : ""}`;
                await sendGraph(undefined, message.requestId);
                return;
              }
              if (!(
                current.graph.canFuse ?? current.graph.breadcrumbs.length === 1
              ))
                throw new Error(
                  "Return to a model's main graph to apply fusion patterns.",
                );
              const path = current.graph.breadcrumbs.at(-1)?.path ?? [];
              if (message.type === "apply-optimization") {
                if (
                  message.backend !== undefined &&
                  message.backend !== "onnx-light" &&
                  message.backend !== "onnxscript"
                )
                  throw new Error("Choose a supported optimization backend.");
                if (message.backend === "onnxscript") {
                  if (
                    message.operation !== "Optimize" ||
                    !Array.isArray(message.patterns) ||
                    message.patterns.length
                  )
                    throw new Error(
                      "ONNX Script uses its default optimizer and rewrite rules.",
                    );
                  if (!vscode.workspace.isTrusted)
                    throw new Error(
                      "Trust this workspace before running ONNX Script.",
                    );
                  if (current.graph.pipeline)
                    throw new Error(
                      "ONNX Script requires a single ONNX model.",
                    );
                  const native = await nativeModel(
                    current,
                    log,
                    () => void sendStatus(),
                  );
                  if (!isCurrent()) return;
                  const bytes = await native.serialize(
                    undefined,
                    current.source?.uri.scheme === "file"
                      ? current.source.uri.fsPath
                      : undefined,
                    false,
                  );
                  if (!isCurrent()) return;
                  const result = await optimizeWithOnnxScript(
                    bytes,
                    getApplicationSettings().pythonPath,
                    context.globalStorageUri.fsPath,
                  );
                  if (!isCurrent()) return;
                  const edited = result.changed
                    ? loadModel(result.bytes, current.graph.name)
                    : native;
                  edited.verify();
                  let replacement = current;
                  if (result.changed) {
                    const retained = edited.clone();
                    replacement = {
                      ...current,
                      native: edited,
                      nativeReload: undefined,
                      graph: edited.graph(false),
                      metadata: undefined,
                      fusionCount: edited.fusionCount(),
                      fusionHistory: edited.fusionHistory(),
                      pythonOptimized: true,
                      reload: () => retained.clone(),
                    };
                  }
                  fusionPreview = undefined;
                  model = replacement;
                  graphRevision++;
                  navigationRequest++;
                  panel.title = `ONNX: ${current.fileName ?? replacement.graph.name}${replacement.pythonOptimized || replacement.fusionCount ? " (modified)" : ""}`;
                  await sendGraph(undefined, message.requestId);
                  if (
                    !active ||
                    model !== replacement ||
                    fusionRequest !== generation + 1
                  )
                    return;
                  log.record(
                    "SUCCEEDED",
                    "Optimize ONNX model",
                    id,
                    result.logs,
                  );
                  await postMessage({
                    type: "optimization-info",
                    requestId: message.requestId,
                    backend: "onnxscript",
                    operation: "Optimize",
                    changed: result.changed,
                    logs: result.logs,
                    matches: [],
                    optimizationReport: null,
                    undoSupported: false,
                    notice: result.changed
                      ? undefined
                      : "No changes. Existing native undo history is retained.",
                    patterns: edited.fusionPatterns(),
                    optimizationPatterns: edited.optimizationPatterns(),
                    scenarios: edited.rewritingScenarios(),
                  });
                  return;
                }
                const native = await nativeModel(
                  current,
                  log,
                  () => void sendStatus(),
                );
                if (!isCurrent()) return;
                const chosen: unknown = message.patterns;
                const catalog = native.optimizationPatterns(path);
                if (
                  message.operation !== "Optimize" ||
                  !Array.isArray(chosen) ||
                  !chosen.length ||
                  !chosen.every(
                    (name: unknown) =>
                      typeof name === "string" &&
                      catalog.some((pattern) => pattern.name === name),
                  ) ||
                  new Set(chosen).size !== chosen.length
                )
                  throw new Error("Choose registered optimization patterns.");
                fusionPreview = undefined;
                const edited = await native.optimize(
                  chosen,
                  "Optimize",
                  current.source?.uri.scheme === "file"
                    ? current.source.uri.fsPath
                    : undefined,
                  path,
                );
                if (!isCurrent()) return;
                const changed = edited.fusionCount() > native.fusionCount();
                const graph = edited.graph(false, path);
                const matches = changed
                  ? (edited.fusionHistory().at(-1)?.matches ?? [])
                  : [];
                const retained = edited.clone();
                const replacement: LoadedModel = {
                  ...current,
                  native: edited,
                  nativeReload: undefined,
                  graph,
                  metadata: undefined,
                  fusionCount: edited.fusionCount(),
                  fusionHistory: edited.fusionHistory(),
                  reload: () => retained.clone(),
                };
                model = replacement;
                graphRevision++;
                navigationRequest++;
                panel.title = `ONNX: ${current.fileName ?? graph.name}${replacement.pythonOptimized || replacement.fusionCount ? " (modified)" : ""}`;
                await sendGraph(undefined, message.requestId);
                if (
                  !active ||
                  model !== replacement ||
                  fusionRequest !== generation + 1
                )
                  return;
                await postMessage({
                  type: "optimization-info",
                  requestId: message.requestId,
                  operation: "Optimize",
                  optimizationReport: edited.optimizationReport(path),
                  changed,
                  matches,
                  patterns: edited.fusionPatterns(path),
                  optimizationPatterns: edited.optimizationPatterns(path),
                  scenarios: edited.rewritingScenarios(path),
                });
                return;
              }
              if (message.type === "apply-rewriting") {
                if (
                  typeof message.scenario !== "string" ||
                  typeof message.all !== "boolean"
                )
                  throw new Error(
                    "Choose a rewriting scenario and an all-steps option.",
                  );
                const native = await nativeModel(
                  current,
                  log,
                  () => void sendStatus(),
                );
                if (!isCurrent()) return;
                const edited = await native.applyRewritingScenario(
                  message.scenario,
                  message.all,
                  current.source?.uri.scheme === "file"
                    ? current.source.uri.fsPath
                    : undefined,
                  path,
                );
                if (!isCurrent()) return;
                const retained = edited.clone();
                const graph = edited.graph(false, path);
                const replacement: LoadedModel = {
                  ...current,
                  native: edited,
                  nativeReload: undefined,
                  graph,
                  metadata: undefined,
                  fusionCount: edited.fusionCount(),
                  fusionHistory: edited.fusionHistory(),
                  reload: () => retained.clone(),
                };
                model = replacement;
                graphRevision++;
                navigationRequest++;
                panel.title = `ONNX: ${current.fileName ?? graph.name} (modified)`;
                await sendGraph(undefined, message.requestId);
                if (
                  !active ||
                  model !== replacement ||
                  fusionRequest !== generation + 1
                )
                  return;
                await postMessage({
                  type: "rewriting-info",
                  requestId: message.requestId,
                  name: message.scenario,
                  patterns: edited.fusionPatterns(path),
                  optimizationPatterns: edited.optimizationPatterns(path),
                  scenarios: edited.rewritingScenarios(path),
                });
                return;
              }
              if (message.type === "apply-fusion") {
                const preview = fusionPreview;
                const candidates: unknown = message.candidates;
                const selectedMatches =
                  preview &&
                  Array.isArray(candidates) &&
                  candidates.length &&
                  new Set(candidates).size === candidates.length &&
                  candidates.every(
                    (id: unknown) =>
                      typeof id === "number" &&
                      Number.isSafeInteger(id) &&
                      id >= 0 &&
                      id < preview.matches.length,
                  )
                    ? candidates.map((id: number) => preview.matches[id])
                    : undefined;
                const anchors = selectedMatches
                  ? selectedMatches.map((match) => match.anchor)
                  : message.anchors;
                if (
                  !preview ||
                  preview.owner !== current ||
                  preview.native !== current.native ||
                  preview.revision !== revision ||
                  preview.id !== message.previewId ||
                  (candidates !== undefined && !selectedMatches) ||
                  !Array.isArray(anchors) ||
                  !anchors.length ||
                  !anchors.every(
                    (anchor: unknown) =>
                      typeof anchor === "number" &&
                      Number.isSafeInteger(anchor) &&
                      preview.anchors.has(anchor),
                  )
                )
                  throw new Error(
                    "The fusion preview is stale or invalid. Preview again.",
                  );
                if (
                  selectedMatches?.some((match) => match.global) &&
                  selectedMatches.length !== 1
                )
                  throw new Error("Apply a global pass individually.");
                const edited = await preview.snapshot.applyFusion(
                  selectedMatches
                    ? selectedMatches.map((match) => match.pattern)
                    : preview.pattern,
                  anchors,
                  path,
                );
                if (!isCurrent()) return;
                const retained = edited.clone();
                const graph = edited.graph(false, path);
                const replacement: LoadedModel = {
                  ...current,
                  native: edited,
                  nativeReload: undefined,
                  graph,
                  metadata: undefined,
                  fusionCount: edited.fusionCount(),
                  fusionHistory: edited.fusionHistory(),
                  reload: () => retained.clone(),
                };
                model = replacement;
                fusionPreview = undefined;
                graphRevision++;
                navigationRequest++;
                panel.title = `ONNX: ${current.fileName ?? graph.name} (modified)`;
                await sendGraph(undefined, message.requestId);
                if (
                  !active ||
                  model !== replacement ||
                  fusionRequest !== generation + 1
                )
                  return;
                await postMessage({
                  type: "fusion-info",
                  requestId: message.requestId,
                  count: anchors.length,
                  patterns: edited.fusionPatterns(path),
                  optimizationPatterns: edited.optimizationPatterns(path),
                  scenarios: edited.rewritingScenarios(path),
                });
                return;
              }
              fusionPreview = undefined;
              const native = await nativeModel(
                current,
                log,
                () => void sendStatus(),
              );
              if (!isCurrent()) return;
              const patterns = native.fusionPatterns(path);
              if (message.type === "fusion-patterns") {
                await postMessage({
                  type: "fusion-patterns",
                  requestId: message.requestId,
                  patterns,
                  optimizationPatterns: native.optimizationPatterns(path),
                  optimizationReport: native.optimizationReport(path),
                  scenarios: native.rewritingScenarios(path),
                });
                return;
              }
              const chosen: unknown = message.patterns ?? [message.pattern];
              if (
                !Array.isArray(chosen) ||
                !chosen.length ||
                !chosen.every(
                  (pattern: unknown) =>
                    typeof pattern === "string" && patterns.includes(pattern),
                ) ||
                new Set(chosen).size !== chosen.length
              )
                throw new Error("Choose applicable fusion patterns.");
              const matches: (FusionMatch & { pattern: string })[] = [];
              let snapshot: NativeModel | undefined;
              for (const pattern of chosen) {
                const preview = await native.previewFusion(
                  pattern,
                  current.source?.uri.scheme === "file"
                    ? current.source.uri.fsPath
                    : undefined,
                  path,
                );
                if (!isCurrent()) return;
                snapshot ??= preview.model;
                matches.push(
                  ...preview.matches.map((match) => ({ ...match, pattern })),
                );
              }
              if (!snapshot) throw new Error("No fusion snapshot was created.");
              fusionPreview = {
                id: generation,
                owner: current,
                native,
                snapshot,
                revision,
                pattern: chosen[0],
                anchors: new Set(matches.map((match) => match.anchor)),
                matches,
              };
              await postMessage({
                type: "fusion-preview",
                requestId: message.requestId,
                previewId: generation,
                matches,
              });
            },
            message.actionId,
          );
        } catch (error) {
          if (!isCurrent()) return;
          const detail = error instanceof Error ? error.message : String(error);
          await postMessage({
            type:
              message.type === "undo-fusion"
                ? "undo-fusion-error"
                : "fusion-error",
            requestId: message.requestId,
            error: detail,
          });
          await vscode.window.showErrorMessage(
            `${message.type === "undo-fusion" ? "Could not undo fusion" : "Could not fuse ONNX model"}: ${detail}`,
          );
        }
      } else if (message?.type === "save") {
        const requestedModel = model;
        try {
          await log.run(
            "Save ONNX model",
            async (id) => {
              const current = requireModel();
              if (
                (message.storage !== undefined &&
                  message.storage !== "embedded" &&
                  message.storage !== "external" &&
                  message.storage !== "reuse") ||
                (message.encrypted !== undefined &&
                  typeof message.encrypted !== "boolean")
              )
                throw new Error("Invalid model save format.");
              const format = {
                external:
                  message.storage === "external" || message.storage === "reuse",
                reuse: message.storage === "reuse",
                encrypted: message.encrypted === true,
              };
              if (current.graph.pipeline && format.encrypted)
                throw new Error(
                  "PipelineProto encryption is not supported by onnx-light.",
                );
              if (format.external && format.encrypted)
                throw new Error(
                  "Encrypted external weights await onnx-light support: https://github.com/xadupre/onnx-light/issues/5533",
                );
              const maxExternalFileSize = message.maxExternalFileSize ?? 0;
              if (
                message.includeRewritings !== undefined &&
                typeof message.includeRewritings !== "boolean"
              )
                throw new Error("Invalid rewriting save choice.");
              if (
                typeof maxExternalFileSize !== "number" ||
                !Number.isSafeInteger(maxExternalFileSize) ||
                (format.external && maxExternalFileSize <= 0)
              )
                throw new Error(
                  "Choose a positive, safe integer maximum external weight file size.",
                );
              const cancelled = () => {
                log.record("CANCELLED", "Save ONNX model", id);
              };
              let password: string | undefined;
              const inline = message.location !== undefined;
              if (format.encrypted) {
                if (
                  inline &&
                  (typeof message.password !== "string" ||
                    !message.password.length ||
                    message.confirmation !== message.password)
                )
                  throw new Error(
                    "Enter and confirm the same non-empty password.",
                  );
                password = inline
                  ? String(message.password)
                  : await vscode.window.showInputBox({
                      title: "Encrypt ONNX model",
                      prompt: "Choose a password for the saved copy.",
                      password: true,
                      ignoreFocusOut: true,
                      validateInput: (value) =>
                        value.length ? undefined : "A password is required.",
                    });
                if (password === undefined) return cancelled();
                const confirmation = inline
                  ? message.confirmation
                  : await vscode.window.showInputBox({
                      title: "Confirm ONNX model password",
                      prompt: "Enter the same password again.",
                      password: true,
                      ignoreFocusOut: true,
                      validateInput: (value) =>
                        value === password
                          ? undefined
                          : "Passwords do not match.",
                    });
                if (confirmation === undefined) return cancelled();
                if (confirmation !== password)
                  throw new Error("Passwords do not match.");
              }
              const fileName = copyFileName(current, format.encrypted);
              if (inline && typeof message.overwrite !== "boolean")
                throw new Error("Invalid overwrite choice.");
              if (!inline) {
                try {
                  await prepareSaveDirectory(saveDirectoryUri());
                } catch (error) {
                  log.record(
                    "FAILED",
                    "Prepare suggested save directory",
                    undefined,
                    error instanceof Error ? error.message : String(error),
                  );
                }
              }
              const destination = inline
                ? locationUri(message.location)
                : await vscode.window.showSaveDialog({
                    title: "Save ONNX model copy",
                    saveLabel: "Save model copy",
                    filters: format.encrypted
                      ? { "Encrypted ONNX models": ["enc"] }
                      : current.graph.pipeline
                        ? { "ONNX pipelines": ["pipe-onnx"] }
                        : { "ONNX models": ["onnx"] },
                    defaultUri: vscode.Uri.joinPath(
                      saveDirectoryUri(),
                      fileName,
                    ),
                  });
              if (!destination || !active || model !== current)
                return cancelled();
              const source = current.source?.uri;
              const originalSources = source
                ? [
                    source,
                    ...(current.sourceWeightLocations ?? []).map((location) =>
                      vscode.Uri.joinPath(source, "..", location),
                    ),
                  ]
                : [];
              await assertSaveDestination(originalSources, destination);
              await prepareSaveDirectory(
                vscode.Uri.joinPath(destination, ".."),
              );
              if (format.reuse) {
                if (
                  !source ||
                  source.scheme !== "file" ||
                  destination.scheme !== "file"
                )
                  throw new Error(
                    "Reusing weights requires a local source and destination.",
                  );
                const [original, target] = await Promise.all([
                  stat(dirname(source.fsPath), { bigint: true }),
                  stat(dirname(destination.fsPath), { bigint: true }),
                ]);
                if (original.dev !== target.dev || original.ino !== target.ino)
                  throw new Error(
                    "Save in the source folder to reuse relative external weights.",
                  );
              }
              const isCurrent = () => active && model === current;
              const native = await nativeModel(current, log, () => {
                void sendStatus();
              });
              const sourcePath =
                current.source?.uri.scheme === "file"
                  ? current.source.uri.fsPath
                  : undefined;
              await postMessage({
                type: "model-progress",
                target: "save",
                actionId: message.actionId,
                stage: "Serializing model and weights",
              });
              const serialized = format.external
                ? await native.serializeFiles(
                    maxExternalFileSize,
                    `${posix.basename(destination.path)}.${randomBytes(12).toString("hex")}.data`,
                    sourcePath,
                    format.reuse,
                    message.includeRewritings ?? false,
                  )
                : {
                    model: await native.serialize(
                      password,
                      sourcePath,
                      message.includeRewritings ?? false,
                    ),
                    weights: [],
                  };
              if (!active || model !== current) return cancelled();
              const sources = source
                ? [
                    ...originalSources,
                    ...native
                      .externalDataLocations()
                      .map((location) =>
                        vscode.Uri.joinPath(source, "..", location),
                      ),
                  ]
                : [];
              await assertSaveDestination(sources, destination);
              if (!isCurrent()) return cancelled();
              const writtenWeights: vscode.Uri[] = [];
              let committed = false;
              try {
                for (const file of serialized.weights) {
                  await postMessage({
                    type: "model-progress",
                    target: "save",
                    actionId: message.actionId,
                    stage: `Writing weight file ${writtenWeights.length + 1} of ${serialized.weights.length}`,
                  });
                  const uri = vscode.Uri.joinPath(destination, "..", file.name);
                  if (
                    !(await writeModelFile(uri, file.bytes, false, isCurrent))
                  )
                    return cancelled();
                  writtenWeights.push(uri);
                  if (!active || model !== current) return cancelled();
                }
                await postMessage({
                  type: "model-progress",
                  target: "save",
                  actionId: message.actionId,
                  stage: "Writing model file",
                });
                const published = await writeModelFile(
                  destination,
                  serialized.model,
                  inline ? message.overwrite === true : true,
                  async () => {
                    if (!isCurrent()) return false;
                    await assertSaveDestination(sources, destination);
                    return isCurrent();
                  },
                );
                if (!published) return cancelled();
                committed = true;
              } finally {
                if (!committed)
                  for (const uri of writtenWeights)
                    await vscode.workspace.fs.delete(uri);
              }
              await updateHistory("save", {
                action: "remember",
                entries: historyEntries(
                  "local",
                  locationText(destination),
                  locationText(vscode.Uri.joinPath(destination, "..")),
                ),
              });
              await postMessage({
                type: "save-info",
                name: posix.basename(destination.path),
              });
            },
            message.actionId,
          );
        } catch (error) {
          if (!active || model !== requestedModel) return;
          const detail = error instanceof Error ? error.message : String(error);
          await postMessage({ type: "save-error", error: detail });
          await vscode.window.showErrorMessage(
            `Could not save ONNX model: ${detail}`,
          );
        }
      } else if (
        message?.type === "navigate" ||
        message?.type === "select-fusion"
      ) {
        const requestedModel = model;
        const requestedNavigation = ++navigationRequest;
        let graph: Graph;
        let selectedNodes: number[] | undefined;
        try {
          graph = await log.run(
            `Navigate graph: ${model?.graph.name ?? "no model loaded"}`,
            async () => {
              const current = requireModel();
              let path = message.path;
              if (message.type === "select-fusion") {
                const operation =
                  typeof message.operation === "number" &&
                  Number.isSafeInteger(message.operation) &&
                  message.operation >= 0
                    ? current.fusionHistory?.[message.operation]
                    : undefined;
                const match =
                  typeof message.match === "number" &&
                  Number.isSafeInteger(message.match) &&
                  message.match >= 0
                    ? operation?.matches[message.match]
                    : undefined;
                if (!operation || !match || match.pattern !== message.pattern)
                  throw new Error("Invalid applied fusion match.");
                path = operation.stage ? [{ pipeline: operation.stage }] : [];
                selectedNodes = match.nodes;
              }
              if (!isGraphPath(path)) throw new Error("Invalid subgraph path.");
              const native = await nativeModel(current, log, () => {
                void sendStatus();
              });
              const graph = native.graph(false, path);
              if (
                message.type === "navigate" &&
                message.nodeIndex !== undefined
              ) {
                if (
                  typeof message.nodeIndex !== "number" ||
                  !Number.isSafeInteger(message.nodeIndex) ||
                  message.nodeIndex < 0 ||
                  !graph.nodes.some(
                    (node) => node.id === `node-${message.nodeIndex}`,
                  )
                )
                  throw new Error("Invalid navigation node index.");
                selectedNodes = [message.nodeIndex];
              }
              return graph;
            },
            message.actionId,
          );
        } catch (error) {
          if (
            !active ||
            model !== requestedModel ||
            navigationRequest !== requestedNavigation
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
          navigationRequest !== requestedNavigation
        )
          return;
        requireModel().graph = graph;
        requireModel().metadata = undefined;
        graphRevision++;
        await sendGraph(selectedNodes);
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
      } else if (message?.type === "verify") {
        const requestedModel = model;
        const requestedGraphRevision = graphRevision;
        const isCurrent = () =>
          active &&
          model === requestedModel &&
          graphRevision === requestedGraphRevision;
        try {
          await log.run(
            `Verify model: ${model?.graph.name ?? "no model loaded"}`,
            async () => {
              const current = requireModel();
              const native = await nativeModel(current, log, () => {
                void sendStatus();
              });
              if (isCurrent()) native.verify();
            },
            message.actionId,
          );
        } catch (error) {
          if (!isCurrent()) return;
          const detail = error instanceof Error ? error.message : String(error);
          await postMessage({ type: "verify-error", error: detail });
          await vscode.window.showErrorMessage(
            `ONNX verification failed: ${detail}`,
          );
          return;
        }
        if (!isCurrent()) return;
        await postMessage({ type: "verify-info" });
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
    hubSearch?.abort();
    exporting?.abort();
    recipeController.abort();
    model = undefined;
    stopHistory();
    stopExportHistory();
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
    context.extension.packageJSON.version,
  );
  if (uri) {
    try {
      const loaded = await readGraph(uri, log);
      if (!active) return;
      model = loaded;
      initialLoading = false;
      if (ready && model) await sendGraph();
      else if (ready) await postMessage({ type: "load-state", pending: false });
      else void sendStatus();
      if (loaded && !uri.path.startsWith(`${mediaRoot.path}/`)) {
        await updateHistory("open", {
          action: "remember",
          entries: historyEntries(
            "local",
            locationText(uri),
            locationText(vscode.Uri.joinPath(uri, "..")),
          ),
        });
      }
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
