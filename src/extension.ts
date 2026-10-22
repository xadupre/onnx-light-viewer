/**
 * Owns panel-local native models and routes webview requests.
 * Only display snapshots cross into the webview; model bytes stay in the host.
 */
import * as vscode from "vscode";
import { randomBytes } from "node:crypto";
import { posix } from "node:path";
import {
  listBackendTests,
  loadBackendTest,
  loadModel,
  onnxLightVersion,
  Graph,
  NativeModel,
  GraphMetadata,
  GraphPath,
} from "./graph";
import { webviewHtml } from "./webview";

const viewType = "onnxLightViewer.graph";

interface LoadedModel {
  graph: Graph;
  native?: NativeModel;
  reload: () => NativeModel;
  metadata?: GraphMetadata;
}

function logStatus(
  output: vscode.OutputChannel,
  status: "RUNNING" | "SUCCEEDED" | "FAILED" | "INVALIDATED",
  action: string,
): void {
  output.appendLine(`[${new Date().toISOString()}] [${status}] ${action}`);
}

async function runAction<T>(
  output: vscode.OutputChannel,
  name: string,
  action: () => T | Promise<T>,
): Promise<T> {
  logStatus(output, "RUNNING", name);
  try {
    const result = await action();
    logStatus(output, "SUCCEEDED", name);
    return result;
  } catch (error) {
    const detail =
      error instanceof Error ? (error.stack ?? error.message) : String(error);
    logStatus(output, "FAILED", `${name}\n${detail}`);
    throw error;
  }
}

async function nativeModel(
  model: LoadedModel,
  output: vscode.OutputChannel,
): Promise<NativeModel> {
  if (!model.native) {
    model.native = await runAction(
      output,
      `Reload native model: ${model.graph.name}`,
      model.reload,
    );
  }
  return model.native;
}

function invalidateModel(
  model: LoadedModel,
  output: vscode.OutputChannel,
): void {
  model.native = undefined;
  logStatus(
    output,
    "INVALIDATED",
    `${model.graph.name}: the next native action will reload the original model; inferred information may need refreshing.`,
  );
}

export function activate(context: vscode.ExtensionContext): void {
  const output = vscode.window.createOutputChannel("ONNX Light Viewer");
  context.subscriptions.push(
    output,
    vscode.commands.registerCommand("onnxLightViewer.showLogs", () =>
      output.show(true),
    ),
  );
  for (const [command, fileName] of [
    ["onnxLightViewer.open", "relu.onnx"],
    ["onnxLightViewer.openSymbolicExample", "tiny-llm.onnx"],
  ]) {
    context.subscriptions.push(
      vscode.commands.registerCommand(command, async () => {
        const model = await readGraph(
          vscode.Uri.joinPath(context.extensionUri, "media", fileName),
          output,
        );
        const panel = vscode.window.createWebviewPanel(
          viewType,
          "ONNX Light Viewer",
          vscode.ViewColumn.One,
          { enableScripts: true },
        );
        configurePanel(context, panel, model, output);
      }),
    );
  }
  context.subscriptions.push(
    vscode.window.registerCustomEditorProvider(viewType, {
      openCustomDocument: (uri) => ({ uri, dispose() {} }),
      resolveCustomEditor: async (document, panel) => {
        const model = await readGraph(document.uri, output);
        configurePanel(context, panel, model, output);
      },
    }),
  );
}

async function readGraph(
  uri: vscode.Uri,
  output: vscode.OutputChannel,
): Promise<LoadedModel> {
  return runAction(output, `Load model: ${uri.path}`, async () => {
    const bytes = await vscode.workspace.fs.readFile(uri);
    const fileName = posix.basename(uri.path);
    const reload = () => loadModel(bytes, fileName);
    const native = reload();
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
        ["node", "attribute", "graph"].every((key) => {
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
 * Navigation invalidates metadata; a ready event resends the current snapshots.
 */
function configurePanel(
  context: vscode.ExtensionContext,
  panel: vscode.WebviewPanel,
  initialModel: LoadedModel,
  output: vscode.OutputChannel,
): void {
  const mediaRoot = vscode.Uri.joinPath(context.extensionUri, "media");
  panel.webview.options = {
    enableScripts: true,
    localResourceRoots: [mediaRoot],
  };
  panel.iconPath = vscode.Uri.joinPath(mediaRoot, "logo.svg");
  let model = initialModel;
  const sendGraph = () =>
    panel.webview.postMessage({
      type: "graph",
      graph: model.graph,
      ...(model.metadata ? { metadata: model.metadata } : {}),
    });
  const messages = panel.webview.onDidReceiveMessage(
    async (message: { type?: string; path?: unknown } | null) => {
      if (message?.type === "show-logs") {
        output.show(true);
      } else if (message?.type === "ready") {
        await sendGraph();
      } else if (message?.type === "open-backend-test") {
        let loadedModel: LoadedModel;
        let name: string;
        try {
          const tests = await runAction(
            output,
            "List backend tests",
            listBackendTests,
          );
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
          if (!selection) return;
          name = selection.label;
          loadedModel = await runAction(
            output,
            `Load backend test: ${name}`,
            () => {
              const reload = () => loadBackendTest(selection.id);
              const native = reload();
              return { graph: native.graph(), native, reload };
            },
          );
        } catch (error) {
          await vscode.window.showErrorMessage(
            `Could not open onnx-light backend test: ${error instanceof Error ? error.message : String(error)}`,
          );
          return;
        }
        model = loadedModel;
        panel.title = `ONNX: ${name}`;
        await sendGraph();
      } else if (message?.type === "open") {
        const selection = await vscode.window.showOpenDialog({
          canSelectMany: false,
          filters: { "ONNX models": ["onnx"] },
          openLabel: "Open ONNX graph",
        });
        if (selection?.[0]) {
          let loadedModel: LoadedModel;
          try {
            loadedModel = await readGraph(selection[0], output);
          } catch (error) {
            await vscode.window.showErrorMessage(
              `Could not open ONNX model: ${error instanceof Error ? error.message : String(error)}`,
            );
            return;
          }
          model = loadedModel;
          panel.title = `ONNX: ${posix.basename(selection[0].path)}`;
          await sendGraph();
        }
      } else if (message?.type === "navigate") {
        let graph: Graph;
        try {
          graph = await runAction(
            output,
            `Navigate graph: ${model.graph.name}`,
            async () => {
              if (!isGraphPath(message.path))
                throw new Error("Invalid subgraph path.");
              const native = await nativeModel(model, output);
              return native.graph(false, message.path);
            },
          );
        } catch (error) {
          const detail = error instanceof Error ? error.message : String(error);
          await panel.webview.postMessage({
            type: "navigation-error",
            error: detail,
          });
          await vscode.window.showErrorMessage(
            `Could not open ONNX subgraph: ${detail}`,
          );
          return;
        }
        model.graph = graph;
        model.metadata = undefined;
        await sendGraph();
      } else if (message?.type === "metadata") {
        let metadata: GraphMetadata;
        try {
          metadata = await runAction(
            output,
            `Refresh metadata: ${model.graph.name}`,
            async () => {
              const native = await nativeModel(model, output);
              return native.metadata(model.graph.breadcrumbs.at(-1)?.path);
            },
          );
        } catch (error) {
          invalidateModel(model, output);
          const detail = error instanceof Error ? error.message : String(error);
          await panel.webview.postMessage({
            type: "metadata-error",
            error: detail,
          });
          await vscode.window.showErrorMessage(
            `Could not analyze ONNX metadata: ${detail}`,
          );
          return;
        }
        model.metadata = metadata;
        await panel.webview.postMessage({ type: "metadata-info", metadata });
      } else if (message?.type === "refresh-shapes") {
        let refreshed: Graph;
        try {
          refreshed = await runAction(
            output,
            `Refresh shapes: ${model.graph.name}`,
            async () => {
              const native = await nativeModel(model, output);
              return native.graph(true, model.graph.breadcrumbs.at(-1)?.path);
            },
          );
        } catch (error) {
          invalidateModel(model, output);
          const detail = error instanceof Error ? error.message : String(error);
          await panel.webview.postMessage({
            type: "shape-error",
            error: detail,
          });
          await vscode.window.showErrorMessage(
            `Could not infer ONNX shapes/types: ${detail}`,
          );
          return;
        }
        model.graph = refreshed;
        await panel.webview.postMessage({
          type: "shape-info",
          tensors: refreshed.tensors,
        });
      }
    },
  );
  panel.onDidDispose(() => messages.dispose());
  const resource = (name: string) =>
    panel.webview.asWebviewUri(vscode.Uri.joinPath(mediaRoot, name)).toString();
  const nonce = randomBytes(16).toString("hex");
  panel.webview.html = webviewHtml(
    resource,
    panel.webview.cspSource,
    nonce,
    onnxLightVersion,
  );
}
