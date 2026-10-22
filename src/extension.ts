import * as vscode from "vscode";
import { randomBytes } from "node:crypto";
import { posix } from "node:path";
import {
  listBackendTests,
  loadBackendTest,
  loadModel,
  Graph,
  NativeModel,
  GraphMetadata,
} from "./graph";
import { webviewHtml } from "./webview";

const viewType = "onnxLightViewer.graph";

interface LoadedModel {
  graph: Graph;
  native: NativeModel;
  metadata?: GraphMetadata;
}

export function activate(context: vscode.ExtensionContext): void {
  for (const [command, fileName] of [
    ["onnxLightViewer.open", "relu.onnx"],
    ["onnxLightViewer.openSymbolicExample", "tiny-llm.onnx"],
  ]) {
    context.subscriptions.push(
      vscode.commands.registerCommand(command, async () => {
        const model = await readGraph(
          vscode.Uri.joinPath(context.extensionUri, "media", fileName),
        );
        const panel = vscode.window.createWebviewPanel(
          viewType,
          "ONNX Light Viewer",
          vscode.ViewColumn.One,
          { enableScripts: true },
        );
        configurePanel(context, panel, model);
      }),
    );
  }
  context.subscriptions.push(
    vscode.window.registerCustomEditorProvider(viewType, {
      openCustomDocument: (uri) => ({ uri, dispose() {} }),
      resolveCustomEditor: async (document, panel) => {
        const model = await readGraph(document.uri);
        configurePanel(context, panel, model);
      },
    }),
  );
}

async function readGraph(uri: vscode.Uri): Promise<LoadedModel> {
  const bytes = await vscode.workspace.fs.readFile(uri);
  const fileName = posix.basename(uri.path);
  const native = loadModel(bytes, fileName);
  return { graph: native.graph(), native };
}

function configurePanel(
  context: vscode.ExtensionContext,
  panel: vscode.WebviewPanel,
  initialModel: LoadedModel,
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
    async (message: { type?: string } | null) => {
      if (message?.type === "ready") {
        await sendGraph();
      } else if (message?.type === "open-backend-test") {
        let loadedModel: LoadedModel;
        let name: string;
        try {
          const selection = await vscode.window.showQuickPick(
            listBackendTests().map((test) => ({
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
          const native = loadBackendTest(selection.id);
          loadedModel = { graph: native.graph(), native };
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
            loadedModel = await readGraph(selection[0]);
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
      } else if (message?.type === "metadata") {
        let metadata: GraphMetadata;
        try {
          metadata = model.native.metadata();
        } catch (error) {
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
          refreshed = model.native.graph(true);
        } catch (error) {
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
  panel.webview.html = webviewHtml(resource, panel.webview.cspSource, nonce);
}
