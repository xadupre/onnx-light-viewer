import * as vscode from "vscode";
import { randomBytes } from "node:crypto";
import { posix } from "node:path";
import { decodeGraph, demoGraph, Graph } from "./graph";
import { webviewHtml } from "./webview";

const viewType = "onnxLightViewer.graph";

export function activate(context: vscode.ExtensionContext): void {
  context.subscriptions.push(
    vscode.commands.registerCommand("onnxLightViewer.open", () => {
      const panel = vscode.window.createWebviewPanel(
        viewType,
        "ONNX Light Viewer",
        vscode.ViewColumn.One,
        { enableScripts: true },
      );
      configurePanel(context, panel, demoGraph);
    }),
    vscode.window.registerCustomEditorProvider(viewType, {
      openCustomDocument: (uri) => ({ uri, dispose() {} }),
      resolveCustomEditor: async (document, panel) => {
        const graph = await readGraph(document.uri);
        configurePanel(context, panel, graph);
      },
    }),
  );
}

async function readGraph(uri: vscode.Uri): Promise<Graph> {
  return decodeGraph(
    await vscode.workspace.fs.readFile(uri),
    posix.basename(uri.path),
  );
}

function configurePanel(
  context: vscode.ExtensionContext,
  panel: vscode.WebviewPanel,
  initialGraph: Graph,
): void {
  const mediaRoot = vscode.Uri.joinPath(context.extensionUri, "media");
  panel.webview.options = {
    enableScripts: true,
    localResourceRoots: [mediaRoot],
  };
  panel.iconPath = vscode.Uri.joinPath(mediaRoot, "logo.svg");
  let graph = initialGraph;
  let isDemo = graph === demoGraph;
  const sendGraph = () =>
    panel.webview.postMessage({ type: "graph", graph, isDemo });
  const messages = panel.webview.onDidReceiveMessage(
    async (message: { type?: string } | null) => {
      if (message?.type === "ready") {
        await sendGraph();
      } else if (message?.type === "open") {
        const selection = await vscode.window.showOpenDialog({
          canSelectMany: false,
          filters: { "ONNX models": ["onnx"] },
          openLabel: "Open ONNX graph",
        });
        if (selection?.[0]) {
          let loadedGraph: Graph;
          try {
            loadedGraph = await readGraph(selection[0]);
          } catch (error) {
            await vscode.window.showErrorMessage(
              `Could not open ONNX model: ${error instanceof Error ? error.message : String(error)}`,
            );
            return;
          }
          graph = loadedGraph;
          isDemo = false;
          panel.title = `ONNX: ${posix.basename(selection[0].path)}`;
          await sendGraph();
        }
      }
    },
  );
  panel.onDidDispose(() => messages.dispose());
  const resource = (name: string) =>
    panel.webview.asWebviewUri(vscode.Uri.joinPath(mediaRoot, name)).toString();
  const nonce = randomBytes(16).toString("hex");
  panel.webview.html = webviewHtml(resource, panel.webview.cspSource, nonce);
}
