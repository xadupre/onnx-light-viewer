<p align="center">
  <img src="media/logo.svg" width="96" alt="ONNX Light Viewer logo">
</p>

# ONNX Light Viewer

A lightweight TypeScript graph explorer for VS Code.

This first prototype opens an interactive graph with a single **Relu** operator,
its input and its output. Select an operator to inspect its connections, use the
zoom controls, or choose **Open ONNX model** to load a real `.onnx` file.

## Run the extension

Requires Node.js 22 or later, npm, and VS Code 1.96 or later, with the `code`
command available in your terminal.

From a Bash terminal, run:

```sh
./start.sh
```

The script installs dependencies on the first run, recompiles the extension,
and opens a new VS Code development window with a real, single-node demo ONNX
file already open in the viewer. Other extensions are disabled in that test
window to avoid conflicts with other ONNX viewers. No F5 or Command Palette
step is needed.
After making changes, close that development window and run the script again.
The script also works when called from another directory.

### Manual launch with F5

```sh
npm ci
npm run compile
```

Open this folder in VS Code and press **F5** to launch the Extension Development
Host. In that window, run **ONNX Light Viewer: Open Graph** from the Command
Palette to see the demo. The **Open ONNX model** button opens a file picker.
Opening a `.onnx` file directly in VS Code also uses the viewer; use **Reopen
Editor With…** if you already have another ONNX editor installed.

Models are decoded locally in the extension host. No model data is uploaded, and
no ONNX runtime or Python installation is required.

## Prototype scope

- Original SVG logo, dark graph canvas, and keyboard-accessible node inspector.
- ONNX protobuf loading, operator connections, graph inputs/outputs, and
  initializer count.
- Simple left-to-right layout for topologically ordered ONNX graphs.
- Zoom in/out and fit-to-graph controls.

This is a starting point, not yet a large-model renderer: the entire file is
decoded in memory and every top-level operator is drawn. Tensor data, attributes,
inferred shapes, nested subgraphs, editing, and external weight files are not
displayed or loaded. Models with external weights can still display their graph.
The fixed dark palette is intentional for this prototype.

## Development

```sh
npm test
npm run format:check
npm run watch
```

The tests load [`test/fixtures/relu.onnx`](test/fixtures/relu.onnx), a small,
valid ONNX model containing one Relu operator with float32 input/output tensors
of shape `[1, 3]` (IR 9, opset 13). It can also be opened manually in the viewer.
To regenerate it, run `python3 test/fixtures/generate_relu.py` with the Python
`onnx` package installed; Python is not required to run the JavaScript tests.

The logo is in [`media/logo.svg`](media/logo.svg), the VS Code integration in
`src/extension.ts`, the model adapter in `src/graph.ts`, and the webview in
`media/viewer.js` and `media/viewer.css`.
