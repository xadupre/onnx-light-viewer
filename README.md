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
command available in your terminal. Building the native binding also requires
CMake 3.24 or later, a C++20 compiler, and the OpenMP and OpenSSL development
packages required by the published onnx-light CMake package.

From a Bash terminal, run:

```sh
./start.sh
```

The script installs dependencies on the first run, recompiles the extension,
and opens a new VS Code development window with `test/models/relu.onnx`
already open in the viewer. A dedicated development workspace explicitly
associates `.onnx` files with ONNX Light Viewer, overriding any user-level
association with Netron. Other extensions are disabled in that test
window to avoid conflicts with other ONNX viewers. No F5 or Command Palette
step is needed.
After making changes, close that development window and run the script again.
The script also works when called from another directory.

Inside a VS Code WSL terminal, the script uses the desktop `code` launcher
instead of the remote terminal CLI, which ignores extension development flags.
The desktop VS Code `bin` directory must be on `PATH`; the script reports an
error if only the remote CLI is available.

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

### Native model loading

The extension loads a Node-API 8 C++ binding linked to **onnx-light's
`onnx_proto` library**. It does not use the npm `onnx-proto` package, protobufjs,
the upstream `onnx` package, a Python process, or a custom ONNX decoder.

The first build downloads the **already published
[onnx-light 0.1.35 C++ release](https://github.com/xadupre/onnx-light/releases/tag/0.1.35)**,
selects the archive for the current platform, and verifies its SHA-256 checksum.
Only the small Node-API binding is compiled; **onnx-light is not rebuilt**.
The downloaded release is cached in `build/native/`, so subsequent iterations
reuse it. There is no automatic fallback to compiling onnx-light from source.

The compiled binding is placed in `dist/native/` and must match the extension
host's operating system and architecture (Linux when using Remote WSL). Node-API
allows it to be loaded by both Node.js and VS Code without an Electron-specific
rebuild. The native binding and dependency licenses are included when packaging
the extension; Python is not needed to build or run it.

## Prototype scope

- Original SVG logo, dark graph canvas, and keyboard-accessible node inspector.
- Full-window workspace with a compact toolbar. The canvas and inspector adapt
  to the editor size; resizing preserves node positions and the current zoom.
- Native onnx-light model loading, operator connections, graph inputs/outputs, and
  initializer count.
- A random horizontal position for each operator on every model load; vertical
  rows follow the node indices, with spacing to keep the nodes readable.
- Zoom in/out using the buttons or mouse wheel over the graph, and a
  fit-to-graph control. Zooming preserves the assigned node positions.

This is a starting point, not yet a large-model renderer: the entire file is
read into memory and every top-level operator is drawn. The native parser skips
inline `raw_data` payloads. Tensor data, attributes,
inferred shapes, nested subgraphs, editing, and external weight files are not
displayed or loaded. Models with external weights can still display their graph.
The fixed dark palette is intentional for this prototype.

## Development

```sh
npm test
npm run format:check
clang-format --dry-run --Werror native/addon.cc
npm run watch
```

The **Tests** GitHub Actions workflow builds the native binding and runs tests
and JavaScript/C++ formatting checks on
every pull request and every push to `main`. Results appear in the PR's
**Checks** tab and the repository's **Actions** tab.

The tests load [`test/models/relu.onnx`](test/models/relu.onnx), a small,
valid ONNX model containing one Relu operator with float32 input/output tensors
of shape `[1, 3]` (IR 9, opset 13). It can also be opened manually in the viewer.
The other small models cover custom operators, dense/sparse initializers,
optional ports, an empty operator list, and intentionally invalid metadata.
To regenerate the model files, run `python3 test/models/generate_models.py` with
the Python `onnx-light` package available; Python is not required to run the tests.

The logo is in [`media/logo.svg`](media/logo.svg), the VS Code integration in
`src/extension.ts`, the native decoder in `native/addon.cc` with its typed entry
point in `src/graph.ts`, and the webview in
`media/viewer.js` and `media/viewer.css`.
