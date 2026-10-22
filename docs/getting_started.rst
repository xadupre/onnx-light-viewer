Getting started
===============

Requirements
------------

* Node.js 22 or later and npm.
* VS Code 1.96 or later, with the desktop ``code`` command on ``PATH``.
* CMake 3.24 or later and a C++20 compiler.
* OpenMP and OpenSSL development packages, required by the published onnx-light
  CMake package.

An internet connection is needed to download npm dependencies and the
:doc:`published native library <native_loading>` on the first build.
Python and ONNX Runtime are not required to build or run the extension.

Launch with start.sh
--------------------

From a Bash terminal in the repository, run:

.. code-block:: bash

   ./start.sh

The script installs npm dependencies when needed, builds the native binding and
TypeScript extension, and opens a new VS Code development window. The small
``test/models/relu.onnx`` model is already displayed; no F5 or Command Palette
step is needed. To call the launcher from another directory, use its full path.

The development workspace associates ``.onnx`` files with ONNX Light Viewer,
overriding user-level associations with other viewers such as Netron.
Other extensions are disabled in this development window.

After changing the application, close the development window and run
``./start.sh`` again. Dependencies and the downloaded onnx-light release are
reused.

WSL
---

Inside a VS Code WSL terminal, the script selects the desktop ``code`` launcher.
The remote terminal CLI ignores extension development flags and cannot be used
to launch this development window.

If the launcher reports that only the remote CLI is available, add the desktop
VS Code ``bin`` directory to ``PATH``, or run from a local terminal with the
required build tools available. The native binding must be built for the
extension host platform: Linux when using Remote WSL.

Using the viewer
----------------

Choose **Open ONNX model** to select a model. Select an operator with the mouse,
or focus it and press Enter or Space, to inspect its name, domain, inputs and
outputs. Use the zoom buttons or hold Ctrl while scrolling the mouse wheel
to zoom in and out. Scrolling without Ctrl moves the view vertically;
Shift+wheel moves it horizontally. Horizontal trackpad scrolling is also
supported. **Fit** recenters the graph and resets zoom and scrolling.

**Backend test** opens a VS Code selector with a search bar for onnx-light's
native C++ backend tests. Type part of a test name, model name or category,
then choose a result with the mouse or Enter. Both single-node and multi-node
model cases are available. Escape cancels without changing the current graph.
The selected case opens in the same viewer and supports normal inspection,
position changes and **Refresh shapes**. No Python or network access is needed.
The catalog is from the pinned onnx-light release; large ``_big_`` cases and
benchmark mode are excluded by the library's default collection settings.

The inspector also shows the imported opset, typed operator attributes and
all connected dense or sparse initializers, without a limit on their number.
Scroll the inspector to reach the entire list. Initializers show only their
type, shape and whether their storage is embedded or external. External
initializers identify their weight file, including separate values/indices
files for sparse tensors. Weight values are not sent to the viewer or displayed,
and external files are never opened.

Attribute defaults provided by ``LightOpSchema`` are marked ``[default]`` and
never replace explicit model values. Attribute arrays still show their first
eight elements and total count when shortened, with bounded preview text.

Click **Refresh shapes** to :doc:`refresh shape and type
information <shape_information>` with onnx-light. Tensor descriptions appear
on the graph and in the side inspector. The viewer logo opens the GitHub
repository.

For a bundled language-model example with symbolic batch/sequence dimensions,
see :doc:`symbolic_example`.

Use **Doc** to open the selected operator's available ``LightOpSchema``
documentation, including formal inputs, outputs, attributes and type
constraints. The dialog identifies the model opset and selected schema version;
**Close** or Escape dismisses it. Documentation is displayed as plain text,
preserving Markdown source safely. **Doc** is disabled for graph inputs/outputs
and operators without available schema documentation.

The graph flows from top to bottom: inputs above the operators, outputs below
them, and connections attached to the top and bottom of each node. Operators
have random horizontal positions on each model load; their vertical rows follow
the node indices. **Refresh positions** restores this layout, including graph
inputs/outputs, and fits the camera without discarding inferred shapes.

The workspace fills the editor and adapts when it is resized. Resizing and
zooming preserve node positions.

Moving and selecting nodes
~~~~~~~~~~~~~~~~~~~~~~~~~~

Drag an operator, input or output to move it. Connections and tensor labels
follow its position, including while zoomed in or out. These changes affect only
the current view; they do not modify or save the ONNX model.

Hold Ctrl, Cmd or Shift while clicking nodes to toggle their selection.
Alternatively, drag a rectangle on the graph background to select nodes;
hold a selection modifier to add to the existing selection. Drag any selected
node to move the entire selection, preserving relative positions.

Click the background to clear the selection. Ctrl+A (Cmd+A on macOS) selects
all operators, inputs and outputs when the graph has focus. Enter or Space
selects a focused node, with the same modifiers for multi-selection. Escape
cancels an active drag or selection rectangle; otherwise it clears the selection.
**Fit** includes all nodes moved outside the original layout.

Opening an ``.onnx`` file directly in VS Code also uses the viewer. Choose
**Reopen Editor With...** if another installed ONNX editor handles the file.

Current limitations
-------------------

This prototype retains the model file in memory and draws every top-level
operator. Native shape inference is explicit, not automatic on opening.
Nested subgraph visualization, editing and external weight files are not
supported. Models with external weights can still display their graph.
The dark palette is currently fixed.
