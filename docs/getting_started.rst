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
outputs. Use the zoom buttons, the mouse wheel, or **Fit** to adjust the view.

The graph flows from top to bottom: inputs above the operators, outputs below
them, and connections attached to the top and bottom of each node. Operators
have random horizontal positions on each model load; their vertical rows follow
the node indices.

The workspace fills the editor and adapts when it is resized. Resizing and
zooming preserve node positions.

Moving and selecting operators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Drag an operator to move it. Connections and tensor labels follow its position,
including while zoomed in or out. These changes affect only the current view;
they do not modify or save the ONNX model.

Hold Ctrl, Cmd or Shift while clicking operators to toggle their selection.
Alternatively, drag a rectangle on the graph background to select operators;
hold a selection modifier to add to the existing selection. Drag any selected
operator to move the entire selection, preserving relative positions.

Click the background to clear the selection. Ctrl+A (Cmd+A on macOS) selects
all operators when the graph has focus. Enter or Space selects a focused
operator, with the same modifiers for multi-selection. Escape cancels an active
drag or selection rectangle; otherwise it clears the selection.
**Fit** includes operators moved outside the original layout.

Opening an ``.onnx`` file directly in VS Code also uses the viewer. Choose
**Reopen Editor With...** if another installed ONNX editor handles the file.

Current limitations
-------------------

This prototype reads the entire model file into memory and draws every
top-level operator. The native parser skips inline ``raw_data`` payloads.
Tensor data, attributes, inferred shapes, nested subgraphs, editing and external
weight files are not displayed or loaded. Models with external weights can
still display their graph. The dark palette is currently fixed.
