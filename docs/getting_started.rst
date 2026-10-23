Getting started
===============

Requirements
------------

* Node.js 22+, npm and VS Code 1.96+ with the desktop ``code`` command on ``PATH``.
* CMake 3.24+, a C++20 compiler and OpenSSL development packages.
* Internet access for the first build's npm dependencies and published
  onnx-light library. Python and ONNX Runtime are not required.

Launch
------

From a Bash terminal in the repository:

.. code-block:: bash

   ./start.sh

This builds the extension and opens a VS Code development window with a small
Relu model. Choose **Open**, then **Local file** to open your own
``.onnx`` file or **Hugging Face** to enter a repository ID, repository URL or
ONNX file URL, then download and cache the selected file.
After changing the application, close that window and run ``./start.sh`` again.

In WSL, use the desktop VS Code ``bin`` directory on ``PATH``, not the remote
terminal CLI. The native binding must match the extension host platform.

See :doc:`features` for the viewer controls and :doc:`development` for tests
and manual builds.

First inspection, without Python
--------------------------------

In the development window opened by ``./start.sh``:

1. Select the **Relu** operator. The inspector shows its ``input``/``output`` tensors,
   float32 types and ``[1, 3]`` shapes. Choose **Doc** to read its schema.
2. Type ``relu`` into **Find node or operator** above the inspector. Select a
   suggestion, or press Enter to center a match at readable size.
3. Choose **Metadata** to inspect static analysis, then **Refresh metadata**.
   **Refresh shapes** recomputes types/shapes separately. Update counts can be
   zero when the model already stores the same information.
4. Use **Logs** for the operation history, **Status** for native-library/errors
   and **Controls** for shortcuts. These actions do not execute the graph.

To explore more than one operator, choose **Open ONNX model**, **Local file**,
then ``test/models/functions.onnx`` in this checkout. Open a **Function** entry
and use breadcrumbs to return. ``test/models/subgraphs.onnx`` demonstrates
nested graph navigation instead. Neither example needs Python or a download.

You can also open a bundled example directly from a Bash terminal after the
initial build. Run this from the repository root, using the desktop ``code``
command:

.. code-block:: bash

   env -u VSCODE_IPC_HOOK_CLI code \
       --new-window --disable-extensions --extensionDevelopmentPath="$PWD" \
       test/models/functions.onnx

Replace the last argument with your own ``.onnx`` path if desired. This explicitly
loads the development extension; running ``code model.onnx`` in an ordinary
window does not install the viewer.

Create a small model to inspect
-------------------------------

This optional example uses the Python ``onnx`` package to create a model; the
viewer itself still needs neither Python nor ONNX Runtime. If you do not already
have ``onnx`` available, use a separate environment:

.. code-block:: bash

   python3 -m venv .venv-model-example
   . .venv-model-example/bin/activate
   python -m pip install onnx

Save the following as ``make_example.py`` in your working directory, then run
``python make_example.py``. It writes a new ``example.onnx`` file.

.. code-block:: python

   import onnx
   from onnx import TensorProto, helper

   graph = helper.make_graph(
       [
           helper.make_node("Add", ["x", "bias"], ["shifted"], name="add_bias"),
           helper.make_node("Relu", ["shifted"], ["y"], name="activate"),
       ],
       "First inspection",
       [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 3])],
       [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", 3])],
       initializer=[
           helper.make_tensor("bias", TensorProto.FLOAT, [3], [0.5, 1.0, 1.5])
       ],
   )
   model = helper.make_model(
       graph,
       ir_version=9,
       opset_imports=[helper.make_opsetid("", 13)],
   )
   onnx.checker.check_model(model)
   onnx.save_model(model, "example.onnx")

Open ``example.onnx`` with **Open ONNX model**, **Local file** in the development
window. Then follow these checks:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Action
     - What to look for
   * - Search ``add_bias`` and select the match.
     - The Add node has inputs ``x`` and ``bias`` and output ``shifted``.
   * - Inspect **Initializers**.
     - ``bias`` is a float32 tensor of shape ``[3]`` with values
       ``0.5, 1, 1.5``. Small embedded values can be previewed.
   * - Choose **Refresh shapes**.
     - The initially unspecified intermediate ``shifted`` gains shape
       ``[batch, 3]``. Input/output shapes retain the symbolic ``batch`` dimension.
   * - Choose **Metadata**.
     - Inspect constant/tag/lifetime information and distinguish last use of the
       weight from outputs that can be released.
   * - Drag the Add node, then choose **Refresh positions**.
     - Only the layout changes; the ONNX file is not rewritten.

Inference updates are kept only in the open viewer. This example saves a file
with Python before opening it; it is not a viewer export or execution command.
