Implemented features
====================

The viewer inspects ONNX graphs locally with onnx-light. Models are not uploaded
or executed, and viewer actions do not save or modify the source file.

Opening models
--------------

* **Open ONNX model** loads a local file. Opening an ``.onnx`` file in VS Code
  also uses the viewer; use **Reopen Editor With...** if another editor handles it.
* **ONNX Light Viewer: Open Graph** opens the bundled Relu demo.
  **ONNX Light Viewer: Open Symbolic Transformer Example** opens a bundled,
  randomly initialized transformer graph with symbolic batch/sequence dimensions.
* The toolbar shows graph counts, refresh progress and errors. Its logo links
  to the GitHub repository.

Zoom, navigation and moving nodes
----------------------------------------

* **+**, **-** or Ctrl+wheel zoom without rearranging nodes. **Fit** recenters
  the view and includes nodes moved outside the original layout.
* Wheel scrolling moves the view vertically; Shift+wheel moves it horizontally.
  Horizontal trackpad scrolling is supported.
* Drag an operator, input or output to move it. Connections and tensor labels
  follow, including while zoomed. Drag a selected node to move the selection.
* Ctrl, Cmd or Shift+click toggles node selection. Drag a rectangle on the
  background to select a group; hold a selection modifier to add to it.
  Click the background to clear selection.
* Ctrl+A (Cmd+A on macOS) selects all nodes when the graph has focus.
  Enter or Space selects a focused node. Escape cancels a drag/selection
  rectangle, or clears selection when no gesture is active.
* **Refresh positions** restores the top-to-bottom layout with new random
  horizontal operator positions and fits the view, preserving inferred shapes.
  Resizing the editor preserves node positions and zoom.

Operator and tensor inspection
------------------------------

Select a node to see its name, domain, imported opset, inputs, outputs and typed
attributes. Schema defaults are marked ``[default]``; long attribute arrays
show a bounded preview.

Graph labels and the inspector show declared tensor types and dimensions,
including symbolic dimensions, scalars and unknown shapes. Dense and sparse
initializers show type, shape and embedded/external storage information,
including external filenames, but not weight values. The initializer list
scrolls without limiting the number of connected parameters.

**Doc** opens the selected operator's available onnx-light schema documentation,
with its schema version, formal inputs/outputs, attributes and type constraints.
It is disabled when no documentation is available. **Close** or Escape dismisses it.

**Refresh shapes** explicitly runs native shape/type inference. It updates
labels and the inspector without resetting selection, positions or the camera.
Unsupported operators or missing information can produce an error; the previous
display remains available. Opening a model does not automatically infer shapes.

Subgraphs
---------

Explore graph attributes of operators such as **If**, **Loop** and **Scan**.
Click a subgraph entry inside the operator's rectangle, or **Open** in the
inspector. Entries support Tab followed by Enter or Space; dragging the
operator's header still moves it. Each graph in a graph-list attribute has its
own entry. Breadcrumbs return to enclosing graphs, including the main graph.

Nested graphs retain enclosing opsets and captured tensor/initializer
information. Captured values appear as input ports; captured initializers stay
in the parameter list. The inspector lists implicit **Captured inputs**.
**Refresh shapes** infers the enclosing model and updates the displayed subgraph,
including intermediate ``value_info`` retained in its native graph.

Metadata and lifetimes
----------------------

**Metadata** switches the inspector to native constant, tag and lifetime
analysis, alongside stored model, graph, node and connected-tensor metadata.
**Refresh metadata** recomputes this analysis for the displayed graph,
even with the metadata inspector closed, without changing the view.

Select an operator to see results releasable after its last reads: their
connections become dashed and tensor names are crossed out. Caller-owned inputs
and weights reaching their last read are listed separately, not marked as
released. Graph outputs are kept. An **Unused** badge identifies unconsumed
results that are not graph outputs.

This is static analysis, not model execution. It does not infer shapes,
evaluate operators or load external weights. Unknown shapes and custom
operators do not prevent metadata inspection.

Backend tests
-------------

Backend tests are ONNX model cases used by onnx-light to check operator and
runtime behavior. They include single-operator cases and multi-node models,
including large Qwen3 cases.

**Backend test** opens a searchable catalog from the bundled native onnx-light
release. Search by test name, model name or category, then press Enter or click
a result. The chosen graph opens in the viewer with normal inspection, moving,
subgraph navigation and shape refresh. Escape cancels the selector.

The viewer opens the test model; it does not run the test, compare reference
outputs, report pass/fail or benchmark it. No Python or network access is needed
to browse these cases, and external weight files are not downloaded or opened.
