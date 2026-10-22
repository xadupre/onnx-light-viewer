Shape and type information
==========================

Reading tensor information
--------------------------

When a model opens, the viewer displays its declared input, output and
intermediate type/shape information, together with dense and sparse initializer
types and dimensions. Information absent from the model is not guessed.

The main graph shows this information below tensor names on connections and
input/output nodes, and inside each operator for its results, including
intermediate values without consumers. Multiple results share a compact line.
Hover a shortened label for its complete description and result names.
Select an operator or a graph input/output to see the same information in the
side inspector's input/output lists.

For tensor types, labels use the form ``FLOAT [N, 3]``:

* ``FLOAT`` or ``INT64`` is the ONNX element type.
* ``N`` is a symbolic dimension, preserved as declared or inferred.
* ``?`` denotes an unknown element type, dimension or rank.
* ``FLOAT []`` is a scalar, not an unknown-rank tensor.
* Zero dimensions remain ``0``.

Non-tensor values, such as sequences, use onnx-light's native type
representation instead of assuming a tensor shape.

Refreshing shapes and types
---------------------------

Click **Refresh shapes** beside **Open ONNX model** to run onnx-light's
native shape/type inference on the currently loaded model. The button is
temporarily disabled while the calculation is running. The toolbar reports
completion or an error.

The new information updates both the graph labels and the side inspector
without resetting node positions, selection, zoom or scrolling. The bundled
Relu demo supports the same action.

The extension retains a native ``ModelProto`` in memory. Inference updates that
model's ``graph.value_info`` for intermediate results and its output
information, so subsequent reads use the updated model rather than a separate
display-only cache. Refreshing does not reread, save or modify the original
ONNX file.

Refreshing positions
--------------------

**Refresh positions** restores the portrait layout: operators return to their
node-index rows with new random horizontal positions, and graph inputs/outputs
return to their original positions. Connections follow the nodes, and the
camera is fitted to the graph. Selection, the inspector and inferred
type/shape information are preserved. This action does not run shape inference.

Native implementation and limitations
-------------------------------------

The binding uses ``onnx_shapes::RegisterShapeFunctions()`` followed by
``core::shapes::InferShapesModel()`` from the already published onnx-light
C++ release. No JavaScript shape rules, upstream ONNX package, Python process
or external service is involved.

Inline ``raw_data`` is borrowed from the retained bytes during inference
without copying the weight payloads. This makes small constant tensors, such
as the shape input of ``Reshape``, available to onnx-light. External weight
files are not loaded. Initializer inspection exports metadata only: shape,
type and embedded/external storage, with the external file name when declared.
No initializer values are displayed, even for tensors with typed inline data.

Inference follows the algorithms and supported operators of that release.
Unsupported operators or missing required information can produce a native
inference error. In that case, the viewer reports the error and keeps the
previous information rather than presenting partial results as a success.
Symbolic dimensions may remain symbolic after a successful refresh.
The :doc:`bundled transformer example <symbolic_example>` demonstrates symbolic
batch and sequence dimensions through attention and feed-forward layers.

Repository link
---------------

The viewer logo in the toolbar is a link to the
`GitHub repository <https://github.com/xadupre/onnx-light-viewer>`_.
It can also be focused with Tab and opened with Enter.
