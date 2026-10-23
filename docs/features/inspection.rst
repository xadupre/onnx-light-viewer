Operator and tensor inspection
==============================

The buttons **Op**, **Data**, **Find** and **Fuse** at the top of the right sidebar
show or hide their panels independently. Visible panels stack vertically,
with **Find** above **Op**, each in a separate bordered box. **Op** contains
operator and tensor details, **Data** contains metadata and lifetime analysis, **Find** searches nodes,
and **Fuse** manages fusion patterns. **Op** and **Find** are initially visible.
The sidebar controls form two rows of three matching buttons:
**Open**, **Save**, **Find**, then **Fuse**, **Op**, **Data**.
**Open** and **Save** toggle their own sidebar panels for file selection and
save options; opening either hides the other. Neither opens a popup file picker.
All sidebar panels have an inverted-color title bar using the compact uppercase
**INSPECTOR** typography.

.. image:: ../_static/inspection.png
   :alt: Reshape inspector showing input and output tensor dimensions and an initializer.
   :width: 240px

Select a node to see its name, domain, imported opset, inputs, outputs and typed
attributes. Scalar and list values appear below the name/type heading without
repeating the attribute name. Schema defaults are marked ``[default]``; long
attribute arrays show a bounded preview.

Graph labels and the inspector show declared tensor types and dimensions,
including symbolic dimensions, scalars and unknown shapes. Dense and sparse
initializers show type, shape and embedded/external storage information,
including external filenames. Dense embedded initializers with fewer than 16
elements also show their available values in the inspector and graph-label
tooltip. Scalars count as one element; 64-bit integers retain their exact precision.
Sparse previews label stored values and indices separately when each component
has fewer than 16 elements, regardless of the logical dense shape. Individual
strings are limited to 128 bytes, with an ellipsis for truncation. External files
and payloads omitted during loading are never read for a preview. The initializer
list scrolls without limiting the number of connected parameters.

Opaque and encoded values
-------------------------

Opaque ports show their domain/name; structured ports show their declared layout,
including model-scoped type references. These are ``TypeProto.Opaque`` and
``StructTypeProto`` in onnx-light, rather than ordinary tensor types.
``EncodedValueProto`` initializers appear in the connected parameter list with
their logical type/shape when present, an **Encoding** layout preview and
embedded/external storage. Affine and structured layouts are supported.
Packed bytes are not displayed as ordinary tensor values or decoded by the viewer.

Open ``examples/quantized-2bit.onnx`` or ``examples/quantized-3bit.onnx`` and select
**Dequantize** to inspect portable Q2_K/Q3_K weight layouts. The 2-bit example
uses an inline declaration; the 3-bit example uses a model-scoped type reference.
These are onnx-light extensions, not standard ONNX or GGUF storage formats.
With onnx-light 0.2.0 Python bindings and their dependencies available, regenerate
both examples from the repository root:

.. code-block:: sh

   python examples/generate_quantized.py

Operator documentation and inference
-------------------------------------

**Doc** opens the selected operator's available onnx-light schema documentation,
with its schema version, formal inputs/outputs, attributes and type constraints.
It is disabled when no documentation is available. **Close** or Escape dismisses it.
Inputs and outputs have separate sections, with spaced parameter headings and
descriptions.
Documentation is rendered as Markdown, including headings, lists, emphasis,
inline and fenced code, tables and HTTP(S) links. Raw HTML remains visible as
text, unsafe links are not activated, and images are shown as text labels
without fetching external resources. The renderer is bundled for offline use.

**Refresh shapes** explicitly runs native shape/type inference. It updates
labels and the inspector without resetting selection, positions or the camera.
Unsupported operators or missing information can produce an error; the previous
display remains available. Opening a model does not automatically infer shapes.
