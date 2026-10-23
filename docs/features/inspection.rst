Operator and tensor inspection
==============================

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
