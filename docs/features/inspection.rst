Operator and tensor inspection
==============================

.. image:: ../_static/inspection.png
   :alt: Reshape inspector showing input and output tensor dimensions and an initializer.
   :width: 240px

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
