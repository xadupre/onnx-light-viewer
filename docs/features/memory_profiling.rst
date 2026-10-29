Memory profiling without execution
==================================

Open **Memory**, beside **Statistics**, to estimate CPU tensor memory without
running the model or changing its retained shapes.

1. For a pipeline, choose a stage. Ordinary models always use the complete root
   graph, even when the Model tab displays a function or subgraph.
2. Fill the dynamic input dimensions. Defaults are **1**; repeated symbolic
   dimensions share a field, and unnamed dimensions use ``input[axis]`` labels.
3. Choose **Estimate memory**. The curve shows estimated device memory for every
   node in execution order. The table below includes existing allocations, new
   output allocations, scratch estimates and their total. **Previous** / **Next**
   exposes every event, with 100 rows per page.

Hover over the curve to inspect memory **before**, **during** or **after** a node.
The readout shows exact bytes, scratch, tensor shapes and SDK allocation buckets:
weights (initializers), inputs, live intermediates and newly allocated outputs,
split by tensor tag. In-place outputs do not allocate another buffer.
Symbolic expressions retain named dimensions; unnamed dimensions use the values
entered above. Symbolic tensor totals exclude scratch and may choose different
in-place reuse than the concrete estimate.

Use **+** / **−** or **Ctrl+wheel** to zoom, then scroll horizontally.
More node graduations and operator types appear as space becomes available.
Crowded labels prioritize **Attention**, then **MatMul**, then
**Normalization**, with other operators sampled every five or ten nodes.
Focus the chart and use **Left** / **Right** to inspect nodes without a mouse.
**Reset zoom**, a new estimate or a stage change returns to the full trace.

.. figure:: ../_static/feature-memory.png
   :alt: Memory chart with zoom controls and numerical and symbolic allocation inspection.

   Inspect SDK allocation categories and expressions at a node's execution stage.

Choose **Local peaks only** to list memory maxima, then click a node name to open
**Model**, select the node and center it at readable zoom. Navigation opens the
estimated root graph or pipeline stage, even if another graph was displayed.
A flat peak is listed once, at its first event; endpoints are compared with their
available neighbor. A constant or single-node trace has one peak.

The native onnx-light analysis runs on an independent snapshot in a background
worker. Lifetime and potential in-place reuse decisions affect the estimate;
model edits invalidate the displayed result. Reading dimensions and estimating
memory happen only on request, not while browsing a hidden tab.

**This is not measured RAM or GPU usage.** Function/control-flow internals,
allocator overhead and fragmentation are excluded. Scratch estimates depend on
registered CPU estimators: zero can mean unavailable. Only dense, fixed-width
tensor inputs are supported. The viewer propagates SDK errors and rejects results
outside JavaScript's exact integer range. A graph without executable nodes has no
per-node peak.

Shape inference and memory arithmetic belong to onnx-light, without viewer-side
operator replacements. Known SDK issues are tracked upstream:
`shape arithmetic <https://github.com/xadupre/onnx-light/issues/5629>`_,
`allocation and scratch arithmetic <https://github.com/xadupre/onnx-light/issues/5630>`_
and `callback lifecycle <https://github.com/xadupre/onnx-light/issues/5631>`_.
The pinned SDK is not protected against those defects by the viewer.
