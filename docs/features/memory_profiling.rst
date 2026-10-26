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

The native onnx-light analysis runs on an independent snapshot in a background
worker. Lifetime and potential in-place reuse decisions affect the estimate;
model edits invalidate the displayed result. Reading dimensions and estimating
memory happen only on request, not while browsing a hidden tab.

**This is not measured RAM or GPU usage.** Function/control-flow internals,
allocator overhead and fragmentation are excluded. Scratch estimates depend on
registered CPU estimators: zero can mean unavailable. Unknown ranks, variable-width
inputs, unresolved sizes and unsafe integer sizes report errors instead of zero.
A graph without executable nodes has no per-node peak.
