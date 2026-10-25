Metadata and lifetimes
======================

.. image:: ../_static/metadata.png
   :alt: Metadata inspector with native constant, tag and tensor lifetime analysis.
   :width: 240px

**Data**, at the top of the right sidebar, toggles native constant, tag and lifetime
analysis, alongside stored model, graph, node and connected-tensor metadata.
Like the other panel toggles, it is highlighted only while its panel is open.
Refresh status and colors belong to **Refresh metadata**, not **Data**.
It does not hide the operator, search or fusion panels.
Metadata sections and stored tensor entries are shown directly, without
collapsible headings. Clicking the background shows metadata for the current
model, graph, function or pipeline.
Pipeline fields and sampling settings appear as individual **Key** / **Value**
rows, not a serialized ``PipelineProto`` block. Only present fields are listed,
including explicit zeros and empty strings; stored metadata uses
``metadata_props.<key>`` rows. Contained models are inspected in their own graphs.
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
