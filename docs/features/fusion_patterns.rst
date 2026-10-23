Fusion patterns
===============

**Fuse**, at the top of the right sidebar, opens the native onnx-light pattern
matcher on the main graph.
The list keeps patterns whose anchor operator occurs in that graph, plus patterns
without an anchor restriction. Matching and rewriting do not execute the model.
Return to the main graph's breadcrumb to enable this control.
The available native patterns are displayed in the sidebar; select one or more
names with normal clicks, then click **Run Matching** while keeping
the main graph visible. Every click reruns matching from scratch for the selected
patterns and clears the previous candidates and selection.
Click a pattern again to uncheck it. **Select all** checks every pattern;
**Clear selection** unchecks them all. No modifier key is needed.

.. code-block:: text

   Fuse -> Choose patterns -> Run Matching
                                        |
                         +--------------+--------------+
                         |                             |
                  Click candidate             Inspect replacements
                         |                             |
                         +--------------+--------------+
                                        |
                     Check candidates -> Fuse selected
                                        |
                                   Save a copy

Each candidate lists its pattern, affected nodes and replacement operators,
tensor names and attributes. Clicking a candidate selects its
nodes in the main graph for inspection. Check or uncheck candidates to choose
which ones to fuse; inspecting or checking a candidate does not change the model.
Both patterns and candidates use collapsible, scrollable lists with checkboxes.

Check individual candidates or choose **Select all compatible**, then click
**Fuse selected**. Candidates from different patterns can be fused together.
**Clear selection** in the candidate list unchecks all candidates without
changing the model.
Overlapping matches
cannot be applied together; the selection count and next step are shown.
Clicking **Fuse** again or **Close** hides the panel and discards the preview
and selection without applying anything, including while matching is running.
All fusions change only the in-memory model; use **Save** to keep the result.
Graph replacement or navigation invalidates the preview.

For example, two inverse transposes can be replaced with an identity:

.. code-block:: text

   Before: x -> Transpose(perm=[1,0]) -> Transpose(perm=[1,0]) -> y
   After:  x ---------------------> Identity -----------------> y

Save and weights
----------------

Use **Save** to write a different model file. The original model and its source
weight files are protected from overwriting.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Choice
     - Result
   * - Embedded weights
     - A self-contained plain or password-encrypted model.
   * - External weights
     - A plain model and fresh weight files; choose their maximum size.
   * - Reuse unchanged external files
     - A plain model referencing the original external weights; only new or
       modified payloads need saving. Save beside the source model so relative
       paths, offsets and lengths remain valid.

The default chunk size is **5120 MiB (5 GiB)**. A tensor cannot span chunks;
small tensors may stay embedded. Weights originally embedded in the source
cannot be reused as external files and are saved normally. Reused external
files must remain available and unchanged: the viewer checks their content when
saving. To move the result elsewhere, choose embedded weights or a full external
copy instead. Separate encrypted external files are not supported upstream.

Fusion materializes the model's weights in native memory before matching, unlike
ordinary lightweight inspection. Large models therefore require enough memory
for their tensor payloads and the pattern's intermediate data.
