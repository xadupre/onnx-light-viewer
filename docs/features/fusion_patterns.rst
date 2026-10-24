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

Each candidate shows its pattern followed by **Before** and **After** sections.
Before lists the affected nodes by name; After lists their replacement operators.
Both show input/output tensor names and inline attributes, so parallel nodes
sharing an input are not mistaken for a chain. Clicking a candidate selects its
nodes in the main graph for inspection. Check or uncheck candidates to choose
which ones to fuse; inspecting or checking a candidate does not change the model.
Both patterns and candidates use collapsible, scrollable lists with checkboxes.
The list includes overlapping matches. A darker background marks each candidate
sharing a node with any earlier candidate, regardless of which ones are checked.
Hover over a darker entry for an explanation; graph highlighting is unchanged.

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

Undo and rewriting plans
-------------------------

**Unfuse**, beside **Fuse**, offers **Unfuse last** and **Unfuse all**.
Unfuse last restores the model before the last **Fuse selected** operation:
one step reverses all candidates applied together. The choices remain open so
you can repeat Unfuse last to undo earlier operations, including operations on
different pipeline stages. Unfuse all reverses every remaining operation at once.
Below the buttons, each applied match is listed separately, even when several
matches use the same pattern. Operation headings retain the undo order, newest
first, with pipeline stage names. The first operation is the next one that
Unfuse last will undo; each undo removes its entries.
Click a match to select only its resulting nodes in the graph, opening the appropriate
pipeline stage when necessary. After further fusions, selection follows the
surviving replacement nodes rather than stale node positions. Patterns with no
remaining nodes cannot be selected.
Undo returns to the current model's root graph and clears cached metadata.
Opening another file starts a new history; previewing and saving do not add steps.
Native recovery preserves the history.

.. figure:: ../_static/fusion-undo.svg
   :alt: Successive Fuse operations can be undone in reverse order; saving chooses a fused graph or the original graph with a concatenated rewriting plan.
   :width: 800px

   Undo history is session-local. Rewriting plans are optional saved variants.

In **Save**, **Include fusion rewritings** selects the saved representation:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Choice
     - Saved content
   * - Unchecked (default)
     - Current fused graph, without ``RewritingProto`` variants.
   * - Checked
     - Original graph plus one named ``onnx-light-viewer.fusions`` plan,
       concatenating the remaining Fuse operations in application order.
       Existing named variants are retained without overwriting their names.

The checked form follows onnx-light's canonical-model/variant contract:
reopening it displays the original graph; onnx-light can replay the saved plan
to obtain the fused graph. It does not persist the viewer's Undo stack.
For pipelines, each modified stage stores its own original graph and ordered
plan. The choice also applies to encrypted saves and external-weight storage.

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
