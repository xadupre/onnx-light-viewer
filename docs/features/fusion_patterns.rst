Fusion patterns
===============

**Fuse**, next to **Save**, runs the native onnx-light pattern matcher on the
main graph.
The list keeps patterns whose anchor operator occurs in that graph, plus patterns
without an anchor restriction. Matching and rewriting do not execute the model.
Return to the main graph's breadcrumb to enable this control.
The available native patterns are displayed in a scrollable list; select a name
to preview it.

.. code-block:: text

   Fuse -> Choose pattern -> Automatic preview
                                        |
                         +--------------+--------------+
                         |                             |
                    Show nodes                Inspect replacements
                         |                             |
                         +--------------+--------------+
                                        |
                      Approve or reject -> Fuse approved
                                        |
                                   Save a copy

Each match lists the affected nodes and the replacement operators, tensor names
and attributes. Choosing a pattern automatically previews its matches and
highlights the first match in the main graph; nothing is approved automatically.
**Show nodes** highlights another match for review.
Check **Approve fusion** for a match or use **Reject** to exclude it.
Choose **Approve all compatible** to approve a disjoint subset of matches.
Overlapping matches cannot be applied together; the approval count is shown.
**Fuse approved** applies only approved matches to the in-memory model.
**Close** discards the preview and approvals without applying anything.
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
