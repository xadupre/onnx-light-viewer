Fusion patterns
===============

**Fuse pattern** runs the native onnx-light pattern matcher on the main graph.
The list keeps patterns whose anchor operator occurs in that graph, plus patterns
without an anchor restriction. Matching and rewriting do not execute the model.
Return to the main graph's breadcrumb to enable this control.

.. code-block:: text

   Fuse pattern -> Choose pattern -> Preview matches
                                        |
                         +--------------+--------------+
                         |                             |
                    Show nodes                Inspect replacements
                         |                             |
                         +--------------+--------------+
                                        |
                          Select matches -> Apply selected
                                        |
                                   Save a copy

Each match lists the affected nodes and the replacement operators, tensor names
and attributes. **Show nodes** selects the affected nodes in the graph.
Check individual matches or choose **Select all compatible**: overlapping
matches cannot be applied together, so this selects a disjoint subset and shows
the selection count. Applying changes only the in-memory model; previewing alone
changes nothing. Graph replacement or navigation invalidates the preview.

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
