Symbolic transformer example
============================

In the Command Palette, run **ONNX Light Viewer: Open Symbolic Transformer
Example**. The model is bundled as ``media/tiny-llm.onnx`` (about 25 KiB), so
opening it requires no download, Python runtime or model-export tools.

This is a tiny decoder-only language-model architecture: one transformer
block, two attention heads, hidden size 16, a ReLU feed-forward layer and a
32-token vocabulary. It includes token/position embeddings, causal masked
self-attention, residual connections, RMS normalization and an output
projection to vocabulary logits. Its reproducible weights are randomly
initialized, not pretrained; it is a graph-inspection example, not a useful
text generator.

Inputs and outputs
------------------

The model uses the symbolic dimensions ``batch_size`` and ``sequence_length``:

* ``input_ids``: ``INT64 [batch_size, sequence_length]``.
* ``position_ids``: ``INT64 [batch_size, sequence_length]``.
* ``attention_mask``:
  ``FLOAT [batch_size, 1, sequence_length, sequence_length]``.
* ``logits``: ``FLOAT [batch_size, sequence_length, 32]``.

For execution, token IDs must be in ``[0, 31]`` and position IDs in ``[0, 127]``.
The mask is an additive causal mask: zero at allowed positions and negative
infinity at masked positions. The viewer does not execute the language model.

Inspecting symbolic results
---------------------------

Some intermediate shapes are declared so that symbolic results appear as soon
as the model opens. Click **Refresh shapes** to infer the remaining information
with onnx-light, then zoom in with Ctrl+wheel and select operators:

* ``query.heads`` has shape ``[batch_size, 2, sequence_length, 8]``.
* ``attention_scores`` has shape
  ``[batch_size, 2, sequence_length, sequence_length]``.
* ``context_merged`` has shape ``[batch_size, sequence_length, 16]``.
* ``feed_forward_expanded`` has shape
  ``[batch_size, sequence_length, 32]``.

The graph and side inspector preserve these symbolic names. Refreshing updates
the model in memory only, not the bundled file. Initializers show their shapes,
types and storage information, without displaying weight values.

Regenerating the model
----------------------

With the Python ``onnx-light`` package available, run from the repository root:

.. code-block:: bash

   python3 examples/generate_tiny_llm.py

The generator uses onnx-light's model helpers and checker, plus Python's
standard library. It does not require upstream ONNX, PyTorch or Transformers.
