Converting models
=================

Choose **Convert** to download a source checkpoint into the application's cache,
convert it offline with an installed exporter and open the ONNX result.
**Download time** and **Conversion time** are shown separately at the top of the
panel, update while running and remain visible after success, failure or cancellation.

.. figure:: ../_static/export-flow.svg
   :alt: The application downloads a pinned source checkpoint into its cache.
         An offline Python exporter reads that snapshot and saves ONNX files.
         The viewer opens the result without executing it.
   :width: 100%

   Conversion is a separate process; inspection never executes the exported graph.

1. Trust the VS Code workspace and select your Python executable in **Settings**.
   Install the desired exporter and its dependencies in that environment yourself.
   The viewer neither installs packages nor changes ``PYTHONPATH``.
2. Open **Convert**, select the exporter and enter a Hugging Face model ID or an
   absolute local checkpoint-directory path. The last used exporter is restored
   across sessions. ``mbext (fast export)`` appears last in the list.
   Choose precision and CPU/CUDA target.
3. The output defaults to ``<download directory>/<model_id>/<exporter>``, using
   the download directory configured in **Settings**. The editable template is
   kept in the field; the command shows the expanded path.
   The download directory is created automatically
   when settings are loaded or saved. Edit the output field to choose another absolute
   **output directory**. If it is not empty, conversion creates a fresh subfolder,
   such as ``modelbuilder-ABC123``, without touching the existing files.
   The displayed command, reproduction script and history use that actual destination.
   Recent model IDs and output directories
   are saved across sessions and suggested in their respective fields.
   Suggestions only fill the field; existing files are never overwritten.
   Review the displayed command, then choose **Convert and open**.
4. The application first downloads the checkpoint, then starts conversion.
   Separate clocks measure these phases, excluding model selection and opening.
   The output window shows live standard output/error and download progress,
   and scrolls automatically to the latest line.
   The subprocess enables progress bars even without a terminal; terminal control
   sequences are removed and carriage-return updates appear as readable lines.
   **Cancel conversion** stops the
   active download or exporter process; closing the viewer also stops it.
   Cached sources and generated files remain available.

After a successful process exit, the viewer opens the exported ``.onnx`` or
``.pipe-onnx`` file. If several models were produced, choose which one to open.
Failures leave the previously displayed model unchanged and appear in the panel
and **Logs**. You can retry immediately, without dismissing the error notification.
Moving to another model while export is running prevents automatic
replacement; the saved files remain available.

Source cache
------------

Sources are kept in ``<Settings download directory>/huggingface``, independently
of conversion output directories. Repeating a conversion or changing exporters
reuses downloaded weights. The application resolves the repository's current
commit before downloading and converts that pinned local snapshot.

The selected Python environment must provide ``huggingface_hub``.
Downloads prefer the standard root safetensors checkpoint, including its
indexed shards, and otherwise use the standard PyTorch checkpoint. Tokenizer
and configuration files are included; alternative weight formats and precision
variants are not downloaded. For repositories with a custom layout, provide
an absolute local checkpoint directory instead.

Local sources skip downloading. Conversion uses local exporter arguments
(``-i`` or Mobius ``--config``) with ``HF_HUB_OFFLINE=1`` and
``TRANSFORMERS_OFFLINE=1``: a missing dependency or file fails explicitly rather
than silently downloading during the conversion phase.

Output directory templates
--------------------------

Use ``base/<model_id>/<exporter>`` to group conversions by model and tool.
Relative paths start in the configured download directory; absolute templates
are also supported. For model ``Qwen/Qwen3-0.6B`` with modelbuilder, this becomes
``<download directory>/base/Qwen/Qwen3-0.6B/modelbuilder``.
``<exporter>`` expands to ``modelbuilder``, ``mobius`` or ``mbext``, not its
display label. For a local checkpoint, ``<model_id>`` is its directory name.
Unsupported placeholders and model IDs containing unsafe path components are
reported as errors.

The command preview uses the expanded path without changing the editable
template. History retains both the template and the actual destination, including
any fresh subfolder created for an existing conversion.

Reproducing a conversion
------------------------

The chosen folder contains ``export_model.py`` alongside the export package.
It replays the same argument vector, Python executable, working directory and
progress settings (``HF_HUB_DISABLE_PROGRESS_BARS=0``, ``TQDM_POSITION=-1``)
and offline conversion settings
without a shell. Run it with Python to reproduce the export; unlike the viewer's
fresh-directory check, subsequent CLI behavior is controlled by the exporter.
The script reads the same cached local snapshot and does not repeat the source download.
The live output retains its most recent 65,536 characters rather than growing
indefinitely. The command display quotes arguments only when shell syntax requires
it; execution always passes the argument vector directly, without a shell.

Exporter entry points
---------------------

mbext is distributed as ``modelbuilder``; Mobius as ``mobius-onnx``.
The upstream modelbuilder module is included with ``onnxruntime-genai``.

.. list-table::
   :header-rows: 1

   * - Selection
     - Python module
   * - mbext
     - ``modelbuilder.builder``
   * - modelbuilder (ONNX Runtime GenAI)
     - ``onnxruntime_genai.models.builder``
   * - Mobius
     - ``mobius build``

For INT4 advanced options, modelbuilder uses ``block_size`` and ``algo_config``;
mbext retains ``int4_block_size`` and ``int4_algo_config``. Modelbuilder requires
the `updated upstream CLI contract
<https://github.com/microsoft/onnxruntime-genai/blob/6b7ff3ca3efa15da219050fdb4ecf8bed67875af/src/python/py/models/README.md#quantization-options>`_.
Older modelbuilder versions accepting only the prefixed names must be upgraded
to use these two settings; the viewer does not detect or fall back to legacy names.
Default commands are unchanged when these settings are left unset.

FP32, FP16, BF16, INT4, INT2 and FP4 are selectable for every exporter and both
CPU/CUDA targets. The viewer does not filter combinations or reset the selected
precision; the exporter reports unsupported values through its normal CLI errors.
Mobius receives ``f32``/``f16`` for FP32/FP16; other precision names are passed
unchanged.
Model support, authentication, runtime compatibility and additional restrictions
depend on the installed exporter. Downloads use the selected Python environment and
credentials; the viewer does not request tokens or enable remote checkpoint code.
Discovery of new model IDs and additional exporters are separate features.
