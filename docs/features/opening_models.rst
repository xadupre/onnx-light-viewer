Opening models
==============

.. image:: ../_static/opening-models.png
   :alt: Model opening control in the viewer toolbar.
   :width: 200px

* **Open** offers **Local file**, **Hugging Face** or **Backend test**.
  Opening an ``.onnx`` file in VS Code
  also uses the viewer; use **Reopen Editor With...** if another editor handles it.
* **ONNX Light Viewer: Open Graph** opens the bundled Relu demo.
  **ONNX Light Viewer: Open Symbolic Transformer Example** opens a bundled,
  randomly initialized transformer graph with symbolic batch/sequence dimensions.
* The toolbar shows the native onnx-light C++ release, graph counts, refresh
  progress and errors. Its logo links to the GitHub repository.

Plain and encrypted ModelProto files
-------------------------------------

The viewer detects onnx-light encrypted model envelopes by their header,
independently of the filename. It reads **ONNXCRY1** (legacy AES-256-CBC),
**ONNXCRY2** (ChaCha20-Poly1305) and **ONNXCRY3** (AES-256-GCM).
Open them through **Local file**, or open an ``.onnx.enc`` file directly in
VS Code. A masked VS Code input asks for the password. Cancelling keeps the
previous model; a wrong password or damaged file produces a visible error.
Passwords never enter the webview, settings or logs. They remain in the
extension host's memory while the model is open so native recovery can reload it.

**Open** and **Save** sit side by side with the same design. **Save** opens a
storage/encryption choice grid:

.. list-table::
   :header-rows: 1

   * - Weight storage
     - Plain
     - Encrypted
   * - Embedded
     - One self-contained ``.onnx`` file
     - One self-contained ``.onnx.enc`` file, including all weights
   * - External
     - An ``.onnx`` file and separate weight files
     - Shown but unavailable pending `onnx-light#5533
       <https://github.com/xadupre/onnx-light/issues/5533>`_

Encrypted output uses **AES-256-GCM / ONNXCRY3** and asks for password confirmation.
The default filename ends in ``-copy.onnx`` or ``-copy.onnx.enc``; choose a
different destination from the source model. Saving always includes the whole
ModelProto, not just the selected graph, with stored metadata, all weights
and any persisted shape inference. It does not change the displayed selection,
layout or camera. A temporary file is written first and renamed only after the
write succeeds; failures and cancellations appear in Logs.

Saving resolves existing external weights relative to the local source model.
Embedded output incorporates them into the saved proto, so encrypted output
protects them too. External output copies and repartitions the weights, without
modifying the originals. Missing weight files fail explicitly before writing.
External weights from virtual sources require opening a local copy with its
weight files available; Hugging Face weight files are not downloaded automatically.

For external output, choose the **maximum weight file size in MiB**
(1 MiB = 1,048,576 bytes). The default is 1024 MiB. This is a file-size limit,
not an I/O buffer size. One tensor cannot span files, so the limit must fit the
largest tensor; an insufficient limit is reported as an error.
Small tensors, strings and sparse components may remain inline according to
the native serializer.
Weight files have unique names alongside the proto and are published first;
the proto is replaced last, after every weight file succeeds. Failure removes
new weight files and leaves an existing destination intact. Existing weight
files are never replaced or deleted.

Plain local files still load with bounded tensor payloads. Saving rereads any
omitted inline data from their source, so the source must remain available and
unchanged; detected changes require reopening it. Encrypted models are decrypted
fully in native memory, and serialization also needs a complete in-memory
payload. Local-file decryption and serialization run on native workers; virtual
files use VS Code's filesystem and the synchronous byte-buffer loader.
Decrypted data is written to disk only when explicitly saving
a plain copy; downloaded encrypted files remain encrypted in the cache.

Hugging Face models
-------------------

Choose **Open**, then **Hugging Face**, and enter a public repository
ID such as ``onnx-community/Kokoro-82M-v1.0-ONNX``, its URL
(``https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX``), or the URL of an
``.onnx`` file. For a repository, the extension lists its ONNX files and lets you
choose a variant; a single ONNX file is selected automatically.

For direct file URLs, both the file page (``/blob/``) and direct download
(``/resolve/``) links are accepted, including nested paths and pinned revisions.
For example:

.. code-block:: text

   https://huggingface.co/hf-internal-testing/tiny-random-bert/blob/main/onnx/model.onnx

The extension downloads the file with a cancellable progress notification, loads
it through the native viewer and retains it in its persistent Hugging Face
cache. Later openings of the same normalized URL use the cached file without
network access when opened by file URL, including after restarting VS Code.
Opening a repository still needs network access to list its files.
Logs distinguish downloads
from cache hits and include the cached file's location. Native inspection,
subgraph navigation and refresh controls work as for a local model.

Only the requested ONNX file is downloaded: external weight files, Python code
and other repository contents are not fetched. Models are neither executed nor
uploaded. Private or gated repositories and authentication tokens are not
supported. Download, parsing and filesystem errors appear in Logs and VS Code
notifications; cancellation or failure keeps the previous model visible.

Partial or unparseable downloads never become cache entries. Different file
paths or revisions have separate entries, while equivalent blob/resolve links
share one. Cached branch URLs are not automatically checked for remote updates;
use a commit-pinned URL for reproducibility, or remove that entry at the location
shown in Logs before downloading an updated branch file.
