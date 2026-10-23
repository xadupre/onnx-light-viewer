Opening models
==============

.. image:: ../_static/opening-models.png
   :alt: Open ONNX model button in the viewer toolbar.
   :width: 200px

* **Open ONNX model** offers **Local file**, **Hugging Face** or **Backend test**.
  Opening an ``.onnx`` file in VS Code
  also uses the viewer; use **Reopen Editor With...** if another editor handles it.
* **ONNX Light Viewer: Open Graph** opens the bundled Relu demo.
  **ONNX Light Viewer: Open Symbolic Transformer Example** opens a bundled,
  randomly initialized transformer graph with symbolic batch/sequence dimensions.
* The toolbar shows the native onnx-light C++ release, graph counts, refresh
  progress and errors. Its logo links to the GitHub repository.

Hugging Face models
-------------------

Choose **Open ONNX model**, then **Hugging Face**, and enter a public repository
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
