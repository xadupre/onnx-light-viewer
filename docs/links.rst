Links
=====

Related repositories in the ONNX ecosystem.

onnx-light projects
-------------------

.. figure:: _static/onnx-light-projects.svg
   :alt: onnx-light decodes models for onnx-light-viewer, onnx-light-cpu provides
         CPU kernels for onnx-light, and onnx-light-ort bridges onnx-light and
         ONNX Runtime in both directions.
   :width: 100%

   Local model inspection, CPU execution, and ONNX Runtime interoperability.

* `onnx-light-viewer <https://github.com/xadupre/onnx-light-viewer>`_: this VS Code
  extension for inspecting ONNX graphs locally without executing models.
* `onnx-light <https://github.com/xadupre/onnx-light>`_: ONNX without protobuf,
  used by this viewer to decode models locally.
* `onnx-light-cpu <https://github.com/xadupre/onnx-light-cpu>`_: optimized CPU
  kernels for onnx-light.
* `onnx-light-ort <https://github.com/xadupre/onnx-light-ort>`_: bidirectional
  interoperability between onnx-light and ONNX Runtime custom operators and
  execution providers.

ONNX and ONNX Runtime
---------------------

.. figure:: _static/onnx-ecosystem.svg
   :alt: ONNX defines the model format consumed by ONNX Runtime;
         ONNX Runtime powers generative AI inference in onnxruntime-genai.
   :width: 100%

   From a shared model format to general inference and generative AI.
   onnx-light reads the same ONNX format; onnx-light-ort connects it to
   ONNX Runtime.

* `onnx <https://github.com/onnx/onnx>`_: the open standard for machine learning
  model interoperability.
* `onnxruntime <https://github.com/microsoft/onnxruntime>`_: a cross-platform
  runtime for machine learning models.
* `onnxruntime-genai <https://github.com/microsoft/onnxruntime-genai>`_:
  generative AI model inference with ONNX Runtime.
