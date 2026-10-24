Links
=====

Related projects and export tools in the ONNX ecosystem.

onnx-light projects
-------------------

.. figure:: _static/onnx-light-projects.svg
   :alt: onnx-light decodes models for onnx-light-viewer, onnx-light-cpu provides
         CPU kernels and onnx-light-kernel-images provides ImageDecoder kernels
         for onnx-light, and onnx-light-ort bridges onnx-light and ONNX Runtime
         in both directions.
   :width: 100%

   Local model inspection, CPU execution, image decoding, and ONNX Runtime
   interoperability.

* `onnx-light-viewer <https://github.com/xadupre/onnx-light-viewer>`_: this VS Code
  extension for inspecting ONNX graphs locally without executing models.
* `onnx-light <https://github.com/xadupre/onnx-light>`_: ONNX without protobuf,
  used by this viewer to decode models locally.
* `onnx-light-cpu <https://github.com/xadupre/onnx-light-cpu>`_: optimized CPU
  kernels for onnx-light.
* `onnx-light-kernel-images <https://github.com/xadupre/onnx-light-kernel-images>`_:
  ImageDecoder kernel extension for onnx-light.
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

ONNX exporters
---------------

Create an ONNX model with an external exporter, then open its ``.onnx`` file
in the viewer. Supported models, operators and target runtimes vary by tool.

.. figure:: _static/onnx-exporters.svg
   :alt: External exporters write an ONNX model and optional external weight
         files. ONNX Light Viewer opens the resulting model for inspection.
   :width: 100%

   Export happens outside the viewer; opening a model does not execute it.

* `PyTorch ONNX <https://docs.pytorch.org/docs/stable/onnx.html>`_:
  ``torch.onnx.export`` exports supported PyTorch models to ONNX.
* `Optimum ONNX <https://github.com/huggingface/optimum-onnx>`_:
  exports supported Hugging Face models for ONNX Runtime.
* `ONNX Runtime GenAI model builder
  <https://github.com/microsoft/onnxruntime-genai/tree/main/src/python/py/models>`_:
  builds ONNX graphs for supported generative AI architectures.
* `mbext <https://github.com/xadupre/mbext>`_:
  extends the GenAI model builder with additional architectures and fast,
  offline export-validation tests.
* `Mobius <https://github.com/onnxruntime/mobius>`_:
  builds ONNX graphs with ONNX Script and pretrained weights rather than
  tracing a PyTorch model.
* `skl2onnx <https://github.com/onnx/sklearn-onnx>`_:
  converts supported scikit-learn estimators and pipelines to ONNX.
* `tf2onnx <https://github.com/onnx/tensorflow-onnx>`_:
  converts TensorFlow and Keras models to ONNX.

These links do not imply built-in export commands. Viewer-side exporter
integration is planned in `issue #131
<https://github.com/xadupre/onnx-light-viewer/issues/131>`_.
