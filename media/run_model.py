"""One-shot ONNX Runtime execution of a viewer-owned model snapshot."""

import json
from pathlib import Path
import sys
from time import perf_counter

if sys.platform == "win32":
    # Match ONNX Runtime's wide-character stderr, including Python tracebacks.
    sys.stderr.reconfigure(encoding="utf-16-le", errors="backslashreplace")

import onnx
from onnx import numpy_helper
import onnxruntime as ort


def main(directory):
    directory = Path(directory)
    settings = json.loads((directory / "settings.json").read_text(encoding="utf-8"))
    options = ort.SessionOptions()
    options.log_severity_level = settings["logSeverity"]
    options.log_verbosity_level = settings["logVerbosity"]
    session = ort.InferenceSession(
        str(directory / "model.onnx"),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    inputs = {}
    for index in range(settings["inputs"]):
        tensor = onnx.TensorProto()
        tensor.ParseFromString((directory / f"input-{index}.pb").read_bytes())
        inputs[tensor.name] = numpy_helper.to_array(tensor)
    start = perf_counter()
    values = session.run(None, inputs)
    time_ms = (perf_counter() - start) * 1000
    outputs = []
    previews = []
    for name, value in inputs.items():
        preview = (
            str(value.reshape(-1).tolist())
            if value.size <= 32
            else "Omitted (preview limit: 32 elements)."
        )
        previews.append({
            "name": name,
            "text": f"{value.dtype}{list(value.shape)} {name}\nValues: {preview}",
        })
    for index, (metadata, value) in enumerate(zip(session.get_outputs(), values)):
        if hasattr(value, "dtype"):
            preview = (
                str(value.reshape(-1).tolist())
                if value.size <= 32
                else "Omitted (preview limit: 32 elements)."
            )
            text = f"{value.dtype}{list(value.shape)} {metadata.name}\nValues: {preview}"
            (directory / f"output-{index}.pb").write_bytes(
                numpy_helper.from_array(value, metadata.name).SerializeToString()
            )
        else:
            text = "Non-tensor output (no tensor preview)."
        outputs.append({"name": metadata.name, "text": text, "hasData": hasattr(value, "dtype")})
    (directory / "result.json").write_text(
        json.dumps({"timeMs": time_ms, "inputs": previews, "outputs": outputs}), encoding="utf-8"
    )


if __name__ == "__main__":
    main(sys.argv[1])
