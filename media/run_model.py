"""ONNX Runtime execution with optional worker-owned session retention."""

import json
import sys
import traceback
from pathlib import Path
from time import perf_counter

sys.stdin.reconfigure(encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8")

if sys.platform == "win32":
    # Match ONNX Runtime's wide-character stderr, including Python tracebacks.
    sys.stderr.reconfigure(encoding="utf-16-le", errors="backslashreplace")

import onnx
import onnxruntime as ort
from onnx import numpy_helper


def preview_values(value):
    preview = []
    for item in value.reshape(-1).tolist():
        if isinstance(item, str) and len(item) > 128:
            item = item[:128] + "…"
        elif isinstance(item, bytes) and len(item) > 128:
            item = item[:128] + b"..."
        preview.append(item)
    return str(preview)


def main(directory, cache=None, key=None):
    directory = Path(directory)
    settings = json.loads((directory / "settings.json").read_text(encoding="utf-8"))
    reused = cache is not None and cache.get("key") == key
    if reused:
        session, load_ms = cache["session"], cache["load_ms"]
    else:
        options = ort.SessionOptions()
        options.log_severity_level = settings["logSeverity"]
        options.log_verbosity_level = settings["logVerbosity"]
        load_start = perf_counter()
        session = ort.InferenceSession(
            str(directory / "model.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        load_ms = (perf_counter() - load_start) * 1000
    retained = settings.get("keepLoaded", False)
    if cache is not None:
        cache.clear()
        if retained:
            cache.update(key=key, session=session, load_ms=load_ms)
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
            preview_values(value)
            if value.size <= 32
            else "Omitted (preview limit: 32 elements)."
        )
        previews.append(
            {
                "name": name,
                "text": f"{value.dtype}{list(value.shape)} {name}\nValues: {preview}",
            }
        )
    for index, (metadata, value) in enumerate(zip(session.get_outputs(), values)):
        if hasattr(value, "dtype"):
            preview = (
                preview_values(value)
                if value.size <= 32
                else "Omitted (preview limit: 32 elements)."
            )
            text = (
                f"{value.dtype}{list(value.shape)} {metadata.name}\nValues: {preview}"
            )
            (directory / f"output-{index}.pb").write_bytes(
                numpy_helper.from_array(value, metadata.name).SerializeToString()
            )
        else:
            text = "Non-tensor output (no tensor preview)."
        outputs.append(
            {"name": metadata.name, "text": text, "hasData": hasattr(value, "dtype")}
        )
    (directory / "result.json").write_text(
        json.dumps(
            {
                "timeMs": time_ms,
                "inputs": previews,
                "outputs": outputs,
                "lifecycle": {
                    "loadMs": load_ms,
                    "reused": reused,
                    "retained": retained,
                },
            }
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    if sys.argv[1] == "--worker":
        cache = {}
        for line in sys.stdin:
            try:
                request = json.loads(line)
                main(request["input"], cache, request["key"])
                print("\x1e" + json.dumps({"type": "done"}), flush=True)
            except Exception as error:  # noqa: BLE001 - report worker request failures to the host
                cache.clear()
                traceback.print_exc()
                print(
                    "\x1e" + json.dumps({"type": "error", "error": str(error)}),
                    flush=True,
                )
    else:
        main(sys.argv[1])
