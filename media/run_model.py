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


def read_profile(session):
    trace = json.loads(Path(session.end_profiling()).read_text(encoding="utf-8"))
    events = [
        event
        for event in trace
        if event.get("cat") == "Node"
        and event.get("name", "").endswith("_kernel_time")
    ]
    events.sort(
        key=lambda event: (
            event.get("pid", 0),
            event.get("tid", 0),
            event.get("ts", 0),
            -event["dur"],
        )
    )
    nodes = {}
    scopes = []
    thread = None
    for event in events:
        current_thread = (event.get("pid"), event.get("tid"))
        if current_thread != thread:
            scopes.clear()
            thread = current_thread
        start = event.get("ts", 0)
        end = start + event["dur"]
        while scopes and (start >= scopes[-1][0] or end > scopes[-1][0]):
            scopes.pop()
        args = event.get("args", {})
        name = event["name"].removesuffix("_kernel_time")
        op = args.get("op_name", "")
        provider = args.get("provider", "")
        index = args.get("node_index")
        key = (
            scopes[-1][1] if scopes else (),
            name,
            op,
            provider,
            index,
        )
        node = nodes.get(key)
        if node is None:
            label = f"{name} [node {index}]" if index is not None else name
            if scopes:
                label = f"{scopes[-1][2]} / {label}"
            node = dict(name=label, op=op, provider=provider, calls=0, durationUs=0)
            nodes[key] = node
        node["calls"] += 1
        node["durationUs"] += event["dur"]
        # Node indices are graph-local; enclosing control-flow kernels scope subgraphs.
        if op in {"If", "Loop", "Scan"} and "ts" in event and event["dur"] > 0:
            scopes.append((end, key, node["name"]))
    return {
        "trace": trace,
        "nodes": sorted(nodes.values(), key=lambda node: node["durationUs"], reverse=True),
    }


def main(directory, cache=None, key=None):
    directory = Path(directory)
    settings = json.loads((directory / "settings.json").read_text(encoding="utf-8"))
    enable_profiling = settings.get("enableProfiling", False)
    reused = not enable_profiling and cache is not None and cache.get("key") == key
    if reused:
        session, load_ms = cache["session"], cache["load_ms"]
    else:
        options = ort.SessionOptions()
        options.log_severity_level = settings["logSeverity"]
        options.log_verbosity_level = settings["logVerbosity"]
        options.enable_profiling = enable_profiling
        options.profile_file_prefix = str(directory / "profile")
        load_start = perf_counter()
        session = ort.InferenceSession(
            str(directory / "model.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        load_ms = (perf_counter() - load_start) * 1000
    retained = settings.get("keepLoaded", False) and not enable_profiling
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
    profiling = read_profile(session) if enable_profiling else None
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
                **({"profiling": profiling} if profiling is not None else {}),
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
