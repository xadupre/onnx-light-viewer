#!/usr/bin/env python3
"""Compare five text-decoder export entry points, without quantization.

Usage:
    python scripts/benchmark_qwen35_exporters.py
    python scripts/benchmark_qwen35_exporters.py --output qwen35-cpu.json \
        --profiles cpu-fp32
    python scripts/benchmark_qwen35_exporters.py --model Qwen/Qwen3-0.6B \
        --output qwen3-cpu.json --profiles cpu-fp32

The default model is Qwen/Qwen3.5-4B. Other models must be text decoders supported
by the selected exporters; this option does not add support to those exporters.

Requires Python >= 3.10, huggingface_hub, onnx and the selected exporters:
modelbuilder (mbext), onnxruntime-genai, mobius-onnx and olive-ai. Install versions
supporting the selected model yourself; PYTHONPATH is inherited, never modified.
FP16/CUDA requires the corresponding exporter/runtime dependencies.

The original HF checkpoint is downloaded once, at an immutable revision, and
shared by all runs. Its stored dtype is not changed. Each exporter loads/casts
it to FP32/CPU or FP16/CUDA during the separately timed creation phase.
Download timing includes revision resolution and cache lookups; cache hits are
identified, not presented as cold network transfers. Recipe downloads are timed
separately as well.

All five entry points export text only, including token embeddings and the LM
head. Vision/embedding sidecar pipelines, quantization and inference are excluded.
Olive uses capture-onnx-graph with ModelBuilder; the official recipe also uses
ModelBuilder, followed by TieWordEmbeddings. These are five entry points, NOT five
independent graph builders. The recipe is adapted to the requested precision,
provider and unpruned logits, retaining its non-quantizing surgery. The pinned
Qwen3.5 recipe is reused as a pipeline template with the selected checkpoint;
changing --model does not select a different model-specific recipe.

Creation time includes a fresh Python process, imports, loading/casting weights,
graph construction/optimization and serialization. Each run has private output
and Olive caches; only the source download is shared. ONNX header validation is
timed separately and does not load external weights. Each selected exporter/profile
is measured once, without warmup. Filesystem caches are not flushed.

The output JSON file is updated after every export, including failures, and is
replaced on subsequent invocations. It contains timings, logs and commands/configs.
A final table of timings and output sizes, plus failure log tails, is printed to
standard output; full logs remain in the JSON report.
Exported models and per-run caches use temporary directories, removed after each
measurement; only the shared download cache and the JSON report are retained.
Runs need substantial RAM, CUDA resources where required, and temporary disk space
for one exported model and its intermediate artifacts. TMPDIR controls its location.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import signal
import subprocess
import sys
import time
from collections import Counter
from copy import deepcopy
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen

DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
RECIPE_REVISION = "47b21648b9bf1a4b7beba2d0b231ceeda76edae8"
EXPORTERS = ("mbext", "modelbuilder", "mobius", "olive", "olive-recipes")
PROFILES = {
    "cpu-fp32": ("cpu", "fp32", "CPUExecutionProvider"),
    "cuda-fp16": ("cuda", "fp16", "CUDAExecutionProvider"),
}
MODULES = {
    "mbext": "modelbuilder",
    "modelbuilder": "onnxruntime_genai",
    "mobius": "mobius",
    "olive": "olive",
    "olive-recipes": "olive",
}


def write_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def download_model(model: str, cache: Path, revision: str) -> dict:
    from huggingface_hub import (
        HfApi,
        hf_hub_download,
        snapshot_download,
        try_to_load_from_cache,
    )

    start = time.perf_counter()
    info = HfApi().model_info(model, revision=revision)
    if not info.sha or not re.fullmatch(r"[a-f0-9]{40}", info.sha):
        raise ValueError("The Hub did not return an immutable model revision.")
    files = {entry.rfilename for entry in info.siblings}
    checkpoint = next(
        (
            name
            for name in ("model.safetensors.index.json", "model.safetensors")
            if name in files
        ),
        None,
    )
    if checkpoint is None:
        raise ValueError("The model has no supported root safetensors checkpoint.")

    def cached(name: str) -> bool:
        return isinstance(
            try_to_load_from_cache(model, name, revision=info.sha, cache_dir=cache), str
        )

    checkpoint_cached = cached(checkpoint)
    if checkpoint.endswith(".index.json"):
        index_path = hf_hub_download(
            model, checkpoint, revision=info.sha, cache_dir=cache
        )
        weights = set(
            json.loads(Path(index_path).read_text(encoding="utf-8"))[
                "weight_map"
            ].values()
        )
        if not weights or not weights <= files:
            raise ValueError("The safetensors index references missing weights.")
    else:
        weights = {checkpoint}
    metadata = {
        name
        for name in files
        if name.endswith(
            (".json", ".txt", ".model", ".tiktoken", ".jinja", ".vocab", ".merges")
        )
    }
    selected = sorted(weights | metadata)
    cache_hit = checkpoint_cached and all(cached(name) for name in selected)
    source = Path(
        snapshot_download(
            model, revision=info.sha, cache_dir=cache, allow_patterns=selected
        )
    ).resolve()
    elapsed = time.perf_counter() - start
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    return {
        "seconds": elapsed,
        "revision": info.sha,
        "source": str(source),
        "cache_hit": cache_hit,
        "files": selected,
        "source_bytes": sum((source / name).stat().st_size for name in selected),
        "stored_dtype": config.get(
            "dtype",
            config.get("torch_dtype", config.get("text_config", {}).get("dtype")),
        ),
    }


def download_recipe(cache: Path, revision: str, profile: str) -> dict:
    directory = "cuda" if profile == "cuda-fp16" else "cpu_and_mobile"
    relative = f"Qwen-Qwen3.5-4B/builtin/{directory}/text.json"
    url = f"https://raw.githubusercontent.com/microsoft/olive-recipes/{revision}/{relative}"
    destination = cache / "benchmark-olive-recipes" / revision / directory / "text.json"
    start = time.perf_counter()
    cache_hit = destination.is_file()
    if not cache_hit:
        with urlopen(url, timeout=60) as response:
            if response.geturl() != url:
                raise ValueError(
                    f"Unexpected Olive recipe redirect: {response.geturl()}"
                )
            config = json.load(response)
        if not isinstance(config, dict):
            raise ValueError("Invalid Olive recipe: expected a JSON object.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        write_json(destination, config)
    config = json.loads(destination.read_text(encoding="utf-8"))
    return {
        "seconds": time.perf_counter() - start,
        "cache_hit": cache_hit,
        "revision": revision,
        "url": url,
        "config": config,
    }


def recipe_config(
    original: dict, source: Path, output: Path, profile: str, cache: Path
) -> dict:
    passes = original.get("passes") if isinstance(original, dict) else None
    builder = passes.get("m") if isinstance(passes, dict) else None
    extra_options = (
        builder.get("extra_options", {}) if isinstance(builder, dict) else None
    )
    engine = original.get("engine", {}) if isinstance(original, dict) else None
    if (
        not isinstance(passes, dict)
        or set(passes) != {"m", "t"}
        or not isinstance(builder, dict)
        or builder.get("type") != "ModelBuilder"
        or passes.get("t")
        != {"type": "GraphSurgeries", "surgeries": [{"surgeon": "TieWordEmbeddings"}]}
        or set(builder)
        - {
            "type",
            "precision",
            "accuracy_level",
            "block_size",
            "algo_config",
            "extra_options",
        }
        or not isinstance(extra_options, dict)
        or set(extra_options) - {"filename", "prune_lm_head"}
        or not isinstance(engine, dict)
        or set(engine) - {"target"}
        or not isinstance(original.get("input_model"), dict)
    ):
        raise ValueError(
            "Upstream recipe changed: review the unquantized benchmark adapter before running."
        )
    config = deepcopy(original)
    builder = config["passes"]["m"]
    device, precision, provider = PROFILES[profile]
    config["input_model"]["model_path"] = str(source)
    builder["precision"] = precision
    for key in ("accuracy_level", "block_size", "algo_config"):
        builder.pop(key, None)
    builder.get("extra_options", {}).pop("prune_lm_head", None)
    config.pop("engine", None)
    config.update(
        {
            "systems": {
                "local": {
                    "type": "LocalSystem",
                    "accelerators": [
                        {
                            "device": "gpu" if device == "cuda" else "cpu",
                            "execution_providers": [provider],
                        },
                    ],
                }
            },
            "host": "local",
            "target": "local",
            "cache_dir": str(cache),
            "output_dir": str(output),
            "no_artifacts": True,
        }
    )
    return config


def export_command(
    exporter: str,
    source: Path,
    output: Path,
    profile: str,
    directory: Path,
    recipe: dict | None,
) -> list[str]:
    device, precision, provider = PROFILES[profile]
    command = [sys.executable, "-u", "-m"]
    if exporter in ("mbext", "modelbuilder"):
        command += [
            (
                "modelbuilder.builder"
                if exporter == "mbext"
                else "onnxruntime_genai.models.builder"
            ),
            "-i",
            str(source),
            "-o",
            str(output),
            "-p",
            precision,
            "-e",
            device,
        ]
        if exporter == "mbext":
            command += ["--extra_options", "hf_remote=false"]
    elif exporter == "mobius":
        command += [
            "mobius",
            "build",
            "--config",
            str(source),
            "--output",
            str(output),
            "--dtype",
            "f32" if precision == "fp32" else "f16",
            "--ep",
            device,
            "--features",
            "text-only",
        ]
    elif exporter == "olive":
        command += [
            "olive",
            "capture-onnx-graph",
            "-m",
            str(source),
            "--use_model_builder",
            "--precision",
            precision,
            "--conversion_device",
            "gpu" if device == "cuda" else "cpu",
            "--execution_provider",
            provider,
            "--output_path",
            str(output),
        ]
    elif exporter == "olive-recipes":
        if recipe is None:
            raise ValueError("Missing pinned recipe.")
        config_path = directory / "recipe.json"
        write_json(
            config_path,
            recipe_config(recipe, source, output, profile, directory / "cache"),
        )
        command += [
            "olive",
            "run",
            "--run-config",
            str(config_path),
            "--output_path",
            str(output),
        ]
    else:
        raise ValueError(f"Unknown exporter: {exporter}")
    return command


def run_process(command: list[str], directory: Path, log: Path, timeout: float) -> None:
    environment = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "PYTHONUNBUFFERED": "1",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "HF_HOME": str(directory / "hf-cache"),
        "HF_HUB_CACHE": str(directory / "hf-cache" / "hub"),
    }
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command,
            cwd=directory,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=os.name != "nt",
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )
        try:
            returncode = process.wait(timeout=timeout)
        except (KeyboardInterrupt, subprocess.TimeoutExpired):
            if os.name == "nt":
                cleanup = subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    check=False,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )
                if cleanup.returncode and process.poll() is None:
                    print(
                        f"taskkill exited with status {cleanup.returncode}; killing the exporter process.",
                        file=stream,
                        flush=True,
                    )
                    process.kill()
            else:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait()
            raise
    if returncode:
        raise RuntimeError(
            f"Exporter exited with status {returncode}; see log in report"
        )


def validate_output(output: Path, precision: str) -> dict:
    import onnx
    from google.protobuf.message import DecodeError

    models = sorted(output.rglob("*.onnx"))
    if not models:
        raise ValueError(f"No ONNX model was created in {output}")
    initializer_elements: Counter = Counter()
    model_names = []
    quantized_ops = {
        "MatMulNBits",
        "MatMulBnb4",
        "QLinearMatMul",
        "MatMulInteger",
        "QLinearConv",
        "ConvInteger",
        "QuantizeLinear",
        "DequantizeLinear",
        "DynamicQuantizeLinear",
    }
    for path in models:
        try:
            model = onnx.load_model(path, load_external_data=False)
        except DecodeError as error:
            raise ValueError(f"Invalid ONNX model: {path}") from error
        if any(node.op_type in quantized_ops for node in model.graph.node):
            raise ValueError(f"Unexpected quantization in {path}")
        external_locations = set()
        for tensor in model.graph.initializer:
            initializer_elements[
                onnx.TensorProto.DataType.Name(tensor.data_type)
            ] += math.prod(tensor.dims)
            external_locations.update(
                entry.value for entry in tensor.external_data if entry.key == "location"
            )
        model_directory = path.parent.resolve()
        for location in external_locations:
            external = (model_directory / location).resolve()
            if not external.is_relative_to(model_directory):
                raise ValueError(f"External weights escape model directory: {location}")
            if not external.is_file() or external.stat().st_size == 0:
                raise ValueError(f"Missing or empty external weights: {external}")
        model_names.append(str(path.relative_to(output)))
    expected = "FLOAT" if precision == "fp32" else "FLOAT16"
    if (
        initializer_elements[expected] == 0
        or initializer_elements["BFLOAT16"]
        or (precision == "fp32" and initializer_elements["FLOAT16"])
        or (
            precision == "fp16"
            and initializer_elements["FLOAT16"] < initializer_elements["FLOAT"]
        )
    ):
        raise ValueError(
            f"Unexpected weight precision for {precision}: {dict(initializer_elements)}"
        )
    return {
        "models": model_names,
        "initializer_elements": dict(initializer_elements),
        "output_bytes": sum(
            path.stat().st_size for path in output.rglob("*") if path.is_file()
        ),
    }


def environment_info() -> dict:
    packages = {}
    for name in (
        "modelbuilder",
        "onnxruntime-genai",
        "onnxruntime-genai-cuda",
        "mobius-onnx",
        "olive-ai",
        "onnx",
        "onnx-light",
        "onnxruntime",
        "onnxruntime-gpu",
        "torch",
        "transformers",
    ):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    origins = {}
    for exporter, module in MODULES.items():
        spec = find_spec(module)
        origins[exporter] = spec.origin if spec else None
    return {
        "python": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "module_origins": origins,
        "pythonpath": os.environ.get("PYTHONPATH"),
        "version_note": "Distribution metadata may be absent or differ from PYTHONPATH source checkouts.",
    }


def print_summary(report: dict) -> None:
    print(f"\nBenchmark summary: {report['model']} ({report['status']})")
    print(
        f"{'Profile':10} {'Exporter':13} {'Status':10} "
        f"{'Creation(s)':>12} {'Validation(s)':>13} {'Output(MiB)':>12}"
    )
    for row in report["results"]:
        creation = (
            f"{row['creation_seconds']:.3f}" if "creation_seconds" in row else "-"
        )
        validation = (
            f"{row['validation_seconds']:.3f}" if "validation_seconds" in row else "-"
        )
        size = f"{row['output_bytes'] / 1024**2:.3f}" if "output_bytes" in row else "-"
        print(
            f"{row['profile']:10} {row['exporter']:13} {row['status']:10} "
            f"{creation:>12} {validation:>13} {size:>12}"
        )
    if not report["results"]:
        print("No exporter results.")
    if "error" in report:
        print(f"Setup error: {report['error']}")
    for row in report["results"]:
        if row["status"] == "ok":
            continue
        print(
            f"\n{row['profile']} {row['exporter']}: {row.get('error', row['status'])}"
        )
        if row.get("log"):
            print(
                "Last exporter log lines (up to 20 lines / 4000 characters; full log in JSON):"
            )
            print("\n".join(row["log"][-4000:].splitlines()[-20:]))
    sys.stdout.flush()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Hugging Face model ID (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("qwen35-benchmark.json"),
        help="JSON results file (default: %(default)s).",
    )
    default_cache = (
        Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(os.environ.get("HF_HUB_CACHE", default_cache)),
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="HF revision; resolved to an immutable commit before downloading.",
    )
    parser.add_argument(
        "--recipe-revision",
        default=RECIPE_REVISION,
        help="Immutable olive-recipes commit (40 hex characters).",
    )
    parser.add_argument(
        "--exporters", choices=EXPORTERS, nargs="+", default=list(EXPORTERS)
    )
    parser.add_argument(
        "--profiles", choices=PROFILES, nargs="+", default=list(PROFILES)
    )
    parser.add_argument(
        "--timeout", type=float, default=3600, help="Seconds per exporter process."
    )
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be positive.")
    if not re.fullmatch(r"[a-f0-9]{40}", args.recipe_revision):
        parser.error("--recipe-revision must be an immutable 40-character commit.")
    output = args.output.expanduser().resolve()
    cache = args.cache_dir.expanduser().resolve()
    report = {
        "model": args.model,
        "component": "text-only decoder with embeddings and LM head",
        "status": "incomplete",
        "environment": environment_info(),
        "arguments": vars(args) | {"output": str(output), "cache_dir": str(cache)},
        "results": [],
    }
    report_path = output
    try:
        print(f"Downloading shared checkpoint: {args.model}", flush=True)
        report["model_download"] = download_model(args.model, cache, args.revision)
        write_json(report_path, report)
        source = Path(report["model_download"]["source"])
        print(
            f"Download: {report['model_download']['seconds']:.3f}s; cache_hit={report['model_download']['cache_hit']}",
            flush=True,
        )
        recipes = {}
        report["recipe_downloads"] = recipes
        if "olive-recipes" in args.exporters:
            for profile in dict.fromkeys(args.profiles):
                recipes[profile] = download_recipe(cache, args.recipe_revision, profile)
                write_json(report_path, report)
                print(
                    f"Recipe download ({profile}): {recipes[profile]['seconds']:.3f}s; cache_hit={recipes[profile]['cache_hit']}",
                    flush=True,
                )
        for profile in dict.fromkeys(args.profiles):
            for exporter in dict.fromkeys(args.exporters):
                with TemporaryDirectory(prefix=f"qwen35-{profile}-{exporter}-") as work:
                    directory = Path(work)
                    artifact = directory / "model"
                    log = directory / "run.log"
                    row = {
                        "exporter": exporter,
                        "profile": profile,
                        "status": "incomplete",
                        "backend": (
                            "ModelBuilder"
                            if exporter in ("modelbuilder", "olive", "olive-recipes")
                            else exporter
                        ),
                    }
                    try:
                        row["command"] = export_command(
                            exporter,
                            source,
                            artifact,
                            profile,
                            directory,
                            recipes.get(profile, {}).get("config"),
                        )
                        if exporter == "olive-recipes":
                            row["recipe_config"] = json.loads(
                                (directory / "recipe.json").read_text(encoding="utf-8")
                            )
                        print(
                            f"{profile:10} {exporter:13}: creating model",
                            flush=True,
                        )
                        start = time.perf_counter()
                        try:
                            run_process(row["command"], directory, log, args.timeout)
                        finally:
                            row["creation_seconds"] = time.perf_counter() - start
                        validation_start = time.perf_counter()
                        try:
                            row.update(validate_output(artifact, PROFILES[profile][1]))
                        finally:
                            row["validation_seconds"] = (
                                time.perf_counter() - validation_start
                            )
                        row["status"] = "ok"
                    except (
                        OSError,
                        ValueError,
                        RuntimeError,
                        ImportError,
                        subprocess.SubprocessError,
                    ) as error:
                        row.update(status="failed", error=str(error))
                        print(
                            f"FAILED {profile} {exporter}: {error}",
                            file=sys.stderr,
                            flush=True,
                        )
                    except KeyboardInterrupt:
                        row["status"] = "cancelled"
                        raise
                    finally:
                        if log.is_file():
                            row["log"] = log.read_text(
                                encoding="utf-8", errors="replace"
                            )
                        report["results"].append(row)
                        write_json(report_path, report)
                elapsed = row.get("creation_seconds")
                duration = "not started" if elapsed is None else f"{elapsed:.3f}s"
                print(
                    f"{profile:10} {exporter:13} creation={duration} {row['status']}",
                    flush=True,
                )
        report["status"] = (
            "failed"
            if any(row["status"] != "ok" for row in report["results"])
            else "ok"
        )
    except KeyboardInterrupt:
        report["status"] = "cancelled"
        return 130
    except (
        OSError,
        ValueError,
        RuntimeError,
        ImportError,
        subprocess.SubprocessError,
    ) as error:
        report.update(status="failed", error=str(error))
        print(f"FAILED benchmark setup: {error}", file=sys.stderr, flush=True)
        return 1
    finally:
        write_json(report_path, report)
        print_summary(report)
        print(f"Results: {report_path}", flush=True)
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
