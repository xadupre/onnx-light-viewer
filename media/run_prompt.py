"""Teacher-forced Hugging Face reference; ONNX engines execute in C++."""

import json
import math
import pathlib
import sys
import time
from copy import copy
from os.path import commonprefix


def compare_logits(actual, reference):
    """Compare every vocabulary entry; relative error is L2 / reference L2."""
    actual = [float(value) for value in actual]
    reference = [float(value) for value in reference]
    if not actual or len(actual) != len(reference):
        raise ValueError("The engine and reference vocabulary sizes differ.")
    if not all(math.isfinite(value) for value in actual + reference):
        raise ValueError("Non-finite vocabulary logits cannot be compared.")
    differences = [abs(a - b) for a, b in zip(actual, reference)]
    l2 = math.hypot(*differences)
    norm = math.hypot(*reference)
    return {
        "maxAbsolute": max(differences),
        "relative": l2 / max(norm, sys.float_info.epsilon),
        "l2": l2,
        "below001": sum(value < 0.01 for value in differences),
        "below01": sum(value < 0.1 for value in differences),
        "below1": sum(value < 1 for value in differences),
        "total": len(differences),
    }


def source_for(request):
    provenance = request.get("provenance") or {}
    reference = request.get("referenceSource")
    if reference and pathlib.Path(reference).is_absolute():
        if not pathlib.Path(reference).is_dir():
            raise ValueError(
                f"Hugging Face reference directory does not exist: {reference}"
            )
        return reference
    source = provenance.get("source") if not reference else None
    if source and pathlib.Path(source).is_dir():
        return source
    identity = reference or provenance.get("model")
    if identity and not pathlib.Path(identity).is_absolute():
        from huggingface_hub import try_to_load_from_cache

        config = try_to_load_from_cache(
            identity,
            "config.json",
            revision=provenance.get("revision") if not reference else None,
        )
        if isinstance(config, str):
            return str(pathlib.Path(config).parent)
        if reference:
            raise ValueError(
                f"No cached Hugging Face config.json is available for {identity}. "
                "Choose a local checkpoint directory or download the checkpoint first."
            )
    directory = pathlib.Path(request["modelPath"]).parent
    if (
        (directory / "config.json").is_file()
        and (
            (directory / "tokenizer.json").is_file()
            or (directory / "tokenizer_config.json").is_file()
        )
        and any(
            (directory / name).is_file()
            for name in (
                "model.safetensors",
                "model.safetensors.index.json",
                "pytorch_model.bin",
                "pytorch_model.bin.index.json",
            )
        )
    ):
        return str(directory)
    raise ValueError(
        "No local Hugging Face source is available. Enter the original model ID "
        "or checkpoint directory in the Prompt panel's Hugging Face reference field. "
        "The checkpoint must already be cached locally; validation does not download weights."
    )


def check_reference(request):
    source = pathlib.Path(source_for(request))
    if not (source / "config.json").is_file():
        raise ValueError(f"Hugging Face reference config.json is missing: {source}")
    if not any(
        path.is_file()
        for path in source.iterdir()
        if path.name in ("tokenizer.json", "vocab.json", "vocab.txt")
        or path.suffix in (".model", ".tiktoken")
        or path.name.startswith("tokenizer.model.")
    ):
        raise ValueError(f"Hugging Face reference tokenizer files are missing: {source}")
    if not any(
        (source / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")
    ):
        index = next(
            (
                source / name
                for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json")
                if (source / name).is_file()
            ),
            None,
        )
        if index is None:
            raise ValueError(
                f"Hugging Face reference checkpoint weights are missing: {source}"
            )
        with index.open(encoding="utf-8") as stream:
            weights = json.load(stream).get("weight_map")
        if not isinstance(weights, dict) or not weights:
            raise ValueError(f"Invalid Hugging Face checkpoint weight index: {index}")
        if not all(isinstance(shard, str) for shard in weights.values()):
            raise ValueError(f"Invalid Hugging Face checkpoint weight index: {index}")
        for shard in sorted(set(weights.values())):
            if not (source / shard).is_file():
                raise ValueError(f"Hugging Face reference weight shard is missing: {shard}")
    return str(source)


class ReferenceDecoder:
    def __init__(self, tokenizer, prompt_ids):
        self.tokenizer = tokenizer
        self.backend = getattr(tokenizer, "backend_tokenizer", None)
        self.stream = None
        self.text = None
        if self.backend is not None:
            from tokenizers import decoders

            stream_type = getattr(decoders, "DecodeStream", None)
            if stream_type is not None and hasattr(stream_type, "__copy__"):
                self.stream = stream_type(skip_special_tokens=False)
                self.stream.step(self.backend, prompt_ids)

    def decode(self, prefix, predicted_id, exported_id):
        if self.stream is not None:
            text = copy(self.stream).step(self.backend, predicted_id) or ""
            self.stream.step(self.backend, exported_id)
            return text
        options = {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}
        if self.text is None:
            self.text = self.tokenizer.decode(prefix, **options).rstrip("\ufffd")
        candidate = self.tokenizer.decode([*prefix, predicted_id], **options).rstrip("\ufffd")
        text = candidate[len(commonprefix((self.text, candidate))):]
        self.text = (
            candidate
            if predicted_id == exported_id
            else self.tokenizer.decode([*prefix, exported_id], **options).rstrip("\ufffd")
        )
        return text


def reference_logits(request, prompt_ids, tokens, logits):
    source = check_reference(request)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        source, local_files_only=True, trust_remote_code=False
    )
    load_ms = (time.perf_counter() - load_start) * 1000
    encoded = tokenizer.encode(request["prompt"], add_special_tokens=False)
    if encoded != list(prompt_ids):
        encoded = tokenizer.encode(request["prompt"], add_special_tokens=True)
        if encoded != list(prompt_ids):
            raise ValueError(
                "Engine and reference tokenizers encoded the prompt differently."
            )
    load_start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        source, local_files_only=True, trust_remote_code=False
    )
    model.eval()
    load_ms += (time.perf_counter() - load_start) * 1000
    reference = []
    discrepancies = [] if logits is not None else None
    prefix = list(prompt_ids)
    decoder = ReferenceDecoder(tokenizer, prompt_ids)
    past = None
    # Each step consumes the exact generated prefix, never an independently
    # generated reference sequence whose divergence would invalidate comparison.
    with torch.inference_mode():
        warmup_start = time.perf_counter()
        warmup = model(
            input_ids=torch.tensor([[prompt_ids[0]]], dtype=torch.long),
            attention_mask=torch.ones((1, 1), dtype=torch.long),
            use_cache=True,
        )
        warmup.logits[0, -1].float().cpu()
        del warmup
        warmup_ms = (time.perf_counter() - warmup_start) * 1000
        for index, token in enumerate(tokens):
            ids = prefix if index == 0 else [prefix[-1]]
            feeds = {
                "input_ids": torch.tensor([ids], dtype=torch.long),
                "attention_mask": torch.ones((1, len(prefix)), dtype=torch.long),
                "use_cache": True,
            }
            if past is not None:
                feeds["past_key_values"] = past
            start = time.perf_counter()
            output = model(**feeds)
            expected_tensor = output.logits[0, -1].float().cpu()
            elapsed = (time.perf_counter() - start) * 1000
            expected = expected_tensor.tolist()
            past = output.past_key_values
            if past is None and index + 1 < len(tokens):
                raise ValueError("The HF reference did not return a decode cache.")
            if not expected or not all(math.isfinite(value) for value in expected):
                raise ValueError(
                    "The HF reference returned empty or non-finite logits."
                )
            if discrepancies is not None:
                discrepancies.append(compare_logits(logits[index], expected))
            predicted_id = max(range(len(expected)), key=expected.__getitem__)
            reference.append(
                {
                    "text": decoder.decode(prefix, predicted_id, token["id"]),
                    "id": predicted_id,
                    "score": float(expected[predicted_id]),
                    "timeMs": elapsed,
                }
            )
            prefix.append(token["id"])
    return {
        "tokens": reference,
        "loadMs": load_ms,
        "warmupMs": warmup_ms,
    }, discrepancies


def main():
    if sys.argv[1] == "--check-reference":
        path = pathlib.Path(sys.argv[2])
        with path.open(encoding="utf-8") as stream:
            request = json.load(stream)
        print("Checking local Hugging Face reference before generation", flush=True)
        request["referenceSource"] = check_reference(request)
        with path.open("w", encoding="utf-8") as stream:
            json.dump(request, stream)
        print("Hugging Face reference files are ready.", flush=True)
        return
    with open(sys.argv[1], encoding="utf-8") as stream:
        request = json.load(stream)
    with open(request["capturePath"], encoding="utf-8") as stream:
        captured = json.load(stream)
    prompt_ids = captured["promptIds"]
    tokens = captured["tokens"]
    logits = captured["logits"]
    if "tokenIds" in request and request["tokenIds"] != [
        token["id"] for token in tokens
    ]:
        raise ValueError(
            "Regenerated tokens no longer match the previous prompt result."
        )
    output = pathlib.Path(sys.argv[2])
    if output.exists():
        with output.open(encoding="utf-8") as stream:
            result = json.load(stream)
    else:
        result = {
            "prompt": request["prompt"],
            "engine": request["engine"],
            "tokens": tokens,
        }
        if logits is None:
            result["limitations"] = [
                "The native SDK did not retain vocabulary logits; discrepancy metrics are unavailable."
            ]
    if request["validate"]:
        print("Validating generated prefixes with the local HF reference", flush=True)
        result["reference"], discrepancies = reference_logits(
            request, prompt_ids, tokens, logits
        )
        if discrepancies is not None:
            result["discrepancies"] = discrepancies
    with output.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, allow_nan=False)


if __name__ == "__main__":
    main()
