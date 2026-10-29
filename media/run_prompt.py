"""Teacher-forced Hugging Face reference; ONNX engines execute in C++."""

import json
import math
import pathlib
import sys
import time


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
    source = provenance.get("source")
    if source and pathlib.Path(source).is_dir():
        return source
    identity = provenance.get("model")
    if identity and not pathlib.Path(identity).is_absolute():
        from huggingface_hub import snapshot_download

        return snapshot_download(
            identity, revision=provenance.get("revision"), local_files_only=True
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
        "No local Hugging Face source is available. Export with source provenance "
        "and keep the downloaded checkpoint/tokenizer cache for reference validation."
    )


def reference_logits(request, prompt_ids, tokens, logits):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    source = source_for(request)
    tokenizer = AutoTokenizer.from_pretrained(
        source, local_files_only=True, trust_remote_code=False
    )
    encoded = tokenizer.encode(request["prompt"], add_special_tokens=False)
    if encoded != list(prompt_ids):
        encoded = tokenizer.encode(request["prompt"], add_special_tokens=True)
        if encoded != list(prompt_ids):
            raise ValueError(
                "Engine and reference tokenizers encoded the prompt differently."
            )
    model = AutoModelForCausalLM.from_pretrained(
        source, local_files_only=True, trust_remote_code=False
    )
    model.eval()
    reference = []
    discrepancies = [] if logits is not None else None
    prefix = list(prompt_ids)
    past = None
    # Each step consumes the exact generated prefix, never an independently
    # generated reference sequence whose divergence would invalidate comparison.
    with torch.inference_mode():
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
                    "text": tokenizer.decode([predicted_id], skip_special_tokens=False),
                    "id": predicted_id,
                    "score": float(expected[predicted_id]),
                    "timeMs": elapsed,
                }
            )
            prefix.append(token["id"])
    return {"tokens": reference}, discrepancies


def main():
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
