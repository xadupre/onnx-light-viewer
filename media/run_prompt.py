"""Bundled causal generation and teacher-forced Hugging Face comparison."""

import json
import math
import pathlib
import sys

CAPTURE_BUDGET = 256 * 1024 * 1024


def retain_row(rows, row, used, limit):
    """Bound Python float/list storage before retaining another vocabulary row."""
    used += len(row) * 40 + 64
    if rows is None or used > limit:
        return None, used
    rows.append(row)
    return rows, used


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
    directory = pathlib.Path(request["modelPath"]).parent
    if (directory / "config.json").is_file() and (
        (directory / "tokenizer.json").is_file()
        or (directory / "tokenizer_config.json").is_file()
    ):
        return str(directory)
    identity = provenance.get("model")
    if identity and not pathlib.Path(identity).is_absolute():
        from huggingface_hub import snapshot_download

        return snapshot_download(
            identity, revision=provenance.get("revision"), local_files_only=True
        )
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
            raise ValueError("Engine and reference tokenizers encoded the prompt differently.")
    model = AutoModelForCausalLM.from_pretrained(
        source, local_files_only=True, trust_remote_code=False
    )
    model.eval()
    reference = []
    discrepancies = [] if logits is not None else None
    prefix = list(prompt_ids)
    # Each step consumes the exact generated prefix, never an independently
    # generated reference sequence whose divergence would invalidate comparison.
    with torch.inference_mode():
        for index, token in enumerate(tokens):
            output = model(
                input_ids=torch.tensor([prefix], dtype=torch.long),
                attention_mask=torch.ones((1, len(prefix)), dtype=torch.long),
                use_cache=False,
            )
            expected = output.logits[0, -1].float().cpu().tolist()
            if not expected or not all(math.isfinite(value) for value in expected):
                raise ValueError("The HF reference returned empty or non-finite logits.")
            if discrepancies is not None:
                discrepancies.append(compare_logits(logits[index], expected))
            predicted_id = max(range(len(expected)), key=expected.__getitem__)
            reference.append(
                {
                    "text": tokenizer.decode([predicted_id], skip_special_tokens=False),
                    "id": predicted_id,
                    "score": float(expected[predicted_id]),
                }
            )
            prefix.append(token["id"])
    return {"tokens": reference}, discrepancies


def generate_genai(request):
    import onnxruntime_genai as og

    directory = pathlib.Path(request["modelPath"]).parent
    with (directory / "genai_config.json").open(encoding="utf-8") as stream:
        config = json.load(stream)
    decoder = config["model"]["decoder"]
    configured_model = decoder.get("filename")
    if configured_model and (
        directory / configured_model
    ).resolve() != pathlib.Path(request["modelPath"]).resolve():
        raise ValueError("The selected model is not the GenAI configuration's decoder.")
    model = og.Model(str(directory))
    tokenizer = og.Tokenizer(model)
    prompt_ids = [int(token) for token in tokenizer.encode(request["prompt"])]
    if not prompt_ids:
        raise ValueError("The tokenizer produced an empty prompt.")
    params = og.GeneratorParams(model)
    params.set_search_options(
        max_length=len(prompt_ids) + request["maxTokens"],
        do_sample=False,
        num_beams=1,
        num_return_sequences=1,
    )
    generator = og.Generator(model, params)
    generator.append_tokens(prompt_ids)
    tokens, logits = [], []
    captured_size = 0
    capture_limit = min(request.get("captureBudget", CAPTURE_BUDGET), CAPTURE_BUDGET)
    stream = tokenizer.create_stream()
    while len(tokens) < request["maxTokens"] and not generator.is_done():
        # get_logits computes the next inference step before search processors.
        # Copy to owned Python numbers before generate_next_token mutates its buffer.
        output = generator.get_logits()
        row = output.reshape(-1, output.shape[-1])[-1].astype("float64").tolist()
        generator.generate_next_token()
        token_id = int(generator.get_next_tokens()[0])
        if token_id < 0 or token_id >= len(row):
            raise ValueError("The engine returned an out-of-vocabulary token.")
        tokens.append(
            {"text": stream.decode(token_id), "id": token_id, "score": row[token_id]}
        )
        logits, captured_size = retain_row(
            logits, row, captured_size, capture_limit
        )
        print(f"Generated {len(tokens)}/{request['maxTokens']} tokens", flush=True)
    return prompt_ids, tokens, logits


def generate_light(request):
    import numpy as np
    from onnx_light import onnx
    from onnx_light.onnx.reference import ReferenceEvaluator
    from transformers import AutoTokenizer

    if pathlib.Path(request["modelPath"]).suffix.lower() == ".pipe-onnx":
        raise ValueError(
            "The onnx-light Python API does not directly execute PipelineProto containers. "
            "Pipeline execution requires an upstream public SDK capability."
        )
    tokenizer = AutoTokenizer.from_pretrained(
        source_for(request), local_files_only=True, trust_remote_code=False
    )
    prompt_ids = [
        int(token)
        for token in tokenizer.encode(request["prompt"], add_special_tokens=True)
    ]
    if not prompt_ids:
        raise ValueError("The tokenizer produced an empty prompt.")
    model = onnx.load(request["modelPath"], load_external_data=True)
    evaluator = ReferenceEvaluator(model)
    generated = evaluator.generate(
        {"input_ids": np.asarray([prompt_ids], dtype=np.int64)},
        max_new_tokens=request["maxTokens"],
        temperature=0.0,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated_ids = np.asarray(generated).tolist()[0][len(prompt_ids):]
    tokens = []
    for generated_id in generated_ids:
        token_id = int(generated_id)
        tokens.append({
            "text": tokenizer.decode([token_id], skip_special_tokens=False),
            "id": token_id,
            "score": None,
        })
        print(f"Generated {len(tokens)}/{request['maxTokens']} tokens", flush=True)
    return prompt_ids, tokens, None


def main():
    with open(sys.argv[1], encoding="utf-8") as stream:
        request = json.load(stream)
    if request.get("reuseCapture"):
        with open(request["capturePath"], encoding="utf-8") as stream:
            captured = json.load(stream)
        prompt_ids = captured["promptIds"]
        tokens = captured["tokens"]
        logits = captured["logits"]
    elif request["engine"] == "onnxruntime-genai":
        prompt_ids, tokens, logits = generate_genai(request)
    elif request["engine"] == "onnx-light":
        prompt_ids, tokens, logits = generate_light(request)
    else:
        raise ValueError("Unsupported prompt engine.")
    if not request.get("reuseCapture"):
        with open(request["capturePath"], "x", encoding="utf-8") as stream:
            json.dump(
                {"promptIds": prompt_ids, "tokens": tokens, "logits": logits},
                stream,
                allow_nan=False,
            )
    if "tokenIds" in request and request["tokenIds"] != [
        token["id"] for token in tokens
    ]:
        raise ValueError("Regenerated tokens no longer match the previous prompt result.")
    result = {
        "prompt": request["prompt"],
        "engine": request["engine"],
        "tokens": tokens,
    }
    if logits is None:
        result["limitations"] = [
            "onnx-light native generation exposes token IDs only. Selected raw scores "
            "and full-vocabulary discrepancy metrics are unavailable; HF validation "
            "still predicts next tokens at the exported prefixes."
        ] if request["engine"] == "onnx-light" else [
            "Full vocabulary logits were unavailable or exceeded the retention budget; discrepancy metrics "
            "are unavailable. HF validation still predicts next tokens at the "
            "exported token prefixes."
        ]
    if request["validate"]:
        print("Validating generated prefixes with the local HF reference", flush=True)
        result["reference"], discrepancies = reference_logits(
            request, prompt_ids, tokens, logits
        )
        if discrepancies is not None:
            result["discrepancies"] = discrepancies
    with open(sys.argv[2], "x", encoding="utf-8") as stream:
        json.dump(result, stream, allow_nan=False)


if __name__ == "__main__":
    main()
