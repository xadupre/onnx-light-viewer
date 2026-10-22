"""Generate a tiny, randomly initialized causal language-model example."""

from math import prod, sqrt
from pathlib import Path
from random import Random

from onnx_light.onnx import TensorProto, checker, helper, save_model


def make_model():
    random = Random(0)
    nodes = []
    initializers = []

    def tensor(name, shape, values, dtype=TensorProto.FLOAT):
        initializers.append(helper.make_tensor(name, dtype, shape, values))
        return name

    def weight(name, shape):
        return tensor(name, shape, [random.uniform(-0.1, 0.1) for _ in range(prod(shape))])

    def node(op, inputs, output, **attributes):
        nodes.append(helper.make_node(op, inputs, [output], name=output, **attributes))
        return output

    def rms_norm(value, name):
        square = node("Mul", [value, value], f"{name}.square")
        mean = node("ReduceMean", [square], f"{name}.mean", axes=[-1], keepdims=1)
        epsilon = tensor(f"{name}.epsilon", [], [1e-5])
        variance = node("Add", [mean, epsilon], f"{name}.variance")
        root = node("Sqrt", [variance], f"{name}.root")
        normalized = node("Div", [value, root], f"{name}.normalized")
        scale = tensor(f"{name}.scale", [16], [1.0] * 16)
        return node("Mul", [normalized, scale], name)

    tokens = weight("token_embedding", [32, 16])
    positions = weight("position_embedding", [128, 16])
    embedded = node("Gather", [tokens, "input_ids"], "token_vectors", axis=0)
    position_vectors = node("Gather", [positions, "position_ids"], "position_vectors", axis=0)
    hidden = node("Add", [embedded, position_vectors], "embeddings")
    normalized = rms_norm(hidden, "attention_norm")
    head_shape = tensor("head_shape", [4], [0, 0, 2, 8], TensorProto.INT64)
    heads = {}
    for name in ("query", "key", "value"):
        projection = weight(f"{name}.weight", [16, 16])
        projected = node("MatMul", [normalized, projection], f"{name}.projection")
        split = node("Reshape", [projected, head_shape], f"{name}.split")
        permutation = [0, 2, 3, 1] if name == "key" else [0, 2, 1, 3]
        heads[name] = node("Transpose", [split], f"{name}.heads", perm=permutation)
    scores = node("MatMul", [heads["query"], heads["key"]], "attention_scores")
    scale = tensor("attention_scale", [], [sqrt(8)])
    scaled = node("Div", [scores, scale], "scaled_scores")
    masked = node("Add", [scaled, "attention_mask"], "masked_scores")
    probabilities = node("Softmax", [masked], "attention_probabilities", axis=-1)
    context = node("MatMul", [probabilities, heads["value"]], "attention_context")
    transposed = node("Transpose", [context], "context_transposed", perm=[0, 2, 1, 3])
    merge_shape = tensor("merge_shape", [3], [0, 0, 16], TensorProto.INT64)
    merged = node("Reshape", [transposed, merge_shape], "context_merged")
    output_projection = weight("attention_output.weight", [16, 16])
    attention = node("MatMul", [merged, output_projection], "attention_output")
    residual = node("Add", [hidden, attention], "attention_residual")
    normalized = rms_norm(residual, "feed_forward_norm")
    up = weight("feed_forward_up.weight", [16, 32])
    expanded = node("MatMul", [normalized, up], "feed_forward_expanded")
    activated = node("Relu", [expanded], "feed_forward_activated")
    down = weight("feed_forward_down.weight", [32, 16])
    contracted = node("MatMul", [activated, down], "feed_forward_output")
    hidden = node("Add", [residual, contracted], "feed_forward_residual")
    final = rms_norm(hidden, "final_norm")
    lm_head = weight("lm_head.weight", [16, 32])
    node("MatMul", [final, lm_head], "logits")

    def value_info(name, dtype, shape):
        return helper.make_tensor_value_info(name, dtype, shape)

    batch, sequence = "batch_size", "sequence_length"
    graph = helper.make_graph(
        nodes,
        "Tiny symbolic transformer",
        [
            value_info("input_ids", TensorProto.INT64, [batch, sequence]),
            value_info("position_ids", TensorProto.INT64, [batch, sequence]),
            value_info("attention_mask", TensorProto.FLOAT, [batch, 1, sequence, sequence]),
        ],
        [value_info("logits", TensorProto.FLOAT, [batch, sequence, 32])],
        initializer=initializers,
        value_info=[
            value_info("token_vectors", TensorProto.FLOAT, [batch, sequence, 16]),
            value_info("query.heads", TensorProto.FLOAT, [batch, 2, sequence, 8]),
            value_info("attention_scores", TensorProto.FLOAT, [batch, 2, sequence, sequence]),
        ],
    )
    model = helper.make_model(
        graph,
        producer_name="onnx-light-viewer",
        ir_version=9,
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.doc_string = (
        "One randomly initialized decoder block, 2 heads, hidden size 16, vocabulary 32. "
        "Not pretrained. Supply an additive causal attention mask (0 allowed, -inf masked)."
    )
    checker.check_model(model)
    return model


if __name__ == "__main__":
    save_model(make_model(), Path(__file__).parents[1] / "media" / "tiny-llm.onnx")
