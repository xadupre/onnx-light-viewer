"""Generate a self-contained three-stage pipeline for manual viewer testing."""

from pathlib import Path

import onnx_light.onnx.helper as oh
from onnx_light.onnx import PipelineProto, TensorProto, checker
from onnx_light.onnx_proto import verify


def make_pipeline():
    def features(name):
        return oh.make_tensor_value_info(
            name, TensorProto.FLOAT, ["batch", "sequence", 3]
        )

    def model(graph):
        result = oh.make_model(
            graph,
            producer_name="onnx-light-viewer",
            ir_version=10,
            opset_imports=[oh.make_opsetid("", 13)],
        )
        checker.check_model(result)
        return result

    embedding = model(
        oh.make_graph(
            [
                oh.make_node(
                    "Gather",
                    ["embedding.transition_table", "input_ids"],
                    ["inputs_embeds"],
                    name="token_embedding",
                    axis=0,
                )
            ],
            "Token embeddings",
            [
                oh.make_tensor_value_info(
                    "input_ids", TensorProto.INT64, ["batch", "sequence"]
                )
            ],
            [features("inputs_embeds")],
            initializer=[
                oh.make_tensor(
                    "embedding.transition_table",
                    TensorProto.FLOAT,
                    [3, 3],
                    [0, 4, 0, 0, 0, 4, 4, 0, 0],
                )
            ],
        )
    )
    prefill = model(
        oh.make_graph(
            [
                oh.make_node(
                    "Transpose",
                    ["inputs_embeds"],
                    ["prefill.transposed"],
                    name="swap_sequence_features",
                    perm=[0, 2, 1],
                ),
                oh.make_node(
                    "Transpose",
                    ["prefill.transposed"],
                    ["logits"],
                    name="restore_sequence_features",
                    perm=[0, 2, 1],
                ),
            ],
            "Prompt logits",
            [features("inputs_embeds")],
            [features("logits")],
        )
    )
    decode = model(
        oh.make_graph(
            [
                oh.make_node(
                    "Identity", ["inputs_embeds"], ["logits"], name="next_token_logits"
                )
            ],
            "Next-token logits",
            [features("inputs_embeds")],
            [features("logits")],
        )
    )
    pipeline = PipelineProto(
        name="Three-token cycle",
        producer_name="onnx-light-viewer",
        pipeline_version=1,
        doc_string=(
            "Tiny deterministic vocabulary: 0 -> 1 -> 2 -> 0. "
            "Prefill contains an identity transpose pair for testing fusion. "
            "Decode has the same logits interface, without persistent state."
        ),
        embedding=embedding,
        prefill=prefill,
        decode=decode,
        max_new_tokens=4,
        temperature=0.0,
        seed=42,
        eos_token_ids=[2],
        pad_token_id=0,
        input_ids_name="input_ids",
        logits_name="logits",
    )
    verify.verify_pipeline(pipeline)
    return pipeline


if __name__ == "__main__":
    data = make_pipeline().SerializeToString()
    for destination in (
        Path(__file__).with_name("tiny.pipe-onnx"),
        Path(__file__).parents[1] / "test" / "models" / "tiny.pipe-onnx",
    ):
        destination.write_bytes(data)
        print(destination)
