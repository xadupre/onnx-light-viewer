from pathlib import Path

import onnx_light.onnx.helper as oh
from onnx_light.onnx import RewritingProto, TensorProto, checker, save_model
from onnx_light.onnx_core.graph_builder import GraphBuilder


model = oh.make_model(
    oh.make_graph(
        [
            oh.make_node("Transpose", ["x"], ["t0"], perm=[1, 0]),
            oh.make_node("Transpose", ["t0"], ["a"], perm=[1, 0]),
            oh.make_node("Transpose", ["a"], ["t1"], perm=[1, 0]),
            oh.make_node("Transpose", ["t1"], ["y"], perm=[1, 0]),
        ],
        "Two execution-provider rewriting scenarios",
        [oh.make_tensor_value_info("x", TensorProto.FLOAT, [2, 2])],
        [oh.make_tensor_value_info("y", TensorProto.FLOAT, [2, 2])],
    ),
    ir_version=9,
    opset_imports=[oh.make_opsetid("", 13)],
)
builder = GraphBuilder(model)
for name, pairs in (("EP1", 1), ("EP2", 2)):
    plan = RewritingProto()
    for index in range(pairs):
        rewrite = plan.rewrites.add()
        rewrite.pattern_name = "TransposeTranspose"
        rewrite.matched_nodes.extend([index, index + 1])
        rewrite.added_nodes.append(
            oh.make_node(
                "Identity",
                ["x" if index == 0 else "a"],
                ["a" if index == 0 else "y"],
                name=f"pair{index}",
            )
        )
        rewrite.added_nodes_positions.append(index)
    model.add_rewriting(name, builder.capture_rewritings(plan))
checker.check_model(model)
save_model(model, Path(__file__).with_name("rewritings-ep1-ep2.onnx"))
