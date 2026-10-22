from pathlib import Path
from struct import pack

from onnx_light.onnx import ModelProto, TensorProto, checker, helper, save_model


def value_info(name):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, [1])


graph = helper.make_graph(
    [helper.make_node("Relu", ["input"], ["output"], name="activation")],
    "A first ONNX graph",
    [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3])],
    [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3])],
)
model = helper.make_model(
    graph,
    producer_name="onnx-light-viewer",
    ir_version=9,
    opset_imports=[helper.make_opsetid("", 13)],
)
checker.check_model(model)
save_model(model, Path(__file__).with_name("relu.onnx"))
save_model(model, Path(__file__).parents[2] / "media" / "relu.onnx")

weight = helper.make_tensor("weight", TensorProto.FLOAT, [1], [1.0])
sparse = helper.make_sparse_tensor(
    helper.make_tensor("sparse", TensorProto.FLOAT, [1], [1.0]),
    helper.make_tensor("indices", TensorProto.INT64, [1], [0]),
    [1],
)
custom_graph = helper.make_graph(
    [
        helper.make_node("Add", ["x", "weight"], ["intermediate"]),
        helper.make_node("Custom", ["intermediate", ""], ["y", ""], domain="example"),
    ],
    "Custom operators",
    [value_info(name) for name in ("x", "weight", "sparse")],
    [value_info("y")],
    initializer=[weight],
    sparse_initializer=[sparse],
)
custom = helper.make_model(
    custom_graph,
    ir_version=9,
    opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid("example", 1)],
)
checker.check_model(custom)
save_model(custom, Path(__file__).with_name("custom.onnx"))

passthrough = helper.make_model(
    helper.make_graph([], "Passthrough", [value_info("x")], [value_info("x")]),
    ir_version=9,
    opset_imports=[helper.make_opsetid("", 13)],
)
checker.check_model(passthrough)
save_model(passthrough, Path(__file__).with_name("passthrough.onnx"))
passthrough.graph.name = ""
save_model(passthrough, Path(__file__).with_name("unnamed.onnx"))

missing_ir = helper.make_model(graph, ir_version=0)
save_model(missing_ir, Path(__file__).with_name("missing-ir.onnx"))
missing_graph = ModelProto()
missing_graph.ir_version = 9
save_model(missing_graph, Path(__file__).with_name("missing-graph.onnx"))
missing_operator = helper.make_model(
    helper.make_graph([helper.make_node("", [], [])], "", [], []),
    ir_version=9,
)
save_model(missing_operator, Path(__file__).with_name("missing-operator.onnx"))

parameter_sparse = helper.make_sparse_tensor(
    helper.make_tensor("preview", TensorProto.FLOAT, [12], list(range(12))),
    helper.make_tensor("indices", TensorProto.INT64, [12], list(range(12))),
    [12],
)
parameter_graph = helper.make_graph(
    [
        helper.make_node(
            "Conv", ["x", "kernel"], ["conv"], kernel_shape=[3, 3], strides=[2, 2]
        ),
        helper.make_node("LeakyRelu", ["conv"], ["activation"], alpha=0.125),
        helper.make_node(
            "Softmax",
            ["activation", "sparse"],
            ["y"],
            domain="example",
            threshold=2**60 + 1,
            text="<b>model text</b>",
            ratios=[0.5, 1.5],
            labels=["left", "right"],
            value=helper.make_tensor("constant", TensorProto.INT64, [2], [3, 7]),
            long_list=list(range(2000)),
            long_floats=[float(index) for index in range(12)],
            long_strings=[f"label{index}" for index in range(12)],
            sparse_value=parameter_sparse,
        ),
    ],
    "Operator parameters",
    [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 5, 5])],
    [value_info("y")],
    initializer=[
        helper.make_tensor("kernel", TensorProto.FLOAT, [1, 1, 3, 3], [0.5] * 9)
    ],
    sparse_initializer=[sparse],
)
parameter_model = helper.make_model(
    parameter_graph,
    ir_version=9,
    opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid("example", 1)],
)
checker.check_model(parameter_model)
save_model(parameter_model, Path(__file__).with_name("parameters.onnx"))

for version in (11, 13):
    softmax = helper.make_model(
        helper.make_graph(
            [helper.make_node("Softmax", ["x"], ["y"])],
            "Versioned schema",
            [value_info("x")],
            [value_info("y")],
        ),
        ir_version=9,
        opset_imports=[helper.make_opsetid("", version)],
    )
    checker.check_model(softmax)
    save_model(softmax, Path(__file__).with_name(f"softmax-{version}.onnx"))

inference = helper.make_model(
    helper.make_graph(
        [
            helper.make_node("Add", ["x", "bias"], ["hidden"]),
            helper.make_node("Reshape", ["hidden", "target"], ["reshaped"]),
            helper.make_node("Transpose", ["reshaped"], ["transposed"], perm=[0, 2, 1]),
            helper.make_node("Cast", ["transposed"], ["y"], to=TensorProto.INT64),
        ],
        "Shape inference",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 3])],
        [helper.make_tensor_value_info("y", TensorProto.INT64, ["N", 1, 3])],
        initializer=[
            helper.make_tensor("bias", TensorProto.FLOAT, [3], [0.0, 0.0, 0.0]),
            helper.make_tensor(
                "target", TensorProto.INT64, [3], pack("<qqq", 0, 3, 1), raw=True
            ),
        ],
    ),
    ir_version=9,
    opset_imports=[helper.make_opsetid("", 13)],
)
checker.check_model(inference)
save_model(inference, Path(__file__).with_name("inference.onnx"))

metadata = helper.make_model(
    helper.make_graph(
        [],
        "Tensor metadata",
        [
            helper.make_tensor_value_info("scalar", TensorProto.FLOAT, []),
            helper.make_tensor_value_info("zero", TensorProto.INT64, [0, "N", None]),
            helper.make_tensor_value_info("unknown_rank", TensorProto.FLOAT, None),
            helper.make_tensor_sequence_value_info("sequence", TensorProto.FLOAT, [2]),
        ],
        [],
    ),
    ir_version=9,
    opset_imports=[helper.make_opsetid("", 13)],
)
save_model(metadata, Path(__file__).with_name("metadata.onnx"))
