from pathlib import Path

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
