from pathlib import Path

from onnx_light.onnx import TensorProto, checker, helper, save_model


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
