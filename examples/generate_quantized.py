"""Generate small models with portable onnx-light 2-bit and 3-bit weights."""

from pathlib import Path

import onnx_light.onnx.helper as oh
from onnx_light.onnx import (
    ExtendedInitializerProto,
    TensorProto,
    TypeProto,
    checker,
    save_model,
)
from onnx_light.onnx_core.quantization import (
    QuantizationFormat,
    make_quantization_plan,
    quantize_tensor_proto,
)


for bits, profile in ((2, QuantizationFormat.Q2_K), (3, QuantizationFormat.Q3_K)):
    weight = oh.make_tensor(
        "weight", TensorProto.FLOAT, [8, 4], [(index - 16) / 16 for index in range(32)]
    )
    encoded = quantize_tensor_proto(
        weight, make_quantization_plan(profile, 32, block_size=32)
    )
    storage_type = TypeProto()
    storage_type.struct_type.CopyFrom(encoded.struct_type)
    model = oh.make_model(
        oh.make_graph(
            [
                oh.make_node(
                    "Dequantize",
                    ["weight"],
                    ["decoded"],
                    domain="ai.rt",
                    dtype=TensorProto.FLOAT,
                ),
                oh.make_node("MatMul", ["x", "decoded"], ["y"]),
            ],
            f"Portable {bits}-bit quantized weights",
            [oh.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 8])],
            [oh.make_tensor_value_info("y", TensorProto.FLOAT, ["N", 4])],
            value_info=[oh.make_value_info("weight", storage_type)],
        ),
        producer_name="onnx-light-viewer",
        opset_imports=[oh.make_opsetid("", 21), oh.make_opsetid("ai.rt", 1)],
    )
    if bits == 3:
        declaration = model.struct_types.add()
        declaration.CopyFrom(encoded.struct_type)
        declaration.type_id = 1
        encoded.struct_type.Clear()
        encoded.struct_type.type_ref = 1
        model.graph.value_info[0].type.struct_type.CopyFrom(encoded.struct_type)
    model.graph.extended_initializer.append(
        ExtendedInitializerProto(encoded_value=encoded)
    )
    checker.check_model(model)
    save_model(model, Path(__file__).with_name(f"quantized-{bits}bit.onnx"))
