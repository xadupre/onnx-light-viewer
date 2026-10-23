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

function_branches = [
    helper.make_graph(
        [helper.make_node(op_type, ["x"], ["branch_value"])],
        f"{op_type} branch",
        [],
        [helper.make_tensor_value_info("branch_value", TensorProto.FLOAT, None)],
    )
    for op_type in ["Relu", "Identity"]
]
local_functions = [
    helper.make_function(
        "example",
        "Transform",
        ["x", "cond"],
        ["y"],
        [
            helper.make_node(
                "If",
                ["cond"],
                ["hidden"],
                then_branch=function_branches[0],
                else_branch=function_branches[1],
            ),
            helper.make_node("Softmax", ["hidden"], ["normalized"]),
            helper.make_node(
                "Identity", ["normalized"], ["y"], domain="example", overload="fast"
            ),
        ],
        [helper.make_opsetid("", 11), helper.make_opsetid("example", 1)],
        value_info=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, None),
            helper.make_tensor_value_info("cond", TensorProto.BOOL, []),
        ],
    ),
    helper.make_function(
        "example",
        "Identity",
        ["x"],
        ["y"],
        [helper.make_node("Neg", ["x"], ["y"])],
        [helper.make_opsetid("", 11)],
    ),
    helper.make_function(
        "example",
        "Identity",
        ["x"],
        ["y"],
        [helper.make_node("Sigmoid", ["x"], ["y"])],
        [helper.make_opsetid("", 11)],
        overload="fast",
    ),
]
function_model = helper.make_model(
    helper.make_graph(
        [
            helper.make_node("Transform", ["x", "cond"], ["first"], domain="example"),
            helper.make_node("Identity", ["first"], ["y"], domain="example"),
        ],
        "Local functions",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3]),
            helper.make_tensor_value_info("cond", TensorProto.BOOL, []),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])],
        initializer=[helper.make_tensor("x", TensorProto.FLOAT, [1, 3], [1, 2, 3])],
    ),
    functions=local_functions,
    ir_version=10,
    opset_imports=[helper.make_opsetid("example", 1)],
)
checker.check_model(function_model)
save_model(function_model, Path(__file__).with_name("functions.onnx"))

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

invalid_graph = helper.make_model(
    helper.make_graph(
        [helper.make_node("Relu", ["missing"], ["y"])],
        "Invalid graph",
        [value_info("x")],
        [value_info("y")],
    ),
    ir_version=9,
    opset_imports=[helper.make_opsetid("", 13)],
)
save_model(invalid_graph, Path(__file__).with_name("invalid-graph.onnx"))

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


def external_tensor(name, dtype, dimensions, location):
    tensor = TensorProto()
    tensor.name = name
    tensor.data_type = dtype
    tensor.dims.extend(dimensions)
    tensor.data_location = TensorProto.EXTERNAL
    entry = tensor.external_data.add()
    entry.key = "location"
    entry.value = location
    return tensor


initializer_metadata = [
    helper.make_tensor(f"weight_{index}", TensorProto.FLOAT, [1], [index + 0.25])
    for index in range(10)
]
initializer_metadata.append(
    external_tensor("external_weight", TensorProto.FLOAT, [1], "weights/decoder.bin")
)
external_sparse = helper.make_sparse_tensor(
    external_tensor("external_sparse", TensorProto.FLOAT, [1], "weights/values.bin"),
    external_tensor("indices", TensorProto.INT64, [1], "weights/indices.bin"),
    [1],
)
initializer_model = helper.make_model(
    helper.make_graph(
        [
            helper.make_node(
                "Inspect",
                ["x", *[tensor.name for tensor in initializer_metadata], "external_sparse"],
                ["y"],
                domain="example",
            )
        ],
        "Initializer metadata",
        [value_info("x")],
        [value_info("y")],
        initializer=initializer_metadata,
        sparse_initializer=[external_sparse],
    ),
    ir_version=9,
    opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid("example", 1)],
)
# External files are intentionally absent: viewing metadata must not open them.
save_model(initializer_model, Path(__file__).with_name("initializers.onnx"))


def raw_initializer(name, dtype, shape, data):
    tensor = TensorProto()
    tensor.name = name
    tensor.data_type = dtype
    tensor.dims.extend(shape)
    tensor.raw_data = data
    return tensor


tiny_initializers = [
    helper.make_tensor("typed_floats", TensorProto.FLOAT, [3], [0.5, -2.0, 3.0]),
    helper.make_tensor("precision_float", TensorProto.FLOAT, [1], [0.1]),
    helper.make_tensor("precision_double", TensorProto.DOUBLE, [1], [0.1]),
    raw_initializer("precision_complex64", TensorProto.COMPLEX64, [1], pack("<ff", 0.1, -0.2)),
    raw_initializer("precision_complex128", TensorProto.COMPLEX128, [1], pack("<dd", 0.1, -0.2)),
    raw_initializer("raw_fifteen", TensorProto.FLOAT, [3, 5], pack("<15f", *range(15))),
    raw_initializer("sixteen", TensorProto.FLOAT, [16], pack("<16f", *range(16))),
    raw_initializer("raw_int64", TensorProto.INT64, [2], pack("<qq", 2**60 + 1, -(2**60) - 3)),
    raw_initializer("int64_fifteen", TensorProto.INT64, [15], pack("<15q", *range(15))),
    helper.make_tensor("uint64_max", TensorProto.UINT64, [1], [2**64 - 1]),
    helper.make_tensor("scalar", TensorProto.DOUBLE, [], [0.125]),
    raw_initializer("booleans", TensorProto.BOOL, [2], bytes([0, 1])),
    helper.make_tensor(
        "strings", TensorProto.STRING, [2], ["<img src=x onerror=alert(1)>", "hello"]
    ),
    helper.make_tensor("long_string", TensorProto.STRING, [1], ["x" * 4096]),
    helper.make_tensor("empty", TensorProto.FLOAT, [0], []),
    raw_initializer("float16", TensorProto.FLOAT16, [2], pack("<HH", 0x3C00, 0xC000)),
    raw_initializer("bfloat16", TensorProto.BFLOAT16, [2], pack("<HH", 0x3F80, 0xC000)),
    raw_initializer("uint4", TensorProto.UINT4, [3], bytes([0xF1, 0x02])),
    raw_initializer("int4", TensorProto.INT4, [3], bytes([0xF8, 0x07])),
    raw_initializer("uint2", TensorProto.UINT2, [5], bytes([0xE4, 0x03])),
    raw_initializer("int2", TensorProto.INT2, [5], bytes([0xE4, 0x03])),
    raw_initializer("float4", TensorProto.FLOAT4E2M1, [2], bytes([0xA2])),
    raw_initializer("float8", TensorProto.FLOAT8E4M3FN, [2], bytes([0x38, 0xC0])),
    raw_initializer(
        "float6", TensorProto.FLOAT6E2M3, [3], (8 | (16 << 6) | (20 << 12)).to_bytes(3, "little")
    ),
    raw_initializer(
        "complex_fifteen", TensorProto.COMPLEX128, [15], pack("<30d", *([0.5, -1.0] * 15))
    ),
    external_tensor("external", TensorProto.FLOAT, [1], "missing/tiny.bin"),
]
tiny_model = helper.make_model(
    helper.make_graph(
        [
            helper.make_node(
                "Inspect",
                [tensor.name for tensor in tiny_initializers]
                + ["tiny_sparse", "large_sparse", "wide_sparse_indices", "sparse_sixteen"],
                ["y"],
                domain="example",
            )
        ],
        "Tiny initializer values",
        [],
        [value_info("y")],
        initializer=tiny_initializers,
        sparse_initializer=[
            helper.make_sparse_tensor(
                helper.make_tensor("tiny_sparse", TensorProto.FLOAT, [2], [0.5, 2.0]),
                helper.make_tensor("indices", TensorProto.INT64, [2], [0, 3]),
                [4],
            ),
            helper.make_sparse_tensor(
                helper.make_tensor("large_sparse", TensorProto.FLOAT, [2], [0.5, 2.0]),
                helper.make_tensor("indices", TensorProto.INT64, [2], [0, 999999]),
                [1000000],
            ),
            helper.make_sparse_tensor(
                helper.make_tensor("wide_sparse_indices", TensorProto.FLOAT, [1], [1.0]),
                helper.make_tensor("indices", TensorProto.INT64, [1, 16], [0] * 16),
                [1] * 16,
            ),
            helper.make_sparse_tensor(
                helper.make_tensor("sparse_sixteen", TensorProto.FLOAT, [16], [1.0] * 16),
                helper.make_tensor("indices", TensorProto.INT64, [16], list(range(16))),
                [1000000],
            ),
        ],
    ),
    ir_version=9,
    opset_imports=[helper.make_opsetid("example", 1)],
)
save_model(tiny_model, Path(__file__).with_name("tiny-initializers.onnx"))

shared_name = "shared;activation"
metadata_model = helper.make_model(
    helper.make_graph(
        [
            helper.make_node(
                "Constant",
                [],
                ["constant"],
                name="constant_source",
                value=helper.make_tensor("value", TensorProto.FLOAT, [1], [2.0]),
            ),
            helper.make_node(
                "Add", ["constant", "bias"], ["constant_sum"], name="constant_add"
            ),
            helper.make_node(
                "Identity", ["input"], [shared_name], name="shared_source"
            ),
            helper.make_node("Shape", [shared_name], ["dimensions"], name="shape"),
            helper.make_node(
                "Identity", [shared_name], ["unused"], name="unused_result"
            ),
            helper.make_node(
                "Add", [shared_name, "constant_sum"], ["output"], name="last_consumer"
            ),
        ],
        "Metadata and result lifetimes",
        [value_info("input")],
        [value_info("output")],
        initializer=[
            external_tensor(
                "bias", TensorProto.FLOAT, [1], "missing/metadata-weights.bin"
            )
        ],
        value_info=[value_info(shared_name)],
    ),
    ir_version=9,
    opset_imports=[helper.make_opsetid("", 13)],
)
for proto, key, value in [
    (metadata_model, "purpose", "metadata <inspection>"),
    (metadata_model.graph, "graph_note", "Keep all entries"),
    (metadata_model.graph.node[5], "node_note", "<script>not executable</script>"),
    (metadata_model.graph.value_info[0], "tensor_note", "A result used by three nodes"),
    (metadata_model.graph.initializer[0], "weight_note", "External payload is absent"),
]:
    entry = proto.metadata_props.add()
    entry.key = key
    entry.value = value
save_model(metadata_model, Path(__file__).with_name("analysis.onnx"))
metadata_model.graph.output.append(value_info(shared_name))
save_model(metadata_model, Path(__file__).with_name("analysis-kept-output.onnx"))

branch_output = helper.make_tensor_value_info(
    "branch_output", TensorProto.FLOAT, ["N", 3]
)
inner_branch = helper.make_graph(
    [helper.make_node("Identity", ["sum"], ["branch_output"])],
    "inner <branch>",
    [],
    [branch_output],
)
then_branch = helper.make_graph(
    [
        helper.make_node("Add", ["x", "bias"], ["sum"]),
        helper.make_node(
            "If",
            ["condition"],
            ["branch_output"],
            then_branch=inner_branch,
            else_branch=inner_branch,
        ),
    ],
    "branch <graph>",
    [],
    [branch_output],
)
else_branch = helper.make_graph(
    [helper.make_node("Identity", ["x"], ["branch_output"])],
    "branch <graph>",
    [],
    [branch_output],
)
empty_branch = helper.make_graph(
    [],
    "empty branch",
    [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 3])],
    [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 3])],
)
shadow_branch = helper.make_graph(
    [helper.make_node("Add", ["x", "bias"], ["shadow_output"])],
    "local scope",
    [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
    [helper.make_tensor_value_info("shadow_output", TensorProto.FLOAT, [2])],
    initializer=[helper.make_tensor("bias", TensorProto.FLOAT, [2], [0.0, 0.0])],
)
subgraphs_model = helper.make_model(
    helper.make_graph(
        [
            helper.make_node(
                "If",
                ["condition"],
                ["y"],
                name="choose_branch",
                then_branch=then_branch,
                else_branch=else_branch,
            ),
            helper.make_node(
                "GraphList",
                ["y"],
                ["z"],
                domain="example",
                branches=[empty_branch, shadow_branch],
            ),
        ],
        "Subgraph navigation",
        [
            helper.make_tensor_value_info("condition", TensorProto.BOOL, []),
            helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 3]),
        ],
        [helper.make_tensor_value_info("z", TensorProto.FLOAT, ["N", 3])],
        initializer=[
            helper.make_tensor("bias", TensorProto.FLOAT, [3], [0.0, 0.0, 0.0])
        ],
    ),
    ir_version=9,
    opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid("example", 1)],
)
checker.check_model(subgraphs_model)
save_model(subgraphs_model, Path(__file__).with_name("subgraphs.onnx"))
if_model = helper.make_model(
    helper.make_graph(
        [subgraphs_model.graph.node[0]],
        "Nested If",
        list(subgraphs_model.graph.input),
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N", 3])],
        initializer=list(subgraphs_model.graph.initializer),
    ),
    ir_version=9,
    opset_imports=[helper.make_opsetid("", 13)],
)
checker.check_model(if_model)
save_model(if_model, Path(__file__).with_name("nested-if.onnx"))
