/** Generate a dependency-free ONNX playground for global passes and optimization. */
const { writeFileSync } = require("node:fs");
const path = require("node:path");

function varint(value) {
  const bytes = [];
  do {
    bytes.push((value % 128) | (value >= 128 ? 128 : 0));
    value = Math.floor(value / 128);
  } while (value);
  return Buffer.from(bytes);
}

const integer = (number, value) =>
  Buffer.concat([varint(number * 8), varint(value)]);
const bytes = (number, value) => {
  const data = Buffer.from(value);
  return Buffer.concat([varint(number * 8 + 2), varint(data.length), data]);
};
const fields = (...values) => Buffer.concat(values.flat());
const features = (name) =>
  fields(
    bytes(1, name),
    bytes(
      2,
      bytes(
        1,
        fields(
          integer(1, 1),
          bytes(2, fields(bytes(1, integer(1, 2)), bytes(1, integer(1, 2)))),
        ),
      ),
    ),
  );
function tensor(name, dims, values) {
  const data = Buffer.alloc(values.length * 4);
  values.forEach((value, index) => data.writeFloatLE(value, index * 4));
  return bytes(
    5,
    fields(
      dims.map((dim) => integer(1, dim)),
      integer(2, 1),
      bytes(8, name),
      bytes(9, data),
    ),
  );
}
function node(name, op, inputs, output, transpose = false) {
  return bytes(
    1,
    fields(
      inputs.map((input) => bytes(1, input)),
      bytes(2, output),
      bytes(3, name),
      bytes(4, op),
      transpose
        ? bytes(
            5,
            fields(
              bytes(1, "perm"),
              integer(8, 1),
              integer(8, 0),
              integer(20, 7),
            ),
          )
        : Buffer.alloc(0),
    ),
  );
}
const model = fields(
  integer(1, 10),
  bytes(2, "onnx-light-viewer"),
  bytes(8, integer(2, 18)),
  bytes(
    7,
    fields(
      node("copy_input", "Identity", ["x"], "copied"),
      node("constant_sum", "Add", ["weight.left", "weight.right"], "weight"),
      node("transpose", "Transpose", ["copied"], "xt", true),
      node("inverse", "Transpose", ["xt"], "restored", true),
      node("matmul", "MatMul", ["restored", "weight"], "product"),
      node("add_bias", "Add", ["product", "bias"], "biased"),
      node("output_transpose", "Transpose", ["biased"], "y", true),
      node("unused_relu", "Relu", ["x"], "unused"),
      bytes(2, "Optimization playground"),
      tensor("weight.left", [2, 2], [1, 0, 0, 1]),
      tensor("weight.right", [2, 2], [1, 0, 0, 1]),
      tensor("bias", [2], [1, 2]),
      bytes(11, features("x")),
      bytes(12, features("y")),
    ),
  ),
);
writeFileSync(path.join(__dirname, "optimization.onnx"), model);
