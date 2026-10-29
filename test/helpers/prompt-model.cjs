const { execFileSync } = require("node:child_process");
const { writeFile, mkdir } = require("node:fs/promises");
const { join } = require("node:path");

function varint(value) {
  const bytes = [];
  do {
    let byte = value & 127;
    value >>>= 7;
    if (value) byte |= 128;
    bytes.push(byte);
  } while (value);
  return Buffer.from(bytes);
}
const integer = (field, value) =>
  Buffer.concat([varint(field * 8), varint(value)]);
const bytes = (field, value) => {
  const data = Buffer.isBuffer(value) ? value : Buffer.from(value);
  return Buffer.concat([varint(field * 8 + 2), varint(data.length), data]);
};
const port = (name, dtype, dimensions) =>
  Buffer.concat([
    bytes(1, name),
    bytes(
      2,
      bytes(
        1,
        Buffer.concat([
          integer(1, dtype),
          bytes(
            2,
            Buffer.concat(
              dimensions.map((dim) =>
                bytes(
                  1,
                  typeof dim === "string" ? bytes(2, dim) : integer(1, dim),
                ),
              ),
            ),
          ),
        ]),
      ),
    ),
  ]);

async function promptModel(
  directory,
  external = false,
  externalLength = false,
) {
  const weights = Buffer.alloc(4 * 4 * 4);
  for (let index = 0; index < 4; ++index)
    weights.writeFloatLE(4, (index * 4 + 1) * 4);
  const model = Buffer.concat([
    integer(1, 9),
    bytes(8, integer(2, 18)),
    bytes(
      7,
      Buffer.concat([
        bytes(
          1,
          Buffer.concat([
            bytes(1, "weights"),
            bytes(1, "input_ids"),
            bytes(2, "logits"),
            bytes(4, "Gather"),
          ]),
        ),
        bytes(2, "Tiny causal generation fixture"),
        bytes(
          5,
          Buffer.concat([
            integer(1, 4),
            integer(1, 4),
            integer(2, 1),
            bytes(8, "weights"),
            ...(external
              ? [
                  integer(14, 1),
                  bytes(
                    13,
                    Buffer.concat([
                      bytes(1, "location"),
                      bytes(2, "weights/decoder.bin"),
                    ]),
                  ),
                  ...(externalLength
                    ? [
                        bytes(
                          13,
                          Buffer.concat([
                            bytes(1, "length"),
                            bytes(2, String(weights.length)),
                          ]),
                        ),
                      ]
                    : []),
                ]
              : [bytes(9, weights)]),
          ]),
        ),
        bytes(11, port("input_ids", 7, ["batch", "sequence"])),
        bytes(12, port("logits", 1, ["batch", "sequence", 4])),
      ]),
    ),
  ]);
  if (external) {
    await mkdir(join(directory, "weights"));
    await writeFile(join(directory, "weights/decoder.bin"), weights);
  }
  await writeFile(join(directory, "model.onnx"), model);
  await writeFile(join(directory, "model.pipe-onnx"), bytes(8, model));
  await writeFile(
    join(directory, "config.json"),
    JSON.stringify({
      architectures: ["GPT2LMHeadModel"],
      eos_token_id: 2,
    }),
  );
  await writeFile(
    join(directory, "tokenizer.json"),
    JSON.stringify({
      version: "1.0",
      truncation: null,
      padding: null,
      added_tokens: [],
      normalizer: null,
      pre_tokenizer: { type: "Whitespace" },
      post_processor: null,
      decoder: null,
      model: {
        type: "BPE",
        vocab: { "<unk>": 0, o: 1, X: 2, H: 3 },
        unk_token: "<unk>",
        merges: [],
      },
    }),
  );
  await writeFile(join(directory, "tokenizer_config.json"), "{}");
  await writeFile(
    join(directory, "genai_config.json"),
    JSON.stringify({
      model: {
        type: "llama",
        vocab_size: 4,
        context_length: 8192,
        bos_token_id: 0,
        eos_token_id: 2,
        pad_token_id: 0,
        decoder: {
          session_options: { intra_op_num_threads: 1 },
          filename: "model.onnx",
          head_size: 4,
          hidden_size: 4,
          num_attention_heads: 1,
          num_key_value_heads: 1,
          num_hidden_layers: 0,
          inputs: { input_ids: "input_ids" },
          outputs: { logits: "logits" },
        },
      },
      search: { max_length: 128, do_sample: false },
    }),
  );
  return join(directory, "model.onnx");
}

function nativeEnvironment() {
  const python = process.platform === "win32" ? "python" : "python3";
  const environment = JSON.parse(
    execFileSync(
      python,
      [
        "-c",
        `import importlib.util, json, pathlib, sys
genai = pathlib.Path(importlib.util.find_spec("onnxruntime_genai").origin).parent
ort = pathlib.Path(importlib.util.find_spec("onnxruntime").origin).parent / "capi"
library = "onnxruntime-genai.dll" if sys.platform == "win32" else "libonnxruntime-genai.dylib" if sys.platform == "darwin" else "libonnxruntime-genai.so"
runtime = next(ort.glob("onnxruntime.dll" if sys.platform == "win32" else "libonnxruntime*.dylib" if sys.platform == "darwin" else "libonnxruntime.so*"))
print(json.dumps({"ONNX_LIGHT_VIEWER_GENAI_LIBRARY": str(genai / library), "ORT_LIB_PATH": str(runtime)}))
`,
      ],
      { encoding: "utf8" },
    ),
  );
  return {
    ...environment,
    ONNX_LIGHT_VIEWER_GENAI_LIBRARY:
      process.env.ONNX_LIGHT_VIEWER_GENAI_LIBRARY ??
      environment.ONNX_LIGHT_VIEWER_GENAI_LIBRARY,
    ORT_LIB_PATH: process.env.ORT_LIB_PATH ?? environment.ORT_LIB_PATH,
  };
}

module.exports = { promptModel, nativeEnvironment };
