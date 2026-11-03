const fs = require("node:fs/promises");
const path = require("node:path");
const { promptModel, nativeEnvironment } = require("../test/helpers/prompt-model.cjs");
const { runPrompt } = require("../dist/prompt");
(async () => {
  const directory = path.join(__dirname, "model");
  await fs.mkdir(directory, { recursive: true });
  const model = await promptModel(directory);
  Object.assign(process.env, nativeEnvironment());
  const result = await runPrompt(
    model,
    {
      engine: "onnxruntime-genai", prompt: "H", maxTokens: 6, repeat: 10,
      keepLoaded: false, sessionSettings: { profile: true },
    },
    "python3", new AbortController().signal, () => {},
  );
  await fs.writeFile(path.join(__dirname, "result.json"), JSON.stringify(result));
  console.log("Real GenAI output:", result.tokens.map(t => t.text).join(""), "repeat:", result.repeat,
    "profile nodes:", result.profiling.nodes.length);
})().catch((error) => { console.error(error); process.exitCode = 1; });
