#include "addon.h"
#include "model_io.h"
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <onnx_core/runtime/generation.h>
#include <onnx_proto/onnx_helper.h>
#include <ort_genai_c.h>
#include <stdexcept>
#include <uv.h>

namespace onnx_light_viewer {
namespace {
namespace rt = ol::core::runtime;
using Clock = std::chrono::steady_clock;

// Each runner process owns one SDK instance; no GenAI objects cross threads.
class Genai {
public:
  Genai(const std::string &path, const std::string &runtime_path) {
    if (!runtime_path.empty() && uv_dlopen(runtime_path.c_str(), &runtime_) != 0) {
      const std::string error = uv_dlerror(&runtime_);
      uv_dlclose(&runtime_);
      throw std::runtime_error("Could not load ORT_LIB_PATH: " + error);
    }
    if (uv_dlopen(path.c_str(), &library_) != 0) {
      const std::string error = uv_dlerror(&library_);
      uv_dlclose(&library_);
      if (runtime_.handle)
        uv_dlclose(&runtime_);
      throw std::runtime_error(
          "Could not load the native ONNX Runtime GenAI library: " + error +
          ". Set ONNX_LIGHT_VIEWER_GENAI_LIBRARY to the C++ SDK shared library's absolute path.");
    }
    try {
#define LOAD(name) Load(#name, name)
      LOAD(OgaResultGetError);
      LOAD(OgaDestroyResult);
      LOAD(OgaCreateModel);
      LOAD(OgaDestroyModel);
      LOAD(OgaCreateTokenizerFromPath);
      LOAD(OgaDestroyTokenizer);
      LOAD(OgaCreateSequences);
      LOAD(OgaDestroySequences);
      LOAD(OgaTokenizerEncode);
      LOAD(OgaSequencesGetSequenceCount);
      LOAD(OgaSequencesGetSequenceData);
      LOAD(OgaCreateTokenizerStream);
      LOAD(OgaDestroyTokenizerStream);
      LOAD(OgaTokenizerStreamDecode);
      LOAD(OgaCreateGeneratorParams);
      LOAD(OgaDestroyGeneratorParams);
      LOAD(OgaGeneratorParamsSetSearchNumber);
      LOAD(OgaGeneratorParamsSetSearchBool);
      LOAD(OgaCreateGenerator);
      LOAD(OgaDestroyGenerator);
      LOAD(OgaGenerator_AppendTokens);
      LOAD(OgaGenerator_IsDone);
      LOAD(OgaGenerator_GetLogits);
      LOAD(OgaGenerator_GenerateNextToken);
      LOAD(OgaGenerator_GetNextTokens);
      LOAD(OgaDestroyTensor);
      LOAD(OgaTensorGetType);
      LOAD(OgaTensorGetShapeRank);
      LOAD(OgaTensorGetShape);
      LOAD(OgaTensorGetData);
      LOAD(OgaShutdown);
#undef LOAD
    } catch (...) {
      uv_dlclose(&library_);
      if (runtime_.handle)
        uv_dlclose(&runtime_);
      throw;
    }
  }
  ~Genai() {
    OgaShutdown();
    uv_dlclose(&library_);
    if (runtime_.handle)
      uv_dlclose(&runtime_);
  }

  void Check(OgaResult *result) {
    if (!result)
      return;
    const std::string error = OgaResultGetError(result);
    OgaDestroyResult(result);
    throw std::runtime_error("ONNX Runtime GenAI: " + error);
  }

  template <typename T> void Load(const char *name, T &function) {
    void *symbol = nullptr;
    if (uv_dlsym(&library_, name, &symbol) != 0)
      throw std::runtime_error("Native GenAI SDK does not expose " + std::string(name) + ": " +
                               uv_dlerror(&library_));
    static_assert(sizeof(function) == sizeof(symbol));
    std::memcpy(&function, &symbol, sizeof(function));
  }

#define API(name) decltype(&::name) name = nullptr
  API(OgaResultGetError);
  API(OgaDestroyResult);
  API(OgaCreateModel);
  API(OgaDestroyModel);
  API(OgaCreateTokenizerFromPath);
  API(OgaDestroyTokenizer);
  API(OgaCreateSequences);
  API(OgaDestroySequences);
  API(OgaTokenizerEncode);
  API(OgaSequencesGetSequenceCount);
  API(OgaSequencesGetSequenceData);
  API(OgaCreateTokenizerStream);
  API(OgaDestroyTokenizerStream);
  API(OgaTokenizerStreamDecode);
  API(OgaCreateGeneratorParams);
  API(OgaDestroyGeneratorParams);
  API(OgaGeneratorParamsSetSearchNumber);
  API(OgaGeneratorParamsSetSearchBool);
  API(OgaCreateGenerator);
  API(OgaDestroyGenerator);
  API(OgaGenerator_AppendTokens);
  API(OgaGenerator_IsDone);
  API(OgaGenerator_GetLogits);
  API(OgaGenerator_GenerateNextToken);
  API(OgaGenerator_GetNextTokens);
  API(OgaDestroyTensor);
  API(OgaTensorGetType);
  API(OgaTensorGetShapeRank);
  API(OgaTensorGetShape);
  API(OgaTensorGetData);
  API(OgaShutdown);
#undef API

private:
  uv_lib_t library_{};
  uv_lib_t runtime_{};
};

template <typename T, typename D> auto Own(T *value, D destroy) {
  return std::unique_ptr<T, D>(value, destroy);
}

int64_t Integer(const Napi::Value &value, int64_t minimum, int64_t maximum) {
  if (!value.IsNumber())
    throw std::runtime_error("Expected an integer prompt argument.");
  const double number = value.As<Napi::Number>().DoubleValue();
  if (!std::isfinite(number) || std::floor(number) != number || number < minimum ||
      number > maximum)
    throw std::runtime_error("Prompt argument is outside its supported integer range.");
  return static_cast<int64_t>(number);
}

std::vector<int64_t> LightTokens(const std::string &path, const std::vector<int64_t> &prompt,
                                 int64_t maximum, std::optional<int64_t> eos) {
  FileDescriptor file(OpenModelFile(path));
  if (file.get() < 0)
    throw std::runtime_error("Could not open the prompt model: " + path);
  ol::utils::FdReadStream stream(file.get());
  const auto directory = ModelFilePath(path).parent_path().string();
  rt::RuntimeContext context;
  rt::GenerationOptions options;
  options.max_new_tokens = maximum;
  options.temperature = 0;
  options.eos_token_id = eos;
  rt::Tensor generated;
  auto extension = ModelFilePath(path).extension().string();
  std::transform(extension.begin(), extension.end(), extension.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  if (extension == ".pipe-onnx") {
    ol::PipelineProto pipeline;
    if (!pipeline.ParseFromStream(stream) || !pipeline.has_prefill())
      throw std::runtime_error("Could not decode the prompt PipelineProto.");
    for (auto *model :
         {pipeline.has_embedding() ? &pipeline.ref_embedding() : nullptr, &pipeline.ref_prefill(),
          pipeline.has_decode() ? &pipeline.ref_decode() : nullptr})
      if (model)
        ol::LoadExternalDataForModel(*model, directory);
    pipeline.set_max_new_tokens(maximum);
    pipeline.set_temperature(0);
    if (eos)
      pipeline.set_eos_token_id(*eos);
    const std::string input =
        pipeline.has_input_ids_name() ? std::string(pipeline.input_ids_name()) : "input_ids";
    rt::RuntimeValueMap feeds;
    feeds.emplace(input,
                  rt::Tensor::FromInt64(input, {1, static_cast<int64_t>(prompt.size())}, prompt));
    generated = rt::Generate(pipeline, context, feeds);
  } else {
    ol::ModelProto model;
    ReadModel(model, stream, {});
    ol::LoadExternalDataForModel(model, directory);
    rt::RuntimeValueMap feeds;
    feeds.emplace("input_ids", rt::Tensor::FromInt64(
                                   "input_ids", {1, static_cast<int64_t>(prompt.size())}, prompt));
    generated = rt::Generate(model, context, feeds, options);
  }
  if (generated.data_type != ol::TensorProto::INT64 || generated.shape.size() != 2 ||
      generated.shape[0] != 1 || generated.shape[1] < static_cast<int64_t>(prompt.size()) ||
      generated.shape[1] - static_cast<int64_t>(prompt.size()) > maximum)
    throw std::runtime_error("Native Generate returned an invalid token sequence.");
  std::vector<int64_t> tokens(static_cast<size_t>(generated.shape[1]));
  std::memcpy(tokens.data(), generated.data.data(), tokens.size() * sizeof(int64_t));
  if (!std::equal(prompt.begin(), prompt.end(), tokens.begin()))
    throw std::runtime_error("Native Generate changed the prompt prefix.");
  tokens.erase(tokens.begin(), tokens.begin() + static_cast<std::ptrdiff_t>(prompt.size()));
  return tokens;
}

Napi::Array Ids(Napi::Env env, const auto &ids) {
  auto array = Napi::Array::New(env, ids.size());
  for (uint32_t index = 0; index < ids.size(); ++index)
    array.Set(index, Napi::Number::New(env, ids[index]));
  return array;
}
} // namespace

Napi::Value RunNativePrompt(const Napi::CallbackInfo &info) {
  const auto env = info.Env();
  try {
    if (info.Length() != 2 || !info[0].IsObject() || !info[1].IsFunction())
      throw std::runtime_error("Expected native prompt options and a token callback.");
    const auto request = info[0].As<Napi::Object>();
    for (const char *key :
         {"engine", "modelPath", "source", "libraryPath", "runtimePath", "prompt"})
      if (!request.Get(key).IsString() ||
          request.Get(key).As<Napi::String>().Utf8Value().find('\0') != std::string::npos)
        throw std::runtime_error("Expected valid native prompt strings.");
    const std::string engine = request.Get("engine").As<Napi::String>();
    if (engine != "onnx-light" && engine != "onnxruntime-genai")
      throw std::runtime_error("Unknown native prompt engine.");
    const auto maximum = Integer(request.Get("maxTokens"), 1, 4096);
    const auto budget = Integer(request.Get("captureBudget"), 0, 256 * 1024 * 1024);
    Genai api(request.Get("libraryPath").As<Napi::String>(),
              request.Get("runtimePath").As<Napi::String>());
    OgaTokenizer *raw_tokenizer = nullptr;
    api.Check(api.OgaCreateTokenizerFromPath(
        request.Get("source").As<Napi::String>().Utf8Value().c_str(), &raw_tokenizer));
    auto tokenizer = Own(raw_tokenizer, api.OgaDestroyTokenizer);
    OgaSequences *raw_sequences = nullptr;
    api.Check(api.OgaCreateSequences(&raw_sequences));
    auto sequences = Own(raw_sequences, api.OgaDestroySequences);
    api.Check(api.OgaTokenizerEncode(tokenizer.get(),
                                     request.Get("prompt").As<Napi::String>().Utf8Value().c_str(),
                                     sequences.get()));
    const auto count = api.OgaSequencesGetSequenceCount(sequences.get(), 0);
    if (!count || count > 1'000'000)
      throw std::runtime_error("The tokenizer produced an empty or oversized prompt.");
    const auto *data = api.OgaSequencesGetSequenceData(sequences.get(), 0);
    const std::vector<int32_t> prompt(data, data + count);
    OgaTokenizerStream *raw_stream = nullptr;
    api.Check(api.OgaCreateTokenizerStream(tokenizer.get(), &raw_stream));
    auto text_stream = Own(raw_stream, api.OgaDestroyTokenizerStream);
    auto result = Napi::Object::New(env);
    result.Set("promptIds", Ids(env, prompt));
    auto tokens = Napi::Array::New(env);
    auto logits = Napi::Array::New(env);
    bool capture = true;
    size_t captured = 0;
    const auto append = [&](int64_t id, const float *row, size_t vocabulary,
                            std::optional<double> elapsed) {
      if (id < 0 || id > std::numeric_limits<int32_t>::max() ||
          (row && static_cast<size_t>(id) >= vocabulary))
        throw std::runtime_error("The native engine returned an out-of-vocabulary token.");
      if (row && !std::isfinite(row[id]))
        throw std::runtime_error("The native engine returned a non-finite token score.");
      const char *text = nullptr;
      api.Check(api.OgaTokenizerStreamDecode(text_stream.get(), static_cast<int32_t>(id), &text));
      auto token = Napi::Object::New(env);
      token.Set("text", text);
      token.Set("id", Napi::Number::New(env, id));
      token.Set("score", row ? Napi::Value(Napi::Number::New(env, row[id])) : env.Null());
      token.Set("timeMs", elapsed ? Napi::Value(Napi::Number::New(env, *elapsed)) : env.Null());
      if (row && capture) {
        // Include the typed capture, its JSON conversion and worst-case numeric text.
        if (vocabulary > (static_cast<size_t>(budget) - captured) / 40) {
          capture = false;
          logits = Napi::Array::New(env);
        } else {
          captured += vocabulary * 40;
          if (!std::all_of(row, row + vocabulary, [](float value) { return std::isfinite(value); }))
            throw std::runtime_error("The native engine returned non-finite vocabulary logits.");
          auto values = Napi::Float32Array::New(env, vocabulary);
          std::memcpy(values.Data(), row, vocabulary * sizeof(float));
          logits.Set(tokens.Length(), values);
        }
      }
      const uint32_t index = tokens.Length();
      tokens.Set(index, token);
      info[1].As<Napi::Function>().Call({token, Napi::Number::New(env, index)});
    };
    if (engine == "onnx-light") {
      const auto eos = request.Get("eosTokenId");
      for (auto id :
           LightTokens(request.Get("modelPath").As<Napi::String>(),
                       std::vector<int64_t>(prompt.begin(), prompt.end()), maximum,
                       eos.IsUndefined() || eos.IsNull()
                           ? std::nullopt
                           : std::optional(Integer(eos, 0, std::numeric_limits<int32_t>::max()))))
        append(id, nullptr, 0, std::nullopt);
      result.Set("logits", env.Null());
    } else {
      OgaModel *raw_model = nullptr;
      api.Check(api.OgaCreateModel(
          ModelFilePath(request.Get("modelPath").As<Napi::String>()).parent_path().string().c_str(),
          &raw_model));
      auto model = Own(raw_model, api.OgaDestroyModel);
      OgaGeneratorParams *raw_params = nullptr;
      api.Check(api.OgaCreateGeneratorParams(model.get(), &raw_params));
      auto params = Own(raw_params, api.OgaDestroyGeneratorParams);
      api.Check(api.OgaGeneratorParamsSetSearchNumber(params.get(), "max_length", count + maximum));
      api.Check(api.OgaGeneratorParamsSetSearchNumber(params.get(), "num_beams", 1));
      api.Check(api.OgaGeneratorParamsSetSearchNumber(params.get(), "num_return_sequences", 1));
      api.Check(api.OgaGeneratorParamsSetSearchBool(params.get(), "do_sample", false));
      OgaGenerator *raw_generator = nullptr;
      api.Check(api.OgaCreateGenerator(model.get(), params.get(), &raw_generator));
      auto generator = Own(raw_generator, api.OgaDestroyGenerator);
      auto start = Clock::now();
      api.Check(api.OgaGenerator_AppendTokens(generator.get(), prompt.data(), prompt.size()));
      while (tokens.Length() < maximum && !api.OgaGenerator_IsDone(generator.get())) {
        OgaTensor *raw_output = nullptr;
        api.Check(api.OgaGenerator_GetLogits(generator.get(), &raw_output));
        auto output = Own(raw_output, api.OgaDestroyTensor);
        OgaElementType type;
        api.Check(api.OgaTensorGetType(output.get(), &type));
        size_t rank;
        api.Check(api.OgaTensorGetShapeRank(output.get(), &rank));
        if (type != OgaElementType_float32 || rank < 1 || rank > 3)
          throw std::runtime_error("GenAI must return float32 last-position logits.");
        std::vector<int64_t> shape(rank);
        api.Check(api.OgaTensorGetShape(output.get(), shape.data(), rank));
        for (size_t axis = 0; axis + 1 < rank; ++axis) {
          if (shape[axis] != 1)
            throw std::runtime_error("Expected one GenAI last-position logit row.");
        }
        if (shape.back() <= 0 || shape.back() > std::numeric_limits<uint32_t>::max())
          throw std::runtime_error("GenAI returned an invalid vocabulary size.");
        void *values = nullptr;
        api.Check(api.OgaTensorGetData(output.get(), &values));
        // GetLogits owns a CPU copy, which remains valid after search mutates its own buffer.
        api.Check(api.OgaGenerator_GenerateNextToken(generator.get()));
        const int32_t *next = nullptr;
        size_t next_count = 0;
        api.Check(api.OgaGenerator_GetNextTokens(generator.get(), &next, &next_count));
        const double elapsed =
            std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        if (next_count != 1)
          throw std::runtime_error("Expected one generated token.");
        append(next[0], static_cast<const float *>(values), static_cast<size_t>(shape.back()),
               elapsed);
        start = Clock::now();
      }
      result.Set("logits", capture ? Napi::Value(logits) : env.Null());
    }
    result.Set("tokens", tokens);
    return result;
  } catch (const std::exception &error) {
    Napi::Error::New(env, error.what()).ThrowAsJavaScriptException();
    return env.Undefined();
  }
}
} // namespace onnx_light_viewer
