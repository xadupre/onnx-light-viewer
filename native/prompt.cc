#include "addon.h"
#include "model_io.h"
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <onnx_core/runtime/generation.h>
#include <onnx_core/runtime/kernels/cast_helper.h>
#include <onnx_core/runtime/runtime_session_loader.h>
#include <onnx_proto/onnx_helper.h>
#include <ort_genai_c.h>
#include <set>
#include <stdexcept>
#include <type_traits>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace onnx_light_viewer {
namespace {
namespace rt = ol::core::runtime;
using Clock = std::chrono::steady_clock;

class Library {
public:
  Library(const std::string &path, const char *description, const char *setting) {
    if (path.empty())
      return;
#ifdef _WIN32
    const auto file = std::filesystem::u8path(path);
    handle_ = LoadLibraryExW(file.c_str(), nullptr,
                             file.is_absolute() ? LOAD_WITH_ALTERED_SEARCH_PATH : 0);
#else
    handle_ = dlopen(path.c_str(), RTLD_LAZY);
#endif
    if (!handle_) {
      const auto error = Error();
      throw std::runtime_error("Could not load " + std::string(description) + ": " + error +
                               ". Set " + setting +
                               " to the C++ SDK shared library's absolute path.");
    }
  }
  ~Library() {
    if (!handle_)
      return;
#ifdef _WIN32
    FreeLibrary(handle_);
#else
    dlclose(handle_);
#endif
  }
  Library(const Library &) = delete;
  Library &operator=(const Library &) = delete;

  template <typename T> void Load(const char *name, T &function) const {
#ifdef _WIN32
    const auto symbol = GetProcAddress(handle_, name);
#else
    const auto symbol = dlsym(handle_, name);
#endif
    if (!symbol) {
      const auto error = Error();
      throw std::runtime_error("Native GenAI SDK does not expose " + std::string(name) + ": " +
                               error);
    }
    static_assert(sizeof(function) == sizeof(symbol));
    std::memcpy(&function, &symbol, sizeof(function));
  }

private:
  static std::string Error() {
#ifdef _WIN32
    return "Windows error " + std::to_string(GetLastError());
#else
    const auto *error = dlerror();
    return error ? error : "Unknown dynamic library error";
#endif
  }
#ifdef _WIN32
  HMODULE handle_ = nullptr;
#else
  void *handle_ = nullptr;
#endif
};

// Each runner process owns one SDK instance; no GenAI objects cross threads.
class Genai {
public:
  Genai(const std::string &path, const std::string &runtime_path)
      : runtime_(runtime_path, "ORT_LIB_PATH", "ORT_LIB_PATH"),
        library_(path, "the native ONNX Runtime GenAI library", "ONNX_LIGHT_VIEWER_GENAI_LIBRARY") {
#define LOAD(name) Load(#name, name)
    LOAD(OgaResultGetError);
    LOAD(OgaDestroyResult);
    LOAD(OgaCreateModel);
    LOAD(OgaCreateConfig);
    LOAD(OgaDestroyConfig);
    LOAD(OgaConfigOverlay);
    LOAD(OgaConfigClearProviders);
    LOAD(OgaConfigAppendProvider);
    LOAD(OgaConfigSetProviderOption);
    LOAD(OgaCreateModelFromConfig);
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
  }
  ~Genai() { OgaShutdown(); }

  void Check(OgaResult *result) {
    if (!result)
      return;
    const std::string error = OgaResultGetError(result);
    OgaDestroyResult(result);
    throw std::runtime_error("ONNX Runtime GenAI: " + error);
  }

  template <typename T> void Load(const char *name, T &function) { library_.Load(name, function); }

#define API(name) decltype(&::name) name = nullptr
  API(OgaResultGetError);
  API(OgaDestroyResult);
  API(OgaCreateModel);
  API(OgaCreateConfig);
  API(OgaDestroyConfig);
  API(OgaConfigOverlay);
  API(OgaConfigClearProviders);
  API(OgaConfigAppendProvider);
  API(OgaConfigSetProviderOption);
  API(OgaCreateModelFromConfig);
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
  Library runtime_;
  Library library_;
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

double Milliseconds(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

class PromptSession {
public:
  explicit PromptSession(const Napi::Object &request)
      : api(request.Get("libraryPath").As<Napi::String>(),
            request.Get("runtimePath").As<Napi::String>()),
        engine(request.Get("engine").As<Napi::String>()),
        key(request.Get("sessionKey").As<Napi::String>()) {
    OgaTokenizer *raw = nullptr;
    api.Check(api.OgaCreateTokenizerFromPath(
        request.Get("source").As<Napi::String>().Utf8Value().c_str(), &raw));
    tokenizer = Own(raw, api.OgaDestroyTokenizer);
    const std::string path = request.Get("modelPath").As<Napi::String>();
    const auto settings = request.Get("sessionSettings").As<Napi::Object>();
    if (engine == "onnxruntime-genai") {
      OgaConfig *raw_config = nullptr;
      api.Check(
          api.OgaCreateConfig(ModelFilePath(path).parent_path().string().c_str(), &raw_config));
      auto config = Own(raw_config, api.OgaDestroyConfig);
      api.Check(api.OgaConfigOverlay(
          config.get(), request.Get("genaiOverlay").As<Napi::String>().Utf8Value().c_str()));
      if (settings.Has("provider")) {
        const std::string provider = settings.Get("provider").As<Napi::String>();
        api.Check(api.OgaConfigClearProviders(config.get()));
        if (provider != "cpu") {
          api.Check(api.OgaConfigAppendProvider(config.get(), provider.c_str()));
          if (settings.Has("deviceId"))
            api.Check(api.OgaConfigSetProviderOption(
                config.get(), provider.c_str(), "device_id",
                std::to_string(Integer(settings.Get("deviceId"), 0, 65535)).c_str()));
          if (settings.Has("gpuMemoryMiB"))
            api.Check(api.OgaConfigSetProviderOption(
                config.get(), provider.c_str(), "gpu_mem_limit",
                std::to_string(Integer(settings.Get("gpuMemoryMiB"), 1, 1'048'576) * 1024 * 1024)
                    .c_str()));
        }
      }
      OgaModel *raw_model = nullptr;
      api.Check(api.OgaCreateModelFromConfig(config.get(), &raw_model));
      model = Own(raw_model, api.OgaDestroyModel);
    } else {
      rt::RuntimeContextOptions context_options;
      if (settings.Has("profile"))
        context_options.events_enabled = settings.Get("profile").As<Napi::Boolean>();
      if (settings.Has("verbose"))
        context_options.verbose = static_cast<int>(Integer(settings.Get("verbose"), 0, 10));
      if (settings.Has("releaseIntermediates"))
        context_options.release_intermediates =
            settings.Get("releaseIntermediates").As<Napi::Boolean>();
      context = std::make_unique<rt::RuntimeContext>(context_options);
      if (settings.Has("threads"))
        session_options.parameters.num_threads =
            static_cast<int32_t>(Integer(settings.Get("threads"), 0, 4096));
      session_options.verbose = context_options.verbose;
      if (settings.Has("checkShapes"))
        session_options.check_shapes = settings.Get("checkShapes").As<Napi::Boolean>();
      if (settings.Has("initialCapacity"))
        session_options.persistent_tensor_initial_capacity =
            static_cast<size_t>(Integer(settings.Get("initialCapacity"), 0, 1'048'576));
      auto extension = ModelFilePath(path).extension().string();
      std::transform(extension.begin(), extension.end(), extension.begin(),
                     [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
      if (settings.Has("sdkLoader") && settings.Get("sdkLoader").As<Napi::Boolean>()) {
        rt::RuntimeLoadOptions loading;
        if (settings.Has("loadOverlap"))
          loading.overlap_preparation = settings.Get("loadOverlap").As<Napi::Boolean>();
        if (settings.Has("loadMemoryMiB")) {
          const auto memory = Integer(settings.Get("loadMemoryMiB"), 0, 1'048'576);
          if (memory)
            loading.memory_limit = static_cast<size_t>(memory) * 1024 * 1024;
        }
        if (extension == ".pipe-onnx")
          loaded_pipeline =
              std::make_unique<rt::LoadedRuntimePipeline>(path, *context, session_options, loading);
        else
          loaded =
              std::make_unique<rt::LoadedRuntimeSession>(path, *context, session_options, loading);
      } else if (extension == ".pipe-onnx") {
        FileDescriptor file(OpenModelFile(path));
        if (file.get() < 0)
          throw std::runtime_error("Could not open the prompt pipeline: " + path);
        ol::utils::FdReadStream stream(file.get());
        pipeline = std::make_unique<ol::PipelineProto>();
        if (!pipeline->ParseFromStream(stream) || !pipeline->has_prefill())
          throw std::runtime_error("Could not decode the prompt PipelineProto.");
        for (auto *stage :
             {pipeline->has_embedding() ? &pipeline->ref_embedding() : nullptr,
              &pipeline->ref_prefill(), pipeline->has_decode() ? &pipeline->ref_decode() : nullptr})
          if (stage)
            ol::LoadExternalDataForModel(*stage, ModelFilePath(path).parent_path().string());
      } else {
        FileDescriptor file(OpenModelFile(path));
        if (file.get() < 0)
          throw std::runtime_error("Could not open the prompt model: " + path);
        ol::utils::FdReadStream stream(file.get());
        plain_model = std::make_unique<ol::ModelProto>();
        ReadModel(*plain_model, stream, {});
        ol::LoadExternalDataForModel(*plain_model, ModelFilePath(path).parent_path().string());
      }
      const auto *descriptor = loaded_pipeline ? &loaded_pipeline->pipeline() : pipeline.get();
      if (descriptor) {
        std::set<std::string> persistent_inputs;
        for (const auto &binding : descriptor->prefill().graph().persistent_bindings())
          persistent_inputs.emplace(binding.input_name());
        for (const auto &initializer : descriptor->prefill().graph().initializer())
          if (persistent_inputs.contains(std::string(initializer.name())))
            initial_feeds.emplace(initializer.name(), rt::TensorFromProto(initializer));
      }
      if (!loaded && !loaded_pipeline)
        prepared = std::make_unique<rt::RuntimeSession>(
            pipeline ? pipeline->prefill() : *plain_model, initial_feeds, session_options);
      auto &base = loaded_pipeline ? loaded_pipeline->prefill_session()
                                   : (loaded ? loaded->session() : *prepared);
      session_options.value_store = base.value_store_owner();
      executor = base.cpu_executor();
      context->set_cpu_executor(executor.get());
    }
  }

  std::vector<int64_t> LightTokens(const std::vector<int32_t> &prompt, int64_t maximum,
                                   const std::optional<std::vector<int64_t>> &eos,
                                   rt::GenerationStepCallback on_step = {},
                                   size_t logit_budget = 0) {
    rt::GenerationOptions options;
    options.max_new_tokens = maximum;
    options.temperature = 0;
    if (eos)
      options.eos_token_ids = *eos;
    const auto *descriptor = loaded_pipeline ? &loaded_pipeline->pipeline() : pipeline.get();
    if (descriptor) {
      if (descriptor->has_input_ids_name())
        options.input_ids_name = descriptor->input_ids_name();
      if (descriptor->has_logits_name())
        options.logits_name = descriptor->logits_name();
      if (descriptor->has_attention_mask_name())
        options.attention_mask_name = descriptor->attention_mask_name();
      if (descriptor->has_position_ids_name())
        options.position_ids_name = descriptor->position_ids_name();
      if (descriptor->has_seed())
        options.seed = static_cast<uint64_t>(descriptor->seed());
      if (descriptor->has_pad_token_id())
        options.pad_token_id = descriptor->pad_token_id();
      if (!eos)
        options.eos_token_ids.assign(descriptor->eos_token_ids().begin(),
                                     descriptor->eos_token_ids().end());
    }
    if (pipeline) {
      pipeline->set_max_new_tokens(maximum);
      pipeline->set_temperature(0);
      if (eos) {
        pipeline->clear_eos_token_ids();
        for (const auto id : *eos)
          pipeline->add_eos_token_ids(id);
      }
    }
    const std::vector<int64_t> ids(prompt.begin(), prompt.end());
    rt::RuntimeValueMap feeds;
    for (const auto &[name, value] : initial_feeds)
      feeds.emplace(name, value.BorrowView());
    feeds.emplace(
        options.input_ids_name,
        rt::Tensor::FromInt64(options.input_ids_name, {1, static_cast<int64_t>(ids.size())}, ids));
    // Plain-pipeline warmup validates stored settings; owning pipelines use explicit options.
    const bool observe = static_cast<bool>(on_step);
    options.on_step = std::move(on_step);
    options.max_logit_bytes = logit_budget;
    const auto generated =
        loaded_pipeline ? rt::Generate(*loaded_pipeline, *context, feeds, options)
        : pipeline ? (observe ? rt::Generate(*pipeline, *context, feeds, options, session_options)
                              : rt::Generate(*pipeline, *context, feeds, session_options))
                   : rt::Generate(loaded ? loaded->model() : *plain_model, *context, feeds, options,
                                  session_options);
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

  template <typename Append>
  void GenaiTokens(const std::vector<int32_t> &prompt, int64_t maximum, Append append) {
    OgaGeneratorParams *raw_params = nullptr;
    api.Check(api.OgaCreateGeneratorParams(model.get(), &raw_params));
    auto params = Own(raw_params, api.OgaDestroyGeneratorParams);
    api.Check(
        api.OgaGeneratorParamsSetSearchNumber(params.get(), "max_length", prompt.size() + maximum));
    api.Check(api.OgaGeneratorParamsSetSearchNumber(params.get(), "num_beams", 1));
    api.Check(api.OgaGeneratorParamsSetSearchNumber(params.get(), "num_return_sequences", 1));
    api.Check(api.OgaGeneratorParamsSetSearchNumber(params.get(), "temperature", 0));
    api.Check(api.OgaGeneratorParamsSetSearchBool(params.get(), "do_sample", false));
    OgaGenerator *raw_generator = nullptr;
    api.Check(api.OgaCreateGenerator(model.get(), params.get(), &raw_generator));
    auto generator = Own(raw_generator, api.OgaDestroyGenerator);
    auto start = Clock::now();
    api.Check(api.OgaGenerator_AppendTokens(generator.get(), prompt.data(), prompt.size()));
    for (int64_t index = 0; index < maximum && !api.OgaGenerator_IsDone(generator.get()); ++index) {
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
      for (size_t axis = 0; axis + 1 < rank; ++axis)
        if (shape[axis] != 1)
          throw std::runtime_error("Expected one GenAI last-position logit row.");
      if (shape.back() <= 0 || shape.back() > std::numeric_limits<uint32_t>::max())
        throw std::runtime_error("GenAI returned an invalid vocabulary size.");
      void *values = nullptr;
      api.Check(api.OgaTensorGetData(output.get(), &values));
      api.Check(api.OgaGenerator_GenerateNextToken(generator.get()));
      const int32_t *next = nullptr;
      size_t next_count = 0;
      api.Check(api.OgaGenerator_GetNextTokens(generator.get(), &next, &next_count));
      const double elapsed = Milliseconds(start);
      if (next_count != 1)
        throw std::runtime_error("Expected one generated token.");
      append(next[0], static_cast<const float *>(values), static_cast<size_t>(shape.back()),
             elapsed);
      start = Clock::now();
    }
  }

  Genai api;
  std::string engine, key;
  std::unique_ptr<OgaTokenizer, decltype(&::OgaDestroyTokenizer)> tokenizer{nullptr, nullptr};
  std::unique_ptr<OgaModel, decltype(&::OgaDestroyModel)> model{nullptr, nullptr};
  std::shared_ptr<rt::CpuExecutor> executor;
  std::unique_ptr<rt::RuntimeContext> context;
  rt::RuntimeSessionOptions session_options;
  rt::RuntimeValueMap initial_feeds;
  std::unique_ptr<rt::LoadedRuntimeSession> loaded;
  std::unique_ptr<rt::LoadedRuntimePipeline> loaded_pipeline;
  std::unique_ptr<ol::PipelineProto> pipeline;
  std::unique_ptr<ol::ModelProto> plain_model;
  std::unique_ptr<rt::RuntimeSession> prepared;
  double load_ms = 0, warmup_ms = 0;
  bool warmed = false;
};

std::shared_ptr<PromptSession> session;

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
    if (info.Length() != 3 || !info[0].IsObject() || !info[1].IsFunction() || !info[2].IsFunction())
      throw std::runtime_error("Expected native prompt options and a token callback.");
    const auto request = info[0].As<Napi::Object>();
    for (const char *key :
         {"engine", "modelPath", "source", "libraryPath", "runtimePath", "prompt", "sessionKey"})
      if (!request.Get(key).IsString() ||
          request.Get(key).As<Napi::String>().Utf8Value().find('\0') != std::string::npos)
        throw std::runtime_error("Expected valid native prompt strings.");
    const std::string engine = request.Get("engine").As<Napi::String>();
    if (engine != "onnx-light" && engine != "onnxruntime-genai")
      throw std::runtime_error("Unknown native prompt engine.");
    const auto maximum = Integer(request.Get("maxTokens"), 1, 4096);
    const auto budget = Integer(request.Get("captureBudget"), 0, 256 * 1024 * 1024);
    const bool reused =
        session && session->key == request.Get("sessionKey").As<Napi::String>().Utf8Value();
    if (!reused) {
      session.reset();
      const auto start = Clock::now();
      session = std::make_shared<PromptSession>(request);
      session->load_ms = Milliseconds(start);
    }
    const auto owner = session;
    auto &api = owner->api;
    OgaSequences *raw_sequences = nullptr;
    api.Check(api.OgaCreateSequences(&raw_sequences));
    auto sequences = Own(raw_sequences, api.OgaDestroySequences);
    api.Check(api.OgaTokenizerEncode(owner->tokenizer.get(),
                                     request.Get("prompt").As<Napi::String>().Utf8Value().c_str(),
                                     sequences.get()));
    const auto count = api.OgaSequencesGetSequenceCount(sequences.get(), 0);
    if (!count || count > 1'000'000)
      throw std::runtime_error("The tokenizer produced an empty or oversized prompt.");
    const auto *data = api.OgaSequencesGetSequenceData(sequences.get(), 0);
    const std::vector<int32_t> prompt(data, data + count);
    const auto eos_value = request.Get("eosTokenIds");
    std::optional<std::vector<int64_t>> eos;
    if (!eos_value.IsUndefined() && !eos_value.IsNull()) {
      if (!eos_value.IsArray())
        throw std::runtime_error("eosTokenIds must be an array of token IDs.");
      const auto values = eos_value.As<Napi::Array>();
      eos.emplace();
      eos->reserve(values.Length());
      for (uint32_t i = 0; i < values.Length(); ++i)
        eos->push_back(Integer(values.Get(i), 0, std::numeric_limits<int32_t>::max()));
    }
    auto lifecycle = Napi::Object::New(env);
    lifecycle.Set("loadMs", owner->load_ms);
    lifecycle.Set("reused", reused);
    lifecycle.Set("retained", request.Get("keepLoaded").ToBoolean());
    info[2].As<Napi::Function>().Call({lifecycle});
    if (!owner->warmed) {
      const auto start = Clock::now();
      const std::vector<int32_t> warmup{prompt.front()};
      if (engine == "onnx-light") {
        owner->LightTokens(warmup, 1, eos);
        owner->context->ClearEvents();
      } else
        owner->GenaiTokens(warmup, 1, [](auto, auto, auto, auto) {});
      owner->warmup_ms = Milliseconds(start);
      owner->warmed = true;
    }
    lifecycle.Set("warmupMs", owner->warmup_ms);
    info[2].As<Napi::Function>().Call({lifecycle});
    OgaTokenizerStream *raw_stream = nullptr;
    api.Check(api.OgaCreateTokenizerStream(owner->tokenizer.get(), &raw_stream));
    auto text_stream = Own(raw_stream, api.OgaDestroyTokenizerStream);
    auto result = Napi::Object::New(env);
    result.Set("promptIds", Ids(env, prompt));
    auto tokens = Napi::Array::New(env);
    auto logits = Napi::Array::New(env);
    bool capture = true;
    size_t captured = 0;
    const auto append = [&](int64_t id, const auto *row, size_t vocabulary,
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
      if (!row) {
        capture = false;
        logits = Napi::Array::New(env);
      }
      if (row && capture) {
        // Include the typed capture, its JSON conversion and worst-case numeric text.
        if (vocabulary > (static_cast<size_t>(budget) - captured) / 40) {
          capture = false;
          logits = Napi::Array::New(env);
        } else {
          captured += vocabulary * 40;
          if (!std::all_of(row, row + vocabulary, [](auto value) { return std::isfinite(value); }))
            throw std::runtime_error("The native engine returned non-finite vocabulary logits.");
          using Value = std::remove_cv_t<std::remove_pointer_t<decltype(row)>>;
          auto values = Napi::TypedArrayOf<Value>::New(env, vocabulary);
          std::memcpy(values.Data(), row, vocabulary * sizeof(Value));
          logits.Set(tokens.Length(), values);
        }
      }
      const uint32_t index = tokens.Length();
      tokens.Set(index, token);
      info[1].As<Napi::Function>().Call({token, Napi::Number::New(env, index)});
    };
    if (engine == "onnx-light") {
      owner->LightTokens(
          prompt, maximum, eos,
          [&](const rt::GenerationStep &step) {
            if (step.token_ids.size() != 1)
              throw std::runtime_error("Expected one generated token from the SDK observer.");
            const auto elapsed =
                std::chrono::duration<double, std::milli>(step.completed_at - step.started_at)
                    .count();
            const auto id = step.token_ids.front();
            if (!step.raw_logits) {
              append(id, static_cast<const float *>(nullptr), 0, elapsed);
              return true;
            }
            const auto &row = *step.raw_logits;
            if (row.shape.size() != 2 || row.shape[0] != 1 || row.shape[1] <= 0 ||
                row.shape[1] > std::numeric_limits<uint32_t>::max())
              throw std::runtime_error("Expected one SDK last-position vocabulary row.");
            const auto vocabulary = static_cast<size_t>(row.shape[1]);
            if (row.data_type == ol::TensorProto::FLOAT)
              append(id, row.AsFloat(), vocabulary, elapsed);
            else if (row.data_type == ol::TensorProto::DOUBLE)
              append(id, row.AsDouble(), vocabulary, elapsed);
            else if (row.data_type == ol::TensorProto::FLOAT16 ||
                     row.data_type == ol::TensorProto::BFLOAT16) {
              std::vector<float> values(vocabulary);
              for (size_t index = 0; index < vocabulary; ++index) {
                uint16_t bits;
                std::memcpy(&bits, row.bytes() + index * sizeof(bits), sizeof(bits));
                values[index] = row.data_type == ol::TensorProto::FLOAT16
                                    ? rt::Float16BitsToFloat(bits)
                                    : rt::Bfloat16BitsToFloat(bits);
              }
              append(id, values.data(), vocabulary, elapsed);
            } else
              throw std::runtime_error("The SDK returned an unsupported generation logit dtype.");
            return true;
          },
          static_cast<size_t>(budget));
      auto events = Napi::Array::New(env);
      for (const auto &event : owner->context->TakeEvents()) {
        if (event.action != rt::RuntimeEventAction::kRunNode)
          continue;
        auto item = Napi::Object::New(env);
        item.Set("operator", event.op_type);
        item.Set("effectiveThreads", event.cpu_effective_threads);
        item.Set("timeMs", static_cast<double>(event.duration_ns) / 1e6);
        events.Set(events.Length(), item);
      }
      result.Set("profileEvents", events);
    } else {
      owner->GenaiTokens(prompt, maximum, append);
    }
    result.Set("logits", capture ? Napi::Value(logits) : env.Null());
    result.Set("tokens", tokens);
    result.Set("lifecycle", lifecycle);
    if (!request.Get("keepLoaded").ToBoolean())
      session.reset();
    return result;
  } catch (const std::exception &error) {
    session.reset();
    Napi::Error::New(env, error.what()).ThrowAsJavaScriptException();
    return env.Undefined();
  }
}

Napi::Value ReleaseNativePrompt(const Napi::CallbackInfo &info) {
  session.reset();
  return info.Env().Undefined();
}
} // namespace onnx_light_viewer
