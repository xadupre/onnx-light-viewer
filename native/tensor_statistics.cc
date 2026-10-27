/** @file On-demand numeric tensor inspection through the SDK. */
#include "graph.h"
#include "model_io.h"
#include "model_storage.h"
#include "tensor_preview.h"
#include <cmath>
#include <onnx_core/runtime/kernels/tensor_statistics.h>
#include <onnx_core/runtime/quantization.h>
#include <stdexcept>

namespace onnx_light_viewer {
namespace {
namespace rt = ol::core::runtime;

Napi::Value NumericValue(Napi::Env env, double value) {
  if (std::isnan(value))
    return Text(env, "NaN");
  if (std::isinf(value))
    return Text(env, value > 0 ? "Infinity" : "-Infinity");
  return Napi::Number::New(env, value);
}

Napi::Array Numbers(Napi::Env env, const std::vector<double> &values) {
  auto result = Napi::Array::New(env, values.size());
  for (size_t index = 0; index < values.size(); ++index)
    result.Set(index, NumericValue(env, values[index]));
  return result;
}

Napi::Array Integers(Napi::Env env, const std::vector<int64_t> &values) {
  auto result = Napi::Array::New(env, values.size());
  for (size_t index = 0; index < values.size(); ++index)
    result.Set(index, Text(env, std::to_string(values[index])));
  return result;
}

class StatisticsWorker : public Napi::AsyncWorker {
public:
  StatisticsWorker(Napi::Env env, std::shared_ptr<StoredModel> source, std::string name,
                   GraphPath path)
      : Napi::AsyncWorker(env), deferred_(Napi::Promise::Deferred::New(env)),
        source_(std::move(source)), snapshot_(source_->model->SharedCopy()), name_(std::move(name)),
        path_(std::move(path)) {}

  Napi::Promise Promise() const { return deferred_.Promise(); }

  void Execute() override {
    try {
      auto selection = GraphScopes(snapshot_, path_);
      for (size_t index = selection.scopes.size(); index > selection.capture_start; --index) {
        const auto &graph = *selection.scopes[index - 1];
        for (int tensor_index = 0; tensor_index < graph.initializer_size(); ++tensor_index) {
          const auto &tensor = graph.initializer(tensor_index);
          if (tensor.name() == name_) {
            if (IsPayloadOmitted(tensor)) {
              RestoreStatisticsPayloads(snapshot_, *source_, source_->file_path, {&tensor});
              // Restoring replaces messages, so resolve the graph and tensor again.
              selection = GraphScopes(snapshot_, path_);
              Compute(selection.scopes[index - 1]->initializer(tensor_index));
            } else {
              Compute(tensor);
            }
            return;
          }
        }
        for (const auto &tensor : graph.extended_initializer())
          if (tensor.has_encoded_value() && tensor.encoded_value().name() == name_) {
            Compute(tensor.encoded_value());
            return;
          }
        for (const auto &tensor : graph.sparse_initializer())
          if (tensor.values().name() == name_)
            throw std::runtime_error("Sparse tensor statistics are not provided by onnx-light.");
        for (int node_index = 0; node_index < graph.node_size(); ++node_index) {
          const auto &node = graph.node(node_index);
          for (const auto &output : node.output())
            if (output == name_) {
              if (node.op_type() == "Constant" &&
                  (node.domain().empty() || node.domain() == "ai.onnx")) {
                for (const auto &attribute : node.attribute()) {
                  if (!attribute.ref_attr_name().empty())
                    continue;
                  if (attribute.has_t()) {
                    if (IsPayloadOmitted(attribute.t())) {
                      const std::string attribute_name(attribute.name());
                      RestoreStatisticsPayloads(snapshot_, *source_, source_->file_path,
                                                {&attribute.t()});
                      selection = GraphScopes(snapshot_, path_);
                      const auto &restored = selection.scopes[index - 1]->node(node_index);
                      for (const auto &value : restored.attribute())
                        if (value.name() == attribute_name && value.has_t()) {
                          Compute(value.t());
                          return;
                        }
                    } else {
                      Compute(attribute.t());
                      return;
                    }
                  }
                  if (attribute.name() == "value_float" && attribute.has_f()) {
                    Compute(rt::Tensor::FromFloat(name_, {}, {attribute.f()}));
                    return;
                  }
                  if (attribute.name() == "value_int" && attribute.has_i()) {
                    Compute(rt::Tensor::FromInt64(name_, {}, {attribute.i()}));
                    return;
                  }
                  if (attribute.name() == "value_floats") {
                    Compute(rt::Tensor::FromFloat(
                        name_, {attribute.floats_size()},
                        std::vector<float>(attribute.floats().begin(), attribute.floats().end())));
                    return;
                  }
                  if (attribute.name() == "value_ints") {
                    Compute(rt::Tensor::FromInt64(
                        name_, {attribute.ints_size()},
                        std::vector<int64_t>(attribute.ints().begin(), attribute.ints().end())));
                    return;
                  }
                }
              }
              throw std::runtime_error(
                  "No stored tensor data. The viewer does not execute the model.");
            }
        }
        // A graph input shadows a captured value of the same name.
        for (const auto &input : graph.input())
          if (input.name() == name_)
            throw std::runtime_error(
                "No stored tensor data. The viewer does not execute the model.");
      }
      throw std::runtime_error("No stored tensor data for " + name_ +
                               ". The viewer does not execute the model.");
    } catch (const std::exception &error) {
      SetError(error.what());
    }
  }

  void OnOK() override {
    auto result = Napi::Object::New(Env());
    result.Set("name", Text(Env(), name_));
    result.Set("dtype", Text(Env(), ol::TensorProto::DataType_Name(statistics_.dtype)));
    result.Set("shape", Integers(Env(), statistics_.shape));
    result.Set("count", Text(Env(), std::to_string(statistics_.count)));
    result.Set("sizeBytes", Text(Env(), std::to_string(statistics_.size_bytes)));
    result.Set("min", NumericValue(Env(), statistics_.min));
    result.Set("max", NumericValue(Env(), statistics_.max));
    result.Set("mean", NumericValue(Env(), statistics_.mean));
    result.Set("median", NumericValue(Env(), *statistics_.median));
    if (statistics_.distinct_count)
      result.Set("distinctCount", Text(Env(), std::to_string(*statistics_.distinct_count)));
    result.Set("quantiles", Numbers(Env(), statistics_.quantiles));
    result.Set("distribution", Integers(Env(), statistics_.distribution));
    result.Set("binEdges", Numbers(Env(), statistics_.bin_edges));
    result.Set("rectangleMap", Numbers(Env(), statistics_.rectangle_map));
    auto shape = Napi::Array::New(Env(), statistics_.rectangle_shape.size());
    for (size_t index = 0; index < statistics_.rectangle_shape.size(); ++index)
      shape.Set(index, Napi::Number::New(Env(), statistics_.rectangle_shape[index]));
    result.Set("rectangleShape", shape);
    deferred_.Resolve(result);
  }

  void OnError(const Napi::Error &error) override { deferred_.Reject(error.Value()); }

private:
  template <typename Proto> void Load(Proto &tensor) {
    if (tensor.data_location() != ol::TensorProto::EXTERNAL && tensor.external_data().empty())
      return;
    if (source_->file_path.empty())
      throw std::runtime_error("External tensor statistics require a local source model.");
    LoadExternalTensor(tensor, ModelFilePath(source_->file_path).parent_path());
  }

  void Compute(const ol::TensorProto &tensor) {
    auto copy = tensor.SharedCopy();
    Load(copy);
    Compute(rt::TensorFromProto(copy));
  }

  void Compute(const rt::Tensor &tensor) {
    statistics_ = rt::ComputeTensorStatistics(tensor, 0, {0.25, 0.5, 0.75}, true, {32, 32});
  }

  std::shared_ptr<const rt::QuantizationParameterCatalogue>
  LoadQuantizationParameters(const std::string &reference) {
    constexpr std::string_view prefix = "onnx_light.quantization.parameters:";
    const auto &graph = snapshot_.graph();
    const auto annotation =
        std::find_if(graph.quantization_annotation().begin(), graph.quantization_annotation().end(),
                     [&](const auto &candidate) {
                       return candidate.tensor_name().value() == std::string(prefix) + reference;
                     });
    if (annotation == graph.quantization_annotation().end())
      return rt::QuantizationParameterCatalogue::Build(snapshot_);

    std::set<std::string> names;
    std::set<std::string> numerical_names;
    std::vector<const ol::TensorProto *> targets;
    for (const auto &mapping : annotation->quant_parameter_tensor_names()) {
      const auto &name = mapping.value().value();
      names.insert(name);
      if (mapping.key().value() != "storage_type" && mapping.key().value() != "logical_type")
        numerical_names.insert(name);
      for (const auto &tensor : graph.initializer())
        if (tensor.name() == name)
          targets.push_back(&tensor);
    }
    RestoreStatisticsPayloads(snapshot_, *source_, source_->file_path, targets);

    auto &initializers = snapshot_.ref_graph().ref_initializer();
    for (auto &tensor : initializers)
      if (names.contains(tensor.name()) && numerical_names.contains(tensor.name())) {
        auto loaded = tensor.SharedCopy();
        Load(loaded);
        tensor = std::move(loaded);
      }

    auto catalogue_model = snapshot_.SharedCopy();
    auto &annotations = catalogue_model.ref_graph().ref_quantization_annotation();
    annotations.clear();
    annotations.push_back(*annotation);
    return rt::QuantizationParameterCatalogue::Build(catalogue_model);
  }

  void Compute(const ol::EncodedValueProto &tensor) {
    auto copy = tensor.SharedCopy();
    Load(copy);
    ol::StructTypeCatalogue catalogue;
    catalogue.Build(snapshot_);
    const auto stored_bytes = copy.raw_data().size();
    std::shared_ptr<const rt::QuantizationParameterCatalogue> parameters;
    if (!copy.parameter_ref().empty()) {
      parameters = LoadQuantizationParameters(copy.parameter_ref());
      copy = rt::MaterializeQuantizedValue(copy, parameters.get(), catalogue);
    }
    statistics_ =
        rt::ComputeTensorStatistics(copy, 0, {0.25, 0.5, 0.75}, true, {32, 32}, catalogue);
    statistics_.size_bytes = stored_bytes;
  }

  Napi::Promise::Deferred deferred_;
  std::shared_ptr<StoredModel> source_;
  ol::ModelProto snapshot_;
  std::string name_;
  GraphPath path_;
  rt::TensorStatistics statistics_;
};
} // namespace

void AddTensorStatisticsMethods(Napi::Env env, Napi::Object result,
                                std::shared_ptr<StoredModel> model) {
  result.Set("tensorStatistics",
             Napi::Function::New(env, [model](const Napi::CallbackInfo &call) -> Napi::Value {
               try {
                 if (call.Length() < 1 || call.Length() > 2 || !call[0].IsString() ||
                     (call.Length() == 2 && !call[1].IsUndefined() && !call[1].IsArray()))
                   throw std::runtime_error("Expected a tensor name and optional graph path.");
                 auto name = call[0].As<Napi::String>().Utf8Value();
                 if (name.empty())
                   throw std::runtime_error("Expected a nonempty tensor name.");
                 auto *worker =
                     new StatisticsWorker(call.Env(), model, std::move(name), ReadPath(call[1]));
                 auto promise = worker->Promise();
                 worker->Queue();
                 return promise;
               } catch (const std::exception &error) {
                 Napi::Error::New(call.Env(), error.what()).ThrowAsJavaScriptException();
                 return call.Env().Undefined();
               }
             }));
}
} // namespace onnx_light_viewer
