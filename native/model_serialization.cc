/** @file Serialization workers and retained-model save methods. */
#include "model_io.h"
#include "model_storage.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <onnx_core/runtime/memory/simple_tensor.h>
#include <onnx_proto/onnx_crypt.h>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

namespace onnx_light_viewer {
namespace {

bool SameWeight(const ol::TensorProto &left, const ol::TensorProto &right) {
  if (left.name() != right.name() || left.data_type() != right.data_type() ||
      left.dims().size() != right.dims().size() ||
      !std::equal(left.dims().begin(), left.dims().end(), right.dims().begin()))
    return false;
  if (left.data_type() == ol::TensorProto::STRING)
    return left.string_data().size() == right.string_data().size() &&
           std::equal(left.string_data().begin(), left.string_data().end(),
                      right.string_data().begin());
  const auto a = ol::core::runtime::TensorFromProto(left);
  const auto b = ol::core::runtime::TensorFromProto(right);
  return a.size_bytes() == b.size_bytes() &&
         (!a.size_bytes() || std::memcmp(a.bytes(), b.bytes(), a.size_bytes()) == 0);
}

/** Serializes an independent snapshot; omitted inline payloads must first be restored. */
class SerializeModelWorker : public Napi::AsyncWorker {
public:
  SerializeModelWorker(Napi::Env env, std::shared_ptr<StoredModel> source,
                       std::optional<std::string> password, std::string file_path,
                       size_t max_external_file_size = 0, std::string external_file_prefix = "",
                       bool reuse_weights = false, bool include_rewritings = false)
      : Napi::AsyncWorker(env), deferred_(Napi::Promise::Deferred::New(env)),
        source_(include_rewritings && source->fusion_base ? source->fusion_base : source),
        snapshot_(source_->model->SharedCopy()), password_(std::move(password)),
        file_path_(std::move(file_path)), max_external_file_size_(max_external_file_size),
        external_file_prefix_(std::move(external_file_prefix)), reuse_weights_(reuse_weights) {
    if (!include_rewritings) {
      snapshot_.clear_rewritings();
    } else if (!source->fusion_plans.empty()) {
      ol::RewritingProto combined;
      for (const auto &plan : source->fusion_plans) {
        for (const auto &edit : plan.rewrites())
          *combined.add_rewrites() = edit.SharedCopy();
        // Each captured plan contains the complete declarations at that step.
        combined.ref_required_opset_imports() = plan.required_opset_imports();
        combined.ref_added_functions() = plan.added_functions();
      }
      std::string name = "onnx-light-viewer.fusions";
      for (size_t suffix = 2; snapshot_.rewritings().contains(name); ++suffix)
        name = "onnx-light-viewer.fusions." + std::to_string(suffix);
      snapshot_.ref_rewritings().emplace(name, std::move(combined));
    }
  }

  Napi::Promise Promise() const { return deferred_.Promise(); }

  void Execute() override {
    try {
      RestoreModelPayloads(snapshot_, *source_, file_path_);
      std::vector<ol::EncodedValueProto *> encoded;
      std::set<const ol::TensorProto *> layout_tensors;
      const auto tensors = ModelSaveTensors(snapshot_, &encoded, &layout_tensors);
      for (auto *value : encoded)
        if (value->has_data_location() && value->data_location() == ol::TensorProto::EXTERNAL) {
          if (file_path_.empty())
            throw std::runtime_error(
                "External weights require a local source model path when saving.");
          LoadExternalTensor(*value, ModelFilePath(file_path_).parent_path());
        }
      auto external_weights = source_->external_weights;
      for (auto *tensor : tensors)
        if ((!layout_tensors.contains(tensor) || tensor->has_data_location()) &&
            tensor->data_location() == ol::TensorProto::EXTERNAL) {
          if (file_path_.empty())
            throw std::runtime_error(
                "External weights require a local source model path when saving.");
          auto reference = tensor->SharedCopy();
          LoadExternalTensor(*tensor, ModelFilePath(file_path_).parent_path());
          external_weights.push_back({std::move(reference), tensor->SharedCopy()});
        }
      std::set<const ol::TensorProto *> reused;
      if (reuse_weights_) {
        using WeightKey = std::tuple<std::string, int32_t, std::vector<int64_t>>;
        const auto weight_key = [](const ol::TensorProto &tensor) -> WeightKey {
          return {tensor.name().value(),
                  tensor.data_type(),
                  {tensor.dims().begin(), tensor.dims().end()}};
        };
        std::map<WeightKey, std::vector<const StoredModel::ExternalWeight *>> weights_by_key;
        for (const auto &weight : external_weights)
          weights_by_key[weight_key(weight.value)].push_back(&weight);
        for (auto *tensor : tensors) {
          if (layout_tensors.contains(tensor))
            continue;
          const auto candidates = weights_by_key.find(weight_key(*tensor));
          if (candidates == weights_by_key.end())
            continue;
          for (const auto *weight : candidates->second)
            if (SameWeight(*tensor, weight->value)) {
              auto fresh = weight->reference.SharedCopy();
              LoadExternalTensor(fresh, ModelFilePath(file_path_).parent_path());
              if (!SameWeight(fresh, weight->value))
                throw std::runtime_error("External weights changed on disk. Reopen the model.");
              tensor->ref_external_data() = weight->reference.external_data();
              tensor->set_data_location(ol::TensorProto::EXTERNAL);
              tensor->clear_raw_data();
              tensor->clear_float_data();
              tensor->clear_double_data();
              tensor->clear_int32_data();
              tensor->clear_int64_data();
              tensor->clear_uint64_data();
              tensor->clear_string_data();
              reused.insert(tensor);
              break;
            }
        }
      }
      if (max_external_file_size_) {
        const ol::SerializeOptions options;
        for (auto *value : tensors) {
          auto &tensor = *value;
          // Layout constants are part of versioned codec descriptors, not ordinary weights.
          if (layout_tensors.contains(value) || reused.contains(value) || tensor.has_raw_data() ||
              tensor.data_type() == ol::TensorProto::STRING ||
              (tensor.float_data().empty() && tensor.double_data().empty() &&
               tensor.int32_data().empty() && tensor.int64_data().empty() &&
               tensor.uint64_data().empty()))
            continue;
          const auto data = ol::core::runtime::TensorFromProto(tensor);
          if (data.size_bytes() < static_cast<size_t>(options.raw_data_threshold))
            continue;
          tensor.set_raw_data(data.bytes(), data.size_bytes());
          tensor.clear_float_data();
          tensor.clear_double_data();
          tensor.clear_int32_data();
          tensor.clear_int64_data();
          tensor.clear_uint64_data();
        }
        // The split serializer requires payloads for every EXTERNAL tensor.
        // Partition only new payloads, then restore the retained references.
        std::map<size_t, ol::TensorProto> references;
        for (size_t i = 0; i < tensors.size(); ++i)
          if (reused.contains(tensors[i])) {
            references.emplace(i, tensors[i]->SharedCopy());
            tensors[i]->set_data_location(ol::TensorProto::DEFAULT);
            tensors[i]->clear_external_data();
          }
        if (!snapshot_.SerializeToString(bytes_, external_files_, max_external_file_size_,
                                         external_file_prefix_, options))
          throw std::runtime_error("Could not serialize the ONNX model with external weights.");
        for (const auto &[name, bytes] : external_files_)
          if (bytes.size() > max_external_file_size_)
            throw std::runtime_error(
                "A tensor exceeds the maximum external weight file size. Choose a larger size.");
        if (!references.empty()) {
          ol::ModelProto partitioned;
          if (!partitioned.ParseFromString(bytes_))
            throw std::runtime_error("Could not restore reused external weight references.");
          const auto partitioned_tensors = ModelSaveTensors(partitioned);
          if (partitioned_tensors.size() != tensors.size())
            throw std::runtime_error("The serializer changed tensor positions.");
          for (const auto &[position, reference] : references) {
            if (partitioned_tensors[position]->name() != reference.name())
              throw std::runtime_error("The serializer changed tensor positions.");
            *partitioned_tensors[position] = reference.SharedCopy();
          }
          std::string finished;
          if (!partitioned.SerializeToString(finished))
            throw std::runtime_error("Could not serialize reused external weight references.");
          bytes_ = std::move(finished);
        }
      } else if (password_) {
#ifdef ONNX_LIGHT_HAS_OPENSSL
        bytes_ = ol::SaveEncryptedModelToString(snapshot_, *password_, "AES-256-GCM");
#else
        throw std::runtime_error("This native build does not support encrypted ONNX models.");
#endif
      } else if (!snapshot_.SerializeToString(bytes_)) {
        throw std::runtime_error("Could not serialize the ONNX model.");
      }
    } catch (const std::exception &error) {
      SetError(error.what());
    }
  }

  void OnOK() override {
    const auto bytes = Napi::Buffer<uint8_t>::Copy(
        Env(), reinterpret_cast<const uint8_t *>(bytes_.data()), bytes_.size());
    if (!max_external_file_size_) {
      deferred_.Resolve(bytes);
      return;
    }
    auto result = Napi::Object::New(Env());
    result.Set("model", bytes);
    auto weights = Napi::Array::New(Env(), external_files_.size());
    uint32_t index = 0;
    for (const auto &[name, data] : external_files_) {
      auto file = Napi::Object::New(Env());
      file.Set("name", name);
      file.Set("bytes", Napi::Buffer<uint8_t>::Copy(
                            Env(), reinterpret_cast<const uint8_t *>(data.data()), data.size()));
      weights.Set(index++, file);
    }
    result.Set("weights", weights);
    deferred_.Resolve(result);
  }
  void OnError(const Napi::Error &error) override { deferred_.Reject(error.Value()); }

private:
  Napi::Promise::Deferred deferred_;
  std::shared_ptr<StoredModel> source_;
  ol::ModelProto snapshot_;
  std::optional<std::string> password_;
  std::string file_path_;
  std::string bytes_;
  size_t max_external_file_size_;
  std::string external_file_prefix_;
  std::unordered_map<std::string, std::string> external_files_;
  bool reuse_weights_;
};

} // namespace

void AddSerializationMethods(Napi::Env env, Napi::Object result,
                             std::shared_ptr<StoredModel> model) {
  result.Set("externalDataLocations",
             Napi::Function::New(env, [model](const Napi::CallbackInfo &call) -> Napi::Value {
               try {
                 if (call.Length() != 0)
                   throw std::runtime_error("Expected no external-data location arguments.");
                 std::set<const ol::TensorProto *> omitted;
                 for (const auto &payload : model->skipped) {
                   const auto *tensor = FindTensor(*model->model, payload);
                   if (!tensor)
                     throw std::runtime_error("Could not locate an omitted tensor payload.");
                   omitted.insert(tensor);
                 }
                 std::set<std::string> locations;
                 for (const auto &weight : model->external_weights)
                   locations.insert(ExternalDataLocation(weight.reference));
                 std::vector<ol::EncodedValueProto *> encoded;
                 for (const auto *tensor : ModelSaveTensors(*model->model, &encoded))
                   if (tensor->data_location() == ol::TensorProto::EXTERNAL &&
                       !omitted.contains(tensor))
                     locations.insert(ExternalDataLocation(*tensor));
                 for (const auto *value : encoded)
                   if (value->data_location() == ol::TensorProto::EXTERNAL)
                     locations.insert(ExternalDataLocation(*value));
                 auto values = Napi::Array::New(call.Env(), locations.size());
                 uint32_t index = 0;
                 for (const auto &location : locations)
                   values.Set(index++, location);
                 return values;
               } catch (const std::exception &error) {
                 Napi::Error::New(call.Env(), error.what()).ThrowAsJavaScriptException();
                 return call.Env().Undefined();
               }
             }));
  result.Set(
      "serializeFiles",
      Napi::Function::New(env, [model](const Napi::CallbackInfo &call) -> Napi::Value {
        try {
          if (call.Length() < 2 || call.Length() > 5 || !call[0].IsNumber() ||
              !call[1].IsString() || (!call[2].IsUndefined() && !call[2].IsString()) ||
              (!call[3].IsUndefined() && !call[3].IsBoolean()) ||
              (!call[4].IsUndefined() && !call[4].IsBoolean()))
            throw std::runtime_error(
                "Expected a maximum external file size, a prefix and an optional source path.");
          const double maximum = call[0].As<Napi::Number>().DoubleValue();
          if (!std::isfinite(maximum) || maximum <= 0 || std::floor(maximum) != maximum ||
              maximum > 9007199254740991.0 ||
              maximum >= static_cast<double>(std::numeric_limits<size_t>::max()))
            throw std::runtime_error("Expected a positive, safe integer external file size.");
          const auto prefix = call[1].As<Napi::String>().Utf8Value();
          if (prefix.empty() || prefix == "." || prefix == ".." ||
              prefix.find_first_of("/\\") != std::string::npos ||
              prefix.find('\0') != std::string::npos)
            throw std::runtime_error("Expected an external file prefix without directories.");
          const auto file_path =
              call[2].IsString() ? call[2].As<Napi::String>().Utf8Value() : model->file_path;
          if (file_path.find('\0') != std::string::npos)
            throw std::runtime_error("Invalid source model file path.");
          auto worker = std::make_unique<SerializeModelWorker>(
              call.Env(), model, std::nullopt, file_path, static_cast<size_t>(maximum), prefix,
              call[3].IsBoolean() && call[3].As<Napi::Boolean>().Value(),
              call[4].IsBoolean() && call[4].As<Napi::Boolean>().Value());
          const auto promise = worker->Promise();
          worker->Queue();
          worker.release();
          return promise;
        } catch (const std::exception &error) {
          Napi::Error::New(call.Env(), error.what()).ThrowAsJavaScriptException();
          return call.Env().Undefined();
        }
      }));
  result.Set("serialize",
             Napi::Function::New(env, [model](const Napi::CallbackInfo &call) -> Napi::Value {
               try {
                 if (call.Length() > 3 || (!call[1].IsUndefined() && !call[1].IsString()) ||
                     (!call[2].IsUndefined() && !call[2].IsBoolean()))
                   throw std::runtime_error("Expected an optional password and source file path.");
                 const auto file_path =
                     call[1].IsString() ? call[1].As<Napi::String>().Utf8Value() : model->file_path;
                 if (file_path.find('\0') != std::string::npos)
                   throw std::runtime_error("Invalid source model file path.");
                 auto worker = std::make_unique<SerializeModelWorker>(
                     call.Env(), model, ReadPassword(call[0]), file_path, 0, "", false,
                     call[2].IsBoolean() && call[2].As<Napi::Boolean>().Value());
                 const auto promise = worker->Promise();
                 worker->Queue();
                 worker.release();
                 return promise;
               } catch (const std::exception &error) {
                 Napi::Error::New(call.Env(), error.what()).ThrowAsJavaScriptException();
                 return call.Env().Undefined();
               }
             }));
}

} // namespace onnx_light_viewer
