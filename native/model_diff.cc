/** @file Read-only projection of an onnx-light model rewriting diff. */
#include "addon.h"
#include "graph.h"
#include "model_io.h"
#include "model_storage.h"
#include "rewriting_trace.h"
#include "tensor_preview.h"
#include <algorithm>
#include <map>
#include <onnx_core/builder/graph_builder.h>
#include <onnx_lib/common/constants.h>
#include <onnx_lib/version_converter/convert.h>
#include <set>
#include <stdexcept>

namespace onnx_light_viewer {
namespace {

struct DiffNode {
  StoredModel::FusionMatch::Node node;
  std::vector<std::pair<std::string, std::string>> attributes;
};

struct DiffStep {
  std::string pattern;
  std::vector<std::string> scope;
  std::vector<DiffNode> before, after;
  std::vector<size_t> removed_initializers, removed_extended_initializers;
  std::vector<std::string> added_initializers, prepared_initializers;
};

DiffNode CaptureNode(const ol::NodeProto &node, std::optional<size_t> index = std::nullopt) {
  DiffNode result{ImpactNode(NodeOrderKey(node), index), {}};
  for (const auto &attribute : node.attribute())
    result.attributes.emplace_back(attribute.name().value(),
                                   AttributePreview(attribute, node.domain()));
  return result;
}

struct ScopeTrace {
  explicit ScopeTrace(const ol::core::builder::GraphBuilder &graph)
      : trace(graph.Nodes(), &nodes) {}
  std::vector<const ol::NodeProto *> nodes;
  RewritingTrace trace;
};

void AlignVersions(ol::ModelProto &source, ol::ModelProto &target) {
  const auto default_opset = [](const ol::ModelProto &model) -> std::optional<int64_t> {
    for (const auto &opset : model.opset_import())
      if (ol::IsOnnxDomain(opset.domain().value()))
        return opset.version();
    return std::nullopt;
  };
  const auto source_opset = default_opset(source);
  const auto target_opset = default_opset(target);
  if (source_opset && target_opset && *source_opset != *target_opset) {
    auto &older = *source_opset < *target_opset ? source : target;
    const ol::version_conversion::DefaultVersionConverter converter;
    older = converter.convert_version(older, ol::OpSetID(std::min(*source_opset, *target_opset)),
                                      ol::OpSetID(std::max(*source_opset, *target_opset)));
  }
  const auto ir_version = std::max(source.ir_version(), target.ir_version());
  source.set_ir_version(ir_version);
  target.set_ir_version(ir_version);
}

std::string WeightDirectory(ol::ModelProto &model, const StoredModel &source) {
  std::vector<ol::EncodedValueProto *> encoded;
  const auto tensors = ModelSaveTensors(model, &encoded, nullptr, false);
  std::set<std::string> locations;
  const auto collect = [&](const auto &value) {
    if (value.has_data_location() && value.data_location() == ol::TensorProto::EXTERNAL)
      locations.insert(ExternalDataLocation(value));
  };
  for (const auto *tensor : tensors)
    collect(*tensor);
  for (const auto *value : encoded)
    collect(*value);
  if (locations.empty())
    return {};
  if (source.file_path.empty())
    throw std::runtime_error("Comparing external weights requires a local path for each model.");
  const auto directory = std::filesystem::absolute(ModelFilePath(source.file_path)).parent_path();
  for (const auto &location : locations)
    CheckedExternalWeightPath(location, directory);
  const auto utf8 = directory.u8string();
  return std::string(utf8.begin(), utf8.end());
}

Napi::Array Strings(Napi::Env env, const std::vector<std::string> &values) {
  auto result = Napi::Array::New(env, values.size());
  for (size_t i = 0; i < values.size(); ++i)
    result.Set(i, values[i]);
  return result;
}

Napi::Array Indices(Napi::Env env, const std::vector<size_t> &values) {
  auto result = Napi::Array::New(env, values.size());
  for (size_t i = 0; i < values.size(); ++i)
    result.Set(i, Napi::Number::New(env, values[i]));
  return result;
}

Napi::Array Nodes(Napi::Env env, const std::vector<DiffNode> &values) {
  auto result = Napi::Array::New(env, values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    const auto &node = values[i].node;
    auto item = Napi::Object::New(env);
    if (node.index)
      item.Set("index", Napi::Number::New(env, *node.index));
    item.Set("name", node.name);
    item.Set("opType", node.op_type);
    item.Set("domain", node.domain);
    item.Set("inputs", Strings(env, node.inputs));
    item.Set("outputs", Strings(env, node.outputs));
    auto attributes = Napi::Array::New(env, values[i].attributes.size());
    for (size_t j = 0; j < values[i].attributes.size(); ++j) {
      auto attribute = Napi::Object::New(env);
      attribute.Set("name", values[i].attributes[j].first);
      attribute.Set("value", values[i].attributes[j].second);
      attributes.Set(j, attribute);
    }
    item.Set("attributes", attributes);
    result.Set(i, item);
  }
  return result;
}

class DiffWorker : public Napi::AsyncWorker {
public:
  DiffWorker(Napi::Env env, std::shared_ptr<StoredModel> source,
             std::shared_ptr<StoredModel> target)
      : Napi::AsyncWorker(env), deferred_(Napi::Promise::Deferred::New(env)),
        source_(std::move(source)), target_(std::move(target)),
        source_snapshot_(source_->model->SharedCopy()),
        target_snapshot_(target_->model->SharedCopy()) {}

  Napi::Promise Promise() const { return deferred_.Promise(); }

  void Execute() override {
    try {
      if (source_->has_training_info || target_->has_training_info)
        throw std::runtime_error(
            "Comparing models with training_info is not supported by onnx-light.");
      RestoreModelPayloads(source_snapshot_, *source_, source_->file_path);
      RestoreModelPayloads(target_snapshot_, *target_, target_->file_path);
      const auto source_dir = WeightDirectory(source_snapshot_, *source_);
      const auto target_dir = WeightDirectory(target_snapshot_, *target_);
      AlignVersions(source_snapshot_, target_snapshot_);
      ol::core::builder::GraphBuilder source(source_snapshot_);
      plan_ = source.DiffRewritings(target_snapshot_, source_dir, target_dir);
      std::map<std::vector<std::string>, std::unique_ptr<ScopeTrace>> scopes;
      for (const auto &edit : plan_.rewrites()) {
        DiffStep step;
        step.pattern = edit.pattern_name().value();
        step.scope.assign(edit.graph_path().begin(), edit.graph_path().end());
        if (!edit.matched_nodes().empty() || !edit.added_nodes().empty()) {
          auto &trace = scopes[step.scope];
          if (!trace) {
            const auto *graph = &source;
            for (const auto &name : step.scope)
              graph = &graph->Subgraph(name);
            trace = std::make_unique<ScopeTrace>(*graph);
          }
          for (const auto index : edit.matched_nodes())
            step.before.push_back(CaptureNode(trace->trace.NodeAt(index), index));
          for (const auto &node : edit.added_nodes())
            step.after.push_back(CaptureNode(node));
          trace->trace.Apply(edit, true);
        }
        step.removed_initializers.assign(edit.removed_initializers().begin(),
                                         edit.removed_initializers().end());
        step.removed_extended_initializers.assign(edit.removed_extended_initializers().begin(),
                                                  edit.removed_extended_initializers().end());
        for (const auto &tensor : edit.added_initializers())
          step.added_initializers.push_back(tensor.name().value());
        for (const auto &call : edit.initializer_calls())
          for (const auto &output : call.output())
            step.prepared_initializers.push_back(output.value());
        steps_.push_back(std::move(step));
      }
    } catch (const std::exception &error) {
      SetError(error.what());
    }
  }

  void OnOK() override {
    auto env = Env();
    auto result = Napi::Object::New(env);
    auto steps = Napi::Array::New(env, steps_.size());
    for (size_t i = 0; i < steps_.size(); ++i) {
      const auto &step = steps_[i];
      auto item = Napi::Object::New(env);
      item.Set("pattern", step.pattern);
      auto impact = Napi::Object::New(env);
      impact.Set("scope", Strings(env, step.scope));
      impact.Set("before", Nodes(env, step.before));
      impact.Set("after", Nodes(env, step.after));
      impact.Set("removedInitializers",
                 Napi::Number::New(env, step.removed_initializers.size() +
                                            step.removed_extended_initializers.size()));
      impact.Set("addedInitializers", Napi::Number::New(env, step.added_initializers.size()));
      item.Set("impact", impact);
      item.Set("removedInitializers", Indices(env, step.removed_initializers));
      item.Set("removedExtendedInitializers", Indices(env, step.removed_extended_initializers));
      item.Set("addedInitializers", Strings(env, step.added_initializers));
      item.Set("preparedInitializers", Strings(env, step.prepared_initializers));
      steps.Set(i, item);
    }
    result.Set("steps", steps);
    auto moves = Napi::Array::New(env, plan_.moved_nodes().size());
    for (size_t i = 0; i < plan_.moved_nodes().size(); ++i) {
      const auto &move = plan_.moved_nodes(i);
      auto item = Napi::Object::New(env);
      item.Set("from", Napi::Number::New(env, move.from_position()));
      item.Set("to", Napi::Number::New(env, move.to_position()));
      item.Set("function", move.function_id().value());
      std::vector<std::string> scope(move.graph_path().begin(), move.graph_path().end());
      item.Set("scope", Strings(env, scope));
      moves.Set(i, item);
    }
    result.Set("moves", moves);
    auto opsets = Napi::Array::New(env, plan_.required_opset_imports().size());
    for (size_t i = 0; i < plan_.required_opset_imports().size(); ++i) {
      const auto &opset = plan_.required_opset_imports(i);
      auto item = Napi::Object::New(env);
      item.Set("domain", opset.domain().value());
      item.Set("version", std::to_string(opset.version()));
      opsets.Set(i, item);
    }
    result.Set("opsets", opsets);
    std::vector<std::string> functions, replaced;
    for (const auto &function : plan_.added_functions())
      functions.push_back(FunctionName(function));
    replaced.assign(plan_.replaced_function_ids().begin(), plan_.replaced_function_ids().end());
    result.Set("functions", Strings(env, functions));
    result.Set("replacedFunctions", Strings(env, replaced));
    deferred_.Resolve(result);
  }

  void OnError(const Napi::Error &error) override { deferred_.Reject(error.Value()); }

private:
  Napi::Promise::Deferred deferred_;
  std::shared_ptr<StoredModel> source_, target_;
  ol::ModelProto source_snapshot_, target_snapshot_;
  ol::RewritingProto plan_;
  std::vector<DiffStep> steps_;
};

} // namespace

Napi::Value DiffModels(const Napi::CallbackInfo &info) {
  try {
    if (info.Length() != 2)
      throw std::runtime_error("Expected two native ONNX model handles.");
    auto *worker =
        new DiffWorker(info.Env(), UnwrapStoredModel(info[0]), UnwrapStoredModel(info[1]));
    const auto promise = worker->Promise();
    worker->Queue();
    return promise;
  } catch (const std::exception &error) {
    Napi::Error::New(info.Env(), error.what()).ThrowAsJavaScriptException();
    return info.Env().Undefined();
  }
}

} // namespace onnx_light_viewer
