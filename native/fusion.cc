/** @file Pattern enumeration, preview and application on independent snapshots. */
#include "model_io.h"
#include "model_storage.h"
#include "tensor_preview.h"
#include <algorithm>
#include <cmath>
#include <map>
#include <onnx_core/builder/graph_graph.h>
#include <onnx_core/builder/pattern_registry.h>
#include <onnx_extensions/patterns/dispatch_table.h>
#include <onnx_extensions/shapes/dispatch_table.h>
#include <onnx_proto/onnx_verify.h>
#include <stdexcept>
#include <tuple>

namespace onnx_light_viewer {
namespace {

using FusionNodeKey = std::tuple<std::string, std::string, std::string, std::string,
                                 std::vector<std::string>, std::vector<std::string>>;

FusionNodeKey NodeOrderKey(const ol::NodeProto &node) {
  std::vector<std::string> inputs, outputs;
  for (const auto &value : node.input())
    inputs.push_back(value.value());
  for (const auto &value : node.output())
    outputs.push_back(value.value());
  return {node.domain().value(), node.op_type().value(), node.overload().value(),
          node.name().value(),   std::move(inputs),      std::move(outputs)};
}

std::vector<size_t> FinalizedNodeOrder(const ol::ModelProto &model,
                                       const std::vector<FusionNodeKey> &inserted_order) {
  const auto &final_nodes = model.graph().node();
  if (final_nodes.size() != inserted_order.size())
    throw std::runtime_error("Model finalization changed the fusion node count.");
  std::map<FusionNodeKey, std::vector<size_t>> final_positions;
  for (size_t i = final_nodes.size(); i-- > 0;)
    final_positions[NodeOrderKey(final_nodes[i])].push_back(i);
  std::vector<size_t> permutation;
  for (size_t i = 0; i < inserted_order.size(); ++i) {
    auto found = final_positions.find(inserted_order[i]);
    if (found == final_positions.end() || found->second.empty())
      throw std::runtime_error("Model finalization changed fusion node identities.");
    const auto position = found->second.back();
    found->second.pop_back();
    permutation.push_back(position);
  }
  return permutation;
}

void FinalizeCapturedPlan(const ol::ModelProto &model, ol::RewritingProto &plan,
                          std::vector<std::vector<StoredModel::FusionMatch>> &history,
                          const std::vector<FusionNodeKey> &inserted_order) {
  const auto permutation = FinalizedNodeOrder(model, inserted_order);
  bool reordered = false;
  for (size_t i = 0; i < permutation.size(); ++i)
    reordered |= permutation[i] != i;
  if (!reordered)
    return;
  for (auto &previous : history)
    for (auto &match : previous) {
      for (auto &node : match.nodes)
        node = permutation.at(node);
      std::sort(match.nodes.begin(), match.nodes.end());
    }
  // Replay must establish the same finalized order before the next edit.
  const auto &final_nodes = model.graph().node();
  auto *reorder = plan.add_rewrites();
  reorder->set_pattern_name("Finalize fusion node order");
  for (size_t i = 0; i < final_nodes.size(); ++i) {
    reorder->add_matched_nodes(i);
    *reorder->add_added_nodes() = final_nodes[i].SharedCopy();
    reorder->add_added_nodes_positions(i);
  }
}

std::vector<size_t> TraceRewriting(const ol::LocalRewritingProto &edit,
                                   std::vector<FusionNodeKey> &nodes,
                                   std::vector<std::vector<StoredModel::FusionMatch>> &history) {
  if (!edit.graph_path().empty())
    return {};
  const std::set<size_t> removed(edit.matched_nodes().begin(), edit.matched_nodes().end());
  if (removed.size() > nodes.size())
    throw std::runtime_error("Invalid rewriting node count.");
  const size_t count = nodes.size() - removed.size() + edit.added_nodes_size();
  std::vector<std::optional<FusionNodeKey>> next(count);
  std::vector<size_t> added;
  for (size_t i = 0; i < edit.added_nodes().size(); ++i) {
    const auto position = edit.added_nodes_positions(i);
    if (position >= count || next[position])
      throw std::runtime_error("Invalid rewriting insertion position.");
    next[position] = NodeOrderKey(edit.added_nodes(i));
    added.push_back(position);
  }
  std::vector<std::vector<size_t>> destinations(nodes.size());
  size_t position = 0;
  for (size_t i = 0; i < nodes.size(); ++i) {
    if (removed.contains(i)) {
      for (const auto target : added)
        if (nodes[i] == *next[target])
          destinations[i].push_back(target);
      if (destinations[i].empty())
        destinations[i] = added;
    } else {
      while (position < count && next[position])
        ++position;
      if (position == count)
        throw std::runtime_error("Invalid rewriting node positions.");
      next[position] = nodes[i];
      destinations[i].push_back(position++);
    }
  }
  for (auto &operation : history)
    for (auto &match : operation) {
      std::set<size_t> mapped;
      for (const auto node : match.nodes)
        mapped.insert(destinations.at(node).begin(), destinations.at(node).end());
      match.nodes.assign(mapped.begin(), mapped.end());
    }
  nodes.clear();
  for (auto &node : next) {
    if (!node)
      throw std::runtime_error("Incomplete rewriting node order.");
    nodes.push_back(std::move(*node));
  }
  return added;
}

Napi::Object FusionNode(Napi::Env env, const ol::NodeProto &node) {
  auto result = Napi::Object::New(env);
  result.Set("opType", node.op_type().value());
  result.Set("domain", node.domain().value());
  result.Set("name", node.name().value());
  const auto names = [&](const auto &values) {
    auto array = Napi::Array::New(env, values.size());
    for (uint32_t i = 0; i < values.size(); ++i)
      array.Set(i, values[i].value());
    return array;
  };
  result.Set("inputs", names(node.input()));
  result.Set("outputs", names(node.output()));
  auto attributes = Napi::Array::New(env, node.attribute_size());
  for (uint32_t i = 0; i < static_cast<uint32_t>(node.attribute_size()); ++i) {
    auto value = Napi::Object::New(env);
    value.Set("name", node.attribute(i).name().value());
    value.Set("value", AttributePreview(node.attribute(i)));
    attributes.Set(i, value);
  }
  result.Set("attributes", attributes);
  return result;
}

class FusionWorker : public Napi::AsyncWorker {
public:
  FusionWorker(Napi::Env env, std::shared_ptr<StoredModel> source,
               std::vector<std::string> patterns, std::optional<std::vector<size_t>> anchors,
               std::string path, std::string scenario = "", bool all = false)
      : Napi::AsyncWorker(env), deferred_(Napi::Promise::Deferred::New(env)),
        source_(std::move(source)),
        canonical_(!scenario.empty() && source_->rewriting_base ? source_->rewriting_base
                                                                : source_),
        snapshot_(canonical_->model->SharedCopy()), patterns_(std::move(patterns)),
        anchors_(std::move(anchors)), path_(std::move(path)),
        weights_(canonical_->external_weights), history_(source_->fusion_history),
        scenario_(std::move(scenario)), all_(all) {}
  Napi::Promise Promise() const { return deferred_.Promise(); }
  void Execute() override {
    try {
      RestoreModelPayloads(snapshot_, *canonical_, path_);
      std::vector<ol::EncodedValueProto *> encoded;
      std::set<const ol::TensorProto *> layout_tensors;
      const auto tensors = ModelSaveTensors(snapshot_, &encoded, &layout_tensors);
      for (auto *value : encoded)
        if (value->has_data_location() && value->data_location() == ol::TensorProto::EXTERNAL) {
          if (path_.empty())
            throw std::runtime_error("Fusion needs a local source path for external weights.");
          LoadExternalTensor(*value, ModelFilePath(path_).parent_path());
        }
      for (auto *tensor : tensors)
        if ((!layout_tensors.contains(tensor) || tensor->has_data_location()) &&
            tensor->data_location() == ol::TensorProto::EXTERNAL) {
          if (path_.empty())
            throw std::runtime_error("Fusion needs a local source path for external weights.");
          auto reference = tensor->SharedCopy();
          LoadExternalTensor(*tensor, ModelFilePath(path_).parent_path());
          weights_.push_back({std::move(reference), tensor->SharedCopy()});
        }
      ol::VerifyModel(snapshot_);
      if (!scenario_.empty()) {
        ApplyScenario();
        return;
      }
      namespace builder = ol::core::builder;
      std::map<std::string, std::unique_ptr<builder::PatternOptimization>> patterns;
      for (const auto &name : patterns_)
        if (!patterns.contains(name))
          patterns.emplace(name, builder::CreateRegisteredPattern(name));
      builder::GraphBuilder graph(snapshot_);
      if (graph.Nodes().size() != snapshot_.graph().node().size())
        throw std::runtime_error("The native builder changed node indices before fusion.");
      for (size_t i = 0; i < graph.Nodes().size(); ++i) {
        const auto &node = graph.Nodes()[i];
        const auto &original = snapshot_.graph().node(i);
        if (node.op_type() != original.op_type() || node.domain() != original.domain() ||
            node.input_size() != original.input_size() ||
            node.output_size() != original.output_size() ||
            !std::equal(node.input().begin(), node.input().end(), original.input().begin()) ||
            !std::equal(node.output().begin(), node.output().end(), original.output().begin()))
          throw std::runtime_error("The native builder changed node indices before fusion.");
      }
      builder::GraphGraph index(graph,
                                std::vector<std::unique_ptr<builder::PatternOptimization>>{});
      std::vector<std::pair<builder::PatternOptimization *, builder::MatchResult>> accepted;
      std::set<size_t> removed;
      std::set<std::pair<std::string, size_t>> requested;
      if (anchors_) {
        for (size_t i = 0; i < anchors_->size(); ++i)
          requested.emplace(patterns_.size() == 1 ? patterns_[0] : patterns_[i], (*anchors_)[i]);
        if (requested.empty() || requested.size() != anchors_->size())
          throw std::runtime_error("Select distinct fusion matches.");
      }
      for (const auto &[name, pattern] : patterns) {
        const auto fast = pattern->FastOpType();
        std::optional<builder::GraphBuilder> trial;
        std::optional<builder::GraphGraph> trial_index;
        for (size_t i = 0; i < graph.Nodes().size(); ++i) {
          if ((anchors_ && !requested.contains({name, i})) ||
              (!fast.empty() && !fast.contains(graph.Nodes()[i].op_type().value())))
            continue;
          auto match = pattern->Match(index, graph.Nodes()[i]);
          if (!match.pattern)
            continue;
          std::vector<size_t> nodes;
          for (const auto *node : match.nodes)
            if (node) {
              const auto position = index.Position(*node);
              if (std::find(nodes.begin(), nodes.end(), position) == nodes.end())
                nodes.push_back(position);
            }
          if (nodes.empty())
            throw std::runtime_error("The pattern returned an empty fusion match.");
          if (anchors_) {
            for (const auto position : nodes)
              if (!removed.insert(position).second)
                throw std::runtime_error("Selected matches overlap. Select disjoint matches.");
            accepted.emplace_back(pattern.get(), std::move(match));
          } else {
            if (!trial) {
              trial.emplace(snapshot_);
              trial_index.emplace(*trial,
                                  std::vector<std::unique_ptr<builder::PatternOptimization>>{});
            }
            const auto trial_match = pattern->Match(*trial_index, trial->Nodes()[i]);
            if (!trial_match.pattern)
              throw std::runtime_error("The fusion match could not be reproduced.");
            matches_.push_back(
                {i, std::move(nodes), pattern->Apply(*trial_index, trial_match.nodes)});
          }
        }
      }
      if (!anchors_)
        return;
      if (accepted.size() != requested.size())
        throw std::runtime_error("The selected fusion preview is no longer applicable.");
      ol::RewritingProto plan;
      auto *edit = plan.add_rewrites();
      edit->set_pattern_name(patterns_.size() == 1 ? patterns_[0] : "Selected fusion patterns");
      for (const auto position : removed)
        edit->add_matched_nodes(position);
      const auto initializer_count = graph.Initializers().size();
      std::map<size_t, std::vector<std::pair<ol::NodeProto, size_t>>> insertions;
      std::vector<StoredModel::FusionMatch> operation;
      for (const auto &[pattern, match] : accepted) {
        const auto match_id = operation.size();
        operation.push_back({pattern->Name(), {}});
        size_t position = graph.Nodes().size();
        if (match.insert_at)
          position = index.Position(*match.insert_at);
        else
          for (const auto *node : match.nodes)
            if (node)
              position = std::min(position, index.Position(*node));
        auto replacement = pattern->Apply(index, match.nodes);
        for (const auto &node : replacement)
          insertions[position].emplace_back(node.SharedCopy(), match_id);
      }
      std::vector<std::vector<size_t>> destinations(graph.Nodes().size());
      std::vector<FusionNodeKey> inserted_order;
      size_t output_position = 0;
      for (size_t i = 0; i <= graph.Nodes().size(); ++i) {
        for (const auto &[node, match_id] : insertions[i]) {
          operation[match_id].nodes.push_back(output_position);
          inserted_order.push_back(NodeOrderKey(node));
          *edit->add_added_nodes() = node.SharedCopy();
          edit->add_added_nodes_positions(output_position++);
        }
        if (i < graph.Nodes().size() && !removed.contains(i)) {
          inserted_order.push_back(NodeOrderKey(graph.Nodes()[i]));
          destinations[i].push_back(output_position++);
        }
      }
      for (size_t i = 0; i < accepted.size(); ++i)
        for (const auto *node : accepted[i].second.nodes)
          if (node)
            destinations[index.Position(*node)] = operation[i].nodes;
      for (auto &previous : history_)
        for (auto &match : previous) {
          std::set<size_t> nodes;
          for (const auto node : match.nodes)
            nodes.insert(destinations.at(node).begin(), destinations.at(node).end());
          match.nodes.assign(nodes.begin(), nodes.end());
        }
      history_.push_back(std::move(operation));
      for (size_t i = initializer_count; i < graph.Initializers().size(); ++i) {
        *edit->add_added_initializers() = graph.Initializers()[i].SharedCopy();
        edit->add_added_initializer_positions(i);
      }
      auto declarations = graph.ToModel(snapshot_.ir_version());
      plan.ref_added_functions() = declarations.functions();
      plan.ref_required_opset_imports() = declarations.opset_import();
      builder::GraphBuilder original(snapshot_);
      auto rewritten = original.ApplyRewritings(plan);
      snapshot_ = rewritten.ToModel(snapshot_.ir_version());
      ol::VerifyModel(snapshot_);
      FinalizeCapturedPlan(snapshot_, plan, history_, inserted_order);
      plan_ = std::move(plan);
    } catch (const std::exception &error) {
      SetError(error.what());
    }
  }
  void OnOK() override {
    auto stored = std::make_shared<StoredModel>();
    stored->backing = source_;
    stored->model = std::make_shared<ol::ModelProto>(std::move(snapshot_));
    stored->file_name = source_->file_name;
    stored->file_path = path_;
    stored->external_weights = std::move(weights_);
    stored->fusion_base = source_->fusion_base;
    stored->fusion_previous = source_->fusion_previous;
    stored->fusion_history = std::move(history_);
    if (!scenario_.empty()) {
      stored->fusion_base = canonical_;
      stored->fusion_previous = source_;
      stored->fusion_plans.push_back(std::move(plan_));
      stored->rewriting_base = canonical_;
      stored->rewriting_name = scenario_;
      stored->rewriting_batches = std::move(batches_);
      deferred_.Resolve(ModelHandle(Env(), stored));
      return;
    }
    for (const auto &plan : source_->fusion_plans)
      stored->fusion_plans.push_back(plan.SharedCopy());
    if (anchors_) {
      stored->fusion_base = source_->fusion_base ? source_->fusion_base : source_;
      stored->fusion_previous = source_;
      stored->fusion_plans.push_back(std::move(plan_));
      deferred_.Resolve(ModelHandle(Env(), stored));
      return;
    }
    stored->rewriting_base = source_->rewriting_base;
    stored->rewriting_name = source_->rewriting_name;
    stored->rewriting_batches = source_->rewriting_batches;
    auto result = Napi::Object::New(Env());
    result.Set("model", ModelHandle(Env(), stored));
    auto matches = Napi::Array::New(Env(), matches_.size());
    for (uint32_t i = 0; i < matches_.size(); ++i) {
      auto item = Napi::Object::New(Env());
      item.Set("anchor", Napi::Number::New(Env(), matches_[i].anchor));
      auto nodes = Napi::Array::New(Env(), matches_[i].nodes.size());
      for (uint32_t j = 0; j < matches_[i].nodes.size(); ++j)
        nodes.Set(j, Napi::Number::New(Env(), matches_[i].nodes[j]));
      item.Set("nodes", nodes);
      auto replacement = Napi::Array::New(Env(), matches_[i].replacement.size());
      for (uint32_t j = 0; j < matches_[i].replacement.size(); ++j)
        replacement.Set(j, FusionNode(Env(), matches_[i].replacement[j]));
      item.Set("replacement", replacement);
      matches.Set(i, item);
    }
    result.Set("matches", matches);
    deferred_.Resolve(result);
  }
  void OnError(const Napi::Error &error) override { deferred_.Reject(error.Value()); }

private:
  void ApplyScenario() {
    if (!source_->fusion_history.empty() && source_->rewriting_name != scenario_)
      throw std::runtime_error("Unfuse current edits before choosing another rewriting scenario.");
    const auto found = snapshot_.rewritings().find(scenario_);
    if (found == snapshot_.rewritings().end())
      throw std::runtime_error("Unknown rewriting scenario: " + scenario_);
    const auto &scenario = found->second;
    const size_t applied =
        source_->rewriting_batches.empty() ? 0 : source_->rewriting_batches.back();
    if (applied >= scenario.rewrites().size())
      throw std::runtime_error("No remaining steps in this rewriting scenario.");
    const size_t target = all_ ? scenario.rewrites().size() : applied + 1;
    batches_ = source_->rewriting_batches;
    batches_.push_back(target);
    plan_ = scenario.SharedCopy();
    plan_.clear_rewrites();
    for (size_t i = 0; i < target; ++i)
      *plan_.add_rewrites() = scenario.rewrites(i).SharedCopy();
    auto rewritten = ol::core::builder::GraphBuilder(snapshot_).ApplyRewritings(plan_);
    std::vector<FusionNodeKey> order;
    for (const auto &node : snapshot_.graph().node())
      order.push_back(NodeOrderKey(node));
    history_.clear();
    size_t step = 0;
    for (const auto end : batches_) {
      history_.emplace_back();
      for (; step < end; ++step) {
        const auto &edit = scenario.rewrites(step);
        auto nodes = TraceRewriting(edit, order, history_);
        history_.back().push_back(
            {edit.pattern_name().empty() ? "Stored rewriting" : edit.pattern_name().value(),
             std::move(nodes)});
      }
    }
    snapshot_ = rewritten.ToModel(snapshot_.ir_version());
    ol::VerifyModel(snapshot_);
    FinalizeCapturedPlan(snapshot_, plan_, history_, order);
  }

  struct Match {
    size_t anchor;
    std::vector<size_t> nodes;
    ol::utils::RepeatedProtoField<ol::NodeProto> replacement;
  };
  Napi::Promise::Deferred deferred_;
  std::shared_ptr<StoredModel> source_;
  std::shared_ptr<StoredModel> canonical_;
  ol::ModelProto snapshot_;
  std::vector<std::string> patterns_;
  std::optional<std::vector<size_t>> anchors_;
  std::string path_;
  std::vector<StoredModel::ExternalWeight> weights_;
  std::vector<Match> matches_;
  ol::RewritingProto plan_;
  std::vector<std::vector<StoredModel::FusionMatch>> history_;
  std::string scenario_;
  bool all_;
  std::vector<size_t> batches_;
};

} // namespace

void AddFusionMethods(Napi::Env env, Napi::Object result, std::shared_ptr<StoredModel> model) {
  result.Set("fusionCount", Napi::Function::New(env, [model](const Napi::CallbackInfo &call) {
               return Napi::Number::New(call.Env(), model->fusion_history.size());
             }));
  result.Set("rewritingScenarios",
             Napi::Function::New(env, [model](const Napi::CallbackInfo &call) -> Napi::Value {
               try {
                 const auto base = model->rewriting_base ? model->rewriting_base
                                   : model->fusion_base  ? model->fusion_base
                                                         : model;
                 auto scenarios = Napi::Array::New(call.Env());
                 for (const auto &[name, plan] : base->model->rewritings()) {
                   auto item = Napi::Object::New(call.Env());
                   item.Set("name", name);
                   const size_t applied =
                       model->rewriting_name == name && !model->rewriting_batches.empty()
                           ? model->rewriting_batches.back()
                           : 0;
                   item.Set("applied", Napi::Number::New(call.Env(), applied));
                   const bool available =
                       (model->fusion_history.empty() || model->rewriting_name == name) &&
                       applied < plan.rewrites().size();
                   item.Set("available", available);
                   std::vector<std::vector<size_t>> selections(plan.rewrites().size());
                   if (model->rewriting_name == name) {
                     size_t index = 0;
                     for (const auto &operation : model->fusion_history)
                       for (const auto &match : operation)
                         selections.at(index++) = match.nodes;
                   }
                   if (available && plan.rewrites(applied).graph_path().empty()) {
                     std::vector<FusionNodeKey> order;
                     for (const auto &node : base->model->graph().node())
                       order.push_back(NodeOrderKey(node));
                     std::vector<std::vector<StoredModel::FusionMatch>> history;
                     for (size_t i = 0; i < applied; ++i)
                       TraceRewriting(plan.rewrites(i), order, history);
                     const auto permutation = FinalizedNodeOrder(*model->model, order);
                     const auto &matched = plan.rewrites(applied).matched_nodes();
                     if (std::all_of(matched.begin(), matched.end(),
                                     [&](const auto node) { return node < permutation.size(); }))
                       for (const auto node : matched)
                         selections[applied].push_back(permutation[node]);
                     std::sort(selections[applied].begin(), selections[applied].end());
                   }
                   auto steps = Napi::Array::New(call.Env(), plan.rewrites_size());
                   for (uint32_t i = 0; i < plan.rewrites().size(); ++i) {
                     auto step = Napi::Object::New(call.Env());
                     step.Set("pattern", plan.rewrites(i).pattern_name().empty()
                                             ? "Stored rewriting"
                                             : plan.rewrites(i).pattern_name().value());
                     auto path = Napi::Array::New(call.Env(), plan.rewrites(i).graph_path_size());
                     for (uint32_t j = 0; j < plan.rewrites(i).graph_path().size(); ++j)
                       path.Set(j, plan.rewrites(i).graph_path(j).value());
                     step.Set("path", path);
                     auto nodes = Napi::Array::New(call.Env(), selections[i].size());
                     for (uint32_t j = 0; j < selections[i].size(); ++j)
                       nodes.Set(j, Napi::Number::New(call.Env(), selections[i][j]));
                     step.Set("nodes", nodes);
                     steps.Set(i, step);
                   }
                   item.Set("steps", steps);
                   scenarios.Set(scenarios.Length(), item);
                 }
                 return scenarios;
               } catch (const std::exception &error) {
                 Napi::Error::New(call.Env(), error.what()).ThrowAsJavaScriptException();
                 return call.Env().Undefined();
               }
             }));
  result.Set(
      "applyRewritingScenario",
      Napi::Function::New(env, [model](const Napi::CallbackInfo &call) -> Napi::Value {
        try {
          if (!call[0].IsString() || call[0].As<Napi::String>().Utf8Value().empty() ||
              !call[1].IsBoolean() || (!call[2].IsUndefined() && !call[2].IsString()))
            throw std::runtime_error(
                "Expected a rewriting scenario, an all-steps choice and optional source path.");
          const auto path =
              call[2].IsString() ? call[2].As<Napi::String>().Utf8Value() : model->file_path;
          if (path.find('\0') != std::string::npos)
            throw std::runtime_error("Invalid source model file path.");
          ol::onnx_shapes::RegisterShapeFunctions();
          auto worker = std::make_unique<FusionWorker>(
              call.Env(), model, std::vector<std::string>{}, std::nullopt, path,
              call[0].As<Napi::String>().Utf8Value(), call[1].As<Napi::Boolean>().Value());
          const auto promise = worker->Promise();
          worker->Queue();
          worker.release();
          return promise;
        } catch (const std::exception &error) {
          Napi::Error::New(call.Env(), error.what()).ThrowAsJavaScriptException();
          return call.Env().Undefined();
        }
      }));
  result.Set("fusionHistory", Napi::Function::New(env, [model](const Napi::CallbackInfo &call) {
               auto operations = Napi::Array::New(call.Env(), model->fusion_history.size());
               for (uint32_t i = 0; i < model->fusion_history.size(); ++i) {
                 auto operation = Napi::Object::New(call.Env());
                 const auto &history = model->fusion_history[i];
                 auto matches = Napi::Array::New(call.Env(), history.size());
                 for (uint32_t j = 0; j < history.size(); ++j) {
                   auto match = Napi::Object::New(call.Env());
                   match.Set("pattern", history[j].pattern);
                   auto nodes = Napi::Array::New(call.Env(), history[j].nodes.size());
                   for (uint32_t k = 0; k < history[j].nodes.size(); ++k)
                     nodes.Set(k, Napi::Number::New(call.Env(), history[j].nodes[k]));
                   match.Set("nodes", nodes);
                   matches.Set(j, match);
                 }
                 operation.Set("matches", matches);
                 operations.Set(i, operation);
               }
               return operations;
             }));
  result.Set("undoFusion",
             Napi::Function::New(env, [model](const Napi::CallbackInfo &call) -> Napi::Value {
               try {
                 if (!model->fusion_previous)
                   throw std::runtime_error("No fusion operation to undo.");
                 return ModelHandle(call.Env(), CloneStoredModel(model->fusion_previous));
               } catch (const std::exception &error) {
                 Napi::Error::New(call.Env(), error.what()).ThrowAsJavaScriptException();
                 return call.Env().Undefined();
               }
             }));
  result.Set("fusionPatterns",
             Napi::Function::New(env, [model](const Napi::CallbackInfo &call) -> Napi::Value {
               try {
                 ol::onnx_patterns::RegisterPatterns();
                 std::set<std::string> operators;
                 for (const auto &node : model->model->graph().node())
                   operators.insert(node.op_type().value());
                 auto values = Napi::Array::New(call.Env());
                 uint32_t i = 0;
                 for (const auto &pattern : ol::core::builder::CreateRegisteredPatterns()) {
                   const auto fast = pattern->FastOpType();
                   if (fast.empty() || std::any_of(fast.begin(), fast.end(), [&](const auto &name) {
                         return operators.contains(name);
                       }))
                     values.Set(i++, pattern->Name());
                 }
                 return values;
               } catch (const std::exception &error) {
                 Napi::Error::New(call.Env(), error.what()).ThrowAsJavaScriptException();
                 return call.Env().Undefined();
               }
             }));
  for (const bool apply : {false, true})
    result.Set(
        apply ? "applyFusion" : "previewFusion",
        Napi::Function::New(env, [model, apply](const Napi::CallbackInfo &call) -> Napi::Value {
          try {
            if ((!call[0].IsString() && !(apply && call[0].IsArray())) ||
                (apply && !call[1].IsArray()) ||
                (!apply && !call[1].IsUndefined() && !call[1].IsString()))
              throw std::runtime_error(
                  "Expected a fusion pattern and selected match indices or source path.");
            ol::onnx_patterns::RegisterPatterns();
            ol::onnx_shapes::RegisterShapeFunctions();
            std::vector<std::string> patterns;
            if (call[0].IsString())
              patterns.push_back(call[0].As<Napi::String>().Utf8Value());
            else {
              const auto values = call[0].As<Napi::Array>();
              if (!values.Length() || values.Length() != call[1].As<Napi::Array>().Length())
                throw std::runtime_error("Expected one fusion pattern per selected match.");
              for (uint32_t i = 0; i < values.Length(); ++i) {
                if (!values.Get(i).IsString())
                  throw std::runtime_error("Invalid fusion pattern.");
                patterns.push_back(values.Get(i).As<Napi::String>().Utf8Value());
              }
            }
            std::optional<std::vector<size_t>> anchors;
            if (apply) {
              anchors.emplace();
              const auto values = call[1].As<Napi::Array>();
              for (uint32_t i = 0; i < values.Length(); ++i) {
                const auto value = values.Get(i);
                if (!value.IsNumber())
                  throw std::runtime_error("Invalid fusion match index.");
                const double index = value.As<Napi::Number>().DoubleValue();
                if (!std::isfinite(index) || index < 0 || std::floor(index) != index ||
                    index >= model->model->graph().node().size())
                  throw std::runtime_error("Invalid fusion match index.");
                anchors->push_back(static_cast<size_t>(index));
              }
            }
            const auto path = !apply && call[1].IsString() ? call[1].As<Napi::String>().Utf8Value()
                                                           : model->file_path;
            if (path.find('\0') != std::string::npos)
              throw std::runtime_error("Invalid source model file path.");
            auto worker = std::make_unique<FusionWorker>(call.Env(), model, std::move(patterns),
                                                         std::move(anchors), path);
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
