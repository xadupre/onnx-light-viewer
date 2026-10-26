/** @file Pattern enumeration, preview and application on independent snapshots. */
#include "model_io.h"
#include "model_storage.h"
#include "rewriting_trace.h"
#include "tensor_preview.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <onnx_core/builder/graph_graph.h>
#include <onnx_core/builder/pattern_registry.h>
#include <onnx_extensions/kernels/kernel_dispatch_table.h>
#include <onnx_extensions/patterns/dispatch_table.h>
#include <onnx_extensions/shapes/dispatch_table.h>
#include <onnx_proto/onnx_verify.h>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

namespace onnx_light_viewer {
namespace {

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
                          const std::vector<FusionNodeKey> &inserted_order,
                          const ol::core::builder::GraphBuilder &builder) {
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
    reorder->ref_added_nodes()[i].ref_attribute() = builder.Nodes()[i].attribute();
    reorder->add_added_nodes_positions(i);
  }
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

using TransformationScopes = std::map<std::vector<std::string>, ol::GraphProto>;

TransformationScopes CaptureScopes(const ol::core::builder::GraphBuilder &builder) {
  // Keep native *_ref attributes so parent replacements reuse existing subgraphs.
  TransformationScopes scopes;
  const auto visit = [&](const auto &self, const auto &scope,
                         const std::vector<std::string> &path) -> void {
    auto &graph = scopes[path];
    for (const auto &node : scope.Nodes())
      *graph.add_node() = node.SharedCopy();
    for (const auto &tensor : scope.Initializers())
      *graph.add_initializer() = tensor.SharedCopy();
    for (const auto &tensor : scope.ExtendedInitializers())
      *graph.add_extended_initializer() = tensor.SharedCopy();
    for (const auto &child : scope.Subgraphs()) {
      auto nested = path;
      nested.push_back(child->name());
      self(self, *child, nested);
    }
  };
  visit(visit, builder, {});
  return scopes;
}

bool CaptureTransformation(const ol::GraphProto &original, const ol::GraphProto &updated,
                           const std::string &name, ol::RewritingProto &plan,
                           const std::vector<std::string> &path) {
  ol::LocalRewritingProto edit;
  edit.set_pattern_name(name);
  for (const auto &scope : path)
    edit.add_graph_path(scope);
  std::map<FusionNodeKey, std::vector<size_t>> positions;
  for (size_t i = 0; i < original.node().size(); ++i)
    positions[NodeOrderKey(original.node(i))].push_back(i);
  std::set<size_t> retained;
  size_t first = 0;
  for (size_t i = 0; i < updated.node().size(); ++i) {
    const auto &node = updated.node(i);
    const auto &candidates = positions[NodeOrderKey(node)];
    const auto found = std::find_if(candidates.begin(), candidates.end(), [&](size_t index) {
      return index >= first && original.node(index).Equals(node);
    });
    if (found != candidates.end()) {
      retained.insert(*found);
      first = *found + 1;
    } else {
      *edit.add_added_nodes() = node.SharedCopy();
      edit.add_added_nodes_positions(i);
    }
  }
  for (size_t i = 0; i < original.node().size(); ++i)
    if (!retained.contains(i))
      edit.add_matched_nodes(i);
  std::set<size_t> retained_initializers;
  std::unordered_map<std::string, size_t> initializer_positions;
  initializer_positions.reserve(original.initializer().size());
  for (size_t i = 0; i < original.initializer().size(); ++i)
    initializer_positions.emplace(original.initializer(i).name().value(), i);
  size_t first_initializer = 0;
  for (size_t i = 0; i < updated.initializer().size(); ++i) {
    const auto &tensor = updated.initializer(i);
    const auto found = initializer_positions.find(tensor.name().value());
    if (found != initializer_positions.end()) {
      const auto position = found->second;
      if (position < first_initializer || !original.initializer(position).Equals(tensor))
        throw std::runtime_error("Cannot capture " + name + ": existing initializer " +
                                 tensor.name().value() + " changed or moved.");
      retained_initializers.insert(position);
      first_initializer = position + 1;
    } else {
      *edit.add_added_initializers() = tensor.SharedCopy();
      edit.add_added_initializer_positions(i);
    }
  }
  for (size_t i = 0; i < original.initializer().size(); ++i)
    if (!retained_initializers.contains(i))
      edit.add_removed_initializers(i);
  size_t next = 0;
  for (size_t i = 0; i < original.extended_initializer().size(); ++i) {
    if (next < updated.extended_initializer().size() &&
        original.extended_initializer(i).Equals(updated.extended_initializer(next))) {
      ++next;
    } else {
      edit.add_removed_extended_initializers(i);
    }
  }
  if (next != updated.extended_initializer().size())
    throw std::runtime_error("Optimization introduced an unsupported extended initializer.");
  if (edit.matched_nodes().empty() && edit.added_nodes().empty() &&
      edit.removed_initializers().empty() && edit.added_initializers().empty() &&
      edit.removed_extended_initializers().empty())
    return false;
  *plan.add_rewrites() = std::move(edit);
  return true;
}

void PreserveFunctionBodies(ol::ModelProto &model, const ol::ModelProto &original) {
  for (auto &function : model.ref_functions())
    for (const auto &source : original.functions())
      if (function.name() == source.name() && function.domain() == source.domain() &&
          function.overload() == source.overload()) {
        function = source.SharedCopy();
        break;
      }
}

bool IsGlobalPass(const std::string &name) {
  return name == "IdentityRemoval" || name == "ConstantFolding" || name == "RemoveUnusedNodes";
}

Napi::Object OptimizationReportValue(Napi::Env env,
                                     const ol::core::builder::OptimizationReport &report,
                                     int64_t wall_time_ns) {
  auto result = Napi::Object::New(env);
  result.Set("iterations", Napi::Number::New(env, report.iterations));
  result.Set("rewrites", Napi::Number::New(env, report.rewrites));
  result.Set("totalTimeNs", Napi::Number::New(env, report.TotalTimeNs()));
  result.Set("wallTimeNs", Napi::Number::New(env, wall_time_ns));
  result.Set("matchingTimeNs", Napi::Number::New(env, report.matching_time_ns));
  result.Set("rewritingTimeNs", Napi::Number::New(env, report.rewriting_time_ns));
  result.Set("cleanupTimeNs", Napi::Number::New(env, report.cleanup_time_ns));
  result.Set("constantFoldingTimeNs", Napi::Number::New(env, report.constant_folding_time_ns));
  result.Set("subgraphOptimizationTimeNs",
             Napi::Number::New(env, report.subgraph_optimization_time_ns));
  auto patterns = Napi::Array::New(env, report.patterns.size());
  for (uint32_t i = 0; i < report.patterns.size(); ++i) {
    const auto &pattern = report.patterns[i];
    auto item = Napi::Object::New(env);
    item.Set("name", pattern.pattern_name);
    item.Set("attempts", Napi::Number::New(env, pattern.attempts));
    item.Set("matches", Napi::Number::New(env, pattern.matches));
    item.Set("matchTimeNs", Napi::Number::New(env, pattern.match_time_ns));
    item.Set("applyTimeNs", Napi::Number::New(env, pattern.apply_time_ns));
    auto reasons = Napi::Array::New(env, pattern.no_matches.size());
    for (uint32_t j = 0; j < pattern.no_matches.size(); ++j) {
      const auto &rejection = pattern.no_matches[j];
      auto reason = Napi::Object::New(env);
      reason.Set("reason", rejection.reason);
      reason.Set("sourceFile", rejection.source_file);
      reason.Set("sourceLine", rejection.source_line);
      reason.Set("occurrences", Napi::Number::New(env, rejection.occurrences));
      reasons.Set(j, reason);
    }
    item.Set("noMatches", reasons);
    patterns.Set(i, item);
  }
  result.Set("patterns", patterns);
  auto subgraphs = Napi::Array::New(env, report.subgraphs.size());
  for (uint32_t i = 0; i < report.subgraphs.size(); ++i) {
    const auto &subgraph = report.subgraphs[i];
    auto item = Napi::Object::New(env);
    auto path = Napi::Array::New(env, subgraph.graph_path.size());
    for (uint32_t j = 0; j < subgraph.graph_path.size(); ++j)
      path.Set(j, subgraph.graph_path[j]);
    item.Set("path", path);
    item.Set("iterations", Napi::Number::New(env, subgraph.iterations));
    item.Set("rewrites", Napi::Number::New(env, subgraph.rewrites));
    item.Set("elapsedTimeNs", Napi::Number::New(env, subgraph.elapsed_time_ns));
    subgraphs.Set(i, item);
  }
  result.Set("subgraphs", subgraphs);
  return result;
}

class FusionWorker : public Napi::AsyncWorker {
public:
  FusionWorker(Napi::Env env, std::shared_ptr<StoredModel> source,
               std::vector<std::string> patterns, std::optional<std::vector<size_t>> anchors,
               std::string path, std::string scenario = "", bool all = false,
               std::string operation = "", std::optional<size_t> diagnostic_anchor = std::nullopt)
      : Napi::AsyncWorker(env), started_(std::chrono::steady_clock::now()),
        deferred_(Napi::Promise::Deferred::New(env)), source_(std::move(source)),
        canonical_(!scenario.empty() && source_->rewriting_base ? source_->rewriting_base
                                                                : source_),
        snapshot_(canonical_->model->SharedCopy()), patterns_(std::move(patterns)),
        anchors_(std::move(anchors)), path_(std::move(path)),
        weights_(canonical_->external_weights), history_(source_->fusion_history),
        scenario_(std::move(scenario)), all_(all), operation_(std::move(operation)),
        diagnostic_anchor_(diagnostic_anchor) {}
  Napi::Promise Promise() const { return deferred_.Promise(); }
  void Execute() override {
    try {
      if (diagnostic_anchor_) {
        const auto &node = snapshot_.graph().node(*diagnostic_anchor_);
        const auto pattern = ol::core::builder::CreateRegisteredPattern(patterns_[0]);
        const auto fast = pattern->FastOpType();
        if (!fast.empty() && !fast.contains(node.op_type().value())) {
          diagnostic_status_ = "op-type-mismatch";
          diagnostic_reason_ = "Operator " + node.op_type().value() +
                               " is incompatible with this pattern's FastOpType anchors:";
          for (const auto &name : fast)
            diagnostic_reason_ += " " + name;
          return;
        }
      }
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
      if (!operation_.empty()) {
        ApplyTransformation();
        return;
      }
      if (!scenario_.empty()) {
        ApplyScenario();
        return;
      }
      if (std::any_of(patterns_.begin(), patterns_.end(), IsGlobalPass)) {
        if (patterns_.size() != 1 || (anchors_ && *anchors_ != std::vector<size_t>{0}))
          throw std::runtime_error("Apply a global pass individually.");
        const auto original = snapshot_.SharedCopy();
        operation_ = patterns_[0];
        ApplyTransformation();
        if (anchors_ && plan_.rewrites().empty())
          throw std::runtime_error("The selected fusion preview is no longer applicable.");
        if (!anchors_) {
          snapshot_ = original.SharedCopy();
          history_ = source_->fusion_history;
          operation_.clear();
          plan_.clear_rewrites();
        }
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
      if (diagnostic_anchor_) {
        const auto match =
            patterns.at(patterns_[0])->Match(index, graph.Nodes()[*diagnostic_anchor_]);
        if (match.pattern) {
          diagnostic_status_ = "matched";
          diagnostic_reason_ = "The pattern matches this node.";
        } else if (match.no_match) {
          diagnostic_status_ = match.no_match->reason.empty() ? "unavailable" : "rejected";
          diagnostic_reason_ = match.no_match->reason.empty()
                                   ? "No match. This pattern did not supply a rejection reason."
                                   : match.no_match->reason;
          diagnostic_file_ = match.no_match->source_file;
          diagnostic_line_ = match.no_match->source_line;
          if (match.no_match->candidate)
            diagnostic_candidate_ = index.Position(*match.no_match->candidate);
        } else {
          diagnostic_status_ = "unavailable";
          diagnostic_reason_ = "No match. This pattern did not supply a rejection diagnostic.";
        }
        return;
      }
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
      FinalizeCapturedPlan(snapshot_, plan, history_, inserted_order, rewritten);
      plan_ = std::move(plan);
    } catch (const std::exception &error) {
      SetError(error.what());
    }
  }
  void OnOK() override {
    if (diagnostic_anchor_) {
      auto result = Napi::Object::New(Env());
      result.Set("pattern", patterns_[0]);
      result.Set("anchor", Napi::Number::New(Env(), *diagnostic_anchor_));
      result.Set("status", diagnostic_status_);
      result.Set("reason", diagnostic_reason_);
      if (diagnostic_candidate_)
        result.Set("candidate", Napi::Number::New(Env(), *diagnostic_candidate_));
      if (!diagnostic_file_.empty()) {
        auto source = Napi::Object::New(Env());
        source.Set("file", diagnostic_file_);
        source.Set("line", diagnostic_line_);
        result.Set("source", source);
      }
      deferred_.Resolve(result);
      return;
    }
    if (!operation_.empty() && plan_.rewrites().empty()) {
      auto stored = CloneStoredModel(source_);
      if (optimization_report_)
        stored->optimization_report = std::move(optimization_report_);
      ResolveModel(stored);
      return;
    }
    auto stored = std::make_shared<StoredModel>();
    stored->backing = source_;
    stored->model = std::make_shared<ol::ModelProto>(std::move(snapshot_));
    stored->file_name = source_->file_name;
    stored->file_path = path_;
    stored->external_weights = std::move(weights_);
    stored->fusion_base = source_->fusion_base;
    stored->fusion_previous = source_->fusion_previous;
    stored->fusion_history = std::move(history_);
    stored->optimization_report = std::move(optimization_report_);
    if (!scenario_.empty()) {
      stored->fusion_base = canonical_;
      stored->fusion_previous = source_;
      stored->fusion_plans.push_back(std::move(plan_));
      stored->rewriting_base = canonical_;
      stored->rewriting_name = scenario_;
      stored->rewriting_batches = std::move(batches_);
      ResolveModel(stored);
      return;
    }
    for (const auto &plan : source_->fusion_plans)
      stored->fusion_plans.push_back(plan.SharedCopy());
    if (anchors_ || !operation_.empty()) {
      stored->fusion_base = source_->fusion_base ? source_->fusion_base : source_;
      stored->fusion_previous = source_;
      stored->fusion_plans.push_back(std::move(plan_));
      ResolveModel(stored);
      return;
    }
    stored->rewriting_base = source_->rewriting_base;
    stored->rewriting_name = source_->rewriting_name;
    stored->rewriting_batches = source_->rewriting_batches;
    stored->optimization_report = source_->optimization_report;
    stored->optimization_wall_time_ns = source_->optimization_wall_time_ns;
    auto result = Napi::Object::New(Env());
    result.Set("model", ModelHandle(Env(), stored));
    auto matches = Napi::Array::New(Env(), matches_.size());
    for (uint32_t i = 0; i < matches_.size(); ++i) {
      auto item = Napi::Object::New(Env());
      item.Set("anchor", Napi::Number::New(Env(), matches_[i].anchor));
      if (patterns_.size() == 1 && IsGlobalPass(patterns_[0]))
        item.Set("global", true);
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
  void ApplyTransformation() {
    namespace builder = ol::core::builder;
    const auto original = snapshot_.SharedCopy();
    builder::GraphBuilder graph(snapshot_);
    auto scopes = CaptureScopes(graph);
    const auto capture = [&](const std::string &name) {
      auto updated = snapshot_.SharedCopy();
      *updated.mutable_graph() = graph.BuildGraph();
      auto updated_scopes = CaptureScopes(graph);
      // Rewrite children before removing enclosing values they previously captured.
      for (auto scope = updated_scopes.rbegin(); scope != updated_scopes.rend(); ++scope) {
        const auto before = scopes.find(scope->first);
        if (before == scopes.end())
          throw std::runtime_error("Global cleanup introduced an unsupported subgraph.");
        CaptureTransformation(before->second, scope->second, name, plan_, scope->first);
      }
      if (!anchors_ && patterns_.size() == 1 && IsGlobalPass(patterns_[0]) &&
          !plan_.rewrites().empty()) {
        Match match;
        match.anchor = 0;
        std::set<size_t> affected;
        std::set<FusionNodeKey> replacements;
        const auto &root = scopes.at({});
        std::map<std::string, std::vector<size_t>> owners;
        for (size_t i = 0; i < root.node().size(); ++i)
          for (const auto &attribute : root.node(i).attribute())
            if (attribute.name().value().ends_with("_ref")) {
              if (attribute.type() == ol::AttributeProto::STRING)
                owners[attribute.s().value()].push_back(i);
              else if (attribute.type() == ol::AttributeProto::STRINGS)
                for (const auto &value : attribute.strings())
                  owners[std::string(value)].push_back(i);
            }
        for (const auto &edit : plan_.rewrites()) {
          if (edit.graph_path().empty()) {
            affected.insert(edit.matched_nodes().begin(), edit.matched_nodes().end());
            for (const auto &node : edit.added_nodes())
              replacements.insert(NodeOrderKey(node));
          } else {
            const auto &child = edit.graph_path(0).value();
            const auto found = owners.find(child);
            if (found != owners.end())
              for (const auto i : found->second) {
                affected.insert(i);
                replacements.insert(NodeOrderKey(root.node(i)));
              }
          }
        }
        match.nodes.assign(affected.begin(), affected.end());
        for (const auto &node : updated.graph().node())
          if (replacements.contains(NodeOrderKey(node)))
            match.replacement.push_back(node.SharedCopy());
        matches_.push_back(std::move(match));
      }
      scopes = std::move(updated_scopes);
      snapshot_ = std::move(updated);
      graph = builder::GraphBuilder(snapshot_);
    };
    if (operation_ == "IdentityRemoval") {
      if (graph.RemoveIdentityNodes())
        capture("IdentityRemoval");
    } else if (operation_ == "ConstantFolding") {
      if (graph.ConstantFold())
        capture("ConstantFolding");
    } else if (operation_ == "RemoveUnusedNodes") {
      graph.RemoveUnusedNodes();
      capture("RemoveUnusedNodes");
    } else {
      {
        builder::GraphGraph canonicalizer(
            graph, std::vector<std::unique_ptr<builder::PatternOptimization>>{});
        canonicalizer.Canonicalize();
      }
      capture("Canonicalize");
      std::vector<std::unique_ptr<builder::PatternOptimization>> chosen;
      for (const auto &name : patterns_)
        chosen.push_back(builder::CreateRegisteredPattern(name));
      builder::GraphGraph optimizer(graph, std::move(chosen));
      optimization_report_ = std::make_shared<builder::OptimizationReport>();
      auto rewrites = optimizer.Optimize(-1, optimization_report_.get());
      plan_.ref_added_functions() = rewrites.added_functions();
      plan_.ref_required_opset_imports() = rewrites.required_opset_imports();
      for (const auto &edit : rewrites.rewrites())
        *plan_.add_rewrites() = edit.SharedCopy();
      const auto optimized_graph = graph.BuildGraph();
      scopes = CaptureScopes(graph);
      snapshot_ = graph.ToModel(snapshot_.ir_version());
      *snapshot_.mutable_graph() = optimized_graph;
      PreserveFunctionBodies(snapshot_, original);
      {
        builder::GraphGraph canonicalizer(
            graph, std::vector<std::unique_ptr<builder::PatternOptimization>>{});
        canonicalizer.Canonicalize();
      }
      capture("Canonicalize");
    }
    if (plan_.rewrites().empty())
      return;
    if (!plan_.added_functions().empty()) {
      ol::ModelProto declarations;
      declarations.mutable_graph()->set_name("Optimization function declarations");
      declarations.ref_struct_types() = snapshot_.struct_types();
      declarations.ref_opset_import() = snapshot_.opset_import();
      declarations.ref_functions() = snapshot_.functions();
      for (const auto &function : plan_.added_functions()) {
        auto found = std::find_if(declarations.ref_functions().begin(),
                                  declarations.ref_functions().end(), [&](const auto &other) {
                                    return function.name() == other.name() &&
                                           function.domain() == other.domain() &&
                                           function.overload() == other.overload();
                                  });
        if (found == declarations.ref_functions().end())
          *declarations.add_functions() = function.SharedCopy();
        else
          *found = function.SharedCopy();
      }
      const auto normalized = builder::GraphBuilder(declarations).ToModel(original.ir_version());
      for (auto &function : plan_.ref_added_functions()) {
        const auto found = std::find_if(
            normalized.functions().begin(), normalized.functions().end(), [&](const auto &other) {
              return function.name() == other.name() && function.domain() == other.domain() &&
                     function.overload() == other.overload();
            });
        if (found == normalized.functions().end())
          throw std::runtime_error("Optimization lost a function declaration.");
        function = found->SharedCopy();
      }
    }
    for (const auto &function : snapshot_.functions()) {
      const auto found = std::find_if(
          plan_.added_functions().begin(), plan_.added_functions().end(), [&](const auto &other) {
            return function.name() == other.name() && function.domain() == other.domain() &&
                   function.overload() == other.overload();
          });
      if (found == plan_.added_functions().end())
        *plan_.add_added_functions() = function.SharedCopy();
      else {
        auto expected = function.SharedCopy();
        auto actual = found->SharedCopy();
        const auto sort_imports = [](auto &definition) {
          std::vector<ol::OperatorSetIdProto> imports(definition.opset_import().begin(),
                                                      definition.opset_import().end());
          std::vector<std::pair<std::string, size_t>> order;
          for (size_t i = 0; i < imports.size(); ++i)
            order.emplace_back(imports[i].domain().value(), i);
          std::sort(order.begin(), order.end());
          for (size_t i = 0; i < order.size(); ++i)
            definition.ref_opset_import()[i] = std::move(imports[order[i].second]);
        };
        sort_imports(expected);
        sort_imports(actual);
        std::string difference;
        if (!expected.Equals(actual, &difference))
          throw std::runtime_error("Optimization produced conflicting function declarations: " +
                                   function.name().value() + ": " + difference);
      }
    }
    for (const auto &opset : snapshot_.opset_import()) {
      const auto found = std::find_if(
          plan_.required_opset_imports().begin(), plan_.required_opset_imports().end(),
          [&](const auto &other) { return opset.domain().value() == other.domain().value(); });
      if (found == plan_.required_opset_imports().end())
        *plan_.add_required_opset_imports() = opset;
      else if (opset.version() != found->version())
        throw std::runtime_error("Optimization produced conflicting opset declarations.");
    }
    RewritingTrace trace(original.graph());
    history_.emplace_back();
    for (const auto &edit : plan_.rewrites()) {
      auto nodes = trace.Apply(edit);
      history_.back().push_back({edit.pattern_name().value(), std::move(nodes)});
    }
    const auto order = trace.Finish(history_);
    auto replayed = builder::GraphBuilder(original).ApplyRewritings(plan_);
    snapshot_ = replayed.ToModel(original.ir_version());
    ol::VerifyModel(snapshot_);
    FinalizeCapturedPlan(snapshot_, plan_, history_, order, replayed);
  }

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
    FinalizeCapturedPlan(snapshot_, plan_, history_, order, rewritten);
  }

  void ResolveModel(const std::shared_ptr<StoredModel> &stored) {
    auto result = ModelHandle(Env(), stored);
    if (operation_ == "Optimize" && stored->optimization_report)
      stored->optimization_wall_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                              std::chrono::steady_clock::now() - started_)
                                              .count();
    deferred_.Resolve(result);
  }

  struct Match {
    size_t anchor;
    std::vector<size_t> nodes;
    ol::utils::RepeatedProtoField<ol::NodeProto> replacement;
  };
  const std::chrono::steady_clock::time_point started_;
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
  std::string operation_;
  std::shared_ptr<ol::core::builder::OptimizationReport> optimization_report_;
  std::optional<size_t> diagnostic_anchor_;
  std::optional<size_t> diagnostic_candidate_;
  std::string diagnostic_status_;
  std::string diagnostic_reason_;
  std::string diagnostic_file_;
  uint_least32_t diagnostic_line_ = 0;
};

} // namespace

void AddFusionMethods(Napi::Env env, Napi::Object result, std::shared_ptr<StoredModel> model) {
  result.Set("optimizationReport",
             Napi::Function::New(env, [model](const Napi::CallbackInfo &call) -> Napi::Value {
               if (!model->optimization_report)
                 return call.Env().Null();
               return OptimizationReportValue(call.Env(), *model->optimization_report,
                                              model->optimization_wall_time_ns);
             }));
  result.Set("explainFusion",
             Napi::Function::New(env, [model](const Napi::CallbackInfo &call) -> Napi::Value {
               try {
                 if (!call[0].IsString() || !call[1].IsNumber() ||
                     (!call[2].IsUndefined() && !call[2].IsString()))
                   throw std::runtime_error(
                       "Expected a native pattern, node index and optional source path.");
                 const auto anchor = call[1].As<Napi::Number>().DoubleValue();
                 if (!std::isfinite(anchor) || anchor < 0 || std::floor(anchor) != anchor ||
                     anchor >= model->model->graph().node().size())
                   throw std::runtime_error("Invalid diagnostic node index.");
                 ol::onnx_patterns::RegisterPatterns();
                 ol::onnx_shapes::RegisterShapeFunctions();
                 const auto pattern = call[0].As<Napi::String>().Utf8Value();
                 ol::core::builder::CreateRegisteredPattern(pattern);
                 const auto path =
                     call[2].IsString() ? call[2].As<Napi::String>().Utf8Value() : model->file_path;
                 if (path.find('\0') != std::string::npos)
                   throw std::runtime_error("Invalid source model file path.");
                 auto worker = std::make_unique<FusionWorker>(
                     call.Env(), model, std::vector<std::string>{pattern}, std::nullopt, path, "",
                     false, "", static_cast<size_t>(anchor));
                 const auto promise = worker->Promise();
                 worker->Queue();
                 worker.release();
                 return promise;
               } catch (const std::exception &error) {
                 Napi::Error::New(call.Env(), error.what()).ThrowAsJavaScriptException();
                 return call.Env().Undefined();
               }
             }));
  result.Set("optimizationPatterns", Napi::Function::New(env, [](const Napi::CallbackInfo &call) {
               ol::onnx_patterns::RegisterPatterns();
               auto values = Napi::Array::New(call.Env());
               for (const auto &pattern : ol::core::builder::CreateRegisteredPatterns()) {
                 auto item = Napi::Object::New(call.Env());
                 item.Set("name", pattern->Name());
                 item.Set("priority", pattern->priority);
                 item.Set("device", ol::core::symbolic::DeviceName(pattern->device));
                 values.Set(values.Length(), item);
               }
               return values;
             }));
  result.Set(
      "optimize", Napi::Function::New(env, [model](const Napi::CallbackInfo &call) -> Napi::Value {
        try {
          if (!call[0].IsArray() || !call[1].IsString() ||
              (!call[2].IsUndefined() && !call[2].IsString()))
            throw std::runtime_error(
                "Expected selected patterns, an optimization mode and optional source path.");
          const auto operation = call[1].As<Napi::String>().Utf8Value();
          if (operation != "Optimize" && !IsGlobalPass(operation))
            throw std::runtime_error("Unknown optimization mode.");
          ol::onnx_patterns::RegisterPatterns();
          ol::onnx_shapes::RegisterShapeFunctions();
          ol::onnx_kernels::RegisterKernelFunctions();
          std::vector<std::string> patterns;
          std::set<std::string> unique;
          const auto selected = call[0].As<Napi::Array>();
          for (uint32_t i = 0; i < selected.Length(); ++i) {
            if (!selected.Get(i).IsString())
              throw std::runtime_error("Invalid optimization pattern.");
            auto name = selected.Get(i).As<Napi::String>().Utf8Value();
            ol::core::builder::CreateRegisteredPattern(name);
            if (!unique.insert(name).second)
              throw std::runtime_error("Duplicate optimization pattern.");
            patterns.push_back(std::move(name));
          }
          if (operation != "Optimize" && !patterns.empty())
            throw std::runtime_error("Global passes do not accept fusion patterns.");
          const auto path =
              call[2].IsString() ? call[2].As<Napi::String>().Utf8Value() : model->file_path;
          if (path.find('\0') != std::string::npos)
            throw std::runtime_error("Invalid source model file path.");
          auto worker = std::make_unique<FusionWorker>(call.Env(), model, std::move(patterns),
                                                       std::nullopt, path, "", false, operation);
          const auto promise = worker->Promise();
          worker->Queue();
          worker.release();
          return promise;
        } catch (const std::exception &error) {
          Napi::Error::New(call.Env(), error.what()).ThrowAsJavaScriptException();
          return call.Env().Undefined();
        }
      }));
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
          ol::onnx_kernels::RegisterKernelFunctions();
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
                 for (const auto *name :
                      {"IdentityRemoval", "ConstantFolding", "RemoveUnusedNodes"})
                   values.Set(i++, name);
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
            const bool global = std::any_of(patterns.begin(), patterns.end(), IsGlobalPass);
            if (global)
              ol::onnx_kernels::RegisterKernelFunctions();
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
                    (global ? index != 0 : index >= model->model->graph().node().size()))
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
