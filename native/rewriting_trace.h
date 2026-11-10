#pragma once

#include "model_storage.h"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

namespace onnx_light_viewer {

using FusionNodeKey = std::tuple<std::string, std::string, std::string, std::string,
                                 std::vector<std::string>, std::vector<std::string>>;

inline FusionNodeKey NodeOrderKey(const ol::NodeProto &node) {
  std::vector<std::string> inputs, outputs;
  for (const auto &value : node.input())
    inputs.push_back(value.value());
  for (const auto &value : node.output())
    outputs.push_back(value.value());
  return {node.domain().value(), node.op_type().value(), node.overload().value(),
          node.name().value(),   std::move(inputs),      std::move(outputs)};
}

inline StoredModel::FusionMatch::Node ImpactNode(const FusionNodeKey &key,
                                                 std::optional<size_t> index = std::nullopt) {
  return {index,           std::get<3>(key), std::get<1>(key), std::get<0>(key), std::get<4>(key),
          std::get<5>(key)};
}

template <typename Lookup>
inline std::shared_ptr<const StoredModel::FusionMatch::Impact>
CaptureFusionImpact(const ol::LocalRewritingProto &edit, const Lookup &lookup) {
  auto impact = std::make_shared<StoredModel::FusionMatch::Impact>();
  impact->graph_path.assign(edit.graph_path().begin(), edit.graph_path().end());
  impact->removed_initializers =
      edit.removed_initializers().size() + edit.removed_extended_initializers().size();
  impact->added_initializers = edit.added_initializers().size();
  for (const auto index : edit.matched_nodes()) {
    if (edit.graph_path().empty())
      impact->before.push_back(ImpactNode(lookup(index), index));
    else {
      StoredModel::FusionMatch::Node node;
      node.index = index;
      impact->before.push_back(std::move(node));
    }
  }
  for (const auto &node : edit.added_nodes())
    impact->after.push_back(ImpactNode(NodeOrderKey(node)));
  return impact;
}

inline std::vector<size_t>
TraceRewriting(const ol::LocalRewritingProto &edit, std::vector<FusionNodeKey> &nodes,
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

// Keep stable node identities in an implicit treap; resolve replacement ancestry only once.
class RewritingTrace {
public:
  explicit RewritingTrace(const ol::GraphProto &graph) : RewritingTrace(graph.node()) {}

  explicit RewritingTrace(const ol::utils::RepeatedProtoField<ol::NodeProto> &nodes,
                          std::vector<const ol::NodeProto *> *tracked_nodes = nullptr)
      : tracked_nodes_(tracked_nodes) {
    entries_.reserve(nodes.size());
    for (const auto &node : nodes)
      root_ = Merge(root_, Add(node));
  }

  std::shared_ptr<const StoredModel::FusionMatch::Impact>
  Impact(const ol::LocalRewritingProto &edit) const {
    return CaptureFusionImpact(edit, [&](size_t position) -> const FusionNodeKey & {
      return entries_[NodeId(position)].key;
    });
  }

  const ol::NodeProto &NodeAt(size_t position) const {
    if (!tracked_nodes_)
      throw std::runtime_error("Rewriting nodes were not retained.");
    return *tracked_nodes_->at(NodeId(position));
  }

  std::vector<size_t> Apply(const ol::LocalRewritingProto &edit, bool scoped = false) {
    if (!scoped && !edit.graph_path().empty())
      return {};
    const std::set<size_t> removed(edit.matched_nodes().begin(), edit.matched_nodes().end());
    if (removed.size() > Size(root_) || (!removed.empty() && *removed.rbegin() >= Size(root_)))
      throw std::runtime_error("Invalid rewriting node count.");
    const auto count = Size(root_) - removed.size() + edit.added_nodes().size();
    auto added = std::make_shared<std::vector<size_t>>();
    std::map<size_t, size_t> insertions;
    std::map<FusionNodeKey, std::shared_ptr<std::vector<size_t>>> replacements;
    for (size_t i = 0; i < edit.added_nodes().size(); ++i) {
      const auto position = edit.added_nodes_positions(i);
      if (position >= count || insertions.contains(position))
        throw std::runtime_error("Invalid rewriting insertion position.");
      const auto id = Add(edit.added_nodes(i));
      insertions.emplace(position, id);
      added->push_back(id);
      auto &same = replacements[entries_[id].key];
      if (!same)
        same = std::make_shared<std::vector<size_t>>();
      same->push_back(id);
    }
    for (auto position = removed.rbegin(); position != removed.rend(); ++position) {
      auto [left, rest] = Split(root_, *position);
      auto [id, right] = Split(rest, 1);
      const auto same = replacements.find(entries_[id].key);
      entries_[id].replacements = same == replacements.end() ? added : same->second;
      root_ = Merge(left, right);
    }
    for (const auto &[position, id] : insertions) {
      if (position > Size(root_))
        throw std::runtime_error("Invalid rewriting node positions.");
      auto [left, right] = Split(root_, position);
      root_ = Merge(Merge(left, id), right);
    }
    return *added;
  }

  std::vector<FusionNodeKey> Finish(std::vector<std::vector<StoredModel::FusionMatch>> &history) {
    std::vector<size_t> final;
    const auto visit = [&](const auto &self, size_t id) -> void {
      if (id == none)
        return;
      self(self, entries_[id].left);
      final.push_back(id);
      self(self, entries_[id].right);
    };
    visit(visit, root_);
    using Positions = std::shared_ptr<const std::vector<size_t>>;
    std::vector<Positions> positions(entries_.size());
    for (size_t i = 0; i < final.size(); ++i)
      positions[final[i]] = std::make_shared<std::vector<size_t>>(1, i);
    std::unordered_map<const std::vector<size_t> *, Positions> resolved;
    for (size_t id = entries_.size(); id-- > 0;) {
      const auto &replacement = entries_[id].replacements;
      if (!replacement)
        continue;
      if (replacement->size() == 1) {
        positions[id] = positions.at(replacement->front());
      } else if (const auto found = resolved.find(replacement.get()); found != resolved.end()) {
        positions[id] = found->second;
      } else {
        auto targets = std::make_shared<std::vector<size_t>>();
        for (const auto child : *replacement) {
          const auto &values = *positions.at(child);
          targets->insert(targets->end(), values.begin(), values.end());
        }
        std::sort(targets->begin(), targets->end());
        targets->erase(std::unique(targets->begin(), targets->end()), targets->end());
        positions[id] = targets;
        resolved.emplace(replacement.get(), std::move(targets));
      }
    }
    for (auto &operation : history)
      for (auto &match : operation) {
        std::vector<size_t> targets;
        for (const auto id : match.nodes) {
          const auto &values = *positions.at(id);
          targets.insert(targets.end(), values.begin(), values.end());
        }
        std::sort(targets.begin(), targets.end());
        targets.erase(std::unique(targets.begin(), targets.end()), targets.end());
        match.nodes = std::move(targets);
      }
    std::vector<FusionNodeKey> order;
    order.reserve(final.size());
    for (const auto id : final)
      order.push_back(std::move(entries_[id].key));
    return order;
  }

private:
  static constexpr size_t none = std::numeric_limits<size_t>::max();
  struct Entry {
    FusionNodeKey key;
    uint64_t priority;
    size_t left = none;
    size_t right = none;
    size_t size = 1;
    std::shared_ptr<const std::vector<size_t>> replacements;
  };

  size_t Add(const ol::NodeProto &node) {
    const auto id = entries_.size();
    uint64_t priority = id + UINT64_C(0x9e3779b97f4a7c15);
    priority = (priority ^ (priority >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    priority = (priority ^ (priority >> 27)) * UINT64_C(0x94d049bb133111eb);
    priority ^= priority >> 31;
    entries_.push_back({NodeOrderKey(node), priority, none, none, 1, nullptr});
    if (tracked_nodes_)
      tracked_nodes_->push_back(&node);
    return id;
  }

  size_t NodeId(size_t position) const {
    auto id = root_;
    while (id != none) {
      const auto left = Size(entries_[id].left);
      if (position == left)
        return id;
      if (position < left)
        id = entries_[id].left;
      else {
        position -= left + 1;
        id = entries_[id].right;
      }
    }
    throw std::runtime_error("Invalid impacted node index.");
  }

  size_t Size(size_t id) const { return id == none ? 0 : entries_[id].size; }

  void Update(size_t id) {
    entries_[id].size = 1 + Size(entries_[id].left) + Size(entries_[id].right);
  }

  size_t Merge(size_t left, size_t right) {
    if (left == none)
      return right;
    if (right == none)
      return left;
    if (entries_[left].priority > entries_[right].priority) {
      entries_[left].right = Merge(entries_[left].right, right);
      Update(left);
      return left;
    }
    entries_[right].left = Merge(left, entries_[right].left);
    Update(right);
    return right;
  }

  std::pair<size_t, size_t> Split(size_t root, size_t count) {
    if (root == none)
      return {none, none};
    if (count <= Size(entries_[root].left)) {
      auto [left, right] = Split(entries_[root].left, count);
      entries_[root].left = right;
      Update(root);
      return {left, root};
    }
    auto [left, right] = Split(entries_[root].right, count - Size(entries_[root].left) - 1);
    entries_[root].right = left;
    Update(root);
    return {root, right};
  }

  std::vector<Entry> entries_;
  size_t root_ = none;
  std::vector<const ol::NodeProto *> *tracked_nodes_;
};

} // namespace onnx_light_viewer
