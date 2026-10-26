#include "model.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <onnx_core/compute/compute_context.h>
#include <onnx_extensions/kernels/kernel_dispatch_table.h>
#include <onnx_extensions/shapes/dispatch_table.h>
#include <onnx_proto/onnx_helper.h>
#include <stdexcept>

namespace onnx_light_viewer {
namespace {
constexpr int64_t maxSafeBytes = 9007199254740991LL;

void RootPath(const Napi::CallbackInfo &call, size_t index) {
  if (call.Length() > index + 1 ||
      (call.Length() > index && !call[index].IsUndefined() &&
       (!call[index].IsArray() || call[index].As<Napi::Array>().Length() != 0)))
    throw std::runtime_error(
        "Memory profiling covers the whole model; choose a stage root for pipelines.");
}

struct Dimension {
  std::string key;
  std::string label;
};

std::string DimensionKey(const auto &dimension, int input, int axis) {
  return dimension.has_dim_param() && !dimension.dim_param().empty()
             ? "symbol:" + std::string(dimension.dim_param())
             : "input:" + std::to_string(input) + ":" + std::to_string(axis);
}

std::vector<Dimension> Dimensions(const ol::ModelProto &model) {
  std::map<std::string, std::string> dimensions;
  for (int input = 0; input < model.graph().input_size(); ++input) {
    const auto &value = model.graph().input(input);
    if (!value.type().has_tensor_type() || !value.type().tensor_type().has_shape())
      throw std::runtime_error("Memory estimation requires a known tensor rank for input " +
                               std::string(value.name()) + ".");
    if (!ol::FixedBitWidth(value.type().tensor_type().elem_type()))
      throw std::runtime_error("Memory estimation requires fixed-width tensor inputs.");
    const auto &shape = value.type().tensor_type().shape();
    for (int axis = 0; axis < shape.dim_size(); ++axis) {
      const auto &dimension = shape.dim(axis);
      if (!dimension.has_dim_value() || dimension.dim_value() < 0)
        dimensions.emplace(DimensionKey(dimension, input, axis),
                           dimension.has_dim_param() && !dimension.dim_param().empty()
                               ? std::string(dimension.dim_param())
                               : std::string(value.name()) + "[" + std::to_string(axis) + "]");
    }
  }
  std::vector<Dimension> result;
  for (const auto &[key, label] : dimensions)
    result.push_back({key, label});
  return result;
}

int64_t Bytes(const ol::core::expressions::DimType &value) {
  const auto *bytes = std::get_if<int64_t>(&value);
  if (!bytes || *bytes < 0 || *bytes > maxSafeBytes)
    throw std::runtime_error("Memory size is unresolved or outside the exact integer range: " +
                             ol::core::expressions::dim_to_string(value));
  return *bytes;
}

int64_t AllocationBytes(const std::string &name, ol::TensorProto::DataType type,
                        const ol::core::symbolic::SymShape &shape) {
  const auto width = ol::FixedBitWidth(type);
  if (!width)
    throw std::runtime_error("Memory estimation requires fixed-width tensor allocations: " + name);
  int64_t elements = 1;
  for (const auto &dimension : shape.Dims()) {
    if (!dimension.IsInt() || dimension.AsInt() < 0)
      throw std::runtime_error("Memory size is unresolved for " + name + ".");
    const auto extent = dimension.AsInt();
    if (extent && elements > std::numeric_limits<int64_t>::max() / extent)
      throw std::runtime_error("Tensor shape is too large for exact memory estimation: " + name);
    elements *= extent;
  }
  if (elements > maxSafeBytes * 8 / width)
    throw std::runtime_error("Tensor shape is too large for exact memory estimation: " + name);
  return (elements * width + 7) / 8;
}

void ValidateAllocations(const ol::GraphProto &graph,
                         const ol::core::shapes::ShapesContext &shapes) {
  int64_t total = 0;
  for (const auto &[name, tensor] : shapes.Tensors()) {
    if (shapes.HasEncodedValue(name) ||
        (shapes.HasType(name) && !shapes.GetType(name).has_tensor_type()))
      throw std::runtime_error("Memory estimation requires dense tensor allocations: " + name);
    const auto bytes = AllocationBytes(
        name, ol::core::symbolic::TensorTypeToDataType(tensor.Dtype()), tensor.Shape());
    if (bytes > std::numeric_limits<int64_t>::max() - total)
      throw std::runtime_error("Tensor allocation sum is too large for native memory analysis.");
    total += bytes;
  }
  const auto require_tensor = [&](const auto &value) {
    const std::string name(value);
    if (!name.empty() && !shapes.Has(name))
      throw std::runtime_error("Memory allocation is unsupported or unresolved for " + name + ".");
  };
  for (const auto &input : graph.input())
    require_tensor(input.name());
  for (const auto &initializer : graph.initializer())
    require_tensor(initializer.name());
  for (const auto &node : graph.node())
    for (const auto &output : node.output())
      require_tensor(output);
  for (const auto &output : graph.output())
    require_tensor(output.name());
}

void ValidateScratchAllocations(const ol::GraphProto &graph,
                                const ol::core::shapes::ShapesContext &shapes) {
  // Mirror the pinned SDK's two estimators before their unchecked products run.
  for (const auto &node : graph.node()) {
    if ((!node.domain().empty() && node.domain() != "ai.onnx") ||
        (node.op_type() != "Attention" && node.op_type() != "LinearAttention"))
      continue;
    const ol::core::symbolic::SymShape empty;
    const auto input = [&](int index) -> const ol::core::symbolic::SymShape & {
      if (index >= node.input_size())
        return empty;
      const std::string name(node.input(index));
      return shapes.Has(name) ? shapes.Get(name).Shape() : empty;
    };
    const auto &query = input(0);
    const auto &key = input(1);
    const std::string name = "scratch for " + std::string(node.op_type());
    if (node.op_type() == "Attention") {
      if (query.Rank() == 4 && key.Rank() == 4)
        AllocationBytes(name, ol::TensorProto::FLOAT, {query[0], query[1], query[2], key[2]});
    } else if (query.Rank() == 3 && key.Rank() == 3 && input(2).Rank() == 3) {
      const auto &state = input(3);
      if (state.Rank() == 4)
        AllocationBytes(name, ol::TensorProto::FLOAT, state);
      else
        AllocationBytes(name, ol::TensorProto::FLOAT, {query[0], key[2], input(2)[2]});
    }
  }
}

class MemoryWorker : public Napi::AsyncWorker {
public:
  MemoryWorker(Napi::Env env, std::shared_ptr<StoredModel> source,
               std::map<std::string, int64_t> dimensions)
      : Napi::AsyncWorker(env), deferred_(Napi::Promise::Deferred::New(env)),
        source_(std::move(source)), snapshot_(source_->model->SharedCopy()),
        dimensions_(std::move(dimensions)) {}

  Napi::Promise Promise() const { return deferred_.Promise(); }

  void Execute() override {
    try {
      auto &graph = snapshot_.ref_graph();
      if (graph.sparse_initializer_size() || graph.extended_initializer_size())
        throw std::runtime_error("Memory estimation requires dense tensor initializers.");
      for (const auto &tensor : graph.initializer())
        AllocationBytes(std::string(tensor.name()), tensor.data_type(),
                        ol::core::symbolic::ShapeFromTensorProtoDims(tensor));
      for (int input = 0; input < graph.input_size(); ++input) {
        auto &shape = graph.ref_input()[input].ref_type().ref_tensor_type().ref_shape();
        std::vector<ol::core::symbolic::SymDim> dimensions;
        dimensions.reserve(shape.dim_size());
        for (int axis = 0; axis < shape.dim_size(); ++axis) {
          auto &dimension = shape.ref_dim()[axis];
          if (!dimension.has_dim_value() || dimension.dim_value() < 0)
            dimension.set_dim_value(dimensions_.at(DimensionKey(dimension, input, axis)));
          dimensions.emplace_back(dimension.dim_value());
        }
        AllocationBytes(std::string(graph.input(input).name()),
                        graph.input(input).type().tensor_type().elem_type(),
                        ol::core::symbolic::SymShape(dimensions));
      }
      for (auto *values : {&graph.ref_output(), &graph.ref_value_info()})
        for (auto &value : *values) {
          if (!value.type().has_tensor_type() || !value.type().tensor_type().has_shape())
            continue;
          auto &type = value.ref_type().ref_tensor_type();
          bool unknown = false;
          for (auto &dimension : type.ref_shape().ref_dim()) {
            if (dimension.has_dim_param()) {
              const auto found = dimensions_.find("symbol:" + std::string(dimension.dim_param()));
              if (found != dimensions_.end())
                dimension.set_dim_value(found->second);
            }
            unknown |= !dimension.has_dim_value() &&
                       (!dimension.has_dim_param() || dimension.dim_param().empty());
          }
          if (unknown)
            type.clear_shape();
        }
      ol::core::compute::ComputeContext context;
      context.ComputeShapes(snapshot_);
      // Native memory arithmetic assumes concrete, bounded allocations.
      ValidateAllocations(graph, context.Shapes());
      ValidateScratchAllocations(graph, context.Shapes());
      context.ComputeValueAndNodeTags(graph);
      context.ComputeInPlaceReuseGraph(graph, context.Shapes(), false, context.ValueTags());
      context.ComputePeakMemory(graph, ol::core::symbolic::Device::kCPU);
      for (size_t index = 0; index < context.Memory().size(); ++index) {
        const auto &memory = context.NodeMemory(index);
        const auto scalar = [&](const char *key) {
          return Bytes(std::get<ol::core::expressions::DimType>(memory.at(key)));
        };
        const auto existing = scalar(ol::core::compute::kNodeMemoryAlreadyAllocatedBytesKey);
        const auto outputs = scalar(ol::core::compute::kNodeMemoryOutputAllocationBytesKey);
        const auto tensors = scalar(ol::core::compute::kNodeMemoryTotalBytesKey);
        const auto scratch = context.NodePeakMemory(index);
        if (scratch < 0 || scratch > maxSafeBytes - tensors)
          throw std::runtime_error("Estimated memory exceeds the exact integer range.");
        events_.push_back({existing, outputs, scratch, tensors + scratch});
        peak_ = std::max(peak_, tensors + scratch);
      }
    } catch (const std::exception &error) {
      SetError(error.what());
    }
  }

  void OnOK() override {
    auto result = Napi::Object::New(Env());
    result.Set("name", std::string(snapshot_.graph().name()));
    result.Set("device", "CPU");
    result.Set("peakBytes", events_.empty() ? Napi::Value(Env().Null())
                                            : Napi::Value(Napi::Number::New(Env(), peak_)));
    auto events = Napi::Array::New(Env(), events_.size());
    for (uint32_t index = 0; index < events_.size(); ++index) {
      const auto &node = snapshot_.graph().node(index);
      const auto &event = events_[index];
      auto item = Napi::Object::New(Env());
      item.Set("index", index);
      item.Set("node", std::string(node.name()));
      item.Set("opType", std::string(node.op_type()));
      item.Set("existingBytes", Napi::Number::New(Env(), event.existing));
      item.Set("outputBytes", Napi::Number::New(Env(), event.outputs));
      item.Set("scratchBytes", Napi::Number::New(Env(), event.scratch));
      item.Set("deviceBytes", Napi::Number::New(Env(), event.total));
      events.Set(index, item);
    }
    result.Set("events", events);
    deferred_.Resolve(result);
  }

  void OnError(const Napi::Error &error) override { deferred_.Reject(error.Value()); }

private:
  struct Event {
    int64_t existing, outputs, scratch, total;
  };
  Napi::Promise::Deferred deferred_;
  std::shared_ptr<StoredModel> source_;
  ol::ModelProto snapshot_;
  std::map<std::string, int64_t> dimensions_;
  std::vector<Event> events_;
  int64_t peak_ = 0;
};
} // namespace

void AddMemoryProfileMethods(Napi::Env env, Napi::Object result,
                             std::shared_ptr<StoredModel> model) {
  result.Set("memoryProfileInputs",
             Napi::Function::New(env, [model](const Napi::CallbackInfo &call) -> Napi::Value {
               try {
                 RootPath(call, 0);
                 auto values = Napi::Array::New(call.Env());
                 for (const auto &dimension : Dimensions(*model->model)) {
                   auto item = Napi::Object::New(call.Env());
                   item.Set("key", dimension.key);
                   item.Set("label", dimension.label);
                   item.Set("defaultValue", 1);
                   values.Set(values.Length(), item);
                 }
                 return values;
               } catch (const std::exception &error) {
                 Napi::Error::New(call.Env(), error.what()).ThrowAsJavaScriptException();
                 return call.Env().Undefined();
               }
             }));
  result.Set(
      "memoryProfile",
      Napi::Function::New(env, [model](const Napi::CallbackInfo &call) -> Napi::Value {
        try {
          RootPath(call, 1);
          if (call.Length() < 1 || !call[0].IsObject() || call[0].IsArray() || call[0].IsNull())
            throw std::runtime_error("Expected dynamic dimension values for the whole model.");
          const auto values = call[0].As<Napi::Object>();
          const auto expected = Dimensions(*model->model);
          if (values.GetPropertyNames().Length() != expected.size())
            throw std::runtime_error("Supply exactly the model's dynamic dimensions.");
          std::map<std::string, int64_t> dimensions;
          for (const auto &dimension : expected) {
            const auto value = values.Get(dimension.key);
            const double number = value.IsNumber() ? value.As<Napi::Number>().DoubleValue() : 0;
            if (!std::isfinite(number) || number < 1 || number > maxSafeBytes ||
                std::floor(number) != number)
              throw std::runtime_error("Expected a positive safe integer for " + dimension.label);
            dimensions.emplace(dimension.key, static_cast<int64_t>(number));
          }
          ol::onnx_shapes::RegisterShapeFunctions();
          ol::onnx_shapes::RegisterPeakMemoryFunctions();
          ol::onnx_kernels::RegisterKernelFunctions();
          auto worker = new MemoryWorker(call.Env(), model, std::move(dimensions));
          const auto promise = worker->Promise();
          worker->Queue();
          return promise;
        } catch (const std::exception &error) {
          Napi::Error::New(call.Env(), error.what()).ThrowAsJavaScriptException();
          return call.Env().Undefined();
        }
      }));
}
} // namespace onnx_light_viewer
