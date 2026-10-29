#include "model.h"
#include <algorithm>
#include <cmath>
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
      const auto bind = [&](auto &graph, bool named) {
        for (int input = 0; input < graph.input_size(); ++input) {
          auto &shape = graph.ref_input()[input].ref_type().ref_tensor_type().ref_shape();
          for (int axis = 0; axis < shape.dim_size(); ++axis) {
            auto &dimension = shape.ref_dim()[axis];
            // Anonymous dimensions have no symbolic spelling.
            if ((!dimension.has_dim_value() || dimension.dim_value() < 0) &&
                (named || !dimension.has_dim_param() || dimension.dim_param().empty()))
              dimension.set_dim_value(dimensions_.at(DimensionKey(dimension, input, axis)));
          }
        }
        for (auto *values : {&graph.ref_output(), &graph.ref_value_info()})
          for (auto &value : *values) {
            if (!value.type().has_tensor_type() || !value.type().tensor_type().has_shape())
              continue;
            auto &type = value.ref_type().ref_tensor_type();
            bool unknown = false;
            for (auto &dimension : type.ref_shape().ref_dim()) {
              if (named && dimension.has_dim_param()) {
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
      };
      auto symbolic_snapshot = snapshot_.SharedCopy();
      bind(symbolic_snapshot.ref_graph(), false);
      ol::core::compute::ComputeContext symbolic;
      symbolic.ComputeShapes(symbolic_snapshot);
      symbolic.ComputeValueAndNodeTags(symbolic_snapshot.graph());
      symbolic.ComputeInPlaceReuseGraph(symbolic_snapshot.graph(), symbolic.Shapes(), false,
                                        symbolic.ValueTags());
      auto &graph = snapshot_.ref_graph();
      bind(graph, true);
      ol::core::compute::ComputeContext context;
      context.ComputeShapes(snapshot_);
      context.ComputeValueAndNodeTags(graph);
      context.set_events_enabled(true);
      context.ComputeInPlaceReuseGraph(graph, context.Shapes(), false, context.ValueTags());
      context.ComputePeakMemory(graph, ol::core::symbolic::Device::kCPU);
      const auto describe = [&](const auto &ports) {
        std::vector<Tensor> tensors;
        for (const auto &port : ports) {
          const std::string name(port);
          if (name.empty())
            continue;
          Tensor tensor;
          tensor.name = name;
          if (context.Shapes().Has(name)) {
            const auto &descriptor = context.Shapes().Get(name);
            tensor.type = ol::TensorProto::DataType_Name(
                ol::core::symbolic::TensorTypeToDataType(descriptor.Dtype()));
            tensor.shape = descriptor.Shape().ToString();
          }
          const auto tag = context.ValueTags().find(name);
          if (tag != context.ValueTags().end())
            tensor.tag = tag->second;
          tensors.push_back(std::move(tensor));
        }
        return tensors;
      };
      for (size_t index = 0; index < context.Memory().size(); ++index) {
        const auto &memory = context.NodeMemory(index);
        for (const auto &[key, entry] : memory) {
          if (const auto *value = std::get_if<ol::core::expressions::DimType>(&entry)) {
            Bytes(*value);
          } else {
            for (const auto &[tag, amount] : std::get<ol::core::compute::TaggedMemory>(entry))
              Bytes(amount);
          }
        }
        const auto scalar = [&](const char *key) {
          return Bytes(std::get<ol::core::expressions::DimType>(memory.at(key)));
        };
        const auto existing = scalar(ol::core::compute::kNodeMemoryAlreadyAllocatedBytesKey);
        const auto outputs = scalar(ol::core::compute::kNodeMemoryOutputAllocationBytesKey);
        const auto tensors = scalar(ol::core::compute::kNodeMemoryTotalBytesKey);
        const auto scratch = context.NodePeakMemory(index);
        if (scratch < 0 || scratch > maxSafeBytes - tensors)
          throw std::runtime_error("Estimated memory exceeds the exact integer range.");
        const auto &node = graph.node(index);
        events_.push_back({existing,
                           outputs,
                           scratch,
                           tensors + scratch,
                           {},
                           describe(node.input()),
                           describe(node.output()),
                           memory,
                           symbolic.NodeMemory(index)});
        peak_ = std::max(peak_, tensors + scratch);
      }
      for (const auto &decision : context.Events())
        events_.at(decision.node_index).decisions.push_back(decision);
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
      item.Set(
          "afterBytes",
          Napi::Number::New(Env(), Bytes(std::get<ol::core::expressions::DimType>(event.memory.at(
                                       ol::core::compute::kNodeMemoryAfterBytesKey)))));
      auto decisions = Napi::Array::New(Env(), event.decisions.size());
      for (uint32_t i = 0; i < event.decisions.size(); ++i) {
        const auto &decision = event.decisions[i];
        auto value = Napi::Object::New(Env());
        value.Set("action", ol::core::compute::ComputeEventActionName(decision.action));
        value.Set("name", decision.name);
        if (decision.action == ol::core::compute::ComputeEventAction::kInPlace) {
          value.Set("input", std::string(node.input(decision.input_index)));
          value.Set("output", std::string(node.output(decision.output_index)));
          value.Set("kind", decision.kind == ol::core::compute::InPlaceReuseKind::kEqual
                                ? "equal"
                                : "greater");
        }
        decisions.Set(i, value);
      }
      item.Set("decisions", decisions);
      item.Set("inputs", TensorDetails(event.inputs));
      item.Set("outputs", TensorDetails(event.output_tensors));
      item.Set("memory", MemoryDetails(event.memory, false));
      item.Set("symbolicMemory", MemoryDetails(event.symbolic_memory, true));
      events.Set(index, item);
    }
    result.Set("events", events);
    deferred_.Resolve(result);
  }

  void OnError(const Napi::Error &error) override { deferred_.Reject(error.Value()); }

private:
  struct Tensor {
    std::string name, type, shape, tag;
  };
  struct Event {
    int64_t existing, outputs, scratch, total;
    ol::core::compute::ComputeEventLog decisions;
    std::vector<Tensor> inputs, output_tensors;
    ol::core::compute::NodeMemoryProfile memory, symbolic_memory;
  };
  Napi::Object MemoryDetails(const ol::core::compute::NodeMemoryProfile &memory, bool symbolic) {
    auto result = Napi::Object::New(Env());
    const auto value = [&](const ol::core::expressions::DimType &amount) -> Napi::Value {
      if (symbolic)
        return Napi::String::New(Env(), ol::core::expressions::dim_to_string(amount));
      return Napi::Number::New(Env(), Bytes(amount));
    };
    for (const auto &[key, entry] : memory) {
      if (const auto *scalar = std::get_if<ol::core::expressions::DimType>(&entry)) {
        result.Set(key, value(*scalar));
      } else {
        auto tags = Napi::Object::New(Env());
        for (const auto &[tag, amount] : std::get<ol::core::compute::TaggedMemory>(entry))
          tags.Set(tag, value(amount));
        result.Set(key, tags);
      }
    }
    return result;
  }
  Napi::Array TensorDetails(const std::vector<Tensor> &tensors) {
    auto result = Napi::Array::New(Env(), tensors.size());
    for (uint32_t index = 0; index < tensors.size(); ++index) {
      const auto &tensor = tensors[index];
      auto item = Napi::Object::New(Env());
      item.Set("name", tensor.name);
      for (const auto &[key, value] :
           {std::pair{"type", tensor.type}, std::pair{"shape", tensor.shape},
            std::pair{"tag", tensor.tag}})
        item.Set(key, value.empty() ? Napi::Value(Env().Null())
                                    : Napi::Value(Napi::String::New(Env(), value)));
      result.Set(index, item);
    }
    return result;
  }
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
