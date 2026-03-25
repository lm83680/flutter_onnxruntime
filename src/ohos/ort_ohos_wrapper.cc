#include "ort_ohos_wrapper.h"

#include "ohos_session_manager.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <node_api.h>

namespace {

struct ClonedTensor {
  std::vector<uint8_t> buffer;
  Ort::Value value{nullptr};
};

class OhosTensorStore {
public:
  OhosTensorStore() : next_tensor_id_(1), memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {}

  std::string CreateTensor(const std::string &source_type, const std::string &data_json,
                           const std::vector<int64_t> &shape) {
    if (source_type == "float32") {
      const std::vector<double> values = ParseNumberArray(data_json);
      std::vector<float> data;
      data.reserve(values.size());
      for (const double value : values) {
        data.push_back(static_cast<float>(value));
      }
      return CreateNumericTensor<float>(source_type, data, shape);
    }

    if (source_type == "int32") {
      const std::vector<double> values = ParseNumberArray(data_json);
      std::vector<int32_t> data;
      data.reserve(values.size());
      for (const double value : values) {
        data.push_back(static_cast<int32_t>(std::llround(value)));
      }
      return CreateNumericTensor<int32_t>(source_type, data, shape);
    }

    if (source_type == "int64") {
      const std::vector<double> values = ParseNumberArray(data_json);
      std::vector<int64_t> data;
      data.reserve(values.size());
      for (const double value : values) {
        data.push_back(static_cast<int64_t>(std::llround(value)));
      }
      return CreateNumericTensor<int64_t>(source_type, data, shape);
    }

    throw std::runtime_error("当前 OHOS 仅支持 float32/int32/int64 张量");
  }

  std::string StoreOutputTensor(Ort::Value &&tensor) {
    std::lock_guard<std::mutex> lock(mutex_);

    const std::string tensor_id = GenerateTensorId();
    tensors_[tensor_id] = std::make_unique<Ort::Value>(std::move(tensor));
    const Ort::TensorTypeAndShapeInfo tensor_info = tensors_[tensor_id]->GetTensorTypeAndShapeInfo();
    tensor_shapes_[tensor_id] = tensor_info.GetShape();
    tensor_types_[tensor_id] = ElementTypeToString(tensor_info.GetElementType());
    return tensor_id;
  }

  std::string CreateTensorFromBinaryFile(const std::string &source_type, const std::string &file_path, const std::vector<int64_t> &shape) {
    std::ifstream input(file_path, std::ios::binary);
    if (!input.good()) {
      throw std::runtime_error("二进制张量文件不存在");
    }

    input.seekg(0, std::ios::end);
    const std::streamoff file_size = input.tellg();
    input.seekg(0, std::ios::beg);
    if (file_size <= 0) {
      throw std::runtime_error("二进制张量文件为空");
    }

    std::vector<uint8_t> file_bytes(static_cast<size_t>(file_size));
    input.read(reinterpret_cast<char *>(file_bytes.data()), file_size);

    if (source_type == "float32") {
      return CreateTensorFromRawBytes<float>(source_type, file_bytes, shape);
    }

    if (source_type == "int32") {
      return CreateTensorFromRawBytes<int32_t>(source_type, file_bytes, shape);
    }

    if (source_type == "int64") {
      return CreateTensorFromRawBytes<int64_t>(source_type, file_bytes, shape);
    }

    throw std::runtime_error("当前 OHOS 仅支持 float32/int32/int64 张量");
  }

  ClonedTensor CloneTensor(const std::string &tensor_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto tensor_it = tensors_.find(tensor_id);
    auto type_it = tensor_types_.find(tensor_id);
    auto shape_it = tensor_shapes_.find(tensor_id);
    if (tensor_it == tensors_.end() || type_it == tensor_types_.end() || shape_it == tensor_shapes_.end()) {
      throw std::runtime_error("未找到对应的 OrtValue");
    }

    const std::string &tensor_type = type_it->second;
    const std::vector<int64_t> &shape = shape_it->second;
    const size_t element_count = tensor_it->second->GetTensorTypeAndShapeInfo().GetElementCount();

    if (tensor_type == "float32") {
      float *data = tensor_it->second->GetTensorMutableData<float>();
      return CloneNumericTensor<float>(data, element_count, shape);
    }

    if (tensor_type == "int32") {
      int32_t *data = tensor_it->second->GetTensorMutableData<int32_t>();
      return CloneNumericTensor<int32_t>(data, element_count, shape);
    }

    if (tensor_type == "int64") {
      int64_t *data = tensor_it->second->GetTensorMutableData<int64_t>();
      return CloneNumericTensor<int64_t>(data, element_count, shape);
    }

    throw std::runtime_error("当前 OHOS 仅支持 float32/int32/int64 张量");
  }

  std::string GetTensorDataJson(const std::string &tensor_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto tensor_it = tensors_.find(tensor_id);
    auto type_it = tensor_types_.find(tensor_id);
    auto shape_it = tensor_shapes_.find(tensor_id);
    if (tensor_it == tensors_.end() || type_it == tensor_types_.end() || shape_it == tensor_shapes_.end()) {
      throw std::runtime_error("未找到对应的 OrtValue");
    }

    std::ostringstream stream;
    stream << "{\"dataType\":\"" << EscapeJson(type_it->second) << "\",\"shape\":" << ToJsonIntArray(shape_it->second)
           << ",\"data\":";

    const size_t element_count = tensor_it->second->GetTensorTypeAndShapeInfo().GetElementCount();
    if (type_it->second == "float32") {
      float *data = tensor_it->second->GetTensorMutableData<float>();
      stream << ToJsonNumericArray(data, element_count);
    } else if (type_it->second == "int32") {
      int32_t *data = tensor_it->second->GetTensorMutableData<int32_t>();
      stream << ToJsonNumericArray(data, element_count);
    } else if (type_it->second == "int64") {
      int64_t *data = tensor_it->second->GetTensorMutableData<int64_t>();
      stream << ToJsonNumericArray(data, element_count);
    } else {
      throw std::runtime_error("当前 OHOS 仅支持 float32/int32/int64 张量");
    }

    stream << "}";
    return stream.str();
  }

  std::string WriteTensorDataToBinaryFile(const std::string &tensor_id, const std::string &file_path) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto tensor_it = tensors_.find(tensor_id);
    auto type_it = tensor_types_.find(tensor_id);
    auto shape_it = tensor_shapes_.find(tensor_id);
    if (tensor_it == tensors_.end() || type_it == tensor_types_.end() || shape_it == tensor_shapes_.end()) {
      throw std::runtime_error("未找到对应的 OrtValue");
    }

    const size_t element_count = tensor_it->second->GetTensorTypeAndShapeInfo().GetElementCount();
    size_t element_size = 0;
    const void *data_ptr = nullptr;

    if (type_it->second == "float32") {
      data_ptr = tensor_it->second->GetTensorMutableData<float>();
      element_size = sizeof(float);
    } else if (type_it->second == "int32") {
      data_ptr = tensor_it->second->GetTensorMutableData<int32_t>();
      element_size = sizeof(int32_t);
    } else if (type_it->second == "int64") {
      data_ptr = tensor_it->second->GetTensorMutableData<int64_t>();
      element_size = sizeof(int64_t);
    } else {
      throw std::runtime_error("当前 OHOS 仅支持 float32/int32/int64 张量");
    }

    std::ofstream output(file_path, std::ios::binary | std::ios::trunc);
    if (!output.good()) {
      throw std::runtime_error("无法写入张量二进制文件");
    }

    output.write(reinterpret_cast<const char *>(data_ptr), static_cast<std::streamsize>(element_count * element_size));
    output.flush();

    std::ostringstream stream;
    stream << "{\"dataType\":\"" << EscapeJson(type_it->second) << "\",\"shape\":" << ToJsonIntArray(shape_it->second) << "}";
    return stream.str();
  }

  bool ReleaseTensor(const std::string &tensor_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    const bool removed = tensors_.erase(tensor_id) > 0;
    tensor_types_.erase(tensor_id);
    tensor_shapes_.erase(tensor_id);
    tensor_buffers_.erase(tensor_id);
    return removed;
  }

  std::string GetTensorType(const std::string &tensor_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto iterator = tensor_types_.find(tensor_id);
    if (iterator == tensor_types_.end()) {
      throw std::runtime_error("未找到对应的 OrtValue");
    }
    return iterator->second;
  }

  std::vector<int64_t> GetTensorShape(const std::string &tensor_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto iterator = tensor_shapes_.find(tensor_id);
    if (iterator == tensor_shapes_.end()) {
      throw std::runtime_error("未找到对应的 OrtValue");
    }
    return iterator->second;
  }

private:
  template <typename T>
  std::string CreateTensorFromRawBytes(const std::string &source_type, const std::vector<uint8_t> &raw_bytes, const std::vector<int64_t> &shape) {
    if (raw_bytes.size() % sizeof(T) != 0) {
      throw std::runtime_error("二进制张量文件大小与数据类型不匹配");
    }

    const size_t element_count = raw_bytes.size() / sizeof(T);
    std::vector<T> data(element_count);
    std::memcpy(data.data(), raw_bytes.data(), raw_bytes.size());
    return CreateNumericTensor<T>(source_type, data, shape);
  }

  template <typename T>
  std::string CreateNumericTensor(const std::string &source_type, const std::vector<T> &data, const std::vector<int64_t> &shape) {
    ValidateTensorShape(shape, data.size());

    std::lock_guard<std::mutex> lock(mutex_);
    const std::string tensor_id = GenerateTensorId();

    std::vector<uint8_t> buffer(data.size() * sizeof(T));
    std::memcpy(buffer.data(), data.data(), buffer.size());
    T *tensor_data = reinterpret_cast<T *>(buffer.data());

    auto tensor = Ort::Value::CreateTensor<T>(memory_info_, tensor_data, data.size(), shape.data(), shape.size());
    tensors_[tensor_id] = std::make_unique<Ort::Value>(std::move(tensor));
    tensor_types_[tensor_id] = source_type;
    tensor_shapes_[tensor_id] = shape;
    tensor_buffers_[tensor_id] = std::move(buffer);
    return tensor_id;
  }

  template <typename T>
  ClonedTensor CloneNumericTensor(const T *data, size_t element_count, const std::vector<int64_t> &shape) {
    std::vector<uint8_t> buffer(element_count * sizeof(T));
    std::memcpy(buffer.data(), data, buffer.size());
    T *cloned_data = reinterpret_cast<T *>(buffer.data());

    ClonedTensor tensor;
    tensor.value = Ort::Value::CreateTensor<T>(memory_info_, cloned_data, element_count, shape.data(), shape.size());
    tensor.buffer = std::move(buffer);
    return tensor;
  }

  std::string GenerateTensorId() { return "tensor_" + std::to_string(next_tensor_id_++); }

  static void ValidateTensorShape(const std::vector<int64_t> &shape, size_t element_count) {
    int64_t expected_count = 1;
    for (const int64_t dimension : shape) {
      if (dimension < 0) {
        throw std::runtime_error("shape 不能包含负数维度");
      }
      expected_count *= dimension;
    }
    if (static_cast<size_t>(expected_count) != element_count) {
      throw std::runtime_error("张量 shape 与数据长度不匹配");
    }
  }

  static std::vector<double> ParseNumberArray(const std::string &json) {
    std::vector<double> values;

    if (!json.empty() && json.front() == '{') {
      const std::regex value_pattern(R"(:\s*(-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?))");
      auto begin = std::sregex_iterator(json.begin(), json.end(), value_pattern);
      auto end = std::sregex_iterator();
      for (auto iterator = begin; iterator != end; ++iterator) {
        values.push_back(std::stod((*iterator)[1].str()));
      }
      return values;
    }

    const std::regex number_pattern(R"(-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)");
    auto begin = std::sregex_iterator(json.begin(), json.end(), number_pattern);
    auto end = std::sregex_iterator();
    for (auto iterator = begin; iterator != end; ++iterator) {
      values.push_back(std::stod((*iterator).str()));
    }
    return values;
  }

  static std::string ElementTypeToString(ONNXTensorElementDataType element_type) {
    switch (element_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return "float32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return "int32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return "int64";
    default:
      throw std::runtime_error("当前 OHOS 仅支持 float32/int32/int64 张量");
    }
  }

  static std::string EscapeJson(const std::string &value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (const char ch : value) {
      switch (ch) {
      case '\\':
        escaped += "\\\\";
        break;
      case '"':
        escaped += "\\\"";
        break;
      case '\n':
        escaped += "\\n";
        break;
      case '\r':
        escaped += "\\r";
        break;
      case '\t':
        escaped += "\\t";
        break;
      default:
        escaped += ch;
        break;
      }
    }
    return escaped;
  }

  static std::string ToJsonIntArray(const std::vector<int64_t> &values) {
    std::ostringstream stream;
    stream << "[";
    for (size_t index = 0; index < values.size(); ++index) {
      if (index > 0) {
        stream << ",";
      }
      stream << values[index];
    }
    stream << "]";
    return stream.str();
  }

  template <typename T>
  static std::string ToJsonNumericArray(const T *data, size_t count) {
    std::ostringstream stream;
    stream << "[";
    for (size_t index = 0; index < count; ++index) {
      if (index > 0) {
        stream << ",";
      }
      stream << data[index];
    }
    stream << "]";
    return stream.str();
  }

  std::map<std::string, std::unique_ptr<Ort::Value>> tensors_;
  std::map<std::string, std::string> tensor_types_;
  std::map<std::string, std::vector<int64_t>> tensor_shapes_;
  std::map<std::string, std::vector<uint8_t>> tensor_buffers_;
  int next_tensor_id_;
  std::mutex mutex_;
  Ort::MemoryInfo memory_info_;
};

OhosSessionManager g_session_manager;
OhosTensorStore g_tensor_store;
thread_local std::string g_response_buffer;

std::string EscapeJson(const std::string &value) {
  std::string escaped;
  escaped.reserve(value.size());
  for (const char ch : value) {
    switch (ch) {
    case '\\':
      escaped += "\\\\";
      break;
    case '"':
      escaped += "\\\"";
      break;
    case '\n':
      escaped += "\\n";
      break;
    case '\r':
      escaped += "\\r";
      break;
    case '\t':
      escaped += "\\t";
      break;
    default:
      escaped += ch;
      break;
    }
  }
  return escaped;
}

std::string ErrorJson(const std::string &code, const std::string &message) {
  return "{\"error\":{\"code\":\"" + EscapeJson(code) + "\",\"message\":\"" + EscapeJson(message) + "\"}}";
}

bool FileExists(const char *path) {
  std::ifstream file(path);
  return file.good();
}

int ExtractIntOption(const std::string &json, const std::string &key) {
  const std::string marker = "\"" + key + "\":";
  const size_t position = json.find(marker);
  if (position == std::string::npos) {
    return -1;
  }
  const size_t value_start = position + marker.size();
  return std::stoi(json.substr(value_start));
}

std::vector<int64_t> ParseShape(const std::string &shape_json) {
  std::vector<int64_t> shape;
  const std::regex number_pattern(R"(-?\d+)");
  auto begin = std::sregex_iterator(shape_json.begin(), shape_json.end(), number_pattern);
  auto end = std::sregex_iterator();
  for (auto iterator = begin; iterator != end; ++iterator) {
    shape.push_back(std::stoll((*iterator).str()));
  }
  return shape;
}

std::vector<std::pair<std::string, std::string>> ParseInputRefs(const std::string &inputs_json) {
  std::vector<std::pair<std::string, std::string>> input_refs;
  const std::regex input_pattern(R"json("([^"]+)":\{"valueId":"([^"]+)"\})json");
  auto begin = std::sregex_iterator(inputs_json.begin(), inputs_json.end(), input_pattern);
  auto end = std::sregex_iterator();
  for (auto iterator = begin; iterator != end; ++iterator) {
    input_refs.emplace_back((*iterator)[1].str(), (*iterator)[2].str());
  }
  return input_refs;
}

std::string ToJsonStringArray(const std::vector<std::string> &items) {
  std::ostringstream stream;
  stream << "[";
  for (size_t index = 0; index < items.size(); ++index) {
    if (index > 0) {
      stream << ",";
    }
    stream << "\"" << EscapeJson(items[index]) << "\"";
  }
  stream << "]";
  return stream.str();
}

std::string ToJsonIntArray(const std::vector<int64_t> &values) {
  std::ostringstream stream;
  stream << "[";
  for (size_t index = 0; index < values.size(); ++index) {
    if (index > 0) {
      stream << ",";
    }
    stream << values[index];
  }
  stream << "]";
  return stream.str();
}

std::string BuildTensorRefJson(const std::string &value_id, const std::string &data_type, const std::vector<int64_t> &shape) {
  std::ostringstream stream;
  stream << "[\"" << EscapeJson(value_id) << "\",\"" << EscapeJson(data_type) << "\",[";
  for (size_t index = 0; index < shape.size(); ++index) {
    if (index > 0) {
      stream << ",";
    }
    stream << shape[index];
  }
  stream << "]]";
  return stream.str();
}

napi_value CreateUtf8String(napi_env env, const std::string &value) {
  napi_value result = nullptr;
  napi_create_string_utf8(env, value.c_str(), value.size(), &result);
  return result;
}

std::string GetStringArgument(napi_env env, napi_callback_info info, size_t index) {
  size_t argc = index + 1;
  napi_value args[3] = {nullptr};
  napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
  if (argc <= index) {
    throw std::runtime_error("原生模块参数不足");
  }

  size_t length = 0;
  napi_get_value_string_utf8(env, args[index], nullptr, 0, &length);
  std::string result(length + 1, '\0');
  napi_get_value_string_utf8(env, args[index], result.data(), result.size(), &length);
  result.resize(length);
  return result;
}

napi_value WrapStringMethod(napi_env env, napi_callback_info info,
                            const std::function<std::string(napi_env, napi_callback_info)> &handler) {
  try {
    return CreateUtf8String(env, handler(env, info));
  } catch (const std::exception &exception) {
    napi_throw_error(env, nullptr, exception.what());
    return nullptr;
  }
}

napi_value WrapIntMethod(napi_env env, napi_callback_info info,
                         const std::function<int32_t(napi_env, napi_callback_info)> &handler) {
  try {
    napi_value result = nullptr;
    napi_create_int32(env, handler(env, info), &result);
    return result;
  } catch (const std::exception &exception) {
    napi_throw_error(env, nullptr, exception.what());
    return nullptr;
  }
}

napi_value NapiGetPlatformVersion(napi_env env, napi_callback_info info) {
  return WrapStringMethod(env, info, [](napi_env, napi_callback_info) { return std::string(ortGetPlatformVersion()); });
}

napi_value NapiGetAvailableProvidersJson(napi_env env, napi_callback_info info) {
  return WrapStringMethod(env, info,
                          [](napi_env, napi_callback_info) { return std::string(ortGetAvailableProvidersJson()); });
}

napi_value NapiCreateSession(napi_env env, napi_callback_info info) {
  return WrapStringMethod(env, info, [](napi_env inner_env, napi_callback_info inner_info) {
    const std::string model_path = GetStringArgument(inner_env, inner_info, 0);
    const std::string session_options_json = GetStringArgument(inner_env, inner_info, 1);
    return std::string(ortCreateSession(model_path.c_str(), session_options_json.c_str()));
  });
}

napi_value NapiCreateOrtValue(napi_env env, napi_callback_info info) {
  return WrapStringMethod(env, info, [](napi_env inner_env, napi_callback_info inner_info) {
    const std::string source_type = GetStringArgument(inner_env, inner_info, 0);
    const std::string data_json = GetStringArgument(inner_env, inner_info, 1);
    const std::string shape_json = GetStringArgument(inner_env, inner_info, 2);
    return std::string(ortCreateOrtValue(source_type.c_str(), data_json.c_str(), shape_json.c_str()));
  });
}

napi_value NapiCreateOrtValueFromBinaryFile(napi_env env, napi_callback_info info) {
  return WrapStringMethod(env, info, [](napi_env inner_env, napi_callback_info inner_info) {
    const std::string source_type = GetStringArgument(inner_env, inner_info, 0);
    const std::string file_path = GetStringArgument(inner_env, inner_info, 1);
    const std::string shape_json = GetStringArgument(inner_env, inner_info, 2);
    return std::string(ortCreateOrtValueFromBinaryFile(source_type.c_str(), file_path.c_str(), shape_json.c_str()));
  });
}

napi_value NapiGetOrtValueData(napi_env env, napi_callback_info info) {
  return WrapStringMethod(env, info, [](napi_env inner_env, napi_callback_info inner_info) {
    const std::string value_id = GetStringArgument(inner_env, inner_info, 0);
    return std::string(ortGetOrtValueData(value_id.c_str()));
  });
}

napi_value NapiWriteOrtValueDataToBinaryFile(napi_env env, napi_callback_info info) {
  return WrapStringMethod(env, info, [](napi_env inner_env, napi_callback_info inner_info) {
    const std::string value_id = GetStringArgument(inner_env, inner_info, 0);
    const std::string file_path = GetStringArgument(inner_env, inner_info, 1);
    return std::string(ortWriteOrtValueDataToBinaryFile(value_id.c_str(), file_path.c_str()));
  });
}

napi_value NapiRunInference(napi_env env, napi_callback_info info) {
  return WrapStringMethod(env, info, [](napi_env inner_env, napi_callback_info inner_info) {
    const std::string session_id = GetStringArgument(inner_env, inner_info, 0);
    const std::string inputs_json = GetStringArgument(inner_env, inner_info, 1);
    const std::string run_options_json = GetStringArgument(inner_env, inner_info, 2);
    return std::string(ortRunInference(session_id.c_str(), inputs_json.c_str(), run_options_json.c_str()));
  });
}

napi_value NapiCloseSession(napi_env env, napi_callback_info info) {
  return WrapIntMethod(env, info, [](napi_env inner_env, napi_callback_info inner_info) {
    const std::string session_id = GetStringArgument(inner_env, inner_info, 0);
    return ortCloseSession(session_id.c_str());
  });
}

napi_value NapiReleaseOrtValue(napi_env env, napi_callback_info info) {
  return WrapIntMethod(env, info, [](napi_env inner_env, napi_callback_info inner_info) {
    const std::string value_id = GetStringArgument(inner_env, inner_info, 0);
    return ortReleaseOrtValue(value_id.c_str());
  });
}

napi_value Init(napi_env env, napi_value exports) {
  static const napi_property_descriptor properties[] = {
      {"ortGetPlatformVersion", nullptr, NapiGetPlatformVersion, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"ortGetAvailableProvidersJson", nullptr, NapiGetAvailableProvidersJson, nullptr, nullptr, nullptr, napi_default,
       nullptr},
      {"ortCreateSession", nullptr, NapiCreateSession, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"ortCreateOrtValue", nullptr, NapiCreateOrtValue, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"ortCreateOrtValueFromBinaryFile", nullptr, NapiCreateOrtValueFromBinaryFile, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"ortGetOrtValueData", nullptr, NapiGetOrtValueData, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"ortWriteOrtValueDataToBinaryFile", nullptr, NapiWriteOrtValueDataToBinaryFile, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"ortRunInference", nullptr, NapiRunInference, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"ortCloseSession", nullptr, NapiCloseSession, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"ortReleaseOrtValue", nullptr, NapiReleaseOrtValue, nullptr, nullptr, nullptr, napi_default, nullptr},
  };
  napi_define_properties(env, exports, sizeof(properties) / sizeof(properties[0]), properties);
  return exports;
}

} // namespace

NAPI_MODULE(flutter_onnxruntime_ohos, Init)

const char *ortGetPlatformVersion() {
  g_response_buffer = "OpenHarmony";
  return g_response_buffer.c_str();
}

const char *ortGetAvailableProvidersJson() {
  try {
    g_response_buffer = ToJsonStringArray(Ort::GetAvailableProviders());
  } catch (const std::exception &exception) {
    g_response_buffer = ErrorJson("ORT_PROVIDER_ERROR", exception.what());
  }
  return g_response_buffer.c_str();
}

const char *ortCreateSession(const char *model_path, const char *session_options_json) {
  try {
    if (model_path == nullptr || std::string(model_path).empty()) {
      g_response_buffer = ErrorJson("INVALID_ARGUMENT", "模型路径不能为空");
      return g_response_buffer.c_str();
    }

    if (!FileExists(model_path)) {
      g_response_buffer = ErrorJson("FILE_NOT_FOUND", "模型文件不存在");
      return g_response_buffer.c_str();
    }

    Ort::SessionOptions session_options;
    const std::string raw_options = session_options_json == nullptr ? "" : session_options_json;
    const int intra_threads = ExtractIntOption(raw_options, "intraOpNumThreads");
    if (intra_threads > 0) {
      session_options.SetIntraOpNumThreads(intra_threads);
    }

    const int inter_threads = ExtractIntOption(raw_options, "interOpNumThreads");
    if (inter_threads > 0) {
      session_options.SetInterOpNumThreads(inter_threads);
    }

    const std::string session_id = g_session_manager.CreateSession(model_path, &session_options);
    const std::vector<std::string> input_names = g_session_manager.GetInputNames(session_id);
    const std::vector<std::string> output_names = g_session_manager.GetOutputNames(session_id);
    g_response_buffer = "{\"sessionId\":\"" + EscapeJson(session_id) + "\",\"inputNames\":" + ToJsonStringArray(input_names) +
                        ",\"outputNames\":" + ToJsonStringArray(output_names) + "}";
  } catch (const Ort::Exception &exception) {
    g_response_buffer = ErrorJson("SESSION_CREATION_FAILED", exception.what());
  } catch (const std::exception &exception) {
    g_response_buffer = ErrorJson("SESSION_CREATION_FAILED", exception.what());
  }

  return g_response_buffer.c_str();
}

const char *ortCreateOrtValue(const char *source_type, const char *data_json, const char *shape_json) {
  try {
    if (source_type == nullptr || data_json == nullptr || shape_json == nullptr) {
      g_response_buffer = ErrorJson("INVALID_ARGUMENT", "创建 OrtValue 的参数不完整");
      return g_response_buffer.c_str();
    }

    const std::vector<int64_t> shape = ParseShape(shape_json);
    const std::string value_id = g_tensor_store.CreateTensor(source_type, data_json, shape);
    g_response_buffer = "{\"valueId\":\"" + EscapeJson(value_id) + "\",\"dataType\":\"" + EscapeJson(source_type) +
                        "\",\"shape\":" + ToJsonIntArray(shape) + "}";
  } catch (const Ort::Exception &exception) {
    g_response_buffer = ErrorJson("ORT_VALUE_CREATION_FAILED", exception.what());
  } catch (const std::exception &exception) {
    g_response_buffer = ErrorJson("ORT_VALUE_CREATION_FAILED", exception.what());
  }

  return g_response_buffer.c_str();
}

const char *ortCreateOrtValueFromBinaryFile(const char *source_type, const char *file_path, const char *shape_json) {
  try {
    if (source_type == nullptr || file_path == nullptr || shape_json == nullptr) {
      g_response_buffer = ErrorJson("INVALID_ARGUMENT", "创建二进制 OrtValue 的参数不完整");
      return g_response_buffer.c_str();
    }

    const std::vector<int64_t> shape = ParseShape(shape_json);
    const std::string value_id = g_tensor_store.CreateTensorFromBinaryFile(source_type, file_path, shape);
    g_response_buffer = "{\"valueId\":\"" + EscapeJson(value_id) + "\",\"dataType\":\"" + EscapeJson(source_type) +
                        "\",\"shape\":" + ToJsonIntArray(shape) + "}";
  } catch (const Ort::Exception &exception) {
    g_response_buffer = ErrorJson("ORT_VALUE_CREATION_FAILED", exception.what());
  } catch (const std::exception &exception) {
    g_response_buffer = ErrorJson("ORT_VALUE_CREATION_FAILED", exception.what());
  }

  return g_response_buffer.c_str();
}

const char *ortGetOrtValueData(const char *value_id) {
  try {
    if (value_id == nullptr || std::string(value_id).empty()) {
      g_response_buffer = ErrorJson("INVALID_ARGUMENT", "valueId 不能为空");
      return g_response_buffer.c_str();
    }
    g_response_buffer = g_tensor_store.GetTensorDataJson(value_id);
  } catch (const Ort::Exception &exception) {
    g_response_buffer = ErrorJson("ORT_VALUE_READ_FAILED", exception.what());
  } catch (const std::exception &exception) {
    g_response_buffer = ErrorJson("ORT_VALUE_READ_FAILED", exception.what());
  }
  return g_response_buffer.c_str();
}

const char *ortWriteOrtValueDataToBinaryFile(const char *value_id, const char *file_path) {
  try {
    if (value_id == nullptr || std::string(value_id).empty() || file_path == nullptr || std::string(file_path).empty()) {
      g_response_buffer = ErrorJson("INVALID_ARGUMENT", "valueId 和 filePath 不能为空");
      return g_response_buffer.c_str();
    }

    g_response_buffer = g_tensor_store.WriteTensorDataToBinaryFile(value_id, file_path);
  } catch (const Ort::Exception &exception) {
    g_response_buffer = ErrorJson("ORT_VALUE_WRITE_FAILED", exception.what());
  } catch (const std::exception &exception) {
    g_response_buffer = ErrorJson("ORT_VALUE_WRITE_FAILED", exception.what());
  }

  return g_response_buffer.c_str();
}

const char *ortRunInference(const char *session_id, const char *inputs_json, const char *run_options_json) {
  try {
    if (session_id == nullptr || std::string(session_id).empty()) {
      g_response_buffer = ErrorJson("INVALID_ARGUMENT", "sessionId 不能为空");
      return g_response_buffer.c_str();
    }

    if (inputs_json == nullptr || std::string(inputs_json).empty()) {
      g_response_buffer = ErrorJson("INVALID_ARGUMENT", "inputs 不能为空");
      return g_response_buffer.c_str();
    }

    std::vector<std::pair<std::string, std::string>> input_refs = ParseInputRefs(inputs_json);
    if (input_refs.empty()) {
      g_response_buffer = ErrorJson("INVALID_ARGUMENT", "inputs 中未找到有效的 OrtValue 引用");
      return g_response_buffer.c_str();
    }

    Ort::RunOptions run_options;
    const std::string raw_run_options = run_options_json == nullptr ? "" : run_options_json;
    const int log_severity_level = ExtractIntOption(raw_run_options, "logSeverityLevel");
    if (log_severity_level >= 0) {
      run_options.SetRunLogSeverityLevel(log_severity_level);
    }
    const int log_verbosity_level = ExtractIntOption(raw_run_options, "logVerbosityLevel");
    if (log_verbosity_level >= 0) {
      run_options.SetRunLogVerbosityLevel(log_verbosity_level);
    }

    std::vector<ClonedTensor> cloned_inputs;
    std::vector<Ort::Value> input_tensors;
    std::vector<std::string> input_names;
    cloned_inputs.reserve(input_refs.size());
    input_tensors.reserve(input_refs.size());
    input_names.reserve(input_refs.size());

    for (const auto &input_ref : input_refs) {
      cloned_inputs.push_back(g_tensor_store.CloneTensor(input_ref.second));
      input_tensors.push_back(std::move(cloned_inputs.back().value));
      input_names.push_back(input_ref.first);
    }

    std::vector<Ort::Value> outputs =
        g_session_manager.RunInference(session_id, input_names, input_tensors, &run_options);
    std::vector<std::string> output_names = g_session_manager.GetOutputNames(session_id);

    std::ostringstream stream;
    stream << "{";
    for (size_t index = 0; index < outputs.size(); ++index) {
      const std::string value_id = g_tensor_store.StoreOutputTensor(std::move(outputs[index]));
      const std::string data_type = g_tensor_store.GetTensorType(value_id);
      const std::vector<int64_t> shape = g_tensor_store.GetTensorShape(value_id);
      const std::string output_name =
          index < output_names.size() ? output_names[index] : ("output_" + std::to_string(index));

      if (index > 0) {
        stream << ",";
      }
      stream << "\"" << EscapeJson(output_name) << "\":" << BuildTensorRefJson(value_id, data_type, shape);
    }
    stream << "}";
    g_response_buffer = stream.str();
  } catch (const Ort::Exception &exception) {
    g_response_buffer = ErrorJson("RUN_INFERENCE_FAILED", exception.what());
  } catch (const std::exception &exception) {
    g_response_buffer = ErrorJson("RUN_INFERENCE_FAILED", exception.what());
  }

  return g_response_buffer.c_str();
}

int ortCloseSession(const char *session_id) {
  if (session_id == nullptr || std::string(session_id).empty()) {
    return 0;
  }
  return g_session_manager.CloseSession(session_id) ? 1 : 0;
}

int ortReleaseOrtValue(const char *value_id) {
  if (value_id == nullptr || std::string(value_id).empty()) {
    return 0;
  }
  return g_tensor_store.ReleaseTensor(value_id) ? 1 : 0;
}
