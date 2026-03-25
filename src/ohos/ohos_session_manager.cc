#include "ohos_session_manager.h"

#include <stdexcept>

OhosSessionManager::OhosSessionManager()
    : next_session_id_(1), env_(ORT_LOGGING_LEVEL_WARNING, "FlutterOnnxRuntimeOhos") {}

OhosSessionManager::~OhosSessionManager() {
  std::lock_guard<std::mutex> lock(mutex_);
  sessions_.clear();
}

std::string OhosSessionManager::CreateSession(const char *model_path, Ort::SessionOptions *options) {
  std::lock_guard<std::mutex> lock(mutex_);

  std::string session_id = GenerateSessionId();
  Ort::SessionOptions session_options;
  if (options != nullptr) {
    session_options = std::move(*options);
  }

  auto ort_session = std::make_unique<Ort::Session>(env_, model_path, session_options);
  OhosSessionInfo session_info;
  session_info.session = std::move(ort_session);

  Ort::AllocatorWithDefaultOptions allocator;
  const size_t input_count = session_info.session->GetInputCount();
  for (size_t index = 0; index < input_count; ++index) {
    auto input_name = session_info.session->GetInputNameAllocated(index, allocator);
    session_info.input_names.emplace_back(input_name.get());
  }

  const size_t output_count = session_info.session->GetOutputCount();
  for (size_t index = 0; index < output_count; ++index) {
    auto output_name = session_info.session->GetOutputNameAllocated(index, allocator);
    session_info.output_names.emplace_back(output_name.get());
  }

  sessions_[session_id] = std::move(session_info);
  return session_id;
}

std::vector<Ort::Value> OhosSessionManager::RunInference(const std::string &session_id,
                                                         const std::vector<std::string> &input_names,
                                                         std::vector<Ort::Value> &input_tensors,
                                                         Ort::RunOptions *options) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto iterator = sessions_.find(session_id);
  if (iterator == sessions_.end()) {
    throw std::runtime_error("未找到对应的 session");
  }

  std::vector<const char *> input_name_refs;
  input_name_refs.reserve(input_names.size());
  for (const std::string &input_name : input_names) {
    input_name_refs.push_back(input_name.c_str());
  }

  std::vector<const char *> output_name_refs;
  output_name_refs.reserve(iterator->second.output_names.size());
  for (const std::string &output_name : iterator->second.output_names) {
    output_name_refs.push_back(output_name.c_str());
  }

  Ort::RunOptions default_options;
  Ort::RunOptions &run_options = options == nullptr ? default_options : *options;
  return iterator->second.session->Run(run_options, input_name_refs.data(), input_tensors.data(), input_tensors.size(),
                                       output_name_refs.data(), output_name_refs.size());
}

bool OhosSessionManager::CloseSession(const std::string &session_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  return sessions_.erase(session_id) > 0;
}

bool OhosSessionManager::HasSession(const std::string &session_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  return sessions_.find(session_id) != sessions_.end();
}

std::vector<std::string> OhosSessionManager::GetInputNames(const std::string &session_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto iterator = sessions_.find(session_id);
  if (iterator == sessions_.end()) {
    return {};
  }
  return iterator->second.input_names;
}

std::vector<std::string> OhosSessionManager::GetOutputNames(const std::string &session_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto iterator = sessions_.find(session_id);
  if (iterator == sessions_.end()) {
    return {};
  }
  return iterator->second.output_names;
}

std::string OhosSessionManager::GenerateSessionId() {
  return "session_" + std::to_string(next_session_id_++);
}
