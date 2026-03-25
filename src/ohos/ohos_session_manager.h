#ifndef FLUTTER_ONNXRUNTIME_OHOS_SESSION_MANAGER_H_
#define FLUTTER_ONNXRUNTIME_OHOS_SESSION_MANAGER_H_

#include <map>
#include <memory>
#include <mutex>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

struct OhosSessionInfo {
  std::unique_ptr<Ort::Session> session;
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
};

class OhosSessionManager {
public:
  OhosSessionManager();
  ~OhosSessionManager();

  std::string CreateSession(const char *model_path, Ort::SessionOptions *options = nullptr);
  std::vector<Ort::Value> RunInference(const std::string &session_id, const std::vector<std::string> &input_names,
                                       std::vector<Ort::Value> &input_tensors, Ort::RunOptions *options = nullptr);
  bool CloseSession(const std::string &session_id);
  bool HasSession(const std::string &session_id);
  std::vector<std::string> GetInputNames(const std::string &session_id);
  std::vector<std::string> GetOutputNames(const std::string &session_id);

private:
  std::string GenerateSessionId();

  std::map<std::string, OhosSessionInfo> sessions_;
  int next_session_id_;
  std::mutex mutex_;
  Ort::Env env_;
};

#endif // FLUTTER_ONNXRUNTIME_OHOS_SESSION_MANAGER_H_
