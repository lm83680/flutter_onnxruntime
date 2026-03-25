#ifndef FLUTTER_ONNXRUNTIME_OHOS_WRAPPER_H_
#define FLUTTER_ONNXRUNTIME_OHOS_WRAPPER_H_

#if defined(_WIN32)
#define ORT_OHOS_EXPORT __declspec(dllexport)
#else
#define ORT_OHOS_EXPORT __attribute__((visibility("default"))) __attribute__((used))
#endif

#ifdef __cplusplus
extern "C" {
#endif

ORT_OHOS_EXPORT const char *ortGetPlatformVersion();
ORT_OHOS_EXPORT const char *ortGetAvailableProvidersJson();
ORT_OHOS_EXPORT const char *ortCreateSession(const char *model_path, const char *session_options_json);
ORT_OHOS_EXPORT const char *ortCreateOrtValue(const char *source_type, const char *data_json, const char *shape_json);
ORT_OHOS_EXPORT const char *ortCreateOrtValueFromBinaryFile(const char *source_type, const char *file_path, const char *shape_json);
ORT_OHOS_EXPORT const char *ortGetOrtValueData(const char *value_id);
ORT_OHOS_EXPORT const char *ortWriteOrtValueDataToBinaryFile(const char *value_id, const char *file_path);
ORT_OHOS_EXPORT const char *ortRunInference(const char *session_id, const char *inputs_json, const char *run_options_json);
ORT_OHOS_EXPORT int ortCloseSession(const char *session_id);
ORT_OHOS_EXPORT int ortReleaseOrtValue(const char *value_id);

#ifdef __cplusplus
}
#endif

#endif // FLUTTER_ONNXRUNTIME_OHOS_WRAPPER_H_
