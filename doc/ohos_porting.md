# OHOS 适配说明

日期：2026-03-25

执行者：Codex

## 当前进度

当前仓库已经补齐 OHOS 插件骨架，并接入了最小原生包装层：

- Dart 侧继续复用现有 `MethodChannel('flutter_onnxruntime')` 协议。
- OHOS 侧新增 `ohos/` 插件目录，使用 ETS 方法通道实现。
- 原生侧新增 `src/` CMake 工程，负责链接 OHOS 版 `libonnxruntime.so`。

## 已实现能力

当前 OHOS 侧已接通以下方法：

- `getPlatformVersion`
- `getAvailableProviders`
- `createSession`
- `createOrtValue`
- `getOrtValueData`
- `releaseOrtValue`
- `runInference`
- `closeSession`

以下方法暂未实现：

- `getMetadata`
- `getInputInfo`
- `getOutputInfo`
- `convertOrtValue`

## 预编译库放置方式

当前 CMake 优先按以下顺序查找 ONNX Runtime：

- 显式传入的 `OHOS_ONNXRUNTIME_ROOT_DIR`
- 仓库内置路径 `third_party/onnxruntime-ohos-arm64-v8a-1.16.3`

目录结构要求如下：

```text
$OHOS_ONNXRUNTIME_ROOT_DIR/
  include/
    onnxruntime_cxx_api.h
    ...
  lib/
    libonnxruntime.so
```

例如：

```bash
export OHOS_ONNXRUNTIME_ROOT_DIR=/absolute/path/to/onnxruntime-ohos-arm64-v8a-1.16.3
```

当前仓库已内置一份可用于验证的预编译目录：

```text
third_party/onnxruntime-ohos-arm64-v8a-1.16.3
```

## 当前设计取舍

- 先打通 `Session` 生命周期，确认 OHOS 平台可成功加载模型与链接 ORT。
- 当前 `OrtValue` 与推理执行已落地最小可用版本，首批仅覆盖 `float32/int32/int64` 三类数值张量。
- 原生层当前使用轻量字符串 JSON 作为 ETS 与 C++ 间的交换格式，便于快速验证链路。
- 原生导出层已从普通共享库调整为 OHOS N-API addon，避免 ArkTS 在运行时导入 `.so` 时出现插件初始化失败。

## 运行时问题修复

针对真机上出现的 `MissingPluginException(No implementation found for method createSession on channel flutter_onnxruntime)`，本轮已完成以下修复：

- 保留现有 `ortCreateSession`、`ortCreateOrtValue`、`ortRunInference` 等 C++ 实现，外层补充 N-API `exports` 注册。
- 通过 `NAPI_MODULE(flutter_onnxruntime_ohos, Init)` 为 `libflutter_onnxruntime_ohos.so` 提供标准模块注册入口。
- 在 `src/CMakeLists.txt` 中显式链接 `ace_napi.z`，解决 `napi_*` 符号在 OHOS 链接阶段缺失的问题。

当前已验证：

- `flutter analyze` 通过
- `flutter build hap -v` 通过
- 最新 HAP 已重新签名生成

当前仍待验证：

- 真机或模拟器重新安装最新 HAP 后，确认 `createSession` 注册成功并完成一次实际推理

## 目标能力边界

本次 OHOS 适配不追求全平台功能完全一致，优先与 `README.md` 中的 iOS 实现状态对齐：

- 需要支持：`CPU Inference`、`EP Configuration`、`Input/Output names`、`Data Type Conversion`、`Inference on Emulator`、`FP16 Support`
- 允许暂不支持：`Input/Output Info`、`Model Metadata`

## 下一步建议

建议按下面顺序继续推进：

1. 在真机或模拟器上执行一次 `addition_model.ort` 端到端推理，确认运行时结果正确。
2. 继续补齐 `getMetadata`、`getInputInfo`、`getOutputInfo`。
3. 扩展 `OrtValue` 数据类型覆盖范围，并补上 `convertOrtValue`。
4. 按 iOS 对齐范围补齐剩余必要能力。
