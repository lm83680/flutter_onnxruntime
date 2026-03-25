cmake_minimum_required(VERSION 3.10)

project(flutter_onnxruntime_ohos VERSION 0.0.1 LANGUAGES CXX)

set(ONNXRUNTIME_ROOT_DIR "" CACHE PATH "ONNX Runtime OHOS 根目录")
set(ONNXRUNTIME_DEFAULT_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/onnxruntime-ohos-arm64-v8a-1.16.3")

if(NOT ONNXRUNTIME_ROOT_DIR AND DEFINED ENV{OHOS_ONNXRUNTIME_ROOT_DIR} AND NOT "$ENV{OHOS_ONNXRUNTIME_ROOT_DIR}" STREQUAL "")
  set(ONNXRUNTIME_ROOT_DIR "$ENV{OHOS_ONNXRUNTIME_ROOT_DIR}" CACHE PATH "ONNX Runtime OHOS 根目录" FORCE)
elseif(NOT ONNXRUNTIME_ROOT_DIR AND EXISTS "${ONNXRUNTIME_DEFAULT_ROOT_DIR}")
  set(ONNXRUNTIME_ROOT_DIR "${ONNXRUNTIME_DEFAULT_ROOT_DIR}" CACHE PATH "ONNX Runtime OHOS 根目录" FORCE)
endif()

set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT_DIR}/include")
set(ONNXRUNTIME_LIBRARY "${ONNXRUNTIME_ROOT_DIR}/lib/libonnxruntime.so")

if(NOT EXISTS "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_cxx_api.h" OR NOT EXISTS "${ONNXRUNTIME_LIBRARY}")
  message(FATAL_ERROR "未找到 OHOS 版 ONNX Runtime。请设置 OHOS_ONNXRUNTIME_ROOT_DIR，或在 third_party/ 下提供 onnxruntime-ohos-arm64-v8a-1.16.3。")
endif()

add_library(flutter_onnxruntime_ohos SHARED
  "ohos/ohos_session_manager.cc"
  "ohos/ort_ohos_wrapper.cc"
)

set_target_properties(flutter_onnxruntime_ohos PROPERTIES
  OUTPUT_NAME "flutter_onnxruntime_ohos"
)

target_compile_features(flutter_onnxruntime_ohos PRIVATE cxx_std_17)
target_compile_definitions(flutter_onnxruntime_ohos PRIVATE DART_SHARED_LIB)

target_include_directories(
  flutter_onnxruntime_ohos
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/ohos"
    "${ONNXRUNTIME_INCLUDE_DIR}"
)

target_link_libraries(
  flutter_onnxruntime_ohos
  PRIVATE
    "${ONNXRUNTIME_LIBRARY}"
    ace_napi.z
)
