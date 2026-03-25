// Copyright (c) MASIC AI
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

// ignore_for_file: constant_identifier_names

import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';

import 'package:flutter_onnxruntime/src/flutter_onnxruntime_platform_interface.dart';
import 'package:flutter_onnxruntime/src/ort_provider.dart';
import 'package:flutter_onnxruntime/src/ort_session.dart';

class OnnxRuntime {
  Future<String?> getPlatformVersion() {
    return FlutterOnnxruntimePlatform.instance.getPlatformVersion();
  }

  /// Create an ONNX Runtime session with the given model path
  Future<OrtSession> createSession(String modelPath, {OrtSessionOptions? options}) async {
    final result = await FlutterOnnxruntimePlatform.instance.createSession(
      modelPath,
      sessionOptions: options?.toMap() ?? {},
    );
    return OrtSession.fromMap(result);
  }

  /// Create an ONNX Runtime session from an asset model file
  ///
  /// This will extract the asset to a temporary file and use that path
  Future<OrtSession> createSessionFromAsset(String assetKey, {OrtSessionOptions? options}) async {
    if (kIsWeb) {
      throw UnsupportedError('flutter_onnxruntime does not support Web platform');
    }

    final directory = await getTemporaryDirectory();
    final fileName = assetKey.split('/').last;
    final filePath = '${directory.path}${Platform.pathSeparator}$fileName';

    final file = File(filePath);
    if (!await file.exists()) {
      await Directory(directory.path).create(recursive: true);
      final data = await rootBundle.load(assetKey);
      await file.writeAsBytes(data.buffer.asUint8List());
    }

    return createSession(filePath, options: options);
  }

  /// Get the available providers
  ///
  /// Returns a list of the available providers
  Future<List<OrtProvider>> getAvailableProviders() async {
    final providers = await FlutterOnnxruntimePlatform.instance.getAvailableProviders();
    return providers.map((p) {
      final provider = OrtProvider.values.firstWhere(
        (e) => e.name == p,
        orElse: () => throw ArgumentError('Provider $p is not a valid OrtProvider.'),
      );
      return provider;
    }).toList();
  }
}
