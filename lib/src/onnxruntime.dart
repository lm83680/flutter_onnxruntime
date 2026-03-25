// Copyright (c) MASIC AI
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

// ignore_for_file: constant_identifier_names

import 'dart:convert';
import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:crypto/crypto.dart';
import 'package:path_provider/path_provider.dart';

import 'package:flutter_onnxruntime/src/flutter_onnxruntime_platform_interface.dart';
import 'package:flutter_onnxruntime/src/ort_provider.dart';
import 'package:flutter_onnxruntime/src/ort_session.dart';

class OnnxRuntime {
  static final Map<String, String> _assetHashCache = <String, String>{};

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
    final metadataPath = '$filePath.meta.json';

    await Directory(directory.path).create(recursive: true);
    final ByteData data = await rootBundle.load(assetKey);
    final Uint8List modelBytes = data.buffer.asUint8List();
    final String latestHash = _resolveAssetHash(assetKey, modelBytes);

    final File file = File(filePath);
    final _ModelVersionMeta? existingMeta = await _readModelVersionMeta(metadataPath);
    final bool modelUpToDate = await file.exists() && existingMeta?.hash == latestHash;

    if (!modelUpToDate) {
      await file.writeAsBytes(modelBytes, flush: true);
      await _writeModelVersionMeta(metadataPath, _ModelVersionMeta(hash: latestHash));
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

  static String _resolveAssetHash(String assetKey, Uint8List bytes) {
    final String? cachedHash = _assetHashCache[assetKey];
    if (cachedHash != null) {
      return cachedHash;
    }

    final String hash = sha256.convert(bytes).toString();
    _assetHashCache[assetKey] = hash;
    return hash;
  }

  static Future<_ModelVersionMeta?> _readModelVersionMeta(String metadataPath) async {
    final File metadataFile = File(metadataPath);
    if (!await metadataFile.exists()) {
      return null;
    }

    try {
      final String rawText = await metadataFile.readAsString();
      final Object? decoded = jsonDecode(rawText);
      if (decoded is! Map<String, dynamic>) {
        return null;
      }

      final String? hash = decoded['hash'] as String?;
      if (hash == null || hash.isEmpty) {
        return null;
      }
      return _ModelVersionMeta(hash: hash);
    } catch (_) {
      return null;
    }
  }

  static Future<void> _writeModelVersionMeta(String metadataPath, _ModelVersionMeta metadata) async {
    final File metadataFile = File(metadataPath);
    final String jsonText = jsonEncode(<String, dynamic>{
      'hash': metadata.hash,
    });
    await metadataFile.writeAsString(jsonText, flush: true);
  }
}

class _ModelVersionMeta {
  final String hash;

  const _ModelVersionMeta({required this.hash});
}
