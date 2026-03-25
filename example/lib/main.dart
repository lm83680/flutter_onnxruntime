import 'dart:async';
import 'dart:io';
import 'dart:isolate';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';

TransferableTypedData _buildInputTensorData(String imagePath, int width, int height) {
  final Uint8List imageBytes = File(imagePath).readAsBytesSync();
  final img.Image? decodedImage = img.decodeImage(imageBytes);
  if (decodedImage == null) {
    throw Exception('无法解码选中的图片');
  }

  final double scale = math.min(width / decodedImage.width, height / decodedImage.height);
  final int resizedWidth = math.max(1, (decodedImage.width * scale).round());
  final int resizedHeight = math.max(1, (decodedImage.height * scale).round());
  final img.Image resizedImage = img.copyResize(decodedImage, width: resizedWidth, height: resizedHeight);
  final img.Image letterboxedImage = img.Image(width: width, height: height);
  img.fill(letterboxedImage, color: img.ColorRgb8(114, 114, 114));

  final int offsetX = ((width - resizedWidth) / 2).floor();
  final int offsetY = ((height - resizedHeight) / 2).floor();
  img.compositeImage(letterboxedImage, resizedImage, dstX: offsetX, dstY: offsetY);

  final Float32List inputData = Float32List(1 * 3 * width * height);
  int pixelIndex = 0;
  for (int channel = 0; channel < 3; channel++) {
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final pixel = letterboxedImage.getPixel(x, y);
        final double value = switch (channel) {
              0 => pixel.r.toDouble(),
              1 => pixel.g.toDouble(),
              _ => pixel.b.toDouble(),
            } /
            255.0;
        inputData[pixelIndex++] = value;
      }
    }
  }

  return TransferableTypedData.fromList([inputData.buffer.asUint8List()]);
}

class PredictionMetrics {
  final double confidence;
  final int preprocessMs;
  final int inferenceMs;
  final int totalMs;

  const PredictionMetrics({
    required this.confidence,
    required this.preprocessMs,
    required this.inferenceMs,
    required this.totalMs,
  });
}

void _debugLogTensorPreview(String label, List<double> values, {int previewCount = 8}) {
  if (values.isEmpty) {
    debugPrint('$label -> empty');
    return;
  }

  final int count = math.min(previewCount, values.length);
  final List<String> preview = values.take(count).map((value) => value.toStringAsFixed(6)).toList(growable: false);
  final double minValue = values.reduce(math.min);
  final double maxValue = values.reduce(math.max);
  debugPrint('$label -> len=${values.length}, min=${minValue.toStringAsFixed(6)}, max=${maxValue.toStringAsFixed(6)}, first=$preview');
}

double _normalizeScoreValue(double rawValue) {
  if (rawValue >= 0.0 && rawValue <= 1.0) {
    return rawValue;
  }
  return 1.0 / (1.0 + math.exp(-rawValue));
}

double _extractBestScoreFromValues(List<double> values, List<int> shape, int classCount, int classIndex) {
  if (shape.length != 4) {
    return 0.0;
  }

  final int channels = shape[1];
  final int gridHeight = shape[2];
  final int gridWidth = shape[3];
  final int attributesPerAnchor = 5 + classCount;
  if (channels % attributesPerAnchor != 0) {
    return 0.0;
  }

  final int anchorCount = channels ~/ attributesPerAnchor;
  final int gridSize = gridHeight * gridWidth;
  double bestScore = 0.0;

  for (int anchorIndex = 0; anchorIndex < anchorCount; anchorIndex++) {
    final int anchorBase = anchorIndex * attributesPerAnchor;
    for (int gridIndex = 0; gridIndex < gridSize; gridIndex++) {
      final double objectness = _normalizeScoreValue(values[(anchorBase + 4) * gridSize + gridIndex]);
      final double classScore = _normalizeScoreValue(values[(anchorBase + 5 + classIndex) * gridSize + gridIndex]);
      bestScore = math.max(bestScore, objectness * classScore);
    }
  }

  return bestScore;
}

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: const CatDetectorPage(),
      theme: ThemeData(colorScheme: ColorScheme.fromSeed(seedColor: Colors.orange)),
    );
  }
}

class CatDetectorPage extends StatefulWidget {
  const CatDetectorPage({super.key});

  @override
  State<CatDetectorPage> createState() => _CatDetectorPageState();
}

class _CatDetectorPageState extends State<CatDetectorPage> {
  static const String _modelAssetPath = 'assets/models/is_cat.onnx';
  static const int _classCount = 80;
  static const int _catClassIndex = 15;
  static const int _inputWidth = 640;
  static const int _inputHeight = 640;

  final OnnxRuntime _onnxRuntime = OnnxRuntime();
  final ImagePicker _imagePicker = ImagePicker();

  OrtSession? _session;
  bool _isLoading = false;
  String _resultText = '点击按钮后选择图片，完成后会显示置信度和总耗时';

  @override
  void initState() {
    super.initState();
    unawaited(_initializeSession());
  }

  Future<void> _initializeSession() async {
    try {
      _session ??= await _onnxRuntime.createSessionFromAsset(_modelAssetPath);
    } catch (error) {
      _setStatus('模型初始化失败：$error');
    }
  }

  @override
  void dispose() {
    unawaited(_session?.close());
    super.dispose();
  }

  Future<void> _pickAndPredict() async {
    try {
      final XFile? file = await _imagePicker.pickImage(
        source: ImageSource.gallery,
        requestFullMetadata: false,
        maxWidth: 640,
        maxHeight: 640,
      );
      if (file == null) {
        return;
      }

      setState(() {
        _isLoading = true;
        _resultText = '正在检测中...';
      });

      final PredictionMetrics metrics = await _runPrediction(file.path);

      final String summary =
          '置信度：${metrics.confidence.toStringAsFixed(6)}\n预处理耗时：${metrics.preprocessMs} ms\n检测耗时：${metrics.inferenceMs} ms\n总耗时：${metrics.totalMs} ms';
      debugPrint(summary);

      if (!mounted) {
        return;
      }
      setState(() {
        _resultText = summary;
        _isLoading = false;
      });
    } catch (error) {
      _setStatus('检测失败：$error');
    }
  }

  Future<PredictionMetrics> _runPrediction(String imagePath) async {
    final List<OrtValue> createdTensors = <OrtValue>[];
    final Stopwatch totalStopwatch = Stopwatch()..start();
    final Stopwatch preprocessStopwatch = Stopwatch();
    final Stopwatch inferenceStopwatch = Stopwatch();
    try {
      await _initializeSession();
      preprocessStopwatch.start();
      final OrtValue inputTensor = await _createInputTensor(imagePath);
      preprocessStopwatch.stop();
      createdTensors.add(inputTensor);

      inferenceStopwatch.start();
      final String inputName = _session!.inputNames.first;
      final Map<String, OrtValue> outputs = await _session!.run(<String, OrtValue>{inputName: inputTensor});
      createdTensors.addAll(outputs.values);
      final double confidence = await _extractCatConfidence(outputs);
      inferenceStopwatch.stop();
      totalStopwatch.stop();

      return PredictionMetrics(
        confidence: confidence,
        preprocessMs: preprocessStopwatch.elapsedMilliseconds,
        inferenceMs: inferenceStopwatch.elapsedMilliseconds,
        totalMs: totalStopwatch.elapsedMilliseconds,
      );
    } finally {
      for (final OrtValue tensor in createdTensors) {
        await tensor.dispose();
      }
    }
  }

  Future<OrtValue> _createInputTensor(String imagePath) async {
    final TransferableTypedData transferableData = await Isolate.run(() => _buildInputTensorData(imagePath, _inputWidth, _inputHeight));
    final Uint8List rawBytes = transferableData.materialize().asUint8List();
    final Float32List inputData = rawBytes.buffer.asFloat32List(rawBytes.offsetInBytes, rawBytes.lengthInBytes ~/ Float32List.bytesPerElement);
    _debugLogTensorPreview(
      'InputTensor(non-OHOS)',
      inputData.take(32).map((value) => value.toDouble()).toList(growable: false),
      previewCount: 8,
    );
    return OrtValue.fromList(inputData, const <int>[1, 3, _inputHeight, _inputWidth]);
  }

  Future<double> _extractOutputBestScore(OrtValue output, String outputName, List<int> shape, int classIndex) async {
    final Float32List values = await output.asFloat32List();
    final List<double> doubleValues = values.map((value) => value.toDouble()).toList(growable: false);
    _debugLogTensorPreview('OutputTensor:$outputName shape=$shape', doubleValues, previewCount: 8);
    return Isolate.run(() => _extractBestScoreFromValues(doubleValues, shape, _classCount, classIndex));
  }

  Future<double> _extractCatConfidence(Map<String, OrtValue> outputs) async {
    double bestScore = 0.0;

    for (final String outputName in _session!.outputNames) {
      final OrtValue? output = outputs[outputName];
      if (output == null) {
        continue;
      }

      final double outputBestScore = await _extractOutputBestScore(output, outputName, output.shape, _catClassIndex);
      bestScore = math.max(bestScore, outputBestScore);
    }

    return bestScore;
  }

  void _setStatus(String message) {
    if (!mounted) {
      return;
    }
    setState(() {
      _resultText = message;
      _isLoading = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Cat Detector Benchmark')),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: <Widget>[
              ElevatedButton(
                onPressed: _isLoading ? null : _pickAndPredict,
                child: _isLoading
                    ? const SizedBox(
                        width: 18,
                        height: 18,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      )
                    : const Text('选择图片并检测'),
              ),
              const SizedBox(height: 24),
              Text(
                _resultText,
                textAlign: TextAlign.center,
                style: Theme.of(context).textTheme.titleMedium,
              ),
            ],
          ),
        ),
      ),
    );
  }
}
