import 'dart:async';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import 'package:path_provider/path_provider.dart';

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
  static const List<String> _classNames = ['cat', 'tool', 'person'];
  static const double _catThreshold = 0.25;
  static const int _inputWidth = 640;
  static const int _inputHeight = 640;
  static const MethodChannel _nativeChannel = MethodChannel('flutter_onnxruntime');

  final OnnxRuntime _onnxRuntime = OnnxRuntime();
  final ImagePicker _imagePicker = ImagePicker();

  OrtSession? _session;
  Uint8List? _selectedImageBytes;
  String _predictionText = '请选择一张图片';
  double? _catConfidence;
  bool _isLoading = false;
  final String _modelInfoText = '模型输入：images [1, 3, 640, 640]\n模型类别：cat / tool / person';
  bool? _isOhos;

  @override
  void initState() {
    super.initState();
    unawaited(_initializeSession());
  }

  Future<void> _initializeSession() async {
    try {
      _session ??= await _onnxRuntime.createSessionFromAsset(_modelAssetPath);
      _isOhos ??= await _onnxRuntime.getPlatformVersion() == 'OpenHarmony';
    } catch (error) {
      _setStatus('模型初始化失败：$error');
    }
  }

  @override
  void dispose() {
    unawaited(_session?.close());
    super.dispose();
  }

  Future<void> _pickImage() async {
    try {
      final XFile? file = await _imagePicker.pickImage(source: ImageSource.gallery, requestFullMetadata: false);
      if (file == null) {
        return;
      }

      final Uint8List bytes = await file.readAsBytes();
      if (!mounted) {
        return;
      }
      setState(() {
        _selectedImageBytes = bytes;
        _predictionText = '图片已选择，可以开始预测';
        _catConfidence = null;
      });
    } catch (error) {
      _setStatus('选图失败：$error');
    }
  }

  Future<void> _predictCat() async {
    if (_selectedImageBytes == null) {
      _setStatus('请先选择一张图片');
      return;
    }

    final createdTensors = <OrtValue>[];

    setState(() {
      _isLoading = true;
      _predictionText = '正在预测中...';
    });

    try {
      await _initializeSession();
      final OrtValue inputTensor = await _createInputTensor(_selectedImageBytes!);
      createdTensors.add(inputTensor);

      final String inputName = _session!.inputNames.first;
      final Map<String, OrtValue> outputs = await _session!.run({inputName: inputTensor});
      createdTensors.addAll(outputs.values);

      final double catScore = await _extractCatConfidence(outputs);
      final bool hasCat = catScore >= _catThreshold;

      if (!mounted) {
        return;
      }
      setState(() {
        _catConfidence = catScore;
        _predictionText = hasCat ? '检测到猫咪' : '未检测到猫咪';
      });
    } catch (error) {
      _setStatus('预测失败：$error');
    } finally {
      for (final tensor in createdTensors) {
        await tensor.dispose();
      }

      if (mounted) {
        setState(() {
          _isLoading = false;
        });
      }
    }
  }

  Future<OrtValue> _createInputTensor(Uint8List imageBytes) async {
    final img.Image? decodedImage = img.decodeImage(imageBytes);
    if (decodedImage == null) {
      throw Exception('无法解码选中的图片');
    }

    final img.Image resizedImage = img.copyResize(decodedImage, width: _inputWidth, height: _inputHeight);
    final Float32List inputData = Float32List(1 * 3 * _inputWidth * _inputHeight);

    int pixelIndex = 0;
    for (int channel = 0; channel < 3; channel++) {
      for (int y = 0; y < _inputHeight; y++) {
        for (int x = 0; x < _inputWidth; x++) {
          final pixel = resizedImage.getPixel(x, y);
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

    if (_isOhos == true) {
      final Directory tempDirectory = await getTemporaryDirectory();
      final File inputFile = File('${tempDirectory.path}/cat_detector_input.bin');
      await inputFile.writeAsBytes(inputData.buffer.asUint8List(), flush: true);

      final Map<Object?, Object?>? result = await _nativeChannel.invokeMethod<Map<Object?, Object?>>('createOrtValueFromBinaryFile', {
        'sourceType': 'float32',
        'filePath': inputFile.path,
        'shape': const [1, 3, _inputHeight, _inputWidth],
      });
      return OrtValue.fromMap(Map<String, dynamic>.from(result ?? <String, dynamic>{}));
    }

    return OrtValue.fromList(inputData, const [1, 3, _inputHeight, _inputWidth]);
  }

  Future<double> _extractCatConfidence(Map<String, OrtValue> outputs) async {
    const int catClassIndex = 0;
    double bestScore = 0.0;

    for (final String outputName in _session!.outputNames) {
      final OrtValue? output = outputs[outputName];
      if (output == null) {
        continue;
      }

      final List<double> values = await _readOutputValues(output, outputName);
      final List<int> shape = output.shape;

      if (shape.length != 4) {
        continue;
      }

      final int channels = shape[1];
      final int gridHeight = shape[2];
      final int gridWidth = shape[3];
      final int attributesPerAnchor = 5 + _classNames.length;
      if (channels % attributesPerAnchor != 0) {
        continue;
      }

      final int anchorCount = channels ~/ attributesPerAnchor;
      final int gridSize = gridHeight * gridWidth;

      for (int anchorIndex = 0; anchorIndex < anchorCount; anchorIndex++) {
        final int anchorBase = anchorIndex * attributesPerAnchor;
        for (int gridIndex = 0; gridIndex < gridSize; gridIndex++) {
          final double objectness = _normalizeScore(values[(anchorBase + 4) * gridSize + gridIndex]);
          final double classScore = _normalizeScore(values[(anchorBase + 5 + catClassIndex) * gridSize + gridIndex]);
          bestScore = math.max(bestScore, objectness * classScore);
        }
      }
    }

    return bestScore;
  }

  Future<List<double>> _readOutputValues(OrtValue output, String outputName) async {
    if (_isOhos != true) {
      final List<dynamic> rawValues = await output.asFlattenedList();
      return rawValues.map((value) => (value as num).toDouble()).toList();
    }

    final Directory tempDirectory = await getTemporaryDirectory();
    final File outputFile = File('${tempDirectory.path}/$outputName.bin');
    final Map<Object?, Object?>? metadata = await _nativeChannel.invokeMethod<Map<Object?, Object?>>('writeOrtValueDataToBinaryFile', {
      'valueId': output.id,
      'filePath': outputFile.path,
    });

    final Map<String, dynamic> metadataMap = Map<String, dynamic>.from(metadata ?? <String, dynamic>{});
    if (metadataMap['dataType'] != 'float32') {
      throw Exception('当前猫咪检测 Demo 仅支持 float32 输出');
    }

    final Uint8List rawBytes = await outputFile.readAsBytes();
    final Float32List floatValues = rawBytes.buffer.asFloat32List(rawBytes.offsetInBytes, rawBytes.lengthInBytes ~/ Float32List.bytesPerElement);
    return floatValues.map((value) => value.toDouble()).toList();
  }

  double _normalizeScore(double rawValue) {
    if (rawValue >= 0.0 && rawValue <= 1.0) {
      return rawValue;
    }
    return 1.0 / (1.0 + math.exp(-rawValue));
  }

  void _setStatus(String message) {
    if (!mounted) {
      return;
    }
    setState(() {
      _predictionText = message;
      _isLoading = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Cat Detector Demo')),
      body: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Expanded(
              child: DecoratedBox(
                decoration: BoxDecoration(
                  color: Colors.grey.shade100,
                  borderRadius: BorderRadius.circular(16),
                  border: Border.all(color: Colors.grey.shade300),
                ),
                child: _selectedImageBytes == null
                    ? const Center(child: Text('请选择一张待检测图片'))
                    : ClipRRect(
                        borderRadius: BorderRadius.circular(16),
                        child: Image.memory(_selectedImageBytes!, fit: BoxFit.contain),
                      ),
              ),
            ),
            const SizedBox(height: 16),
            Text(_modelInfoText, style: Theme.of(context).textTheme.bodySmall),
            const SizedBox(height: 12),
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(_predictionText, style: Theme.of(context).textTheme.titleMedium),
                    const SizedBox(height: 8),
                    Text(
                      _catConfidence == null ? '猫咪置信度：-' : '猫咪置信度：${_catConfidence!.toStringAsFixed(4)}',
                      style: Theme.of(context).textTheme.bodyMedium,
                    ),
                    const SizedBox(height: 4),
                    Text('判定阈值：${_catThreshold.toStringAsFixed(2)}', style: Theme.of(context).textTheme.bodySmall),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 12),
            Row(
              children: [
                Expanded(
                  child: OutlinedButton(
                    onPressed: _isLoading ? null : _pickImage,
                    child: const Text('从相册选择'),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: ElevatedButton(
                    onPressed: _isLoading ? null : _predictCat,
                    child: _isLoading
                        ? const SizedBox(
                            width: 18,
                            height: 18,
                            child: CircularProgressIndicator(strokeWidth: 2),
                          )
                        : const Text('开始预测'),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
