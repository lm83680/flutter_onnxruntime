// Copyright (c) MASIC AI
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:flutter_onnxruntime/src/flutter_onnxruntime_platform_interface.dart';
import 'package:plugin_platform_interface/plugin_platform_interface.dart';

class MockFlutterOnnxruntimePlatform with MockPlatformInterfaceMixin implements FlutterOnnxruntimePlatform {
  // Store values for assertions
  Map<String, dynamic>? lastRunInputs;

  @override
  Future<String?> getPlatformVersion() => Future.value('42');

  @override
  Future<Map<String, dynamic>> createSession(String modelPath, {Map<String, dynamic>? sessionOptions}) {
    return Future.value({
      'sessionId': 'test_session_id',
      'inputNames': ['input1', 'input2'],
      'outputNames': ['output1'],
    });
  }

  @override
  Future<Map<String, dynamic>> runInference(
    String sessionId,
    Map<String, OrtValue> inputs, {
    Map<String, dynamic>? runOptions,
  }) {
    // Store the inputs for later assertions
    lastRunInputs = {
      for (final entry in inputs.entries) entry.key: {'valueId': entry.value.id},
    };

    // Add shape info for each input
    for (final entry in inputs.entries) {
      lastRunInputs!['${entry.key}_shape'] = entry.value.shape;
    }

    // Return mock outputs - now with OrtValue properties
    return Future.value({
      'output1': [
        'tensor_001',
        'float32',
        [2, 2],
      ],
    });
  }

  @override
  Future<Map<String, dynamic>> createOrtValue(String sourceType, dynamic data, List<int> shape) {
    // Return a mock OrtValue map
    return Future.value({
      'valueId': 'test_value_id_${DateTime.now().millisecondsSinceEpoch}',
      'dataType': sourceType,
      'shape': shape,
    });
  }

  @override
  Future<Map<String, dynamic>> createOrtValueFromBinaryFile(String sourceType, String filePath, List<int> shape) {
    return Future.value({
      'valueId': 'binary_test_value_id_${DateTime.now().millisecondsSinceEpoch}',
      'dataType': sourceType,
      'shape': shape,
    });
  }

  @override
  Future<void> closeSession(String sessionId) => Future.value();

  @override
  Future<Map<String, dynamic>> getMetadata(String sessionId) {
    return Future.value({
      'producerName': 'Test Producer',
      'graphName': 'Test Graph',
      'domain': 'test.domain',
      'description': 'Test Description',
      'version': 1,
      'customMetadataMap': {},
    });
  }

  @override
  Future<List<Map<String, dynamic>>> getInputInfo(String sessionId) {
    return Future.value([
      {
        'name': 'input1',
        'shape': [2, 2],
        'type': 'FLOAT',
      },
      {
        'name': 'input2',
        'shape': [2, 2],
        'type': 'FLOAT',
      },
    ]);
  }

  @override
  Future<List<Map<String, dynamic>>> getOutputInfo(String sessionId) {
    return Future.value([
      {
        'name': 'output1',
        'shape': [2, 2],
        'type': 'FLOAT',
      },
    ]);
  }

  @override
  Future<Map<String, dynamic>> convertOrtValue(String valueId, String targetType) {
    return Future.value({
      'valueId': valueId,
      'dataType': targetType,
      'shape': [2, 2],
    });
  }

  @override
  Future<Map<String, dynamic>> getOrtValueData(String valueId) {
    return Future.value({
      'data': [1.0, 2.0, 3.0, 4.0],
      'shape': [2, 2],
    });
  }

  @override
  Future<void> releaseOrtValue(String valueId) => Future.value();

  @override
  Future<Map<String, dynamic>> writeOrtValueDataToBinaryFile(String valueId, String filePath) {
    return Future.value({
      'dataType': 'float32',
      'shape': [2, 2]
    });
  }

  @override
  Future<List<String>> getAvailableProviders() => Future.value(['CPU']);
}

class ConversionTrackingMock extends MockFlutterOnnxruntimePlatform {
  String? lastConvertedValueId;
  String? lastConvertedTargetType;

  @override
  Future<Map<String, dynamic>> convertOrtValue(String valueId, String targetType) {
    // Track the conversion operation for assertion
    lastConvertedValueId = valueId;
    lastConvertedTargetType = targetType;

    return Future.value({
      'valueId': 'converted_$valueId',
      'dataType': targetType,
      'shape': [2, 2],
    });
  }
}

class ProviderOptionsMock extends MockFlutterOnnxruntimePlatform {
  @override
  Future<List<String>> getAvailableProviders() => Future.value(['CUDA', 'CPU', 'CORE_ML']);

  @override
  Future<Map<String, dynamic>> createSession(String modelPath, {Map<String, dynamic>? sessionOptions}) {
    // Verify options are passed correctly
    final providers = sessionOptions?['providers'] as List<dynamic>?;
    if (providers != null && providers.contains('CUDA') && providers.contains('CPU')) {
      return Future.value({
        'sessionId': 'provider_test_session',
        'inputNames': ['input1'],
        'outputNames': ['output1'],
      });
    }

    // Default response
    return Future.value({
      'sessionId': 'test_session_id',
      'inputNames': ['input1', 'input2'],
      'outputNames': ['output1', 'output2'],
    });
  }
}

class ArrayShapeMock extends MockFlutterOnnxruntimePlatform {
  final Map<String, Map<String, dynamic>> dataStorage = {};

  void setTensorData(String valueId, List<dynamic> data, List<int> shape) {
    dataStorage[valueId] = {'data': data, 'shape': shape};
  }

  @override
  Future<Map<String, dynamic>> getOrtValueData(String valueId) {
    if (dataStorage.containsKey(valueId)) {
      return Future.value(dataStorage[valueId]!);
    }
    return Future.value({'data': [], 'shape': []});
  }
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  late OrtSession session;
  late MockFlutterOnnxruntimePlatform mockPlatform;
  final FlutterOnnxruntimePlatform initialPlatform = FlutterOnnxruntimePlatform.instance;

  setUp(() async {
    mockPlatform = MockFlutterOnnxruntimePlatform();
    FlutterOnnxruntimePlatform.instance = mockPlatform;

    // Create a session for testing
    final onnxRuntime = OnnxRuntime();
    session = await onnxRuntime.createSession('test_model.onnx');
  });

  tearDown(() {
    FlutterOnnxruntimePlatform.instance = initialPlatform;
  });

  group('OrtSession with OrtValue integration', () {
    test('run() should accept only OrtValue objects as inputs', () async {
      // Create OrtValue tensors
      final tensor1 = await OrtValue.fromList(Float32List.fromList([1.0, 2.0, 3.0, 4.0]), [2, 2]);
      final tensor2 = await OrtValue.fromList(Float32List.fromList([5.0, 6.0, 7.0, 8.0]), [2, 2]);

      // Use OrtValue objects in inputs map
      final inputs = {'input1': tensor1, 'input2': tensor2};

      // Run inference with OrtValues
      final outputs = await session.run(inputs);

      // Verify that the inputs were correctly processed
      expect(mockPlatform.lastRunInputs, isNotNull);

      // Check that the valueId and shape were passed correctly for input1
      expect(mockPlatform.lastRunInputs!['input1'], isA<Map<String, dynamic>>());
      expect(mockPlatform.lastRunInputs!['input1']['valueId'], tensor1.id);
      expect(mockPlatform.lastRunInputs!['input1_shape'], tensor1.shape);

      // Check that the valueId and shape were passed correctly for input2
      expect(mockPlatform.lastRunInputs!['input2'], isA<Map<String, dynamic>>());
      expect(mockPlatform.lastRunInputs!['input2']['valueId'], tensor2.id);
      expect(mockPlatform.lastRunInputs!['input2_shape'], tensor2.shape);

      // Verify output format
      expect(outputs, isA<Map<String, OrtValue>>());
      expect(outputs.keys, contains('output1'));
      expect(outputs['output1']!.id, 'tensor_001');
      expect(outputs['output1']!.shape, [2, 2]);
    });

    test('run() should accept string tensors as inputs', () async {
      // Create OrtValue tensors including a string tensor
      final tensor1 = await OrtValue.fromList(['word1', 'word2', 'word3', 'word4'], [2, 2]);
      final tensor2 = await OrtValue.fromList(Float32List.fromList([5.0, 6.0, 7.0, 8.0]), [2, 2]);

      // Use OrtValue objects in inputs map
      final inputs = {'input1': tensor1, 'input2': tensor2};

      // Run inference with OrtValues
      final outputs = await session.run(inputs);

      // Verify that the inputs were correctly processed
      expect(mockPlatform.lastRunInputs, isNotNull);

      // Check that the valueId and shape were passed correctly for string tensor
      expect(mockPlatform.lastRunInputs!['input1'], isA<Map<String, dynamic>>());
      expect(mockPlatform.lastRunInputs!['input1']['valueId'], tensor1.id);
      expect(mockPlatform.lastRunInputs!['input1_shape'], tensor1.shape);

      // Verify output format
      expect(outputs, isA<Map<String, OrtValue>>());
    });

    test('should properly clean up OrtValue resources', () async {
      // This test verifies that dispose() can be called after using in session.run()

      // Create an OrtValue tensor
      final tensor = await OrtValue.fromList(Float32List.fromList([1.0, 2.0, 3.0, 4.0]), [2, 2]);

      // Use OrtValue in session.run()
      await session.run({'input1': tensor});

      // Should be able to dispose without errors
      await expectLater(tensor.dispose(), completes);
    });

    test('run output should be OrtValue that can be used in subsequent operations', () async {
      // Use custom mock that tracks conversion operations
      final conversionMock = ConversionTrackingMock();
      FlutterOnnxruntimePlatform.instance = conversionMock;

      // Setup the createSession and runInference methods
      final onnxRuntime = OnnxRuntime();
      final testSession = await onnxRuntime.createSession('test_model.onnx');

      // Create input tensor
      final inputTensor = await OrtValue.fromList(Float32List.fromList([1.0, 2.0, 3.0, 4.0]), [2, 2]);

      // Run inference
      final outputs = await testSession.run({'input1': inputTensor});

      // Get the output tensor
      final outputTensor = outputs['output1']!;

      // Use the output tensor in another operation (e.g., convert to int32)
      await outputTensor.to(OrtDataType.int32);

      // Verify that the conversion was called with the right parameters
      expect(conversionMock.lastConvertedValueId, 'tensor_001');
      expect(conversionMock.lastConvertedTargetType, 'int32');

      // Restore original mock
      FlutterOnnxruntimePlatform.instance = mockPlatform;
    });

    test('OrtValue.asList() returns multi-dimensional data while asFlattenedList() returns flat data', () async {
      // Set up a specialized mock platform interface
      final arrayShapeMock = ArrayShapeMock();
      FlutterOnnxruntimePlatform.instance = arrayShapeMock;

      // Create a mock tensor
      final tensor = OrtValue.fromMap({
        'valueId': 'test_tensor_id',
        'dataType': 'float32',
        'shape': [2, 3],
      });

      // Set up the mock data
      arrayShapeMock.setTensorData('test_tensor_id', [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);

      // Get data using both methods
      final shapedData = await tensor.asList();
      final flatData = await tensor.asFlattenedList();

      // Verify shaped data has the right structure (2x3 matrix)
      expect(shapedData, isA<List>());
      expect(shapedData.length, 2);
      expect(shapedData[0], isA<List>());
      expect(shapedData[0].length, 3);
      expect(shapedData[0][0], 1.0);
      expect(shapedData[0][1], 2.0);
      expect(shapedData[0][2], 3.0);
      expect(shapedData[1][0], 4.0);
      expect(shapedData[1][1], 5.0);
      expect(shapedData[1][2], 6.0);

      // Verify flat data is a simple 1D list
      expect(flatData, isA<List>());
      expect(flatData.length, 6);
      expect(flatData, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

      // Clean up
      FlutterOnnxruntimePlatform.instance = initialPlatform;
    });

    test('OrtValue.asList() correctly reshapes string tensor data', () async {
      // Set up a specialized mock platform interface
      final arrayShapeMock = ArrayShapeMock();
      FlutterOnnxruntimePlatform.instance = arrayShapeMock;

      // Create a mock string tensor
      final tensor = OrtValue.fromMap({
        'valueId': 'test_string_tensor_id',
        'dataType': 'string',
        'shape': [2, 2],
      });

      // Set up the mock string data
      arrayShapeMock.setTensorData('test_string_tensor_id', ['hello', 'world', 'string', 'tensor'], [2, 2]);

      // Get data using both methods
      final shapedData = await tensor.asList();
      final flatData = await tensor.asFlattenedList();

      // Verify shaped data has the right structure (2x2 matrix)
      expect(shapedData, isA<List>());
      expect(shapedData.length, 2);
      expect(shapedData[0], isA<List>());
      expect(shapedData[0].length, 2);
      expect(shapedData[0][0], 'hello');
      expect(shapedData[0][1], 'world');
      expect(shapedData[1][0], 'string');
      expect(shapedData[1][1], 'tensor');

      // Verify flat data is a simple 1D list
      expect(flatData, isA<List>());
      expect(flatData.length, 4);
      expect(flatData, ['hello', 'world', 'string', 'tensor']);

      // Clean up
      FlutterOnnxruntimePlatform.instance = initialPlatform;
    });
  });

  group('Provider and Session Options Integration', () {
    test('uses available providers in session options', () async {
      // Create a specialized mock for this test
      final providersMock = ProviderOptionsMock();
      FlutterOnnxruntimePlatform.instance = providersMock;

      // Create runtime and check available providers
      final onnxRuntime = OnnxRuntime();
      final availableProviders = await onnxRuntime.getAvailableProviders();

      // Verify we get our expected providers
      expect(availableProviders, containsAll([OrtProvider.CUDA, OrtProvider.CPU, OrtProvider.CORE_ML]));

      // Create session with preferred providers
      final options = OrtSessionOptions(
        providers: [OrtProvider.CUDA, OrtProvider.CPU], // Prioritize CUDA, fallback to CPU
        deviceId: 0,
      );

      final session = await onnxRuntime.createSession('test_model.onnx', options: options);

      // Verify we get the special session id that indicates providers were correctly used
      expect(session, isNotNull);
      expect(session.id, 'provider_test_session');

      // Restore the original mock
      FlutterOnnxruntimePlatform.instance = mockPlatform;
    });
  });
}
