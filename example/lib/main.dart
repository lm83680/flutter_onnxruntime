// Copyright (c) MASIC AI
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import 'package:flutter/material.dart';
import 'dart:async';

import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});

  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  final _flutterOnnxruntimePlugin = OnnxRuntime();
  final _aController = TextEditingController(); // Controller for the first input
  final _bController = TextEditingController(); // Controller for the second input
  String _result = ''; // Variable to store the result
  OrtSession? _session;

  @override
  void initState() {
    super.initState();
    _initializeSession();
  }

  Future<void> _initializeSession() async {
    try {
      _session = await _flutterOnnxruntimePlugin.createSessionFromAsset('assets/models/addition_model.ort');
    } catch (error) {
      if (!mounted) {
        return;
      }
      setState(() {
        _result = 'Init Error: $error';
      });
      rethrow;
    }
  }

  @override
  void dispose() {
    _session?.close();
    _aController.dispose();
    _bController.dispose();
    super.dispose();
  }

  // Run a simple Addition Model
  Future<void> _runAddition(double a, double b) async {
    final createdInputs = <OrtValue>[];
    final createdOutputs = <OrtValue>[];

    try {
      if (_session == null) {
        await _initializeSession();
      }

      final inputs = {
        'A': await OrtValue.fromList([a], [1]),
        'B': await OrtValue.fromList([b], [1]),
      };
      createdInputs.addAll(inputs.values);

      final outputs = await _session!.run(inputs);
      createdOutputs.addAll(outputs.values);

      final outputData = await outputs['C']!.asList();
      if (!mounted) {
        return;
      }
      setState(() {
        _result = outputData[0].toString();
      });
    } catch (error) {
      if (!mounted) {
        return;
      }
      setState(() {
        _result = 'Error: $error';
      });
    } finally {
      for (final tensor in createdInputs) {
        await tensor.dispose();
      }
      for (final tensor in createdOutputs) {
        await tensor.dispose();
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: const Text('Addition Model with Onnxruntime Demo')),
        body: Center(
          child: Padding(
            padding: const EdgeInsets.all(32.0),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: <Widget>[
                TextField(
                  controller: _aController,
                  keyboardType: TextInputType.number,
                  decoration: InputDecoration(labelText: 'Enter number A'),
                ),
                TextField(
                  controller: _bController,
                  keyboardType: TextInputType.number,
                  decoration: InputDecoration(labelText: 'Enter number B'),
                ),
                ElevatedButton(
                  onPressed: () async {
                    final a = double.tryParse(_aController.text) ?? 0.0;
                    final b = double.tryParse(_bController.text) ?? 0.0;
                    await _runAddition(a, b); // Run the addition when the button is pressed
                  },
                  child: Text('Add'),
                ),
                SizedBox(height: 20),
                Text('Result: $_result'), // Display the result
              ],
            ),
          ),
        ),
      ),
    );
  }
}
