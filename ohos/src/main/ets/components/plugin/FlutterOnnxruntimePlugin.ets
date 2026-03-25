import {
  FlutterPlugin,
  FlutterPluginBinding,
  MethodCall,
  MethodCallHandler,
  MethodChannel,
  MethodResult,
} from '@ohos/flutter_ohos';
import {
  ortCloseSession,
  ortCreateOrtValue,
  ortCreateSession,
  ortGetAvailableProvidersJson,
  ortGetOrtValueData,
  ortGetPlatformVersion,
  ortReleaseOrtValue,
  ortRunInference,
} from 'libflutter_onnxruntime_ohos.so';

class NativeErrorDetail {
  code: string = 'NATIVE_ERROR';
  message: string = '原生调用失败';
}

class NativeErrorPayload {
  error: NativeErrorDetail | null = null;
}

class OrtValueResultPayload {
  valueId: string = '';
  dataType: string = '';
  shape: Array<number> = [];
}

class OrtValueDataPayload {
  dataType: string = '';
  shape: Array<number> = [];
  data: Array<number> = [];
}

export default class FlutterOnnxruntimePlugin implements FlutterPlugin, MethodCallHandler {
  private channel: MethodChannel | null = null;

  getUniqueClassName(): string {
    return "FlutterOnnxruntimePlugin"
  }

  onAttachedToEngine(binding: FlutterPluginBinding): void {
    this.channel = new MethodChannel(binding.getBinaryMessenger(), "flutter_onnxruntime");
    this.channel.setMethodCallHandler(this)
  }

  onDetachedFromEngine(binding: FlutterPluginBinding): void {
    if (this.channel != null) {
      this.channel.setMethodCallHandler(null)
    }
  }

  private parseJson(jsonText: string): Object {
    try {
      return JSON.parse(jsonText) as Object;
    } catch (error) {
      const detail = new NativeErrorDetail();
      detail.code = 'JSON_PARSE_ERROR';
      detail.message = `${error}`;
      const payload = new NativeErrorPayload();
      payload.error = detail;
      return payload;
    }
  }

  private getError(payload: Object): Map<string, string> | null {
    const rawPayload = payload as NativeErrorPayload;
    if (rawPayload.error === null || rawPayload.error === undefined) {
      return null;
    }
    const errorMap = new Map<string, string>();
    errorMap.set('code', rawPayload.error.code ?? 'NATIVE_ERROR');
    errorMap.set('message', rawPayload.error.message ?? '原生调用失败');
    return errorMap;
  }

  private normalizeValue(value: Object | null): Object | string | number | boolean | Array<Object | string | number | boolean> | null {
    if (value === null || value === undefined) {
      return null;
    }

    if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
      return value;
    }

    if (value instanceof Float32Array || value instanceof Float64Array || value instanceof Int32Array) {
      const normalizedArray: Array<number> = [];
      for (let index = 0; index < value.length; index += 1) {
        normalizedArray.push(Number(value[index]));
      }
      return normalizedArray;
    }

    if (value instanceof Array) {
      const normalizedArray: Array<Object | string | number | boolean> = [];
      for (let index = 0; index < value.length; index += 1) {
        normalizedArray.push(this.normalizeValue(value[index] as Object) as Object | string | number | boolean);
      }
      return normalizedArray;
    }

    if (value instanceof Map) {
      const normalizedObject: Record<string, Object | string | number | boolean | Array<Object | string | number | boolean> | null> = {};
      value.forEach((mapValue: Object, mapKey: Object) => {
        normalizedObject[`${mapKey}`] = this.normalizeValue(mapValue);
      });
      return normalizedObject as Object;
    }

    return value;
  }

  private toJsonText(value: Object | null, fallback: string): string {
    if (value === null || value === undefined) {
      return fallback;
    }
    return JSON.stringify(this.normalizeValue(value));
  }

  onMethodCall(call: MethodCall, result: MethodResult): void {
    if (call.method === "getPlatformVersion") {
      result.success(ortGetPlatformVersion());
      return;
    }

    if (call.method === "getAvailableProviders") {
      const payload = this.parseJson(ortGetAvailableProvidersJson());
      const error = this.getError(payload);
      if (error !== null) {
        result.error(error.get('code')!, error.get('message')!, null);
        return;
      }
      result.success(payload);
      return;
    }

    if (call.method === "createSession") {
      const modelPath: string = `${call.argument('modelPath') ?? ''}`;
      const sessionOptionsValue: Object | null = call.hasArgument('sessionOptions') ? call.argument('sessionOptions') as Object : null;
      const sessionOptionsJson = this.toJsonText(sessionOptionsValue, '{}');
      const payload = this.parseJson(ortCreateSession(modelPath, sessionOptionsJson));
      const error = this.getError(payload);
      if (error !== null) {
        result.error(error.get('code')!, error.get('message')!, null);
        return;
      }
      result.success(payload);
      return;
    }

    if (call.method === "closeSession") {
      const sessionId: string = `${call.argument('sessionId') ?? ''}`;
      const closed = ortCloseSession(sessionId) === 1;
      if (!closed) {
        result.error("INVALID_SESSION", "未找到需要关闭的 session", null);
        return;
      }
      result.success(null);
      return;
    }

    if (call.method === "createOrtValue") {
      const sourceType: string = `${call.argument('sourceType') ?? ''}`;
      const dataValue: Object | null = call.hasArgument('data') ? call.argument('data') as Object : null;
      const shapeValue: Object | null = call.hasArgument('shape') ? call.argument('shape') as Object : null;
      const payload = this.parseJson(
        ortCreateOrtValue(
          sourceType,
          this.toJsonText(dataValue, '[]'),
          this.toJsonText(shapeValue, '[]'),
        ),
      );
      const error = this.getError(payload);
      if (error !== null) {
        result.error(error.get('code')!, error.get('message')!, null);
        return;
      }
      result.success(payload as OrtValueResultPayload);
      return;
    }

    if (call.method === "getOrtValueData") {
      const valueId: string = `${call.argument('valueId') ?? ''}`;
      const payload = this.parseJson(ortGetOrtValueData(valueId));
      const error = this.getError(payload);
      if (error !== null) {
        result.error(error.get('code')!, error.get('message')!, null);
        return;
      }
      result.success(payload as OrtValueDataPayload);
      return;
    }

    if (call.method === "releaseOrtValue") {
      const valueId: string = `${call.argument('valueId') ?? ''}`;
      const released = ortReleaseOrtValue(valueId) === 1;
      if (!released) {
        result.error("INVALID_ORT_VALUE", "未找到需要释放的 OrtValue", null);
        return;
      }
      result.success(null);
      return;
    }

    if (call.method === "runInference") {
      const sessionId: string = `${call.argument('sessionId') ?? ''}`;
      const inputsValue: Object | null = call.hasArgument('inputs') ? call.argument('inputs') as Object : null;
      const runOptionsValue: Object | null = call.hasArgument('runOptions') ? call.argument('runOptions') as Object : null;
      const payload = this.parseJson(
        ortRunInference(
          sessionId,
          this.toJsonText(inputsValue, '{}'),
          this.toJsonText(runOptionsValue, '{}'),
        ),
      );
      const error = this.getError(payload);
      if (error !== null) {
        result.error(error.get('code')!, error.get('message')!, null);
        return;
      }
      result.success(payload);
      return;
    }

    result.notImplemented()
  }
}
