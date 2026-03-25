declare module 'libflutter_onnxruntime_ohos.so' {
  export function ortGetPlatformVersion(): string;
  export function ortGetAvailableProvidersJson(): string;
  export function ortCreateSession(modelPath: string, sessionOptionsJson: string): string;
  export function ortCreateOrtValue(sourceType: string, dataJson: string, shapeJson: string): string;
  export function ortCreateOrtValueFromBinaryFile(sourceType: string, filePath: string, shapeJson: string): string;
  export function ortGetOrtValueData(valueId: string): string;
  export function ortWriteOrtValueDataToBinaryFile(valueId: string, filePath: string): string;
  export function ortRunInference(sessionId: string, inputsJson: string, runOptionsJson: string): string;
  export function ortCloseSession(sessionId: string): number;
  export function ortReleaseOrtValue(valueId: string): number;
}
