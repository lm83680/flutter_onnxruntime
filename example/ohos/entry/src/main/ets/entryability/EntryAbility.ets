import { FlutterAbility, FlutterEngine, Log } from '@ohos/flutter_ohos';
import FlutterOnnxruntimePlugin from 'flutter_onnxruntime';
import { GeneratedPluginRegistrant } from '../plugins/GeneratedPluginRegistrant';

export default class EntryAbility extends FlutterAbility {
  configureFlutterEngine(flutterEngine: FlutterEngine) {
    super.configureFlutterEngine(flutterEngine)
    GeneratedPluginRegistrant.registerWith(flutterEngine)
    try {
      flutterEngine.getPlugins()?.add(new FlutterOnnxruntimePlugin());
    } catch (error) {
      Log.e('EntryAbility', 'Manual registration for flutter_onnxruntime failed', error);
    }
  }
}
