// Walk-test RSSI logger for the student Flutter app (Campaign 1).
//
// Add to pubspec.yaml:  flutter_blue_plus: ^1.32.0, path_provider: ^2.1.0
// Android: BLUETOOTH_SCAN + ACCESS_FINE_LOCATION permissions; iOS: NSBluetoothAlwaysUsageDescription.
//
// The operator types the point metadata (point_id, distance, inside/outside,
// partition, LOS to screen, posture) and presses "Log 30 s". Every advertisement
// carrying the station's manufacturer ID is appended to walktest.csv in the
// app documents folder. Export the file afterwards (share sheet / adb pull).

import 'dart:async';
import 'dart:io';

import 'package:flutter_blue_plus/flutter_blue_plus.dart';
import 'package:path_provider/path_provider.dart';

const int kCompanyId = 0xFFFF; // replace with the assigned company ID in production

class WalkPoint {
  final String sessionId, roomId, deviceId, posture, pointId, partitionType;
  final bool insideRoom, losToScreen;
  final double distanceM;
  WalkPoint({
    required this.sessionId, required this.roomId, required this.deviceId,
    required this.posture, required this.pointId, required this.partitionType,
    required this.insideRoom, required this.losToScreen, required this.distanceM,
  });
}

Future<int> logPoint(WalkPoint p, {Duration window = const Duration(seconds: 30)}) async {
  final dir = await getApplicationDocumentsDirectory();
  final file = File('${dir.path}/walktest.csv');
  if (!await file.exists()) {
    await file.writeAsString(
        'session_id,room_id,device_id,posture,point_id,inside_room,partition_type,'
        'distance_m,los_to_screen,t_unix_ms,rssi_dbm\n');
  }
  final sink = file.openWrite(mode: FileMode.append);
  var count = 0;
  final sub = FlutterBluePlus.onScanResults.listen((results) {
    for (final r in results) {
      if (!r.advertisementData.manufacturerData.containsKey(kCompanyId)) continue;
      sink.writeln([
        p.sessionId, p.roomId, p.deviceId, p.posture, p.pointId,
        p.insideRoom ? 1 : 0, p.partitionType, p.distanceM,
        p.losToScreen ? 1 : 0, DateTime.now().millisecondsSinceEpoch, r.rssi,
      ].join(','));
      count++;
    }
  });
  // continuousUpdates: report every advertisement, not just the first per device
  await FlutterBluePlus.startScan(
      timeout: window, continuousUpdates: true, androidScanMode: AndroidScanMode.lowLatency);
  await Future.delayed(window);
  await sub.cancel();
  await sink.flush();
  await sink.close();
  return count; // show to operator; < 50 samples in 30 s suggests a problem
}
