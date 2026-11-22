// Mode-I chirp for latency campaigns (Campaign 2), Android.
//
// Add to pubspec.yaml: flutter_ble_peripheral: ^1.2.0, crypto: ^3.0.3, http: ^1.2.0
//
// iOS NOTE: CoreBluetooth apps cannot advertise manufacturer-specific data
// (foreground or background). On iOS put the 4-byte tag in a 128-bit service
// UUID instead and adapt station_logger.py to read service UUIDs.
//
// Flow (genuine):   GET /nonce -> HMAC -> advertise for 1 s.
// Flow (A10 relay): absent phone GETs /nonce, computes the tag with ITS key,
//                   sends the tag (hex) over the relay WebSocket; the accomplice
//                   phone calls advertiseTag() on receipt.

import 'dart:convert';
import 'dart:typed_data';

import 'package:crypto/crypto.dart';
import 'package:flutter_ble_peripheral/flutter_ble_peripheral.dart';
import 'package:http/http.dart' as http;

const int kCompanyId = 0xFFFF;

Uint8List chirpTag(List<int> deviceTestKey, String nonceHex) {
  final mac = Hmac(sha256, deviceTestKey).convert(utf8.encode(nonceHex)).bytes;
  return Uint8List.fromList(mac.sublist(0, 4)); // Truncate8 (hex) = 4 bytes
}

Future<void> advertiseTag(Uint8List tag, {Duration hold = const Duration(seconds: 1)}) async {
  final adv = AdvertiseData(manufacturerId: kCompanyId, manufacturerData: tag, includeDeviceName: false);
  final settings = AdvertiseSettings(
      advertiseMode: AdvertiseMode.advertiseModeLowLatency,
      txPowerLevel: AdvertiseTxPower.advertiseTxPowerMedium,
      connectable: false);
  await FlutterBlePeripheral().start(advertiseData: adv, advertiseSettings: settings);
  await Future.delayed(hold);
  await FlutterBlePeripheral().stop();
}

Future<void> genuineTrial(String server, String deviceId, String network, List<int> key) async {
  final r = await http.get(Uri.parse('$server/nonce?kind=genuine&device_id=$deviceId&network=$network'));
  final nonce = jsonDecode(r.body)['nonce'] as String;
  await advertiseTag(chirpTag(key, nonce));
}
