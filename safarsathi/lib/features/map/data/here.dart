// lib/features/map/data/here.dart
//
// Where the phone is — from its own GPS, which needs no signal.
//
// GPS is a receiver: it listens to satellites and sends nothing, so it works
// in a valley with no bars at all. That is what makes "you are here" on an
// offline map possible, and why it is worth the one new permission.
//
// THE POSITION NEVER LEAVES THE PHONE from here. It is drawn on the map and,
// only when the person presses the SOS button and then Send in their own SMS
// app, written into a text. Nothing in this file has a network path.
//
// Asked for only while the app is open ("while using the app"), never in the
// background.

import 'package:geolocator/geolocator.dart';

import '../../discovery/data/geo.dart';

/// Where locating stands, each state with a different thing to say or do.
enum HereState {
  /// Never asked. The map shows its button and says nothing yet.
  notAsked,

  /// The phone's location switch is off. Fixable in Settings, not here.
  serviceOff,

  /// Refused this time. Asking again is allowed.
  denied,

  /// Refused with "don't ask again", or blocked in Settings. Only the app's
  /// settings page can change it, so the screen offers that instead.
  blocked,

  /// Allowed; waiting for a fix. A cold GPS in a valley can take a minute.
  locating,

  /// Allowed and fixed.
  found,
}

class HereFix {
  final LatLng at;

  /// Radius of the 68% circle, metres. Shown, never hidden: ±200 m is a
  /// different claim from ±5 m, and the map draws the difference.
  final double accuracyM;
  final DateTime time;

  const HereFix({required this.at, required this.accuracyM, required this.time});
}

/// The state a permission answer and a service switch add up to.
///
/// Pure, so every combination is tested without a phone.
HereState stateFor({
  required bool serviceOn,
  required LocationPermission permission,
}) {
  if (!serviceOn) return HereState.serviceOff;
  return switch (permission) {
    LocationPermission.denied => HereState.denied,
    LocationPermission.deniedForever => HereState.blocked,
    LocationPermission.unableToDetermine => HereState.notAsked,
    LocationPermission.whileInUse ||
    LocationPermission.always => HereState.locating,
  };
}

/// Everything the screens need from location. Abstract so tests can script
/// a phone's answers.
abstract class LocationSource {
  /// Where permission stands. With [ask], asks if it has not been refused
  /// for good — only ever in response to a tap.
  Future<HereState> check({bool ask = false});

  /// Fixes as the phone moves, while listened to.
  Stream<HereFix> watch();

  /// A fix within [timeout], else the last known one, else null.
  Future<HereFix?> once({Duration timeout});

  Future<void> openLocationSettings();
  Future<void> openAppSettings();
}

class DeviceLocation implements LocationSource {
  const DeviceLocation();

  static HereFix _fix(Position p) => HereFix(
    at: LatLng(p.latitude, p.longitude),
    accuracyM: p.accuracy,
    time: p.timestamp,
  );

  @override
  Future<HereState> check({bool ask = false}) async {
    final serviceOn = await Geolocator.isLocationServiceEnabled();
    var permission = await Geolocator.checkPermission();
    if (ask && serviceOn && permission == LocationPermission.denied) {
      permission = await Geolocator.requestPermission();
    }
    final state = stateFor(serviceOn: serviceOn, permission: permission);
    // Never asked reads as `denied` on Android. Until the person taps, say
    // nothing rather than "refused".
    if (!ask && state == HereState.denied) return HereState.notAsked;
    return state;
  }

  @override
  Stream<HereFix> watch() => Geolocator.getPositionStream(
    locationSettings: const LocationSettings(
      accuracy: LocationAccuracy.high,
      // Five metres: enough to follow a walk through a village without
      // redrawing for every jitter of a stationary phone.
      distanceFilter: 5,
    ),
  ).map(_fix);

  @override
  Future<HereFix?> once({Duration timeout = const Duration(seconds: 20)}) async {
    try {
      final p = await Geolocator.getCurrentPosition(
        locationSettings: LocationSettings(
          accuracy: LocationAccuracy.high,
          timeLimit: timeout,
        ),
      );
      return _fix(p);
    } on Object {
      // No fix in time — a valley, a building. The last one known is still
      // worth sending, with its age said out loud.
      final last = await Geolocator.getLastKnownPosition();
      return last == null ? null : _fix(last);
    }
  }

  @override
  Future<void> openLocationSettings() => Geolocator.openLocationSettings();

  @override
  Future<void> openAppSettings() => Geolocator.openAppSettings();
}

/// "±12 m, 20 s ago" — the two things that decide how far to trust a dot.
String describeFix(HereFix fix, {DateTime? now}) {
  final age = (now ?? DateTime.now()).difference(fix.time);
  final when = age.inSeconds < 60
      ? '${age.inSeconds.clamp(0, 59)} s ago'
      : age.inMinutes < 60
      ? '${age.inMinutes} min ago'
      : '${age.inHours} h ago';
  return '±${fix.accuracyM.round()} m, $when';
}
