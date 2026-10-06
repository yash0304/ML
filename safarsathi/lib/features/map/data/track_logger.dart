// lib/features/map/data/track_logger.dart
//
// The route log — #30b. Where the phone went, from its own GPS, which needs
// no signal: Polarsteps' one idea that survives the offline constraint whole.
//
// NOT BACKGROUND LOCATION. DECISIONS (23 Sep): location only while the app is
// in use. The log runs as a foreground service — started by the person,
// with a notification saying "SafarSathi is logging your route" for as long
// as it runs — which Android counts as in use. So there is still no
// ACCESS_BACKGROUND_LOCATION. It logs with the screen off, stops when
// switched off, and stops when the app is closed from Recents.

import 'dart:async';

import 'package:geolocator/geolocator.dart';

import '../../../core/database/app_database.dart';
import '../../discovery/data/geo.dart';
import '../../trips/data/timeline.dart';
import 'here.dart';

/// Fixes as the phone moves, with the service notification while listened.
abstract class TrackSource {
  Stream<HereFix> track();
}

class DeviceTrackSource implements TrackSource {
  const DeviceTrackSource();

  @override
  Stream<HereFix> track() => Geolocator.getPositionStream(
    locationSettings: AndroidSettings(
      accuracy: LocationAccuracy.high,
      // The thinning in keepFix decides what is stored; this only spares
      // the phone from waking for every few metres.
      distanceFilter: 50,
      intervalDuration: const Duration(seconds: 30),
      foregroundNotificationConfig: const ForegroundNotificationConfig(
        notificationTitle: 'SafarSathi is logging your route',
        notificationText:
            'From the GPS, which needs no signal. Turn it off on the '
            'Timeline.',
        notificationChannelName: 'Route log',
        setOngoing: true,
      ),
    ),
  ).map(
    (p) => HereFix(
      at: LatLng(p.latitude, p.longitude),
      accuracyM: p.accuracy,
      time: p.timestamp,
    ),
  );
}

class TrackLogger {
  final AppDatabase db;
  final LocationSource location;
  final TrackSource source;

  TrackLogger({
    required this.db,
    required this.location,
    this.source = const DeviceTrackSource(),
  });

  StreamSubscription<HereFix>? _sub;
  LatLng? _lastAt;
  DateTime? _lastTime;
  int? _tripId;

  bool get running => _sub != null;

  /// Starts logging to [tripId]. Returns a sentence when it cannot — and
  /// asks for location only when [ask] (a tap), never on its own.
  Future<String?> start(int tripId, {bool ask = true}) async {
    if (_sub != null && _tripId == tripId) return null;
    await stop();
    final state = await location.check(ask: ask);
    if (state == HereState.serviceOff) {
      return 'Location is switched off on the phone.';
    }
    if (state == HereState.blocked) {
      return 'Location is blocked for SafarSathi in the phone\'s Settings.';
    }
    if (state != HereState.locating && state != HereState.found) {
      return 'Location was not allowed, so the route cannot be logged.';
    }
    _tripId = tripId;
    _sub = source.track().listen(
      (fix) => _take(tripId, fix),
      onError: (_) {},
    );
    return null;
  }

  Future<void> _take(int tripId, HereFix fix) async {
    if (!keepFix(
      at: fix.at,
      accuracyM: fix.accuracyM,
      time: fix.time,
      lastAt: _lastAt,
      lastTime: _lastTime,
    )) {
      return;
    }
    _lastAt = fix.at;
    _lastTime = fix.time;
    await addFix(
      db,
      tripId: tripId,
      at: fix.at,
      accuracyM: fix.accuracyM,
      time: fix.time,
    );
  }

  Future<void> stop() async {
    await _sub?.cancel();
    _sub = null;
    _tripId = null;
  }
}
