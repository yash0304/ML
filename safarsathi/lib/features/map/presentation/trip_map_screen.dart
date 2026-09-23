// lib/features/map/presentation/trip_map_screen.dart
//
// Looking at the map you downloaded.
//
// This screen exists because `TripMap` did not have one. The widget was built
// at #24, tested, and wired to nothing — so the app could download 346 MB of
// tiles, report them accurately on the cache screen, and offer no way at all
// to see them. A download nobody can look at is not an offline map; it is a
// directory.

import 'package:drift/drift.dart' show OrderingTerm;
import 'package:drift/drift.dart' as drift;
import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart' as fm;
import 'package:latlong2/latlong.dart' as ll;

import '../../../core/database/app_database.dart';
import '../../../core/theme/motion.dart';
import '../../../core/theme/app_tokens.dart';
import '../../discovery/data/geo.dart';
import '../../discovery/data/polyline.dart';
import '../data/here.dart';
import '../data/tile_provider.dart';
import '../data/tile_store.dart';
import 'trip_map.dart';

/// Everything the map needs, read once.
class TripMapView {
  final List<LatLng> route;
  final List<({String name, LatLng at})> stops;
  final int tileCount;

  const TripMapView({
    required this.route,
    required this.stops,
    required this.tileCount,
  });

  bool get hasTiles => tileCount > 0;
  bool get isEmpty => route.isEmpty && stops.isEmpty;
}

/// Reads the trip's stops and its decoded route lines.
Future<TripMapView> readTripMap(
  AppDatabase db,
  int tripId, {
  required String providerId,
}) async {
  final stops =
      await (db.select(db.stops)
            ..where((s) => s.tripId.equals(tripId))
            ..orderBy([(s) => OrderingTerm(expression: s.sequenceOrder)]))
          .get();

  final legs =
      await (db.select(db.legs)
            ..where((l) => l.tripId.equals(tripId))
            ..orderBy([(l) => OrderingTerm(expression: l.sequenceOrder)]))
          .get();

  // Every leg's line, end to end, as one run of points. Drawing them as one
  // polyline is right here: they are one journey, and the joins are stops.
  final route = <LatLng>[];
  for (final leg in legs) {
    final encoded = leg.routePolyline;
    if (encoded == null || encoded.isEmpty) continue;
    route.addAll(Polyline.decode(encoded));
  }

  final count = drift.countAll();
  final usage =
      db.selectOnly(db.mapTiles)
        ..addColumns([count])
        ..where(db.mapTiles.provider.equals(providerId));

  return TripMapView(
    route: route,
    stops: [
      for (final s in stops)
        if (s.lat != null && s.lon != null)
          (name: s.name, at: LatLng(s.lat!, s.lon!)),
    ],
    tileCount: (await usage.getSingle()).read(count) ?? 0,
  );
}

class TripMapScreen extends StatefulWidget {
  final Future<TripMapView> Function() load;
  final MapTileProvider provider;
  final TileStore store;

  /// Opens the download screen, for when there is nothing to show yet.
  final VoidCallback? onDownload;

  /// The phone's GPS. Null leaves "you are here" off entirely — the tests
  /// that are about tiles, and any build without a location source.
  final LocationSource? location;

  const TripMapScreen({
    super.key,
    required this.load,
    required this.provider,
    required this.store,
    this.onDownload,
    this.location,
  });

  @override
  State<TripMapScreen> createState() => _TripMapScreenState();
}

class _TripMapScreenState extends State<TripMapScreen> {
  // LOADED ONCE. This used to call load() inside build, which was harmless
  // while nothing rebuilt the screen. A moving dot rebuilds it every few
  // metres, and each rebuild would have re-read the stops, the route and the
  // tile count from the database.
  late final Future<TripMapView> _view = widget.load();

  final _controller = fm.MapController();
  StreamSubscription<HereFix>? _sub;
  HereState _state = HereState.notAsked;
  HereFix? _me;
  bool _centreOnNextFix = false;

  @override
  void initState() {
    super.initState();
    // Already allowed from an earlier visit: start without a prompt. Never
    // asks here — only a tap on the button asks.
    final location = widget.location;
    if (location != null) {
      location.check().then((state) {
        if (!mounted) return;
        setState(() => _state = state);
        if (state == HereState.locating) _start(centre: false);
      });
    }
  }

  @override
  void dispose() {
    _sub?.cancel();
    super.dispose();
  }

  void _start({required bool centre}) {
    _centreOnNextFix = centre;
    _sub ??= widget.location!.watch().listen(
      (fix) {
        if (!mounted) return;
        setState(() {
          _me = fix;
          _state = HereState.found;
        });
        if (_centreOnNextFix) {
          _centreOnNextFix = false;
          _recentre();
        }
      },
      onError: (_) {
        // The stream ends when location is switched off mid-walk. Say so,
        // and let the button start it again.
        if (!mounted) return;
        _sub = null;
        setState(() => _state = HereState.serviceOff);
      },
    );
  }

  void _recentre() {
    final me = _me;
    if (me == null) return;
    try {
      _controller.move(
        ll.LatLng(me.at.lat, me.at.lon),
        _controller.camera.zoom,
      );
    } on Object {
      // The map has not laid out yet; the next fix will centre it.
      _centreOnNextFix = true;
    }
  }

  Future<void> _onButton() async {
    final location = widget.location!;
    switch (_state) {
      case HereState.found:
        _recentre();
      case HereState.serviceOff:
        await location.openLocationSettings();
        final state = await location.check();
        if (mounted) setState(() => _state = state);
        if (state == HereState.locating) _start(centre: true);
      case HereState.blocked:
        await location.openAppSettings();
      case HereState.notAsked || HereState.denied || HereState.locating:
        final state = await location.check(ask: true);
        if (!mounted) return;
        setState(() => _state = _me != null ? HereState.found : state);
        if (state == HereState.locating) _start(centre: true);
    }
  }

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);

    return Scaffold(
      backgroundColor: c.paper,
      appBar: AppBar(
        title: const Text('The map'),
        backgroundColor: c.paper,
        foregroundColor: c.ink,
        elevation: 0,
      ),
      body: FutureBuilder<TripMapView>(
        future: _view,
        builder: (context, snap) {
          final view = snap.data;
          if (view == null) return const SizedBox();

          return Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Expanded(
                child: Stack(
                  children: [
                    TripMap(
                      provider: widget.provider,
                      store: widget.store,
                      route: view.route,
                      stops: view.stops,
                      hasTiles: view.hasTiles,
                      height: double.infinity,
                      me: _me,
                      controller: _controller,
                    ),
                    if (widget.location != null && view.hasTiles)
                      Positioned(
                        right: AppTokens.gutter,
                        // Clear of the attribution strip.
                        bottom: AppTokens.s32,
                        child: _HereButton(state: _state, onTap: _onButton),
                      ),
                  ],
                ),
              ),
              if (widget.location != null && view.hasTiles)
                _HereLine(state: _state, fix: _me),
              _Footer(view: view, onDownload: widget.onDownload),
            ],
          );
        },
      ),
    );
  }
}

/// The target button. Its icon says what a tap will do.
class _HereButton extends StatelessWidget {
  final HereState state;
  final VoidCallback onTap;
  const _HereButton({required this.state, required this.onTap});

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    final icon = switch (state) {
      HereState.found => Icons.my_location,
      HereState.locating => Icons.location_searching,
      HereState.serviceOff || HereState.blocked => Icons.location_disabled,
      HereState.notAsked || HereState.denied => Icons.location_searching,
    };
    return Semantics(
      button: true,
      label: state == HereState.found ? 'Centre on me' : 'Show where I am',
      child: PressScale(
        onTap: onTap,
        child: Container(
          width: 48,
          height: 48,
          decoration: BoxDecoration(
            color: c.paper,
            shape: BoxShape.circle,
            border: Border.all(color: c.ink),
          ),
          child: Icon(icon, color: c.signal),
        ),
      ),
    );
  }
}

/// One line under the map saying where locating stands — each state its own
/// sentence, because "nothing is happening" and "GPS is warming up in a
/// valley" look identical otherwise.
class _HereLine extends StatelessWidget {
  final HereState state;
  final HereFix? fix;
  const _HereLine({required this.state, this.fix});

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    final (text, caution) = switch (state) {
      HereState.notAsked => (
        'Tap the target to show where you are. GPS needs no signal.',
        false,
      ),
      HereState.locating => (
        'Finding you. GPS works with no signal, but the first fix can take '
            'a minute in a valley or indoors.',
        false,
      ),
      HereState.found => ('You are here — ${describeFix(fix!)}.', false),
      HereState.serviceOff => (
        'Location is switched off on this phone. Tap the target to turn it '
            'on.',
        true,
      ),
      HereState.denied => (
        'SafarSathi was not allowed to see where you are. Tap the target to '
            'ask again.',
        true,
      ),
      HereState.blocked => (
        'Location is blocked for SafarSathi. Tap the target to open its '
            'settings.',
        true,
      ),
    };
    return Container(
      padding: const EdgeInsets.fromLTRB(
        AppTokens.gutter,
        AppTokens.s8,
        AppTokens.gutter,
        0,
      ),
      color: c.paper,
      child: Text(
        text,
        style: AppTokens.captionStyle.copyWith(
          color: caution ? c.cautionMark : c.muted,
        ),
      ),
    );
  }
}

class _Footer extends StatelessWidget {
  final TripMapView view;
  final VoidCallback? onDownload;

  const _Footer({required this.view, this.onDownload});

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);

    return Container(
      padding: const EdgeInsets.all(AppTokens.gutter),
      decoration: BoxDecoration(
        color: c.paper,
        border: Border(
          top: BorderSide(color: c.rule, width: AppTokens.hairline),
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            view.hasTiles
                ? '${view.tileCount} tiles on this phone. Nothing here asks '
                      'the network — pan outside what you downloaded and it '
                      'goes blank, which is the honest answer.'
                : 'No tiles downloaded yet, so there is nothing to draw.',
            style: AppTokens.captionStyle.copyWith(
              color: view.hasTiles ? c.muted : c.cautionMark,
            ),
          ),
          if (!view.hasTiles && onDownload != null) ...[
            const SizedBox(height: AppTokens.s12),
            PressScale(
              onTap: onDownload,
              child: Container(
                width: double.infinity,
                padding: const EdgeInsets.symmetric(vertical: AppTokens.s12),
                alignment: Alignment.center,
                decoration: BoxDecoration(
                  color: c.signal,
                  border: Border.all(color: c.ink),
                  borderRadius: BorderRadius.circular(AppTokens.radiusSoft),
                ),
                child: Text(
                  'DOWNLOAD THE MAP',
                  style: AppTokens.stencilStyle.copyWith(
                    fontSize: 11,
                    color: c.paper,
                  ),
                ),
              ),
            ),
          ],
        ],
      ),
    );
  }
}
