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
import 'package:flutter/material.dart';

import '../../../core/database/app_database.dart';
import '../../../core/theme/motion.dart';
import '../../../core/theme/app_tokens.dart';
import '../../discovery/data/geo.dart';
import '../../discovery/data/polyline.dart';
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

class TripMapScreen extends StatelessWidget {
  final Future<TripMapView> Function() load;
  final MapTileProvider provider;
  final TileStore store;

  /// Opens the download screen, for when there is nothing to show yet.
  final VoidCallback? onDownload;

  const TripMapScreen({
    super.key,
    required this.load,
    required this.provider,
    required this.store,
    this.onDownload,
  });

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
        future: load(),
        builder: (context, snap) {
          final view = snap.data;
          if (view == null) return const SizedBox();

          return Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Expanded(
                child: TripMap(
                  provider: provider,
                  store: store,
                  route: view.route,
                  stops: view.stops,
                  hasTiles: view.hasTiles,
                  height: double.infinity,
                ),
              ),
              _Footer(view: view, onDownload: onDownload),
            ],
          );
        },
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
