// lib/features/map/presentation/trip_map.dart
//
// The map itself — issue #24.
//
// Renders from the local cache only. Three things are non-negotiable here:
// attribution on every map, an honest word when there is no key, and an
// honest word when there are no tiles. A blank grey rectangle is
// indistinguishable from a bug.

import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart' as fm;
// flutter_map takes latlong2's LatLng, which is a different class from this
// app's own. Aliased so the two never get confused at a call site.
import 'package:latlong2/latlong.dart' as ll;

import '../../../core/theme/app_tokens.dart';
import '../../discovery/data/geo.dart';
import '../data/tile_provider.dart';
import '../data/tile_store.dart';
import 'offline_tile_provider.dart';

class TripMap extends StatelessWidget {
  final MapTileProvider provider;
  final TileStore store;

  /// The route, decoded. Empty draws no line.
  final List<LatLng> route;

  /// Where the stops are, in order.
  final List<({String name, LatLng at})> stops;

  /// True when this region has tiles. False renders the explanation rather
  /// than an empty map.
  final bool hasTiles;

  final double height;

  const TripMap({
    super.key,
    required this.provider,
    required this.store,
    this.route = const [],
    this.stops = const [],
    this.hasTiles = true,
    this.height = 260,
  });

  LatLng get _centre {
    if (route.isNotEmpty) {
      final box = boundsOf(route);
      return LatLng((box.north + box.south) / 2, (box.east + box.west) / 2);
    }
    if (stops.isNotEmpty) return stops.first.at;
    return const LatLng(25.5788, 91.8933);
  }

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);

    if (!provider.isConfigured) {
      return _Notice(
        height: height,
        title: 'Maps are off in this build',
        body: provider.configurationHint,
      );
    }
    if (!hasTiles) {
      return _Notice(
        height: height,
        title: 'Nothing downloaded for this trip yet',
        body:
            'Download the map on WiFi before you leave. After that it works '
            'with no signal at all.',
      );
    }

    return SizedBox(
      height: height,
      child: Stack(
        children: [
          fm.FlutterMap(
            options: fm.MapOptions(
              initialCenter: ll.LatLng(_centre.lat, _centre.lon),
              initialZoom: 11,
              minZoom: 10,
              maxZoom: 16,
              backgroundColor: c.stone,
            ),
            children: [
              fm.TileLayer(
                tileProvider: OfflineTileProvider(
                  store: store,
                  providerId: provider.id,
                ),
                tileDimension: provider.tileSize,
                // A URL template is required by the widget but never used:
                // OfflineTileProvider ignores it and reads from disk. The
                // placeholder keeps the real one, and the key, out of the
                // widget tree entirely.
                urlTemplate: 'offline://{z}/{x}/{y}',
                retinaMode: false,
              ),
              if (route.length > 1)
                fm.PolylineLayer(
                  polylines: [
                    fm.Polyline(
                      points: [
                        for (final p in route) ll.LatLng(p.lat, p.lon),
                      ],
                      strokeWidth: 3,
                      color: c.signal,
                    ),
                  ],
                ),
              if (stops.isNotEmpty)
                fm.MarkerLayer(
                  markers: [
                    for (var i = 0; i < stops.length; i++)
                      fm.Marker(
                        point: ll.LatLng(stops[i].at.lat, stops[i].at.lon),
                        width: 22,
                        height: 22,
                        child: _StopPin(index: i + 1),
                      ),
                  ],
                ),
            ],
          ),

          // ATTRIBUTION, ON EVERY MAP. Required by the provider's terms and
          // by OpenStreetMap's licence; not a styling choice.
          Positioned(
            left: 0,
            right: 0,
            bottom: 0,
            child: Container(
              color: c.paper.withValues(alpha: 0.82),
              padding: const EdgeInsets.symmetric(
                horizontal: AppTokens.s8,
                vertical: 3,
              ),
              child: Text(
                provider.attribution,
                style: AppTokens.captionStyle.copyWith(
                  color: c.muted,
                  fontSize: 10,
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _StopPin extends StatelessWidget {
  final int index;
  const _StopPin({required this.index});

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    return Container(
      alignment: Alignment.center,
      decoration: BoxDecoration(
        color: c.paper,
        border: Border.all(color: c.ink, width: 1.5),
        shape: BoxShape.circle,
      ),
      child: Text(
        '$index',
        style: AppTokens.numberStyle.copyWith(color: c.ink, fontSize: 11),
      ),
    );
  }
}

/// Stands in for a map that cannot be drawn, and says which of the two
/// reasons it is.
class _Notice extends StatelessWidget {
  final double height;
  final String title;
  final String body;

  const _Notice({
    required this.height,
    required this.title,
    required this.body,
  });

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    return Container(
      height: height,
      width: double.infinity,
      color: c.stone,
      padding: const EdgeInsets.all(AppTokens.s24),
      alignment: Alignment.center,
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(Icons.map_outlined, size: 28, color: c.muted),
          const SizedBox(height: AppTokens.s12),
          Text(
            title,
            textAlign: TextAlign.center,
            style: AppTokens.rowTitleStyle.copyWith(color: c.ink),
          ),
          const SizedBox(height: AppTokens.s4),
          Text(
            body,
            textAlign: TextAlign.center,
            style: AppTokens.captionStyle.copyWith(color: c.muted),
          ),
        ],
      ),
    );
  }
}
