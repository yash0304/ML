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
import '../data/here.dart';
import '../data/tile_downloader.dart'
    show defaultMinZoom, defaultMaxZoom;
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

  /// Where the phone is, from its own GPS. Null draws no dot.
  final HereFix? me;

  /// Lets the screen recentre on [me] without rebuilding the map.
  final fm.MapController? controller;

  /// One place to mark and open on — a diary entry's page. Null for the
  /// whole-trip map.
  final LatLng? place;

  /// Where to open, when the trip's bottom level is not the right one. A
  /// single place opens at the closest level downloaded for it.
  final int? initialZoom;

  const TripMap({
    super.key,
    required this.provider,
    required this.store,
    this.route = const [],
    this.stops = const [],
    this.hasTiles = true,
    this.height = 260,
    this.me,
    this.controller,
    this.place,
    this.initialZoom,
  });

  LatLng get _centre {
    if (place != null) return place!;
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

    // TILES FIRST, KEY SECOND, AND THE ORDER IS THE WHOLE POINT.
    //
    // `OfflineTileProvider` reads files off this phone and never looks at a
    // key. Checking `isConfigured` first meant a phone holding a complete
    // downloaded map refused to draw it and said "maps are off" — which is
    // exactly the state somebody is in after restoring a backup onto a new
    // handset, standing in a valley, needing the map they already paid for.
    // A key is needed to FETCH tiles. It is needed for nothing else.
    if (!hasTiles) {
      return _Notice(
        height: height,
        title: provider.isConfigured
            ? 'Nothing downloaded for this trip yet'
            : 'Maps are off until you add a key',
        body: provider.isConfigured
            ? 'Download the map on WiFi before you leave. After that it works '
                  'with no signal at all.'
            : provider.configurationHint,
      );
    }

    return SizedBox(
      height: height,
      child: Stack(
        children: [
          fm.FlutterMap(
            mapController: controller,
            options: fm.MapOptions(
              initialCenter: ll.LatLng(_centre.lat, _centre.lon),
              // OPENS INSIDE THE BAND THAT WAS ACTUALLY DOWNLOADED.
              //
              // This opened at 11 while the downloader only ever fetches
              // defaultMinZoom..defaultMaxZoom, which is 12 to 15. Zoom 11
              // is a level for which not one tile has ever existed on any
              // phone, so the map drew the route and the stops over blank
              // paper and looked broken — reported as "all maps downloaded
              // but it is not at all readable".
              //
              // Two literals in two files that were never checked against
              // each other. They are now the same constants.
              initialZoom: (initialZoom ?? defaultMinZoom).toDouble(),
              // One step out is still legible: tiles below minNativeZoom are
              // scaled rather than dropped. Further out would need 64 tiles
              // to cover what one covers, which is a stutter, not a map.
              minZoom: defaultMinZoom - 1,
              maxZoom: defaultMaxZoom + 2,
              backgroundColor: c.stone,
            ),
            children: [
              fm.TileLayer(
                tileProvider: OfflineTileProvider(
                  store: store,
                  providerId: provider.id,
                ),
                // THE GRID SIZE, NOT THE IMAGE SIZE. This read tileSize
                // (512, the pixels in a retina tile) and flutter_map divides
                // the map's 256-based pixel bounds by it to pick x and y —
                // so every index came out halved and the layer asked for z12
                // tiles by z11 numbers. Nothing downloaded was ever at those
                // coordinates, so a complete 204 MB map drew as blank paper.
                tileDimension: provider.gridSize,
                // Outside this band flutter_map scales the nearest tile it
                // has instead of asking for a level nothing was downloaded
                // for. Blurry beats blank: past 15 it is soft but readable,
                // and the alternative is an empty screen.
                minNativeZoom: defaultMinZoom,
                maxNativeZoom: defaultMaxZoom,
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
              // Ink, not red: red is the SOS tab's alone. The pin's tip is
              // the point, so the marker sits above it.
              if (place != null)
                fm.MarkerLayer(
                  markers: [
                    fm.Marker(
                      point: ll.LatLng(place!.lat, place!.lon),
                      width: 36,
                      height: 36,
                      alignment: Alignment.topCenter,
                      child: Icon(
                        Icons.place,
                        key: const Key('place-pin'),
                        size: 36,
                        color: c.ink,
                      ),
                    ),
                  ],
                ),
              // YOU, LAST, SO NOTHING DRAWS OVER YOU. The circle is the fix's
              // own accuracy in metres on the ground: a wide one says "you
              // are somewhere in here", which is the truth when the GPS is
              // still settling in a valley.
              if (me != null) ...[
                fm.CircleLayer(
                  circles: [
                    fm.CircleMarker(
                      point: ll.LatLng(me!.at.lat, me!.at.lon),
                      radius: me!.accuracyM,
                      useRadiusInMeter: true,
                      color: c.signal.withValues(alpha: 0.14),
                      borderColor: c.signal.withValues(alpha: 0.5),
                      borderStrokeWidth: 1,
                    ),
                  ],
                ),
                fm.MarkerLayer(
                  markers: [
                    fm.Marker(
                      point: ll.LatLng(me!.at.lat, me!.at.lon),
                      width: 18,
                      height: 18,
                      child: Semantics(
                        label: 'You are here',
                        child: Container(
                          key: const Key('you-are-here'),
                          decoration: BoxDecoration(
                            color: c.signal,
                            shape: BoxShape.circle,
                            border: Border.all(color: c.paper, width: 3),
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
              ],
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
