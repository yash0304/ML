// lib/features/map/data/map_download.dart
//
// Working out what a trip's maps would cost, and fetching them — issue #24.
//
// Tiles cover the same corridor #23 already computes, so the box a leg needs
// is the box its places came from. Where a leg has no route yet, the
// straight line between its stops stands in, widened by the corridor.

import 'package:drift/drift.dart';

import '../../../core/database/app_database.dart';
import '../../discovery/data/corridor.dart';
import '../../discovery/data/geo.dart';
import '../../discovery/data/polyline.dart';
import 'tile_downloader.dart';
import 'tile_math.dart';

/// What downloading a trip's maps would involve, stated before anything runs.
class MapEstimate {
  final int legCount;
  final int legsWithoutCoordinates;
  final int tileCount;
  final int alreadyHave;

  const MapEstimate({
    required this.legCount,
    required this.legsWithoutCoordinates,
    required this.tileCount,
    required this.alreadyHave,
  });

  int get toFetch => tileCount - alreadyHave;
  int get estimatedBytes => toFetch * averageTileBytes;
  String get estimatedSize => describeBytes(estimatedBytes);
  bool get isComplete => tileCount > 0 && toFetch == 0;
  bool get hasNothingToDo => tileCount == 0;
}

/// The box each leg needs covering, in itinerary order.
///
/// A leg already routed uses its real polyline. One not yet routed falls back
/// to the straight line between its stops, which is wrong in the mountains but
/// is the only thing available before #22 has run, and errs wide rather than
/// narrow.
Future<List<BoundingBox>> tripBoxes(AppDatabase db, int tripId) async {
  final legs =
      await (db.select(db.legs)
            ..where((l) => l.tripId.equals(tripId))
            ..orderBy([(l) => OrderingTerm(expression: l.sequenceOrder)]))
          .get();

  final stops = await (db.select(
    db.stops,
  )..where((s) => s.tripId.equals(tripId))).get();
  final byId = {for (final s in stops) s.id: s};

  final boxes = <BoundingBox>[];
  for (final leg in legs) {
    final from = byId[leg.fromStopId];
    final to = byId[leg.toStopId];
    if (from?.lat == null ||
        from?.lon == null ||
        to?.lat == null ||
        to?.lon == null) {
      continue;
    }

    final polyline = leg.routePolyline;
    final points = polyline == null
        ? [LatLng(from!.lat!, from.lon!), LatLng(to!.lat!, to.lon!)]
        : Polyline.decode(polyline);
    if (points.length < 2) continue;

    boxes.add(Corridor(points, bufferKm: leg.corridorKm).queryBox);
  }
  return boxes;
}

/// How many legs cannot be covered at all, so the screen can say so rather
/// than quietly downloading less than the user expects.
Future<int> legsMissingCoordinates(AppDatabase db, int tripId) async {
  final legs = await (db.select(
    db.legs,
  )..where((l) => l.tripId.equals(tripId))).get();
  final stops = await (db.select(
    db.stops,
  )..where((s) => s.tripId.equals(tripId))).get();
  final byId = {for (final s in stops) s.id: s};

  var missing = 0;
  for (final leg in legs) {
    final from = byId[leg.fromStopId];
    final to = byId[leg.toStopId];
    if (from?.lat == null ||
        from?.lon == null ||
        to?.lat == null ||
        to?.lon == null) {
      missing++;
    }
  }
  return missing;
}

class MapDownload {
  final AppDatabase db;
  final TileDownloader downloader;

  const MapDownload({required this.db, required this.downloader});

  Future<MapEstimate> estimate(
    int tripId, {
    int minZoom = defaultMinZoom,
    int maxZoom = defaultMaxZoom,
  }) async {
    final boxes = await tripBoxes(db, tripId);
    final missing = await legsMissingCoordinates(db, tripId);

    // Adjacent legs overlap, so counting per box would double-count the
    // terrain between two stops. Deduplicated by tile.
    final tiles = <TileCoordinate>{};
    for (final box in boxes) {
      tiles.addAll(tilesForBox(box, minZoom: minZoom, maxZoom: maxZoom));
    }

    var have = 0;
    for (final tile in tiles) {
      if (await downloader.store.has(downloader.provider.id, tile)) have++;
    }

    return MapEstimate(
      legCount: boxes.length + missing,
      legsWithoutCoordinates: missing,
      tileCount: tiles.length,
      alreadyHave: have,
    );
  }

  /// Downloads every leg's tiles, reporting progress across the whole trip.
  Stream<TileProgress> download(
    int tripId, {
    int minZoom = defaultMinZoom,
    int maxZoom = defaultMaxZoom,
  }) async* {
    final boxes = await tripBoxes(db, tripId);

    final tiles = <TileCoordinate>{};
    for (final box in boxes) {
      tiles.addAll(tilesForBox(box, minZoom: minZoom, maxZoom: maxZoom));
    }
    final ordered = tiles.toList();

    // One combined bounding box would cover the rectangle enclosing the whole
    // trip, which for a loop is mostly terrain nobody drives through. Instead
    // the union of the legs' own tiles is downloaded, one at a time.
    yield* downloader.downloadTiles(ordered);
  }
}
