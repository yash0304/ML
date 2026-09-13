// lib/features/map/data/tile_math.dart
//
// Slippy-map arithmetic: the Web Mercator tile scheme every raster provider
// uses. Pure functions, no dependency on which provider serves them.

import 'dart:math' as math;

import '../../discovery/data/geo.dart';

/// Web Mercator is undefined beyond this latitude — the projection stretches
/// to infinity at the poles, so every tile scheme cuts it off here.
const mercatorLimit = 85.05112878;

class TileCoordinate {
  final int z;
  final int x;
  final int y;

  const TileCoordinate(this.z, this.x, this.y);

  @override
  String toString() => '$z/$x/$y';

  @override
  bool operator ==(Object other) =>
      other is TileCoordinate && other.z == z && other.x == x && other.y == y;

  @override
  int get hashCode => Object.hash(z, x, y);
}

/// Which tile a coordinate falls in, at [zoom].
TileCoordinate tileFor(LatLng point, int zoom) {
  final scale = 1 << zoom;
  final lat = point.lat.clamp(-mercatorLimit, mercatorLimit);
  final latRad = lat * math.pi / 180;

  final x = ((point.lon + 180) / 360 * scale).floor();
  final y =
      ((1 - math.log(math.tan(latRad) + 1 / math.cos(latRad)) / math.pi) /
              2 *
              scale)
          .floor();

  // A point exactly on the eastern or southern edge lands one past the last
  // tile, which does not exist.
  return TileCoordinate(zoom, x.clamp(0, scale - 1), y.clamp(0, scale - 1));
}

/// The north-west corner of a tile, which is what a renderer positions by.
LatLng tileNorthWest(TileCoordinate tile) {
  final scale = 1 << tile.z;
  final lon = tile.x / scale * 360 - 180;
  final n = math.pi - 2 * math.pi * tile.y / scale;
  final lat = 180 / math.pi * math.atan(0.5 * (math.exp(n) - math.exp(-n)));
  return LatLng(lat, lon);
}

/// Every tile covering [box] between the two zooms, inclusive.
///
/// TILE COUNT GROWS AS 4^z. A corridor at zoom 14 is a few hundred tiles; the
/// same corridor at zoom 18 is tens of thousands. This is why the caller shows
/// the number before anything downloads.
List<TileCoordinate> tilesForBox(
  BoundingBox box, {
  required int minZoom,
  required int maxZoom,
}) {
  if (minZoom > maxZoom) return const [];

  final out = <TileCoordinate>[];
  for (var z = minZoom; z <= maxZoom; z++) {
    final northWest = tileFor(LatLng(box.north, box.west), z);
    final southEast = tileFor(LatLng(box.south, box.east), z);

    for (var x = northWest.x; x <= southEast.x; x++) {
      for (var y = northWest.y; y <= southEast.y; y++) {
        out.add(TileCoordinate(z, x, y));
      }
    }
  }
  return out;
}

/// How many tiles [tilesForBox] would produce, without building the list.
///
/// Counting by multiplication rather than by generating tens of thousands of
/// objects, because this runs on every keystroke of a zoom slider.
int countTilesForBox(
  BoundingBox box, {
  required int minZoom,
  required int maxZoom,
}) {
  if (minZoom > maxZoom) return 0;

  var total = 0;
  for (var z = minZoom; z <= maxZoom; z++) {
    final northWest = tileFor(LatLng(box.north, box.west), z);
    final southEast = tileFor(LatLng(box.south, box.east), z);
    total += (southEast.x - northWest.x + 1) * (southEast.y - northWest.y + 1);
  }
  return total;
}

/// Rough bytes for a tile count.
///
/// A 512px raster tile of terrain averages somewhere near 25 KB; empty ocean
/// is a fraction of that and a dense town is more. The estimate screen says it
/// is an estimate rather than pretending otherwise.
const averageTileBytes = 25 * 1024;

String describeBytes(int bytes) {
  if (bytes < 1024) return '$bytes B';
  if (bytes < 1024 * 1024) return '${(bytes / 1024).round()} KB';
  final mb = bytes / (1024 * 1024);
  return mb < 10 ? '${mb.toStringAsFixed(1)} MB' : '${mb.round()} MB';
}
