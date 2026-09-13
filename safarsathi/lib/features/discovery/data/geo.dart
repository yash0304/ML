// lib/features/discovery/data/geo.dart
//
// The small amount of spherical geometry this app needs, written out.
//
// Pure functions over doubles, so every one of them is arguable in a test
// rather than only observable on a map. Nothing here touches the network or
// the database.

import 'dart:math' as math;

/// Metres per degree of latitude. Constant enough at any latitude — the
/// variation across the whole planet is about 0.6%, which is centimetres over
/// the distances this app measures.
const _metresPerDegreeLat = 111132.0;

const earthRadiusMetres = 6371008.8;

class LatLng {
  final double lat;
  final double lon;
  const LatLng(this.lat, this.lon);

  @override
  String toString() =>
      'LatLng(${lat.toStringAsFixed(6)}, ${lon.toStringAsFixed(6)})';

  @override
  bool operator ==(Object other) =>
      other is LatLng && other.lat == lat && other.lon == lon;

  @override
  int get hashCode => Object.hash(lat, lon);
}

/// A bounding box, in the order Overpass wants it: south, west, north, east.
class BoundingBox {
  final double south;
  final double west;
  final double north;
  final double east;

  const BoundingBox({
    required this.south,
    required this.west,
    required this.north,
    required this.east,
  });

  /// Overpass and most OSM tooling take `(south,west,north,east)`.
  String get overpassString =>
      '${south.toStringAsFixed(6)},${west.toStringAsFixed(6)},'
      '${north.toStringAsFixed(6)},${east.toStringAsFixed(6)}';

  double get heightDegrees => north - south;
  double get widthDegrees => east - west;

  /// Rough area in square kilometres. Used to refuse a query so large it is
  /// certainly a bug, and which would cost the Overpass volunteers real money.
  double get approxAreaSqKm {
    final midLat = (north + south) / 2;
    final heightKm = heightDegrees * _metresPerDegreeLat / 1000;
    final widthKm =
        widthDegrees *
        _metresPerDegreeLat *
        math.cos(midLat * math.pi / 180) /
        1000;
    return (heightKm * widthKm).abs();
  }

  bool contains(LatLng p) =>
      p.lat >= south && p.lat <= north && p.lon >= west && p.lon <= east;
}

/// Great-circle distance in metres.
double haversineMetres(LatLng a, LatLng b) {
  final dLat = _rad(b.lat - a.lat);
  final dLon = _rad(b.lon - a.lon);
  final lat1 = _rad(a.lat);
  final lat2 = _rad(b.lat);

  final h =
      math.sin(dLat / 2) * math.sin(dLat / 2) +
      math.cos(lat1) * math.cos(lat2) * math.sin(dLon / 2) * math.sin(dLon / 2);
  return 2 * earthRadiusMetres * math.asin(math.min(1.0, math.sqrt(h)));
}

/// Shortest distance in metres from [p] to the SEGMENT a→b, not to its ends.
///
/// A road running dead straight for 40 km has two vertices. Measuring to the
/// nearer vertex would report a dhaba halfway along it as 20 km off the route
/// when it is sitting on it.
///
/// Projects onto a local tangent plane anchored at `a`, which is accurate to
/// well under a metre over the segment lengths a road polyline contains.
double distanceToSegmentMetres(LatLng p, LatLng a, LatLng b) {
  final scale = math.cos(_rad(a.lat));

  const ax = 0.0;
  const ay = 0.0;
  final bx = (b.lon - a.lon) * scale;
  final by = b.lat - a.lat;
  final px = (p.lon - a.lon) * scale;
  final py = p.lat - a.lat;

  final dx = bx - ax;
  final dy = by - ay;
  final lengthSquared = dx * dx + dy * dy;

  // A degenerate segment — the same point twice, which OSRM does emit — is
  // just a point.
  if (lengthSquared == 0) return haversineMetres(p, a);

  // How far along the segment the closest point sits, clamped to its ends.
  var t = ((px - ax) * dx + (py - ay) * dy) / lengthSquared;
  t = t.clamp(0.0, 1.0);

  final closest = LatLng(a.lat + dy * t, a.lon + (dx * t) / scale);
  return haversineMetres(p, closest);
}

/// Where along the segment the closest point to [p] falls, 0 at `a`, 1 at `b`.
double fractionAlongSegment(LatLng p, LatLng a, LatLng b) {
  final scale = math.cos(_rad(a.lat));
  final bx = (b.lon - a.lon) * scale;
  final by = b.lat - a.lat;
  final px = (p.lon - a.lon) * scale;
  final py = p.lat - a.lat;

  final lengthSquared = bx * bx + by * by;
  if (lengthSquared == 0) return 0;
  return (((px * bx) + (py * by)) / lengthSquared).clamp(0.0, 1.0);
}

/// The box containing every point, with no padding.
BoundingBox boundsOf(List<LatLng> points) {
  if (points.isEmpty) {
    throw ArgumentError('No points to bound.');
  }
  var south = points.first.lat;
  var north = points.first.lat;
  var west = points.first.lon;
  var east = points.first.lon;

  for (final p in points) {
    if (p.lat < south) south = p.lat;
    if (p.lat > north) north = p.lat;
    if (p.lon < west) west = p.lon;
    if (p.lon > east) east = p.lon;
  }
  return BoundingBox(south: south, west: west, north: north, east: east);
}

/// Grows a box by [km] on every side.
///
/// LONGITUDE DEGREES SHRINK WITH LATITUDE. Three kilometres is 3/111 degrees
/// of latitude but 3/(111·cos φ) degrees of longitude. At Shillong's 25.6°
/// that is a 10% difference; ignoring it makes the corridor narrower than
/// advertised on the east–west axis, and the places the user was promised are
/// simply missing with no error anywhere.
BoundingBox padBox(BoundingBox box, double km) {
  final latPad = km * 1000 / _metresPerDegreeLat;

  // Widen using whichever edge is furthest from the equator, so the padding is
  // at least [km] everywhere in the box rather than only at its middle.
  final worstLat = math.max(box.north.abs(), box.south.abs());
  final scale = math.cos(_rad(worstLat));
  // Near a pole the scale collapses and the padding would explode. Nothing
  // this app does goes there, but a clamp beats an infinity.
  final lonPad = scale < 0.01
      ? 180.0
      : km * 1000 / (_metresPerDegreeLat * scale);

  return BoundingBox(
    south: (box.south - latPad).clamp(-90.0, 90.0),
    north: (box.north + latPad).clamp(-90.0, 90.0),
    west: (box.west - lonPad).clamp(-180.0, 180.0),
    east: (box.east + lonPad).clamp(-180.0, 180.0),
  );
}

double _rad(double degrees) => degrees * math.pi / 180;
