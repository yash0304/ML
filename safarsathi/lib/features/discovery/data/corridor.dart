// lib/features/discovery/data/corridor.dart
//
// A route, a band either side of it, and where each place falls along it.
//
// This is what makes "coming up in 12 km" possible, and what separates a dhaba
// on your road from a dhaba on the parallel valley road 9 km away that you
// would have to leave the route for an hour to reach.

import 'geo.dart';

/// Where a point sits relative to a route.
class CorridorPosition {
  /// Shortest distance from the point to the route line, in kilometres.
  final double offRouteKm;

  /// How far along the route the closest point sits, in kilometres from the
  /// start. This is the ordering the discovery screen uses.
  final double alongRouteKm;

  const CorridorPosition({
    required this.offRouteKm,
    required this.alongRouteKm,
  });
}

class Corridor {
  /// The decoded route.
  final List<LatLng> route;

  /// Half-width of the band, in kilometres. 3 km by default: far enough to
  /// catch a place signposted off the highway, near enough that reaching it is
  /// a detour and not an expedition.
  final double bufferKm;

  Corridor(this.route, {this.bufferKm = 3.0})
    : assert(bufferKm > 0, 'A corridor with no width contains nothing.');

  /// Cumulative distance to each vertex, in kilometres. Computed once because
  /// every point queried walks the whole route.
  late final List<double> _cumulativeKm = _buildCumulative();

  List<double> _buildCumulative() {
    final out = <double>[0];
    for (var i = 1; i < route.length; i++) {
      out.add(out[i - 1] + haversineMetres(route[i - 1], route[i]) / 1000);
    }
    return out;
  }

  /// Total route length in kilometres.
  double get lengthKm => route.length < 2 ? 0 : _cumulativeKm.last;

  /// The box to query, the route's own bounds grown by the buffer.
  BoundingBox get queryBox => padBox(boundsOf(route), bufferKm);

  /// Where [point] falls. Null when the route has fewer than two points, since
  /// there is no line to measure against.
  CorridorPosition? locate(LatLng point) {
    if (route.length < 2) return null;

    var bestMetres = double.infinity;
    var bestAlongKm = 0.0;

    for (var i = 0; i < route.length - 1; i++) {
      final a = route[i];
      final b = route[i + 1];

      // DISTANCE TO A ROUTE IS DISTANCE TO A SEGMENT, never to a vertex. A
      // road running straight for 40 km has two vertices, and measuring to the
      // nearer one would put a dhaba halfway along it 20 km off the route.
      final metres = distanceToSegmentMetres(point, a, b);
      if (metres >= bestMetres) continue;

      bestMetres = metres;
      final t = fractionAlongSegment(point, a, b);
      bestAlongKm =
          _cumulativeKm[i] + (_cumulativeKm[i + 1] - _cumulativeKm[i]) * t;
    }

    return CorridorPosition(
      offRouteKm: bestMetres / 1000,
      alongRouteKm: bestAlongKm,
    );
  }

  bool covers(LatLng point) {
    final position = locate(point);
    return position != null && position.offRouteKm <= bufferKm;
  }

  /// Positions for every point inside the band, in the order you will reach
  /// them. Points outside the band are dropped: the bounding box is a
  /// rectangle and the corridor is not, so a query returns corners the route
  /// never goes near.
  List<({T item, CorridorPosition position})> place<T>(
    List<T> items,
    LatLng Function(T) locationOf,
  ) {
    final out = <({T item, CorridorPosition position})>[];
    for (final item in items) {
      final position = locate(locationOf(item));
      if (position == null || position.offRouteKm > bufferKm) continue;
      out.add((item: item, position: position));
    }
    out.sort(
      (a, b) => a.position.alongRouteKm.compareTo(b.position.alongRouteKm),
    );
    return out;
  }
}
