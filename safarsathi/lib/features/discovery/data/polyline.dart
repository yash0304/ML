// lib/features/discovery/data/polyline.dart
//
// The Google encoded-polyline codec, written out rather than pulled in.
//
// Forty lines of bit-shifting that has to be exactly right, over a format that
// has not changed since 2005. A package dependency would mean trusting
// somebody else's forty lines and carrying their release cycle.
//
// PRECISION IS AN EXPLICIT ARGUMENT. OSRM's older API returns precision 5;
// the v5 API returns 6 when asked. Getting it wrong scales the whole route by
// ten and puts Shillong in the Bay of Bengal — a failure that looks like a
// bug in the map, not a bug in a number.

import 'geo.dart';

class Polyline {
  Polyline._();

  /// Decodes [encoded] into points.
  ///
  /// A malformed string yields what it managed to read rather than throwing:
  /// half a route still draws, and the caller checks the length.
  static List<LatLng> decode(String encoded, {int precision = 5}) {
    final factor = _factorFor(precision);
    final points = <LatLng>[];

    var index = 0;
    var lat = 0;
    var lon = 0;

    while (index < encoded.length) {
      final dLat = _readValue(encoded, index);
      if (dLat == null) break;
      index = dLat.nextIndex;
      lat += dLat.value;

      final dLon = _readValue(encoded, index);
      if (dLon == null) break;
      index = dLon.nextIndex;
      lon += dLon.value;

      points.add(LatLng(lat / factor, lon / factor));
    }
    return points;
  }

  static String encode(List<LatLng> points, {int precision = 5}) {
    final factor = _factorFor(precision);
    final buffer = StringBuffer();

    var lastLat = 0;
    var lastLon = 0;

    for (final p in points) {
      final lat = (p.lat * factor).round();
      final lon = (p.lon * factor).round();
      _writeValue(buffer, lat - lastLat);
      _writeValue(buffer, lon - lastLon);
      lastLat = lat;
      lastLon = lon;
    }
    return buffer.toString();
  }

  static double _factorFor(int precision) {
    var factor = 1.0;
    for (var i = 0; i < precision; i++) {
      factor *= 10;
    }
    return factor;
  }

  static ({int value, int nextIndex})? _readValue(String encoded, int start) {
    var index = start;
    var shift = 0;
    var result = 0;
    int byte;

    do {
      if (index >= encoded.length) return null;
      byte = encoded.codeUnitAt(index++) - 63;
      result |= (byte & 0x1f) << shift;
      shift += 5;
    } while (byte >= 0x20);

    // The low bit is the sign, and the value is inverted when it is set.
    final value = (result & 1) != 0 ? ~(result >> 1) : result >> 1;
    return (value: value, nextIndex: index);
  }

  static void _writeValue(StringBuffer buffer, int value) {
    var v = value < 0 ? ~(value << 1) : value << 1;
    while (v >= 0x20) {
      buffer.writeCharCode((0x20 | (v & 0x1f)) + 63);
      v >>= 5;
    }
    buffer.writeCharCode(v + 63);
  }
}
