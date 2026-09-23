// lib/features/contacts/data/place_location.dart
//
// Where a diary entry is, as a person can actually supply it, and the
// Google Maps handoff for getting there.
//
// NO GEOCODING. Turning "Civil Hospital, Shillong" into a point needs a
// server, and a guessed point is worse than none: it would sit the hospital
// at the wrong kilometre on every leg it touched. So a position comes from
// the person — pasted from Google Maps, or where they are standing — and the
// app only reads it.

import '../../discovery/data/geo.dart';

/// A position read from pasted text, or the one sentence saying why not.
class ParsedLocation {
  final LatLng? at;
  final String? problem;
  const ParsedLocation._(this.at, this.problem);

  static const empty = ParsedLocation._(null, null);
  const ParsedLocation.ok(LatLng at) : this._(at, null);
  const ParsedLocation.bad(String problem) : this._(null, problem);

  bool get isEmpty => at == null && problem == null;
}

const _num = r'(-?\d{1,3}(?:\.\d+)?)';

/// Reads what Google Maps gives you, in each of the shapes it gives it:
///
///  * `25.567739, 91.881081` — press and hold a spot; this is what appears
///    in the search box, and the most reliable thing to paste
///  * `25°34'03.9"N 91°52'51.9"E` — shown on a dropped pin
///  * a long google.com/maps link: `!3d…!4d…`, `@lat,lon,17z`, `?q=lat,lon`
///  * `geo:lat,lon`
///
/// A maps.app.goo.gl short link holds no coordinates at all — only Google's
/// server knows where it points — so it is refused with what to do instead.
ParsedLocation parseLocation(String raw) {
  var text = raw.trim();
  if (text.isEmpty) return ParsedLocation.empty;

  if (text.contains('goo.gl')) {
    return const ParsedLocation.bad(
      'That is a short link, which has no coordinates in it. In Google Maps, '
      'press and hold the place, then copy the numbers shown at the top.',
    );
  }

  try {
    text = Uri.decodeFull(text);
  } on Object {
    // A stray % that is not an escape: read it as typed.
  }

  final patterns = [
    // The place's own pin. Checked before "@", which is only where the
    // camera was when the link was copied.
    RegExp('!3d$_num!4d$_num'),
    RegExp('[?&](?:q|query|ll|destination|daddr|center)=$_num\\s*,\\s*$_num'),
    RegExp('@$_num,$_num'),
    RegExp('geo:$_num,$_num'),
    RegExp('^$_num\\s*°?\\s*[,; ]\\s*$_num\\s*°?\$'),
  ];
  for (final p in patterns) {
    final m = p.firstMatch(text);
    if (m != null) {
      return _checked(double.parse(m[1]!), double.parse(m[2]!));
    }
  }

  // Degrees with a hemisphere: decimal ("25.5677° N, 91.8810° E") or
  // degrees-minutes-seconds ("25°34'03.9"N 91°52'51.9"E").
  final hemi = RegExp(
    r'''(\d{1,3})(?:\.(\d+))?°\s*(?:(\d{1,2})['′]\s*(?:([\d.]+)["″]\s*)?)?([NS])[,\s]+(\d{1,3})(?:\.(\d+))?°\s*(?:(\d{1,2})['′]\s*(?:([\d.]+)["″]\s*)?)?([EW])''',
  ).firstMatch(text);
  if (hemi != null) {
    double part(int deg, int frac, int min, int sec, int dir) {
      final whole = double.parse(
        '${hemi[deg]}${hemi[frac] == null ? '' : '.${hemi[frac]}'}',
      );
      final m = hemi[min] == null ? 0.0 : double.parse(hemi[min]!);
      final s = hemi[sec] == null ? 0.0 : double.parse(hemi[sec]!);
      final value = whole + m / 60 + s / 3600;
      return (hemi[dir] == 'S' || hemi[dir] == 'W') ? -value : value;
    }

    return _checked(part(1, 2, 3, 4, 5), part(6, 7, 8, 9, 10));
  }

  return const ParsedLocation.bad(
    'Could not find a location in that. Paste two numbers like '
    '25.5677, 91.8810 — in Google Maps, press and hold the place and copy '
    'the numbers shown at the top.',
  );
}

ParsedLocation _checked(double lat, double lon) {
  if (lat.abs() > 90 || lon.abs() > 180) {
    return ParsedLocation.bad('$lat, $lon is not a place on Earth.');
  }
  // A blank read as a number lands in the Gulf of Guinea. Same rule as the
  // importer: refused, never trusted.
  if (lat == 0 && lon == 0) {
    return const ParsedLocation.bad('0, 0 is not a place, it is a blank.');
  }
  return ParsedLocation.ok(LatLng(lat, lon));
}

/// How a position is written back into the field: six decimals is about
/// ten centimetres, more than any pin is worth, and round-trips exactly.
String formatLocation(LatLng at) =>
    '${at.lat.toStringAsFixed(6)}, ${at.lon.toStringAsFixed(6)}';

/// Directions to [to] in Google Maps. With no origin, Google Maps starts
/// from wherever the phone is — which is the whole request.
String directionsUrl(LatLng to) =>
    'https://www.google.com/maps/dir/?api=1&destination='
    '${to.lat.toStringAsFixed(6)},${to.lon.toStringAsFixed(6)}';

/// A search by name, for an entry with no saved position. Said on screen to
/// be a search: Google may find a different place of the same name.
String mapsSearchUrl(String name, String? stopName) {
  final query = stopName == null || name.contains(stopName)
      ? name
      : '$name, $stopName';
  return 'https://www.google.com/maps/search/?api=1&query='
      '${Uri.encodeQueryComponent(query)}';
}

/// "1.2 km" / "350 m", straight line.
String describeDistance(double metres) => metres < 1000
    ? '${(metres / 10).round() * 10} m'
    : '${(metres / 1000).toStringAsFixed(metres < 10000 ? 1 : 0)} km';
