// lib/core/util/sun.dart
//
// When the light goes — offline, from a latitude, a longitude and a date.
//
// WHY THIS IS HERE. In the Khasi hills in October it is dark soon after five,
// and the hill roads after dark are the real hazard of the trip: no lights,
// no barriers, fog, and a shared sumo that will not wait. "Leaves 15:30,
// sunset 17:08" is the sentence that decides whether to take the later one.
// It needs no network — only arithmetic — so it belongs in an app that has
// no network on the road.
//
// NOAA's general solar position equations. Accurate to about a minute, which
// is all a decision about daylight needs; pinned against an independent
// implementation (astral) in the tests.

import 'dart:math' as math;

import 'package:flutter/foundation.dart' show visibleForTesting;

class SunTimes {
  /// UTC instants. Null only where the sun does not rise or set that day —
  /// never in India, but the maths is general and says so rather than lying.
  final DateTime? sunrise;
  final DateTime? sunset;

  const SunTimes({this.sunrise, this.sunset});
}

/// Sunrise and sunset at [lat], [lon] on the calendar [date] (its year,
/// month and day; the time of day is ignored).
///
/// "Sunrise" and "sunset" are the standard ones: the upper limb of the sun on
/// the horizon, allowing for refraction (zenith 90.833°). Civil twilight adds
/// roughly twenty usable minutes after sunset; that margin is deliberately
/// not counted here.
SunTimes sunTimes(double lat, double lon, DateTime date) {
  final day = DateTime.utc(date.year, date.month, date.day);
  final dayOfYear = day.difference(DateTime.utc(date.year)).inDays + 1;
  final yearDays = _isLeap(date.year) ? 366 : 365;

  // Fractional year at local solar noon, in radians.
  final gamma = 2 * math.pi / yearDays * (dayOfYear - 1);

  // Equation of time (minutes) and solar declination (radians).
  final eqTime = 229.18 *
      (0.000075 +
          0.001868 * math.cos(gamma) -
          0.032077 * math.sin(gamma) -
          0.014615 * math.cos(2 * gamma) -
          0.040849 * math.sin(2 * gamma));
  final decl = 0.006918 -
      0.399912 * math.cos(gamma) +
      0.070257 * math.sin(gamma) -
      0.006758 * math.cos(2 * gamma) +
      0.000907 * math.sin(2 * gamma) -
      0.002697 * math.cos(3 * gamma) +
      0.00148 * math.sin(3 * gamma);

  final latRad = lat * math.pi / 180;
  const zenith = 90.833 * math.pi / 180;
  final cosHa = math.cos(zenith) / (math.cos(latRad) * math.cos(decl)) -
      math.tan(latRad) * math.tan(decl);
  if (cosHa < -1 || cosHa > 1) return const SunTimes();

  final haDegrees = math.acos(cosHa) * 180 / math.pi;
  final riseMinutes = 720 - 4 * (lon + haDegrees) - eqTime;
  final setMinutes = 720 - 4 * (lon - haDegrees) - eqTime;

  DateTime at(double minutes) =>
      day.add(Duration(microseconds: (minutes * 60e6).round()));

  return SunTimes(sunrise: at(riseMinutes), sunset: at(setMinutes));
}

bool _isLeap(int y) => (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;

/// Pins every clock time to one offset, for the golden images.
///
/// A golden rendered on a UTC machine said "sunrise 23:46" — true in UTC,
/// absurd on a screen about Meghalaya — and would have failed on any machine
/// in India. Set once in the golden harness; never set in the app.
@visibleForTesting
Duration? clockOffsetOverride;

/// "17:08", in the phone's own time zone unless [offset] says otherwise.
///
/// The offset is injectable so the tests read Indian time without depending
/// on the time zone of the machine running them.
String clockTime(DateTime utc, {Duration? offset}) {
  final shift = offset ?? clockOffsetOverride;
  final local = shift == null ? utc.toLocal() : utc.toUtc().add(shift);
  return '${local.hour.toString().padLeft(2, '0')}:'
      '${local.minute.toString().padLeft(2, '0')}';
}

/// How [when] sits against the day's light, as a sentence — or null when it
/// is comfortably inside daylight and nothing needs saying.
///
/// Thirty minutes before sunset counts as cutting it close: in a valley the
/// light goes before the sun does.
String? daylightWarning(DateTime when, SunTimes sun, {required String verb}) {
  final set = sun.sunset;
  final rise = sun.sunrise;
  if (set == null || rise == null) return null;
  final t = when.toUtc();

  if (t.isAfter(set)) {
    final mins = _minutes(t.difference(set));
    return '$verb ${_span(mins)} after sunset — in the dark.';
  }
  if (t.isBefore(rise)) {
    final mins = _minutes(rise.difference(t));
    return '$verb ${_span(mins)} before sunrise — in the dark.';
  }
  final left = _minutes(set.difference(t));
  if (left <= 30) {
    return '$verb only ${_span(left)} before sunset.';
  }
  return null;
}

/// Rounded, not truncated. The database keeps whole seconds and the sunset
/// has fractions of one, so 45 minutes after sunset comes back as 44:59 —
/// and "44 min" for a time somebody typed as 45 reads as a bug.
int _minutes(Duration d) => (d.inSeconds / 60).round();

String _span(int minutes) {
  if (minutes < 60) return '$minutes min';
  final h = minutes ~/ 60;
  final m = minutes % 60;
  return m == 0 ? '$h h' : '$h h $m min';
}
