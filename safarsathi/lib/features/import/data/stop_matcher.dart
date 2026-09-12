// lib/features/import/data/stop_matcher.dart
//
// Matching a place name in a spreadsheet against the trip's stops.
//
// AN UNMATCHED NAME IS NOT AN ERROR. The row imports as a trip-level contact
// and the preview says so. Refusing a good phone number because a place was
// spelled differently would be the feature working against its own purpose:
// the number is what the user needs at 9pm outside a locked homestay, not the
// stop association.

/// One of the trip's stops, as far as matching is concerned.
class StopCandidate {
  final int id;
  final String name;
  const StopCandidate(this.id, this.name);
}

class StopMatch {
  final int? stopId;
  final String? stopName;

  /// True when the sheet named a place and nothing on the trip resembled it.
  /// The preview surfaces this; it does not block the row.
  final bool unmatched;

  const StopMatch({this.stopId, this.stopName, this.unmatched = false});

  static const none = StopMatch();
}

/// Lowercased, letters and digits only, so `Cherrapunji (Sohra)` and
/// `cherrapunji` collapse together before any distance is measured.
String _squash(String s) =>
    s.toLowerCase().replaceAll(RegExp(r'[^a-z0-9]'), '');

/// Levenshtein distance, iterative with two rows.
///
/// Written out rather than pulled from a package: it is fifteen lines, it runs
/// against a handful of stop names, and the app's whole point is to have no
/// dependency it does not need.
int levenshtein(String a, String b) {
  if (a == b) return 0;
  if (a.isEmpty) return b.length;
  if (b.isEmpty) return a.length;

  var previous = List<int>.generate(b.length + 1, (i) => i);
  var current = List<int>.filled(b.length + 1, 0);

  for (var i = 0; i < a.length; i++) {
    current[0] = i + 1;
    for (var j = 0; j < b.length; j++) {
      final cost = a.codeUnitAt(i) == b.codeUnitAt(j) ? 0 : 1;
      current[j + 1] = [
        current[j] + 1,
        previous[j + 1] + 1,
        previous[j] + cost,
      ].reduce((x, y) => x < y ? x : y);
    }
    final swap = previous;
    previous = current;
    current = swap;
  }
  return previous[b.length];
}

/// The distance we will forgive for a name of this length.
///
/// Scaled, because a fixed threshold is wrong at both ends: allowing two edits
/// makes `Puri` match `Pune`, and allowing only one makes `Cherrapunji` miss
/// `Cherrapunjee`. Roughly a fifth of the name, floor 1 for anything over four
/// characters, cap 3.
int _tolerance(int length) {
  if (length <= 4) return 0;
  final scaled = length ~/ 5;
  return scaled.clamp(1, 3);
}

/// Resolves [raw] against [stops].
///
/// Exact squashed match first, then containment either way (so `Shillong` in
/// the sheet finds `Shillong, Meghalaya` on the trip), then the closest
/// candidate within tolerance.
StopMatch matchStop(String raw, List<StopCandidate> stops) {
  final needle = _squash(raw);
  if (needle.isEmpty) return StopMatch.none;
  if (stops.isEmpty) {
    return StopMatch(unmatched: true, stopName: raw.trim());
  }

  for (final s in stops) {
    if (_squash(s.name) == needle) {
      return StopMatch(stopId: s.id, stopName: s.name);
    }
  }

  for (final s in stops) {
    final hay = _squash(s.name);
    if (hay.isEmpty) continue;
    if (hay.contains(needle) || needle.contains(hay)) {
      return StopMatch(stopId: s.id, stopName: s.name);
    }
  }

  StopCandidate? best;
  var bestDistance = 1 << 30;
  for (final s in stops) {
    final d = levenshtein(needle, _squash(s.name));
    if (d < bestDistance) {
      bestDistance = d;
      best = s;
    }
  }

  if (best != null && bestDistance <= _tolerance(needle.length)) {
    return StopMatch(stopId: best.id, stopName: best.name);
  }

  return StopMatch(unmatched: true, stopName: raw.trim());
}
