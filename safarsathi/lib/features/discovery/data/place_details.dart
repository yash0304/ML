// lib/features/discovery/data/place_details.dart
//
// What OpenStreetMap knows about a place beyond its name and number — kept,
// instead of thrown away.
//
// ASKED FOR: "menus for the fast food joints between the stops." There is no
// source for menus this app could use: OpenStreetMap does not carry them,
// and the delivery apps that do have no public API, forbid scraping, and do
// not list a roadside dhaba in the Khasi hills anyway. What OpenStreetMap
// often does carry is the next best thing for deciding where to stop: what
// kind of food, whether there is anything vegetarian, and the hours. The
// download fetched all of it and kept none; the rawTags column existed and
// was never written.

import 'dart:convert';

/// The tags worth carrying offline. Kept small on purpose: the whole tag set
/// of a place can run to dozens of keys nobody reads at a roadside.
const keptTags = {
  'amenity',
  'cuisine',
  'diet:vegetarian',
  'diet:vegan',
  // Rarely tagged — few places anywhere carry it, and in the Khasi hills
  // probably none — but where it exists it is the one answer a Jain
  // traveller needs, so it is kept and shown.
  'diet:jain',
  'opening_hours',
  'description',
};

/// The kept tags of [tags], as the string stored in `Pois.rawTags`, or null
/// when there is nothing worth storing.
String? encodeKeptTags(Map<String, String> tags) {
  final kept = {
    for (final e in tags.entries)
      if (keptTags.contains(e.key) && e.value.trim().isNotEmpty)
        e.key: e.value.trim(),
  };
  return kept.isEmpty ? null : jsonEncode(kept);
}

Map<String, String> decodeKeptTags(String? raw) {
  if (raw == null || raw.isEmpty) return const {};
  try {
    final decoded = jsonDecode(raw);
    if (decoded is! Map) return const {};
    return {for (final e in decoded.entries) '${e.key}': '${e.value}'};
  } on Object {
    // A damaged row loses its details, never the place.
    return const {};
  }
}

String _title(String s) {
  final words = s.replaceAll('_', ' ').trim().split(RegExp(r'\s+'));
  return words
      .map((w) => w.isEmpty ? w : w[0].toUpperCase() + w.substring(1))
      .join(' ');
}

/// "Fast food"; "Café"; null for anything that is not somewhere to eat.
String? kindOfFood(Map<String, String> tags) => switch (tags['amenity']) {
  'fast_food' => 'Fast food',
  'cafe' => 'Café',
  'restaurant' => 'Restaurant',
  _ => null,
};

/// "Indian, Chinese, Momo" from OSM's semicolon list.
String? cuisineOf(Map<String, String> tags) {
  final raw = tags['cuisine'];
  if (raw == null || raw.trim().isEmpty) return null;
  final parts = raw
      .split(RegExp(r'[;,]'))
      .map((p) => p.trim())
      .where((p) => p.isNotEmpty)
      .map(_title)
      .toList();
  return parts.isEmpty ? null : parts.join(', ');
}

/// What the place says about vegetarian food — "yes" and "only" are very
/// different answers for a Jain or vegetarian traveller, and "no" is worth
/// knowing before the stop, not after.
String? vegOf(Map<String, String> tags) {
  final vegan = tags['diet:vegan'];
  return switch (tags['diet:vegetarian']) {
    'only' => 'Pure veg',
    'yes' => vegan == 'yes' || vegan == 'only' ? 'Veg and vegan' : 'Veg options',
    'limited' => 'Some veg',
    'no' => 'No veg options',
    _ => vegan == 'yes' || vegan == 'only' ? 'Vegan options' : null,
  };
}

/// Jain food, when the place says. "No" is shown too: for a Jain traveller
/// it is worth knowing before the stop, not at the counter.
String? jainOf(Map<String, String> tags) => switch (tags['diet:jain']) {
  'only' => 'Jain only',
  'yes' => 'Jain food',
  'no' => 'No Jain food',
  _ => null,
};

/// OSM's hours, made readable without pretending to parse them: "Mo-Su
/// 09:00-21:00" becomes "Mon–Sun 09:00–21:00". The full grammar has public
/// holidays, sunset offsets and exceptions; showing a half-parsed version as
/// if it were certain would be worse than showing what was written.
String? hoursOf(Map<String, String> tags) {
  final raw = tags['opening_hours'];
  if (raw == null || raw.trim().isEmpty) return null;
  if (raw.trim() == '24/7') return 'Open 24 hours';
  const days = {
    'Mo': 'Mon', 'Tu': 'Tue', 'We': 'Wed', 'Th': 'Thu',
    'Fr': 'Fri', 'Sa': 'Sat', 'Su': 'Sun', 'PH': 'holidays',
  };
  var out = raw.trim();
  days.forEach((k, v) => out = out.replaceAll(RegExp('\\b$k\\b'), v));
  return out.replaceAll('-', '–').replaceAll(';', ',');
}

/// One line for a list row: "Fast food · Indian, Momo · Veg options".
/// Null when nothing beyond the category is known.
String? foodLine(Map<String, String> tags) {
  final parts = [
    kindOfFood(tags),
    cuisineOf(tags),
    vegOf(tags),
    jainOf(tags),
  ].whereType<String>().toList();
  return parts.isEmpty ? null : parts.join(' · ');
}

/// Somewhere a vegetarian can eat — options, limited, or pure veg.
bool servesVeg(Map<String, String> tags) =>
    const {'yes', 'only', 'limited'}.contains(tags['diet:vegetarian']) ||
    const {'yes', 'only'}.contains(tags['diet:vegan']);

/// Somewhere that says it serves Jain food.
bool servesJain(Map<String, String> tags) =>
    const {'yes', 'only'}.contains(tags['diet:jain']);
