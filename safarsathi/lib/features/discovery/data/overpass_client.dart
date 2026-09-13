// lib/features/discovery/data/overpass_client.dart
//
// Places along the route, from OpenStreetMap — issue #21.
//
// RUNS ONCE PER LEG AT SETUP, ON WIFI. Never on a timer, never while moving,
// never in the background. The whole app rests on that.
//
// THE TRUST RULE REACHING ITS LEAST TRUSTWORTHY SOURCE: a phone number from
// OSM is `communityOsm` and can never be anything else. It is a number a
// stranger typed into a public wiki and it may be a decade old. It carries the
// amber dot, it is never pre-confirmed, and saving one into the diary lands it
// as `userEntered` — the same tier as something typed by hand.

import 'dart:convert';

import 'package:http/http.dart' as http;

import 'geo.dart';
import 'poi_category.dart';

/// Identifies the app to the Overpass volunteers, as their usage policy asks.
/// An anonymous flood is how a free service gets an IP blocked.
const overpassUserAgent = 'SafarSathi/0.1 (offline travel companion)';

const overpassEndpoint = 'https://overpass-api.de/api/interpreter';

/// A query larger than this is a bug, not a trip.
///
/// A 3 km corridor around a 100 km leg is roughly 700 km². Fifteen thousand
/// leaves generous headroom for a long leg and still refuses a request that
/// would ask a volunteer-funded server to scan half a state.
const maxQueryAreaSqKm = 15000.0;

class OverpassException implements Exception {
  final String message;
  const OverpassException(this.message);
  @override
  String toString() => message;
}

/// A place, before it reaches the database.
class PoiDraft {
  final String osmId;
  final String name;
  final String category;
  final LatLng location;

  /// Phone numbers found on the element, in the order the tags were read.
  final List<({String raw, String sourceTag})> phones;

  /// The raw tag set, kept so a future version can show what it knew without
  /// a second query.
  final Map<String, String> tags;

  const PoiDraft({
    required this.osmId,
    required this.name,
    required this.category,
    required this.location,
    this.phones = const [],
    this.tags = const {},
  });
}

/// The OSM tag keys that hold a phone number, in preference order.
const _phoneTags = ['phone', 'contact:phone', 'contact:mobile', 'mobile'];

class OverpassClient {
  /// Injected so every test runs without a network, and so the real client is
  /// the only thing that ever has to be trusted with an endpoint.
  final Future<String> Function(String query) fetch;

  OverpassClient({Future<String> Function(String query)? fetch})
    : fetch = fetch ?? _fetchOverHttp;

  static Future<String> _fetchOverHttp(String query) async {
    final response = await http.post(
      Uri.parse(overpassEndpoint),
      headers: const {
        'User-Agent': overpassUserAgent,
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: {'data': query},
    );

    if (response.statusCode == 429 || response.statusCode == 504) {
      // Overpass says these plainly: too many requests, or the query timed
      // out. Both mean back off, not retry immediately.
      throw const OverpassException(
        'OpenStreetMap is busy right now. Try again in a minute.',
      );
    }
    if (response.statusCode != 200) {
      throw OverpassException(
        'OpenStreetMap returned ${response.statusCode}.',
      );
    }
    return response.body;
  }

  /// Builds the Overpass QL for a box and a set of categories.
  ///
  /// `nwr` covers nodes, ways and relations in one pass, and `center` gives a
  /// single coordinate for a way rather than its whole outline — a petrol
  /// pump is a pin, not a polygon.
  static String buildQuery(
    BoundingBox box, {
    List<String> categoryKeys = defaultPoiCategoryKeys,
    int timeoutSeconds = 60,
  }) {
    final area = box.approxAreaSqKm;
    if (area > maxQueryAreaSqKm) {
      throw OverpassException(
        'That area is ${area.round()} square kilometres, which is too large '
        'to ask OpenStreetMap for. Shorten the leg or narrow the corridor.',
      );
    }

    final filters = <String>[];
    for (final key in categoryKeys) {
      final category = poiCategoryByKey(key);
      if (category == null) continue;
      for (final filter in category.filters) {
        final parts = filter.split('=');
        if (parts.length != 2) continue;
        filters.add('  nwr["${parts[0]}"="${parts[1]}"](${box.overpassString});');
      }
    }

    if (filters.isEmpty) {
      throw const OverpassException('No categories selected.');
    }

    return '[out:json][timeout:$timeoutSeconds];\n'
        '(\n${filters.join('\n')}\n);\n'
        'out center tags;';
  }

  /// Parses an Overpass JSON response.
  static List<PoiDraft> parse(String body) {
    final Map<String, dynamic> json;
    try {
      json = jsonDecode(body) as Map<String, dynamic>;
    } on Object {
      throw const OverpassException(
        'OpenStreetMap sent something this app could not read.',
      );
    }

    final elements = json['elements'];
    if (elements is! List) {
      throw const OverpassException(
        'OpenStreetMap sent a response with no places in it.',
      );
    }

    final out = <PoiDraft>[];
    for (final element in elements) {
      if (element is! Map) continue;

      final tags = <String, String>{};
      final rawTags = element['tags'];
      if (rawTags is Map) {
        for (final entry in rawTags.entries) {
          tags['${entry.key}'] = '${entry.value}';
        }
      }

      final category = categoryForTags(tags);
      if (category == null) continue;

      // A place with no name is not usable. "Unnamed fuel station, 14 km
      // ahead" tells you nothing you can act on or ask for.
      final name = tags['name'] ?? tags['name:en'];
      if (name == null || name.trim().isEmpty) continue;

      final location = _locationOf(element);
      if (location == null) continue;

      final phones = <({String raw, String sourceTag})>[];
      for (final key in _phoneTags) {
        final value = tags[key];
        if (value == null || value.trim().isEmpty) continue;
        // One element can list several numbers, semicolon-separated. That is
        // the OSM convention and it is common on hospitals.
        for (final part in value.split(';')) {
          if (part.trim().isEmpty) continue;
          phones.add((raw: part.trim(), sourceTag: key));
        }
      }

      out.add(
        PoiDraft(
          osmId: '${element['type']}/${element['id']}',
          name: name.trim(),
          category: category,
          location: location,
          phones: phones,
          tags: tags,
        ),
      );
    }
    return out;
  }

  /// A node carries its own coordinates; a way or relation carries a `center`
  /// because the query asked for `out center`.
  static LatLng? _locationOf(Map element) {
    final lat = element['lat'];
    final lon = element['lon'];
    if (lat is num && lon is num) {
      return LatLng(lat.toDouble(), lon.toDouble());
    }
    final centre = element['center'];
    if (centre is Map) {
      final cLat = centre['lat'];
      final cLon = centre['lon'];
      if (cLat is num && cLon is num) {
        return LatLng(cLat.toDouble(), cLon.toDouble());
      }
    }
    return null;
  }

  /// Queries and parses in one go.
  Future<List<PoiDraft>> search(
    BoundingBox box, {
    List<String> categoryKeys = defaultPoiCategoryKeys,
  }) async {
    final query = buildQuery(box, categoryKeys: categoryKeys);
    return parse(await fetch(query));
  }
}
