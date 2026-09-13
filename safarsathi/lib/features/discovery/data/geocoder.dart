// lib/features/discovery/data/geocoder.dart
//
// Turning a place name into coordinates, so the corridor has something to
// route between.
//
// NOMINATIM'S USAGE POLICY IS STRICT AND THIS CLIENT OBEYS IT:
//   - a real User-Agent identifying the app; an anonymous client gets blocked,
//   - at most one request per second, enforced here rather than hoped for,
//   - no bulk or automated use. One lookup, when a person taps a button, for
//     a place they typed.
// Breaking any of these gets an IP banned, and it would be deserved.

import 'dart:convert';

import 'package:http/http.dart' as http;

import 'geo.dart';

const nominatimEndpoint = 'https://nominatim.openstreetmap.org/search';
const nominatimUserAgent = 'SafarSathi/0.1 (offline travel companion)';

/// The policy floor. Held to by the client itself.
const nominatimMinInterval = Duration(seconds: 1);

class GeocodeException implements Exception {
  final String message;
  const GeocodeException(this.message);
  @override
  String toString() => message;
}

/// One candidate. Never saved without the user picking it.
class GeocodeResult {
  final String displayName;
  final LatLng location;

  /// OSM's own type words: `city`, `village`, `hamlet`, `administrative`.
  /// Shown so two places with the same name can be told apart.
  final String? kind;

  const GeocodeResult({
    required this.displayName,
    required this.location,
    this.kind,
  });

  /// The first comma-separated part, which is nearly always the place itself.
  String get shortName => displayName.split(',').first.trim();
}

class Geocoder {
  final Future<String> Function(Uri url) fetch;

  /// Injected so a test can move time without waiting a real second.
  final DateTime Function() clock;
  final Future<void> Function(Duration) sleep;

  DateTime? _lastRequest;

  Geocoder({
    Future<String> Function(Uri url)? fetch,
    DateTime Function()? clock,
    Future<void> Function(Duration)? sleep,
  }) : fetch = fetch ?? _fetchOverHttp,
       clock = clock ?? DateTime.now,
       sleep = sleep ?? Future.delayed;

  static Future<String> _fetchOverHttp(Uri url) async {
    final response = await http.get(
      url,
      headers: const {'User-Agent': nominatimUserAgent},
    );
    if (response.statusCode == 429) {
      throw const GeocodeException(
        'Too many lookups too quickly. Wait a moment and try again.',
      );
    }
    if (response.statusCode != 200) {
      throw GeocodeException('The lookup service returned ${response.statusCode}.');
    }
    return response.body;
  }

  static Uri buildUrl(String query, {String? countryCode, int limit = 5}) =>
      Uri.parse(nominatimEndpoint).replace(
        queryParameters: {
          'q': query,
          'format': 'jsonv2',
          'limit': '$limit',
          'addressdetails': '0',
          // Narrowing to the stop's own country removes most of the
          // same-name confusion before the user has to read anything.
          if (countryCode != null) 'countrycodes': countryCode.toLowerCase(),
        },
      );

  static List<GeocodeResult> parse(String body) {
    final dynamic json;
    try {
      json = jsonDecode(body);
    } on Object {
      throw const GeocodeException(
        'The lookup service sent something this app could not read.',
      );
    }
    if (json is! List) {
      throw const GeocodeException('The lookup service sent no results.');
    }

    final out = <GeocodeResult>[];
    for (final entry in json) {
      if (entry is! Map) continue;
      final lat = double.tryParse('${entry['lat']}');
      final lon = double.tryParse('${entry['lon']}');
      final name = entry['display_name'];
      if (lat == null || lon == null || name is! String) continue;

      out.add(
        GeocodeResult(
          displayName: name,
          location: LatLng(lat, lon),
          kind: entry['type'] as String?,
        ),
      );
    }
    return out;
  }

  /// Looks up [query], waiting out the policy interval first if need be.
  ///
  /// NOTHING IS SAVED HERE. The caller shows the candidates and the user picks
  /// one. Silently accepting the first result is how a trip ends up routed to
  /// a village in Karnataka with the same name as the town you meant.
  Future<List<GeocodeResult>> search(
    String query, {
    String? countryCode,
  }) async {
    if (query.trim().isEmpty) return const [];

    final last = _lastRequest;
    if (last != null) {
      final since = clock().difference(last);
      if (since < nominatimMinInterval) {
        await sleep(nominatimMinInterval - since);
      }
    }
    _lastRequest = clock();

    return parse(
      await fetch(buildUrl(query.trim(), countryCode: countryCode)),
    );
  }
}

/// Parses a typed coordinate pair.
///
/// The manual way in, for when the lookup is wrong or the place has no name a
/// geocoder knows — which describes a good share of the homestays this app
/// exists for.
LatLng? parseLatLon(String input) {
  final text = input.trim();
  if (text.isEmpty) return null;

  final parts = text.split(RegExp(r'[,\s]+'));
  if (parts.length != 2) return null;

  final lat = double.tryParse(parts[0]);
  final lon = double.tryParse(parts[1]);
  if (lat == null || lon == null) return null;
  if (lat < -90 || lat > 90 || lon < -180 || lon > 180) return null;

  return LatLng(lat, lon);
}

String formatLatLon(LatLng p) =>
    '${p.lat.toStringAsFixed(5)}, ${p.lon.toStringAsFixed(5)}';
