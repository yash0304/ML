// lib/features/discovery/data/osrm_client.dart
//
// The route between two stops — issue #22.
//
// THIS IS THE ONLY LIVE ROUTING CALL IN THE APP, and it happens once per leg
// at setup, on WiFi. There is no re-route, no traffic, no turn-by-turn. What
// gets stored is a line on a map and a distance, and both are then frozen.
//
// Rome2Rio was one of the five apps this project set out to absorb, and it is
// the one that could not be: it depends on live operator schedule databases.
// What survives is this — the shape of the road, downloaded before you go.

import 'dart:convert';

import 'package:http/http.dart' as http;

import 'geo.dart';
import 'polyline.dart';

/// The public demo server. It has no SLA and asks for light use, which is
/// exactly what one call per leg at setup is.
const osrmEndpoint = 'https://router.project-osrm.org';

const osrmUserAgent = 'SafarSathi/0.1 (offline travel companion)';

class OsrmException implements Exception {
  final String message;
  const OsrmException(this.message);
  @override
  String toString() => message;
}

class RouteResult {
  /// Encoded at precision 5, which is what is stored and what the map draws.
  final String encodedPolyline;
  final List<LatLng> points;
  final double distanceKm;
  final Duration duration;

  const RouteResult({
    required this.encodedPolyline,
    required this.points,
    required this.distanceKm,
    required this.duration,
  });
}

class OsrmClient {
  /// Injected, so every test runs without a network.
  final Future<String> Function(Uri url) fetch;

  OsrmClient({Future<String> Function(Uri url)? fetch})
    : fetch = fetch ?? _fetchOverHttp;

  static Future<String> _fetchOverHttp(Uri url) async {
    final response = await http.get(
      url,
      headers: const {'User-Agent': osrmUserAgent},
    );
    if (response.statusCode != 200) {
      throw OsrmException('The routing service returned ${response.statusCode}.');
    }
    return response.body;
  }

  /// Builds the request URL.
  ///
  /// Coordinates go in LON,LAT order — OSRM follows GeoJSON, not the lat-first
  /// convention everything else in this app uses. Swapping them does not
  /// error; it silently routes somewhere else entirely.
  static Uri buildUrl(LatLng from, LatLng to, {String profile = 'driving'}) {
    final coords =
        '${from.lon.toStringAsFixed(6)},${from.lat.toStringAsFixed(6)};'
        '${to.lon.toStringAsFixed(6)},${to.lat.toStringAsFixed(6)}';
    return Uri.parse(
      '$osrmEndpoint/route/v1/$profile/$coords'
      '?overview=full&geometries=polyline&alternatives=false&steps=false',
    );
  }

  /// Parses an OSRM `route` response.
  ///
  /// `geometries=polyline` means PRECISION 5. Asking for `polyline6` would
  /// return precision 6, and decoding one as the other scales the entire route
  /// by ten — a failure that looks like a broken map rather than a wrong
  /// number, which is why the request pins the format rather than guessing.
  static RouteResult parse(String body) {
    final Map<String, dynamic> json;
    try {
      json = jsonDecode(body) as Map<String, dynamic>;
    } on Object {
      throw const OsrmException(
        'The routing service sent something this app could not read.',
      );
    }

    final code = json['code'];
    if (code == 'NoRoute') {
      throw const OsrmException(
        'No road route between those two places. Check the coordinates, or '
        'leave this leg without a line.',
      );
    }
    if (code != 'Ok') {
      throw OsrmException('The routing service said: $code.');
    }

    final routes = json['routes'];
    if (routes is! List || routes.isEmpty) {
      throw const OsrmException('The routing service returned no route.');
    }

    final route = routes.first;
    if (route is! Map) {
      throw const OsrmException('The routing service returned no route.');
    }

    final geometry = route['geometry'];
    if (geometry is! String || geometry.isEmpty) {
      throw const OsrmException('That route came back with no line to draw.');
    }

    final distance = route['distance'];
    final duration = route['duration'];

    return RouteResult(
      encodedPolyline: geometry,
      points: Polyline.decode(geometry),
      distanceKm: distance is num ? distance.toDouble() / 1000 : 0,
      duration: Duration(
        seconds: duration is num ? duration.round() : 0,
      ),
    );
  }

  Future<RouteResult> route(
    LatLng from,
    LatLng to, {
    String profile = 'driving',
  }) async => parse(await fetch(buildUrl(from, to, profile: profile)));
}
