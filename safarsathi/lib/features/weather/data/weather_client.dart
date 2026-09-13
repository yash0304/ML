// lib/features/weather/data/weather_client.dart
//
// The weather snapshot — issue #26.
//
// Borrowed from Windy's offline mode: download the forecast for a region
// BEFORE losing connection, then it is static. Not live weather. A frozen
// picture of what was expected, with the date it was taken stamped on it.
//
// Open-Meteo: free, no API key, no account, non-commercial use covered. That
// last part matters beyond cost — it is the one network source in this app
// that blocks on nothing anybody has to sign up for.

import 'dart:convert';

import 'package:http/http.dart' as http;

import '../../discovery/data/geo.dart';

const openMeteoEndpoint = 'https://api.open-meteo.com/v1/forecast';

class WeatherException implements Exception {
  final String message;
  const WeatherException(this.message);
  @override
  String toString() => message;
}

/// One day, as fetched.
class DailyForecast {
  final DateTime date;
  final String condition;
  final double? tempMinC;
  final double? tempMaxC;
  final double? rainMm;

  const DailyForecast({
    required this.date,
    required this.condition,
    this.tempMinC,
    this.tempMaxC,
    this.rainMm,
  });
}

/// How old a snapshot is, in bands the UI colours by.
enum Staleness { fresh, ageing, stale }

/// A FORECAST MUST NEVER LOOK CURRENT. `cachedAt` is non-nullable in the
/// schema for exactly this reason; this turns it into something the screen can
/// say out loud.
Staleness stalenessOf(DateTime cachedAt, {DateTime? now}) {
  final age = (now ?? DateTime.now()).difference(cachedAt);
  if (age < const Duration(days: 2)) return Staleness.fresh;
  if (age < const Duration(days: 7)) return Staleness.ageing;
  return Staleness.stale;
}

/// The age in words, which is what actually appears beside a temperature.
///
/// "Taken 5 days ago" is a fact a person can weigh. A timestamp is not, and a
/// bare temperature is a lie by omission — somebody packs on it.
String describeAge(DateTime cachedAt, {DateTime? now}) {
  final age = (now ?? DateTime.now()).difference(cachedAt);
  if (age.inMinutes < 60) return 'taken just now';
  if (age.inHours < 24) {
    final h = age.inHours;
    return 'taken $h ${h == 1 ? 'hour' : 'hours'} ago';
  }
  final d = age.inDays;
  return 'taken $d ${d == 1 ? 'day' : 'days'} ago';
}

/// WMO weather codes, which is what Open-Meteo returns, in plain words.
///
/// The full table has 100 entries and most never occur where anyone travels.
/// These are the bands that change what you pack.
String conditionForWmoCode(int code) => switch (code) {
  0 => 'Clear',
  1 || 2 => 'Mostly clear',
  3 => 'Overcast',
  45 || 48 => 'Fog',
  51 || 53 || 55 => 'Drizzle',
  56 || 57 => 'Freezing drizzle',
  61 => 'Light rain',
  63 => 'Rain',
  65 => 'Heavy rain',
  66 || 67 => 'Freezing rain',
  71 || 73 || 75 || 77 => 'Snow',
  80 || 81 => 'Rain showers',
  82 => 'Violent rain showers',
  85 || 86 => 'Snow showers',
  95 => 'Thunderstorm',
  96 || 99 => 'Thunderstorm with hail',
  _ => 'Unknown',
};

class WeatherClient {
  final Future<String> Function(Uri url) fetch;

  WeatherClient({Future<String> Function(Uri url)? fetch})
    : fetch = fetch ?? _fetchOverHttp;

  static Future<String> _fetchOverHttp(Uri url) async {
    final response = await http.get(url);
    if (response.statusCode != 200) {
      throw WeatherException(
        'The forecast service returned ${response.statusCode}.',
      );
    }
    return response.body;
  }

  static Uri buildUrl(LatLng at, {required DateTime from, required DateTime to}) {
    String day(DateTime d) =>
        '${d.year}-${d.month.toString().padLeft(2, '0')}-'
        '${d.day.toString().padLeft(2, '0')}';

    return Uri.parse(openMeteoEndpoint).replace(
      queryParameters: {
        'latitude': at.lat.toStringAsFixed(4),
        'longitude': at.lon.toStringAsFixed(4),
        'daily':
            'weather_code,temperature_2m_max,temperature_2m_min,'
            'precipitation_sum',
        'timezone': 'auto',
        'start_date': day(from),
        'end_date': day(to),
      },
    );
  }

  static List<DailyForecast> parse(String body) {
    final Map<String, dynamic> json;
    try {
      json = jsonDecode(body) as Map<String, dynamic>;
    } on Object {
      throw const WeatherException(
        'The forecast service sent something this app could not read.',
      );
    }

    if (json['error'] == true) {
      throw WeatherException('The forecast service said: ${json['reason']}.');
    }

    final daily = json['daily'];
    if (daily is! Map) {
      throw const WeatherException('That forecast came back with no days in it.');
    }

    final dates = daily['time'];
    if (dates is! List || dates.isEmpty) {
      throw const WeatherException('That forecast came back with no days in it.');
    }

    List<dynamic> column(String key) {
      final value = daily[key];
      return value is List ? value : const [];
    }

    final codes = column('weather_code');
    final maxima = column('temperature_2m_max');
    final minima = column('temperature_2m_min');
    final rain = column('precipitation_sum');

    double? at(List<dynamic> list, int i) {
      if (i >= list.length) return null;
      final value = list[i];
      return value is num ? value.toDouble() : null;
    }

    final out = <DailyForecast>[];
    for (var i = 0; i < dates.length; i++) {
      final date = DateTime.tryParse('${dates[i]}');
      if (date == null) continue;

      final code = i < codes.length && codes[i] is num
          ? (codes[i] as num).round()
          : -1;

      out.add(
        DailyForecast(
          date: DateTime(date.year, date.month, date.day),
          condition: conditionForWmoCode(code),
          tempMaxC: at(maxima, i),
          tempMinC: at(minima, i),
          rainMm: at(rain, i),
        ),
      );
    }
    return out;
  }

  Future<List<DailyForecast>> forecast(
    LatLng at, {
    required DateTime from,
    required DateTime to,
  }) async => parse(await fetch(buildUrl(at, from: from, to: to)));
}
