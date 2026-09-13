// test/geocoder_weather_test.dart — stop coordinates and issue #26.

import 'package:drift/drift.dart' hide isNull, isNotNull;
import 'package:drift/native.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:safarsathi/core/database/app_database.dart';
import 'package:safarsathi/features/discovery/data/geo.dart';
import 'package:safarsathi/features/discovery/data/geocoder.dart';
import 'package:safarsathi/features/trips/data/trip_editor.dart';
import 'package:safarsathi/features/weather/data/weather_client.dart';
import 'package:safarsathi/features/weather/data/weather_sync.dart';

const _nominatimBody = '''
[
  {"lat":"25.5788","lon":"91.8933","type":"city",
   "display_name":"Shillong, East Khasi Hills, Meghalaya, India"},
  {"lat":"12.9716","lon":"77.5946","type":"village",
   "display_name":"Shillong, Karnataka, India"}
]
''';

const _forecastBody = '''
{"latitude":25.5,"longitude":91.9,
 "daily":{
   "time":["2026-10-01","2026-10-02","2026-10-03"],
   "weather_code":[61,3,95],
   "temperature_2m_max":[24.1,22.8,21.0],
   "temperature_2m_min":[17.2,16.9,16.1],
   "precipitation_sum":[12.4,0.0,38.6]
 }}
''';

void main() {
  group('geocoding', () {
    test('the URL asks for what the parser expects', () {
      final url = Geocoder.buildUrl('Shillong', countryCode: 'IN');
      expect(url.queryParameters['q'], 'Shillong');
      expect(url.queryParameters['format'], 'jsonv2');
      // Narrowing by country removes most same-name confusion before the user
      // has to read anything.
      expect(url.queryParameters['countrycodes'], 'in');
    });

    test('results carry a display name so two Shillongs can be told apart', () {
      final results = Geocoder.parse(_nominatimBody);

      expect(results.length, 2);
      expect(results.first.shortName, 'Shillong');
      expect(results.first.displayName, contains('Meghalaya'));
      expect(results.last.displayName, contains('Karnataka'));
      expect(results.first.location.lat, closeTo(25.5788, 0.0001));
    });

    test('a result missing coordinates is dropped', () {
      final results = Geocoder.parse(
        '[{"display_name":"Nowhere"},{"lat":"1","lon":"2",'
        '"display_name":"Somewhere"}]',
      );
      expect(results.single.displayName, 'Somewhere');
    });

    test('malformed JSON fails with a sentence', () {
      expect(
        () => Geocoder.parse('<html>502</html>'),
        throwsA(isA<GeocodeException>()),
      );
    });

    test('an empty query does not call out at all', () async {
      var called = false;
      final geocoder = Geocoder(
        fetch: (_) async {
          called = true;
          return '[]';
        },
      );

      expect(await geocoder.search('   '), isEmpty);
      expect(called, isFalse);
    });

    test('THE CLIENT RATE-LIMITS ITSELF TO ONE PER SECOND', () async {
      // Nominatim's policy, enforced here rather than hoped for. Breaking it
      // gets an IP banned, and it would be deserved.
      var now = DateTime(2026, 9, 13, 12, 0, 0);
      final waits = <Duration>[];

      final geocoder = Geocoder(
        fetch: (_) async => _nominatimBody,
        clock: () => now,
        sleep: (d) async {
          waits.add(d);
          now = now.add(d);
        },
      );

      await geocoder.search('Shillong');
      expect(waits, isEmpty);

      // A second lookup 200 ms later has to wait out the remaining 800.
      now = now.add(const Duration(milliseconds: 200));
      await geocoder.search('Dawki');

      expect(waits.single.inMilliseconds, 800);
    });

    test('a lookup a second later does not wait', () async {
      var now = DateTime(2026, 9, 13, 12, 0, 0);
      final waits = <Duration>[];
      final geocoder = Geocoder(
        fetch: (_) async => _nominatimBody,
        clock: () => now,
        sleep: (d) async => waits.add(d),
      );

      await geocoder.search('Shillong');
      now = now.add(const Duration(seconds: 2));
      await geocoder.search('Dawki');

      expect(waits, isEmpty);
    });
  });

  group('typed coordinates', () {
    test('a comma pair parses', () {
      final p = parseLatLon('25.5788, 91.8933')!;
      expect(p.lat, closeTo(25.5788, 0.0001));
      expect(p.lon, closeTo(91.8933, 0.0001));
    });

    test('a space-separated pair parses too', () {
      expect(parseLatLon('25.5788 91.8933'), isNotNull);
    });

    test('a negative pair parses', () {
      final p = parseLatLon('-33.86,151.2')!;
      expect(p.lat, closeTo(-33.86, 0.001));
    });

    test('out of range is refused, not clamped', () {
      // Clamping would put a stop somewhere real and wrong.
      expect(parseLatLon('91, 0'), isNull);
      expect(parseLatLon('0, 181'), isNull);
    });

    test('nonsense is null', () {
      expect(parseLatLon(''), isNull);
      expect(parseLatLon('Shillong'), isNull);
      expect(parseLatLon('25.5'), isNull);
      expect(parseLatLon('1,2,3'), isNull);
    });

    test('round-trips through the display form', () {
      const p = LatLng(25.5788, 91.8933);
      expect(parseLatLon(formatLatLon(p))!.lat, closeTo(p.lat, 0.00001));
    });
  });

  group('forecast parsing', () {
    test('a date range parses to one entry per day', () {
      final days = WeatherClient.parse(_forecastBody);

      expect(days.length, 3);
      expect(days.first.date, DateTime(2026, 10, 1));
      expect(days.first.tempMaxC, 24.1);
      expect(days.first.rainMm, 12.4);
    });

    test('WMO codes become words that change what you pack', () {
      final days = WeatherClient.parse(_forecastBody);
      expect(days[0].condition, 'Light rain');
      expect(days[1].condition, 'Overcast');
      expect(days[2].condition, 'Thunderstorm');
    });

    test('an unknown code says so rather than guessing', () {
      expect(conditionForWmoCode(999), 'Unknown');
    });

    test('a missing column leaves nulls rather than failing', () {
      final days = WeatherClient.parse(
        '{"daily":{"time":["2026-10-01"],"weather_code":[0]}}',
      );
      expect(days.single.tempMaxC, isNull);
      expect(days.single.condition, 'Clear');
    });

    test('an error response is reported with its reason', () {
      expect(
        () => WeatherClient.parse(
          '{"error":true,"reason":"Invalid date range"}',
        ),
        throwsA(
          isA<WeatherException>().having(
            (e) => e.message,
            'message',
            contains('Invalid date range'),
          ),
        ),
      );
    });

    test('a response with no days is refused', () {
      expect(
        () => WeatherClient.parse('{"daily":{"time":[]}}'),
        throwsA(isA<WeatherException>()),
      );
    });

    test('the URL asks for the four fields the parser reads', () {
      final url = WeatherClient.buildUrl(
        const LatLng(25.5788, 91.8933),
        from: DateTime(2026, 10, 1),
        to: DateTime(2026, 10, 5),
      );
      expect(url.queryParameters['daily'], contains('weather_code'));
      expect(url.queryParameters['daily'], contains('precipitation_sum'));
      expect(url.queryParameters['start_date'], '2026-10-01');
      expect(url.queryParameters['end_date'], '2026-10-05');
    });
  });

  group('staleness', () {
    final taken = DateTime(2026, 10, 1, 9);

    test('A FORECAST MUST NEVER LOOK CURRENT', () {
      // Three bands, because a five-day-old forecast shown as today's weather
      // is worse than no forecast: somebody packs on it.
      expect(stalenessOf(taken, now: taken.add(const Duration(hours: 6))),
          Staleness.fresh);
      expect(stalenessOf(taken, now: taken.add(const Duration(days: 3))),
          Staleness.ageing);
      expect(stalenessOf(taken, now: taken.add(const Duration(days: 9))),
          Staleness.stale);
    });

    test('the age reads in words, not as a timestamp', () {
      expect(describeAge(taken, now: taken.add(const Duration(minutes: 5))),
          'taken just now');
      expect(describeAge(taken, now: taken.add(const Duration(hours: 3))),
          'taken 3 hours ago');
      expect(describeAge(taken, now: taken.add(const Duration(hours: 1))),
          'taken 1 hour ago');
      expect(describeAge(taken, now: taken.add(const Duration(days: 5))),
          'taken 5 days ago');
      expect(describeAge(taken, now: taken.add(const Duration(days: 1))),
          'taken 1 day ago');
    });
  });

  group('storing a snapshot', () {
    late AppDatabase db;
    late TripEditor editor;
    late int tripId;
    late Stop stop;

    setUp(() async {
      db = AppDatabase(NativeDatabase.memory());
      editor = TripEditor(db);
      tripId = await editor.createTrip(name: 'Meghalaya');
      final id = await editor.addStop(
        tripId,
        StopDraft(
          name: 'Shillong',
          arrivalDate: DateTime(2026, 10, 1),
          departureDate: DateTime(2026, 10, 3),
        ),
      );
      await (db.update(db.stops)..where((s) => s.id.equals(id))).write(
        const StopsCompanion(lat: Value(25.5788), lon: Value(91.8933)),
      );
      stop = await (db.select(
        db.stops,
      )..where((s) => s.id.equals(id))).getSingle();
    });

    tearDown(() => db.close());

    WeatherSync syncWith([String? body]) => WeatherSync(
      db: db,
      client: WeatherClient(fetch: (_) async => body ?? _forecastBody),
    );

    test('every stored day carries the same cachedAt', () async {
      final taken = DateTime(2026, 9, 25, 10);
      await syncWith().syncStop(stop, now: taken);

      final rows = await db.select(db.weatherSnapshots).get();
      expect(rows.length, 3);
      expect(rows.every((r) => r.cachedAt == taken), isTrue);
    });

    test('re-syncing replaces rather than duplicating', () async {
      await syncWith().syncStop(stop);
      await syncWith().syncStop(stop);
      expect((await db.select(db.weatherSnapshots).get()).length, 3);
    });

    test('a stop with no coordinates says so', () async {
      await (db.update(db.stops)..where((s) => s.id.equals(stop.id))).write(
        const StopsCompanion(lat: Value(null), lon: Value(null)),
      );
      final bare = await (db.select(
        db.stops,
      )..where((s) => s.id.equals(stop.id))).getSingle();

      await expectLater(
        syncWith().syncStop(bare),
        throwsA(
          isA<WeatherException>().having(
            (e) => e.message,
            'message',
            contains('no coordinates'),
          ),
        ),
      );
    });

    test('the trip view reports the age of each stop', () async {
      final taken = DateTime(2026, 9, 25, 10);
      await syncWith().syncStop(stop, now: taken);

      final view = await syncWith().watchTripWeather(tripId).first;
      expect(view.single.stopName, 'Shillong');
      expect(view.single.days.length, 3);
      expect(view.single.cachedAt, taken);
      expect(
        view.single.staleness(now: taken.add(const Duration(days: 9))),
        Staleness.stale,
      );
      expect(
        view.single.ageDescription(now: taken.add(const Duration(days: 3))),
        'taken 3 days ago',
      );
    });

    test('a stop with nothing stored reports no age at all', () async {
      final view = await syncWith().watchTripWeather(tripId).first;
      expect(view.single.isEmpty, isTrue);
      expect(view.single.cachedAt, isNull);
      expect(view.single.staleness(), isNull);
    });
  });
}
