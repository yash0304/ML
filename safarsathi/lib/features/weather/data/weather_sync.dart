// lib/features/weather/data/weather_sync.dart
//
// Storing and reading the snapshot — issue #26.

import 'package:drift/drift.dart';

import '../../../core/database/app_database.dart';
import '../../discovery/data/geo.dart';
import 'weather_client.dart';

/// A stored snapshot with its age worked out.
class StopWeather {
  final int stopId;
  final String stopName;
  final List<WeatherSnapshot> days;

  /// The newest `cachedAt` across the stored days. Null when nothing is
  /// stored for this stop.
  final DateTime? cachedAt;

  const StopWeather({
    required this.stopId,
    required this.stopName,
    required this.days,
    this.cachedAt,
  });

  bool get isEmpty => days.isEmpty;

  Staleness? staleness({DateTime? now}) =>
      cachedAt == null ? null : stalenessOf(cachedAt!, now: now);

  String? ageDescription({DateTime? now}) =>
      cachedAt == null ? null : describeAge(cachedAt!, now: now);
}

class WeatherSync {
  final AppDatabase db;
  final WeatherClient client;

  WeatherSync({required this.db, WeatherClient? client})
    : client = client ?? WeatherClient();

  /// Fetches and stores the forecast for one stop, over the days it is
  /// occupied.
  ///
  /// Replaces rather than accumulating: a re-sync is the user asking for what
  /// is expected NOW, and the unique key on (stop, date) means a merge would
  /// fail anyway.
  Future<int> syncStop(Stop stop, {DateTime? now}) async {
    if (stop.lat == null || stop.lon == null) {
      throw const WeatherException(
        'This stop has no coordinates yet, so there is nowhere to ask about.',
      );
    }

    final from = stop.arrivalDate ?? (now ?? DateTime.now());
    final to = stop.departureDate ?? from.add(Duration(days: stop.nights));

    final days = await client.forecast(
      LatLng(stop.lat!, stop.lon!),
      from: from,
      to: to.isBefore(from) ? from : to,
    );

    // Stamped once for the whole fetch, so every day of one snapshot reports
    // the same age. Per-row timestamps would drift by milliseconds and read
    // as though some days were fresher than others.
    final cachedAt = now ?? DateTime.now();

    await db.transaction(() async {
      await (db.delete(
        db.weatherSnapshots,
      )..where((w) => w.stopId.equals(stop.id))).go();

      for (final day in days) {
        await db
            .into(db.weatherSnapshots)
            .insert(
              WeatherSnapshotsCompanion.insert(
                stopId: stop.id,
                forDate: day.date,
                condition: day.condition,
                cachedAt: cachedAt,
                tempMinC: Value(day.tempMinC),
                tempMaxC: Value(day.tempMaxC),
                rainMm: Value(day.rainMm),
              ),
            );
      }
    });

    return days.length;
  }

  /// Every stop's weather for a trip, in itinerary order.
  Stream<List<StopWeather>> watchTripWeather(int tripId) {
    final tick = db
        .customSelect(
          'SELECT 1',
          readsFrom: {db.stops, db.weatherSnapshots},
        )
        .watch();

    return tick.asyncMap((_) async {
      final stops =
          await (db.select(db.stops)
                ..where((s) => s.tripId.equals(tripId))
                ..orderBy([(s) => OrderingTerm(expression: s.sequenceOrder)]))
              .get();

      final out = <StopWeather>[];
      for (final stop in stops) {
        final days =
            await (db.select(db.weatherSnapshots)
                  ..where((w) => w.stopId.equals(stop.id))
                  ..orderBy([(w) => OrderingTerm(expression: w.forDate)]))
                .get();

        DateTime? newest;
        for (final d in days) {
          if (newest == null || d.cachedAt.isAfter(newest)) newest = d.cachedAt;
        }

        out.add(
          StopWeather(
            stopId: stop.id,
            stopName: stop.name,
            days: days,
            cachedAt: newest,
          ),
        );
      }
      return out;
    });
  }
}
