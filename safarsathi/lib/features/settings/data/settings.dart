// lib/features/settings/data/settings.dart
//
// The handful of things worth making adjustable — issue #35.
//
// Stored in the database rather than shared preferences, so there is still
// exactly one place this app keeps state and exactly one thing to back up.
//
// EVERY OPTION IS A THING THAT CAN BE WRONG. There are three.

import 'package:drift/drift.dart';
import 'package:flutter/material.dart';

import '../../../core/database/app_database.dart';

class SettingKeys {
  SettingKeys._();

  /// system | day | lamp. A phone in a pocket does not know it is night in a
  /// valley, which is why this overrides the system rather than following it.
  static const themeMode = 'themeMode';

  /// Default corridor half-width in kilometres, for new legs.
  static const corridorKm = 'corridorKm';
}

ThemeMode themeModeFromName(String? name) => switch (name) {
  'day' => ThemeMode.light,
  'lamp' => ThemeMode.dark,
  _ => ThemeMode.system,
};

String themeModeName(ThemeMode mode) => switch (mode) {
  ThemeMode.light => 'day',
  ThemeMode.dark => 'lamp',
  ThemeMode.system => 'system',
};

class SettingsRepository {
  final AppDatabase db;
  const SettingsRepository(this.db);

  Future<String?> read(String key) async {
    final row = await (db.select(
      db.appSettings,
    )..where((s) => s.key.equals(key))).getSingleOrNull();
    return row?.value;
  }

  Stream<String?> watch(String key) => (db.select(
    db.appSettings,
  )..where((s) => s.key.equals(key))).watchSingleOrNull().map((r) => r?.value);

  Future<void> write(String key, String value) => db
      .into(db.appSettings)
      .insertOnConflictUpdate(
        AppSettingsCompanion.insert(key: key, value: value),
      );

  Stream<ThemeMode> watchThemeMode() =>
      watch(SettingKeys.themeMode).map(themeModeFromName);

  Future<void> setThemeMode(ThemeMode mode) =>
      write(SettingKeys.themeMode, themeModeName(mode));

  Stream<double> watchCorridorKm() => watch(
    SettingKeys.corridorKm,
  ).map((v) => double.tryParse(v ?? '') ?? 3.0);

  Future<void> setCorridorKm(double km) =>
      write(SettingKeys.corridorKm, km.toString());
}

/// What one trip is holding, for the cache screen.
class CacheSummary {
  final int tripId;
  final String tripName;
  final int poiCount;
  final int routedLegCount;
  final int legCount;
  final int weatherDayCount;
  final DateTime? lastSyncedAt;

  const CacheSummary({
    required this.tripId,
    required this.tripName,
    required this.poiCount,
    required this.routedLegCount,
    required this.legCount,
    required this.weatherDayCount,
    this.lastSyncedAt,
  });

  bool get isEmpty =>
      poiCount == 0 && routedLegCount == 0 && weatherDayCount == 0;
}

Stream<List<CacheSummary>> watchCacheSummaries(AppDatabase db) {
  final tick = db
      .customSelect(
        'SELECT 1',
        readsFrom: {db.trips, db.legs, db.pois, db.weatherSnapshots, db.stops},
      )
      .watch();

  return tick.asyncMap((_) async {
    final trips = await db.select(db.trips).get();
    final out = <CacheSummary>[];

    for (final trip in trips) {
      final legs = await (db.select(
        db.legs,
      )..where((l) => l.tripId.equals(trip.id))).get();
      final pois = await (db.select(
        db.pois,
      )..where((p) => p.tripId.equals(trip.id))).get();

      final stops = await (db.select(
        db.stops,
      )..where((s) => s.tripId.equals(trip.id))).get();
      final stopIds = [for (final s in stops) s.id];
      final weather = stopIds.isEmpty
          ? <WeatherSnapshot>[]
          : await (db.select(
              db.weatherSnapshots,
            )..where((w) => w.stopId.isIn(stopIds))).get();

      DateTime? newest;
      for (final leg in legs) {
        final at = leg.lastSyncedAt;
        if (at != null && (newest == null || at.isAfter(newest))) newest = at;
      }

      out.add(
        CacheSummary(
          tripId: trip.id,
          tripName: trip.name,
          poiCount: pois.length,
          routedLegCount: legs.where((l) => l.routePolyline != null).length,
          legCount: legs.length,
          weatherDayCount: weather.length,
          lastSyncedAt: newest,
        ),
      );
    }
    return out;
  });
}

/// Drops everything downloaded for a trip, leaving everything typed.
///
/// The distinction is the whole point: contacts, expenses, the itinerary and
/// the checklist are the user's work. Routes, places and forecasts came off
/// the network and can come off it again.
Future<void> clearTripCache(AppDatabase db, int tripId) async {
  await db.transaction(() async {
    await (db.delete(db.pois)..where((p) => p.tripId.equals(tripId))).go();

    await (db.update(db.legs)..where((l) => l.tripId.equals(tripId))).write(
      const LegsCompanion(
        routePolyline: Value(null),
        distanceKm: Value(null),
        lastSyncedAt: Value(null),
      ),
    );

    final stops = await (db.select(
      db.stops,
    )..where((s) => s.tripId.equals(tripId))).get();
    final stopIds = [for (final s in stops) s.id];
    if (stopIds.isNotEmpty) {
      await (db.delete(
        db.weatherSnapshots,
      )..where((w) => w.stopId.isIn(stopIds))).go();
    }
  });
}

/// One row of the call history.
class CallLogEntry {
  final int id;
  final String action;
  final DateTime occurredAt;
  final String contactName;
  final String phoneRaw;

  const CallLogEntry({
    required this.id,
    required this.action,
    required this.occurredAt,
    required this.contactName,
    required this.phoneRaw,
  });
}

/// What the diary already logs, read back — newest first.
///
/// A COPY COUNTS AS AN ACTION. The dial happens in the Android dialer after a
/// paste, so the app never sees the call itself; recording only "calls" would
/// show an empty history to someone who used the app all day.
Stream<List<CallLogEntry>> watchCallHistory(AppDatabase db, int tripId) {
  final query = db.select(db.callLogs).join([
    innerJoin(db.contacts, db.contacts.id.equalsExp(db.callLogs.contactId)),
  ])
    ..where(db.callLogs.tripId.equals(tripId))
    ..orderBy([
      OrderingTerm(
        expression: db.callLogs.occurredAt,
        mode: OrderingMode.desc,
      ),
    ]);

  return query.watch().map(
    (rows) => [
      for (final row in rows)
        CallLogEntry(
          id: row.readTable(db.callLogs).id,
          action: row.readTable(db.callLogs).action,
          occurredAt: row.readTable(db.callLogs).occurredAt,
          contactName: row.readTable(db.contacts).name,
          phoneRaw: row.readTable(db.contacts).phoneRaw,
        ),
    ],
  );
}
