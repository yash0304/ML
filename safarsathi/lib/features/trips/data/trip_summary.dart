// lib/features/trips/data/trip_summary.dart
//
// What the Trip screen needs, gathered in one place so the screen can take a
// stream and stay ignorant of the database. Same seam as the diary.

import 'package:drift/drift.dart';

import '../../../core/database/app_database.dart';
import 'trip_editor.dart';

class StopSummary {
  final int id;
  final String name;
  final int sequenceOrder;
  final int nights;
  final DateTime? arrivalDate;
  final int diaryCount;
  final bool isCurrent;

  const StopSummary({
    required this.id,
    required this.name,
    required this.sequenceOrder,
    required this.nights,
    required this.diaryCount,
    required this.isCurrent,
    this.arrivalDate,
  });
}

class LegSummary {
  final String fromName;
  final String toName;
  final String? mode;
  final double? distanceKm;
  final DateTime? plannedDeparture;
  final String? note;

  /// Null when this leg has never been synced. The milestone cap goes muted
  /// rather than green, so an unprepared leg is visible at a glance.
  final DateTime? lastSyncedAt;

  const LegSummary({
    required this.fromName,
    required this.toName,
    this.mode,
    this.distanceKm,
    this.plannedDeparture,
    this.note,
    this.lastSyncedAt,
  });
}

class TripSummary {
  final String name;
  final DateTime? startDate;
  final DateTime? endDate;
  final List<StopSummary> stops;
  final LegSummary? nextLeg;

  const TripSummary({
    required this.name,
    required this.stops,
    this.startDate,
    this.endDate,
    this.nextLeg,
  });

  int get legCount => stops.length <= 1 ? 0 : stops.length - 1;
}

/// One stream over everything the Trip screen shows.
///
/// The tick query exists because this summary reads four tables and a Drift
/// stream only fires for the tables its own query touches. Watching `trips`
/// alone looked correct for as long as nothing could edit a stop; the moment
/// #16 shipped, adding a stop would have left this screen stale with no error
/// anywhere. `readsFrom` names the real dependency set.
Stream<TripSummary> watchTripSummary(
  AppDatabase db,
  int tripId, {
  int? currentStopId,
}) {
  final tick = db
      .customSelect(
        'SELECT 1',
        readsFrom: {db.trips, db.stops, db.legs, db.contacts},
      )
      .watch();

  return tick.asyncMap((_) async {
    final trip = await (db.select(
      db.trips,
    )..where((t) => t.id.equals(tripId))).getSingle();

    final stops =
        await (db.select(db.stops)
              ..where((s) => s.tripId.equals(tripId))
              ..orderBy([(s) => OrderingTerm(expression: s.sequenceOrder)]))
            .get();

    final contacts = await (db.select(
      db.contacts,
    )..where((c) => c.tripId.equals(tripId))).get();

    // Where the app thinks you are. The caller may override it; otherwise it
    // follows today's date against the stops' own dates (#19).
    final here = currentStopId ?? currentStopOf(stops)?.id;

    int countFor(int stopId) =>
        contacts.where((c) => c.stopId == stopId || c.stopId == null).length;

    final summaries = [
      for (final s in stops)
        StopSummary(
          id: s.id,
          name: s.name,
          sequenceOrder: s.sequenceOrder,
          nights: s.nights,
          arrivalDate: s.arrivalDate,
          diaryCount: countFor(s.id),
          isCurrent: s.id == here,
        ),
    ];

    // The leg leaving the stop you are standing at. Falls back to the first
    // leg when no current stop is known.
    LegSummary? next;
    final legs =
        await (db.select(db.legs)
              ..where((l) => l.tripId.equals(tripId))
              ..orderBy([(l) => OrderingTerm(expression: l.sequenceOrder)]))
            .get();
    if (legs.isNotEmpty) {
      final byId = {for (final s in stops) s.id: s.name};
      final leg = legs.firstWhere(
        (l) => here == null || l.fromStopId == here,
        orElse: () => legs.first,
      );
      next = LegSummary(
        fromName: byId[leg.fromStopId] ?? '—',
        toName: byId[leg.toStopId] ?? '—',
        mode: leg.mode,
        distanceKm: leg.distanceKm,
        plannedDeparture: leg.plannedDeparture,
        note: leg.note,
        lastSyncedAt: leg.lastSyncedAt,
      );
    }

    return TripSummary(
      name: trip.name,
      startDate: trip.startDate,
      endDate: trip.endDate,
      stops: summaries,
      nextLeg: next,
    );
  });
}
