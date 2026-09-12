// lib/features/trips/data/trip_summary.dart
//
// What the Trip screen needs, gathered in one place so the screen can take a
// stream and stay ignorant of the database. Same seam as the diary.

import 'package:drift/drift.dart';

import '../../../core/database/app_database.dart';

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
/// Rebuilds whenever any of the underlying tables changes, which is what
/// keeps the readiness banner and the per-stop counts honest.
Stream<TripSummary> watchTripSummary(
  AppDatabase db,
  int tripId, {
  int? currentStopId,
}) {
  final query = db.select(db.trips)..where((t) => t.id.equals(tripId));

  return query.watchSingle().asyncMap((trip) async {
    final stops =
        await (db.select(db.stops)
              ..where((s) => s.tripId.equals(tripId))
              ..orderBy([(s) => OrderingTerm(expression: s.sequenceOrder)]))
            .get();

    final contacts = await (db.select(
      db.contacts,
    )..where((c) => c.tripId.equals(tripId))).get();

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
          isCurrent: s.id == currentStopId,
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
        (l) => currentStopId == null || l.fromStopId == currentStopId,
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
