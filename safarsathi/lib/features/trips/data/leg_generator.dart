// lib/features/trips/data/leg_generator.dart
//
// Legs derive from stops. There is no such thing as a leg the user created
// directly: one exists between every pair of consecutive stops, and the whole
// of issue #17 is keeping that true as stops are added, reordered and removed.
//
// THE RULE THAT EARNS THIS FILE: a leg whose from/to pair is unchanged KEEPS
// ITS ROW, and therefore keeps its cached route polyline, its distance and its
// lastSyncedAt. Regenerating by deleting every leg and inserting fresh ones is
// shorter by four lines and throws away the only data in this app that needed
// a network connection to obtain. Reordering the last two stops of a ten-stop
// trip must leave the first eight legs untouched.

import 'package:drift/drift.dart';

import '../../../core/database/app_database.dart';

/// What a regeneration would do, without doing it. Returned so the reasoning
/// is testable in isolation from the database.
class LegPlan {
  /// Pairs that need a new row, in sequence order.
  final List<({int from, int to, int order})> insert;

  /// Existing legs whose pair survived, with the order they now sit at.
  final List<({int id, int order})> reorder;

  /// Legs whose pair no longer exists anywhere in the itinerary.
  final List<int> delete;

  const LegPlan({
    required this.insert,
    required this.reorder,
    required this.delete,
  });

  bool get isEmpty => insert.isEmpty && reorder.isEmpty && delete.isEmpty;
}

/// Only what matters for matching. Keeps the planner free of Drift rows so it
/// can be tested with literals.
typedef ExistingLeg = ({int id, int fromStopId, int toStopId, int order});

/// Works out the minimum set of changes that makes [legs] match [stopIds].
///
/// Matching is by the (from, to) PAIR, never by sequence position. A leg is
/// the road between two places; which number it sits at in the itinerary is
/// incidental, and matching on position would discard a cached route every
/// time a stop was inserted earlier in the trip.
LegPlan planLegs(List<int> stopIds, List<ExistingLeg> legs) {
  final wanted = <({int from, int to, int order})>[];
  for (var i = 0; i < stopIds.length - 1; i++) {
    wanted.add((from: stopIds[i], to: stopIds[i + 1], order: i + 1));
  }

  // A pair can legitimately repeat: Shillong → Cherrapunji → Shillong →
  // Cherrapunji is a real itinerary. So candidates are held per pair and
  // consumed one at a time rather than looked up once.
  final byPair = <String, List<ExistingLeg>>{};
  for (final leg in legs) {
    byPair.putIfAbsent('${leg.fromStopId}>${leg.toStopId}', () => []).add(leg);
  }

  final insert = <({int from, int to, int order})>[];
  final reorder = <({int id, int order})>[];
  final used = <int>{};

  for (final w in wanted) {
    final candidates = byPair['${w.from}>${w.to}'];
    if (candidates != null && candidates.isNotEmpty) {
      final match = candidates.removeAt(0);
      used.add(match.id);
      if (match.order != w.order) reorder.add((id: match.id, order: w.order));
    } else {
      insert.add(w);
    }
  }

  final delete = [
    for (final leg in legs)
      if (!used.contains(leg.id)) leg.id,
  ];

  return LegPlan(insert: insert, reorder: reorder, delete: delete);
}

/// Applies the plan. One transaction: a half-regenerated itinerary is worse
/// than an out-of-date one.
Future<LegPlan> regenerateLegs(AppDatabase db, int tripId) async {
  return db.transaction(() async {
    final stops =
        await (db.select(db.stops)
              ..where((s) => s.tripId.equals(tripId))
              ..orderBy([(s) => OrderingTerm(expression: s.sequenceOrder)]))
            .get();

    final existing =
        await (db.select(db.legs)..where((l) => l.tripId.equals(tripId))).get();

    final plan = planLegs(
      [for (final s in stops) s.id],
      [
        for (final l in existing)
          (
            id: l.id,
            fromStopId: l.fromStopId,
            toStopId: l.toStopId,
            order: l.sequenceOrder,
          ),
      ],
    );

    // PLANNED STOPS OUTLIVE THEIR LEG. A stop inserted between Shillong and
    // Sohra replaces that leg with two; the viewpoint planned on it belongs
    // on the first new one (same start), or else the one with the same end.
    // Remembered before the delete, because the delete nulls their legId.
    final gone = {
      for (final l in existing)
        if (plan.delete.contains(l.id)) l.id: l,
    };
    final orphans = gone.isEmpty
        ? const <PlannedStop>[]
        : await (db.select(db.plannedStops)
                ..where((p) => p.legId.isIn(gone.keys.toList())))
              .get();

    for (final id in plan.delete) {
      await (db.delete(db.legs)..where((l) => l.id.equals(id))).go();
    }
    for (final r in plan.reorder) {
      await (db.update(db.legs)..where((l) => l.id.equals(r.id))).write(
        LegsCompanion(sequenceOrder: Value(r.order)),
      );
    }
    for (final i in plan.insert) {
      await db.into(db.legs).insert(
        LegsCompanion.insert(
          tripId: tripId,
          fromStopId: i.from,
          toStopId: i.to,
          sequenceOrder: i.order,
        ),
      );
    }

    if (orphans.isNotEmpty) {
      final now =
          await (db.select(db.legs)..where((l) => l.tripId.equals(tripId)))
              .get();
      for (final p in orphans) {
        final old = gone[p.legId]!;
        Leg? home;
        for (final l in now) {
          if (l.fromStopId == old.fromStopId) home = l;
        }
        if (home == null) {
          for (final l in now) {
            if (l.toStopId == old.toStopId) home = l;
          }
        }
        // No leg shares either end: it stays on the trip, legless, and the
        // plan still lists it. Losing a planned stop silently is the one
        // outcome not allowed.
        if (home != null) {
          await (db.update(db.plannedStops)..where((x) => x.id.equals(p.id)))
              .write(PlannedStopsCompanion(legId: Value(home.id)));
        }
      }
    }

    return plan;
  });
}
