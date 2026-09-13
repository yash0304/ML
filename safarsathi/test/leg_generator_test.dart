// test/leg_generator_test.dart — issue #17
//
// The planner is pure, so the rule that earns this issue — a leg whose pair
// is unchanged keeps its row, and therefore its cached route — is checkable
// without a database.

import 'package:drift/drift.dart' hide isNull, isNotNull;
import 'package:drift/native.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:safarsathi/core/database/app_database.dart';
import 'package:safarsathi/features/trips/data/leg_generator.dart';

ExistingLeg leg(int id, int from, int to, int order) =>
    (id: id, fromStopId: from, toStopId: to, order: order);

void main() {
  group('planning', () {
    test('a fresh trip gets a leg between each consecutive pair', () {
      final plan = planLegs([1, 2, 3], const []);

      expect(plan.insert.length, 2);
      expect(plan.insert[0], (from: 1, to: 2, order: 1));
      expect(plan.insert[1], (from: 2, to: 3, order: 2));
      expect(plan.delete, isEmpty);
    });

    test('a single stop has no legs', () {
      expect(planLegs([1], const []).isEmpty, isTrue);
      expect(planLegs([], const []).isEmpty, isTrue);
    });

    test('nothing changes when the itinerary has not', () {
      final plan = planLegs([1, 2, 3], [leg(10, 1, 2, 1), leg(11, 2, 3, 2)]);
      expect(plan.isEmpty, isTrue);
    });

    test('REORDERING THE TAIL LEAVES EARLIER LEGS UNTOUCHED', () {
      // The rule this whole file exists for. Legs 10 and 11 carry cached
      // routes that cost a network connection to obtain; a regeneration that
      // deletes and reinserts them throws that away for nothing.
      final before = [
        leg(10, 1, 2, 1),
        leg(11, 2, 3, 2),
        leg(12, 3, 4, 3),
        leg(13, 4, 5, 4),
      ];
      final plan = planLegs([1, 2, 3, 5, 4], before);

      expect(plan.delete, isNot(contains(10)));
      expect(plan.delete, isNot(contains(11)));
      // 3→4 and 4→5 are gone; 3→5 and 5→4 are new.
      expect(plan.delete, containsAll([12, 13]));
      expect(plan.insert.map((i) => (i.from, i.to)), [(3, 5), (5, 4)]);
    });

    test('inserting a stop at the front keeps every later leg', () {
      final before = [leg(10, 1, 2, 1), leg(11, 2, 3, 2)];
      final plan = planLegs([9, 1, 2, 3], before);

      expect(plan.delete, isEmpty);
      expect(plan.insert.single, (from: 9, to: 1, order: 1));
      // Both survivors shift down one.
      expect(plan.reorder, containsAll([(id: 10, order: 2), (id: 11, order: 3)]));
    });

    test('removing a middle stop joins its neighbours', () {
      final before = [leg(10, 1, 2, 1), leg(11, 2, 3, 2)];
      final plan = planLegs([1, 3], before);

      expect(plan.delete, containsAll([10, 11]));
      expect(plan.insert.single, (from: 1, to: 3, order: 1));
    });

    test('a pair that legitimately repeats gets two legs', () {
      // Shillong → Cherrapunji → Shillong → Cherrapunji is a real itinerary.
      final plan = planLegs([1, 2, 1, 2], const []);
      expect(plan.insert.length, 3);
      expect(plan.insert.map((i) => (i.from, i.to)), [(1, 2), (2, 1), (1, 2)]);
    });

    test('a repeated pair consumes existing rows one at a time', () {
      final before = [leg(10, 1, 2, 1)];
      final plan = planLegs([1, 2, 1, 2], before);

      // The first 1→2 reuses leg 10; the second needs a new row.
      expect(plan.delete, isEmpty);
      expect(plan.insert.map((i) => (i.from, i.to)), [(2, 1), (1, 2)]);
    });

    test('a leg whose stops are gone entirely is deleted', () {
      final plan = planLegs([1, 2], [leg(10, 7, 8, 1)]);
      expect(plan.delete, [10]);
    });
  });

  group('against the database', () {
    late AppDatabase db;
    late int tripId;

    setUp(() async {
      db = AppDatabase(NativeDatabase.memory());
      tripId = await db
          .into(db.trips)
          .insert(TripsCompanion.insert(name: 'Meghalaya'));
    });

    tearDown(() => db.close());

    Future<int> stop(String name, int order) => db.into(db.stops).insert(
      StopsCompanion.insert(
        tripId: tripId,
        name: name,
        sequenceOrder: order,
        countryCode: 'IN',
      ),
    );

    test('a cached route survives a reorder of later stops', () async {
      final a = await stop('Shillong', 1);
      final b = await stop('Cherrapunji', 2);
      final c = await stop('Dawki', 3);
      final d = await stop('Mawlynnong', 4);

      await regenerateLegs(db, tripId);

      // Pretend the first leg was synced on WiFi at setup.
      final first = await (db.select(db.legs)
            ..where((l) => l.fromStopId.equals(a) & l.toStopId.equals(b)))
          .getSingle();
      await (db.update(db.legs)..where((l) => l.id.equals(first.id))).write(
        LegsCompanion(
          routePolyline: const Value('encoded'),
          distanceKm: const Value(54.0),
          lastSyncedAt: Value(DateTime(2026, 9, 1)),
        ),
      );

      // Swap the last two stops.
      await (db.update(db.stops)..where((s) => s.id.equals(d))).write(
        const StopsCompanion(sequenceOrder: Value(3)),
      );
      await (db.update(db.stops)..where((s) => s.id.equals(c))).write(
        const StopsCompanion(sequenceOrder: Value(4)),
      );
      await regenerateLegs(db, tripId);

      final after = await (db.select(
        db.legs,
      )..where((l) => l.id.equals(first.id))).getSingleOrNull();

      expect(after, isNotNull);
      expect(after!.routePolyline, 'encoded');
      expect(after.distanceKm, 54.0);
      expect(after.lastSyncedAt, DateTime(2026, 9, 1));
    });

    test('legs match the itinerary after every change', () async {
      await stop('A', 1);
      await stop('B', 2);
      await regenerateLegs(db, tripId);
      expect((await db.select(db.legs).get()).length, 1);

      await stop('C', 3);
      await regenerateLegs(db, tripId);
      expect((await db.select(db.legs).get()).length, 2);
    });

    test('legs are numbered densely and in order', () async {
      for (var i = 1; i <= 4; i++) {
        await stop('S$i', i);
      }
      await regenerateLegs(db, tripId);

      final legs = await (db.select(db.legs)..orderBy([
            (l) => OrderingTerm(expression: l.sequenceOrder),
          ]))
          .get();
      expect(legs.map((l) => l.sequenceOrder), [1, 2, 3]);
    });
  });
}
