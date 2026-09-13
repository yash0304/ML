// test/trip_editor_test.dart — issues #16 and #19
//
// The acceptance criterion from the backlog, spelled out: build the full
// Meghalaya itinerary and check the two Shillong rows coexist.

import 'package:drift/drift.dart' hide isNull, isNotNull;
import 'package:drift/native.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:safarsathi/core/database/app_database.dart';
import 'package:safarsathi/features/contacts/data/contacts_dao.dart';
import 'package:safarsathi/features/trips/data/trip_editor.dart';

void main() {
  late AppDatabase db;
  late TripEditor editor;

  setUp(() {
    db = AppDatabase(NativeDatabase.memory());
    editor = TripEditor(db);
  });

  tearDown(() => db.close());

  Future<List<Stop>> stopsOf(int tripId) => editor.stopsOf(tripId);

  group('trips', () {
    test('creating a trip makes it the active one', () async {
      final id = await editor.createTrip(name: 'Meghalaya');
      final trip = await (db.select(
        db.trips,
      )..where((t) => t.id.equals(id))).getSingle();
      expect(trip.isActive, isTrue);
    });

    test('EXACTLY ONE trip is ever active', () async {
      final a = await editor.createTrip(name: 'Meghalaya');
      final b = await editor.createTrip(name: 'Ladakh');
      await editor.setActiveTrip(a);

      final all = await db.select(db.trips).get();
      expect(all.where((t) => t.isActive).map((t) => t.id), [a]);

      await editor.setActiveTrip(b);
      final after = await db.select(db.trips).get();
      expect(after.where((t) => t.isActive).map((t) => t.id), [b]);
    });

    test('deleting the active trip promotes another', () async {
      await editor.createTrip(name: 'Meghalaya');
      final b = await editor.createTrip(name: 'Ladakh');
      await editor.deleteTrip(b);

      final all = await db.select(db.trips).get();
      expect(all.length, 1);
      expect(all.single.isActive, isTrue);
    });

    test('ensureActiveTrip promotes when nothing is flagged', () async {
      // Every trip predating #16 is in exactly this state.
      await db.into(db.trips).insert(TripsCompanion.insert(name: 'Old'));
      expect(
        (await db.select(db.trips).get()).any((t) => t.isActive),
        isFalse,
      );

      await ensureActiveTrip(db);
      expect((await db.select(db.trips).get()).single.isActive, isTrue);
    });

    test('deleting a trip takes its stops and contacts with it', () async {
      final id = await editor.createTrip(name: 'Meghalaya');
      final stop = await editor.addStop(id, const StopDraft(name: 'Shillong'));
      await db.into(db.contacts).insert(
        ContactsCompanion.insert(
          name: 'Rina',
          phoneRaw: '+91 90000 00001',
          tripId: Value(id),
          stopId: Value(stop),
        ),
      );

      await editor.deleteTrip(id);
      expect(await db.select(db.stops).get(), isEmpty);
      expect(await db.select(db.contacts).get(), isEmpty);
    });
  });

  group('stops', () {
    late int tripId;
    setUp(() async {
      tripId = await editor.createTrip(name: 'Meghalaya');
    });

    test('THE MEGHALAYA ITINERARY: Shillong appears twice', () async {
      for (final name in [
        'Shillong',
        'Cherrapunji',
        'Shillong',
        'Dawki',
        'Mawlynnong',
      ]) {
        await editor.addStop(tripId, StopDraft(name: name));
      }

      final stops = await stopsOf(tripId);
      expect(stops.map((s) => s.name), [
        'Shillong',
        'Cherrapunji',
        'Shillong',
        'Dawki',
        'Mawlynnong',
      ]);

      final shillongs = stops.where((s) => s.name == 'Shillong').toList();
      expect(shillongs.length, 2);
      expect(shillongs.first.id, isNot(shillongs.last.id));
      expect(shillongs.first.sequenceOrder, 1);
      expect(shillongs.last.sequenceOrder, 3);

      // Four legs, and the two Shillong departures go to different places.
      final legs = await (db.select(db.legs)..orderBy([
            (l) => OrderingTerm(expression: l.sequenceOrder),
          ]))
          .get();
      expect(legs.length, 4);
      expect(legs[0].fromStopId, shillongs.first.id);
      expect(legs[2].fromStopId, shillongs.last.id);
    });

    test('stops are numbered densely as they are added', () async {
      for (var i = 1; i <= 4; i++) {
        await editor.addStop(tripId, StopDraft(name: 'S$i'));
      }
      expect(
        (await stopsOf(tripId)).map((s) => s.sequenceOrder),
        [1, 2, 3, 4],
      );
    });

    test('reordering renumbers densely', () async {
      for (final n in ['A', 'B', 'C', 'D']) {
        await editor.addStop(tripId, StopDraft(name: n));
      }
      await editor.reorderStops(tripId, 3, 0); // D to the front

      final stops = await stopsOf(tripId);
      expect(stops.map((s) => s.name), ['D', 'A', 'B', 'C']);
      expect(stops.map((s) => s.sequenceOrder), [1, 2, 3, 4]);
    });

    test('deleting a stop renumbers and rejoins the legs', () async {
      for (final n in ['A', 'B', 'C']) {
        await editor.addStop(tripId, StopDraft(name: n));
      }
      final b = (await stopsOf(tripId))[1];
      await editor.deleteStop(tripId, b.id);

      final stops = await stopsOf(tripId);
      expect(stops.map((s) => s.name), ['A', 'C']);
      expect(stops.map((s) => s.sequenceOrder), [1, 2]);

      final leg = await db.select(db.legs).getSingle();
      expect(leg.fromStopId, stops.first.id);
      expect(leg.toStopId, stops.last.id);
    });

    test('DELETING A STOP KEEPS ITS CONTACTS, UNATTACHED', () async {
      // A number you have dialled and confirmed does not stop being a real
      // number because you dropped the stop from the plan.
      final stop = await editor.addStop(
        tripId,
        const StopDraft(name: 'Shillong'),
      );
      await db.into(db.contacts).insert(
        ContactsCompanion.insert(
          name: 'Rina',
          phoneRaw: '+91 90000 00001',
          tripId: Value(tripId),
          stopId: Value(stop),
          callConfirmed: const Value(true),
        ),
      );

      expect(await editor.contactsAt(stop), 1);
      await editor.deleteStop(tripId, stop);

      final contact = await db.select(db.contacts).getSingle();
      expect(contact.name, 'Rina');
      expect(contact.stopId, isNull);
      expect(contact.callConfirmed, isTrue);
    });

    test('nights derive from the dates when both are set', () {
      const draft = StopDraft(name: 'Shillong', nights: 9);
      final dated = draft.copyWith(
        arrivalDate: DateTime(2026, 10, 1),
        departureDate: DateTime(2026, 10, 3),
      );
      expect(dated.effectiveNights, 2);
      expect(dated.isOvernight, isTrue);
    });

    test('nights are typed when the dates are not both set', () {
      const draft = StopDraft(name: 'Shillong', nights: 2);
      expect(draft.effectiveNights, 2);
      expect(
        draft.copyWith(arrivalDate: DateTime(2026, 10, 1)).effectiveNights,
        2,
      );
    });

    test('a reversed date range is zero nights, never negative', () {
      const draft = StopDraft(name: 'X');
      final bad = draft.copyWith(
        arrivalDate: DateTime(2026, 10, 5),
        departureDate: DateTime(2026, 10, 1),
      );
      expect(bad.effectiveNights, 0);
    });

    test('tags round-trip through storage', () async {
      final id = await editor.addStop(
        tripId,
        const StopDraft(name: 'Sohra', activityTags: ['caves', 'rain']),
      );
      final saved = await (db.select(
        db.stops,
      )..where((s) => s.id.equals(id))).getSingle();
      expect(saved.activityTags, 'caves,rain');
      expect(parseTags(saved.activityTags), ['caves', 'rain']);
    });

    test('tag parsing tolerates what people actually type', () {
      expect(parseTags('trek, caves,'), ['trek', 'caves']);
      expect(parseTags(''), isEmpty);
      expect(parseTags('  '), isEmpty);
    });
  });

  group('current stop', () {
    Stop row(int id, String name, int order, {DateTime? from, DateTime? to}) =>
        Stop(
          id: id,
          tripId: 1,
          name: name,
          sequenceOrder: order,
          countryCode: 'IN',
          arrivalDate: from,
          departureDate: to,
          nights: 0,
          activityTags: '',
        );

    final stops = [
      row(
        1,
        'Shillong',
        1,
        from: DateTime(2026, 10, 1),
        to: DateTime(2026, 10, 3),
      ),
      row(
        2,
        'Cherrapunji',
        2,
        from: DateTime(2026, 10, 3),
        to: DateTime(2026, 10, 5),
      ),
      row(3, 'Dawki', 3, from: DateTime(2026, 10, 5)),
    ];

    test('mid-trip, the stop whose window contains today', () {
      expect(currentStopOf(stops, now: DateTime(2026, 10, 4))?.name,
          'Cherrapunji');
    });

    test('before the trip, the first stop', () {
      expect(currentStopOf(stops, now: DateTime(2026, 9, 20))?.name,
          'Shillong');
    });

    test('after the trip, the last one reached', () {
      expect(currentStopOf(stops, now: DateTime(2026, 11, 1))?.name, 'Dawki');
    });

    test('A TRIP WITH NO DATES STILL HAS A CURRENT STOP', () {
      // Returning null here would make the diary's stop-scope toggle vanish
      // on exactly the itineraries planned in a hurry.
      final undated = [row(1, 'A', 1), row(2, 'B', 2)];
      expect(currentStopOf(undated, now: DateTime(2026, 10, 4))?.name, 'A');
    });

    test('no stops means no current stop', () {
      expect(currentStopOf(const [], now: DateTime(2026, 10, 4)), isNull);
    });

    test('the boundary day belongs to the stop that starts it', () {
      // Shillong leaves on the 3rd and Cherrapunji arrives on the 3rd. The
      // earlier window wins, which is the honest answer: you are still there
      // until you go.
      expect(currentStopOf(stops, now: DateTime(2026, 10, 3))?.name,
          'Shillong');
    });
  });

  group('active trip context', () {
    test('follows the active trip and its derived stop', () async {
      final id = await editor.createTrip(name: 'Meghalaya');
      await editor.addStop(
        id,
        StopDraft(
          name: 'Shillong',
          arrivalDate: DateTime.now().subtract(const Duration(days: 1)),
          departureDate: DateTime.now().add(const Duration(days: 1)),
        ),
      );

      final context = await watchActiveTripContext(db).first;
      expect(context?.tripId, id);
      expect(context?.name, 'Meghalaya');
      expect(context?.currentStopName, 'Shillong');
    });

    test('is null when there is no trip', () async {
      expect(await watchActiveTripContext(db).first, isNull);
    });

    test('country comes from the stop, not the trip', () async {
      final id = await editor.createTrip(name: 'Alps');
      await editor.addStop(
        id,
        const StopDraft(name: 'Innsbruck', countryCode: 'AT'),
      );
      final context = await watchActiveTripContext(db).first;
      expect(context?.countryCode, 'AT');
    });
  });

  test('a saved contact category survives the editor untouched', () async {
    // Guards against a future "tidy up" that rewrites contacts on stop edits.
    final tripId = await editor.createTrip(name: 'T');
    final stop = await editor.addStop(tripId, const StopDraft(name: 'S'));
    await db.into(db.contacts).insert(
      ContactsCompanion.insert(
        name: 'Homestay',
        phoneRaw: '1',
        tripId: Value(tripId),
        stopId: Value(stop),
        category: const Value(ContactCategory.accommodation),
      ),
    );

    await editor.updateStop(
      tripId,
      StopDraft(id: stop, name: 'S renamed', nights: 2),
    );

    final contact = await db.select(db.contacts).getSingle();
    expect(contact.category, ContactCategory.accommodation);
    expect(contact.stopId, stop);
  });
}
