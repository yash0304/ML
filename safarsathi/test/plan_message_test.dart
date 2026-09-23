// test/plan_message_test.dart — the message somebody at home receives.

import 'package:drift/drift.dart' hide isNull, isNotNull;
import 'package:drift/native.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:safarsathi/core/database/app_database.dart';
import 'package:safarsathi/features/contacts/data/contacts_dao.dart';
import 'package:safarsathi/features/trips/data/plan_message.dart';
import 'package:safarsathi/features/trips/data/trip_editor.dart';

void main() {
  late AppDatabase db;
  late TripEditor editor;
  late int tripId, shillong, sohra, dawki;

  setUp(() async {
    db = AppDatabase(NativeDatabase.memory());
    editor = TripEditor(db);
    tripId = await editor.createTrip(name: 'Meghalaya');
    await (db.update(db.trips)..where((t) => t.id.equals(tripId))).write(
      TripsCompanion(
        startDate: Value(DateTime(2026, 10, 1)),
        endDate: Value(DateTime(2026, 10, 5)),
      ),
    );
    shillong = await editor.addStop(tripId, StopDraft(
        name: 'Shillong', arrivalDate: DateTime(2026, 10, 1), nights: 2));
    sohra = await editor.addStop(tripId, StopDraft(
        name: 'Sohrra', arrivalDate: DateTime(2026, 10, 3), nights: 1));
    dawki = await editor.addStop(tripId, StopDraft(
        name: 'Dawki', arrivalDate: DateTime(2026, 10, 5)));
  });
  tearDown(() => db.close());

  Future<void> stay(String name, int stopId, String phone) =>
      db.into(db.contacts).insert(ContactsCompanion.insert(
        tripId: Value(tripId), stopId: Value(stopId), name: name,
        phoneRaw: phone, category: const Value(ContactCategory.accommodation),
      ));

  test('the whole message, as it will be read', () async {
    await stay('Hotel Pinewood', shillong, '0364 222 3116');
    await db.into(db.travellers).insert(
        TravellersCompanion.insert(tripId: tripId, name: 'Yash'));
    await db.into(db.travellers).insert(
        TravellersCompanion.insert(tripId: tripId, name: 'Chhaya'));
    final legs = await (db.select(db.legs)
          ..orderBy([(l) => OrderingTerm(expression: l.sequenceOrder)]))
        .get();
    await (db.update(db.legs)..where((l) => l.id.equals(legs.first.id))).write(
      LegsCompanion(
        mode: const Value('Shared sumo'),
        plannedDeparture: Value(DateTime(2026, 10, 3, 9, 30)),
      ),
    );

    expect(await buildPlanMessage(db, tripId), '''
Meghalaya — 1 Oct–5 Oct 2026
Travelling: Yash, Chhaya

Where we sleep each night:
1 Oct · Shillong (2 nights)
  Stay: Hotel Pinewood, 0364 222 3116
3 Oct · Sohrra (1 night)
  Stay: not saved yet
5 Oct · Dawki (passing through)

Getting between them:
3 Oct 09:30 · Shillong → Sohrra · Shared sumo
Sohrra → Dawki

If we do not answer, we are probably out of signal — the number for each night's stay is above.''');
  });

  test('A NIGHT WITH NO STAY SAYS SO rather than being left out', () async {
    // Somebody at home reading "Sohra" with nothing under it would assume
    // it was arranged. It is worth them knowing that it is not.
    expect(await buildPlanMessage(db, tripId),
        contains('Sohrra (1 night)\n  Stay: not saved yet'));
  });

  test('a stop passed through lists no stay line at all', () async {
    final msg = await buildPlanMessage(db, tripId);
    expect(msg, contains('Dawki (passing through)'));
    expect(msg.split('Dawki (passing through)').last,
        isNot(contains('Stay:')));
    expect(dawki, isNonZero);
  });

  test('only stays — a restaurant is not where you sleep', () async {
    await db.into(db.contacts).insert(ContactsCompanion.insert(
      tripId: Value(tripId), stopId: Value(sohra), name: 'Orange Roots',
      phoneRaw: '1', category: const Value(ContactCategory.restaurant),
    ));
    expect(await buildPlanMessage(db, tripId), isNot(contains('Orange Roots')));
  });

  test('a trip with no dates still makes a message', () async {
    final bare = await editor.createTrip(name: 'Somewhere');
    await editor.addStop(bare, const StopDraft(name: 'Ooty'));
    final msg = await buildPlanMessage(db, bare);
    expect(msg, startsWith('Somewhere\n'));
    expect(msg, contains('Ooty (passing through)'));
  });
}
