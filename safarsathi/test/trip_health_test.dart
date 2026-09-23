// test/trip_health_test.dart
//
// Sample data and duplicates — found in a real trip a week before departure:
// the demo's made-up numbers (two seeded as confirmed, one as an emergency
// contact) and every imported stay listed three times.

import 'package:drift/drift.dart' hide isNull, isNotNull;
import 'package:drift/native.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:safarsathi/core/database/app_database.dart';
import 'package:safarsathi/features/contacts/data/contacts_dao.dart';
import 'package:safarsathi/features/dev/dev_seed.dart';
import 'package:safarsathi/features/trips/data/trip_health.dart';

void main() {
  late AppDatabase db;
  setUp(() => db = AppDatabase(NativeDatabase.memory()));
  tearDown(() => db.close());

  group('the demo', () {
    test('NEVER SEEDS A NUMBER THAT LOOKS VERIFIED OR SITS ON THE SOS TAB',
        () async {
      final demo = await createDemoTrip(db);
      final seeded = await (db.select(db.contacts)
            ..where((c) => c.tripId.equals(demo.tripId)))
          .get();
      expect(seeded, isNotEmpty);
      expect(seeded.where((c) => c.callConfirmed), isEmpty,
          reason: 'a sample may show the diary; it may never look trusted');
      expect(seeded.where((c) => c.isEmergency), isEmpty);
    });

    test('everything it seeds is recognised as sample data', () async {
      final demo = await createDemoTrip(db);
      final health = await watchTripHealth(db, demo.tripId).first;
      final contacts = await (db.select(db.contacts)
            ..where((c) => c.tripId.equals(demo.tripId)))
          .get();
      final expenses = await (db.select(db.expenses)
            ..where((e) => e.tripId.equals(demo.tripId)))
          .get();
      expect(health.sampleContacts.length, contacts.length);
      expect(health.sampleExpenses.length, expenses.length);
    });
  });

  group('sample data in a real trip', () {
    late int tripId;
    setUp(() async {
      tripId = await db.into(db.trips).insert(
            TripsCompanion.insert(name: 'Meghalaya'));
    });

    Future<int> contact(String name, String phone,
            {bool confirmed = false, bool emergency = false, int? stopId,
            String category = ContactCategory.other}) =>
        db.into(db.contacts).insert(ContactsCompanion.insert(
          tripId: Value(tripId), stopId: Value(stopId), name: name,
          phoneRaw: phone, category: Value(category),
          callConfirmed: Value(confirmed), isEmergency: Value(emergency),
        ));

    test('THE OLD DEMO\'S CONFIRMED EMERGENCY FAKE IS FOUND AND NAMED',
        () async {
      // Exactly what was on the phone: seeded by an older build.
      await contact('Bah Rothell · homestay owner', '+91 90000 00007',
          confirmed: true, emergency: true);
      await contact('Kongthong homestay', '+91 90000 00001', confirmed: true);
      await contact('Kongthong Travellers Nest', '+919856060347');

      final health = await watchTripHealth(db, tripId).first;
      expect(health.sampleContacts.map((c) => c.name),
          containsAll(['Bah Rothell · homestay owner', 'Kongthong homestay']));
      expect(health.sampleConfirmed, 2);
      expect(health.sampleEmergency, 1);
    });

    test('found by the number even if renamed', () async {
      await contact('My driver', '+91 90000 00002');
      expect((await watchTripHealth(db, tripId).first).sampleContacts,
          hasLength(1));
    });

    test('A REAL NUMBER IS NEVER TAKEN FOR A SAMPLE', () async {
      await contact('Kongthong Travellers Nest', '+919856060347');
      await contact('Shillong guesthouse', '+91 98631 00003'); // demo's NAME
      await contact('Near miss', '+91 90000 00008'); // not one of the seven
      expect((await watchTripHealth(db, tripId).first).hasSamples, isFalse);
    });

    test('sample expenses are matched to the paisa', () async {
      final me = await db.into(db.travellers).insert(
          TravellersCompanion.insert(tripId: tripId, name: 'You'));
      Future<void> spend(String what, int paise) =>
          db.into(db.expenses).insert(ExpensesCompanion.insert(
                tripId: tripId, description: what, amountMinor: paise,
                paidById: me, spentAt: Value(DateTime(2026, 10, 2))));
      await spend('Fuel', 210000); // the demo's
      await spend('Fuel', 180000); // a real one, same word
      final health = await watchTripHealth(db, tripId).first;
      expect(health.sampleExpenses.single.amountMinor, 210000);
    });

    test('removing takes the samples and nothing else', () async {
      await contact('Bah Rothell · homestay owner', '+91 90000 00007',
          confirmed: true, emergency: true);
      final keep = await contact('Kongthong Travellers Nest', '+919856060347');
      await removeSamples(db, tripId);
      final left = await db.select(db.contacts).get();
      expect(left.map((c) => c.id), [keep]);
    });
  });

  group('duplicates', () {
    late int tripId;
    setUp(() async {
      tripId = await db.into(db.trips).insert(
            TripsCompanion.insert(name: 'Meghalaya'));
    });

    Future<int> c(String name, String phone,
            {bool confirmed = false, int? stopId}) =>
        db.into(db.contacts).insert(ContactsCompanion.insert(
          tripId: Value(tripId), stopId: Value(stopId), name: name,
          phoneRaw: phone, phoneE164: Value(phone),
          category: const Value(ContactCategory.accommodation),
          callConfirmed: Value(confirmed),
        ));

    test('three imports of the same stay leave two extras', () async {
      for (var i = 0; i < 3; i++) {
        await c('Alpha Guest House Sohra', '+918974804455');
      }
      final health = await watchTripHealth(db, tripId).first;
      expect(health.duplicateExtras, hasLength(2));
    });

    test('THE CONFIRMED COPY IS THE ONE KEPT', () async {
      await c('Alpha Guest House Sohra', '+918974804455');
      final called =
          await c('Alpha Guest House Sohra', '+918974804455', confirmed: true);
      await c('Alpha Guest House Sohra', '+918974804455');
      await removeDuplicates(db, tripId);
      final left = await db.select(db.contacts).get();
      expect(left.single.id, called);
    });

    test('otherwise the oldest is kept', () async {
      final first = await c('Alpha', '+918974804455');
      await c('Alpha', '+918974804455');
      await removeDuplicates(db, tripId);
      expect((await db.select(db.contacts).get()).single.id, first);
    });

    test('same number at a different stop is not a duplicate', () async {
      final stop = await db.into(db.stops).insert(StopsCompanion.insert(
          tripId: tripId, name: 'Sohrra', sequenceOrder: 1,
          countryCode: 'IN'));
      await c('Food Planet', '+917005379187');
      await c('Food Planet', '+917005379187', stopId: stop);
      expect((await watchTripHealth(db, tripId).first).duplicateExtras,
          isEmpty);
    });

    test('different names on one number are not duplicates', () async {
      await c('112 — ERSS', '112');
      await c('Unified emergency', '112');
      expect((await watchTripHealth(db, tripId).first).duplicateExtras,
          isEmpty);
    });

    test('a sample is counted once, as a sample, not also as a duplicate',
        () async {
      await c('Kongthong homestay', '+91 90000 00001');
      await c('Kongthong homestay', '+91 90000 00001');
      final health = await watchTripHealth(db, tripId).first;
      expect(health.sampleContacts, hasLength(2));
      expect(health.duplicateExtras, isEmpty);
    });
  });
}
