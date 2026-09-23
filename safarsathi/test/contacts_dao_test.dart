// Issue #5 acceptance.
//
// The ordering and filter tests are the easy half. The half that matters is
// the one asserting that nothing can reach this table already verified.

// drift exports its own isNull / isNotNull expression helpers, which collide
// with the matchers of the same name.
import 'package:drift/drift.dart' hide isNull, isNotNull;
import 'package:drift/native.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:safarsathi/core/database/app_database.dart';
import 'package:safarsathi/features/contacts/data/contacts_dao.dart';

void main() {
  late AppDatabase db;
  late ContactsDao dao;
  late int tripId;
  late int kongthongId;
  late int shillongId;

  setUp(() async {
    db = AppDatabase(NativeDatabase.memory());
    dao = db.contactsDao;
    tripId = await db
        .into(db.trips)
        .insert(TripsCompanion.insert(name: 'Meghalaya'));
    kongthongId = await db
        .into(db.stops)
        .insert(
          StopsCompanion.insert(
            tripId: tripId,
            name: 'Kongthong',
            sequenceOrder: 1,
            countryCode: 'IN',
          ),
        );
    shillongId = await db
        .into(db.stops)
        .insert(
          StopsCompanion.insert(
            tripId: tripId,
            name: 'Shillong',
            sequenceOrder: 2,
            countryCode: 'IN',
          ),
        );
  });

  tearDown(() async => db.close());

  Future<int> addContact(
    String name,
    String phone, {
    int? stopId,
    String category = 'other',
    String? note,
    bool pinned = false,
    bool confirmed = false,
    bool emergency = false,
    String? e164,
  }) {
    return dao.insertContact(
      ContactsCompanion.insert(
        name: name,
        phoneRaw: phone,
        tripId: Value(tripId),
        stopId: Value(stopId),
        category: Value(category),
        note: Value(note),
        isPinned: Value(pinned),
        callConfirmed: Value(confirmed),
        isEmergency: Value(emergency),
        phoneE164: Value(e164),
        tier: Value(
          confirmed
              ? ContactTier.userVerified.name
              : ContactTier.userEntered.name,
        ),
      ),
    );
  }

  Future<List<String>> namesFor(ContactFilter f) async =>
      (await dao.watchContacts(f).first).map((c) => c.name).toList();

  // -------------------------------------------------------------------------

  group('ordering', () {
    test('pinned first, then confirmed, then alphabetical', () async {
      await addContact('Zeta unconfirmed', '1');
      await addContact('Alpha unconfirmed', '2');
      await addContact('Yankee confirmed', '3', confirmed: true);
      await addContact('Bravo confirmed', '4', confirmed: true);
      await addContact('Zulu pinned', '5', pinned: true);

      expect(await namesFor(ContactFilter(tripId: tripId)), [
        'Zulu pinned',
        'Bravo confirmed',
        'Yankee confirmed',
        'Alpha unconfirmed',
        'Zeta unconfirmed',
      ]);
    });

    test('a never-called number is not sunk to the bottom', () async {
      // The ordering is deliberately not by recency. If it were, everything
      // never called would sink — which is exactly what the readiness system
      // is trying to surface.
      final called = await addContact('Aaa called', '1');
      await addContact('Bbb never called', '2');
      await dao.logCall(contactId: called, tripId: tripId, action: 'call');

      expect(await namesFor(ContactFilter(tripId: tripId)), [
        'Aaa called',
        'Bbb never called',
      ]);
    });
  });

  group('filters', () {
    test('emergency contacts never appear in the diary feed', () async {
      await addContact('Homestay', '1');
      await addContact('Local ambulance friend', '2', emergency: true);

      expect(await namesFor(ContactFilter(tripId: tripId)), ['Homestay']);
    });

    test('stop scoping keeps trip-wide contacts', () async {
      // Your driver is not tied to one stop, but you still need him while
      // standing in Kongthong.
      await addContact('Homestay', '1', stopId: kongthongId);
      await addContact('Guesthouse', '2', stopId: shillongId);
      await addContact('Driver', '3');

      final names = await namesFor(
        ContactFilter(tripId: tripId, stopId: kongthongId),
      );
      expect(names, containsAll(['Homestay', 'Driver']));
      expect(names, isNot(contains('Guesthouse')));
    });

    test('category filter', () async {
      await addContact('Homestay', '1', category: 'accommodation');
      await addContact('Chemist', '2', category: 'pharmacy');

      expect(
        await namesFor(
          ContactFilter(tripId: tripId, category: 'accommodation'),
        ),
        ['Homestay'],
      );
    });

    test('search matches name, both numbers, note and category', () async {
      await addContact(
        'Bah Rothell',
        '+91 98560 41122',
        e164: '+919856041122',
        note: 'whistling village host',
        category: 'accommodation',
      );
      await addContact('Someone else', '+91 11111 11111');

      for (final term in [
        'Rothell',
        '98560',
        '+919856',
        'whistling',
        'accommodation',
      ]) {
        expect(
          await namesFor(ContactFilter(tripId: tripId, searchTerm: term)),
          ['Bah Rothell'],
          reason: 'search term "$term" should match',
        );
      }
    });

    test('filters compose', () async {
      await addContact(
        'Kongthong homestay',
        '1',
        stopId: kongthongId,
        category: 'accommodation',
      );
      await addContact(
        'Kongthong chemist',
        '2',
        stopId: kongthongId,
        category: 'pharmacy',
      );
      await addContact(
        'Shillong homestay',
        '3',
        stopId: shillongId,
        category: 'accommodation',
      );

      expect(
        await namesFor(
          ContactFilter(
            tripId: tripId,
            stopId: kongthongId,
            category: 'accommodation',
            searchTerm: 'homestay',
          ),
        ),
        ['Kongthong homestay'],
      );
    });
  });

  group('the readiness count', () {
    test('drops the moment a contact is confirmed', () async {
      final a = await addContact('Homestay', '1');
      await addContact('Guesthouse', '2');

      expect(await dao.watchUnconfirmedCount(tripId).first, 2);
      await dao.markConfirmed(a);
      expect(await dao.watchUnconfirmedCount(tripId).first, 1);
    });

    test('confirming promotes the tier and stamps the time', () async {
      final id = await addContact('Homestay', '1');
      await dao.markConfirmed(id);

      final row = await (db.select(
        db.contacts,
      )..where((c) => c.id.equals(id))).getSingle();
      expect(row.tier, ContactTier.userVerified.name);
      expect(row.callConfirmed, isTrue);
      expect(row.confirmedAt, isNotNull);
      expect(ContactTier.parse(row.tier).isTrusted, isTrue);
    });
  });

  group('import', () {
    ContactsCompanion row(String name, {bool confirmed = false}) =>
        ContactsCompanion.insert(
          name: name,
          phoneRaw: '+91 90000 0000$name'.substring(0, 15),
          tripId: Value(tripId),
          // A malicious or careless caller claiming the row is verified.
          callConfirmed: Value(confirmed),
          tier: Value(
            confirmed
                ? ContactTier.userVerified.name
                : ContactTier.userEntered.name,
          ),
        );

    test(
      'cannot produce a confirmed contact, whatever the caller asks',
      () async {
        // THE INVARIANT THE WHOLE TRUST SYSTEM RESTS ON. A number in a
        // spreadsheet is still an unverified number.
        await dao.insertBatch(
          [row('A', confirmed: true), row('B', confirmed: true)],
          ImportBatchesCompanion.insert(
            fileName: 'numbers.xlsx',
            tripId: Value(tripId),
          ),
        );

        final all = await db.select(db.contacts).get();
        expect(all, hasLength(2));
        for (final c in all) {
          expect(
            c.callConfirmed,
            isFalse,
            reason: '${c.name} arrived confirmed',
          );
          expect(c.tier, ContactTier.userEntered.name);
          expect(c.confirmedAt, isNull);
          expect(ContactTier.parse(c.tier).isTrusted, isFalse);
        }
        expect(await dao.watchUnconfirmedCount(tripId).first, 2);
      },
    );

    test('a batch that fails part way leaves nothing behind', () async {
      // One bad file must roll back whole rather than leaving half a
      // spreadsheet in the database.
      await expectLater(
        dao.insertBatch([
          row('A'),
          // No name and no phone: violates NOT NULL.
          const ContactsCompanion(),
        ], ImportBatchesCompanion.insert(fileName: 'broken.csv')),
        throwsA(anything),
      );

      expect(await db.select(db.contacts).get(), isEmpty);
      expect(await db.select(db.importBatches).get(), isEmpty);
    });

    test('rollback removes the contacts and the batch row', () async {
      await dao.insertBatch(
        [row('A'), row('B')],
        ImportBatchesCompanion.insert(
          fileName: 'numbers.xlsx',
          tripId: Value(tripId),
        ),
      );
      final batch = await db.select(db.importBatches).getSingle();

      await dao.rollbackImport(batch.id);

      expect(await db.select(db.contacts).get(), isEmpty);
      expect(await db.select(db.importBatches).get(), isEmpty);
    });
  });

  group('logging', () {
    test('a copy is logged like any other outbound action', () async {
      // The dial happens in the Android dialer after a paste, so without
      // this the record of who was reached would rot.
      final id = await addContact('Homestay', '1');
      await dao.logCall(contactId: id, tripId: tripId, action: 'copy');

      final logs = await db.select(db.callLogs).get();
      expect(logs, hasLength(1));
      expect(logs.single.action, 'copy');

      final row = await (db.select(
        db.contacts,
      )..where((c) => c.id.equals(id))).getSingle();
      expect(row.callCount, 1);
      expect(row.lastCalledAt, isNotNull);
    });

    test('the count accumulates across actions', () async {
      final id = await addContact('Homestay', '1');
      for (final action in ['copy', 'dialer', 'call', 'whatsapp']) {
        await dao.logCall(contactId: id, tripId: tripId, action: action);
      }
      final row = await (db.select(
        db.contacts,
      )..where((c) => c.id.equals(id))).getSingle();
      expect(row.callCount, 4);
    });
  });

  group('other reads and writes', () {
    test('pinning moves a contact to the top', () async {
      await addContact('Aaa', '1');
      final z = await addContact('Zzz', '2');
      await dao.togglePin(z, true);

      expect(await namesFor(ContactFilter(tripId: tripId)), ['Zzz', 'Aaa']);
    });

    test('emergency contacts have their own stream', () async {
      await addContact('Homestay', '1');
      await addContact('Ambulance friend', '2', emergency: true);

      final items = await dao.watchTripEmergencyContacts(tripId).first;
      expect(items.map((c) => c.name), ['Ambulance friend']);
    });

    test('helplines are filtered to the trip countries', () async {
      Future<void> seed(String country, String number) => db
          .into(db.emergencyHelplines)
          .insert(
            EmergencyHelplinesCompanion.insert(
              countryCode: country,
              serviceType: 'all',
              label: 'All emergencies',
              number: number,
              sourceNote: 'test',
            ),
          );
      await seed('IN', '112');
      await seed('DE', '110');

      final lines = await dao.watchEmergencyHelplines(['IN']).first;
      expect(lines.map((h) => h.number), ['112']);
    });

    test('findByE164 is scoped to the trip when asked', () async {
      await addContact('Homestay', '+91 98560 41122', e164: '+919856041122');

      expect(await dao.findByE164('+919856041122'), isNotNull);
      expect(await dao.findByE164('+919856041122', tripId: tripId), isNotNull);
      expect(await dao.findByE164('+910000000000'), isNull);
    });

    test('deleting a contact removes it from the feed', () async {
      final id = await addContact('Homestay', '1');
      await dao.deleteContact(id);
      expect(await namesFor(ContactFilter(tripId: tripId)), isEmpty);
    });

    test('DELETING A CONTACT TAKES ITS CALL HISTORY WITH IT', () async {
      // The delete dialog promises this; a call log left pointing at nothing
      // would be a row the app can never show or clear.
      final id = await addContact('Homestay', '1');
      await dao.logCall(contactId: id, tripId: tripId, action: 'call');
      await dao.deleteContact(id);
      expect(await db.select(db.callLogs).get(), isEmpty);
    });
  });
}
