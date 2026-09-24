// test/readiness_test.dart — issue #20
//
// The acceptance criterion: a trip with one unconfirmed homestay reads
// not-ready; confirm it, reads ready.

import 'package:drift/drift.dart' hide isNull, isNotNull;
import 'package:drift/native.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:safarsathi/core/database/app_database.dart';
import 'package:safarsathi/features/contacts/data/contacts_dao.dart';
import 'package:safarsathi/features/trips/data/readiness.dart';
import 'package:safarsathi/features/trips/data/stay.dart';
import 'package:safarsathi/features/trips/data/trip_editor.dart';

void main() {
  late AppDatabase db;
  late TripEditor editor;
  late int tripId;

  setUp(() async {
    db = AppDatabase(NativeDatabase.memory());
    editor = TripEditor(db);
    tripId = await editor.createTrip(name: 'Meghalaya');
  });

  tearDown(() => db.close());

  Future<int> overnight(String name) =>
      editor.addStop(tripId, StopDraft(name: name, nights: 2));

  Future<int> passingThrough(String name) =>
      editor.addStop(tripId, StopDraft(name: name));

  Future<int> contact(
    int? stopId, {
    String name = 'Homestay',
    String category = ContactCategory.accommodation,
    bool confirmed = false,
    bool stay = false,
  }) async {
    final id = await db.into(db.contacts).insert(
      ContactsCompanion.insert(
        name: name,
        phoneRaw: '+91 90000 00001',
        tripId: Value(tripId),
        stopId: Value(stopId),
        category: Value(category),
        callConfirmed: Value(confirmed),
      ),
    );
    // Where you sleep is chosen, never inferred from what is saved.
    if (stay) await setStay(db, stopId!, id);
    return id;
  }

  Future<Readiness> check() => watchReadiness(db, tripId).first;

  test('THE ACCEPTANCE TEST: unconfirmed blocks, confirming clears', () async {
    final stop = await overnight('Shillong');
    final id = await contact(stop, stay: true);

    var readiness = await check();
    expect(readiness.isReady, isFalse);
    expect(readiness.blocking.single.stopName, 'Shillong');
    expect(
      readiness.blocking.single.label,
      'Call and confirm Homestay, where you are staying in Shillong.',
    );

    await db.contactsDao.markConfirmed(id);

    readiness = await check();
    expect(readiness.isReady, isTrue);
  });

  test('AN ABSENT NUMBER BLOCKS TOO, and says something different', () async {
    // Absence is the more dangerous case and the one apps usually say nothing
    // about at all.
    await overnight('Mawlynnong');

    final readiness = await check();
    expect(readiness.isReady, isFalse);
    final item = readiness.blocking.single;
    expect(item.missing, isTrue);
    expect(item.contactId, isNull);
    expect(item.label, 'No accommodation number for Mawlynnong.');
  });

  test('a stop you only pass through never blocks', () async {
    await passingThrough('Dawki');
    expect((await check()).isReady, isTrue);
  });

  test('a non-accommodation number does not clear a stop', () async {
    // A confirmed taxi driver is not a place to sleep.
    final stop = await overnight('Shillong');
    await contact(
      stop,
      name: 'Biren',
      category: ContactCategory.transport,
      confirmed: true,
    );

    final readiness = await check();
    expect(readiness.blocking.single.missing, isTrue);
  });

  test('a trip-wide number does not clear a specific stop', () async {
    await overnight('Shillong');
    await contact(null, category: ContactCategory.accommodation);

    expect((await check()).blocking.single.missing, isTrue);
  });

  test('SAVED IS NOT CHOSEN: options at a stop ask which one', () async {
    // A sheet of guest houses is a list of options. Found on the phone:
    // Bramhome showed as the Shillong bed for somebody staying elsewhere.
    final stop = await overnight('Shillong');
    await contact(stop, name: 'Bramhome', confirmed: true);
    await contact(stop, name: 'J P Guest House');

    final item = (await check()).blocking.single;
    expect(item.label,
        'Choose where you are staying in Shillong — 2 places are saved there.');
    expect(item.contactId, isNull);
  });

  test('even one saved option is a question, not an answer', () async {
    final stop = await overnight('Sohra');
    await contact(stop, name: 'Alpha Guest House');
    expect((await check()).blocking.single.label,
        'Choose where you are staying in Sohra — is it Alpha Guest House?');
  });

  test('A CONFIRMED NUMBER YOU ARE NOT STAYING AT CLEARS NOTHING', () async {
    final stop = await overnight('Shillong');
    await contact(stop, name: 'Bramhome', confirmed: true);
    final mine = await contact(stop, name: 'Mine', stay: true);

    final item = (await check()).blocking.single;
    expect(item.contactId, mine);
    expect(item.label, contains('Call and confirm Mine'));

    await db.contactsDao.markConfirmed(mine);
    expect((await check()).isReady, isTrue);
  });

  test('every overnight stop is checked independently', () async {
    final a = await overnight('Shillong');
    await overnight('Cherrapunji');
    await overnight('Dawki');
    await contact(a, confirmed: true, stay: true);

    final readiness = await check();
    expect(readiness.openCount, 2);
    expect(
      readiness.blocking.map((i) => i.stopName),
      ['Cherrapunji', 'Dawki'],
    );
  });

  test('a trip with no stops is ready', () async {
    expect((await check()).isReady, isTrue);
  });

  group('the checklist', () {
    test('a blocking item is written per open problem', () async {
      await overnight('Shillong');
      await overnight('Dawki');
      await syncReadinessChecklist(db, tripId);

      final items = await db.select(db.checklistItems).get();
      expect(items.length, 2);
      expect(items.every((i) => i.isBlocking), isTrue);
      expect(items.every((i) => i.isGenerated), isTrue);
      expect(items.every((i) => !i.isDone), isTrue);
    });

    test('syncing twice does not duplicate anything', () async {
      await overnight('Shillong');
      await syncReadinessChecklist(db, tripId);
      await syncReadinessChecklist(db, tripId);

      expect((await db.select(db.checklistItems).get()).length, 1);
    });

    test('an item stops blocking once its number is confirmed', () async {
      final stop = await overnight('Shillong');
      final id = await contact(stop, stay: true);
      await syncReadinessChecklist(db, tripId);

      await db.contactsDao.markConfirmed(id);
      await syncReadinessChecklist(db, tripId);

      final item = await db.select(db.checklistItems).getSingle();
      // Ticked off rather than deleted: something that got handled should
      // stay visible.
      expect(item.isDone, isTrue);
    });

    test('the label follows the problem as it changes', () async {
      final stop = await overnight('Shillong');
      await syncReadinessChecklist(db, tripId);
      expect(
        (await db.select(db.checklistItems).getSingle()).label,
        contains('No accommodation number'),
      );

      final id = await contact(stop);
      await syncReadinessChecklist(db, tripId);
      expect(
        (await db.select(db.checklistItems).getSingle()).label,
        contains('Choose where you are staying'),
      );

      await setStay(db, stop, id);
      await syncReadinessChecklist(db, tripId);
      expect(
        (await db.select(db.checklistItems).getSingle()).label,
        contains('Call and confirm'),
      );
    });

    test('REGENERATION NEVER DISCARDS A USER EDIT', () async {
      // The isUserEdited column exists for exactly this, and this is the
      // first generator to run, so the rule is settled here.
      final stop = await overnight('Shillong');
      await syncReadinessChecklist(db, tripId);

      final item = await db.select(db.checklistItems).getSingle();
      await (db.update(db.checklistItems)..where((i) => i.id.equals(item.id)))
          .write(
            const ChecklistItemsCompanion(
              label: Value('Ring Rina, she answers after 7pm'),
              isUserEdited: Value(true),
            ),
          );

      await contact(stop);
      await syncReadinessChecklist(db, tripId);

      final after = await db.select(db.checklistItems).getSingle();
      expect(after.label, 'Ring Rina, she answers after 7pm');
    });

    test('a user-edited item is not silently ticked off either', () async {
      final stop = await overnight('Shillong');
      final id = await contact(stop, stay: true);
      await syncReadinessChecklist(db, tripId);

      final item = await db.select(db.checklistItems).getSingle();
      await (db.update(db.checklistItems)..where((i) => i.id.equals(item.id)))
          .write(const ChecklistItemsCompanion(isUserEdited: Value(true)));

      await db.contactsDao.markConfirmed(id);
      await syncReadinessChecklist(db, tripId);

      expect((await db.select(db.checklistItems).getSingle()).isDone, isFalse);
    });

    test('a blocking item points at the contact it is about', () async {
      final stop = await overnight('Shillong');
      final id = await contact(stop, stay: true);
      await syncReadinessChecklist(db, tripId);

      final item = await db.select(db.checklistItems).getSingle();
      expect(item.contactId, id);
      expect(item.stopId, stop);
    });
  });
}
