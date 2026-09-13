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
  }) => db.into(db.contacts).insert(
    ContactsCompanion.insert(
      name: name,
      phoneRaw: '+91 90000 00001',
      tripId: Value(tripId),
      stopId: Value(stopId),
      category: Value(category),
      callConfirmed: Value(confirmed),
    ),
  );

  Future<Readiness> check() => watchReadiness(db, tripId).first;

  test('THE ACCEPTANCE TEST: unconfirmed blocks, confirming clears', () async {
    final stop = await overnight('Shillong');
    final id = await contact(stop);

    var readiness = await check();
    expect(readiness.isReady, isFalse);
    expect(readiness.blocking.single.stopName, 'Shillong');
    expect(
      readiness.blocking.single.label,
      'Call and confirm the Shillong accommodation number.',
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

  test('one confirmed number clears the stop, however many are unconfirmed',
      () async {
    final stop = await overnight('Shillong');
    await contact(stop, name: 'Old number');
    await contact(stop, name: 'New number', confirmed: true);

    expect((await check()).isReady, isTrue);
  });

  test('every overnight stop is checked independently', () async {
    final a = await overnight('Shillong');
    await overnight('Cherrapunji');
    await overnight('Dawki');
    await contact(a, confirmed: true);

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
      final id = await contact(stop);
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

      await contact(stop);
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
      final id = await contact(stop);
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
      final id = await contact(stop);
      await syncReadinessChecklist(db, tripId);

      final item = await db.select(db.checklistItems).getSingle();
      expect(item.contactId, id);
      expect(item.stopId, stop);
    });
  });
}
