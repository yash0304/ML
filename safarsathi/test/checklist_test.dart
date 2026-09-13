// test/checklist_test.dart — issue #29
//
// The rule that is the whole issue: a manual edit survives regeneration.

import 'package:drift/drift.dart' hide isNull, isNotNull;
import 'package:drift/native.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:safarsathi/core/database/app_database.dart';
import 'package:safarsathi/features/checklist/data/checklist_dao.dart';
import 'package:safarsathi/features/checklist/data/checklist_generator.dart';
import 'package:safarsathi/features/trips/data/readiness.dart';
import 'package:safarsathi/features/trips/data/trip_editor.dart';

void main() {
  group('generation rules', () {
    Stop stop(String name, {int nights = 0, String tags = ''}) => Stop(
      id: name.hashCode,
      tripId: 1,
      name: name,
      sequenceOrder: 1,
      countryCode: 'IN',
      nights: nights,
      activityTags: tags,
    );

    test('a trip with no tags still packs the basics', () {
      final items = generatePackList([stop('Shillong', nights: 2)]);
      final labels = items.map((i) => i.label);

      expect(labels, contains('Toothbrush and paste'));
      expect(labels, contains('ID'));
      expect(labels, contains('Any medicines you take'));
    });

    test('tags add their own items', () {
      final items = generatePackList([
        stop('Sohra', nights: 2, tags: 'caves,rain'),
      ]);
      final labels = items.map((i) => i.label);

      expect(labels, contains('Headtorch'));
      expect(labels, contains('Dry bag for the phone'));
      expect(labels, isNot(contains('Sleeping bag')));
    });

    test('an item names the tags that produced it', () {
      final items = generatePackList([
        stop('Sohra', nights: 1, tags: 'caves'),
      ]);
      final torch = items.firstWhere((i) => i.label == 'Headtorch');
      expect(torch.sourceTags, {'caves'});
    });

    test('an item asked for twice is packed once, crediting both tags', () {
      final items = generatePackList([
        stop('Sohra', nights: 1, tags: 'caves'),
        stop('Camp', nights: 1, tags: 'camping'),
      ]);

      final torches = items.where((i) => i.label == 'Headtorch').toList();
      expect(torches.length, 1);
      expect(torches.single.sourceTags, {'caves', 'camping'});
    });

    test('a count scales with the nights carrying that tag, not the trip', () {
      final items = generatePackList([
        stop('Shillong', nights: 5, tags: 'city'),
        stop('Trek base', nights: 2, tags: 'trek'),
      ]);

      // Two nights of trekking, so two pairs of trekking socks — not seven.
      final socks = items.firstWhere((i) => i.label == 'Trekking socks');
      expect(socks.quantity, '2');
    });

    test('clothes scale with the whole trip, plus one', () {
      final items = generatePackList([
        stop('A', nights: 2),
        stop('B', nights: 1),
      ]);
      final clothes = items.firstWhere((i) => i.label == 'Changes of clothes');
      expect(clothes.quantity, '4');
    });

    test('things you pack one of carry no count', () {
      final items = generatePackList([stop('A', nights: 3)]);
      expect(items.firstWhere((i) => i.label == 'ID').quantity, isNull);
    });

    test('an unknown tag is ignored rather than crashing', () {
      final items = generatePackList([
        stop('A', nights: 1, tags: 'paragliding'),
      ]);
      expect(items, isNotEmpty);
    });
  });

  group('against the database', () {
    late AppDatabase db;
    late TripEditor editor;
    late ChecklistDao dao;
    late int tripId;

    setUp(() async {
      db = AppDatabase(NativeDatabase.memory());
      editor = TripEditor(db);
      dao = ChecklistDao(db);
      tripId = await editor.createTrip(name: 'Meghalaya');
      await editor.addStop(
        tripId,
        const StopDraft(
          name: 'Cherrapunji',
          nights: 2,
          activityTags: ['caves', 'rain'],
        ),
      );
    });

    tearDown(() => db.close());

    Future<List<ChecklistItem>> pack() async =>
        (await watchChecklist(db, tripId).first).pack;

    test('regenerating twice does not duplicate anything', () async {
      await regeneratePackList(db, tripId);
      final first = (await pack()).length;

      await regeneratePackList(db, tripId);
      expect((await pack()).length, first);
    });

    test('AN EDITED LABEL AND COUNT SURVIVE REGENERATION', () async {
      await regeneratePackList(db, tripId);
      final torch = (await pack()).firstWhere((i) => i.label == 'Headtorch');

      await dao.edit(torch.id, label: 'Headtorch + the red one', quantity: '2');
      await regeneratePackList(db, tripId);

      final after = await pack();
      expect(after.map((i) => i.label), contains('Headtorch + the red one'));
      expect(
        after.firstWhere((i) => i.label == 'Headtorch + the red one').quantity,
        '2',
      );
      // And the generator does not re-add the original alongside it.
      expect(after.where((i) => i.label == 'Headtorch'), isEmpty);
    });

    test('A REMOVED GENERATED ITEM STAYS REMOVED', () async {
      // Deleting the row would simply bring it back on the next run, so the
      // remove marks it edited and done instead.
      await regeneratePackList(db, tripId);
      final socks = (await pack()).firstWhere(
        (i) => i.label == 'Quick-dry trousers',
      );

      await dao.remove(socks);
      await regeneratePackList(db, tripId);

      final after = (await pack()).firstWhere(
        (i) => i.label == 'Quick-dry trousers',
      );
      expect(after.isDone, isTrue);
      expect(after.isUserEdited, isTrue);
    });

    test('an item the user wrote is really deleted', () async {
      final id = await dao.addOwn(tripId, 'Spare specs');
      final item = (await pack()).firstWhere((i) => i.id == id);
      await dao.remove(item);

      expect((await pack()).where((i) => i.id == id), isEmpty);
    });

    test('an item the user added survives regeneration', () async {
      await dao.addOwn(tripId, 'Spare specs');
      await regeneratePackList(db, tripId);
      expect((await pack()).map((i) => i.label), contains('Spare specs'));
    });

    test('TICKING IS NOT AN EDIT', () async {
      // Everyone ticks things off. Treating that as taking ownership would
      // freeze the whole list the first time someone packed a toothbrush.
      await regeneratePackList(db, tripId);
      final item = (await pack()).first;
      await dao.setDone(item.id, true);

      final after = (await pack()).firstWhere((i) => i.id == item.id);
      expect(after.isDone, isTrue);
      expect(after.isUserEdited, isFalse);
    });

    test('an item whose tag is gone is dropped', () async {
      await regeneratePackList(db, tripId);
      expect((await pack()).map((i) => i.label), contains('Headtorch'));

      final stop = (await editor.stopsOf(tripId)).single;
      await editor.updateStop(
        tripId,
        StopDraft(id: stop.id, name: stop.name, nights: 2),
      );
      await regeneratePackList(db, tripId);

      expect((await pack()).map((i) => i.label), isNot(contains('Headtorch')));
    });

    test('blocking items and pack items live in one list, separated',
        () async {
      await regeneratePackList(db, tripId);
      await syncReadinessChecklist(db, tripId);

      final view = await watchChecklist(db, tripId).first;
      expect(view.blocking, isNotEmpty);
      expect(view.pack, isNotEmpty);
      expect(view.blocking.every((i) => i.isBlocking), isTrue);
      expect(view.pack.every((i) => !i.isBlocking), isTrue);
    });

    test('regenerating the pack list does not disturb blocking items',
        () async {
      await syncReadinessChecklist(db, tripId);
      final before = (await watchChecklist(db, tripId).first).blocking.length;

      await regeneratePackList(db, tripId);
      expect((await watchChecklist(db, tripId).first).blocking.length, before);
    });

    test('progress counts both sections', () async {
      await regeneratePackList(db, tripId);
      await syncReadinessChecklist(db, tripId);

      var view = await watchChecklist(db, tripId).first;
      expect(view.progress, 0);
      expect(view.isReady, isFalse);

      for (final item in [...view.blocking, ...view.pack]) {
        await dao.setDone(item.id, true);
      }
      view = await watchChecklist(db, tripId).first;
      expect(view.progress, 1.0);
      expect(view.isReady, isTrue);
    });

    test('an empty list reads as zero progress, not complete', () async {
      final other = await editor.createTrip(name: 'Empty');
      final view = await watchChecklist(db, other).first;
      expect(view.total, 0);
      expect(view.progress, 0);
    });
  });

  group('schema v2', () {
    late AppDatabase db;

    setUp(() => db = AppDatabase(NativeDatabase.memory()));
    tearDown(() => db.close());

    test('generatorKey exists on the checklist table', () async {
      final columns = await db
          .customSelect("PRAGMA table_info('checklist_items')")
          .get();
      expect(
        columns.map((r) => r.read<String>('name')),
        contains('generator_key'),
      );
    });

    test('the schema version was bumped, not edited', () {
      // The rule from DECISIONS.md: a shipped migration is never changed.
      // Adding a column means a new version, and this is the reminder.
      expect(db.schemaVersion, 2);
    });

    test('A ROW WRITTEN BEFORE v2 IS ADOPTED, NOT DUPLICATED', () async {
      // Simulates what is already on Yash's phone: generated rows with a null
      // key. They must be matched by label once, then carry a key.
      final editor = TripEditor(db);
      final tripId = await editor.createTrip(name: 'Meghalaya');
      await editor.addStop(
        tripId,
        const StopDraft(name: 'Sohra', nights: 1, activityTags: ['caves']),
      );

      await db
          .into(db.checklistItems)
          .insert(
            ChecklistItemsCompanion.insert(
              tripId: tripId,
              label: 'Headtorch',
              sourceTags: const Value('caves'),
            ),
          );

      await regeneratePackList(db, tripId);

      final torches = (await watchChecklist(db, tripId).first).pack
          .where((i) => i.label == 'Headtorch')
          .toList();
      expect(torches.length, 1);
      expect(torches.single.generatorKey, 'Headtorch');
    });
  });
}
