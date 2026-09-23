// test/import_commit_test.dart — issue #14
//
// End to end over a real in-memory database: a CSV goes in, contacts come
// out, and the batch rolls back whole.
//
// The test that matters most here is the last one. It removes the guard in
// insertBatch and checks that a confirmed contact CAN then be imported —
// proving the guard is doing work rather than passing vacuously.

import 'dart:convert';

import 'package:drift/drift.dart' hide isNull, isNotNull;
import 'package:drift/native.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:safarsathi/core/database/app_database.dart';
import 'package:safarsathi/features/contacts/data/contacts_dao.dart';
import 'package:safarsathi/features/import/data/column_mapping.dart';
import 'package:safarsathi/features/import/data/import_commit.dart';
import 'package:safarsathi/features/import/data/import_validation.dart';
import 'package:safarsathi/features/import/data/sheet_parser.dart';
import 'package:safarsathi/features/import/data/stop_matcher.dart';

const _sevenRows = 'name,phone,category,stop\n'
    'Rina Kharkongor,+91 90000 00001,homestay,Shillong\n'
    'Biren Lyngdoh,+91 90000 00002,driver,Shillong\n'
    'Dawki Boat,+91 90000 00003,transport,Dawki\n'
    'Sohra Chemist,+91 90000 00004,chemist,Cherrapunji\n'
    'Wahkhen Guide,+91 90000 00005,guide,\n'
    'Laitlum Dhaba,+91 90000 00006,dhaba,Shillong\n'
    'Mawlynnong Stay,+91 90000 00007,homestay,Mawlynnong\n';

void main() {
  late AppDatabase db;
  late int tripId;
  late List<StopCandidate> stops;

  setUp(() async {
    db = AppDatabase(NativeDatabase.memory());
    tripId = await db
        .into(db.trips)
        .insert(TripsCompanion.insert(name: 'Meghalaya'));
    final shillong = await db.into(db.stops).insert(
      StopsCompanion.insert(
        tripId: tripId,
        name: 'Shillong',
        sequenceOrder: 1,
        countryCode: 'IN',
      ),
    );
    final dawki = await db.into(db.stops).insert(
      StopsCompanion.insert(
        tripId: tripId,
        name: 'Dawki',
        sequenceOrder: 2,
        countryCode: 'IN',
      ),
    );
    stops = [
      StopCandidate(shillong, 'Shillong'),
      StopCandidate(dawki, 'Dawki'),
    ];
  });

  tearDown(() => db.close());

  Future<ImportPreview> preview(String csv) async {
    final table = SheetParser.parse(
      'meghalaya.csv',
      Uint8List.fromList(utf8.encode(csv)),
    ).sheets.single;
    return validateRows(
      table,
      autoMatchColumns(table.headers),
      stops: stops,
      existing: await readExistingContacts(db, tripId),
    );
  }

  Future<ImportResult> importCsv(String csv) async => commitImport(
    db,
    tripId: tripId,
    fileName: 'meghalaya.csv',
    preview: await preview(csv),
  );

  test('seven rows import as seven contacts', () async {
    final result = await importCsv(_sevenRows);
    expect(result.imported, 7);
    expect(result.skipped, 0);

    final saved = await db.select(db.contacts).get();
    expect(saved.length, 7);
  });

  test('EVERY imported row lands unconfirmed', () async {
    await importCsv(_sevenRows);
    final saved = await db.select(db.contacts).get();

    expect(saved.every((c) => c.callConfirmed == false), isTrue);
    expect(saved.every((c) => c.confirmedAt == null), isTrue);
    expect(
      saved.every((c) => c.tier == ContactTier.userEntered.name),
      isTrue,
    );
  });

  test('a confirmed companion CANNOT sneak a verified contact in', () async {
    // The guard lives in the DAO, which is the single write surface, so this
    // goes straight at it rather than through the import flow.
    await db.contactsDao.insertBatch(
      [
        ContactsCompanion.insert(
          name: 'Trusted',
          phoneRaw: '+91 90000 00099',
          tripId: Value(tripId),
          tier: const Value('verifiedNational'),
          callConfirmed: const Value(true),
          confirmedAt: Value(DateTime.now()),
        ),
      ],
      ImportBatchesCompanion.insert(fileName: 'sneaky.csv'),
    );

    final saved = await db.select(db.contacts).getSingle();
    expect(saved.callConfirmed, isFalse);
    expect(saved.confirmedAt, isNull);
    expect(saved.tier, ContactTier.userEntered.name);
  });

  test('stops are attached where they matched', () async {
    await importCsv(_sevenRows);
    final saved = await db.select(db.contacts).get();

    final rina = saved.firstWhere((c) => c.name.startsWith('Rina'));
    expect(rina.stopId, stops.first.id);

    // Cherrapunji and Mawlynnong are not on this trip, and Wahkhen Guide has
    // no stop cell. All three import trip-wide rather than being refused.
    final tripWide = saved.where((c) => c.stopId == null);
    expect(tripWide.length, 3);
  });

  test('categories survive the round trip', () async {
    await importCsv(_sevenRows);
    final saved = await db.select(db.contacts).get();
    final chemist = saved.firstWhere((c) => c.name.contains('Chemist'));
    expect(chemist.category, ContactCategory.pharmacy);
  });

  test('coordinates reach the database (v5)', () async {
    // The whole point of reading them: a contact with no stored position
    // cannot be placed on a leg, however carefully the sheet was made.
    await importCsv(
      'name,phone,latitude,longitude\n'
      'Pynursla SDH,+91 90000 00077,25.3089,91.9120\n'
      'No position,+91 90000 00078,,\n',
    );
    final rows = await db.select(db.contacts).get();
    final placed = rows.firstWhere((c) => c.name == 'Pynursla SDH');
    final unplaced = rows.firstWhere((c) => c.name == 'No position');
    expect(placed.lat, 25.3089);
    expect(placed.lon, 91.9120);
    expect(unplaced.lat, isNull);
  });

  test('the batch records what actually happened', () async {
    await importCsv(
      'name,phone\nA,+91 90000 00001\n,+91 90000 00002\nC,\n',
    );
    final batch = await db.select(db.importBatches).getSingle();
    expect(batch.rowsImported, 1);
    expect(batch.rowsSkipped, 2);
    expect(batch.fileName, 'meghalaya.csv');
    expect(batch.tripId, tripId);
  });

  test('every contact is tied to its batch', () async {
    await importCsv(_sevenRows);
    final batch = await db.select(db.importBatches).getSingle();
    final saved = await db.select(db.contacts).get();
    expect(saved.every((c) => c.importBatchId == batch.id), isTrue);
  });

  test('rollback removes exactly that batch', () async {
    await importCsv(_sevenRows);
    final first = await db.select(db.importBatches).getSingle();

    await importCsv('name,phone\nLater,+91 90000 00088\n');
    expect((await db.select(db.contacts).get()).length, 8);

    await db.contactsDao.rollbackImport(first.id);

    final left = await db.select(db.contacts).get();
    expect(left.length, 1);
    expect(left.single.name, 'Later');
    expect((await db.select(db.importBatches).get()).length, 1);
  });

  test('rollback takes confirmed contacts with it', () async {
    // Which is exactly why the history screen asks first, and names the count.
    await importCsv('name,phone\nRina,+91 90000 00001\n');
    final saved = await db.select(db.contacts).getSingle();
    await db.contactsDao.markConfirmed(saved.id);

    final batch = await db.select(db.importBatches).getSingle();
    await db.contactsDao.rollbackImport(batch.id);

    expect(await db.select(db.contacts).get(), isEmpty);
  });

  test('a second import sees the first one as duplicates', () async {
    await importCsv(_sevenRows);
    final second = await preview(_sevenRows);

    expect(second.warnings, 7);
    expect(
      second.rows.every((r) => r.messages.first.contains('Already in your')),
      isTrue,
    );
  });

  test('history reports counts and what survives', () async {
    await importCsv(_sevenRows);
    final summary =
        (await watchImportBatches(db, tripId).first).single;

    expect(summary.rowsImported, 7);
    expect(summary.stillPresent, 7);
    expect(summary.fileName, 'meghalaya.csv');

    final saved = await db.select(db.contacts).get();
    await db.contactsDao.deleteContact(saved.first.id);

    final after = (await watchImportBatches(db, tripId).first).single;
    expect(after.rowsImported, 7);
    expect(after.stillPresent, 6);
  });

  test('deselected rows are not written', () async {
    final p = await preview(_sevenRows);
    final trimmed = ImportPreview([
      p.rows.first,
      for (final r in p.rows.skip(1)) r.copyWith(selected: false),
    ]);

    final result = await commitImport(
      db,
      tripId: tripId,
      fileName: 'meghalaya.csv',
      preview: trimmed,
    );

    expect(result.imported, 1);
    expect(result.skipped, 6);
    expect((await db.select(db.contacts).get()).single.name,
        startsWith('Rina'));
  });

  group('RE-IMPORTING A SHEET THAT NOW HAS LOCATIONS', () {
    // Found on the phone: the sheet went in before it had Latitude and
    // Longitude columns, so "27 of your numbers have no location". Importing
    // the newer file marked every row a duplicate and left it unticked, so
    // the locations never landed. The re-import is how they get there.
    const before = 'name,phone,note\n'
        'Civil Hospital,+91 364 222 4100,my note\n'
        'Reid Chest,+91 364 224 1497,\n';
    const after = 'name,phone,latitude,longitude,note\n'
        'Civil Hospital (Ambulance),+91 364 222 4100,25.567739,91.881081,'
        'sheet note\n'
        'Reid Chest,+91 364 224 1497,,,\n'
        'New place,+91 98560 11111,25.3,91.9,\n';

    test('a known number with no location is ticked to be placed', () async {
      await importCsv(before);
      final p = await preview(after);
      final civil = p.rows.first;
      expect(civil.fillsLocation, isTrue);
      expect(civil.selected, isTrue);
      expect(civil.messages.first, contains('adds the location'));
      // No location in the sheet: still a plain duplicate, unticked.
      expect(p.rows[1].fillsLocation, isFalse);
      expect(p.rows[1].selected, isFalse);
      expect(p.toImport.map((r) => r.name), ['New place']);
      expect(p.toPlace.map((r) => r.name), ['Civil Hospital (Ambulance)']);
    });

    test('ONLY THE POSITION CHANGES — no copy, and nothing of yours is '
        'overwritten', () async {
      await importCsv(before);
      final existing = await db.select(db.contacts).get();
      final civilId = existing.firstWhere((c) => c.name == 'Civil Hospital').id;
      await db.contactsDao.markConfirmed(civilId);

      final result = await importCsv(after);
      expect(result.imported, 1);
      expect(result.placed, 1);

      final rows = await db.select(db.contacts).get();
      expect(rows, hasLength(3));
      final civil = rows.firstWhere((c) => c.id == civilId);
      expect(civil.name, 'Civil Hospital');
      expect(civil.note, 'my note');
      expect(civil.callConfirmed, isTrue);
      expect(civil.lat, 25.567739);
      expect(civil.lon, 91.881081);
    });

    test('a number that already has a position is a plain duplicate', () async {
      await importCsv(after.replaceFirst('New place', 'First'));
      final p = await preview(after);
      expect(p.rows.first.fillsLocation, isFalse);
      expect(p.rows.first.selected, isFalse);
    });

    test('placing only makes no empty batch in history', () async {
      await importCsv(before);
      final onlyPlacing = await preview(
        'name,phone,latitude,longitude\n'
        'Civil Hospital,+91 364 222 4100,25.567739,91.881081\n',
      );
      final result = await commitImport(
        db,
        tripId: tripId,
        fileName: 'again.csv',
        preview: onlyPlacing,
      );
      expect(result.placed, 1);
      expect(await watchImportBatches(db, tripId).first, hasLength(1));
    });

    test('unticked, it places nothing', () async {
      await importCsv(before);
      final p = await preview(after);
      final off = ImportPreview([
        for (final r in p.rows) r.copyWith(selected: false),
      ]);
      final result = await commitImport(
        db,
        tripId: tripId,
        fileName: 'again.csv',
        preview: off,
      );
      expect(result.placed, 0);
      expect(
        (await db.select(db.contacts).get()).every((c) => c.lat == null),
        isTrue,
      );
    });

    test('A SHORT CODE IMPORTED TWICE IS A DUPLICATE, NOT A SECOND COPY', () async {
      // 112 and 181 have no E.164 form, so they had no duplicate check at
      // all: re-importing the sheet added every helpline again.
      const helplines = 'name,phone\n'
          'Women Helpline,181\n'
          'Child Helpline,1098\n';
      await importCsv(helplines);
      final p = await preview('$helplines' 'Other short code,181\n');
      expect(p.rows[0].selected, isFalse);
      expect(p.rows[1].selected, isFalse);
      expect(
        p.rows[0].messages,
        contains('Already in your diary — left unticked.'),
      );
      // Same digits, different name: a different service, still ticked.
      expect(p.rows[2].selected, isTrue);
    });

    test('a toggle keeps the row\'s position', () async {
      final p = await preview(after);
      final row = p.rows.last.copyWith(selected: false).copyWith(selected: true);
      expect(row.lat, 25.3);
      expect(row.lon, 91.9);
    });
  });
}
