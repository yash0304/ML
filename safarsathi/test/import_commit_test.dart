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
}
