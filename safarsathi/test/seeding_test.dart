// Issue #4 acceptance. The rules these pin are the ones where getting it
// wrong is dangerous rather than merely wrong.

import 'package:drift/drift.dart' hide isNull;
import 'package:drift/native.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:safarsathi/core/database/app_database.dart';
import 'package:safarsathi/core/database/seeding.dart';
import 'package:safarsathi/features/emergency/data/emergency_seed.dart';

void main() {
  late AppDatabase db;

  setUp(() {
    db = AppDatabase(NativeDatabase.memory());
  });

  tearDown(() async {
    await db.close();
  });

  Future<List<EmergencyHelpline>> seeded() =>
      db.select(db.emergencyHelplines).get();

  test('seeding twice does not duplicate a single row', () async {
    // Seeding runs again after a reinstall and after every migration.
    await seedReferenceData(db);
    final first = await seeded();

    await seedReferenceData(db);
    final second = await seeded();

    expect(second, hasLength(first.length));
    expect(
      second.map((r) => r.id).toSet(),
      first.map((r) => r.id).toSet(),
      reason: 're-seeding must not disturb ids',
    );
  });

  test('fourteen rows are seeded', () async {
    await seedReferenceData(db);
    expect(await seeded(), hasLength(14));
  });

  test('the four flagged numbers never reach the database', () async {
    // 1930, 1078, 1033 and 104 are widely cited on aggregator sites but were
    // not confirmed from a .gov.in source. 104 is state-operated and not live
    // everywhere. Issue #36 verifies them; until then they must not appear.
    await seedReferenceData(db);
    final numbers = (await seeded()).map((r) => r.number).toSet();

    for (final flagged in ['1930', '1078', '1033', '104']) {
      expect(
        numbers,
        isNot(contains(flagged)),
        reason: '$flagged is unverified and must not ship',
      );
    }
  });

  test('every seeded number carries a source the UI can show', () async {
    // THE NON-NEGOTIABLE. A number without provenance is a number the user
    // cannot judge.
    await seedReferenceData(db);
    for (final row in await seeded()) {
      expect(
        row.sourceNote.trim(),
        isNotEmpty,
        reason: '${row.number} (${row.label}) has no source',
      );
      expect(row.needsVerification, isFalse);
    }
  });

  test('the state-level list ships empty', () async {
    // No authoritative combined dataset exists across 28 states and 8 union
    // territories. Guessed coverage is worse than none. Issue #37 fills this
    // manually, per portal.
    await seedReferenceData(db);
    for (final row in await seeded()) {
      expect(row.regionCode, isNull);
    }
  });

  test('112 is present with its government source', () async {
    await seedReferenceData(db);
    final row = await (db.select(
      db.emergencyHelplines,
    )..where((h) => h.number.equals('112'))).getSingle();

    expect(row.label, 'All emergencies');
    expect(row.sourceNote, contains('112.gov.in'));
    expect(row.sourceUrl, 'https://112.gov.in');
  });

  test('a corrected source note reaches an existing install', () async {
    // Upsert rather than insert-or-ignore, so fixing provenance in a later
    // version is not stranded on devices that already seeded.
    await seedReferenceData(db);
    final before = await (db.select(
      db.emergencyHelplines,
    )..where((h) => h.number.equals('112'))).getSingle();

    // Simulate an older install carrying stale text.
    await (db.update(
      db.emergencyHelplines,
    )..where((h) => h.number.equals('112'))).write(
      const EmergencyHelplinesCompanion(
        sourceNote: Value('stale text from an older version'),
      ),
    );

    await seedReferenceData(db);
    final after = await (db.select(
      db.emergencyHelplines,
    )..where((h) => h.number.equals('112'))).getSingle();

    expect(after.sourceNote, contains('112.gov.in'));
    expect(after.id, before.id, reason: 'the row is updated, not replaced');
    expect(await seeded(), hasLength(14));
  });

  test('the seed file itself keeps the flagged entries on record', () async {
    // They stay in the file so a future session can see they were considered
    // and rejected, rather than quietly forgotten.
    final flagged = emergencySeed.where((h) => h.needsVerification).toList();
    expect(flagged, hasLength(4));
    expect(flagged.map((h) => h.number).toSet(), {
      '1930',
      '1078',
      '1033',
      '104',
    });
  });

  test('no two seeded entries collide on the natural key', () async {
    // 102 and 108 are both ambulance lines, which is legitimate. Two rows
    // sharing country + number + service would not be.
    final keys = emergencySeed
        .where((h) => !h.needsVerification)
        .map((h) => '${h.countryCode}|${h.number}|${h.serviceType}')
        .toList();
    expect(keys.toSet(), hasLength(keys.length));
  });
}
