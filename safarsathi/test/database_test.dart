import 'dart:io';
// Issue #2 acceptance. These assert against a real SQLite database rather
// than against generated Dart, because what matters is the schema sqlite
// actually creates.

// drift exports its own `isNull` expression helper, which collides with
// the matcher of the same name.
import 'package:drift/drift.dart' hide isNull;
import 'package:drift/native.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:safarsathi/core/database/app_database.dart';

void main() {
  late AppDatabase db;

  setUp(() {
    db = AppDatabase(NativeDatabase.memory());
  });

  tearDown(() async {
    await db.close();
  });

  Future<List<String>> tableNames() async {
    final rows = await db
        .customSelect(
          "SELECT name FROM sqlite_master "
          "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'",
        )
        .get();
    return rows.map((r) => r.read<String>('name')).toList()..sort();
  }

  Future<String> createSqlFor(String table) async {
    final row = await db
        .customSelect(
          "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
          variables: [Variable<String>(table)],
        )
        .getSingle();
    return row.read<String>('sql');
  }

  test('every table exists', () async {
    final names = await tableNames();
    expect(names, hasLength(19));
    for (final expected in [
      'trips',
      'stops',
      'legs',
      'pois',
      'poi_contacts',
      'contacts',
      'import_batches',
      'call_logs',
      'emergency_helplines',
      'checklist_items',
      'weather_snapshots',
      'travellers',
      'expenses',
      'expense_splits',
      'timeline_entries',
      'trusted_contacts',
      // Added at v3 for #35.
      'app_settings',
      // Added at v4 for #24.
      'map_tiles',
      // Added at v7: stops planned on the way.
      'planned_stops',
    ]) {
      expect(names, contains(expected));
    }
  });

  test('foreign keys are switched on', () async {
    // SQLite defaults this to OFF. Every ON DELETE CASCADE in the schema is
    // inert without the pragma in beforeOpen.
    final row = await db.customSelect('PRAGMA foreign_keys').getSingle();
    expect(row.data.values.first, 1);
  });

  test('deleting a trip cascades to its stops and contacts', () async {
    final tripId = await db
        .into(db.trips)
        .insert(TripsCompanion.insert(name: 'Meghalaya'));
    await db
        .into(db.stops)
        .insert(
          StopsCompanion.insert(
            tripId: tripId,
            name: 'Kongthong',
            sequenceOrder: 1,
            countryCode: 'IN',
          ),
        );
    await db
        .into(db.contacts)
        .insert(
          ContactsCompanion.insert(
            name: 'Homestay',
            phoneRaw: '+91 98560 41122',
            tripId: Value(tripId),
          ),
        );

    await (db.delete(db.trips)..where((t) => t.id.equals(tripId))).go();

    expect(await db.select(db.stops).get(), isEmpty);
    expect(await db.select(db.contacts).get(), isEmpty);
  });

  test('a contact cannot arrive already confirmed', () async {
    // THE CORE INVARIANT. Nothing reaches this table verified — not an
    // import, not a POI save, not a form.
    final id = await db
        .into(db.contacts)
        .insert(
          ContactsCompanion.insert(
            name: 'Guesthouse',
            phoneRaw: '+91 99540 22187',
          ),
        );
    final row = await (db.select(
      db.contacts,
    )..where((c) => c.id.equals(id))).getSingle();

    expect(row.tier, 'userEntered');
    expect(row.callConfirmed, isFalse);
    expect(row.confirmedAt, isNull);
  });

  test('seeding the same helpline twice is rejected', () async {
    // This is what makes first-launch seeding at #4 idempotent rather than
    // merely careful. Seeding runs again after a reinstall or a migration.
    Future<int> seed() => db
        .into(db.emergencyHelplines)
        .insert(
          EmergencyHelplinesCompanion.insert(
            countryCode: 'IN',
            serviceType: 'all',
            label: 'All emergencies',
            number: '112',
            sourceNote: '112.gov.in, Ministry of Home Affairs',
          ),
        );

    await seed();
    await expectLater(seed(), throwsA(isA<SqliteException>()));

    final rows = await db.select(db.emergencyHelplines).get();
    expect(rows, hasLength(1));
  });

  test('a weather snapshot cannot exist without its age', () async {
    // A snapshot without a cachedAt is a forecast pretending to be current.
    final sql = await createSqlFor('weather_snapshots');
    expect(sql, contains('cached_at'));
    expect(
      RegExp(r'cached_at[^,]*NOT NULL').hasMatch(sql),
      isTrue,
      reason: 'cachedAt must be non-nullable: $sql',
    );
  });

  test('expense splits sum exactly to the expense', () async {
    // Money is stored in minor units precisely so a three-way split of an
    // odd amount loses nothing.
    final tripId = await db
        .into(db.trips)
        .insert(TripsCompanion.insert(name: 'Meghalaya'));
    final people = <int>[];
    for (final name in ['You', 'Ankit', 'Priya']) {
      people.add(
        await db
            .into(db.travellers)
            .insert(TravellersCompanion.insert(tripId: tripId, name: name)),
      );
    }

    const totalMinor = 320011; // ₹3,200.11 — deliberately not divisible by 3
    final expenseId = await db
        .into(db.expenses)
        .insert(
          ExpensesCompanion.insert(
            tripId: tripId,
            description: 'Taxi to Kongthong',
            amountMinor: totalMinor,
            paidById: people.first,
          ),
        );

    final base = totalMinor ~/ people.length;
    final remainder = totalMinor % people.length;
    for (var i = 0; i < people.length; i++) {
      await db
          .into(db.expenseSplits)
          .insert(
            ExpenseSplitsCompanion.insert(
              expenseId: expenseId,
              travellerId: people[i],
              shareMinor: base + (i < remainder ? 1 : 0),
            ),
          );
    }

    final splits = await db.select(db.expenseSplits).get();
    final sum = splits.fold<int>(0, (a, s) => a + s.shareMinor);
    expect(sum, totalMinor, reason: 'not one paisa may go missing');
  });

  test('one traveller cannot hold two shares of one expense', () async {
    final tripId = await db
        .into(db.trips)
        .insert(TripsCompanion.insert(name: 'Meghalaya'));
    final who = await db
        .into(db.travellers)
        .insert(TravellersCompanion.insert(tripId: tripId, name: 'Ankit'));
    final expenseId = await db
        .into(db.expenses)
        .insert(
          ExpensesCompanion.insert(
            tripId: tripId,
            description: 'Dinner',
            amountMinor: 86000,
            paidById: who,
          ),
        );

    Future<int> addShare() => db
        .into(db.expenseSplits)
        .insert(
          ExpenseSplitsCompanion.insert(
            expenseId: expenseId,
            travellerId: who,
            shareMinor: 43000,
          ),
        );

    await addShare();
    await expectLater(addShare(), throwsA(isA<SqliteException>()));
  });

  test('a POI attaches to a stop XOR a leg', () async {
    final tripId = await db
        .into(db.trips)
        .insert(TripsCompanion.insert(name: 'Meghalaya'));
    final stopId = await db
        .into(db.stops)
        .insert(
          StopsCompanion.insert(
            tripId: tripId,
            name: 'Kongthong',
            sequenceOrder: 1,
            countryCode: 'IN',
          ),
        );

    Future<int> insertPoi({int? stop, int? leg}) => db
        .into(db.pois)
        .insert(
          PoisCompanion.insert(
            tripId: tripId,
            name: 'Tea stall',
            category: 'restaurant',
            lat: 25.30,
            lon: 91.73,
            stopId: Value(stop),
            legId: Value(leg),
          ),
        );

    // Attached to a stop: fine.
    await insertPoi(stop: stopId);

    // Attached to neither answers no query, so it must be rejected.
    await expectLater(insertPoi(), throwsA(isA<SqliteException>()));

    // Attached to both answers two queries wrongly. Also rejected.
    final legId = await db
        .into(db.legs)
        .insert(
          LegsCompanion.insert(
            tripId: tripId,
            fromStopId: stopId,
            toStopId: stopId,
            sequenceOrder: 1,
          ),
        );
    await expectLater(
      insertPoi(stop: stopId, leg: legId),
      throwsA(isA<SqliteException>()),
    );
  });

  group('migrations', () {
    test('every version has a step and they run in ascending order', () async {
      // Not a behaviour test — a reminder. Steps that run out of order work
      // by luck until a later one depends on an earlier one's column, and
      // then they fail only on the phones that skipped a version.
      final db = AppDatabase(NativeDatabase.memory());
      addTearDown(db.close);
      expect(db.schemaVersion, 8);
    });

    test('A REAL v4 DATABASE UPGRADES TO v5 WITH ITS CONTACTS INTACT', () async {
      // The phone this ships to holds a v4 database with a diary in it. An
      // in-memory database is created at the current version and never
      // exercises onUpgrade at all, so this builds a genuine v4 file: create
      // it, remove what v5 added, stamp it v4, close it, and reopen it with
      // the current code. That is the exact path an install-over-the-top
      // takes.
      final dir = await Directory.systemTemp.createTemp('upgrade');
      addTearDown(() => dir.delete(recursive: true));
      final file = File('${dir.path}/v4.sqlite');

      final v4 = AppDatabase(NativeDatabase(file));
      final tripId = await v4
          .into(v4.trips)
          .insert(TripsCompanion.insert(name: 'Meghalaya'));
      await v4
          .into(v4.contacts)
          .insert(
            ContactsCompanion.insert(
              tripId: Value(tripId),
              name: 'Nazareth Hospital',
              phoneRaw: '0364 222 4052',
              phoneE164: const Value('+913642224052'),
              category: const Value('hospital'),
              callConfirmed: const Value(true),
            ),
          );
      await v4.customStatement('ALTER TABLE contacts DROP COLUMN lat');
      await v4.customStatement('ALTER TABLE contacts DROP COLUMN lon');
      // And what v6 added, or the v6 step finds its column already there.
      await v4.customStatement('ALTER TABLE stops DROP COLUMN stay_contact_id');
      await v4.customStatement('DROP TABLE planned_stops');
      await v4.customStatement('ALTER TABLE legs DROP COLUMN driver_contact_id');
      await v4.customStatement('ALTER TABLE legs DROP COLUMN vehicle_number');
      await v4.customStatement('PRAGMA user_version = 4');
      await v4.close();

      final v5 = AppDatabase(NativeDatabase(file));
      addTearDown(v5.close);

      final rows = await v5.select(v5.contacts).get();
      expect(rows, hasLength(1));
      expect(rows.single.name, 'Nazareth Hospital');
      // A confirmation is the most expensive thing in the diary — a phone
      // call made on purpose. It must survive the upgrade.
      expect(rows.single.callConfirmed, isTrue);
      expect(rows.single.lat, isNull);
      expect(rows.single.lon, isNull);

      // And the new columns are writable.
      await (v5.update(v5.contacts)..where((c) => c.id.equals(rows.single.id)))
          .write(const ContactsCompanion(lat: Value(25.57), lon: Value(91.88)));
      final after = await v5.select(v5.contacts).getSingle();
      expect(after.lat, 25.57);

      final version = await v5.customSelect('PRAGMA user_version').getSingle();
      expect(version.read<int>('user_version'), 8);
    });

    test('A REAL v5 DATABASE UPGRADES: stops kept, no stay chosen',
        () async {
      // The phone holds a v5 trip with stops, guest houses and positions.
      // The upgrade must keep every one and choose nothing: a stay picked
      // from whatever sorted first is the bug v6 exists to fix.
      final dir = await Directory.systemTemp.createTemp('upgrade6');
      addTearDown(() => dir.delete(recursive: true));
      final file = File('${dir.path}/v5.sqlite');

      final v5 = AppDatabase(NativeDatabase(file));
      final tripId = await v5
          .into(v5.trips)
          .insert(TripsCompanion.insert(name: 'Meghalaya'));
      final stopId = await v5.into(v5.stops).insert(
        StopsCompanion.insert(
          tripId: tripId,
          name: 'Shillong',
          sequenceOrder: 1,
          countryCode: 'IN',
          nights: const Value(2),
        ),
      );
      await v5.into(v5.contacts).insert(
        ContactsCompanion.insert(
          tripId: Value(tripId),
          stopId: Value(stopId),
          name: 'Bramhome Guest House',
          phoneRaw: '0364 222 6683',
          category: const Value('accommodation'),
          lat: const Value(25.575219),
          lon: const Value(91.882741),
        ),
      );
      await v5.customStatement('ALTER TABLE stops DROP COLUMN stay_contact_id');
      await v5.customStatement('DROP TABLE planned_stops');
      await v5.customStatement('ALTER TABLE legs DROP COLUMN driver_contact_id');
      await v5.customStatement('ALTER TABLE legs DROP COLUMN vehicle_number');
      await v5.customStatement('PRAGMA user_version = 5');
      await v5.close();

      final v6 = AppDatabase(NativeDatabase(file));
      addTearDown(v6.close);

      final stop = await v6.select(v6.stops).getSingle();
      expect(stop.name, 'Shillong');
      expect(stop.nights, 2);
      expect(stop.stayContactId, isNull);
      final contact = await v6.select(v6.contacts).getSingle();
      expect(contact.lat, 25.575219);

      await (v6.update(v6.stops)..where((s) => s.id.equals(stop.id)))
          .write(StopsCompanion(stayContactId: Value(contact.id)));
      expect((await v6.select(v6.stops).getSingle()).stayContactId,
          contact.id);

      final version = await v6.customSelect('PRAGMA user_version').getSingle();
      expect(version.read<int>('user_version'), 8);
    });

    test('v3 created app_settings with its key as the primary key', () async {
      final db = AppDatabase(NativeDatabase.memory());
      addTearDown(db.close);

      final columns = await db
          .customSelect("PRAGMA table_info('app_settings')")
          .get();
      final pk = columns.where((r) => r.read<int>('pk') > 0);

      expect(pk.map((r) => r.read<String>('name')), ['key']);
    });

    test('a setting round-trips and upserts rather than duplicating', () async {
      final db = AppDatabase(NativeDatabase.memory());
      addTearDown(db.close);

      await db
          .into(db.appSettings)
          .insertOnConflictUpdate(
            AppSettingsCompanion.insert(key: 'themeMode', value: 'lamp'),
          );
      await db
          .into(db.appSettings)
          .insertOnConflictUpdate(
            AppSettingsCompanion.insert(key: 'themeMode', value: 'day'),
          );

      final rows = await db.select(db.appSettings).get();
      expect(rows.length, 1);
      expect(rows.single.value, 'day');
    });
  });

  test('v4 created map_tiles with its identity unique, not just its id',
      () async {
    // Two providers' tiles must never be mistaken for each other, and the
    // same tile must never be counted twice.
    final db = AppDatabase(NativeDatabase.memory());
    addTearDown(db.close);

    await db.customStatement(
      "INSERT INTO map_tiles (provider, z, x, y, bytes) "
      "VALUES ('maptiler', 12, 1, 1, 100)",
    );

    await expectLater(
      db.customStatement(
        "INSERT INTO map_tiles (provider, z, x, y, bytes) "
        "VALUES ('maptiler', 12, 1, 1, 200)",
      ),
      throwsA(anything),
      reason: 'the same tile twice',
    );

    // A different provider, same coordinates, is a different tile.
    await db.customStatement(
      "INSERT INTO map_tiles (provider, z, x, y, bytes) "
      "VALUES ('stadia', 12, 1, 1, 100)",
    );
    expect((await db.select(db.mapTiles).get()).length, 2);
  });
}
