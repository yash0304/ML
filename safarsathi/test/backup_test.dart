// test/backup_test.dart — issue #56.
//
// A backup is only worth anything if it comes back exactly. The round-trip
// test is the one that matters; the exclusions are the ones that matter for
// safety.

import 'dart:convert';

import 'package:drift/drift.dart' hide isNull, isNotNull;
import 'package:drift/native.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:safarsathi/core/database/app_database.dart';
import 'package:safarsathi/features/backup/data/backup.dart';
import 'package:safarsathi/features/contacts/data/contacts_dao.dart';
import 'package:safarsathi/features/settings/data/settings.dart';
import 'package:safarsathi/features/trips/data/trip_editor.dart';

void main() {
  late AppDatabase db;
  late TripEditor editor;

  setUp(() {
    db = AppDatabase(NativeDatabase.memory());
    editor = TripEditor(db);
  });
  tearDown(() => db.close());

  /// A trip with something of every kind in it.
  Future<int> seedTrip() async {
    final tripId = await editor.createTrip(name: 'Meghalaya');
    final shillong = await editor.addStop(
      tripId,
      const StopDraft(name: 'Shillong', nights: 2),
    );
    await editor.addStop(
      tripId,
      const StopDraft(name: 'Cherrapunji', nights: 1),
    );

    await db.into(db.contacts).insert(
      ContactsCompanion.insert(
        tripId: Value(tripId),
        stopId: Value(shillong),
        name: 'Kongthong homestay',
        phoneRaw: '+91 90000 00001',
        phoneE164: const Value('+919000000001'),
        // Called, answered, marked. This is the bit that must survive.
        tier: Value(ContactTier.userVerified.name),
        callConfirmed: const Value(true),
      ),
    );
    await db.into(db.contacts).insert(
      ContactsCompanion.insert(
        tripId: Value(tripId),
        name: 'Sohra chemist',
        phoneRaw: '03637 000000',
        tier: Value(ContactTier.userEntered.name),
      ),
    );

    await db.into(db.checklistItems).insert(
      ChecklistItemsCompanion.insert(tripId: tripId, label: 'Headtorch'),
    );
    return tripId;
  }

  group('the round trip', () {
    test('A BACKUP RESTORED ONTO AN EMPTY APP IS THE SAME APP', () async {
      final tripId = await seedTrip();
      final before = await db.select(db.contacts).get();
      final json = await exportBackup(db);

      // The uninstall, in effect.
      final fresh = AppDatabase(NativeDatabase.memory());
      addTearDown(fresh.close);

      await restoreBackup(fresh, readBackup(json));

      final after = await fresh.select(fresh.contacts).get();
      expect(after, hasLength(before.length));
      expect(
        after.map((c) => c.name).toSet(),
        before.map((c) => c.name).toSet(),
      );
      expect((await fresh.select(fresh.trips).get()).single.name, 'Meghalaya');
      expect(await fresh.select(fresh.stops).get(), hasLength(2));
      expect(await fresh.select(fresh.checklistItems).get(), hasLength(1));
      expect(tripId, isPositive);
    });

    test('CONFIRMATIONS COME BACK, and this is deliberate', () async {
      // Everywhere else in this app, bulk entry is forced to unconfirmed.
      // A backup is the app's own record of calls the user made; dropping it
      // would force re-calling twenty places after replacing a phone, which
      // would make backups useless. ISSUE_56_Backup.md has the reasoning.
      await seedTrip();
      final json = await exportBackup(db);

      final fresh = AppDatabase(NativeDatabase.memory());
      addTearDown(fresh.close);
      await restoreBackup(fresh, readBackup(json));

      final restored = await fresh.select(fresh.contacts).get();
      final homestay = restored.firstWhere(
        (c) => c.name == 'Kongthong homestay',
      );
      expect(homestay.callConfirmed, isTrue);
      expect(homestay.tier, ContactTier.userVerified.name);

      final chemist = restored.firstWhere((c) => c.name == 'Sohra chemist');
      expect(chemist.callConfirmed, isFalse);
      expect(chemist.tier, ContactTier.userEntered.name);
    });

    test('ids and their foreign keys survive', () async {
      final tripId = await seedTrip();
      final json = await exportBackup(db);

      final fresh = AppDatabase(NativeDatabase.memory());
      addTearDown(fresh.close);
      await restoreBackup(fresh, readBackup(json));

      final trip = (await fresh.select(fresh.trips).get()).single;
      expect(trip.id, tripId);
      final stops = await fresh.select(fresh.stops).get();
      expect(stops.every((s) => s.tripId == tripId), isTrue);
      // The leg between the two stops points at real rows.
      final legs = await fresh.select(fresh.legs).get();
      expect(legs, hasLength(1));
      final ids = stops.map((s) => s.id).toSet();
      expect(ids.contains(legs.single.fromStopId), isTrue);
      expect(ids.contains(legs.single.toStopId), isTrue);
    });

    test('restoring REPLACES rather than merging', () async {
      await seedTrip();
      final json = await exportBackup(db);

      // Another trip arrives after the backup was taken.
      await editor.createTrip(name: 'Ladakh');
      expect(await db.select(db.trips).get(), hasLength(2));

      await restoreBackup(db, readBackup(json));

      final trips = await db.select(db.trips).get();
      expect(trips, hasLength(1));
      expect(trips.single.name, 'Meghalaya');
    });
  });

  group('what a backup refuses to carry', () {
    test('EMERGENCY NUMBERS ARE NEVER IN A BACKUP', () async {
      // The one exclusion that is not about size. These are seeded by the app
      // from sources recorded in sourceNote. A file must never be able to put
      // a number on the emergency screen.
      await db.into(db.emergencyHelplines).insert(
        EmergencyHelplinesCompanion.insert(
          countryCode: 'IN',
          serviceType: 'all',
          number: '112',
          label: 'All emergencies',
          sourceNote: '112.gov.in',
        ),
      );

      final json = await exportBackup(db);
      expect(json, isNot(contains('112')));
      expect(json, isNot(contains('emergencyHelplines')));
      expect(backupTableOrder, isNot(contains('emergencyHelplines')));
    });

    test('every excluded table really is excluded', () {
      // Keeps the documented list and the real one from drifting apart.
      for (final table in excludedFromBackup) {
        expect(
          backupTableOrder,
          isNot(contains(table)),
          reason: '$table must not be backed up',
        );
      }
    });

    test('downloaded places and forecasts are left behind', () async {
      final tripId = await seedTrip();
      final stopId = (await db.select(db.stops).get()).first.id;
      final legId = (await db.select(db.legs).get()).single.id;
      await db.into(db.pois).insert(
        PoisCompanion.insert(
          tripId: tripId,
          // Exactly one of stop or leg — the schema's own CHECK constraint.
          legId: Value(legId),
          name: 'IOC Umroi',
          category: ContactCategory.fuel,
          lat: 25.4,
          lon: 91.8,
        ),
      );
      await db.into(db.weatherSnapshots).insert(
        WeatherSnapshotsCompanion.insert(
          stopId: stopId,
          forDate: DateTime(2026, 10, 2),
          condition: 'Heavy rain',
          cachedAt: DateTime(2026, 9, 30),
        ),
      );

      final json = await exportBackup(db);
      expect(json, isNot(contains('IOC Umroi')));
      expect(json, isNot(contains('Heavy rain')));
    });

    test("A LEG'S DOWNLOADED ROUTE IS NOT CARRIED EITHER", () async {
      final tripId = await seedTrip();
      await (db.update(db.legs)..where((l) => l.tripId.equals(tripId))).write(
        LegsCompanion(
          routePolyline: const Value('abc_polyline_data'),
          distanceKm: const Value(54),
          lastSyncedAt: Value(DateTime(2026, 9, 30)),
        ),
      );

      final json = await exportBackup(db);
      expect(json, isNot(contains('abc_polyline_data')));

      final fresh = AppDatabase(NativeDatabase.memory());
      addTearDown(fresh.close);
      await restoreBackup(fresh, readBackup(json));

      // The leg comes back reading as not downloaded, which is the truth:
      // the places that went with it are not there either.
      final leg = (await fresh.select(fresh.legs).get()).single;
      expect(leg.routePolyline, isNull);
      expect(leg.lastSyncedAt, isNull);
    });

    test('THE MAP KEY DOES NOT TRAVEL IN A BACKUP', () async {
      final settings = SettingsRepository(db);
      await settings.setMapTilerKey('a-real-looking-key');
      await settings.setCorridorKm(5);

      final json = await exportBackup(db);
      expect(json, isNot(contains('a-real-looking-key')));
      // But the settings that are not secrets do come along.
      expect(json, contains(SettingKeys.corridorKm));
    });

    test('restoring never clears a key already on the phone', () async {
      final settings = SettingsRepository(db);
      final json = await exportBackup(db);
      await settings.setMapTilerKey('typed-on-this-phone');

      await restoreBackup(db, readBackup(json));
      expect(await settings.readMapTilerKey(), 'typed-on-this-phone');
    });
  });

  group('reading a file', () {
    test('a summary is available before anything is written', () async {
      await seedTrip();
      final contents = readBackup(await exportBackup(db));

      expect(contents.trips, 1);
      expect(contents.stops, 2);
      expect(contents.contacts, 2);
      // The number the restore screen shows, because it is the one thing a
      // restore takes on trust.
      expect(contents.confirmedContacts, 1);
      expect(contents.checklistItems, 1);
      expect(contents.isEmpty, isFalse);
    });

    test('nonsense is refused with a sentence, not a crash', () {
      expect(
        () => readBackup('this is not json'),
        throwsA(
          isA<BackupException>().having(
            (e) => e.message,
            'message',
            contains('not even JSON'),
          ),
        ),
      );
    });

    test('some other JSON file is refused', () {
      expect(
        () => readBackup('{"hello": "world"}'),
        throwsA(
          isA<BackupException>().having(
            (e) => e.message,
            'message',
            contains('not a SafarSathi backup'),
          ),
        ),
      );
    });

    test('A BACKUP FROM A NEWER APP IS REFUSED, NOT PARTLY RESTORED', () {
      // Columns this build has never heard of cannot be restored faithfully,
      // and a silent partial restore is worse than no restore.
      final future = jsonEncode({
        'format': 'safarsathi.backup',
        'formatVersion': 1,
        'schemaVersion': AppDatabase.currentSchemaVersion + 1,
        'createdAt': DateTime(2026, 9, 13).toIso8601String(),
        'tables': const <String, dynamic>{},
      });

      expect(
        () => readBackup(future),
        throwsA(
          isA<BackupException>().having(
            (e) => e.message,
            'message',
            contains('newer version of the app'),
          ),
        ),
      );
    });

    test('a backup from an older schema is accepted', () {
      // Every migration in this project has been additive, so the missing
      // columns take their defaults.
      final old = jsonEncode({
        'format': 'safarsathi.backup',
        'formatVersion': 1,
        'schemaVersion': 1,
        'createdAt': DateTime(2026, 9, 13).toIso8601String(),
        'tables': const <String, dynamic>{},
      });
      expect(readBackup(old).schemaVersion, 1);
    });

    test('a damaged section says which one', () {
      final broken = jsonEncode({
        'format': 'safarsathi.backup',
        'formatVersion': 1,
        'schemaVersion': AppDatabase.currentSchemaVersion,
        'createdAt': DateTime(2026, 9, 13).toIso8601String(),
        'tables': {'contacts': 'not a list'},
      });
      expect(
        () => readBackup(broken),
        throwsA(
          isA<BackupException>().having(
            (e) => e.message,
            'message',
            contains('"contacts"'),
          ),
        ),
      );
    });

    test('an empty app makes a valid, empty backup', () async {
      final contents = readBackup(await exportBackup(db));
      expect(contents.isEmpty, isTrue);
      expect(contents.contacts, 0);
    });
  });

  group('the filename', () {
    test('carries the date, because that is the question asked of it', () {
      expect(
        backupFileName(now: DateTime(2026, 10, 2, 7, 5)),
        'safarsathi-2026-10-02-0705.json',
      );
    });
  });
}
