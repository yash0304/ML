// lib/core/database/app_database.dart
//
// SQLite via Drift. THE ONLY SOURCE OF TRUTH.
//
// The sync layer writes into this database and then disappears. Nothing in
// the presentation layer may call it, and no widget may make a network call.
// If you find yourself writing `http` inside a screen, stop.

import 'package:drift/drift.dart';
import 'package:drift_flutter/drift_flutter.dart';

import '../../features/contacts/data/contacts_dao.dart';
import 'tables.dart';

// Re-exported so a DAO can declare @DriftAccessor(tables: [...]) with a
// single import of this file.
export 'tables.dart';

part 'app_database.g.dart';

@DriftDatabase(
  tables: [
    Trips,
    Stops,
    Legs,
    PlannedStops,
    Pois,
    PoiContacts,
    Contacts,
    ImportBatches,
    CallLogs,
    EmergencyHelplines,
    ChecklistItems,
    WeatherSnapshots,
    Travellers,
    Expenses,
    ExpenseSplits,
    TimelineEntries,
    TrustedContacts,
    AppSettings,
    MapTiles,
  ],
  daos: [ContactsDao],
)
class AppDatabase extends _$AppDatabase {
  /// Pass an executor in tests; production opens the on-device file.
  AppDatabase([QueryExecutor? executor]) : super(executor ?? _openOnDevice());

  /// Exposed as a constant so a backup file can be checked against it
  /// without opening a database.
  static const currentSchemaVersion = 8;

  @override
  int get schemaVersion => currentSchemaVersion;

  @override
  MigrationStrategy get migration => MigrationStrategy(
    onCreate: (Migrator m) async {
      await m.createAll();
    },

    // NEVER edit a shipped migration, and never add a destructive fallback in
    // release builds. Each version's step is additive and stays as written.
    onUpgrade: (Migrator m, int from, int to) async {
      // STEPS RUN IN ASCENDING VERSION ORDER, always. These two happen to be
      // independent, but a v3 step that assumes v2's column exists would
      // silently fail if it ran first, and only on the phones that had been
      // sitting on v1.
      if (from < 2) {
        // v2 adds ChecklistItems.generatorKey. Matching a generated item by
        // its label meant a rename produced a duplicate: the generator found
        // no row for its rule and inserted a second copy beside the user's.
        // Existing rows get null and are re-keyed on the next regeneration.
        await m.addColumn(checklistItems, checklistItems.generatorKey);
      }
      if (from < 3) {
        // v3 adds the AppSettings key-value table (#35). Nothing reads it
        // before it exists, so an empty table is a complete migration.
        await m.createTable(appSettings);
      }
      if (from < 4) {
        // v4 adds the MapTiles index (#24). The tiles themselves are files;
        // an empty index simply means nothing has been downloaded yet.
        await m.createTable(mapTiles);
      }
      if (from < 5) {
        // v5 adds Contacts.lat/lon, so a number with a known position can be
        // placed on a leg at its distance along the road. Existing contacts
        // get null, which is the truth: nobody recorded where they were.
        await m.addColumn(contacts, contacts.lat);
        await m.addColumn(contacts, contacts.lon);
      }
      if (from < 6) {
        // v6 adds Stops.stayContactId: which saved stay is the one you sleep
        // at. Existing stops get null — not decided — rather than a guess
        // from whichever accommodation number sorted first.
        await m.addColumn(stops, stops.stayContactId);
      }
      if (from < 7) {
        // v7 adds PlannedStops: viewpoints and other stops on the way that
        // go home in the plan. A new table; nothing existing changes.
        await m.createTable(plannedStops);
      }
      if (from < 8) {
        // v8 adds who is driving each leg and the vehicle's number. Existing
        // legs get neither: nobody recorded them.
        await m.addColumn(legs, legs.driverContactId);
        await m.addColumn(legs, legs.vehicleNumber);
      }
    },

    beforeOpen: (details) async {
      // SQLite has foreign keys OFF by default. Every ON DELETE CASCADE in
      // tables.dart is inert without this line — deleting a trip would leave
      // orphaned stops, contacts and expenses behind, silently.
      await customStatement('PRAGMA foreign_keys = ON');
    },
  );
}

QueryExecutor _openOnDevice() => driftDatabase(name: 'safarsathi');
