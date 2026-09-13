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
  ],
  daos: [ContactsDao],
)
class AppDatabase extends _$AppDatabase {
  /// Pass an executor in tests; production opens the on-device file.
  AppDatabase([QueryExecutor? executor]) : super(executor ?? _openOnDevice());

  @override
  int get schemaVersion => 2;

  @override
  MigrationStrategy get migration => MigrationStrategy(
    onCreate: (Migrator m) async {
      await m.createAll();
    },

    // NEVER edit a shipped migration, and never add a destructive fallback in
    // release builds. Each version's step is additive and stays as written.
    onUpgrade: (Migrator m, int from, int to) async {
      if (from < 2) {
        // v2 adds ChecklistItems.generatorKey. Matching a generated item by
        // its label meant a rename produced a duplicate: the generator found
        // no row for its rule and inserted a second copy beside the user's.
        // Existing rows get null and are re-keyed on the next regeneration.
        await m.addColumn(checklistItems, checklistItems.generatorKey);
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
