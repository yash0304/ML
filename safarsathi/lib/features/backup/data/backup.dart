// lib/features/backup/data/backup.dart
//
// Getting the user's work off this phone, and back onto one — issue #56.
//
// This app's entire value is a directory typed by hand before a trip, and its
// only copy lives on one handset. A dropped phone in Meghalaya has exactly
// the same effect as an uninstall, and there is no network to have synced
// anything to.
//
// WHAT GOES IN IS THE SAME DISTINCTION `clearTripCache` ALREADY DRAWS: what
// you typed, not what came off the network. Places, forecasts, tiles and
// route lines are all re-downloadable and would bloat the file into
// uselessness. Emergency helplines are excluded for a different and much
// harder reason — see `_excluded` below.

import 'dart:convert';

import 'package:drift/drift.dart';

import '../../../core/database/app_database.dart';
import '../../settings/data/settings.dart';

/// The wrapper's own version, bumped when the file's shape changes rather
/// than when the database's does.
const backupFormatVersion = 1;
const backupFormatName = 'safarsathi.backup';

class BackupException implements Exception {
  final String message;
  const BackupException(this.message);
  @override
  String toString() => message;
}

/// What a backup says it holds, read before anything is written.
class BackupContents {
  final DateTime createdAt;
  final int schemaVersion;

  final int trips;
  final int stops;
  final int contacts;

  /// How many of those contacts the file claims were confirmed by calling
  /// them. Shown on the restore screen, because this is the one thing a
  /// restore takes on trust that no other import path does.
  final int confirmedContacts;

  final int checklistItems;
  final int expenses;
  final int callLogs;

  /// The parsed tables, kept so the restore does not re-parse.
  final Map<String, List<Map<String, dynamic>>> tables;

  const BackupContents({
    required this.createdAt,
    required this.schemaVersion,
    required this.trips,
    required this.stops,
    required this.contacts,
    required this.confirmedContacts,
    required this.checklistItems,
    required this.expenses,
    required this.callLogs,
    required this.tables,
  });

  bool get isEmpty => trips == 0 && contacts == 0;
}

/// Tables a backup deliberately does not carry, and why.
///
/// `emergencyHelplines` is the one that matters. Those numbers are seeded by
/// the app from sources recorded in `sourceNote`, and the whole trust model
/// rests on that. A FILE MUST NEVER BE ABLE TO PUT A NUMBER ON THE EMERGENCY
/// SCREEN — not by hand-editing the JSON, and not by sending somebody a
/// "backup" to restore. They are re-seeded on first launch anyway, so
/// excluding them costs nothing at all.
///
/// The rest is simply re-downloadable: `pois`, `poiContacts`,
/// `weatherSnapshots` and `mapTiles` all came off the network, and tiles alone
/// would make the file tens of megabytes.
const excludedFromBackup = {
  'emergencyHelplines',
  'pois',
  'poiContacts',
  'weatherSnapshots',
  'mapTiles',
};

/// Settings that stay on the phone they were typed on.
///
/// A backup is a file people mail to themselves and copy between handsets, and
/// a map key sitting inside one travels further than anybody intends.
/// Re-entering it is a single paste.
const _excludedSettings = {SettingKeys.mapTilerKey};

/// Everything the backup carries, in an order that satisfies the foreign keys
/// on the way in. Reversed, it satisfies them on the way out.
const backupTableOrder = [
  'trips',
  'stops',
  'legs',
  'importBatches',
  'contacts',
  'callLogs',
  'checklistItems',
  'travellers',
  'expenses',
  'expenseSplits',
  'timelineEntries',
  'trustedContacts',
  'appSettings',
];

/// Reads the whole database into one JSON document.
Future<String> exportBackup(AppDatabase db) async {
  final tables = <String, List<Map<String, dynamic>>>{};

  for (final name in backupTableOrder) {
    tables[name] = await _readTable(db, name);
  }

  return const JsonEncoder.withIndent('  ').convert({
    'format': backupFormatName,
    'formatVersion': backupFormatVersion,
    'schemaVersion': db.schemaVersion,
    'createdAt': DateTime.now().toUtc().toIso8601String(),
    'tables': tables,
  });
}

Future<List<Map<String, dynamic>>> _readTable(
  AppDatabase db,
  String name,
) async {
  switch (name) {
    case 'trips':
      return [for (final r in await db.select(db.trips).get()) r.toJson()];
    case 'stops':
      return [for (final r in await db.select(db.stops).get()) r.toJson()];
    case 'legs':
      // The route line, its distance and the sync stamp came off OSRM with
      // the places that are already excluded. A restored leg reads as not
      // downloaded, which is the truth.
      return [
        for (final r in await db.select(db.legs).get())
          r
              .copyWith(
                routePolyline: const Value(null),
                distanceKm: const Value(null),
                lastSyncedAt: const Value(null),
              )
              .toJson(),
      ];
    case 'importBatches':
      return [
        for (final r in await db.select(db.importBatches).get()) r.toJson(),
      ];
    case 'contacts':
      return [for (final r in await db.select(db.contacts).get()) r.toJson()];
    case 'callLogs':
      return [for (final r in await db.select(db.callLogs).get()) r.toJson()];
    case 'checklistItems':
      return [
        for (final r in await db.select(db.checklistItems).get()) r.toJson(),
      ];
    case 'travellers':
      return [for (final r in await db.select(db.travellers).get()) r.toJson()];
    case 'expenses':
      return [for (final r in await db.select(db.expenses).get()) r.toJson()];
    case 'expenseSplits':
      return [
        for (final r in await db.select(db.expenseSplits).get()) r.toJson(),
      ];
    case 'timelineEntries':
      return [
        for (final r in await db.select(db.timelineEntries).get()) r.toJson(),
      ];
    case 'trustedContacts':
      return [
        for (final r in await db.select(db.trustedContacts).get()) r.toJson(),
      ];
    case 'appSettings':
      return [
        for (final r in await db.select(db.appSettings).get())
          if (!_excludedSettings.contains(r.key)) r.toJson(),
      ];
    default:
      throw BackupException('Nothing knows how to back up "$name".');
  }
}

/// Parses and checks a backup file without touching the database.
BackupContents readBackup(String source) {
  final Object? decoded;
  try {
    decoded = jsonDecode(source);
  } on FormatException {
    throw const BackupException(
      'That is not a backup file — it is not even JSON.',
    );
  }

  if (decoded is! Map<String, dynamic>) {
    throw const BackupException('That file is not a SafarSathi backup.');
  }
  if (decoded['format'] != backupFormatName) {
    throw const BackupException(
      'That file is not a SafarSathi backup. Pick the .json this app wrote.',
    );
  }

  final schema = decoded['schemaVersion'];
  if (schema is! int) {
    throw const BackupException('That backup does not say which version '
        'wrote it, so it cannot be restored safely.');
  }
  // A NEWER FILE IS REFUSED. Columns this build has never heard of cannot be
  // restored faithfully, and a silent partial restore is the worst outcome
  // available — worse than not restoring at all.
  if (schema > AppDatabase.currentSchemaVersion) {
    throw BackupException(
      'That backup came from a newer version of the app (database $schema, '
      'this build understands ${AppDatabase.currentSchemaVersion}). Update '
      'the app first.',
    );
  }

  final rawTables = decoded['tables'];
  if (rawTables is! Map<String, dynamic>) {
    throw const BackupException('That backup has no tables in it.');
  }

  final tables = <String, List<Map<String, dynamic>>>{};
  for (final name in backupTableOrder) {
    final rows = rawTables[name];
    if (rows == null) {
      tables[name] = const [];
      continue;
    }
    if (rows is! List) {
      throw BackupException('The "$name" section of that backup is damaged.');
    }
    tables[name] = [
      for (final row in rows)
        if (row is Map<String, dynamic>)
          row
        else
          throw BackupException('A row in "$name" is damaged.'),
    ];
  }

  final contacts = tables['contacts']!;
  final createdAt = DateTime.tryParse('${decoded['createdAt']}');

  return BackupContents(
    createdAt: createdAt ?? DateTime.fromMillisecondsSinceEpoch(0),
    schemaVersion: schema,
    trips: tables['trips']!.length,
    stops: tables['stops']!.length,
    contacts: contacts.length,
    confirmedContacts: contacts
        .where((c) => c['call_confirmed'] == true || c['callConfirmed'] == true)
        .length,
    checklistItems: tables['checklistItems']!.length,
    expenses: tables['expenses']!.length,
    callLogs: tables['callLogs']!.length,
    tables: tables,
  );
}

/// What is on the phone right now, for the side-by-side on the restore
/// screen. A restore that lands on top of a newer trip is the case worth
/// being careful about.
Future<BackupContents> currentContents(AppDatabase db) async {
  final contacts = await db.select(db.contacts).get();
  return BackupContents(
    createdAt: DateTime.now(),
    schemaVersion: db.schemaVersion,
    trips: (await db.select(db.trips).get()).length,
    stops: (await db.select(db.stops).get()).length,
    contacts: contacts.length,
    confirmedContacts: contacts.where((c) => c.callConfirmed).length,
    checklistItems: (await db.select(db.checklistItems).get()).length,
    expenses: (await db.select(db.expenses).get()).length,
    callLogs: (await db.select(db.callLogs).get()).length,
    tables: const {},
  );
}

/// Wipes what the backup covers and writes it back, in one transaction.
///
/// REPLACES, NEVER MERGES. Merging would mean duplicate detection across nine
/// tables with foreign keys between them, and a half-merged database is worse
/// than either outcome. The screen says exactly what is about to be lost.
///
/// `callConfirmed` and the trust tier come back exactly as they were. That is
/// a deliberate exception to the rule `insertBatch` enforces everywhere else,
/// and the reasoning is in ISSUE_56_Backup.md: a backup is this app's own
/// record of calls you made, not a spreadsheet somebody sent you. The restore
/// screen states what it is trusting.
Future<void> restoreBackup(AppDatabase db, BackupContents backup) async {
  await db.transaction(() async {
    // Reverse order, so nothing is deleted while a row still points at it.
    for (final name in backupTableOrder.reversed) {
      await _clearTable(db, name);
    }
    for (final name in backupTableOrder) {
      await _writeTable(db, name, backup.tables[name] ?? const []);
    }
  });
}

Future<void> _clearTable(AppDatabase db, String name) async {
  switch (name) {
    case 'trips':
      await db.delete(db.trips).go();
    case 'stops':
      await db.delete(db.stops).go();
    case 'legs':
      await db.delete(db.legs).go();
    case 'importBatches':
      await db.delete(db.importBatches).go();
    case 'contacts':
      await db.delete(db.contacts).go();
    case 'callLogs':
      await db.delete(db.callLogs).go();
    case 'checklistItems':
      await db.delete(db.checklistItems).go();
    case 'travellers':
      await db.delete(db.travellers).go();
    case 'expenses':
      await db.delete(db.expenses).go();
    case 'expenseSplits':
      await db.delete(db.expenseSplits).go();
    case 'timelineEntries':
      await db.delete(db.timelineEntries).go();
    case 'trustedContacts':
      await db.delete(db.trustedContacts).go();
    case 'appSettings':
      await (db.delete(
        db.appSettings,
      )..where((s) => s.key.isNotIn(_excludedSettings.toList()))).go();
    default:
      throw BackupException('Nothing knows how to clear "$name".');
  }
}

Future<void> _writeTable(
  AppDatabase db,
  String name,
  List<Map<String, dynamic>> rows,
) async {
  if (rows.isEmpty) return;

  switch (name) {
    case 'trips':
      await db.batch(
        (b) => b.insertAll(db.trips, [for (final r in rows) Trip.fromJson(r)]),
      );
    case 'stops':
      await db.batch(
        (b) => b.insertAll(db.stops, [for (final r in rows) Stop.fromJson(r)]),
      );
    case 'legs':
      await db.batch(
        (b) => b.insertAll(db.legs, [for (final r in rows) Leg.fromJson(r)]),
      );
    case 'importBatches':
      await db.batch(
        (b) => b.insertAll(db.importBatches, [
          for (final r in rows) ImportBatche.fromJson(r),
        ]),
      );
    case 'contacts':
      await db.batch(
        (b) => b.insertAll(db.contacts, [
          for (final r in rows) Contact.fromJson(r),
        ]),
      );
    case 'callLogs':
      await db.batch(
        (b) => b.insertAll(db.callLogs, [
          for (final r in rows) CallLog.fromJson(r),
        ]),
      );
    case 'checklistItems':
      await db.batch(
        (b) => b.insertAll(db.checklistItems, [
          for (final r in rows) ChecklistItem.fromJson(r),
        ]),
      );
    case 'travellers':
      await db.batch(
        (b) => b.insertAll(db.travellers, [
          for (final r in rows) Traveller.fromJson(r),
        ]),
      );
    case 'expenses':
      await db.batch(
        (b) => b.insertAll(db.expenses, [
          for (final r in rows) Expense.fromJson(r),
        ]),
      );
    case 'expenseSplits':
      await db.batch(
        (b) => b.insertAll(db.expenseSplits, [
          for (final r in rows) ExpenseSplit.fromJson(r),
        ]),
      );
    case 'timelineEntries':
      await db.batch(
        (b) => b.insertAll(db.timelineEntries, [
          for (final r in rows) TimelineEntry.fromJson(r),
        ]),
      );
    case 'trustedContacts':
      await db.batch(
        (b) => b.insertAll(db.trustedContacts, [
          for (final r in rows) TrustedContact.fromJson(r),
        ]),
      );
    case 'appSettings':
      await db.batch(
        (b) => b.insertAll(db.appSettings, [
          for (final r in rows)
            if (!_excludedSettings.contains(r['key'])) AppSetting.fromJson(r),
        ]),
      );
    default:
      throw BackupException('Nothing knows how to restore "$name".');
  }
}

/// The filename a backup is offered under.
///
/// Dated, because the commonest question about a backup file is how old it is,
/// and a file manager showing six identically-named files answers nothing.
String backupFileName({DateTime? now}) {
  final d = (now ?? DateTime.now());
  String two(int n) => n.toString().padLeft(2, '0');
  return 'safarsathi-${d.year}-${two(d.month)}-${two(d.day)}-'
      '${two(d.hour)}${two(d.minute)}.json';
}

