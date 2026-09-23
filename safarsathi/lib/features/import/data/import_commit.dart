// lib/features/import/data/import_commit.dart
//
// The only part of the import flow that writes.
//
// Everything up to here has been in memory: parsed, mapped, validated,
// previewed. This turns the selected rows into one transaction, and it is
// deliberately thin — the transaction and the tier guard both live in
// ContactsDao.insertBatch, which is the single write surface for contacts.

import 'package:drift/drift.dart';

import '../../../core/database/app_database.dart';
import '../../contacts/data/multi_add.dart' show typedBatchLabel;
import 'import_validation.dart';

class ImportResult {
  final int imported;
  final int skipped;

  /// Entries already in the diary that gained a position.
  final int placed;

  const ImportResult({
    required this.imported,
    required this.skipped,
    this.placed = 0,
  });
}

/// Writes the selected rows of [preview] as one batch.
///
/// Note what is NOT passed: no tier, no confirmation flag. `insertBatch`
/// forces every row to `userEntered` with `callConfirmed = false` whatever the
/// caller says, so there is no argument here that could get it wrong. A number
/// in a spreadsheet is an unverified number.
Future<ImportResult> commitImport(
  AppDatabase db, {
  required int tripId,
  required String fileName,
  String? sheetName,
  required ImportPreview preview,
}) async {
  final rows = preview.toImport;
  final toPlace = preview.toPlace;
  final skipped = preview.total - rows.length - toPlace.length;

  final entries = [
    for (final r in rows)
      ContactsCompanion.insert(
        name: r.name,
        phoneRaw: r.phoneRaw,
        tripId: Value(tripId),
        stopId: Value(r.stopId),
        phoneE164: Value(r.phoneE164),
        note: Value(r.note),
        category: Value(r.category),
        isEmergency: Value(r.isEmergency),
        hasWhatsapp: Value(r.hasWhatsapp),
        lat: Value(r.lat),
        lon: Value(r.lon),
      ),
  ];

  var placed = 0;
  await db.transaction(() async {
    // A sheet that only placed existing entries makes no batch: there would
    // be nothing in it to undo, and an empty line in Import history.
    if (entries.isNotEmpty) {
      await db.contactsDao.insertBatch(
        entries,
        ImportBatchesCompanion.insert(
          fileName: fileName,
          tripId: Value(tripId),
          sheetName: Value(sheetName),
          rowsImported: Value(rows.length),
          rowsSkipped: Value(skipped),
        ),
      );
    }

    // ONLY THE POSITION, AND ONLY WHERE THERE WAS NONE. The name, note, tier
    // and confirmation of the diary's entry are the person's, and a sheet
    // does not get to change them. `lat IS NULL` also means a position the
    // person set by hand is never overwritten.
    for (final r in toPlace) {
      placed += await (db.update(db.contacts)..where(
            (c) =>
                c.tripId.equals(tripId) &
                c.phoneE164.equals(r.phoneE164!) &
                c.lat.isNull(),
          ))
          .write(ContactsCompanion(lat: Value(r.lat), lon: Value(r.lon)));
    }
  });

  return ImportResult(imported: rows.length, skipped: skipped, placed: placed);
}

/// What the diary already holds, for duplicate detection in the preview.
///
/// Read once when the preview is built rather than queried per row: sixty rows
/// against sixty queries is sixty round trips for a set that fits in memory.
Future<ExistingContacts> readExistingContacts(
  AppDatabase db,
  int tripId,
) async {
  final rows = await (db.select(
    db.contacts,
  )..where((c) => c.tripId.equals(tripId))).get();

  return ExistingContacts(
    e164: {
      for (final c in rows)
        if (c.phoneE164 != null) c.phoneE164!,
    },
    nameAndDigits: {
      for (final c in rows)
        if (c.phoneE164 == null) nameAndDigitsKey(c.name, c.phoneRaw),
    },
    unplacedE164: {
      for (final c in rows)
        if (c.phoneE164 != null && c.lat == null) c.phoneE164!,
    },
    squashedNames: {
      for (final c in rows)
        c.name.toLowerCase().replaceAll(RegExp(r'[^a-z0-9]'), ''),
    },
  );
}

/// A batch as the history screen shows it.
class ImportBatchSummary {
  final int id;
  final String fileName;
  final String? sheetName;
  final int rowsImported;
  final int rowsSkipped;
  final DateTime importedAt;

  /// How many of that batch's contacts are still in the diary. A batch whose
  /// rows were deleted one by one should not offer to roll back nine numbers
  /// that are already gone.
  final int stillPresent;

  const ImportBatchSummary({
    required this.id,
    required this.fileName,
    required this.rowsImported,
    required this.rowsSkipped,
    required this.importedAt,
    required this.stillPresent,
    this.sheetName,
  });

  /// True for a multi-add session (#10) rather than a file.
  ///
  /// Both go through `insertBatch` and both can be rolled back, but the
  /// history should not tell somebody they "imported" numbers they sat and
  /// typed. Checked here so no call site compares the label itself.
  bool get wasTyped => fileName == typedBatchLabel;
}

/// Every batch on this trip, newest first, with a live count of surviving rows.
Stream<List<ImportBatchSummary>> watchImportBatches(AppDatabase db, int tripId) {
  final batches =
      db.select(db.importBatches)
        ..where((b) => b.tripId.equals(tripId))
        ..orderBy([
          (b) => OrderingTerm(
            expression: b.importedAt,
            mode: OrderingMode.desc,
          ),
        ]);

  return batches.watch().asyncMap((list) async {
    final out = <ImportBatchSummary>[];
    for (final b in list) {
      final count = await (db.selectOnly(db.contacts)
            ..addColumns([db.contacts.id.count()])
            ..where(db.contacts.importBatchId.equals(b.id)))
          .map((r) => r.read(db.contacts.id.count()) ?? 0)
          .getSingle();
      out.add(
        ImportBatchSummary(
          id: b.id,
          fileName: b.fileName,
          sheetName: b.sheetName,
          rowsImported: b.rowsImported,
          rowsSkipped: b.rowsSkipped,
          importedAt: b.importedAt,
          stillPresent: count,
        ),
      );
    }
    return out;
  });
}
