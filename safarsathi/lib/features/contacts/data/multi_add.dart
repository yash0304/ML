// lib/features/contacts/data/multi_add.dart
//
// Several contacts at once, typed rather than imported — issue #10.
//
// The rows land through ContactsDao.insertBatch, the same function the file
// import uses. That is deliberate: insertBatch FORCES userEntered and
// callConfirmed = false on every row whatever the caller passes, so there is
// no argument this screen could get wrong. A second path that re-implemented
// the guard is how the invariant erodes.

import 'package:drift/drift.dart';
import 'package:phone_numbers_parser/phone_numbers_parser.dart' show IsoCode;

import '../../../core/database/app_database.dart';
import 'contacts_dao.dart';
import 'phone_normaliser.dart';

/// What a multi-add batch is called in the import history.
///
/// Compared through [ImportBatchSummaryTyped.wasTyped] rather than as a bare
/// string at each call site.
const typedBatchLabel = 'Typed in the app';

/// One line of the sheet, exactly as typed.
class MultiAddRow {
  final String name;
  final String phoneRaw;
  final String category;

  /// Set only when the number parsed and validated.
  final String? phoneE164;

  /// A sentence when the number looks wrong. Warns; never blocks.
  final String? warning;

  /// An entry already in the diary with this number.
  final String? duplicateOf;

  /// An earlier row of this same sheet with this number. Typing the same
  /// driver twice off a booking confirmation is easy, and the second one is
  /// the row you are looking at.
  final bool duplicateInSheet;

  const MultiAddRow({
    this.name = '',
    this.phoneRaw = '',
    this.category = ContactCategory.other,
    this.phoneE164,
    this.warning,
    this.duplicateOf,
    this.duplicateInSheet = false,
  });

  /// Nothing typed at all — the trailing row, and any row emptied again.
  bool get isBlank => name.trim().isEmpty && phoneRaw.trim().isEmpty;

  /// Has both halves, so it will be saved.
  bool get isReady => name.trim().isNotEmpty && phoneRaw.trim().isNotEmpty;

  /// Started but unfinished. Skipped rather than rejected.
  bool get isPartial => !isBlank && !isReady;

  String get missing => name.trim().isEmpty ? 'Needs a name' : 'Needs a number';

  MultiAddRow copyWith({
    String? name,
    String? phoneRaw,
    String? category,
    String? phoneE164,
    String? warning,
    String? duplicateOf,
    bool? duplicateInSheet,
    bool clearPhoneE164 = false,
    bool clearWarning = false,
    bool clearDuplicate = false,
  }) => MultiAddRow(
    name: name ?? this.name,
    phoneRaw: phoneRaw ?? this.phoneRaw,
    category: category ?? this.category,
    phoneE164: clearPhoneE164 ? null : (phoneE164 ?? this.phoneE164),
    warning: clearWarning ? null : (warning ?? this.warning),
    duplicateOf: clearDuplicate ? null : (duplicateOf ?? this.duplicateOf),
    duplicateInSheet: duplicateInSheet ?? this.duplicateInSheet,
  );
}

/// The whole sheet, and the arithmetic the screen puts on its button.
class MultiAddSheet {
  final List<MultiAddRow> rows;
  const MultiAddSheet(this.rows);

  /// A sheet always ends in one blank row, so there is always somewhere to
  /// type next and no button to hunt for.
  factory MultiAddSheet.empty() => const MultiAddSheet([MultiAddRow()]);

  List<MultiAddRow> get ready => [for (final r in rows) if (r.isReady) r];
  int get readyCount => ready.length;
  int get partialCount => rows.where((r) => r.isPartial).length;
  bool get hasAnything => rows.any((r) => !r.isBlank);
  bool get canSave => readyCount > 0;

  /// Re-runs the checks that depend on other rows, and keeps exactly one
  /// blank row at the end.
  MultiAddSheet settled() {
    final seen = <String, int>{};
    final out = <MultiAddRow>[];

    for (var i = 0; i < rows.length; i++) {
      final row = rows[i];
      final key = row.phoneE164;
      // Only a normalised number can be compared. Two differently-typed
      // spellings of the same number are the same number; two unparseable
      // strings are not necessarily anything.
      final dupeHere = key != null && seen.containsKey(key);
      if (key != null) seen.putIfAbsent(key, () => i);
      out.add(row.copyWith(duplicateInSheet: dupeHere));
    }

    while (out.length > 1 && out[out.length - 1].isBlank && out[out.length - 2].isBlank) {
      out.removeLast();
    }
    if (out.isEmpty || !out.last.isBlank) {
      // The new row inherits the category above it: most sessions are one
      // kind at a time, so setting it once and having it stick is the fast
      // path, and overriding a row costs one tap.
      out.add(MultiAddRow(category: out.isEmpty ? ContactCategory.other : out.last.category));
    }
    return MultiAddSheet(out);
  }
}

/// Normalises one row's number and looks it up in the diary.
Future<MultiAddRow> checkRow(
  MultiAddRow row, {
  required Future<Contact?> Function(String e164) findDuplicate,
  IsoCode country = IsoCode.IN,
}) async {
  if (row.phoneRaw.trim().isEmpty) {
    return row.copyWith(
      clearPhoneE164: true,
      clearWarning: true,
      clearDuplicate: true,
    );
  }

  final phone = PhoneNormaliser.normalise(row.phoneRaw, country: country);
  String? duplicate;
  if (phone.e164 != null) {
    final hit = await findDuplicate(phone.e164!);
    duplicate = hit?.name;
  }

  return MultiAddRow(
    name: row.name,
    phoneRaw: row.phoneRaw,
    category: row.category,
    phoneE164: phone.e164,
    warning: phone.warning,
    duplicateOf: duplicate,
    duplicateInSheet: row.duplicateInSheet,
  );
}

class MultiAddResult {
  final int added;
  final int skipped;
  const MultiAddResult({required this.added, required this.skipped});
}

/// Writes the ready rows as one batch.
///
/// Note what is NOT passed: no tier, no confirmation flag. Typing a number
/// does not make it work, and quickly typing eight is exactly when somebody
/// might assume otherwise.
Future<MultiAddResult> commitMultiAdd(
  AppDatabase db, {
  required int tripId,
  required MultiAddSheet sheet,
}) async {
  final rows = sheet.ready;
  if (rows.isEmpty) {
    return const MultiAddResult(added: 0, skipped: 0);
  }

  final entries = [
    for (final r in rows)
      ContactsCompanion.insert(
        name: r.name.trim(),
        phoneRaw: r.phoneRaw.trim(),
        tripId: Value(tripId),
        phoneE164: Value(r.phoneE164),
        category: Value(r.category),
      ),
  ];

  await db.contactsDao.insertBatch(
    entries,
    ImportBatchesCompanion.insert(
      fileName: typedBatchLabel,
      tripId: Value(tripId),
      rowsImported: Value(rows.length),
      rowsSkipped: Value(sheet.partialCount),
    ),
  );

  return MultiAddResult(added: rows.length, skipped: sheet.partialCount);
}
