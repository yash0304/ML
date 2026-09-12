// lib/features/import/data/import_validation.dart
//
// Turning mapped rows into something the user can look at and decide about,
// before anything reaches the database.
//
// Nothing here writes. The whole validation pass is pure over a list of rows
// plus a snapshot of what is already in the diary, which is what makes the
// preview honest: what you see is exactly what the commit will do.

import 'package:phone_numbers_parser/phone_numbers_parser.dart' show IsoCode;

import '../../contacts/data/contacts_dao.dart';
import '../../contacts/data/phone_normaliser.dart';
import 'column_mapping.dart';
import 'sheet_parser.dart';
import 'stop_matcher.dart';

/// How a row will be treated.
enum RowState {
  /// Nothing to say. It will import.
  ready,

  /// It will import, and the user should look at it first.
  warning,

  /// There is nothing to save. Cannot be selected.
  skip,
}

class ValidatedRow {
  /// The line in the original file, as a spreadsheet numbers it.
  final int sourceRow;

  final String name;
  final String phoneRaw;
  final String? phoneE164;
  final String category;
  final String? note;
  final int? stopId;
  final String? stopName;
  final bool isEmergency;
  final bool hasWhatsapp;

  final RowState state;

  /// One sentence per problem, each naming the specific thing that is wrong.
  /// "Row 5 has no number" is actionable; "invalid row" is not.
  final List<String> messages;

  /// Whether this row will be written. Skipped rows are always false and
  /// cannot be turned on.
  final bool selected;

  const ValidatedRow({
    required this.sourceRow,
    required this.name,
    required this.phoneRaw,
    required this.state,
    this.phoneE164,
    this.category = ContactCategory.other,
    this.note,
    this.stopId,
    this.stopName,
    this.isEmergency = false,
    this.hasWhatsapp = false,
    this.messages = const [],
    this.selected = true,
  });

  bool get canSelect => state != RowState.skip;

  ValidatedRow copyWith({bool? selected}) => ValidatedRow(
    sourceRow: sourceRow,
    name: name,
    phoneRaw: phoneRaw,
    phoneE164: phoneE164,
    category: category,
    note: note,
    stopId: stopId,
    stopName: stopName,
    isEmergency: isEmergency,
    hasWhatsapp: hasWhatsapp,
    state: state,
    messages: messages,
    selected: state == RowState.skip ? false : (selected ?? this.selected),
  );
}

class ImportPreview {
  final List<ValidatedRow> rows;
  const ImportPreview(this.rows);

  int get total => rows.length;
  int get ready => rows.where((r) => r.state == RowState.ready).length;
  int get warnings => rows.where((r) => r.state == RowState.warning).length;
  int get skipped => rows.where((r) => r.state == RowState.skip).length;
  int get selected => rows.where((r) => r.selected).length;

  List<ValidatedRow> get toImport => [
    for (final r in rows)
      if (r.selected) r,
  ];
}

/// What the diary already holds, passed in rather than queried here so the
/// validation stays pure and testable.
class ExistingContacts {
  final Set<String> e164;
  final Set<String> squashedNames;
  const ExistingContacts({required this.e164, required this.squashedNames});

  static const empty = ExistingContacts(e164: {}, squashedNames: {});
}

/// Values a sheet uses for yes. Anything else is no — a blank cell must not
/// silently mark a contact as an emergency number.
const _truthy = {'y', 'yes', 'true', '1', 'haan', 'ha', 'x', 'sos'};

bool _isTruthy(String raw) => _truthy.contains(raw.trim().toLowerCase());

ImportPreview validateRows(
  SheetTable table,
  ColumnMapping mapping, {
  List<StopCandidate> stops = const [],
  ExistingContacts existing = ExistingContacts.empty,
  IsoCode country = IsoCode.IN,
}) {
  final rows = <ValidatedRow>[];

  // Duplicates are detected against the diary AND against earlier rows of the
  // same file. A sheet listing the same homestay under two stops is the
  // common case, not an edge case.
  final seenInFile = <String>{};

  for (var i = 0; i < table.rows.length; i++) {
    final cells = table.rows[i];
    final sourceRow = i < table.sourceRowNumbers.length
        ? table.sourceRowNumbers[i]
        : i + 2;

    String at(ImportField f) {
      final index = mapping[f];
      if (index == null || index >= cells.length) return '';
      return cells[index].trim();
    }

    final name = at(ImportField.name);
    final phoneRaw = at(ImportField.phone);
    final messages = <String>[];

    if (name.isEmpty && phoneRaw.isEmpty) {
      rows.add(
        ValidatedRow(
          sourceRow: sourceRow,
          name: '',
          phoneRaw: '',
          state: RowState.skip,
          selected: false,
          messages: const ['Nothing in this row.'],
        ),
      );
      continue;
    }
    if (name.isEmpty) {
      rows.add(
        ValidatedRow(
          sourceRow: sourceRow,
          name: '',
          phoneRaw: phoneRaw,
          state: RowState.skip,
          selected: false,
          messages: const ['No name. A number with no name is unusable.'],
        ),
      );
      continue;
    }
    if (phoneRaw.isEmpty) {
      rows.add(
        ValidatedRow(
          sourceRow: sourceRow,
          name: name,
          phoneRaw: '',
          state: RowState.skip,
          selected: false,
          messages: const ['No number.'],
        ),
      );
      continue;
    }

    final phone = PhoneNormaliser.normalise(phoneRaw, country: country);

    // A number that will not normalise STILL IMPORTS, carrying the raw text.
    // The diary already shows an amber dot on it, so nothing is claimed that
    // is not true, and the user keeps the only record they have of it.
    if (!phone.normalised) {
      messages.add('Could not read this as a phone number. Saved as typed.');
    }

    final key = phone.e164;
    if (key != null) {
      if (existing.e164.contains(key)) {
        messages.add('Already in your diary.');
      } else if (seenInFile.contains(key)) {
        messages.add('Appears earlier in this file too.');
      }
      seenInFile.add(key);
    } else if (existing.squashedNames.contains(_squashName(name))) {
      messages.add('A contact with this name is already in your diary.');
    }

    final stopRaw = at(ImportField.stopName);
    final match = stopRaw.isEmpty
        ? StopMatch.none
        : matchStop(stopRaw, stops);
    if (match.unmatched) {
      messages.add(
        'No stop on this trip called "$stopRaw". Importing trip-wide.',
      );
    }

    final categoryRaw = at(ImportField.category);
    final category = _category(categoryRaw);
    if (categoryRaw.isNotEmpty && category == null) {
      messages.add('Category "$categoryRaw" is not one we know. Filed under '
          'Other.');
    }

    rows.add(
      ValidatedRow(
        sourceRow: sourceRow,
        name: name,
        phoneRaw: phone.raw,
        phoneE164: phone.e164,
        category: category ?? ContactCategory.other,
        note: _orNull(at(ImportField.note)),
        stopId: match.stopId,
        stopName: match.stopId == null ? null : match.stopName,
        isEmergency: _isTruthy(at(ImportField.isEmergency)),
        hasWhatsapp: _isTruthy(at(ImportField.whatsapp)),
        // A warning is information, not a veto: these rows arrive selected.
        state: messages.isEmpty ? RowState.ready : RowState.warning,
        messages: messages,
      ),
    );
  }

  return ImportPreview(rows);
}

String? _orNull(String s) => s.isEmpty ? null : s;

String _squashName(String s) =>
    s.toLowerCase().replaceAll(RegExp(r'[^a-z0-9]'), '');

/// Matches a sheet's category text against the app's own list, on the same
/// squashed key the column matcher uses, plus the handful of words people
/// actually write.
String? _category(String raw) {
  if (raw.trim().isEmpty) return null;
  final key = squashHeader(raw);
  if (key.isEmpty) return null;

  for (final c in ContactCategory.all) {
    if (squashHeader(c) == key) return c;
    if (squashHeader(ContactCategory.labels[c] ?? '') == key) return c;
  }

  const synonyms = <String, String>{
    'hotel': ContactCategory.accommodation,
    'homestay': ContactCategory.accommodation,
    'stay': ContactCategory.accommodation,
    'guesthouse': ContactCategory.accommodation,
    'lodge': ContactCategory.accommodation,
    'resort': ContactCategory.accommodation,
    'driver': ContactCategory.transport,
    'taxi': ContactCategory.transport,
    'cab': ContactCategory.transport,
    'car': ContactCategory.transport,
    'mechanic': ContactCategory.transport,
    'food': ContactCategory.restaurant,
    'dhaba': ContactCategory.restaurant,
    'cafe': ContactCategory.restaurant,
    'doctor': ContactCategory.hospital,
    'clinic': ContactCategory.hospital,
    'med': ContactCategory.hospital,
    'chemist': ContactCategory.pharmacy,
    'medical': ContactCategory.pharmacy,
    'petrol': ContactCategory.fuel,
    'pump': ContactCategory.fuel,
    'petrolpump': ContactCategory.fuel,
    'diesel': ContactCategory.fuel,
    'local': ContactCategory.localContact,
    'friend': ContactCategory.localContact,
    'family': ContactCategory.localContact,
  };
  return synonyms[key];
}
