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

  /// Where the place is, when the sheet said. Both or neither.
  final double? lat;
  final double? lon;

  final RowState state;

  /// One sentence per problem, each naming the specific thing that is wrong.
  /// "Row 5 has no number" is actionable; "invalid row" is not.
  final List<String> messages;

  /// Whether this row will be written. Skipped rows are always false and
  /// cannot be turned on.
  final bool selected;

  /// The number is already in the diary WITHOUT a position, and this row has
  /// one. Ticked, it adds the position to that entry rather than making a
  /// second copy. Re-importing a sheet that gained Latitude and Longitude
  /// columns is how a diary gets placed on the road, so this is the point
  /// of re-importing, not a duplicate.
  final bool fillsLocation;

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
    this.lat,
    this.lon,
    this.messages = const [],
    this.selected = true,
    this.fillsLocation = false,
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
    // Carried through a toggle. Before this, ticking a row off and on again
    // quietly dropped its position.
    lat: lat,
    lon: lon,
    fillsLocation: fillsLocation,
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

  /// Rows that become new diary entries.
  List<ValidatedRow> get toImport => [
    for (final r in rows)
      if (r.selected && !r.fillsLocation) r,
  ];

  /// Rows that only add a position to an entry already in the diary.
  List<ValidatedRow> get toPlace => [
    for (final r in rows)
      if (r.selected && r.fillsLocation) r,
  ];
}

/// What the diary already holds, passed in rather than queried here so the
/// validation stays pure and testable.
class ExistingContacts {
  final Set<String> e164;
  final Set<String> squashedNames;

  /// The subset of [e164] with no position saved.
  final Set<String> unplacedE164;

  /// Name and digits together, for numbers with no E.164 form — 112, 181,
  /// 1098. Without this a short code had no duplicate check at all, and
  /// re-importing a sheet added every helpline again.
  final Set<String> nameAndDigits;

  const ExistingContacts({
    required this.e164,
    required this.squashedNames,
    this.unplacedE164 = const {},
    this.nameAndDigits = const {},
  });

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

    // A DUPLICATE ARRIVES UNTICKED. It used to arrive ticked with a warning,
    // which meant importing the same sheet twice put every stay in the diary
    // twice — found on a phone as each homestay listed three times. The row
    // is still shown and can still be ticked; importing a copy is now a
    // choice rather than the default.
    final position = _position(
      at(ImportField.latitude),
      at(ImportField.longitude),
    );

    var duplicate = false;
    var fillsLocation = false;
    final key = phone.e164;
    if (key != null) {
      if (existing.e164.contains(key) &&
          !seenInFile.contains(key) &&
          position.lat != null &&
          existing.unplacedE164.contains(key)) {
        messages.add('Already in your diary with no location — this adds the '
            'location to it. No second copy is made.');
        fillsLocation = true;
      } else if (existing.e164.contains(key)) {
        messages.add('Already in your diary — left unticked.');
        duplicate = true;
      } else if (seenInFile.contains(key)) {
        messages.add('Appears earlier in this file too — left unticked.');
        duplicate = true;
      }
      seenInFile.add(key);
    } else {
      final nameKey = nameAndDigitsKey(name, phone.raw);
      if (existing.nameAndDigits.contains(nameKey)) {
        messages.add('Already in your diary — left unticked.');
        duplicate = true;
      } else if (seenInFile.contains(nameKey)) {
        messages.add('Appears earlier in this file too — left unticked.');
        duplicate = true;
      } else if (existing.squashedNames.contains(_squashName(name))) {
        messages.add('A contact with this name is already in your diary.');
      }
      seenInFile.add(nameKey);
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

    if (position.problem != null) messages.add(position.problem!);

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
        lat: position.lat,
        lon: position.lon,
        // A warning is information, not a veto: these rows arrive selected —
        // except a duplicate, which arrives unticked.
        state: messages.isEmpty ? RowState.ready : RowState.warning,
        messages: messages,
        selected: !duplicate,
        fillsLocation: fillsLocation,
      ),
    );
  }

  return ImportPreview(rows);
}

String? _orNull(String s) => s.isEmpty ? null : s;

/// A coordinate pair from two cells, or a sentence saying why not.
///
/// BOTH OR NEITHER, AND NEVER GUESSED. A latitude with no longitude is not
/// half a position, it is no position; keeping it would put the place on the
/// equator's meridian. And 0,0 is the Gulf of Guinea, which is where a blank
/// cell read as a number ends up — so it is refused rather than trusted.
/// A bad position costs the row its place on the road, never the row itself:
/// the name and number still import.
({double? lat, double? lon, String? problem}) _position(
  String latRaw,
  String lonRaw,
) {
  if (latRaw.isEmpty && lonRaw.isEmpty) {
    return (lat: null, lon: null, problem: null);
  }
  final lat = double.tryParse(latRaw.replaceAll(',', '.'));
  final lon = double.tryParse(lonRaw.replaceAll(',', '.'));
  const unplaced = 'Imported without a position, so it will not appear on '
      'the road between stops.';

  if (lat == null || lon == null) {
    return (
      lat: null,
      lon: null,
      problem: 'Could not read "$latRaw, $lonRaw" as a location. $unplaced',
    );
  }
  if (lat.abs() > 90 || lon.abs() > 180) {
    return (
      lat: null,
      lon: null,
      problem: '$lat, $lon is not a place on Earth. $unplaced',
    );
  }
  if (lat == 0 && lon == 0) {
    return (
      lat: null,
      lon: null,
      problem: '0, 0 is a blank read as a number, not a place. $unplaced',
    );
  }
  return (lat: lat, lon: lon, problem: null);
}

String _squashName(String s) =>
    s.toLowerCase().replaceAll(RegExp(r'[^a-z0-9]'), '');

/// The duplicate key for a number with no E.164 form: same name, same digits.
/// Both, because "181" alone is every women's helpline in the country.
String nameAndDigitsKey(String name, String phone) =>
    '${_squashName(name)}|${phone.replaceAll(RegExp(r'\D'), '')}';

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
