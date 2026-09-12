// lib/features/import/data/column_mapping.dart
//
// Which column in the user's sheet means what.
//
// Real sheets do not use the template's headers. They say "Mobile", "Contact
// No.", "Ph", "Place". Auto-matching is what makes import feel like the app
// understood the file rather than demanding a particular one.

enum ImportField {
  name,
  phone,
  category,
  note,
  stopName,
  isEmergency,
  whatsapp,
}

extension ImportFieldInfo on ImportField {
  /// Shown on the mapping screen.
  String get label => switch (this) {
    ImportField.name => 'Name',
    ImportField.phone => 'Phone',
    ImportField.category => 'Category',
    ImportField.note => 'Note',
    ImportField.stopName => 'Stop',
    ImportField.isEmergency => 'Emergency',
    ImportField.whatsapp => 'WhatsApp',
  };

  /// Only a name and a number make a contact. Everything else is decoration
  /// the sheet may or may not carry.
  bool get isRequired => this == ImportField.name || this == ImportField.phone;

  /// Header spellings seen in the wild, squashed the same way as the file's
  /// own headers before comparison. The Hinglish and abbreviated forms are
  /// here because that is what a homestay list actually looks like.
  List<String> get aliases => switch (this) {
    ImportField.name => const [
      'name',
      'contactname',
      'fullname',
      'person',
      'who',
      'nam',
    ],
    ImportField.phone => const [
      'phone',
      'phonenumber',
      'mobile',
      'mobileno',
      'mobilenumber',
      'contact',
      'contactno',
      'contactnumber',
      'number',
      'no',
      'ph',
      'cell',
      'tel',
      'telephone',
      'whatsappnumber',
    ],
    ImportField.category => const [
      'category',
      'type',
      'kind',
      'tag',
      'group',
    ],
    ImportField.note => const [
      'note',
      'notes',
      'remark',
      'remarks',
      'comment',
      'comments',
      'detail',
      'details',
      'address',
    ],
    ImportField.stopName => const [
      'stop',
      'stopname',
      'place',
      'placename',
      'location',
      'city',
      'town',
      'village',
      'destination',
      'where',
    ],
    ImportField.isEmergency => const [
      'emergency',
      'isemergency',
      'sos',
      'urgent',
      'critical',
    ],
    ImportField.whatsapp => const [
      'whatsapp',
      'haswhatsapp',
      'wa',
      'onwhatsapp',
    ],
  };
}

/// Lowercased with everything that is not a letter or a digit removed, so
/// `Phone Number`, `phone_number` and `PHONE-NUMBER` all collapse to the same
/// key. Matching on the raw string would fail on a single stray space.
String squashHeader(String header) =>
    header.toLowerCase().replaceAll(RegExp(r'[^a-z0-9]'), '');

/// Field to column index. A field the sheet does not carry is simply absent.
typedef ColumnMapping = Map<ImportField, int>;

/// Best guess at what each column means.
///
/// A column is never claimed twice: the first field to match an unclaimed
/// column wins, in enum declaration order. Without that rule a sheet with both
/// `Phone` and `WhatsApp Number` maps the same column to two fields, and the
/// user gets a duplicate they never asked for.
ColumnMapping autoMatchColumns(List<String> headers) {
  final squashed = [for (final h in headers) squashHeader(h)];
  final mapping = <ImportField, int>{};
  final claimed = <int>{};

  // Exact alias hits first, across all fields, before any prefix guessing —
  // otherwise `no` inside `notes` steals the column `phone` wanted.
  for (final field in ImportField.values) {
    for (var i = 0; i < squashed.length; i++) {
      if (claimed.contains(i) || squashed[i].isEmpty) continue;
      if (field.aliases.contains(squashed[i])) {
        mapping[field] = i;
        claimed.add(i);
        break;
      }
    }
  }

  // Then containment, for headers like `guest name (hindi)` or `phone 1`.
  for (final field in ImportField.values) {
    if (mapping.containsKey(field)) continue;
    for (var i = 0; i < squashed.length; i++) {
      if (claimed.contains(i) || squashed[i].isEmpty) continue;
      final hit = field.aliases.any(
        (a) => a.length >= 4 && squashed[i].contains(a),
      );
      if (hit) {
        mapping[field] = i;
        claimed.add(i);
        break;
      }
    }
  }

  return mapping;
}

/// The fields a mapping still needs before the preview can be built.
List<ImportField> missingRequired(ColumnMapping mapping) => [
  for (final f in ImportField.values)
    if (f.isRequired && !mapping.containsKey(f)) f,
];
