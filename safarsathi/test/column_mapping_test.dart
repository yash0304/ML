// test/column_mapping_test.dart — issue #12

import 'package:flutter_test/flutter_test.dart';
import 'package:safarsathi/features/import/data/column_mapping.dart';

void main() {
  test('the template headers all match', () {
    final m = autoMatchColumns([
      'name',
      'phone',
      'category',
      'stop',
      'note',
      'emergency',
    ]);
    expect(m[ImportField.name], 0);
    expect(m[ImportField.phone], 1);
    expect(m[ImportField.category], 2);
    expect(m[ImportField.stopName], 3);
    expect(m[ImportField.note], 4);
    expect(m[ImportField.isEmergency], 5);
  });

  test('case, spacing and punctuation do not matter', () {
    final m = autoMatchColumns(['Full Name', 'PHONE-NUMBER', 'Place Name']);
    expect(m[ImportField.name], 0);
    expect(m[ImportField.phone], 1);
    expect(m[ImportField.stopName], 2);
  });

  test('the abbreviations a real sheet uses match', () {
    final m = autoMatchColumns(['Nam', 'Mobile No.', 'Remarks']);
    expect(m[ImportField.name], 0);
    expect(m[ImportField.phone], 1);
    expect(m[ImportField.note], 2);
  });

  test('no column is claimed by two fields', () {
    final m = autoMatchColumns(['Name', 'Phone', 'WhatsApp Number']);
    final claimed = m.values.toList();
    expect(claimed.toSet().length, claimed.length);
    expect(m[ImportField.phone], 1);
  });

  test('"no" inside "notes" does not steal the phone column', () {
    // Exact alias hits are resolved before any containment guessing. Without
    // that ordering, `no` matches inside `notes` and the phone column loses.
    final m = autoMatchColumns(['Name', 'Notes', 'Number']);
    expect(m[ImportField.note], 1);
    expect(m[ImportField.phone], 2);
  });

  test('a column nothing recognises is simply left unmapped', () {
    final m = autoMatchColumns(['Name', 'Phone', 'Aadhaar']);
    expect(m.values, isNot(contains(2)));
  });

  test('missing required fields are reported by name', () {
    expect(missingRequired(autoMatchColumns(['Phone'])), [ImportField.name]);
    expect(missingRequired(autoMatchColumns(['Name'])), [ImportField.phone]);
    expect(missingRequired(autoMatchColumns(['Name', 'Phone'])), isEmpty);
  });

  test('only name and phone are required', () {
    final required = ImportField.values.where((f) => f.isRequired).toSet();
    expect(required, {ImportField.name, ImportField.phone});
  });
}
