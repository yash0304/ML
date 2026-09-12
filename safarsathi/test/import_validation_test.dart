// test/import_validation_test.dart — issue #13
//
// The acceptance criterion from the backlog, spelled out: feed a file with a
// blank name, a bad number and a duplicate; each flags correctly and the rest
// still imports.

import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:safarsathi/features/contacts/data/contacts_dao.dart';
import 'package:safarsathi/features/import/data/column_mapping.dart';
import 'package:safarsathi/features/import/data/import_validation.dart';
import 'package:safarsathi/features/import/data/sheet_parser.dart';
import 'package:safarsathi/features/import/data/stop_matcher.dart';

SheetTable table(String csv) => SheetParser.parse(
  'x.csv',
  Uint8List.fromList(utf8.encode(csv)),
).sheets.single;

ImportPreview run(
  String csv, {
  List<StopCandidate> stops = const [],
  ExistingContacts existing = ExistingContacts.empty,
}) {
  final t = table(csv);
  return validateRows(
    t,
    autoMatchColumns(t.headers),
    stops: stops,
    existing: existing,
  );
}

void main() {
  test('a clean row is ready and selected', () {
    final p = run('name,phone\nRina,+91 90000 00001\n');
    final row = p.rows.single;

    expect(row.state, RowState.ready);
    expect(row.selected, isTrue);
    expect(row.messages, isEmpty);
    expect(row.phoneE164, '+919000000001');
  });

  test('the mixed file: each problem flags, the rest still imports', () {
    final p = run(
      'name,phone\n'
      'Rina,+91 90000 00001\n' // clean
      ',+91 90000 00002\n' // no name
      'Bad,not-a-number\n' // will not normalise
      'Rina again,+91 90000 00001\n' // duplicate of row 2
      'Biren,+91 90000 00003\n', // clean
    );

    expect(p.total, 5);
    expect(p.ready, 2);
    expect(p.warnings, 2);
    expect(p.skipped, 1);
    // Everything except the nameless row will be written.
    expect(p.selected, 4);
    expect(p.toImport.length, 4);
  });

  test('a row with no name is skipped and cannot be selected', () {
    final row = run('name,phone\n,+91 90000 00001\n').rows.single;
    expect(row.state, RowState.skip);
    expect(row.selected, isFalse);
    expect(row.canSelect, isFalse);
    expect(row.copyWith(selected: true).selected, isFalse);
    expect(row.messages.single, contains('No name'));
  });

  test('a row with no number is skipped', () {
    final row = run('name,phone\nRina,\n').rows.single;
    expect(row.state, RowState.skip);
    expect(row.messages.single, contains('No number'));
  });

  test('a number that will not normalise still imports, as typed', () {
    // The amber dot already says the number is unverified. Refusing to save
    // it would lose the only record the user has of it.
    final row = run('name,phone\nRina,call the shop\n').rows.single;
    expect(row.state, RowState.warning);
    expect(row.selected, isTrue);
    expect(row.phoneRaw, 'call the shop');
    expect(row.phoneE164, isNull);
  });

  test('a duplicate of an existing diary entry warns but imports', () {
    final p = run(
      'name,phone\nRina,+91 90000 00001\n',
      existing: const ExistingContacts(
        e164: {'+919000000001'},
        squashedNames: {'rina'},
      ),
    );
    final row = p.rows.single;
    expect(row.state, RowState.warning);
    expect(row.selected, isTrue);
    expect(row.messages.single, contains('Already in your diary'));
  });

  test('a duplicate within the same file is caught', () {
    // A sheet listing the same homestay under two stops is the common case.
    final p = run(
      'name,phone\nRina,+91 90000 00001\nRina,+91 90000 00001\n',
    );
    expect(p.rows.first.state, RowState.ready);
    expect(p.rows.last.messages.single, contains('earlier in this file'));
  });

  test('the row number in a message is the line the user sees', () {
    final p = run('name,phone\nA,+91 90000 00001\n\n\n,+91 90000 00002\n');
    expect(p.rows.last.sourceRow, 5);
  });

  group('stops', () {
    const stops = [StopCandidate(1, 'Shillong'), StopCandidate(2, 'Dawki')];

    test('a matched stop is attached', () {
      final row = run(
        'name,phone,stop\nRina,+91 90000 00001,Shillong\n',
        stops: stops,
      ).rows.single;
      expect(row.stopId, 1);
      expect(row.state, RowState.ready);
    });

    test('an unmatched stop warns but still imports trip-wide', () {
      final row = run(
        'name,phone,stop\nRina,+91 90000 00001,Guwahati\n',
        stops: stops,
      ).rows.single;
      expect(row.stopId, isNull);
      expect(row.selected, isTrue);
      expect(row.messages.single, contains('Guwahati'));
    });
  });

  group('categories', () {
    test('the app\'s own names match', () {
      final row = run(
        'name,phone,category\nA,+91 90000 00001,accommodation\n',
      ).rows.single;
      expect(row.category, ContactCategory.accommodation);
    });

    test('the words people actually write match', () {
      for (final (word, expected) in [
        ('Homestay', ContactCategory.accommodation),
        ('driver', ContactCategory.transport),
        ('Dhaba', ContactCategory.restaurant),
        ('petrol pump', ContactCategory.fuel),
        ('chemist', ContactCategory.pharmacy),
      ]) {
        final row = run(
          'name,phone,category\nA,+91 90000 00001,$word\n',
        ).rows.single;
        expect(row.category, expected, reason: word);
      }
    });

    test('an unknown category files under Other and says so', () {
      final row = run(
        'name,phone,category\nA,+91 90000 00001,zzz\n',
      ).rows.single;
      expect(row.category, ContactCategory.other);
      expect(row.messages.single, contains('zzz'));
    });
  });

  group('flags', () {
    test('the yes spellings a sheet uses all read as yes', () {
      for (final word in ['y', 'Yes', 'TRUE', '1', 'haan']) {
        final row = run(
          'name,phone,emergency\nA,+91 90000 00001,$word\n',
        ).rows.single;
        expect(row.isEmergency, isTrue, reason: word);
      }
    });

    test('a blank cell never marks a contact as an emergency number', () {
      final row = run(
        'name,phone,emergency\nA,+91 90000 00001,\n',
      ).rows.single;
      expect(row.isEmergency, isFalse);
    });

    test('anything unrecognised reads as no, not as yes', () {
      final row = run(
        'name,phone,emergency\nA,+91 90000 00001,maybe\n',
      ).rows.single;
      expect(row.isEmergency, isFalse);
    });
  });
}
