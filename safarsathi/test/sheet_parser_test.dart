// test/sheet_parser_test.dart — issue #11
//
// The parser takes bytes, so every one of these runs in plain Dart with a
// string literal. No file picker, no platform channel, no fixture files.

import 'dart:convert';
import 'dart:typed_data';

import 'package:excel/excel.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:safarsathi/features/import/data/sheet_parser.dart';

Uint8List csv(String text) => Uint8List.fromList(utf8.encode(text));

void main() {
  group('CSV', () {
    test('reads a header and its rows', () {
      final book = SheetParser.parse(
        'contacts.csv',
        csv('name,phone\nRina,+91 90000 00001\nBiren,+91 90000 00002\n'),
      );
      final sheet = book.sheets.single;

      expect(sheet.headers, ['name', 'phone']);
      expect(sheet.rowCount, 2);
      expect(sheet.rows.first, ['Rina', '+91 90000 00001']);
      expect(book.needsSheetChoice, isFalse);
    });

    test('row numbers survive blank rows, so "row 5" means row 5', () {
      final book = SheetParser.parse(
        'contacts.csv',
        csv('name,phone\nA,1\n\n\nD,4\n'),
      );
      final sheet = book.sheets.single;

      expect(sheet.rowCount, 2);
      // Line 2 is A. Lines 3 and 4 are blank. Line 5 is D.
      expect(sheet.sourceRowNumbers, [2, 5]);
    });

    test('a short row pads instead of throwing', () {
      final book = SheetParser.parse(
        'c.csv',
        csv('name,phone,note\nRina,+91 90000 00001\n'),
      );
      expect(book.sheets.single.rows.single, ['Rina', '+91 90000 00001', '']);
    });

    test('a long row is truncated to the header width', () {
      final book = SheetParser.parse('c.csv', csv('name,phone\nA,1,2,3\n'));
      expect(book.sheets.single.rows.single, ['A', '1']);
    });

    test('a trailing comma does not create a phantom column', () {
      final book = SheetParser.parse('c.csv', csv('name,phone,\nA,1,\n'));
      expect(book.sheets.single.headers, ['name', 'phone']);
    });

    test('leading blank lines do not become the header', () {
      final book = SheetParser.parse('c.csv', csv('\n\nname,phone\nA,1\n'));
      final sheet = book.sheets.single;
      expect(sheet.headers, ['name', 'phone']);
      expect(sheet.sourceRowNumbers, [4]);
    });

    test('windows line endings read the same as unix ones', () {
      final book = SheetParser.parse(
        'c.csv',
        csv('name,phone\r\nA,1\r\n'),
      );
      expect(book.sheets.single.rows.single, ['A', '1']);
    });

    test('a file with nothing in it says so in a sentence', () {
      expect(
        () => SheetParser.parse('c.csv', csv('\n\n')),
        throwsA(
          isA<SheetParseException>().having(
            (e) => e.message,
            'message',
            contains('no rows'),
          ),
        ),
      );
    });

    test('an unsupported extension is refused by name', () {
      expect(
        () => SheetParser.parse('scan.pdf', csv('x')),
        throwsA(
          isA<SheetParseException>().having(
            (e) => e.message,
            'message',
            contains('scan.pdf'),
          ),
        ),
      );
    });
  });

  group('XLSX', () {
    Uint8List workbook(Map<String, List<List<Object?>>> sheets) {
      final book = Excel.createExcel();
      final defaultSheet = book.getDefaultSheet()!;
      for (final entry in sheets.entries) {
        final sheet = book[entry.key];
        for (final row in entry.value) {
          sheet.appendRow([
            for (final cell in row)
              switch (cell) {
                null => TextCellValue(''),
                final String s => TextCellValue(s),
                final int i => IntCellValue(i),
                final double d => DoubleCellValue(d),
                _ => TextCellValue(cell.toString()),
              },
          ]);
        }
      }
      if (!sheets.containsKey(defaultSheet)) book.delete(defaultSheet);
      return Uint8List.fromList(book.encode()!);
    }

    test('yields the same shape as a CSV', () {
      final book = SheetParser.parse(
        'contacts.xlsx',
        workbook({
          'Contacts': [
            ['name', 'phone'],
            ['Rina', '+91 90000 00001'],
          ],
        }),
      );
      final sheet = book.sheets.single;
      expect(sheet.headers, ['name', 'phone']);
      expect(sheet.rows.single, ['Rina', '+91 90000 00001']);
    });

    test('a phone number stored as a float renders as digits', () {
      // Excel stores a number typed without a leading + as a float. Left
      // alone it comes back as 9.87654321E9, normalises to nothing, and looks
      // like a corrupt file. This is the likeliest way a real sheet breaks.
      final book = SheetParser.parse(
        'contacts.xlsx',
        workbook({
          'Sheet1': [
            ['name', 'phone'],
            ['Rina', 9876543210.0],
          ],
        }),
      );
      expect(book.sheets.single.rows.single[1], '9876543210');
    });

    test('several sheets ask to be chosen between', () {
      final book = SheetParser.parse(
        'trip.xlsx',
        workbook({
          'Stays': [
            ['name', 'phone'],
            ['A', '1'],
          ],
          'Drivers': [
            ['name', 'phone'],
            ['B', '2'],
          ],
        }),
      );
      expect(book.needsSheetChoice, isTrue);
      expect(book.sheets.map((s) => s.name), containsAll(['Stays', 'Drivers']));
    });

    test('an empty sheet is dropped rather than offered', () {
      final book = SheetParser.parse(
        'trip.xlsx',
        workbook({
          'Real': [
            ['name', 'phone'],
            ['A', '1'],
          ],
          'Blank': [],
        }),
      );
      expect(book.sheets.map((s) => s.name), ['Real']);
    });
  });
}
