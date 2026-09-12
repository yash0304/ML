// lib/features/import/data/sheet_parser.dart
//
// Turns a picked file into rows the rest of the import flow can reason about.
//
// THIS PARSES BYTES, NOT PATHS. No file_picker import, no dart:io, no platform
// channel — which is why the whole of issue #11 is testable in a plain Dart
// test with a string literal, and why it will keep working if the file picker
// is ever replaced.

import 'dart:convert';
import 'dart:typed_data';

import 'package:csv/csv.dart';
import 'package:excel/excel.dart';

/// Raised when a file cannot be read at all. Carries a sentence meant for the
/// user, not a stack trace — a wrong file picked at a dhaba should say so in
/// words.
class SheetParseException implements Exception {
  final String message;
  const SheetParseException(this.message);
  @override
  String toString() => message;
}

/// One sheet's worth of rows, with the header row separated out.
class SheetTable {
  /// For CSV this is the file name; for XLSX the sheet's own name.
  final String name;

  final List<String> headers;

  /// Data rows only, each padded or truncated to `headers.length`.
  final List<List<String>> rows;

  /// The 1-based line each row occupied in the original file, header
  /// included. `rows[i]` came from line `sourceRowNumbers[i]`.
  ///
  /// THIS IS THE POINT OF THIS CLASS. Every error message downstream says
  /// "row 14", and row 14 has to mean what the user sees in Excel — not the
  /// index after blank rows were dropped. Losing this turns every warning
  /// into a scavenger hunt through a sixty-line sheet.
  final List<int> sourceRowNumbers;

  const SheetTable({
    required this.name,
    required this.headers,
    required this.rows,
    required this.sourceRowNumbers,
  });

  bool get isEmpty => rows.isEmpty;
  int get rowCount => rows.length;
}

class ParsedWorkbook {
  final String fileName;

  /// One entry for CSV. Possibly several for XLSX, in which case the user is
  /// asked which one.
  final List<SheetTable> sheets;

  const ParsedWorkbook({required this.fileName, required this.sheets});

  bool get needsSheetChoice => sheets.length > 1;
}

class SheetParser {
  SheetParser._();

  /// Dispatches on the file extension. An unknown extension is an error with
  /// a sentence rather than a guess — parsing a PDF as CSV produces one row
  /// of garbage and no explanation.
  static ParsedWorkbook parse(String fileName, Uint8List bytes) {
    final lower = fileName.toLowerCase();
    if (lower.endsWith('.csv') || lower.endsWith('.txt')) {
      return parseCsv(fileName, bytes);
    }
    if (lower.endsWith('.xlsx') || lower.endsWith('.xlsm')) {
      return parseXlsx(fileName, bytes);
    }
    throw SheetParseException(
      'SafarSathi can read .csv and .xlsx files. '
      '"$fileName" is neither.',
    );
  }

  static ParsedWorkbook parseCsv(String fileName, Uint8List bytes) {
    // allowMalformed keeps a file saved in some Windows codepage readable
    // rather than throwing on the first stray byte. A mangled character in a
    // name is recoverable; a refused file is not.
    final text = utf8.decode(bytes, allowMalformed: true);

    final raw = const CsvToListConverter(
      shouldParseNumbers: false,
      eol: '\n',
    ).convert(text.replaceAll('\r\n', '\n'));

    final table = _tableFrom(fileName, raw);
    if (table == null) {
      throw const SheetParseException(
        'That file has no rows in it. Check it opens in a spreadsheet first.',
      );
    }
    return ParsedWorkbook(fileName: fileName, sheets: [table]);
  }

  static ParsedWorkbook parseXlsx(String fileName, Uint8List bytes) {
    final Excel book;
    try {
      book = Excel.decodeBytes(bytes);
    } on Object {
      throw const SheetParseException(
        'That file could not be opened as a spreadsheet. If it came from '
        'Google Sheets, export it as CSV instead.',
      );
    }

    final sheets = <SheetTable>[];
    for (final entry in book.tables.entries) {
      final rows = [
        for (final row in entry.value.rows) [for (final cell in row) _cell(cell)],
      ];
      final table = _tableFrom(entry.key, rows);
      if (table != null) sheets.add(table);
    }

    if (sheets.isEmpty) {
      throw const SheetParseException(
        'Every sheet in that workbook is empty.',
      );
    }
    return ParsedWorkbook(fileName: fileName, sheets: sheets);
  }

  /// Takes the first non-blank row as the header and keeps the 1-based
  /// original line number of every row after it.
  static SheetTable? _tableFrom(String name, List<List<dynamic>> raw) {
    var headerIndex = -1;
    for (var i = 0; i < raw.length; i++) {
      if (_hasContent(raw[i])) {
        headerIndex = i;
        break;
      }
    }
    if (headerIndex < 0) return null;

    final headers = [
      for (final cell in raw[headerIndex]) _text(cell).trim(),
    ];
    // A trailing comma in a CSV gives a phantom empty column on every row.
    while (headers.isNotEmpty && headers.last.isEmpty) {
      headers.removeLast();
    }
    if (headers.isEmpty) return null;

    final rows = <List<String>>[];
    final numbers = <int>[];
    for (var i = headerIndex + 1; i < raw.length; i++) {
      if (!_hasContent(raw[i])) continue;
      rows.add(_fit(raw[i], headers.length));
      numbers.add(i + 1); // 1-based, as a spreadsheet shows it.
    }

    return SheetTable(
      name: name,
      headers: headers,
      rows: rows,
      sourceRowNumbers: numbers,
    );
  }

  /// Pads a short row and truncates a long one, so a row missing its last two
  /// cells reads as two empty fields rather than throwing a range error.
  static List<String> _fit(List<dynamic> row, int width) => [
    for (var i = 0; i < width; i++) i < row.length ? _text(row[i]).trim() : '',
  ];

  static bool _hasContent(List<dynamic> row) =>
      row.any((c) => _text(c).trim().isNotEmpty);

  static String _cell(dynamic cell) {
    if (cell == null) return '';
    // The excel package wraps values; its `value` is the typed cell content.
    final value = cell is Data ? cell.value : cell;
    return _text(value);
  }

  /// Renders a cell as the user typed it, with one specific rescue.
  static String _text(dynamic value) {
    if (value == null) return '';

    // EXCEL STORES A PHONE NUMBER TYPED WITHOUT A LEADING + AS A FLOAT.
    // Left alone it comes back as "9.87654321E9", which normalises to
    // nothing and looks like a corrupt file. This is the single most likely
    // way a real sheet of Indian mobile numbers breaks on import.
    if (value is double) {
      if (value == value.roundToDouble() && value.abs() < 1e17) {
        return value.toInt().toString();
      }
      return value.toString();
    }
    if (value is int) return value.toString();

    final text = value.toString();
    // The typed wrappers in the excel package stringify to their content for
    // text and to the raw value for everything else; a null-ish sentinel is
    // not useful to show.
    return text == 'null' ? '' : text;
  }
}
