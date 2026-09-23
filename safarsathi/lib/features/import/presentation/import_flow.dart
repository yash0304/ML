// lib/features/import/presentation/import_flow.dart
//
// Drives pick → parse → (choose sheet) → map → preview → commit.
//
// The file picker lives HERE and nowhere else. Every step below it takes plain
// values, so all of the parsing, matching and validation stays testable
// without a platform channel.

import 'dart:typed_data';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';

import '../../../core/database/app_database.dart';
import '../../../core/theme/app_tokens.dart';
import '../../../core/theme/motion.dart';
import '../../../core/widgets/retro.dart';
import '../data/column_mapping.dart';
import '../data/import_commit.dart';
import '../data/import_validation.dart';
import '../data/sheet_parser.dart';
import '../data/stop_matcher.dart';
import 'column_mapping_screen.dart';
import 'import_preview_screen.dart';

/// Header line offered on the More tab, for someone starting a sheet from
/// scratch. Not a downloaded file: a release build cannot write to shared
/// storage without a permission this app refuses to ask for, and a line of
/// text on the clipboard solves the same problem.
const importTemplateHeader = 'name,phone,category,stop,note,emergency';

class ImportFlow {
  final AppDatabase db;
  final int tripId;

  /// Injected so a test can drive the whole flow without a file picker.
  final Future<({String name, List<int> bytes})?> Function() pickFile;

  ImportFlow({required this.db, required this.tripId, Future<({String name, List<int> bytes})?> Function()? pickFile})
    : pickFile = pickFile ?? _pickWithFilePicker;

  static Future<({String name, List<int> bytes})?> _pickWithFilePicker() async {
    final file = await FilePicker.pickFile(
      type: FileType.custom,
      allowedExtensions: const ['csv', 'xlsx', 'xlsm', 'txt'],
      dialogTitle: 'Pick a contacts sheet',
    );
    if (file == null) return null;
    // Read the BYTES, never a path. On Android a picked file often lives
    // behind a content:// URI with no readable filesystem path at all.
    return (name: file.name, bytes: await file.readAsBytes());
  }

  Future<void> start(BuildContext context) async {
    final picked = await pickFile();
    if (picked == null || !context.mounted) return;

    final ParsedWorkbook workbook;
    try {
      workbook = SheetParser.parse(
        picked.name,
        Uint8List.fromList(picked.bytes),
      );
    } on SheetParseException catch (e) {
      if (context.mounted) _say(context, e.message);
      return;
    }

    if (!context.mounted) return;

    var sheet = workbook.sheets.first;
    if (workbook.needsSheetChoice) {
      final chosen = await _chooseSheet(context, workbook.sheets);
      if (chosen == null || !context.mounted) return;
      sheet = chosen;
    }

    final stops = await _stops();
    final existing = await readExistingContacts(db, tripId);
    if (!context.mounted) return;

    await Navigator.of(context).push(
      MaterialPageRoute<void>(
        builder: (mapContext) => ColumnMappingScreen(
          table: sheet,
          initial: autoMatchColumns(sheet.headers),
          onContinue: (mapping) {
            final preview = validateRows(
              sheet,
              mapping,
              stops: stops,
              existing: existing,
            );
            Navigator.of(mapContext).pushReplacement(
              MaterialPageRoute<void>(
                builder: (previewContext) => ImportPreviewHost(
                  preview: preview,
                  fileName: workbook.fileName,
                  sheetName: workbook.needsSheetChoice ? sheet.name : null,
                  onCommit: (p) async {
                    final result = await commitImport(
                      db,
                      tripId: tripId,
                      fileName: workbook.fileName,
                      sheetName: workbook.needsSheetChoice ? sheet.name : null,
                      preview: p,
                    );
                    return result;
                  },
                ),
              ),
            );
          },
        ),
      ),
    );
  }

  Future<List<StopCandidate>> _stops() async {
    final rows = await (db.select(
      db.stops,
    )..where((s) => s.tripId.equals(tripId))).get();
    return [for (final s in rows) StopCandidate(s.id, s.name)];
  }

  Future<SheetTable?> _chooseSheet(
    BuildContext context,
    List<SheetTable> sheets,
  ) {
    final c = AppTokens.of(context);
    return showModalBottomSheet<SheetTable>(
      context: context,
      backgroundColor: c.paper,
      builder: (sheetContext) => SafeArea(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const StencilLabel('Which sheet?'),
            for (final s in sheets)
              PressScale(
                onTap: () => Navigator.of(sheetContext).pop(s),
                child: Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: AppTokens.gutter,
                    vertical: AppTokens.s16,
                  ),
                  decoration: BoxDecoration(
                    border: Border(
                      bottom: BorderSide(
                        color: c.rule,
                        width: AppTokens.hairline,
                      ),
                    ),
                  ),
                  child: Row(
                    children: [
                      Expanded(
                        child: Text(
                          s.name,
                          style: AppTokens.rowTitleStyle.copyWith(color: c.ink),
                        ),
                      ),
                      Text(
                        '${s.rowCount}',
                        style: AppTokens.numberStyle.copyWith(color: c.muted),
                      ),
                      const SizedBox(width: AppTokens.s4),
                      Text(
                        'rows',
                        style: AppTokens.captionStyle.copyWith(color: c.muted),
                      ),
                    ],
                  ),
                ),
              ),
            const SizedBox(height: AppTokens.s16),
          ],
        ),
      ),
    );
  }

  void _say(BuildContext context, String message) {
    Haptics.reject();
    final c = AppTokens.of(context);
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        backgroundColor: c.ink,
        content: Text(
          message,
          style: AppTokens.captionStyle.copyWith(color: c.paper),
        ),
      ),
    );
  }
}

/// Wraps the preview screen so a finished commit reports back and leaves the
/// flow, rather than stranding the user on a list of rows that already landed.
class ImportPreviewHost extends StatelessWidget {
  final ImportPreview preview;
  final String fileName;
  final String? sheetName;
  final Future<ImportResult> Function(ImportPreview) onCommit;

  const ImportPreviewHost({
    super.key,
    required this.preview,
    required this.fileName,
    required this.onCommit,
    this.sheetName,
  });

  @override
  Widget build(BuildContext context) {
    return ImportPreviewScreen(
      preview: preview,
      fileName: fileName,
      sheetName: sheetName,
      onCommit: (p) async {
        final result = await onCommit(p);
        final n = result.imported;
        if (!context.mounted) return n;
        Haptics.confirm();
        final c = AppTokens.of(context);
        final messenger = ScaffoldMessenger.of(context);
        Navigator.of(context).pop();
        messenger.showSnackBar(
          SnackBar(
            backgroundColor: c.ink,
            content: Text(
              importResultMessage(result),
              style: AppTokens.captionStyle.copyWith(color: c.paper),
            ),
          ),
        );
        return n;
      },
    );
  }
}

/// What the import did, in one sentence per kind of change.
String importResultMessage(ImportResult r) {
  final parts = [
    if (r.imported > 0)
      '${r.imported} ${r.imported == 1 ? 'contact' : 'contacts'} imported, '
          'all unconfirmed. Call them, then mark them.',
    if (r.placed > 0)
      '${r.placed} ${r.placed == 1 ? 'entry' : 'entries'} in your diary '
          'now ${r.placed == 1 ? 'has a location' : 'have locations'}.',
  ];
  return parts.isEmpty ? 'Nothing changed.' : parts.join(' ');
}
