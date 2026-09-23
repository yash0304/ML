// test/import_screens_test.dart — issues #12 and #13, the screens.
//
// No database here. Both screens take plain values, which is the constraint
// established at #6 and the reason these run in milliseconds.

import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:safarsathi/core/theme/app_tokens.dart';
import 'package:safarsathi/features/import/data/column_mapping.dart';
import 'package:safarsathi/features/import/data/import_commit.dart';
import 'package:safarsathi/features/import/data/import_validation.dart';
import 'package:safarsathi/features/import/data/sheet_parser.dart';
import 'package:safarsathi/features/import/presentation/column_mapping_screen.dart';
import 'package:safarsathi/features/import/presentation/import_flow.dart'
    show importResultMessage;
import 'package:safarsathi/features/import/presentation/import_preview_screen.dart';

SheetTable table(String csv) => SheetParser.parse(
  'contacts.csv',
  Uint8List.fromList(utf8.encode(csv)),
).sheets.single;

Widget wrap(Widget child, {Brightness brightness = Brightness.light}) =>
    MaterialApp(
      theme: brightness == Brightness.light ? AppTokens.light : AppTokens.dark,
      home: child,
    );

void main() {
  group('column mapping', () {
    testWidgets('shows a sample value so the guess can be checked', (
      tester,
    ) async {
      final t = table('name,phone\nRina Kharkongor,+91 90000 00001\n');
      await tester.pumpWidget(
        wrap(
          ColumnMappingScreen(
            table: t,
            initial: autoMatchColumns(t.headers),
            onContinue: (_) {},
          ),
        ),
      );

      expect(find.text('e.g. Rina Kharkongor'), findsOneWidget);
      expect(find.text('e.g. +91 90000 00001'), findsOneWidget);
    });

    testWidgets('will not continue while a required field is unmatched', (
      tester,
    ) async {
      var continued = false;
      final t = table('nickname,digits\nA,1\n');
      await tester.pumpWidget(
        wrap(
          ColumnMappingScreen(
            table: t,
            initial: autoMatchColumns(t.headers),
            onContinue: (_) => continued = true,
          ),
        ),
      );

      // The warning sits below the field list, so scroll it into view.
      await tester.scrollUntilVisible(
        find.textContaining('Still need'),
        200,
        scrollable: find.byType(Scrollable).first,
      );
      expect(find.textContaining('Still need'), findsOneWidget);

      await tester.tap(find.text('See what will be imported'));
      await tester.pump();
      expect(continued, isFalse);
    });

    testWidgets('continues once name and phone are matched', (tester) async {
      ColumnMapping? got;
      final t = table('name,phone\nA,1\n');
      await tester.pumpWidget(
        wrap(
          ColumnMappingScreen(
            table: t,
            initial: autoMatchColumns(t.headers),
            onContinue: (m) => got = m,
          ),
        ),
      );

      expect(find.textContaining('Still need'), findsNothing);
      await tester.tap(find.text('See what will be imported'));
      await tester.pump();
      expect(got?[ImportField.name], 0);
      expect(got?[ImportField.phone], 1);
    });

    testWidgets('the row count and sheet name are stated up front', (
      tester,
    ) async {
      final t = table('name,phone\nA,1\nB,2\nC,3\n');
      await tester.pumpWidget(
        wrap(
          ColumnMappingScreen(
            table: t,
            initial: autoMatchColumns(t.headers),
            onContinue: (_) {},
          ),
        ),
      );
      expect(find.textContaining('3 rows'), findsOneWidget);
    });
  });

  group('preview', () {
    ImportPreview previewOf(String csv) {
      final t = table(csv);
      return validateRows(t, autoMatchColumns(t.headers));
    }

    Future<void> pumpPreview(
      WidgetTester tester,
      ImportPreview p, {
      Future<int> Function(ImportPreview)? onCommit,
      Brightness brightness = Brightness.light,
    }) => tester.pumpWidget(
      wrap(
        ImportPreviewScreen(
          preview: p,
          fileName: 'contacts.csv',
          onCommit: onCommit ?? (_) async => p.selected,
        ),
        brightness: brightness,
      ),
    );

    testWidgets('THE BUTTON SAYS WHAT WILL HAPPEN: new entries and '
        'locations added are counted apart', (tester) async {
      final t = table(
        'name,phone,latitude,longitude\n'
        'Civil Hospital,+91 364 222 4100,25.567739,91.881081\n'
        'Woodland,+91 364 222 5240,25.566285,91.889846\n'
        'New place,+91 98560 11111,25.3,91.9\n',
      );
      final p = validateRows(
        t,
        autoMatchColumns(t.headers),
        existing: const ExistingContacts(
          e164: {'+913642224100', '+913642225240'},
          squashedNames: {},
          unplacedE164: {'+913642224100', '+913642225240'},
        ),
      );
      await pumpPreview(tester, p);
      expect(find.text('Import 1 contact + 2 locations'), findsOneWidget);
      expect(find.textContaining('stays if the batch is undone'), findsOneWidget);
    });

    test('the message after importing names both kinds of change', () {
      expect(
        importResultMessage(
          const ImportResult(imported: 0, skipped: 17, placed: 50),
        ),
        '50 entries in your diary now have locations.',
      );
      expect(
        importResultMessage(
          const ImportResult(imported: 1, skipped: 0, placed: 1),
        ),
        '1 contact imported, all unconfirmed. Call them, then mark them. '
        '1 entry in your diary now has a location.',
      );
    });

    testWidgets('the counts are stated in the header', (tester) async {
      await pumpPreview(
        tester,
        previewOf(
          'name,phone\nA,+91 90000 00001\nB,bad\n,+91 90000 00003\n',
        ),
      );

      expect(find.text('READ'), findsOneWidget);
      expect(find.text('READY'), findsOneWidget);
      expect(find.text('CHECK'), findsOneWidget);
      expect(find.text('SKIPPED'), findsOneWidget);
    });

    testWidgets('the invariant is restated in words', (tester) async {
      await pumpPreview(tester, previewOf('name,phone\nA,+91 90000 00001\n'));
      expect(find.textContaining('lands unconfirmed'), findsOneWidget);
      expect(find.textContaining('undone in one action'), findsOneWidget);
    });

    testWidgets('a warning names the specific problem', (tester) async {
      await pumpPreview(tester, previewOf('name,phone\nA,not a number\n'));
      expect(
        find.textContaining('Could not read this as a phone number'),
        findsOneWidget,
      );
    });

    testWidgets('the button counts what is selected', (tester) async {
      await pumpPreview(
        tester,
        previewOf('name,phone\nA,+91 90000 00001\nB,+91 90000 00002\n'),
      );
      expect(find.text('Import 2 contacts'), findsOneWidget);
    });

    testWidgets('deselecting a row updates the button', (tester) async {
      await pumpPreview(
        tester,
        previewOf('name,phone\nA,+91 90000 00001\nB,+91 90000 00002\n'),
      );
      await tester.tap(find.text('A'));
      await tester.pump();
      expect(find.text('Import 1 contact'), findsOneWidget);
    });

    testWidgets('a skipped row cannot be selected', (tester) async {
      await pumpPreview(tester, previewOf('name,phone\n,+91 90000 00001\n'));
      expect(find.text('Nothing selected'), findsOneWidget);

      await tester.tap(find.text('(no name)'));
      await tester.pump();
      expect(find.text('Nothing selected'), findsOneWidget);
    });

    testWidgets('committing passes only the selected rows', (tester) async {
      ImportPreview? committed;
      await pumpPreview(
        tester,
        previewOf('name,phone\nA,+91 90000 00001\nB,+91 90000 00002\n'),
        onCommit: (p) async {
          committed = p;
          return p.selected;
        },
      );

      await tester.tap(find.text('B'));
      await tester.pump();
      await tester.tap(find.text('Import 1 contact'));
      await tester.pump();

      expect(committed?.toImport.map((r) => r.name), ['A']);
    });

    testWidgets('the row number shown is the line in the file', (
      tester,
    ) async {
      // Two data rows, on lines 2 and 5 of the file. The count block also
      // renders a "2", so look inside the rows rather than the whole screen.
      final p = previewOf(
        'name,phone\nA,+91 90000 00001\n\n\nD,+91 90000 00004\n',
      );
      expect(p.rows.map((r) => r.sourceRow), [2, 5]);

      await pumpPreview(tester, p);
      expect(
        find.descendant(of: find.byType(IntrinsicHeight), matching: find.text('2')),
        findsOneWidget,
      );
      expect(
        find.descendant(of: find.byType(IntrinsicHeight), matching: find.text('5')),
        findsOneWidget,
      );
    });

    testWidgets('lays out at phone width in both themes', (tester) async {
      tester.view.physicalSize = const Size(400, 800);
      tester.view.devicePixelRatio = 1.0;
      addTearDown(tester.view.reset);

      final p = previewOf(
        'name,phone,stop\n'
        'A very long homestay name that will not fit,+91 90000 00001,Shillong\n'
        'B,not a number\n'
        ',+91 90000 00003\n',
      );

      for (final b in Brightness.values) {
        await pumpPreview(tester, p, brightness: b);
        expect(tester.takeException(), isNull);
      }
    });
  });
}
