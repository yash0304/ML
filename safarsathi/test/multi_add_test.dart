// test/multi_add_test.dart — issue #10.
//
// The invariant first: bulk entry must never produce a confirmed contact.
// Everything else on this screen is convenience; that one is the app.

import 'dart:async';

import 'package:drift/drift.dart' hide isNull, isNotNull;
import 'package:drift/native.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:safarsathi/core/database/app_database.dart';
import 'package:safarsathi/core/theme/app_tokens.dart';
import 'package:safarsathi/features/contacts/data/contacts_dao.dart';
import 'package:safarsathi/features/contacts/data/multi_add.dart';
import 'package:safarsathi/features/contacts/presentation/multi_add_screen.dart';
import 'package:safarsathi/features/import/data/import_commit.dart';
import 'package:safarsathi/features/trips/data/trip_editor.dart';

MultiAddRow r(String name, String phone, {String? e164, String? dupe}) =>
    MultiAddRow(
      name: name,
      phoneRaw: phone,
      phoneE164: e164,
      duplicateOf: dupe,
    );

Widget wrap(Widget child) => MaterialApp(theme: AppTokens.light, home: child);

/// What the Android back gesture does, and what PopScope intercepts.
Future<void> goBack(WidgetTester tester) async {
  final context = tester.element(find.byType(MultiAddScreen));
  // Not awaited: the pop resolves across frames, and awaiting it before
  // pumping means waiting for something only pumping can deliver.
  unawaited(Navigator.of(context).maybePop());
  await tester.pumpAndSettle();
}

/// Puts [child] behind a route that can actually be popped.
Widget pushable(Widget child) => MaterialApp(
  theme: AppTokens.light,
  home: Builder(
    builder: (context) => Scaffold(
      body: TextButton(
        onPressed: () => Navigator.of(context).push(
          MaterialPageRoute<void>(builder: (_) => child),
        ),
        child: const Text('open'),
      ),
    ),
  ),
);

void useTallSurface(WidgetTester tester) {
  tester.view.physicalSize = const Size(420, 2000);
  tester.view.devicePixelRatio = 1.0;
  addTearDown(tester.view.reset);
}

void main() {
  group('the sheet', () {
    test('starts with one blank row so there is somewhere to type', () {
      final sheet = MultiAddSheet.empty();
      expect(sheet.rows, hasLength(1));
      expect(sheet.rows.single.isBlank, isTrue);
      expect(sheet.canSave, isFalse);
    });

    test('GROWS ITSELF when the last row is typed into', () {
      final sheet = MultiAddSheet([r('Kongthong homestay', '9000000001')])
          .settled();
      expect(sheet.rows, hasLength(2));
      expect(sheet.rows.last.isBlank, isTrue);
    });

    test('never keeps two blank rows at the end', () {
      final sheet = MultiAddSheet([
        r('A', '1'),
        const MultiAddRow(),
        const MultiAddRow(),
        const MultiAddRow(),
      ]).settled();
      expect(sheet.rows, hasLength(2));
    });

    test('A NEW ROW INHERITS THE CATEGORY ABOVE IT', () {
      // Most sessions are one kind at a time. Setting it once and having it
      // stick is the fast path.
      final sheet = MultiAddSheet([
        r('Shillong guesthouse', '9000000001').copyWith(
          category: ContactCategory.accommodation,
        ),
      ]).settled();
      expect(sheet.rows.last.category, ContactCategory.accommodation);
    });

    test('counts what will save and what will be left behind', () {
      final sheet = MultiAddSheet([
        r('A', '9000000001'),
        r('B', ''), // partial
        r('', '9000000003'), // partial
        r('D', '9000000004'),
        const MultiAddRow(),
      ]).settled();

      expect(sheet.readyCount, 2);
      expect(sheet.partialCount, 2);
      expect(sheet.canSave, isTrue);
    });

    test('a partial row says which half is missing', () {
      expect(r('Bah Rothell', '').missing, 'Needs a number');
      expect(r('', '9000000001').missing, 'Needs a name');
    });

    test('SPOTS A NUMBER TYPED TWICE ON THE SAME SHEET', () {
      // Easy to do off a booking confirmation, and the second one is the row
      // you are looking at.
      final sheet = MultiAddSheet([
        r('Wanshai', '+91 90000 00002', e164: '+919000000002'),
        r('Wanshai driver', '09000000002', e164: '+919000000002'),
      ]).settled();

      expect(sheet.rows[0].duplicateInSheet, isFalse);
      expect(sheet.rows[1].duplicateInSheet, isTrue);
    });

    test('two unparseable numbers are not assumed to be the same', () {
      final sheet = MultiAddSheet([
        r('A', 'ask at the desk'),
        r('B', 'ask at the desk'),
      ]).settled();
      expect(sheet.rows.every((row) => !row.duplicateInSheet), isTrue);
    });
  });

  group('checking a row', () {
    test('normalises the number and finds it in the diary', () async {
      final checked = await checkRow(
        r('Kongthong homestay', '09000000001'),
        findDuplicate: (e164) async => Contact(
          id: 1,
          name: 'Bah Rothell',
          phoneRaw: '+91 90000 00001',
          category: ContactCategory.accommodation,
          tier: ContactTier.userEntered.name,
          callConfirmed: false,
          isPinned: false,
          isEmergency: false,
          hasWhatsapp: false,
          callCount: 0,
          createdAt: DateTime(2026, 9, 12),
        ),
      );

      expect(checked.phoneE164, '+919000000001');
      expect(checked.duplicateOf, 'Bah Rothell');
    });

    test('AN ODD NUMBER WARNS AND STILL SAVES', () async {
      final checked = await checkRow(
        r('Signboard', '123'),
        findDuplicate: (_) async => null,
      );
      // Refusing would lose the only record somebody has of it.
      expect(checked.isReady, isTrue);
      expect(checked.warning, isNotNull);
    });

    test('emptying the number clears everything it had said', () async {
      final checked = await checkRow(
        r('A', '').copyWith(warning: 'stale', duplicateOf: 'stale'),
        findDuplicate: (_) async => null,
      );
      expect(checked.warning, isNull);
      expect(checked.duplicateOf, isNull);
      expect(checked.phoneE164, isNull);
    });
  });

  group('committing', () {
    late AppDatabase db;
    late int tripId;

    setUp(() async {
      db = AppDatabase(NativeDatabase.memory());
      tripId = await TripEditor(db).createTrip(name: 'Meghalaya');
    });
    tearDown(() => db.close());

    test('NOTHING TYPED HERE IS EVER CONFIRMED', () async {
      final result = await commitMultiAdd(
        db,
        tripId: tripId,
        sheet: MultiAddSheet([
          r('Kongthong homestay', '9000000001', e164: '+919000000001'),
          r('Wanshai', '9000000002', e164: '+919000000002'),
          const MultiAddRow(),
        ]).settled(),
      );

      expect(result.added, 2);

      final saved = await db.select(db.contacts).get();
      expect(saved, hasLength(2));
      for (final contact in saved) {
        expect(contact.callConfirmed, isFalse);
        expect(contact.tier, ContactTier.userEntered.name);
        expect(contact.confirmedAt, isNull);
      }
    });

    test('partial rows are skipped, not written', () async {
      final result = await commitMultiAdd(
        db,
        tripId: tripId,
        sheet: MultiAddSheet([
          r('Kongthong homestay', '9000000001'),
          r('No number yet', ''),
          r('', '9000000003'),
        ]).settled(),
      );

      expect(result.added, 1);
      expect(result.skipped, 2);
      expect(await db.select(db.contacts).get(), hasLength(1));
    });

    test('an empty sheet writes nothing at all, not an empty batch', () async {
      final result = await commitMultiAdd(
        db,
        tripId: tripId,
        sheet: MultiAddSheet.empty(),
      );
      expect(result.added, 0);
      expect(await db.select(db.importBatches).get(), isEmpty);
    });

    test('IT IS RECORDED AS A BATCH, SO IT CAN BE UNDONE', () async {
      await commitMultiAdd(
        db,
        tripId: tripId,
        sheet: MultiAddSheet([
          r('A', '9000000001'),
          r('B', '9000000002'),
        ]).settled(),
      );

      final batches = await watchImportBatches(db, tripId).first;
      expect(batches, hasLength(1));
      // The history must not tell somebody they "imported" what they typed.
      expect(batches.single.wasTyped, isTrue);
      expect(batches.single.fileName, typedBatchLabel);

      await db.contactsDao.rollbackImport(batches.single.id);
      expect(await db.select(db.contacts).get(), isEmpty);
    });

    test('a file import is not marked as typed', () async {
      await db.contactsDao.insertBatch(
        [ContactsCompanion.insert(name: 'A', phoneRaw: '1', tripId: Value(tripId))],
        ImportBatchesCompanion.insert(
          fileName: 'bookings.csv',
          tripId: Value(tripId),
          rowsImported: const Value(1),
        ),
      );
      final batches = await watchImportBatches(db, tripId).first;
      expect(batches.single.wasTyped, isFalse);
    });
  });

  group('the screen', () {
    /// The screen itself, with no app around it.
    MultiAddScreen bare({
      Future<MultiAddRow> Function(MultiAddRow)? check,
      Future<MultiAddResult> Function(MultiAddSheet)? onSave,
    }) => MultiAddScreen(
      check: check ?? (row) async => row,
      onSave:
          onSave ??
          (sheet) async => MultiAddResult(
            added: sheet.readyCount,
            skipped: sheet.partialCount,
          ),
    );

    Widget screen({
      Future<MultiAddRow> Function(MultiAddRow)? check,
      Future<MultiAddResult> Function(MultiAddSheet)? onSave,
    }) => wrap(bare(check: check, onSave: onSave));

    testWidgets('opens with one row and nothing to save', (tester) async {
      useTallSurface(tester);
      await tester.pumpWidget(screen());
      await tester.pump();

      expect(find.text('Name'), findsOneWidget);
      expect(find.text('Nothing to save yet'), findsOneWidget);
    });

    testWidgets('THE UNCONFIRMED PROMISE IS STATED ON THE SCREEN', (
      tester,
    ) async {
      useTallSurface(tester);
      await tester.pumpWidget(screen());
      await tester.pump();

      expect(
        find.textContaining('saves unconfirmed, with the amber dot'),
        findsOneWidget,
      );
    });

    testWidgets('typing a row grows the sheet and counts the button', (
      tester,
    ) async {
      useTallSurface(tester);
      await tester.pumpWidget(screen());
      await tester.pump();

      await tester.enterText(find.byType(TextField).at(0), 'Kongthong');
      await tester.pump();
      await tester.enterText(find.byType(TextField).at(1), '9000000001');
      await tester.pumpAndSettle();

      expect(find.text('Save 1 entry'), findsOneWidget);
      // A second row appeared to type into.
      expect(find.byType(TextField), findsNWidgets(4));
    });

    testWidgets('the button pluralises', (tester) async {
      useTallSurface(tester);
      await tester.pumpWidget(screen());
      await tester.pump();

      await tester.enterText(find.byType(TextField).at(0), 'A');
      await tester.enterText(find.byType(TextField).at(1), '9000000001');
      await tester.pumpAndSettle();
      await tester.enterText(find.byType(TextField).at(2), 'B');
      await tester.enterText(find.byType(TextField).at(3), '9000000002');
      await tester.pumpAndSettle();

      expect(find.text('Save 2 entries'), findsOneWidget);
    });

    testWidgets('an unfinished row is announced, not rejected', (tester) async {
      useTallSurface(tester);
      await tester.pumpWidget(screen());
      await tester.pump();

      await tester.enterText(find.byType(TextField).at(0), 'A');
      await tester.enterText(find.byType(TextField).at(1), '9000000001');
      await tester.pumpAndSettle();
      await tester.enterText(find.byType(TextField).at(2), 'Name only');
      await tester.pumpAndSettle();

      expect(find.text('Needs a number'), findsOneWidget);
      expect(find.textContaining('will be left behind'), findsOneWidget);
      expect(find.text('Save 1 entry'), findsOneWidget);
    });

    testWidgets('a duplicate in the diary is named', (tester) async {
      useTallSurface(tester);
      await tester.pumpWidget(
        screen(
          check: (row) async => row.copyWith(
            phoneE164: '+919000000001',
            duplicateOf: 'Bah Rothell',
          ),
        ),
      );
      await tester.pump();

      await tester.enterText(find.byType(TextField).at(0), 'Kongthong');
      await tester.enterText(find.byType(TextField).at(1), '9000000001');
      await tester.pumpAndSettle();

      expect(
        find.text('Already in the diary as Bah Rothell'),
        findsOneWidget,
      );
    });

    testWidgets('saving hands over only the ready rows', (tester) async {
      useTallSurface(tester);
      MultiAddSheet? saved;
      await tester.pumpWidget(
        screen(
          onSave: (sheet) async {
            saved = sheet;
            return MultiAddResult(added: sheet.readyCount, skipped: 0);
          },
        ),
      );
      await tester.pump();

      await tester.enterText(find.byType(TextField).at(0), 'A');
      await tester.enterText(find.byType(TextField).at(1), '9000000001');
      await tester.pumpAndSettle();

      await tester.tap(find.text('Save 1 entry'));
      await tester.pumpAndSettle();

      expect(saved?.readyCount, 1);
      expect(saved?.ready.single.name, 'A');
    });

    testWidgets('LEAVING WITH TYPED ROWS ASKS FIRST', (tester) async {
      useTallSurface(tester);
      // Pushed rather than used as `home`, so popping is a real thing the
      // navigator could do. As `home` it is the first route and can never
      // pop, which would make this assertion true for the wrong reason.
      await tester.pumpWidget(pushable(bare()));
      await tester.pump();
      await tester.tap(find.text('open'));
      await tester.pumpAndSettle();

      await tester.enterText(find.byType(TextField).at(0), 'A');
      await tester.enterText(find.byType(TextField).at(1), '9000000001');
      await tester.pumpAndSettle();

      // Eight numbers typed and lost to a back gesture is the same harm the
      // swipe actions were designed against.
      await goBack(tester);

      expect(find.text('Throw these away?'), findsOneWidget);
      expect(find.textContaining('1 entry is ready'), findsOneWidget);
      expect(find.byType(MultiAddScreen), findsOneWidget);

      await tester.tap(find.text('Keep typing'));
      await tester.pumpAndSettle();
      expect(find.byType(MultiAddScreen), findsOneWidget);
    });

    testWidgets('and throwing away really leaves', (tester) async {
      useTallSurface(tester);
      await tester.pumpWidget(pushable(bare()));
      await tester.pump();
      await tester.tap(find.text('open'));
      await tester.pumpAndSettle();

      await tester.enterText(find.byType(TextField).at(0), 'A');
      await tester.enterText(find.byType(TextField).at(1), '9000000001');
      await tester.pumpAndSettle();

      await goBack(tester);
      await tester.tap(find.text('Throw away'));
      await tester.pumpAndSettle();

      expect(find.byType(MultiAddScreen), findsNothing);
    });

    testWidgets('an untouched sheet leaves without a question', (tester) async {
      useTallSurface(tester);
      await tester.pumpWidget(pushable(bare()));
      await tester.pump();
      await tester.tap(find.text('open'));
      await tester.pumpAndSettle();

      await goBack(tester);

      expect(find.text('Throw these away?'), findsNothing);
      expect(find.byType(MultiAddScreen), findsNothing);
    });

    testWidgets('a row can be removed', (tester) async {
      useTallSurface(tester);
      await tester.pumpWidget(screen());
      await tester.pump();

      await tester.enterText(find.byType(TextField).at(0), 'A');
      await tester.enterText(find.byType(TextField).at(1), '9000000001');
      await tester.pumpAndSettle();
      expect(find.text('Save 1 entry'), findsOneWidget);

      await tester.tap(find.byIcon(Icons.close).first);
      await tester.pumpAndSettle();
      expect(find.text('Nothing to save yet'), findsOneWidget);
    });
  });
}
