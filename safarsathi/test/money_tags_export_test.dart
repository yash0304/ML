// test/money_tags_export_test.dart — what each spend was for, and the
// ledger as a file.
//
// "Downloading the expenses from the Money tab, and tag money spends on
// every line item."

import 'package:drift/native.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:safarsathi/core/database/app_database.dart';
import 'package:safarsathi/core/theme/app_tokens.dart';
import 'package:safarsathi/features/money/data/expense_editor.dart';
import 'package:safarsathi/features/money/data/expense_export.dart';
import 'package:safarsathi/features/money/data/expense_tags.dart';
import 'package:safarsathi/features/money/data/money_summary.dart';
import 'package:safarsathi/features/money/presentation/expense_form_screen.dart';
import 'package:safarsathi/features/money/presentation/money_screen.dart';
import 'package:safarsathi/features/trips/data/trip_editor.dart';

Traveller person(int id, String name) =>
    Traveller(id: id, tripId: 1, name: name, isSelf: id == 1);

LedgerEntry line(int id, String what, int paise, String? tag) => LedgerEntry(
  id: id,
  description: what,
  amountMinor: paise,
  paidByName: 'Yash',
  splitCount: 2,
  spentAt: DateTime(2026, 10, 2),
  tag: tag,
);

void main() {
  group('tags', () {
    test('a first guess from what was typed', () {
      expect(guessTag('Taxi Shillong to Sohra'), 'transport');
      expect(guessTag('Diesel at Umroi'), 'fuel');
      expect(guessTag('Lunch at Laitlum dhaba'), 'food');
      expect(guessTag('Alpha Guest House, 1 night'), 'stay');
      expect(guessTag('Nohkalikai entry ticket'), 'entry');
      expect(guessTag('Guide for the root bridge'), 'guide');
      expect(guessTag('Something unusual'), isNull);
    });

    test('labels, untagged included', () {
      expect(tagLabel('food'), 'Food');
      expect(tagLabel(null), 'Untagged');
    });

    test('where it went adds up to the whole, largest first', () {
      final s = MoneySummary(
        totalMinor: 70000,
        travellerCount: 2,
        balances: const [],
        settlements: const [],
        ledger: [
          line(1, 'Lunch', 20000, 'food'),
          line(2, 'Taxi', 40000, 'transport'),
          line(3, 'Tea', 5000, 'food'),
          line(4, 'Something', 5000, null),
        ],
      );
      final t = s.byTag;
      expect(t.map((x) => x.tag), ['transport', 'food', null]);
      expect(t[1].totalMinor, 25000);
      expect(t[1].count, 2);
      expect(t.fold<int>(0, (a, x) => a + x.totalMinor), 70000);
    });
  });

  group('the form', () {
    final travellers = [person(1, 'Yash'), person(2, 'Priya')];

    Future<ExpenseDraft?> fill(
      WidgetTester tester, {
      ExpenseDraft? existing,
      required Future<void> Function() act,
    }) async {
      tester.view.physicalSize = const Size(420, 1600);
      tester.view.devicePixelRatio = 1.0;
      addTearDown(tester.view.reset);
      ExpenseDraft? saved;
      await tester.pumpWidget(
        MaterialApp(
          theme: AppTokens.light,
          home: ExpenseFormScreen(
            travellers: travellers,
            existing: existing,
            onSave: (d) async => saved = d,
          ),
        ),
      );
      await act();
      await tester.tap(
        find.text(existing == null ? 'Add expense' : 'Save expense'),
      );
      await tester.pump();
      return saved;
    }

    testWidgets('THE TAG FOLLOWS WHAT YOU TYPE until you pick one', (
      tester,
    ) async {
      final saved = await fill(
        tester,
        act: () async {
          await tester.enterText(find.byType(TextField).first, 'Taxi to Sohra');
          await tester.enterText(find.byType(TextField).at(1), '2500');
          await tester.pump();
        },
      );
      expect(saved?.category, 'transport');
    });

    testWidgets('a tag you tap wins over the guess', (tester) async {
      final saved = await fill(
        tester,
        act: () async {
          await tester.tap(find.byKey(const Key('tag-food')));
          await tester.enterText(find.byType(TextField).first, 'Taxi snacks');
          await tester.enterText(find.byType(TextField).at(1), '200');
          await tester.pump();
        },
      );
      expect(saved?.category, 'food');
    });

    testWidgets('tapping the lit tag clears it', (tester) async {
      final saved = await fill(
        tester,
        act: () async {
          await tester.enterText(find.byType(TextField).first, 'Taxi');
          await tester.enterText(find.byType(TextField).at(1), '200');
          await tester.pump();
          await tester.tap(find.byKey(const Key('tag-transport')));
          await tester.pump();
        },
      );
      expect(saved?.category, isNull);
    });

    testWidgets('AN OLD UNTAGGED EXPENSE OPENS WITH A GUESS LIT', (
      tester,
    ) async {
      // Saved before tags existed. Opening it and pressing Save is the tag.
      final saved = await fill(
        tester,
        existing: ExpenseDraft(
          id: 5,
          description: 'Taxi, Shillong to Cherrapunji',
          amountMinor: 320011,
          paidById: 1,
          shares: const {1: 160006, 2: 160005},
          spentAt: DateTime(2026, 10, 2),
        ),
        act: () async {},
      );
      expect(saved?.category, 'transport');
    });

    testWidgets('an edited expense keeps its tag', (tester) async {
      final saved = await fill(
        tester,
        existing: ExpenseDraft(
          id: 4,
          description: 'Lunch',
          amountMinor: 20000,
          paidById: 1,
          shares: const {1: 10000, 2: 10000},
          spentAt: DateTime(2026, 10, 2),
          category: 'shopping',
        ),
        act: () async {
          await tester.enterText(find.byType(TextField).first, 'Lunch thali');
          await tester.pump();
        },
      );
      expect(saved?.category, 'shopping');
    });
  });

  group('the Money tab', () {
    Future<void> pump(WidgetTester tester, {Future<void> Function()? onExport}) async {
      tester.view.physicalSize = const Size(420, 2400);
      tester.view.devicePixelRatio = 1.0;
      addTearDown(tester.view.reset);
      await tester.pumpWidget(
        MaterialApp(
          theme: AppTokens.light,
          home: MoneyScreen(
            onExport: onExport,
            summary: Stream.value(
              MoneySummary(
                totalMinor: 65000,
                travellerCount: 2,
                balances: const [],
                settlements: const [],
                ledger: [
                  line(1, 'Lunch', 20000, 'food'),
                  line(2, 'Taxi', 40000, 'transport'),
                  line(3, 'Tea', 5000, 'food'),
                ],
              ),
            ),
          ),
        ),
      );
      await tester.pump();
    }

    testWidgets('every line says its tag; the totals say where it went', (
      tester,
    ) async {
      await pump(tester);
      expect(find.textContaining('Food · Yash paid'), findsNWidgets(2));
      expect(find.text('WHERE IT WENT'), findsOneWidget);
      expect(find.text('Transport · 1'), findsOneWidget);
      expect(find.text('Food · 2'), findsOneWidget);
    });

    testWidgets('a tag filters the ledger, and SHOW ALL undoes it', (
      tester,
    ) async {
      await pump(tester);
      await tester.tap(find.byKey(const Key('tag-total-food')));
      await tester.pump();
      expect(find.text('Taxi'), findsNothing);
      expect(find.text('Lunch'), findsOneWidget);
      expect(find.text('LEDGER · FOOD'), findsOneWidget);

      await tester.tap(find.byKey(const Key('ledger-show-all')));
      await tester.pump();
      expect(find.text('Taxi'), findsOneWidget);
    });

    testWidgets('untagged lines offer to be tagged in one tap', (
      tester,
    ) async {
      tester.view.physicalSize = const Size(420, 2400);
      tester.view.devicePixelRatio = 1.0;
      addTearDown(tester.view.reset);
      var asked = 0;
      await tester.pumpWidget(
        MaterialApp(
          theme: AppTokens.light,
          home: MoneyScreen(
            onTagUntagged: () async {
              asked++;
              return 1;
            },
            summary: Stream.value(
              MoneySummary(
                totalMinor: 25000,
                travellerCount: 2,
                balances: const [],
                settlements: const [],
                ledger: [
                  line(1, 'Lunch', 20000, null),
                  line(2, 'Odd thing', 5000, null),
                ],
              ),
            ),
          ),
        ),
      );
      await tester.pump();
      await tester.tap(find.byKey(const Key('tag-untagged')));
      await tester.pump();
      expect(asked, 1);
      expect(find.textContaining('Tagged 1 of 2'), findsOneWidget);
    });

    testWidgets('EXPORT is offered and asks for the file', (tester) async {
      var exports = 0;
      await pump(tester, onExport: () async => exports++);
      await tester.tap(find.byKey(const Key('money-export')));
      expect(exports, 1);
    });
  });

  group('the exported file', () {
    late AppDatabase db;
    late int tripId, yash, priya, sohra;

    setUp(() async {
      db = AppDatabase(NativeDatabase.memory());
      final editor = TripEditor(db);
      tripId = await editor.createTrip(name: 'Meghalaya, Oct');
      sohra = await editor.addStop(tripId, const StopDraft(name: 'Sohra'));
      yash = await db.into(db.travellers).insert(
        TravellersCompanion.insert(tripId: tripId, name: 'Yash'),
      );
      priya = await db.into(db.travellers).insert(
        TravellersCompanion.insert(tripId: tripId, name: 'Priya'),
      );
      final money = ExpenseEditor(db);
      await money.saveExpense(
        tripId,
        ExpenseDraft(
          description: 'Taxi, Shillong to "Sohra"',
          amountMinor: 320011,
          paidById: yash,
          shares: {yash: 160006, priya: 160005},
          spentAt: DateTime(2026, 10, 2),
          category: 'transport',
          stopId: sohra,
        ),
      );
      await money.saveExpense(
        tripId,
        ExpenseDraft(
          description: 'Tea',
          amountMinor: 4000,
          paidById: priya,
          shares: {priya: 4000},
          spentAt: DateTime(2026, 10, 1),
        ),
      );
    });
    tearDown(() => db.close());

    test('THE WHOLE LEDGER, as a spreadsheet opens it', () async {
      final csv = await expensesCsv(db, tripId);
      expect(csv.startsWith('﻿'), isTrue, reason: 'Excel needs the BOM');
      final lines = csv.substring(1).split('\n');
      expect(lines[0],
          'Date,What for,Tag,Amount (INR),Paid by,Yash share,Priya share,Stop');
      // Oldest first, as a ledger reads.
      expect(lines[1], '2026-10-01,Tea,Untagged,40.00,Priya,,40.00,');
      // Quotes and a comma in the description survive.
      expect(
        lines[2],
        '2026-10-02,"Taxi, Shillong to ""Sohra""",Transport,3200.11,Yash,'
        '1600.06,1600.05,Sohra',
      );
      expect(csv, contains('Total,,,3240.11'));
      expect(csv, contains('By tag,,,Amount (INR)\nTransport,,,3200.11\n'
          'Untagged,,,40.00'));
      // Priya owes Yash her share of the taxi, less nothing: 1600.05.
      expect(csv, contains('Priya,Yash,,1600.05'));
    });

    test('a file name that sorts and says what it is', () {
      expect(
        expensesFileName('Meghalaya, Oct', DateTime(2026, 10, 6)),
        'Meghalaya-Oct-expenses-2026-10-06.csv',
      );
      expect(expensesFileName('…', DateTime(2026, 10, 6)),
          'Trip-expenses-2026-10-06.csv');
    });

    test('UNTAGGED ONES ARE TAGGED FROM WHAT THEY SAY — only on a tap, '
        'and never over a chosen tag', () async {
      final money = ExpenseEditor(db);
      await money.saveExpense(
        tripId,
        ExpenseDraft(
          description: 'Diesel',
          amountMinor: 210000,
          paidById: yash,
          shares: {yash: 210000},
          spentAt: DateTime(2026, 10, 3),
        ),
      );
      await money.saveExpense(
        tripId,
        ExpenseDraft(
          description: 'Lunch',
          amountMinor: 30000,
          paidById: yash,
          shares: {yash: 30000},
          spentAt: DateTime(2026, 10, 3),
          category: 'shopping', // odd, but chosen
        ),
      );
      // Tea (untagged → food), Diesel (→ fuel), Lunch stays shopping,
      // the taxi keeps transport.
      expect(await money.tagUntaggedFromDescriptions(tripId), 2);
      final byName = {
        for (final e in await db.select(db.expenses).get())
          e.description: e.category,
      };
      expect(byName['Tea'], 'food');
      expect(byName['Diesel'], 'fuel');
      expect(byName['Lunch'], 'shopping');
      expect(byName['Taxi, Shillong to "Sohra"'], 'transport');
    });

    test('tags reach the database', () async {
      final rows = await db.select(db.expenses).get();
      expect(rows.map((e) => e.category), containsAll(['transport', null]));
    });
  });
}
