// test/currency_test.dart — #39, spending in more than one currency.
//
// The base is rupees. An expense keeps its own currency, amount and the rate
// it was saved with; balances settle exactly in paise.

import 'dart:convert';
import 'dart:io';

import 'package:drift/native.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:safarsathi/core/database/app_database.dart';
import 'package:safarsathi/core/theme/app_tokens.dart';
import 'package:safarsathi/features/backup/data/backup.dart';
import 'package:safarsathi/features/money/data/currency.dart';
import 'package:safarsathi/features/money/data/expense_editor.dart';
import 'package:safarsathi/features/money/data/expense_export.dart';
import 'package:safarsathi/features/money/data/money_summary.dart';
import 'package:safarsathi/features/money/presentation/currencies_screen.dart';
import 'package:safarsathi/features/money/presentation/expense_form_screen.dart';
import 'package:safarsathi/features/trips/data/trip_editor.dart';

CurrencyRate eur({double rate = 92.4}) => CurrencyRate(
  id: 1,
  tripId: 1,
  code: 'EUR',
  rateToBase: rate,
  capturedAt: DateTime(2026, 10, 6),
);

void main() {
  group('reading and writing money', () {
    test('foreign amounts with their sign, rupees as always', () {
      expect(formatMoney(1250, 'EUR'), '€12.50');
      expect(formatMoney(125000, 'BDT'), '৳1,250.00');
      expect(formatMoney(34000, 'CZK'), 'CZK 340.00');
      expect(formatMoney(12345678, 'INR'), '₹1,23,456.78');
    });

    test('a rate is a positive number; a code is three letters', () {
      expect(parseRate('92.40'), 92.4);
      expect(parseRate('92,40'), 92.4);
      expect(parseRate('0'), isNull);
      expect(parseRate('abc'), isNull);
      expect(parseCurrencyCode(' eur '), 'EUR');
      expect(parseCurrencyCode('EURO'), isNull);
    });

    test('the rate reads with its date', () {
      expect(describeRate(eur()), '1 EUR = ₹92.40 · saved 6 Oct');
    });
  });

  group('the ledger', () {
    late AppDatabase db;
    late int tripId, a, b, c;
    late ExpenseEditor money;

    setUp(() async {
      db = AppDatabase(NativeDatabase.memory());
      tripId = await TripEditor(db).createTrip(name: 'Europe');
      Future<int> person(String n) => db.into(db.travellers).insert(
        TravellersCompanion.insert(tripId: tripId, name: n),
      );
      a = await person('Yash');
      b = await person('Priya');
      c = await person('Ankit');
      money = ExpenseEditor(db);
    });
    tearDown(() => db.close());

    test('A EURO DINNER SPLIT THREE WAYS SETTLES EXACTLY IN PAISE', () async {
      // €100.00 split 33.34 / 33.33 / 33.33 at an awkward rate.
      await money.saveExpense(
        tripId,
        ExpenseDraft(
          description: 'Dinner',
          amountMinor: 10000,
          paidById: a,
          shares: {a: 3334, b: 3333, c: 3333},
          spentAt: DateTime(2026, 10, 6),
          currency: 'EUR',
          rateToBase: 92.4137,
          rateCapturedAt: DateTime(2026, 10, 5),
        ),
      );
      final s = await watchMoneySummary(db, tripId).first;
      final net = s.balances.fold<int>(0, (x, y) => x + y.netMinor);
      expect(net, 0, reason: 'what one is owed, the others owe, to the paisa');
      final e = s.ledger.single;
      expect(e.currency, 'EUR');
      expect(e.originalMinor, 10000);
      expect(
        e.amountMinor,
        toBaseMinor(3334, 92.4137) + 2 * toBaseMinor(3333, 92.4137),
      );
    });

    test('CHANGING A RATE NEVER REWRITES WHAT WAS ALREADY SPENT', () async {
      await saveCurrencyRate(db, tripId: tripId, code: 'EUR', rate: 90);
      await money.saveExpense(
        tripId,
        ExpenseDraft(
          description: 'Train',
          amountMinor: 5000,
          paidById: a,
          shares: {a: 5000},
          spentAt: DateTime(2026, 10, 6),
          currency: 'EUR',
          rateToBase: 90,
        ),
      );
      final before = (await watchMoneySummary(db, tripId).first).totalMinor;
      await saveCurrencyRate(db, tripId: tripId, code: 'EUR', rate: 95);

      expect((await db.select(db.currencyRates).get()), hasLength(1));
      expect((await db.select(db.currencyRates).getSingle()).rateToBase, 95);
      expect((await watchMoneySummary(db, tripId).first).totalMinor, before);
    });

    test('an INR-only trip reads exactly as before', () async {
      await money.saveExpense(
        tripId,
        ExpenseDraft(
          description: 'Tea',
          amountMinor: 4000,
          paidById: a,
          shares: {a: 2000, b: 2000},
          spentAt: DateTime(2026, 10, 6),
        ),
      );
      final s = await watchMoneySummary(db, tripId).first;
      expect(s.totalMinor, 4000);
      expect(s.ledger.single.isForeign, isFalse);
    });

    test('the export shows the rupees and what was spent', () async {
      await money.saveExpense(
        tripId,
        ExpenseDraft(
          description: 'Museum',
          amountMinor: 2000,
          paidById: b,
          shares: {b: 1000, c: 1000},
          spentAt: DateTime(2026, 10, 6),
          currency: 'EUR',
          rateToBase: 92.4,
        ),
      );
      final csv = await expensesCsv(db, tripId);
      expect(csv, contains('Museum,Untagged,1848.00,EUR,20.00,92.4,Priya,'
          ',924.00,924.00,'));
    });

    test('INR is not a rate', () async {
      expect(
        () => saveCurrencyRate(db, tripId: tripId, code: 'INR', rate: 1),
        throwsArgumentError,
      );
    });

    test('a backup carries the rates; a v8 backup restores without them',
        () async {
      await saveCurrencyRate(db, tripId: tripId, code: 'BDT', rate: 0.72);
      final fresh = AppDatabase(NativeDatabase.memory());
      addTearDown(fresh.close);
      await restoreBackup(fresh, readBackup(await exportBackup(db)));
      expect((await fresh.select(fresh.currencyRates).getSingle()).code, 'BDT');

      final json = jsonDecode(await exportBackup(db)) as Map<String, dynamic>;
      json['schemaVersion'] = 8;
      (json['tables'] as Map).remove('currencyRates');
      final older = AppDatabase(NativeDatabase.memory());
      addTearDown(older.close);
      await restoreBackup(older, readBackup(jsonEncode(json)));
      expect(await older.select(older.currencyRates).get(), isEmpty);
    });
  });

  test('A REAL v8 DATABASE UPGRADES TO v9', () async {
    final dir = await Directory.systemTemp.createTemp('upgrade9');
    addTearDown(() => dir.delete(recursive: true));
    final file = File('${dir.path}/v8.sqlite');
    final v8 = AppDatabase(NativeDatabase(file));
    await v8.into(v8.trips).insert(TripsCompanion.insert(name: 'Meghalaya'));
    await v8.customStatement('DROP TABLE currency_rates');
    await v8.customStatement('PRAGMA user_version = 8');
    await v8.close();

    final v9 = AppDatabase(NativeDatabase(file));
    addTearDown(v9.close);
    expect((await v9.select(v9.trips).getSingle()).name, 'Meghalaya');
    await saveCurrencyRate(v9, tripId: 1, code: 'EUR', rate: 92.4);
    expect(await v9.select(v9.currencyRates).get(), hasLength(1));
  });

  group('the screens', () {
    final travellers = [
      const Traveller(id: 1, tripId: 1, name: 'Yash', isSelf: true),
      const Traveller(id: 2, tripId: 1, name: 'Priya', isSelf: false),
    ];

    Future<ExpenseDraft?> form(
      WidgetTester tester, {
      List<CurrencyRate> currencies = const [],
      Future<void> Function()? act,
    }) async {
      tester.view.physicalSize = const Size(420, 1800);
      tester.view.devicePixelRatio = 1.0;
      addTearDown(tester.view.reset);
      ExpenseDraft? saved;
      await tester.pumpWidget(
        MaterialApp(
          theme: AppTokens.light,
          home: ExpenseFormScreen(
            travellers: travellers,
            currencies: currencies,
            onSave: (d) async => saved = d,
          ),
        ),
      );
      await act?.call();
      await tester.tap(find.text('Add expense'));
      await tester.pump();
      return saved;
    }

    testWidgets('a trip with no saved currency sees no choice', (
      tester,
    ) async {
      await form(
        tester,
        act: () async {
          await tester.enterText(find.byType(TextField).first, 'Tea');
          await tester.enterText(find.byType(TextField).at(1), '40');
          await tester.pump();
        },
      );
      expect(find.byKey(const Key('currency-INR')), findsNothing);
      expect(find.byKey(const Key('expense-in-rupees')), findsNothing);
    });

    testWidgets('PICKING EUROS SAYS WHAT IT COMES TO IN RUPEES, and keeps '
        'the rate', (tester) async {
      final saved = await form(
        tester,
        currencies: [eur()],
        act: () async {
          await tester.tap(find.byKey(const Key('currency-EUR')));
          await tester.enterText(find.byType(TextField).first, 'Dinner');
          await tester.enterText(find.byType(TextField).at(1), '40');
          await tester.pump();
          expect(
            find.textContaining('= ₹3,696 at 1 EUR = ₹92.40'),
            findsOneWidget,
          );
          expect(find.text('€20.00'), findsNWidgets(2));
        },
      );
      expect(saved?.currency, 'EUR');
      expect(saved?.amountMinor, 4000);
      expect(saved?.rateToBase, 92.4);
      expect(saved?.rateCapturedAt, DateTime(2026, 10, 6));
    });

    testWidgets('the currencies screen refuses nonsense and saves a rate', (
      tester,
    ) async {
      final saved = <(String, double)>[];
      await tester.pumpWidget(
        MaterialApp(
          theme: AppTokens.light,
          home: CurrenciesScreen(
            rates: Stream.value([eur()]),
            onSave: (code, rate) async => saved.add((code, rate)),
            onDelete: (_) async {},
          ),
        ),
      );
      await tester.pump();
      expect(find.text('1 EUR = ₹92.40 · saved 6 Oct'), findsOneWidget);

      await tester.enterText(find.byKey(const Key('currency-code')), 'inr');
      await tester.enterText(find.byKey(const Key('currency-rate')), '1');
      await tester.tap(find.byKey(const Key('currency-save')));
      await tester.pump();
      expect(find.textContaining('need no rate'), findsOneWidget);

      await tester.enterText(find.byKey(const Key('currency-code')), 'bdt');
      await tester.enterText(find.byKey(const Key('currency-rate')), '0.72');
      await tester.tap(find.byKey(const Key('currency-save')));
      await tester.pump();
      expect(saved, [('BDT', 0.72)]);
    });
  });
}
