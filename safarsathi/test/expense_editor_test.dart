// test/expense_editor_test.dart — issues #31 and #32
//
// The ledger has to balance exactly. Every test here is some version of that.

import 'package:drift/native.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:safarsathi/core/database/app_database.dart';
import 'package:safarsathi/features/money/data/expense_editor.dart';
import 'package:safarsathi/features/money/data/money_summary.dart';
import 'package:safarsathi/features/money/data/settlement.dart';
import 'package:safarsathi/features/trips/data/trip_editor.dart';

void main() {
  group('parsing what people type', () {
    test('plain rupees', () {
      expect(parseRupees('340'), 34000);
      expect(parseRupees('0'), 0);
    });

    test('paise', () {
      expect(parseRupees('340.50'), 34050);
      expect(parseRupees('340.5'), 34050);
      expect(parseRupees('.75'), 75);
    });

    test('Indian grouping and a rupee sign', () {
      expect(parseRupees('1,234.50'), 123450);
      expect(parseRupees('₹340'), 34000);
      expect(parseRupees('  ₹ 1,23,456  '), 12345600);
    });

    test('MORE THAN TWO DECIMALS TRUNCATE, never round up', () {
      // Rounding 340.567 to 340.57 invents a paisa nobody spent, and the
      // ledger would be out by it forever.
      expect(parseRupees('340.567'), 34056);
      expect(parseRupees('340.999'), 34099);
    });

    test('nonsense is null, not zero', () {
      // Silently recording zero would put a free taxi in the ledger.
      expect(parseRupees(''), isNull);
      expect(parseRupees('abc'), isNull);
      expect(parseRupees('1.2.3'), isNull);
      expect(parseRupees('₹'), isNull);
    });

    test('round-trips through the field text', () {
      for (final minor in [0, 1, 99, 100, 34050, 12345600]) {
        expect(parseRupees(rupeesFieldText(minor)), minor, reason: '$minor');
      }
    });
  });

  group('even splits', () {
    test('a clean division', () {
      expect(evenSplit(30000, [1, 2, 3]).values, [10000, 10000, 10000]);
    });

    test('THE REMAINDER IS HANDED OUT, never lost', () {
      final split = evenSplit(10000, [1, 2, 3]);
      expect(split.values.reduce((a, b) => a + b), 10000);
      expect(split.values, [3334, 3333, 3333]);
    });

    test('sums back to the total for every party size', () {
      for (var n = 1; n <= 12; n++) {
        final ids = [for (var i = 0; i < n; i++) i];
        final split = evenSplit(320011, ids);
        expect(
          split.values.fold(0, (a, b) => a + b),
          320011,
          reason: 'n = $n',
        );
      }
    });

    test('nobody to split between is empty, not a crash', () {
      expect(evenSplit(10000, const []), isEmpty);
    });
  });

  group('simplify debts', () {
    test('THE THREE-PERSON MEGHALAYA LEDGER settles in two payments', () {
      // Yash pays the ₹3,200.11 taxi and ₹2,100 of fuel; Priya pays the
      // ₹1,500 cave guide; Ankit pays the ₹860 dinner. All three ways.
      const total = 320011 + 210000 + 150000 + 86000;
      final share = evenShares(total, 3);

      final balances = [
        Balance(
          travellerId: 1,
          name: 'Yash',
          netMinor: 320011 + 210000 - share[0],
        ),
        Balance(travellerId: 2, name: 'Priya', netMinor: 150000 - share[1]),
        Balance(travellerId: 3, name: 'Ankit', netMinor: 86000 - share[2]),
      ];

      final settlements = simplifyDebts(balances);

      expect(settlements.length, 2);
      expect(settlements.every((s) => s.toName == 'Yash'), isTrue);
      // The payments clear exactly what Yash is owed.
      expect(
        settlements.fold(0, (a, s) => a + s.amountMinor),
        balances.first.netMinor,
      );
    });

    test('A CYCLE COLLAPSES TO NOTHING', () {
      // A owes B, B owes C, C owes A, all the same amount. Everyone is square
      // and no payment should be proposed at all.
      final settlements = simplifyDebts(const [
        Balance(travellerId: 1, name: 'A', netMinor: 0),
        Balance(travellerId: 2, name: 'B', netMinor: 0),
        Balance(travellerId: 3, name: 'C', netMinor: 0),
      ]);
      expect(settlements, isEmpty);
    });

    test('payments always sum back to the credits', () {
      final balances = [
        const Balance(travellerId: 1, name: 'A', netMinor: 15000),
        const Balance(travellerId: 2, name: 'B', netMinor: 4500),
        const Balance(travellerId: 3, name: 'C', netMinor: -12000),
        const Balance(travellerId: 4, name: 'D', netMinor: -7500),
      ];
      final settlements = simplifyDebts(balances);

      expect(settlements.fold(0, (a, s) => a + s.amountMinor), 19500);
      // Never more payments than there are people.
      expect(settlements.length, lessThanOrEqualTo(balances.length));
    });
  });

  group('against the database', () {
    late AppDatabase db;
    late ExpenseEditor money;
    late int tripId;
    late int yash;
    late int priya;
    late int ankit;

    setUp(() async {
      db = AppDatabase(NativeDatabase.memory());
      money = ExpenseEditor(db);
      tripId = await TripEditor(db).createTrip(name: 'Meghalaya');
      yash = await money.addTraveller(tripId, 'Yash');
      priya = await money.addTraveller(tripId, 'Priya');
      ankit = await money.addTraveller(tripId, 'Ankit');
    });

    tearDown(() => db.close());

    Future<int> spend(
      String what,
      int minor,
      int paidBy, {
      List<int>? between,
    }) => money.saveExpense(
      tripId,
      ExpenseDraft(
        description: what,
        amountMinor: minor,
        paidById: paidBy,
        shares: evenSplit(minor, between ?? [yash, priya, ankit]),
        spentAt: DateTime(2026, 10, 3),
      ),
    );

    test('an expense and its splits land together', () async {
      final id = await spend('Taxi', 320011, yash);

      final expense = await (db.select(
        db.expenses,
      )..where((e) => e.id.equals(id))).getSingle();
      expect(expense.amountMinor, 320011);

      final shares = await money.sharesOf(id);
      expect(shares.length, 3);
      expect(shares.values.fold(0, (a, b) => a + b), 320011);
    });

    test('AN UNBALANCED SPLIT IS REFUSED', () async {
      expect(
        () => money.saveExpense(
          tripId,
          ExpenseDraft(
            description: 'Wrong',
            amountMinor: 10000,
            paidById: yash,
            shares: {yash: 3000, priya: 3000},
            spentAt: DateTime(2026, 10, 3),
          ),
        ),
        throwsArgumentError,
      );
      expect(await db.select(db.expenses).get(), isEmpty);
    });

    test('editing replaces the splits wholesale', () async {
      final id = await spend('Taxi', 30000, yash);
      expect((await money.sharesOf(id)).length, 3);

      await money.saveExpense(
        tripId,
        ExpenseDraft(
          id: id,
          description: 'Taxi, just us two',
          amountMinor: 30000,
          paidById: yash,
          shares: evenSplit(30000, [yash, priya]),
          spentAt: DateTime(2026, 10, 3),
        ),
      );

      final shares = await money.sharesOf(id);
      expect(shares.length, 2);
      expect(shares.containsKey(ankit), isFalse);
      expect(shares.values.fold(0, (a, b) => a + b), 30000);
    });

    test('deleting an expense takes its splits', () async {
      final id = await spend('Taxi', 30000, yash);
      await money.deleteExpense(id);
      expect(await db.select(db.expenseSplits).get(), isEmpty);
    });

    test('A TRAVELLER IN THE LEDGER CANNOT BE REMOVED', () async {
      // Cascading would quietly change everyone else's balance: the expense
      // keeps its total but loses a share.
      await spend('Taxi', 30000, yash);
      final priyaRow = (await money.travellersOf(tripId)).firstWhere(
        (t) => t.id == priya,
      );

      expect(
        () => money.deleteTraveller(priyaRow),
        throwsA(isA<TravellerInUse>()),
      );
      expect((await money.travellersOf(tripId)).length, 3);
    });

    test('a traveller with nothing against them can be removed', () async {
      final spare = await money.addTraveller(tripId, 'Guest');
      final row = (await money.travellersOf(
        tripId,
      )).firstWhere((t) => t.id == spare);

      await money.deleteTraveller(row);
      expect((await money.travellersOf(tripId)).length, 3);
    });

    test('the payer counts even when they are not in the split', () async {
      await money.saveExpense(
        tripId,
        ExpenseDraft(
          description: 'Treat',
          amountMinor: 30000,
          paidById: yash,
          shares: evenSplit(30000, [priya, ankit]),
          spentAt: DateTime(2026, 10, 3),
        ),
      );

      final row = (await money.travellersOf(tripId)).firstWhere(
        (t) => t.id == yash,
      );
      expect(
        () => money.deleteTraveller(row),
        throwsA(isA<TravellerInUse>()),
      );
    });

    group('the summary', () {
      test('balances net out to zero across everyone', () async {
        await spend('Taxi', 320011, yash);
        await spend('Guide', 150000, priya);
        await spend('Dinner', 86000, ankit);

        final summary = await watchMoneySummary(db, tripId).first;
        expect(
          summary.balances.fold(0, (a, b) => a + b.netMinor),
          0,
        );
        expect(summary.totalMinor, 320011 + 150000 + 86000);
      });

      test('the settle-up clears every balance exactly', () async {
        await spend('Taxi', 320011, yash);
        await spend('Guide', 150000, priya);
        await spend('Dinner', 86000, ankit);

        final summary = await watchMoneySummary(db, tripId).first;
        final owed = {for (final b in summary.balances) b.name: b.netMinor};

        for (final s in summary.settlements) {
          owed[s.fromName] = (owed[s.fromName] ?? 0) + s.amountMinor;
          owed[s.toName] = (owed[s.toName] ?? 0) - s.amountMinor;
        }
        expect(owed.values.every((v) => v == 0), isTrue);
      });

      test('THE SUMMARY UPDATES WHEN A TRAVELLER IS ADDED', () async {
        // The stream used to watch only the expenses table, so adding a
        // traveller left the screen stale. Invisible until #31 made it
        // possible to add one.
        final before = await watchMoneySummary(db, tripId).first;
        expect(before.travellerCount, 3);

        final stream = watchMoneySummary(db, tripId);
        final next = stream.skip(1).first;
        await money.addTraveller(tripId, 'Guest');

        expect((await next).travellerCount, 4);
      });

      test('the ledger is newest first', () async {
        await money.saveExpense(
          tripId,
          ExpenseDraft(
            description: 'Old',
            amountMinor: 1000,
            paidById: yash,
            shares: {yash: 1000},
            spentAt: DateTime(2026, 10, 1),
          ),
        );
        await money.saveExpense(
          tripId,
          ExpenseDraft(
            description: 'New',
            amountMinor: 1000,
            paidById: yash,
            shares: {yash: 1000},
            spentAt: DateTime(2026, 10, 5),
          ),
        );

        final summary = await watchMoneySummary(db, tripId).first;
        expect(summary.ledger.map((e) => e.description), ['New', 'Old']);
      });
    });
  });
}
