// test/live_screens_test.dart — screens that keep up, and survive a scroll.
//
// Two bugs reported from the phone, one cause each:
//
//  * "I am not getting expenses updated directly; I need to exit the app and
//    go back to the Money tab." Every screen ticked on the same SQL,
//    'SELECT 1', and drift shares an active stream between queries with the
//    same SQL — readsFrom is not part of the key. So the Money tab shared the
//    active-trip stream's trigger and only moved when a stop changed.
//
//  * "On the home page, scrolling down and back up, it comes up blank." The
//    readiness stream allowed one listener ever; a section scrolled out of a
//    ListView and back listens again, which threw.
//
// The tests here run streams side by side, the way the app does — the thing
// the per-screen tests never did.

import 'dart:async';

import 'package:drift/native.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:safarsathi/core/database/app_database.dart';
import 'package:safarsathi/core/database/watch_tables.dart';
import 'package:safarsathi/core/theme/app_tokens.dart';
import 'package:safarsathi/core/util/streams.dart';
import 'package:safarsathi/features/money/data/expense_editor.dart';
import 'package:safarsathi/features/money/data/money_summary.dart';
import 'package:safarsathi/features/trips/data/readiness.dart';
import 'package:safarsathi/features/trips/data/trip_editor.dart';
import 'package:safarsathi/features/trips/data/trip_summary.dart';
import 'package:safarsathi/features/trips/presentation/trip_screen.dart';

/// A stream any number of widgets can listen to, each getting [value] at
/// once — what a Drift stream does, without a database in a widget test.
Stream<T> replaying<T>(T value) =>
    Stream.multi((c) => c.add(value), isBroadcast: true);

void main() {
  group('the Money tab keeps up', () {
    late AppDatabase db;
    late int tripId, me;

    setUp(() async {
      db = AppDatabase(NativeDatabase.memory());
      tripId = await TripEditor(db).createTrip(name: 'Meghalaya');
      me = await db.into(db.travellers).insert(
        TravellersCompanion.insert(tripId: tripId, name: 'Yash'),
      );
    });
    tearDown(() => db.close());

    Future<void> spend(String what, int paise) =>
        ExpenseEditor(db).saveExpense(
          tripId,
          ExpenseDraft(
            description: what,
            amountMinor: paise,
            paidById: me,
            shares: {me: paise},
            spentAt: DateTime(2026, 10, 2),
          ),
        );

    test('WHY: drift shares "SELECT 1" streams whatever they read', () async {
      // Pinned so the reason for watchTables cannot quietly stop being true
      // or be forgotten. If drift ever keys on readsFrom, this fails and the
      // helper's comment needs updating — the helper itself stays correct.
      final trips = db
          .customSelect('SELECT 1', readsFrom: {db.trips})
          .watch()
          .listen((_) {});
      var fired = 0;
      final expenses = db
          .customSelect('SELECT 1', readsFrom: {db.expenses})
          .watch()
          .listen((_) => fired++);
      await pumpEventQueue();
      final before = fired;
      await spend('Tea', 5000);
      await pumpEventQueue();
      expect(fired, before, reason: 'shared the trips stream, so missed it');
      await trips.cancel();
      await expenses.cancel();
    });

    test('A SAVED EXPENSE REACHES THE MONEY TAB with other screens open',
        () async {
      // The active-trip stream is the first thing the app listens to.
      final trip = watchActiveTripContext(db).listen((_) {});
      final readiness = watchReadiness(db, tripId).listen((_) {});
      final totals = <int>[];
      final money = watchMoneySummary(
        db,
        tripId,
      ).listen((s) => totals.add(s.totalMinor));
      await pumpEventQueue();

      await spend('Lunch', 30000);
      await pumpEventQueue();
      expect(totals.last, 30000);

      await spend('Taxi', 250000);
      await pumpEventQueue();
      expect(totals.last, 280000);

      await trip.cancel();
      await readiness.cancel();
      await money.cancel();
    });

    test('different table sets get different streams', () async {
      var stops = 0;
      var expenses = 0;
      final a = watchTables(db, {db.stops}).listen((_) => stops++);
      final b = watchTables(db, {db.expenses}).listen((_) => expenses++);
      await pumpEventQueue();
      await spend('Tea', 5000);
      await pumpEventQueue();
      expect(expenses, 2, reason: 'once on listen, once on the write');
      expect(stops, 1, reason: 'stops were not written');
      await a.cancel();
      await b.cancel();
    });
  });

  test('THE READINESS STREAM CAN BE LISTENED TO AGAIN', () async {
    // What a ListView does to a section scrolled away and back.
    final s = combineLatest2(replaying(1), replaying(2), (int a, int b) => a + b);
    final first = await s.first;
    final second = await s.first;
    expect([first, second], [3, 3]);
  });

  testWidgets('SCROLLING THE TRIP PAGE DOWN AND BACK UP KEEPS IT ALL', (
    tester,
  ) async {
    tester.view.physicalSize = const Size(420, 700);
    tester.view.devicePixelRatio = 1.0;
    addTearDown(tester.view.reset);

    await tester.pumpWidget(
      MaterialApp(
        theme: AppTokens.light,
        home: TripScreen(
          trip: replaying(
            TripSummary(
              name: 'Meghalaya',
              stops: [
                for (var i = 1; i <= 8; i++)
                  StopSummary(
                    id: i,
                    name: 'Stop $i',
                    sequenceOrder: i,
                    nights: 1,
                    diaryCount: 2,
                    isCurrent: false,
                  ),
              ],
            ),
          ),
          unconfirmedCount: replaying(0),
          // Built the way the app builds it.
          readiness: combineLatest2(
            replaying(
              const Readiness([
                ReadinessItem(
                  stopId: 1,
                  stopName: 'Stop 1',
                  label: 'Choose where you are staying in Stop 1.',
                ),
              ]),
            ),
            replaying(0),
            (Readiness r, int _) => r,
          ),
        ),
      ),
    );
    await tester.pump();
    await tester.pump();
    expect(find.text('Choose where you are staying in Stop 1.'), findsOneWidget);

    await tester.drag(find.byType(ListView), const Offset(0, -3000));
    await tester.pump();
    await tester.drag(find.byType(ListView), const Offset(0, 3000));
    await tester.pump();
    await tester.pump();

    expect(tester.takeException(), isNull);
    expect(find.text('Choose where you are staying in Stop 1.'), findsOneWidget);
    expect(find.text('Meghalaya'), findsOneWidget);
  });
}
