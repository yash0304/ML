// test/checklist_money_screens_test.dart — the screens for #29 and #31.

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:safarsathi/core/database/app_database.dart';
import 'package:safarsathi/core/theme/app_tokens.dart';
import 'package:safarsathi/features/checklist/data/checklist_dao.dart';
import 'package:safarsathi/features/checklist/presentation/checklist_screen.dart';
import 'package:safarsathi/features/money/data/expense_editor.dart';
import 'package:safarsathi/features/money/presentation/expense_form_screen.dart';
import 'package:safarsathi/features/money/presentation/travellers_screen.dart';

Widget wrap(Widget child, {Brightness brightness = Brightness.light}) =>
    MaterialApp(
      theme: brightness == Brightness.light ? AppTokens.light : AppTokens.dark,
      home: child,
    );

ChecklistItem item(
  int id,
  String label, {
  bool done = false,
  bool blocking = false,
  String? quantity,
  String tags = '',
  bool edited = false,
}) => ChecklistItem(
  id: id,
  tripId: 1,
  label: label,
  quantity: quantity,
  sourceTags: tags,
  isDone: done,
  isBlocking: blocking,
  isGenerated: true,
  isUserEdited: edited,
  sortOrder: id,
);

Traveller person(int id, String name) =>
    Traveller(id: id, tripId: 1, name: name, isSelf: id == 1);

void main() {
  group('checklist', () {
    Widget screen(ChecklistView view) => wrap(
      ChecklistScreen(
        checklist: Stream.value(view),
        onToggle: (_, _) async {},
        onRemove: (_) async {},
        onEdit: (_, _, _) async {},
        onAdd: (_) async {},
      ),
    );

    testWidgets('blocking items come first, in their own section', (
      tester,
    ) async {
      await tester.pumpWidget(
        screen(
          ChecklistView(
            blocking: [item(1, 'Call the Shillong homestay', blocking: true)],
            pack: [item(2, 'Headtorch', tags: 'caves')],
          ),
        ),
      );
      await tester.pump();

      expect(find.text('BEFORE YOU LEAVE SIGNAL'), findsOneWidget);
      expect(find.text('PACK'), findsOneWidget);
      expect(find.text('BLOCKING'), findsOneWidget);
    });

    testWidgets('an item shows the tags that produced it', (tester) async {
      await tester.pumpWidget(
        screen(
          ChecklistView(
            blocking: const [],
            pack: [item(2, 'Headtorch', tags: 'caves,camping')],
          ),
        ),
      );
      await tester.pump();
      expect(find.text('caves · camping'), findsOneWidget);
    });

    testWidgets('the readiness marker never says "readiness"', (tester) async {
      // That tag is generator bookkeeping, not something to show a person.
      await tester.pumpWidget(
        screen(
          ChecklistView(
            blocking: [item(1, 'Call it', blocking: true, tags: 'readiness')],
            pack: const [],
          ),
        ),
      );
      await tester.pump();
      expect(find.text('readiness'), findsNothing);
    });

    testWidgets('a done item is struck through', (tester) async {
      await tester.pumpWidget(
        screen(
          ChecklistView(
            blocking: const [],
            pack: [item(2, 'Toothbrush', done: true)],
          ),
        ),
      );
      await tester.pump();

      final text = tester.widget<Text>(find.text('Toothbrush'));
      expect(text.style?.decoration, TextDecoration.lineThrough);
    });

    testWidgets('tapping an item toggles it', (tester) async {
      int? toggled;
      bool? to;
      await tester.pumpWidget(
        wrap(
          ChecklistScreen(
            checklist: Stream.value(
              ChecklistView(blocking: const [], pack: [item(2, 'Headtorch')]),
            ),
            onToggle: (i, v) async {
              toggled = i.id;
              to = v;
            },
            onRemove: (_) async {},
            onEdit: (_, _, _) async {},
            onAdd: (_) async {},
          ),
        ),
      );
      await tester.pump();
      await tester.tap(find.text('Headtorch'));
      await tester.pump();

      expect(toggled, 2);
      expect(to, isTrue);
    });

    testWidgets('a quantity renders beside the item', (tester) async {
      await tester.pumpWidget(
        screen(
          ChecklistView(
            blocking: const [],
            pack: [item(2, 'Trekking socks', quantity: '3')],
          ),
        ),
      );
      await tester.pump();
      expect(find.text('3'), findsOneWidget);
    });

    testWidgets('an edited item is marked as the user\'s', (tester) async {
      await tester.pumpWidget(
        screen(
          ChecklistView(
            blocking: const [],
            pack: [item(2, 'My own torch', edited: true)],
          ),
        ),
      );
      await tester.pump();
      expect(find.text('YOURS'), findsOneWidget);
    });

    testWidgets('the promise that edits survive is on the screen', (
      tester,
    ) async {
      await tester.pumpWidget(
        screen(
          ChecklistView(blocking: const [], pack: [item(2, 'Headtorch')]),
        ),
      );
      await tester.pump();
      expect(find.textContaining('stays changed'), findsOneWidget);
      expect(find.textContaining('No weather is looked up'), findsOneWidget);
    });

    testWidgets('an empty pack list explains how to fill it', (tester) async {
      await tester.pumpWidget(
        screen(const ChecklistView(blocking: [], pack: [])),
      );
      await tester.pump();
      expect(find.textContaining('Tag your stops'), findsOneWidget);
    });
  });

  group('expense form', () {
    final travellers = [
      person(1, 'Yash'),
      person(2, 'Priya'),
      person(3, 'Ankit'),
    ];

    Widget form({
      ExpenseDraft? existing,
      Future<void> Function(ExpenseDraft)? onSave,
    }) => wrap(
      ExpenseFormScreen(
        travellers: travellers,
        existing: existing,
        onSave: onSave ?? (_) async {},
      ),
    );

    testWidgets('will not save without a description and an amount', (
      tester,
    ) async {
      var saved = false;
      await tester.pumpWidget(form(onSave: (_) async => saved = true));

      expect(find.text('Not ready yet'), findsOneWidget);
      await tester.tap(find.text('Not ready yet'));
      await tester.pump();
      expect(saved, isFalse);
    });

    testWidgets('an even split is offered and adds up exactly', (
      tester,
    ) async {
      ExpenseDraft? saved;
      await tester.pumpWidget(form(onSave: (d) async => saved = d));

      await tester.enterText(find.byType(TextField).first, 'Taxi');
      await tester.enterText(find.byType(TextField).at(1), '100');
      await tester.pump();

      expect(find.text('Shares add up exactly.'), findsOneWidget);
      await tester.tap(find.text('Add expense'));
      await tester.pump();

      expect(saved?.amountMinor, 10000);
      expect(saved?.shares.values.fold(0, (int a, int b) => a + b), 10000);
      expect(saved?.balances, isTrue);
    });

    testWidgets('deselecting someone re-splits the rest', (tester) async {
      ExpenseDraft? saved;
      await tester.pumpWidget(form(onSave: (d) async => saved = d));

      await tester.enterText(find.byType(TextField).first, 'Taxi');
      await tester.enterText(find.byType(TextField).at(1), '100');
      await tester.pump();

      await tester.tap(find.text('Ankit'));
      await tester.pump();
      await tester.tap(find.text('Add expense'));
      await tester.pump();

      expect(saved?.shares.length, 2);
      expect(saved?.shares.values.fold(0, (int a, int b) => a + b), 10000);
    });

    testWidgets('AN UNBALANCED HAND-SET SPLIT CANNOT BE SAVED', (
      tester,
    ) async {
      await tester.pumpWidget(form());

      await tester.enterText(find.byType(TextField).first, 'Taxi');
      await tester.enterText(find.byType(TextField).at(1), '100');
      await tester.pump();

      await tester.tap(find.text('SET BY HAND'));
      await tester.pump();

      // Three share fields now follow description and amount.
      await tester.enterText(find.byType(TextField).at(2), '10');
      await tester.pump();

      expect(find.textContaining('still to allocate'), findsOneWidget);
      expect(find.text('Not ready yet'), findsOneWidget);
    });

    testWidgets('nobody sharing is refused and says so', (tester) async {
      await tester.pumpWidget(form());
      await tester.enterText(find.byType(TextField).first, 'Taxi');
      await tester.enterText(find.byType(TextField).at(1), '100');
      await tester.pump();

      for (final name in ['Yash', 'Priya', 'Ankit']) {
        await tester.tap(find.text(name).last);
        await tester.pump();
      }
      expect(find.text('Nobody is sharing this yet.'), findsOneWidget);
      expect(find.text('Not ready yet'), findsOneWidget);
    });

    testWidgets('an existing uneven split opens in hand-set mode', (
      tester,
    ) async {
      // Reverting to even on open would silently rewrite a deliberate
      // arrangement.
      await tester.pumpWidget(
        form(
          existing: ExpenseDraft(
            id: 1,
            description: 'Room',
            amountMinor: 30000,
            paidById: 1,
            shares: const {1: 20000, 2: 10000},
            spentAt: DateTime(2026, 10, 3),
          ),
        ),
      );
      await tester.pump();
      expect(find.text('BACK TO EVEN'), findsOneWidget);
    });

    testWidgets('lays out at phone width in both themes', (tester) async {
      tester.view.physicalSize = const Size(400, 800);
      tester.view.devicePixelRatio = 1.0;
      addTearDown(tester.view.reset);

      for (final b in Brightness.values) {
        await tester.pumpWidget(
          wrap(
            ExpenseFormScreen(travellers: travellers, onSave: (_) async {}),
            brightness: b,
          ),
        );
        expect(tester.takeException(), isNull);
      }
    });
  });

  group('travellers', () {
    testWidgets('says there are no accounts and nothing is sent', (
      tester,
    ) async {
      await tester.pumpWidget(
        wrap(
          TravellersScreen(
            travellers: Stream.value([person(1, 'Yash')]),
            onAdd: (_) async {},
            onRename: (_, _) async {},
            onDelete: (_) async {},
          ),
        ),
      );
      await tester.pump();

      expect(find.text('Yash'), findsOneWidget);
      expect(find.textContaining('nothing syncs'), findsOneWidget);
    });

    testWidgets('a refused delete explains why, and keeps the person', (
      tester,
    ) async {
      await tester.pumpWidget(
        wrap(
          TravellersScreen(
            travellers: Stream.value([person(1, 'Priya')]),
            onAdd: (_) async {},
            onRename: (_, _) async {},
            onDelete: (_) async => throw const TravellerInUse('Priya', 3),
          ),
        ),
      );
      await tester.pump();

      await tester.tap(find.byIcon(Icons.close));
      await tester.pump();

      expect(find.textContaining('Priya is in 3 expenses'), findsOneWidget);
      expect(find.text('Priya'), findsOneWidget);
    });
  });
}
