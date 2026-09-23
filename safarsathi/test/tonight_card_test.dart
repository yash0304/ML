// test/tonight_card_test.dart — the card as a person sees it.

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:safarsathi/core/database/app_database.dart';
import 'package:safarsathi/core/theme/app_tokens.dart';
import 'package:safarsathi/core/util/sun.dart';
import 'package:safarsathi/features/contacts/data/entry_draft.dart';
import 'package:safarsathi/features/contacts/presentation/entry_form_screen.dart';
import 'package:safarsathi/features/trips/data/tonight.dart';
import 'package:safarsathi/features/trips/data/trip_summary.dart';
import 'package:safarsathi/features/trips/presentation/trip_screen.dart';

Stop stop(String name) => Stop(
  id: 5,
  tripId: 1,
  name: name,
  sequenceOrder: 3,
  countryCode: 'IN',
  nights: 1,
  activityTags: '',
  lat: 25.2027,
  lon: 91.9146,
  arrivalDate: DateTime(2026, 10, 4),
);

Contact stay(int id, String name, {bool confirmed = false, String? note}) =>
    Contact(
      id: id,
      stopId: 5,
      name: name,
      phoneRaw: '+91 90000 0000$id',
      category: 'accommodation',
      tier: 'userEntered',
      callConfirmed: confirmed,
      isPinned: false,
      isEmergency: false,
      hasWhatsapp: false,
      callCount: 0,
      createdAt: DateTime(2026, 9, 22),
      note: note,
    );

void main() {
  setUpAll(() => clockOffsetOverride = const Duration(hours: 5, minutes: 30));
  tearDownAll(() => clockOffsetOverride = null);

  Future<void> pump(
    WidgetTester tester,
    Tonight? t, {
    void Function(Contact)? onOpen,
    void Function(int)? onAdd,
  }) async {
    tester.view.physicalSize = const Size(420, 1400);
    tester.view.devicePixelRatio = 1.0;
    addTearDown(tester.view.reset);
    await tester.pumpWidget(
      MaterialApp(
        theme: AppTokens.light,
        home: TripScreen(
          trip: Stream.value(const TripSummary(name: 'Meghalaya', stops: [])),
          unconfirmedCount: Stream.value(0),
          tonight: Stream.value(t),
          onOpenContact: onOpen,
          onAddStay: onAdd,
        ),
      ),
    );
    await tester.pump();
    await tester.pump();
  }

  Tonight tonight({List<Contact> stays = const [],
      TonightKind kind = TonightKind.tonight}) => Tonight(
    kind: kind,
    stop: stop('Mawlynnong'),
    night: DateTime(2026, 10, 4),
    stays: stays,
    sun: sunTimes(25.2027, 91.9146, DateTime(2026, 10, 4)),
  );

  testWidgets('tonight names the stop, the stay, the number and the note',
      (tester) async {
    await pump(tester, tonight(stays: [
      stay(1, 'Iartong Guest House', confirmed: true,
          note: 'Owner Bah Iar. Pay cash.'),
    ]));
    expect(find.text('TONIGHT · MAWLYNNONG'), findsOneWidget);
    expect(find.text('Iartong Guest House'), findsOneWidget);
    expect(find.text('+91 90000 00001'), findsOneWidget);
    expect(find.text('Owner Bah Iar. Pay cash.'), findsOneWidget);
    expect(find.textContaining('Sunset at Mawlynnong 17:0'), findsOneWidget);
  });

  testWidgets('before the trip it says first night, with the date', (
    tester,
  ) async {
    await pump(tester, tonight(kind: TonightKind.firstNight));
    expect(find.text('FIRST NIGHT · MAWLYNNONG, 4 OCT'), findsOneWidget);
  });

  testWidgets('no stay saved says so and offers to add one at that stop',
      (tester) async {
    int? addedAt;
    await pump(tester, tonight(), onAdd: (id) => addedAt = id);
    expect(find.text('No stay for Mawlynnong in your diary yet. Add one.'),
        findsOneWidget);
    await tester.tap(find.textContaining('No stay for Mawlynnong'));
    expect(addedAt, 5);
  });

  testWidgets('tapping a stay opens it', (tester) async {
    Contact? opened;
    await pump(tester, tonight(stays: [stay(3, 'Nangroi Homestay')]),
        onOpen: (c) => opened = c);
    await tester.tap(find.text('Nangroi Homestay'));
    expect(opened?.id, 3);
  });

  testWidgets('at most two stays on the first screen', (tester) async {
    await pump(tester, tonight(stays: [
      for (var i = 1; i <= 4; i++) stay(i, 'Stay $i'),
    ]));
    expect(find.text('Stay 2'), findsOneWidget);
    expect(find.text('Stay 3'), findsNothing);
  });

  testWidgets('no night on the plan means no card at all', (tester) async {
    await pump(tester, null);
    expect(find.textContaining('TONIGHT'), findsNothing);
    expect(find.textContaining('FIRST NIGHT'), findsNothing);
  });

  testWidgets('"add one" opens the form already at that stop, as a stay',
      (tester) async {
    final saved = <EntryDraft>[];
    tester.view.physicalSize = const Size(420, 1400);
    tester.view.devicePixelRatio = 1.0;
    addTearDown(tester.view.reset);
    await tester.pumpWidget(MaterialApp(
      theme: AppTokens.light,
      home: EntryFormScreen(
        stops: const [StopOption(5, 'Mawlynnong')],
        initialStopId: 5,
        initialCategory: 'accommodation',
        onSave: (d) async => saved.add(d),
      ),
    ));
    await tester.enterText(find.descendant(
        of: find.byKey(const Key('field-name')),
        matching: find.byType(TextField)), 'Iartong Guest House');
    await tester.enterText(find.descendant(
        of: find.byKey(const Key('field-number')),
        matching: find.byType(TextField)), '+91 98630 12345');
    await tester.tap(find.text('Save to diary'));
    await tester.pumpAndSettle();
    expect(saved.single.stopId, 5);
    expect(saved.single.category, 'accommodation');
  });
}
