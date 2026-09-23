// test/trip_health_banner_test.dart — the warning, as it reads.

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:safarsathi/core/database/app_database.dart';
import 'package:safarsathi/core/theme/app_tokens.dart';
import 'package:safarsathi/features/trips/data/trip_health.dart';
import 'package:safarsathi/features/trips/data/trip_summary.dart';
import 'package:safarsathi/features/trips/presentation/trip_screen.dart';

Contact contact(int id, String name, String phone,
        {bool confirmed = false, bool emergency = false}) =>
    Contact(
      id: id, name: name, phoneRaw: phone, category: 'other',
      tier: 'userEntered', callConfirmed: confirmed, isPinned: false,
      isEmergency: emergency, hasWhatsapp: false, callCount: 0,
      createdAt: DateTime(2026, 9, 1),
    );

void main() {
  late int samplesRemoved;
  late int copiesRemoved;

  Future<void> pump(WidgetTester tester, TripHealth health) async {
    samplesRemoved = 0;
    copiesRemoved = 0;
    tester.view.physicalSize = const Size(420, 1400);
    tester.view.devicePixelRatio = 1.0;
    addTearDown(tester.view.reset);
    await tester.pumpWidget(MaterialApp(
      theme: AppTokens.light,
      home: TripScreen(
        trip: Stream.value(const TripSummary(name: 'Meghalaya', stops: [])),
        unconfirmedCount: Stream.value(0),
        health: Stream.value(health),
        onRemoveSamples: () async => samplesRemoved++,
        onRemoveDuplicates: () async => copiesRemoved++,
      ),
    ));
    await tester.pump();
    await tester.pump();
  }

  testWidgets('THE WARNING SAYS WHY IT MATTERS: CONFIRMED, AND ON THE SOS TAB',
      (tester) async {
    await pump(tester, TripHealth(sampleContacts: [
      contact(1, 'Bah Rothell', '+91 90000 00007',
          confirmed: true, emergency: true),
      contact(2, 'Kongthong homestay', '+91 90000 00001', confirmed: true),
      contact(3, 'Village guide', '+91 90000 00004'),
    ]));
    expect(
      find.text('3 sample entries from an older version\'s demo are still in '
          'this trip. The '
          'numbers are made up — and 2 are marked confirmed and 1 is on the '
          'SOS tab.'),
      findsOneWidget,
    );
  });

  testWidgets('removing asks first, then removes', (tester) async {
    await pump(tester, TripHealth(sampleContacts: [
      contact(1, 'Village guide', '+91 90000 00004'),
    ]));
    await tester.tap(find.text('Remove them'));
    await tester.pumpAndSettle();
    expect(samplesRemoved, 0, reason: 'nothing deleted before confirming');
    expect(find.textContaining('Only the demo\'s own'), findsOneWidget);
    await tester.tap(find.text('Remove'));
    await tester.pumpAndSettle();
    expect(samplesRemoved, 1);
  });

  testWidgets('"not now" deletes nothing', (tester) async {
    await pump(tester, TripHealth(sampleContacts: [
      contact(1, 'Village guide', '+91 90000 00004'),
    ]));
    await tester.tap(find.text('Remove them'));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Not now'));
    await tester.pumpAndSettle();
    expect(samplesRemoved, 0);
  });

  testWidgets('duplicates are counted and removed on confirm', (tester) async {
    await pump(tester, TripHealth(duplicateExtras: [
      contact(4, 'Alpha', '+918974804455'),
      contact(5, 'Alpha', '+918974804455'),
    ]));
    expect(find.textContaining('2 entries appear more than once'),
        findsOneWidget);
    await tester.tap(find.text('Keep one of each'));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Remove copies'));
    await tester.pumpAndSettle();
    expect(copiesRemoved, 1);
  });

  testWidgets('a healthy trip shows no banner at all', (tester) async {
    await pump(tester, const TripHealth());
    expect(find.textContaining('sample'), findsNothing);
    expect(find.textContaining('more than once'), findsNothing);
  });
}
