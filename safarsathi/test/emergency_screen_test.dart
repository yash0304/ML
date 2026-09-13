// Issue #50 — the emergency screen.
//
// THE EXEMPT SCREEN. Half of what these tests check is the absence of things:
// no grain, no stamps, no ticket edges, no milestones. Anything a user could
// mistake for a verification mark would defeat the tier system, and the whole
// reason this screen exists is that the tiering can be trusted.

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:safarsathi/core/database/app_database.dart';
import 'package:safarsathi/core/theme/app_tokens.dart';
import 'package:safarsathi/core/widgets/retro.dart';
import 'package:safarsathi/features/contacts/data/contacts_dao.dart';
import 'package:safarsathi/features/emergency/presentation/emergency_screen.dart';

EmergencyHelpline line(
  String number,
  String label,
  String source, {
  int id = 1,
}) {
  return EmergencyHelpline(
    id: id,
    countryCode: 'IN',
    serviceType: 'all',
    label: label,
    number: number,
    sourceNote: source,
    needsVerification: false,
    tier: 'verifiedNational',
  );
}

Contact localContact({
  String name = 'Bah Rothell',
  bool confirmed = true,
  int id = 1,
}) {
  return Contact(
    id: id,
    name: name,
    phoneRaw: '+91 90000 00007',
    category: ContactCategory.localContact,
    tier: confirmed
        ? ContactTier.userVerified.name
        : ContactTier.userEntered.name,
    callConfirmed: confirmed,
    isPinned: false,
    isEmergency: true,
    hasWhatsapp: false,
    callCount: 0,
    createdAt: DateTime(2026, 9, 12),
  );
}

void main() {
  late List<String> called;
  late List<String> copied;

  Future<void> pumpEmergency(
    WidgetTester tester, {
    List<EmergencyHelpline> helplines = const [],
    List<Contact> locals = const [],
  }) async {
    called = [];
    copied = [];
    tester.view.physicalSize = const Size(420, 1400);
    tester.view.devicePixelRatio = 1.0;
    addTearDown(tester.view.reset);

    await tester.pumpWidget(
      MaterialApp(
        theme: AppTokens.light,
        home: EmergencyScreen(
          helplines: Stream.value(helplines),
          localContacts: Stream.value(locals),
          placeLabel: 'India · Kongthong',
          onCall: (n) async => called.add(n),
          onCopy: (n) async => copied.add(n),
        ),
      ),
    );
    await tester.pump();
    await tester.pump();
  }

  group('official helplines', () {
    testWidgets('render with their number, label and government source', (
      tester,
    ) async {
      await pumpEmergency(
        tester,
        helplines: [
          line(
            '112',
            'All emergencies',
            '112.gov.in, Ministry of Home Affairs',
          ),
        ],
      );
      expect(find.text('112'), findsOneWidget);
      expect(find.text('All emergencies'), findsOneWidget);
      // Provenance under every bundled number, always.
      expect(find.textContaining('112.gov.in'), findsOneWidget);
    });

    testWidgets('tapping the row places the call', (tester) async {
      // Here the tap CALLS. A copy-and-paste dance is a liability in an
      // emergency.
      await pumpEmergency(
        tester,
        helplines: [line('112', 'All emergencies', '112.gov.in')],
      );
      await tester.tap(find.text('All emergencies'));
      await tester.pump();
      expect(called, ['112']);
      expect(copied, isEmpty);
    });

    testWidgets('copy is present but secondary', (tester) async {
      await pumpEmergency(
        tester,
        helplines: [line('112', 'All emergencies', '112.gov.in')],
      );
      await tester.tap(find.bySemanticsLabel('Copy All emergencies'));
      await tester.pump();
      expect(copied, ['112']);
    });
  });

  group('the two sections never mix', () {
    testWidgets('each has its own visible header', (tester) async {
      // A government short code and a number somebody typed must never sit in
      // the same list.
      await pumpEmergency(
        tester,
        helplines: [line('112', 'All emergencies', '112.gov.in')],
        locals: [localContact()],
      );
      expect(find.text('OFFICIAL HELPLINES'), findsOneWidget);
      expect(find.text('YOUR LOCAL CONTACTS'), findsOneWidget);
    });

    testWidgets('a section with nothing in it does not appear', (tester) async {
      await pumpEmergency(
        tester,
        helplines: [line('112', 'All emergencies', '112.gov.in')],
      );
      expect(find.text('YOUR LOCAL CONTACTS'), findsNothing);
    });

    testWidgets('an unconfirmed local contact keeps its trust dot', (
      tester,
    ) async {
      await pumpEmergency(tester, locals: [localContact(confirmed: false)]);
      final dots = tester.widgetList<Container>(find.byType(Container)).where((
        box,
      ) {
        final d = box.decoration;
        return d is BoxDecoration &&
            d.shape == BoxShape.circle &&
            d.color == AppColors.day.cautionMark;
      });
      expect(dots, hasLength(1));
    });
  });

  group('the exemption holds', () {
    testWidgets('no ephemera anywhere on the screen', (tester) async {
      await pumpEmergency(
        tester,
        helplines: [
          line('112', 'All emergencies', '112.gov.in'),
          line('108', 'Ambulance', 'Legacy line', id: 2),
        ],
        locals: [localContact()],
      );

      expect(find.byType(GrainOverlay), findsNothing);
      expect(find.byType(StampBadge), findsNothing);
      expect(find.byType(TicketCard), findsNothing);
      expect(find.byType(MilestoneMarker), findsNothing);
      expect(find.byType(HazardStripe), findsNothing);
    });

    testWidgets('THE DIARY GAINING A STAMP DID NOT LEAK ONE IN HERE', (
      tester,
    ) async {
      // #44 put a confirmation stamp on every diary row. This surface takes
      // confirmed local contacts too, and the whole point of #48 is that a
      // decoration added anywhere else never arrives here by accident.
      await pumpEmergency(
        tester,
        helplines: [line('112', 'All emergencies', '112.gov.in')],
        locals: [localContact(confirmed: true), localContact(id: 9)],
      );

      expect(find.byType(StampBadge), findsNothing);
      expect(find.text('CONFIRMED'), findsNothing);
    });

    testWidgets('NOTHING SWIPES ON THIS SCREEN EITHER', (tester) async {
      // #45 gave diary rows swipe actions. A hidden gesture on the screen
      // somebody reaches for in an emergency is the wrong kind of clever.
      await pumpEmergency(
        tester,
        helplines: [line('112', 'All emergencies', '112.gov.in')],
        locals: [localContact()],
      );

      expect(find.byType(Dismissible), findsNothing);
    });

    testWidgets('the missing state list is explained, not silently absent', (
      tester,
    ) async {
      await pumpEmergency(
        tester,
        helplines: [line('112', 'All emergencies', '112.gov.in')],
      );
      expect(
        find.textContaining('28 states and 8 union territories'),
        findsOneWidget,
      );
      expect(find.textContaining('worse than none'), findsOneWidget);
    });
  });
}
