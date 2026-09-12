// Issue #9 — the entry screen and the confirmation.
//
// Promoting a number from userEntered to userVerified is the single action
// the whole trust system depends on: the amber dot, the readiness count and
// the blocking checklist items all read the boolean it sets. These tests
// cover what the user sees and what the app promises.

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:safarsathi/core/database/app_database.dart';
import 'package:safarsathi/core/theme/app_tokens.dart';
import 'package:safarsathi/core/widgets/retro.dart';
import 'package:safarsathi/features/contacts/data/contacts_dao.dart';
import 'package:safarsathi/features/contacts/presentation/entry_screen.dart';

Contact entry({
  String name = 'Kongthong homestay',
  String phoneRaw = '+91 90000 00001',
  ContactTier tier = ContactTier.userEntered,
  bool confirmed = false,
  DateTime? confirmedAt,
  String? note,
  int? importBatchId,
  DateTime? lastCalledAt,
  int callCount = 0,
}) {
  return Contact(
    id: 1,
    name: name,
    phoneRaw: phoneRaw,
    category: ContactCategory.accommodation,
    tier: tier.name,
    callConfirmed: confirmed,
    confirmedAt: confirmedAt,
    isPinned: false,
    isEmergency: false,
    hasWhatsapp: false,
    callCount: callCount,
    lastCalledAt: lastCalledAt,
    note: note,
    importBatchId: importBatchId,
    createdAt: DateTime(2026, 9, 12),
  );
}

void main() {
  late List<bool> confirmations;

  Future<void> pumpEntry(
    WidgetTester tester,
    Contact contact, {
    String? stopName,
    Future<String> Function(Contact)? onCopy,
    Future<void> Function(Contact)? onCall,
    bool withConfirm = true,
  }) async {
    confirmations = [];
    tester.view.physicalSize = const Size(420, 1400);
    tester.view.devicePixelRatio = 1.0;
    addTearDown(tester.view.reset);

    await tester.pumpWidget(
      MaterialApp(
        theme: AppTokens.light,
        home: EntryScreen(
          contact: contact,
          stopName: stopName,
          onCopy: onCopy ?? (c) async => c.phoneRaw,
          onOpenDialer: () async {},
          onCall: onCall,
          onConfirm: withConfirm
              ? (c, {required confirmed}) async => confirmations.add(confirmed)
              : null,
          onEdit: (_) {},
        ),
      ),
    );
    await tester.pump();
  }

  /// The stamp is drawn at 0.85 opacity once landed, and not rendered at all
  /// before the first transition.
  bool stampIsShowing(WidgetTester tester) {
    final opacities = tester
        .widgetList<Opacity>(
          find.descendant(
            of: find.byType(StampBadge),
            matching: find.byType(Opacity),
          ),
        )
        .toList();
    return opacities.any((o) => o.opacity > 0.1);
  }

  group('the number and its provenance', () {
    testWidgets('the number is the headline and is selectable', (tester) async {
      await pumpEntry(tester, entry());
      expect(find.byType(SelectableText), findsOneWidget);

      final text = tester.widget<SelectableText>(find.byType(SelectableText));
      expect(text.data, '+91 90000 00001');
      expect(text.style!.fontSize, greaterThan(20));
      expect(text.style!.fontFamily, 'CourierPrime');
    });

    testWidgets('a typed number says it is not confirmed', (tester) async {
      await pumpEntry(tester, entry());
      // "Typed by you" also appears as the Source row, so match the whole
      // provenance line.
      expect(find.text('Typed by you · not confirmed'), findsOneWidget);
    });

    testWidgets('a number from map data says nobody has checked it', (
      tester,
    ) async {
      await pumpEntry(tester, entry(tier: ContactTier.communityOsm));
      expect(find.textContaining('open map data'), findsOneWidget);
      expect(find.textContaining('nobody has checked it'), findsOneWidget);
    });

    testWidgets('a government short code says it is authoritative', (
      tester,
    ) async {
      await pumpEntry(
        tester,
        entry(
          name: 'All emergencies',
          phoneRaw: '112',
          tier: ContactTier.verifiedNational,
        ),
      );
      expect(find.textContaining('Authoritative'), findsOneWidget);
      // Nothing for the user to assert about a government short code: it is
      // authoritative by provenance, not by anyone having dialled it.
      expect(find.text('Mark confirmed'), findsNothing);
    });

    testWidgets('a confirmed number says when, and by whom', (tester) async {
      await pumpEntry(
        tester,
        entry(
          tier: ContactTier.userVerified,
          confirmed: true,
          confirmedAt: DateTime(2026, 9, 11),
        ),
      );
      expect(find.textContaining('Confirmed by you, 11 Sep'), findsOneWidget);
      expect(find.textContaining('you reached this number'), findsOneWidget);
    });
  });

  group('confirming', () {
    testWidgets('says plainly that the user has to have called it', (
      tester,
    ) async {
      await pumpEntry(tester, entry());
      expect(
        find.textContaining('Only after you have actually called it'),
        findsOneWidget,
      );
      expect(find.textContaining('cannot tell whether a call'), findsOneWidget);
    });

    testWidgets('the stamp is absent until the entry is confirmed', (
      tester,
    ) async {
      await pumpEntry(tester, entry());
      expect(stampIsShowing(tester), isFalse);
    });

    testWidgets('confirming lands the stamp and turns the line green', (
      tester,
    ) async {
      await pumpEntry(tester, entry());

      await tester.tap(find.text('Mark confirmed'));
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 400));

      expect(confirmations, [true]);
      expect(stampIsShowing(tester), isTrue);
      expect(find.textContaining('you reached this number'), findsOneWidget);
      expect(find.textContaining('not confirmed'), findsNothing);
    });

    testWidgets('a confirmed entry renders its stamp without animating', (
      tester,
    ) async {
      // Scrolling past a confirmed entry has to be silent and still.
      await pumpEntry(
        tester,
        entry(tier: ContactTier.userVerified, confirmed: true),
      );
      expect(stampIsShowing(tester), isTrue);
    });

    testWidgets('the confirmation can be cleared again', (tester) async {
      // A number that worked in October may not work in November.
      await pumpEntry(
        tester,
        entry(tier: ContactTier.userVerified, confirmed: true),
      );
      expect(find.text('Clear confirmation'), findsOneWidget);

      await tester.tap(find.text('Clear confirmation'));
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 400));

      expect(confirmations, [false]);
      expect(find.text('Typed by you · not confirmed'), findsOneWidget);
    });

    testWidgets('there is no confirm button when the caller offers none', (
      tester,
    ) async {
      await pumpEntry(tester, entry(), withConfirm: false);
      await tester.tap(find.text('Mark confirmed'));
      await tester.pump();
      expect(confirmations, isEmpty);
    });
  });

  group('the record', () {
    testWidgets('shows where the entry is attached', (tester) async {
      await pumpEntry(tester, entry(), stopName: 'Kongthong');
      expect(find.text('Kongthong'), findsOneWidget);
    });

    testWidgets('says whole trip when it is attached to none', (tester) async {
      await pumpEntry(tester, entry());
      expect(find.text('Whole trip'), findsOneWidget);
    });

    testWidgets('an imported entry says where it came from', (tester) async {
      await pumpEntry(tester, entry(importBatchId: 3));
      expect(find.text('Imported from a sheet'), findsOneWidget);
    });

    testWidgets('the last action is labelled as an action, not a call', (
      tester,
    ) async {
      // A copy counts, because the dial happens in the Android dialer.
      await pumpEntry(tester, entry());
      expect(find.text('LAST ACTION'), findsOneWidget);
      expect(find.text('Never used'), findsOneWidget);
    });

    testWidgets('a used entry shows when and how often', (tester) async {
      await pumpEntry(
        tester,
        entry(lastCalledAt: DateTime(2026, 10, 3), callCount: 4),
      );
      expect(find.textContaining('3 Oct'), findsOneWidget);
      expect(find.textContaining('4 in total'), findsOneWidget);
    });
  });

  group('actions', () {
    testWidgets('copying shows the number in a toast', (tester) async {
      await pumpEntry(tester, entry());
      await tester.tap(find.text('Copy number'));
      await tester.pump();
      await tester.pump();

      expect(find.text('Copied'), findsOneWidget);
      // Once as the headline, once in the toast.
      expect(find.text('+91 90000 00001'), findsNWidgets(2));
    });

    testWidgets('a failed action says what went wrong', (tester) async {
      await pumpEntry(
        tester,
        entry(),
        onCall: (_) async => throw Exception('No dialer on this phone.'),
      );
      await tester.tap(find.text('Call'));
      await tester.pump();
      await tester.pump();

      expect(find.text('Could not open that'), findsOneWidget);
      expect(find.textContaining('No dialer'), findsOneWidget);
    });
  });
}
