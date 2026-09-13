// test/diary_interactions_test.dart — issues #44, #45 and #46.
//
// Three interactions, and a static golden cannot see any of them. The one
// that matters most is negative: scrolling a list of already-confirmed
// contacts must fire no haptic and land no stamp.

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:safarsathi/core/database/app_database.dart';
import 'package:safarsathi/core/theme/app_tokens.dart';
import 'package:safarsathi/core/widgets/retro.dart';
import 'package:safarsathi/features/contacts/data/contacts_dao.dart';
import 'package:safarsathi/features/contacts/presentation/diary_screen.dart';
import 'package:safarsathi/features/contacts/presentation/diary_widgets.dart';

Contact row({
  required int id,
  String? name,
  bool confirmed = false,
  bool pinned = false,
}) => Contact(
  id: id,
  name: name ?? 'Contact $id',
  phoneRaw: '+91 90000 0000$id',
  category: ContactCategory.accommodation,
  tier: confirmed ? ContactTier.userVerified.name : ContactTier.userEntered.name,
  callConfirmed: confirmed,
  isPinned: pinned,
  isEmergency: false,
  hasWhatsapp: false,
  callCount: 0,
  createdAt: DateTime(2026, 9, 12),
);

Widget wrap(Widget child) => MaterialApp(theme: AppTokens.light, home: child);

/// The entries list. The category thumb index is a ListView too, so a bare
/// `find.byType(ListView)` is ambiguous on this screen.
final entriesList = find.byType(ListView).first;

/// Mirrors `_CacheStamp.findKey`, which is private to the screen.
const stampKey = ValueKey('diary-cache-stamp');

/// Counts every haptic the framework is asked for, so "nothing buzzed" is
/// something a test can actually assert rather than hope for.
class HapticSpy {
  final calls = <String>[];

  void install(WidgetTester tester) {
    tester.binding.defaultBinaryMessenger.setMockMethodCallHandler(
      SystemChannels.platform,
      (call) async {
        if (call.method == 'HapticFeedback.vibrate') {
          calls.add('${call.arguments}');
        }
        return null;
      },
    );
    addTearDown(
      () => tester.binding.defaultBinaryMessenger.setMockMethodCallHandler(
        SystemChannels.platform,
        null,
      ),
    );
  }
}

Widget diary({
  required List<Contact> contacts,
  Future<void> Function(Contact, bool)? onTogglePin,
  Future<String> Function(Contact)? onCopy,
  Stream<String>? cacheStamp,
}) => wrap(
  DiaryScreen(
    watchContacts: (_) => Stream.value(contacts),
    unconfirmedCount: Stream.value(
      contacts.where((c) => !c.callConfirmed).length,
    ),
    tripId: 1,
    tripName: 'Meghalaya',
    onCopy: onCopy ?? (c) async => c.phoneRaw,
    onTogglePin: onTogglePin,
    cacheStamp: cacheStamp,
  ),
);

void main() {
  group('#44 — the stamp lands once', () {
    testWidgets('a confirmed row carries the stamp and no amber dot', (
      tester,
    ) async {
      await tester.pumpWidget(
        wrap(
          Scaffold(
            body: DiaryEntry(contact: row(id: 1, confirmed: true), lineNumber: 1),
          ),
        ),
      );
      await tester.pump();

      expect(find.byType(StampBadge), findsOneWidget);
      expect(find.text('CONFIRMED'), findsOneWidget);
      // The dot is gone from the tree, not merely invisible: its semantics
      // would otherwise still announce "Not confirmed yet".
      expect(find.bySemanticsLabel('Not confirmed yet'), findsNothing);
    });

    testWidgets('an unconfirmed row carries the dot and no stamp', (
      tester,
    ) async {
      await tester.pumpWidget(
        wrap(Scaffold(body: DiaryEntry(contact: row(id: 1), lineNumber: 1))),
      );
      await tester.pump();

      // Mounted but collapsed to nothing — so it can see the transition
      // later without announcing "Confirmed" in the meantime.
      expect(find.byType(StampBadge), findsOneWidget);
      expect(find.text('CONFIRMED'), findsNothing);
      expect(find.bySemanticsLabel('Confirmed'), findsNothing);
      expect(find.bySemanticsLabel('Not confirmed yet'), findsOneWidget);
    });

    testWidgets('BUILDING A CONFIRMED ROW FIRES NO HAPTIC', (tester) async {
      final spy = HapticSpy()..install(tester);

      await tester.pumpWidget(
        wrap(
          Scaffold(
            body: DiaryEntry(contact: row(id: 1, confirmed: true), lineNumber: 1),
          ),
        ),
      );
      await tester.pumpAndSettle();

      // Already stamped when it arrived. Nothing happened, so nothing buzzes.
      expect(spy.calls, isEmpty);
    });

    testWidgets('THE FALSE TO TRUE TRANSITION FIRES EXACTLY ONE HAPTIC', (
      tester,
    ) async {
      final spy = HapticSpy()..install(tester);
      final contact = row(id: 1);

      await tester.pumpWidget(
        wrap(Scaffold(body: DiaryEntry(contact: contact, lineNumber: 1))),
      );
      await tester.pump();
      expect(spy.calls, isEmpty);

      await tester.pumpWidget(
        wrap(
          Scaffold(
            body: DiaryEntry(contact: row(id: 1, confirmed: true), lineNumber: 1),
          ),
        ),
      );
      await tester.pumpAndSettle();

      expect(spy.calls, hasLength(1));
      expect(spy.calls.single, contains('mediumImpact'));
    });

    testWidgets('SCROLLING PAST CONFIRMED ROWS IS SILENT AND STILL', (
      tester,
    ) async {
      // The bug this guards: ListView recycles elements, so without a key
      // per row a confirmed row's element gets handed an unconfirmed contact
      // and back again, StampBadge reads that as a confirmation, and the
      // phone buzzes at somebody who is only scrolling.
      final spy = HapticSpy()..install(tester);
      final contacts = [
        for (var i = 1; i <= 30; i++) row(id: i, confirmed: i.isEven),
      ];

      await tester.pumpWidget(diary(contacts: contacts));
      await tester.pump();
      spy.calls.clear();

      await tester.fling(entriesList, const Offset(0, -600), 1200);
      await tester.pumpAndSettle();
      await tester.fling(entriesList, const Offset(0, 600), 1200);
      await tester.pumpAndSettle();

      expect(spy.calls, isEmpty);
    });

    testWidgets('every row is keyed on its contact id', (tester) async {
      await tester.pumpWidget(
        diary(contacts: [row(id: 7), row(id: 9, confirmed: true)]),
      );
      await tester.pump();

      final keys = tester
          .widgetList<DiaryEntry>(find.byType(DiaryEntry))
          .map((e) => e.key)
          .toList();
      expect(keys, [const ValueKey(7), const ValueKey(9)]);
    });
  });

  group('#45 — swipe, springing back', () {
    testWidgets('swiping right calls, and the row stays', (tester) async {
      var copied = 0;
      await tester.pumpWidget(
        diary(
          contacts: [row(id: 1, name: 'Kongthong homestay')],
          onTogglePin: (_, _) async {},
          onCopy: (c) async {
            copied++;
            return c.phoneRaw;
          },
        ),
      );
      await tester.pump();

      await tester.drag(find.text('Kongthong homestay'), const Offset(320, 0));
      await tester.pumpAndSettle();

      expect(copied, 1);
      // NOTHING IS EVER REMOVED BY A GESTURE.
      expect(find.text('Kongthong homestay'), findsOneWidget);
    });

    testWidgets('swiping left pins, and the row stays', (tester) async {
      Contact? pinnedContact;
      bool? pinnedTo;

      await tester.pumpWidget(
        diary(
          contacts: [row(id: 1, name: 'Kongthong homestay')],
          onTogglePin: (c, p) async {
            pinnedContact = c;
            pinnedTo = p;
          },
        ),
      );
      await tester.pump();

      await tester.drag(find.text('Kongthong homestay'), const Offset(-320, 0));
      await tester.pumpAndSettle();

      expect(pinnedContact?.id, 1);
      expect(pinnedTo, isTrue);
      expect(find.text('Kongthong homestay'), findsOneWidget);
    });

    testWidgets('an already-pinned row offers to unpin', (tester) async {
      bool? pinnedTo;
      await tester.pumpWidget(
        diary(
          contacts: [row(id: 1, name: 'Kongthong homestay', pinned: true)],
          onTogglePin: (_, p) async => pinnedTo = p,
        ),
      );
      await tester.pump();

      await tester.drag(find.text('Kongthong homestay'), const Offset(-320, 0));
      await tester.pumpAndSettle();
      expect(pinnedTo, isFalse);
    });

    testWidgets('crossing the threshold buzzes once, not per pixel', (
      tester,
    ) async {
      final spy = HapticSpy()..install(tester);
      await tester.pumpWidget(
        diary(
          contacts: [row(id: 1, name: 'Kongthong homestay')],
          onTogglePin: (_, _) async {},
        ),
      );
      await tester.pump();
      spy.calls.clear();

      final gesture = await tester.startGesture(
        tester.getCenter(find.text('Kongthong homestay')),
      );
      for (var i = 0; i < 20; i++) {
        await gesture.moveBy(const Offset(16, 0));
        await tester.pump();
      }
      // Well past the threshold after twenty small moves, and still one buzz.
      expect(spy.calls, hasLength(1));

      await gesture.up();
      await tester.pumpAndSettle();
    });

    testWidgets('WITHOUT HANDLERS THERE IS NO SWIPE AT ALL', (tester) async {
      await tester.pumpWidget(
        wrap(Scaffold(body: DiaryEntry(contact: row(id: 1), lineNumber: 1))),
      );
      await tester.pump();
      expect(find.byType(Dismissible), findsNothing);
    });
  });

  group('#46 — over-scroll shows what is cached', () {
    testWidgets('it takes no height until it is dragged for', (tester) async {
      await tester.pumpWidget(
        diary(
          contacts: [row(id: 1)],
          cacheStamp: Stream.value('Cached 26/09/2026 · 87 places'),
        ),
      );
      await tester.pump();

      // Present in the tree but collapsed, so nothing shifts on an ordinary
      // scroll.
      expect(tester.getSize(find.byKey(stampKey)).height, 0);
    });

    testWidgets('dragging past the top reveals the cache sentence', (
      tester,
    ) async {
      await tester.pumpWidget(
        diary(
          contacts: [row(id: 1)],
          cacheStamp: Stream.value('Cached 26/09/2026 · 87 places'),
        ),
      );
      await tester.pump();

      final gesture = await tester.startGesture(
        tester.getCenter(entriesList),
      );
      await gesture.moveBy(const Offset(0, 160));
      await tester.pump();

      expect(tester.getSize(find.byKey(stampKey)).height, greaterThan(0));
      expect(find.text('Cached 26/09/2026 · 87 places'), findsOneWidget);

      await gesture.up();
      await tester.pumpAndSettle();
    });

    testWidgets('NO PULL TO REFRESH, EVER', (tester) async {
      // A spinner would promise the one thing this app is built never to do.
      await tester.pumpWidget(
        diary(contacts: [row(id: 1)], cacheStamp: Stream.value('Cached')),
      );
      await tester.pump();
      expect(find.byType(RefreshIndicator), findsNothing);
    });

    testWidgets('a screen given no cache stream shows no stamp', (
      tester,
    ) async {
      await tester.pumpWidget(diary(contacts: [row(id: 1)]));
      await tester.pump();
      expect(find.byKey(stampKey), findsNothing);
    });
  });
}
