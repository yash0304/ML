// Issue #6 acceptance, at the widget level.
//
// These build the diary's parts from plain data rather than from a database.
// Ordering and filtering are already covered by the DAO tests; what matters
// here is what the user actually sees — above all, that a verified number and
// an unverified one never look the same.

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:safarsathi/core/database/app_database.dart';
import 'package:safarsathi/core/theme/app_tokens.dart';
import 'package:safarsathi/features/contacts/data/contacts_dao.dart';
import 'package:safarsathi/features/contacts/presentation/diary_widgets.dart';

Contact contact({
  int id = 1,
  String name = 'Kongthong homestay',
  String phoneRaw = '+91 90000 00001',
  String category = ContactCategory.accommodation,
  ContactTier tier = ContactTier.userEntered,
  bool confirmed = false,
  bool pinned = false,
  String? note,
}) {
  return Contact(
    id: id,
    name: name,
    phoneRaw: phoneRaw,
    category: category,
    tier: tier.name,
    callConfirmed: confirmed,
    isPinned: pinned,
    isEmergency: false,
    hasWhatsapp: false,
    callCount: 0,
    note: note,
    createdAt: DateTime(2026, 9, 12),
  );
}

Future<void> pump(
  WidgetTester tester,
  Widget child, {
  Brightness brightness = Brightness.light,
}) async {
  await tester.pumpWidget(
    MaterialApp(
      theme: brightness == Brightness.dark ? AppTokens.dark : AppTokens.light,
      home: Scaffold(body: child),
    ),
  );
  await tester.pump();
}

/// The trust dot is the only thing in the app painted in cautionMark.
int trustDots(WidgetTester tester, AppColors palette) {
  return tester.widgetList<Container>(find.byType(Container)).where((box) {
    final d = box.decoration;
    return d is BoxDecoration &&
        d.shape == BoxShape.circle &&
        d.color == palette.cautionMark;
  }).length;
}

void main() {
  group('the trust dot', () {
    testWidgets('appears on an unverified entry', (tester) async {
      await pump(tester, DiaryEntry(contact: contact(), lineNumber: 1));
      expect(trustDots(tester, AppColors.day), 1);
    });

    testWidgets('is absent once the user has confirmed the number', (
      tester,
    ) async {
      await pump(
        tester,
        DiaryEntry(
          contact: contact(tier: ContactTier.userVerified, confirmed: true),
          lineNumber: 1,
        ),
      );
      expect(trustDots(tester, AppColors.day), 0);
    });

    testWidgets('appears on a number that came from map data', (tester) async {
      // communityOsm is never trusted, whatever else is true of it.
      await pump(
        tester,
        DiaryEntry(
          contact: contact(tier: ContactTier.communityOsm),
          lineNumber: 1,
        ),
      );
      expect(trustDots(tester, AppColors.day), 1);
    });

    testWidgets('is absent on a government short code', (tester) async {
      await pump(
        tester,
        DiaryEntry(
          contact: contact(
            name: 'All emergencies',
            phoneRaw: '112',
            tier: ContactTier.verifiedNational,
          ),
          lineNumber: 1,
        ),
      );
      expect(trustDots(tester, AppColors.day), 0);
    });

    testWidgets('still reads as amber at night', (tester) async {
      await pump(
        tester,
        DiaryEntry(contact: contact(), lineNumber: 1),
        brightness: Brightness.dark,
      );
      expect(trustDots(tester, AppColors.night), 1);
    });
  });

  group('an entry line', () {
    testWidgets('shows the margin number, the name and the number', (
      tester,
    ) async {
      await pump(
        tester,
        DiaryEntry(contact: contact(note: 'Bah Rothell'), lineNumber: 3),
      );
      expect(find.text('03'), findsOneWidget);
      expect(find.text('Kongthong homestay'), findsOneWidget);
      expect(find.text('+91 90000 00001'), findsOneWidget);
      expect(find.textContaining('Bah Rothell'), findsOneWidget);
    });

    testWidgets('sets the number larger than the meta line', (tester) async {
      // In a diary the number is the content, and it has to survive being
      // read in poor light before being pasted into the dialer.
      await pump(
        tester,
        DiaryEntry(contact: contact(note: 'a note'), lineNumber: 1),
      );
      final number = tester.widget<Text>(find.text('+91 90000 00001'));
      final meta = tester.widget<Text>(find.textContaining('a note'));
      expect(number.style!.fontSize!, greaterThan(meta.style!.fontSize!));
    });

    testWidgets('numbers use tabular figures so they align down the page', (
      tester,
    ) async {
      await pump(tester, DiaryEntry(contact: contact(), lineNumber: 1));
      final number = tester.widget<Text>(find.text('+91 90000 00001'));
      expect(
        number.style!.fontFeatures?.any((f) => f.feature == 'tnum'),
        isTrue,
      );
    });

    testWidgets('a pinned entry shows its pin', (tester) async {
      await pump(
        tester,
        DiaryEntry(contact: contact(pinned: true), lineNumber: 1),
      );
      expect(find.byIcon(Icons.push_pin), findsOneWidget);
    });

    testWidgets('tapping the line asks to copy', (tester) async {
      var copied = 0;
      await pump(
        tester,
        DiaryEntry(contact: contact(), lineNumber: 1, onCopy: () => copied++),
      );
      await tester.tap(find.byType(DiaryEntry));
      await tester.pump();
      expect(copied, 1);
    });
  });

  group('the thumb index', () {
    testWidgets('reports the category it was flipped to', (tester) async {
      String? chosen = 'unset';
      await pump(
        tester,
        CategoryIndex(selected: null, onSelect: (c) => chosen = c),
      );
      await tester.tap(find.text('STAY'));
      await tester.pump();
      expect(chosen, ContactCategory.accommodation);
    });

    testWidgets('ALL clears the category', (tester) async {
      String? chosen = 'unset';
      await pump(
        tester,
        CategoryIndex(
          selected: ContactCategory.accommodation,
          onSelect: (c) => chosen = c,
        ),
      );
      await tester.tap(find.text('ALL'));
      await tester.pump();
      expect(chosen, isNull);
    });
  });

  group('the readiness banner', () {
    testWidgets('stays hidden when nothing is unconfirmed', (tester) async {
      // A permanent banner becomes wallpaper and stops being read.
      await pump(
        tester,
        ReadinessBanner(unconfirmedCount: Stream<int>.value(0)),
      );
      expect(find.textContaining('not confirmed'), findsNothing);
    });

    testWidgets('shows the count when something is unconfirmed', (
      tester,
    ) async {
      await pump(
        tester,
        ReadinessBanner(unconfirmedCount: Stream<int>.value(4)),
      );
      expect(find.text('4'), findsOneWidget);
      expect(find.textContaining('numbers not confirmed'), findsOneWidget);
    });

    testWidgets('reads as singular for one', (tester) async {
      await pump(
        tester,
        ReadinessBanner(unconfirmedCount: Stream<int>.value(1)),
      );
      expect(find.textContaining('number not confirmed'), findsOneWidget);
    });
  });

  group('empty states', () {
    testWidgets('give direction when the diary is empty', (tester) async {
      await pump(tester, const DiaryEmptyState(searching: false));
      expect(find.text('Nothing in the diary yet.'), findsOneWidget);
      expect(find.textContaining('import a sheet'), findsOneWidget);
    });

    testWidgets('say something different when a search misses', (tester) async {
      await pump(tester, const DiaryEmptyState(searching: true));
      expect(find.text('Nothing matches that.'), findsOneWidget);
      expect(find.textContaining('shorter search'), findsOneWidget);
    });
  });

  group('layout', () {
    testWidgets('a long name and note do not overflow at phone width', (
      tester,
    ) async {
      tester.view.physicalSize = const Size(400, 800);
      tester.view.devicePixelRatio = 1.0;
      addTearDown(tester.view.reset);

      await pump(
        tester,
        DiaryEntry(
          contact: contact(
            name: 'Kongthong whistling village homestay, Bah Rothell family',
            note: 'Ask for the room at the back, reception closes at 22:00',
          ),
          lineNumber: 12,
        ),
      );
      expect(tester.takeException(), isNull);
    });
  });
}
