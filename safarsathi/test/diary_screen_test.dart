// The diary screen, driven by plain streams.
//
// No database here on purpose. The DAO's ordering, filtering and search are
// covered against real SQLite in contacts_dao_test.dart; what this file
// checks is that the screen renders what arrives and asks for the right thing
// when the user searches or flips a tab.
//
// Keeping Drift out of widget tests is not just convenience. A widget test
// cannot close a Drift database — close() awaits work the fake clock never
// advances, and the test hangs until the runner is killed — and cancelling a
// query stream leaves cleanup timers the framework then reports as pending.

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:safarsathi/core/database/app_database.dart';
import 'package:safarsathi/core/theme/app_tokens.dart';
import 'package:safarsathi/features/contacts/data/contacts_dao.dart';
import 'package:safarsathi/features/contacts/presentation/diary_screen.dart';

Contact entry(
  String name,
  String phone, {
  int id = 0,
  String category = ContactCategory.other,
  bool confirmed = false,
  int? stopId,
}) {
  return Contact(
    id: id == 0 ? name.hashCode : id,
    name: name,
    phoneRaw: phone,
    category: category,
    tier: confirmed
        ? ContactTier.userVerified.name
        : ContactTier.userEntered.name,
    callConfirmed: confirmed,
    isPinned: false,
    isEmergency: false,
    hasWhatsapp: false,
    callCount: 0,
    stopId: stopId,
    createdAt: DateTime(2026, 9, 12),
  );
}

/// Applies the same rules as the DAO, in memory, so the screen can be driven
/// without SQLite. Records the filters it was asked for.
class FakeDiary {
  final List<Contact> all;
  final filters = <ContactFilter>[];

  FakeDiary(this.all);

  Stream<List<Contact>> watch(ContactFilter f) {
    filters.add(f);
    var rows = all.where((c) => !c.isEmergency);
    if (f.category != null) {
      rows = rows.where((c) => c.category == f.category);
    }
    if (f.stopId != null) {
      rows = rows.where((c) => c.stopId == f.stopId || c.stopId == null);
    }
    final term = f.searchTerm.trim().toLowerCase();
    if (term.isNotEmpty) {
      rows = rows.where(
        (c) =>
            c.name.toLowerCase().contains(term) ||
            c.phoneRaw.toLowerCase().contains(term),
      );
    }
    final list = rows.toList()
      ..sort((a, b) {
        if (a.callConfirmed != b.callConfirmed) {
          return a.callConfirmed ? -1 : 1;
        }
        return a.name.compareTo(b.name);
      });
    return Stream<List<Contact>>.value(list);
  }

  int get unconfirmed => all.where((c) => !c.callConfirmed).length;
}

void main() {
  /// A filter change swaps in a new stream, which delivers on a microtask.
  /// One frame paints the empty StreamBuilder; the next paints the rows.
  Future<void> settle(WidgetTester tester) async {
    await tester.pump();
    await tester.pump();
  }

  Future<FakeDiary> pumpDiary(
    WidgetTester tester,
    List<Contact> rows, {
    String? currentStopName,
    int? currentStopId,
  }) async {
    final fake = FakeDiary(rows);
    await tester.pumpWidget(
      MaterialApp(
        theme: AppTokens.light,
        home: DiaryScreen(
          watchContacts: fake.watch,
          unconfirmedCount: Stream<int>.value(fake.unconfirmed),
          tripId: 1,
          tripName: 'Meghalaya',
          currentStopId: currentStopId,
          currentStopName: currentStopName,
          onCopy: (_) {},
        ),
      ),
    );
    await tester.pump();
    await tester.pump();
    return fake;
  }

  testWidgets('renders entries with margin numbers and a page footer', (
    tester,
  ) async {
    await pumpDiary(tester, [
      entry('Zeta', '+91 90000 00001'),
      entry('Alpha', '+91 90000 00002'),
      entry('Yankee', '+91 90000 00003', confirmed: true),
    ]);

    expect(find.text('01'), findsOneWidget);
    expect(find.text('02'), findsOneWidget);
    expect(find.text('03'), findsOneWidget);
    expect(find.textContaining('3 entries'), findsOneWidget);
    expect(find.text('Diary'), findsOneWidget);
  });

  testWidgets('the readiness banner counts only unconfirmed entries', (
    tester,
  ) async {
    await pumpDiary(tester, [
      entry('Homestay', '+91 90000 00001'),
      entry('Guesthouse', '+91 90000 00002'),
      entry('Driver', '+91 90000 00003', confirmed: true),
    ]);

    expect(find.textContaining('numbers not confirmed'), findsOneWidget);
    expect(find.text('2'), findsOneWidget);
  });

  testWidgets('the banner stays away when everything is confirmed', (
    tester,
  ) async {
    await pumpDiary(tester, [
      entry('Homestay', '+91 90000 00001', confirmed: true),
    ]);
    expect(find.textContaining('not confirmed'), findsNothing);
  });

  testWidgets('an empty diary gives direction', (tester) async {
    await pumpDiary(tester, []);
    expect(find.text('Nothing in the diary yet.'), findsOneWidget);
    expect(find.textContaining('import a sheet'), findsOneWidget);
  });

  testWidgets('searching narrows the page, and clearing restores it', (
    tester,
  ) async {
    await pumpDiary(tester, [
      entry('Kongthong homestay', '+91 90000 00001'),
      entry('Driver', '+91 90000 00002'),
    ]);
    expect(find.text('Driver'), findsOneWidget);

    await tester.enterText(find.byType(TextField), 'homestay');
    await settle(tester);
    expect(find.text('Kongthong homestay'), findsOneWidget);
    expect(find.text('Driver'), findsNothing);

    await tester.tap(find.byIcon(Icons.close));
    await settle(tester);
    expect(find.text('Driver'), findsOneWidget);
  });

  testWidgets('the thumb index filters by category', (tester) async {
    final fake = await pumpDiary(tester, [
      entry(
        'Kongthong homestay',
        '+91 90000 00001',
        category: ContactCategory.accommodation,
      ),
      entry('Driver', '+91 90000 00002', category: ContactCategory.transport),
    ]);

    await tester.tap(find.text('STAY'));
    await settle(tester);

    expect(fake.filters.last.category, ContactCategory.accommodation);
    expect(find.text('Kongthong homestay'), findsOneWidget);
    expect(find.text('Driver'), findsNothing);

    await tester.tap(find.text('ALL'));
    await settle(tester);
    expect(fake.filters.last.category, isNull);
    expect(find.text('Driver'), findsOneWidget);
  });

  testWidgets('a missed search says so differently from an empty diary', (
    tester,
  ) async {
    await pumpDiary(tester, [entry('Driver', '+91 90000 00002')]);
    await tester.enterText(find.byType(TextField), 'zzzz');
    await settle(tester);
    expect(find.text('Nothing matches that.'), findsOneWidget);
  });

  testWidgets('the stop-scope toggle keeps trip-wide entries', (tester) async {
    // Your driver is not tied to one stop, but you still need him while
    // standing in Kongthong.
    final fake = await pumpDiary(
      tester,
      [
        entry('Kongthong homestay', '+91 90000 00001', stopId: 7),
        entry('Shillong guesthouse', '+91 90000 00003', stopId: 9),
        entry('Driver', '+91 90000 00002'),
      ],
      currentStopId: 7,
      currentStopName: 'Kongthong',
    );

    await tester.tap(find.byIcon(Icons.travel_explore));
    await settle(tester);

    expect(fake.filters.last.stopId, 7);
    expect(find.text('Kongthong homestay'), findsOneWidget);
    expect(find.text('Driver'), findsOneWidget);
    expect(find.text('Shillong guesthouse'), findsNothing);
  });

  testWidgets('no scope toggle when the trip has no current stop', (
    tester,
  ) async {
    await pumpDiary(tester, [entry('Driver', '+91 90000 00002')]);
    expect(find.byIcon(Icons.travel_explore), findsNothing);
  });

  testWidgets('the whole screen fits a phone', (tester) async {
    tester.view.physicalSize = const Size(400, 800);
    tester.view.devicePixelRatio = 1.0;
    addTearDown(tester.view.reset);

    await pumpDiary(tester, [
      entry('Kongthong homestay', '+91 90000 00001'),
      entry('Driver', '+91 90000 00002', confirmed: true),
    ]);
    expect(tester.takeException(), isNull);
  });

  testWidgets('and fits it in the night palette too', (tester) async {
    tester.view.physicalSize = const Size(400, 800);
    tester.view.devicePixelRatio = 1.0;
    addTearDown(tester.view.reset);

    final fake = FakeDiary([
      entry('Kongthong homestay', '+91 90000 00001'),
      entry('Driver', '+91 90000 00002', confirmed: true),
    ]);
    await tester.pumpWidget(
      MaterialApp(
        theme: AppTokens.dark,
        home: DiaryScreen(
          watchContacts: fake.watch,
          unconfirmedCount: Stream<int>.value(fake.unconfirmed),
          tripId: 1,
          tripName: 'Meghalaya',
          onCopy: (_) {},
        ),
      ),
    );
    await tester.pump();
    await tester.pump();
    expect(tester.takeException(), isNull);
    expect(find.text('Diary'), findsOneWidget);
  });
}
