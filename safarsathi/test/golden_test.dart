// Renders the real screens to PNGs so they can be looked at without a phone.
//
// Run with:  flutter test --update-goldens test/golden_test.dart
//
// Fonts are loaded from the bundled assets on purpose. Without this the test
// renderer substitutes its own font and the images tell you nothing about
// whether Inter and Archivo Narrow are wired correctly — which is the one
// thing about this design no other test can check.

import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:safarsathi/core/database/app_database.dart';
import 'package:safarsathi/core/theme/app_tokens.dart';
import 'package:safarsathi/features/contacts/data/contacts_dao.dart';
import 'package:safarsathi/features/contacts/data/entry_draft.dart';
import 'package:safarsathi/features/contacts/presentation/diary_screen.dart';
import 'package:safarsathi/features/contacts/presentation/entry_form_screen.dart';
import 'package:safarsathi/features/contacts/presentation/entry_screen.dart';
import 'package:safarsathi/features/emergency/presentation/emergency_screen.dart';
import 'package:safarsathi/features/trips/data/trip_summary.dart';
import 'package:safarsathi/features/money/data/money_summary.dart';
import 'package:safarsathi/features/money/data/settlement.dart';
import 'package:safarsathi/features/money/presentation/money_screen.dart';
import 'package:safarsathi/features/trips/presentation/trip_screen.dart';

Future<void> loadRealFonts() async {
  const families = {
    'Jost': ['assets/fonts/Jost.ttf'],
    'Archivo': ['assets/fonts/Archivo.ttf'],
    'CourierPrime': [
      'assets/fonts/CourierPrime-Regular.ttf',
      'assets/fonts/CourierPrime-Bold.ttf',
    ],
  };
  for (final entry in families.entries) {
    final loader = FontLoader(entry.key);
    for (final asset in entry.value) {
      loader.addFont(rootBundle.load(asset));
    }
    await loader.load();
  }

  // Material's icon font ships with the SDK, not with the app, so the test
  // renderer does not have it and every icon renders as a tofu box. Walk up
  // from the test binary to find it; skip if it is not there, since goldens
  // are a dev convenience and should not fail a checkout elsewhere.
  var dir = File(Platform.resolvedExecutable).parent;
  for (var i = 0; i < 8; i++) {
    final candidate = File(
      '${dir.path}/bin/cache/artifacts/material_fonts/'
      'MaterialIcons-Regular.otf',
    );
    if (candidate.existsSync()) {
      final loader = FontLoader(
        'MaterialIcons',
      )..addFont(Future.value(candidate.readAsBytesSync().buffer.asByteData()));
      await loader.load();
      break;
    }
    dir = dir.parent;
  }
}

Contact row(
  String name,
  String phone, {
  required int id,
  String category = ContactCategory.other,
  String tier = 'userEntered',
  bool confirmed = false,
  bool pinned = false,
  String? note,
  int? stopId,
}) {
  return Contact(
    id: id,
    name: name,
    phoneRaw: phone,
    category: category,
    tier: tier,
    callConfirmed: confirmed,
    isPinned: pinned,
    isEmergency: false,
    hasWhatsapp: false,
    callCount: 0,
    note: note,
    stopId: stopId,
    createdAt: DateTime(2026, 9, 12),
  );
}

final demo = <Contact>[
  row(
    'Kongthong homestay',
    '+91 90000 00001',
    id: 1,
    category: ContactCategory.accommodation,
    tier: 'userVerified',
    confirmed: true,
    pinned: true,
    note: 'Bah Rothell',
  ),
  row(
    'Wanshai · driver',
    '+91 90000 00002',
    id: 2,
    category: ContactCategory.transport,
    tier: 'userVerified',
    confirmed: true,
    note: 'Whole trip',
  ),
  row(
    'Shillong guesthouse',
    '+91 90000 00003',
    id: 3,
    category: ContactCategory.accommodation,
    note: 'Front desk',
  ),
  row(
    'Bah Deiwan · guide',
    '+91 90000 00004',
    id: 4,
    category: ContactCategory.guide,
    note: 'Village walk',
  ),
  row(
    'Civil Hospital Shillong',
    '0364 900 0005',
    id: 5,
    category: ContactCategory.hospital,
    tier: 'communityOsm',
    note: 'From map data',
  ),
  row(
    'IOC pump · Sohra',
    '+91 90000 00006',
    id: 6,
    category: ContactCategory.fuel,
    note: 'Last fuel',
  ),
];

Stream<List<Contact>> Function(ContactFilter) feed(List<Contact> rows) {
  return (f) {
    var out = rows.where((c) => f.category == null || c.category == f.category);
    final term = f.searchTerm.trim().toLowerCase();
    if (term.isNotEmpty) {
      out = out.where((c) => c.name.toLowerCase().contains(term));
    }
    final list = out.toList()
      ..sort((a, b) {
        if (a.isPinned != b.isPinned) return a.isPinned ? -1 : 1;
        if (a.callConfirmed != b.callConfirmed) {
          return a.callConfirmed ? -1 : 1;
        }
        return a.name.compareTo(b.name);
      });
    return Stream<List<Contact>>.value(list);
  };
}

void main() {
  setUpAll(loadRealFonts);

  Future<void> shoot(
    WidgetTester tester,
    String name, {
    required ThemeData theme,
    List<Contact> rows = const [],
  }) async {
    tester.view.physicalSize = const Size(840, 1780);
    tester.view.devicePixelRatio = 2.0;
    addTearDown(tester.view.reset);

    await tester.pumpWidget(
      MaterialApp(
        debugShowCheckedModeBanner: false,
        theme: theme,
        home: DiaryScreen(
          watchContacts: feed(rows),
          unconfirmedCount: Stream<int>.value(
            rows.where((c) => !c.callConfirmed).length,
          ),
          tripId: 1,
          tripName: 'Meghalaya · 1–5 Oct',
          currentStopId: 4,
          currentStopName: 'Kongthong',
          onCopy: (c) async => c.phoneRaw,
          onAdd: () {},
        ),
      ),
    );
    await tester.pump();
    await tester.pump();

    await expectLater(
      find.byType(MaterialApp),
      matchesGoldenFile('goldens/$name.png'),
    );
  }

  testWidgets('diary — paper', (tester) async {
    await shoot(tester, 'diary_paper', theme: AppTokens.light, rows: demo);
  });

  testWidgets('diary — lamp', (tester) async {
    await shoot(tester, 'diary_lamp', theme: AppTokens.dark, rows: demo);
  });

  testWidgets('diary — empty', (tester) async {
    await shoot(tester, 'diary_empty', theme: AppTokens.light);
  });

  testWidgets('diary — after a copy', (tester) async {
    tester.view.physicalSize = const Size(840, 1780);
    tester.view.devicePixelRatio = 2.0;
    addTearDown(tester.view.reset);

    await tester.pumpWidget(
      MaterialApp(
        debugShowCheckedModeBanner: false,
        theme: AppTokens.light,
        home: DiaryScreen(
          watchContacts: feed(demo),
          unconfirmedCount: Stream<int>.value(4),
          tripId: 1,
          tripName: 'Meghalaya · 1–5 Oct',
          currentStopId: 4,
          currentStopName: 'Kongthong',
          onCopy: (c) async => c.phoneRaw,
          onOpenDialer: () async {},
          onAdd: () {},
        ),
      ),
    );
    await tester.pump();
    await tester.pump();

    await tester.tap(find.text('Shillong guesthouse'));
    await tester.pump();
    await tester.pump();

    await expectLater(
      find.byType(MaterialApp),
      matchesGoldenFile('goldens/diary_copied.png'),
    );
  });

  testWidgets('new entry form', (tester) async {
    tester.view.physicalSize = const Size(840, 1780);
    tester.view.devicePixelRatio = 2.0;
    addTearDown(tester.view.reset);

    await tester.pumpWidget(
      MaterialApp(
        debugShowCheckedModeBanner: false,
        theme: AppTokens.light,
        home: EntryFormScreen(
          stops: const [
            StopOption(1, 'Shillong'),
            StopOption(2, 'Cherrapunji'),
            StopOption(3, 'Kongthong'),
          ],
          findDuplicate: (_) async => demo[2],
          onSave: (EntryDraft d) async {},
        ),
      ),
    );
    await tester.pump();

    await tester.enterText(
      find.descendant(
        of: find.byKey(const Key('field-name')),
        matching: find.byType(TextField),
      ),
      'Sohra tea stall · Kong Bina',
    );
    await tester.enterText(
      find.descendant(
        of: find.byKey(const Key('field-number')),
        matching: find.byType(TextField),
      ),
      '+91 90000 00003',
    );
    await tester.pump();
    await tester.pump();

    await expectLater(
      find.byType(MaterialApp),
      matchesGoldenFile('goldens/entry_form.png'),
    );
  });

  Future<void> shootEntry(
    WidgetTester tester,
    String name, {
    required Contact contact,
    bool confirmAfterPump = false,
  }) async {
    tester.view.physicalSize = const Size(840, 1780);
    tester.view.devicePixelRatio = 2.0;
    addTearDown(tester.view.reset);

    await tester.pumpWidget(
      MaterialApp(
        debugShowCheckedModeBanner: false,
        theme: AppTokens.light,
        home: EntryScreen(
          contact: contact,
          stopName: 'Shillong · nights 1 and 4',
          onCopy: (c) async => c.phoneRaw,
          onOpenDialer: () async {},
          onCall: (_) async {},
          onChat: (_) async {},
          onConfirm: (_, {required confirmed}) async {},
          onEdit: (_) {},
        ),
      ),
    );
    await tester.pump();

    if (confirmAfterPump) {
      await tester.tap(find.text('Mark confirmed'));
      await tester.pump();
      // Past the 380ms stamp, so the image shows it settled.
      await tester.pump(const Duration(milliseconds: 500));
    }

    await expectLater(
      find.byType(MaterialApp),
      matchesGoldenFile('goldens/$name.png'),
    );
  }

  testWidgets('entry — unconfirmed', (tester) async {
    await shootEntry(tester, 'entry_unconfirmed', contact: demo[2]);
  });

  testWidgets('entry — stamp landed', (tester) async {
    await shootEntry(
      tester,
      'entry_confirmed',
      contact: demo[2],
      confirmAfterPump: true,
    );
  });

  Future<void> shootScreen(
    WidgetTester tester,
    String name,
    Widget screen,
  ) async {
    tester.view.physicalSize = const Size(840, 1780);
    tester.view.devicePixelRatio = 2.0;
    addTearDown(tester.view.reset);

    await tester.pumpWidget(
      MaterialApp(
        debugShowCheckedModeBanner: false,
        theme: AppTokens.light,
        home: screen,
      ),
    );
    await tester.pump();
    await tester.pump();

    await expectLater(
      find.byType(MaterialApp),
      matchesGoldenFile('goldens/$name.png'),
    );
  }

  testWidgets('emergency', (tester) async {
    EmergencyHelpline h(String n, String l, String s, int id) =>
        EmergencyHelpline(
          id: id,
          countryCode: 'IN',
          serviceType: 'all',
          label: l,
          number: n,
          sourceNote: s,
          needsVerification: false,
          tier: 'verifiedNational',
        );

    await shootScreen(
      tester,
      'emergency',
      EmergencyScreen(
        placeLabel: 'India · Kongthong',
        helplines: Stream.value([
          h(
            '112',
            'All emergencies',
            '112.gov.in, Ministry of Home Affairs',
            1,
          ),
          h('108', 'Ambulance', 'Legacy line, active alongside 112', 2),
          h('101', 'Fire', 'Legacy line, active alongside 112', 3),
          h('181', 'Women helpline', 'india.gov.in helpline directory', 4),
          h('1098', 'Child helpline', 'india.gov.in helpline directory', 5),
        ]),
        localContacts: Stream.value([
          Contact(
            id: 9,
            name: 'Bah Rothell · homestay owner',
            phoneRaw: '+91 90000 00007',
            category: ContactCategory.localContact,
            tier: ContactTier.userVerified.name,
            callConfirmed: true,
            isPinned: false,
            isEmergency: true,
            hasWhatsapp: false,
            callCount: 0,
            createdAt: DateTime(2026, 9, 12),
          ),
        ]),
        onCall: (_) async {},
        onCopy: (_) async {},
      ),
    );
  });

  testWidgets('trip', (tester) async {
    await shootScreen(
      tester,
      'trip',
      TripScreen(
        unconfirmedCount: Stream.value(4),
        trip: Stream.value(
          TripSummary(
            name: 'Meghalaya · demo',
            startDate: DateTime(2026, 10, 1),
            endDate: DateTime(2026, 10, 5),
            nextLeg: LegSummary(
              fromName: 'Cherrapunji',
              toName: 'Kongthong',
              mode: 'Shared taxi',
              distanceKm: 56,
              plannedDeparture: DateTime(2026, 10, 3, 9, 30),
              note: 'Wanshai has the pickup point',
            ),
            stops: [
              StopSummary(
                id: 1,
                name: 'Shillong',
                sequenceOrder: 1,
                nights: 1,
                diaryCount: 4,
                isCurrent: false,
                arrivalDate: DateTime(2026, 10, 1),
              ),
              StopSummary(
                id: 2,
                name: 'Cherrapunji',
                sequenceOrder: 2,
                nights: 1,
                diaryCount: 3,
                isCurrent: false,
                arrivalDate: DateTime(2026, 10, 2),
              ),
              StopSummary(
                id: 3,
                name: 'Kongthong',
                sequenceOrder: 3,
                nights: 2,
                diaryCount: 5,
                isCurrent: true,
                arrivalDate: DateTime(2026, 10, 3),
              ),
            ],
          ),
        ),
      ),
    );
  });

  testWidgets('money', (tester) async {
    const you = Balance(travellerId: 1, name: 'You', netMinor: 481341);
    const ankit = Balance(travellerId: 2, name: 'Ankit', netMinor: -155337);
    const priya = Balance(travellerId: 3, name: 'Priya', netMinor: -326004);
    const balances = [you, ankit, priya];

    await shootScreen(
      tester,
      'money',
      MoneyScreen(
        summary: Stream.value(
          MoneySummary(
            totalMinor: 1206011,
            travellerCount: 3,
            balances: balances,
            settlements: simplifyDebts(balances),
            ledger: [
              LedgerEntry(
                id: 1,
                description: 'Homestay · 2 nights',
                amountMinor: 440000,
                paidByName: 'You',
                splitCount: 3,
                spentAt: DateTime(2026, 10, 3),
              ),
              LedgerEntry(
                id: 2,
                description: 'Cave guide',
                amountMinor: 150000,
                paidByName: 'Priya',
                splitCount: 3,
                spentAt: DateTime(2026, 10, 3),
              ),
              LedgerEntry(
                id: 3,
                description: 'Taxi · Shillong to Cherrapunji',
                amountMinor: 320011,
                paidByName: 'You',
                splitCount: 3,
                spentAt: DateTime(2026, 10, 2),
              ),
              LedgerEntry(
                id: 4,
                description: 'Fuel',
                amountMinor: 210000,
                paidByName: 'You',
                splitCount: 3,
                spentAt: DateTime(2026, 10, 2),
              ),
              LedgerEntry(
                id: 5,
                description: 'Dinner at Sohra',
                amountMinor: 86000,
                paidByName: 'Ankit',
                splitCount: 3,
                spentAt: DateTime(2026, 10, 2),
              ),
            ],
          ),
        ),
      ),
    );
  });
}
