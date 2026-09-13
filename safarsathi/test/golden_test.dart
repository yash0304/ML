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
import 'package:safarsathi/features/trips/data/readiness.dart';
import 'package:safarsathi/features/trips/data/trip_editor.dart';
import 'package:safarsathi/features/trips/presentation/itinerary_screen.dart';
import 'package:safarsathi/features/trips/presentation/leg_list_screen.dart';
import 'package:safarsathi/features/trips/presentation/stop_form_screen.dart';
import 'package:safarsathi/features/trips/presentation/trip_list_screen.dart';
import 'package:safarsathi/features/checklist/data/checklist_dao.dart';
import 'package:safarsathi/features/checklist/presentation/checklist_screen.dart';
import 'package:safarsathi/features/money/data/expense_editor.dart';
import 'package:safarsathi/features/money/presentation/expense_form_screen.dart';
import 'package:safarsathi/features/import/data/column_mapping.dart';
import 'package:safarsathi/features/import/data/import_commit.dart';
import 'package:safarsathi/features/import/data/import_validation.dart';
import 'package:safarsathi/features/import/data/sheet_parser.dart';
import 'package:safarsathi/features/import/data/stop_matcher.dart';
import 'package:safarsathi/features/import/presentation/column_mapping_screen.dart';
import 'package:safarsathi/features/import/presentation/import_history_screen.dart';
import 'package:safarsathi/features/import/presentation/import_preview_screen.dart';
import 'package:safarsathi/features/import/presentation/more_screen.dart';
import 'dart:convert';

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

  // ---------------------------------------------------------------------
  // Import (#11–#15)
  // ---------------------------------------------------------------------

  // A sheet with everything wrong with it that a real sheet has: a homestay
  // already in the diary, a number Excel mangled into a float, a stop that is
  // not on the trip, a row with no name, and one with no number at all.
  const messySheet = 'Guest Name,Mobile No.,Type,Place,Remarks\n'
      'Rina Kharkongor,+91 90000 00001,Homestay,Shillong,Blue gate\n'
      'Biren Lyngdoh,9000000002,Driver,Shillong,Innova\n'
      'Dawki Boat,+91 90000 00003,Transport,Dawki,\n'
      'Sohra Chemist,call the shop,Chemist,Cherrapunji,Ask at the market\n'
      ',+91 90000 00005,Guide,Shillong,\n'
      'Mawlynnong Stay,,Homestay,Mawlynnong,\n'
      'Rina Kharkongor,+91 90000 00001,Homestay,Dawki,Same person\n';

  SheetTable messyTable() => SheetParser.parse(
    'meghalaya-contacts.csv',
    Uint8List.fromList(utf8.encode(messySheet)),
  ).sheets.single;

  testWidgets('import — column mapping', (tester) async {
    final table = messyTable();
    await shootScreen(
      tester,
      'import_mapping',
      ColumnMappingScreen(
        table: table,
        initial: autoMatchColumns(table.headers),
        onContinue: (_) {},
      ),
    );
  });

  testWidgets('import — preview', (tester) async {
    final table = messyTable();
    final preview = validateRows(
      table,
      autoMatchColumns(table.headers),
      stops: const [
        StopCandidate(1, 'Shillong'),
        StopCandidate(2, 'Dawki'),
        StopCandidate(3, 'Cherrapunji'),
      ],
      existing: const ExistingContacts(
        e164: {'+919000000001'},
        squashedNames: {'rinakharkongor'},
      ),
    );

    await shootScreen(
      tester,
      'import_preview',
      ImportPreviewScreen(
        preview: preview,
        fileName: 'meghalaya-contacts.csv',
        onCommit: (_) async => 0,
      ),
    );
  });

  testWidgets('import — history', (tester) async {
    await shootScreen(
      tester,
      'import_history',
      ImportHistoryScreen(
        onRollback: (_) async {},
        batches: Stream.value([
          ImportBatchSummary(
            id: 1,
            fileName: 'meghalaya-contacts.csv',
            rowsImported: 5,
            rowsSkipped: 2,
            importedAt: DateTime(2026, 9, 28),
            stillPresent: 5,
          ),
          ImportBatchSummary(
            id: 2,
            fileName: 'trip-planning.xlsx',
            sheetName: 'Drivers',
            rowsImported: 9,
            rowsSkipped: 0,
            importedAt: DateTime(2026, 9, 21),
            stillPresent: 7,
          ),
        ]),
      ),
    );
  });

  testWidgets('more', (tester) async {
    await shootScreen(
      tester,
      'more',
      MoreScreen(
        contactCount: Stream.value(23),
        onTrips: () {},
        onItinerary: () {},
        onLegs: () {},
        onChecklist: () {},
        onTravellers: () {},
        onImport: () {},
        onHistory: () {},
        onTemplate: () {},
      ),
    );
  });

  // ---------------------------------------------------------------------
  // Trip structure (#16–#20)
  // ---------------------------------------------------------------------

  Stop stopRow(
    int id,
    String name,
    int order, {
    int nights = 0,
    String tags = '',
    DateTime? arrival,
  }) => Stop(
    id: id,
    tripId: 1,
    name: name,
    sequenceOrder: order,
    countryCode: 'IN',
    nights: nights,
    activityTags: tags,
    arrivalDate: arrival,
  );

  testWidgets('itinerary', (tester) async {
    await shootScreen(
      tester,
      'itinerary',
      ItineraryScreen(
        tripName: 'Meghalaya, October',
        onReorder: (_, _) async {},
        onEdit: (_) {},
        onAdd: () {},
        onEditTrip: () {},
        stops: Stream.value([
          stopRow(
            1,
            'Shillong',
            1,
            nights: 2,
            tags: 'city,rain',
            arrival: DateTime(2026, 10, 1),
          ),
          stopRow(
            2,
            'Cherrapunji',
            2,
            nights: 2,
            tags: 'caves,rain,trek',
            arrival: DateTime(2026, 10, 3),
          ),
          // The same place again, which is the rule this screen exists to
          // make visible.
          stopRow(
            3,
            'Shillong',
            3,
            nights: 1,
            tags: 'city',
            arrival: DateTime(2026, 10, 5),
          ),
          stopRow(4, 'Dawki', 4, arrival: DateTime(2026, 10, 6)),
          stopRow(
            5,
            'Mawlynnong',
            5,
            nights: 1,
            tags: 'homestay',
            arrival: DateTime(2026, 10, 6),
          ),
        ]),
      ),
    );
  });

  testWidgets('stop form', (tester) async {
    await shootScreen(
      tester,
      'stop_form',
      StopFormScreen(
        existing: StopDraft(
          id: 2,
          name: 'Cherrapunji',
          nights: 2,
          activityTags: const ['caves', 'rain'],
          arrivalDate: DateTime(2026, 10, 3),
          departureDate: DateTime(2026, 10, 5),
          note: 'Blue gate past the church',
        ),
        contactsHere: () async => 3,
        onDelete: () async {},
        onSave: (_) async {},
      ),
    );
  });

  testWidgets('trips', (tester) async {
    await shootScreen(
      tester,
      'trips',
      TripListScreen(
        onActivate: (_) async {},
        onOpen: (_) {},
        onDelete: (_) async {},
        onCreate: () {},
        trips: Stream.value([
          Trip(
            id: 1,
            name: 'Meghalaya, October',
            startDate: DateTime(2026, 10, 1),
            endDate: DateTime(2026, 10, 5),
            baseCurrency: 'INR',
            isActive: true,
            createdAt: DateTime(2026, 9, 1),
          ),
          Trip(
            id: 2,
            name: 'Ladakh, next summer',
            baseCurrency: 'INR',
            isActive: false,
            createdAt: DateTime(2026, 8, 1),
          ),
        ]),
      ),
    );
  });

  testWidgets('legs', (tester) async {
    await shootScreen(
      tester,
      'legs',
      LegListScreen(
        onOpen: (_) {},
        legs: Stream.value([
          LegRow(
            id: 1,
            fromName: 'Shillong',
            toName: 'Cherrapunji',
            mode: 'Shared sumo',
            plannedDeparture: DateTime(2026, 10, 3, 7, 30),
            isBooked: true,
            distanceKm: 54,
          ),
          const LegRow(
            id: 2,
            fromName: 'Cherrapunji',
            toName: 'Shillong',
            isBooked: false,
            distanceKm: 54,
          ),
          const LegRow(
            id: 3,
            fromName: 'Shillong',
            toName: 'Dawki',
            mode: 'Taxi',
            isBooked: false,
            distanceKm: 82,
          ),
        ]),
      ),
    );
  });

  testWidgets('trip — not ready', (tester) async {
    // The Trip screen carrying the pre-departure block, which is the payoff
    // for the whole trust system.
    await shootScreen(
      tester,
      'trip_blocked',
      TripScreen(
        unconfirmedCount: Stream.value(2),
        onEditItinerary: () {},
        readiness: Stream.value(
          const Readiness([
            ReadinessItem(
              stopId: 2,
              stopName: 'Cherrapunji',
              contactId: 9,
              label: 'Call and confirm the Cherrapunji accommodation number.',
            ),
            ReadinessItem(
              stopId: 5,
              stopName: 'Mawlynnong',
              missing: true,
              label: 'No accommodation number for Mawlynnong.',
            ),
          ]),
        ),
        trip: Stream.value(
          TripSummary(
            name: 'Meghalaya, October',
            startDate: DateTime(2026, 10, 1),
            endDate: DateTime(2026, 10, 6),
            nextLeg: const LegSummary(
              fromName: 'Shillong',
              toName: 'Cherrapunji',
              mode: 'Shared sumo',
              distanceKm: 54,
            ),
            stops: [
              const StopSummary(
                id: 1,
                name: 'Shillong',
                sequenceOrder: 1,
                nights: 2,
                diaryCount: 6,
                isCurrent: true,
              ),
              const StopSummary(
                id: 2,
                name: 'Cherrapunji',
                sequenceOrder: 2,
                nights: 2,
                diaryCount: 3,
                isCurrent: false,
              ),
              const StopSummary(
                id: 5,
                name: 'Mawlynnong',
                sequenceOrder: 3,
                nights: 1,
                diaryCount: 0,
                isCurrent: false,
              ),
            ],
          ),
        ),
      ),
    );
  });

  // ---------------------------------------------------------------------
  // Checklist and expense entry (#29, #31)
  // ---------------------------------------------------------------------

  ChecklistItem listItem(
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

  testWidgets('checklist', (tester) async {
    await shootScreen(
      tester,
      'checklist',
      ChecklistScreen(
        onToggle: (_, _) async {},
        onRemove: (_) async {},
        onEdit: (_, _, _) async {},
        onAdd: (_) async {},
        onRegenerate: () async {},
        checklist: Stream.value(
          ChecklistView(
            blocking: [
              listItem(
                1,
                'Call and confirm the Cherrapunji accommodation number.',
                blocking: true,
                tags: 'readiness',
              ),
              listItem(
                2,
                'No accommodation number for Mawlynnong.',
                blocking: true,
                tags: 'readiness',
              ),
            ],
            pack: [
              listItem(3, 'Changes of clothes', quantity: '6', done: true),
              listItem(4, 'Toothbrush and paste', done: true),
              listItem(5, 'Any medicines you take'),
              listItem(6, 'Headtorch', tags: 'caves'),
              listItem(7, 'Spare batteries', tags: 'caves'),
              listItem(8, 'Poncho or rain jacket', tags: 'rain'),
              listItem(9, 'Dry bag for the phone', tags: 'rain'),
              listItem(10, 'Trekking socks', quantity: '2', tags: 'trek'),
              listItem(11, 'Leech socks', tags: 'trek'),
              listItem(
                12,
                'Rina says bring cash, no card machine',
                tags: 'homestay',
                edited: true,
              ),
            ],
          ),
        ),
      ),
    );
  });

  testWidgets('expense form', (tester) async {
    await shootScreen(
      tester,
      'expense_form',
      ExpenseFormScreen(
        travellers: [
          const Traveller(id: 1, tripId: 1, name: 'Yash', isSelf: true),
          const Traveller(id: 2, tripId: 1, name: 'Priya', isSelf: false),
          const Traveller(id: 3, tripId: 1, name: 'Ankit', isSelf: false),
        ],
        existing: ExpenseDraft(
          id: 1,
          description: 'Taxi, Shillong to Cherrapunji',
          amountMinor: 320011,
          paidById: 1,
          shares: evenSplit(320011, const [1, 2, 3]),
          spentAt: DateTime(2026, 10, 3),
        ),
        onSave: (_) async {},
        onDelete: () async {},
      ),
    );
  });
}
