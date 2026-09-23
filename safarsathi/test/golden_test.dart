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
import 'package:safarsathi/core/util/sun.dart'
    show clockOffsetOverride, sunTimes;
import 'package:safarsathi/features/trips/data/tonight.dart';
import 'package:safarsathi/features/emergency/presentation/sos_panel.dart';
import 'package:safarsathi/features/map/data/here.dart';
import 'package:safarsathi/features/contacts/data/contacts_dao.dart';
import 'package:safarsathi/features/contacts/data/entry_draft.dart';
import 'package:safarsathi/features/contacts/presentation/diary_screen.dart';
import 'package:safarsathi/features/contacts/presentation/entry_form_screen.dart';
import 'package:safarsathi/features/backup/data/backup.dart';
import 'package:safarsathi/features/backup/presentation/backup_screen.dart';
import 'package:safarsathi/features/contacts/data/multi_add.dart';
import 'package:safarsathi/features/contacts/presentation/entry_screen.dart';
import 'package:safarsathi/features/contacts/presentation/multi_add_screen.dart';
import 'package:safarsathi/features/emergency/presentation/emergency_screen.dart';
import 'package:safarsathi/features/trips/data/trip_summary.dart';
import 'package:safarsathi/features/money/data/money_summary.dart';
import 'package:safarsathi/features/money/data/settlement.dart';
import 'package:safarsathi/features/money/presentation/money_screen.dart';
import 'package:safarsathi/features/trips/presentation/trip_screen.dart';
import 'package:safarsathi/features/trips/data/readiness.dart';
import 'package:safarsathi/features/trips/data/trip_editor.dart';
import 'package:safarsathi/features/trips/data/stop_detail.dart';
import 'package:safarsathi/features/trips/presentation/itinerary_screen.dart';
import 'package:safarsathi/features/trips/presentation/leg_detail_screen.dart';
import 'package:safarsathi/features/trips/presentation/stop_detail_screen.dart';
import 'package:safarsathi/features/trips/presentation/leg_list_screen.dart';
import 'package:safarsathi/features/trips/presentation/stop_form_screen.dart';
import 'package:safarsathi/features/trips/presentation/trip_list_screen.dart';
import 'package:safarsathi/features/checklist/data/checklist_dao.dart';
import 'package:safarsathi/features/checklist/presentation/checklist_screen.dart';
import 'package:safarsathi/features/money/data/expense_editor.dart';
import 'package:safarsathi/features/money/presentation/expense_form_screen.dart';
import 'package:safarsathi/features/settings/data/settings.dart';
import 'package:safarsathi/features/settings/presentation/settings_screen.dart';
import 'package:safarsathi/features/weather/data/weather_sync.dart';
import 'package:safarsathi/features/weather/presentation/weather_screen.dart';
import 'package:safarsathi/features/map/data/map_download.dart';
import 'package:safarsathi/features/map/data/tile_downloader.dart';
import 'package:safarsathi/features/map/data/tile_provider.dart';
import 'package:safarsathi/features/map/presentation/map_download_screen.dart';
import 'package:safarsathi/features/sync/data/sync_error.dart';
import 'package:safarsathi/features/sync/data/trip_sync.dart';
import 'package:safarsathi/features/sync/presentation/sync_screen.dart';
import 'package:safarsathi/features/discovery/data/discovery.dart';
import 'package:safarsathi/features/discovery/presentation/discovery_screen.dart';
import 'package:safarsathi/features/discovery/presentation/poi_detail_screen.dart';

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
  // Clock times in the images are Indian time on every machine — see
  // clockOffsetOverride.
  setUpAll(() => clockOffsetOverride = const Duration(hours: 5, minutes: 30));
  tearDownAll(() => clockOffsetOverride = null);

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
          // Frozen, so confirming in this test does not stamp the image with
          // the day it was rendered.
          clock: () => DateTime(2026, 9, 28),
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
        sosPanel: SosPanel(
          trusted: Stream.value(const [
            TrustedContact(
              id: 1,
              name: 'Mummy',
              phoneE164: '+919825012345',
              notifyOnArrival: false,
              escalate: false,
              escalateAfterMinutes: 30,
            ),
          ]),
          location: const _NoLocation(),
          nearStop: () async => 'Kongthong',
          openSms: (_) async => true,
          share: (_) async {},
          pickPerson: () async => null,
          addPerson: (_, _) async {},
          removePerson: (_) async {},
        ),
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
        // The two shortcuts the app now opens on. Wired here so the golden
        // covers the section rather than the null case.
        onViewMap: () {},
        onLegs: () {},
        onOpenContact: (_) {},
        onAddStay: (_) {},
        onSharePlan: () {},
        tonight: Stream.value(
          Tonight(
            kind: TonightKind.tonight,
            stop: Stop(
              id: 3,
              tripId: 1,
              name: 'Kongthong',
              sequenceOrder: 3,
              countryCode: 'IN',
              nights: 2,
              activityTags: '',
              lat: 25.3309,
              lon: 91.8238,
              arrivalDate: DateTime(2026, 10, 3),
            ),
            night: DateTime(2026, 10, 3),
            stays: [
              diaryContact(
                id: 9,
                name: 'Kongthong Travellers Nest',
                category: 'accommodation',
                phone: '+91 90000 00019',
                note: 'Owner meets the sumo at the village gate.',
              ),
            ],
            sun: sunTimes(25.3309, 91.8238, DateTime(2026, 10, 3)),
          ),
        ),
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
        onChecklist: () {},
        onTravellers: () {},
        onWeather: () {},
        onSettings: () {},
        onMap: () {},
        onSync: () {},
        onImport: () {},
        onMultiAdd: () {},
        onPickFromPhone: () {},
        onBackup: () {},
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

  // ---------------------------------------------------------------------
  // Setup: weather (#26) and settings (#35)
  // ---------------------------------------------------------------------

  testWidgets('weather', (tester) async {
    // Taken a week and a bit ago, so the staleness treatment is what the
    // image actually shows — the one rule this screen exists to enforce.
    final taken = DateTime(2026, 9, 22, 8);
    final now = DateTime(2026, 10, 1, 9);

    WeatherSnapshot day(
      int id,
      int stopId,
      int date,
      String condition,
      double min,
      double max,
      double rain,
    ) => WeatherSnapshot(
      id: id,
      stopId: stopId,
      forDate: DateTime(2026, 10, date),
      condition: condition,
      cachedAt: stopId == 1 ? taken : now.subtract(const Duration(hours: 5)),
      tempMinC: min,
      tempMaxC: max,
      rainMm: rain,
    );

    await shootScreen(
      tester,
      'weather',
      WeatherScreen(
        now: now,
        onRefresh: () async {},
        weather: Stream.value([
          StopWeather(
            stopId: 1,
            stopName: 'Shillong',
            cachedAt: taken,
            days: [
              day(1, 1, 1, 'Light rain', 17.2, 24.1, 12.4),
              day(2, 1, 2, 'Overcast', 16.9, 22.8, 0),
              day(3, 1, 3, 'Thunderstorm', 16.1, 21.0, 38.6),
            ],
          ),
          StopWeather(
            stopId: 2,
            stopName: 'Cherrapunji',
            cachedAt: now.subtract(const Duration(hours: 5)),
            days: [
              day(4, 2, 3, 'Heavy rain', 15.8, 20.2, 61.0),
              day(5, 2, 4, 'Rain showers', 16.0, 21.4, 24.5),
            ],
          ),
          const StopWeather(stopId: 3, stopName: 'Dawki', days: []),
        ]),
      ),
    );
  });

  testWidgets('settings', (tester) async {
    await shootScreen(
      tester,
      'settings',
      SettingsScreen(
        themeMode: Stream.value(ThemeMode.system),
        onThemeMode: (_) async {},
        corridorKm: Stream.value(3.0),
        onCorridorKm: (_) async {},
        onClearCache: (_) async {},
        onCallHistory: () {},
        // Empty, which is the state the first user was actually in.
        mapKey: Stream.value(''),
        onMapKey: (_) async {},
        caches: Stream.value(const [
          CacheSummary(
            tripId: 1,
            tripName: 'Meghalaya, October',
            poiCount: 87,
            routedLegCount: 3,
            legCount: 4,
            weatherDayCount: 18,
          ),
          CacheSummary(
            tripId: 2,
            tripName: 'Ladakh, next summer',
            poiCount: 0,
            routedLegCount: 0,
            legCount: 6,
            weatherDayCount: 0,
          ),
        ]),
      ),
    );
  });

  testWidgets('call history', (tester) async {
    await shootScreen(
      tester,
      'call_history',
      CallHistoryScreen(
        history: Stream.value([
          CallLogEntry(
            id: 1,
            action: 'dialer',
            occurredAt: DateTime(2026, 10, 3, 18, 42),
            contactName: 'Rina Kharkongor',
            phoneRaw: '+91 90000 00001',
          ),
          CallLogEntry(
            id: 2,
            action: 'copy',
            occurredAt: DateTime(2026, 10, 3, 18, 41),
            contactName: 'Rina Kharkongor',
            phoneRaw: '+91 90000 00001',
          ),
          CallLogEntry(
            id: 3,
            action: 'whatsapp',
            occurredAt: DateTime(2026, 10, 3, 14, 5),
            contactName: 'Biren Lyngdoh',
            phoneRaw: '+91 90000 00002',
          ),
        ]),
      ),
    );
  });

  // ---------------------------------------------------------------------
  // Offline map (#24)
  // ---------------------------------------------------------------------

  // NO DATABASE IN THIS TEST. The first draft opened one and closed it in a
  // tearDown, which is the trap this project recorded at #6: a widget test
  // cannot close a Drift database — close() awaits work the fake clock never
  // advances, and the test hangs until the runner gives up ten minutes later.
  // The screen takes callbacks and a stream, so it never needed one.
  testWidgets('map download', (tester) async {
    await shootScreen(
      tester,
      'map_download',
      MapDownloadScreen(
        provider: const MapTilerRaster(apiKey: 'golden-not-a-real-key'),
        estimate: () async => const MapEstimate(
          legCount: 4,
          legsWithoutCoordinates: 1,
          tileCount: 1284,
          alreadyHave: 412,
        ),
        download: () => const Stream<TileProgress>.empty(),
        usage: Stream.value((count: 412, bytes: 9 * 1024 * 1024)),
        onClear: () async {},
      ),
    );
  });

  testWidgets('map download — no key', (tester) async {
    // The state a build without a key lands in. It must read as a decision,
    // not as a broken screen.
    await shootScreen(
      tester,
      'map_no_key',
      MapDownloadScreen(
        provider: const MapTilerRaster(apiKey: ''),
        estimate: () async => const MapEstimate(
          legCount: 4,
          legsWithoutCoordinates: 0,
          tileCount: 1284,
          alreadyHave: 0,
        ),
        download: () => const Stream<TileProgress>.empty(),
        usage: Stream.value((count: 0, bytes: 0)),
        onClear: () async {},
      ),
    );
  });

  // ---------------------------------------------------------------------
  // The sync orchestrator (#25)
  // ---------------------------------------------------------------------

  testWidgets('sync', (tester) async {
    await shootScreen(
      tester,
      'sync',
      SyncScreen(
        estimateSize: () async => 'about 24 MB, nearly all of it map',
        run: () => const Stream<SyncProgress>.empty(),
        plan: () async => TripSyncPlan(
          legsWithoutCoordinates: 1,
          stopsWithoutCoordinates: 0,
          tasks: [
            for (final leg in const [
              'Shillong → Cherrapunji',
              'Cherrapunji → Shillong',
              'Shillong → Dawki',
            ])
              SyncTask(kind: SyncKind.corridor, subject: leg, legId: 1),
            for (final stop in const [
              'Shillong',
              'Cherrapunji',
              'Dawki',
              'Mawlynnong',
            ])
              SyncTask(kind: SyncKind.weather, subject: stop, stopId: 1),
            const SyncTask(kind: SyncKind.tiles, subject: 'whole trip'),
          ],
        ),
      ),
    );
  });

  testWidgets('sync — partly failed', (tester) async {
    // What a run looks like when Overpass was busy: everything else went
    // through and the screen says which piece did not.
    //
    // This one taps the button rather than using shootScreen, because the
    // failure list only exists after a run — a golden of the untapped screen
    // would show nothing it claims to.
    tester.view.physicalSize = const Size(840, 1780);
    tester.view.devicePixelRatio = 2.0;
    addTearDown(tester.view.reset);

    const failed = SyncTask(
      kind: SyncKind.corridor,
      subject: 'Shillong → Cherrapunji',
      legId: 1,
    );

    await tester.pumpWidget(
      MaterialApp(
        debugShowCheckedModeBanner: false,
        theme: AppTokens.light,
        home: SyncScreen(
          estimateSize: () async => 'the maps are already here',
          plan: () async => const TripSyncPlan(
            legsWithoutCoordinates: 0,
            stopsWithoutCoordinates: 0,
            tasks: [
              failed,
              SyncTask(kind: SyncKind.weather, subject: 'Shillong', stopId: 1),
              SyncTask(kind: SyncKind.tiles, subject: 'whole trip'),
            ],
          ),
          run: () => Stream.fromIterable([
            SyncProgress(
              done: 3,
              total: 3,
              failures: [
                SyncFailure(
                  failed,
                  describeSyncError(
                    'OpenStreetMap is busy right now. Try again in a minute.',
                  ),
                ),
              ],
            ),
          ]),
        ),
      ),
    );
    await tester.pumpAndSettle();
    await tester.tap(find.textContaining('Download everything'));
    await tester.pumpAndSettle();

    await expectLater(
      find.byType(MaterialApp),
      matchesGoldenFile('goldens/sync_failed.png'),
    );
  });

  // ---------------------------------------------------------------------
  // What is along the way (#27, #28)
  // ---------------------------------------------------------------------

  PoiContact osmPhone(int id, String raw) => PoiContact(
    id: id,
    poiId: id,
    phoneRaw: raw,
    phoneE164: raw.replaceAll(' ', ''),
    tier: 'communityOsm',
    sourceTag: 'phone',
  );

  CorridorPlace corridorPlace({
    required int id,
    required String name,
    required String category,
    required double along,
    required double off,
    String? phone,
    Map<String, String> tags = const {},
  }) => CorridorPlace(
    id: id,
    name: name,
    category: category,
    lat: 25.4,
    lon: 91.8,
    alongRouteKm: along,
    offRouteKm: off,
    osmId: 'node/$id',
    phones: phone == null ? const [] : [osmPhone(id, phone)],
    tags: tags,
  );

  testWidgets('discovery', (tester) async {
    await shootScreen(
      tester,
      'discovery',
      DiscoveryScreen(
        onOpen: (_) {},
        discovery: Stream.value(
          LegDiscovery(
            legId: 1,
            fromName: 'Shillong',
            toName: 'Cherrapunji',
            distanceKm: 54,
            lastSyncedAt: DateTime(2026, 9, 20),
            places: [
              corridorPlace(
                id: 1,
                name: 'IOC Umroi',
                category: ContactCategory.fuel,
                along: 6,
                off: 0.1,
                phone: '+91 364 111 1111',
              ),
              corridorPlace(
                id: 2,
                name: 'Mawkdok viewpoint',
                category: ContactCategory.other,
                along: 14,
                off: 0.3,
              ),
              corridorPlace(
                id: 3,
                name: 'Sohra PHC',
                category: ContactCategory.hospital,
                along: 21,
                off: 0.4,
                phone: '+91 364 222 2222',
              ),
              corridorPlace(
                id: 4,
                name: 'Laitlum dhaba',
                category: ContactCategory.restaurant,
                along: 38,
                off: 2.1,
                tags: const {
                  'amenity': 'fast_food',
                  'cuisine': 'indian;momo',
                  'diet:vegetarian': 'yes',
                },
              ),
            ],
          ),
        ),
      ),
    );
  });

  testWidgets('place detail', (tester) async {
    await shootScreen(
      tester,
      'poi_detail',
      PoiDetailScreen(
        place: corridorPlace(
          id: 3,
          name: 'Sohra Primary Health Centre',
          category: ContactCategory.hospital,
          along: 21,
          off: 0.4,
          phone: '+91 364 222 2222',
        ),
        onCopy: (_) async {},
        onOpenDialer: (_) async {},
        onSave: (_) async {},
        onOpenMaps: () async {},
      ),
    );
  });

  // -- the two detail screens, #51 and #52 ---------------------------------

  final shotNow = DateTime(2026, 9, 28, 18);

  WeatherSnapshot forecast(
    int id,
    DateTime date,
    String condition, {
    required DateTime cachedAt,
    double? rain,
    double min = 17,
    double max = 24,
  }) => WeatherSnapshot(
    id: id,
    stopId: 4,
    forDate: date,
    condition: condition,
    tempMinC: min,
    tempMaxC: max,
    rainMm: rain,
    cachedAt: cachedAt,
  );

  StopDetail stopDetail({required DateTime cachedAt, DateTime? synced}) =>
      StopDetail(
        stop: Stop(
          id: 4,
          tripId: 1,
          name: 'Cherrapunji',
          sequenceOrder: 3,
          nights: 2,
          countryCode: 'IN',
          activityTags: 'trek,caves,rain',
          arrivalDate: DateTime(2026, 10, 2),
          lat: 25.2702,
          lon: 91.7323,
        ),
        weather: [
          forecast(
            1,
            DateTime(2026, 10, 2),
            'Heavy rain',
            cachedAt: cachedAt,
            rain: 41,
          ),
          forecast(
            2,
            DateTime(2026, 10, 3),
            'Rain showers',
            cachedAt: cachedAt,
            rain: 12,
            min: 18,
            max: 25,
          ),
          forecast(
            3,
            DateTime(2026, 10, 4),
            'Overcast',
            cachedAt: cachedAt,
            min: 19,
            max: 26,
          ),
        ],
        weatherCachedAt: cachedAt,
        diaryCount: 5,
        unconfirmedCount: 2,
        checklistCount: 14,
        checklistDone: 6,
        nearbyPlaceCount: 9,
        lastSyncedAt: synced,
      );

  testWidgets('stop detail', (tester) async {
    await shootScreen(
      tester,
      'stop_detail',
      StopDetailScreen(
        now: shotNow,
        detail: Stream.value(
          stopDetail(
            cachedAt: shotNow.subtract(const Duration(hours: 5)),
            synced: DateTime(2026, 9, 26),
          ),
        ),
        onEdit: () {},
        onOpenDiary: () {},
        onOpenChecklist: () {},
        onTags: (_) async {},
      ),
    );
  });

  /// The one the whole screen is designed against: a forecast old enough to
  /// mislead. The stamp and the sentence both have to be visible in the image.
  testWidgets('stop detail — stale forecast', (tester) async {
    await shootScreen(
      tester,
      'stop_detail_stale',
      StopDetailScreen(
        now: shotNow,
        detail: Stream.value(
          stopDetail(cachedAt: shotNow.subtract(const Duration(days: 5))),
        ),
        onEdit: () {},
        onOpenDiary: () {},
        onOpenChecklist: () {},
        onTags: (_) async {},
      ),
    );
  });

  testWidgets('leg detail', (tester) async {
    await shootScreen(
      tester,
      'leg_detail',
      LegDetailScreen(
        onEditTransport: () {},
        onSeeAll: () {},
        onOpenPlace: (_) {},
        onOpenContact: (_) {},
        transport: Stream.value(
          LegTransport(
            mode: 'Shared sumo',
            plannedDeparture: DateTime(2026, 10, 2, 7, 30),
            plannedArrival: DateTime(2026, 10, 2, 9, 45),
            note: 'Bara Bazar stand. Ask for the Sohra counter, not Mawsynram.',
          ),
        ),
        discovery: Stream.value(
          LegDiscovery(
            legId: 1,
            fromName: 'Shillong',
            toName: 'Cherrapunji',
            distanceKm: 54,
            lastSyncedAt: DateTime(2026, 9, 26),
            // Your own numbers, as a sheet with coordinates leaves them: one
            // on the road with its source in the note, and the arrival stop
            // led by the hospital. One confirmed, so both trust states show.
            onTheWay: [
              LegContact(
                contact: diaryContact(
                  id: 1,
                  name: 'Laitlyngkot PHC',
                  category: ContactCategory.hospital,
                  phone: '+91 90000 00011',
                  note: '24x7. Govt source: East Khasi Hills District',
                ),
                alongRouteKm: 22,
                offRouteKm: 0.2,
              ),
            ],
            atDestination: [
              diaryContact(
                id: 2,
                name: 'Sohra Community Health Centre',
                category: ContactCategory.hospital,
                phone: '+91 90000 00012',
                note: 'Nearest govt facility for Nohkalikai and Seven Sisters.',
                confirmed: true,
              ),
              diaryContact(
                id: 3,
                name: 'Cherrapunjee Holiday Resort',
                category: ContactCategory.accommodation,
                phone: '+91 90000 00013',
                note: 'Google listing only — not call-tested.',
              ),
            ],
            places: [
              corridorPlace(
                id: 1,
                name: 'IOC Umroi',
                category: ContactCategory.fuel,
                along: 6,
                off: 0.1,
                phone: '+91 364 111 1111',
              ),
              corridorPlace(
                id: 2,
                name: 'Mawkdok Dympep viewpoint',
                category: ContactCategory.other,
                along: 14,
                off: 0.3,
              ),
              corridorPlace(
                id: 3,
                name: 'Sohra PHC',
                category: ContactCategory.hospital,
                along: 21,
                off: 0.4,
                phone: '+91 364 222 2222',
              ),
              corridorPlace(
                id: 4,
                name: 'Laitlum dhaba',
                category: ContactCategory.restaurant,
                along: 38,
                off: 2.1,
                tags: const {
                  'amenity': 'fast_food',
                  'cuisine': 'indian;momo',
                  'diet:vegetarian': 'yes',
                },
              ),
              corridorPlace(
                id: 5,
                name: 'Nohkalikai turning',
                category: ContactCategory.other,
                along: 47,
                off: 0.2,
              ),
              corridorPlace(
                id: 6,
                name: 'Sohra market',
                category: ContactCategory.other,
                along: 53,
                off: 0.1,
              ),
            ],
          ),
        ),
      ),
    );
  });

  // -- multi-add, #10 -------------------------------------------------------

  testWidgets('multi add', (tester) async {
    // Typed in for real rather than mocked into place, so the image is of the
    // screen somebody would actually be looking at part way through: two rows
    // down, a duplicate caught, one half-finished, and a blank waiting.
    tester.view.physicalSize = const Size(840, 1780);
    tester.view.devicePixelRatio = 2.0;
    addTearDown(tester.view.reset);

    await tester.pumpWidget(
      MaterialApp(
        debugShowCheckedModeBanner: false,
        theme: AppTokens.light,
        home: MultiAddScreen(
          check: (row) async => row.copyWith(
            phoneE164: '+91${row.phoneRaw.replaceAll(RegExp(r"[^0-9]"), "")}',
            duplicateOf: row.name.contains('Wanshai') ? 'Wanshai' : null,
          ),
          onSave: (_) async => const MultiAddResult(added: 0, skipped: 0),
        ),
      ),
    );
    await tester.pump();

    Future<void> type(int row, String name, String phone) async {
      await tester.enterText(find.byType(TextField).at(row * 2), name);
      if (phone.isNotEmpty) {
        await tester.enterText(find.byType(TextField).at(row * 2 + 1), phone);
      }
      await tester.pumpAndSettle();
    }

    await type(0, 'Kongthong homestay', '+91 90000 00001');
    await type(1, 'Wanshai · driver', '+91 90000 00002');
    await type(2, 'Sohra chemist', '');

    await expectLater(
      find.byType(MaterialApp),
      matchesGoldenFile('goldens/multi_add.png'),
    );
  });

  // -- backup, #56 ----------------------------------------------------------

  testWidgets('backup — a file chosen, waiting to be restored', (tester) async {
    // The state worth looking at is after a file has been picked, so the
    // golden taps the real button rather than mocking the screen into place.
    final picked = BackupContents(
      createdAt: DateTime.utc(2026, 9, 28, 14, 30),
      schemaVersion: 4,
      trips: 1,
      stops: 6,
      contacts: 23,
      confirmedContacts: 17,
      checklistItems: 21,
      expenses: 9,
      callLogs: 34,
      tables: const {},
    );
    final current = BackupContents(
      createdAt: DateTime.utc(2026, 10, 1),
      schemaVersion: 4,
      trips: 1,
      stops: 2,
      contacts: 4,
      confirmedContacts: 0,
      checklistItems: 3,
      expenses: 0,
      callLogs: 1,
      tables: const {},
    );

    tester.view.physicalSize = const Size(840, 1780);
    tester.view.devicePixelRatio = 2.0;
    addTearDown(tester.view.reset);

    await tester.pumpWidget(
      MaterialApp(
        debugShowCheckedModeBanner: false,
        theme: AppTokens.light,
        home: BackupScreen(
          onExport: () async => null,
          onPick: () async => picked,
          onCurrent: () async => current,
          onRestore: (_) async {},
        ),
      ),
    );
    await tester.pumpAndSettle();

    await tester.tap(find.text('CHOOSE A BACKUP FILE'));
    await tester.pumpAndSettle();

    await expectLater(
      find.byType(MaterialApp),
      matchesGoldenFile('goldens/backup.png'),
    );
  });
}

/// A diary row for the goldens, with everything the table requires filled.
Contact diaryContact({
  required int id,
  required String name,
  required String category,
  required String phone,
  String? note,
  bool confirmed = false,
}) => Contact(
  id: id,
  name: name,
  phoneRaw: phone,
  category: category,
  tier: confirmed ? 'userVerified' : 'userEntered',
  callConfirmed: confirmed,
  isPinned: false,
  isEmergency: false,
  hasWhatsapp: false,
  callCount: 0,
  createdAt: DateTime(2026, 9, 22),
  note: note,
);

/// A location source for the goldens that never finds anything.
class _NoLocation implements LocationSource {
  const _NoLocation();
  @override
  Future<HereState> check({bool ask = false}) async => HereState.notAsked;
  @override
  Stream<HereFix> watch() => const Stream.empty();
  @override
  Future<HereFix?> once({Duration timeout = Duration.zero}) async => null;
  @override
  Future<void> openAppSettings() async {}
  @override
  Future<void> openLocationSettings() async {}
}
