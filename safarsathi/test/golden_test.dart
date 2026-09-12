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
}
