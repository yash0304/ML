// Issue #8 — the entry form.
//
// The rules that matter here are the ones about what the app refuses to do:
// it will not save a nameless or numberless entry, it will not block on a
// duplicate, and it will not let a confirmation survive a change of digits.

import 'package:drift/native.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:safarsathi/core/database/app_database.dart';
import 'package:safarsathi/core/theme/app_tokens.dart';
import 'package:safarsathi/features/contacts/data/contacts_dao.dart';
import 'package:safarsathi/features/contacts/data/entry_draft.dart';
import 'package:safarsathi/features/contacts/presentation/entry_form_screen.dart';

Contact existingEntry({
  String name = 'Kongthong homestay',
  String phoneRaw = '+91 98560 41122',
  String? phoneE164 = '+919856041122',
  bool confirmed = false,
}) {
  return Contact(
    id: 42,
    name: name,
    phoneRaw: phoneRaw,
    phoneE164: phoneE164,
    category: ContactCategory.accommodation,
    tier: confirmed
        ? ContactTier.userVerified.name
        : ContactTier.userEntered.name,
    callConfirmed: confirmed,
    isPinned: false,
    isEmergency: false,
    hasWhatsapp: false,
    callCount: 0,
    createdAt: DateTime(2026, 9, 12),
  );
}

void main() {
  group('the form', () {
    late List<EntryDraft> saved;

    Future<void> pumpForm(
      WidgetTester tester, {
      Contact? existing,
      Contact? duplicate,
      List<StopOption> stops = const [],
    }) async {
      saved = [];
      // The default 800x600 surface is shorter than the form; the save
      // button lands off-screen and taps miss it.
      tester.view.physicalSize = const Size(420, 1400);
      tester.view.devicePixelRatio = 1.0;
      addTearDown(tester.view.reset);

      await tester.pumpWidget(
        MaterialApp(
          theme: AppTokens.light,
          home: EntryFormScreen(
            existing: existing,
            stops: stops,
            findDuplicate: (_) async => duplicate,
            onSave: (d) async => saved.add(d),
          ),
        ),
      );
      await tester.pump();
    }

    Future<void> save(WidgetTester tester, String label) async {
      await tester.ensureVisible(find.text(label));
      await tester.pump();
      await tester.tap(find.text(label));
      await tester.pump();
    }

    Future<void> type(WidgetTester tester, String field, String text) async {
      await tester.enterText(
        find.descendant(
          of: find.byKey(Key('field-${field.toLowerCase()}')),
          matching: find.byType(TextField),
        ),
        text,
      );
      await tester.pump();
      await tester.pump();
    }

    testWidgets('refuses an entry with no name', (tester) async {
      await pumpForm(tester);
      await type(tester, 'Number', '+91 98560 41122');
      await save(tester, 'Save to diary');

      expect(find.text('An entry needs a name.'), findsOneWidget);
      expect(saved, isEmpty);
    });

    testWidgets('refuses an entry with no number', (tester) async {
      await pumpForm(tester);
      await type(tester, 'Name', 'Homestay');
      await save(tester, 'Save to diary');

      expect(find.text('An entry needs a number.'), findsOneWidget);
      expect(saved, isEmpty);
    });

    testWidgets('saves both the raw and the normalised number', (tester) async {
      await pumpForm(tester);
      await type(tester, 'Name', 'Kongthong homestay');
      await type(tester, 'Number', '98560 41122');
      await save(tester, 'Save to diary');

      expect(saved.single.phoneRaw, '98560 41122');
      expect(saved.single.phoneE164, '+919856041122');
    });

    testWidgets('an unreadable number warns but still saves', (tester) async {
      await pumpForm(tester);
      await type(tester, 'Name', 'Sohra tea stall');
      await type(tester, 'Number', '9856');

      expect(
        find.textContaining('does not look like a complete'),
        findsOneWidget,
      );

      await save(tester, 'Save to diary');

      expect(saved, hasLength(1));
      expect(saved.single.phoneRaw, '9856');
      expect(saved.single.phoneE164, isNull);
    });

    testWidgets('a duplicate is a warning, not a block', (tester) async {
      await pumpForm(tester, duplicate: existingEntry(name: 'Bah Rothell'));
      await type(tester, 'Name', 'Kongthong homestay');
      await type(tester, 'Number', '+91 98560 41122');

      expect(find.textContaining('Already in your diary'), findsOneWidget);
      expect(find.textContaining('Bah Rothell'), findsOneWidget);

      await save(tester, 'Save to diary');
      expect(saved, hasLength(1));
    });

    testWidgets('editing an entry is not a duplicate of itself', (
      tester,
    ) async {
      final entry = existingEntry();
      await pumpForm(tester, existing: entry, duplicate: entry);
      await type(tester, 'Number', '+91 98560 41122');

      expect(find.textContaining('Already in your diary'), findsNothing);
    });

    testWidgets('says plainly that a new entry lands unconfirmed', (
      tester,
    ) async {
      await pumpForm(tester);
      expect(find.textContaining('As unconfirmed'), findsOneWidget);
      expect(find.textContaining('does not make it work'), findsOneWidget);
    });

    testWidgets('a confirmed entry keeps its confirmation when only the '
        'name changes', (tester) async {
      await pumpForm(tester, existing: existingEntry(confirmed: true));
      await type(tester, 'Name', 'Kongthong homestay, Bah Rothell');

      expect(find.textContaining('Stays confirmed'), findsOneWidget);

      await save(tester, 'Save changes');
      expect(saved.single.resetConfirmation, isFalse);
    });

    testWidgets('changing the digits clears the confirmation, and says so', (
      tester,
    ) async {
      // A confirmation means "I called THIS number and it worked". Change the
      // digits and that is no longer true.
      await pumpForm(tester, existing: existingEntry(confirmed: true));
      await type(tester, 'Number', '+91 98560 49999');

      expect(
        find.textContaining('Changing the digits clears that'),
        findsOneWidget,
      );

      await save(tester, 'Save changes');
      expect(saved.single.resetConfirmation, isTrue);
    });

    testWidgets('a stop can be chosen, and whole-trip is the default', (
      tester,
    ) async {
      await pumpForm(
        tester,
        stops: const [StopOption(7, 'Kongthong'), StopOption(9, 'Shillong')],
      );
      await type(tester, 'Name', 'Homestay');
      await type(tester, 'Number', '+91 98560 41122');

      await tester.tap(find.text('KONGTHONG'));
      await tester.pump();
      await save(tester, 'Save to diary');

      expect(saved.single.stopId, 7);
    });
  });

  group('writing a draft', () {
    late AppDatabase db;
    late ContactsDao dao;
    late int tripId;

    setUp(() async {
      db = AppDatabase(NativeDatabase.memory());
      dao = db.contactsDao;
      tripId = await db
          .into(db.trips)
          .insert(TripsCompanion.insert(name: 'Meghalaya'));
    });

    tearDown(() async => db.close());

    test('a new entry always lands unconfirmed', () async {
      await saveEntry(
        dao,
        const EntryDraft(
          name: 'Homestay',
          phoneRaw: '+91 98560 41122',
          phoneE164: '+919856041122',
        ),
        tripId: tripId,
      );

      final row = await db.select(db.contacts).getSingle();
      expect(row.tier, ContactTier.userEntered.name);
      expect(row.callConfirmed, isFalse);
      expect(row.confirmedAt, isNull);
    });

    test('an edit that keeps the number keeps the confirmation', () async {
      final id = await saveEntry(
        dao,
        const EntryDraft(name: 'Homestay', phoneRaw: '+91 98560 41122'),
        tripId: tripId,
      );
      await dao.markConfirmed(id);

      await saveEntry(
        dao,
        EntryDraft(
          id: id,
          name: 'Homestay, Bah Rothell',
          phoneRaw: '+91 98560 41122',
        ),
        tripId: tripId,
      );

      final row = await db.select(db.contacts).getSingle();
      expect(row.name, 'Homestay, Bah Rothell');
      expect(row.callConfirmed, isTrue);
      expect(row.tier, ContactTier.userVerified.name);
    });

    test('an edit that changes the number drops the confirmation', () async {
      final id = await saveEntry(
        dao,
        const EntryDraft(name: 'Homestay', phoneRaw: '+91 98560 41122'),
        tripId: tripId,
      );
      await dao.markConfirmed(id);

      await saveEntry(
        dao,
        EntryDraft(
          id: id,
          name: 'Homestay',
          phoneRaw: '+91 98560 49999',
          resetConfirmation: true,
        ),
        tripId: tripId,
      );

      final row = await db.select(db.contacts).getSingle();
      expect(row.callConfirmed, isFalse);
      expect(row.tier, ContactTier.userEntered.name);
      expect(row.confirmedAt, isNull);
      expect(await dao.watchUnconfirmedCount(tripId).first, 1);
    });

    test('an empty note is stored as nothing, not as blank text', () async {
      await saveEntry(
        dao,
        const EntryDraft(
          name: 'Homestay',
          phoneRaw: '+91 98560 41122',
          note: '   ',
        ),
        tripId: tripId,
      );
      expect((await db.select(db.contacts).getSingle()).note, isNull);
    });
  });
}
