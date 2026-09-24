// test/stay_test.dart — where you actually sleep at each stop.
//
// "It defaults to Bramhome Guest House, in which I won't be staying. I will
// be staying in different places — from where can I mention that?"
//
// Saved is not chosen. These tests pin that a stop's stay is the person's
// choice, made from the stop, the Trip page or the guest house's own page,
// and that nothing picks one for them from what happens to be saved.

import 'dart:convert';

import 'package:drift/drift.dart' hide isNull, isNotNull;
import 'package:drift/native.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:safarsathi/core/database/app_database.dart';
import 'package:safarsathi/core/theme/app_tokens.dart';
import 'package:safarsathi/features/backup/data/backup.dart';
import 'package:safarsathi/features/contacts/data/contacts_dao.dart';
import 'package:safarsathi/features/contacts/presentation/entry_screen.dart';
import 'package:safarsathi/features/trips/data/stay.dart';
import 'package:safarsathi/features/trips/data/stop_detail.dart';
import 'package:safarsathi/features/trips/data/trip_editor.dart';
import 'package:safarsathi/features/trips/presentation/stay_picker_sheet.dart';
import 'package:safarsathi/features/trips/presentation/stop_detail_screen.dart';

Contact _contact(int id, String name, {bool confirmed = false}) => Contact(
  id: id,
  name: name,
  phoneRaw: '+91 90000 0000$id',
  category: ContactCategory.accommodation,
  tier: confirmed ? 'userVerified' : 'userEntered',
  callConfirmed: confirmed,
  isPinned: false,
  isEmergency: false,
  hasWhatsapp: false,
  callCount: 0,
  createdAt: DateTime(2026, 9, 20),
  stopId: 1,
);

Stop _stop({int nights = 2, int? stayId}) => Stop(
  id: 1,
  tripId: 1,
  name: 'Shillong',
  sequenceOrder: 1,
  nights: nights,
  countryCode: 'IN',
  activityTags: '',
  stayContactId: stayId,
);

void main() {
  group('the data', () {
    late AppDatabase db;
    late TripEditor editor;
    late int tripId, shillong;

    setUp(() async {
      db = AppDatabase(NativeDatabase.memory());
      editor = TripEditor(db);
      tripId = await editor.createTrip(name: 'Meghalaya');
      shillong = await editor.addStop(
        tripId,
        const StopDraft(name: 'Shillong', nights: 2),
      );
    });
    tearDown(() => db.close());

    Future<int> stay(String name, {int? batch, int? stopId}) =>
        db.into(db.contacts).insert(
          ContactsCompanion.insert(
            tripId: Value(tripId),
            stopId: Value(stopId ?? shillong),
            name: name,
            phoneRaw: '+91 90000 00001',
            category: const Value(ContactCategory.accommodation),
            importBatchId: Value(batch),
          ),
        );

    Future<Stop> reload() => (db.select(
      db.stops,
    )..where((s) => s.id.equals(shillong))).getSingle();

    test('a chosen stay is found; a deleted one reads as not decided', () async {
      final id = await stay('J P Guest House');
      await setStay(db, shillong, id);
      final diary = await db.select(db.contacts).get();
      expect(chosenStay(await reload(), diary)?.name, 'J P Guest House');

      await db.contactsDao.deleteContact(id);
      expect(chosenStay(await reload(), await db.select(db.contacts).get()),
          isNull);
    });

    test('clearing goes back to not decided', () async {
      await setStay(db, shillong, await stay('J P'));
      await setStay(db, shillong, null);
      expect((await reload()).stayContactId, isNull);
    });

    test('THE FIRST STAY YOU TYPE AT A STOP IS YOUR STAY', () async {
      final id = await stay('Our homestay');
      expect(await adoptFirstStay(db, id), isTrue);
      expect((await reload()).stayContactId, id);
    });

    test('AN IMPORTED OPTION NEVER CHOOSES ITSELF', () async {
      final batch = await db.into(db.importBatches).insert(
        ImportBatchesCompanion.insert(fileName: 'contacts.xlsx'),
      );
      final id = await stay('Bramhome Guest House', batch: batch);
      expect(await adoptFirstStay(db, id), isFalse);
      expect((await reload()).stayContactId, isNull);
    });

    test('a second typed stay does not replace a choice, or make one', () async {
      final first = await stay('First');
      await stay('Second');
      expect(await adoptFirstStay(db, first), isFalse,
          reason: 'two saved: which one is a question');
      await setStay(db, shillong, first);
      final third = await stay('Third');
      expect(await adoptFirstStay(db, third), isFalse);
      expect((await reload()).stayContactId, first);
    });

    test('a stop you pass through has no stay to adopt', () async {
      final dawki = await editor.addStop(
        tripId,
        const StopDraft(name: 'Dawki'),
      );
      final id = await stay('Boat guy', stopId: dawki);
      expect(await adoptFirstStay(db, id), isFalse);
    });

    test('the stop page reads the stay and counts the options', () async {
      await stay('Bramhome');
      final mine = await stay('J P');
      var detail = await watchStopDetail(db, shillong).first;
      expect(detail.stay, isNull);
      expect(detail.stayOptionCount, 2);

      await setStay(db, shillong, mine);
      detail = await watchStopDetail(db, shillong).first;
      expect(detail.stay?.name, 'J P');
    });

    test('A BACKUP KEEPS THE CHOICE, and restores though stops come before '
        'contacts', () async {
      // Stops.stayContactId has no foreign key for exactly this: a backup
      // writes stops first, and a reference would fail every restore.
      final mine = await stay('J P');
      await setStay(db, shillong, mine);
      final json = await exportBackup(db);

      final fresh = AppDatabase(NativeDatabase.memory());
      addTearDown(fresh.close);
      await restoreBackup(fresh, readBackup(json));
      final stop = await (fresh.select(
        fresh.stops,
      )..where((s) => s.id.equals(shillong))).getSingle();
      expect(stop.stayContactId, mine);
    });

    test('a v5 backup, which has no stay at all, still restores', () async {
      final json = jsonDecode(await exportBackup(db)) as Map<String, dynamic>;
      json['schemaVersion'] = 5;
      for (final s in (json['tables']['stops'] as List)) {
        (s as Map).remove('stayContactId');
      }

      final fresh = AppDatabase(NativeDatabase.memory());
      addTearDown(fresh.close);
      await restoreBackup(fresh, readBackup(jsonEncode(json)));
      expect((await fresh.select(fresh.stops).getSingle()).stayContactId,
          isNull);
    });
  });

  group('the picker', () {
    late StayChoice? chosen;

    Future<void> pump(WidgetTester tester, {int? current}) async {
      chosen = null;
      tester.view.physicalSize = const Size(420, 1200);
      tester.view.devicePixelRatio = 1.0;
      addTearDown(tester.view.reset);
      await tester.pumpWidget(
        MaterialApp(
          theme: AppTokens.light,
          home: Scaffold(
            body: StayPickerList(
              stopName: 'Shillong',
              options: [
                _contact(1, 'Bramhome Guest House', confirmed: true),
                _contact(2, 'J P Guest House'),
              ],
              currentId: current,
              onChoice: (c) => chosen = c,
            ),
          ),
        ),
      );
    }

    testWidgets('lists what is saved and says what choosing means', (
      tester,
    ) async {
      await pump(tester);
      expect(find.text('STAYING IN SHILLONG'), findsOneWidget);
      expect(find.text('Bramhome Guest House'), findsOneWidget);
      expect(find.text('J P Guest House'), findsOneWidget);
      expect(find.textContaining('has to be confirmed'), findsOneWidget);
      // Nothing chosen, nothing to clear.
      expect(find.byKey(const Key('stay-clear')), findsNothing);
    });

    testWidgets('tapping one picks it', (tester) async {
      await pump(tester);
      await tester.tap(find.text('J P Guest House'));
      expect((chosen as StayPicked).contactId, 2);
    });

    testWidgets('the current one is marked, and can be cleared', (
      tester,
    ) async {
      await pump(tester, current: 2);
      final jp = find.byKey(const Key('stay-option-2'));
      expect(
        find.descendant(
          of: jp,
          matching: find.byIcon(Icons.radio_button_checked),
        ),
        findsOneWidget,
      );
      await tester.tap(find.byKey(const Key('stay-clear')));
      expect(chosen, isA<StayCleared>());
    });

    testWidgets('somewhere not saved yet can be added', (tester) async {
      await pump(tester);
      await tester.tap(find.byKey(const Key('stay-add-new')));
      expect(chosen, isA<StayAddNew>());
    });
  });

  group('the stop page', () {
    Future<void> pump(
      WidgetTester tester, {
      Contact? stay,
      int options = 2,
      int nights = 2,
      VoidCallback? onChoose,
    }) async {
      tester.view.physicalSize = const Size(420, 2400);
      tester.view.devicePixelRatio = 1.0;
      addTearDown(tester.view.reset);
      await tester.pumpWidget(
        MaterialApp(
          theme: AppTokens.light,
          home: StopDetailScreen(
            detail: Stream.value(
              StopDetail(
                stop: _stop(nights: nights, stayId: stay?.id),
                weather: const [],
                diaryCount: 2,
                unconfirmedCount: 1,
                checklistCount: 0,
                checklistDone: 0,
                nearbyPlaceCount: 0,
                stay: stay,
                stayOptionCount: options,
              ),
            ),
            onChooseStay: onChoose ?? () {},
          ),
        ),
      );
      await tester.pump();
    }

    testWidgets('not decided says so, with how many are saved', (
      tester,
    ) async {
      var tapped = 0;
      await pump(tester, onChoose: () => tapped++);
      expect(find.text('STAYING AT'), findsOneWidget);
      expect(find.textContaining('Not decided. 2 places are saved here'),
          findsOneWidget);
      await tester.tap(find.byKey(const Key('stop-choose-stay')));
      expect(tapped, 1);
    });

    testWidgets('a chosen stay shows its number and whether it is confirmed', (
      tester,
    ) async {
      await pump(tester, stay: _contact(2, 'J P Guest House'));
      expect(find.text('J P Guest House'), findsOneWidget);
      expect(find.text('+91 90000 00002'), findsOneWidget);
      expect(find.textContaining('call it before you go'), findsOneWidget);
      expect(find.text('CHANGE'), findsOneWidget);
    });

    testWidgets('a stop you pass through has no "staying at"', (tester) async {
      await pump(tester, nights: 0);
      expect(find.text('STAYING AT'), findsNothing);
    });
  });

  group('the guest house page', () {
    testWidgets('"I am staying here" makes it the stay, and undoes', (
      tester,
    ) async {
      final calls = <bool>[];
      Future<void> pump(bool isStay) async {
        tester.view.physicalSize = const Size(420, 1600);
        tester.view.devicePixelRatio = 1.0;
        addTearDown(tester.view.reset);
        await tester.pumpWidget(
          MaterialApp(
            theme: AppTokens.light,
            home: EntryScreen(
              contact: _contact(2, 'J P Guest House'),
              stopName: 'Shillong',
              stay: (stopName: 'Shillong', isStay: isStay),
              onStayHere: (v) async => calls.add(v),
            ),
          ),
        );
        await tester.pump();
      }

      await pump(false);
      expect(find.text('I am staying here in Shillong'), findsOneWidget);
      await tester.tap(find.byKey(const Key('entry-stay-here')));
      await tester.pump();

      await pump(true);
      expect(find.textContaining('Where you are staying in Shillong'),
          findsOneWidget);
      await tester.tap(find.byKey(const Key('entry-stay-here')));
      await tester.pump();
      expect(calls, [true, false]);
    });
  });
}
