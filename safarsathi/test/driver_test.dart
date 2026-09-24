// test/driver_test.dart — who is taking you from one stop to the next.
//
// "Who is taking us from one stop to another — hope that can also be there,
// like taxi or transport or something."
//
// A leg now knows its driver (a diary entry) and the vehicle's number. Both
// go in the plan sent home, and in an SOS text on a travel day.

import 'dart:convert';
import 'dart:io';

import 'package:drift/drift.dart' hide isNull, isNotNull;
import 'package:drift/native.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:safarsathi/core/database/app_database.dart';
import 'package:safarsathi/core/theme/app_tokens.dart';
import 'package:safarsathi/features/backup/data/backup.dart';
import 'package:safarsathi/features/contacts/data/contacts_dao.dart';
import 'package:safarsathi/features/emergency/data/sos.dart';
import 'package:safarsathi/features/trips/data/driver.dart';
import 'package:safarsathi/features/trips/data/plan_message.dart';
import 'package:safarsathi/features/trips/data/trip_editor.dart';
import 'package:safarsathi/features/trips/presentation/leg_detail_screen.dart';
import 'package:safarsathi/features/trips/presentation/leg_form_screen.dart';
import 'package:safarsathi/features/discovery/data/discovery.dart';

Contact person(int id, String name, String category) => Contact(
  id: id,
  name: name,
  phoneRaw: '+91 98560 1234$id',
  category: category,
  tier: 'userEntered',
  callConfirmed: false,
  isPinned: false,
  isEmergency: false,
  hasWhatsapp: false,
  callCount: 0,
  createdAt: DateTime(2026, 9, 20),
);

Leg leg(int id, {DateTime? departs, int? driver}) => Leg(
  id: id,
  tripId: 1,
  fromStopId: id,
  toStopId: id + 1,
  sequenceOrder: id,
  isBooked: false,
  corridorKm: 3,
  plannedDeparture: departs,
  driverContactId: driver,
);

void main() {
  group('choosing', () {
    test('transport numbers come first, then local contacts, nothing else',
        () {
      final options = driverOptions([
        person(1, 'Zed Homestay', ContactCategory.accommodation),
        person(2, 'Bah Kyn (fixer)', ContactCategory.localContact),
        person(3, 'Anthony', ContactCategory.transport),
        person(4, 'Civil Hospital', ContactCategory.hospital),
      ]);
      expect(options.map((c) => c.name), ['Anthony', 'Bah Kyn (fixer)']);
    });

    test('a deleted driver reads as nobody chosen', () {
      expect(driverOf(leg(1, driver: 9), [person(3, 'Anthony', 'transport')]),
          isNull);
      expect(
        driverOf(leg(1, driver: 3), [person(3, 'Anthony', 'transport')])?.name,
        'Anthony',
      );
    });
  });

  group('today\'s leg', () {
    final legs = [
      leg(1, departs: DateTime(2026, 10, 2, 8)),
      leg(2, departs: DateTime(2026, 10, 2, 15)),
      leg(3, departs: DateTime(2026, 10, 3, 9)),
    ];

    test('before setting off, the first of the day', () {
      expect(legToday(legs, now: DateTime(2026, 10, 2, 6))?.id, 1);
    });

    test('once moving, the latest leg that has left', () {
      expect(legToday(legs, now: DateTime(2026, 10, 2, 16))?.id, 2);
    });

    test('a day with no leg has none', () {
      expect(legToday(legs, now: DateTime(2026, 10, 4, 12)), isNull);
    });
  });

  group('the line', () {
    test('whatever is known, in one line', () {
      final anthony = person(3, 'Anthony', 'transport');
      expect(
        rideLine(mode: 'Taxi', vehicle: 'ML 05 A 1234', driver: anthony),
        'Taxi ML 05 A 1234 · driver Anthony, +91 98560 12343',
      );
      expect(rideLine(mode: 'Taxi'), 'Taxi');
      expect(rideLine(driver: anthony), 'driver Anthony, +91 98560 12343');
      expect(rideLine(), isNull);
    });

    test('THE SOS TEXT SAYS WHAT YOU ARE TRAVELLING IN', () {
      final msg = sosMessage(
        nearStop: 'Sohra',
        ride: 'Taxi ML 05 A 1234 · driver Anthony, +91 98560 12343',
      );
      expect(
        msg,
        contains('Travelling: Taxi ML 05 A 1234 · driver Anthony, '
            '+91 98560 12343\nSent from SafarSathi.'),
      );
      expect(sosMessage(nearStop: 'Sohra'), isNot(contains('Travelling')));
    });
  });

  group('the database', () {
    late AppDatabase db;
    late TripEditor editor;
    late int tripId;

    setUp(() async {
      db = AppDatabase(NativeDatabase.memory());
      editor = TripEditor(db);
      tripId = await editor.createTrip(name: 'Meghalaya');
      for (final (i, n) in ['Guwahati', 'Shillong', 'Sohra', 'Dawki'].indexed) {
        await editor.addStop(
          tripId,
          StopDraft(name: n, nights: i == 3 ? 0 : 1),
        );
      }
    });
    tearDown(() => db.close());

    Future<int> contact(String name) => db.into(db.contacts).insert(
      ContactsCompanion.insert(
        tripId: Value(tripId),
        name: name,
        phoneRaw: '+91 98560 12345',
        category: const Value(ContactCategory.transport),
      ),
    );

    Future<List<Leg>> legs() => (db.select(db.legs)
          ..orderBy([(l) => OrderingTerm(expression: l.sequenceOrder)]))
        .get();

    test('THE WHOLE TRIP FILLS ONLY THE LEGS WITH NOBODY', () async {
      final anthony = await contact('Anthony');
      final boatman = await contact('Dawki boatman');
      final all = await legs();
      await setDriver(db, all.last.id, boatman);
      await setDriver(db, all.first.id, anthony);

      expect(
        await legsWithoutDriver(db, tripId, exceptLegId: all.first.id),
        hasLength(1),
      );
      expect(await setDriverWhereMissing(db, tripId, anthony), 1);
      final after = await legs();
      expect(after.map((l) => l.driverContactId),
          [anthony, anthony, boatman]);
    });

    test('THE PLAN SENT HOME SAYS WHO IS DRIVING AND IN WHAT', () async {
      final anthony = await contact('Anthony');
      final first = (await legs()).first;
      await editor.updateLeg(
        first.id,
        mode: 'Taxi',
        isBooked: true,
        vehicleNumber: 'ML 05 A 1234',
      );
      await setDriver(db, first.id, anthony);

      final msg = await buildPlanMessage(db, tripId);
      expect(
        msg,
        contains('Guwahati → Shillong · Taxi ML 05 A 1234\n'
            '  Driver: Anthony, +91 98560 12345'),
      );
    });

    test('the leg screen reads the driver and the vehicle', () async {
      final anthony = await contact('Anthony');
      final first = (await legs()).first;
      await editor.updateLeg(first.id, isBooked: false,
          vehicleNumber: 'ML 05 A 1234');
      await setDriver(db, first.id, anthony);

      final t = await watchLegTransport(db, first.id).first;
      expect(t.driver?.name, 'Anthony');
      expect(t.vehicleNumber, 'ML 05 A 1234');
      expect(t.isEmpty, isFalse);
    });

    test('a backup keeps both; a v7 backup without them restores', () async {
      final anthony = await contact('Anthony');
      final first = (await legs()).first;
      await editor.updateLeg(first.id, isBooked: false,
          vehicleNumber: 'ML 05 A 1234');
      await setDriver(db, first.id, anthony);

      final fresh = AppDatabase(NativeDatabase.memory());
      addTearDown(fresh.close);
      await restoreBackup(fresh, readBackup(await exportBackup(db)));
      final back = await (fresh.select(
        fresh.legs,
      )..where((l) => l.id.equals(first.id))).getSingle();
      expect(back.driverContactId, anthony);
      expect(back.vehicleNumber, 'ML 05 A 1234');

      final json = jsonDecode(await exportBackup(db)) as Map<String, dynamic>;
      json['schemaVersion'] = 7;
      for (final l in (json['tables']['legs'] as List)) {
        (l as Map)
          ..remove('driverContactId')
          ..remove('vehicleNumber');
      }
      final older = AppDatabase(NativeDatabase.memory());
      addTearDown(older.close);
      await restoreBackup(older, readBackup(jsonEncode(json)));
      expect((await older.select(older.legs).get()).first.driverContactId,
          isNull);
    });
  });

  test('A REAL v7 DATABASE UPGRADES TO v8 with its legs intact', () async {
    final dir = await Directory.systemTemp.createTemp('upgrade8');
    addTearDown(() => dir.delete(recursive: true));
    final file = File('${dir.path}/v7.sqlite');

    final v7 = AppDatabase(NativeDatabase(file));
    final editor = TripEditor(v7);
    final trip = await editor.createTrip(name: 'Meghalaya');
    await editor.addStop(trip, const StopDraft(name: 'Shillong'));
    await editor.addStop(trip, const StopDraft(name: 'Sohra'));
    await editor.updateLeg(
      (await v7.select(v7.legs).getSingle()).id,
      mode: 'Taxi',
      isBooked: true,
    );
    await v7.customStatement('ALTER TABLE legs DROP COLUMN driver_contact_id');
    await v7.customStatement('ALTER TABLE legs DROP COLUMN vehicle_number');
    await v7.customStatement('PRAGMA user_version = 7');
    await v7.close();

    final v8 = AppDatabase(NativeDatabase(file));
    addTearDown(v8.close);
    final l = await v8.select(v8.legs).getSingle();
    expect(l.mode, 'Taxi');
    expect(l.isBooked, isTrue);
    expect(l.driverContactId, isNull);
    expect(l.vehicleNumber, isNull);
    final version = await v8.customSelect('PRAGMA user_version').getSingle();
    expect(version.read<int>('user_version'), 8);
  });

  group('the screens', () {
    void tall(WidgetTester tester) {
      tester.view.physicalSize = const Size(420, 2000);
      tester.view.devicePixelRatio = 1.0;
      addTearDown(tester.view.reset);
    }

    Future<void> pumpLeg(
      WidgetTester tester,
      LegTransport t, {
      VoidCallback? onChoose,
      void Function(Contact)? onOpen,
    }) async {
      tall(tester);
      await tester.pumpWidget(
        MaterialApp(
          theme: AppTokens.light,
          home: LegDetailScreen(
            discovery: Stream.value(
              const LegDiscovery(
                legId: 1,
                fromName: 'Shillong',
                toName: 'Sohra',
                places: [],
              ),
            ),
            transport: Stream.value(t),
            onChooseDriver: onChoose,
            onOpenDriver: onOpen,
          ),
        ),
      );
      // Two streams: the leg, then its transport.
      await tester.pump();
      await tester.pump();
    }

    testWidgets('nobody chosen asks who is taking you', (tester) async {
      var chose = 0;
      await pumpLeg(tester, const LegTransport(), onChoose: () => chose++);
      expect(find.textContaining('Who is taking you?'), findsOneWidget);
      await tester.tap(find.byKey(const Key('leg-choose-driver')));
      expect(chose, 1);
    });

    testWidgets('a driver shows with their number, and opens to call', (
      tester,
    ) async {
      Contact? opened;
      var chose = 0;
      await pumpLeg(
        tester,
        LegTransport(
          mode: 'Taxi',
          vehicleNumber: 'ML 05 A 1234',
          driver: person(3, 'Anthony', 'transport'),
        ),
        onChoose: () => chose++,
        onOpen: (c) => opened = c,
      );
      expect(find.text('Anthony'), findsOneWidget);
      expect(find.text('+91 98560 12343'), findsOneWidget);
      expect(find.text('ML 05 A 1234'), findsOneWidget);
      await tester.tap(find.byKey(const Key('leg-open-driver')));
      await tester.tap(find.byKey(const Key('leg-change-driver')));
      expect(opened?.id, 3);
      expect(chose, 1);
    });

    testWidgets('the leg form saves the vehicle number, in capitals', (
      tester,
    ) async {
      tall(tester);
      String? saved;
      await tester.pumpWidget(
        MaterialApp(
          theme: AppTokens.light,
          home: LegFormScreen(
            fromName: 'Shillong',
            toName: 'Sohra',
            onSave: ({
              mode,
              plannedDeparture,
              plannedArrival,
              required isBooked,
              note,
              vehicleNumber,
            }) async => saved = vehicleNumber,
          ),
        ),
      );
      await tester.enterText(
        find.byKey(const Key('leg-vehicle')),
        'ml 05 a 1234',
      );
      await tester.tap(find.text('Save'));
      await tester.pump();
      expect(saved, 'ML 05 A 1234');
    });
  });
}
