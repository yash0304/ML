// test/check_in_reminders_test.dart — #34, when nobody has checked in.

import 'package:drift/drift.dart' hide isNull, isNotNull;
import 'package:drift/native.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:safarsathi/core/database/app_database.dart';
import 'package:safarsathi/core/theme/app_tokens.dart';
import 'package:safarsathi/features/emergency/data/check_in.dart';
import 'package:safarsathi/features/emergency/data/check_in_reminders.dart';
import 'package:safarsathi/features/emergency/presentation/check_in_screen.dart';
import 'package:safarsathi/features/map/data/here.dart';
import 'package:safarsathi/features/trips/data/trip_editor.dart';

import 'here_test.dart' show FakeLocation;

Leg leg(int id, int to, {DateTime? departs, DateTime? arrives}) => Leg(
  id: id,
  tripId: 1,
  fromStopId: to - 1,
  toStopId: to,
  sequenceOrder: id,
  isBooked: false,
  corridorKm: 3,
  plannedDeparture: departs,
  plannedArrival: arrives,
);

TrustedContact person(int minutes, {bool escalate = true}) => TrustedContact(
  id: minutes,
  name: 'Maa',
  phoneE164: '+919800000000',
  notifyOnArrival: true,
  escalate: escalate,
  escalateAfterMinutes: minutes,
);

void main() {
  final departs = DateTime(2026, 10, 2, 8);
  final arrives = DateTime(2026, 10, 2, 12);
  const names = {2: 'Sohra'};

  test('two reminders after a planned arrival: arrival + buffer, then twice',
      () {
    final r = checkInReminders(
      legs: [leg(5, 2, departs: departs, arrives: arrives)],
      stopNames: names,
      checkIns: const {},
      now: DateTime(2026, 10, 2, 9),
    );
    expect(r.map((x) => x.at), [
      DateTime(2026, 10, 2, 14),
      DateTime(2026, 10, 2, 16),
    ]);
    expect(r.first.title, 'Reached Sohra?');
    expect(r.last.escalation, isTrue);
    expect(r.last.title, 'Still no check-in from Sohra');
    expect(r.first.payload, 'checkin:2');
    // Stable ids, so rescheduling replaces.
    expect(r.map((x) => x.id), [10, 11]);
  });

  test('A CHECK-IN AFTER SETTING OFF CANCELS THEM', () {
    expect(
      checkInReminders(
        legs: [leg(5, 2, departs: departs, arrives: arrives)],
        stopNames: names,
        checkIns: {2: DateTime(2026, 10, 2, 12, 30)},
        now: DateTime(2026, 10, 2, 9),
      ),
      isEmpty,
    );
  });

  test('a check-in from an earlier visit does not count', () {
    // Shillong is often first and last; the first night's check-in is not
    // this evening's.
    expect(
      checkInReminders(
        legs: [leg(5, 2, departs: departs, arrives: arrives)],
        stopNames: names,
        checkIns: {2: DateTime(2026, 9, 30, 18)},
        now: DateTime(2026, 10, 2, 9),
      ),
      hasLength(2),
    );
  });

  test('no planned arrival, no reminder; past times are dropped', () {
    expect(
      checkInReminders(
        legs: [leg(5, 2, departs: departs)],
        stopNames: names,
        checkIns: const {},
        now: DateTime(2026, 10, 2, 9),
      ),
      isEmpty,
    );
    final late = checkInReminders(
      legs: [leg(5, 2, departs: departs, arrives: arrives)],
      stopNames: names,
      checkIns: const {},
      now: DateTime(2026, 10, 2, 15),
    );
    expect(late.single.escalation, isTrue);
  });

  test('the buffer is the shortest a trusted person asked for', () {
    expect(reminderBuffer(const []), 120);
    expect(reminderBuffer([person(180), person(90)]), 90);
    expect(reminderBuffer([person(30, escalate: false)]), 120);
  });

  test('a tapped reminder names its stop', () {
    expect(stopFromPayload('checkin:7'), 7);
    expect(stopFromPayload('other'), isNull);
    expect(stopFromPayload(null), isNull);
  });

  group('from the database', () {
    late AppDatabase db;
    late int tripId, sohra;

    setUp(() async {
      db = AppDatabase(NativeDatabase.memory());
      final editor = TripEditor(db);
      tripId = await editor.createTrip(name: 'Meghalaya');
      await editor.setActiveTrip(tripId);
      await editor.addStop(tripId, const StopDraft(name: 'Shillong'));
      sohra = await editor.addStop(tripId, const StopDraft(name: 'Sohra'));
      final legId = (await db.select(db.legs).getSingle()).id;
      await (db.update(db.legs)..where((l) => l.id.equals(legId))).write(
        LegsCompanion(
          plannedDeparture: Value(departs),
          plannedArrival: Value(arrives),
        ),
      );
    });
    tearDown(() => db.close());

    test('REMINDERS OFF SCHEDULES NOTHING', () async {
      expect(
        await remindersDue(db, enabled: false, now: DateTime(2026, 10, 2, 9)),
        isEmpty,
      );
    });

    test('on, it schedules for the active trip until you check in', () async {
      final due = await remindersDue(
        db,
        enabled: true,
        now: DateTime(2026, 10, 2, 9),
      );
      expect(due.map((r) => r.stopName), ['Sohra', 'Sohra']);

      await recordCheckIn(
        db,
        tripId: tripId,
        stopId: sohra,
        stopName: 'Sohra',
        at: DateTime(2026, 10, 2, 12, 40),
      );
      expect(
        await remindersDue(db, enabled: true, now: DateTime(2026, 10, 2, 13)),
        isEmpty,
      );
    });
  });

  testWidgets('the switch asks, and says so when notifications are refused', (
    tester,
  ) async {
    tester.view.physicalSize = const Size(420, 1600);
    tester.view.devicePixelRatio = 1.0;
    addTearDown(tester.view.reset);
    final asked = <bool>[];
    await tester.pumpWidget(
      MaterialApp(
        theme: AppTokens.light,
        home: CheckInScreen(
          stops: const [(id: 2, name: 'Sohra')],
          trusted: Stream.value(const []),
          location: FakeLocation(HereState.notAsked),
          stayAt: (_) async => null,
          openSms: (_) async => true,
          share: (_) async {},
          onRecorded: (_, _) async {},
          remindersOn: Stream.value(false),
          onReminders: (on) async {
            asked.add(on);
            return 'Notifications are off for SafarSathi.';
          },
        ),
      ),
    );
    await tester.pump();
    expect(find.text('Remind me if I forget'), findsOneWidget);
    expect(find.textContaining('nothing is ever sent without you'),
        findsOneWidget);
    await tester.tap(find.byKey(const Key('checkin-reminders')));
    await tester.pump();
    expect(asked, [true]);
    expect(find.text('Notifications are off for SafarSathi.'), findsOneWidget);
  });
}
