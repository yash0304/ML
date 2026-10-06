// test/check_in_test.dart — #33, "reached safely" to the trusted people.

import 'package:drift/native.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:safarsathi/core/database/app_database.dart';
import 'package:safarsathi/core/theme/app_tokens.dart';
import 'package:safarsathi/features/discovery/data/geo.dart';
import 'package:safarsathi/features/emergency/data/check_in.dart';
import 'package:safarsathi/features/emergency/presentation/check_in_screen.dart';
import 'package:safarsathi/features/map/data/here.dart';
import 'package:safarsathi/features/trips/data/trip_editor.dart';

import 'here_test.dart' show FakeLocation;

class _Fixed extends FakeLocation {
  _Fixed() : super(HereState.locating);
  @override
  Future<HereFix?> once({Duration timeout = const Duration(seconds: 20)}) async =>
      HereFix(
        at: const LatLng(25.27183, 91.73271),
        accuracyM: 9,
        time: DateTime(2026, 10, 2, 17, 20),
      );
}

Leg leg(int id, int to, DateTime departs) => Leg(
  id: id,
  tripId: 1,
  fromStopId: to - 1,
  toStopId: to,
  sequenceOrder: id,
  isBooked: false,
  corridorKm: 3,
  plannedDeparture: departs,
);

void main() {
  group('the message', () {
    test('reached, when, where, and where tonight is', () {
      final m = checkInMessage(
        stopName: 'Sohra',
        at: DateTime(2026, 10, 2, 17, 20),
        fix: HereFix(
          at: const LatLng(25.27183, 91.73271),
          accuracyM: 9,
          time: DateTime(2026, 10, 2, 17, 20),
        ),
        stay: 'Alpha Guest House, +91 89748 04455',
        now: DateTime(2026, 10, 2, 17, 20, 10),
      );
      expect(m, '''
Reached Sohra safely — 17:20, 2 Oct.
Where I am: 25.27183, 91.73271 (±9 m, 10 s ago)
Map: https://maps.google.com/?q=25.27183,91.73271
Staying at: Alpha Guest House, +91 89748 04455
Sent from SafarSathi.''');
    });

    test('no GPS and no stay: still a check-in', () {
      expect(
        checkInMessage(stopName: 'Dawki', at: DateTime(2026, 10, 4, 9, 5)),
        'Reached Dawki safely — 09:05, 4 Oct.\nSent from SafarSathi.',
      );
    });

    test('it defaults to where today\'s leg arrives, else where you are', () {
      final legs = [
        leg(1, 2, DateTime(2026, 10, 2, 8)),
        leg(2, 3, DateTime(2026, 10, 3, 9)),
      ];
      expect(
        checkInStopId(
          legs: legs,
          currentStopId: 1,
          now: DateTime(2026, 10, 2, 18),
        ),
        2,
      );
      expect(
        checkInStopId(
          legs: legs,
          currentStopId: 1,
          now: DateTime(2026, 10, 5, 12),
        ),
        1,
      );
    });
  });

  test('a check-in is recorded as an arrival, latest per stop', () async {
    final db = AppDatabase(NativeDatabase.memory());
    addTearDown(db.close);
    final editor = TripEditor(db);
    final trip = await editor.createTrip(name: 'Meghalaya');
    final sohra = await editor.addStop(trip, const StopDraft(name: 'Sohra'));
    await recordCheckIn(
      db,
      tripId: trip,
      stopId: sohra,
      stopName: 'Sohra',
      at: DateTime(2026, 10, 2, 17),
    );
    await recordCheckIn(
      db,
      tripId: trip,
      stopId: sohra,
      stopName: 'Sohra',
      at: DateTime(2026, 10, 2, 18),
    );
    final seen = await watchCheckIns(db, trip).first;
    expect(seen, {sohra: DateTime(2026, 10, 2, 18)});
    final row = (await db.select(db.timelineEntries).get()).first;
    expect(row.kind, 'arrival');
    expect(row.title, 'Checked in at Sohra');
  });

  group('the screen', () {
    late List<Uri> opened;
    late List<String> recorded;

    Future<void> pump(
      WidgetTester tester, {
      List<TrustedContact> people = const [],
      LocationSource? location,
    }) async {
      opened = [];
      recorded = [];
      tester.view.physicalSize = const Size(420, 1400);
      tester.view.devicePixelRatio = 1.0;
      addTearDown(tester.view.reset);
      await tester.pumpWidget(
        MaterialApp(
          theme: AppTokens.light,
          home: CheckInScreen(
            stops: const [(id: 1, name: 'Shillong'), (id: 2, name: 'Sohra')],
            initialStopId: 2,
            trusted: Stream.value(people),
            location: location ?? _Fixed(),
            stayAt: (id) async => id == 2 ? 'Alpha Guest House' : null,
            openSms: (uri) async {
              opened.add(uri);
              return true;
            },
            share: (_) async {},
            onRecorded: (stop, fix) async => recorded.add(stop.name),
            clock: () => DateTime(2026, 10, 2, 17, 20),
          ),
        ),
      );
      await tester.pump();
    }

    testWidgets('NO ONE TO TELL SAYS WHERE TO CHOOSE THEM', (tester) async {
      await pump(tester);
      expect(find.textContaining('Choose people on the SOS tab'),
          findsOneWidget);
    });

    testWidgets('texting someone opens SMS with the check-in, and records it',
        (tester) async {
      await pump(
        tester,
        people: const [
          TrustedContact(
            id: 7,
            name: 'Maa',
            phoneE164: '+919800000000',
            notifyOnArrival: true,
            escalate: true,
            escalateAfterMinutes: 120,
          ),
        ],
      );
      await tester.tap(find.byKey(const Key('checkin-to-7')));
      await tester.pumpAndSettle();
      final body = Uri.decodeComponent(opened.single.query);
      expect(opened.single.path, '%2B919800000000');
      expect(body, contains('Reached Sohra safely — 17:20, 2 Oct.'));
      expect(body, contains('Staying at: Alpha Guest House'));
      expect(body, contains('25.27183, 91.73271'));
      expect(recorded, ['Sohra']);
      expect(find.byKey(const Key('checkin-done')), findsOneWidget);
    });

    testWidgets('location can be left out, and is then never asked for', (
      tester,
    ) async {
      final phone = FakeLocation(HereState.locating);
      await pump(
        tester,
        location: phone,
        people: const [
          TrustedContact(
            id: 7,
            name: 'Maa',
            phoneE164: '+919800000000',
            notifyOnArrival: true,
            escalate: true,
            escalateAfterMinutes: 120,
          ),
        ],
      );
      await tester.tap(find.byKey(const Key('checkin-location')));
      await tester.pump();
      await tester.tap(find.byKey(const Key('checkin-stop-1')));
      await tester.pump();
      await tester.tap(find.byKey(const Key('checkin-to-7')));
      await tester.pumpAndSettle();
      expect(phone.asks, isEmpty);
      expect(Uri.decodeComponent(opened.single.query),
          isNot(contains('Where I am')));
      expect(recorded, ['Shillong']);
    });
  });
}
