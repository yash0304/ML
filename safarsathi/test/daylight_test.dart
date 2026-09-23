// test/daylight_test.dart
//
// Sunset on the leg and the stop — the part of the sun maths a person sees.

import 'package:drift/drift.dart' hide isNull, isNotNull;
import 'package:drift/native.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:safarsathi/core/database/app_database.dart';
import 'package:safarsathi/core/theme/app_tokens.dart';
import 'package:safarsathi/core/util/sun.dart';
import 'package:safarsathi/features/discovery/data/discovery.dart';
import 'package:safarsathi/features/trips/data/trip_editor.dart';
import 'package:safarsathi/features/trips/presentation/leg_detail_screen.dart';

void main() {
  group('the leg', () {
    late AppDatabase db;
    late int legId;
    late int sohraId;

    setUp(() async {
      db = AppDatabase(NativeDatabase.memory());
      final editor = TripEditor(db);
      final trip = await editor.createTrip(name: 'Meghalaya');
      await editor.addStop(
        trip,
        const StopDraft(name: 'Shillong', lat: 25.5764, lon: 91.8838),
      );
      sohraId = await editor.addStop(
        trip,
        const StopDraft(name: 'Sohrra', lat: 25.2718, lon: 91.7327),
      );
      legId = (await db.select(db.legs).getSingle()).id;
    });
    tearDown(() => db.close());

    Future<void> times({DateTime? leaves, DateTime? arrives}) =>
        (db.update(db.legs)..where((l) => l.id.equals(legId))).write(
          LegsCompanion(
            plannedDeparture: Value(leaves),
            plannedArrival: Value(arrives),
          ),
        );

    test('the sunset is the destination\'s, on the arrival day', () async {
      final arrives = DateTime(2026, 10, 3, 12);
      await times(arrives: arrives);
      final t = await watchLegTransport(db, legId).first;
      final expected = sunTimes(25.2718, 91.7327, arrives).sunset;
      expect(t.arriveSun?.sunset, expected);
      expect(t.toName, 'Sohrra');
    });

    test('ARRIVING AFTER DARK IS SAID', () async {
      final sunset = sunTimes(25.2718, 91.7327, DateTime(2026, 10, 3)).sunset!;
      await times(arrives: sunset.add(const Duration(minutes: 45)).toLocal());
      final t = await watchLegTransport(db, legId).first;
      expect(t.arrivalWarning, contains('45 min after sunset'));
    });

    test('a midday leg says nothing', () async {
      await times(
        leaves: DateTime(2026, 10, 3, 9),
        arrives: DateTime(2026, 10, 3, 11),
      );
      final t = await watchLegTransport(db, legId).first;
      expect(t.departureWarning, isNull);
      expect(t.arrivalWarning, isNull);
    });

    test('with no times typed, the destination\'s own date still gives a '
        'sunset', () async {
      await (db.update(db.stops)..where((s) => s.id.equals(sohraId))).write(
        StopsCompanion(arrivalDate: Value(DateTime(2026, 10, 3))),
      );
      final t = await watchLegTransport(db, legId).first;
      expect(t.arriveSun?.sunset, isNotNull);
    });

    test('THE LEG FOLLOWS A STOP THAT MOVES', () async {
      // It used to watch the legs table alone. Moving a stop would have left
      // the old sunset on screen.
      await times(arrives: DateTime(2026, 10, 3, 12));
      final seen = <DateTime?>[];
      final sub = watchLegTransport(db, legId).listen(
        (t) => seen.add(t.arriveSun?.sunset),
      );
      await pumpEventQueue();
      await (db.update(db.stops)..where((s) => s.id.equals(sohraId))).write(
        // Far west: Kanyakumari. The sunset moves by over an hour.
        const StopsCompanion(lat: Value(8.0883), lon: Value(77.5385)),
      );
      await pumpEventQueue();
      await sub.cancel();

      expect(seen.length, greaterThanOrEqualTo(2));
      expect(seen.last, isNot(seen.first));
    });
  });

  group('the screen', () {
    Future<void> pump(WidgetTester tester, LegTransport t) async {
      tester.view.physicalSize = const Size(420, 1600);
      tester.view.devicePixelRatio = 1.0;
      addTearDown(tester.view.reset);
      await tester.pumpWidget(
        MaterialApp(
          theme: AppTokens.light,
          home: LegDetailScreen(
            discovery: Stream.value(
              const LegDiscovery(
                legId: 1,
                fromName: 'Shillong',
                toName: 'Sohrra',
                places: [],
              ),
            ),
            transport: Stream.value(t),
          ),
        ),
      );
      // Two frames: the transport stream is built inside the leg stream's
      // builder, so it subscribes one frame later.
      await tester.pump();
      await tester.pump();
    }

    testWidgets('the sunset row names the place and the time', (tester) async {
      final sun = sunTimes(25.2718, 91.7327, DateTime(2026, 10, 3));
      await pump(tester, LegTransport(arriveSun: sun, toName: 'Sohrra'));
      expect(find.text('Sunset, Sohrra'), findsOneWidget);
      expect(find.text(clockTime(sun.sunset!)), findsOneWidget);
    });

    testWidgets('it shows even when nothing about the leg is typed', (
      tester,
    ) async {
      final sun = sunTimes(25.2718, 91.7327, DateTime(2026, 10, 3));
      await pump(tester, LegTransport(arriveSun: sun, toName: 'Sohrra'));
      expect(find.text('Nothing recorded for this leg yet.'), findsOneWidget);
      expect(find.text('Sunset, Sohrra'), findsOneWidget);
    });

    testWidgets('an after-dark arrival is on screen', (tester) async {
      final sun = sunTimes(25.2718, 91.7327, DateTime(2026, 10, 3));
      await pump(
        tester,
        LegTransport(
          mode: 'Shared sumo',
          plannedArrival: sun.sunset!.add(const Duration(minutes: 70)),
          arriveSun: sun,
          toName: 'Sohrra',
        ),
      );
      expect(find.textContaining('1 h 10 min after sunset'), findsOneWidget);
    });
  });
}
