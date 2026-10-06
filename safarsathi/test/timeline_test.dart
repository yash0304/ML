// test/timeline_test.dart — #30, the trip as it went.

import 'dart:async';

import 'package:drift/native.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:safarsathi/core/database/app_database.dart';
import 'package:safarsathi/core/theme/app_tokens.dart';
import 'package:safarsathi/features/discovery/data/geo.dart';
import 'package:safarsathi/features/map/data/here.dart';
import 'package:safarsathi/features/map/data/track_logger.dart';
import 'package:safarsathi/features/map/presentation/trip_map_screen.dart';
import 'package:safarsathi/features/trips/data/timeline.dart';
import 'package:safarsathi/features/trips/data/trip_editor.dart';
import 'package:safarsathi/features/trips/presentation/timeline_note_screen.dart';
import 'package:safarsathi/features/trips/presentation/timeline_screen.dart';

import 'here_test.dart' show FakeLocation;

const shillong = LatLng(25.5788, 91.8933);
const sohra = LatLng(25.2718, 91.7327);

TimelineEntry entry(
  int id,
  String kind,
  DateTime at, {
  int? stopId,
  LatLng? where,
  String? body,
  String? photos,
}) => TimelineEntry(
  id: id,
  tripId: 1,
  stopId: stopId,
  kind: kind,
  body: body,
  lat: where?.lat,
  lon: where?.lon,
  photoPaths: photos,
  occurredAt: at,
);

Stop stop(int id, String name, LatLng at) => Stop(
  id: id,
  tripId: 1,
  name: name,
  sequenceOrder: id,
  countryCode: 'IN',
  nights: 1,
  activityTags: '',
  lat: at.lat,
  lon: at.lon,
);

class _Track implements TrackSource {
  final controller = StreamController<HereFix>();
  @override
  Stream<HereFix> track() => controller.stream;
}

void main() {
  final stops = {1: stop(1, 'Shillong', shillong), 2: stop(2, 'Sohra', sohra)};

  group('building the timeline', () {
    test('days in order; arrivals and notes, never the fixes themselves', () {
      final days = buildTimeline([
        entry(3, 'note', DateTime(2026, 10, 2, 16), body: 'Fog at Nohkalikai'),
        entry(1, 'arrival', DateTime(2026, 10, 1, 13), stopId: 1),
        entry(2, 'arrival', DateTime(2026, 10, 2, 12), stopId: 2),
        entry(4, 'fix', DateTime(2026, 10, 2, 9), where: shillong),
      ], stops);
      expect(days.map((d) => d.day), [
        DateTime(2026, 10, 1),
        DateTime(2026, 10, 2),
      ]);
      expect(days[1].items, hasLength(2));
      expect((days[1].items.first as TimelineArrival).stopName, 'Sohra');
      expect((days[1].items.last as TimelineEntry).body, 'Fog at Nohkalikai');
    });

    test('WITH NO TRACK, THE DISTANCE IS A STRAIGHT LINE AND SAYS SO', () {
      final days = buildTimeline([
        entry(1, 'arrival', DateTime(2026, 10, 1, 13), stopId: 1),
        entry(2, 'arrival', DateTime(2026, 10, 2, 12), stopId: 2),
      ], stops);
      final a = days[1].items.single as TimelineArrival;
      expect(a.straightLine, isTrue);
      expect(a.km, closeTo(37.5, 1.5));
      expect(a.since, const Duration(hours: 23));
    });

    test('with a logged track, the road driven and the day\'s total', () {
      final days = buildTimeline([
        entry(1, 'arrival', DateTime(2026, 10, 2, 8), stopId: 1),
        entry(10, 'fix', DateTime(2026, 10, 2, 9), where: shillong),
        entry(11, 'fix', DateTime(2026, 10, 2, 10),
            where: const LatLng(25.45, 91.75)),
        entry(12, 'fix', DateTime(2026, 10, 2, 11), where: sohra),
        entry(2, 'arrival', DateTime(2026, 10, 2, 12), stopId: 2),
      ], stops);
      final a = days.single.items.last as TimelineArrival;
      expect(a.straightLine, isFalse);
      expect(a.km, greaterThan(37.5), reason: 'a bend is longer than a line');
      expect(days.single.loggedKm, closeTo(a.km!, 0.001));
    });

    test('a fix is kept every 150 m or 10 minutes, and only if precise', () {
      final t = DateTime(2026, 10, 2, 9);
      expect(keepFix(at: shillong, accuracyM: 12, time: t), isTrue);
      expect(keepFix(at: shillong, accuracyM: 250, time: t), isFalse);
      expect(
        keepFix(
          at: const LatLng(25.5789, 91.8933),
          accuracyM: 12,
          time: t.add(const Duration(minutes: 1)),
          lastAt: shillong,
          lastTime: t,
        ),
        isFalse,
      );
      expect(
        keepFix(
          at: const LatLng(25.5789, 91.8933),
          accuracyM: 12,
          time: t.add(const Duration(minutes: 11)),
          lastAt: shillong,
          lastTime: t,
        ),
        isTrue,
      );
      expect(
        keepFix(
          at: const LatLng(25.5810, 91.8933),
          accuracyM: 12,
          time: t.add(const Duration(minutes: 1)),
          lastAt: shillong,
          lastTime: t,
        ),
        isTrue,
      );
    });
  });

  group('the route log', () {
    late AppDatabase db;
    late int tripId;
    setUp(() async {
      db = AppDatabase(NativeDatabase.memory());
      tripId = await TripEditor(db).createTrip(name: 'Meghalaya');
    });
    tearDown(() => db.close());

    test('stores thinned fixes, and stops', () async {
      final track = _Track();
      final logger = TrackLogger(
        db: db,
        location: FakeLocation(HereState.locating),
        source: track,
      );
      expect(await logger.start(tripId), isNull);
      final t = DateTime(2026, 10, 2, 9);
      track.controller
        ..add(HereFix(at: shillong, accuracyM: 10, time: t))
        ..add(HereFix(
          at: const LatLng(25.5789, 91.8933),
          accuracyM: 10,
          time: t.add(const Duration(seconds: 30)),
        ))
        ..add(HereFix(
          at: const LatLng(25.5900, 91.8933),
          accuracyM: 10,
          time: t.add(const Duration(minutes: 2)),
        ));
      await pumpEventQueue();
      expect(await loggedTrack(db, tripId), hasLength(2));
      await logger.stop();
      expect(logger.running, isFalse);
    });

    test('THE TRIP MAP GETS THE ROAD ACTUALLY DRIVEN', () async {
      await addFix(db, tripId: tripId, at: shillong, accuracyM: 9,
          time: DateTime(2026, 10, 2, 9));
      await addFix(db, tripId: tripId, at: sohra, accuracyM: 9,
          time: DateTime(2026, 10, 2, 11));
      final view = await readTripMap(db, tripId, providerId: 'maptiler');
      expect(view.track, hasLength(2));
    });

    test('REFUSED LOCATION IS SAID, and nothing runs', () async {
      final logger = TrackLogger(
        db: db,
        location: FakeLocation(HereState.notAsked,
            afterAsking: HereState.denied),
        source: _Track(),
      );
      expect(await logger.start(tripId), contains('not allowed'));
      expect(logger.running, isFalse);
    });
  });

  group('the screen', () {
    Widget photo(String p) => Text('PHOTO $p');

    testWidgets('THE LOGGING SWITCH STATES ITS BATTERY COST', (tester) async {
      tester.view.physicalSize = const Size(420, 1600);
      tester.view.devicePixelRatio = 1.0;
      addTearDown(tester.view.reset);
      final asked = <bool>[];
      await tester.pumpWidget(
        MaterialApp(
          theme: AppTokens.light,
          home: TimelineScreen(
            timeline: Stream.value(const []),
            photo: photo,
            logging: Stream.value(false),
            onLogging: (on) async {
              asked.add(on);
              return null;
            },
          ),
        ),
      );
      await tester.pump();
      expect(find.textContaining('about 4% of the battery a day'),
          findsOneWidget);
      expect(find.textContaining('Nothing yet'), findsOneWidget);
      await tester.tap(find.byKey(const Key('timeline-logging')));
      await tester.pump();
      expect(asked, [true]);
    });

    testWidgets('arrivals are caps, notes are dots; a note can be deleted', (
      tester,
    ) async {
      tester.view.physicalSize = const Size(420, 1600);
      tester.view.devicePixelRatio = 1.0;
      addTearDown(tester.view.reset);
      final deleted = <int>[];
      await tester.pumpWidget(
        MaterialApp(
          theme: AppTokens.light,
          home: TimelineScreen(
            photo: photo,
            onDelete: (n) async => deleted.add(n.id),
            timeline: Stream.value(
              buildTimeline([
                entry(1, 'arrival', DateTime(2026, 10, 1, 13), stopId: 1),
                entry(2, 'arrival', DateTime(2026, 10, 2, 12), stopId: 2),
                entry(3, 'note', DateTime(2026, 10, 2, 16),
                    body: 'Fog at Nohkalikai', photos: '/a.jpg'),
              ], stops),
            ),
          ),
        ),
      );
      await tester.pump();
      expect(find.byKey(const Key('rail-cap')), findsNWidgets(2));
      expect(find.byKey(const Key('rail-dot')), findsOneWidget);
      expect(find.text('Reached Sohra · 12:00'), findsOneWidget);
      expect(find.textContaining('km in a straight line'), findsOneWidget);
      expect(find.text('PHOTO /a.jpg'), findsOneWidget);

      await tester.longPress(find.text('Fog at Nohkalikai'));
      await tester.pumpAndSettle();
      await tester.tap(find.byKey(const Key('timeline-delete-confirm')));
      await tester.pumpAndSettle();
      expect(deleted, [3]);
    });

    testWidgets('a note needs words or a photo', (tester) async {
      tester.view.physicalSize = const Size(420, 1200);
      tester.view.devicePixelRatio = 1.0;
      addTearDown(tester.view.reset);
      final saved = <(String, List<String>)>[];
      await tester.pumpWidget(
        MaterialApp(
          theme: AppTokens.light,
          home: TimelineNoteScreen(
            photo: photo,
            pickPhoto: ({required camera}) async =>
                camera ? '/cam.jpg' : '/gal.jpg',
            onSave: (t, p) async => saved.add((t, p)),
          ),
        ),
      );
      await tester.tap(find.byKey(const Key('note-save')));
      await tester.pump();
      expect(saved, isEmpty);

      await tester.tap(find.byKey(const Key('note-photo-camera')));
      await tester.pump();
      expect(find.text('PHOTO /cam.jpg'), findsOneWidget);
      await tester.tap(find.byKey(const Key('note-save')));
      await tester.pumpAndSettle();
      expect(saved.single.$2, ['/cam.jpg']);
    });
  });

  test('notes are stored with their photos', () async {
    final db = AppDatabase(NativeDatabase.memory());
    addTearDown(db.close);
    final trip = await TripEditor(db).createTrip(name: 'M');
    await addTimelineNote(
      db,
      tripId: trip,
      text: '  Root bridge  ',
      photoPaths: ['/a.jpg', '/b.jpg'],
    );
    final row = await db.select(db.timelineEntries).getSingle();
    expect(row.body, 'Root bridge');
    expect(photosOf(row), ['/a.jpg', '/b.jpg']);
  });
}
