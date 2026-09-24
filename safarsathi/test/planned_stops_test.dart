// test/planned_stops_test.dart — viewpoints and other stops on the way.
//
// "In between stops as well as view points also should be provided somewhere
// where we can send the same to home as well."
//
// Three things are pinned here: sights are downloaded, a place on the road
// can be put in the plan (or typed), and the plan that goes home names them
// under their leg, in road order. Plus the one failure that must not
// happen: a planned stop vanishing because the itinerary was edited.

import 'dart:convert';
import 'dart:io';

import 'package:drift/native.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:safarsathi/core/database/app_database.dart';
import 'package:safarsathi/core/theme/app_tokens.dart';
import 'package:safarsathi/features/backup/data/backup.dart';
import 'package:safarsathi/features/discovery/data/corridor.dart';
import 'package:safarsathi/features/discovery/data/discovery.dart';
import 'package:safarsathi/features/discovery/data/geo.dart';
import 'package:safarsathi/features/discovery/data/overpass_client.dart';
import 'package:safarsathi/features/discovery/data/poi_category.dart';
import 'package:safarsathi/features/discovery/presentation/discovery_screen.dart';
import 'package:safarsathi/features/discovery/presentation/poi_detail_screen.dart';
import 'package:safarsathi/features/trips/data/plan_message.dart';
import 'package:safarsathi/features/trips/data/planned_stops.dart';
import 'package:safarsathi/features/trips/data/trip_editor.dart';
import 'package:safarsathi/features/trips/presentation/leg_detail_screen.dart';
import 'package:safarsathi/features/trips/presentation/planned_stop_dialog.dart';

const shillongAt = LatLng(25.5788, 91.8933);
const sohraAt = LatLng(25.2718, 91.7327);

CorridorPlace viewpoint({
  String name = 'Mawkdok Dympep viewpoint',
  String osmId = 'node/77',
}) => CorridorPlace(
  id: 1,
  name: name,
  category: 'viewpoint',
  lat: 25.4400,
  lon: 91.7800,
  alongRouteKm: 20,
  offRouteKm: 0.2,
  osmId: osmId,
);

PlannedStop plannedRow(
  int id,
  String name, {
  double? lat,
  double? lon,
  int? legId = 1,
  String? osmId,
  String? note,
  DateTime? at,
}) => PlannedStop(
  id: id,
  tripId: 1,
  legId: legId,
  name: name,
  lat: lat,
  lon: lon,
  osmId: osmId,
  note: note,
  createdAt: at ?? DateTime(2026, 9, 24, 10, id),
);

void main() {
  group('sights are downloaded', () {
    test('viewpoints, waterfalls and caves are asked for by default', () {
      expect(defaultPoiCategoryKeys, contains('viewpoint'));
      final q = OverpassClient.buildQuery(
        const BoundingBox(north: 25.6, south: 25.2, east: 91.9, west: 91.7),
      );
      expect(q, contains('"tourism"="viewpoint"'));
      expect(q, contains('"waterway"="waterfall"'));
      expect(q, contains('"natural"="cave_entrance"'));
    });

    test('a waterfall and a cave read as Sights', () {
      expect(categoryForTags({'waterway': 'waterfall'}), 'viewpoint');
      expect(categoryForTags({'natural': 'cave_entrance'}), 'viewpoint');
      expect(placeCategoryLabel('viewpoint'), 'Sights');
      // The diary's own words still win where it has them.
      expect(placeCategoryLabel('restaurant'), 'Food');
    });
  });

  group('road order', () {
    test('placed stops by kilometre, then typed ones in the order added', () {
      final corridor = Corridor([shillongAt, sohraAt]);
      final ordered = orderPlanned(
        [
          plannedRow(1, 'Lunch somewhere', at: DateTime(2026, 9, 24, 9)),
          plannedRow(2, 'Near Sohra', lat: 25.30, lon: 91.75),
          plannedRow(3, 'Near Shillong', lat: 25.55, lon: 91.88),
          plannedRow(4, 'Ask the driver', at: DateTime(2026, 9, 24, 11)),
        ],
        corridor,
      );
      expect(ordered.map((p) => p.stop.name), [
        'Near Shillong',
        'Near Sohra',
        'Lunch somewhere',
        'Ask the driver',
      ]);
      expect(ordered.first.alongRouteKm, lessThan(ordered[1].alongRouteKm!));
      expect(ordered[2].alongRouteKm, isNull);
    });
  });

  group('the database', () {
    late AppDatabase db;
    late TripEditor editor;
    late int tripId, shillong, sohra;

    setUp(() async {
      db = AppDatabase(NativeDatabase.memory());
      editor = TripEditor(db);
      tripId = await editor.createTrip(name: 'Meghalaya');
      shillong = await editor.addStop(
        tripId,
        StopDraft(
          name: 'Shillong',
          nights: 1,
          arrivalDate: DateTime(2026, 10, 1),
          lat: shillongAt.lat,
          lon: shillongAt.lon,
        ),
      );
      sohra = await editor.addStop(
        tripId,
        StopDraft(
          name: 'Sohra',
          nights: 1,
          arrivalDate: DateTime(2026, 10, 2),
          lat: sohraAt.lat,
          lon: sohraAt.lon,
        ),
      );
    });
    tearDown(() => db.close());

    Future<Leg> onlyLeg() => (db.select(db.legs)).getSingle();

    test('a place is planned once, and taken out again', () async {
      final leg = await onlyLeg();
      await planPlace(db, tripId: tripId, legId: leg.id, place: viewpoint());
      await planPlace(db, tripId: tripId, legId: leg.id, place: viewpoint());
      expect(await db.select(db.plannedStops).get(), hasLength(1));
      expect(
        await isPlacePlanned(db, legId: leg.id, osmId: 'node/77'),
        isTrue,
      );

      await unplanPlace(db, legId: leg.id, osmId: 'node/77');
      expect(await db.select(db.plannedStops).get(), isEmpty);
    });

    test('the leg screen sees planned stops and which places they are', () async {
      final leg = await onlyLeg();
      await planPlace(db, tripId: tripId, legId: leg.id, place: viewpoint());
      await planStop(db, tripId: tripId, legId: leg.id, name: 'Lunch');
      final d = await watchLegDiscovery(db, leg.id).first;
      expect(d.planned.map((p) => p.stop.name),
          ['Mawkdok Dympep viewpoint', 'Lunch']);
      expect(d.plannedOsmIds, {'node/77'});
    });

    test('A STOP INSERTED MID-TRIP MOVES THE PLAN ONTO THE NEW LEG', () async {
      // Shillong → Sohra becomes Shillong → Laitlum → Sohra. The old leg is
      // gone; the viewpoint planned on it must not go with it.
      final leg = await onlyLeg();
      final id = await planStop(
        db,
        tripId: tripId,
        legId: leg.id,
        name: 'Elephant Falls',
      );
      final laitlum = await editor.addStop(
        tripId,
        const StopDraft(name: 'Laitlum'),
      );
      await editor.reorderStops(tripId, 2, 1);

      final legs = await db.select(db.legs).get();
      expect(legs.any((l) => l.id == leg.id), isFalse);
      final moved = await (db.select(
        db.plannedStops,
      )..where((p) => p.id.equals(id))).getSingle();
      final home = legs.firstWhere((l) => l.id == moved.legId);
      expect(home.fromStopId, shillong);
      expect(home.toStopId, laitlum);
    });

    test('with no leg left to hold it, it stays on the trip, not deleted',
        () async {
      final leg = await onlyLeg();
      await planStop(db, tripId: tripId, legId: leg.id, name: 'Elephant Falls');
      await editor.deleteStop(tripId, sohra);

      final left = await db.select(db.plannedStops).getSingle();
      expect(left.legId, isNull);
      expect(await buildPlanMessage(db, tripId),
          contains('Also planned:\n  - Elephant Falls'));
    });

    test('THE PLAN SENT HOME NAMES THE STOPS UNDER THEIR LEG, in road order',
        () async {
      final leg = await onlyLeg();
      await planStop(db, tripId: tripId, legId: leg.id, name: 'Lunch',
          note: 'ask for the thali');
      await planPlace(db, tripId: tripId, legId: leg.id, place: viewpoint());
      final msg = await buildPlanMessage(db, tripId);
      expect(
        msg,
        contains(
          'Shillong → Sohra\n'
          '  - Mawkdok Dympep viewpoint, ',
        ),
      );
      expect(msg, contains(' km in\n  - Lunch — ask for the thali'));
    });

    test('a backup carries the plan', () async {
      final leg = await onlyLeg();
      await planPlace(db, tripId: tripId, legId: leg.id, place: viewpoint());
      final json = await exportBackup(db);

      final fresh = AppDatabase(NativeDatabase.memory());
      addTearDown(fresh.close);
      await restoreBackup(fresh, readBackup(json));
      final back = await fresh.select(fresh.plannedStops).getSingle();
      expect(back.name, 'Mawkdok Dympep viewpoint');
      expect(back.legId, leg.id);
    });

    test('a v6 backup, which has no plan section, still restores', () async {
      final json = jsonDecode(await exportBackup(db)) as Map<String, dynamic>;
      json['schemaVersion'] = 6;
      (json['tables'] as Map).remove('plannedStops');
      final fresh = AppDatabase(NativeDatabase.memory());
      addTearDown(fresh.close);
      await restoreBackup(fresh, readBackup(jsonEncode(json)));
      expect(await fresh.select(fresh.plannedStops).get(), isEmpty);
      expect(await fresh.select(fresh.stops).get(), hasLength(2));
    });
  });

  test('A REAL v6 DATABASE UPGRADES TO v7 with its trip intact', () async {
    final dir = await Directory.systemTemp.createTemp('upgrade7');
    addTearDown(() => dir.delete(recursive: true));
    final file = File('${dir.path}/v6.sqlite');

    final v6 = AppDatabase(NativeDatabase(file));
    await v6.into(v6.trips).insert(TripsCompanion.insert(name: 'Meghalaya'));
    await v6.customStatement('DROP TABLE planned_stops');
    await v6.customStatement('ALTER TABLE legs DROP COLUMN driver_contact_id');
    await v6.customStatement('ALTER TABLE legs DROP COLUMN vehicle_number');
    await v6.customStatement('PRAGMA user_version = 6');
    await v6.close();

    final v7 = AppDatabase(NativeDatabase(file));
    addTearDown(v7.close);
    expect((await v7.select(v7.trips).getSingle()).name, 'Meghalaya');
    await v7.into(v7.plannedStops).insert(
      PlannedStopsCompanion.insert(tripId: 1, name: 'Nohkalikai Falls'),
    );
    expect(await v7.select(v7.plannedStops).get(), hasLength(1));
    final version = await v7.customSelect('PRAGMA user_version').getSingle();
    expect(version.read<int>('user_version'), 8);
  });

  group('the screens', () {
    void tall(WidgetTester tester) {
      tester.view.physicalSize = const Size(420, 2000);
      tester.view.devicePixelRatio = 1.0;
      addTearDown(tester.view.reset);
    }

    LegDiscovery legWith(List<PlannedOnTheWay> planned) => LegDiscovery(
      legId: 1,
      fromName: 'Shillong',
      toName: 'Sohra',
      places: [viewpoint()],
      lastSyncedAt: DateTime(2026, 9, 20),
      planned: planned,
    );

    testWidgets('the leg says how to plan when nothing is planned', (
      tester,
    ) async {
      tall(tester);
      var adds = 0;
      await tester.pumpWidget(
        MaterialApp(
          theme: AppTokens.light,
          home: LegDetailScreen(
            discovery: Stream.value(legWith(const [])),
            transport: Stream.value(const LegTransport()),
            onAddPlanned: () => adds++,
          ),
        ),
      );
      await tester.pump();
      expect(find.text('PLANNED STOPS ON THE WAY'), findsOneWidget);
      expect(find.textContaining('add it to the plan'), findsOneWidget);
      await tester.tap(find.byKey(const Key('leg-add-planned')));
      expect(adds, 1);
    });

    testWidgets('planned stops show their kilometre, and can go', (
      tester,
    ) async {
      tall(tester);
      final removed = <int>[];
      final directed = <int>[];
      await tester.pumpWidget(
        MaterialApp(
          theme: AppTokens.light,
          home: LegDetailScreen(
            discovery: Stream.value(
              legWith([
                PlannedOnTheWay(
                  plannedRow(5, 'Mawkdok Dympep viewpoint',
                      lat: 25.44, lon: 91.78),
                  alongRouteKm: 21.4,
                ),
                PlannedOnTheWay(plannedRow(6, 'Lunch', note: 'thali')),
              ]),
            ),
            transport: Stream.value(const LegTransport()),
            onRemovePlanned: (s) async => removed.add(s.id),
            onDirectionsTo: (s) => directed.add(s.id),
          ),
        ),
      );
      await tester.pump();
      expect(find.text('21 km in'), findsOneWidget);
      expect(find.text('thali'), findsOneWidget);

      await tester.tap(find.text('Mawkdok Dympep viewpoint').first);
      // Lunch has no position: no directions to offer.
      await tester.tap(find.text('Lunch'));
      await tester.tap(find.byKey(const Key('planned-remove-6')));
      expect(directed, [5]);
      expect(removed, [6]);
    });

    testWidgets('a place can be added to the plan from its page', (
      tester,
    ) async {
      tall(tester);
      final toggles = <bool>[];
      await tester.pumpWidget(
        MaterialApp(
          theme: AppTokens.light,
          home: PoiDetailScreen(
            place: viewpoint(),
            onCopy: (_) async {},
            onOpenDialer: (_) async {},
            onSave: (_) async {},
            onTogglePlan: (v) async => toggles.add(v),
          ),
        ),
      );
      await tester.pump();
      expect(find.text('Sights'), findsOneWidget);
      expect(find.text('Add to the plan for this leg'), findsOneWidget);
      await tester.tap(find.byKey(const Key('poi-plan-toggle')));
      await tester.pump();
      expect(find.text('In the plan · tap to take out'), findsOneWidget);
      await tester.tap(find.byKey(const Key('poi-plan-toggle')));
      await tester.pump();
      expect(toggles, [true, false]);
    });

    testWidgets('the road list marks what is in the plan', (tester) async {
      tall(tester);
      await tester.pumpWidget(
        MaterialApp(
          theme: AppTokens.light,
          home: DiscoveryScreen(
            discovery: Stream.value(
              legWith([
                PlannedOnTheWay(
                  plannedRow(5, 'Mawkdok Dympep viewpoint', osmId: 'node/77'),
                ),
              ]),
            ),
            onOpen: (_) {},
          ),
        ),
      );
      await tester.pump();
      expect(find.text('IN THE PLAN'), findsOneWidget);
      expect(
        find.descendant(of: find.byType(Wrap), matching: find.text('SIGHTS')),
        findsOneWidget,
      );
    });

    testWidgets('the typed stop needs a name, and a location that reads', (
      tester,
    ) async {
      PlannedStopInput? result;
      await tester.pumpWidget(
        MaterialApp(
          theme: AppTokens.light,
          home: Builder(
            builder: (context) => TextButton(
              onPressed: () async => result = await showPlannedStopDialog(
                context,
                legName: 'Shillong → Sohra',
              ),
              child: const Text('open'),
            ),
          ),
        ),
      );
      await tester.tap(find.text('open'));
      await tester.pumpAndSettle();

      await tester.tap(find.byKey(const Key('planned-add')));
      await tester.pump();
      expect(find.text('A stop needs a name.'), findsOneWidget);

      await tester.enterText(
        find.byKey(const Key('planned-name')),
        'Nohkalikai Falls',
      );
      await tester.enterText(
        find.byKey(const Key('planned-location')),
        'https://maps.app.goo.gl/x',
      );
      await tester.tap(find.byKey(const Key('planned-add')));
      await tester.pump();
      expect(find.textContaining('short link'), findsOneWidget);

      await tester.enterText(
        find.byKey(const Key('planned-location')),
        '25.2743, 91.6872',
      );
      await tester.enterText(find.byKey(const Key('planned-note')), 'Sunset');
      await tester.tap(find.byKey(const Key('planned-add')));
      await tester.pumpAndSettle();
      expect(result?.name, 'Nohkalikai Falls');
      expect(result?.lat, 25.2743);
      expect(result?.note, 'Sunset');
    });
  });
}
