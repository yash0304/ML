// test/trip_map_screen_test.dart
//
// The screen that shows the downloaded map.
//
// It exists because `TripMap` was built at #24 and wired to nothing: the app
// could fetch 346 MB of tiles, count them correctly, and offer no way to look
// at them. These tests are mostly about the states where there is nothing to
// draw, because those are the ones that were previously silent.

import 'dart:io';
import 'dart:typed_data';

import 'package:drift/native.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:safarsathi/core/database/app_database.dart';
import 'package:safarsathi/core/theme/app_tokens.dart';
import 'package:safarsathi/features/discovery/data/geo.dart';
import 'package:safarsathi/features/map/data/tile_math.dart';
import 'package:safarsathi/features/map/data/tile_provider.dart';
import 'package:safarsathi/features/map/data/tile_store.dart';
import 'package:safarsathi/features/map/presentation/trip_map_screen.dart';
import 'package:safarsathi/features/trips/data/trip_editor.dart';

Widget wrap(Widget child) => MaterialApp(theme: AppTokens.light, home: child);

void main() {
  late AppDatabase db;
  late Directory root;
  late TileStore store;

  setUp(() async {
    db = AppDatabase(NativeDatabase.memory());
    root = await Directory.systemTemp.createTemp('trip-map');
    store = TileStore(db: db, root: root);
  });
  tearDown(() async {
    await db.close();
    if (root.existsSync()) root.deleteSync(recursive: true);
  });

  group('reading what to draw', () {
    test('stops without coordinates are left out, not drawn at zero', () async {
      final editor = TripEditor(db);
      final tripId = await editor.createTrip(name: 'Meghalaya');
      await editor.addStop(
        tripId,
        const StopDraft(name: 'Shillong', lat: 25.57, lon: 91.88),
      );
      // Never located. Drawing this at (0, 0) would put a marker in the
      // Atlantic and a route line across Africa.
      await editor.addStop(tripId, const StopDraft(name: 'Sohra'));

      final view = await readTripMap(db, tripId, providerId: 'p');
      expect(view.stops, hasLength(1));
      expect(view.stops.single.name, 'Shillong');
      expect(view.route, isEmpty);
      expect(view.hasTiles, isFalse);
    });

    test('counts only this provider\'s tiles', () async {
      final editor = TripEditor(db);
      final tripId = await editor.createTrip(name: 'Meghalaya');

      await store.write('maptiler-outdoor-v2', const TileCoordinate(12, 1, 1),
          Uint8List.fromList([1]));
      await store.write('some-other-provider', const TileCoordinate(12, 2, 2),
          Uint8List.fromList([1]));

      final view = await readTripMap(
        db,
        tripId,
        providerId: 'maptiler-outdoor-v2',
      );
      expect(view.tileCount, 1);
      expect(view.hasTiles, isTrue);
    });
  });

  group('the screen', () {
    Widget screen(TripMapView view, {VoidCallback? onDownload}) => wrap(
      TripMapScreen(
        provider: const MapTilerRaster(apiKey: 'test-key'),
        store: store,
        load: () async => view,
        onDownload: onDownload,
      ),
    );

    testWidgets('WITH NOTHING DOWNLOADED IT SAYS SO, AND OFFERS THE FIX', (
      tester,
    ) async {
      var asked = 0;
      await tester.pumpWidget(
        screen(
          const TripMapView(route: [], stops: [], tileCount: 0),
          onDownload: () => asked++,
        ),
      );
      await tester.pumpAndSettle();

      expect(find.textContaining('No tiles downloaded yet'), findsOneWidget);
      await tester.tap(find.text('DOWNLOAD THE MAP'));
      await tester.pump();
      expect(asked, 1);
    });

    testWidgets('with tiles it reports the count and the honest limit', (
      tester,
    ) async {
      await tester.pumpWidget(
        screen(
          const TripMapView(
            route: [],
            stops: [(name: 'Shillong', at: LatLng(25.57, 91.88))],
            tileCount: 3424,
          ),
        ),
      );
      await tester.pumpAndSettle();

      expect(find.textContaining('3424 tiles on this phone'), findsOneWidget);
      // The promise the whole app rests on, restated where the map is.
      expect(
        find.textContaining('Nothing here asks the network'),
        findsOneWidget,
      );
      // And what it does at the edge, rather than a silent grey.
      expect(find.textContaining('goes blank'), findsOneWidget);
      expect(find.text('DOWNLOAD THE MAP'), findsNothing);
    });

    testWidgets('a build with no key says that instead of drawing grey', (
      tester,
    ) async {
      await tester.pumpWidget(
        wrap(
          TripMapScreen(
            provider: const MapTilerRaster(apiKey: ''),
            store: store,
            load: () async =>
                const TripMapView(route: [], stops: [], tileCount: 0),
          ),
        ),
      );
      await tester.pumpAndSettle();
      expect(find.textContaining('Maps are off'), findsOneWidget);
    });
  });
}
