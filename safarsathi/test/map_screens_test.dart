// test/map_screens_test.dart — issue #24, the screens.

import 'dart:io';

import 'package:drift/native.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:safarsathi/core/database/app_database.dart';
import 'package:safarsathi/core/theme/app_tokens.dart';
import 'package:safarsathi/features/discovery/data/geo.dart';
import 'package:safarsathi/features/map/data/map_download.dart';
import 'package:safarsathi/features/map/data/tile_downloader.dart';
import 'package:safarsathi/features/map/data/tile_provider.dart';
import 'package:safarsathi/features/map/data/tile_store.dart';
import 'package:safarsathi/features/map/presentation/map_download_screen.dart';
import 'package:safarsathi/features/map/presentation/trip_map.dart';

Widget wrap(Widget child, {Brightness brightness = Brightness.light}) =>
    MaterialApp(
      theme: brightness == Brightness.light ? AppTokens.light : AppTokens.dark,
      home: Scaffold(body: child),
    );

// Distinct numbers throughout, so a finder cannot match the wrong figure.
const _estimate = MapEstimate(
  legCount: 4,
  legsWithoutCoordinates: 0,
  tileCount: 862,
  alreadyHave: 120,
);

void main() {
  late AppDatabase db;
  late Directory root;
  late TileStore store;

  setUp(() async {
    db = AppDatabase(NativeDatabase.memory());
    root = await Directory.systemTemp.createTemp('tiles');
    store = TileStore(db: db, root: root);
  });

  tearDown(() async {
    await db.close();
    if (root.existsSync()) await root.delete(recursive: true);
  });

  group('the map widget', () {
    testWidgets('NO KEY SAYS SO, RATHER THAN SHOWING A GREY RECTANGLE', (
      tester,
    ) async {
      await tester.pumpWidget(
        wrap(
          TripMap(
            provider: const MapTilerRaster(apiKey: ''),
            store: store,
          ),
        ),
      );

      expect(find.text('Maps are off in this build'), findsOneWidget);
      expect(find.textContaining('MAPTILER_KEY'), findsOneWidget);
    });

    testWidgets('no tiles downloaded says what to do about it', (tester) async {
      await tester.pumpWidget(
        wrap(
          TripMap(
            provider: const MapTilerRaster(apiKey: 'test-key'),
            store: store,
            hasTiles: false,
          ),
        ),
      );

      expect(
        find.text('Nothing downloaded for this trip yet'),
        findsOneWidget,
      );
      expect(find.textContaining('no signal at all'), findsOneWidget);
    });

    testWidgets('ATTRIBUTION RENDERS ON A DRAWN MAP', (tester) async {
      // Required by MapTiler's terms and OpenStreetMap's licence. Not a
      // styling choice, so it is asserted rather than assumed.
      await tester.pumpWidget(
        wrap(
          TripMap(
            provider: const MapTilerRaster(apiKey: 'test-key'),
            store: store,
            route: const [LatLng(25.5788, 91.8933), LatLng(25.2702, 91.7323)],
          ),
        ),
      );
      await tester.pump();

      expect(
        find.text('© MapTiler © OpenStreetMap contributors'),
        findsOneWidget,
      );
    });

    testWidgets('a notice carries no attribution, having no map', (
      tester,
    ) async {
      await tester.pumpWidget(
        wrap(
          TripMap(provider: const MapTilerRaster(apiKey: ''), store: store),
        ),
      );
      expect(find.textContaining('© MapTiler'), findsNothing);
    });
  });

  group('the download screen', () {
    Widget screen({
      MapTileProvider provider = const MapTilerRaster(apiKey: 'test-key'),
      MapEstimate estimate = _estimate,
      Stream<TileProgress> Function()? download,
      ({int count, int bytes}) usage = (count: 0, bytes: 0),
      Future<void> Function()? onClear,
    }) => wrap(
      MapDownloadScreen(
        provider: provider,
        estimate: () async => estimate,
        download: download ?? () => const Stream<TileProgress>.empty(),
        usage: Stream.value(usage),
        onClear: onClear ?? () async {},
      ),
    );

    testWidgets('THE BUTTON STATES THE COUNT AND THE SIZE UP FRONT', (
      tester,
    ) async {
      // An app that starts pulling data on someone's hotel WiFi is an app
      // they stop trusting the moment they notice.
      await tester.pumpWidget(screen());
      await tester.pumpAndSettle();

      expect(find.textContaining('Download 742 tiles'), findsOneWidget);
      expect(find.text('862'), findsOneWidget, reason: 'total');
      expect(find.text('120'), findsOneWidget, reason: 'already have');
      expect(find.text('742'), findsOneWidget, reason: 'to fetch');
      expect(find.textContaining('is an estimate'), findsOneWidget);
    });

    testWidgets('nothing downloads until the button is pressed', (
      tester,
    ) async {
      var started = false;
      await tester.pumpWidget(
        screen(
          download: () {
            started = true;
            return const Stream<TileProgress>.empty();
          },
        ),
      );
      await tester.pumpAndSettle();
      expect(started, isFalse);
    });

    testWidgets('pressing it reports progress', (tester) async {
      await tester.pumpWidget(
        screen(
          download: () => Stream.fromIterable(const [
            TileProgress(done: 0, total: 3),
            TileProgress(done: 2, total: 3),
            TileProgress(done: 3, total: 3),
          ]),
        ),
      );
      await tester.pumpAndSettle();

      await tester.tap(find.textContaining('Download 742 tiles'));
      await tester.pumpAndSettle();

      expect(find.text('3 of 3'), findsOneWidget);
    });

    testWidgets('AN ALREADY-COMPLETE TRIP OFFERS NOTHING', (tester) async {
      await tester.pumpWidget(
        screen(
          estimate: const MapEstimate(
            legCount: 4,
            legsWithoutCoordinates: 0,
            tileCount: 862,
            alreadyHave: 862,
          ),
        ),
      );
      await tester.pumpAndSettle();
      expect(find.text('Already downloaded'), findsOneWidget);
    });

    testWidgets('no key disables the button and explains', (tester) async {
      await tester.pumpWidget(
        screen(provider: const MapTilerRaster(apiKey: '')),
      );
      await tester.pumpAndSettle();

      expect(find.text('No map provider'), findsOneWidget);
      expect(find.textContaining('MAPTILER_KEY'), findsOneWidget);
    });

    testWidgets('LEGS WITHOUT COORDINATES ARE NAMED, NOT SKIPPED SILENTLY', (
      tester,
    ) async {
      await tester.pumpWidget(
        screen(
          estimate: const MapEstimate(
            legCount: 4,
            legsWithoutCoordinates: 2,
            tileCount: 400,
            alreadyHave: 0,
          ),
        ),
      );
      await tester.pumpAndSettle();
      expect(find.textContaining('2 legs are skipped'), findsOneWidget);
    });

    testWidgets('a trip with no coordinates at all says what to fix', (
      tester,
    ) async {
      await tester.pumpWidget(
        screen(
          estimate: const MapEstimate(
            legCount: 3,
            legsWithoutCoordinates: 3,
            tileCount: 0,
            alreadyHave: 0,
          ),
        ),
      );
      await tester.pumpAndSettle();

      expect(find.textContaining('Set them on each stop first'), findsOneWidget);
      expect(find.text('Nothing to download yet'), findsOneWidget);
    });

    testWidgets('A FAILURE SAYS WHAT SURVIVED IT', (tester) async {
      await tester.pumpWidget(
        screen(
          download: () => Stream<TileProgress>.error(
            const TileDownloadException('The provider rejected the key.'),
          ),
        ),
      );
      await tester.pumpAndSettle();

      await tester.tap(find.textContaining('Download 742 tiles'));
      await tester.pumpAndSettle();

      expect(find.textContaining('rejected the key'), findsOneWidget);
      expect(find.textContaining('carry on from there'), findsOneWidget);
    });

    testWidgets('the cache line states tiles and bytes', (tester) async {
      await tester.pumpWidget(
        screen(usage: (count: 500, bytes: 21 * 1024 * 1024)),
      );
      await tester.pumpAndSettle();
      expect(find.text('500 tiles · 21 MB'), findsOneWidget);
    });

    testWidgets('it says the map never asks the network again', (tester) async {
      await tester.pumpWidget(screen());
      await tester.pumpAndSettle();
      expect(
        find.textContaining('never asks the network again'),
        findsOneWidget,
      );
    });

    testWidgets('lays out at phone width in both themes', (tester) async {
      tester.view.physicalSize = const Size(400, 800);
      tester.view.devicePixelRatio = 1.0;
      addTearDown(tester.view.reset);

      for (final b in Brightness.values) {
        await tester.pumpWidget(
          wrap(
            MapDownloadScreen(
              provider: const MapTilerRaster(apiKey: 'test-key'),
              estimate: () async => _estimate,
              download: () => const Stream<TileProgress>.empty(),
              usage: Stream.value((count: 12, bytes: 4096)),
              onClear: () async {},
            ),
            brightness: b,
          ),
        );
        await tester.pumpAndSettle();
        expect(tester.takeException(), isNull);
      }
    });
  });
}
