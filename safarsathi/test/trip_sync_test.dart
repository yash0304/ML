// test/trip_sync_test.dart — issue #25.
//
// Every network client is faked, so this exercises the orchestration itself:
// what gets planned, in what order, and what happens when one piece fails.

import 'dart:io';

import 'package:drift/drift.dart' hide isNull, isNotNull;
import 'package:drift/native.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:safarsathi/core/database/app_database.dart';
import 'package:safarsathi/features/discovery/data/corridor_sync.dart';
import 'package:safarsathi/features/discovery/data/geo.dart';
import 'package:safarsathi/features/discovery/data/osrm_client.dart';
import 'package:safarsathi/features/discovery/data/overpass_client.dart';
import 'package:safarsathi/features/discovery/data/polyline.dart';
import 'package:safarsathi/features/map/data/map_download.dart';
import 'package:safarsathi/features/map/data/tile_downloader.dart';
import 'package:safarsathi/features/map/data/tile_provider.dart';
import 'package:safarsathi/features/map/data/tile_store.dart';
import 'package:safarsathi/features/sync/data/trip_sync.dart';
import 'package:safarsathi/features/trips/data/trip_editor.dart';
import 'package:safarsathi/features/weather/data/weather_client.dart';
import 'package:safarsathi/features/weather/data/weather_sync.dart';

const shillong = LatLng(25.5788, 91.8933);
const cherrapunji = LatLng(25.2702, 91.7323);

String get _osrmBody =>
    '{"code":"Ok","routes":[{"geometry":'
    '"${Polyline.encode(const [shillong, cherrapunji])}",'
    '"distance":54000.0,"duration":5400.0}]}';

const _overpassBody = '''
{"elements":[{"type":"node","id":1,"lat":25.42,"lon":91.81,
  "tags":{"amenity":"restaurant","name":"Dhaba","phone":"+91 364 111 1111"}}]}
''';

const _forecastBody = '''
{"daily":{"time":["2026-10-01"],"weather_code":[61],
 "temperature_2m_max":[24.1],"temperature_2m_min":[17.2],
 "precipitation_sum":[12.4]}}
''';

void main() {
  late AppDatabase db;
  late Directory root;
  late TileStore store;
  late TripEditor editor;
  late int tripId;

  setUp(() async {
    db = AppDatabase(NativeDatabase.memory());
    root = await Directory.systemTemp.createTemp('sync');
    store = TileStore(db: db, root: root);
    editor = TripEditor(db);

    tripId = await editor.createTrip(name: 'Meghalaya');
    await editor.addStop(
      tripId,
      const StopDraft(name: 'Shillong', lat: 25.5788, lon: 91.8933),
    );
    await editor.addStop(
      tripId,
      const StopDraft(name: 'Cherrapunji', lat: 25.2702, lon: 91.7323),
    );
  });

  tearDown(() async {
    await db.close();
    if (root.existsSync()) await root.delete(recursive: true);
  });

  TripSync syncWith({
    String? osrm,
    Future<String> Function(String)? overpass,
    Future<String> Function(Uri)? forecast,
    Future<Uint8List?> Function(String)? tiles,
    MapTileProvider provider = const MapTilerRaster(apiKey: 'test-key'),
  }) {
    final downloader = TileDownloader(
      store: store,
      provider: provider,
      delay: Duration.zero,
      fetch: tiles ?? (_) async => Uint8List.fromList(List.filled(16, 2)),
    );

    return TripSync(
      db: db,
      corridor: CorridorSync(
        db: db,
        osrm: OsrmClient(fetch: (_) async => osrm ?? _osrmBody),
        overpass: OverpassClient(
          fetch: overpass ?? (_) async => _overpassBody,
        ),
      ),
      weather: WeatherSync(
        db: db,
        client: WeatherClient(
          fetch: forecast ?? (_) async => _forecastBody,
        ),
      ),
      map: MapDownload(db: db, downloader: downloader),
    );
  }

  group('planning', () {
    test('one corridor task per leg, one weather per stop, one map', () async {
      final plan = await syncWith().plan(tripId);

      expect(plan.countOf(SyncKind.corridor), 1);
      expect(plan.countOf(SyncKind.weather), 2);
      expect(plan.countOf(SyncKind.tiles), 1);
      expect(plan.total, 4);
    });

    test('ROUTE AND PLACES ARE ONE TASK, NOT TWO', () async {
      // CorridorSync.syncLeg routes the leg and queries its box in a single
      // call and writes both in one transaction. Two rows would report
      // progress that corresponds to no work.
      final plan = await syncWith().plan(tripId);
      expect(
        plan.tasks.where((t) => t.kind == SyncKind.corridor).length,
        1,
      );
    });

    test('EVERY CORRIDOR RUNS BEFORE THE MAP', () async {
      // Tiles derive their box from the route. With no polyline that falls
      // back to the straight line, which in these hills is another valley.
      await editor.addStop(
        tripId,
        const StopDraft(name: 'Dawki', lat: 25.1932, lon: 92.0207),
      );
      final plan = await syncWith().plan(tripId);

      final lastCorridor = plan.tasks.lastIndexWhere(
        (t) => t.kind == SyncKind.corridor,
      );
      final tiles = plan.tasks.indexWhere((t) => t.kind == SyncKind.tiles);
      expect(tiles, greaterThan(lastCorridor));
    });

    test('a task names what it is working on', () async {
      final plan = await syncWith().plan(tripId);
      final corridor = plan.tasks.firstWhere(
        (t) => t.kind == SyncKind.corridor,
      );
      expect(corridor.subject, 'Shillong → Cherrapunji');
      expect(corridor.label, contains('Shillong → Cherrapunji'));
    });

    test('legs and stops without coordinates are counted, not skipped quietly',
        () async {
      final stops = await editor.stopsOf(tripId);
      await (db.update(db.stops)..where((s) => s.id.equals(stops.last.id)))
          .write(const StopsCompanion(lat: Value(null), lon: Value(null)));

      final plan = await syncWith().plan(tripId);
      expect(plan.legsWithoutCoordinates, 1);
      expect(plan.stopsWithoutCoordinates, 1);
      expect(plan.countOf(SyncKind.corridor), 0);
      expect(plan.countOf(SyncKind.tiles), 0);
      // The one stop that does have coordinates still gets its weather.
      expect(plan.countOf(SyncKind.weather), 1);
    });

    test('a trip with nothing usable plans nothing', () async {
      await db.update(db.stops).write(
        const StopsCompanion(lat: Value(null), lon: Value(null)),
      );
      final plan = await syncWith().plan(tripId);
      expect(plan.isEmpty, isTrue);
    });
  });

  group('running', () {
    test('one press does routes, places, weather and tiles', () async {
      final progress = await syncWith().run(tripId).last;

      expect(progress.isDone, isTrue);
      expect(progress.hadFailures, isFalse);

      final leg = await db.select(db.legs).getSingle();
      expect(leg.routePolyline, isNotNull);
      expect(leg.lastSyncedAt, isNotNull);
      expect(await db.select(db.pois).get(), isNotEmpty);
      expect(await db.select(db.weatherSnapshots).get(), isNotEmpty);
      expect((await store.usage()).count, greaterThan(0));
    });

    test('progress names the item, and rises to the total', () async {
      final seen = await syncWith().run(tripId).toList();

      expect(seen.first.done, 0);
      expect(seen.last.done, seen.last.total);
      expect(
        seen.any((p) => p.current?.subject == 'Shillong → Cherrapunji'),
        isTrue,
      );
      for (var i = 1; i < seen.length; i++) {
        expect(seen[i].done, greaterThanOrEqualTo(seen[i - 1].done));
      }
    });

    test('ONE FAILURE DOES NOT STOP THE RUN', () async {
      // Overpass being busy must not cost the tiles or the forecast. A sync
      // that silently does 60% is worse than one that says which 40% failed.
      final progress = await syncWith(
        overpass: (_) async =>
            throw const OverpassException('OpenStreetMap is busy right now.'),
      ).run(tripId).last;

      expect(progress.isDone, isTrue);
      expect(progress.failures.length, 1);
      expect(progress.failures.single.task.kind, SyncKind.corridor);
      expect(progress.failures.single.reason, contains('busy'));

      // The rest still happened.
      expect(await db.select(db.weatherSnapshots).get(), isNotEmpty);
      expect((await store.usage()).count, greaterThan(0));
    });

    test('a failure carries the task it belongs to', () async {
      final progress = await syncWith(
        forecast: (_) async =>
            throw const WeatherException('The forecast service is down.'),
      ).run(tripId).last;

      expect(progress.failures.length, 2, reason: 'one per stop');
      expect(
        progress.failures.every((f) => f.kind == SyncKind.weather),
        isTrue,
      );
      // The corridor still ran.
      expect(await db.select(db.pois).get(), isNotEmpty);
    });

    test('a tile failure is one failure, not one per tile', () async {
      final progress = await syncWith(
        provider: const MapTilerRaster(apiKey: ''),
      ).run(tripId).last;

      expect(progress.failures.length, 1);
      expect(progress.failures.single.task.kind, SyncKind.tiles);
    });

    test('RE-RUNNING AFTER A FAILURE FETCHES ONLY WHAT IS MISSING', () async {
      // Nothing tracks its own position: every underlying step skips work
      // already done, which cannot get out of step with what is on disk.
      await syncWith(
        overpass: (_) async => throw const OverpassException('busy'),
      ).run(tripId).last;

      final tilesAfterFirst = (await store.usage()).count;
      expect(tilesAfterFirst, greaterThan(0));

      var tileCalls = 0;
      final second = await syncWith(
        tiles: (_) async {
          tileCalls++;
          return Uint8List.fromList([1]);
        },
      ).run(tripId).last;

      expect(second.hadFailures, isFalse);
      expect(tileCalls, 0, reason: 'tiles were already on disk');
      expect(await db.select(db.pois).get(), isNotEmpty);
    });

    test('an empty plan runs to completion immediately', () async {
      await db.update(db.stops).write(
        const StopsCompanion(lat: Value(null), lon: Value(null)),
      );
      final progress = await syncWith().run(tripId).last;

      expect(progress.total, 0);
      expect(progress.isDone, isTrue);
      expect(progress.fraction, 1.0);
    });

    test('the estimate speaks in words, not false precision', () async {
      final map = MapDownload(
        db: db,
        downloader: TileDownloader(
          store: store,
          provider: const MapTilerRaster(apiKey: 'k'),
          delay: Duration.zero,
          fetch: (_) async => Uint8List.fromList([1]),
        ),
      );

      final before = await estimateSyncSize(map, tripId);
      expect(before, contains('nearly all of it map'));

      await syncWith().run(tripId).last;
      expect(await estimateSyncSize(map, tripId), contains('already here'));
    });
  });
}

extension on SyncFailure {
  SyncKind get kind => task.kind;
}
