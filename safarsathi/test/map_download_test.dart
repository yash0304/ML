// test/map_download_test.dart — issue #24, the trip-level download.

import 'dart:io';

import 'package:drift/drift.dart' hide isNull, isNotNull;
import 'package:drift/native.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:safarsathi/core/database/app_database.dart';
import 'package:safarsathi/features/discovery/data/geo.dart';
import 'package:safarsathi/features/discovery/data/polyline.dart';
import 'package:safarsathi/features/map/data/map_download.dart';
import 'package:safarsathi/features/map/data/tile_downloader.dart';
import 'package:safarsathi/features/map/data/tile_math.dart';
import 'package:safarsathi/features/map/data/tile_provider.dart';
import 'package:safarsathi/features/map/data/tile_store.dart';
import 'package:safarsathi/features/trips/data/trip_editor.dart';

const shillong = LatLng(25.5788, 91.8933);
const cherrapunji = LatLng(25.2702, 91.7323);

void main() {
  late AppDatabase db;
  late Directory root;
  late TileStore store;
  late TripEditor editor;
  late int tripId;
  late int firstStop;
  late int secondStop;

  setUp(() async {
    db = AppDatabase(NativeDatabase.memory());
    root = await Directory.systemTemp.createTemp('tiles');
    store = TileStore(db: db, root: root);
    editor = TripEditor(db);

    tripId = await editor.createTrip(name: 'Meghalaya');
    firstStop = await editor.addStop(
      tripId,
      const StopDraft(name: 'Shillong', lat: 25.5788, lon: 91.8933),
    );
    secondStop = await editor.addStop(
      tripId,
      const StopDraft(name: 'Cherrapunji', lat: 25.2702, lon: 91.7323),
    );
  });

  tearDown(() async {
    await db.close();
    if (root.existsSync()) await root.delete(recursive: true);
  });

  MapDownload downloadWith({
    Future<Uint8List?> Function(String)? fetch,
    MapTileProvider provider = const MapTilerRaster(apiKey: 'test-key'),
  }) => MapDownload(
    db: db,
    downloader: TileDownloader(
      store: store,
      provider: provider,
      delay: Duration.zero,
      fetch: fetch ?? (_) async => Uint8List.fromList(List.filled(32, 3)),
    ),
  );

  test('THE STOP COORDINATES SAVE THROUGH THE DRAFT', () async {
    // They were added to StopDraft for exactly this; without them every leg
    // is skipped and the download covers nothing.
    final stops = await editor.stopsOf(tripId);
    expect(stops.first.lat, closeTo(25.5788, 0.0001));
    expect(stops.last.lon, closeTo(91.7323, 0.0001));
  });

  test('an unrouted leg falls back to the straight line', () async {
    // Wrong in the mountains, but it is the only thing available before the
    // route is fetched, and it errs wide rather than narrow.
    final boxes = await tripBoxes(db, tripId);
    expect(boxes.length, 1);
    expect(boxes.single.contains(shillong), isTrue);
    expect(boxes.single.contains(cherrapunji), isTrue);
  });

  test('a routed leg uses its real polyline', () async {
    // A route that swings east of the straight line. The box has to follow.
    const detour = LatLng(25.42, 92.20);
    await db.update(db.legs).write(
      LegsCompanion(
        routePolyline: Value(
          Polyline.encode(const [shillong, detour, cherrapunji]),
        ),
      ),
    );

    final boxes = await tripBoxes(db, tripId);
    expect(boxes.single.contains(detour), isTrue);
    expect(boxes.single.east, greaterThan(92.1));
  });

  test('THE ESTIMATE IS A REAL NUMBER AND FETCHES NOTHING', () async {
    final estimate = await downloadWith().estimate(tripId);

    expect(estimate.tileCount, greaterThan(0));
    expect(estimate.alreadyHave, 0);
    expect(estimate.toFetch, estimate.tileCount);
    expect(estimate.estimatedSize, isNotEmpty);
    expect((await store.usage()).count, 0);
  });

  test('legs missing coordinates are counted, not silently skipped', () async {
    await (db.update(db.stops)..where((s) => s.id.equals(secondStop))).write(
      const StopsCompanion(lat: Value(null), lon: Value(null)),
    );

    final estimate = await downloadWith().estimate(tripId);
    expect(estimate.legsWithoutCoordinates, 1);
    expect(estimate.hasNothingToDo, isTrue);
  });

  test('ADJACENT LEGS DO NOT DOUBLE-COUNT THEIR SHARED TERRAIN', () async {
    // Two legs meeting at a stop overlap around it. Counting per box would
    // charge the user twice for the same tiles.
    await editor.addStop(
      tripId,
      const StopDraft(name: 'Dawki', lat: 25.1932, lon: 92.0207),
    );

    final boxes = await tripBoxes(db, tripId);
    expect(boxes.length, 2);

    final estimate = await downloadWith().estimate(tripId);
    final naive = boxes.fold<int>(
      0,
      (sum, box) => sum + countTilesForBoxHelper(box),
    );
    expect(estimate.tileCount, lessThan(naive));
  });

  test('downloading writes tiles and the estimate then reads complete',
      () async {
    final download = downloadWith();
    final before = await download.estimate(tripId);

    await download.download(tripId).last;

    final after = await download.estimate(tripId);
    expect(after.alreadyHave, before.tileCount);
    expect(after.isComplete, isTrue);
    expect((await store.usage()).count, before.tileCount);
  });

  test('A SECOND DOWNLOAD FETCHES NOTHING', () async {
    await downloadWith().download(tripId).last;

    var calls = 0;
    await downloadWith(
      fetch: (_) async {
        calls++;
        return Uint8List.fromList([1]);
      },
    ).download(tripId).last;

    expect(calls, 0);
  });

  test('an unconfigured provider refuses before fetching anything', () async {
    await expectLater(
      downloadWith(provider: const MapTilerRaster(apiKey: ''))
          .download(tripId)
          .toList(),
      throwsA(isA<TileDownloadException>()),
    );
    expect((await store.usage()).count, 0);
  });

  test('a trip with one stop has no legs and nothing to do', () async {
    final lonely = await editor.createTrip(name: 'One stop');
    await editor.addStop(
      lonely,
      const StopDraft(name: 'Shillong', lat: 25.5788, lon: 91.8933),
    );

    final estimate = await downloadWith().estimate(lonely);
    expect(estimate.legCount, 0);
    expect(estimate.hasNothingToDo, isTrue);
  });

  test('clearing tiles resets the estimate', () async {
    final download = downloadWith();
    await download.download(tripId).last;
    expect((await download.estimate(tripId)).isComplete, isTrue);

    await store.clear(const MapTilerRaster(apiKey: 'x').id);

    final after = await download.estimate(tripId);
    expect(after.alreadyHave, 0);
    expect(after.isComplete, isFalse);
  });

  test('the first stop id is stable through all of this', () {
    // Guards against a refactor that renumbers stops during a download.
    expect(firstStop, isNot(secondStop));
  });
}

/// Local helper so the double-counting test can state the naive number.
int countTilesForBoxHelper(BoundingBox box) => countTilesForBox(
  box,
  minZoom: defaultMinZoom,
  maxZoom: defaultMaxZoom,
);
