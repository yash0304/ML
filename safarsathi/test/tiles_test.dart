// test/tiles_test.dart — issue #24.
//
// The maths, the provider abstraction, the store and the downloader. Nothing
// here reaches the network, and nothing here needs a key.

import 'dart:io';
import 'dart:typed_data';

import 'package:drift/native.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:safarsathi/core/database/app_database.dart';
import 'package:safarsathi/features/discovery/data/geo.dart';
import 'package:safarsathi/features/map/data/tile_downloader.dart';
import 'package:safarsathi/features/map/data/tile_math.dart';
import 'package:safarsathi/features/map/data/tile_provider.dart';
import 'package:safarsathi/features/map/data/tile_store.dart';

const shillong = LatLng(25.5788, 91.8933);

void main() {
  group('tile maths', () {
    test('a known coordinate lands on a known tile', () {
      // Worked through by hand against the standard formula:
      //   x = floor((91.8933 + 180) / 360 · 4096) = 3093
      //   y = floor((1 − ln(tan φ + sec φ)/π) / 2 · 4096) = 1746
      expect(tileFor(shillong, 12), const TileCoordinate(12, 3093, 1746));
    });

    test('zoom 0 is one tile for the whole world', () {
      expect(tileFor(shillong, 0), const TileCoordinate(0, 0, 0));
      expect(tileFor(const LatLng(-40, -170), 0), const TileCoordinate(0, 0, 0));
    });

    test('the north-west corner round-trips back to its own tile', () {
      const tile = TileCoordinate(12, 2988, 1706);
      expect(tileFor(tileNorthWest(tile), 12), tile);
    });

    test('LATITUDE CLAMPS AT THE MERCATOR LIMIT', () {
      // Web Mercator is undefined at the poles: the tangent blows up and y
      // goes to infinity. Every tile scheme cuts off at 85.05.
      final north = tileFor(const LatLng(89.9, 0), 4);
      final south = tileFor(const LatLng(-89.9, 0), 4);

      expect(north.y, 0);
      expect(south.y, 15);
      expect(north.y, isNot(isNaN));
    });

    test('a point on the far edge stays inside the grid', () {
      final tile = tileFor(const LatLng(-85.05112878, 180), 3);
      expect(tile.x, lessThanOrEqualTo(7));
      expect(tile.y, lessThanOrEqualTo(7));
    });

    test('a box yields every tile covering it', () {
      const box = BoundingBox(south: 25.5, west: 91.8, north: 25.6, east: 91.9);
      final tiles = tilesForBox(box, minZoom: 12, maxZoom: 12);

      expect(tiles, isNotEmpty);
      expect(tiles.every((t) => t.z == 12), isTrue);
      expect(tiles.toSet().length, tiles.length, reason: 'no duplicates');
      // Every corner of the box is covered.
      for (final corner in [
        const LatLng(25.5, 91.8),
        const LatLng(25.6, 91.9),
        const LatLng(25.5, 91.9),
      ]) {
        expect(tiles, contains(tileFor(corner, 12)));
      }
    });

    test('TILE COUNT GROWS AS 4^z', () {
      // The reason the zoom band is capped and the count is shown before
      // anything downloads.
      const box = BoundingBox(south: 25.2, west: 91.7, north: 25.6, east: 92.0);

      final at12 = countTilesForBox(box, minZoom: 12, maxZoom: 12);
      final at14 = countTilesForBox(box, minZoom: 14, maxZoom: 14);
      final at18 = countTilesForBox(box, minZoom: 18, maxZoom: 18);

      expect(at14, greaterThan(at12 * 10));
      expect(at18, greaterThan(at14 * 100));
    });

    test('counting matches generating', () {
      const box = BoundingBox(south: 25.2, west: 91.7, north: 25.6, east: 92.0);
      expect(
        countTilesForBox(box, minZoom: 11, maxZoom: 13),
        tilesForBox(box, minZoom: 11, maxZoom: 13).length,
      );
    });

    test('an inverted zoom range is empty, not a crash', () {
      const box = BoundingBox(south: 25.5, west: 91.8, north: 25.6, east: 91.9);
      expect(tilesForBox(box, minZoom: 14, maxZoom: 12), isEmpty);
      expect(countTilesForBox(box, minZoom: 14, maxZoom: 12), 0);
    });

    test('byte sizes read in words', () {
      expect(describeBytes(512), '512 B');
      expect(describeBytes(2048), '2 KB');
      expect(describeBytes(1536 * 1024), '1.5 MB');
      expect(describeBytes(42 * 1024 * 1024), '42 MB');
    });
  });

  group('the provider abstraction', () {
    test('A BUILD WITH NO KEY DISABLES THE MAP AND SAYS WHY', () {
      // Never a crash, never silent. A blank grey rectangle is
      // indistinguishable from a bug.
      const provider = MapTilerRaster(apiKey: '');

      expect(provider.isConfigured, isFalse);
      // The hint has to name the thing missing AND where to fix it.
      expect(provider.configurationHint, contains('MapTiler key'));
      expect(provider.configurationHint, contains('Settings'));
      expect(
        () => provider.urlFor(12, 3093, 1746),
        throwsA(isA<TileProviderNotConfigured>()),
      );
    });

    test('a configured provider builds a tile URL', () {
      const provider = MapTilerRaster(apiKey: 'test-key-not-real');
      final url = provider.urlFor(12, 3093, 1746);

      expect(url, contains('/12/3093/1746'));
      expect(url, contains('key=test-key-not-real'));
      expect(provider.isConfigured, isTrue);
    });

    test('ATTRIBUTION IS PART OF THE INTERFACE', () {
      // Every provider's terms require it. A map rendering without it is a
      // licence violation, not a styling choice, so a new implementation
      // cannot compile without supplying one.
      const provider = MapTilerRaster(apiKey: 'x');
      expect(provider.attribution, contains('MapTiler'));
      expect(provider.attribution, contains('OpenStreetMap'));
    });

    test('the id distinguishes styles, so caches cannot collide', () {
      const outdoor = MapTilerRaster(apiKey: 'x');
      const streets = MapTilerRaster(apiKey: 'x', style: 'streets-v2');
      expect(outdoor.id, isNot(streets.id));
    });

    test('the null provider is configured-false and throws on use', () {
      const provider = NoTileProvider();
      expect(provider.isConfigured, isFalse);
      expect(
        () => provider.urlFor(1, 1, 1),
        throwsA(isA<TileProviderNotConfigured>()),
      );
    });

    test('NO REAL KEY IS COMPILED INTO THE TEST BUILD', () {
      // Nothing passes --dart-define here, so the app's own provider must be
      // unconfigured. If this ever fails, a key reached a tracked file.
      expect(activeTileProvider.isConfigured, isFalse);
      expect(MapTilerRaster.buildTimeKey, isEmpty);
    });

    test('A KEY TYPED ON THE PHONE CONFIGURES THE PROVIDER', () {
      // The whole point of the runtime path: an APK built with no secret can
      // still be given a key by the person holding it.
      expect(tileProviderFor(null).isConfigured, isFalse);
      expect(tileProviderFor('').isConfigured, isFalse);
      expect(tileProviderFor('   ').isConfigured, isFalse);

      final typed = tileProviderFor('  abc123  ');
      expect(typed.isConfigured, isTrue);
      expect(typed.urlFor(9, 1, 2), contains('key=abc123'));
    });

    test('the cache id does not move when the key does', () {
      // A rotated key must not orphan a downloaded trip.
      expect(tileProviderFor('one').id, tileProviderFor('two').id);
      expect(tileProviderFor('one').id, activeTileProvider.id);
    });

    test('the hint sends the user somewhere they can act', () {
      expect(activeTileProvider.configurationHint, contains('Settings'));
    });
  });

  group('webp, without stranding what is already downloaded', () {
    late AppDatabase db;
    late Directory root;
    late TileStore store;

    setUp(() async {
      db = AppDatabase(NativeDatabase.memory());
      root = await Directory.systemTemp.createTemp('tiles-format');
      store = TileStore(db: db, root: root);
    });
    tearDown(() async {
      await db.close();
      if (root.existsSync()) root.deleteSync(recursive: true);
    });

    test('the provider now asks for webp', () {
      const provider = MapTilerRaster(apiKey: 'k');
      expect(provider.format, 'webp');
      expect(provider.urlFor(12, 1, 2), contains('@2x.webp'));
      expect(provider.urlFor(12, 1, 2), isNot(contains('.png')));
    });

    test('THE CACHE ID DOES NOT MOVE WITH THE FORMAT', () {
      // Folding the format into the id would strand every tile downloaded
      // before the switch — 346 MB of them, on a phone, a fortnight before
      // a trip.
      expect(
        const MapTilerRaster(apiKey: 'k', format: 'webp').id,
        const MapTilerRaster(apiKey: 'k', format: 'png').id,
      );
    });

    test('A TILE DOWNLOADED AS PNG IS STILL FOUND AFTER THE SWITCH', () async {
      const tile = TileCoordinate(12, 100, 200);
      // Written the old way, before this change existed.
      await store.write('maptiler-outdoor-v2', tile, Uint8List.fromList([1, 2]));

      // Read the new way, with webp preferred.
      expect(await store.has('maptiler-outdoor-v2', tile), isTrue);
      expect(await store.read('maptiler-outdoor-v2', tile), [1, 2]);
    });

    test('a webp tile wins when both happen to exist', () async {
      const tile = TileCoordinate(12, 100, 200);
      await store.write('p', tile, Uint8List.fromList([1]), format: 'png');
      await store.write('p', tile, Uint8List.fromList([2]), format: 'webp');
      expect(await store.read('p', tile), [2]);
    });

    test('new downloads land as webp', () async {
      const tile = TileCoordinate(12, 100, 200);
      await store.write('p', tile, Uint8List.fromList([9]), format: 'webp');
      expect(store.fileFor('p', tile, format: 'webp').existsSync(), isTrue);
      expect(store.fileFor('p', tile, format: 'png').existsSync(), isFalse);
    });

    test('a missing tile is still missing in either format', () async {
      expect(
        await store.has('p', const TileCoordinate(1, 1, 1)),
        isFalse,
      );
    });
  });

  group('the store', () {
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

    Uint8List bytes(int n) => Uint8List.fromList(List.filled(n, 7));

    test('a written tile reads back', () async {
      const tile = TileCoordinate(12, 2988, 1706);
      await store.write('maptiler', tile, bytes(120));

      expect(await store.has('maptiler', tile), isTrue);
      expect((await store.read('maptiler', tile))!.length, 120);
    });

    test('A MISSING TILE IS NULL, NOT AN ERROR', () async {
      // It renders blank. There is no network path in this class at all —
      // not a fallback, not a timeout — and that absence is the guarantee.
      expect(
        await store.read('maptiler', const TileCoordinate(12, 1, 1)),
        isNull,
      );
    });

    test('two providers do not see each other\'s tiles', () async {
      const tile = TileCoordinate(12, 2988, 1706);
      await store.write('maptiler', tile, bytes(100));

      expect(await store.has('stadia', tile), isFalse);
      expect((await store.usage('maptiler')).count, 1);
      expect((await store.usage('stadia')).count, 0);
    });

    test('usage counts tiles and bytes', () async {
      await store.write('maptiler', const TileCoordinate(12, 1, 1), bytes(100));
      await store.write('maptiler', const TileCoordinate(12, 1, 2), bytes(250));

      final usage = await store.usage();
      expect(usage.count, 2);
      expect(usage.bytes, 350);
    });

    test('rewriting a tile does not double-count it', () async {
      const tile = TileCoordinate(12, 1, 1);
      await store.write('maptiler', tile, bytes(100));
      await store.write('maptiler', tile, bytes(180));

      final usage = await store.usage();
      expect(usage.count, 1);
      expect(usage.bytes, 180);
    });

    test('CLEARING REMOVES THE FILES, NOT JUST THE INDEX', () async {
      // Clearing the index alone would leave hundreds of megabytes on the
      // phone reporting as zero, which is the worst possible answer to
      // "clear the cache".
      const tile = TileCoordinate(12, 1, 1);
      await store.write('maptiler', tile, bytes(100));
      final file = store.fileFor('maptiler', tile);
      expect(file.existsSync(), isTrue);

      await store.clear();

      expect(file.existsSync(), isFalse);
      expect((await store.usage()).count, 0);
    });

    test('clearing one provider leaves the other alone', () async {
      await store.write('maptiler', const TileCoordinate(12, 1, 1), bytes(10));
      await store.write('stadia', const TileCoordinate(12, 1, 1), bytes(10));

      await store.clear('maptiler');

      expect((await store.usage('maptiler')).count, 0);
      expect((await store.usage('stadia')).count, 1);
    });
  });

  group('the downloader', () {
    late AppDatabase db;
    late Directory root;
    late TileStore store;

    const box = BoundingBox(
      south: 25.57,
      west: 91.88,
      north: 25.59,
      east: 91.90,
    );
    const provider = MapTilerRaster(apiKey: 'test-key-not-real');

    setUp(() async {
      db = AppDatabase(NativeDatabase.memory());
      root = await Directory.systemTemp.createTemp('tiles');
      store = TileStore(db: db, root: root);
    });

    tearDown(() async {
      await db.close();
      if (root.existsSync()) await root.delete(recursive: true);
    });

    TileDownloader downloaderWith({
      Future<Uint8List?> Function(String)? fetch,
    }) => TileDownloader(
      store: store,
      provider: provider,
      delay: Duration.zero,
      fetch: fetch ?? (_) async => Uint8List.fromList(List.filled(64, 1)),
    );

    test('THE PLAN IS SHOWN BEFORE ANYTHING DOWNLOADS', () async {
      final plan = await downloaderWith().plan(box, minZoom: 12, maxZoom: 13);

      expect(plan.total, greaterThan(0));
      expect(plan.alreadyHave, 0);
      expect(plan.toFetch, plan.total);
      expect(plan.estimatedSize, isNotEmpty);
      expect((await store.usage()).count, 0, reason: 'planning fetches nothing');
    });

    test('downloading writes every tile', () async {
      final downloader = downloaderWith();
      final progress = await downloader
          .download(box, minZoom: 12, maxZoom: 12)
          .last;

      final plan = await downloader.plan(box, minZoom: 12, maxZoom: 12);
      expect(progress.done, progress.total);
      expect(plan.isComplete, isTrue);
      expect((await store.usage()).count, progress.total);
    });

    test('A SECOND DOWNLOAD FETCHES NOTHING', () async {
      // Resumable by construction: a tile on disk is skipped, so an
      // interrupted download restarts at 80% rather than at nothing.
      await downloaderWith().download(box, minZoom: 12, maxZoom: 12).last;

      var calls = 0;
      await downloaderWith(
        fetch: (_) async {
          calls++;
          return Uint8List.fromList([1]);
        },
      ).download(box, minZoom: 12, maxZoom: 12).last;

      expect(calls, 0);
    });

    test('progress runs from zero to the total', () async {
      final seen = await downloaderWith()
          .download(box, minZoom: 12, maxZoom: 12)
          .toList();

      expect(seen.first.done, 0);
      expect(seen.last.done, seen.last.total);
      expect(seen.last.fraction, 1.0);
      for (var i = 1; i < seen.length; i++) {
        expect(seen[i].done, greaterThanOrEqualTo(seen[i - 1].done));
      }
    });

    test('a missing tile is counted, not fatal', () async {
      // Ocean, or beyond the style's coverage. Normal.
      final progress = await downloaderWith(fetch: (_) async => null)
          .download(box, minZoom: 12, maxZoom: 12)
          .last;

      expect(progress.failed, progress.total);
      expect(progress.done, progress.total);
      expect((await store.usage()).count, 0);
    });

    test('A REJECTED KEY STOPS THE WHOLE DOWNLOAD', () async {
      // It affects every remaining tile, so continuing would be thousands of
      // guaranteed failures against someone else's rate limit.
      final downloader = downloaderWith(
        fetch: (_) async =>
            throw const TileDownloadException('key rejected'),
      );

      await expectLater(
        downloader.download(box, minZoom: 12, maxZoom: 12).toList(),
        throwsA(isA<TileDownloadException>()),
      );
    });

    test('AN UNCONFIGURED PROVIDER REFUSES BEFORE FETCHING', () async {
      final downloader = TileDownloader(
        store: store,
        provider: const MapTilerRaster(apiKey: ''),
        fetch: (_) async => Uint8List.fromList([1]),
      );

      await expectLater(
        downloader.download(box).toList(),
        throwsA(
          isA<TileDownloadException>().having(
            (e) => e.message,
            'message',
            contains('MapTiler key'),
          ),
        ),
      );
    });

    test('the default zoom band is the one the docs promise', () {
      expect(defaultMinZoom, 12);
      expect(defaultMaxZoom, 15);
    });
  });
}
