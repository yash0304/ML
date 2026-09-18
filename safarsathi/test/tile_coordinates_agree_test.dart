// test/tile_coordinates_agree_test.dart
//
// THE SEAM THAT BROKE, PINNED.
//
// The downloader decides which tiles to fetch using tile_math. The renderer
// decides which tiles to ask for using flutter_map's own grid arithmetic.
// Nothing ever checked that those two agree, and for one released build they
// did not: the layer was handed the retina IMAGE size (512) where it wanted
// the GRID size (256), so it divided the map's 256-based pixel bounds by
// twice too much and asked for z12 tiles by z11 numbering.
//
// Every request missed. 3424 tiles and 204 MB on the phone rendered as blank
// paper — while the route and the markers, positioned by the projection
// rather than by tiles, drew perfectly and made it look like a styling
// problem.
//
// These tests fail if the two halves ever disagree again.

import 'dart:io';
import 'dart:typed_data';

import 'package:drift/native.dart';
import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart' as fm;
import 'package:latlong2/latlong.dart' as ll;
import 'package:flutter_test/flutter_test.dart';

import 'package:safarsathi/core/database/app_database.dart';
import 'package:safarsathi/core/theme/app_tokens.dart';
import 'package:safarsathi/features/discovery/data/geo.dart';
import 'package:safarsathi/features/map/data/tile_math.dart';
import 'package:safarsathi/features/map/data/tile_provider.dart';
import 'package:safarsathi/features/map/data/tile_store.dart';
import 'package:safarsathi/features/map/presentation/trip_map.dart';

/// Records every coordinate the layer asks for, and serves a blank image so
/// nothing touches the disk or the network.
class _RecordingTileProvider extends fm.TileProvider {
  final List<TileCoordinate> asked = [];

  @override
  ImageProvider getImage(fm.TileCoordinates coordinates, fm.TileLayer options) {
    asked.add(TileCoordinate(coordinates.z, coordinates.x, coordinates.y));
    return MemoryImage(_onePixelPng);
  }
}

final _onePixelPng = Uint8List.fromList(const [
  0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, //
  0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
  0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
  0x08, 0x06, 0x00, 0x00, 0x00, 0x1F, 0x15, 0xC4,
  0x89, 0x00, 0x00, 0x00, 0x0A, 0x49, 0x44, 0x41,
  0x54, 0x78, 0x9C, 0x63, 0x00, 0x01, 0x00, 0x00,
  0x05, 0x00, 0x01, 0x0D, 0x0A, 0x2D, 0xB4, 0x00,
  0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE,
  0x42, 0x60, 0x82,
]);

const _shillong = LatLng(25.5788, 91.8933);

void main() {
  late AppDatabase db;
  late Directory root;
  late TileStore store;

  setUp(() async {
    db = AppDatabase(NativeDatabase.memory());
    root = await Directory.systemTemp.createTemp('coords');
    store = TileStore(db: db, root: root);
  });

  tearDown(() async {
    await db.close();
    if (root.existsSync()) await root.delete(recursive: true);
  });

  Future<List<TileCoordinate>> askedFor(
    WidgetTester tester, {
    required int gridSize,
  }) async {
    final recorder = _RecordingTileProvider();

    await tester.pumpWidget(
      MaterialApp(
        theme: AppTokens.light,
        home: Scaffold(
          body: SizedBox(
            width: 400,
            height: 400,
            child: fm.FlutterMap(
              options: fm.MapOptions(
                initialCenter: ll.LatLng(_shillong.lat, _shillong.lon),
                initialZoom: 12,
              ),
              children: [
                fm.TileLayer(
                  tileProvider: recorder,
                  tileDimension: gridSize,
                  urlTemplate: 'offline://{z}/{x}/{y}',
                ),
              ],
            ),
          ),
        ),
      ),
    );
    await tester.pump();
    return recorder.asked;
  }

  testWidgets('the renderer asks for the tile the downloader would store', (
    tester,
  ) async {
    final asked = await askedFor(tester, gridSize: 256);
    expect(asked, isNotEmpty);

    // The tile the DOWNLOADER would have fetched for the centre of the view.
    final expected = tileFor(_shillong, 12);

    expect(
      asked.where((t) => t.z == 12).map((t) => '${t.x}/${t.y}').toList(),
      contains('${expected.x}/${expected.y}'),
      reason: 'the layer must ask for the same x/y that tile_math stored; '
          'if it does not, every downloaded tile is a miss',
    );
  });

  testWidgets('THE IMAGE SIZE HALVES EVERY INDEX — the original bug', (
    tester,
  ) async {
    // Handing the layer 512 (the retina image size) instead of 256 (the grid
    // size) is what shipped. This asserts the failure mode directly, so the
    // test explains itself if anyone is tempted to "fix" gridSize back.
    final correct = await askedFor(tester, gridSize: 256);
    final broken = await askedFor(tester, gridSize: 512);

    final expected = tileFor(_shillong, 12);
    final correctCentre = correct.where((t) => t.z == 12);
    final brokenCentre = broken.where((t) => t.z == 12);

    expect(
      correctCentre.map((t) => '${t.x}/${t.y}'),
      contains('${expected.x}/${expected.y}'),
    );
    expect(
      brokenCentre.map((t) => '${t.x}/${t.y}'),
      isNot(contains('${expected.x}/${expected.y}')),
      reason: 'the image size must not be used as the grid size',
    );
  });

  test('a retina provider keeps the standard grid', () {
    // 512 pixels of detail over the same 256-pixel square of ground.
    const provider = MapTilerRaster(apiKey: 'k');
    expect(provider.tileSize, 512);
    expect(provider.gridSize, 256);
  });

  testWidgets('TripMap hands the layer the grid size, not the image size', (
    tester,
  ) async {
    await tester.pumpWidget(
      MaterialApp(
        theme: AppTokens.light,
        home: Scaffold(
          body: TripMap(
            provider: const MapTilerRaster(apiKey: 'k'),
            store: store,
            stops: const [(name: 'Shillong', at: _shillong)],
          ),
        ),
      ),
    );

    final layer = tester.widget<fm.TileLayer>(find.byType(fm.TileLayer));
    expect(layer.tileDimension, const MapTilerRaster(apiKey: 'k').gridSize);
  });
}
