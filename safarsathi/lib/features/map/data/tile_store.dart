// lib/features/map/data/tile_store.dart
//
// Where downloaded tiles live — issue #24.
//
// Files on disk, indexed in SQLite. The index exists so the cache screen can
// state a real number without walking a directory tree, and so a re-download
// knows what to skip.

import 'dart:io';

import 'package:drift/drift.dart';

import '../../../core/database/app_database.dart';
import 'tile_math.dart';

/// Reads and writes tiles. Takes the directory rather than finding it, so a
/// test can point it at a temporary folder.
class TileStore {
  final AppDatabase db;

  /// Root under which `<provider>/<z>/<x>/<y>.png` hangs.
  final Directory root;

  const TileStore({required this.db, required this.root});

  File fileFor(String provider, TileCoordinate tile) => File(
    '${root.path}/$provider/${tile.z}/${tile.x}/${tile.y}.png',
  );

  /// Reads a tile, or null when it was never downloaded.
  ///
  /// THERE IS NO NETWORK PATH HERE. Not a fallback, not a timeout. A tile that
  /// was not downloaded returns null and renders blank, and the only way to be
  /// certain of that on a mountain road is for the code to be absent.
  Future<Uint8List?> read(String provider, TileCoordinate tile) async {
    final file = fileFor(provider, tile);
    if (!file.existsSync()) return null;
    try {
      return await file.readAsBytes();
    } on FileSystemException {
      // A truncated or unreadable file is a missing tile, not a crash.
      return null;
    }
  }

  Future<bool> has(String provider, TileCoordinate tile) async =>
      fileFor(provider, tile).existsSync();

  Future<void> write(
    String provider,
    TileCoordinate tile,
    Uint8List bytes,
  ) async {
    final file = fileFor(provider, tile);
    await file.parent.create(recursive: true);
    await file.writeAsBytes(bytes, flush: true);

    // The conflict target is the UNIQUE KEY, not the primary key.
    // `insertOnConflictUpdate` defaults to conflicting on `id`, which an
    // autoIncrement insert never supplies — so re-writing a tile threw a
    // constraint error instead of updating. It only bites when a tile is
    // written twice, which happens whenever a file was cleared but its index
    // row survived.
    await db
        .into(db.mapTiles)
        .insert(
          MapTilesCompanion.insert(
            provider: provider,
            z: tile.z,
            x: tile.x,
            y: tile.y,
            bytes: bytes.length,
          ),
          onConflict: DoUpdate(
            (_) => MapTilesCompanion(
              bytes: Value(bytes.length),
              fetchedAt: Value(DateTime.now()),
            ),
            target: [
              db.mapTiles.provider,
              db.mapTiles.z,
              db.mapTiles.x,
              db.mapTiles.y,
            ],
          ),
        );
  }

  /// How many tiles and how many bytes are held, per provider.
  Future<({int count, int bytes})> usage([String? provider]) async {
    final query = db.selectOnly(db.mapTiles)
      ..addColumns([db.mapTiles.id.count(), db.mapTiles.bytes.sum()]);
    if (provider != null) {
      query.where(db.mapTiles.provider.equals(provider));
    }

    final row = await query.getSingle();
    return (
      count: row.read(db.mapTiles.id.count()) ?? 0,
      bytes: row.read(db.mapTiles.bytes.sum()) ?? 0,
    );
  }

  Stream<({int count, int bytes})> watchUsage([String? provider]) {
    final query = db.selectOnly(db.mapTiles)
      ..addColumns([db.mapTiles.id.count(), db.mapTiles.bytes.sum()]);
    if (provider != null) {
      query.where(db.mapTiles.provider.equals(provider));
    }
    return query.watchSingle().map(
      (row) => (
        count: row.read(db.mapTiles.id.count()) ?? 0,
        bytes: row.read(db.mapTiles.bytes.sum()) ?? 0,
      ),
    );
  }

  /// Removes every tile, files included.
  ///
  /// The index alone would leave hundreds of megabytes on the phone reporting
  /// as zero, which is the worst possible answer to "clear the cache".
  Future<void> clear([String? provider]) async {
    final directory = provider == null
        ? root
        : Directory('${root.path}/$provider');
    if (directory.existsSync()) {
      await directory.delete(recursive: true);
    }

    if (provider == null) {
      await db.delete(db.mapTiles).go();
    } else {
      await (db.delete(
        db.mapTiles,
      )..where((t) => t.provider.equals(provider))).go();
    }
  }
}
