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

  /// Root under which `<provider>/<z>/<x>/<y>.<ext>` hangs.
  final Directory root;

  const TileStore({required this.db, required this.root});

  /// The formats a tile may be stored in, newest preference first.
  ///
  /// WEBP WAS ADDED WITHOUT INVALIDATING ANYTHING ALREADY ON DISK. Tiles
  /// downloaded before the switch are PNG and are still perfectly good; making
  /// the reader miss them would have turned a working offline map blank and
  /// forced a 346 MB re-download, which is precisely the cost this change
  /// exists to avoid. So a read tries both, and only new downloads are WebP.
  static const formats = ['webp', 'png'];

  File fileFor(String provider, TileCoordinate tile, {String format = 'png'}) =>
      File('${root.path}/$provider/${tile.z}/${tile.x}/${tile.y}.$format');

  /// The file actually on disk for this tile, whatever format it is in.
  File? storedFile(String provider, TileCoordinate tile) {
    for (final format in formats) {
      final file = fileFor(provider, tile, format: format);
      if (file.existsSync()) return file;
    }
    return null;
  }

  /// Reads a tile, or null when it was never downloaded.
  ///
  /// THERE IS NO NETWORK PATH HERE. Not a fallback, not a timeout. A tile that
  /// was not downloaded returns null and renders blank, and the only way to be
  /// certain of that on a mountain road is for the code to be absent.
  Future<Uint8List?> read(String provider, TileCoordinate tile) async {
    final file = storedFile(provider, tile);
    if (file == null) return null;
    try {
      return await file.readAsBytes();
    } on FileSystemException {
      // A truncated or unreadable file is a missing tile, not a crash.
      return null;
    }
  }

  Future<bool> has(String provider, TileCoordinate tile) async =>
      storedFile(provider, tile) != null;

  Future<void> write(
    String provider,
    TileCoordinate tile,
    Uint8List bytes, {
    String format = 'png',
  }) async {
    final file = fileFor(provider, tile, format: format);
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
