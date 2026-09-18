// lib/features/backup/data/full_backup.dart
//
// The backup that carries the downloaded map with it — issue #57.
//
// The JSON backup at #56 deliberately leaves out everything re-downloadable,
// which keeps the file small and is the right default. It also means a phone
// that dies takes 346 MB of tiles and every place along every road with it,
// and the only way back is a WiFi connection and twenty minutes.
//
// This is the other half: the same JSON, plus the tile files, as one archive.
//
// NOTHING HERE HOLDS THE WHOLE THING IN MEMORY. The zip is written to disk a
// file at a time and read back the same way. The obvious implementation —
// build the bytes, hand them to the save dialog — works fine on the test trip
// and takes the app down on a real one.

import 'dart:convert';
import 'dart:io';

import 'package:archive/archive_io.dart';
import 'package:drift/drift.dart';

import '../../../core/database/app_database.dart';
import 'backup.dart';

/// Where the database part of the archive lives inside it.
const backupEntryName = 'safarsathi.json';

/// The prefix every tile entry sits under.
const tilesEntryPrefix = 'tiles/';

class FullBackupPlan {
  final int tileCount;
  final int tileBytes;

  const FullBackupPlan({required this.tileCount, required this.tileBytes});

  bool get hasTiles => tileCount > 0;
}

/// What a full backup would contain, before writing anything.
Future<FullBackupPlan> planFullBackup(AppDatabase db) async {
  final count = db.mapTiles.id.count();
  final bytes = db.mapTiles.bytes.sum();
  final query = db.selectOnly(db.mapTiles)..addColumns([count, bytes]);
  final row = await query.getSingle();
  return FullBackupPlan(
    tileCount: row.read(count) ?? 0,
    tileBytes: row.read(bytes) ?? 0,
  );
}

/// Writes the archive to [destination] and returns it.
///
/// [tileRoot] is the directory tiles hang under — the same one `TileStore`
/// was given. Missing entirely is fine: a backup taken before anything was
/// downloaded is still a valid backup.
Future<File> writeFullBackup(
  AppDatabase db, {
  required File destination,
  required Directory tileRoot,
  void Function(int done, int total)? onProgress,
}) async {
  final json = await exportBackup(db);

  // The JSON goes in first, so a truncated archive still has the part that
  // cannot be re-downloaded at the front of it.
  final jsonFile = File('${destination.parent.path}/$backupEntryName');
  await jsonFile.writeAsString(json, flush: true);

  final tiles = tileRoot.existsSync()
      ? tileRoot
            .listSync(recursive: true)
            .whereType<File>()
            .toList(growable: false)
      : const <File>[];

  final encoder = ZipFileEncoder();
  encoder.create(destination.path);
  try {
    await encoder.addFile(jsonFile, backupEntryName);

    var done = 0;
    for (final tile in tiles) {
      // Stored relative to the tile root, so the restore can put it back
      // wherever this phone happens to keep its documents directory — which
      // is not the same path on every device, or even across reinstalls.
      final relative = tile.path.substring(tileRoot.path.length + 1);
      // STORE, DO NOT DEFLATE. These are PNG and WebP: already compressed,
      // so deflating them spends minutes of phone CPU to save nothing.
      await encoder.addFile(
        tile,
        '$tilesEntryPrefix$relative',
        ZipFileEncoder.STORE,
      );
      done++;
      if (done % 50 == 0 || done == tiles.length) {
        onProgress?.call(done, tiles.length);
      }
    }
  } finally {
    await encoder.close();
    if (jsonFile.existsSync()) await jsonFile.delete();
  }

  return destination;
}

class FullBackupContents {
  final BackupContents database;
  final int tileCount;

  const FullBackupContents({required this.database, required this.tileCount});
}

/// Reads an archive far enough to describe it, without unpacking anything.
Future<FullBackupContents> readFullBackup(File archive) async {
  final input = InputFileStream(archive.path);
  try {
    final zip = ZipDecoder().decodeBuffer(input);

    final entry = zip.files.where((f) => f.name == backupEntryName).firstOrNull;
    if (entry == null) {
      throw const BackupException(
        'That archive has no SafarSathi backup inside it.',
      );
    }

    final database = readBackup(utf8.decode(entry.content as List<int>));
    final tiles = zip.files
        .where((f) => f.isFile && f.name.startsWith(tilesEntryPrefix))
        .length;

    return FullBackupContents(
      database: database.withTiles(tiles),
      tileCount: tiles,
    );
  } finally {
    await input.close();
  }
}

/// Restores both halves: the database, then the tiles.
///
/// The database goes first deliberately. If the tiles fail halfway — the
/// phone fills up, the file is truncated — the trip, the numbers and the
/// checklist are already back, and the map is the part that can be fetched
/// again.
Future<void> restoreFullBackup(
  AppDatabase db,
  File archive, {
  required Directory tileRoot,
  void Function(int done, int total)? onProgress,
}) async {
  final input = InputFileStream(archive.path);
  try {
    final zip = ZipDecoder().decodeBuffer(input);

    final entry = zip.files.where((f) => f.name == backupEntryName).firstOrNull;
    if (entry == null) {
      throw const BackupException(
        'That archive has no SafarSathi backup inside it.',
      );
    }
    await restoreBackup(db, readBackup(utf8.decode(entry.content as List<int>)));

    final tiles = zip.files
        .where((f) => f.isFile && f.name.startsWith(tilesEntryPrefix))
        .toList(growable: false);

    var done = 0;
    for (final tile in tiles) {
      final relative = tile.name.substring(tilesEntryPrefix.length);
      // A zip entry naming its way out of the tile directory is not something
      // this app ever writes, so it is something to refuse rather than
      // resolve.
      if (relative.contains('..')) continue;

      final out = File('${tileRoot.path}/$relative');
      await out.parent.create(recursive: true);
      await out.writeAsBytes(tile.content as List<int>, flush: false);
      done++;
      if (done % 50 == 0 || done == tiles.length) {
        onProgress?.call(done, tiles.length);
      }
    }

    // The index is rebuilt from what actually landed rather than trusted from
    // the file: a half-written archive should report the tiles it has, not
    // the ones it meant to have.
    await reindexTiles(db, tileRoot);
  } finally {
    await input.close();
  }
}

/// Rebuilds the `mapTiles` index by walking the tile directory.
Future<void> reindexTiles(AppDatabase db, Directory tileRoot) async {
  await db.delete(db.mapTiles).go();
  if (!tileRoot.existsSync()) return;

  final rows = <MapTilesCompanion>[];
  for (final file in tileRoot.listSync(recursive: true).whereType<File>()) {
    final parts = file.path
        .substring(tileRoot.path.length + 1)
        .split(Platform.pathSeparator);
    // <provider>/<z>/<x>/<y>.<ext>
    if (parts.length < 4) continue;
    final provider = parts.sublist(0, parts.length - 3).join('/');
    final z = int.tryParse(parts[parts.length - 3]);
    final x = int.tryParse(parts[parts.length - 2]);
    final y = int.tryParse(parts.last.split('.').first);
    if (z == null || x == null || y == null) continue;

    rows.add(
      MapTilesCompanion.insert(
        provider: provider,
        z: z,
        x: x,
        y: y,
        bytes: file.lengthSync(),
      ),
    );
  }

  if (rows.isEmpty) return;
  await db.batch((b) => b.insertAll(db.mapTiles, rows));
}

/// The filename a full backup is offered under.
String fullBackupFileName({DateTime? now}) {
  final d = now ?? DateTime.now();
  String two(int n) => n.toString().padLeft(2, '0');
  return 'safarsathi-full-${d.year}-${two(d.month)}-${two(d.day)}-'
      '${two(d.hour)}${two(d.minute)}.zip';
}
