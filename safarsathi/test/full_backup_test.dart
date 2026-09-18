// test/full_backup_test.dart — issue #57.
//
// The JSON backup leaves the map behind on purpose. This is the archive that
// does not, and the only question that matters about it is whether a phone
// restored from one has a working offline map without touching the network.

import 'dart:io';
import 'dart:typed_data';

import 'package:archive/archive_io.dart';
import 'package:drift/native.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:safarsathi/core/database/app_database.dart';
import 'package:safarsathi/features/backup/data/backup.dart';
import 'package:safarsathi/features/backup/data/full_backup.dart';
import 'package:safarsathi/features/map/data/tile_math.dart';
import 'package:safarsathi/features/map/data/tile_store.dart';
import 'package:safarsathi/features/trips/data/trip_editor.dart';

void main() {
  late AppDatabase db;
  late Directory work;
  late Directory tiles;
  late TileStore store;

  setUp(() async {
    db = AppDatabase(NativeDatabase.memory());
    work = await Directory.systemTemp.createTemp('full-backup');
    tiles = Directory('${work.path}/tiles')..createSync(recursive: true);
    store = TileStore(db: db, root: tiles);
  });

  tearDown(() async {
    await db.close();
    if (work.existsSync()) work.deleteSync(recursive: true);
  });

  Future<void> seed({int tileCount = 6}) async {
    final editor = TripEditor(db);
    final tripId = await editor.createTrip(name: 'Meghalaya');
    await editor.addStop(tripId, const StopDraft(name: 'Shillong', nights: 1));
    await editor.addStop(tripId, const StopDraft(name: 'Sohra', nights: 1));

    for (var i = 0; i < tileCount; i++) {
      await store.write(
        'maptiler-outdoor-v2',
        TileCoordinate(12, 100 + i, 200),
        Uint8List.fromList(List.filled(64, i)),
        format: i.isEven ? 'webp' : 'png',
      );
    }
  }

  group('the plan', () {
    test('says what it would carry before writing anything', () async {
      await seed(tileCount: 4);
      final plan = await planFullBackup(db);
      expect(plan.tileCount, 4);
      expect(plan.tileBytes, 4 * 64);
      expect(plan.hasTiles, isTrue);
    });

    test('a trip with no map is still a valid backup', () async {
      await seed(tileCount: 0);
      expect((await planFullBackup(db)).hasTiles, isFalse);
    });
  });

  group('the round trip', () {
    test('A RESTORED PHONE HAS ITS MAP, WITH NO NETWORK', () async {
      await seed(tileCount: 6);
      final archive = File('${work.path}/backup.zip');
      await writeFullBackup(db, destination: archive, tileRoot: tiles);

      expect(archive.existsSync(), isTrue);

      // A new phone: new database, empty tile directory.
      final fresh = AppDatabase(NativeDatabase.memory());
      addTearDown(fresh.close);
      final freshTiles = Directory('${work.path}/restored')
        ..createSync(recursive: true);
      final freshStore = TileStore(db: fresh, root: freshTiles);

      await restoreFullBackup(fresh, archive, tileRoot: freshTiles);

      expect((await fresh.select(fresh.trips).get()).single.name, 'Meghalaya');
      expect(await fresh.select(fresh.stops).get(), hasLength(2));

      // Every tile readable, byte for byte, from disk alone.
      for (var i = 0; i < 6; i++) {
        final bytes = await freshStore.read(
          'maptiler-outdoor-v2',
          TileCoordinate(12, 100 + i, 200),
        );
        expect(bytes, isNotNull, reason: 'tile $i missing');
        expect(bytes!.first, i);
      }
    });

    test('BOTH TILE FORMATS SURVIVE, under their own names', () async {
      await seed(tileCount: 2);
      final archive = File('${work.path}/backup.zip');
      await writeFullBackup(db, destination: archive, tileRoot: tiles);

      final freshTiles = Directory('${work.path}/restored')
        ..createSync(recursive: true);
      final fresh = AppDatabase(NativeDatabase.memory());
      addTearDown(fresh.close);
      await restoreFullBackup(fresh, archive, tileRoot: freshTiles);

      final store = TileStore(db: fresh, root: freshTiles);
      expect(
        store.fileFor('maptiler-outdoor-v2', const TileCoordinate(12, 100, 200),
            format: 'webp').existsSync(),
        isTrue,
      );
      expect(
        store.fileFor('maptiler-outdoor-v2', const TileCoordinate(12, 101, 200),
            format: 'png').existsSync(),
        isTrue,
      );
    });

    test('THE INDEX IS REBUILT FROM WHAT LANDED, not from the file', () async {
      // A half-written archive should report the tiles it has, not the ones
      // it meant to have — the cache screen saying 3,424 over an empty
      // directory is the worst possible answer.
      await seed(tileCount: 5);
      final archive = File('${work.path}/backup.zip');
      await writeFullBackup(db, destination: archive, tileRoot: tiles);

      final fresh = AppDatabase(NativeDatabase.memory());
      addTearDown(fresh.close);
      final freshTiles = Directory('${work.path}/restored')
        ..createSync(recursive: true);
      await restoreFullBackup(fresh, archive, tileRoot: freshTiles);

      final indexed = await fresh.select(fresh.mapTiles).get();
      expect(indexed, hasLength(5));
      expect(indexed.every((t) => t.provider == 'maptiler-outdoor-v2'), isTrue);
      expect(indexed.every((t) => t.bytes == 64), isTrue);

      // And the usage figure the cache screen reads is the real one.
      final store = TileStore(db: fresh, root: freshTiles);
      final usage = await store.usage('maptiler-outdoor-v2');
      expect(usage.count, 5);
      expect(usage.bytes, 5 * 64);
    });

    test('reindexing an empty directory reports nothing, not stale rows', () async {
      await seed(tileCount: 3);
      expect((await db.select(db.mapTiles).get()), hasLength(3));

      final empty = Directory('${work.path}/nothing')..createSync();
      await reindexTiles(db, empty);
      expect(await db.select(db.mapTiles).get(), isEmpty);
    });
  });

  group('reading one before committing to it', () {
    test('describes both halves', () async {
      await seed(tileCount: 7);
      final archive = File('${work.path}/backup.zip');
      await writeFullBackup(db, destination: archive, tileRoot: tiles);

      final contents = await readFullBackup(archive);
      expect(contents.tileCount, 7);
      expect(contents.database.trips, 1);
      expect(contents.database.stops, 2);
    });

    test('a zip that is not a SafarSathi backup is refused', () async {
      await seed(tileCount: 1);
      // An archive with tiles but no database in it.
      final odd = File('${work.path}/odd.zip');
      await writeTilesOnlyZip(odd, tiles);

      expect(
        () => readFullBackup(odd),
        throwsA(
          isA<BackupException>().having(
            (e) => e.message,
            'message',
            contains('no SafarSathi backup'),
          ),
        ),
      );
    });

    test('the json half alone still restores when there is no map', () async {
      await seed(tileCount: 0);
      final archive = File('${work.path}/backup.zip');
      await writeFullBackup(db, destination: archive, tileRoot: tiles);

      final contents = await readFullBackup(archive);
      expect(contents.tileCount, 0);
      expect(contents.database.trips, 1);
    });
  });

  group('the filename', () {
    test('says it is the full one, and when', () {
      expect(
        fullBackupFileName(now: DateTime(2026, 10, 2, 7, 5)),
        'safarsathi-full-2026-10-02-0705.zip',
      );
    });
  });
}

/// A zip with tiles in it but no database — something a person could
/// plausibly hand the app by mistake.
Future<void> writeTilesOnlyZip(File destination, Directory tiles) async {
  final encoder = ZipFileEncoder();
  encoder.create(destination.path);
  try {
    for (final file in tiles.listSync(recursive: true).whereType<File>()) {
      final relative = file.path.substring(tiles.path.length + 1);
      await encoder.addFile(file, '$tilesEntryPrefix$relative');
    }
  } finally {
    await encoder.close();
  }
}
