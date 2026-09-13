// lib/features/sync/data/trip_sync.dart
//
// Everything a trip needs, in one press — issue #25.
//
// Every piece already worked. What was missing is that the user had to visit
// three separate screens, in the right order, before leaving WiFi. That is not
// a feature gap; it is the difference between a trip that is prepared and one
// that is not.
//
// NO RIVERPOD. The backlog said this screen would justify it. What it holds is
// a list of tasks and an index into it, for the lifetime of one route — a
// StatefulWidget over a stream does that. See DECISIONS.md.

import 'package:drift/drift.dart';

import '../../../core/database/app_database.dart';
import '../../discovery/data/corridor_sync.dart';
import '../../map/data/map_download.dart';
import '../../map/data/tile_math.dart';
import '../../weather/data/weather_sync.dart';
import 'sync_error.dart';

/// What kind of work a task is. Ordered as they run.
///
/// `corridor` is ONE task, not two. `CorridorSync.syncLeg` routes the leg and
/// queries its box in a single call and writes both in one transaction —
/// listing "route" and "places" as separate rows would report progress that
/// does not correspond to any work.
enum SyncKind { corridor, tiles, weather }

extension SyncKindLabel on SyncKind {
  String get label => switch (this) {
    SyncKind.corridor => 'Route and what is along it',
    SyncKind.tiles => 'Map',
    SyncKind.weather => 'Weather',
  };
}

/// One unit of work, named so progress can say what it is doing rather than
/// only how far along it is.
class SyncTask {
  final SyncKind kind;
  final String subject;

  /// Set for leg work.
  final int? legId;

  /// Set for stop work.
  final int? stopId;

  const SyncTask({
    required this.kind,
    required this.subject,
    this.legId,
    this.stopId,
  });

  String get label => '${kind.label} · $subject';
}

class SyncFailure {
  final SyncTask task;
  final SyncError error;
  const SyncFailure(this.task, this.error);

  /// The sentence to show. Never the raw exception — see sync_error.dart.
  String get reason => error.message;
}

class SyncProgress {
  final int done;
  final int total;

  /// What is being worked on right now, or null when finished.
  final SyncTask? current;

  final List<SyncFailure> failures;

  const SyncProgress({
    required this.done,
    required this.total,
    this.current,
    this.failures = const [],
  });

  double get fraction => total == 0 ? 1 : done / total;
  bool get isDone => done >= total;
  bool get hadFailures => failures.isNotEmpty;
}

class TripSyncPlan {
  final List<SyncTask> tasks;

  /// Legs whose stops have no coordinates. They cannot be worked at all, and
  /// the screen says so rather than quietly doing less than expected.
  final int legsWithoutCoordinates;

  /// Stops with no coordinates, for the same reason.
  final int stopsWithoutCoordinates;

  const TripSyncPlan({
    required this.tasks,
    required this.legsWithoutCoordinates,
    required this.stopsWithoutCoordinates,
  });

  int get total => tasks.length;
  bool get isEmpty => tasks.isEmpty;

  int countOf(SyncKind kind) => tasks.where((t) => t.kind == kind).length;
}

class TripSync {
  final AppDatabase db;
  final CorridorSync corridor;
  final WeatherSync weather;
  final MapDownload map;

  const TripSync({
    required this.db,
    required this.corridor,
    required this.weather,
    required this.map,
  });

  /// What a run would do, without doing any of it.
  Future<TripSyncPlan> plan(int tripId) async {
    final legs =
        await (db.select(db.legs)
              ..where((l) => l.tripId.equals(tripId))
              ..orderBy([(l) => OrderingTerm(expression: l.sequenceOrder)]))
            .get();

    final stops =
        await (db.select(db.stops)
              ..where((s) => s.tripId.equals(tripId))
              ..orderBy([(s) => OrderingTerm(expression: s.sequenceOrder)]))
            .get();
    final byId = {for (final s in stops) s.id: s};

    final tasks = <SyncTask>[];
    var legsMissing = 0;

    for (final leg in legs) {
      final from = byId[leg.fromStopId];
      final to = byId[leg.toStopId];
      final usable =
          from?.lat != null &&
          from?.lon != null &&
          to?.lat != null &&
          to?.lon != null;

      if (!usable) {
        legsMissing++;
        continue;
      }

      // ORDER MATTERS. Tiles derive their box from the route, and with no
      // polyline that falls back to the straight line — which in the
      // Meghalaya hills is a different valley. So every corridor runs first.
      tasks.add(
        SyncTask(
          kind: SyncKind.corridor,
          subject: '${from!.name} → ${to!.name}',
          legId: leg.id,
        ),
      );
    }

    var stopsMissing = 0;
    for (final stop in stops) {
      if (stop.lat == null || stop.lon == null) {
        stopsMissing++;
        continue;
      }
      tasks.add(
        SyncTask(
          kind: SyncKind.weather,
          subject: stop.name,
          stopId: stop.id,
        ),
      );
    }

    // Tiles last: by far the largest download, and a failure there should not
    // cost the small useful things that come before it.
    if (legs.length > legsMissing) {
      tasks.add(const SyncTask(kind: SyncKind.tiles, subject: 'whole trip'));
    }

    return TripSyncPlan(
      tasks: tasks,
      legsWithoutCoordinates: legsMissing,
      stopsWithoutCoordinates: stopsMissing,
    );
  }

  /// Runs the plan.
  ///
  /// ONE FAILURE DOES NOT STOP THE RUN. Each task is independent and a failure
  /// is recorded and stepped over: Overpass being busy should not cost the
  /// tiles, and a stop outside the forecast range should not cost the route. A
  /// sync that silently does 60% is worse than one that says which 40% is
  /// missing, so the failures come back with the progress.
  ///
  /// Resumable by construction. Nothing here tracks its own position; every
  /// underlying step already skips work already done, so re-running after a
  /// failure only costs the missing parts. That cannot get out of step with
  /// what is actually on disk, which a resume cursor can.
  Stream<SyncProgress> run(int tripId) async* {
    final plan = await this.plan(tripId);
    final failures = <SyncFailure>[];
    var done = 0;

    yield SyncProgress(done: 0, total: plan.total);

    for (final task in plan.tasks) {
      yield SyncProgress(
        done: done,
        total: plan.total,
        current: task,
        failures: List.of(failures),
      );

      try {
        switch (task.kind) {
          case SyncKind.corridor:
            await corridor.syncLeg(task.legId!);

          case SyncKind.weather:
            final stop = await (db.select(
              db.stops,
            )..where((s) => s.id.equals(task.stopId!))).getSingle();
            await weather.syncStop(stop);

          case SyncKind.tiles:
            await for (final _ in map.download(tripId)) {
              // Tile-level progress is not surfaced here; the whole tile
              // download is one row in this list. The map screen shows the
              // per-tile detail for anyone who wants it.
            }
        }
      } on Object catch (e) {
        failures.add(SyncFailure(task, describeSyncError(e)));
      }

      done++;
      yield SyncProgress(
        done: done,
        total: plan.total,
        current: done >= plan.total ? null : task,
        failures: List.of(failures),
      );
    }
  }

}

/// Rough total size of a run, for the button.
///
/// Only the tiles are worth estimating: a route is a few kilobytes, a place
/// list is tens, and a forecast is smaller still. Saying "about 21 MB" and
/// meaning the tiles is honest; itemising three negligible things is noise.
Future<String> estimateSyncSize(MapDownload map, int tripId) async {
  final estimate = await map.estimate(tripId);
  if (estimate.toFetch == 0) return 'the maps are already here';
  return 'about ${describeBytes(estimate.estimatedBytes)}, nearly all of it map';
}
