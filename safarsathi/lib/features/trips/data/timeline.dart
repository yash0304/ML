// lib/features/trips/data/timeline.dart
//
// The trip as it went — #30. SCREENS.md §9.
//
// The rail is the road. Arrivals (check-ins, #33) are milestone caps on it;
// notes and photos are plain dots. GPS fixes, when the route log is on, are
// never shown one by one — they become the distance driven between
// arrivals, and the line on the map.

import 'package:drift/drift.dart';

import '../../../core/database/app_database.dart';
import '../../../core/database/watch_tables.dart';
import '../../discovery/data/geo.dart';

/// Timeline kinds, as stored in TimelineEntries.kind.
abstract final class TimelineKind {
  static const arrival = 'arrival';
  static const note = 'note';
  static const fix = 'fix';
}

/// An arrival with how far and how long since the one before.
class TimelineArrival {
  final TimelineEntry entry;
  final String stopName;

  /// Kilometres since the previous arrival. Null for the first.
  final double? km;

  /// True when [km] is a straight line between the two places, not the road
  /// driven — said on screen, because in these hills the two differ a lot.
  final bool straightLine;

  final Duration? since;

  const TimelineArrival({
    required this.entry,
    required this.stopName,
    this.km,
    this.straightLine = false,
    this.since,
  });
}

/// One day of the trip, oldest first within it.
class TimelineDay {
  final DateTime day;

  /// Arrivals ([TimelineArrival]) and notes ([TimelineEntry]) in time order.
  final List<Object> items;

  /// Road driven that day, from the route log. Zero when it was off.
  final double loggedKm;

  const TimelineDay({
    required this.day,
    required this.items,
    this.loggedKm = 0,
  });
}

DateTime _dayOf(DateTime t) => DateTime(t.year, t.month, t.day);

LatLng? _at(TimelineEntry e) =>
    e.lat == null || e.lon == null ? null : LatLng(e.lat!, e.lon!);

double _pathKm(List<LatLng> points) {
  var m = 0.0;
  for (var i = 1; i < points.length; i++) {
    m += haversineMetres(points[i - 1], points[i]);
  }
  return m / 1000;
}

/// Days in order, built from every timeline row of a trip. Pure.
List<TimelineDay> buildTimeline(
  List<TimelineEntry> entries,
  Map<int, Stop> stops,
) {
  final sorted = [...entries]
    ..sort((a, b) => a.occurredAt.compareTo(b.occurredAt));
  final fixes = [
    for (final e in sorted)
      if (e.kind == TimelineKind.fix && _at(e) != null) e,
  ];

  LatLng? placeOf(TimelineEntry e) {
    final own = _at(e);
    if (own != null) return own;
    final s = stops[e.stopId];
    return s?.lat == null || s?.lon == null ? null : LatLng(s!.lat!, s.lon!);
  }

  final byDay = <DateTime, List<Object>>{};
  final kmByDay = <DateTime, double>{};
  TimelineEntry? previous;

  for (final e in sorted) {
    final day = _dayOf(e.occurredAt);
    if (e.kind == TimelineKind.arrival) {
      double? km;
      var straight = false;
      Duration? since;
      if (previous != null) {
        since = e.occurredAt.difference(previous.occurredAt);
        final between = [
          for (final f in fixes)
            if (f.occurredAt.isAfter(previous.occurredAt) &&
                !f.occurredAt.isAfter(e.occurredAt))
              _at(f)!,
        ];
        if (between.length >= 2) {
          km = _pathKm(between);
        } else {
          final a = placeOf(previous);
          final b = placeOf(e);
          if (a != null && b != null) {
            km = haversineMetres(a, b) / 1000;
            straight = true;
          }
        }
      }
      byDay.putIfAbsent(day, () => []).add(
        TimelineArrival(
          entry: e,
          stopName: stops[e.stopId]?.name ?? e.title ?? 'Arrived',
          km: km,
          straightLine: straight,
          since: since,
        ),
      );
      previous = e;
    } else if (e.kind == TimelineKind.note) {
      byDay.putIfAbsent(day, () => []).add(e);
    } else if (e.kind == TimelineKind.fix) {
      byDay.putIfAbsent(day, () => []);
    }
  }

  // Road driven per day: consecutive fixes on the same day.
  for (var i = 1; i < fixes.length; i++) {
    final a = fixes[i - 1];
    final b = fixes[i];
    final day = _dayOf(b.occurredAt);
    if (_dayOf(a.occurredAt) != day) continue;
    kmByDay[day] =
        (kmByDay[day] ?? 0) + haversineMetres(_at(a)!, _at(b)!) / 1000;
  }

  final days = byDay.keys.toList()..sort();
  return [
    for (final d in days)
      TimelineDay(day: d, items: byDay[d]!, loggedKm: kmByDay[d] ?? 0),
  ];
}

/// Whether a new GPS fix is worth keeping: far enough or long enough since
/// the last kept one, and precise enough to mean something. Pure.
///
/// One fix per 150 m or 10 minutes is enough to draw a road and measure it,
/// and keeps a long day's drive to a few hundred rows.
bool keepFix({
  required LatLng at,
  required double accuracyM,
  required DateTime time,
  LatLng? lastAt,
  DateTime? lastTime,
}) {
  if (accuracyM > 100) return false;
  if (lastAt == null || lastTime == null) return true;
  if (haversineMetres(lastAt, at) >= 150) return true;
  return time.difference(lastTime) >= const Duration(minutes: 10);
}

Stream<List<TimelineDay>> watchTimeline(AppDatabase db, int tripId) =>
    watchTables(db, {db.timelineEntries, db.stops}).asyncMap((_) async {
      final entries = await (db.select(
        db.timelineEntries,
      )..where((t) => t.tripId.equals(tripId))).get();
      final stops = await (db.select(
        db.stops,
      )..where((s) => s.tripId.equals(tripId))).get();
      return buildTimeline(entries, {for (final s in stops) s.id: s});
    });

/// A note, with photos already copied into the app's folder.
Future<int> addTimelineNote(
  AppDatabase db, {
  required int tripId,
  int? stopId,
  required String text,
  List<String> photoPaths = const [],
  DateTime? at,
}) => db.into(db.timelineEntries).insert(
  TimelineEntriesCompanion.insert(
    tripId: tripId,
    stopId: Value(stopId),
    kind: TimelineKind.note,
    body: Value(text.trim().isEmpty ? null : text.trim()),
    photoPaths: Value(photoPaths.isEmpty ? null : photoPaths.join('\n')),
    occurredAt: at ?? DateTime.now(),
  ),
);

Future<void> deleteTimelineEntry(AppDatabase db, int id) =>
    (db.delete(db.timelineEntries)..where((t) => t.id.equals(id))).go();

/// Stores one GPS fix of the route log.
Future<int> addFix(
  AppDatabase db, {
  required int tripId,
  required LatLng at,
  required double accuracyM,
  required DateTime time,
}) => db.into(db.timelineEntries).insert(
  TimelineEntriesCompanion.insert(
    tripId: tripId,
    kind: TimelineKind.fix,
    lat: Value(at.lat),
    lon: Value(at.lon),
    accuracyM: Value(accuracyM),
    occurredAt: time,
  ),
);

/// The logged track of a trip, oldest first, for the map.
Future<List<LatLng>> loggedTrack(AppDatabase db, int tripId) async => [
  for (final e
      in await (db.select(db.timelineEntries)
            ..where(
              (t) => t.tripId.equals(tripId) & t.kind.equals(TimelineKind.fix),
            )
            ..orderBy([(t) => OrderingTerm(expression: t.occurredAt)]))
          .get())
    if (e.lat != null && e.lon != null) LatLng(e.lat!, e.lon!),
];

/// Photo paths of an entry.
List<String> photosOf(TimelineEntry e) => [
  for (final p in (e.photoPaths ?? '').split('\n'))
    if (p.trim().isNotEmpty) p.trim(),
];
