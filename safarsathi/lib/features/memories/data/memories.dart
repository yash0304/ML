// lib/features/memories/data/memories.dart
//
// The trip's photos, as an album — the Memories tab.
//
// NO NEW TABLE. A memory is a timeline note that carries photos, so the same
// picture shows on the Timeline where it happened and in the album, and
// deleting it in one place deletes it in both. The photo files live in the
// app's own folder on this phone; the database holds their paths.

import 'package:drift/drift.dart';

import '../../../core/database/app_database.dart';
import '../../../core/database/watch_tables.dart';
import '../../trips/data/timeline.dart';

class Memory {
  final TimelineEntry entry;
  final String path;

  /// Where it was filed, if anywhere.
  final String? place;

  const Memory({required this.entry, required this.path, this.place});

  String? get caption => entry.body;
  DateTime get at => entry.occurredAt;
}

class MemoryDay {
  final DateTime day;

  /// The places the day's photos were filed under, in order, without
  /// repeats: "Cherrapunji · Dawki".
  final List<String> places;
  final List<Memory> photos;

  const MemoryDay({
    required this.day,
    required this.places,
    required this.photos,
  });
}

/// Every photo on the trip, one by one, grouped by day, oldest first — the
/// order the trip happened in.
List<MemoryDay> buildMemories(
  List<TimelineEntry> entries,
  Map<int, Stop> stops,
) {
  final withPhotos = [
    for (final e in entries)
      if (e.kind != TimelineKind.fix && photosOf(e).isNotEmpty) e,
  ]..sort((a, b) => a.occurredAt.compareTo(b.occurredAt));

  final days = <DateTime, List<Memory>>{};
  for (final e in withPhotos) {
    final day = DateTime(
      e.occurredAt.year,
      e.occurredAt.month,
      e.occurredAt.day,
    );
    final place = stops[e.stopId]?.name;
    for (final p in photosOf(e)) {
      (days[day] ??= []).add(Memory(entry: e, path: p, place: place));
    }
  }
  return [
    for (final MapEntry(key: day, value: photos) in days.entries)
      MemoryDay(
        day: day,
        places: [
          for (final (i, m) in photos.indexed)
            if (m.place != null &&
                !photos.take(i).any((o) => o.place == m.place))
              m.place!,
        ],
        photos: photos,
      ),
  ];
}

Stream<List<MemoryDay>> watchMemories(AppDatabase db, int tripId) =>
    watchTables(db, {db.timelineEntries, db.stops}).asyncMap((_) async {
      final entries = await (db.select(db.timelineEntries)
            ..where(
              (t) =>
                  t.tripId.equals(tripId) &
                  t.photoPaths.isNotNull() &
                  t.kind.equals(TimelineKind.fix).not(),
            ))
          .get();
      final stops = await (db.select(
        db.stops,
      )..where((s) => s.tripId.equals(tripId))).get();
      return buildMemories(entries, {for (final s in stops) s.id: s});
    });

/// When to file photos added under [stop]. Added on the day, they are
/// filed now. Added afterwards — the album filled in at home — they go
/// under the day the trip reached that stop, so the album keeps the trip's
/// order rather than the order of tidying up.
DateTime memoryDate(Stop? stop, DateTime now) {
  final arrived = stop?.arrivalDate;
  if (arrived == null) return now;
  final from = DateTime(arrived.year, arrived.month, arrived.day);
  final left = stop!.departureDate ?? from.add(Duration(days: stop.nights));
  final until = DateTime(left.year, left.month, left.day + 1);
  if (!now.isBefore(from) && now.isBefore(until)) return now;
  return DateTime(from.year, from.month, from.day, 12);
}

/// Takes one photo out of its note. A note left with no photo and no words
/// goes too; one with words stays on the Timeline as a plain note.
Future<void> removeMemory(AppDatabase db, Memory m) async {
  final rest = [
    for (final p in photosOf(m.entry))
      if (p != m.path) p,
  ];
  if (rest.isEmpty && (m.entry.body ?? '').trim().isEmpty) {
    await deleteTimelineEntry(db, m.entry.id);
    return;
  }
  await (db.update(
    db.timelineEntries,
  )..where((t) => t.id.equals(m.entry.id))).write(
    TimelineEntriesCompanion(
      photoPaths: Value(rest.isEmpty ? null : rest.join('\n')),
    ),
  );
}

/// Changes the words under a photo — they belong to its note, so every
/// photo added with it shares them.
Future<void> setMemoryCaption(AppDatabase db, int entryId, String text) =>
    (db.update(db.timelineEntries)..where((t) => t.id.equals(entryId))).write(
      TimelineEntriesCompanion(
        body: Value(text.trim().isEmpty ? null : text.trim()),
      ),
    );
