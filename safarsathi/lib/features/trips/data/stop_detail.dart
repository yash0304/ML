// lib/features/trips/data/stop_detail.dart
//
// Everything one stop holds — issue #51. SCREENS.md §10.

import 'package:drift/drift.dart';

import '../../../core/database/app_database.dart';
import '../../weather/data/weather_client.dart';
import '../../contacts/data/contacts_dao.dart' show ContactCategory;
import 'trip_editor.dart';

class StopDetail {
  final Stop stop;

  /// The frozen forecast, oldest day first.
  final List<WeatherSnapshot> weather;

  /// Newest `cachedAt` across those days, or null when nothing is stored.
  final DateTime? weatherCachedAt;

  /// Diary entries attached to this stop.
  final int diaryCount;

  /// How many of those still carry the amber dot. The number that decides
  /// whether this stop blocks departure.
  final int unconfirmedCount;

  /// Checklist items scoped to this stop.
  final int checklistCount;
  final int checklistDone;

  /// Corridor places on the legs either side of this stop.
  final int nearbyPlaceCount;

  /// When the legs touching this stop were last downloaded.
  final DateTime? lastSyncedAt;

  /// Where you said you sleep here, or null when not decided.
  final Contact? stay;

  /// Accommodation numbers saved here — what can be chosen.
  final int stayOptionCount;

  const StopDetail({
    required this.stop,
    required this.weather,
    required this.diaryCount,
    required this.unconfirmedCount,
    required this.checklistCount,
    required this.checklistDone,
    required this.nearbyPlaceCount,
    this.weatherCachedAt,
    this.lastSyncedAt,
    this.stay,
    this.stayOptionCount = 0,
  });

  List<String> get tags => parseTags(stop.activityTags);
  bool get isOvernight => stop.nights > 0;
  bool get hasCoordinates => stop.lat != null && stop.lon != null;

  Staleness? weatherStaleness({DateTime? now}) => weatherCachedAt == null
      ? null
      : stalenessOf(weatherCachedAt!, now: now);

  String? weatherAge({DateTime? now}) =>
      weatherCachedAt == null ? null : describeAge(weatherCachedAt!, now: now);
}

Stream<StopDetail> watchStopDetail(AppDatabase db, int stopId) {
  final tick = db
      .customSelect(
        'SELECT 1',
        readsFrom: {
          db.stops,
          db.legs,
          db.pois,
          db.contacts,
          db.checklistItems,
          db.weatherSnapshots,
        },
      )
      .watch();

  return tick.asyncMap((_) async {
    final stop = await (db.select(
      db.stops,
    )..where((s) => s.id.equals(stopId))).getSingle();

    final weather =
        await (db.select(db.weatherSnapshots)
              ..where((w) => w.stopId.equals(stopId))
              ..orderBy([(w) => OrderingTerm(expression: w.forDate)]))
            .get();

    DateTime? newest;
    for (final day in weather) {
      if (newest == null || day.cachedAt.isAfter(newest)) newest = day.cachedAt;
    }

    final contacts = await (db.select(
      db.contacts,
    )..where((c) => c.stopId.equals(stopId))).get();

    final checklist = await (db.select(
      db.checklistItems,
    )..where((i) => i.stopId.equals(stopId))).get();

    // Legs touching this stop, either arriving or leaving. Their corridor is
    // what "nearby" means for a stop that has no corridor of its own.
    final legs = await (db.select(db.legs)..where(
          (l) => l.fromStopId.equals(stopId) | l.toStopId.equals(stopId),
        ))
        .get();

    DateTime? synced;
    for (final leg in legs) {
      final at = leg.lastSyncedAt;
      if (at != null && (synced == null || at.isAfter(synced))) synced = at;
    }

    final places = legs.isEmpty
        ? 0
        : (await (db.select(
                db.pois,
              )..where((p) => p.legId.isIn([for (final l in legs) l.id])))
              .get())
              .length;

    return StopDetail(
      stop: stop,
      weather: weather,
      weatherCachedAt: newest,
      diaryCount: contacts.length,
      unconfirmedCount: contacts.where((c) => !c.callConfirmed).length,
      checklistCount: checklist.length,
      checklistDone: checklist.where((i) => i.isDone).length,
      nearbyPlaceCount: places,
      lastSyncedAt: synced,
      // By id, not among this stop's contacts: a stay whose entry has since
      // been attached elsewhere is still the stay until changed here.
      stay: stop.stayContactId == null
          ? null
          : await (db.select(
              db.contacts,
            )..where((c) => c.id.equals(stop.stayContactId!))).getSingleOrNull(),
      stayOptionCount: contacts
          .where((c) => c.category == ContactCategory.accommodation)
          .length,
    );
  });
}
