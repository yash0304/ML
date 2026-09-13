// lib/features/discovery/data/corridor_sync.dart
//
// One leg's worth of downloading — the part of #25 that #21 to #23 can carry
// on their own, before the orchestrator and the map exist.
//
// Route the leg, query its corridor, place every result along the line, and
// write the lot. Runs once, at setup, when the user presses a button that says
// what it is about to do.

import 'package:drift/drift.dart';

import '../../../core/database/app_database.dart';
import 'corridor.dart';
import 'geo.dart';
import 'osrm_client.dart';
import 'overpass_client.dart';
import 'poi_category.dart';

class LegSyncResult {
  final int legId;
  final double distanceKm;
  final int poisFound;
  final int phonesFound;

  const LegSyncResult({
    required this.legId,
    required this.distanceKm,
    required this.poisFound,
    required this.phonesFound,
  });
}

/// What a sync is about to cost, shown BEFORE it runs.
///
/// A user on a hotel WiFi deserves to know how much they are about to pull and
/// from whom. An app that just starts downloading is an app you stop trusting
/// the moment you notice.
class SyncEstimate {
  final int legCount;
  final double totalRouteKm;
  final double corridorAreaSqKm;

  const SyncEstimate({
    required this.legCount,
    required this.totalRouteKm,
    required this.corridorAreaSqKm,
  });

  /// Deliberately vague, because it is a guess. Route lines are small; the
  /// places are the bulk, and how many exist is unknowable before asking.
  String get roughSize =>
      corridorAreaSqKm < 500 ? 'well under a megabyte' : 'a megabyte or two';
}

class CorridorSync {
  final AppDatabase db;
  final OsrmClient osrm;
  final OverpassClient overpass;

  CorridorSync({
    required this.db,
    OsrmClient? osrm,
    OverpassClient? overpass,
  }) : osrm = osrm ?? OsrmClient(),
       overpass = overpass ?? OverpassClient();

  /// Downloads one leg: its route, then the places along it.
  ///
  /// The whole leg is written in one transaction at the end, so a failure
  /// halfway leaves the leg exactly as it was rather than half-synced with a
  /// `lastSyncedAt` implying otherwise.
  Future<LegSyncResult> syncLeg(
    int legId, {
    List<String> categoryKeys = defaultPoiCategoryKeys,
  }) async {
    final leg = await (db.select(
      db.legs,
    )..where((l) => l.id.equals(legId))).getSingle();

    final stops = await (db.select(
      db.stops,
    )..where((s) => s.id.isIn([leg.fromStopId, leg.toStopId]))).get();
    final byId = {for (final s in stops) s.id: s};

    final from = byId[leg.fromStopId];
    final to = byId[leg.toStopId];
    if (from?.lat == null ||
        from?.lon == null ||
        to?.lat == null ||
        to?.lon == null) {
      throw const OsrmException(
        'This leg has a stop with no coordinates yet, so there is nothing to '
        'route between.',
      );
    }

    final result = await osrm.route(
      LatLng(from!.lat!, from.lon!),
      LatLng(to!.lat!, to.lon!),
    );

    final corridor = Corridor(result.points, bufferKm: leg.corridorKm);
    final found = await overpass.search(
      corridor.queryBox,
      categoryKeys: categoryKeys,
    );
    final placed = found.isEmpty
        ? const <({PoiDraft item, CorridorPosition position})>[]
        : corridor.place(found, (p) => p.location);

    var phones = 0;

    await db.transaction(() async {
      await (db.update(db.legs)..where((l) => l.id.equals(legId))).write(
        LegsCompanion(
          routePolyline: Value(result.encodedPolyline),
          distanceKm: Value(result.distanceKm),
          lastSyncedAt: Value(DateTime.now()),
        ),
      );

      // Replaced wholesale. A re-sync is the user asking for what is there
      // now, and merging would leave places that have since closed.
      await (db.delete(db.pois)..where((p) => p.legId.equals(legId))).go();

      for (final entry in placed) {
        final poiId = await db
            .into(db.pois)
            .insert(
              PoisCompanion.insert(
                tripId: leg.tripId,
                legId: Value(legId),
                name: entry.item.name,
                category: entry.item.category,
                lat: entry.item.location.lat,
                lon: entry.item.location.lon,
                osmId: Value(entry.item.osmId),
                distanceAlongRouteKm: Value(entry.position.alongRouteKm),
                distanceOffRouteKm: Value(entry.position.offRouteKm),
              ),
            );

        for (final phone in entry.item.phones) {
          // THE TIER IS FORCED HERE and nowhere else decides it. A number a
          // stranger typed into a public wiki is `communityOsm`, always.
          await db
              .into(db.poiContacts)
              .insert(
                PoiContactsCompanion.insert(
                  poiId: poiId,
                  phoneRaw: phone.raw,
                  tier: const Value('communityOsm'),
                  sourceTag: Value(phone.sourceTag),
                ),
              );
          phones++;
        }
      }
    });

    return LegSyncResult(
      legId: legId,
      distanceKm: result.distanceKm,
      poisFound: placed.length,
      phonesFound: phones,
    );
  }

  /// What syncing this trip would involve, without doing any of it.
  ///
  /// Uses the straight-line distance between stops, since the real route is
  /// exactly what has not been fetched yet. It underestimates, and the screen
  /// says so rather than pretending to a precision it does not have.
  Future<SyncEstimate> estimate(int tripId) async {
    final legs = await (db.select(
      db.legs,
    )..where((l) => l.tripId.equals(tripId))).get();
    final stops = await (db.select(
      db.stops,
    )..where((s) => s.tripId.equals(tripId))).get();
    final byId = {for (final s in stops) s.id: s};

    var totalKm = 0.0;
    var area = 0.0;

    for (final leg in legs) {
      final from = byId[leg.fromStopId];
      final to = byId[leg.toStopId];
      if (from?.lat == null ||
          from?.lon == null ||
          to?.lat == null ||
          to?.lon == null) {
        continue;
      }
      final km =
          haversineMetres(
            LatLng(from!.lat!, from.lon!),
            LatLng(to!.lat!, to.lon!),
          ) /
          1000;
      totalKm += km;
      area += km * leg.corridorKm * 2;
    }

    return SyncEstimate(
      legCount: legs.length,
      totalRouteKm: totalKm,
      corridorAreaSqKm: area,
    );
  }
}
