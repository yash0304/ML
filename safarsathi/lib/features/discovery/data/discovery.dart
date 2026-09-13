// lib/features/discovery/data/discovery.dart
//
// What is along a leg, as the screen needs it — issue #27.

import 'package:drift/drift.dart';

import '../../../core/database/app_database.dart';

/// One place on the corridor, with everything the list and detail need.
class CorridorPlace {
  final int id;
  final String name;
  final String category;
  final double lat;
  final double lon;

  /// How far along the leg it sits. The ordering, and the "in 12 km" line.
  final double alongRouteKm;

  /// How far off the road it is. The detour, which is the other half of
  /// deciding whether to stop.
  final double offRouteKm;

  final String? osmId;

  /// Numbers OpenStreetMap holds for it. ALWAYS `communityOsm`.
  final List<PoiContact> phones;

  const CorridorPlace({
    required this.id,
    required this.name,
    required this.category,
    required this.lat,
    required this.lon,
    required this.alongRouteKm,
    required this.offRouteKm,
    this.osmId,
    this.phones = const [],
  });

  bool get hasPhone => phones.isNotEmpty;
}

class LegDiscovery {
  final int legId;
  final String fromName;
  final String toName;
  final double? distanceKm;
  final DateTime? lastSyncedAt;
  final List<CorridorPlace> places;

  const LegDiscovery({
    required this.legId,
    required this.fromName,
    required this.toName,
    required this.places,
    this.distanceKm,
    this.lastSyncedAt,
  });

  bool get isSynced => lastSyncedAt != null;

  /// Only the categories this leg actually has. Offering eleven chips when
  /// four match anything is a filter that mostly disappoints.
  List<String> get categoriesPresent {
    final seen = <String>{for (final p in places) p.category};
    return seen.toList()..sort();
  }
}

/// Everything along one leg, ordered by how far along it you will find it.
///
/// ORDERED BY DISTANCE ALONG THE ROUTE, NOT BY DISTANCE FROM YOU. "Coming up
/// in 12 km" is the useful sentence while moving; "0.2 km away" treats the
/// road as a plane, and a place 200 m off across a gorge is an hour of
/// driving.
Stream<LegDiscovery> watchLegDiscovery(AppDatabase db, int legId) {
  final tick = db
      .customSelect(
        'SELECT 1',
        readsFrom: {db.legs, db.stops, db.pois, db.poiContacts},
      )
      .watch();

  return tick.asyncMap((_) async {
    final leg = await (db.select(
      db.legs,
    )..where((l) => l.id.equals(legId))).getSingle();

    final stops = await (db.select(
      db.stops,
    )..where((s) => s.id.isIn([leg.fromStopId, leg.toStopId]))).get();
    final byId = {for (final s in stops) s.id: s.name};

    final pois =
        await (db.select(db.pois)
              ..where((p) => p.legId.equals(legId))
              ..orderBy([
                (p) => OrderingTerm(expression: p.distanceAlongRouteKm),
              ]))
            .get();

    final contacts = pois.isEmpty
        ? <PoiContact>[]
        : await (db.select(db.poiContacts)
                ..where((c) => c.poiId.isIn([for (final p in pois) p.id])))
              .get();

    final byPoi = <int, List<PoiContact>>{};
    for (final c in contacts) {
      byPoi.putIfAbsent(c.poiId, () => []).add(c);
    }

    return LegDiscovery(
      legId: legId,
      fromName: byId[leg.fromStopId] ?? '—',
      toName: byId[leg.toStopId] ?? '—',
      distanceKm: leg.distanceKm,
      lastSyncedAt: leg.lastSyncedAt,
      places: [
        for (final p in pois)
          CorridorPlace(
            id: p.id,
            name: p.name,
            category: p.category,
            lat: p.lat,
            lon: p.lon,
            alongRouteKm: p.distanceAlongRouteKm ?? 0,
            offRouteKm: p.distanceOffRouteKm ?? 0,
            osmId: p.osmId,
            phones: byPoi[p.id] ?? const [],
          ),
      ],
    );
  });
}

/// Saves a place's number into the diary.
///
/// IT LANDS AS `communityOsm`, NOT `userEntered`. The backlog said the latter;
/// both carry the amber dot so the trust outcome is identical, but the entry
/// screen reads provenance out loud — "Typed by you" versus "From open map
/// data, nobody has checked it". Marking an OSM number as typed by the user
/// would make the app lie about where it came from, which in an app built on
/// knowing that is the one thing it must not do.
///
/// It becomes `userVerified` the moment the user calls it and says so, exactly
/// like any other number.
Future<int> savePlaceAsContact(
  AppDatabase db, {
  required int tripId,
  required CorridorPlace place,
  required PoiContact phone,
  int? stopId,
}) => db
    .into(db.contacts)
    .insert(
      ContactsCompanion.insert(
        name: place.name,
        phoneRaw: phone.phoneRaw,
        tripId: Value(tripId),
        stopId: Value(stopId),
        phoneE164: Value(phone.phoneE164),
        category: Value(place.category),
        tier: const Value('communityOsm'),
        note: Value(
          place.osmId == null
              ? 'From map data'
              : 'From map data (${place.osmId})',
        ),
      ),
    );

/// The Google Maps handoff, for reviews and photos this app does not carry.
///
/// Offered, never followed automatically, and the screen says it needs signal:
/// a dead tap on a mountain road with no explanation is worse than no button.
String googleMapsUrl(CorridorPlace place) =>
    'https://www.google.com/maps/search/?api=1&query='
    '${place.lat.toStringAsFixed(6)},${place.lon.toStringAsFixed(6)}';
