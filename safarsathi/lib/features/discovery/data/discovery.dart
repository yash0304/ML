// lib/features/discovery/data/discovery.dart
//
// What is along a leg, as the screen needs it — issue #27.

import 'package:drift/drift.dart';

import '../../../core/database/app_database.dart';
import '../../contacts/data/contacts_dao.dart';
import 'place_details.dart';
import 'corridor.dart';
import 'geo.dart';
import 'polyline.dart';

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

  /// The OSM tags kept for it: food type, cuisine, veg, hours, description.
  /// Empty for places downloaded before these were kept.
  final Map<String, String> tags;

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
    this.tags = const {},
  });

  bool get hasPhone => phones.isNotEmpty;
}

/// One of the user's own diary contacts, placed on a leg.
class LegContact {
  final Contact contact;
  final double alongRouteKm;
  final double offRouteKm;

  const LegContact({
    required this.contact,
    required this.alongRouteKm,
    required this.offRouteKm,
  });
}

/// A hospital or pharmacy near a stop that has none of its own.
class NearbyHelp {
  final Contact contact;

  /// As the crow flies. In these hills the road is longer, and the screen
  /// says so rather than letting 13 km read as a quarter of an hour.
  final double straightLineKm;

  const NearbyHelp({required this.contact, required this.straightLineKm});
}

class LegDiscovery {
  final int legId;
  final String fromName;
  final String toName;
  final double? distanceKm;
  final DateTime? lastSyncedAt;
  final List<CorridorPlace> places;

  /// The user's own numbers that lie along this road, in the order they come
  /// up. Only contacts with a stored position can be here; ones attached to
  /// either end of the leg are left to those stops, so the start of every
  /// leg is not buried under the whole of the town being left.
  final List<LegContact> onTheWay;

  /// The user's own numbers at the stop this leg arrives at, the ones that
  /// matter if something goes wrong on arrival first.
  final List<Contact> atDestination;

  /// False when there is no line to measure along — no route yet and no
  /// coordinates on one of the stops — so the screen can say why "on the
  /// way" is empty rather than implying there is nothing there.
  final bool canPlace;

  /// Diary contacts that could have been on this road but have no position
  /// saved. Without this, an empty "on the way" reads as "nothing is there"
  /// when the truth is "nothing here knows where anything is" — which is the
  /// state of every contact imported before coordinates were read.
  final int unplaced;

  /// Whether anything in this trip's diary has a position at all. The
  /// "no location saved" hint is only true while this is false: once a
  /// sheet with coordinates is in, a few helplines with no place — 1363,
  /// an embassy line — must not nag on every leg forever.
  final bool anyPlaced;

  /// The closest hospitals and pharmacies to the arrival stop, when that stop
  /// has neither of its own in the diary. "Kongthong has no pharmacy; the
  /// nearest is in Pynursla" is the sentence this exists to say.
  final List<NearbyHelp> nearestHelp;

  const LegDiscovery({
    required this.legId,
    required this.fromName,
    required this.toName,
    required this.places,
    this.distanceKm,
    this.lastSyncedAt,
    this.onTheWay = const [],
    this.atDestination = const [],
    this.canPlace = true,
    this.unplaced = 0,
    this.anyPlaced = false,
    this.nearestHelp = const [],
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
        // CONTACTS IS HERE ON PURPOSE. Importing a sheet, confirming a number
        // or editing a note all change what this leg shows, and a Drift
        // stream only fires for the tables it names — leave this out and the
        // leg screen goes stale with no error anywhere.
        readsFrom: {db.legs, db.stops, db.pois, db.poiContacts, db.contacts},
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
    final stopById = {for (final s in stops) s.id: s};

    final diary = await (db.select(
      db.contacts,
    )..where((c) => c.tripId.equals(leg.tripId))).get();

    final corridor = legCorridor(
      leg,
      from: stopById[leg.fromStopId],
      to: stopById[leg.toStopId],
    );

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
            tags: decodeKeptTags(p.rawTags),
          ),
      ],
      onTheWay: corridor == null
          ? const []
          : contactsOnTheWay(
              corridor,
              diary,
              endStopIds: {leg.fromStopId, leg.toStopId},
            ),
      atDestination: contactsAtStop(diary, leg.toStopId),
      canPlace: corridor != null,
      unplaced: unplacedContacts(
        diary,
        endStopIds: {leg.fromStopId, leg.toStopId},
      ),
      anyPlaced: diary.any((c) => c.lat != null && c.lon != null),
      nearestHelp: nearestHelpTo(stopById[leg.toStopId], diary),
    );
  });
}

/// The line a leg's places are measured along.
///
/// The real route when there is one. Before routing has run, the straight
/// line between the two stops stands in — wrong in the hills, but the same
/// fallback the map download uses, and it errs wide rather than narrow.
/// Null only when neither is available.
Corridor? legCorridor(Leg leg, {Stop? from, Stop? to}) {
  final polyline = leg.routePolyline;
  final List<LatLng> points;
  if (polyline != null) {
    points = Polyline.decode(polyline);
  } else if (from?.lat != null &&
      from?.lon != null &&
      to?.lat != null &&
      to?.lon != null) {
    points = [LatLng(from!.lat!, from.lon!), LatLng(to!.lat!, to.lon!)];
  } else {
    return null;
  }
  if (points.length < 2) return null;
  return Corridor(points, bufferKm: leg.corridorKm);
}

/// Diary contacts inside [corridor], in the order the road reaches them.
///
/// Contacts at either end of the leg are left out: they belong to those
/// stops, and on a leg out of Shillong they would otherwise fill the top of
/// the list with twenty-seven city numbers at kilometre nought.
List<LegContact> contactsOnTheWay(
  Corridor corridor,
  List<Contact> diary, {
  required Set<int> endStopIds,
}) {
  final placeable = [
    for (final c in diary)
      if (c.lat != null && c.lon != null && !endStopIds.contains(c.stopId)) c,
  ];
  return [
    for (final hit in corridor.place(
      placeable,
      (c) => LatLng(c.lat!, c.lon!),
    ))
      LegContact(
        contact: hit.item,
        alongRouteKm: hit.position.alongRouteKm,
        offRouteKm: hit.position.offRouteKm,
      ),
  ];
}

/// Contacts that might belong on a leg but carry no position.
///
/// Emergency numbers are not counted: 112 has no location and never will,
/// and "3 numbers could not be placed" when all three are helplines would
/// send someone looking for a fix that does not exist.
///
/// Nor are short codes and toll-free lines, emergency-flagged or not: 181,
/// 1098 and 1800 11 1363 are services, not places, and a sheet does not
/// always tick them as emergency numbers.
int unplacedContacts(List<Contact> diary, {required Set<int> endStopIds}) => [
  for (final c in diary)
    if ((c.lat == null || c.lon == null) &&
        !c.isEmergency &&
        !isServiceNumber(c.phoneE164 ?? c.phoneRaw) &&
        !endStopIds.contains(c.stopId))
      c,
].length;

/// A number that reaches a service rather than a place: a short code
/// (112, 181, 1098, 1363) or an Indian toll-free 1800 line.
bool isServiceNumber(String phone) {
  final digits = phone.replaceAll(RegExp(r'\D'), '');
  if (digits.length <= 6) return true;
  return digits.startsWith('1800') || digits.startsWith('911800');
}

/// Help categories: what somebody needs within the hour, not the evening.
const helpCategories = {ContactCategory.hospital, ContactCategory.pharmacy};

/// How far to look for help from a stop that has none. Pynursla to Kongthong
/// is about 13 km in a straight line; Jowai to Dawki about 30.
const nearestHelpRadiusKm = 35.0;

/// The closest hospitals and pharmacies to [stop], nearest first — but ONLY
/// when the stop has no hospital or pharmacy of its own in the diary.
///
/// Shown for every stop, this would list Shillong's suburbs under Shillong,
/// which already has twenty-seven numbers. It earns its place only where the
/// answer to "where is the nearest chemist" is somewhere else.
List<NearbyHelp> nearestHelpTo(
  Stop? stop,
  List<Contact> diary, {
  int limit = 3,
}) {
  if (stop == null || stop.lat == null || stop.lon == null) return const [];
  final hasOwn = diary.any(
    (c) => c.stopId == stop.id && helpCategories.contains(c.category),
  );
  if (hasOwn) return const [];

  final here = LatLng(stop.lat!, stop.lon!);
  final found = <NearbyHelp>[
    for (final c in diary)
      if (c.stopId != stop.id &&
          helpCategories.contains(c.category) &&
          c.lat != null &&
          c.lon != null)
        NearbyHelp(
          contact: c,
          straightLineKm: haversineMetres(here, LatLng(c.lat!, c.lon!)) / 1000,
        ),
  ].where((h) => h.straightLineKm <= nearestHelpRadiusKm).toList()
    ..sort((a, b) => a.straightLineKm.compareTo(b.straightLineKm));

  return found.take(limit).toList();
}

/// The order a person arriving somewhere needs things in: help first, then a
/// bed, then food. Anything unlisted follows, alphabetically.
const arrivalOrder = [
  ContactCategory.emergency,
  ContactCategory.hospital,
  ContactCategory.pharmacy,
  ContactCategory.accommodation,
  ContactCategory.restaurant,
  ContactCategory.transport,
];

/// Diary contacts attached to [stopId], help first.
List<Contact> contactsAtStop(List<Contact> diary, int stopId) {
  int rank(Contact c) {
    final i = arrivalOrder.indexOf(c.category);
    return i == -1 ? arrivalOrder.length : i;
  }

  return [
    for (final c in diary)
      if (c.stopId == stopId) c,
  ]..sort((a, b) {
      final byRank = rank(a).compareTo(rank(b));
      return byRank != 0
          ? byRank
          : a.name.toLowerCase().compareTo(b.name.toLowerCase());
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
