// test/corridor_sync_test.dart — the three issues working together.
//
// Both clients are faked, so this runs with no network and exercises the part
// that matters: what actually lands in the database.

import 'package:drift/drift.dart' hide isNull, isNotNull;
import 'package:drift/native.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:safarsathi/core/database/app_database.dart';
import 'package:safarsathi/features/discovery/data/corridor_sync.dart';
import 'package:safarsathi/features/discovery/data/discovery.dart';
import 'package:safarsathi/features/discovery/data/geo.dart';
import 'package:safarsathi/features/discovery/data/osrm_client.dart';
import 'package:safarsathi/features/discovery/data/overpass_client.dart';
import 'package:safarsathi/features/discovery/data/polyline.dart';
import 'package:safarsathi/features/trips/data/trip_editor.dart';

const shillong = LatLng(25.5788, 91.8933);
const cherrapunji = LatLng(25.2702, 91.7323);

/// A straight line between the two, as OSRM would return it.
String get _route => Polyline.encode(const [
  shillong,
  LatLng(25.42, 91.81),
  cherrapunji,
]);

String osrmBody({double distanceMetres = 54000}) =>
    '{"code":"Ok","routes":[{"geometry":"$_route",'
    '"distance":$distanceMetres,"duration":5400.0}]}';

/// A dhaba right on the line, a hospital just off it, and a fuel station in
/// the corner of the bounding box that the road never goes near.
const _overpassBody = '''
{"elements": [
  {"type":"node","id":1,"lat":25.42,"lon":91.81,
   "tags":{"amenity":"restaurant","name":"Roadside dhaba",
           "phone":"+91 364 111 1111"}},
  {"type":"node","id":2,"lat":25.4215,"lon":91.812,
   "tags":{"amenity":"hospital","name":"Sohra PHC",
           "contact:phone":"+91 364 222 2222;+91 364 222 3333"}},
  {"type":"node","id":3,"lat":25.58,"lon":91.73,
   "tags":{"amenity":"fuel","name":"Far corner pump"}}
]}
''';

void main() {
  late AppDatabase db;
  late TripEditor editor;
  late int tripId;
  late int legId;

  setUp(() async {
    db = AppDatabase(NativeDatabase.memory());
    editor = TripEditor(db);
    tripId = await editor.createTrip(name: 'Meghalaya');

    final from = await editor.addStop(
      tripId,
      const StopDraft(name: 'Shillong'),
    );
    await editor.addStop(tripId, const StopDraft(name: 'Cherrapunji'));

    // Coordinates are not part of StopDraft yet; they arrive at #24 with the
    // map picker. Until then the sync needs them set directly.
    await (db.update(db.stops)..where((s) => s.id.equals(from))).write(
      const StopsCompanion(lat: Value(25.5788), lon: Value(91.8933)),
    );
    final stops = await editor.stopsOf(tripId);
    await (db.update(db.stops)..where((s) => s.id.equals(stops.last.id)))
        .write(
          const StopsCompanion(lat: Value(25.2702), lon: Value(91.7323)),
        );

    legId = (await db.select(db.legs).getSingle()).id;
  });

  tearDown(() => db.close());

  CorridorSync syncWith({String? overpass, String? osrm}) => CorridorSync(
    db: db,
    osrm: OsrmClient(fetch: (_) async => osrm ?? osrmBody()),
    overpass: OverpassClient(fetch: (_) async => overpass ?? _overpassBody),
  );

  test('WHAT A FOOD PLACE SERVES SURVIVES THE DOWNLOAD', () async {
    // The download fetched cuisine, veg and hours and kept none of it; the
    // rawTags column existed and nothing wrote to it.
    await syncWith(
      overpass: '{"elements": [{"type":"node","id":7,"lat":25.42,'
          '"lon":91.81,"tags":{"amenity":"fast_food","name":"Momo Point",'
          '"cuisine":"indian;chinese;momo","diet:vegetarian":"yes",'
          '"opening_hours":"Mo-Su 09:00-21:00","operator":"not kept"}}]}',
    ).syncLeg(legId);

    final place = (await watchLegDiscovery(db, legId).first).places.single;
    expect(place.tags['cuisine'], 'indian;chinese;momo');
    expect(place.tags['diet:vegetarian'], 'yes');
    expect(place.tags['opening_hours'], 'Mo-Su 09:00-21:00');
    expect(place.tags.containsKey('operator'), isFalse,
        reason: 'only the tags worth carrying offline are kept');
  });

  test('a synced leg gets its route, distance and timestamp', () async {
    final result = await syncWith().syncLeg(legId);

    final leg = await (db.select(
      db.legs,
    )..where((l) => l.id.equals(legId))).getSingle();

    expect(leg.routePolyline, isNotNull);
    expect(leg.distanceKm, closeTo(54, 0.1));
    expect(leg.lastSyncedAt, isNotNull);
    expect(result.distanceKm, closeTo(54, 0.1));
  });

  test('places inside the corridor are stored, the far one is not', () async {
    final result = await syncWith().syncLeg(legId);

    final pois = await db.select(db.pois).get();
    expect(result.poisFound, 2);
    expect(pois.map((p) => p.name), containsAll(['Roadside dhaba', 'Sohra PHC']));
    // The bounding box is a rectangle and the corridor is not, so a query
    // returns corners the route never goes near.
    expect(pois.map((p) => p.name), isNot(contains('Far corner pump')));
  });

  test('each place records how far along and how far off the route', () async {
    await syncWith().syncLeg(legId);

    final dhaba = await (db.select(
      db.pois,
    )..where((p) => p.name.equals('Roadside dhaba'))).getSingle();

    // It sits on a vertex, so it is on the line and roughly halfway along.
    expect(dhaba.distanceOffRouteKm, lessThan(0.05));
    expect(dhaba.distanceAlongRouteKm, closeTo(19, 3));
  });

  test('a place is attached to its leg, never to a stop', () async {
    await syncWith().syncLeg(legId);
    final pois = await db.select(db.pois).get();

    expect(pois.every((p) => p.legId == legId), isTrue);
    expect(pois.every((p) => p.stopId == null), isTrue);
  });

  test('EVERY OSM NUMBER LANDS AS communityOsm', () async {
    // The trust invariant reaching the least trustworthy source the app has:
    // a number a stranger typed into a public wiki.
    final result = await syncWith().syncLeg(legId);
    final contacts = await db.select(db.poiContacts).get();

    expect(result.phonesFound, 3);
    expect(contacts.every((c) => c.tier == 'communityOsm'), isTrue);
  });

  test('a semicolon list becomes several numbers, each with its tag', () async {
    await syncWith().syncLeg(legId);

    final phc = await (db.select(
      db.pois,
    )..where((p) => p.name.equals('Sohra PHC'))).getSingle();
    final contacts = await (db.select(
      db.poiContacts,
    )..where((c) => c.poiId.equals(phc.id))).get();

    expect(contacts.length, 2);
    expect(contacts.every((c) => c.sourceTag == 'contact:phone'), isTrue);
  });

  test('RE-SYNCING REPLACES, it does not accumulate', () async {
    // A re-sync is the user asking for what is there now. Merging would leave
    // places that have since closed.
    await syncWith().syncLeg(legId);
    await syncWith().syncLeg(legId);

    expect((await db.select(db.pois).get()).length, 2);
    expect((await db.select(db.poiContacts).get()).length, 3);
  });

  test('a re-sync with fewer results drops the ones that are gone', () async {
    await syncWith().syncLeg(legId);
    await syncWith(
      overpass: '{"elements": [{"type":"node","id":1,"lat":25.42,'
          '"lon":91.81,"tags":{"amenity":"restaurant",'
          '"name":"Roadside dhaba"}}]}',
    ).syncLeg(legId);

    final pois = await db.select(db.pois).get();
    expect(pois.length, 1);
    expect(pois.single.name, 'Roadside dhaba');
  });

  test('A FAILED SYNC LEAVES THE LEG EXACTLY AS IT WAS', () async {
    // Half-synced with a lastSyncedAt implying otherwise is worse than
    // not synced at all.
    final sync = CorridorSync(
      db: db,
      osrm: OsrmClient(fetch: (_) async => osrmBody()),
      overpass: OverpassClient(
        fetch: (_) async => throw const OverpassException('busy'),
      ),
    );

    await expectLater(sync.syncLeg(legId), throwsA(isA<OverpassException>()));

    final leg = await (db.select(
      db.legs,
    )..where((l) => l.id.equals(legId))).getSingle();
    expect(leg.lastSyncedAt, isNull);
    expect(leg.routePolyline, isNull);
    expect(await db.select(db.pois).get(), isEmpty);
  });

  test('a stop with no coordinates says so instead of routing nowhere',
      () async {
    await (db.update(db.stops)).write(
      const StopsCompanion(lat: Value(null), lon: Value(null)),
    );

    await expectLater(
      syncWith().syncLeg(legId),
      throwsA(
        isA<OsrmException>().having(
          (e) => e.message,
          'message',
          contains('no coordinates'),
        ),
      ),
    );
  });

  test('a leg with no route found reports it and writes nothing', () async {
    await expectLater(
      syncWith(osrm: '{"code":"NoRoute","routes":[]}').syncLeg(legId),
      throwsA(isA<OsrmException>()),
    );
    expect(await db.select(db.pois).get(), isEmpty);
  });

  group('the estimate', () {
    test('says what a sync would cover, before running it', () async {
      final estimate = await syncWith().estimate(tripId);

      expect(estimate.legCount, 1);
      // Straight-line, which underestimates — the real route is the thing not
      // yet fetched.
      expect(estimate.totalRouteKm, closeTo(37, 3));
      expect(estimate.corridorAreaSqKm, greaterThan(0));
      expect(estimate.roughSize, isNotEmpty);
    });

    test('a leg with no coordinates is skipped, not counted as zero', () async {
      await (db.update(db.stops)).write(
        const StopsCompanion(lat: Value(null), lon: Value(null)),
      );
      final estimate = await syncWith().estimate(tripId);

      expect(estimate.legCount, 1);
      expect(estimate.totalRouteKm, 0);
    });
  });
}
