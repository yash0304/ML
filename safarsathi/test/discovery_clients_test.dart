// test/discovery_clients_test.dart — issues #21 and #22.
//
// Both clients take an injected fetch, so every one of these runs with no
// network. The fixtures are trimmed real responses.

import 'package:flutter_test/flutter_test.dart';

import 'package:safarsathi/features/discovery/data/geo.dart';
import 'package:safarsathi/features/discovery/data/osrm_client.dart';
import 'package:safarsathi/features/discovery/data/overpass_client.dart';
import 'package:safarsathi/features/discovery/data/poi_category.dart';

const shillongBox = BoundingBox(
  south: 25.55,
  west: 91.85,
  north: 25.60,
  east: 91.92,
);

void main() {
  group('Overpass query', () {
    test('names the box and the categories', () {
      final query = OverpassClient.buildQuery(
        shillongBox,
        categoryKeys: const ['fuel'],
      );

      expect(query, contains('[out:json]'));
      expect(query, contains('amenity"="fuel'));
      expect(query, contains(shillongBox.overpassString));
      // nwr catches nodes, ways and relations; `out center` gives a way a
      // single pin rather than its whole outline.
      expect(query, contains('nwr'));
      expect(query, contains('out center tags'));
    });

    test('a category with several tags emits all of them', () {
      final query = OverpassClient.buildQuery(
        shillongBox,
        categoryKeys: const ['hospital'],
      );
      expect(query, contains('amenity"="hospital'));
      expect(query, contains('amenity"="clinic'));
      expect(query, contains('amenity"="doctors'));
    });

    test('AN OVERSIZED BOX IS REFUSED', () {
      // Overpass is free infrastructure paid for by volunteers. A query this
      // big is a bug, and it costs someone else money.
      const huge = BoundingBox(south: 8, west: 68, north: 37, east: 97);
      expect(
        () => OverpassClient.buildQuery(huge),
        throwsA(
          isA<OverpassException>().having(
            (e) => e.message,
            'message',
            contains('too large'),
          ),
        ),
      );
    });

    test('no categories is refused rather than silently returning nothing', () {
      expect(
        () => OverpassClient.buildQuery(shillongBox, categoryKeys: const []),
        throwsA(isA<OverpassException>()),
      );
    });

    test('an unknown category key is skipped, not fatal', () {
      final query = OverpassClient.buildQuery(
        shillongBox,
        categoryKeys: const ['fuel', 'teleportation'],
      );
      expect(query, contains('amenity"="fuel'));
    });
  });

  group('Overpass parsing', () {
    const body = '''
{
  "version": 0.6,
  "elements": [
    {
      "type": "node", "id": 1, "lat": 25.5788, "lon": 91.8933,
      "tags": {"amenity": "fuel", "name": "IOC Police Bazar",
               "phone": "+91 364 222 1111"}
    },
    {
      "type": "way", "id": 2,
      "center": {"lat": 25.5700, "lon": 91.8800},
      "tags": {"amenity": "hospital", "name": "Civil Hospital",
               "contact:phone": "+91 364 222 2222;+91 364 222 3333"}
    },
    {
      "type": "node", "id": 3, "lat": 25.5600, "lon": 91.8700,
      "tags": {"amenity": "fuel"}
    },
    {
      "type": "node", "id": 4, "lat": 25.5500, "lon": 91.8600,
      "tags": {"amenity": "bench", "name": "A bench"}
    },
    {
      "type": "node", "id": 5,
      "tags": {"amenity": "pharmacy", "name": "No coordinates here"}
    }
  ]
}
''';

    test('a node and a way both yield a place with a coordinate', () {
      final pois = OverpassClient.parse(body);
      final names = pois.map((p) => p.name);

      expect(names, contains('IOC Police Bazar'));
      expect(names, contains('Civil Hospital'));

      final hospital = pois.firstWhere((p) => p.name == 'Civil Hospital');
      // A way carries `center` rather than its own lat/lon.
      expect(hospital.location.lat, closeTo(25.57, 0.001));
    });

    test('AN UNNAMED PLACE IS DROPPED', () {
      // "Unnamed fuel station, 14 km ahead" tells you nothing you can act on
      // or ask a local for.
      final pois = OverpassClient.parse(body);
      expect(pois.where((p) => p.osmId == 'node/3'), isEmpty);
    });

    test('a category the app does not know is dropped', () {
      final pois = OverpassClient.parse(body);
      expect(pois.where((p) => p.name == 'A bench'), isEmpty);
    });

    test('an element with no coordinate anywhere is dropped', () {
      final pois = OverpassClient.parse(body);
      expect(pois.where((p) => p.osmId == 'node/5'), isEmpty);
    });

    test('phone and contact:phone are both read, with their source', () {
      final pois = OverpassClient.parse(body);

      final fuel = pois.firstWhere((p) => p.name == 'IOC Police Bazar');
      expect(fuel.phones.single.raw, '+91 364 222 1111');
      expect(fuel.phones.single.sourceTag, 'phone');

      final hospital = pois.firstWhere((p) => p.name == 'Civil Hospital');
      expect(hospital.phones.first.sourceTag, 'contact:phone');
    });

    test('a semicolon-separated list becomes several numbers', () {
      // The OSM convention, and common on hospitals.
      final pois = OverpassClient.parse(body);
      final hospital = pois.firstWhere((p) => p.name == 'Civil Hospital');

      expect(hospital.phones.length, 2);
      expect(hospital.phones[1].raw, '+91 364 222 3333');
    });

    test('A HOTEL WITH A RESTAURANT READS AS SOMEWHERE TO SLEEP', () {
      // The more specific tag wins. `tourism=hotel` says what the place IS;
      // `amenity=restaurant`, which half of them also carry, says what it
      // also does. Backwards, this files every guest house on the route under
      // Food, and at 8pm that is the wrong answer.
      expect(
        categoryForTags(const {'tourism': 'hotel', 'amenity': 'restaurant'}),
        'accommodation',
      );
    });

    test('a hospital with a pharmacy reads as a hospital', () {
      expect(
        categoryForTags(const {'amenity': 'hospital', 'shop': 'chemist'}),
        'hospital',
      );
    });

    test('malformed JSON fails with a sentence, not a crash', () {
      expect(
        () => OverpassClient.parse('not json at all'),
        throwsA(
          isA<OverpassException>().having(
            (e) => e.message,
            'message',
            contains('could not read'),
          ),
        ),
      );
    });

    test('a response with no elements list is refused', () {
      expect(
        () => OverpassClient.parse('{"version": 0.6}'),
        throwsA(isA<OverpassException>()),
      );
    });

    test('an empty elements list is simply no places', () {
      expect(OverpassClient.parse('{"elements": []}'), isEmpty);
    });

    test('search runs the query through the injected fetch', () async {
      String? sent;
      final client = OverpassClient(
        fetch: (query) async {
          sent = query;
          return body;
        },
      );

      final pois = await client.search(
        shillongBox,
        categoryKeys: const ['fuel', 'hospital'],
      );
      expect(sent, contains('amenity"="fuel'));
      expect(pois, isNotEmpty);
    });
  });

  group('OSRM', () {
    test('COORDINATES GO IN LON,LAT ORDER', () {
      // OSRM follows GeoJSON, not the lat-first convention everything else in
      // this app uses. Swapping them does not error; it silently routes
      // somewhere else entirely.
      final url = OsrmClient.buildUrl(
        const LatLng(25.5788, 91.8933),
        const LatLng(25.2702, 91.7323),
      );
      expect(url.path, contains('91.893300,25.578800;91.732300,25.270200'));
    });

    test('the request pins the geometry format', () {
      final url = OsrmClient.buildUrl(
        const LatLng(25.5, 91.5),
        const LatLng(25.2, 91.7),
      );
      // geometries=polyline means precision 5. Leaving it to the server to
      // decide is how a route ends up ten times too big.
      expect(url.query, contains('geometries=polyline'));
      expect(url.query, contains('overview=full'));
    });

    test('a route parses to a line and a distance', () {
      const body = '''
{"code": "Ok",
 "routes": [{"geometry": "_p~iF~ps|U_ulLnnqC",
             "distance": 54321.0, "duration": 4500.5}],
 "waypoints": []}
''';
      final result = OsrmClient.parse(body);

      expect(result.distanceKm, closeTo(54.3, 0.1));
      expect(result.duration.inMinutes, 75);
      expect(result.points.length, 2);
      expect(result.encodedPolyline, '_p~iF~ps|U_ulLnnqC');
    });

    test('NoRoute says what to do about it', () {
      expect(
        () => OsrmClient.parse('{"code": "NoRoute", "routes": []}'),
        throwsA(
          isA<OsrmException>().having(
            (e) => e.message,
            'message',
            contains('No road route'),
          ),
        ),
      );
    });

    test('another error code is reported rather than swallowed', () {
      expect(
        () => OsrmClient.parse('{"code": "InvalidQuery"}'),
        throwsA(
          isA<OsrmException>().having(
            (e) => e.message,
            'message',
            contains('InvalidQuery'),
          ),
        ),
      );
    });

    test('a route with no geometry is refused', () {
      expect(
        () => OsrmClient.parse(
          '{"code": "Ok", "routes": [{"distance": 1.0}]}',
        ),
        throwsA(isA<OsrmException>()),
      );
    });

    test('malformed JSON fails with a sentence', () {
      expect(
        () => OsrmClient.parse('<html>502</html>'),
        throwsA(isA<OsrmException>()),
      );
    });

    test('route runs the URL through the injected fetch', () async {
      Uri? sent;
      final client = OsrmClient(
        fetch: (url) async {
          sent = url;
          return '{"code":"Ok","routes":[{"geometry":"_p~iF~ps|U",'
              '"distance":1000.0,"duration":60.0}]}';
        },
      );

      final result = await client.route(
        const LatLng(25.5, 91.5),
        const LatLng(25.2, 91.7),
      );
      expect(sent!.host, 'router.project-osrm.org');
      expect(result.distanceKm, 1.0);
    });
  });
}
