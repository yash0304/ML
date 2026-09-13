// test/geo_test.dart and the corridor — issues #22 and #23.
//
// All pure maths, so every rule is arguable here rather than only observable
// on a map that does not exist yet.

import 'package:flutter_test/flutter_test.dart';

import 'package:safarsathi/features/discovery/data/corridor.dart';
import 'package:safarsathi/features/discovery/data/geo.dart';
import 'package:safarsathi/features/discovery/data/polyline.dart';

// Real places, so a wrong answer is recognisable rather than abstract.
const shillong = LatLng(25.5788, 91.8933);
const cherrapunji = LatLng(25.2702, 91.7323);
const dawki = LatLng(25.1932, 92.0207);

void main() {
  group('haversine', () {
    test('Shillong to Cherrapunji is about 37 km as the crow flies', () {
      final km = haversineMetres(shillong, cherrapunji) / 1000;
      expect(km, closeTo(37, 2));
    });

    test('a point is zero from itself', () {
      expect(haversineMetres(shillong, shillong), 0);
    });

    test('one degree of latitude is about 111 km anywhere', () {
      for (final lat in [0.0, 25.0, 50.0, 70.0]) {
        final km =
            haversineMetres(LatLng(lat, 0), LatLng(lat + 1, 0)) / 1000;
        expect(km, closeTo(111.2, 0.5), reason: 'at $lat');
      }
    });

    test('a degree of longitude shrinks towards the poles', () {
      final atEquator =
          haversineMetres(const LatLng(0, 0), const LatLng(0, 1)) / 1000;
      final atShillong =
          haversineMetres(const LatLng(25.6, 0), const LatLng(25.6, 1)) / 1000;

      expect(atEquator, closeTo(111.2, 0.5));
      expect(atShillong, closeTo(111.2 * 0.902, 1));
    });
  });

  group('distance to a segment', () {
    // A road running dead east for a degree, at Shillong's latitude.
    const a = LatLng(25.6, 91.0);
    const b = LatLng(25.6, 92.0);

    test('MEASURES TO THE SEGMENT, NOT THE NEARER VERTEX', () {
      // A point sitting right on the line, halfway along. Measured to the
      // nearer vertex it would read as 50 km off the route.
      const onTheLine = LatLng(25.6, 91.5);

      expect(distanceToSegmentMetres(onTheLine, a, b), lessThan(20));
      expect(haversineMetres(onTheLine, a) / 1000, closeTo(50, 2));
    });

    test('a point beside the middle reads its perpendicular distance', () {
      // Roughly 1.1 km north of the line, halfway along.
      const beside = LatLng(25.61, 91.5);
      expect(distanceToSegmentMetres(beside, a, b) / 1000, closeTo(1.11, 0.1));
    });

    test('past the end, it measures to the end', () {
      const past = LatLng(25.6, 92.5);
      expect(
        distanceToSegmentMetres(past, a, b),
        closeTo(haversineMetres(past, b), 50),
      );
    });

    test('a zero-length segment is just a point', () {
      // OSRM does emit repeated coordinates.
      expect(
        distanceToSegmentMetres(shillong, a, a),
        closeTo(haversineMetres(shillong, a), 1),
      );
    });

    test('the fraction along runs 0 to 1 and clamps outside', () {
      expect(fractionAlongSegment(a, a, b), closeTo(0, 0.001));
      expect(fractionAlongSegment(b, a, b), closeTo(1, 0.001));
      expect(
        fractionAlongSegment(const LatLng(25.6, 91.5), a, b),
        closeTo(0.5, 0.01),
      );
      expect(fractionAlongSegment(const LatLng(25.6, 90.0), a, b), 0);
      expect(fractionAlongSegment(const LatLng(25.6, 93.0), a, b), 1);
    });
  });

  group('bounding boxes', () {
    test('bounds contain every point', () {
      final box = boundsOf([shillong, cherrapunji, dawki]);
      for (final p in [shillong, cherrapunji, dawki]) {
        expect(box.contains(p), isTrue);
      }
      expect(box.north, shillong.lat);
      expect(box.south, dawki.lat);
    });

    test('an empty list has no bounds', () {
      expect(() => boundsOf(const []), throwsArgumentError);
    });

    test('PADDING WIDENS MORE IN LONGITUDE THAN IN LATITUDE', () {
      // The bug this guards: three kilometres is 3/111 degrees of latitude but
      // 3/(111·cos φ) degrees of longitude. Ignoring it makes the corridor
      // narrower than advertised east–west, and places simply go missing.
      final box = padBox(
        const BoundingBox(south: 25.6, west: 91.0, north: 25.6, east: 91.0),
        3,
      );

      final latPad = box.north - 25.6;
      final lonPad = box.east - 91.0;
      expect(lonPad, greaterThan(latPad));
      expect(lonPad / latPad, closeTo(1 / 0.902, 0.02));
    });

    test('the padding is really the distance asked for', () {
      final box = padBox(
        const BoundingBox(south: 25.6, west: 91.0, north: 25.6, east: 91.0),
        3,
      );
      final northKm =
          haversineMetres(const LatLng(25.6, 91.0), LatLng(box.north, 91.0)) /
          1000;
      final eastKm =
          haversineMetres(const LatLng(25.6, 91.0), LatLng(25.6, box.east)) /
          1000;

      expect(northKm, closeTo(3, 0.1));
      expect(eastKm, closeTo(3, 0.1));
    });

    test('padding never runs off the ends of the world', () {
      final box = padBox(
        const BoundingBox(south: 89.9, west: 179.9, north: 89.95, east: 179.95),
        50,
      );
      expect(box.north, lessThanOrEqualTo(90));
      expect(box.east, lessThanOrEqualTo(180));
    });

    test('the Overpass string is south, west, north, east', () {
      const box = BoundingBox(south: 1, west: 2, north: 3, east: 4);
      expect(box.overpassString, '1.000000,2.000000,3.000000,4.000000');
    });

    test('area is roughly right for a known box', () {
      // One degree by one degree at Shillong: about 111 by 100 km.
      const box = BoundingBox(south: 25.1, west: 91.0, north: 26.1, east: 92.0);
      expect(box.approxAreaSqKm, closeTo(11100, 800));
    });
  });

  group('polyline codec', () {
    test('the canonical Google example decodes', () {
      // From Google's own documentation, precision 5.
      final points = Polyline.decode('_p~iF~ps|U_ulLnnqC_mqNvxq`@');
      expect(points.length, 3);
      expect(points[0].lat, closeTo(38.5, 0.0001));
      expect(points[0].lon, closeTo(-120.2, 0.0001));
      expect(points[1].lat, closeTo(40.7, 0.0001));
      expect(points[1].lon, closeTo(-120.95, 0.0001));
      expect(points[2].lat, closeTo(43.252, 0.0001));
      expect(points[2].lon, closeTo(-126.453, 0.0001));
    });

    test('round-trips at precision 5', () {
      const route = [shillong, cherrapunji, dawki];
      final decoded = Polyline.decode(Polyline.encode(route));

      for (var i = 0; i < route.length; i++) {
        expect(decoded[i].lat, closeTo(route[i].lat, 0.00001));
        expect(decoded[i].lon, closeTo(route[i].lon, 0.00001));
      }
    });

    test('round-trips at precision 6', () {
      const route = [shillong, cherrapunji];
      final decoded = Polyline.decode(
        Polyline.encode(route, precision: 6),
        precision: 6,
      );
      for (var i = 0; i < route.length; i++) {
        expect(decoded[i].lat, closeTo(route[i].lat, 0.000001));
      }
    });

    test('DECODING AT THE WRONG PRECISION IS OFF BY TEN', () {
      // The failure that puts Shillong in the Bay of Bengal, and which looks
      // like a broken map rather than a wrong number.
      final encoded = Polyline.encode([shillong], precision: 6);
      final wrong = Polyline.decode(encoded);
      expect(wrong.first.lat, closeTo(shillong.lat * 10, 0.01));
    });

    test('negative coordinates survive', () {
      const southern = [LatLng(-33.86, 151.2), LatLng(-37.81, 144.96)];
      final decoded = Polyline.decode(Polyline.encode(southern));
      expect(decoded[0].lat, closeTo(-33.86, 0.00001));
      expect(decoded[1].lon, closeTo(144.96, 0.00001));
    });

    test('an empty route encodes and decodes to nothing', () {
      expect(Polyline.encode(const []), '');
      expect(Polyline.decode(''), isEmpty);
    });

    test('a truncated string yields what it could read, not an exception', () {
      final full = Polyline.encode([shillong, cherrapunji, dawki]);
      final points = Polyline.decode(full.substring(0, full.length - 2));
      expect(points.length, lessThan(3));
    });
  });

  group('corridor', () {
    // A straight run east at Shillong's latitude, about 100 km long.
    final straight = Corridor(const [
      LatLng(25.6, 91.0),
      LatLng(25.6, 92.0),
    ]);

    test('length matches the great-circle distance', () {
      expect(straight.lengthKm, closeTo(100.3, 1));
    });

    test('a point on the line is zero off route, and halfway along', () {
      final position = straight.locate(const LatLng(25.6, 91.5))!;
      expect(position.offRouteKm, lessThan(0.05));
      expect(position.alongRouteKm, closeTo(50, 1));
    });

    test('distance along route is cumulative and monotonic', () {
      final a = straight.locate(const LatLng(25.6, 91.2))!;
      final b = straight.locate(const LatLng(25.6, 91.6))!;
      expect(a.alongRouteKm, lessThan(b.alongRouteKm));
    });

    test('the query box is the route grown by the buffer', () {
      final box = straight.queryBox;
      final northKm =
          haversineMetres(const LatLng(25.6, 91.5), LatLng(box.north, 91.5)) /
          1000;
      expect(northKm, closeTo(3, 0.1));
    });

    test('a place just outside the band is excluded', () {
      // 3 km buffer. Roughly 4 km north of the line.
      expect(straight.covers(const LatLng(25.636, 91.5)), isFalse);
      expect(straight.covers(const LatLng(25.62, 91.5)), isTrue);
    });

    test('placing sorts by distance along the route and drops the rest', () {
      final places = [
        (name: 'far end', at: const LatLng(25.6, 91.9)),
        (name: 'near start', at: const LatLng(25.6, 91.1)),
        (name: 'off route', at: const LatLng(25.8, 91.5)),
        (name: 'middle', at: const LatLng(25.61, 91.5)),
      ];

      final placed = straight.place(places, (p) => p.at);
      expect(placed.map((p) => p.item.name), [
        'near start',
        'middle',
        'far end',
      ]);
    });

    test('a route of one point cannot locate anything', () {
      final stub = Corridor(const [LatLng(25.6, 91.0)]);
      expect(stub.locate(shillong), isNull);
      expect(stub.lengthKm, 0);
    });

    test('a corridor with no width is refused', () {
      expect(() => Corridor(const [shillong, dawki], bufferKm: 0), throwsA(isA<AssertionError>()));
    });

    test('a real three-leg route measures plausibly', () {
      final route = Corridor(const [shillong, cherrapunji, dawki]);
      expect(route.lengthKm, closeTo(37 + 31, 5));

      // Cherrapunji is a vertex, so it is on the route and 37 km along it.
      final position = route.locate(cherrapunji)!;
      expect(position.offRouteKm, lessThan(0.01));
      expect(position.alongRouteKm, closeTo(37, 2));
    });
  });
}
