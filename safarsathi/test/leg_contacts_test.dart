// test/leg_contacts_test.dart
//
// Your own numbers, placed on the road between stops.
//
// Asked for from the road: "check getting between the stops and put these
// phone numbers of hospitals / guest houses there". The sheet those came from
// carries a latitude and longitude for every row, and the corridor maths that
// places OpenStreetMap results along a leg is generic — so a diary contact
// with a position goes on the leg at its real kilometre, not into a note.

import 'package:drift/drift.dart' hide isNull, isNotNull;
import 'package:drift/native.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:safarsathi/core/database/app_database.dart';
import 'package:safarsathi/features/contacts/data/contacts_dao.dart';
import 'package:safarsathi/features/discovery/data/discovery.dart';
import 'package:safarsathi/features/discovery/data/geo.dart';
import 'package:safarsathi/features/discovery/data/polyline.dart';
import 'package:safarsathi/features/trips/data/trip_editor.dart';

const shillong = LatLng(25.5788, 91.8933);
const sohra = LatLng(25.2702, 91.7323);

void main() {
  late AppDatabase db;
  late TripEditor editor;
  late int tripId;
  late int shillongId;
  late int sohraId;
  late int legId;

  setUp(() async {
    db = AppDatabase(NativeDatabase.memory());
    editor = TripEditor(db);
    tripId = await editor.createTrip(name: 'Meghalaya');
    shillongId = await editor.addStop(
      tripId,
      StopDraft(name: 'Shillong', lat: shillong.lat, lon: shillong.lon),
    );
    sohraId = await editor.addStop(
      tripId,
      StopDraft(name: 'Sohrra', lat: sohra.lat, lon: sohra.lon),
    );
    legId = (await db.select(db.legs).getSingle()).id;
  });

  tearDown(() => db.close());

  /// A point [t] of the way along the straight line from Shillong to Sohra.
  LatLng along(double t) => LatLng(
    shillong.lat + (sohra.lat - shillong.lat) * t,
    shillong.lon + (sohra.lon - shillong.lon) * t,
  );

  Future<void> contact(
    String name, {
    LatLng? at,
    int? stopId,
    int? trip,
    String category = ContactCategory.other,
  }) => db.into(db.contacts).insert(
    ContactsCompanion.insert(
      tripId: Value(trip ?? tripId),
      stopId: Value(stopId),
      name: name,
      phoneRaw: '+91 90000 00000',
      category: Value(category),
      lat: Value(at?.lat),
      lon: Value(at?.lon),
    ),
  );

  Future<LegDiscovery> leg() => watchLegDiscovery(db, legId).first;

  group('on the way', () {
    test('a contact on the road appears, at its distance along it', () async {
      await contact('Halfway dhaba', at: along(0.5));
      final found = (await leg()).onTheWay.single;

      expect(found.contact.name, 'Halfway dhaba');
      // Shillong to Sohra is about 38 km in a straight line.
      expect(found.alongRouteKm, closeTo(19, 2));
      expect(found.offRouteKm, lessThan(0.1));
    });

    test('they come in the order the road reaches them', () async {
      await contact('Later', at: along(0.8));
      await contact('Sooner', at: along(0.2));
      await contact('Middle', at: along(0.5));

      expect(
        (await leg()).onTheWay.map((c) => c.contact.name),
        ['Sooner', 'Middle', 'Later'],
      );
    });

    test('a place well off the road is left out', () async {
      // Jowai, forty-odd kilometres east. Not on this leg.
      await contact('Jowai hospital', at: const LatLng(25.4681, 92.2609));
      expect((await leg()).onTheWay, isEmpty);
    });

    test('CONTACTS AT EITHER END ARE LEFT TO THOSE STOPS', () async {
      // Otherwise the top of every leg out of Shillong is twenty-seven city
      // numbers at kilometre nought, burying the one hospital halfway.
      await contact('Shillong hotel', at: shillong, stopId: shillongId);
      await contact('Sohra chemist', at: sohra, stopId: sohraId);
      await contact('Halfway dhaba', at: along(0.5));

      expect(
        (await leg()).onTheWay.map((c) => c.contact.name),
        ['Halfway dhaba'],
      );
    });

    test('a contact with no position cannot be placed and is not', () async {
      await contact('Somewhere');
      expect((await leg()).onTheWay, isEmpty);
    });

    test('another trip\'s contacts never appear', () async {
      final other = await editor.createTrip(name: 'Another trip');
      await contact('Not ours', at: along(0.5), trip: other);
      expect((await leg()).onTheWay, isEmpty);
    });

    test('THE REAL ROUTE WINS OVER THE STRAIGHT LINE', () async {
      // A road that swings well west before coming back. A place on that
      // swing is on the road; a place on the straight line is not, once the
      // road is known.
      const swing = LatLng(25.42, 91.60);
      await (db.update(db.legs)..where((l) => l.id.equals(legId))).write(
        LegsCompanion(
          routePolyline: Value(Polyline.encode(const [shillong, swing, sohra])),
        ),
      );
      await contact('On the swing', at: swing);
      await contact('On the straight line', at: along(0.5));

      expect(
        (await leg()).onTheWay.map((c) => c.contact.name),
        ['On the swing'],
      );
    });

    test('with no route and no coordinates, it says it cannot place', () async {
      await (db.update(db.stops)..where((s) => s.id.equals(sohraId))).write(
        const StopsCompanion(lat: Value(null), lon: Value(null)),
      );
      await contact('Halfway dhaba', at: along(0.5));

      final result = await leg();
      expect(result.canPlace, isFalse);
      expect(result.onTheWay, isEmpty);
    });
  });

  group('what could not be placed', () {
    test('contacts with no position are counted, so empty is explained',
        () async {
      await contact('Imported before coordinates');
      await contact('Also no position');
      final result = await leg();
      expect(result.onTheWay, isEmpty);
      expect(result.unplaced, 2);
    });

    test('emergency numbers are not counted — 112 has no location', () async {
      await db.into(db.contacts).insert(
        ContactsCompanion.insert(
          tripId: Value(tripId),
          name: 'ERSS',
          phoneRaw: '112',
          isEmergency: const Value(true),
        ),
      );
      expect((await leg()).unplaced, 0);
    });

    test('contacts at either end are not counted either', () async {
      await contact('Shillong hotel', stopId: shillongId);
      await contact('Sohra chemist', stopId: sohraId);
      expect((await leg()).unplaced, 0);
    });
  });

  group('the "no location" hint', () {
    test('anyPlaced is false while nothing in the diary has a position',
        () async {
      await contact('Imported before coordinates');
      expect((await leg()).anyPlaced, isFalse);
    });

    test('ONE PLACED CONTACT ANYWHERE ENDS THE HINT FOR EVERY LEG', () async {
      // Found by running the real sheet end to end: the tourist helplines
      // and the Bangladesh commission line have no place and never will.
      // Without this, "re-import with coordinates" would show on every leg
      // after the re-import had already been done.
      await contact('1363 tourist helpline');
      await contact('Somewhere far', at: const LatLng(26.18, 91.75));
      final result = await leg();
      expect(result.anyPlaced, isTrue);
      expect(result.unplaced, 1);
    });
  });

  group('nearest help', () {
    // Kongthong, from the sheet: "no restaurant or pharmacy listed — nearest
    // pharmacy and hospital are in Pynursla".
    const pynursla = LatLng(25.3104, 91.8987);

    test('a destination with no hospital or pharmacy gets the closest ones',
        () async {
      await contact('MK Pharmacy', at: pynursla,
          category: ContactCategory.pharmacy);
      await contact('Far hospital', at: const LatLng(25.57, 91.88),
          category: ContactCategory.hospital);

      final help = (await leg()).nearestHelp;
      expect(help.map((h) => h.contact.name).first, 'MK Pharmacy');
      expect(help.first.straightLineKm, lessThan(35));
      // Nearest first.
      for (var i = 1; i < help.length; i++) {
        expect(help[i].straightLineKm,
            greaterThanOrEqualTo(help[i - 1].straightLineKm));
      }
    });

    test('A DESTINATION WITH ITS OWN HOSPITAL GETS NONE', () async {
      // Otherwise Shillong's suburbs would be listed under Shillong, which
      // already carries twenty-seven numbers.
      await contact('Sohra CHC', stopId: sohraId,
          category: ContactCategory.hospital);
      await contact('MK Pharmacy', at: pynursla,
          category: ContactCategory.pharmacy);
      expect((await leg()).nearestHelp, isEmpty);
    });

    test('only hospitals and pharmacies count as help', () async {
      await contact('A homestay', at: pynursla,
          category: ContactCategory.accommodation);
      expect((await leg()).nearestHelp, isEmpty);
    });

    test('nothing beyond the radius', () async {
      // Guwahati, ~100 km from Sohra.
      await contact('Guwahati hospital', at: const LatLng(26.18, 91.75),
          category: ContactCategory.hospital);
      expect((await leg()).nearestHelp, isEmpty);
    });

    test('at most three', () async {
      for (var i = 0; i < 5; i++) {
        await contact('Pharmacy $i',
            at: LatLng(pynursla.lat + i * 0.001, pynursla.lon),
            category: ContactCategory.pharmacy);
      }
      expect((await leg()).nearestHelp, hasLength(3));
    });
  });

  group('at the destination', () {
    test('contacts at the arrival stop, help first', () async {
      await contact('Zed Cafe', stopId: sohraId,
          category: ContactCategory.restaurant);
      await contact('Sohra CHC', stopId: sohraId,
          category: ContactCategory.hospital);
      await contact('Police', stopId: sohraId,
          category: ContactCategory.emergency);
      await contact('Erica Pharmacy', stopId: sohraId,
          category: ContactCategory.pharmacy);
      await contact('Homestay', stopId: sohraId,
          category: ContactCategory.accommodation);

      expect(
        (await leg()).atDestination.map((c) => c.name),
        ['Police', 'Sohra CHC', 'Erica Pharmacy', 'Homestay', 'Zed Cafe'],
      );
    });

    test('needs no coordinates: the stop is enough', () async {
      await contact('Sohra CHC', stopId: sohraId,
          category: ContactCategory.hospital);
      expect((await leg()).atDestination.single.name, 'Sohra CHC');
    });

    test('the departure stop is not the destination', () async {
      await contact('Shillong hotel', stopId: shillongId);
      expect((await leg()).atDestination, isEmpty);
    });
  });

  test('THE LEG UPDATES WHEN A CONTACT IS ADDED', () async {
    // The stream reads four tables and a Drift stream only fires for the
    // ones its query names. Without contacts in readsFrom, importing a sheet
    // would leave an open leg screen showing the old, empty list.
    final updates = watchLegDiscovery(db, legId);
    final seen = <int>[];
    final sub = updates.listen((l) => seen.add(l.onTheWay.length));

    await pumpEventQueue();
    await contact('Halfway dhaba', at: along(0.5));
    await pumpEventQueue();
    await sub.cancel();

    expect(seen.first, 0);
    expect(seen.last, 1);
  });
}
