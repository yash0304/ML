// test/discovery_test.dart — issues #27 and #28.

import 'package:drift/drift.dart' hide isNull, isNotNull;
import 'package:drift/native.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:safarsathi/core/database/app_database.dart';
import 'package:safarsathi/features/contacts/data/contacts_dao.dart';
import 'package:safarsathi/features/discovery/data/discovery.dart';
import 'package:safarsathi/features/trips/data/trip_editor.dart';

void main() {
  late AppDatabase db;
  late TripEditor editor;
  late int tripId;
  late int legId;

  setUp(() async {
    db = AppDatabase(NativeDatabase.memory());
    editor = TripEditor(db);
    tripId = await editor.createTrip(name: 'Meghalaya');
    await editor.addStop(
      tripId,
      const StopDraft(name: 'Shillong', lat: 25.5788, lon: 91.8933),
    );
    await editor.addStop(
      tripId,
      const StopDraft(name: 'Cherrapunji', lat: 25.2702, lon: 91.7323),
    );
    legId = (await db.select(db.legs).getSingle()).id;
  });

  tearDown(() => db.close());

  Future<int> place(
    String name,
    String category,
    double along, {
    double off = 0.1,
    String? phone,
  }) async {
    final id = await db
        .into(db.pois)
        .insert(
          PoisCompanion.insert(
            tripId: tripId,
            legId: Value(legId),
            name: name,
            category: category,
            lat: 25.4,
            lon: 91.8,
            osmId: const Value('node/1'),
            distanceAlongRouteKm: Value(along),
            distanceOffRouteKm: Value(off),
          ),
        );
    if (phone != null) {
      await db
          .into(db.poiContacts)
          .insert(
            PoiContactsCompanion.insert(
              poiId: id,
              phoneRaw: phone,
              phoneE164: Value(phone.replaceAll(' ', '')),
              tier: const Value('communityOsm'),
              sourceTag: const Value('phone'),
            ),
          );
    }
    return id;
  }

  group('the list', () {
    test('ORDERED BY DISTANCE ALONG THE ROUTE', () async {
      // Not by distance from you. "Coming up in 12 km" is the useful sentence
      // while moving; a place 200 m away across a gorge is an hour of driving.
      await place('Far dhaba', ContactCategory.restaurant, 40, off: 0.1);
      await place('Near pump', ContactCategory.fuel, 5, off: 2.5);
      await place('Middle clinic', ContactCategory.hospital, 20, off: 0.3);

      final legRow = await watchLegDiscovery(db, legId).first;
      expect(legRow.places.map((p) => p.name), [
        'Near pump',
        'Middle clinic',
        'Far dhaba',
      ]);
    });

    test('a place carries both distances', () async {
      await place('Dhaba', ContactCategory.restaurant, 12.4, off: 1.7);
      final found = (await watchLegDiscovery(db, legId).first).places.single;

      expect(found.alongRouteKm, closeTo(12.4, 0.01));
      expect(found.offRouteKm, closeTo(1.7, 0.01));
    });

    test('the leg names both its ends', () async {
      final legRow = await watchLegDiscovery(db, legId).first;
      expect(legRow.fromName, 'Shillong');
      expect(legRow.toName, 'Cherrapunji');
    });

    test('only the categories actually present are offered', () async {
      // A chip nobody can match is a filter that only disappoints.
      await place('Pump', ContactCategory.fuel, 5);
      await place('Dhaba', ContactCategory.restaurant, 10);

      final legRow = await watchLegDiscovery(db, legId).first;
      expect(legRow.categoriesPresent, [
        ContactCategory.fuel,
        ContactCategory.restaurant,
      ]);
      expect(
        legRow.categoriesPresent,
        isNot(contains(ContactCategory.pharmacy)),
      );
    });

    test('phones come back with their place', () async {
      await place(
        'Clinic',
        ContactCategory.hospital,
        20,
        phone: '+91 364 222 2222',
      );
      final found = (await watchLegDiscovery(db, legId).first).places.single;

      expect(found.hasPhone, isTrue);
      expect(found.phones.single.phoneRaw, '+91 364 222 2222');
      expect(found.phones.single.tier, 'communityOsm');
    });

    test('a place with no phone reports none', () async {
      await place('Viewpoint', ContactCategory.other, 8);
      expect(
        (await watchLegDiscovery(db, legId).first).places.single.hasPhone,
        isFalse,
      );
    });

    test('an unsynced leg is distinguishable from an empty one', () async {
      // Two different reasons for an empty list, and confusing them sends the
      // user to the wrong screen.
      var legRow = await watchLegDiscovery(db, legId).first;
      expect(legRow.places, isEmpty);
      expect(legRow.isSynced, isFalse);

      await (db.update(db.legs)..where((l) => l.id.equals(legId))).write(
        LegsCompanion(lastSyncedAt: Value(DateTime.now())),
      );
      legRow = await watchLegDiscovery(db, legId).first;
      expect(legRow.places, isEmpty);
      expect(legRow.isSynced, isTrue);
    });

    test('another leg\'s places do not leak in', () async {
      await editor.addStop(
        tripId,
        const StopDraft(name: 'Dawki', lat: 25.1932, lon: 92.0207),
      );
      final other = (await db.select(db.legs).get()).last;

      await place('On this leg', ContactCategory.fuel, 5);
      await db.into(db.pois).insert(
        PoisCompanion.insert(
          tripId: tripId,
          legId: Value(other.id),
          name: 'On the other leg',
          category: ContactCategory.fuel,
          lat: 25.2,
          lon: 92.0,
        ),
      );

      final legRow = await watchLegDiscovery(db, legId).first;
      expect(legRow.places.map((p) => p.name), ['On this leg']);
    });
  });

  group('saving a place', () {
    test('IT LANDS AS communityOsm, NOT userEntered', () async {
      // The backlog said userEntered. Both carry the amber dot, so the trust
      // outcome is identical — but the entry screen reads provenance out
      // loud, and "Typed by you" about a number a stranger put in a public
      // wiki is the one thing this app must not say.
      await place(
        'Sohra PHC',
        ContactCategory.hospital,
        20,
        phone: '+91 364 222 2222',
      );
      final found = (await watchLegDiscovery(db, legId).first).places.single;

      await savePlaceAsContact(
        db,
        tripId: tripId,
        place: found,
        phone: found.phones.single,
      );

      final saved = await db.select(db.contacts).getSingle();
      expect(saved.tier, ContactTier.communityOsm.name);
      expect(saved.tier, isNot(ContactTier.userEntered.name));
    });

    test('it is never confirmed', () async {
      await place('PHC', ContactCategory.hospital, 20, phone: '+91 1');
      final found = (await watchLegDiscovery(db, legId).first).places.single;
      await savePlaceAsContact(
        db,
        tripId: tripId,
        place: found,
        phone: found.phones.single,
      );

      final saved = await db.select(db.contacts).getSingle();
      expect(saved.callConfirmed, isFalse);
      expect(saved.confirmedAt, isNull);
    });

    test('the note records where it came from', () async {
      await place('PHC', ContactCategory.hospital, 20, phone: '+91 1');
      final found = (await watchLegDiscovery(db, legId).first).places.single;
      await savePlaceAsContact(
        db,
        tripId: tripId,
        place: found,
        phone: found.phones.single,
      );

      final saved = await db.select(db.contacts).getSingle();
      expect(saved.note, contains('map data'));
      expect(saved.note, contains('node/1'));
    });

    test('the category and both number forms carry over', () async {
      await place(
        'Pump',
        ContactCategory.fuel,
        5,
        phone: '+91 364 111 1111',
      );
      final found = (await watchLegDiscovery(db, legId).first).places.single;
      await savePlaceAsContact(
        db,
        tripId: tripId,
        place: found,
        phone: found.phones.single,
      );

      final saved = await db.select(db.contacts).getSingle();
      expect(saved.category, ContactCategory.fuel);
      expect(saved.phoneRaw, '+91 364 111 1111');
      expect(saved.phoneE164, '+913641111111');
    });

    test('a saved place shows up in the diary', () async {
      await place('PHC', ContactCategory.hospital, 20, phone: '+91 90000 1');
      final found = (await watchLegDiscovery(db, legId).first).places.single;
      await savePlaceAsContact(
        db,
        tripId: tripId,
        place: found,
        phone: found.phones.single,
      );

      final diary = await db.contactsDao
          .watchContacts(ContactFilter(tripId: tripId))
          .first;
      expect(diary.map((c) => c.name), contains('PHC'));
    });

    test('CONFIRMING IT STILL PROMOTES IT NORMALLY', () async {
      // The path out of communityOsm is unchanged: call it and say so.
      await place('PHC', ContactCategory.hospital, 20, phone: '+91 1');
      final found = (await watchLegDiscovery(db, legId).first).places.single;
      final id = await savePlaceAsContact(
        db,
        tripId: tripId,
        place: found,
        phone: found.phones.single,
      );

      await db.contactsDao.markConfirmed(id);
      final saved = await db.select(db.contacts).getSingle();
      expect(saved.tier, ContactTier.userVerified.name);
      expect(saved.callConfirmed, isTrue);
    });
  });

  test('the maps link points at the coordinates', () {
    const found = CorridorPlace(
      id: 1,
      name: 'Dhaba',
      category: 'restaurant',
      lat: 25.4,
      lon: 91.8,
      alongRouteKm: 12,
      offRouteKm: 0.2,
    );
    final url = googleMapsUrl(found);

    expect(url, contains('25.400000,91.800000'));
    expect(url, startsWith('https://www.google.com/maps/'));
  });
}
