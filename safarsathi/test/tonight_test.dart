// test/tonight_test.dart
//
// Which bed is tonight's. The edge days are the whole difficulty: the day
// you move is the day the answer changes, and the day it matters most.

import 'package:drift/drift.dart' hide isNull, isNotNull;
import 'package:drift/native.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:safarsathi/core/database/app_database.dart';
import 'package:safarsathi/features/contacts/data/contacts_dao.dart';
import 'package:safarsathi/features/trips/data/stay.dart';
import 'package:safarsathi/features/trips/data/tonight.dart';
import 'package:safarsathi/features/trips/data/trip_editor.dart';

void main() {
  late AppDatabase db;
  late int tripId;
  late int shillong, sohra, mawlynnong, dawki;

  // The real shape: Shillong 1–3, Sohra 3–4, Mawlynnong 4–5, then a day
  // passing through Dawki on the way home, sleeping nowhere.
  setUp(() async {
    db = AppDatabase(NativeDatabase.memory());
    final editor = TripEditor(db);
    tripId = await editor.createTrip(name: 'Meghalaya');
    shillong = await editor.addStop(tripId, StopDraft(
      name: 'Shillong', lat: 25.5764, lon: 91.8838,
      arrivalDate: DateTime(2026, 10, 1), departureDate: DateTime(2026, 10, 3),
      nights: 2,
    ));
    sohra = await editor.addStop(tripId, StopDraft(
      name: 'Sohrra', lat: 25.2718, lon: 91.7327,
      arrivalDate: DateTime(2026, 10, 3), departureDate: DateTime(2026, 10, 4),
      nights: 1,
    ));
    mawlynnong = await editor.addStop(tripId, StopDraft(
      name: 'Mawlynnong', lat: 25.2027, lon: 91.9146,
      arrivalDate: DateTime(2026, 10, 4), departureDate: DateTime(2026, 10, 5),
      nights: 1,
    ));
    dawki = await editor.addStop(tripId, StopDraft(
      name: 'Dawki', lat: 25.1849, lon: 92.0223,
      arrivalDate: DateTime(2026, 10, 5), departureDate: DateTime(2026, 10, 5),
    ));
  });
  tearDown(() => db.close());

  Future<Tonight?> on(int month, int day, {int hour = 12}) =>
      watchTonight(db, tripId, now: DateTime(2026, month, day, hour)).first;

  test('before the trip it is the first night, dated', () async {
    final t = await on(9, 23);
    expect(t?.kind, TonightKind.firstNight);
    expect(t?.stop.name, 'Shillong');
    expect(t?.night, DateTime(2026, 10, 1));
  });

  test('a middle night of a two-night stop', () async {
    final t = await on(10, 2);
    expect(t?.kind, TonightKind.tonight);
    expect(t?.stop.name, 'Shillong');
  });

  test('THE DAY YOU MOVE, TONIGHT IS THE NEXT STOP — not the one you wake in',
      () async {
    // 3 October: breakfast in Shillong, bed in Sohra. The "current stop"
    // logic says Shillong all day; the night is Sohra's.
    expect((await on(10, 3, hour: 7))?.stop.name, 'Sohrra');
    expect((await on(10, 4, hour: 7))?.stop.name, 'Mawlynnong');
  });

  test('a stop passed through is never tonight', () async {
    // 5 October: leave Mawlynnong, through Dawki, home. No bed on the plan.
    expect(await on(10, 5), isNull);
  });

  test('after the trip there is no card at all', () async {
    expect(await on(10, 9), isNull);
  });

  test('nights with no departure date are counted from arrival', () async {
    await (db.update(db.stops)..where((s) => s.id.equals(sohra))).write(
      const StopsCompanion(departureDate: Value(null), nights: Value(1)),
    );
    expect((await on(10, 3))?.stop.name, 'Sohrra');
  });

  group('the stays', () {
    Future<int> stay(String name, int stopId, {bool confirmed = false,
        String category = ContactCategory.accommodation}) =>
        db.into(db.contacts).insert(ContactsCompanion.insert(
          tripId: Value(tripId), stopId: Value(stopId), name: name,
          phoneRaw: '+91 90000 00000', category: Value(category),
          callConfirmed: Value(confirmed),
        ));

    test('only accommodation at that stop, confirmed first', () async {
      await stay('Zed Homestay', mawlynnong);
      await stay('Iartong Guest House', mawlynnong, confirmed: true);
      await stay('A restaurant', mawlynnong,
          category: ContactCategory.restaurant);
      await stay('Somewhere else', shillong);

      final t = await on(10, 4);
      expect(t!.options.map((c) => c.name),
          ['Iartong Guest House', 'Zed Homestay']);
      // Saved is not chosen: nothing is tonight's bed until you say so.
      expect(t.stay, isNull);
    });

    test('THE CHOSEN STAY IS TONIGHT\'S, WHATEVER SORTS FIRST', () async {
      await stay('Bramhome Guest House', mawlynnong, confirmed: true);
      final mine = await stay('Zed Homestay', mawlynnong);
      await setStay(db, mawlynnong, mine);
      final t = await on(10, 4);
      expect(t!.stay?.name, 'Zed Homestay');
    });

    test('a chosen stay since deleted reads as not decided', () async {
      final mine = await stay('Zed Homestay', mawlynnong);
      await setStay(db, mawlynnong, mine);
      await db.contactsDao.deleteContact(mine);
      expect((await on(10, 4))!.stay, isNull);
    });

    test('THE CARD UPDATES WHEN A STAY IS SAVED', () async {
      final seen = <int>[];
      final sub = watchTonight(db, tripId, now: DateTime(2026, 10, 4))
          .listen((t) => seen.add(t?.options.length ?? -1));
      await pumpEventQueue();
      await stay('Iartong Guest House', mawlynnong);
      await pumpEventQueue();
      await sub.cancel();
      expect(seen.first, 0);
      expect(seen.last, 1);
    });
  });

  test('sunset is the stop\'s, on that night', () async {
    final t = await on(10, 4);
    expect(t?.sun?.sunset, isNotNull);
    expect(dawki, isNonZero); // declared, so the fixture reads as the trip
  });
}
