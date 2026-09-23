// test/place_location_test.dart — where a diary entry is, and getting there.
//
// "From this screen I should have a map so that I can have a direction from
// my current location to this place." The position has to come from
// somewhere a person can actually get it — Google Maps — so the reading of
// what Google Maps hands over is most of what is tested here.

import 'dart:io';
import 'dart:typed_data';

import 'package:drift/drift.dart' show Value;
import 'package:drift/native.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:safarsathi/core/database/app_database.dart';
import 'package:safarsathi/core/theme/app_tokens.dart';
import 'package:safarsathi/features/contacts/data/contacts_dao.dart';
import 'package:safarsathi/features/contacts/data/place_location.dart';
import 'package:safarsathi/features/contacts/presentation/entry_screen.dart';
import 'package:safarsathi/features/discovery/data/discovery.dart';
import 'package:safarsathi/features/discovery/data/geo.dart';
import 'package:safarsathi/features/map/data/here.dart';
import 'package:safarsathi/features/map/data/tile_math.dart';
import 'package:safarsathi/features/map/data/tile_provider.dart';
import 'package:safarsathi/features/map/data/tile_store.dart';
import 'package:safarsathi/features/map/presentation/place_map.dart';

import 'here_test.dart' show FakeLocation;

const civil = LatLng(25.567739, 91.881081);

void expectAt(ParsedLocation p, double lat, double lon) {
  expect(p.problem, isNull);
  expect(p.at!.lat, closeTo(lat, 1e-6));
  expect(p.at!.lon, closeTo(lon, 1e-6));
}

Contact contactOf({double? lat, double? lon, String name = 'Civil Hospital'}) =>
    Contact(
      id: 7,
      name: name,
      phoneRaw: '+913642224100',
      phoneE164: '+913642224100',
      category: ContactCategory.hospital,
      tier: ContactTier.userEntered.name,
      callConfirmed: false,
      isPinned: false,
      isEmergency: false,
      hasWhatsapp: false,
      callCount: 0,
      createdAt: DateTime(2026, 9, 12),
      lat: lat,
      lon: lon,
    );

void main() {
  group('reading what Google Maps gives you', () {
    test('the numbers from press-and-hold', () {
      expectAt(parseLocation('25.567739, 91.881081'), 25.567739, 91.881081);
      expectAt(parseLocation(' 25.567739,91.881081 '), 25.567739, 91.881081);
      expectAt(parseLocation('25.567739 91.881081'), 25.567739, 91.881081);
    });

    test('degrees, minutes and seconds from a dropped pin', () {
      final p = parseLocation('25°34\'03.9"N 91°52\'51.9"E');
      expect(p.at!.lat, closeTo(25.567750, 1e-5));
      expect(p.at!.lon, closeTo(91.881083, 1e-5));
    });

    test('decimal degrees with a hemisphere', () {
      expectAt(parseLocation('25.5677° N, 91.8810° E'), 25.5677, 91.8810);
    });

    test('A LONG LINK READS THE PIN, NOT WHERE THE CAMERA WAS', () {
      // "@" is the view when the link was copied; "!3d!4d" is the place.
      expectAt(
        parseLocation(
          'https://www.google.com/maps/place/Civil+Hospital/'
          '@25.5600,91.8700,15z/data=!4m6!3m5!1s0x0:0x0!8m2!'
          '3d25.567739!4d91.881081',
        ),
        25.567739,
        91.881081,
      );
      expectAt(
        parseLocation('https://www.google.com/maps/@25.5677,91.881,17z'),
        25.5677,
        91.881,
      );
      expectAt(
        parseLocation('https://maps.google.com/?q=25.567739%2C91.881081'),
        25.567739,
        91.881081,
      );
      expectAt(parseLocation('geo:25.567739,91.881081'), 25.567739, 91.881081);
    });

    test('A SHORT LINK IS REFUSED WITH WHAT TO DO INSTEAD', () {
      // maps.app.goo.gl holds no coordinates; only Google's server knows.
      final p = parseLocation('https://maps.app.goo.gl/AbCdEf123');
      expect(p.at, isNull);
      expect(p.problem, contains('press and hold'));
    });

    test('nonsense, out of range and 0,0 are refused', () {
      expect(parseLocation('Civil Hospital Shillong').problem, isNotNull);
      expect(parseLocation('95.0, 91.0').problem, contains('not a place'));
      expect(parseLocation('0, 0').problem, contains('blank'));
    });

    test('empty is no location, not a problem', () {
      expect(parseLocation('  ').isEmpty, isTrue);
    });

    test('a saved position is written back so it reads again exactly', () {
      final text = formatLocation(civil);
      expect(text, '25.567739, 91.881081');
      expectAt(parseLocation(text), civil.lat, civil.lon);
    });
  });

  group('the handoff to Google Maps', () {
    test('directions start from wherever the phone is', () {
      final url = directionsUrl(civil);
      expect(url, startsWith('https://www.google.com/maps/dir/?api=1'));
      expect(url, contains('destination=25.567739,91.881081'));
      expect(url, isNot(contains('origin=')));
    });

    test('a search adds the stop, once', () {
      expect(
        mapsSearchUrl('Reid Chest Hospital', 'Shillong'),
        contains('query=Reid+Chest+Hospital%2C+Shillong'),
      );
      expect(
        mapsSearchUrl('Police Control Room, Shillong', 'Shillong'),
        endsWith('query=Police+Control+Room%2C+Shillong'),
      );
    });

    test('distances read as a person says them', () {
      expect(describeDistance(348), '350 m');
      expect(describeDistance(3420), '3.4 km');
      expect(describeDistance(41200), '41 km');
    });
  });

  group('the entry page', () {
    late List<String> opened;

    Future<void> pump(WidgetTester tester, Contact contact) async {
      opened = [];
      tester.view.physicalSize = const Size(420, 1600);
      tester.view.devicePixelRatio = 1.0;
      addTearDown(tester.view.reset);
      await tester.pumpWidget(
        MaterialApp(
          theme: AppTokens.light,
          home: EntryScreen(
            contact: contact,
            stopName: 'Shillong',
            onOpenMaps: (url) async => opened.add(url),
            placeMap: (_, place) => Text(
              'MAP ${place.lat},${place.lon}',
              key: const Key('stub-map'),
            ),
          ),
        ),
      );
      await tester.pump();
    }

    testWidgets('a placed entry shows its map and gives directions', (
      tester,
    ) async {
      await pump(tester, contactOf(lat: civil.lat, lon: civil.lon));
      expect(find.text('MAP 25.567739,91.881081'), findsOneWidget);
      expect(find.textContaining('starting from where you are'), findsOneWidget);

      await tester.tap(find.byKey(const Key('entry-directions')));
      await tester.pump();
      expect(opened.single, directionsUrl(civil));
    });

    testWidgets('an unplaced entry says how to place it, and offers a '
        'search that says it is one', (tester) async {
      await pump(tester, contactOf(name: 'Reid Chest Hospital'));
      expect(find.byKey(const Key('stub-map')), findsNothing);
      expect(find.textContaining('No location saved'), findsOneWidget);
      expect(find.text('Search in Google Maps'), findsOneWidget);
      expect(find.textContaining('A search by name'), findsOneWidget);

      await tester.tap(find.byKey(const Key('entry-directions')));
      await tester.pump();
      expect(opened.single, mapsSearchUrl('Reid Chest Hospital', 'Shillong'));
    });
  });

  group('the place map', () {
    late AppDatabase db;
    late Directory root;
    late TileStore store;
    const provider = MapTilerRaster(apiKey: 'k');

    setUp(() async {
      db = AppDatabase(NativeDatabase.memory());
      root = await Directory.systemTemp.createTemp('place');
      store = TileStore(db: db, root: root);
    });
    tearDown(() async {
      await db.close();
      if (root.existsSync()) await root.delete(recursive: true);
    });

    Future<void> pump(WidgetTester tester, {LocationSource? location}) async {
      tester.view.physicalSize = const Size(420, 900);
      tester.view.devicePixelRatio = 1.0;
      addTearDown(tester.view.reset);
      await tester.pumpWidget(
        MaterialApp(
          theme: AppTokens.light,
          home: Scaffold(
            body: PlaceMap(
              provider: provider,
              store: store,
              place: civil,
              location: location,
            ),
          ),
        ),
      );
      await tester.pump();
      await tester.pump();
    }

    testWidgets('A PLACE THE DOWNLOAD DOES NOT REACH SAYS SO, not blank', (
      tester,
    ) async {
      await pump(tester);
      expect(find.textContaining('does not reach this place'), findsOneWidget);
      expect(find.byKey(const Key('place-pin')), findsNothing);
    });

    testWidgets('with tiles there, the pin is drawn', (tester) async {
      await tester.runAsync(
        () => store.write(
          provider.id,
          tileFor(civil, 13),
          Uint8List.fromList([0]),
        ),
      );
      await pump(tester);
      expect(find.byKey(const Key('place-pin')), findsOneWidget);
    });

    testWidgets('location refused is said, and nothing else happens', (
      tester,
    ) async {
      final phone = FakeLocation(HereState.notAsked,
          afterAsking: HereState.denied);
      await pump(tester, location: phone);
      await tester.tap(find.byKey(const Key('place-map-locate')));
      await tester.pump();
      await tester.pump();
      expect(phone.asks, [true]);
      expect(find.text('Location was not allowed.'), findsOneWidget);
    });

    testWidgets('a fix on a drawn map frames you and the place together', (
      tester,
    ) async {
      await tester.runAsync(
        () => store.write(provider.id, tileFor(civil, 14), Uint8List(1)),
      );
      await pump(
        tester,
        location: _FixedLocation(
          HereFix(
            at: const LatLng(25.5764, 91.8818),
            accuracyM: 9,
            time: DateTime.now(),
          ),
        ),
      );
      await tester.tap(find.byKey(const Key('place-map-locate')));
      await tester.pump();
      await tester.pump();
      expect(find.byKey(const Key('you-are-here')), findsOneWidget);
      expect(find.byKey(const Key('place-pin')), findsOneWidget);
      expect(find.byKey(const Key('place-map-distance')), findsOneWidget);
    });

    testWidgets('a fix gives the straight-line distance, even where no map '
        'is drawn', (tester) async {
      final phone = _FixedLocation(
        HereFix(
          // Police Bazar, about 1 km north.
          at: const LatLng(25.5764, 91.8818),
          accuracyM: 9,
          time: DateTime.now(),
        ),
      );
      await pump(tester, location: phone);
      await tester.tap(find.byKey(const Key('place-map-locate')));
      await tester.pump();
      await tester.pump();
      expect(
        find.textContaining('from you in a straight line'),
        findsOneWidget,
      );
      expect(find.textContaining('970 m'), findsOneWidget);
    });
  });

  group('what the leg counts as unplaced', () {
    test('short codes and toll-free lines are services, not places', () {
      expect(isServiceNumber('112'), isTrue);
      expect(isServiceNumber('+91181'), isTrue);
      expect(isServiceNumber('1098'), isTrue);
      expect(isServiceNumber('1800 11 1363'), isTrue);
      expect(isServiceNumber('+911800111363'), isTrue);
      expect(isServiceNumber('+913642224100'), isFalse);
      expect(isServiceNumber('+917827170170'), isFalse);
    });

    test('they are not counted as numbers waiting for a location', () {
      Contact c(int id, String phone) => contactOf().copyWith(
            id: id,
            phoneRaw: phone,
            phoneE164: const Value(null),
          );
      final diary = [
        c(1, '181'),
        c(2, '1800-599-2026'),
        c(3, '0364-2241497'),
      ];
      expect(unplacedContacts(diary, endStopIds: const {}), 1);
    });
  });
}

class _FixedLocation extends FakeLocation {
  final HereFix fix;
  _FixedLocation(this.fix) : super(HereState.locating);

  @override
  Future<HereFix?> once({Duration timeout = const Duration(seconds: 20)}) async =>
      fix;
}
