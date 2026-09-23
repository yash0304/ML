// test/here_test.dart — "you are here" on the offline map.

import 'dart:async';
import 'dart:io';

import 'package:drift/native.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:geolocator/geolocator.dart' show LocationPermission;

import 'package:safarsathi/core/database/app_database.dart';
import 'package:safarsathi/core/theme/app_tokens.dart';
import 'package:safarsathi/features/discovery/data/geo.dart';
import 'package:safarsathi/features/map/data/here.dart';
import 'package:safarsathi/features/map/data/tile_provider.dart';
import 'package:safarsathi/features/map/data/tile_store.dart';
import 'package:safarsathi/features/map/presentation/trip_map_screen.dart';

/// A phone whose answers the test writes.
class FakeLocation implements LocationSource {
  HereState state;
  HereState afterAsking;
  final fixes = StreamController<HereFix>.broadcast();
  final asks = <bool>[];
  var appSettingsOpened = 0;
  var locationSettingsOpened = 0;

  FakeLocation(this.state, {HereState? afterAsking})
      : afterAsking = afterAsking ?? state;

  @override
  Future<HereState> check({bool ask = false}) async {
    asks.add(ask);
    if (ask) state = afterAsking;
    return state;
  }

  @override
  Stream<HereFix> watch() => fixes.stream;

  @override
  Future<HereFix?> once({Duration timeout = const Duration(seconds: 20)}) async =>
      null;

  @override
  Future<void> openAppSettings() async => appSettingsOpened++;

  @override
  Future<void> openLocationSettings() async => locationSettingsOpened++;
}

final sohraFix = HereFix(
  at: const LatLng(25.2718, 91.7327),
  accuracyM: 12,
  time: DateTime.now(),
);

void main() {
  group('permission adds up to one state', () {
    test('location switched off wins over everything', () {
      for (final p in LocationPermission.values) {
        expect(stateFor(serviceOn: false, permission: p), HereState.serviceOff);
      }
    });

    test('each permission answer', () {
      HereState s(LocationPermission p) => stateFor(serviceOn: true, permission: p);
      expect(s(LocationPermission.denied), HereState.denied);
      expect(s(LocationPermission.deniedForever), HereState.blocked);
      expect(s(LocationPermission.whileInUse), HereState.locating);
      expect(s(LocationPermission.always), HereState.locating);
      expect(s(LocationPermission.unableToDetermine), HereState.notAsked);
    });
  });

  test('a fix is described by how precise and how old it is', () {
    final fix = HereFix(
      at: const LatLng(0, 0),
      accuracyM: 12.4,
      time: DateTime(2026, 10, 3, 17, 0, 0),
    );
    expect(describeFix(fix, now: DateTime(2026, 10, 3, 17, 0, 20)),
        '±12 m, 20 s ago');
    expect(describeFix(fix, now: DateTime(2026, 10, 3, 17, 7)),
        '±12 m, 7 min ago');
  });

  group('the map screen', () {
    late AppDatabase db;
    late Directory root;
    late TileStore store;
    var loads = 0;

    setUp(() async {
      db = AppDatabase(NativeDatabase.memory());
      root = await Directory.systemTemp.createTemp('here');
      store = TileStore(db: db, root: root);
      loads = 0;
    });
    tearDown(() async {
      await db.close();
      if (root.existsSync()) await root.delete(recursive: true);
    });

    Future<void> pump(WidgetTester tester, LocationSource? location) async {
      tester.view.physicalSize = const Size(420, 900);
      tester.view.devicePixelRatio = 1.0;
      addTearDown(tester.view.reset);
      await tester.pumpWidget(
        MaterialApp(
          theme: AppTokens.light,
          home: TripMapScreen(
            provider: const MapTilerRaster(apiKey: 'k'),
            store: store,
            location: location,
            load: () async {
              loads++;
              return const TripMapView(
                route: [],
                stops: [(name: 'Sohrra', at: LatLng(25.2718, 91.7327))],
                tileCount: 3424,
              );
            },
          ),
        ),
      );
      await tester.pump();
      await tester.pump();
    }

    testWidgets('NOTHING IS ASKED ON OPEN — only a tap asks', (tester) async {
      final phone = FakeLocation(HereState.notAsked,
          afterAsking: HereState.locating);
      await pump(tester, phone);
      expect(phone.asks, [false]);
      expect(find.textContaining('Tap the target'), findsOneWidget);
    });

    testWidgets('a tap asks, then the first fix puts you on the map',
        (tester) async {
      final phone = FakeLocation(HereState.notAsked,
          afterAsking: HereState.locating);
      await pump(tester, phone);

      await tester.tap(find.bySemanticsLabel('Show where I am'));
      await tester.pump();
      expect(phone.asks.last, isTrue);
      expect(find.textContaining('Finding you'), findsOneWidget);

      phone.fixes.add(sohraFix);
      await tester.pump();
      await tester.pump();
      expect(find.byKey(const Key('you-are-here')), findsOneWidget);
      expect(find.textContaining('You are here — ±12 m'), findsOneWidget);
    });

    testWidgets('allowed on an earlier visit, it starts without a prompt',
        (tester) async {
      final phone = FakeLocation(HereState.locating);
      await pump(tester, phone);
      expect(phone.asks, [false]);
      phone.fixes.add(sohraFix);
      await tester.pump();
      expect(find.byKey(const Key('you-are-here')), findsOneWidget);
    });

    testWidgets('blocked for good, the tap opens the app\'s settings',
        (tester) async {
      final phone = FakeLocation(HereState.blocked);
      await pump(tester, phone);
      expect(find.textContaining('blocked for SafarSathi'), findsOneWidget);
      await tester.tap(find.bySemanticsLabel('Show where I am'));
      await tester.pump();
      expect(phone.appSettingsOpened, 1);
    });

    testWidgets('location switched off says so and opens its setting',
        (tester) async {
      final phone = FakeLocation(HereState.serviceOff);
      await pump(tester, phone);
      expect(find.textContaining('switched off on this phone'), findsOneWidget);
      await tester.tap(find.bySemanticsLabel('Show where I am'));
      await tester.pump();
      expect(phone.locationSettingsOpened, 1);
    });

    testWidgets('THE MAP IS LOADED ONCE, HOWEVER OFTEN THE DOT MOVES',
        (tester) async {
      // load() used to sit inside build. With a dot rebuilding the screen
      // every few metres, that would have been a database read per step.
      final phone = FakeLocation(HereState.locating);
      await pump(tester, phone);
      for (var i = 0; i < 5; i++) {
        phone.fixes.add(HereFix(
          at: LatLng(25.2718 + i * 0.0001, 91.7327),
          accuracyM: 10,
          time: DateTime.now(),
        ));
        await tester.pump();
      }
      expect(loads, 1);
    });

    testWidgets('with no location source there is no button at all',
        (tester) async {
      await pump(tester, null);
      expect(find.bySemanticsLabel('Show where I am'), findsNothing);
      expect(find.textContaining('Tap the target'), findsNothing);
    });
  });
}
