// test/place_details_test.dart — what a food place serves, as it reads.

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:safarsathi/core/theme/app_tokens.dart';
import 'package:safarsathi/features/discovery/data/discovery.dart';
import 'package:safarsathi/features/discovery/data/place_details.dart';
import 'package:safarsathi/features/discovery/presentation/poi_detail_screen.dart';

void main() {
  group('keeping', () {
    test('only the tags worth carrying are stored', () {
      final raw = encodeKeptTags({
        'amenity': 'fast_food',
        'cuisine': 'momo',
        'operator': 'x',
        'wheelchair': 'no',
      });
      expect(decodeKeptTags(raw), {'amenity': 'fast_food', 'cuisine': 'momo'});
    });

    test('nothing worth keeping stores nothing', () {
      expect(encodeKeptTags({'operator': 'x'}), isNull);
    });

    test('a damaged row loses its details, never the place', () {
      expect(decodeKeptTags('{not json'), isEmpty);
      expect(decodeKeptTags(null), isEmpty);
    });
  });

  group('reading', () {
    test('cuisine: a readable list', () {
      expect(cuisineOf({'cuisine': 'indian;chinese;momo'}),
          'Indian, Chinese, Momo');
      expect(cuisineOf({'cuisine': 'regional_indian'}), 'Regional Indian');
    });

    test('VEG: "yes" AND "only" ARE DIFFERENT ANSWERS', () {
      expect(vegOf({'diet:vegetarian': 'only'}), 'Pure veg');
      expect(vegOf({'diet:vegetarian': 'yes'}), 'Veg options');
      expect(vegOf({'diet:vegetarian': 'no'}), 'No veg options');
      expect(vegOf({'diet:vegetarian': 'yes', 'diet:vegan': 'yes'}),
          'Veg and vegan');
      expect(vegOf({}), isNull);
    });

    test('hours: readable, not pretend-parsed', () {
      expect(hoursOf({'opening_hours': 'Mo-Su 09:00-21:00'}),
          'Mon–Sun 09:00–21:00');
      expect(hoursOf({'opening_hours': 'Mo-Sa 08:00-20:00; Su off'}),
          'Mon–Sat 08:00–20:00, Sun off');
      expect(hoursOf({'opening_hours': '24/7'}), 'Open 24 hours');
    });

    test('one line for a row', () {
      expect(
        foodLine({
          'amenity': 'fast_food',
          'cuisine': 'indian;momo',
          'diet:vegetarian': 'yes',
        }),
        'Fast food · Indian, Momo · Veg options',
      );
      expect(foodLine({'amenity': 'fuel'}), isNull);
    });
  });

  testWidgets('the place screen shows what they serve, and says it is not a '
      'menu', (tester) async {
    tester.view.physicalSize = const Size(420, 1600);
    tester.view.devicePixelRatio = 1.0;
    addTearDown(tester.view.reset);
    await tester.pumpWidget(MaterialApp(
      theme: AppTokens.light,
      home: PoiDetailScreen(
        place: const CorridorPlace(
          id: 1,
          name: 'Momo Point',
          category: 'restaurant',
          lat: 25.42,
          lon: 91.81,
          alongRouteKm: 21,
          offRouteKm: 0.1,
          tags: {
            'amenity': 'fast_food',
            'cuisine': 'indian;momo',
            'diet:vegetarian': 'only',
            'opening_hours': 'Mo-Su 09:00-21:00',
          },
        ),
        onCopy: (_) async {},
        onOpenDialer: (_) async {},
        onSave: (_) async {},
      ),
    ));
    await tester.pump();

    expect(find.text('WHAT THEY SERVE'), findsOneWidget);
    expect(find.text('Fast food'), findsOneWidget);
    expect(find.text('Indian, Momo'), findsOneWidget);
    expect(find.text('Pure veg'), findsOneWidget);
    expect(find.text('Mon–Sun 09:00–21:00'), findsOneWidget);
    expect(find.textContaining('does not carry menus'), findsOneWidget);
  });

  testWidgets('a place with nothing known shows no empty section', (
    tester,
  ) async {
    await tester.pumpWidget(MaterialApp(
      theme: AppTokens.light,
      home: PoiDetailScreen(
        place: const CorridorPlace(
          id: 2, name: 'IOC Umroi', category: 'fuel', lat: 25.6, lon: 91.9,
          alongRouteKm: 6, offRouteKm: 0.1,
        ),
        onCopy: (_) async {},
        onOpenDialer: (_) async {},
        onSave: (_) async {},
      ),
    ));
    await tester.pump();
    expect(find.text('WHAT THEY SERVE'), findsNothing);
  });
}
