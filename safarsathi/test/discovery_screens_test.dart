// test/discovery_screens_test.dart — issues #27 and #28, the screens.

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:safarsathi/core/database/app_database.dart';
import 'package:safarsathi/core/theme/app_tokens.dart';
import 'package:safarsathi/features/contacts/data/contacts_dao.dart';
import 'package:safarsathi/features/discovery/data/discovery.dart';
import 'package:safarsathi/features/discovery/presentation/discovery_screen.dart';
import 'package:safarsathi/features/discovery/presentation/poi_detail_screen.dart';

void useTallSurface(WidgetTester tester) {
  tester.view.physicalSize = const Size(430, 2400);
  tester.view.devicePixelRatio = 1.0;
  addTearDown(tester.view.reset);
}

Widget wrap(Widget child, {Brightness brightness = Brightness.light}) =>
    MaterialApp(
      theme: brightness == Brightness.light ? AppTokens.light : AppTokens.dark,
      home: child,
    );

PoiContact phone(String raw) => PoiContact(
  id: 1,
  poiId: 1,
  phoneRaw: raw,
  phoneE164: raw.replaceAll(' ', ''),
  tier: 'communityOsm',
  sourceTag: 'phone',
);

CorridorPlace placeOf({
  int id = 1,
  String name = 'Sohra PHC',
  String category = ContactCategory.hospital,
  double along = 20,
  double off = 0.4,
  bool withPhone = true,
}) => CorridorPlace(
  id: id,
  name: name,
  category: category,
  lat: 25.4,
  lon: 91.8,
  alongRouteKm: along,
  offRouteKm: off,
  osmId: 'node/1',
  phones: withPhone ? [phone('+91 364 222 2222')] : const [],
);

LegDiscovery legOf({
  List<CorridorPlace>? places,
  bool synced = true,
}) => LegDiscovery(
  legId: 1,
  fromName: 'Shillong',
  toName: 'Cherrapunji',
  distanceKm: 54,
  lastSyncedAt: synced ? DateTime(2026, 9, 20) : null,
  places:
      places ??
      [
        placeOf(id: 1, name: 'IOC pump', category: ContactCategory.fuel,
            along: 6, off: 0.1),
        placeOf(id: 2, name: 'Sohra PHC', along: 21, off: 0.4),
        placeOf(
          id: 3,
          name: 'Laitlum dhaba',
          category: ContactCategory.restaurant,
          along: 38,
          off: 2.1,
          withPhone: false,
        ),
      ],
);

void main() {
  group('the discovery list', () {
    Widget screen(LegDiscovery leg, {void Function(CorridorPlace)? onOpen}) =>
        wrap(
          DiscoveryScreen(
            discovery: Stream.value(leg),
            onOpen: onOpen ?? (_) {},
          ),
        );

    testWidgets('places render in route order with their kilometres', (
      tester,
    ) async {
      useTallSurface(tester);
      await tester.pumpWidget(screen(legOf()));
      await tester.pump();

      expect(find.text('IOC pump'), findsOneWidget);
      expect(find.text('Sohra PHC'), findsOneWidget);
      expect(find.text('Laitlum dhaba'), findsOneWidget);
      // The milestone numerals.
      expect(find.text('6'), findsOneWidget);
      expect(find.text('21'), findsOneWidget);
      expect(find.text('38'), findsOneWidget);
    });

    testWidgets('a place on the road says so rather than showing 0.1 km', (
      tester,
    ) async {
      useTallSurface(tester);
      await tester.pumpWidget(screen(legOf()));
      await tester.pump();

      expect(find.text('on the road'), findsOneWidget);
      expect(find.text('2.1 km off the road'), findsOneWidget);
    });

    testWidgets('only the categories present are offered as chips', (
      tester,
    ) async {
      useTallSurface(tester);
      await tester.pumpWidget(screen(legOf()));
      await tester.pump();

      // Scoped to the chip row: each milestone cap also carries its
      // category, so an unscoped finder matches twice.
      Finder chip(String label) => find.descendant(
        of: find.byType(Wrap),
        matching: find.text(label),
      );

      expect(chip('ALL'), findsOneWidget);
      expect(chip('FUEL'), findsOneWidget);
      expect(chip('HOSPITAL'), findsOneWidget);
      expect(chip('FOOD'), findsOneWidget);
      // Nothing on this leg is a chemist.
      expect(chip('PHARMACY'), findsNothing);
    });

    testWidgets('a chip filters the list', (tester) async {
      useTallSurface(tester);
      await tester.pumpWidget(screen(legOf()));
      await tester.pump();

      await tester.tap(
        find.descendant(of: find.byType(Wrap), matching: find.text('FUEL')),
      );
      await tester.pump();

      expect(find.text('IOC pump'), findsOneWidget);
      expect(find.text('Sohra PHC'), findsNothing);
    });

    group('diet chips', () {
      Finder chip(String label) => find.descendant(
        of: find.byType(Wrap),
        matching: find.text(label),
      );

      CorridorPlace food(int id, String name, Map<String, String> tags) =>
          CorridorPlace(
            id: id,
            name: name,
            category: ContactCategory.restaurant,
            lat: 25.4,
            lon: 91.8,
            alongRouteKm: 10.0 + id,
            offRouteKm: 0.1,
            tags: tags,
          );

      testWidgets('NO JAIN CHIP WHEN NOTHING ON THE LEG SAYS JAIN', (
        tester,
      ) async {
        useTallSurface(tester);
        await tester.pumpWidget(screen(legOf()));
        await tester.pump();
        expect(chip('VEG'), findsNothing);
        expect(chip('JAIN'), findsNothing);
      });

      testWidgets('Veg and Jain filter the list, and stack', (tester) async {
        useTallSurface(tester);
        await tester.pumpWidget(screen(legOf(places: [
          food(1, 'Momo Point', {'amenity': 'fast_food'}),
          food(2, 'Green Leaf', {'diet:vegetarian': 'only'}),
          food(3, 'Marwari Bhojanalaya',
              {'diet:vegetarian': 'only', 'diet:jain': 'yes'}),
        ])));
        await tester.pump();

        await tester.tap(chip('VEG'));
        await tester.pump();
        expect(find.text('Momo Point'), findsNothing);
        expect(find.text('Green Leaf'), findsOneWidget);
        expect(find.text('Marwari Bhojanalaya'), findsOneWidget);

        await tester.tap(chip('JAIN'));
        await tester.pump();
        expect(find.text('Green Leaf'), findsNothing);
        expect(find.text('Marwari Bhojanalaya'), findsOneWidget);

        // Off again brings the rest back.
        await tester.tap(chip('VEG'));
        await tester.tap(chip('JAIN'));
        await tester.pump();
        expect(find.text('Momo Point'), findsOneWidget);
      });

      testWidgets('an empty filter says so', (tester) async {
        useTallSurface(tester);
        await tester.pumpWidget(screen(legOf(places: [
          placeOf(id: 1, name: 'IOC pump', category: ContactCategory.fuel),
          food(2, 'Green Leaf', {'diet:vegetarian': 'only'}),
        ])));
        await tester.pump();

        await tester.tap(chip('FUEL'));
        await tester.tap(chip('VEG'));
        await tester.pump();
        expect(find.text('Nothing on this leg matches those filters.'),
            findsOneWidget);
      });
    });

    testWidgets('tapping a place opens it', (tester) async {
      useTallSurface(tester);
      CorridorPlace? opened;
      await tester.pumpWidget(screen(legOf(), onOpen: (p) => opened = p));
      await tester.pump();

      await tester.tap(find.text('Sohra PHC'));
      await tester.pump();
      expect(opened?.id, 2);
    });

    testWidgets('AN UNSYNCED LEG READS DIFFERENTLY FROM AN EMPTY ONE', (
      tester,
    ) async {
      // Confusing the two sends the user to the wrong screen.
      await tester.pumpWidget(screen(legOf(places: const [], synced: false)));
      await tester.pump();
      expect(find.textContaining('has not been downloaded'), findsOneWidget);

      await tester.pumpWidget(screen(legOf(places: const [], synced: true)));
      await tester.pump();
      expect(find.textContaining('simply has nothing tagged'), findsOneWidget);
    });

    testWidgets('it says numbers here are unchecked', (tester) async {
      useTallSurface(tester);
      await tester.pumpWidget(screen(legOf()));
      await tester.pump();
      expect(find.textContaining('carries the amber dot'), findsOneWidget);
    });
  });

  group('the place detail', () {
    Widget detail({
      CorridorPlace? place,
      bool alreadySaved = false,
      Future<void> Function(PoiContact)? onSave,
      Future<void> Function()? onOpenMaps,
    }) => wrap(
      PoiDetailScreen(
        place: place ?? placeOf(),
        alreadySaved: alreadySaved,
        onCopy: (_) async {},
        onOpenDialer: (_) async {},
        onSave: onSave ?? (_) async {},
        onOpenMaps: onOpenMaps ?? () async {},
      ),
    );

    testWidgets('both distances are shown', (tester) async {
      useTallSurface(tester);
      await tester.pumpWidget(detail());
      await tester.pump();

      expect(find.text('20 km'), findsOneWidget);
      expect(find.text('0.4 km'), findsOneWidget);
    });

    testWidgets('THE PROVENANCE IS IN THE SAME WORDS THE DIARY USES', (
      tester,
    ) async {
      // The two screens must never disagree about what this tier means.
      useTallSurface(tester);
      await tester.pumpWidget(detail());
      await tester.pump();

      expect(
        find.textContaining('From open map data · nobody has checked it'),
        findsOneWidget,
      );
    });

    testWidgets('SAVING SAYS IT WILL NOT MAKE THE NUMBER TRUSTED', (
      tester,
    ) async {
      // The one place a user might expect saving to change the tier.
      useTallSurface(tester);
      await tester.pumpWidget(detail());
      await tester.pump();

      expect(find.text('Save to the diary'), findsOneWidget);
      expect(find.textContaining('keep the amber dot'), findsOneWidget);
    });

    testWidgets('saving reports the phone it saved', (tester) async {
      useTallSurface(tester);
      PoiContact? saved;
      await tester.pumpWidget(detail(onSave: (p) async => saved = p));
      await tester.pump();

      await tester.tap(find.text('Save to the diary'));
      await tester.pump();
      expect(saved?.phoneRaw, '+91 364 222 2222');
    });

    testWidgets('an already-saved place does not offer to save again', (
      tester,
    ) async {
      useTallSurface(tester);
      await tester.pumpWidget(detail(alreadySaved: true));
      await tester.pump();

      expect(find.text('Already in your diary.'), findsOneWidget);
      expect(find.text('Save to the diary'), findsNothing);
    });

    testWidgets('a place with no number says so and offers no save', (
      tester,
    ) async {
      useTallSurface(tester);
      await tester.pumpWidget(detail(place: placeOf(withPhone: false)));
      await tester.pump();

      expect(find.textContaining('No number on the map'), findsOneWidget);
      expect(find.text('Save to the diary'), findsNothing);
    });

    testWidgets('THE MAPS LINK SAYS IT NEEDS SIGNAL, BEFORE THE TAP', (
      tester,
    ) async {
      // A dead tap on a mountain road with no explanation is worse than no
      // button.
      useTallSurface(tester);
      await tester.pumpWidget(detail());
      await tester.pump();

      expect(find.text('Open in Google Maps'), findsOneWidget);
      expect(find.textContaining('Needs signal'), findsOneWidget);
    });

    testWidgets('copy and dialer are both offered, copy first', (
      tester,
    ) async {
      useTallSurface(tester);
      await tester.pumpWidget(detail());
      await tester.pump();

      final copy = tester.getTopLeft(find.text('COPY')).dx;
      final dialer = tester.getTopLeft(find.text('DIALER')).dx;
      expect(copy, lessThan(dialer));
    });

    testWidgets('lays out at phone width in both themes', (tester) async {
      tester.view.physicalSize = const Size(400, 800);
      tester.view.devicePixelRatio = 1.0;
      addTearDown(tester.view.reset);

      for (final b in Brightness.values) {
        await tester.pumpWidget(
          wrap(
            PoiDetailScreen(
              place: placeOf(name: 'A rather long place name for a dhaba'),
              onCopy: (_) async {},
              onOpenDialer: (_) async {},
              onSave: (_) async {},
              onOpenMaps: () async {},
            ),
            brightness: b,
          ),
        );
        await tester.pump();
        expect(tester.takeException(), isNull);
      }
    });
  });
}
