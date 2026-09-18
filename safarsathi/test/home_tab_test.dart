// test/home_tab_test.dart
//
// WHICH SCREEN THE APP OPENS ON, AND WHAT IS ONE TAP AWAY.
//
// Asked for directly: the trip should be the first thing on screen, and the
// map and the legs should be reachable from it rather than buried four taps
// deep under More. Tab order is the kind of thing a refactor reorders by
// accident and nobody notices until they open the app on a mountain road, so
// it is pinned here rather than left to whoever edits the list next.

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:safarsathi/core/theme/app_tokens.dart';
import 'package:safarsathi/core/widgets/app_shell.dart';
import 'package:safarsathi/features/trips/data/trip_summary.dart';
import 'package:safarsathi/features/trips/presentation/trip_screen.dart';

Widget _wrap(Widget child) =>
    MaterialApp(theme: AppTokens.light, home: child);

final _summary = TripSummary(
  name: 'Meghalaya',
  startDate: DateTime(2026, 10),
  endDate: DateTime(2026, 10, 5),
  stops: const [],
);

void main() {
  group('the app opens on the trip', () {
    testWidgets('the first destination is the one shown', (tester) async {
      var shown = '';
      await tester.pumpWidget(
        _wrap(
          AppShell(
            destinations: [
              ShellDestination(
                label: 'Trip',
                icon: Icons.route_outlined,
                screen: Builder(
                  builder: (_) {
                    shown = 'Trip';
                    return const Text('trip screen');
                  },
                ),
              ),
              const ShellDestination(
                label: 'Diary',
                icon: Icons.menu_book_outlined,
                screen: Text('diary screen'),
              ),
            ],
          ),
        ),
      );

      expect(shown, 'Trip');
    });
  });

  group('the map and the legs are on the first screen', () {
    testWidgets('both are offered when wired', (tester) async {
      var map = 0;
      var legs = 0;

      await tester.pumpWidget(
        _wrap(
          TripScreen(
            trip: Stream.value(_summary),
            unconfirmedCount: Stream.value(0),
            onViewMap: () => map++,
            onLegs: () => legs++,
          ),
        ),
      );
      await tester.pump();

      expect(find.text('The map'), findsOneWidget);
      expect(find.text('Getting between stops'), findsOneWidget);

      await tester.tap(find.text('The map'));
      await tester.tap(find.text('Getting between stops'));
      expect(map, 1);
      expect(legs, 1);
    });

    testWidgets('the section is absent rather than dead when unwired', (
      tester,
    ) async {
      // The golden harness and the widget tests build this screen without
      // either callback. A row that looks tappable and does nothing is worse
      // than no row.
      await tester.pumpWidget(
        _wrap(
          TripScreen(
            trip: Stream.value(_summary),
            unconfirmedCount: Stream.value(0),
          ),
        ),
      );
      await tester.pump();

      expect(find.text('On the road'.toUpperCase()), findsNothing);
      expect(find.text('The map'), findsNothing);
      expect(find.text('Getting between stops'), findsNothing);
    });
  });
}
