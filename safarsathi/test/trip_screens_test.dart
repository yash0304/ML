// test/trip_screens_test.dart — the editor screens for #16, #18 and #20.
//
// No database. Every screen takes plain values or a stream, which is the seam
// established at #6 and the reason these run in milliseconds.

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:safarsathi/core/database/app_database.dart';
import 'package:safarsathi/core/theme/app_tokens.dart';
import 'package:safarsathi/features/trips/data/readiness.dart';
import 'package:safarsathi/features/trips/data/trip_editor.dart';
import 'package:safarsathi/features/trips/presentation/itinerary_screen.dart';
import 'package:safarsathi/features/trips/presentation/leg_form_screen.dart';
import 'package:safarsathi/features/trips/presentation/leg_list_screen.dart';
import 'package:safarsathi/features/trips/presentation/readiness_panel.dart';
import 'package:safarsathi/features/trips/presentation/stop_form_screen.dart';

Widget wrap(Widget child, {Brightness brightness = Brightness.light}) =>
    MaterialApp(
      theme: brightness == Brightness.light ? AppTokens.light : AppTokens.dark,
      home: child,
    );

Stop stop(int id, String name, int order, {int nights = 0, String tags = ''}) =>
    Stop(
      id: id,
      tripId: 1,
      name: name,
      sequenceOrder: order,
      countryCode: 'IN',
      nights: nights,
      activityTags: tags,
    );

void main() {
  group('itinerary', () {
    testWidgets('lists stops in order with their nights', (tester) async {
      await tester.pumpWidget(
        wrap(
          ItineraryScreen(
            tripName: 'Meghalaya',
            stops: Stream.value([
              stop(1, 'Shillong', 1, nights: 2),
              stop(2, 'Cherrapunji', 2, nights: 1),
              stop(3, 'Dawki', 3),
            ]),
            onReorder: (_, _) async {},
            onEdit: (_) {},
            onAdd: () {},
          ),
        ),
      );
      await tester.pump();

      expect(find.text('Shillong'), findsOneWidget);
      expect(find.text('2 nights'), findsOneWidget);
      expect(find.text('1 night'), findsOneWidget);
      // A stop you do not sleep at says so rather than showing a bare zero.
      expect(find.text('passing through'), findsOneWidget);
    });

    testWidgets('THE SAME PLACE TWICE RENDERS AS TWO ROWS', (tester) async {
      await tester.pumpWidget(
        wrap(
          ItineraryScreen(
            tripName: 'Meghalaya',
            stops: Stream.value([
              stop(1, 'Shillong', 1),
              stop(2, 'Cherrapunji', 2),
              stop(3, 'Shillong', 3),
            ]),
            onReorder: (_, _) async {},
            onEdit: (_) {},
            onAdd: () {},
          ),
        ),
      );
      await tester.pump();

      expect(find.text('Shillong'), findsNWidgets(2));
      expect(find.text('1'), findsOneWidget);
      expect(find.text('3'), findsOneWidget);
    });

    testWidgets('tapping a stop opens it', (tester) async {
      Stop? opened;
      await tester.pumpWidget(
        wrap(
          ItineraryScreen(
            tripName: 'Meghalaya',
            stops: Stream.value([stop(1, 'Shillong', 1)]),
            onReorder: (_, _) async {},
            onEdit: (s) => opened = s,
            onAdd: () {},
          ),
        ),
      );
      await tester.pump();
      await tester.tap(find.text('Shillong'));
      await tester.pump();

      expect(opened?.id, 1);
    });

    testWidgets('an empty itinerary explains what to do', (tester) async {
      await tester.pumpWidget(
        wrap(
          ItineraryScreen(
            tripName: 'Meghalaya',
            stops: Stream.value(const []),
            onReorder: (_, _) async {},
            onEdit: (_) {},
            onAdd: () {},
          ),
        ),
      );
      await tester.pump();
      expect(find.textContaining('No stops yet'), findsOneWidget);
    });

    testWidgets('tags are shown so the checklist inputs are visible', (
      tester,
    ) async {
      await tester.pumpWidget(
        wrap(
          ItineraryScreen(
            tripName: 'Meghalaya',
            stops: Stream.value([
              stop(1, 'Sohra', 1, nights: 1, tags: 'caves,rain'),
            ]),
            onReorder: (_, _) async {},
            onEdit: (_) {},
            onAdd: () {},
          ),
        ),
      );
      await tester.pump();
      expect(find.text('caves · rain'), findsOneWidget);
    });
  });

  group('stop form', () {
    testWidgets('will not save a stop with no name', (tester) async {
      var saved = false;
      await tester.pumpWidget(
        wrap(StopFormScreen(onSave: (_) async => saved = true)),
      );

      await tester.tap(find.text('Add stop'));
      await tester.pump();
      expect(saved, isFalse);
    });

    testWidgets('saves the typed name', (tester) async {
      StopDraft? saved;
      await tester.pumpWidget(
        wrap(StopFormScreen(onSave: (d) async => saved = d)),
      );

      await tester.enterText(find.byType(TextField).first, '  Shillong  ');
      await tester.tap(find.text('Add stop'));
      await tester.pump();

      expect(saved?.name, 'Shillong');
    });

    testWidgets('says out loud that a place may repeat', (tester) async {
      await tester.pumpWidget(wrap(StopFormScreen(onSave: (_) async {})));
      expect(
        find.textContaining('can appear more than once'),
        findsOneWidget,
      );
    });

    testWidgets('an existing stop opens with its values', (tester) async {
      await tester.pumpWidget(
        wrap(
          StopFormScreen(
            existing: const StopDraft(
              id: 7,
              name: 'Cherrapunji',
              nights: 2,
              activityTags: ['caves'],
            ),
            onSave: (_) async {},
          ),
        ),
      );

      expect(find.text('Cherrapunji'), findsOneWidget);
      expect(find.text('Edit stop'), findsOneWidget);
      expect(find.text('Save stop'), findsOneWidget);
    });

    testWidgets('a tag toggles on and off', (tester) async {
      StopDraft? saved;
      await tester.pumpWidget(
        wrap(
          StopFormScreen(
            existing: const StopDraft(id: 7, name: 'Sohra'),
            onSave: (d) async => saved = d,
          ),
        ),
      );

      await tester.tap(find.text('CAVES'));
      await tester.pump();
      await tester.tap(find.text('Save stop'));
      await tester.pump();
      expect(saved?.activityTags, ['caves']);
    });

    testWidgets('lays out at phone width in both themes', (tester) async {
      tester.view.physicalSize = const Size(400, 800);
      tester.view.devicePixelRatio = 1.0;
      addTearDown(tester.view.reset);

      for (final b in Brightness.values) {
        await tester.pumpWidget(
          wrap(
            StopFormScreen(
              existing: const StopDraft(
                id: 1,
                name: 'A stop with a rather long name indeed',
                nights: 3,
              ),
              onSave: (_) async {},
            ),
            brightness: b,
          ),
        );
        expect(tester.takeException(), isNull);
      }
    });
  });

  group('leg form', () {
    testWidgets('says the times are typed, not looked up', (tester) async {
      await tester.pumpWidget(
        wrap(
          LegFormScreen(
            fromName: 'Shillong',
            toName: 'Dawki',
            onSave:
                ({
                  mode,
                  plannedDeparture,
                  plannedArrival,
                  required isBooked,
                  note,
                  vehicleNumber,
                }) async {},
          ),
        ),
      );

      expect(find.text('Shillong → Dawki'), findsOneWidget);
      expect(find.textContaining('Typed, not looked up'), findsOneWidget);
    });

    testWidgets('a mode and the booked flag round-trip', (tester) async {
      String? savedMode;
      bool? savedBooked;
      await tester.pumpWidget(
        wrap(
          LegFormScreen(
            fromName: 'A',
            toName: 'B',
            onSave:
                ({
                  mode,
                  plannedDeparture,
                  plannedArrival,
                  required isBooked,
                  note,
                  vehicleNumber,
                }) async {
                  savedMode = mode;
                  savedBooked = isBooked;
                },
          ),
        ),
      );

      await tester.tap(find.text('SHARED SUMO'));
      await tester.pump();
      await tester.tap(find.text('Not booked yet'));
      await tester.pump();
      await tester.tap(find.text('Save'));
      await tester.pump();

      expect(savedMode, 'Shared sumo');
      expect(savedBooked, isTrue);
    });
  });

  group('leg list', () {
    testWidgets('a leg with no transport set is flagged', (tester) async {
      await tester.pumpWidget(
        wrap(
          LegListScreen(
            legs: Stream.value(const [
              LegRow(id: 1, fromName: 'A', toName: 'B', isBooked: false),
              LegRow(
                id: 2,
                fromName: 'B',
                toName: 'C',
                mode: 'Taxi',
                isBooked: true,
              ),
            ]),
            onOpen: (_) {},
          ),
        ),
      );
      await tester.pump();

      expect(find.text('No transport set'), findsOneWidget);
      expect(find.text('Taxi'), findsOneWidget);
      expect(find.text('BOOKED'), findsOneWidget);
    });

    testWidgets('A LEG WITH NO KILOMETRES SAYS WHY, RATHER THAN NOTHING', (
      tester,
    ) async {
      // From the phone, 18 Sep: five legs, and the two touching Sohrra showed
      // an empty space where the distance goes. Both were blocked on the same
      // thing — that stop had no coordinates — and the screen said neither
      // which stop nor that it mattered.
      await tester.pumpWidget(
        wrap(
          LegListScreen(
            legs: Stream.value(const [
              LegRow(
                id: 1,
                fromName: 'Kongthong',
                toName: 'Sohrra',
                isBooked: false,
                stopsWithoutLocation: ['Sohrra'],
              ),
              LegRow(
                id: 2,
                fromName: 'Shillong',
                toName: 'Guwahati',
                isBooked: false,
                distanceKm: 105,
              ),
              LegRow(
                id: 3,
                fromName: 'Guwahati',
                toName: 'Shillong',
                isBooked: false,
              ),
            ]),
            onOpen: (_) {},
          ),
        ),
      );
      await tester.pump();

      // Names the stop, because that is the thing to go and fix.
      expect(find.text('Sohrra has no location yet'), findsOneWidget);
      // A located leg that simply has not been fetched reads differently:
      // one needs a download, the other needs you.
      expect(find.text('Not downloaded yet'), findsOneWidget);
      // And a leg that did download still just shows its distance.
      expect(find.text('105 km'), findsOneWidget);
    });

    testWidgets('both ends missing names both', (tester) async {
      await tester.pumpWidget(
        wrap(
          LegListScreen(
            legs: Stream.value(const [
              LegRow(
                id: 1,
                fromName: 'Kongthong',
                toName: 'Sohrra',
                isBooked: false,
                stopsWithoutLocation: ['Kongthong', 'Sohrra'],
              ),
            ]),
            onOpen: (_) {},
          ),
        ),
      );
      await tester.pump();
      expect(
        find.text('Kongthong and Sohrra have no location yet'),
        findsOneWidget,
      );
    });

    testWidgets('legs cannot be added here, and it says why', (tester) async {
      await tester.pumpWidget(
        wrap(LegListScreen(legs: Stream.value(const []), onOpen: (_) {})),
      );
      await tester.pump();
      expect(find.textContaining('add the stops instead'), findsOneWidget);
    });
  });

  group('readiness panel', () {
    testWidgets('a ready trip says so plainly', (tester) async {
      await tester.pumpWidget(
        wrap(
          Scaffold(
            body: ReadinessPanel(readiness: Stream.value(const Readiness([]))),
          ),
        ),
      );
      await tester.pump();
      expect(find.textContaining('has a number you have called'),
          findsOneWidget);
    });

    testWidgets('an unconfirmed number reads BLOCKING', (tester) async {
      await tester.pumpWidget(
        wrap(
          Scaffold(
            body: ReadinessPanel(
              readiness: Stream.value(
                const Readiness([
                  ReadinessItem(
                    stopId: 1,
                    stopName: 'Shillong',
                    contactId: 9,
                    label: 'Call and confirm the Shillong accommodation '
                        'number.',
                  ),
                ]),
              ),
            ),
          ),
        ),
      );
      await tester.pump();

      expect(find.text('BLOCKING'), findsOneWidget);
      expect(find.text('BEFORE YOU LEAVE SIGNAL'), findsOneWidget);
    });

    testWidgets('an ABSENT number reads differently from an unconfirmed one', (
      tester,
    ) async {
      await tester.pumpWidget(
        wrap(
          Scaffold(
            body: ReadinessPanel(
              readiness: Stream.value(
                const Readiness([
                  ReadinessItem(
                    stopId: 1,
                    stopName: 'Mawlynnong',
                    missing: true,
                    label: 'No accommodation number for Mawlynnong.',
                  ),
                ]),
              ),
            ),
          ),
        ),
      );
      await tester.pump();

      expect(find.text('NOTHING TO CALL'), findsOneWidget);
      expect(find.text('BLOCKING'), findsNothing);
    });

    testWidgets('the panel never uses emergency red', (tester) async {
      // Red belongs to the emergency tab. A call you have not made yet is a
      // thing to do before you leave, not a crisis.
      await tester.pumpWidget(
        wrap(
          Scaffold(
            body: ReadinessPanel(
              readiness: Stream.value(
                const Readiness([
                  ReadinessItem(stopId: 1, stopName: 'S', label: 'Call it.'),
                ]),
              ),
            ),
          ),
        ),
      );
      await tester.pump();

      final context = tester.element(find.byType(ReadinessPanel));
      final colours = AppTokens.of(context);
      final texts = tester.widgetList<Text>(find.byType(Text));
      for (final t in texts) {
        expect(t.style?.color, isNot(colours.emergency));
      }
    });
  });
}
