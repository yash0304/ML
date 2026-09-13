// test/setup_screens_test.dart — the place picker, weather and settings.

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:safarsathi/core/database/app_database.dart';
import 'package:safarsathi/core/theme/app_tokens.dart';
import 'package:safarsathi/features/discovery/data/geo.dart';
import 'package:safarsathi/features/discovery/data/geocoder.dart';
import 'package:safarsathi/features/settings/data/settings.dart';
import 'package:safarsathi/features/settings/presentation/settings_screen.dart';
import 'package:safarsathi/features/trips/data/trip_editor.dart';
import 'package:safarsathi/features/trips/presentation/place_picker_sheet.dart';
import 'package:safarsathi/features/trips/presentation/stop_form_screen.dart';
import 'package:safarsathi/features/weather/data/weather_sync.dart';
import 'package:safarsathi/features/weather/presentation/weather_screen.dart';

/// These screens are long lists and most assertions are below a phone's fold.
/// A tall surface is simpler and less brittle than scrolling to each one; the
/// phone-width layout is covered by its own test.
void useTallSurface(WidgetTester tester) {
  tester.view.physicalSize = const Size(420, 2200);
  tester.view.devicePixelRatio = 1.0;
  addTearDown(tester.view.reset);
}

Widget wrap(Widget child, {Brightness brightness = Brightness.light}) =>
    MaterialApp(
      theme: brightness == Brightness.light ? AppTokens.light : AppTokens.dark,
      home: child,
    );

const _candidates = [
  GeocodeResult(
    displayName: 'Shillong, East Khasi Hills, Meghalaya, India',
    location: LatLng(25.5788, 91.8933),
    kind: 'city',
  ),
  GeocodeResult(
    displayName: 'Shillong, Karnataka, India',
    location: LatLng(12.9716, 77.5946),
    kind: 'village',
  ),
];

WeatherSnapshot day(
  int id,
  DateTime date,
  String condition,
  DateTime cachedAt,
) => WeatherSnapshot(
  id: id,
  stopId: 1,
  forDate: date,
  condition: condition,
  cachedAt: cachedAt,
  tempMinC: 17,
  tempMaxC: 24,
  rainMm: 12,
);

void main() {
  group('place picker', () {
    testWidgets('NOTHING IS SAVED UNTIL A CANDIDATE IS PICKED', (tester) async {
      useTallSurface(tester);
      // The whole reason this screen exists rather than taking the first
      // result: two real places called Shillong, 2,500 km apart.
      await tester.pumpWidget(
        wrap(
          PlacePickerSheet(
            initialQuery: 'Shillong',
            search: (_, _) async => _candidates,
          ),
        ),
      );

      await tester.tap(find.text('Search'));
      await tester.pumpAndSettle();

      // The hint text also mentions Meghalaya, so match the full result line.
      expect(
        find.text('Shillong, East Khasi Hills, Meghalaya, India'),
        findsOneWidget,
      );
      expect(find.text('Shillong, Karnataka, India'), findsOneWidget);
      expect(find.text('CITY'), findsOneWidget);
      expect(find.text('VILLAGE'), findsOneWidget);
    });

    testWidgets('a lookup that finds nothing suggests what to do', (
      tester,
    ) async {
      await tester.pumpWidget(
        wrap(
          PlacePickerSheet(
            initialQuery: 'Nowhere at all',
            search: (_, _) async => const [],
          ),
        ),
      );

      await tester.tap(find.text('Search'));
      await tester.pumpAndSettle();
      expect(find.textContaining('Nothing found'), findsOneWidget);
    });

    testWidgets('a failed lookup shows the reason, not a crash', (
      tester,
    ) async {
      await tester.pumpWidget(
        wrap(
          PlacePickerSheet(
            initialQuery: 'Shillong',
            search: (_, _) async =>
                throw const GeocodeException('The lookup service is down.'),
          ),
        ),
      );

      await tester.tap(find.text('Search'));
      await tester.pumpAndSettle();
      expect(find.textContaining('lookup service is down'), findsOneWidget);
      expect(tester.takeException(), isNull);
    });

    testWidgets('typed coordinates offer a button only once valid', (
      tester,
    ) async {
      await tester.pumpWidget(
        wrap(
          PlacePickerSheet(
            initialQuery: '',
            search: (_, _) async => const [],
          ),
        ),
      );

      expect(find.text('Use these coordinates'), findsNothing);

      await tester.enterText(find.byType(TextField).last, '25.5788, 91.8933');
      await tester.pump();
      expect(find.text('Use these coordinates'), findsOneWidget);

      await tester.enterText(find.byType(TextField).last, '999, 999');
      await tester.pump();
      expect(find.text('Use these coordinates'), findsNothing);
    });

    testWidgets('it says the lookup is the network moment', (tester) async {
      await tester.pumpWidget(
        wrap(
          PlacePickerSheet(initialQuery: '', search: (_, _) async => const []),
        ),
      );
      expect(find.textContaining('only moment this screen uses the network'),
          findsOneWidget);
    });
  });

  group('the stop form coordinate row', () {
    testWidgets('A STOP WITH NO COORDINATES SAYS WHAT THAT COSTS', (
      tester,
    ) async {
      useTallSurface(tester);
      await tester.pumpWidget(
        wrap(
          StopFormScreen(
            existing: const StopDraft(id: 1, name: 'Shillong'),
            onSave: (_) async {},
            onPickPlace: (_, _) async => null,
          ),
        ),
      );

      expect(find.text('Not set — find it'), findsOneWidget);
      expect(find.textContaining('cannot be downloaded'), findsOneWidget);
    });

    testWidgets('a stop with coordinates shows them', (tester) async {
      useTallSurface(tester);
      await tester.pumpWidget(
        wrap(
          StopFormScreen(
            existing: const StopDraft(
              id: 1,
              name: 'Shillong',
              lat: 25.5788,
              lon: 91.8933,
            ),
            onSave: (_) async {},
            onPickPlace: (_, _) async => null,
          ),
        ),
      );
      expect(find.text('25.57880, 91.89330'), findsOneWidget);
    });

    testWidgets('picking a place saves its coordinates', (tester) async {
      useTallSurface(tester);
      StopDraft? saved;
      await tester.pumpWidget(
        wrap(
          StopFormScreen(
            existing: const StopDraft(id: 1, name: 'Shillong'),
            onSave: (d) async => saved = d,
            onPickPlace: (_, _) async => const LatLng(25.5788, 91.8933),
          ),
        ),
      );

      await tester.tap(find.text('Not set — find it'));
      await tester.pumpAndSettle();
      await tester.tap(find.text('Save stop'));
      await tester.pump();

      expect(saved?.lat, closeTo(25.5788, 0.0001));
      expect(saved?.hasCoordinates, isTrue);
    });

    testWidgets('the section is absent when no picker is wired', (
      tester,
    ) async {
      await tester.pumpWidget(
        wrap(
          StopFormScreen(
            existing: const StopDraft(id: 1, name: 'Shillong'),
            onSave: (_) async {},
          ),
        ),
      );
      expect(find.text('WHERE IT IS'), findsNothing);
    });
  });

  group('weather', () {
    final taken = DateTime(2026, 10, 1, 9);

    Widget screen(List<StopWeather> stops, {DateTime? now}) => wrap(
      WeatherScreen(weather: Stream.value(stops), now: now),
    );

    testWidgets('THE AGE IS STATED BESIDE EVERY STOP', (tester) async {
      await tester.pumpWidget(
        screen(
          [
            StopWeather(
              stopId: 1,
              stopName: 'Shillong',
              cachedAt: taken,
              days: [day(1, DateTime(2026, 10, 2), 'Light rain', taken)],
            ),
          ],
          now: taken.add(const Duration(days: 3)),
        ),
      );
      await tester.pump();

      expect(find.text('TAKEN 3 DAYS AGO'), findsOneWidget);
    });

    testWidgets('BEYOND A WEEK IT SAYS SO IN CAUTION', (tester) async {
      // A five-day-old forecast shown as today's weather is worse than no
      // forecast, because somebody packs on it.
      await tester.pumpWidget(
        screen(
          [
            StopWeather(
              stopId: 1,
              stopName: 'Shillong',
              cachedAt: taken,
              days: [day(1, DateTime(2026, 10, 2), 'Light rain', taken)],
            ),
          ],
          now: taken.add(const Duration(days: 9)),
        ),
      );
      await tester.pump();

      expect(find.textContaining('Over a week old'), findsOneWidget);

      final context = tester.element(find.byType(WeatherScreen));
      final c = AppTokens.of(context);
      final label = tester.widget<Text>(find.text('TAKEN 9 DAYS AGO'));
      expect(label.style?.color, c.cautionMark);
    });

    testWidgets('a fresh forecast does not shout', (tester) async {
      await tester.pumpWidget(
        screen(
          [
            StopWeather(
              stopId: 1,
              stopName: 'Shillong',
              cachedAt: taken,
              days: [day(1, DateTime(2026, 10, 2), 'Clear', taken)],
            ),
          ],
          now: taken.add(const Duration(hours: 4)),
        ),
      );
      await tester.pump();
      expect(find.textContaining('Over a week old'), findsNothing);
    });

    testWidgets('nothing downloaded explains what to do', (tester) async {
      await tester.pumpWidget(
        screen(const [
          StopWeather(stopId: 1, stopName: 'Shillong', days: []),
        ]),
      );
      await tester.pump();
      expect(find.textContaining('Fetch it on WiFi'), findsOneWidget);
    });

    testWidgets('it says this is a picture, not live weather', (tester) async {
      await tester.pumpWidget(
        screen(
          [
            StopWeather(
              stopId: 1,
              stopName: 'Shillong',
              cachedAt: taken,
              days: [day(1, DateTime(2026, 10, 2), 'Clear', taken)],
            ),
          ],
          now: taken,
        ),
      );
      await tester.pump();
      expect(find.textContaining('not live weather'), findsOneWidget);
    });
  });

  group('settings', () {
    Widget screen({
      ThemeMode mode = ThemeMode.system,
      List<CacheSummary> caches = const [],
      Future<void> Function(ThemeMode)? onThemeMode,
      Future<void> Function(int)? onClear,
    }) => wrap(
      SettingsScreen(
        themeMode: Stream.value(mode),
        onThemeMode: onThemeMode ?? (_) async {},
        corridorKm: Stream.value(3.0),
        onCorridorKm: (_) async {},
        caches: Stream.value(caches),
        onClearCache: onClear ?? (_) async {},
        onCallHistory: () {},
      ),
    );

    testWidgets('the theme override offers three states', (tester) async {
      await tester.pumpWidget(screen());
      await tester.pump();

      expect(find.text('FOLLOW PHONE'), findsOneWidget);
      expect(find.text('PAPER'), findsOneWidget);
      expect(find.text('LAMP'), findsOneWidget);
    });

    testWidgets('choosing a theme reports it', (tester) async {
      ThemeMode? chosen;
      await tester.pumpWidget(screen(onThemeMode: (m) async => chosen = m));
      await tester.pump();

      await tester.tap(find.text('LAMP'));
      await tester.pump();
      expect(chosen, ThemeMode.dark);
    });

    testWidgets('a cache row states what is actually held', (tester) async {
      await tester.pumpWidget(
        screen(
          caches: const [
            CacheSummary(
              tripId: 1,
              tripName: 'Meghalaya',
              poiCount: 42,
              routedLegCount: 3,
              legCount: 4,
              weatherDayCount: 15,
            ),
          ],
        ),
      );
      await tester.pump();

      expect(
        find.text('42 places · 3/4 legs routed · 15 forecast days'),
        findsOneWidget,
      );
      expect(find.text('CLEAR'), findsOneWidget);
    });

    testWidgets('a trip with nothing downloaded offers no clear', (
      tester,
    ) async {
      await tester.pumpWidget(
        screen(
          caches: const [
            CacheSummary(
              tripId: 1,
              tripName: 'Ladakh',
              poiCount: 0,
              routedLegCount: 0,
              legCount: 2,
              weatherDayCount: 0,
            ),
          ],
        ),
      );
      await tester.pump();

      expect(find.text('Nothing downloaded'), findsOneWidget);
      expect(find.text('CLEAR'), findsNothing);
    });

    testWidgets('CLEARING SAYS WHAT SURVIVES IT', (tester) async {
      useTallSurface(tester);
      await tester.pumpWidget(
        screen(
          caches: const [
            CacheSummary(
              tripId: 1,
              tripName: 'Meghalaya',
              poiCount: 42,
              routedLegCount: 3,
              legCount: 4,
              weatherDayCount: 15,
            ),
          ],
        ),
      );
      await tester.pump();

      // The distinction is the whole point: their work stays, the downloads go.
      expect(find.textContaining('are yours and stay put'), findsOneWidget);

      await tester.tap(find.text('CLEAR'));
      await tester.pumpAndSettle();
      // The row behind the dialog also says "42 places", so match the
      // dialog's own wording.
      expect(find.textContaining('Removes 42 places'), findsOneWidget);
      expect(find.textContaining('need WiFi to get them back'), findsOneWidget);
    });

    testWidgets('call history says a copy counts', (tester) async {
      await tester.pumpWidget(screen());
      await tester.pump();
      expect(find.textContaining('Copies count too'), findsOneWidget);
    });
  });

  group('call history', () {
    testWidgets('reads back with the action in words', (tester) async {
      await tester.pumpWidget(
        wrap(
          CallHistoryScreen(
            history: Stream.value([
              CallLogEntry(
                id: 1,
                action: 'copy',
                occurredAt: DateTime(2026, 10, 3, 14, 5),
                contactName: 'Rina Kharkongor',
                phoneRaw: '+91 90000 00001',
              ),
              CallLogEntry(
                id: 2,
                action: 'dialer',
                occurredAt: DateTime(2026, 10, 3, 14, 6),
                contactName: 'Rina Kharkongor',
                phoneRaw: '+91 90000 00001',
              ),
            ]),
          ),
        ),
      );
      await tester.pump();

      expect(find.text('COPIED'), findsOneWidget);
      expect(find.text('OPENED THE DIALER'), findsOneWidget);
      expect(find.text('03/10  14:05'), findsOneWidget);
    });

    testWidgets('empty explains that copies will fill it', (tester) async {
      await tester.pumpWidget(
        wrap(CallHistoryScreen(history: Stream.value(const []))),
      );
      await tester.pump();
      expect(find.textContaining('Copying a number counts'), findsOneWidget);
    });
  });
}
