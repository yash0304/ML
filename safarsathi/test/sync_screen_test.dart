// test/sync_screen_test.dart — issue #25, the screen.

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:safarsathi/core/theme/app_tokens.dart';
import 'package:safarsathi/features/sync/data/trip_sync.dart';
import 'package:safarsathi/features/sync/presentation/sync_screen.dart';

/// This screen is a long list and the failure section sits below a phone's
/// fold. A tall surface is less brittle than scrolling to each assertion; the
/// phone-width layout has its own test.
void useTallSurface(WidgetTester tester) {
  tester.view.physicalSize = const Size(420, 2000);
  tester.view.devicePixelRatio = 1.0;
  addTearDown(tester.view.reset);
}

Widget wrap(Widget child, {Brightness brightness = Brightness.light}) =>
    MaterialApp(
      theme: brightness == Brightness.light ? AppTokens.light : AppTokens.dark,
      home: child,
    );

const _shillongToSohra = SyncTask(
  kind: SyncKind.corridor,
  subject: 'Shillong → Cherrapunji',
  legId: 1,
);

TripSyncPlan planOf({
  int legs = 3,
  int stops = 4,
  bool tiles = true,
  int legsMissing = 0,
  int stopsMissing = 0,
}) => TripSyncPlan(
  legsWithoutCoordinates: legsMissing,
  stopsWithoutCoordinates: stopsMissing,
  tasks: [
    for (var i = 0; i < legs; i++)
      SyncTask(kind: SyncKind.corridor, subject: 'Leg $i', legId: i),
    for (var i = 0; i < stops; i++)
      SyncTask(kind: SyncKind.weather, subject: 'Stop $i', stopId: i),
    if (tiles) const SyncTask(kind: SyncKind.tiles, subject: 'whole trip'),
  ],
);

void main() {
  Widget screen({
    TripSyncPlan? plan,
    Stream<SyncProgress> Function()? run,
    String size = 'about 21 MB, nearly all of it map',
  }) => wrap(
    SyncScreen(
      plan: () async => plan ?? planOf(),
      run: run ?? () => const Stream<SyncProgress>.empty(),
      estimateSize: () async => size,
    ),
  );

  testWidgets('the plan is stated before anything runs', (tester) async {
    await tester.pumpWidget(screen());
    await tester.pumpAndSettle();

    expect(find.text('3 legs'), findsOneWidget);
    expect(find.text('4 stops'), findsOneWidget);
    expect(find.text('the whole corridor'), findsOneWidget);
    expect(find.textContaining('nearly all of it map'), findsOneWidget);
    expect(find.text('Download everything (8 steps)'), findsOneWidget);
  });

  testWidgets('one of a thing reads as singular', (tester) async {
    // "1 legs" on the screen people see last before leaving reads as
    // sloppiness, and a trip with one leg is common.
    await tester.pumpWidget(screen(plan: planOf(legs: 1, stops: 1)));
    await tester.pumpAndSettle();

    expect(find.text('1 leg'), findsOneWidget);
    expect(find.text('1 stop'), findsOneWidget);
  });

  testWidgets('NOTHING RUNS UNTIL THE BUTTON IS PRESSED', (tester) async {
    var started = false;
    await tester.pumpWidget(
      screen(
        run: () {
          started = true;
          return const Stream<SyncProgress>.empty();
        },
      ),
    );
    await tester.pumpAndSettle();
    expect(started, isFalse);
  });

  testWidgets('PROGRESS NAMES THE ITEM, not just a percentage', (
    tester,
  ) async {
    // "Route and what is along it · Shillong → Cherrapunji" is something a
    // person can wait through. "43%" is not.
    await tester.pumpWidget(
      screen(
        run: () => Stream.fromIterable(const [
          SyncProgress(done: 1, total: 8, current: _shillongToSohra),
        ]),
      ),
    );
    await tester.pumpAndSettle();

    await tester.tap(find.textContaining('Download everything'));
    await tester.pumpAndSettle();

    expect(find.text('1 of 8'), findsOneWidget);
    expect(
      find.textContaining('Shillong → Cherrapunji'),
      findsOneWidget,
    );
  });

  testWidgets('A FAILURE IS LISTED WITH ITS REASON', (tester) async {
    useTallSurface(tester);
    await tester.pumpWidget(
      screen(
        run: () => Stream.fromIterable(const [
          SyncProgress(
            done: 8,
            total: 8,
            failures: [
              SyncFailure(
                _shillongToSohra,
                'OpenStreetMap is busy right now. Try again in a minute.',
              ),
            ],
          ),
        ]),
      ),
    );
    await tester.pumpAndSettle();

    await tester.tap(find.textContaining('Download everything'));
    await tester.pumpAndSettle();

    expect(find.text('WHAT DID NOT WORK'), findsOneWidget);
    expect(find.textContaining('OpenStreetMap is busy'), findsOneWidget);
    expect(find.textContaining('retries only these'), findsOneWidget);
  });

  testWidgets('a failure never renders in emergency red', (tester) async {
    // A leg that did not download is a thing to retry, not a crisis.
    await tester.pumpWidget(
      screen(
        run: () => Stream.fromIterable(const [
          SyncProgress(
            done: 8,
            total: 8,
            failures: [SyncFailure(_shillongToSohra, 'busy')],
          ),
        ]),
      ),
    );
    await tester.pumpAndSettle();
    await tester.tap(find.textContaining('Download everything'));
    await tester.pumpAndSettle();

    final context = tester.element(find.byType(SyncScreen));
    final c = AppTokens.of(context);
    for (final text in tester.widgetList<Text>(find.byType(Text))) {
      expect(text.style?.color, isNot(c.emergency));
    }
  });

  testWidgets('a clean finish says you can turn the data off', (tester) async {
    await tester.pumpWidget(
      screen(
        run: () =>
            Stream.fromIterable(const [SyncProgress(done: 8, total: 8)]),
      ),
    );
    await tester.pumpAndSettle();

    await tester.tap(find.textContaining('Download everything'));
    await tester.pumpAndSettle();

    expect(find.textContaining('turn the data off'), findsOneWidget);
    expect(find.text('WHAT DID NOT WORK'), findsNothing);
  });

  testWidgets('SKIPPED LEGS AND STOPS ARE NAMED', (tester) async {
    await tester.pumpWidget(
      screen(plan: planOf(legsMissing: 2, stopsMissing: 1)),
    );
    await tester.pumpAndSettle();

    expect(
      find.textContaining('2 legs and 1 stop will be skipped'),
      findsOneWidget,
    );
  });

  testWidgets('a trip with no coordinates refuses and says why', (
    tester,
  ) async {
    await tester.pumpWidget(
      screen(
        plan: planOf(
          legs: 0,
          stops: 0,
          tiles: false,
          legsMissing: 3,
          stopsMissing: 4,
        ),
      ),
    );
    await tester.pumpAndSettle();

    expect(find.textContaining('find it on the map first'), findsOneWidget);
    expect(find.text('Nothing to download yet'), findsOneWidget);
  });

  testWidgets('an empty trip says that rather than blaming coordinates', (
    tester,
  ) async {
    await tester.pumpWidget(
      screen(plan: planOf(legs: 0, stops: 0, tiles: false)),
    );
    await tester.pumpAndSettle();
    expect(find.text('This trip has no stops yet.'), findsOneWidget);
  });

  testWidgets('it says the app stops asking the network afterwards', (
    tester,
  ) async {
    await tester.pumpWidget(screen());
    await tester.pumpAndSettle();
    expect(
      find.textContaining('never asks the network again'),
      findsOneWidget,
    );
  });

  testWidgets('lays out at phone width in both themes', (tester) async {
    tester.view.physicalSize = const Size(400, 800);
    tester.view.devicePixelRatio = 1.0;
    addTearDown(tester.view.reset);

    for (final b in Brightness.values) {
      await tester.pumpWidget(
        wrap(
          SyncScreen(
            plan: () async => planOf(legsMissing: 1),
            run: () => const Stream<SyncProgress>.empty(),
            estimateSize: () async => 'about 21 MB',
          ),
          brightness: b,
        ),
      );
      await tester.pumpAndSettle();
      expect(tester.takeException(), isNull);
    }
  });
}
