// test/retro_primitives_test.dart — issues #41 and #42.
//
// Both were built as the screens needed them rather than up front, so both
// arrive here with the code already written and only the verification the
// backlog asked for missing. Two things it names specifically: that
// reduce-motion zeroes the animations and leaves the haptics working, and
// that the perforation path survives three widths.

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:safarsathi/core/theme/app_tokens.dart';
import 'package:safarsathi/core/theme/motion.dart';
import 'package:safarsathi/core/widgets/retro.dart';

Widget wrap(Widget child, {bool reduceMotion = false}) => MaterialApp(
  theme: AppTokens.light,
  home: MediaQuery(
    data: MediaQueryData(disableAnimations: reduceMotion),
    child: Scaffold(body: child),
  ),
);

void main() {
  group('#41 — reduce motion', () {
    testWidgets('collapses every ordinary duration to zero', (tester) async {
      late BuildContext ctx;
      await tester.pumpWidget(
        wrap(
          Builder(
            builder: (context) {
              ctx = context;
              return const SizedBox();
            },
          ),
          reduceMotion: true,
        ),
      );

      expect(Motion.reduced(ctx), isTrue);
      for (final d in [Motion.instant, Motion.quick, Motion.base, Motion.sheet]) {
        expect(Motion.d(ctx, d), Duration.zero);
      }
    });

    testWidgets('leaves the durations alone when it is off', (tester) async {
      late BuildContext ctx;
      await tester.pumpWidget(
        wrap(
          Builder(
            builder: (context) {
              ctx = context;
              return const SizedBox();
            },
          ),
        ),
      );

      expect(Motion.reduced(ctx), isFalse);
      expect(Motion.d(ctx, Motion.base), Motion.base);
      expect(Motion.stampDuration(ctx), Motion.stamp);
    });

    testWidgets('THE STAMP SHORTENS RATHER THAN VANISHING', (tester) async {
      // The one exception. Reduce-motion must not take away the only visible
      // acknowledgement that a number has been confirmed.
      late BuildContext ctx;
      await tester.pumpWidget(
        wrap(
          Builder(
            builder: (context) {
              ctx = context;
              return const SizedBox();
            },
          ),
          reduceMotion: true,
        ),
      );

      expect(Motion.stampDuration(ctx), Motion.quick);
      expect(Motion.stampDuration(ctx), isNot(Duration.zero));
    });

    testWidgets('HAPTICS ARE NOT ANIMATION AND ARE NOT SUPPRESSED', (
      tester,
    ) async {
      final calls = <MethodCall>[];
      tester.binding.defaultBinaryMessenger.setMockMethodCallHandler(
        SystemChannels.platform,
        (call) async {
          calls.add(call);
          return null;
        },
      );
      addTearDown(
        () => tester.binding.defaultBinaryMessenger.setMockMethodCallHandler(
          SystemChannels.platform,
          null,
        ),
      );

      var tapped = 0;
      await tester.pumpWidget(
        wrap(
          PressScale(onTap: () => tapped++, child: const Text('Tap me')),
          reduceMotion: true,
        ),
      );

      await tester.tap(find.text('Tap me'));
      await tester.pumpAndSettle();

      expect(tapped, 1);
      // Somebody who has turned animation off still needs to feel the press.
      expect(
        calls.where((c) => c.method == 'HapticFeedback.vibrate'),
        isNotEmpty,
      );
    });
  });

  group('#42 — the perforation path', () {
    /// The clipper is private, so the path is exercised through the widget
    /// that owns it: if the geometry threw or produced nothing, the card
    /// would not lay out.
    Future<void> ticketAt(
      WidgetTester tester,
      double width,
      TicketEdge edge,
    ) async {
      // The default 800px test surface is narrower than the widest case, and
      // an overflow there would fail for the wrong reason.
      tester.view.physicalSize = Size(width + 40, 400);
      tester.view.devicePixelRatio = 1.0;
      addTearDown(tester.view.reset);

      return tester.pumpWidget(
        wrap(
          Center(
            child: SizedBox(
              width: width,
              child: TicketCard(
                edge: edge,
                child: const SizedBox(height: 40, child: Text('Ticket')),
              ),
            ),
          ),
        ),
      );
    }

    // A narrow phone, a normal one, and a tablet-ish width. The loop that
    // walks the notches steps by `perfPitch` and must terminate cleanly when
    // the card is narrower than one whole notch.
    for (final width in const [72.0, 360.0, 900.0]) {
      testWidgets('lays out at ${width.round()}px on both edges', (
        tester,
      ) async {
        await ticketAt(tester, width, TicketEdge.both);
        expect(tester.takeException(), isNull);
        expect(find.text('Ticket'), findsOneWidget);
        expect(tester.getSize(find.byType(TicketCard)).width, width);
      });
    }

    testWidgets('a card narrower than one notch still renders', (tester) async {
      // `pitch / 2 + r <= width` is false here, so the notch loop runs zero
      // times and the path has to close as a plain rectangle rather than
      // throwing or collapsing.
      await ticketAt(tester, 8, TicketEdge.both);
      expect(tester.takeException(), isNull);
      expect(find.byType(TicketCard), findsOneWidget);
    });

    testWidgets('every edge option renders', (tester) async {
      for (final edge in TicketEdge.values) {
        await ticketAt(tester, 320, edge);
        expect(tester.takeException(), isNull, reason: '$edge threw');
      }
    });
  });

  group('#42 — the primitives exist and paint in both themes', () {
    for (final theme in const [Brightness.light, Brightness.dark]) {
      testWidgets('all seven render in ${theme.name}', (tester) async {
        await tester.pumpWidget(
          MaterialApp(
            theme: theme == Brightness.light
                ? AppTokens.light
                : AppTokens.dark,
            home: const Scaffold(
              body: SingleChildScrollView(
                child: Column(
                  children: [
                    StencilLabel('Stencil'),
                    TicketCard(edge: TicketEdge.both, child: Text('Ticket')),
                    MilestoneMarker(
                      numeral: '14',
                      unit: 'KM',
                      place: 'CHERRAPUNJI',
                    ),
                    StampBadge(label: 'Stamp'),
                    SizedBox(height: 20, child: HazardStripe()),
                    SizedBox(
                      height: 40,
                      child: GrainOverlay(child: SizedBox.expand()),
                    ),
                    RollingDigits(value: '7', style: AppTokens.numberStyle),
                  ],
                ),
              ),
            ),
          ),
        );
        await tester.pumpAndSettle();

        expect(tester.takeException(), isNull);
        expect(find.byType(StencilLabel), findsOneWidget);
        expect(find.byType(TicketCard), findsOneWidget);
        expect(find.byType(MilestoneMarker), findsOneWidget);
        expect(find.byType(StampBadge), findsOneWidget);
        expect(find.byType(HazardStripe), findsOneWidget);
        expect(find.byType(GrainOverlay), findsOneWidget);
        expect(find.byType(RollingDigits), findsOneWidget);
      });
    }

    test('THE NIGHT PALETTE DROPS THE GRAIN, per #47', () {
      expect(AppColors.day.grainOpacity, 0.03);
      expect(AppColors.night.grainOpacity, 0.02);
    });
  });
}
