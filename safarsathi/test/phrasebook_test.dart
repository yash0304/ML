// test/phrasebook_test.dart — #40, a few phrases offline.

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:safarsathi/core/theme/app_tokens.dart';
import 'package:safarsathi/features/phrasebook/data/phrases.dart';
import 'package:safarsathi/features/phrasebook/presentation/phrasebook_screen.dart';

void main() {
  group('the phrases', () {
    test('a trip\'s languages come first, the rest keep their order', () {
      expect(languagesFor({'IN'}).take(2).map((l) => l.code), ['hi', 'kha']);
      expect(languagesFor({'AT'}).first.code, 'de');
      expect(languagesFor({'CH'}).take(3).map((l) => l.code), [
        'de',
        'fr',
        'it',
      ]);
      expect(languagesFor({}).map((l) => l.code), [
        for (final l in languages) l.code,
      ]);
      expect(languagesFor({'IN', 'BD'}).length, languages.length);
    });

    test('codes are unique and every language has phrases', () {
      final codes = {for (final l in languages) l.code};
      expect(codes.length, languages.length);
      for (final l in languages) {
        expect(l.phraseCount, greaterThan(0), reason: l.name);
        for (final g in l.groups) {
          for (final p in g.phrases) {
            expect(p.english.trim(), isNotEmpty);
            expect(p.text.trim(), isNotEmpty);
          }
        }
      }
    });

    test('scripts a reader may not read carry how to say them', () {
      for (final code in ['hi', 'bn']) {
        final l = languages.firstWhere((l) => l.code == code);
        for (final g in l.groups) {
          for (final p in g.phrases) {
            expect(p.say, isNotNull, reason: '${l.name}: ${p.english}');
          }
        }
      }
    });

    test('no emergency numbers: they belong to the SOS tab and its sources',
        () {
      final digits = RegExp(r'\d{3,}');
      for (final l in languages) {
        for (final g in l.groups) {
          for (final p in g.phrases) {
            expect(
              digits.hasMatch('${p.english} ${p.text} ${p.say} ${p.note}'),
              isFalse,
              reason: '${l.name}: ${p.english}',
            );
          }
        }
      }
    });

    test('every full language can ask for vegetarian food and for help', () {
      for (final l in languages.where((l) => l.code != 'kha')) {
        final english = [
          for (final g in l.groups) ...g.phrases.map((p) => p.english),
        ];
        expect(english, contains('I am vegetarian'), reason: l.name);
        expect(english, contains('No onion, no garlic'), reason: l.name);
        expect(english, contains('Help!'), reason: l.name);
      }
    });
  });

  group('the screen', () {
    Future<void> pump(WidgetTester t, Set<String> countries) async {
      t.view.physicalSize = const Size(400, 900);
      t.view.devicePixelRatio = 1;
      addTearDown(t.view.reset);
      await t.pumpWidget(
        MaterialApp(
          theme: AppTokens.light,
          home: PhrasebookScreen(countries: countries),
        ),
      );
    }

    testWidgets('opens on the trip\'s language and says it is unchecked',
        (t) async {
      await pump(t, {'IN'});
      expect(find.text('नमस्ते'), findsOneWidget);
      expect(find.text('Namaste'), findsOneWidget);
      expect(
        find.textContaining('not checked by a native speaker'),
        findsOneWidget,
      );
    });

    testWidgets('switching language shows its phrases and note', (t) async {
      await pump(t, {'IN'});
      await t.tap(find.byKey(const Key('lang-kha')));
      await t.pumpAndSettle();
      expect(find.text('Khublei'), findsOneWidget);
      expect(find.textContaining('reliably taught'), findsOneWidget);
      expect(find.text('नमस्ते'), findsNothing);
    });

    testWidgets('an Austrian trip opens on German', (t) async {
      await pump(t, {'AT'});
      expect(find.text('Hallo / Guten Tag'), findsOneWidget);
    });

    testWidgets('tapping a phrase shows it large', (t) async {
      await pump(t, {'IN'});
      await t.tap(find.byKey(const Key('phrase-Thank you')));
      await t.pumpAndSettle();
      final shown = t.widget<Text>(find.byKey(const Key('show-phrase')));
      expect(shown.data, 'धन्यवाद');
      expect(shown.style!.fontSize, greaterThan(40));
      expect(find.text('Dhanyavaad'), findsOneWidget);
    });

    testWidgets('dark theme draws too', (t) async {
      t.view.physicalSize = const Size(400, 900);
      t.view.devicePixelRatio = 1;
      addTearDown(t.view.reset);
      await t.pumpWidget(
        MaterialApp(
          theme: AppTokens.dark,
          home: const PhrasebookScreen(countries: {'FR'}),
        ),
      );
      expect(find.text('Bonjour'), findsOneWidget);
      expect(t.takeException(), isNull);
    });
  });
}
