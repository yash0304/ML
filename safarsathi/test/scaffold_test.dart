// Scaffold checks for issue #1. Cheap, and they pin the things most likely to
// break silently: the palettes, the type roles, and the no-shadow rule.

import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:safarsathi/core/theme/app_tokens.dart';

/// WCAG 2.1 relative luminance.
double _luminance(Color c) {
  double channel(double v) =>
      v <= 0.03928 ? v / 12.92 : math.pow((v + 0.055) / 1.055, 2.4).toDouble();

  return 0.2126 * channel(c.r) + 0.7152 * channel(c.g) + 0.0722 * channel(c.b);
}

/// WCAG contrast ratio between two opaque colours.
double contrast(Color a, Color b) {
  final la = _luminance(a);
  final lb = _luminance(b);
  final hi = math.max(la, lb);
  final lo = math.min(la, lb);
  return (hi + 0.05) / (lo + 0.05);
}

void main() {
  // The screen-level tests live in diary_screen_test.dart, driven by plain
  // streams. They cannot live here: constructing the real AppDatabase inside
  // a widget test means closing it, and close() awaits work the fake clock
  // never advances — the test hangs until the runner is killed.

  test('both palettes are distinct and complete', () {
    expect(AppColors.day.paper, isNot(AppColors.night.paper));
    expect(AppColors.day.ink, isNot(AppColors.night.ink));
    // Red is rationed to the emergency surface, so it must stay distinct from
    // signal in both themes or the rationing means nothing.
    expect(AppColors.day.emergency, isNot(AppColors.day.signal));
    expect(AppColors.night.emergency, isNot(AppColors.night.signal));
  });

  test('caution splits into a text token and a graphics token by day', () {
    // The brighter mark fails AA as text on warm paper, which is exactly why
    // it may only be used for the 7px dot and the hazard stripe.
    expect(AppColors.day.caution, isNot(AppColors.day.cautionMark));
    // At night the dark ground carries it and no split is needed.
    expect(AppColors.night.caution, AppColors.night.cautionMark);
  });

  test('text colours meet WCAG AA on their own ground', () {
    for (final c in [AppColors.day, AppColors.night]) {
      expect(contrast(c.ink, c.paper), greaterThanOrEqualTo(4.5));
      expect(contrast(c.muted, c.paper), greaterThanOrEqualTo(4.5));
      expect(contrast(c.signal, c.paper), greaterThanOrEqualTo(4.5));
      expect(contrast(c.caution, c.paper), greaterThanOrEqualTo(4.5));
      expect(contrast(c.emergency, c.paper), greaterThanOrEqualTo(4.5));
    }
  });

  test('secondary text also meets AA on the raised surface', () {
    // muted carries phone numbers, captions and provenance, and it sits on
    // `stone` in chips, the search field and category avatars. Checking it
    // only against `paper` is how the first version of this palette shipped
    // a value that failed.
    for (final c in [AppColors.day, AppColors.night]) {
      expect(contrast(c.muted, c.stone), greaterThanOrEqualTo(4.5));
      expect(contrast(c.ink, c.stone), greaterThanOrEqualTo(4.5));
      expect(contrast(c.signal, c.stone), greaterThanOrEqualTo(4.5));
    }
  });

  test('cautionMark clears the 3:1 floor for non-text graphics', () {
    for (final c in [AppColors.day, AppColors.night]) {
      expect(contrast(c.cautionMark, c.paper), greaterThanOrEqualTo(3.0));
    }
  });

  test('every numeric style uses tabular figures', () {
    for (final style in [
      AppTokens.numberStyle,
      AppTokens.milestoneStyle,
      AppTokens.badgeStyle,
    ]) {
      expect(
        style.fontFeatures?.any((f) => f.feature == 'tnum'),
        isTrue,
        reason: 'digits must align down a column',
      );
    }
  });

  test('every variable-font style pins its weight axis', () {
    // Jost and Archivo are variable. Without an explicit wght axis the whole
    // app silently renders at one weight. Courier Prime is static and ships
    // two real weight files, so it is exempt — and must stay exempt, or a
    // variation would be set on a face that has no axis to move.
    for (final style in [
      AppTokens.stencilStyle,
      AppTokens.milestoneStyle,
      AppTokens.stampStyle,
      AppTokens.titleStyle,
      AppTokens.rowTitleStyle,
      AppTokens.captionStyle,
    ]) {
      final axes = style.fontVariations ?? const <FontVariation>[];
      expect(
        axes.any((v) => v.axis == 'wght'),
        isTrue,
        reason: 'variable fonts need an explicit wght axis',
      );
    }
  });

  test('the number styles are the typewriter, and carry no axis', () {
    for (final style in [AppTokens.numberStyle, AppTokens.badgeStyle]) {
      expect(style.fontFamily, 'CourierPrime');
      expect(style.fontVariations ?? const [], isEmpty);
      expect(style.fontWeight, isNotNull);
    }
  });

  test('words and signage use their own faces', () {
    expect(AppTokens.titleStyle.fontFamily, 'Archivo');
    expect(AppTokens.rowTitleStyle.fontFamily, 'Archivo');
    expect(AppTokens.captionStyle.fontFamily, 'Archivo');
    expect(AppTokens.stencilStyle.fontFamily, 'Jost');
    expect(AppTokens.milestoneStyle.fontFamily, 'Jost');
    expect(AppTokens.stampStyle.fontFamily, 'Jost');
  });

  test('no theme paints a drop shadow', () {
    for (final theme in [AppTokens.light, AppTokens.dark]) {
      expect(theme.appBarTheme.elevation, 0);
      expect(theme.cardTheme.elevation, 0);
      expect(theme.floatingActionButtonTheme.elevation, 0);
      expect(theme.bottomSheetTheme.elevation, 0);
    }
  });
}
