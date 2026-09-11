// lib/core/theme/app_tokens.dart  —  v2 "Milestone"
//
// Single source of truth for colour, type and geometry. No hard-coded values
// in screens. Supersedes app_tokens_v1_superseded.dart.
//
// PALETTE RATIONALE (see DESIGN_VISUAL_v2.md §2):
// Anchored on Indian highway milestone markers — warm stone, coloured cap,
// stencilled ink. Hue already carries meaning on the road:
//   green  = go / confirmed / official
//   amber  = caution / unverified
//   red    = emergency, and nothing else
// Red appears ONLY on the emergency tab. If it starts showing up elsewhere,
// it stops meaning emergency.
//
// TWO THEMES. Night is functional, not cosmetic — this app's defining use is
// a village at 11pm with no signal.
//
// Compiled and analysed clean as of issue #1.

import 'package:flutter/material.dart';

// ---------------------------------------------------------------------------
// COLOUR
// ---------------------------------------------------------------------------

@immutable
class AppColors extends ThemeExtension<AppColors> {
  final Color paper; // ground
  final Color stone; // raised: chips, avatars, search, ticket body
  final Color rule; // hairlines, perforations, borders
  final Color ink; // primary text
  final Color muted; // secondary text, numbers, inactive icons

  final Color signal; // official, confirmed, primary action
  final Color signalSoft; // selected chip, stamp ink bed

  /// Caution TEXT and icons. Meets 4.5:1 on paper.
  final Color caution;

  /// Caution GRAPHICS only — the 7px trust dot and the hazard stripe.
  /// Brighter than [caution] so it reads amber at seven pixels; held to the
  /// 3:1 non-text floor, which is why it must never carry text.
  final Color cautionMark;
  final Color cautionSoft; // readiness banner bed

  final Color emergency; // emergency tab ONLY
  final Color emergencySoft;

  /// Grain is felt, not seen. 3% by day, 2% at night.
  final double grainOpacity;

  const AppColors({
    required this.paper,
    required this.stone,
    required this.rule,
    required this.ink,
    required this.muted,
    required this.signal,
    required this.signalSoft,
    required this.caution,
    required this.cautionMark,
    required this.cautionSoft,
    required this.emergency,
    required this.emergencySoft,
    required this.grainOpacity,
  });

  /// Day — "Paper". Contrast on `paper`: ink 15.0:1, muted 4.9:1,
  /// signal 6.6:1, caution 4.7:1, emergency 6.2:1, cautionMark 3.1:1.
  static const day = AppColors(
    paper: Color(0xFFFAF7F0),
    stone: Color(0xFFEFEAE0),
    rule: Color(0xFFD8D1C2),
    ink: Color(0xFF191E1A),
    muted: Color(0xFF6B7670),
    signal: Color(0xFF1F6B4A),
    signalSoft: Color(0xFFDCEBE3),
    caution: Color(0xFFA35F10),
    cautionMark: Color(0xFFC77B1E),
    cautionSoft: Color(0xFFF6EBD6),
    emergency: Color(0xFFB32B23),
    emergencySoft: Color(0xFFF7DEDB),
    grainOpacity: 0.03,
  );

  /// Night — "Lamp". Warm charcoal, never neutral black: the warmth is what
  /// keeps it the same app after dark.
  static const night = AppColors(
    paper: Color(0xFF14120E),
    stone: Color(0xFF1F1C17),
    rule: Color(0xFF332E26),
    ink: Color(0xFFEDE6D7),
    muted: Color(0xFF9A9287),
    signal: Color(0xFF4FBF8B),
    signalSoft: Color(0xFF17301F),
    caution: Color(0xFFE0A040),
    cautionMark: Color(0xFFE0A040), // the dark ground carries it; no split needed
    cautionSoft: Color(0xFF33260F),
    emergency: Color(0xFFFF6B5E),
    emergencySoft: Color(0xFF3A1512),
    grainOpacity: 0.02,
  );

  @override
  AppColors copyWith({
    Color? paper,
    Color? stone,
    Color? rule,
    Color? ink,
    Color? muted,
    Color? signal,
    Color? signalSoft,
    Color? caution,
    Color? cautionMark,
    Color? cautionSoft,
    Color? emergency,
    Color? emergencySoft,
    double? grainOpacity,
  }) {
    return AppColors(
      paper: paper ?? this.paper,
      stone: stone ?? this.stone,
      rule: rule ?? this.rule,
      ink: ink ?? this.ink,
      muted: muted ?? this.muted,
      signal: signal ?? this.signal,
      signalSoft: signalSoft ?? this.signalSoft,
      caution: caution ?? this.caution,
      cautionMark: cautionMark ?? this.cautionMark,
      cautionSoft: cautionSoft ?? this.cautionSoft,
      emergency: emergency ?? this.emergency,
      emergencySoft: emergencySoft ?? this.emergencySoft,
      grainOpacity: grainOpacity ?? this.grainOpacity,
    );
  }

  @override
  AppColors lerp(ThemeExtension<AppColors>? other, double t) {
    if (other is! AppColors) return this;
    Color c(Color a, Color b) => Color.lerp(a, b, t)!;
    return AppColors(
      paper: c(paper, other.paper),
      stone: c(stone, other.stone),
      rule: c(rule, other.rule),
      ink: c(ink, other.ink),
      muted: c(muted, other.muted),
      signal: c(signal, other.signal),
      signalSoft: c(signalSoft, other.signalSoft),
      caution: c(caution, other.caution),
      cautionMark: c(cautionMark, other.cautionMark),
      cautionSoft: c(cautionSoft, other.cautionSoft),
      emergency: c(emergency, other.emergency),
      emergencySoft: c(emergencySoft, other.emergencySoft),
      grainOpacity: grainOpacity + (other.grainOpacity - grainOpacity) * t,
    );
  }
}

// ---------------------------------------------------------------------------
// TYPE, GEOMETRY, THEME
// ---------------------------------------------------------------------------

class AppTokens {
  AppTokens._();

  /// Colours for the current theme. Screens read every colour through this.
  static AppColors of(BuildContext context) =>
      Theme.of(context).extension<AppColors>() ?? AppColors.day;

  // --- Families ---
  // ARCHIVO NARROW SPEAKS. INTER READS.
  // Anything read as a sentence is Inter. Anything stencilled onto an object
  // — a label, a marker numeral, a stamp — is Archivo Narrow. See §3.1.
  static const _body = 'Inter';
  static const _stencil = 'ArchivoNarrow';

  // Both families are bundled as VARIABLE fonts — Google Fonts no longer
  // publishes static instances for either. `fontWeight` alone does not
  // reliably move the 'wght' axis, so every style below sets the axis
  // explicitly as well. Drop the fontVariations and the type silently
  // renders at a single weight.
  static const _w400 = <FontVariation>[FontVariation('wght', 400)];
  static const _w500 = <FontVariation>[FontVariation('wght', 500)];
  static const _w600 = <FontVariation>[FontVariation('wght', 600)];
  static const _w700 = <FontVariation>[FontVariation('wght', 700)];

  // Phone numbers, distances and amounts get tabular figures so digits align
  // down the column and can be read at a glance rather than parsed. It is
  // also what lets RollingDigits animate without the row width jittering.
  static const _tabular = <FontFeature>[FontFeature.tabularFigures()];

  // --- Stencil roles (Archivo Narrow) ---

  /// Section headers, field labels, tab labels.
  static const stencilStyle = TextStyle(
    fontFamily: _stencil,
    fontSize: 13,
    fontWeight: FontWeight.w700,
    fontVariations: _w700,
    letterSpacing: 1.2,
    height: 1.0,
  );

  /// The km numeral on a milestone marker.
  static const milestoneStyle = TextStyle(
    fontFamily: _stencil,
    fontSize: 28,
    fontWeight: FontWeight.w700,
    fontVariations: _w700,
    height: 1.0,
    fontFeatures: _tabular,
  );

  /// Stamp marks: CONFIRMED, CACHED, IMPORTED, ROLLED BACK.
  static const stampStyle = TextStyle(
    fontFamily: _stencil,
    fontSize: 11,
    fontWeight: FontWeight.w700,
    fontVariations: _w700,
    letterSpacing: 1.6,
    height: 1.0,
  );

  /// Emergency number badges.
  static const badgeStyle = TextStyle(
    fontFamily: _stencil,
    fontSize: 14,
    fontWeight: FontWeight.w700,
    fontVariations: _w700,
    height: 1.0,
    fontFeatures: _tabular,
  );

  // --- Body roles (Inter) ---

  static const titleStyle = TextStyle(
    fontFamily: _body,
    fontSize: 16,
    fontWeight: FontWeight.w600,
    fontVariations: _w600,
    height: 1.3,
  );

  static const rowTitleStyle = TextStyle(
    fontFamily: _body,
    fontSize: 15,
    fontWeight: FontWeight.w500,
    fontVariations: _w500,
    height: 1.3,
  );

  static const numberStyle = TextStyle(
    fontFamily: _body,
    fontSize: 14,
    fontWeight: FontWeight.w400,
    fontVariations: _w400,
    fontFeatures: _tabular,
  );

  static const captionStyle = TextStyle(
    fontFamily: _body,
    fontSize: 12.5,
    fontWeight: FontWeight.w400,
    fontVariations: _w400,
    height: 1.35,
  );

  // --- Geometry ---
  // Paper is cut, not rounded. Sharp on anything ticket- or stamp-shaped.
  static const radiusSharp = 0.0;
  static const radiusSoft = 4.0;
  static const hairline = 1.0;
  static const gutter = 16.0;
  static const tapTarget = 48.0;

  // Spacing scale. Use these, not arbitrary numbers.
  static const s4 = 4.0;
  static const s8 = 8.0;
  static const s12 = 12.0;
  static const s16 = 16.0;
  static const s24 = 24.0;
  static const s32 = 32.0;

  // Perforation geometry, shared by PerforationPainter.
  static const perfRadius = 4.0;
  static const perfPitch = 10.0;

  // --- ThemeData ---

  static ThemeData theme(AppColors c) {
    final isDark = c == AppColors.night;
    final base = isDark ? ThemeData.dark() : ThemeData.light();

    return base.copyWith(
      extensions: [c],
      scaffoldBackgroundColor: c.paper,
      canvasColor: c.paper,
      dividerColor: c.rule,
      dividerTheme: DividerThemeData(
        color: c.rule,
        thickness: hairline,
        space: hairline,
      ),
      colorScheme: base.colorScheme.copyWith(
        surface: c.paper,
        primary: c.signal,
        onPrimary: c.paper,
        error: c.emergency,
        onSurface: c.ink,
      ),
      textTheme: base.textTheme.apply(
        fontFamily: _body,
        bodyColor: c.ink,
        displayColor: c.ink,
      ),
      // THERE ARE NO DROP SHADOWS IN THIS APP. Print does not have them.
      // Separation is a 1px rule and a ground shift; the two elements that
      // genuinely float take a 1px ink border instead. See §2.3.
      appBarTheme: AppBarTheme(
        backgroundColor: c.paper,
        foregroundColor: c.ink,
        elevation: 0,
        scrolledUnderElevation: 0,
        titleSpacing: gutter,
      ),
      cardTheme: CardThemeData(
        color: c.stone,
        elevation: 0,
        margin: EdgeInsets.zero,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(radiusSharp),
          side: BorderSide(color: c.rule, width: hairline),
        ),
      ),
      floatingActionButtonTheme: FloatingActionButtonThemeData(
        backgroundColor: c.signal,
        foregroundColor: c.paper,
        elevation: 0,
        highlightElevation: 0,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(radiusSoft),
          side: BorderSide(color: c.ink, width: hairline),
        ),
      ),
      bottomSheetTheme: BottomSheetThemeData(
        backgroundColor: c.paper,
        elevation: 0,
        modalElevation: 0,
        shape: RoundedRectangleBorder(
          borderRadius: const BorderRadius.vertical(
            top: Radius.circular(radiusSoft),
          ),
          side: BorderSide(color: c.ink, width: hairline),
        ),
      ),
      chipTheme: base.chipTheme.copyWith(
        backgroundColor: c.stone,
        selectedColor: c.signalSoft,
        side: BorderSide(color: c.rule, width: hairline),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(radiusSoft),
        ),
        showCheckmark: false,
      ),
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: c.stone,
        isDense: true,
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(radiusSoft),
          borderSide: BorderSide(color: c.rule, width: hairline),
        ),
      ),
      splashFactory: NoSplash.splashFactory, // press scale + haptic instead
    );
  }

  static ThemeData get light => theme(AppColors.day);
  static ThemeData get dark => theme(AppColors.night);
}
