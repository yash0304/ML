// lib/core/theme/app_tokens.dart
//
// Single source of truth for colour and type. No hard-coded values in screens.
//
// PALETTE RATIONALE (see DESIGN.md):
// Anchored on Indian highway milestone markers — white stone, coloured cap.
// That gives a colour system where hue already carries meaning on the road:
//   green  = go / confirmed / official
//   amber  = caution / unverified
//   red    = emergency, and nothing else
// Red appears ONLY on the emergency tab. If it starts showing up elsewhere,
// it stops meaning emergency.

import 'package:flutter/material.dart';

class AppTokens {
  AppTokens._();

  // --- Surfaces ---
  static const surface = Color(0xFFFFFFFF);
  static const surfaceRaised = Color(0xFFF1F4F1);

  // --- Ink ---
  static const ink = Color(0xFF1A211C);
  static const muted = Color(0xFF6B7670);

  // --- Signal: official, confirmed, primary action ---
  static const signal = Color(0xFF1F6B4A);
  static const signalSoft = Color(0xFFDCEBE3);

  // --- Caution: unverified, not yet confirmed ---
  static const caution = Color(0xFFC77B1E);
  static const cautionSoft = Color(0xFFFBF0DD);

  // --- Emergency: emergency tab ONLY ---
  static const emergency = Color(0xFFB32B23);
  static const emergencySoft = Color(0xFFFAE3E1);

  // --- Type ---
  // One family throughout. Phone numbers get tabular figures so digits
  // align down the column and can be read at a glance, not parsed.
  static const _family = 'Inter';

  static const titleStyle = TextStyle(
    fontFamily: _family,
    fontSize: 16,
    fontWeight: FontWeight.w600,
    color: ink,
    height: 1.3,
  );

  static const sectionStyle = TextStyle(
    fontFamily: _family,
    fontSize: 13,
    fontWeight: FontWeight.w600,
    color: muted,
    letterSpacing: 0.2,
  );

  static const rowTitleStyle = TextStyle(
    fontFamily: _family,
    fontSize: 15,
    fontWeight: FontWeight.w500,
    color: ink,
  );

  static const numberStyle = TextStyle(
    fontFamily: _family,
    fontSize: 14,
    fontWeight: FontWeight.w400,
    color: muted,
    fontFeatures: [FontFeature.tabularFigures()],
  );

  static const captionStyle = TextStyle(
    fontFamily: _family,
    fontSize: 12.5,
    fontWeight: FontWeight.w400,
    color: muted,
    height: 1.35,
  );
}
