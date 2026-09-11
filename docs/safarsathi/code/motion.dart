// lib/core/theme/motion.dart  —  v2 "Milestone"
//
// THE GOVERNING CONSTRAINT: nothing animates at rest.
// If the user is not touching the screen and no state has just changed, zero
// frames are scheduled. This app runs for days between charges with no signal;
// an idle animation is a battery cost with no user attached to it.
//
// That rules out ambient motion, looping texture, animated map layers and
// parallax. What is left — and what actually makes an app feel alive — is
// response: every touch answers immediately, in the hand as well as on screen.
//
// See DESIGN_VISUAL_v2.md §5. Never compiled.

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

class Motion {
  Motion._();

  static const instant = Duration(milliseconds: 90); // press-in
  static const quick = Duration(milliseconds: 140); // chip swap, dot clear
  static const base = Duration(milliseconds: 220); // route, reorder
  static const sheet = Duration(milliseconds: 280); // modal sheet
  static const stamp = Duration(milliseconds: 380); // the confirmation stamp

  static const standard = Curves.easeOutCubic;

  static const spring = SpringDescription(
    mass: 1,
    stiffness: 380,
    damping: 26,
  );

  static const pressScale = 0.97;

  /// True when the platform asks for reduced motion.
  static bool reduced(BuildContext context) =>
      MediaQuery.of(context).disableAnimations;

  /// Collapses a duration to zero under reduce-motion.
  ///
  /// The one exception in the app is the confirmation stamp, which becomes a
  /// [quick] cross-fade rather than vanishing — the user must still get a
  /// visible confirmation. Call [stampDuration] for that case.
  static Duration d(BuildContext context, Duration value) =>
      reduced(context) ? Duration.zero : value;

  static Duration stampDuration(BuildContext context) =>
      reduced(context) ? quick : stamp;
}

/// Fixed haptic vocabulary. Small enough to stay meaningful.
///
/// Never fires on scroll, never on a stream rebuild, never when data merely
/// arrives. Haptics are NOT suppressed by reduce-motion — they are feedback,
/// not animation.
class Haptics {
  Haptics._();

  /// Chip, tab or filter change.
  static void select() => HapticFeedback.selectionClick();

  /// Row press-in; sheet open; pin toggle; swipe threshold crossed.
  static void light() => HapticFeedback.lightImpact();

  /// The confirmation stamp landing; import commit.
  static void confirm() => HapticFeedback.mediumImpact();

  /// Dialing an emergency number; destructive confirm (rollback, delete).
  static void grave() => HapticFeedback.heavyImpact();

  /// Validation error on an import row. Two light taps, 80ms apart.
  static Future<void> reject() async {
    await HapticFeedback.lightImpact();
    await Future<void>.delayed(const Duration(milliseconds: 80));
    await HapticFeedback.lightImpact();
  }
}

/// Wraps every tappable in the app. Scale 0.97 in over [Motion.instant],
/// back over [Motion.quick], with a haptic on press-in.
///
/// Material ink splashes are disabled app-wide (`NoSplash.splashFactory`);
/// this is what replaces them. A splash spreads outward from a finger; a
/// press-scale reads as paper being pushed, which is the right metaphor here.
class PressScale extends StatefulWidget {
  final Widget child;
  final VoidCallback? onTap;
  final VoidCallback? onLongPress;

  /// Haptic fired on press-in. Defaults to [Haptics.light].
  /// Emergency dial rows pass [Haptics.grave].
  final VoidCallback? feedback;

  const PressScale({
    super.key,
    required this.child,
    this.onTap,
    this.onLongPress,
    this.feedback,
  });

  @override
  State<PressScale> createState() => _PressScaleState();
}

class _PressScaleState extends State<PressScale> {
  bool _down = false;

  void _set(bool down) {
    if (_down == down) return;
    setState(() => _down = down);
    if (down) (widget.feedback ?? Haptics.light)();
  }

  @override
  Widget build(BuildContext context) {
    final enabled = widget.onTap != null || widget.onLongPress != null;
    return GestureDetector(
      behavior: HitTestBehavior.opaque,
      onTapDown: enabled ? (_) => _set(true) : null,
      onTapUp: enabled ? (_) => _set(false) : null,
      onTapCancel: enabled ? () => _set(false) : null,
      onTap: widget.onTap,
      onLongPress: widget.onLongPress,
      child: AnimatedScale(
        scale: _down ? Motion.pressScale : 1.0,
        duration: Motion.d(context, _down ? Motion.instant : Motion.quick),
        curve: Motion.standard,
        child: widget.child,
      ),
    );
  }
}
