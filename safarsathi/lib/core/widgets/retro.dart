// lib/core/widgets/retro.dart  —  v2 "Milestone"
//
// The retro primitives. Every one of them is a geometric shape drawn in code:
// a notch, a perforation, a band, a hairline. No illustration, no photographic
// texture, no image assets beyond one 4 KB grain tile.
//
// THE EXEMPTION RULE (DESIGN_VISUAL_v2.md §0):
//   Ephemera may carry category, provenance, place and delight.
//   Ephemera may NEVER carry trust.
// Nothing in this file may be used on the emergency tab or as a trust marker.
// Trust is the amber dot and plain type, exactly as DESIGN.md §4 specifies.
//
// Compiled and analysed clean as of issue #1.

import 'dart:math' as math;
import 'package:flutter/material.dart';
import '../theme/app_tokens.dart';
import '../theme/motion.dart';

// ---------------------------------------------------------------------------
// STENCIL LABEL — every section header in the app
// ---------------------------------------------------------------------------

class StencilLabel extends StatelessWidget {
  final String text;

  /// Hairline rule running from the label to the right edge.
  final bool ruled;
  final Color? color;

  const StencilLabel(this.text, {super.key, this.ruled = true, this.color});

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    return Padding(
      padding: const EdgeInsets.fromLTRB(
        AppTokens.gutter,
        AppTokens.s24,
        AppTokens.gutter,
        AppTokens.s8,
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          Text(
            text.toUpperCase(),
            style: AppTokens.stencilStyle.copyWith(color: color ?? c.muted),
          ),
          if (ruled) ...[
            const SizedBox(width: AppTokens.s12),
            Expanded(
              child: Container(height: AppTokens.hairline, color: c.rule),
            ),
          ],
        ],
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// TICKET CARD — trips, legs, expenses, import batches
// ---------------------------------------------------------------------------

/// Which edges carry perforation notches.
enum TicketEdge { bottom, top, both, none }

/// A bus- or railway-ticket stub: perforated edge, hairline border, no shadow.
///
/// For objects the user considers ONE AT A TIME — a trip, a leg, an expense.
/// Never for a list scanned under stress: the dialer keeps its dense rows,
/// because a perforated contact list would be noise.
class TicketCard extends StatelessWidget {
  final Widget child;
  final TicketEdge edge;
  final EdgeInsets padding;
  final VoidCallback? onTap;

  const TicketCard({
    super.key,
    required this.child,
    this.edge = TicketEdge.bottom,
    this.padding = const EdgeInsets.all(AppTokens.s16),
    this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    final clipper = _TicketClipper(edge);

    return PressScale(
      onTap: onTap,
      child: ClipPath(
        clipper: clipper,
        child: CustomPaint(
          foregroundPainter: _TicketBorderPainter(edge, c.rule),
          child: Container(color: c.stone, padding: padding, child: child),
        ),
      ),
    );
  }
}

/// Outline of a ticket stub: a rectangle with semicircular notches bitten out
/// of the chosen edges. Shared by the clipper and the border painter so the
/// drawn outline always follows the clip exactly.
Path _ticketPath(Size size, TicketEdge edge) {
  const r = AppTokens.perfRadius;
  const pitch = AppTokens.perfPitch;

  final top = edge == TicketEdge.top || edge == TicketEdge.both;
  final bottom = edge == TicketEdge.bottom || edge == TicketEdge.both;
  final path = Path()..moveTo(0, 0);

  // Top edge, left to right. A clockwise arc between two points on the top
  // edge bulges downward, into the card, which is the notch we want.
  if (top) {
    for (var x = pitch / 2; x + r <= size.width; x += pitch) {
      path.lineTo(x - r, 0);
      path.arcToPoint(
        Offset(x + r, 0),
        radius: const Radius.circular(r),
        clockwise: true,
      );
    }
  }
  path.lineTo(size.width, 0);
  path.lineTo(size.width, size.height);

  // Bottom edge, right to left, so the same clockwise arc bulges upward.
  if (bottom) {
    for (var x = size.width - pitch / 2; x - r >= 0; x -= pitch) {
      path.lineTo(x + r, size.height);
      path.arcToPoint(
        Offset(x - r, size.height),
        radius: const Radius.circular(r),
        clockwise: true,
      );
    }
  }
  path.lineTo(0, size.height);
  path.close();
  return path;
}

class _TicketClipper extends CustomClipper<Path> {
  final TicketEdge edge;
  const _TicketClipper(this.edge);

  @override
  Path getClip(Size size) => _ticketPath(size, edge);

  @override
  bool shouldReclip(_TicketClipper old) => old.edge != edge;
}

class _TicketBorderPainter extends CustomPainter {
  final TicketEdge edge;
  final Color rule;
  const _TicketBorderPainter(this.edge, this.rule);

  @override
  void paint(Canvas canvas, Size size) {
    canvas.drawPath(
      _ticketPath(size, edge),
      Paint()
        ..style = PaintingStyle.stroke
        ..strokeWidth = AppTokens.hairline
        ..color = rule,
    );
  }

  @override
  bool shouldRepaint(_TicketBorderPainter old) =>
      old.edge != edge || old.rule != rule;
}

// ---------------------------------------------------------------------------
// MILESTONE MARKER — screen headers, route-discovery group headers
// ---------------------------------------------------------------------------

/// The kilometre stone: warm slab, domed top, coloured cap band, stencilled
/// numeral. The cap colour carries meaning like every other colour in this app
/// — [AppColors.signal] for a confirmed or cached leg, [AppColors.muted] for
/// one not yet synced. Never [AppColors.emergency].
class MilestoneMarker extends StatelessWidget {
  final String numeral; // "14"
  final String unit; // "KM"
  final String place; // "CHERRAPUNJI"
  final Color? capColor;

  const MilestoneMarker({
    super.key,
    required this.numeral,
    required this.place,
    this.unit = 'KM',
    this.capColor,
  });

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    final cap = capColor ?? c.signal;

    return Container(
      // Wide enough for a long place name set in Jost, which is a Futura
      // revival and not a condensed face — CHERRAPUNJI truncated at 92.
      width: 106,
      decoration: BoxDecoration(
        color: c.stone,
        border: Border.all(color: c.rule, width: AppTokens.hairline),
        borderRadius: const BorderRadius.vertical(
          top: Radius.circular(44),
          bottom: Radius.circular(AppTokens.radiusSharp),
        ),
      ),
      clipBehavior: Clip.antiAlias,
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: double.infinity,
            color: cap,
            padding: const EdgeInsets.fromLTRB(
              AppTokens.s8,
              AppTokens.s12,
              AppTokens.s8,
              AppTokens.s8,
            ),
            child: Text(
              place.toUpperCase(),
              textAlign: TextAlign.center,
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: AppTokens.stencilStyle.copyWith(
                fontSize: 10,
                color: c.paper,
              ),
            ),
          ),
          Padding(
            padding: const EdgeInsets.symmetric(vertical: AppTokens.s8),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  numeral,
                  style: AppTokens.milestoneStyle.copyWith(color: c.ink),
                ),
                Text(
                  unit.toUpperCase(),
                  style: AppTokens.stencilStyle.copyWith(
                    fontSize: 9,
                    color: c.muted,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// STAMP BADGE — the signature moment
// ---------------------------------------------------------------------------

/// A rubber stamp landing in a permit book.
///
/// Used for CONFIRMED, CACHED, IMPORTED, ROLLED BACK. It animates ONLY on the
/// transition into [landed]; a rebuild of an already-stamped row renders it
/// statically, because nothing animates at rest.
///
/// It is a status mark on an action the user took, never a trust mark: a
/// CONFIRMED stamp is the visible receipt of `markConfirmed`, while the thing
/// that actually tells the user a number is unverified remains the amber dot.
class StampBadge extends StatefulWidget {
  final String label;
  final bool landed;
  final Color? inkColor;

  const StampBadge({
    super.key,
    required this.label,
    this.landed = true,
    this.inkColor,
  });

  @override
  State<StampBadge> createState() => _StampBadgeState();
}

class _StampBadgeState extends State<StampBadge>
    with SingleTickerProviderStateMixin {
  late final AnimationController _ctl = AnimationController(
    vsync: this,
    duration: Motion.stamp,
    value: widget.landed ? 1 : 0,
  );

  @override
  void didUpdateWidget(StampBadge old) {
    super.didUpdateWidget(old);
    if (widget.landed && !old.landed) {
      _ctl.duration = Motion.stampDuration(context);
      // The haptic fires at contact, not at animation start.
      Future<void>.delayed(_ctl.duration! ~/ 2, () {
        if (mounted) Haptics.confirm();
      });
      _ctl.forward(from: 0);
    } else if (!widget.landed && old.landed) {
      _ctl.value = 0;
    }
  }

  @override
  void dispose() {
    _ctl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    final ink = widget.inkColor ?? c.signal;

    // The Semantics sits INSIDE the builder's child, not around the whole
    // badge: the builder drops that child entirely before the stamp lands,
    // and a label left outside would survive the drop and announce
    // "Confirmed" on a contact nobody confirmed.
    return AnimatedBuilder(
      animation: _ctl,
      builder: (context, child) {
        // Not merely invisible: gone. A stamp at zero opacity would still
        // announce "Confirmed" to a screen reader on a contact nobody has
        // confirmed, which is the same lie the amber dot was making in the
        // other direction.
        if (_ctl.value == 0) return const SizedBox.shrink();

        final t = Curves.elasticOut.transform(_ctl.value);
        // -8deg settling to -3deg, ink bleeding in to 0.85.
        final degrees = -8 + 5 * t;
        // THE STAMP OWNS ITS OWN SPACE, growing with the landing.
        //
        // Callers must mount this unconditionally and let `landed` drive
        // it — `if (confirmed) StampBadge(landed: confirmed)` looks right
        // and can never animate, because the widget only ever exists in
        // the landed state and so never sees the false -> true transition
        // that fires the stamp and the haptic. (DIALER_RETRO_PATCH.md
        // edit 8 spells it exactly that way; it is wrong.) Collapsing to
        // zero width here is what makes unconditional mounting free.
        return Align(
          widthFactor: _ctl.value,
          child: Opacity(
            opacity: (_ctl.value * 0.85).clamp(0.0, 0.85).toDouble(),
            child: Transform.rotate(
              angle: degrees * math.pi / 180,
              child: Transform.scale(scale: 0.9 + 0.1 * t, child: child),
            ),
          ),
        );
      },
      child: Semantics(
        label: widget.label,
        child: Container(
          padding: const EdgeInsets.symmetric(
            horizontal: AppTokens.s8,
            vertical: AppTokens.s4,
          ),
          decoration: BoxDecoration(
            border: Border.all(color: ink, width: 1.5),
            borderRadius: BorderRadius.circular(2),
          ),
          child: Text(
            widget.label.toUpperCase(),
            style: AppTokens.stampStyle.copyWith(color: ink),
          ),
        ),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// HAZARD STRIPE — the readiness banner's left edge, and nowhere else
// ---------------------------------------------------------------------------

/// 6px 45-degree bars. Rationed exactly the way red is: one element in the
/// whole app uses it.
class HazardStripe extends StatelessWidget {
  final double width;
  const HazardStripe({super.key, this.width = 6});

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    return SizedBox(
      width: width,
      child: CustomPaint(
        painter: _HazardPainter(c.cautionMark, c.cautionSoft),
        child: const SizedBox.expand(),
      ),
    );
  }
}

class _HazardPainter extends CustomPainter {
  final Color bar;
  final Color bed;
  const _HazardPainter(this.bar, this.bed);

  @override
  void paint(Canvas canvas, Size size) {
    canvas.drawRect(Offset.zero & size, Paint()..color = bed);
    final paint = Paint()
      ..color = bar
      ..strokeWidth = 3
      ..style = PaintingStyle.stroke;
    for (var y = -size.width; y < size.height + size.width; y += 6) {
      canvas.drawLine(Offset(0, y), Offset(size.width, y - size.width), paint);
    }
  }

  @override
  bool shouldRepaint(_HazardPainter old) => old.bar != bar || old.bed != bed;
}

// ---------------------------------------------------------------------------
// GRAIN — felt, not seen
// ---------------------------------------------------------------------------

/// One 128x128 tile under 4 KB, at 3% by day and 2% at night.
/// Three hard rules: never above text, never on the emergency tab,
/// never animated. Wrap the scaffold body, below every text layer.
class GrainOverlay extends StatelessWidget {
  final Widget child;
  const GrainOverlay({super.key, required this.child});

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    return Stack(
      children: [
        Positioned.fill(
          child: IgnorePointer(
            child: Opacity(
              opacity: c.grainOpacity,
              child: const DecoratedBox(
                decoration: BoxDecoration(
                  image: DecorationImage(
                    image: AssetImage('assets/texture/grain_128.png'),
                    repeat: ImageRepeat.repeat,
                  ),
                ),
              ),
            ),
          ),
        ),
        child,
      ],
    );
  }
}

// ---------------------------------------------------------------------------
// ROLLING DIGITS — counts and totals
// ---------------------------------------------------------------------------

/// Rolls each digit position on change. Only possible without the row width
/// jittering because every numeric style in this app is tabular.
class RollingDigits extends StatelessWidget {
  final String value;
  final TextStyle style;

  const RollingDigits({super.key, required this.value, required this.style});

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        for (var i = 0; i < value.length; i++)
          AnimatedSwitcher(
            duration: Motion.d(context, Motion.quick),
            switchInCurve: Motion.standard,
            transitionBuilder: (child, anim) => ClipRect(
              child: SlideTransition(
                position: Tween(
                  begin: const Offset(0, 0.6),
                  end: Offset.zero,
                ).animate(anim),
                child: FadeTransition(opacity: anim, child: child),
              ),
            ),
            child: Text(
              value[i],
              key: ValueKey('$i-${value[i]}'),
              style: style,
            ),
          ),
      ],
    );
  }
}
