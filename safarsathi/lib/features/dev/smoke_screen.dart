// A throwaway screen that exists to prove the scaffold actually works on a
// device: both fonts, both themes, the grain tile, the retro primitives and
// the haptics. It is deleted at backlog #6 when the diary replaces it.
//
// If the stencil labels below do not look condensed, the font family name in
// pubspec.yaml does not match app_tokens.dart and everything built after this
// will be sitting on a silent fallback.

import 'package:flutter/material.dart';

import '../../core/theme/app_tokens.dart';
import '../../core/theme/motion.dart';
import '../../core/widgets/retro.dart';

class SmokeScreen extends StatefulWidget {
  const SmokeScreen({super.key});

  @override
  State<SmokeScreen> createState() => _SmokeScreenState();
}

class _SmokeScreenState extends State<SmokeScreen> {
  bool _stamped = false;

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    final isNight = c.paper == AppColors.night.paper;

    return Scaffold(
      appBar: AppBar(
        title: Text('Scaffold check', style: AppTokens.titleStyle),
        bottom: PreferredSize(
          preferredSize: const Size.fromHeight(1),
          child: Container(height: AppTokens.hairline, color: c.rule),
        ),
      ),
      body: GrainOverlay(
        child: ListView(
          padding: const EdgeInsets.only(bottom: AppTokens.s32),
          children: [
            StencilLabel(isNight ? 'Lamp — night' : 'Paper — day'),
            Padding(
              padding: const EdgeInsets.symmetric(
                horizontal: AppTokens.gutter,
              ),
              child: Text(
                'Switch the phone between light and dark to swap palettes.',
                style: AppTokens.captionStyle.copyWith(color: c.muted),
              ),
            ),

            const StencilLabel('Type'),
            _Line('Title', AppTokens.titleStyle, 'Meghalaya · October', c.ink),
            _Line('Row', AppTokens.rowTitleStyle, 'Kongthong homestay', c.ink),
            _Line('Number', AppTokens.numberStyle, '+91 98560 41122', c.muted),
            _Line('Caption', AppTokens.captionStyle, 'Source: 112.gov.in', c.muted),
            _Line('Badge', AppTokens.badgeStyle, '112 · 108 · 1098', c.emergency),

            const StencilLabel('Milestone marker'),
            Padding(
              padding: const EdgeInsets.symmetric(
                horizontal: AppTokens.gutter,
              ),
              child: Row(
                children: [
                  const MilestoneMarker(numeral: '54', place: 'Sohra'),
                  const SizedBox(width: AppTokens.s16),
                  Expanded(
                    child: Text(
                      'Cap is signal green when the leg is cached, muted when '
                      'it is not. Never red.',
                      style: AppTokens.captionStyle.copyWith(color: c.muted),
                    ),
                  ),
                ],
              ),
            ),

            const StencilLabel('Ticket card'),
            Padding(
              padding: const EdgeInsets.symmetric(
                horizontal: AppTokens.gutter,
              ),
              child: TicketCard(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Kongthong → Cherrapunji',
                      style: AppTokens.titleStyle.copyWith(color: c.ink),
                    ),
                    const SizedBox(height: AppTokens.s4),
                    Text(
                      'Shared taxi · 56 km · leaves 09:30',
                      style: AppTokens.numberStyle.copyWith(color: c.muted),
                    ),
                  ],
                ),
              ),
            ),

            const StencilLabel('Hazard stripe'),
            Padding(
              padding: const EdgeInsets.symmetric(
                horizontal: AppTokens.gutter,
              ),
              child: IntrinsicHeight(
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    const SizedBox(width: 6, child: HazardStripe()),
                    Expanded(
                      child: Container(
                        color: c.cautionSoft,
                        padding: const EdgeInsets.all(AppTokens.s12),
                        child: Text(
                          '4 numbers not confirmed. Call before you leave '
                          'signal.',
                          style: AppTokens.captionStyle.copyWith(color: c.ink),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),

            const StencilLabel('Stamp and haptics'),
            Padding(
              padding: const EdgeInsets.symmetric(
                horizontal: AppTokens.gutter,
              ),
              child: Row(
                children: [
                  PressScale(
                    onTap: () => setState(() => _stamped = !_stamped),
                    child: Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: AppTokens.s16,
                        vertical: AppTokens.s12,
                      ),
                      decoration: BoxDecoration(
                        color: c.signal,
                        border: Border.all(color: c.ink),
                        borderRadius: BorderRadius.circular(
                          AppTokens.radiusSoft,
                        ),
                      ),
                      child: Text(
                        _stamped ? 'RESET' : 'MARK CONFIRMED',
                        style: AppTokens.stencilStyle.copyWith(color: c.paper),
                      ),
                    ),
                  ),
                  const SizedBox(width: AppTokens.s24),
                  StampBadge(label: 'Confirmed', landed: _stamped),
                ],
              ),
            ),

            const StencilLabel('Rolling digits'),
            Padding(
              padding: const EdgeInsets.symmetric(
                horizontal: AppTokens.gutter,
              ),
              child: RollingDigits(
                value: _stamped ? '3' : '4',
                style: AppTokens.milestoneStyle.copyWith(color: c.ink),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _Line extends StatelessWidget {
  final String label;
  final TextStyle style;
  final String sample;
  final Color color;

  const _Line(this.label, this.style, this.sample, this.color);

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    return Padding(
      padding: const EdgeInsets.fromLTRB(
        AppTokens.gutter,
        AppTokens.s8,
        AppTokens.gutter,
        AppTokens.s8,
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.baseline,
        textBaseline: TextBaseline.alphabetic,
        children: [
          SizedBox(
            width: 78,
            child: Text(
              label.toUpperCase(),
              style: AppTokens.stencilStyle.copyWith(
                fontSize: 10,
                color: c.muted,
              ),
            ),
          ),
          Expanded(child: Text(sample, style: style.copyWith(color: color))),
        ],
      ),
    );
  }
}
