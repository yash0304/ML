// lib/features/emergency/presentation/emergency_screen.dart
//
// THE EXEMPT SCREEN. Read DESIGN_VISUAL_v2.md §0 before changing anything.
//
// No grain. No ruled paper. No thumb index. No stamps. No ticket edges. No
// swipe actions. Everywhere else in this app ephemera carries category and
// place; here it carries nothing, because anything a user could mistake for a
// verification mark would defeat the whole tier system.
//
// AND HERE THE TAP CALLS. Copy-first is a planning workflow — in an emergency
// a copy-and-paste dance is a liability. Copy is demoted to a secondary icon,
// for reading a number out to somebody else.

import 'package:flutter/material.dart';

import '../../../core/database/app_database.dart';
import '../../../core/theme/app_tokens.dart';
import '../../../core/theme/motion.dart';
import '../../contacts/data/contacts_dao.dart';

class EmergencyScreen extends StatelessWidget {
  /// Bundled Tier 1 numbers, filtered to the countries this trip touches.
  final Stream<List<EmergencyHelpline>> helplines;

  /// Trip contacts the user marked emergency-relevant.
  final Stream<List<Contact>> localContacts;

  final String? placeLabel;

  /// Places the call. Fires the heavy haptic itself.
  final Future<void> Function(String number)? onCall;
  final Future<void> Function(String number)? onCopy;

  /// "Text my location", first on the tab. Optional so the screen's own
  /// tests and goldens can render without a location source.
  final Widget? sosPanel;

  const EmergencyScreen({
    super.key,
    required this.helplines,
    required this.localContacts,
    this.placeLabel,
    this.onCall,
    this.onCopy,
    this.sosPanel,
  });

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);

    // Deliberately no GrainOverlay. See the file header.
    return Scaffold(
      backgroundColor: c.paper,
      body: SafeArea(
        bottom: false,
        child: ListView(
          padding: const EdgeInsets.only(bottom: AppTokens.s24),
          children: [
            _header(c),
            // FIRST, ABOVE THE HELPLINES. 112 is one tap below either way;
            // telling your own people where you are is the thing no
            // helpline can do for you.
            ?sosPanel,
            _officialSection(c),
            _localSection(c),
            _stateNote(c),
          ],
        ),
      ),
    );
  }

  Widget _header(AppColors c) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(
        AppTokens.gutter,
        AppTokens.s12,
        AppTokens.gutter,
        0,
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          if (placeLabel != null)
            Text(
              placeLabel!.toUpperCase(),
              style: AppTokens.stencilStyle.copyWith(
                fontSize: 10,
                letterSpacing: 1.6,
                color: c.muted,
              ),
            ),
          const SizedBox(height: 2),
          Text('Emergency', style: AppTokens.titleStyle.copyWith(color: c.ink)),
          Text(
            'Tap the number to call. No copying step.',
            style: AppTokens.captionStyle.copyWith(color: c.muted),
          ),
        ],
      ),
    );
  }

  Widget _officialSection(AppColors c) {
    return StreamBuilder<List<EmergencyHelpline>>(
      stream: helplines,
      builder: (context, snap) {
        final lines = snap.data ?? const <EmergencyHelpline>[];
        if (lines.isEmpty) return const SizedBox.shrink();
        return Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const _SectionHeader(label: 'Official helplines'),
            for (final line in lines)
              _EmergencyRow(
                badge: line.number,
                label: line.label,
                // Provenance under every bundled number, always. The user can
                // see where it came from and judge it for themselves.
                sub: line.sourceNote,
                onCall: onCall == null ? null : () => onCall!(line.number),
                onCopy: onCopy == null ? null : () => onCopy!(line.number),
              ),
          ],
        );
      },
    );
  }

  Widget _localSection(AppColors c) {
    return StreamBuilder<List<Contact>>(
      stream: localContacts,
      builder: (context, snap) {
        final items = snap.data ?? const <Contact>[];
        if (items.isEmpty) return const SizedBox.shrink();
        return Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // A visible header, always. A government short code and a number
            // somebody typed must never sit in the same list.
            const _SectionHeader(label: 'Your local contacts'),
            for (final contact in items)
              _EmergencyRow(
                badge: _initials(contact.name),
                badgeIsNumber: false,
                label: contact.name,
                sub: contact.phoneRaw,
                untrusted: !ContactTier.parse(contact.tier).isTrusted,
                onCall: onCall == null
                    ? null
                    : () => onCall!(contact.phoneE164 ?? contact.phoneRaw),
                onCopy: onCopy == null
                    ? null
                    : () => onCopy!(contact.phoneE164 ?? contact.phoneRaw),
              ),
          ],
        );
      },
    );
  }

  static String _initials(String name) {
    final parts = name.trim().split(RegExp(r'\s+'));
    final letters = parts
        .where((p) => p.isNotEmpty)
        .take(2)
        .map((p) => p[0].toUpperCase())
        .join();
    return letters.isEmpty ? '?' : letters;
  }

  Widget _stateNote(AppColors c) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(
        AppTokens.gutter,
        AppTokens.s24,
        AppTokens.gutter,
        0,
      ),
      child: Text(
        'State and union territory helplines are not listed. No authoritative '
        'combined list exists across 28 states and 8 union territories, and a '
        'guessed number here would be worse than none.',
        style: AppTokens.captionStyle.copyWith(color: c.muted),
      ),
    );
  }
}

class _SectionHeader extends StatelessWidget {
  final String label;
  const _SectionHeader({required this.label});

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
        children: [
          Text(
            label.toUpperCase(),
            style: AppTokens.stencilStyle.copyWith(
              fontSize: 10.5,
              color: c.muted,
            ),
          ),
          const SizedBox(width: AppTokens.s12),
          Expanded(
            child: Container(height: AppTokens.hairline, color: c.rule),
          ),
        ],
      ),
    );
  }
}

class _EmergencyRow extends StatelessWidget {
  final String badge;
  final bool badgeIsNumber;
  final String label;
  final String sub;
  final bool untrusted;
  final VoidCallback? onCall;
  final VoidCallback? onCopy;

  const _EmergencyRow({
    required this.badge,
    this.badgeIsNumber = true,
    required this.label,
    required this.sub,
    this.untrusted = false,
    this.onCall,
    this.onCopy,
  });

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);

    return PressScale(
      // The tap CALLS. Heavy, because calling 112 should feel heavier than
      // calling your homestay.
      onTap: onCall,
      feedback: Haptics.grave,
      child: Container(
        padding: const EdgeInsets.symmetric(
          horizontal: AppTokens.gutter,
          vertical: AppTokens.s8,
        ),
        decoration: BoxDecoration(
          border: Border(
            bottom: BorderSide(color: c.rule, width: AppTokens.hairline),
          ),
        ),
        child: Row(
          children: [
            Container(
              width: 46,
              height: 46,
              alignment: Alignment.center,
              decoration: BoxDecoration(
                color: badgeIsNumber ? c.emergencySoft : c.stone,
                borderRadius: BorderRadius.circular(AppTokens.radiusSoft),
              ),
              child: Text(
                badge,
                style: AppTokens.badgeStyle.copyWith(
                  fontSize: badge.length > 4 ? 11 : 14,
                  color: badgeIsNumber ? c.emergency : c.muted,
                ),
              ),
            ),
            const SizedBox(width: AppTokens.s12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                mainAxisSize: MainAxisSize.min,
                children: [
                  Row(
                    children: [
                      Flexible(
                        child: Text(
                          label,
                          style: AppTokens.rowTitleStyle.copyWith(color: c.ink),
                          overflow: TextOverflow.ellipsis,
                        ),
                      ),
                      if (untrusted)
                        Container(
                          width: 7,
                          height: 7,
                          margin: const EdgeInsets.only(left: 6),
                          decoration: BoxDecoration(
                            color: c.cautionMark,
                            shape: BoxShape.circle,
                          ),
                        ),
                    ],
                  ),
                  Text(
                    sub,
                    style: AppTokens.captionStyle.copyWith(
                      fontSize: 11.5,
                      color: c.muted,
                    ),
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                  ),
                ],
              ),
            ),
            // Secondary, for reading a number out to somebody else.
            if (onCopy != null)
              _IconButton(
                icon: Icons.copy_rounded,
                tone: c.muted,
                border: c.rule,
                semantic: 'Copy $label',
                onTap: onCopy,
              ),
            const SizedBox(width: 6),
            _IconButton(
              icon: Icons.call,
              tone: c.emergency,
              border: c.emergency,
              semantic: 'Call $label',
              onTap: onCall,
              feedback: Haptics.grave,
            ),
          ],
        ),
      ),
    );
  }
}

class _IconButton extends StatelessWidget {
  final IconData icon;
  final Color tone;
  final Color border;
  final String semantic;
  final VoidCallback? onTap;
  final VoidCallback? feedback;

  const _IconButton({
    required this.icon,
    required this.tone,
    required this.border,
    required this.semantic,
    this.onTap,
    this.feedback,
  });

  @override
  Widget build(BuildContext context) {
    return Semantics(
      button: true,
      label: semantic,
      child: PressScale(
        onTap: onTap,
        feedback: feedback ?? Haptics.light,
        child: Container(
          padding: const EdgeInsets.all(9),
          decoration: BoxDecoration(
            border: Border.all(color: border),
            borderRadius: BorderRadius.circular(AppTokens.radiusSoft),
          ),
          child: Icon(icon, size: 17, color: tone),
        ),
      ),
    );
  }
}
