// lib/features/contacts/presentation/diary_widgets.dart
//
// The parts of the diary. Separated from the screen so each can be tested on
// its own and so #7 can wire actions without touching layout.
//
// THE TRUST DOT IS THE POINT. A number from a spreadsheet and 112 must never
// look the same. Verified renders plain; unverified carries the amber mark.
// No exceptions, and nothing else in this file may use that colour.

import 'package:flutter/material.dart';

import '../../../core/database/app_database.dart';
import '../../../core/theme/app_tokens.dart';
import '../../../core/theme/motion.dart';
import '../../../core/widgets/retro.dart';
import '../data/contacts_dao.dart';

/// One ruled line of the diary.
///
/// The number is set larger than the meta line on purpose: in a diary the
/// number is the content, and since the dial happens in the Android dialer
/// after a paste, legibility of the digits outranks everything else here.
class DiaryEntry extends StatelessWidget {
  final Contact contact;

  /// 1-based position on the page, rendered in the margin.
  final int lineNumber;

  /// Wired at #7. Until then the row is inert by design.
  final VoidCallback? onCopy;
  final VoidCallback? onOpen;

  const DiaryEntry({
    super.key,
    required this.contact,
    required this.lineNumber,
    this.onCopy,
    this.onOpen,
  });

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    final trusted = ContactTier.parse(contact.tier).isTrusted;
    final meta = _meta();

    return PressScale(
      onTap: onCopy,
      onLongPress: onOpen,
      child: DecoratedBox(
        decoration: BoxDecoration(
          border: Border(
            bottom: BorderSide(color: c.rule, width: AppTokens.hairline),
          ),
        ),
        child: IntrinsicHeight(
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              _Margin(lineNumber: lineNumber),
              Expanded(
                child: Padding(
                  padding: const EdgeInsets.fromLTRB(
                    AppTokens.s12,
                    AppTokens.s8,
                    AppTokens.s8,
                    AppTokens.s12,
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Row(
                        children: [
                          if (contact.isPinned)
                            Padding(
                              padding: const EdgeInsets.only(
                                right: AppTokens.s4,
                              ),
                              child: Icon(
                                Icons.push_pin,
                                size: 12,
                                color: c.muted,
                              ),
                            ),
                          Flexible(
                            child: Text(
                              contact.name,
                              style: AppTokens.rowTitleStyle.copyWith(
                                color: c.ink,
                              ),
                              overflow: TextOverflow.ellipsis,
                            ),
                          ),
                          // The whole point of the tier system. Unconfirmed
                          // numbers carry a caution mark; confirmed ones
                          // carry nothing.
                          if (!trusted) const _TrustDot(),
                        ],
                      ),
                      const SizedBox(height: 1),
                      Text(
                        contact.phoneRaw,
                        style: AppTokens.numberStyle.copyWith(
                          fontSize: 15,
                          color: c.ink,
                        ),
                      ),
                      if (meta != null)
                        Padding(
                          padding: const EdgeInsets.only(top: 1),
                          child: Text(
                            meta,
                            style: AppTokens.captionStyle.copyWith(
                              fontSize: 11.5,
                              color: c.muted,
                            ),
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                          ),
                        ),
                    ],
                  ),
                ),
              ),
              _CopyAffordance(onTap: onCopy),
            ],
          ),
        ),
      ),
    );
  }

  String? _meta() {
    final parts = <String>[
      if (contact.note != null && contact.note!.trim().isNotEmpty)
        contact.note!.trim(),
      ContactCategory.labels[contact.category] ?? contact.category,
    ];
    return parts.isEmpty ? null : parts.join(' · ');
  }
}

/// The ruled notebook margin: a number, and a doubled hairline down its edge.
/// The ruled notebook margin: a line number, and a doubled hairline down its
/// edge.
class _Margin extends StatelessWidget {
  final int lineNumber;
  const _Margin({required this.lineNumber});

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    // Two hairlines a couple of pixels apart, drawn as siblings. An earlier
    // version used a border plus a boxShadow for the second rule, which
    // spreads behind the whole box and paints a solid column instead.
    return Row(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        SizedBox(
          width: 26,
          child: Padding(
            padding: const EdgeInsets.only(top: AppTokens.s8),
            child: Text(
              lineNumber.toString().padLeft(2, '0'),
              textAlign: TextAlign.center,
              style: AppTokens.stencilStyle.copyWith(
                fontSize: 9.5,
                color: c.muted,
              ),
            ),
          ),
        ),
        Container(width: AppTokens.hairline, color: c.rule),
        const SizedBox(width: 2),
        Container(width: AppTokens.hairline, color: c.rule),
      ],
    );
  }
}

class _TrustDot extends StatelessWidget {
  const _TrustDot();

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    return Padding(
      padding: const EdgeInsets.only(left: 6),
      child: Tooltip(
        message: 'Not confirmed yet',
        child: Semantics(
          label: 'Not confirmed yet',
          child: Container(
            width: 7,
            height: 7,
            decoration: BoxDecoration(
              // cautionMark, not caution: this is a 7px graphic held to the
              // 3:1 floor, and it has to read amber at that size.
              color: c.cautionMark,
              shape: BoxShape.circle,
            ),
          ),
        ),
      ),
    );
  }
}

class _CopyAffordance extends StatelessWidget {
  final VoidCallback? onTap;
  const _CopyAffordance({this.onTap});

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    return Padding(
      padding: const EdgeInsets.fromLTRB(AppTokens.s4, 0, AppTokens.s12, 0),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(Icons.copy_rounded, size: 17, color: c.signal),
          const SizedBox(height: 2),
          Text(
            'Copy',
            style: AppTokens.stencilStyle.copyWith(
              fontSize: 8.5,
              letterSpacing: 1.0,
              color: c.signal,
            ),
          ),
        ],
      ),
    );
  }
}

/// The thumb index down the right edge. Flipped the way you would flip a real
/// diary, standing in a village, one-handed.
///
/// #49 narrows this to the categories the trip actually uses — eleven do not
/// fit a phone edge, so it scrolls for now.
class CategoryIndex extends StatelessWidget {
  final String? selected;
  final ValueChanged<String?> onSelect;

  const CategoryIndex({
    super.key,
    required this.selected,
    required this.onSelect,
  });

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    return Container(
      width: 30,
      decoration: BoxDecoration(
        border: Border(
          left: BorderSide(color: c.rule, width: AppTokens.hairline),
        ),
      ),
      child: ListView(
        padding: EdgeInsets.zero,
        children: [
          _Tab(
            label: 'All',
            active: selected == null,
            onTap: () => onSelect(null),
          ),
          for (final category in ContactCategory.pickerOrder)
            _Tab(
              label: ContactCategory.shortLabels[category] ?? category,
              active: selected == category,
              onTap: () => onSelect(category),
            ),
        ],
      ),
    );
  }
}

class _Tab extends StatelessWidget {
  final String label;
  final bool active;
  final VoidCallback onTap;

  const _Tab({required this.label, required this.active, required this.onTap});

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    return PressScale(
      onTap: onTap,
      feedback: Haptics.select,
      child: Container(
        height: 62,
        alignment: Alignment.center,
        decoration: BoxDecoration(
          color: active ? c.signal : c.stone,
          border: Border(
            bottom: BorderSide(color: c.rule, width: AppTokens.hairline),
          ),
        ),
        child: RotatedBox(
          quarterTurns: 3,
          child: Text(
            label.toUpperCase(),
            maxLines: 1,
            overflow: TextOverflow.clip,
            style: AppTokens.stencilStyle.copyWith(
              fontSize: 9.5,
              color: active ? c.paper : c.muted,
            ),
          ),
        ),
      ),
    );
  }
}

/// Renders only when something is unconfirmed. A permanent banner becomes
/// wallpaper and stops being read.
class ReadinessBanner extends StatelessWidget {
  /// A stream of the unconfirmed count rather than the DAO itself, so the
  /// widget can be rendered and tested without a database.
  final Stream<int> unconfirmedCount;

  const ReadinessBanner({super.key, required this.unconfirmedCount});

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    return StreamBuilder<int>(
      stream: unconfirmedCount,
      builder: (context, snap) {
        final n = snap.data ?? 0;
        if (n == 0) return const SizedBox.shrink();
        return Padding(
          padding: const EdgeInsets.fromLTRB(
            AppTokens.gutter,
            AppTokens.s8,
            AppTokens.gutter,
            AppTokens.s4,
          ),
          child: IntrinsicHeight(
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                // The only ephemera permitted on this element.
                const HazardStripe(),
                Expanded(
                  child: Container(
                    color: c.cautionSoft,
                    padding: const EdgeInsets.symmetric(
                      horizontal: AppTokens.s12,
                      vertical: AppTokens.s8,
                    ),
                    child: Row(
                      children: [
                        RollingDigits(
                          value: '$n',
                          style: AppTokens.numberStyle.copyWith(
                            fontSize: 13,
                            fontWeight: FontWeight.w600,
                            color: c.ink,
                          ),
                        ),
                        const SizedBox(width: AppTokens.s4),
                        Expanded(
                          child: Text(
                            '${n == 1 ? "number" : "numbers"} not confirmed. '
                            'Call before you leave signal.',
                            style: AppTokens.captionStyle.copyWith(
                              fontSize: 12.5,
                              color: c.ink,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ],
            ),
          ),
        );
      },
    );
  }
}

/// Empty states give direction, not a shrug.
class DiaryEmptyState extends StatelessWidget {
  final bool searching;
  final VoidCallback? onAdd;

  const DiaryEmptyState({super.key, required this.searching, this.onAdd});

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(AppTokens.s32),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            StampBadge(
              label: searching ? 'No match' : 'No entries',
              inkColor: c.muted,
            ),
            const SizedBox(height: AppTokens.s24),
            Text(
              searching ? 'Nothing matches that.' : 'Nothing in the diary yet.',
              style: AppTokens.titleStyle.copyWith(color: c.ink),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: AppTokens.s8),
            Text(
              searching
                  ? 'Try a shorter search, or clear the category tab.'
                  : 'Add your homestay, driver and guide numbers here, '
                        'or import a sheet.',
              style: AppTokens.captionStyle.copyWith(color: c.muted),
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
    );
  }
}

/// Confirms a copy and offers the dialer, so the whole workflow is two taps
/// and a paste.
///
/// Not a SnackBar: the number has to be readable in tabular figures and the
/// action has to be a stencil mark, and bending Material's snackbar to that
/// costs more than drawing it.
class CopyToast extends StatelessWidget {
  /// The number that was copied, or null when [message] is an error.
  final String? number;
  final String? message;
  final VoidCallback? onOpenDialer;

  const CopyToast({super.key, this.number, this.message, this.onOpenDialer});

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    return Container(
      decoration: BoxDecoration(
        color: c.ink,
        borderRadius: BorderRadius.circular(AppTokens.radiusSoft),
      ),
      padding: const EdgeInsets.symmetric(
        horizontal: AppTokens.s12,
        vertical: AppTokens.s8,
      ),
      child: Row(
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  number != null ? 'Copied' : 'Could not open that',
                  style: AppTokens.captionStyle.copyWith(color: c.paper),
                ),
                Text(
                  number ?? message ?? '',
                  style: AppTokens.numberStyle.copyWith(
                    fontSize: 15,
                    fontWeight: FontWeight.w600,
                    color: c.paper,
                  ),
                ),
              ],
            ),
          ),
          if (number != null && onOpenDialer != null) ...[
            const SizedBox(width: AppTokens.s8),
            PressScale(
              onTap: onOpenDialer,
              child: Container(
                padding: const EdgeInsets.symmetric(
                  horizontal: AppTokens.s8,
                  vertical: 6,
                ),
                decoration: BoxDecoration(
                  border: Border.all(color: c.paper),
                  borderRadius: BorderRadius.circular(3),
                ),
                child: Text(
                  'Open dialer',
                  style: AppTokens.stencilStyle.copyWith(
                    fontSize: 9.5,
                    color: c.paper,
                  ),
                ),
              ),
            ),
          ],
        ],
      ),
    );
  }
}
