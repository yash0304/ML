// lib/features/trips/presentation/readiness_panel.dart
//
// The pre-departure block, on the Trip screen — issue #20.
//
// This is the payoff for the whole trust system. Everything else records
// whether a number has been dialled; this is the part that says the trip is
// not ready yet, and names what is missing.
//
// It uses cautionMark, not emergency red. A number you have not called yet is
// a thing to do before you leave, not a crisis, and spending the emergency
// colour here would blunt it on the screen where it has to carry weight.

import 'package:flutter/material.dart';

import '../../../core/theme/app_tokens.dart';
import '../../../core/widgets/retro.dart';
import '../data/readiness.dart';

class ReadinessPanel extends StatelessWidget {
  final Stream<Readiness> readiness;

  /// Opens the diary filtered to that stop, so the fix is one tap away rather
  /// than a hunt through eleven categories.
  final void Function(int stopId)? onOpenStop;

  const ReadinessPanel({
    super.key,
    required this.readiness,
    this.onOpenStop,
  });

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);

    return StreamBuilder<Readiness>(
      stream: readiness,
      builder: (context, snap) {
        final data = snap.data;
        if (data == null) return const SizedBox.shrink();

        if (data.isReady) {
          return Padding(
            padding: const EdgeInsets.fromLTRB(
              AppTokens.gutter,
              AppTokens.s8,
              AppTokens.gutter,
              AppTokens.s8,
            ),
            child: Row(
              children: [
                Icon(Icons.check, size: 16, color: c.signal),
                const SizedBox(width: AppTokens.s8),
                Expanded(
                  child: Text(
                    'Every overnight stop has a number you have called.',
                    style: AppTokens.captionStyle.copyWith(color: c.signal),
                  ),
                ),
              ],
            ),
          );
        }

        return Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const StencilLabel('Before you leave signal'),
            for (final item in data.blocking)
              _BlockingRow(
                item: item,
                onTap: onOpenStop == null
                    ? null
                    : () => onOpenStop!(item.stopId),
              ),
            Padding(
              padding: const EdgeInsets.fromLTRB(
                AppTokens.gutter,
                AppTokens.s8,
                AppTokens.gutter,
                0,
              ),
              child: Text(
                'The trip does not read ready while any of these are open. '
                'Choose where you are staying at each stop, call that '
                'number, then mark it confirmed.'
                '${onOpenStop == null ? '' : ' Tap one to choose.'}',
                style: AppTokens.captionStyle.copyWith(color: c.muted),
              ),
            ),
          ],
        );
      },
    );
  }
}

class _BlockingRow extends StatelessWidget {
  final ReadinessItem item;
  final VoidCallback? onTap;

  const _BlockingRow({required this.item, this.onTap});

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    return GestureDetector(
      onTap: onTap,
      behavior: HitTestBehavior.opaque,
      child: Container(
        padding: const EdgeInsets.symmetric(
          horizontal: AppTokens.gutter,
          vertical: AppTokens.s12,
        ),
        decoration: BoxDecoration(
          border: Border(
            bottom: BorderSide(color: c.rule, width: AppTokens.hairline),
          ),
        ),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // The empty stencil box, same mark the checklist uses. An open
            // blocking item is an unticked box, not an error icon.
            Container(
              width: 18,
              height: 18,
              margin: const EdgeInsets.only(top: 1),
              decoration: BoxDecoration(border: Border.all(color: c.ink)),
            ),
            const SizedBox(width: AppTokens.s12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    item.label,
                    style: AppTokens.rowTitleStyle.copyWith(color: c.ink),
                  ),
                  const SizedBox(height: 2),
                  Text(
                    item.missing
                        // Absence is the more dangerous case and the one apps
                        // usually say nothing about at all.
                        ? 'NOTHING TO CALL'
                        : 'BLOCKING',
                    style: AppTokens.stencilStyle.copyWith(
                      fontSize: 9,
                      color: c.cautionMark,
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
