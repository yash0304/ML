// lib/features/trips/presentation/stay_picker_sheet.dart
//
// "Where are you staying in Shillong?" — one choice per stop.
//
// The list is the accommodation numbers already saved at the stop. Picking
// one is a statement about the trip, not about the number: it is what the
// Tonight card shows, what the plan sent home names, and what has to be
// confirmed before the trip reads as ready. The sheet says so, so nobody
// picks casually.

import 'package:flutter/material.dart';

import '../../../core/database/app_database.dart';
import '../../../core/theme/app_tokens.dart';
import '../../../core/widgets/retro.dart';
import '../../contacts/presentation/diary_widgets.dart' show TrustDot;

/// What the person chose.
sealed class StayChoice {
  const StayChoice();
}

class StayPicked extends StayChoice {
  final int contactId;
  const StayPicked(this.contactId);
}

/// Back to not decided.
class StayCleared extends StayChoice {
  const StayCleared();
}

/// Somewhere not in the diary yet: add it, and it becomes the stay.
class StayAddNew extends StayChoice {
  const StayAddNew();
}

Future<StayChoice?> showStayPicker(
  BuildContext context, {
  required String stopName,
  required List<Contact> options,
  int? currentId,
  String? title,
  String? explainer,
  String? addLabel,
}) {
  final c = AppTokens.of(context);
  return showModalBottomSheet<StayChoice>(
    context: context,
    backgroundColor: c.paper,
    isScrollControlled: true,
    builder: (sheetContext) => SafeArea(
      child: ConstrainedBox(
        constraints: BoxConstraints(
          maxHeight: MediaQuery.of(sheetContext).size.height * 0.8,
        ),
        child: StayPickerList(
          stopName: stopName,
          options: options,
          currentId: currentId,
          title: title,
          explainer: explainer,
          addLabel: addLabel,
          onChoice: (choice) => Navigator.of(sheetContext).pop(choice),
        ),
      ),
    ),
  );
}

/// The sheet's body, separate so it can be tested and drawn without a route.
class StayPickerList extends StatelessWidget {
  final String stopName;
  final List<Contact> options;
  final int? currentId;
  final ValueChanged<StayChoice> onChoice;

  /// The same list picks a driver for a leg; these replace the stay wording.
  final String? title;
  final String? explainer;
  final String? addLabel;

  const StayPickerList({
    super.key,
    required this.stopName,
    required this.options,
    required this.onChoice,
    this.currentId,
    this.title,
    this.explainer,
    this.addLabel,
  });

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    final caption = AppTokens.captionStyle.copyWith(color: c.muted);

    return ListView(
      shrinkWrap: true,
      padding: const EdgeInsets.only(bottom: AppTokens.s16),
      children: [
        StencilLabel(title ?? 'Staying in $stopName'),
        Padding(
          padding: const EdgeInsets.fromLTRB(
            AppTokens.gutter,
            0,
            AppTokens.gutter,
            AppTokens.s8,
          ),
          child: Text(
            explainer ??
                'The one you pick is tonight\'s bed on the Trip page, the '
                    'stay in the plan you send home, and the number that has '
                    'to be confirmed before the trip is ready. The others '
                    'stay in the diary.',
            style: caption,
          ),
        ),
        for (final o in options)
          _Option(
            key: Key('stay-option-${o.id}'),
            contact: o,
            chosen: o.id == currentId,
            onTap: () => onChoice(StayPicked(o.id)),
          ),
        _Action(
          key: const Key('stay-add-new'),
          icon: Icons.add,
          label: addLabel ?? 'Somewhere else — add it',
          color: c.signal,
          onTap: () => onChoice(const StayAddNew()),
        ),
        if (currentId != null)
          _Action(
            key: const Key('stay-clear'),
            icon: Icons.remove_circle_outline,
            label: 'Not decided yet',
            color: c.muted,
            onTap: () => onChoice(const StayCleared()),
          ),
      ],
    );
  }
}

class _Option extends StatelessWidget {
  final Contact contact;
  final bool chosen;
  final VoidCallback onTap;

  const _Option({
    super.key,
    required this.contact,
    required this.chosen,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    final note = contact.note?.trim();
    return InkWell(
      onTap: onTap,
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
          children: [
            Icon(
              chosen ? Icons.radio_button_checked : Icons.radio_button_off,
              size: 20,
              color: chosen ? c.signal : c.muted,
            ),
            const SizedBox(width: AppTokens.s12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    contact.name,
                    style: AppTokens.rowTitleStyle.copyWith(color: c.ink),
                  ),
                  Text(
                    contact.phoneRaw,
                    style: AppTokens.numberStyle.copyWith(color: c.ink),
                  ),
                  if (note != null && note.isNotEmpty)
                    Text(
                      note,
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                      style: AppTokens.captionStyle.copyWith(color: c.muted),
                    ),
                ],
              ),
            ),
            if (!contact.callConfirmed) const TrustDot(),
          ],
        ),
      ),
    );
  }
}

class _Action extends StatelessWidget {
  final IconData icon;
  final String label;
  final Color color;
  final VoidCallback onTap;

  const _Action({
    super.key,
    required this.icon,
    required this.label,
    required this.color,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      child: Padding(
        padding: const EdgeInsets.symmetric(
          horizontal: AppTokens.gutter,
          vertical: AppTokens.s12,
        ),
        child: Row(
          children: [
            Icon(icon, size: 20, color: color),
            const SizedBox(width: AppTokens.s12),
            Expanded(
              child: Text(
                label,
                style: AppTokens.rowTitleStyle.copyWith(color: color),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
