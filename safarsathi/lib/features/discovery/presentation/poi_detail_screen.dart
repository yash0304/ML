// lib/features/discovery/presentation/poi_detail_screen.dart
//
// One place on the corridor — issue #28.
//
// THE WHOLE SCREEN IS ABOUT PROVENANCE. A number here was typed into a public
// map by a stranger and may be a decade old. It renders with the amber dot and
// with what that means spelled out, and saving it into the diary keeps it
// `communityOsm` — never `userEntered`, because the app must not tell you that
// you typed something a stranger did.

import 'package:flutter/material.dart';

import '../../../core/database/app_database.dart';
import '../../../core/theme/app_tokens.dart';
import '../../../core/theme/motion.dart';
import '../../../core/widgets/retro.dart';
import '../data/place_details.dart';
import '../data/discovery.dart';
import '../data/poi_category.dart' show placeCategoryLabel;

class PoiDetailScreen extends StatelessWidget {
  final CorridorPlace place;

  final Future<void> Function(PoiContact phone) onCopy;
  final Future<void> Function(PoiContact phone) onOpenDialer;
  final Future<void> Function(PoiContact phone) onSave;

  /// Hands off to Google Maps. Null hides the row.
  final Future<void> Function()? onOpenMaps;

  /// True once this place's number is already in the diary, so the screen
  /// does not offer to save it twice.
  final bool alreadySaved;

  /// Whether this place is in the plan for its leg.
  final bool planned;

  /// Puts it in the leg's plan (true) or takes it out (false). Null hides
  /// the button.
  final Future<void> Function(bool plan)? onTogglePlan;

  const PoiDetailScreen({
    super.key,
    required this.place,
    required this.onCopy,
    required this.onOpenDialer,
    required this.onSave,
    this.onOpenMaps,
    this.alreadySaved = false,
    this.planned = false,
    this.onTogglePlan,
  });

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);

    return Scaffold(
      backgroundColor: c.paper,
      appBar: AppBar(
        title: Text(placeCategoryLabel(place.category)),
        backgroundColor: c.paper,
        foregroundColor: c.ink,
        elevation: 0,
      ),
      body: ListView(
        padding: const EdgeInsets.only(bottom: AppTokens.s32),
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(
              AppTokens.gutter,
              AppTokens.s8,
              AppTokens.gutter,
              0,
            ),
            child: Text(
              place.name,
              style: AppTokens.titleStyle.copyWith(color: c.ink),
            ),
          ),
          if (onTogglePlan != null)
            _PlanToggle(planned: planned, onToggle: onTogglePlan!),

          if (_hasFood(place)) ...[
            const StencilLabel('What they serve'),
            if (kindOfFood(place.tags) != null)
              _Field(label: 'Kind', value: kindOfFood(place.tags)!),
            if (cuisineOf(place.tags) != null)
              _Field(label: 'Food', value: cuisineOf(place.tags)!),
            if (vegOf(place.tags) != null)
              _Field(label: 'Veg', value: vegOf(place.tags)!),
            if (jainOf(place.tags) != null)
              _Field(label: 'Jain', value: jainOf(place.tags)!),
            if (hoursOf(place.tags) != null)
              _Field(label: 'Hours', value: hoursOf(place.tags)!),
            if ((place.tags['description'] ?? '').trim().isNotEmpty)
              Padding(
                padding: const EdgeInsets.fromLTRB(
                  AppTokens.gutter,
                  AppTokens.s8,
                  AppTokens.gutter,
                  0,
                ),
                child: Text(
                  place.tags['description']!.trim(),
                  style: AppTokens.captionStyle.copyWith(color: c.ink),
                ),
              ),
            Padding(
              padding: const EdgeInsets.fromLTRB(
                AppTokens.gutter,
                AppTokens.s8,
                AppTokens.gutter,
                0,
              ),
              child: Text(
                // SAID, because it was asked for. No source this app can
                // use carries menus for places like this one.
                'From OpenStreetMap, which does not carry menus — this is '
                'what it knows. Hours may be out of date; the note you add '
                'when you save it can say what you find.',
                style: AppTokens.captionStyle.copyWith(color: c.muted),
              ),
            ),
          ],

          const StencilLabel('Where'),
          _Field(
            label: 'Along the leg',
            value: '${place.alongRouteKm.round()} km',
          ),
          _Field(
            label: 'Off the road',
            value: place.offRouteKm < 0.2
                ? 'on it'
                : '${place.offRouteKm.toStringAsFixed(1)} km',
          ),

          if (place.hasPhone) ...[
            const StencilLabel('Number'),
            for (final phone in place.phones)
              _PhoneRow(
                phone: phone,
                onCopy: () => onCopy(phone),
                onOpenDialer: () => onOpenDialer(phone),
              ),

            // THE PROVENANCE, in the same words the entry screen uses, so the
            // two never disagree about what this tier means.
            Padding(
              padding: const EdgeInsets.fromLTRB(
                AppTokens.gutter,
                AppTokens.s12,
                AppTokens.gutter,
                0,
              ),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Container(
                    width: 7,
                    height: 7,
                    margin: const EdgeInsets.only(top: 5),
                    decoration: BoxDecoration(
                      color: c.cautionMark,
                      shape: BoxShape.circle,
                    ),
                  ),
                  const SizedBox(width: AppTokens.s8),
                  Expanded(
                    child: Text(
                      'From open map data · nobody has checked it.\n\n'
                      'Somebody typed this into a public map, possibly years '
                      'ago. Call it before you rely on it.',
                      style: AppTokens.captionStyle.copyWith(
                        color: c.cautionMark,
                      ),
                    ),
                  ),
                ],
              ),
            ),

            const StencilLabel('Keep it'),
            if (alreadySaved)
              Padding(
                padding: const EdgeInsets.symmetric(
                  horizontal: AppTokens.gutter,
                ),
                child: Text(
                  'Already in your diary.',
                  style: AppTokens.captionStyle.copyWith(color: c.signal),
                ),
              )
            else
              Padding(
                padding: const EdgeInsets.symmetric(
                  horizontal: AppTokens.gutter,
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    PressScale(
                      onTap: () {
                        Haptics.light();
                        onSave(place.phones.first);
                      },
                      child: Container(
                        height: 44,
                        alignment: Alignment.center,
                        decoration: BoxDecoration(
                          color: c.signal,
                          border: Border.all(color: c.ink),
                          borderRadius: BorderRadius.circular(
                            AppTokens.radiusSoft,
                          ),
                        ),
                        child: Text(
                          'Save to the diary',
                          style: AppTokens.stencilStyle.copyWith(
                            fontSize: 10.5,
                            color: c.paper,
                          ),
                        ),
                      ),
                    ),
                    const SizedBox(height: AppTokens.s8),
                    Text(
                      // Said plainly, because this is the one place a user
                      // might expect saving to make a number trustworthy.
                      'It will keep the amber dot and still read "from open '
                      'map data". Calling it and marking it confirmed is what '
                      'changes that.',
                      style: AppTokens.captionStyle.copyWith(color: c.muted),
                    ),
                  ],
                ),
              ),
          ] else
            Padding(
              padding: const EdgeInsets.fromLTRB(
                AppTokens.gutter,
                AppTokens.s16,
                AppTokens.gutter,
                0,
              ),
              child: Text(
                'No number on the map for this one.',
                style: AppTokens.captionStyle.copyWith(color: c.muted),
              ),
            ),

          if (onOpenMaps != null) ...[
            const StencilLabel('Reviews and photos'),
            PressScale(
              onTap: () {
                Haptics.light();
                onOpenMaps!();
              },
              child: Container(
                padding: const EdgeInsets.symmetric(
                  horizontal: AppTokens.gutter,
                  vertical: AppTokens.s16,
                ),
                decoration: BoxDecoration(
                  border: Border(
                    bottom: BorderSide(
                      color: c.rule,
                      width: AppTokens.hairline,
                    ),
                  ),
                ),
                child: Row(
                  children: [
                    Icon(Icons.open_in_new, size: 18, color: c.ink),
                    const SizedBox(width: AppTokens.s16),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'Open in Google Maps',
                            style: AppTokens.rowTitleStyle.copyWith(
                              color: c.ink,
                            ),
                          ),
                          const SizedBox(height: 2),
                          Text(
                            // Said up front. A dead tap on a mountain road
                            // with no explanation is worse than no button.
                            'Needs signal. This app does not carry reviews '
                            'or photos.',
                            style: AppTokens.captionStyle.copyWith(
                              color: c.muted,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ],
      ),
    );
  }
}

class _PhoneRow extends StatelessWidget {
  final PoiContact phone;
  final VoidCallback onCopy;
  final VoidCallback onOpenDialer;

  const _PhoneRow({
    required this.phone,
    required this.onCopy,
    required this.onOpenDialer,
  });

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);

    return Container(
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
          Expanded(
            child: Text(
              phone.phoneRaw,
              style: AppTokens.numberStyle.copyWith(
                color: c.ink,
                fontSize: 16,
              ),
            ),
          ),
          // Copy first, as everywhere in this app: the dial happens in the
          // Android dialer after a paste.
          _Action(label: 'Copy', onTap: onCopy),
          const SizedBox(width: AppTokens.s8),
          _Action(label: 'Dialer', onTap: onOpenDialer),
        ],
      ),
    );
  }
}

class _Action extends StatelessWidget {
  final String label;
  final VoidCallback onTap;
  const _Action({required this.label, required this.onTap});

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    return PressScale(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(
          horizontal: AppTokens.s12,
          vertical: AppTokens.s8,
        ),
        decoration: BoxDecoration(
          border: Border.all(color: c.rule),
          borderRadius: BorderRadius.circular(AppTokens.radiusSoft),
        ),
        child: Text(
          label.toUpperCase(),
          style: AppTokens.stencilStyle.copyWith(fontSize: 10, color: c.ink),
        ),
      ),
    );
  }
}

class _Field extends StatelessWidget {
  final String label;
  final String value;
  const _Field({required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    return Container(
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
          Expanded(
            child: Text(
              label,
              style: AppTokens.captionStyle.copyWith(color: c.muted),
            ),
          ),
          Text(
            value,
            style: AppTokens.numberStyle.copyWith(color: c.ink),
          ),
        ],
      ),
    );
  }
}

/// Whether there is anything to say under "What they serve".
bool _hasFood(CorridorPlace place) =>
    kindOfFood(place.tags) != null ||
    cuisineOf(place.tags) != null ||
    vegOf(place.tags) != null ||
    jainOf(place.tags) != null ||
    hoursOf(place.tags) != null;

/// "Add to the plan for this leg" — a viewpoint, a lunch stop — or, once
/// added, the way back out. Holds its own state so the tap shows at once.
class _PlanToggle extends StatefulWidget {
  final bool planned;
  final Future<void> Function(bool plan) onToggle;
  const _PlanToggle({required this.planned, required this.onToggle});

  @override
  State<_PlanToggle> createState() => _PlanToggleState();
}

class _PlanToggleState extends State<_PlanToggle> {
  late bool _planned = widget.planned;
  bool _busy = false;

  Future<void> _tap() async {
    if (_busy) return;
    setState(() => _busy = true);
    await widget.onToggle(!_planned);
    if (!mounted) return;
    setState(() {
      _busy = false;
      _planned = !_planned;
    });
  }

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    return Padding(
      padding: const EdgeInsets.fromLTRB(
        AppTokens.gutter,
        AppTokens.s12,
        AppTokens.gutter,
        0,
      ),
      child: PressScale(
        key: const Key('poi-plan-toggle'),
        onTap: _tap,
        child: Container(
          padding: const EdgeInsets.symmetric(vertical: AppTokens.s12),
          decoration: BoxDecoration(
            color: _planned ? null : c.signal,
            border: Border.all(color: _planned ? c.rule : c.ink),
            borderRadius: BorderRadius.circular(AppTokens.radiusSoft),
          ),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(
                _planned ? Icons.check : Icons.flag_outlined,
                size: 16,
                color: _planned ? c.signal : c.paper,
              ),
              const SizedBox(width: 6),
              Text(
                _planned
                    ? 'In the plan · tap to take out'
                    : 'Add to the plan for this leg',
                style: AppTokens.stencilStyle.copyWith(
                  fontSize: 10.5,
                  color: _planned ? c.signal : c.paper,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
