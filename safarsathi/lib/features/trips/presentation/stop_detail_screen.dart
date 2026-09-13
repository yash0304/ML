// lib/features/trips/presentation/stop_detail_screen.dart
//
// One stop — issue #51. SCREENS.md §10, Windy's snapshot pattern.
//
// The forecast opens the screen and its age opens the forecast. Under three
// days it is muted; past three days it turns caution and a sentence spells out
// what that means. A stale forecast that looks current is the failure mode
// this whole screen is designed against.

import 'package:flutter/material.dart';

import '../../../core/database/app_database.dart';
import '../../../core/theme/app_tokens.dart';
import '../../../core/theme/motion.dart';
import '../../../core/widgets/retro.dart';
import '../../weather/data/weather_client.dart';
import '../data/stop_detail.dart';
import '../data/trip_editor.dart';

class StopDetailScreen extends StatelessWidget {
  final Stream<StopDetail> detail;

  final VoidCallback? onEdit;
  final VoidCallback? onOpenDiary;
  final VoidCallback? onOpenChecklist;

  /// Changing tags here regenerates the checklist, which is why the link is
  /// stated on screen rather than left implied.
  final Future<void> Function(List<String> tags)? onTags;

  /// Frozen in tests and goldens so an age never drifts.
  final DateTime? now;

  const StopDetailScreen({
    super.key,
    required this.detail,
    this.onEdit,
    this.onOpenDiary,
    this.onOpenChecklist,
    this.onTags,
    this.now,
  });

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);

    return StreamBuilder<StopDetail>(
      stream: detail,
      builder: (context, snap) {
        final data = snap.data;

        return Scaffold(
          backgroundColor: c.paper,
          appBar: AppBar(
            title: Text(data?.stop.name ?? 'Stop'),
            backgroundColor: c.paper,
            foregroundColor: c.ink,
            elevation: 0,
            actions: [
              if (onEdit != null)
                IconButton(
                  onPressed: onEdit,
                  icon: const Icon(Icons.edit_outlined),
                  color: c.muted,
                ),
            ],
          ),
          body: data == null
              ? const SizedBox()
              : ListView(
                  padding: const EdgeInsets.only(bottom: AppTokens.s32),
                  children: [
                    _Header(detail: data),
                    _WeatherSection(detail: data, now: now),
                    _WhatIsHere(
                      detail: data,
                      onOpenDiary: onOpenDiary,
                      onOpenChecklist: onOpenChecklist,
                    ),
                    _Tags(detail: data, onTags: onTags),
                    _Cached(detail: data),
                  ],
                ),
        );
      },
    );
  }
}

class _Header extends StatelessWidget {
  final StopDetail detail;
  const _Header({required this.detail});

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    final arrival = detail.stop.arrivalDate;

    return Padding(
      padding: const EdgeInsets.fromLTRB(
        AppTokens.gutter,
        AppTokens.s8,
        AppTokens.gutter,
        0,
      ),
      child: Text(
        [
          if (arrival != null)
            '${arrival.day.toString().padLeft(2, '0')}/'
                '${arrival.month.toString().padLeft(2, '0')}',
          detail.isOvernight
              ? '${detail.stop.nights} '
                    '${detail.stop.nights == 1 ? 'night' : 'nights'}'
              : 'passing through',
          if (!detail.hasCoordinates) 'no coordinates yet',
        ].join(' · '),
        style: AppTokens.captionStyle.copyWith(
          color: detail.hasCoordinates ? c.muted : c.cautionMark,
        ),
      ),
    );
  }
}

class _WeatherSection extends StatelessWidget {
  final StopDetail detail;
  final DateTime? now;
  const _WeatherSection({required this.detail, this.now});

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    final staleness = detail.weatherStaleness(now: now);
    final stale = staleness == Staleness.stale;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Row(
          children: [
            const Expanded(child: StencilLabel('Weather')),
            if (detail.weatherCachedAt != null)
              Padding(
                padding: const EdgeInsets.only(right: AppTokens.gutter),
                child: Text(
                  detail.weatherAge(now: now)!.toUpperCase(),
                  style: AppTokens.stencilStyle.copyWith(
                    fontSize: 9,
                    color: stale ? c.cautionMark : c.muted,
                  ),
                ),
              ),
          ],
        ),

        if (detail.weather.isEmpty)
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: AppTokens.gutter),
            child: Text(
              'No forecast downloaded for this stop.',
              style: AppTokens.captionStyle.copyWith(color: c.muted),
            ),
          )
        else ...[
          if (stale)
            Padding(
              padding: const EdgeInsets.fromLTRB(
                AppTokens.gutter,
                0,
                AppTokens.gutter,
                AppTokens.s8,
              ),
              child: Text(
                'More than three days old. Weather here turns over faster '
                'than that — read it as the season, not the day.',
                style: AppTokens.captionStyle.copyWith(color: c.cautionMark),
              ),
            ),
          for (final day in detail.weather) _DayRow(day: day, stale: stale),
        ],
      ],
    );
  }
}

class _DayRow extends StatelessWidget {
  final WeatherSnapshot day;
  final bool stale;
  const _DayRow({required this.day, required this.stale});

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    final ink = stale ? c.muted : c.ink;
    final d = day.forDate;

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
          SizedBox(
            width: 54,
            child: Text(
              '${d.day.toString().padLeft(2, '0')}/'
              '${d.month.toString().padLeft(2, '0')}',
              style: AppTokens.numberStyle.copyWith(color: c.muted),
            ),
          ),
          Expanded(
            child: Text(
              day.condition,
              style: AppTokens.rowTitleStyle.copyWith(color: ink),
            ),
          ),
          if (day.rainMm != null && day.rainMm! > 0) ...[
            Text(
              '${day.rainMm!.round()}',
              style: AppTokens.numberStyle.copyWith(color: ink),
            ),
            Text('mm', style: AppTokens.captionStyle.copyWith(color: c.muted)),
            const SizedBox(width: AppTokens.s12),
          ],
          if (day.tempMinC != null && day.tempMaxC != null)
            Text(
              '${day.tempMinC!.round()}–${day.tempMaxC!.round()}°',
              style: AppTokens.numberStyle.copyWith(color: ink),
            ),
        ],
      ),
    );
  }
}

class _WhatIsHere extends StatelessWidget {
  final StopDetail detail;
  final VoidCallback? onOpenDiary;
  final VoidCallback? onOpenChecklist;

  const _WhatIsHere({
    required this.detail,
    this.onOpenDiary,
    this.onOpenChecklist,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        const StencilLabel('What is here'),
        _CountRow(
          label: 'Diary entries',
          value: '${detail.diaryCount}',
          // The amber count, because this is the number that decides whether
          // the stop blocks departure.
          warning: detail.unconfirmedCount > 0
              ? '${detail.unconfirmedCount} unconfirmed'
              : null,
          onTap: onOpenDiary,
        ),
        _CountRow(
          label: 'Checklist',
          value: detail.checklistCount == 0
              ? '0'
              : '${detail.checklistDone} of ${detail.checklistCount}',
          onTap: onOpenChecklist,
        ),
        _CountRow(
          label: 'Places on the roads either side',
          value: '${detail.nearbyPlaceCount}',
        ),
      ],
    );
  }
}

class _CountRow extends StatelessWidget {
  final String label;
  final String value;
  final String? warning;
  final VoidCallback? onTap;

  const _CountRow({
    required this.label,
    required this.value,
    this.warning,
    this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    return PressScale(
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
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    label,
                    style: AppTokens.rowTitleStyle.copyWith(color: c.ink),
                  ),
                  if (warning != null)
                    Padding(
                      padding: const EdgeInsets.only(top: 2),
                      child: Text(
                        warning!,
                        style: AppTokens.captionStyle.copyWith(
                          color: c.cautionMark,
                        ),
                      ),
                    ),
                ],
              ),
            ),
            Text(
              value,
              style: AppTokens.numberStyle.copyWith(color: c.ink),
            ),
            // The chevron's space is reserved whether or not there is one, so
            // the counts read as a column rather than as three numbers that
            // happen to be near the right edge.
            Padding(
              padding: const EdgeInsets.only(left: AppTokens.s8),
              child: onTap == null
                  ? const SizedBox(width: 18)
                  : Icon(Icons.chevron_right, size: 18, color: c.muted),
            ),
          ],
        ),
      ),
    );
  }
}

class _Tags extends StatelessWidget {
  final StopDetail detail;
  final Future<void> Function(List<String>)? onTags;

  const _Tags({required this.detail, this.onTags});

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    final tags = detail.tags;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        const StencilLabel('What happens here'),
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: AppTokens.gutter),
          child: Wrap(
            spacing: AppTokens.s8,
            runSpacing: AppTokens.s8,
            children: [
              for (final tag in knownActivityTags)
                _Tag(
                  label: tag,
                  on: tags.contains(tag),
                  onTap: onTags == null
                      ? null
                      : () {
                          Haptics.select();
                          final next = List.of(tags);
                          next.contains(tag)
                              ? next.remove(tag)
                              : next.add(tag);
                          onTags!(next);
                        },
                ),
            ],
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
            // THE LINK IS MADE VISIBLE rather than left implied. Editing tags
            // here is how the packing list changes, and nobody would guess
            // that from a row of chips.
            'These build the packing checklist. Adding "caves" puts a '
            'headtorch on it; anything you have already changed by hand '
            'stays as you left it.',
            style: AppTokens.captionStyle.copyWith(color: c.muted),
          ),
        ),
      ],
    );
  }
}

class _Tag extends StatelessWidget {
  final String label;
  final bool on;
  final VoidCallback? onTap;
  const _Tag({required this.label, required this.on, this.onTap});

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
          color: on ? c.signal : Colors.transparent,
          border: Border.all(color: on ? c.ink : c.rule),
          borderRadius: BorderRadius.circular(AppTokens.radiusSoft),
        ),
        child: Text(
          label.toUpperCase(),
          style: AppTokens.stencilStyle.copyWith(
            fontSize: 10,
            color: on ? c.paper : c.muted,
          ),
        ),
      ),
    );
  }
}

class _Cached extends StatelessWidget {
  final StopDetail detail;
  const _Cached({required this.detail});

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    final synced = detail.lastSyncedAt;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        const StencilLabel('Cached here'),
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: AppTokens.gutter),
          child: Text(
            synced == null
                ? 'The roads either side of this stop have not been '
                      'downloaded yet.'
                : 'Roads either side downloaded '
                      '${synced.day.toString().padLeft(2, '0')}/'
                      '${synced.month.toString().padLeft(2, '0')}/'
                      '${synced.year}, with ${detail.nearbyPlaceCount} '
                      'places along them.',
            style: AppTokens.captionStyle.copyWith(
              color: synced == null ? c.cautionMark : c.muted,
            ),
          ),
        ),
      ],
    );
  }
}
