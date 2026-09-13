// lib/features/weather/presentation/weather_screen.dart
//
// The frozen forecast — issue #26.
//
// THE WHOLE SCREEN IS BUILT AROUND ONE RULE: a snapshot must never look
// current. The age is stated in words beside every stop, and beyond a week it
// renders in caution. A five-day-old forecast shown as today's weather is
// worse than no forecast, because somebody packs on it.

import 'package:flutter/material.dart';

import '../../../core/theme/app_tokens.dart';
import '../../../core/database/app_database.dart';
import '../../../core/widgets/retro.dart';
import '../data/weather_client.dart';
import '../data/weather_sync.dart';

class WeatherScreen extends StatelessWidget {
  final Stream<List<StopWeather>> weather;

  /// Re-fetches everything. Offered rather than automatic, because fetching
  /// is the one moment this app touches the network and the user decides when.
  final Future<void> Function()? onRefresh;

  /// Frozen in tests and goldens so an age never drifts.
  final DateTime? now;

  const WeatherScreen({
    super.key,
    required this.weather,
    this.onRefresh,
    this.now,
  });

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);

    return Scaffold(
      backgroundColor: c.paper,
      appBar: AppBar(
        title: const Text('Weather'),
        backgroundColor: c.paper,
        foregroundColor: c.ink,
        elevation: 0,
        actions: [
          if (onRefresh != null)
            IconButton(
              onPressed: onRefresh,
              icon: const Icon(Icons.refresh),
              color: c.muted,
              tooltip: 'Fetch again, on WiFi',
            ),
        ],
      ),
      body: StreamBuilder<List<StopWeather>>(
        stream: weather,
        builder: (context, snap) {
          final stops = snap.data;
          if (stops == null) return const SizedBox();

          if (stops.every((s) => s.isEmpty)) {
            return Center(
              child: Padding(
                padding: const EdgeInsets.all(AppTokens.s32),
                child: Text(
                  'No forecast downloaded yet. Fetch it on WiFi before you '
                  'leave; after that it is frozen, and the app will tell you '
                  'how old it is.',
                  textAlign: TextAlign.center,
                  style: AppTokens.captionStyle.copyWith(color: c.muted),
                ),
              ),
            );
          }

          return ListView(
            padding: const EdgeInsets.only(bottom: AppTokens.s32),
            children: [
              for (final stop in stops) _StopBlock(stop: stop, now: now),
              const StencilLabel('About this forecast'),
              Padding(
                padding: const EdgeInsets.symmetric(
                  horizontal: AppTokens.gutter,
                ),
                child: Text(
                  'This is a picture taken once, not live weather. It does '
                  'not update on the road and it is not meant to — the app '
                  'makes no network call while you are moving.',
                  style: AppTokens.captionStyle.copyWith(color: c.muted),
                ),
              ),
            ],
          );
        },
      ),
    );
  }
}

class _StopBlock extends StatelessWidget {
  final StopWeather stop;
  final DateTime? now;

  const _StopBlock({required this.stop, this.now});

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    final staleness = stop.staleness(now: now);

    // Stale reads in caution. Ageing is muted but stated. Fresh is quiet.
    final ageColour = switch (staleness) {
      Staleness.stale => c.cautionMark,
      Staleness.ageing => c.muted,
      _ => c.muted,
    };

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Padding(
          padding: const EdgeInsets.fromLTRB(
            AppTokens.gutter,
            AppTokens.s24,
            AppTokens.gutter,
            AppTokens.s8,
          ),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [
              Expanded(
                child: Text(
                  stop.stopName,
                  style: AppTokens.titleStyle.copyWith(
                    color: c.ink,
                    fontSize: 18,
                  ),
                ),
              ),
              if (stop.cachedAt != null)
                Text(
                  // The age in words. A timestamp is not something a person
                  // weighs at a glance.
                  stop.ageDescription(now: now)!.toUpperCase(),
                  style: AppTokens.stencilStyle.copyWith(
                    fontSize: 9,
                    color: ageColour,
                  ),
                ),
            ],
          ),
        ),

        if (staleness == Staleness.stale)
          Padding(
            padding: const EdgeInsets.fromLTRB(
              AppTokens.gutter,
              0,
              AppTokens.gutter,
              AppTokens.s8,
            ),
            child: Text(
              'More than three days old. Treat this as a rough idea of the '
              'season, not a forecast.',
              style: AppTokens.captionStyle.copyWith(color: c.cautionMark),
            ),
          ),

        if (stop.isEmpty)
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: AppTokens.gutter),
            child: Text(
              'Nothing downloaded for this stop.',
              style: AppTokens.captionStyle.copyWith(color: c.muted),
            ),
          )
        else
          for (final day in stop.days) _DayRow(day: day, stale: staleness),
      ],
    );
  }
}

class _DayRow extends StatelessWidget {
  final WeatherSnapshot day;
  final Staleness? stale;

  const _DayRow({required this.day, this.stale});

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    // A stale forecast's numbers go muted too, so the whole row reads as
    // something to weigh rather than something to rely on.
    final ink = stale == Staleness.stale ? c.muted : c.ink;
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
            Text(
              'mm',
              style: AppTokens.captionStyle.copyWith(color: c.muted),
            ),
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
