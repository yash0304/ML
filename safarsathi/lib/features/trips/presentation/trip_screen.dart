// lib/features/trips/presentation/trip_screen.dart
//
// The trip, read only. SCREENS.md §4. Building and editing a trip is #16.
//
// The milestone cap carries cache state: signal green when the next leg has
// been downloaded, muted when it has not. An unprepared leg should be visible
// without reading anything.

import 'package:flutter/material.dart';

import '../../../core/theme/app_tokens.dart';
import '../../../core/widgets/retro.dart';
import '../../contacts/presentation/diary_widgets.dart';
import '../data/trip_summary.dart';

class TripScreen extends StatelessWidget {
  final Stream<TripSummary> trip;
  final Stream<int> unconfirmedCount;

  const TripScreen({
    super.key,
    required this.trip,
    required this.unconfirmedCount,
  });

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);

    return Scaffold(
      backgroundColor: c.paper,
      body: GrainOverlay(
        child: SafeArea(
          bottom: false,
          child: StreamBuilder<TripSummary>(
            stream: trip,
            builder: (context, snap) {
              final data = snap.data;
              if (data == null) return const SizedBox();
              return ListView(
                padding: const EdgeInsets.only(bottom: AppTokens.s24),
                children: [
                  _header(c, data),
                  if (data.nextLeg != null) ...[
                    const StencilLabel('Next leg'),
                    _nextLeg(c, data.nextLeg!),
                  ],
                  const SizedBox(height: AppTokens.s16),
                  ReadinessBanner(unconfirmedCount: unconfirmedCount),
                  const StencilLabel('Stops'),
                  for (final stop in data.stops) _stopTicket(c, stop),
                  _footer(c, data),
                ],
              );
            },
          ),
        ),
      ),
    );
  }

  Widget _header(AppColors c, TripSummary trip) {
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
          Text(
            _dateRange(trip).toUpperCase(),
            style: AppTokens.stencilStyle.copyWith(
              fontSize: 10,
              letterSpacing: 1.6,
              color: c.muted,
            ),
          ),
          const SizedBox(height: 2),
          Text(trip.name, style: AppTokens.titleStyle.copyWith(color: c.ink)),
          Text(
            '${trip.stops.length} '
            '${trip.stops.length == 1 ? "stop" : "stops"} · '
            '${trip.legCount} ${trip.legCount == 1 ? "leg" : "legs"}',
            style: AppTokens.captionStyle.copyWith(color: c.muted),
          ),
        ],
      ),
    );
  }

  static String _dateRange(TripSummary trip) {
    if (trip.startDate == null) return 'Not dated yet';
    final start = _date(trip.startDate!);
    if (trip.endDate == null) return start;
    return '$start – ${_date(trip.endDate!)}';
  }

  static String _date(DateTime d) {
    const months = [
      'Jan',
      'Feb',
      'Mar',
      'Apr',
      'May',
      'Jun',
      'Jul',
      'Aug',
      'Sep',
      'Oct',
      'Nov',
      'Dec',
    ];
    return '${d.day} ${months[d.month - 1]}';
  }

  Widget _nextLeg(AppColors c, LegSummary leg) {
    final cached = leg.lastSyncedAt != null;
    final km = leg.distanceKm;

    return Padding(
      padding: const EdgeInsets.fromLTRB(
        AppTokens.gutter,
        AppTokens.s4,
        AppTokens.gutter,
        0,
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          MilestoneMarker(
            numeral: km == null ? '—' : km.round().toString(),
            place: leg.toName,
            // Muted when the leg has never been synced, so an unprepared leg
            // reads at a glance.
            capColor: cached ? c.signal : c.muted,
          ),
          const SizedBox(width: AppTokens.s12),
          Expanded(
            child: Padding(
              padding: const EdgeInsets.only(top: AppTokens.s4),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    '${leg.fromName} → ${leg.toName}',
                    style: AppTokens.titleStyle.copyWith(
                      fontSize: 15,
                      color: c.ink,
                    ),
                  ),
                  const SizedBox(height: AppTokens.s4),
                  Text(
                    [
                      if (leg.mode != null) leg.mode!,
                      if (leg.plannedDeparture != null)
                        'leaves ${_time(leg.plannedDeparture!)}',
                    ].join(', '),
                    style: AppTokens.captionStyle.copyWith(color: c.muted),
                  ),
                  if (leg.note != null)
                    Padding(
                      padding: const EdgeInsets.only(top: 2),
                      child: Text(
                        leg.note!,
                        style: AppTokens.captionStyle.copyWith(color: c.muted),
                      ),
                    ),
                  const SizedBox(height: AppTokens.s8),
                  StampBadge(
                    label: cached
                        ? 'Cached ${_date(leg.lastSyncedAt!)}'
                        : 'Not downloaded',
                    inkColor: cached ? c.signal : c.muted,
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  static String _time(DateTime d) =>
      '${d.hour.toString().padLeft(2, '0')}:'
      '${d.minute.toString().padLeft(2, '0')}';

  Widget _stopTicket(AppColors c, StopSummary stop) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(
        AppTokens.gutter,
        0,
        AppTokens.gutter,
        AppTokens.s12,
      ),
      child: TicketCard(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Expanded(
                  child: Text(
                    stop.name,
                    style: AppTokens.titleStyle.copyWith(
                      fontSize: 15,
                      color: c.ink,
                    ),
                  ),
                ),
                Text(
                  stop.isCurrent
                      ? 'STOP ${stop.sequenceOrder} · HERE'
                      : 'STOP ${stop.sequenceOrder}',
                  style: AppTokens.stencilStyle.copyWith(
                    fontSize: 9,
                    color: stop.isCurrent ? c.signal : c.muted,
                  ),
                ),
              ],
            ),
            const SizedBox(height: AppTokens.s8),
            Container(height: AppTokens.hairline, color: c.rule),
            const SizedBox(height: AppTokens.s8),
            Row(
              children: [
                Expanded(
                  child: _Field(label: 'Nights', value: stop.nights.toString()),
                ),
                Expanded(
                  child: _Field(label: 'Diary', value: '${stop.diaryCount}'),
                ),
                Expanded(
                  child: _Field(
                    label: 'Arrive',
                    value: stop.arrivalDate == null
                        ? '—'
                        : _date(stop.arrivalDate!),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _footer(AppColors c, TripSummary trip) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(
        AppTokens.gutter,
        AppTokens.s8,
        AppTokens.gutter,
        0,
      ),
      child: Text(
        'Building and editing a trip comes later. This one is placeholder '
        'data so the rest of the app has something to stand on.',
        style: AppTokens.captionStyle.copyWith(color: c.muted),
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
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label.toUpperCase(),
          style: AppTokens.stencilStyle.copyWith(fontSize: 9, color: c.muted),
        ),
        const SizedBox(height: 2),
        Text(value, style: AppTokens.numberStyle.copyWith(color: c.ink)),
      ],
    );
  }
}
