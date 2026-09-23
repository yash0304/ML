// lib/features/trips/presentation/trip_screen.dart
//
// The trip, read only. SCREENS.md §4. Editing happens in the itinerary screen
// behind the EDIT rule, so the thing you look at on the road stays a thing you
// look at rather than a thing you can knock out of shape with a stray tap.
//
// The milestone cap carries cache state: signal green when the next leg has
// been downloaded, muted when it has not. An unprepared leg should be visible
// without reading anything.

import 'package:flutter/material.dart';

import '../../../core/theme/app_tokens.dart';
import '../../../core/widgets/retro.dart';
import '../../contacts/presentation/diary_widgets.dart';
import '../../../core/database/app_database.dart';
import '../../../core/util/sun.dart';
import '../data/tonight.dart';
import '../data/readiness.dart';
import '../data/trip_summary.dart';
import 'readiness_panel.dart';

class TripScreen extends StatelessWidget {
  final Stream<TripSummary> trip;
  final Stream<int> unconfirmedCount;

  /// The pre-departure block (#20). Optional, so the golden harness and the
  /// widget tests can render the screen without building one.
  final Stream<Readiness>? readiness;

  /// Opens the itinerary editor (#16).
  final VoidCallback? onEditItinerary;

  /// The downloaded map, drawn. Was four taps away behind More, which is
  /// three too many for the thing the app exists to do.
  final VoidCallback? onViewMap;

  /// The legs, and the places found along each one.
  final VoidCallback? onLegs;

  /// Where you sleep tonight (or the first night, before the trip).
  final Stream<Tonight?>? tonight;

  /// Opens a diary entry — a stay on the Tonight card.
  final void Function(Contact contact)? onOpenContact;

  /// "No stay saved here yet — add one", starting at that stop.
  final void Function(int stopId)? onAddStay;

  /// Sends the plan — each night's stop and stay — to someone at home.
  final VoidCallback? onSharePlan;

  const TripScreen({
    super.key,
    required this.trip,
    required this.unconfirmedCount,
    this.readiness,
    this.onEditItinerary,
    this.onViewMap,
    this.onLegs,
    this.tonight,
    this.onOpenContact,
    this.onAddStay,
    this.onSharePlan,
  });

  /// The section rule, with a way into the editor when one is wired.
  Widget _stopsHeader(AppColors c) {
    if (onEditItinerary == null) return const StencilLabel('Stops');
    return Row(
      children: [
        const Expanded(child: StencilLabel('Stops')),
        Padding(
          padding: const EdgeInsets.only(right: AppTokens.gutter),
          child: GestureDetector(
            onTap: onEditItinerary,
            child: Text(
              'EDIT',
              style: AppTokens.stencilStyle.copyWith(
                fontSize: 9.5,
                color: c.signal,
              ),
            ),
          ),
        ),
      ],
    );
  }

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
                  // TONIGHT FIRST. At six in the evening on a dark road the
                  // question is where the bed is and how to ring them; the
                  // next leg is tomorrow's problem.
                  if (tonight != null)
                    _TonightCard(
                      tonight: tonight!,
                      onOpen: onOpenContact,
                      onAdd: onAddStay,
                    ),
                  if (data.nextLeg != null) ...[
                    const StencilLabel('Next leg'),
                    _nextLeg(c, data.nextLeg!),
                  ],
                  if (onViewMap != null || onLegs != null) ...[
                    const StencilLabel('On the road'),
                    if (onViewMap != null)
                      _Shortcut(
                        icon: Icons.map_outlined,
                        title: 'The map',
                        subtitle: 'What you downloaded, drawn. No signal '
                            'needed.',
                        onTap: onViewMap!,
                      ),
                    if (onLegs != null)
                      _Shortcut(
                        icon: Icons.directions_bus_outlined,
                        title: 'Getting between stops',
                        subtitle: 'Each leg, how you are travelling it, and '
                            'what is along the way.',
                        onTap: onLegs!,
                      ),
                  ],
                  if (onSharePlan != null)
                    _Shortcut(
                      icon: Icons.send_outlined,
                      title: 'Send the plan home',
                      subtitle: 'Each night\'s stop and a number there that '
                          'is not yours — for when yours is out of signal.',
                      onTap: onSharePlan!,
                    ),
                  const SizedBox(height: AppTokens.s16),
                  if (readiness != null)
                    ReadinessPanel(readiness: readiness!)
                  else
                    ReadinessBanner(unconfirmedCount: unconfirmedCount),
                  _stopsHeader(c),
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
        'This screen is read only. Change the itinerary from EDIT above, or '
        'from the More tab.',
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

/// A way into another screen, from the one the app opens on.
///
/// Deliberately the same shape as the rows on More: the two lists are the
/// same kind of thing, and only their placement says which is reached often.
class _Shortcut extends StatelessWidget {
  final IconData icon;
  final String title;
  final String subtitle;
  final VoidCallback onTap;

  const _Shortcut({
    required this.icon,
    required this.title,
    required this.subtitle,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    return InkWell(
      onTap: onTap,
      child: Padding(
        padding: const EdgeInsets.fromLTRB(
          AppTokens.gutter,
          AppTokens.s12,
          AppTokens.gutter,
          AppTokens.s12,
        ),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Icon(icon, size: 20, color: c.muted),
            const SizedBox(width: AppTokens.s12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    title,
                    style: AppTokens.titleStyle.copyWith(
                      fontSize: 15,
                      color: c.ink,
                    ),
                  ),
                  const SizedBox(height: 2),
                  Text(
                    subtitle,
                    style: AppTokens.captionStyle.copyWith(color: c.muted),
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

/// Tonight's bed: the stop, the stays saved there, and when the light goes.
class _TonightCard extends StatelessWidget {
  final Stream<Tonight?> tonight;
  final void Function(Contact)? onOpen;
  final void Function(int stopId)? onAdd;

  const _TonightCard({required this.tonight, this.onOpen, this.onAdd});

  /// Two is enough on the first screen; the rest are one tap into the diary.
  static const shown = 2;

  static const _months = [
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
  ];

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    return StreamBuilder<Tonight?>(
      stream: tonight,
      builder: (context, snap) {
        final t = snap.data;
        // After the trip, or on a night with no bed on the plan, the card is
        // absent rather than pointing at somebody else's bed.
        if (t == null) return const SizedBox.shrink();

        final label = t.kind == TonightKind.tonight
            ? 'Tonight · ${t.stop.name}'
            : 'First night · ${t.stop.name}, '
                  '${t.night.day} ${_months[t.night.month - 1]}';

        return Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            StencilLabel(label),
            if (t.stays.isEmpty)
              InkWell(
                onTap: onAdd == null ? null : () => onAdd!(t.stop.id),
                child: Padding(
                  padding: const EdgeInsets.fromLTRB(
                    AppTokens.gutter,
                    0,
                    AppTokens.gutter,
                    AppTokens.s8,
                  ),
                  child: Text(
                    'No stay for ${t.stop.name} in your diary yet.'
                    '${onAdd == null ? '' : ' Add one.'}',
                    style: AppTokens.captionStyle.copyWith(
                      color: c.cautionMark,
                    ),
                  ),
                ),
              )
            else
              for (final stay in t.stays.take(shown))
                _StayRow(
                  stay: stay,
                  onTap: onOpen == null ? null : () => onOpen!(stay),
                ),
            if (t.sun?.sunset != null)
              Padding(
                padding: const EdgeInsets.fromLTRB(
                  AppTokens.gutter,
                  AppTokens.s8,
                  AppTokens.gutter,
                  0,
                ),
                child: Text(
                  'Sunset at ${t.stop.name} ${clockTime(t.sun!.sunset!)}.',
                  style: AppTokens.captionStyle.copyWith(color: c.muted),
                ),
              ),
          ],
        );
      },
    );
  }
}

class _StayRow extends StatelessWidget {
  final Contact stay;
  final VoidCallback? onTap;
  const _StayRow({required this.stay, this.onTap});

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    final note = stay.note?.trim();
    return InkWell(
      onTap: onTap,
      child: Padding(
        padding: const EdgeInsets.fromLTRB(
          AppTokens.gutter,
          AppTokens.s4,
          AppTokens.gutter,
          AppTokens.s8,
        ),
        child: Row(
          children: [
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    stay.name,
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: AppTokens.rowTitleStyle.copyWith(color: c.ink),
                  ),
                  const SizedBox(height: 2),
                  Text(
                    stay.phoneRaw,
                    style: AppTokens.numberStyle.copyWith(color: c.ink),
                  ),
                  if (note != null && note.isNotEmpty)
                    Text(
                      note,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: AppTokens.captionStyle.copyWith(color: c.muted),
                    ),
                ],
              ),
            ),
            if (!stay.callConfirmed) const TrustDot(),
          ],
        ),
      ),
    );
  }
}
