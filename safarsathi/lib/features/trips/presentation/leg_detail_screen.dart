// lib/features/trips/presentation/leg_detail_screen.dart
//
// One leg — issue #52. SCREENS.md §11, Rome2Rio reduced.
//
// Rome2Rio was one of the five apps this project set out to absorb and the one
// that could not be: it depends on live operator schedule databases. What
// survives is this — what you were told, typed by you, plus the road itself
// and what is on it.

import 'package:flutter/material.dart';

import '../../../core/database/app_database.dart';
import '../../../core/theme/app_tokens.dart';
import '../../../core/theme/motion.dart';
import '../../../core/widgets/retro.dart';
import '../../contacts/data/contacts_dao.dart';
import '../../discovery/data/discovery.dart';

class LegDetailScreen extends StatelessWidget {
  final Stream<LegDiscovery> discovery;

  /// Typed transport details, which live on the leg row itself.
  final Stream<LegTransport> transport;

  final VoidCallback? onEditTransport;
  final VoidCallback? onSeeAll;
  final void Function(CorridorPlace place)? onOpenPlace;

  /// How many places to show inline before deferring to the full list.
  static const inlineLimit = 5;

  const LegDetailScreen({
    super.key,
    required this.discovery,
    required this.transport,
    this.onEditTransport,
    this.onSeeAll,
    this.onOpenPlace,
  });

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);

    return StreamBuilder<LegDiscovery>(
      stream: discovery,
      builder: (context, snap) {
        final leg = snap.data;

        return Scaffold(
          backgroundColor: c.paper,
          appBar: AppBar(
            title: Text(
              leg == null ? 'Leg' : '${leg.fromName} → ${leg.toName}',
            ),
            backgroundColor: c.paper,
            foregroundColor: c.ink,
            elevation: 0,
          ),
          body: leg == null
              ? const SizedBox()
              : ListView(
                  padding: const EdgeInsets.only(bottom: AppTokens.s32),
                  children: [
                    _Transport(
                      transport: transport,
                      onEdit: onEditTransport,
                    ),
                    _OnTheRoad(
                      leg: leg,
                      onOpenPlace: onOpenPlace,
                      onSeeAll: onSeeAll,
                    ),
                    _Cached(leg: leg),
                  ],
                ),
        );
      },
    );
  }
}

/// The typed half of a leg, as the screen needs it.
class LegTransport {
  final String? mode;
  final DateTime? plannedDeparture;
  final DateTime? plannedArrival;
  final bool isBooked;
  final String? note;

  const LegTransport({
    this.mode,
    this.plannedDeparture,
    this.plannedArrival,
    this.isBooked = false,
    this.note,
  });

  bool get isEmpty =>
      mode == null &&
      plannedDeparture == null &&
      plannedArrival == null &&
      note == null;

  factory LegTransport.fromRow(Leg leg) => LegTransport(
    mode: leg.mode,
    plannedDeparture: leg.plannedDeparture,
    plannedArrival: leg.plannedArrival,
    isBooked: leg.isBooked,
    note: leg.note,
  );
}

/// The typed half of one leg. Separate from [watchLegDiscovery] because the
/// two halves change for different reasons: this one only when a person edits
/// it, the other when a download lands.
Stream<LegTransport> watchLegTransport(AppDatabase db, int legId) =>
    (db.select(db.legs)..where((l) => l.id.equals(legId)))
        .watchSingle()
        .map(LegTransport.fromRow);

class _Transport extends StatelessWidget {
  final Stream<LegTransport> transport;
  final VoidCallback? onEdit;

  const _Transport({required this.transport, this.onEdit});

  static String _time(DateTime d) =>
      '${d.day.toString().padLeft(2, '0')}/'
      '${d.month.toString().padLeft(2, '0')}  '
      '${d.hour.toString().padLeft(2, '0')}:'
      '${d.minute.toString().padLeft(2, '0')}';

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);

    return StreamBuilder<LegTransport>(
      stream: transport,
      builder: (context, snap) {
        final t = snap.data ?? const LegTransport();

        return Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Row(
              children: [
                const Expanded(child: StencilLabel('Transport')),
                if (onEdit != null)
                  Padding(
                    padding: const EdgeInsets.only(right: AppTokens.gutter),
                    child: GestureDetector(
                      onTap: onEdit,
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
            ),

            if (t.isEmpty)
              Padding(
                padding: const EdgeInsets.symmetric(
                  horizontal: AppTokens.gutter,
                ),
                child: Text(
                  'Nothing recorded for this leg yet.',
                  style: AppTokens.captionStyle.copyWith(color: c.muted),
                ),
              )
            else ...[
              _Field(label: 'Mode', value: t.mode ?? '—'),
              _Field(
                label: 'Departs',
                value: t.plannedDeparture == null
                    ? '—'
                    : _time(t.plannedDeparture!),
              ),
              _Field(
                label: 'Arrives',
                value: t.plannedArrival == null
                    ? '—'
                    : _time(t.plannedArrival!),
              ),
              _Field(label: 'Booked', value: t.isBooked ? 'Yes' : 'Not yet'),
              if (t.note != null)
                Padding(
                  padding: const EdgeInsets.fromLTRB(
                    AppTokens.gutter,
                    AppTokens.s12,
                    AppTokens.gutter,
                    0,
                  ),
                  child: Text(
                    t.note!,
                    style: AppTokens.captionStyle.copyWith(color: c.ink),
                  ),
                ),
            ],

            Padding(
              padding: const EdgeInsets.fromLTRB(
                AppTokens.gutter,
                AppTokens.s8,
                AppTokens.gutter,
                0,
              ),
              child: Text(
                // SAID, rather than implying a lookup that cannot happen
                // offline. Rome2Rio needs live operator databases; this app
                // holds what you were told.
                'Typed by you. There is no timetable to look up — this app '
                'makes no network call on the road.',
                style: AppTokens.captionStyle.copyWith(color: c.muted),
              ),
            ),
          ],
        );
      },
    );
  }
}

class _OnTheRoad extends StatelessWidget {
  final LegDiscovery leg;
  final void Function(CorridorPlace)? onOpenPlace;
  final VoidCallback? onSeeAll;

  const _OnTheRoad({required this.leg, this.onOpenPlace, this.onSeeAll});

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    final shown = leg.places.take(LegDetailScreen.inlineLimit).toList();
    final more = leg.places.length - shown.length;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Row(
          children: [
            const Expanded(child: StencilLabel('On the road')),
            if (onSeeAll != null && leg.places.isNotEmpty)
              Padding(
                padding: const EdgeInsets.only(right: AppTokens.gutter),
                child: GestureDetector(
                  onTap: onSeeAll,
                  child: Text(
                    'SEE ALL',
                    style: AppTokens.stencilStyle.copyWith(
                      fontSize: 9.5,
                      color: c.signal,
                    ),
                  ),
                ),
              ),
          ],
        ),

        if (leg.places.isEmpty)
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: AppTokens.gutter),
            child: Text(
              leg.isSynced
                  ? 'Nothing tagged along this road in OpenStreetMap.'
                  : 'This leg has not been downloaded yet.',
              style: AppTokens.captionStyle.copyWith(
                color: leg.isSynced ? c.muted : c.cautionMark,
              ),
            ),
          )
        else ...[
          for (final (i, place) in shown.indexed)
            _PlaceRow(
              place: place,
              first: i == 0,
              // The rail stays closed at the bottom when more places follow
              // further along, because the road does.
              last: i == shown.length - 1 && more == 0,
              onTap: onOpenPlace == null ? null : () => onOpenPlace!(place),
            ),
          if (more > 0)
            Padding(
              padding: const EdgeInsets.fromLTRB(
                AppTokens.gutter,
                AppTokens.s12,
                AppTokens.gutter,
                0,
              ),
              child: Text(
                '$more more further along.',
                style: AppTokens.captionStyle.copyWith(color: c.muted),
              ),
            ),
        ],
      ],
    );
  }
}

class _PlaceRow extends StatelessWidget {
  final CorridorPlace place;
  final VoidCallback? onTap;
  final bool first;
  final bool last;

  const _PlaceRow({
    required this.place,
    required this.first,
    required this.last,
    this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);

    return PressScale(
      onTap: onTap,
      child: Container(
        // Horizontal padding only. The vertical padding belongs to the text
        // columns, so the milestone rail can run the full height of the row
        // and meet the rail of the row below it across the hairline.
        padding: const EdgeInsets.symmetric(horizontal: AppTokens.gutter),
        decoration: BoxDecoration(
          border: Border(
            bottom: BorderSide(color: c.rule, width: AppTokens.hairline),
          ),
        ),
        child: IntrinsicHeight(
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // "Coming up in 12 km" is more useful while moving than a
              // straight-line distance on a map.
              SizedBox(
                width: 48,
                child: Align(
                  alignment: Alignment.centerRight,
                  child: Text(
                    '${place.alongRouteKm.round()} km',
                    style: AppTokens.numberStyle.copyWith(color: c.muted),
                  ),
                ),
              ),
              // EACH ON ITS OWN MILESTONE MARKER, per SCREENS.md §11. The road
              // runs down the page and the places hang off it in order, which
              // is the one thing a flat list cannot say.
              _Milestone(first: first, last: last),
              Expanded(
                child: Padding(
                  padding: const EdgeInsets.symmetric(
                    vertical: AppTokens.s12,
                  ),
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        place.name,
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: AppTokens.rowTitleStyle.copyWith(color: c.ink),
                      ),
                      const SizedBox(height: 2),
                      Text(
                        ContactCategory.labels[place.category] ??
                            place.category,
                        style: AppTokens.stencilStyle.copyWith(
                          fontSize: 9,
                          color: c.muted,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
              // ANYTHING CARRYING A NUMBER FROM OPEN MAP DATA KEEPS THE AMBER
              // DOT. Nothing community-contributed is ever shown as verified.
              if (place.hasPhone)
                Align(
                  alignment: Alignment.centerRight,
                  child: Container(
                    width: 7,
                    height: 7,
                    decoration: BoxDecoration(
                      color: c.cautionMark,
                      shape: BoxShape.circle,
                    ),
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }
}

/// The rail the places hang off: a hairline down the page with a stop notch
/// at each one. Open at the top of the first row and the bottom of the last,
/// because the road carries on past both ends of what is listed.
class _Milestone extends StatelessWidget {
  final bool first;
  final bool last;
  const _Milestone({required this.first, required this.last});

  static const width = 28.0;
  static const _notch = 7.0;

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    return SizedBox(
      width: width,
      child: Stack(
        alignment: Alignment.center,
        children: [
          Positioned.fill(
            child: Align(
              alignment: Alignment.center,
              child: CustomPaint(
                painter: _RailPainter(
                  colour: c.rule,
                  openTop: first,
                  openBottom: last,
                ),
                child: const SizedBox.expand(),
              ),
            ),
          ),
          Container(
            width: _notch,
            height: _notch,
            decoration: BoxDecoration(
              color: c.paper,
              border: Border.all(color: c.ink, width: 1.2),
            ),
          ),
        ],
      ),
    );
  }
}

class _RailPainter extends CustomPainter {
  final Color colour;
  final bool openTop;
  final bool openBottom;

  const _RailPainter({
    required this.colour,
    required this.openTop,
    required this.openBottom,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final x = size.width / 2;
    final mid = size.height / 2;
    final paint = Paint()
      ..color = colour
      ..strokeWidth = 1;
    if (!openTop) canvas.drawLine(Offset(x, 0), Offset(x, mid), paint);
    if (!openBottom) {
      canvas.drawLine(Offset(x, mid), Offset(x, size.height), paint);
    }
  }

  @override
  bool shouldRepaint(_RailPainter old) =>
      old.colour != colour ||
      old.openTop != openTop ||
      old.openBottom != openBottom;
}

class _Cached extends StatelessWidget {
  final LegDiscovery leg;
  const _Cached({required this.leg});

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    final synced = leg.lastSyncedAt;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        const StencilLabel('Cached for this leg'),
        _Field(
          label: 'Route',
          value: leg.distanceKm == null
              ? 'not downloaded'
              : '${leg.distanceKm!.round()} km',
        ),
        _Field(label: 'Places', value: '${leg.places.length}'),
        _Field(
          label: 'Downloaded',
          value: synced == null
              ? 'never'
              : '${synced.day.toString().padLeft(2, '0')}/'
                    '${synced.month.toString().padLeft(2, '0')}/'
                    '${synced.year}',
        ),
        Padding(
          padding: const EdgeInsets.fromLTRB(
            AppTokens.gutter,
            AppTokens.s12,
            AppTokens.gutter,
            0,
          ),
          child: Text(
            // Which is also how you decide what to delete when the phone
            // fills up, per SCREENS.md §11.
            'Map tiles are counted for the whole trip rather than per leg — '
            'adjacent legs share the ground between them. Settings has the '
            'total.',
            style: AppTokens.captionStyle.copyWith(color: c.muted),
          ),
        ),
      ],
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
          Text(value, style: AppTokens.numberStyle.copyWith(color: c.ink)),
        ],
      ),
    );
  }
}
