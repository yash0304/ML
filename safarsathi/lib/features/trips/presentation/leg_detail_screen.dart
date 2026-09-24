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
import '../../../core/util/sun.dart';
import '../../contacts/data/contacts_dao.dart';
import '../../contacts/presentation/diary_widgets.dart' show TrustDot;
import '../../discovery/data/place_details.dart' show foodLine;
import '../../discovery/data/discovery.dart';
import '../data/planned_stops.dart' show PlannedOnTheWay;
import '../../discovery/data/poi_category.dart' show placeCategoryLabel;

class LegDetailScreen extends StatelessWidget {
  final Stream<LegDiscovery> discovery;

  /// Typed transport details, which live on the leg row itself.
  final Stream<LegTransport> transport;

  final VoidCallback? onEditTransport;
  final VoidCallback? onSeeAll;
  final void Function(CorridorPlace place)? onOpenPlace;

  /// Opens one of the user's own diary entries, where calling, copying and
  /// confirming already live.
  final void Function(Contact contact)? onOpenContact;

  /// Adds a stop on the way by name. Null hides the button.
  final VoidCallback? onAddPlanned;

  /// Choose or change who is driving this leg.
  final VoidCallback? onChooseDriver;

  /// Opens the driver's diary entry, to call them.
  final void Function(Contact driver)? onOpenDriver;

  /// Takes a planned stop out of the plan.
  final Future<void> Function(PlannedStop stop)? onRemovePlanned;

  /// Directions to a planned stop that has a position.
  final void Function(PlannedStop stop)? onDirectionsTo;

  /// How many places to show inline before deferring to the full list.
  static const inlineLimit = 5;

  /// How many destination numbers to show before pointing at the diary.
  /// Enough for the help-first ones to all be visible; not so many that a
  /// big town pushes the road off the screen.
  static const destinationLimit = 8;

  const LegDetailScreen({
    super.key,
    required this.discovery,
    required this.transport,
    this.onEditTransport,
    this.onSeeAll,
    this.onOpenPlace,
    this.onOpenContact,
    this.onAddPlanned,
    this.onChooseDriver,
    this.onOpenDriver,
    this.onRemovePlanned,
    this.onDirectionsTo,
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
                      onChooseDriver: onChooseDriver,
                      onOpenDriver: onOpenDriver,
                    ),
                    // WHAT YOU MEAN TO STOP FOR, right under how you are
                    // travelling: it is the plan for the day, and what goes
                    // home in the message.
                    _Planned(
                      leg: leg,
                      onAdd: onAddPlanned,
                      onRemove: onRemovePlanned,
                      onDirections: onDirectionsTo,
                    ),
                    // BETWEEN THE STOPS FIRST, THEN THE DESTINATION. Your own
                    // numbers on the way lead, being the ones chosen and
                    // imported on purpose; the map's places along the road
                    // follow directly. The destination's list used to sit in
                    // between — a town's worth of numbers — and pushed what
                    // is actually on the road off the screen: "I am not able
                    // to see many stops between two stops."
                    _YourNumbersOnTheWay(leg: leg, onOpen: onOpenContact),
                    _OnTheRoad(
                      leg: leg,
                      onOpenPlace: onOpenPlace,
                      onSeeAll: onSeeAll,
                    ),
                    _YourNumbersAtDestination(
                      leg: leg,
                      onOpen: onOpenContact,
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

  /// The light at each end, on the day of each end. Null where a stop has no
  /// coordinates or there is no date to reckon from.
  final SunTimes? departSun;
  final SunTimes? arriveSun;

  /// Where the arrival sunset is measured, for the row that shows it.
  final String? toName;

  /// Who is taking you, and in what.
  final Contact? driver;
  final String? vehicleNumber;

  const LegTransport({
    this.mode,
    this.plannedDeparture,
    this.plannedArrival,
    this.isBooked = false,
    this.note,
    this.departSun,
    this.arriveSun,
    this.toName,
    this.driver,
    this.vehicleNumber,
  });

  /// "Leaves 40 min after sunset — in the dark." Null when the times are fine
  /// or unknown.
  String? get departureWarning => plannedDeparture == null || departSun == null
      ? null
      : daylightWarning(plannedDeparture!, departSun!, verb: 'Leaves');

  String? get arrivalWarning => plannedArrival == null || arriveSun == null
      ? null
      : daylightWarning(plannedArrival!, arriveSun!, verb: 'Arrives');

  bool get isEmpty =>
      mode == null &&
      plannedDeparture == null &&
      plannedArrival == null &&
      note == null &&
      vehicleNumber == null;

  factory LegTransport.fromRow(
    Leg leg, {
    Stop? from,
    Stop? to,
    Contact? driver,
  }) {
    SunTimes? sunAt(Stop? stop, DateTime? day) =>
        stop?.lat == null || stop?.lon == null || day == null
        ? null
        : sunTimes(stop!.lat!, stop.lon!, day);

    return LegTransport(
      mode: leg.mode,
      plannedDeparture: leg.plannedDeparture,
      plannedArrival: leg.plannedArrival,
      isBooked: leg.isBooked,
      note: leg.note,
      departSun: sunAt(from, leg.plannedDeparture),
      // With no arrival time typed, the day of departure — or failing that
      // the day the destination stop begins — still gives a sunset worth
      // knowing before setting off.
      arriveSun: sunAt(
        to,
        leg.plannedArrival ?? leg.plannedDeparture ?? to?.arrivalDate,
      ),
      toName: to?.name,
      driver: driver,
      vehicleNumber: leg.vehicleNumber,
    );
  }
}

/// The typed half of one leg. Separate from [watchLegDiscovery] because the
/// two halves change for different reasons: this one only when a person edits
/// it, the other when a download lands.
///
/// Reads stops too, for their coordinates, so it ticks on both tables: a
/// stream that watched only `legs` would keep an old sunset after a stop was
/// moved.
Stream<LegTransport> watchLegTransport(AppDatabase db, int legId) => db
    // Contacts too: the driver's name and number come from the diary.
    .customSelect('SELECT 1', readsFrom: {db.legs, db.stops, db.contacts})
    .watch()
    .asyncMap((_) async {
      final leg = await (db.select(
        db.legs,
      )..where((l) => l.id.equals(legId))).getSingle();
      final stops = await (db.select(
        db.stops,
      )..where((s) => s.id.isIn([leg.fromStopId, leg.toStopId]))).get();
      final byId = {for (final s in stops) s.id: s};
      return LegTransport.fromRow(
        leg,
        from: byId[leg.fromStopId],
        to: byId[leg.toStopId],
        driver: leg.driverContactId == null
            ? null
            : await (db.select(db.contacts)
                    ..where((c) => c.id.equals(leg.driverContactId!)))
                  .getSingleOrNull(),
      );
    });

class _Transport extends StatelessWidget {
  final Stream<LegTransport> transport;
  final VoidCallback? onEdit;
  final VoidCallback? onChooseDriver;
  final void Function(Contact driver)? onOpenDriver;

  const _Transport({
    required this.transport,
    this.onEdit,
    this.onChooseDriver,
    this.onOpenDriver,
  });

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
              if (t.vehicleNumber != null)
                _Field(label: 'Vehicle', value: t.vehicleNumber!),
            ],
            // WHO IS TAKING YOU — shown even with nothing else typed: the
            // driver is often known before the times are.
            _DriverRow(
              driver: t.driver,
              onChoose: onChooseDriver,
              onOpen: onOpenDriver,
            ),
            // THE LIGHT, SHOWN EVEN WITH NOTHING TYPED. Whether the leg can
            // be done before dark is worth knowing before the times are
            // decided — that is when it changes the decision.
            if (t.arriveSun?.sunset != null)
              _Field(
                label: t.toName == null ? 'Sunset' : 'Sunset, ${t.toName}',
                value: clockTime(t.arriveSun!.sunset!),
              ),
            for (final warning in [t.departureWarning, t.arrivalWarning])
              if (warning != null)
                Padding(
                  padding: const EdgeInsets.fromLTRB(
                    AppTokens.gutter,
                    AppTokens.s8,
                    AppTokens.gutter,
                    0,
                  ),
                  child: Text(
                    warning,
                    style: AppTokens.captionStyle.copyWith(
                      color: c.cautionMark,
                    ),
                  ),
                ),
            if (!t.isEmpty) ...[
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
                        // "Fast food · Indian, Momo · Veg options" when the
                        // map knows it — what decides the stop — and the bare
                        // category when it does not.
                        foodLine(place.tags) ??
                            placeCategoryLabel(place.category),
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
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

class _DriverRow extends StatelessWidget {
  final Contact? driver;
  final VoidCallback? onChoose;
  final void Function(Contact)? onOpen;

  const _DriverRow({this.driver, this.onChoose, this.onOpen});

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    final d = driver;
    if (d == null && onChoose == null) return const SizedBox.shrink();

    return Container(
      padding: const EdgeInsets.fromLTRB(
        AppTokens.gutter,
        AppTokens.s12,
        AppTokens.gutter,
        AppTokens.s12,
      ),
      decoration: BoxDecoration(
        border: Border(
          bottom: BorderSide(color: c.rule, width: AppTokens.hairline),
        ),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Expanded(
            child: d == null
                ? InkWell(
                    key: const Key('leg-choose-driver'),
                    onTap: onChoose,
                    child: Text(
                      'Who is taking you? Choose the driver from your diary.',
                      style: AppTokens.captionStyle.copyWith(
                        color: c.cautionMark,
                      ),
                    ),
                  )
                : InkWell(
                    key: const Key('leg-open-driver'),
                    onTap: onOpen == null ? null : () => onOpen!(d),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Driver',
                          style: AppTokens.captionStyle.copyWith(
                            color: c.muted,
                          ),
                        ),
                        Row(
                          children: [
                            Flexible(
                              child: Text(
                                d.name,
                                style: AppTokens.rowTitleStyle.copyWith(
                                  color: c.ink,
                                ),
                              ),
                            ),
                            if (!d.callConfirmed) const TrustDot(),
                          ],
                        ),
                        Text(
                          d.phoneRaw,
                          style: AppTokens.numberStyle.copyWith(color: c.ink),
                        ),
                      ],
                    ),
                  ),
          ),
          if (d != null && onChoose != null)
            PressScale(
              key: const Key('leg-change-driver'),
              onTap: onChoose,
              child: Text(
                'CHANGE',
                style: AppTokens.stencilStyle.copyWith(
                  fontSize: 9.5,
                  color: c.signal,
                ),
              ),
            ),
        ],
      ),
    );
  }
}

/// Stops planned on this road — picked from the places along it, or typed.
class _Planned extends StatelessWidget {
  final LegDiscovery leg;
  final VoidCallback? onAdd;
  final Future<void> Function(PlannedStop)? onRemove;
  final void Function(PlannedStop)? onDirections;

  const _Planned({
    required this.leg,
    this.onAdd,
    this.onRemove,
    this.onDirections,
  });

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    final planned = leg.planned;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Row(
          children: [
            const Expanded(child: StencilLabel('Planned stops on the way')),
            if (onAdd != null)
              Padding(
                padding: const EdgeInsets.only(
                  right: AppTokens.gutter,
                  top: AppTokens.s16,
                ),
                child: PressScale(
                  key: const Key('leg-add-planned'),
                  onTap: onAdd,
                  child: Text(
                    'ADD',
                    style: AppTokens.stencilStyle.copyWith(
                      fontSize: 10.5,
                      color: c.signal,
                    ),
                  ),
                ),
              ),
          ],
        ),
        if (planned.isEmpty)
          const _Explain(
            'None yet. Tap a viewpoint or any place under "On the road" and '
            'add it to the plan, or add one by name. They go home in the '
            'plan you send.',
          )
        else
          for (final p in planned)
            _PlannedRow(
              planned: p,
              onRemove: onRemove == null ? null : () => onRemove!(p.stop),
              onDirections:
                  onDirections == null ||
                      p.stop.lat == null ||
                      p.stop.lon == null
                  ? null
                  : () => onDirections!(p.stop),
            ),
      ],
    );
  }
}

class _PlannedRow extends StatelessWidget {
  final PlannedOnTheWay planned;
  final VoidCallback? onRemove;
  final VoidCallback? onDirections;

  const _PlannedRow({required this.planned, this.onRemove, this.onDirections});

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    final s = planned.stop;
    final km = planned.alongRouteKm;
    final detail = [
      if (km != null) '${km.round()} km in',
      if (s.category != null) placeCategoryLabel(s.category!),
      if (s.note != null) s.note!,
    ].join(' · ');

    return InkWell(
      onTap: onDirections,
      child: Padding(
        padding: const EdgeInsets.fromLTRB(
          AppTokens.gutter,
          AppTokens.s8,
          AppTokens.s8,
          AppTokens.s8,
        ),
        child: Row(
          children: [
            Icon(Icons.flag_outlined, size: 18, color: c.signal),
            const SizedBox(width: AppTokens.s12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    s.name,
                    style: AppTokens.rowTitleStyle.copyWith(color: c.ink),
                  ),
                  if (detail.isNotEmpty)
                    Text(
                      detail,
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                      style: AppTokens.captionStyle.copyWith(color: c.muted),
                    ),
                ],
              ),
            ),
            if (onDirections != null)
              Icon(Icons.directions, size: 18, color: c.muted),
            if (onRemove != null)
              IconButton(
                key: Key('planned-remove-${s.id}'),
                tooltip: 'Take out of the plan',
                onPressed: onRemove,
                icon: Icon(Icons.close, size: 18, color: c.muted),
              ),
          ],
        ),
      ),
    );
  }
}

/// The user's own numbers along this leg, on the same milestone rail as the
/// map's places, so "the hospital is 38 km in" reads the way the road does.
class _YourNumbersOnTheWay extends StatelessWidget {
  final LegDiscovery leg;
  final void Function(Contact)? onOpen;

  const _YourNumbersOnTheWay({required this.leg, this.onOpen});

  @override
  Widget build(BuildContext context) {
    final found = leg.onTheWay;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        const StencilLabel('Your numbers on the way'),
        if (found.isEmpty)
          _Explain(
            // Three different empties, and each one says a different thing
            // to do. Collapsing them into "nothing here" would be the one
            // answer that is always wrong.
            !leg.canPlace
                ? 'This leg has no route and one of its stops has no '
                      'location, so nothing can be placed along it yet.'
                : !leg.anyPlaced && leg.unplaced > 0
                ? '${leg.unplaced} of your '
                      '${leg.unplaced == 1 ? 'number has' : 'numbers have'} '
                      'no location saved, so '
                      '${leg.unplaced == 1 ? 'it' : 'they'} cannot be put on '
                      'this road. Import your sheet again with Latitude and '
                      'Longitude columns — entries already here get the '
                      'location added, with no copies — or paste one into '
                      'any entry with Edit.'
                : 'None of your numbers lie along this road.',
            caution: !leg.canPlace || (!leg.anyPlaced && leg.unplaced > 0),
          )
        else
          for (final (i, hit) in found.indexed)
            _ContactRow(
              contact: hit.contact,
              km: hit.alongRouteKm,
              first: i == 0,
              last: i == found.length - 1,
              onTap: onOpen == null ? null : () => onOpen!(hit.contact),
            ),
      ],
    );
  }
}

/// The user's own numbers at the stop this leg arrives at, help first.
class _YourNumbersAtDestination extends StatelessWidget {
  final LegDiscovery leg;
  final void Function(Contact)? onOpen;

  const _YourNumbersAtDestination({required this.leg, this.onOpen});

  @override
  Widget build(BuildContext context) {
    final all = leg.atDestination;
    final shown = all.take(LegDetailScreen.destinationLimit).toList();
    final more = all.length - shown.length;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        StencilLabel('Your numbers at ${leg.toName}'),
        if (all.isEmpty)
          _Explain('Nothing in your diary at ${leg.toName} yet.')
        else ...[
          for (final contact in shown)
            _ContactRow(
              contact: contact,
              onTap: onOpen == null ? null : () => onOpen!(contact),
            ),
          if (more > 0)
            _Explain('$more more at ${leg.toName} — all of them are in the '
                'diary.'),
        ],
        if (leg.nearestHelp.isNotEmpty) ...[
          StencilLabel('Nearest help to ${leg.toName}'),
          _Explain(
            '${leg.toName} has no hospital or pharmacy in your diary. These '
            'are the closest ones that do — distances are straight-line, and '
            'the road through these hills is longer.',
          ),
          for (final help in leg.nearestHelp)
            _ContactRow(
              contact: help.contact,
              away: help.straightLineKm,
              onTap: onOpen == null ? null : () => onOpen!(help.contact),
            ),
        ],
      ],
    );
  }
}

class _Explain extends StatelessWidget {
  final String text;
  final bool caution;
  const _Explain(this.text, {this.caution = false});

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    return Padding(
      padding: const EdgeInsets.fromLTRB(
        AppTokens.gutter,
        AppTokens.s8,
        AppTokens.gutter,
        AppTokens.s4,
      ),
      child: Text(
        text,
        style: AppTokens.captionStyle.copyWith(
          color: caution ? c.cautionMark : c.muted,
        ),
      ),
    );
  }
}

/// One diary contact: what it is, the number as the user has it, and the
/// note — which is where the sheet's hours, source and warnings went.
///
/// With [km], it hangs off the milestone rail like a map place. Without, it
/// is a plain row, because a number at the destination has no distance.
class _ContactRow extends StatelessWidget {
  final Contact contact;
  final double? km;

  /// Straight-line distance from a stop, for nearest-help rows. Shown in the
  /// second line rather than on the rail: it is not a point on this road.
  final double? away;
  final bool first;
  final bool last;
  final VoidCallback? onTap;

  const _ContactRow({
    required this.contact,
    this.km,
    this.away,
    this.first = false,
    this.last = false,
    this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    final note = contact.note?.trim();
    final category =
        ContactCategory.labels[contact.category] ?? contact.category;

    final text = Padding(
      padding: const EdgeInsets.symmetric(vertical: AppTokens.s12),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            contact.name,
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
            style: AppTokens.rowTitleStyle.copyWith(color: c.ink),
          ),
          const SizedBox(height: 2),
          Text(
            [
              category.toUpperCase(),
              contact.phoneRaw,
              if (away != null) '${away!.round()} KM AWAY',
            ].join(' · '),
            style: AppTokens.stencilStyle.copyWith(fontSize: 9, color: c.muted),
          ),
          if (note != null && note.isNotEmpty) ...[
            const SizedBox(height: 4),
            Text(
              note,
              // Two lines: enough for "24x7. Google listing only — not
              // call-tested", which is what decides whether to trust the
              // number. The rest is one tap away.
              maxLines: 2,
              overflow: TextOverflow.ellipsis,
              style: AppTokens.captionStyle.copyWith(color: c.muted),
            ),
          ],
        ],
      ),
    );

    return PressScale(
      onTap: onTap,
      child: Container(
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
              if (km != null) ...[
                SizedBox(
                  width: 48,
                  child: Align(
                    alignment: Alignment.centerRight,
                    child: Text(
                      '${km!.round()} km',
                      style: AppTokens.numberStyle.copyWith(color: c.muted),
                    ),
                  ),
                ),
                _Milestone(first: first, last: last),
              ],
              Expanded(child: text),
              // THE SAME AMBER DOT AS EVERYWHERE ELSE: imported is not
              // verified. It clears when the number is called and confirmed,
              // from the entry this row opens.
              if (!contact.callConfirmed)
                const Align(
                  alignment: Alignment.centerRight,
                  child: TrustDot(),
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
