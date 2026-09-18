// lib/features/trips/presentation/leg_list_screen.dart
//
// The legs between stops, as a list you can open — issues #17 and #18.
//
// Legs are never created here. They exist because two stops are consecutive,
// and they appear and disappear as the itinerary changes. What this screen
// offers is the part only a person can supply: how you are getting there.

import 'package:drift/drift.dart' show OrderingTerm;
import 'package:flutter/material.dart';

import '../../../core/database/app_database.dart';
import '../../../core/theme/app_tokens.dart';

class LegRow {
  final int id;
  final int poiCount;
  final String fromName;
  final String toName;
  final String? mode;
  final DateTime? plannedDeparture;
  final bool isBooked;
  final double? distanceKm;

  /// Stops on this leg that have no coordinates yet, by name.
  ///
  /// A leg cannot be routed without both ends, so this is the difference
  /// between "nobody has downloaded this yet" and "nothing can download it
  /// until you do something". The row said neither and simply showed a blank
  /// space where the kilometres go.
  final List<String> stopsWithoutLocation;

  const LegRow({
    required this.id,
    this.poiCount = 0,
    required this.fromName,
    required this.toName,
    required this.isBooked,
    this.mode,
    this.plannedDeparture,
    this.distanceKm,
    this.stopsWithoutLocation = const [],
  });

  bool get canBeRouted => stopsWithoutLocation.isEmpty;
  bool get isDownloaded => distanceKm != null;

  /// What this leg is waiting for, or null when it is waiting for nothing.
  String? get blockedBy {
    if (canBeRouted) return isDownloaded ? null : 'Not downloaded yet';
    final missing = stopsWithoutLocation;
    return missing.length == 1
        ? '${missing.single} has no location yet'
        : '${missing.join(' and ')} have no location yet';
  }
}

/// Reads from both tables, because a leg's label is two stop names.
Stream<List<LegRow>> watchLegSummaries(AppDatabase db, int tripId) {
  final tick = db
      .customSelect('SELECT 1', readsFrom: {db.legs, db.stops})
      .watch();

  return tick.asyncMap((_) async {
    final legs =
        await (db.select(db.legs)
              ..where((l) => l.tripId.equals(tripId))
              ..orderBy([(l) => OrderingTerm(expression: l.sequenceOrder)]))
            .get();

    final stops = await (db.select(
      db.stops,
    )..where((s) => s.tripId.equals(tripId))).get();
    final byId = {for (final s in stops) s.id: s.name};
    // A leg needs BOTH ends located before anything can be fetched for it.
    final located = {
      for (final s in stops) s.id: s.lat != null && s.lon != null,
    };

    final pois = await (db.select(
      db.pois,
    )..where((p) => p.tripId.equals(tripId))).get();
    final poisPerLeg = <int, int>{};
    for (final p in pois) {
      if (p.legId != null) {
        poisPerLeg[p.legId!] = (poisPerLeg[p.legId!] ?? 0) + 1;
      }
    }

    return [
      for (final l in legs)
        LegRow(
          id: l.id,
          poiCount: poisPerLeg[l.id] ?? 0,
          fromName: byId[l.fromStopId] ?? '—',
          toName: byId[l.toStopId] ?? '—',
          mode: l.mode,
          plannedDeparture: l.plannedDeparture,
          isBooked: l.isBooked,
          distanceKm: l.distanceKm,
          stopsWithoutLocation: [
            for (final id in {l.fromStopId, l.toStopId})
              if (located[id] == false) byId[id] ?? 'A stop',
          ],
        ),
    ];
  });
}

class LegListScreen extends StatelessWidget {
  final Stream<List<LegRow>> legs;
  final void Function(int legId) onOpen;

  /// Opens what is along the leg (#27). Optional so tests and goldens can
  /// render the list without one.
  final void Function(int legId)? onDiscover;

  const LegListScreen({
    super.key,
    required this.legs,
    required this.onOpen,
    this.onDiscover,
  });

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);

    return Scaffold(
      backgroundColor: c.paper,
      appBar: AppBar(
        title: const Text('Getting between stops'),
        backgroundColor: c.paper,
        foregroundColor: c.ink,
        elevation: 0,
      ),
      body: StreamBuilder<List<LegRow>>(
        stream: legs,
        builder: (context, snap) {
          final list = snap.data;
          if (list == null) return const SizedBox();
          if (list.isEmpty) {
            return Center(
              child: Padding(
                padding: const EdgeInsets.all(AppTokens.s32),
                child: Text(
                  'Legs appear on their own once a trip has two stops. '
                  'You cannot add one directly — add the stops instead.',
                  textAlign: TextAlign.center,
                  style: AppTokens.captionStyle.copyWith(color: c.muted),
                ),
              ),
            );
          }
          return ListView.builder(
            itemCount: list.length,
            itemBuilder: (context, i) => _LegRowTile(
              row: list[i],
              onTap: () => onOpen(list[i].id),
              onDiscover: onDiscover == null
                  ? null
                  : () => onDiscover!(list[i].id),
            ),
          );
        },
      ),
    );
  }
}

class _LegRowTile extends StatelessWidget {
  final LegRow row;
  final VoidCallback onTap;
  final VoidCallback? onDiscover;

  const _LegRowTile({
    required this.row,
    required this.onTap,
    this.onDiscover,
  });

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    final d = row.plannedDeparture;

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
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    '${row.fromName} → ${row.toName}',
                    style: AppTokens.rowTitleStyle.copyWith(color: c.ink),
                  ),
                  const SizedBox(height: 2),
                  Text(
                    row.mode ?? 'No transport set',
                    style: AppTokens.captionStyle.copyWith(
                      color: row.mode == null ? c.cautionMark : c.muted,
                    ),
                  ),
                  if (d != null)
                    Padding(
                      padding: const EdgeInsets.only(top: 2),
                      child: Text(
                        '${d.day.toString().padLeft(2, '0')}/'
                        '${d.month.toString().padLeft(2, '0')}  '
                        '${d.hour.toString().padLeft(2, '0')}:'
                        '${d.minute.toString().padLeft(2, '0')}',
                        style: AppTokens.numberStyle.copyWith(
                          color: c.muted,
                          fontSize: 11.5,
                        ),
                      ),
                    ),
                ],
              ),
            ),
            Column(
              crossAxisAlignment: CrossAxisAlignment.end,
              children: [
                if (row.distanceKm != null)
                  Text(
                    '${row.distanceKm!.round()} km',
                    style: AppTokens.numberStyle.copyWith(color: c.muted),
                  )
                // A BLANK SPACE IS NOT AN EXPLANATION. A leg with no
                // kilometres is either waiting on a download or waiting on
                // a stop that has no location, and those need different
                // things done about them.
                else if (row.blockedBy != null)
                  ConstrainedBox(
                    constraints: const BoxConstraints(maxWidth: 148),
                    child: Text(
                      row.blockedBy!,
                      textAlign: TextAlign.end,
                      style: AppTokens.captionStyle.copyWith(
                        fontSize: 11.5,
                        color: row.canBeRouted ? c.muted : c.cautionMark,
                      ),
                    ),
                  ),
                if (row.isBooked)
                  Padding(
                    padding: const EdgeInsets.only(top: 2),
                    child: Text(
                      'BOOKED',
                      style: AppTokens.stencilStyle.copyWith(
                        fontSize: 9,
                        color: c.signal,
                      ),
                    ),
                  ),
                if (onDiscover != null && row.poiCount > 0)
                  Padding(
                    padding: const EdgeInsets.only(top: AppTokens.s8),
                    child: GestureDetector(
                      onTap: onDiscover,
                      child: Text(
                        '${row.poiCount} ON THE ROAD',
                        style: AppTokens.stencilStyle.copyWith(
                          fontSize: 9,
                          color: c.signal,
                        ),
                      ),
                    ),
                  ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
