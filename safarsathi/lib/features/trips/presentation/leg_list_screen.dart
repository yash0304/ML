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
  final String fromName;
  final String toName;
  final String? mode;
  final DateTime? plannedDeparture;
  final bool isBooked;
  final double? distanceKm;

  const LegRow({
    required this.id,
    required this.fromName,
    required this.toName,
    required this.isBooked,
    this.mode,
    this.plannedDeparture,
    this.distanceKm,
  });
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

    return [
      for (final l in legs)
        LegRow(
          id: l.id,
          fromName: byId[l.fromStopId] ?? '—',
          toName: byId[l.toStopId] ?? '—',
          mode: l.mode,
          plannedDeparture: l.plannedDeparture,
          isBooked: l.isBooked,
          distanceKm: l.distanceKm,
        ),
    ];
  });
}

class LegListScreen extends StatelessWidget {
  final Stream<List<LegRow>> legs;
  final void Function(int legId) onOpen;

  const LegListScreen({super.key, required this.legs, required this.onOpen});

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
            itemBuilder: (context, i) =>
                _LegRowTile(row: list[i], onTap: () => onOpen(list[i].id)),
          );
        },
      ),
    );
  }
}

class _LegRowTile extends StatelessWidget {
  final LegRow row;
  final VoidCallback onTap;

  const _LegRowTile({required this.row, required this.onTap});

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
              ],
            ),
          ],
        ),
      ),
    );
  }
}
