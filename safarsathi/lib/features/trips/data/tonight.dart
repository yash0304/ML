// lib/features/trips/data/tonight.dart
//
// Where you sleep tonight, and how to reach them.
//
// NOT THE SAME AS THE CURRENT STOP. On the day you move from Sohra to
// Mawlynnong, `currentStopOf` still says Sohra — you are there at breakfast —
// but tonight's bed is in Mawlynnong, and that is the number you need at six
// in the evening on a dark road. So the rule here is the night, not the day:
// the stop whose nights include tonight.

import 'package:drift/drift.dart';

import '../../../core/database/app_database.dart';
import '../../../core/util/sun.dart';
import '../../contacts/data/contacts_dao.dart';

enum TonightKind {
  /// Tonight's bed, during the trip.
  tonight,

  /// Before the trip starts: the first night, so the card is useful while
  /// packing rather than blank until the day.
  firstNight,
}

class Tonight {
  final TonightKind kind;
  final Stop stop;

  /// The night in question — today, or the trip's first night.
  final DateTime night;

  /// Accommodation numbers saved at that stop, confirmed first.
  final List<Contact> stays;

  final SunTimes? sun;

  const Tonight({
    required this.kind,
    required this.stop,
    required this.night,
    required this.stays,
    this.sun,
  });
}

DateTime _day(DateTime d) => DateTime(d.year, d.month, d.day);

/// The last night spent at [s], exclusive: the day you leave.
DateTime? _leaves(Stop s) {
  final a = s.arrivalDate;
  if (a == null) return null;
  if (s.departureDate != null) return _day(s.departureDate!);
  return _day(a).add(Duration(days: s.nights));
}

/// The stop you sleep at on the night of [now], or null — with the first
/// night instead when the trip has not started yet.
///
/// A stop with no nights is passed through, not slept at, and never counts.
/// After the last night, and on a travel night between stops, this is null:
/// better an absent card than a bed that is not yours.
({Stop stop, TonightKind kind, DateTime night})? tonightStopOf(
  List<Stop> stops, {
  DateTime? now,
}) {
  final today = _day(now ?? DateTime.now());
  final sleeping = [
    for (final s in stops)
      if (s.arrivalDate != null &&
          _leaves(s) != null &&
          _leaves(s)!.isAfter(_day(s.arrivalDate!)))
        s,
  ];
  if (sleeping.isEmpty) return null;

  for (final s in sleeping) {
    final from = _day(s.arrivalDate!);
    if (!today.isBefore(from) && today.isBefore(_leaves(s)!)) {
      return (stop: s, kind: TonightKind.tonight, night: today);
    }
  }

  final first = sleeping.reduce(
    (a, b) => a.arrivalDate!.isBefore(b.arrivalDate!) ? a : b,
  );
  if (today.isBefore(_day(first.arrivalDate!))) {
    return (
      stop: first,
      kind: TonightKind.firstNight,
      night: _day(first.arrivalDate!),
    );
  }
  return null;
}

/// Stays at [stopId], confirmed ones first: a number somebody has actually
/// rung is the one to try first at night.
List<Contact> staysAt(List<Contact> diary, int stopId) => [
  for (final c in diary)
    if (c.stopId == stopId && c.category == ContactCategory.accommodation) c,
]..sort((a, b) {
    if (a.callConfirmed != b.callConfirmed) return a.callConfirmed ? -1 : 1;
    return a.name.toLowerCase().compareTo(b.name.toLowerCase());
  });

/// Tonight, live. Ticks on stops and contacts: editing a stop's dates or
/// saving the homestay's number both change what this card should say.
Stream<Tonight?> watchTonight(AppDatabase db, int tripId, {DateTime? now}) =>
    db
        .customSelect('SELECT 1', readsFrom: {db.stops, db.contacts})
        .watch()
        .asyncMap((_) async {
          final stops = await (db.select(db.stops)
                ..where((s) => s.tripId.equals(tripId))
                ..orderBy([(s) => OrderingTerm(expression: s.sequenceOrder)]))
              .get();
          final hit = tonightStopOf(stops, now: now);
          if (hit == null) return null;

          final diary = await (db.select(
            db.contacts,
          )..where((c) => c.tripId.equals(tripId))).get();
          final s = hit.stop;

          return Tonight(
            kind: hit.kind,
            stop: s,
            night: hit.night,
            stays: staysAt(diary, s.id),
            sun: s.lat == null || s.lon == null
                ? null
                : sunTimes(s.lat!, s.lon!, hit.night),
          );
        });
