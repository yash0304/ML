// lib/features/trips/data/plan_message.dart
//
// The trip as one message for somebody at home.
//
// WHY. Someone who is not on the trip should know which town you sleep in
// each night and a number there that is not yours — because yours is the
// one that will be out of signal. This is that, as plain text, for whatever
// the person chooses to send it with.
//
// ONLY WHAT WAS TYPED. No guessed hotels, no "signal is weak at…" that the
// app has no way of knowing. A night with no stay saved says so, which is
// itself worth the reader knowing.

import 'package:drift/drift.dart';

import '../../../core/database/app_database.dart';
import 'stay.dart' show chosenStay;

const _months = [
  'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
];

String _date(DateTime d) => '${d.day} ${_months[d.month - 1]}';

String _time(DateTime d) =>
    '${d.hour.toString().padLeft(2, '0')}:${d.minute.toString().padLeft(2, '0')}';

/// The message, from rows already read. Pure, so the tests can pin it.
String formatPlan({
  required Trip trip,
  required List<Stop> stops,
  required List<Leg> legs,
  required List<Contact> diary,
  List<Traveller> travellers = const [],
}) {
  final out = StringBuffer();
  final byId = {for (final s in stops) s.id: s};

  out.write(trip.name);
  if (trip.startDate != null) {
    out.write(' — ${_date(trip.startDate!)}');
    if (trip.endDate != null) {
      out.write('–${_date(trip.endDate!)} ${trip.endDate!.year}');
    }
  }
  out.writeln();

  if (travellers.isNotEmpty) {
    out.writeln('Travelling: ${travellers.map((t) => t.name).join(', ')}');
  }
  out
    ..writeln()
    ..writeln('Where we sleep each night:');

  for (final s in stops) {
    final nights = s.nights;
    final when = s.arrivalDate == null ? '' : '${_date(s.arrivalDate!)} · ';
    final length = nights == 0
        ? 'passing through'
        : '$nights ${nights == 1 ? 'night' : 'nights'}';
    out.writeln('$when${s.name} ($length)');

    if (nights > 0) {
      // ONLY THE STAY YOU CHOSE. This listed the first two accommodation
      // numbers at the stop, so a sheet of options went home as if both
      // were booked. Somebody at home reading a name will ring it.
      final stay = chosenStay(s, diary);
      out.writeln(
        stay == null
            ? '  Stay: not decided yet'
            : '  Stay: ${stay.name}, ${stay.phoneRaw}',
      );
    }
  }

  final travel = [
    for (final l in legs)
      if (byId[l.fromStopId] != null && byId[l.toStopId] != null) l,
  ];
  if (travel.isNotEmpty) {
    out
      ..writeln()
      ..writeln('Getting between them:');
    for (final l in travel) {
      final parts = <String>[
        if (l.plannedDeparture != null)
          '${_date(l.plannedDeparture!)} ${_time(l.plannedDeparture!)}',
        '${byId[l.fromStopId]!.name} → ${byId[l.toStopId]!.name}',
        if (l.mode != null && l.mode!.trim().isNotEmpty) l.mode!.trim(),
      ];
      out.writeln(parts.join(' · '));
    }
  }

  out
    ..writeln()
    ..writeln(
      'If we do not answer, we are probably out of signal — the number for '
      'each night\'s stay is above.',
    );
  return out.toString().trimRight();
}

/// The message for [tripId], read from the database.
Future<String> buildPlanMessage(AppDatabase db, int tripId) async {
  final trip = await (db.select(
    db.trips,
  )..where((t) => t.id.equals(tripId))).getSingle();
  final stops = await (db.select(db.stops)
        ..where((s) => s.tripId.equals(tripId))
        ..orderBy([(s) => OrderingTerm(expression: s.sequenceOrder)]))
      .get();
  final legs = await (db.select(db.legs)
        ..where((l) => l.tripId.equals(tripId))
        ..orderBy([(l) => OrderingTerm(expression: l.sequenceOrder)]))
      .get();
  final diary = await (db.select(
    db.contacts,
  )..where((c) => c.tripId.equals(tripId))).get();
  final travellers = await (db.select(
    db.travellers,
  )..where((t) => t.tripId.equals(tripId))).get();

  return formatPlan(
    trip: trip,
    stops: stops,
    legs: legs,
    diary: diary,
    travellers: travellers,
  );
}
