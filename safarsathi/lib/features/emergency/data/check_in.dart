// lib/features/emergency/data/check_in.dart
//
// "Reached Sohra safely" — #33.
//
// The quiet half of safety: not an alarm, a reassurance, sent to the same
// trusted people as the SOS text and by the same road — the phone's own SMS
// app, which needs one bar and no data. The app writes it; the person
// presses Send. Nothing is sent by itself, and nothing goes through a
// server (DECISIONS 2026-09-11).
//
// A check-in is recorded as an `arrival` on the trip's timeline, which is
// what #34 looks for before reminding anyone, and what #30 draws.

import 'package:drift/drift.dart';

import '../../../core/database/app_database.dart';
import '../../../core/database/watch_tables.dart';
import '../../map/data/here.dart';
import '../../trips/data/driver.dart' show legToday;

/// The message. Pure, so every shape of it is pinned by a test.
String checkInMessage({
  required String stopName,
  required DateTime at,
  HereFix? fix,
  String? stay,
  DateTime? now,
}) {
  final hh = at.hour.toString().padLeft(2, '0');
  final mm = at.minute.toString().padLeft(2, '0');
  const months = [
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
  ];
  final out = StringBuffer(
    'Reached $stopName safely — $hh:$mm, ${at.day} ${months[at.month - 1]}.\n',
  );
  if (fix != null) {
    final lat = fix.at.lat.toStringAsFixed(5);
    final lon = fix.at.lon.toStringAsFixed(5);
    out
      ..writeln('Where I am: $lat, $lon (${describeFix(fix, now: now)})')
      ..writeln('Map: https://maps.google.com/?q=$lat,$lon');
  }
  if (stay != null) out.writeln('Staying at: $stay');
  out.write('Sent from SafarSathi.');
  return out.toString();
}

/// Which stop a check-in is most likely for: where today's leg arrives, or
/// failing that the stop the trip says you are at.
int? checkInStopId({
  required List<Leg> legs,
  required int? currentStopId,
  DateTime? now,
}) => legToday(legs, now: now)?.toStopId ?? currentStopId;

/// Records a check-in at [stopId]. Called when the person opens the message
/// to send — the app cannot know whether they pressed Send, and says so.
Future<int> recordCheckIn(
  AppDatabase db, {
  required int tripId,
  required int stopId,
  required String stopName,
  HereFix? fix,
  DateTime? at,
}) => db.into(db.timelineEntries).insert(
  TimelineEntriesCompanion.insert(
    tripId: tripId,
    stopId: Value(stopId),
    kind: 'arrival',
    title: Value('Checked in at $stopName'),
    lat: Value(fix?.at.lat),
    lon: Value(fix?.at.lon),
    accuracyM: Value(fix?.accuracyM),
    occurredAt: at ?? DateTime.now(),
  ),
);

/// Stops of [tripId] with a check-in, and when the latest was.
Stream<Map<int, DateTime>> watchCheckIns(AppDatabase db, int tripId) =>
    watchTables(db, {db.timelineEntries}).asyncMap((_) async {
      final rows = await (db.select(db.timelineEntries)..where(
            (t) =>
                t.tripId.equals(tripId) &
                t.kind.equals('arrival') &
                t.stopId.isNotNull(),
          ))
          .get();
      final out = <int, DateTime>{};
      for (final r in rows) {
        final prev = out[r.stopId!];
        if (prev == null || r.occurredAt.isAfter(prev)) {
          out[r.stopId!] = r.occurredAt;
        }
      }
      return out;
    });
