// lib/features/trips/data/driver.dart
//
// Who is taking you from one stop to the next, and in what.
//
// In Meghalaya the answer is usually one taxi and one driver for the whole
// trip, booked through a homestay or a stand in Guwahati. The driver's
// number is the one you ring when the car is not at the gate, and the
// vehicle's registration is the one detail somebody at home can do anything
// with if the day goes wrong. Both go in the plan sent home, and in an SOS
// text on a travel day.

import 'package:drift/drift.dart';

import '../../../core/database/app_database.dart';
import '../../contacts/data/contacts_dao.dart' show ContactCategory;

/// The driver chosen for [leg], or null — none chosen, or since deleted.
Contact? driverOf(Leg leg, List<Contact> diary) {
  final id = leg.driverContactId;
  if (id == null) return null;
  for (final c in diary) {
    if (c.id == id) return c;
  }
  return null;
}

/// Who can be picked: transport numbers first (drivers, taxi stands), then
/// local contacts, who are often the person arranging the car.
List<Contact> driverOptions(List<Contact> diary) {
  int rank(Contact c) => c.category == ContactCategory.transport ? 0 : 1;
  return [
    for (final c in diary)
      if (c.category == ContactCategory.transport ||
          c.category == ContactCategory.localContact)
        c,
  ]..sort((a, b) {
      final r = rank(a).compareTo(rank(b));
      return r != 0 ? r : a.name.toLowerCase().compareTo(b.name.toLowerCase());
    });
}

Future<void> setDriver(AppDatabase db, int legId, int? contactId) =>
    (db.update(db.legs)..where((l) => l.id.equals(legId))).write(
      LegsCompanion(driverContactId: Value(contactId)),
    );

/// Legs of [tripId] with nobody driving yet, other than [exceptLegId] — the
/// ones "same driver for the whole trip" would fill.
Future<List<Leg>> legsWithoutDriver(
  AppDatabase db,
  int tripId, {
  int? exceptLegId,
}) async => [
  for (final l in await (db.select(
    db.legs,
  )..where((l) => l.tripId.equals(tripId))).get())
    if (l.driverContactId == null && l.id != exceptLegId) l,
];

/// Makes [contactId] the driver of every leg that has none. A leg already
/// given a different driver keeps them: "the whole trip" never overwrites
/// a choice made on purpose. Returns how many legs changed.
Future<int> setDriverWhereMissing(
  AppDatabase db,
  int tripId,
  int contactId,
) =>
    (db.update(db.legs)..where(
          (l) => l.tripId.equals(tripId) & l.driverContactId.isNull(),
        ))
        .write(LegsCompanion(driverContactId: Value(contactId)));

/// The leg you are on, or about to be on, today: the latest one that has
/// left, or else today's first. Null on a day with no leg planned.
Leg? legToday(List<Leg> legs, {DateTime? now}) {
  final t = now ?? DateTime.now();
  bool sameDay(DateTime d) =>
      d.year == t.year && d.month == t.month && d.day == t.day;
  final today = [
    for (final l in legs)
      if (l.plannedDeparture != null && sameDay(l.plannedDeparture!)) l,
  ]..sort((a, b) => a.plannedDeparture!.compareTo(b.plannedDeparture!));
  if (today.isEmpty) return null;
  Leg? left;
  for (final l in today) {
    if (!l.plannedDeparture!.isAfter(t)) left = l;
  }
  return left ?? today.first;
}

/// "Taxi ML 05 A 1234 · driver Anthony, +91 98560 12345" — whatever of it is
/// known. Null when nothing is.
String? rideLine({String? mode, String? vehicle, Contact? driver}) {
  final what = [
    if (mode != null && mode.trim().isNotEmpty) mode.trim(),
    if (vehicle != null && vehicle.trim().isNotEmpty) vehicle.trim(),
  ].join(' ');
  final parts = [
    if (what.isNotEmpty) what,
    if (driver != null) 'driver ${driver.name}, ${driver.phoneRaw}',
  ];
  return parts.isEmpty ? null : parts.join(' · ');
}
