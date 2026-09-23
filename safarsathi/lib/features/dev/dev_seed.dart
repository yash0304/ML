// lib/features/dev/dev_seed.dart
//
// TEMPORARY. Trip CRUD is #16; until then the diary needs something to show.
//
// Creates one demo trip and a handful of entries, ONLY when the database has
// no trip at all and ONLY in debug builds.
//
// EVERY NUMBER HERE IS DELIBERATELY, OBVIOUSLY FAKE — +91 90000 000xx — so
// that placeholder data can never be mistaken for a number someone might
// actually dial. Delete this file at #16.

import 'package:drift/drift.dart';
import 'package:flutter/foundation.dart';

import '../../core/database/app_database.dart';
import '../contacts/data/contacts_dao.dart';
import '../money/data/settlement.dart';

class DemoTrip {
  final int tripId;
  final String name;
  final int? currentStopId;
  final String? currentStopName;

  const DemoTrip({
    required this.tripId,
    required this.name,
    this.currentStopId,
    this.currentStopName,
  });
}

/// Returns the trip the diary should open on, creating a demo one in debug
/// builds if the database is empty.
///
/// Returns null in a release build with no trips, so the caller can offer
/// [createDemoTrip] rather than opening a diary with nothing in it. Real trip
/// creation is #16.
Future<DemoTrip?> ensureDemoTrip(AppDatabase db) async {
  final existing = await db.select(db.trips).get();
  if (existing.isNotEmpty) {
    final trip = existing.first;
    final stops =
        await (db.select(db.stops)
              ..where((s) => s.tripId.equals(trip.id))
              ..orderBy([(s) => OrderingTerm(expression: s.sequenceOrder)]))
            .get();
    return DemoTrip(
      tripId: trip.id,
      name: trip.name,
      currentStopId: stops.isEmpty ? null : stops.last.id,
      currentStopName: stops.isEmpty ? null : stops.last.name,
    );
  }

  if (!kDebugMode) return null;
  return createDemoTrip(db);
}

/// Creates the demo trip on demand, in any build.
///
/// A release build has no way to make a trip until #16, so without this an
/// installable APK opens to an empty screen and there is nothing to look at.
Future<DemoTrip> createDemoTrip(AppDatabase db) async {
  final tripId = await db
      .into(db.trips)
      .insert(
        TripsCompanion.insert(
          name: 'Meghalaya · demo',
          isActive: const Value(true),
        ),
      );

  Future<int> stop(String name, int order, int nights, int day) => db
      .into(db.stops)
      .insert(
        StopsCompanion.insert(
          tripId: tripId,
          name: name,
          sequenceOrder: order,
          countryCode: 'IN',
          nights: Value(nights),
          arrivalDate: Value(DateTime(2026, 10, day)),
        ),
      );

  final shillong = await stop('Shillong', 1, 1, 1);
  final cherrapunji = await stop('Cherrapunji', 2, 1, 2);
  final kongthong = await stop('Kongthong', 3, 2, 3);

  Future<void> leg(
    int from,
    int to,
    int order,
    String mode,
    double km,
    int hour,
    String note, {
    bool cached = true,
  }) => db
      .into(db.legs)
      .insert(
        LegsCompanion.insert(
          tripId: tripId,
          fromStopId: from,
          toStopId: to,
          sequenceOrder: order,
          mode: Value(mode),
          distanceKm: Value(km),
          plannedDeparture: Value(DateTime(2026, 10, order + 1, hour, 30)),
          note: Value(note),
          lastSyncedAt: Value(cached ? DateTime(2026, 9, 11) : null),
        ),
      );

  await leg(shillong, cherrapunji, 1, 'Shared taxi', 54, 9, 'Leaves when full');
  // Deliberately left unsynced, so the milestone cap renders muted and an
  // unprepared leg is visible without reading anything.
  await leg(
    cherrapunji,
    kongthong,
    2,
    'Shared taxi',
    56,
    9,
    'Wanshai has the pickup point',
    cached: false,
  );

  Future<void> entry(
    String name,
    String phone, {
    int? stopId,
    String category = ContactCategory.other,
    String? note,
    bool confirmed = false,
    bool pinned = false,
    bool emergency = false,
  }) => db
      .into(db.contacts)
      .insert(
        ContactsCompanion.insert(
          name: name,
          phoneRaw: phone,
          tripId: Value(tripId),
          stopId: Value(stopId),
          category: Value(category),
          note: Value(note),
          isPinned: Value(pinned),
          isEmergency: Value(emergency),
          callConfirmed: Value(confirmed),
          tier: Value(
            confirmed
                ? ContactTier.userVerified.name
                : ContactTier.userEntered.name,
          ),
        ),
      );

  await entry(
    'Kongthong homestay',
    '+91 90000 00001',
    stopId: kongthong,
    category: ContactCategory.accommodation,
    note: 'Sample — not a real number',
    pinned: true,
  );
  await entry(
    'Driver',
    '+91 90000 00002',
    category: ContactCategory.transport,
    note: 'Sample — not a real number',
  );
  await entry(
    'Shillong guesthouse',
    '+91 90000 00003',
    stopId: shillong,
    category: ContactCategory.accommodation,
    note: 'Sample — not a real number',
  );
  await entry(
    'Village guide',
    '+91 90000 00004',
    stopId: kongthong,
    category: ContactCategory.guide,
    note: 'Sample — not a real number',
  );
  await entry(
    'Chemist, Sohra',
    '+91 90000 00005',
    category: ContactCategory.pharmacy,
    note: 'Sample — not a real number',
  );
  await entry(
    'Fuel pump',
    '+91 90000 00006',
    category: ContactCategory.fuel,
    note: 'Sample — not a real number',
  );
  // NEVER CONFIRMED, NEVER EMERGENCY. This entry used to be seeded as a
  // confirmed emergency contact who "can reach a local ambulance faster than
  // 108" — a made-up number that looked call-verified on the SOS tab. People
  // start real trips from the demo and keep its rows, and that is how it
  // reached a real trip a week before departure. A sample may show what the
  // diary looks like; it may never look trustworthy. trip_health.dart finds
  // and removes the old ones.
  await entry(
    'Bah Rothell · homestay owner',
    '+91 90000 00007',
    stopId: kongthong,
    category: ContactCategory.localContact,
    note: 'Sample — not a real number',
  );

  // --- Money -------------------------------------------------------------
  //
  // ONE TRAVELLER, AND THAT IS YOU. The demo used to seed two companions so
  // the settle-up had something to settle, which meant anybody opening the
  // app for the first time found two strangers in their ledger and had to
  // work out they were not real. Adding a traveller is one tap on
  // Money → Travellers, and doing it yourself is a better way to learn the
  // screen than finding it pre-filled.
  final people = <int>[
    await db
        .into(db.travellers)
        .insert(
          TravellersCompanion.insert(
            tripId: tripId,
            name: 'You',
            isSelf: const Value(true),
          ),
        ),
  ];

  Future<void> spend(String what, int amountMinor, int paidBy, int day) async {
    final id = await db
        .into(db.expenses)
        .insert(
          ExpensesCompanion.insert(
            tripId: tripId,
            description: what,
            amountMinor: amountMinor,
            paidById: people[paidBy],
            spentAt: Value(DateTime(2026, 10, day)),
          ),
        );
    // Shares are handed out one paisa at a time so they sum back exactly.
    final shares = evenShares(amountMinor, people.length);
    for (var i = 0; i < people.length; i++) {
      await db
          .into(db.expenseSplits)
          .insert(
            ExpenseSplitsCompanion.insert(
              expenseId: id,
              travellerId: people[i],
              shareMinor: shares[i],
            ),
          );
    }
  }

  // All paid by you, because you are the only one here.
  await spend('Taxi · Shillong to Cherrapunji', 320011, 0, 2);
  await spend('Homestay · 2 nights', 440000, 0, 3);
  await spend('Cave guide', 150000, 0, 3);
  await spend('Dinner at Sohra', 86000, 0, 2);
  await spend('Fuel', 210000, 0, 2);

  return DemoTrip(
    tripId: tripId,
    name: 'Meghalaya · demo',
    currentStopId: kongthong,
    currentStopName: 'Kongthong',
  );
}
