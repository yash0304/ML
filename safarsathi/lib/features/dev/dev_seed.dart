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
/// builds if the database is empty. Returns null in release with no trips.
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

  final tripId = await db
      .into(db.trips)
      .insert(
        TripsCompanion.insert(
          name: 'Meghalaya · demo',
          isActive: const Value(true),
        ),
      );

  Future<int> stop(String name, int order, int nights) => db
      .into(db.stops)
      .insert(
        StopsCompanion.insert(
          tripId: tripId,
          name: name,
          sequenceOrder: order,
          countryCode: 'IN',
          nights: Value(nights),
        ),
      );

  final shillong = await stop('Shillong', 1, 1);
  await stop('Cherrapunji', 2, 1);
  final kongthong = await stop('Kongthong', 3, 2);

  Future<void> entry(
    String name,
    String phone, {
    int? stopId,
    String category = ContactCategory.other,
    String? note,
    bool confirmed = false,
    bool pinned = false,
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
    note: 'Placeholder data',
    confirmed: true,
    pinned: true,
  );
  await entry(
    'Driver',
    '+91 90000 00002',
    category: ContactCategory.transport,
    note: 'Placeholder data',
    confirmed: true,
  );
  await entry(
    'Shillong guesthouse',
    '+91 90000 00003',
    stopId: shillong,
    category: ContactCategory.accommodation,
    note: 'Placeholder data',
  );
  await entry(
    'Village guide',
    '+91 90000 00004',
    stopId: kongthong,
    category: ContactCategory.guide,
    note: 'Placeholder data',
  );
  await entry(
    'Chemist, Sohra',
    '+91 90000 00005',
    category: ContactCategory.pharmacy,
    note: 'Placeholder data',
  );
  await entry(
    'Fuel pump',
    '+91 90000 00006',
    category: ContactCategory.fuel,
    note: 'Placeholder data',
  );

  return DemoTrip(
    tripId: tripId,
    name: 'Meghalaya · demo',
    currentStopId: kongthong,
    currentStopName: 'Kongthong',
  );
}
