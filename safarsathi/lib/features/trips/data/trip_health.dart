// lib/features/trips/data/trip_health.dart
//
// Two things that can quietly make a trip's diary untrustworthy, each with a
// one-tap fix the person approves.
//
// SAMPLE DATA FROM THE DEMO. A trip started from the demo keeps its sample
// rows: seven made-up numbers, two of them seeded as CONFIRMED and one as an
// EMERGENCY contact ("can reach a local ambulance faster than 108"), plus
// five made-up expenses. Found in a real trip a week before departure. A fake
// number that looks call-verified, on the SOS tab, is the single worst thing
// this app could hold — the exact failure its trust system exists to stop.
//
// DUPLICATES. Importing the same sheet twice put every stay on screen three
// times. The importer warned and imported anyway.
//
// Matching is exact, never fuzzy: a sample row is one of the demo's own seven
// numbers or five expenses to the paisa; a duplicate is the same name, same
// number, same stop, same category. Nothing the person typed can be caught.

import '../../../core/database/app_database.dart';

/// The demo's seven numbers, digits only. The +91 90000 000xx block was
/// chosen for the demo precisely because it is not anybody's number.
const demoNumbers = {
  '919000000001',
  '919000000002',
  '919000000003',
  '919000000004',
  '919000000005',
  '919000000006',
  '919000000007',
};

/// The demo's five expenses: description and amount in paise.
const demoExpenses = {
  ('Taxi · Shillong to Cherrapunji', 320011),
  ('Homestay · 2 nights', 440000),
  ('Cave guide', 150000),
  ('Dinner at Sohra', 86000),
  ('Fuel', 210000),
};

String _digits(String s) => s.replaceAll(RegExp(r'\D'), '');

bool isDemoContact(Contact c) =>
    demoNumbers.contains(_digits(c.phoneE164 ?? c.phoneRaw));

bool isDemoExpense(Expense e) =>
    demoExpenses.contains((e.description, e.amountMinor));

class TripHealth {
  final List<Contact> sampleContacts;
  final List<Expense> sampleExpenses;

  /// Every copy beyond the one kept, per group of identical entries.
  final List<Contact> duplicateExtras;

  const TripHealth({
    this.sampleContacts = const [],
    this.sampleExpenses = const [],
    this.duplicateExtras = const [],
  });

  bool get hasSamples => sampleContacts.isNotEmpty || sampleExpenses.isNotEmpty;
  bool get isHealthy => !hasSamples && duplicateExtras.isEmpty;

  /// Sample numbers that look verified or sit on the SOS tab — said out loud,
  /// because that is what makes this urgent rather than tidy.
  int get sampleConfirmed => sampleContacts.where((c) => c.callConfirmed).length;
  int get sampleEmergency => sampleContacts.where((c) => c.isEmergency).length;
}

/// The copies to remove: for each group of identical entries, all but one.
///
/// THE ONE KEPT IS THE CONFIRMED ONE, if any copy was confirmed — a call made
/// on purpose is the most expensive thing in the diary — and otherwise the
/// oldest, which is the one any history hangs off.
List<Contact> duplicateExtras(List<Contact> diary) {
  final groups = <String, List<Contact>>{};
  for (final c in diary) {
    final key = [
      c.name.trim().toLowerCase(),
      _digits(c.phoneE164 ?? c.phoneRaw),
      '${c.stopId}',
      c.category,
    ].join('|');
    groups.putIfAbsent(key, () => []).add(c);
  }
  final extras = <Contact>[];
  for (final group in groups.values) {
    if (group.length < 2) continue;
    group.sort((a, b) {
      if (a.callConfirmed != b.callConfirmed) return a.callConfirmed ? -1 : 1;
      return a.id.compareTo(b.id);
    });
    extras.addAll(group.skip(1));
  }
  return extras;
}

TripHealth checkHealth(List<Contact> diary, List<Expense> expenses) {
  final samples = diary.where(isDemoContact).toList();
  final real = diary.where((c) => !isDemoContact(c)).toList();
  return TripHealth(
    sampleContacts: samples,
    sampleExpenses: expenses.where(isDemoExpense).toList(),
    // Samples are counted once, as samples: removing them is its own fix.
    duplicateExtras: duplicateExtras(real),
  );
}

Stream<TripHealth> watchTripHealth(AppDatabase db, int tripId) => db
    .customSelect('SELECT 1', readsFrom: {db.contacts, db.expenses})
    .watch()
    .asyncMap((_) async {
      final diary = await (db.select(
        db.contacts,
      )..where((c) => c.tripId.equals(tripId))).get();
      final expenses = await (db.select(
        db.expenses,
      )..where((e) => e.tripId.equals(tripId))).get();
      return checkHealth(diary, expenses);
    });

/// Removes the demo's sample contacts and expenses from [tripId].
Future<void> removeSamples(AppDatabase db, int tripId) async {
  final health = await watchTripHealth(db, tripId).first;
  await db.transaction(() async {
    final contactIds = [for (final c in health.sampleContacts) c.id];
    if (contactIds.isNotEmpty) {
      await (db.delete(db.contacts)..where((c) => c.id.isIn(contactIds))).go();
    }
    final expenseIds = [for (final e in health.sampleExpenses) e.id];
    if (expenseIds.isNotEmpty) {
      await (db.delete(db.expenses)..where((e) => e.id.isIn(expenseIds))).go();
    }
  });
}

/// Removes every duplicate copy in [tripId], keeping one of each.
Future<int> removeDuplicates(AppDatabase db, int tripId) async {
  final health = await watchTripHealth(db, tripId).first;
  final ids = [for (final c in health.duplicateExtras) c.id];
  if (ids.isEmpty) return 0;
  await (db.delete(db.contacts)..where((c) => c.id.isIn(ids))).go();
  return ids.length;
}
