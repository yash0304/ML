// lib/features/trips/data/readiness.dart
//
// The pre-departure readiness check — issue #20.
//
// This is where the trust system finally pays off. Every other part of the app
// records whether a number has been dialled; this is the part that refuses to
// call the trip ready until the ones that matter have been.
//
// WHICH ONES MATTER: the accommodation number of every stop you sleep at. A
// lunch stop with no number is an inconvenience. A homestay with no number, at
// 9pm, in a valley with no signal, is a night in the car.

import 'package:drift/drift.dart';

import '../../../core/database/app_database.dart';
import '../../../core/util/streams.dart';
import '../../contacts/data/contacts_dao.dart';

/// One reason the trip is not ready yet.
class ReadinessItem {
  final int stopId;
  final String stopName;

  /// Null when the stop has no accommodation contact at all.
  final int? contactId;

  final String label;

  /// True when the problem is an ABSENT number rather than an unconfirmed one.
  /// The more dangerous of the two, and the one an app usually fails to say
  /// anything about.
  final bool missing;

  const ReadinessItem({
    required this.stopId,
    required this.stopName,
    required this.label,
    this.contactId,
    this.missing = false,
  });
}

class Readiness {
  final List<ReadinessItem> blocking;
  const Readiness(this.blocking);

  bool get isReady => blocking.isEmpty;
  int get openCount => blocking.length;
}

/// Works out what is blocking, from rows already read.
///
/// Pure, so the rule is testable without a database and so the screen can be
/// driven from a stream without either knowing about the other.
Readiness computeReadiness(List<Stop> stops, List<Contact> contacts) {
  final items = <ReadinessItem>[];

  for (final stop in stops) {
    if (stop.nights <= 0) continue; // Not a stop you sleep at.

    final atStop = [
      for (final c in contacts)
        if (c.stopId == stop.id &&
            c.category == ContactCategory.accommodation)
          c,
    ];

    if (atStop.isEmpty) {
      items.add(
        ReadinessItem(
          stopId: stop.id,
          stopName: stop.name,
          missing: true,
          label: 'No accommodation number for ${stop.name}.',
        ),
      );
      continue;
    }

    // One confirmed number clears the stop. A second unconfirmed number for
    // the same homestay is not a second problem.
    if (atStop.any((c) => c.callConfirmed)) continue;

    items.add(
      ReadinessItem(
        stopId: stop.id,
        stopName: stop.name,
        contactId: atStop.first.id,
        label: 'Call and confirm the ${stop.name} accommodation number.',
      ),
    );
  }

  return Readiness(items);
}

/// Live readiness for a trip.
Stream<Readiness> watchReadiness(AppDatabase db, int tripId) {
  final stops =
      (db.select(db.stops)
            ..where((s) => s.tripId.equals(tripId))
            ..orderBy([(s) => OrderingTerm(expression: s.sequenceOrder)]))
          .watch();
  final contacts =
      (db.select(db.contacts)..where((c) => c.tripId.equals(tripId))).watch();

  return combineLatest2(stops, contacts, computeReadiness);
}

/// Writes the blocking items into the checklist so they sit alongside "pack
/// leech socks", which is the correct place for them: a call you have not made
/// is a thing you have not packed.
///
/// REGENERATION NEVER DISCARDS A USER'S EDIT. An item with `isUserEdited` set
/// survives untouched, whatever the generator now thinks. That column exists
/// for exactly this, and the rule is decided here rather than at #29 because
/// this is the first generator to run.
Future<void> syncReadinessChecklist(AppDatabase db, int tripId) async {
  final readiness = await watchReadiness(db, tripId).first;

  await db.transaction(() async {
    final existing =
        await (db.select(db.checklistItems)..where(
              (i) => i.tripId.equals(tripId) & i.isBlocking.equals(true),
            ))
            .get();

    final keep = <int>{};

    for (final item in readiness.blocking) {
      final match = existing.firstWhereOrNullCompat(
        (e) => e.stopId == item.stopId,
      );

      if (match == null) {
        await db.into(db.checklistItems).insert(
          ChecklistItemsCompanion.insert(
            tripId: tripId,
            label: item.label,
            stopId: Value(item.stopId),
            contactId: Value(item.contactId),
            isBlocking: const Value(true),
            sourceTags: const Value('readiness'),
          ),
        );
        continue;
      }

      keep.add(match.id);
      if (match.isUserEdited) continue; // Their wording wins over ours.

      await (db.update(db.checklistItems)..where((i) => i.id.equals(match.id)))
          .write(
            ChecklistItemsCompanion(
              label: Value(item.label),
              contactId: Value(item.contactId),
              isDone: const Value(false),
            ),
          );
    }

    // Anything no longer blocking is done, not deleted — a generated item the
    // user ticked off should stay visible as something that got handled.
    for (final e in existing) {
      if (keep.contains(e.id)) continue;
      if (e.isUserEdited) continue;
      await (db.update(db.checklistItems)..where((i) => i.id.equals(e.id)))
          .write(const ChecklistItemsCompanion(isDone: Value(true)));
    }
  });
}

extension<T> on List<T> {
  T? firstWhereOrNullCompat(bool Function(T) test) {
    for (final e in this) {
      if (test(e)) return e;
    }
    return null;
  }
}
