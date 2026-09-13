// lib/features/checklist/data/checklist_dao.dart
//
// Reading and ticking the checklist — issue #29.
//
// Every write that touches a generated item sets `isUserEdited`, which is what
// makes the generator leave it alone afterwards. That flag is set HERE rather
// than at each call site so a future screen cannot forget it and quietly
// start losing people's edits.

import 'package:drift/drift.dart';

import '../../../core/database/app_database.dart';
import '../../../core/util/streams.dart';

class ChecklistView {
  /// From #20. "Call the homestay" belongs in the same list as "pack leech
  /// socks", which is the correct place for it.
  final List<ChecklistItem> blocking;
  final List<ChecklistItem> pack;

  const ChecklistView({required this.blocking, required this.pack});

  int get total => blocking.length + pack.length;
  int get done =>
      blocking.where((i) => i.isDone).length +
      pack.where((i) => i.isDone).length;

  /// 0 to 1, for the progress hairline. An empty list reads as zero rather
  /// than as complete: nothing packed is not everything packed.
  double get progress => total == 0 ? 0 : done / total;

  bool get isReady => blocking.every((i) => i.isDone);
}

Stream<ChecklistView> watchChecklist(AppDatabase db, int tripId) {
  final query = db.select(db.checklistItems)
    ..where((i) => i.tripId.equals(tripId))
    ..orderBy([
      (i) => OrderingTerm(expression: i.sortOrder),
      (i) => OrderingTerm(expression: i.id),
    ]);

  return query.watch().map(
    (rows) => ChecklistView(
      blocking: [
        for (final r in rows)
          if (r.isBlocking) r,
      ],
      pack: [
        for (final r in rows)
          if (!r.isBlocking) r,
      ],
    ),
  );
}

/// Live count of what is still open, for the badge on the Trip tab.
Stream<int> watchOpenChecklistCount(AppDatabase db, int tripId) =>
    watchChecklist(db, tripId).map((v) => v.total - v.done);

class ChecklistDao {
  final AppDatabase db;
  const ChecklistDao(this.db);

  /// Ticking is NOT an edit. Everyone ticks things off; treating that as
  /// "the user has taken ownership of this row" would freeze the whole list
  /// against regeneration the first time someone packed a toothbrush.
  Future<void> setDone(int id, bool done) =>
      (db.update(db.checklistItems)..where((i) => i.id.equals(id))).write(
        ChecklistItemsCompanion(isDone: Value(done)),
      );

  /// Changing the words or the count IS an edit, and the flag is set here so
  /// no caller can forget.
  Future<void> edit(int id, {required String label, String? quantity}) =>
      (db.update(db.checklistItems)..where((i) => i.id.equals(id))).write(
        ChecklistItemsCompanion(
          label: Value(label),
          quantity: Value(quantity),
          isUserEdited: const Value(true),
        ),
      );

  Future<int> addOwn(int tripId, String label, {String? quantity}) =>
      db.into(db.checklistItems).insert(
        ChecklistItemsCompanion.insert(
          tripId: tripId,
          label: label,
          quantity: Value(quantity),
          isGenerated: const Value(false),
          isUserEdited: const Value(true),
        ),
      );

  /// Removing an item.
  ///
  /// A GENERATED ITEM IS NOT DELETED — it is marked edited and done. Deleting
  /// the row would simply bring it back on the next regeneration, and "I do
  /// not need leech socks" has to survive. An item the user wrote themselves
  /// has no rule behind it, so that one really goes.
  Future<void> remove(ChecklistItem item) async {
    if (!item.isGenerated) {
      await (db.delete(
        db.checklistItems,
      )..where((i) => i.id.equals(item.id))).go();
      return;
    }
    await (db.update(
      db.checklistItems,
    )..where((i) => i.id.equals(item.id))).write(
      const ChecklistItemsCompanion(
        isDone: Value(true),
        isUserEdited: Value(true),
      ),
    );
  }
}

/// The trip is ready when nothing blocking is open. Combines the checklist
/// with itself rather than re-querying, so the two never disagree.
Stream<bool> watchChecklistReady(AppDatabase db, int tripId) =>
    combineLatest2(
      watchChecklist(db, tripId),
      Stream<void>.value(null),
      (ChecklistView v, void _) => v.isReady,
    );
