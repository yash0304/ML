// lib/features/money/data/money_summary.dart
//
// What the Money screen needs, in one stream.

import '../../../core/database/app_database.dart';
import 'settlement.dart';

class LedgerEntry {
  final int id;
  final String description;
  final int amountMinor;
  final String paidByName;
  final int splitCount;
  final DateTime spentAt;

  const LedgerEntry({
    required this.id,
    required this.description,
    required this.amountMinor,
    required this.paidByName,
    required this.splitCount,
    required this.spentAt,
  });
}

class MoneySummary {
  final int totalMinor;
  final int travellerCount;
  final List<Balance> balances;
  final List<Settlement> settlements;
  final List<LedgerEntry> ledger;

  const MoneySummary({
    required this.totalMinor,
    required this.travellerCount,
    required this.balances,
    required this.settlements,
    required this.ledger,
  });

  int get perHeadMinor =>
      travellerCount == 0 ? 0 : totalMinor ~/ travellerCount;

  /// How many payments the simplification saved. The ledger would otherwise
  /// be settled one expense at a time.
  int get paymentsSaved {
    final naive = ledger.fold<int>(0, (a, e) => a + (e.splitCount - 1));
    final saved = naive - settlements.length;
    return saved < 0 ? 0 : saved;
  }
}

Stream<MoneySummary> watchMoneySummary(AppDatabase db, int tripId) {
  // Names the real dependency set. A Drift stream only fires for the tables
  // its own query touches, so watching `expenses` alone left the screen stale
  // whenever a traveller was added or a split edited — invisible until #31
  // made either possible. Same bug the trip summary had at #16.
  final tick = db
      .customSelect(
        'SELECT 1',
        readsFrom: {db.expenses, db.expenseSplits, db.travellers},
      )
      .watch();

  return tick.asyncMap((_) async {
    final rows = await (db.select(
      db.expenses,
    )..where((e) => e.tripId.equals(tripId))).get();

    final travellers = await (db.select(
      db.travellers,
    )..where((t) => t.tripId.equals(tripId))).get();
    final nameOf = {for (final t in travellers) t.id: t.name};

    final splits = await db.select(db.expenseSplits).get();

    // Net position per traveller: what they paid out, less what was theirs.
    final net = {for (final t in travellers) t.id: 0};
    final ledger = <LedgerEntry>[];
    var total = 0;

    for (final e
        in rows.toList()..sort((a, b) => b.spentAt.compareTo(a.spentAt))) {
      final mine = splits.where((s) => s.expenseId == e.id).toList();
      total += e.amountMinor;
      net[e.paidById] = (net[e.paidById] ?? 0) + e.amountMinor;
      for (final s in mine) {
        net[s.travellerId] = (net[s.travellerId] ?? 0) - s.shareMinor;
      }
      ledger.add(
        LedgerEntry(
          id: e.id,
          description: e.description,
          amountMinor: e.amountMinor,
          paidByName: nameOf[e.paidById] ?? '—',
          splitCount: mine.length,
          spentAt: e.spentAt,
        ),
      );
    }

    final balances = [
      for (final t in travellers)
        Balance(travellerId: t.id, name: t.name, netMinor: net[t.id] ?? 0),
    ]..sort((a, b) => b.netMinor.compareTo(a.netMinor));

    return MoneySummary(
      totalMinor: total,
      travellerCount: travellers.length,
      balances: balances,
      settlements: simplifyDebts(balances),
      ledger: ledger,
    );
  });
}
