// lib/features/money/data/money_summary.dart
//
// What the Money screen needs, in one stream.

import '../../../core/database/app_database.dart';
import 'currency.dart';
import 'settlement.dart';
import '../../../core/database/watch_tables.dart';

class LedgerEntry {
  final int id;
  final String description;
  final int amountMinor;
  final String paidByName;
  final int splitCount;
  final DateTime spentAt;

  /// What it was for (an expense_tags key), or null when untagged.
  final String? tag;

  /// The amount as spent, in [currency]. [amountMinor] is always paise.
  final int? originalMinor;
  final String currency;

  bool get isForeign => currency != baseCurrency;

  const LedgerEntry({
    required this.id,
    required this.description,
    required this.amountMinor,
    required this.paidByName,
    required this.splitCount,
    required this.spentAt,
    this.tag,
    this.originalMinor,
    this.currency = baseCurrency,
  });
}

/// One tag's share of the spending.
class TagTotal {
  final String? tag;
  final int totalMinor;
  final int count;
  const TagTotal({this.tag, required this.totalMinor, required this.count});
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

  /// Where it went: each tag's total, largest first, untagged included so
  /// the parts always add up to the whole.
  List<TagTotal> get byTag {
    final total = <String?, int>{};
    final count = <String?, int>{};
    for (final e in ledger) {
      total[e.tag] = (total[e.tag] ?? 0) + e.amountMinor;
      count[e.tag] = (count[e.tag] ?? 0) + 1;
    }
    return [
      for (final k in total.keys)
        TagTotal(tag: k, totalMinor: total[k]!, count: count[k]!),
    ]..sort((a, b) => b.totalMinor.compareTo(a.totalMinor));
  }

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
  final tick = watchTables(db, {
    db.expenses,
    db.expenseSplits,
    db.travellers,
  });

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
      // IN RUPEES, SHARE BY SHARE. A €12.50 dinner split three ways is
      // converted one share at a time, and the expense's rupee amount is
      // their sum — so what the payer is owed is exactly what the others owe,
      // in paise, however the rounding falls. INR at 1.0 is unchanged.
      final base = baseShares(
        {for (final s in mine) s.travellerId: s.shareMinor},
        e.rateToBase,
      );
      final amount = base.isEmpty
          ? toBaseMinor(e.amountMinor, e.rateToBase)
          : base.values.fold(0, (a, b) => a + b);
      total += amount;
      net[e.paidById] = (net[e.paidById] ?? 0) + amount;
      for (final s in base.entries) {
        net[s.key] = (net[s.key] ?? 0) - s.value;
      }
      ledger.add(
        LedgerEntry(
          id: e.id,
          description: e.description,
          amountMinor: amount,
          originalMinor: e.amountMinor,
          currency: e.currency,
          paidByName: nameOf[e.paidById] ?? '—',
          splitCount: mine.length,
          spentAt: e.spentAt,
          tag: e.category,
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
