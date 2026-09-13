// lib/features/money/data/expense_editor.dart
//
// Writing to the ledger — issue #31.
//
// EVERY FIGURE IS INTEGER MINOR UNITS. Paise, not rupees. Nothing anywhere
// holds a double, because a three-way split of ₹100 has to still be ₹100 when
// you add it back up, and a ledger that does not balance is worse than no
// ledger at all.

import 'package:drift/drift.dart';

import '../../../core/database/app_database.dart';
import 'settlement.dart';

/// Parses what a person actually types into exact minor units.
///
/// Has to survive `1,234.50`, `₹340`, `340`, `340.5`, and `340.567` — which
/// TRUNCATES rather than rounding up into money nobody spent. Returns null
/// when there is no number in there at all, so the form can refuse rather
/// than silently recording zero.
int? parseRupees(String input) {
  var text = input.trim();
  if (text.isEmpty) return null;

  // Strip currency marks, spaces and the Indian grouping commas.
  text = text.replaceAll(RegExp(r'[₹$€£,\s]'), '');
  if (text.isEmpty) return null;

  final negative = text.startsWith('-');
  if (negative) text = text.substring(1);

  final parts = text.split('.');
  if (parts.length > 2) return null;

  final whole = parts[0].isEmpty ? '0' : parts[0];
  if (!RegExp(r'^\d+$').hasMatch(whole)) return null;

  var fraction = parts.length == 2 ? parts[1] : '';
  if (fraction.isNotEmpty && !RegExp(r'^\d+$').hasMatch(fraction)) return null;
  // Two digits, truncated. 340.567 is 340.56, not 340.57: the extra paisa
  // would be money the ledger invented.
  fraction = fraction.padRight(2, '0').substring(0, 2);

  final minor = int.parse(whole) * 100 + int.parse(fraction);
  return negative ? -minor : minor;
}

/// Minor units back to a plain editable string. Not `formatRupees` — that one
/// groups and prefixes for display, and putting `₹1,234.50` back in a text
/// field to be edited is unkind.
String rupeesFieldText(int minor) {
  final sign = minor < 0 ? '-' : '';
  final abs = minor.abs();
  return '$sign${abs ~/ 100}.${(abs % 100).toString().padLeft(2, '0')}';
}

/// One expense as the form passes it around.
class ExpenseDraft {
  final int? id;
  final String description;
  final int amountMinor;
  final int paidById;

  /// Traveller id to their share, in minor units. Must sum to [amountMinor].
  final Map<int, int> shares;

  final DateTime spentAt;
  final String? category;
  final int? stopId;
  final String currency;
  final double rateToBase;
  final DateTime? rateCapturedAt;

  const ExpenseDraft({
    required this.description,
    required this.amountMinor,
    required this.paidById,
    required this.shares,
    required this.spentAt,
    this.id,
    this.category,
    this.stopId,
    this.currency = 'INR',
    this.rateToBase = 1.0,
    this.rateCapturedAt,
  });

  int get shareTotal => shares.values.fold(0, (a, b) => a + b);

  /// THE FORM WILL NOT SAVE UNTIL THIS IS TRUE. Shares that do not sum to the
  /// amount put the ledger permanently out by the difference, and nothing
  /// downstream would ever notice.
  bool get balances => shareTotal == amountMinor;

  /// How far out, for the message that tells the user which way to go.
  int get outBy => amountMinor - shareTotal;
}

/// An even split across [travellerIds], remainder handed out one minor unit
/// at a time so the shares always sum back to [amountMinor] exactly.
Map<int, int> evenSplit(int amountMinor, List<int> travellerIds) {
  if (travellerIds.isEmpty) return const {};
  final shares = evenShares(amountMinor, travellerIds.length);
  return {
    for (var i = 0; i < travellerIds.length; i++) travellerIds[i]: shares[i],
  };
}

class TravellerInUse implements Exception {
  final String name;
  final int expenseCount;
  const TravellerInUse(this.name, this.expenseCount);

  @override
  String toString() =>
      '$name appears in $expenseCount ${expenseCount == 1 ? 'expense' : 'expenses'}.';
}

class ExpenseEditor {
  final AppDatabase db;
  const ExpenseEditor(this.db);

  // -- travellers ----------------------------------------------------------

  Stream<List<Traveller>> watchTravellers(int tripId) =>
      (db.select(db.travellers)
            ..where((t) => t.tripId.equals(tripId))
            ..orderBy([(t) => OrderingTerm(expression: t.id)]))
          .watch();

  Future<List<Traveller>> travellersOf(int tripId) =>
      (db.select(db.travellers)
            ..where((t) => t.tripId.equals(tripId))
            ..orderBy([(t) => OrderingTerm(expression: t.id)]))
          .get();

  Future<int> addTraveller(int tripId, String name) => db
      .into(db.travellers)
      .insert(TravellersCompanion.insert(tripId: tripId, name: name.trim()));

  Future<void> renameTraveller(int id, String name) =>
      (db.update(db.travellers)..where((t) => t.id.equals(id))).write(
        TravellersCompanion(name: Value(name.trim())),
      );

  /// How many expenses a traveller is tangled up in, either as payer or as a
  /// share. Used to refuse the delete with a reason.
  Future<int> expenseCountFor(int travellerId) async {
    final paid = await (db.select(
      db.expenses,
    )..where((e) => e.paidById.equals(travellerId))).get();
    final split = await (db.select(
      db.expenseSplits,
    )..where((s) => s.travellerId.equals(travellerId))).get();
    return {...paid.map((e) => e.id), ...split.map((s) => s.expenseId)}.length;
  }

  /// REFUSES rather than cascading.
  ///
  /// The schema cascades a traveller's splits, which would quietly change
  /// everyone else's balance: the expense keeps its total but loses a share,
  /// so the payer is suddenly owed more than they are. Better to say no and
  /// name the number.
  Future<void> deleteTraveller(Traveller traveller) async {
    final count = await expenseCountFor(traveller.id);
    if (count > 0) throw TravellerInUse(traveller.name, count);
    await (db.delete(
      db.travellers,
    )..where((t) => t.id.equals(traveller.id))).go();
  }

  // -- expenses ------------------------------------------------------------

  Future<Map<int, int>> sharesOf(int expenseId) async {
    final rows = await (db.select(
      db.expenseSplits,
    )..where((s) => s.expenseId.equals(expenseId))).get();
    return {for (final r in rows) r.travellerId: r.shareMinor};
  }

  /// Saves an expense and its splits as one transaction.
  ///
  /// The balance check is asserted here as well as in the form. The form is a
  /// convenience; this is the guarantee.
  Future<int> saveExpense(int tripId, ExpenseDraft draft) async {
    if (!draft.balances) {
      throw ArgumentError(
        'Shares total ${draft.shareTotal} but the expense is '
        '${draft.amountMinor}.',
      );
    }

    return db.transaction(() async {
      final companion = ExpensesCompanion(
        tripId: Value(tripId),
        description: Value(draft.description.trim()),
        amountMinor: Value(draft.amountMinor),
        paidById: Value(draft.paidById),
        spentAt: Value(draft.spentAt),
        category: Value(draft.category),
        stopId: Value(draft.stopId),
        currency: Value(draft.currency),
        rateToBase: Value(draft.rateToBase),
        rateCapturedAt: Value(draft.rateCapturedAt),
      );

      final int id;
      if (draft.id == null) {
        id = await db.into(db.expenses).insert(companion);
      } else {
        id = draft.id!;
        await (db.update(
          db.expenses,
        )..where((e) => e.id.equals(id))).write(companion);
        // Replaced wholesale rather than diffed: a split whose traveller was
        // removed from the expense has to go, and the unique key on
        // (expense, traveller) makes a partial update fiddly for no gain.
        await (db.delete(
          db.expenseSplits,
        )..where((s) => s.expenseId.equals(id))).go();
      }

      for (final entry in draft.shares.entries) {
        await db
            .into(db.expenseSplits)
            .insert(
              ExpenseSplitsCompanion.insert(
                expenseId: id,
                travellerId: entry.key,
                shareMinor: entry.value,
              ),
            );
      }
      return id;
    });
  }

  Future<void> deleteExpense(int id) =>
      (db.delete(db.expenses)..where((e) => e.id.equals(id))).go();
}
