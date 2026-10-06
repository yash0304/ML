// lib/features/money/data/currency.dart
//
// Spending in more than one currency, with no live rate — #39.
//
// The base is always INR: balances, settlements and totals are in rupees,
// because that is the money the group settles in at home. An expense keeps
// its own currency and amount, and the rate it was saved with. A rate is
// typed by the person and shown with the day they typed it.

import 'package:drift/drift.dart';

import '../../../core/database/app_database.dart';
import '../../../core/database/watch_tables.dart';
import 'settlement.dart';

const baseCurrency = 'INR';

/// Signs for the currencies a trip from India most often meets. Anything
/// else shows its ISO code, which is never wrong.
const _symbols = <String, String>{
  'INR': '₹',
  'EUR': '€',
  'USD': r'$',
  'GBP': '£',
  'BDT': '৳',
  'THB': '฿',
  'JPY': '¥',
  'NPR': 'Rs ',
  'BTN': 'Nu. ',
  'LKR': 'Rs ',
  'AED': 'AED ',
  'SGD': r'S$',
  'CHF': 'CHF ',
};

/// "€12.50", "৳1,250.00", "CZK 340.00". INR keeps the Indian grouping.
String formatMoney(int minor, String currency, {bool signed = false}) {
  if (currency == baseCurrency) return formatRupees(minor, signed: signed);
  final abs = minor.abs();
  final digits = (abs ~/ 100).toString();
  final grouped = digits.replaceAllMapped(
    RegExp(r'\B(?=(\d{3})+(?!\d))'),
    (_) => ',',
  );
  final sign = minor < 0
      ? '−'
      : signed
      ? '+'
      : '';
  return '$sign${_symbols[currency] ?? '$currency '}$grouped.'
      '${(abs % 100).toString().padLeft(2, '0')}';
}

/// What a sign reads as in a form's prefix.
String currencyPrefix(String currency) =>
    (_symbols[currency] ?? '$currency ').trimRight();

/// [minor] of a currency in paise, at [rate] rupees per unit.
int toBaseMinor(int minor, double rate) => (minor * rate).round();

/// Shares converted one by one. The expense's rupee amount is their sum, so
/// balances settle exactly in paise however the rounding falls.
Map<int, int> baseShares(Map<int, int> shares, double rate) => {
  for (final s in shares.entries) s.key: toBaseMinor(s.value, rate),
};

/// "92.40" or "92,40" → 92.4. Null for anything that is not a positive rate.
double? parseRate(String input) {
  final v = double.tryParse(input.trim().replaceAll(',', '.'));
  if (v == null || v <= 0 || v.isNaN || v.isInfinite) return null;
  return v;
}

/// Three letters, upper case, or null.
String? parseCurrencyCode(String input) {
  final code = input.trim().toUpperCase();
  return RegExp(r'^[A-Z]{3}$').hasMatch(code) ? code : null;
}

Stream<List<CurrencyRate>> watchCurrencyRates(AppDatabase db, int tripId) =>
    watchTables(db, {db.currencyRates}).asyncMap(
      (_) => (db.select(db.currencyRates)
            ..where((r) => r.tripId.equals(tripId))
            ..orderBy([(r) => OrderingTerm(expression: r.code)]))
          .get(),
    );

/// Saves [code] at [rate], stamped now. Replaces an earlier rate for the same
/// code. Expenses already saved keep the rate they were saved with.
Future<void> saveCurrencyRate(
  AppDatabase db, {
  required int tripId,
  required String code,
  required double rate,
  DateTime? now,
}) async {
  if (code == baseCurrency) {
    throw ArgumentError('INR is the base currency; it has no rate.');
  }
  await db.into(db.currencyRates).insert(
    CurrencyRatesCompanion.insert(
      tripId: tripId,
      code: code,
      rateToBase: rate,
      capturedAt: now ?? DateTime.now(),
    ),
    onConflict: DoUpdate(
      (_) => CurrencyRatesCompanion(
        rateToBase: Value(rate),
        capturedAt: Value(now ?? DateTime.now()),
      ),
      target: [db.currencyRates.tripId, db.currencyRates.code],
    ),
  );
}

Future<void> deleteCurrencyRate(AppDatabase db, int id) =>
    (db.delete(db.currencyRates)..where((r) => r.id.equals(id))).go();

/// "1 EUR = ₹92.40 · saved 6 Oct".
String describeRate(CurrencyRate r) {
  const months = [
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
  ];
  final d = r.capturedAt;
  return '1 ${r.code} = ${formatRupees(toBaseMinor(100, r.rateToBase))} · '
      'saved ${d.day} ${months[d.month - 1]}';
}
