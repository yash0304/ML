// lib/features/money/data/expense_export.dart
//
// The ledger as a spreadsheet: every expense, who paid, each person's share,
// the tag, then the totals by tag and who pays whom.
//
// CSV, because every phone and every spreadsheet opens it, and it is the one
// format somebody can paste into the group chat's sheet without the app
// being installed on their side. Amounts are plain numbers in rupees with
// two decimals — "2400.00", not "₹2,400" — so a spreadsheet can add them.

import 'package:drift/drift.dart';

import '../../../core/database/app_database.dart';
import 'currency.dart';
import 'expense_tags.dart';
import 'settlement.dart';

String _cell(Object? v) {
  final s = v?.toString() ?? '';
  if (s.contains(RegExp(r'[",\n\r]'))) return '"${s.replaceAll('"', '""')}"';
  return s;
}

String _row(List<Object?> cells) => cells.map(_cell).join(',');

String _rupees(int minor) {
  final sign = minor < 0 ? '-' : '';
  final abs = minor.abs();
  return '$sign${abs ~/ 100}.${(abs % 100).toString().padLeft(2, '0')}';
}

String _day(DateTime d) =>
    '${d.year}-${d.month.toString().padLeft(2, '0')}-'
    '${d.day.toString().padLeft(2, '0')}';

/// The CSV text for [tripId]. Starts with a byte-order mark: without it,
/// Excel on Windows reads the file as Latin-1 and turns names typed in
/// Hindi or Khasi into mojibake.
Future<String> expensesCsv(AppDatabase db, int tripId) async {
  final travellers = await (db.select(db.travellers)
        ..where((t) => t.tripId.equals(tripId))
        ..orderBy([(t) => OrderingTerm(expression: t.id)]))
      .get();
  final expenses = await (db.select(db.expenses)
        ..where((e) => e.tripId.equals(tripId))
        ..orderBy([
          (e) => OrderingTerm(expression: e.spentAt),
          (e) => OrderingTerm(expression: e.id),
        ]))
      .get();
  final splits = expenses.isEmpty
      ? const <ExpenseSplit>[]
      : await (db.select(db.expenseSplits)..where(
              (s) => s.expenseId.isIn([for (final e in expenses) e.id]),
            ))
            .get();
  final stops = await (db.select(
    db.stops,
  )..where((s) => s.tripId.equals(tripId))).get();

  final nameOf = {for (final t in travellers) t.id: t.name};
  final stopName = {for (final s in stops) s.id: s.name};
  final out = StringBuffer('﻿');

  out.writeln(_row([
    'Date',
    'What for',
    'Tag',
    'Amount (INR)',
    'Currency',
    'Amount spent',
    'Rate to INR',
    'Paid by',
    for (final t in travellers) '${t.name} share',
    'Stop',
  ]));

  var total = 0;
  final byTag = <String?, int>{};
  final net = {for (final t in travellers) t.id: 0};
  for (final e in expenses) {
    // Rupees share by share, exactly as the Money tab counts them (#39).
    final shares = baseShares({
      for (final s in splits)
        if (s.expenseId == e.id) s.travellerId: s.shareMinor,
    }, e.rateToBase);
    final amount = shares.isEmpty
        ? toBaseMinor(e.amountMinor, e.rateToBase)
        : shares.values.fold(0, (a, b) => a + b);
    total += amount;
    byTag[e.category] = (byTag[e.category] ?? 0) + amount;
    net[e.paidById] = (net[e.paidById] ?? 0) + amount;
    for (final s in shares.entries) {
      net[s.key] = (net[s.key] ?? 0) - s.value;
    }
    out.writeln(_row([
      _day(e.spentAt),
      e.description,
      tagLabel(e.category),
      _rupees(amount),
      e.currency,
      _rupees(e.amountMinor),
      e.rateToBase,
      nameOf[e.paidById] ?? '',
      for (final t in travellers)
        shares.containsKey(t.id) ? _rupees(shares[t.id]!) : '',
      e.stopId == null ? '' : stopName[e.stopId] ?? '',
    ]));
  }

  out
    ..writeln()
    ..writeln(_row(['Total', '', '', _rupees(total)]))
    ..writeln()
    ..writeln(_row(['By tag', '', '', 'Amount (INR)']));
  final tags = byTag.entries.toList()..sort((a, b) => b.value - a.value);
  for (final t in tags) {
    out.writeln(_row([tagLabel(t.key), '', '', _rupees(t.value)]));
  }

  final balances = [
    for (final t in travellers)
      Balance(travellerId: t.id, name: t.name, netMinor: net[t.id] ?? 0),
  ];
  out
    ..writeln()
    ..writeln(_row(['Settle up: who pays', 'to whom', '', 'Amount (INR)']));
  final settle = simplifyDebts(balances);
  if (settle.isEmpty) {
    out.writeln(_row(['Everyone is square']));
  } else {
    for (final s in settle) {
      out.writeln(_row([s.fromName, s.toName, '', _rupees(s.amountMinor)]));
    }
  }
  return out.toString();
}

/// A file name that sorts and says what it is: "Meghalaya-expenses-2026-10-06.csv".
String expensesFileName(String tripName, DateTime now) {
  final safe = tripName
      .replaceAll(RegExp(r'[^A-Za-z0-9]+'), '-')
      .replaceAll(RegExp(r'^-+|-+$'), '');
  return '${safe.isEmpty ? 'Trip' : safe}-expenses-${_day(now)}.csv';
}
