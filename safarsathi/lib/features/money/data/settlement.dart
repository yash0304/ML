// lib/features/money/data/settlement.dart
//
// Who owes whom, and the fewest payments that clear it.
//
// All local maths over a list of rows, which is why the whole feature works
// with no signal. Every figure is in MINOR UNITS — paise, cents — because
// floating point accumulates error across a split and this ledger has to
// balance exactly.
//
// THE APP RECORDS A SETTLEMENT, IT NEVER EXECUTES ONE. There is no payment
// integration and there will not be: it would reintroduce a network
// dependency and a compliance surface for something people do with cash or
// UPI in thirty seconds.

class Balance {
  final int travellerId;
  final String name;

  /// Positive when the trip owes them, negative when they owe the trip.
  final int netMinor;

  const Balance({
    required this.travellerId,
    required this.name,
    required this.netMinor,
  });
}

class Settlement {
  final String fromName;
  final String toName;
  final int amountMinor;

  const Settlement({
    required this.fromName,
    required this.toName,
    required this.amountMinor,
  });
}

/// Collapses a set of balances into the fewest payments that clear them.
///
/// Greedy: repeatedly send the largest debt to the largest credit. That is
/// not provably minimal for every graph — the general problem is NP-hard —
/// but it is optimal for the shapes a trip actually produces and it never
/// produces more payments than there are people. What matters more is that it
/// is exact: the payments always sum back to the balances.
List<Settlement> simplifyDebts(List<Balance> balances) {
  final creditors = balances.where((b) => b.netMinor > 0).toList()
    ..sort((a, b) => b.netMinor.compareTo(a.netMinor));
  final debtors = balances.where((b) => b.netMinor < 0).toList()
    ..sort((a, b) => a.netMinor.compareTo(b.netMinor));

  final owed = {for (final b in creditors) b.travellerId: b.netMinor};
  final owes = {for (final b in debtors) b.travellerId: -b.netMinor};
  final nameOf = {for (final b in balances) b.travellerId: b.name};

  final out = <Settlement>[];
  var ci = 0;
  var di = 0;

  while (ci < creditors.length && di < debtors.length) {
    final creditor = creditors[ci];
    final debtor = debtors[di];
    final credit = owed[creditor.travellerId]!;
    final debt = owes[debtor.travellerId]!;
    final amount = credit < debt ? credit : debt;

    if (amount > 0) {
      out.add(
        Settlement(
          fromName: nameOf[debtor.travellerId]!,
          toName: nameOf[creditor.travellerId]!,
          amountMinor: amount,
        ),
      );
      owed[creditor.travellerId] = credit - amount;
      owes[debtor.travellerId] = debt - amount;
    }

    if (owed[creditor.travellerId] == 0) ci++;
    if (owes[debtor.travellerId] == 0) di++;
  }

  return out;
}

/// Splits [totalMinor] across [people], handing the remainder out one unit at
/// a time so the shares always sum back to the total exactly.
///
/// A three-way split of ₹3,200.11 is 106670, 106670, 106671 — never three
/// rounded values that lose a paisa.
List<int> evenShares(int totalMinor, int people) {
  if (people <= 0) return const [];
  final base = totalMinor ~/ people;
  final remainder = totalMinor % people;
  return [for (var i = 0; i < people; i++) base + (i < remainder ? 1 : 0)];
}

/// Formats minor units as rupees. Grouping is Indian: the last three digits,
/// then pairs.
String formatRupees(int minor, {bool signed = false}) {
  final negative = minor < 0;
  final abs = minor.abs();

  // PAISE ARE SHOWN WHENEVER THEY EXIST, and never rounded away.
  //
  // Rounding read better until the split screen put three shares of a
  // ₹3,200.11 taxi next to their total: ₹1,067 three times against ₹3,200,
  // under a line claiming the shares added up exactly. They did, in paise.
  // A ledger that rounds is a ledger that looks wrong at exactly the moment
  // someone checks it.
  final major = abs ~/ 100;
  final paise = abs % 100;
  final digits = major.toString();

  String grouped;
  if (digits.length <= 3) {
    grouped = digits;
  } else {
    final last3 = digits.substring(digits.length - 3);
    var rest = digits.substring(0, digits.length - 3);
    final parts = <String>[];
    while (rest.length > 2) {
      parts.insert(0, rest.substring(rest.length - 2));
      rest = rest.substring(0, rest.length - 2);
    }
    if (rest.isNotEmpty) parts.insert(0, rest);
    grouped = '${parts.join(',')},$last3';
  }

  final sign = negative
      ? '−'
      : signed
      ? '+'
      : '';
  final fraction = paise == 0
      ? ''
      : '.${paise.toString().padLeft(2, '0')}';
  return '$sign₹$grouped$fraction';
}
