// Issue #55 — the settling maths.
//
// This is the part that has to be exactly right. Money is stored in minor
// units precisely so a split never loses a paisa, and a settlement that does
// not sum back to the balances is a bug people would find on a trip, in a
// car park, arguing.

import 'package:flutter_test/flutter_test.dart';

import 'package:safarsathi/features/money/data/settlement.dart';

Balance b(int id, String name, int net) =>
    Balance(travellerId: id, name: name, netMinor: net);

void main() {
  group('even shares', () {
    test('an amount that divides cleanly splits cleanly', () {
      expect(evenShares(90000, 3), [30000, 30000, 30000]);
    });

    test('a remainder is handed out one unit at a time', () {
      // Rs 3,200.11 three ways. Not three rounded values that lose a paisa.
      expect(evenShares(320011, 3), [106671, 106670, 106670]);
    });

    test('shares always sum back to the total', () {
      for (final total in [1, 7, 99, 320011, 1000000, 86000]) {
        for (final people in [1, 2, 3, 4, 5, 7]) {
          final shares = evenShares(total, people);
          expect(
            shares.fold<int>(0, (a, s) => a + s),
            total,
            reason: '$total across $people',
          );
          expect(shares, hasLength(people));
        }
      }
    });

    test('nobody to split across is empty, not a crash', () {
      expect(evenShares(1000, 0), isEmpty);
    });
  });

  group('simplify debts', () {
    test('everyone square produces no payments', () {
      expect(simplifyDebts([b(1, 'You', 0), b(2, 'Ankit', 0)]), isEmpty);
    });

    test('one debtor and one creditor is one payment', () {
      final out = simplifyDebts([b(1, 'You', 5000), b(2, 'Ankit', -5000)]);
      expect(out, hasLength(1));
      expect(out.single.fromName, 'Ankit');
      expect(out.single.toName, 'You');
      expect(out.single.amountMinor, 5000);
    });

    test('two debtors paying one creditor is two payments', () {
      final out = simplifyDebts([
        b(1, 'You', 9000),
        b(2, 'Ankit', -4000),
        b(3, 'Priya', -5000),
      ]);
      expect(out, hasLength(2));
      expect(out.every((s) => s.toName == 'You'), isTrue);
    });

    test('the payments always clear the balances exactly', () {
      final cases = <List<Balance>>[
        [b(1, 'You', 481341), b(2, 'Ankit', -155337), b(3, 'Priya', -326004)],
        [b(1, 'A', 100), b(2, 'B', -33), b(3, 'C', -33), b(4, 'D', -34)],
        [b(1, 'A', 7), b(2, 'B', -7)],
        [b(1, 'A', 50), b(2, 'B', 50), b(3, 'C', -100)],
      ];

      for (final balances in cases) {
        final settlements = simplifyDebts(balances);
        final net = {for (final x in balances) x.name: x.netMinor};
        for (final s in settlements) {
          net[s.fromName] = net[s.fromName]! + s.amountMinor;
          net[s.toName] = net[s.toName]! - s.amountMinor;
        }
        for (final entry in net.entries) {
          expect(
            entry.value,
            0,
            reason: '${entry.key} left holding ${entry.value}',
          );
        }
      }
    });

    test('it never needs more payments than there are people', () {
      final balances = [
        b(1, 'A', 481341),
        b(2, 'B', -155337),
        b(3, 'C', -326004),
      ];
      expect(simplifyDebts(balances).length, lessThan(balances.length));
    });

    test('no payment is ever zero', () {
      final out = simplifyDebts([
        b(1, 'A', 100),
        b(2, 'B', 0),
        b(3, 'C', -100),
      ]);
      expect(out.every((s) => s.amountMinor > 0), isTrue);
    });
  });

  group('formatting', () {
    test('groups the Indian way', () {
      expect(formatRupees(100), '₹1');
      expect(formatRupees(99900), '₹999');
      expect(formatRupees(320000), '₹3,200');
      expect(formatRupees(10000000), '₹1,00,000');
      expect(formatRupees(120601100), '₹12,06,011');
    });

    test('PAISE ARE SHOWN, never rounded away', () {
      // Rounding read better until the split screen put three shares of a
      // ₹3,200.11 taxi beside their total: ₹1,067 three times against
      // ₹3,200, under a line claiming they added up exactly.
      expect(formatRupees(320011), '₹3,200.11');
      expect(formatRupees(106671), '₹1,066.71');
      expect(formatRupees(5), '₹0.05');
    });

    test('a whole amount carries no decimal point', () {
      expect(formatRupees(320000), '₹3,200');
      expect(formatRupees(0), '₹0');
    });

    test('three shares still read as their total', () {
      final shares = evenShares(320011, 3);
      final shown = shares.map(formatRupees).toSet();
      // Two distinct values, and both say their paise, so the arithmetic is
      // checkable on screen.
      expect(shown.length, 2);
      expect(shown.every((s) => s.contains('.')), isTrue);
    });

    test('a negative balance reads as a minus, not a bracket', () {
      expect(formatRupees(-155337), startsWith('−₹'));
    });

    test('a credit can be shown with its sign', () {
      expect(formatRupees(481341, signed: true), startsWith('+₹'));
    });
  });
}
