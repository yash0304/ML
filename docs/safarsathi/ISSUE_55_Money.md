# Issue #55 — Money screen

**Size:** M · **Depends on:** #53 · **Spec:** SCREENS.md §8, DESIGN_VISUAL_v2.md §3

## Why this issue exists

Splitwise was one of the five apps this project set out to absorb, and it is
the one people miss hardest offline — the moment you need to know who owes
whom is at the end of a day, in a room with no signal, with everyone present.

#55 delivers the useful half of #31/#32 against data that already exists.
Entering a new expense stays in #31. Reading the ledger and clearing it is
here.

## What it must show

1. **Balances**, one per traveller, on a ticket stub.
2. **Settle-up above the ledger.** The fewest payments that clear the trip.
   Most ledger apps bury this under a tab; it is the only part anyone acts on,
   so it goes first.
3. **The ledger itself** underneath, newest first.

## Build steps

1. `lib/features/money/data/settlement.dart` — pure functions, no Drift
   import, so the maths is testable without a database:
   - `evenShares(totalMinor, n)` — remainder handed out one minor unit at a
     time, so shares always sum back to the total.
   - `simplifyDebts(balances)` — greedy largest-debt-to-largest-credit.
   - `formatRupees(minor)` — **Indian grouping** (1,23,456), not the Western
     three-digit grouping `NumberFormat` gives you by default.
2. `lib/features/money/data/money_summary.dart` — the stream.
3. `lib/features/money/presentation/money_screen.dart` — layout only.

## The rules that are easy to get wrong

- **Money is integer minor units — paise — everywhere.** Never a double. A
  three-way split of ₹100 is 3334/3333/3333, and it has to still be ₹100 when
  you add it back up.
- **A negative balance is never red.** Red in this app means emergency, and
  owing your brother-in-law ₹400 is not an emergency. Negative renders muted,
  positive in signal green.
- **The app records a settlement, it never executes one.** No payment
  integration, ever — it would reintroduce a network dependency and a
  compliance surface for something people do with cash or UPI in thirty
  seconds.
- **Courier Prime has no rupee glyph.** It is a 1950s typewriter design and ₹
  was adopted in 2010. `numberStyle` needs `fontFamilyFallback: [Archivo]` or
  every amount on this screen renders as a tofu box. Pinned by test.

## Acceptance criteria

- [x] `flutter analyze` clean.
- [x] Even splits sum back to the total for every n from 1 to 12.
- [x] Simplified payments sum back to the balances exactly.
- [x] Rupee amounts use Indian grouping.
- [x] The rupee sign renders — asserted, not eyeballed.
- [x] No red on a negative balance.
- [x] Golden `money.png` committed.
