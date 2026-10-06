# #39 — Multi-currency expenses

**Backlog line:** Manual exchange-rate snapshot per country at setup.
*Depends on #31.*

## Why

A trip that crosses a border (Dawki → Tamabil into Bangladesh, Phuentsholing
into Bhutan, the Europe trip) spends in more than one currency. There is no
live rate offline, so the rate is one the person types at setup, shown with
the date they typed it, and every expense keeps the rate it was saved with.

## What exists

`Expenses.currency` (default INR), `Expenses.rateToBase` (default 1.0) and
`Expenses.rateCapturedAt` have been in the schema since v1, unused.

## Build

1. **Schema v9: `CurrencyRates`.** Columns: trip, ISO code, rate to INR,
   captured at. Unique on (trip, code). In backups after `trips`.
2. **Currencies screen** (Money → CURRENCIES, and More): list each code with
   "1 EUR = ₹92.40 · saved 6 Oct". Add one with a code and a rate. Changing a
   rate never rewrites saved expenses, and the screen says so.
3. **Expense form:** currency chips (INR plus the trip's currencies). The
   amount and shares are in that currency. Under the amount, the INR value
   and the rate used, with its date. On save, the expense stores the rate
   snapshot.
4. **Ledger and balances in INR.** Each share converts on its own
   (`round(share × rate)`); the expense's INR amount is the sum of the
   converted shares. Balances and settlements stay exact in paise.
5. **Ledger rows** show the original amount with its INR value: "€12.50 ·
   ₹1,155.00".
6. **Export** gains Currency, Amount (original) and Rate columns. Shares and
   totals stay in INR.

## Acceptance

- [x] A EUR expense split three ways settles exactly in paise.
- [x] Changing a rate leaves saved expenses unchanged.
- [x] An INR-only trip reads exactly as before.
- [x] A v8 backup restores; a v9 backup carries the rates.
- [x] A real v8 database upgrades.
