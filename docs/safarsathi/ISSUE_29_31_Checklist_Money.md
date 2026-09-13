# Issues #29, #31, #32 — Checklist and the ledger

**Size:** M + M + (already done) · **Depends on:** #16, #20 · **Spec:** SCREENS.md §7, §8

Two half-built things finished. The readiness panel from #20 is the blocking
section of a checklist that had no other sections. The money screen from #55
reads a ledger nothing could write to.

---

## #29 — Checklist generation

**Files:** `features/checklist/data/checklist_generator.dart`,
`presentation/checklist_screen.dart`

### What generates an item

Three inputs, all already in the database: a stop's **activity tags**, its
**nights**, and the trip's **stop count**. No weather lookup at generation
time — the forecast is a cached snapshot and #29 does not depend on one
existing.

Rules are declared as data, not code: a tag maps to a list of items, each with
a quantity that may scale with nights. `trek` gives you boots and a poncho;
`caves` gives a headtorch and spare batteries; `rain` in Meghalaya in October
gives a dry bag, because that is the trip this app was built for.

### The rule that is the whole issue

**A manual edit survives regeneration.** `isUserEdited` already exists and #20
already honours it for blocking items. #29 extends the same rule to generated
pack items: change a label, change a quantity, or delete an item, and
regenerating must not undo it.

Deleting needs care. A deleted generated item would simply come back, so a
delete on a generated item marks it edited-and-done rather than removing the
row. That is the only way "I do not need leech socks" can survive a
regeneration without a tombstone table.

### The screen

- A 3px progress hairline under the app bar, filled in `signal`.
- **`BEFORE YOU LEAVE SIGNAL`** first — the blocking items from #20, carrying
  the amber dot and a `BLOCKING` tag. "Call the homestay" sits in the same
  list as "pack leech socks", which is the correct place for it.
- **`PACK`** — quantity on the right, the tags that produced the item
  underneath. A generated list nobody understands gets ignored.
- The tick is a stencil mark in an 18px box, never a Material checkbox.
  Checked items go muted and struck through.
- A **`WHY THESE`** block naming the inputs and promising edits survive.

## #31 — Expenses and splits

**Files:** `features/money/data/expense_editor.dart`,
`presentation/expense_form_screen.dart`, `travellers_screen.dart`

### Money handling

Every figure is **integer minor units**. The form takes rupees as text and
parses to paise; nothing anywhere holds a double. Parsing has to survive what
people actually type: `1,234.50`, `1234`, `₹340`, `340.5`, `340.567` (which
truncates rather than rounding up into money nobody spent).

### Splitting

Even by default, across the selected travellers, with the remainder handed out
one minor unit at a time so the shares always sum back to the total. Custom
shares are allowed, and the form **will not save until they sum exactly** —
a ledger that does not balance is worse than no ledger.

### Travellers

Named people with no accounts, no invitations and no sync. Deleting a
traveller who appears in the ledger is refused with a reason, not silently
cascaded: `KeyAction.cascade` on `ExpenseSplits.travellerId` would quietly
change everyone else's balance.

### Currency

A per-expense currency with a manual rate and the date it was captured. Shown
with that date wherever it appears, because a rate with no date is a number
pretending to be a fact.

## #32 — Simplify debts

Already built at #55: `simplifyDebts`, greedy largest-debt-to-largest-credit,
exact rather than provably minimal. #31 gives it a real ledger to run on, and
this issue adds the two tests the backlog asks for by name — the three-person
Meghalaya ledger, and a cycle.

---

## Acceptance

**#29** — [ ] Tags and nights produce items. [ ] Generated items show their
source tags. [ ] A manual edit survives regeneration. [ ] A deleted generated
item stays deleted. [ ] Blocking items render in the same list.

**#31** — [ ] Rupee text parses to exact paise. [ ] An even split sums back to
the total. [ ] Custom shares that do not sum cannot be saved. [ ] A traveller
in the ledger cannot be deleted. [ ] A rate is never shown without its date.

**#32** — [ ] The three-person Meghalaya ledger settles in two payments.
[ ] A cycle collapses to nothing.
