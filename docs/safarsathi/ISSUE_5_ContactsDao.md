# Issue #5 — ContactsDao

**Size:** M · **Depends on:** #2 · **Blocks:** #6, #7, #8, #9, #14

The single read/write surface for contact data. The diary never touches the
database directly; it watches these streams.

`docs/safarsathi/code/contacts_dao.dart` was drafted before any of this
compiled, and the schema at #2 was built to its exact column names. It should
drop in with only wiring changes.

---

## Wiring

1. **`app_database.dart` must export `tables.dart`.** The DAO's
   `@DriftAccessor(tables: [Contacts, CallLogs, ...])` needs those classes in
   scope, and the drafted file has a single import of `app_database.dart`.
   Exporting keeps the drafted file unchanged, which is the point.
2. **Register the DAO** in `@DriftDatabase(daos: [ContactsDao])` so the
   database exposes a `contactsDao` getter.
3. Run `build_runner` for the `contacts_dao.g.dart` part file.

---

## Two things in the drafted file that need a decision

### The ordering comment does not match the ordering code

The doc comment says *"Recently-called floats up within each group"*, but
`watchContacts` orders by pinned, then confirmed, then name. There is no
recency term.

**Keep the code, fix the comment.** Two reasons, and the second is the one
that matters:

- The diary is a diary. Page order has to be stable or the margin numbers
  shift under you and "the homestay is entry 3" stops being true.
- `lastCalledAt` is null for anything never called, and SQLite sorts nulls as
  smallest — so `DESC` would sink every never-called number to the bottom of
  its group. Never-called is exactly the category the readiness system is
  trying to push in front of you. The ordering would fight the feature.

Recency belongs in the call log, not in the page order.

### `lastCalledAt` now means "last outbound action"

`logCall` bumps `lastCalledAt` and `callCount` for every action, and since the
copy-first change, `copy` is one of them. That is deliberate — copies have to
be logged or the record of who you reached rots — but it means the field is
really "last outbound action of any kind". The entry screen should label it
that way rather than "Last called".

---

## A wrinkle worth knowing, not worth fixing now

`markConfirmed(id, confirmed: false)` sets the tier back to `userEntered`.
If the contact came from map data it was `communityOsm`, and un-confirming it
loses that provenance. Both tiers render identically — amber dot, untrusted —
so there is no safety impact, and un-confirming is a rare action. Left alone
rather than adding a "previous tier" column for it. Revisit if POI-saved
contacts ever need to be told apart after the fact.

---

## Tests

Ordering and filters are the easy half. The half that matters:

- **Bulk import cannot produce a confirmed contact.** `insertBatch` must land
  every row as `userEntered` / `callConfirmed = false` regardless of what the
  companion asks for. This is the invariant the whole trust system rests on.
- **A failed batch leaves nothing behind.** One bad row rolls the whole
  transaction back rather than leaving half a spreadsheet in the database.
- **`watchUnconfirmedCount` drives the readiness banner**, so it has to
  change the instant a contact is confirmed.
- **Emergency contacts never appear in the main diary feed**, and trip
  contacts marked emergency appear only on the emergency screen.

---

## What this issue found

**The drafted `insertBatch` could import a confirmed contact.**

It passed the caller's companion straight through, adding only the batch id.
A screen that built a row with `callConfirmed: true` would have landed it
verified — and the readiness check, the amber dot and the blocking checklist
items all read that flag. The one rule the project calls non-negotiable was
being enforced nowhere.

It is now enforced in the DAO rather than at the import screen, because the
DAO is the single write surface and a future caller cannot forget. A test
proves the hole exists without the guard: with the override removed it reports
*"A arrived confirmed"*.

## Acceptance criteria

All verified. `flutter analyze` clean, 48 tests green.

- [x] The drafted DAO compiles, with one deliberate change to its body —
      the import invariant above — plus corrected comments.
- [x] Ordering: pinned first, then confirmed, then alphabetical.
- [x] A never-called number is not sunk below a called one.
- [x] Stop scoping includes trip-wide contacts.
- [x] Search matches name, raw number, normalised number, note and category.
- [x] Filters compose: trip + stop + category + search together.
- [x] `watchUnconfirmedCount` emits a new value when a contact is confirmed.
- [x] **Import lands every row unconfirmed, whatever the caller asks.**
- [x] **A batch that fails part way leaves the database untouched.**
- [x] `rollbackImport` removes the contacts and the batch row.
- [x] `logCall` writes a log row and bumps the count, including for `copy`.
- [x] Emergency contacts stay out of the diary feed and have their own stream.
- [x] `flutter analyze` clean, all tests green.
