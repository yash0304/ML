# Issue #2 — Drift database wiring

**Size:** M · **Depends on:** #1 · **Blocks:** #4, #5, #16, #21, #26, #29, #31

The database is the only source of truth in this app. There is no backend and
no runtime network call, so if something is not in SQLite it does not exist.
This issue creates every table the eleven screens need and gets the generated
code compiling.

**The drafted schema files never made it into the repo.** `database_schema.dart`,
`contacts_dialer_schema.dart` and `contacts_schema_addendum.dart` are listed in
HANDOFF.md as drafted but were not among the uploads. The schema here is
reconstructed from PROJECT_RUNDOWN §5, DESIGN.md §2, SCREENS.md, and — for the
contacts tables — from the exact column names `contacts_dao.dart` already uses.
The DAO is the specification for those: if a name here disagrees with it, the
DAO wins, because #5 drops that file in unchanged.

---

## Tables

Fifteen, plus one the original model did not name.

| Table | Holds |
|---|---|
| `Trips` | The trip. Base currency lives here. |
| `Stops` | **Ordered occurrences, not unique places.** Shillong twice is two rows. `countryCode` lives here. |
| `Legs` | Connects consecutive stops. Transport, polyline, corridor width, sync time. |
| `Pois` | Cached places. Attaches to a Stop **or** a Leg, never both. |
| `PoiContacts` | Phone numbers from map tags. Always `communityOsm`. |
| `Contacts` | The diary. Both `phoneRaw` and `phoneE164`. |
| `ImportBatches` | One row per sheet import, so a batch rolls back whole. |
| `CallLogs` | Every call, copy, SMS and chat action. |
| `EmergencyHelplines` | Bundled reference data. Not trip-scoped. |
| `ChecklistItems` | Pack items and blocking readiness items, in one list. |
| `WeatherSnapshots` | Frozen forecast per stop per day, with its age. |
| `Travellers` | **New.** Named people on the trip. No accounts, no sync. |
| `Expenses` | The ledger. |
| `ExpenseSplits` | Who owes what on each expense. |
| `TimelineEntries` | Arrivals, notes, photos, GPS fixes. |
| `TrustedContacts` | Check-in recipients. |

`Travellers` is not in PROJECT_RUNDOWN §5 because the expense feature was
described before it was designed. `ExpenseSplits` has to point at a person,
and inventing a free-text name per split would make balances unjoinable.

---

## Four things this issue must get right

### 1. Foreign keys are off by default in SQLite

Every `ON DELETE CASCADE` in this schema is **inert** unless
`PRAGMA foreign_keys = ON` runs in `beforeOpen`. Without it, deleting a trip
leaves orphaned stops, contacts and expenses behind and nothing complains.

### 2. Money is stored in integer minor units

`amountMinor`, not a double. Floating point accumulates rounding error across
a three-way split, and a ledger that has to balance exactly cannot afford it.
Rupees are stored as paise, euros as cents.

### 3. A POI attaches to a Stop XOR a Leg

Both columns are nullable and exactly one is set. This is enforced by a table
`CHECK` constraint rather than by convention, because a POI with neither
answers no query and a POI with both answers two queries wrongly.

### 4. Seeding must be idempotent

`EmergencyHelplines` carries a unique key on country + number + service.
First-launch seeding will run again after a reinstall or a migration, and it
must not duplicate rows. The unique key is what makes #4 safe rather than
careful.

---

## Steps

1. `lib/core/database/tables.dart` — all sixteen table classes.
2. `lib/core/database/app_database.dart` — `@DriftDatabase`, `schemaVersion 1`,
   `MigrationStrategy` with `onCreate` and the foreign-key pragma.
   **No destructive fallback.** Never edit a shipped migration.
3. `dart run build_runner build --delete-conflicting-outputs`.
4. `flutter analyze` and `flutter test`.

The database opens through `drift_flutter`'s `driftDatabase(name:)` on device
and through an in-memory executor in tests, so tests never touch a file.

---

## Acceptance criteria

All verified in the build container. `flutter analyze` clean, 20 tests green.

- [x] `build_runner` generates `app_database.g.dart` and it compiles.
- [x] `flutter analyze` reports no issues.
- [x] A test opens an in-memory database and finds **all sixteen tables**.
- [x] A test proves `PRAGMA foreign_keys` is on, and that deleting a trip
      cascades to its stops and contacts.
- [x] A test proves a POI with neither `stopId` nor `legId` is rejected, and
      so is one with both.
- [x] A test inserts the same helpline twice and proves the second fails —
      this is what makes #4's seeding idempotent.
- [x] A test proves a three-way split of ₹3,200.11 sums exactly to the total.
- [x] A contact inserted with no explicit tier defaults to `userEntered` and
      `callConfirmed = false`. **Nothing reaches the database confirmed.**
- [ ] On device, the database file is created and the tables are present.
      Still owed — needs a phone.

## What this issue found

**Code generation was silently producing a schema with no foreign keys.**

At `drift_dev` 2.31 under `analyzer` 10, every `references(...)` call was
discarded. `build_runner` reported success and wrote 20 outputs. The only
signal was a warning reading *"This parameter should be a simple class name"*,
scrolling past in a wall of build output, and it was not obviously fatal.

The generated schema had **zero** `REFERENCES` clauses, no unique keys and no
table constraints. Everything else — column names, types, nullability,
defaults — generated correctly, which is what made it look fine.

Upgrading to `drift` / `drift_dev` 2.35, which pulls `analyzer` 13, generates
all 27 foreign keys. That cascade required bumping `drift_flutter` to 0.3.1
and `sqlite3_flutter_libs` to 0.6.0.

The lesson is in how the tests are written: **they assert against
`sqlite_master`, not against generated Dart.** Reading the generator's output
to check the generator is circular. Asking SQLite what it actually created is
not.

---

## Gotchas

- Drift maps `camelCase` to `snake_case`. The `CHECK` constraint is written in
  SQL and must use `stop_id`, not `stopId`.
- `contacts_dao.dart` is dropped in unchanged at #5. Do not rename a contacts
  column to something you prefer.
- Do not add a repository layer. DAO to widget, until a second consumer of the
  same data appears.
