# Issue #4 — Emergency helpline seeding

**Size:** S · **Depends on:** #2 · **Blocks:** #50 (emergency screen), #38 (EU)

Seed the Tier 1 national helplines on first launch, idempotently.

This is the issue where getting it wrong is actually dangerous, so the rules
come before the steps.

---

## The rules

**1. No number ships without a government-domain source.**
Every row carries a `sourceNote` and the UI shows it under the number. The
user can see where it came from and judge for themselves. A row with an empty
`sourceNote` is a bug, pinned by a test.

**2. Do not seed the four flagged numbers.**
`1930` (cyber crime), `1078` (NDMA disaster), `1033` (highway accident) and
`104` (health) are widely cited on aggregator sites but **were not confirmed
from a `.gov.in` source**. `104` in particular is state-operated and not live
everywhere. They stay in the seed file marked `needsVerification: true`, and
the seeder skips anything carrying that flag. Issue #36 verifies them and
flips the flag; nothing else should.

**3. The state list stays empty.**
There is no authoritative combined dataset across 28 states and 8 union
territories. Populating it from aggregator sites or from memory would be worse
than shipping it empty. Issue #37 does it manually, per portal, with
provenance recorded per number.

**4. Nothing here is written from memory.**
The data below is transcribed from `PROJECT_RUNDOWN.md` §4.2, which is the
record of the research that produced it. If a number is not in that table, it
does not go in the seed file.

---

## Source data

Transcribed from PROJECT_RUNDOWN §4.2. Sources are that document's, verbatim.

| Number | Service | Source |
|---|---|---|
| 112 | Single national emergency (ERSS) | 112.gov.in, MHA |
| 100 | Police | legacy, active alongside 112 |
| 101 | Fire | legacy, active alongside 112 |
| 102 | Ambulance | legacy, active alongside 112 |
| 108 | Ambulance | legacy, active alongside 112 |
| 181 | Women Helpline | india.gov.in helpline directory |
| 14490 | NCW 24×7 Women Helpline | ncw.gov.in |
| 1091 | Anti-Obscene Calls Cell | india.gov.in |
| 1098 | Child Helpline | india.gov.in |
| 1363 | Tourist Helpline | india.gov.in |
| 139 | Railway security / medical | india.gov.in |
| 14567 | Senior citizens | india.gov.in |
| 14456 | Disabilities | india.gov.in |
| 14433 | NHRC | india.gov.in |

Fourteen rows seeded. Four more present but skipped.

---

## Idempotency

Seeding runs again after a reinstall or a migration and must not duplicate
rows. Two mechanisms, deliberately belt and braces:

1. `EmergencyHelplines` carries a unique key on
   `(countryCode, number, serviceType)` from #2. The database refuses a
   duplicate regardless of what the seeder does.
2. The seeder uses **upsert on that natural key**, not insert-or-ignore.

Upsert rather than ignore because these are bundled reference data, not user
rows. If a later version corrects a `sourceNote` or a label, the correction
should reach people who already installed the app. Insert-or-ignore would
leave them on the old text forever, with no migration path short of a schema
version bump. Running the seeder twice still produces exactly the same state,
which is what idempotent means.

**The seeder never touches `id`**, so anything referencing a helpline row
survives a re-seed.

---

## Steps

1. `lib/features/emergency/data/emergency_seed.dart` — the const data, with
   provenance per row and the four flagged entries present but marked.
2. `lib/core/database/seeding.dart` — `seedReferenceData(db)`, upserting on
   the natural key and skipping flagged rows.
3. Call it from `main()` before `runApp`, awaited, so the first frame never
   renders an empty emergency screen.
4. Tests.

---

## Acceptance criteria

All verified. `flutter analyze` clean, 29 tests green.

- [x] Seeding twice leaves the row count unchanged, and ids are undisturbed.
- [x] Exactly fourteen rows are seeded.
- [x] **None of 1930, 1078, 1033 or 104 is present**, and all four are still
      on record in the seed file with their flag.
- [x] Every seeded row has a non-empty `sourceNote`.
- [x] No seeded row carries `needsVerification: true`.
- [x] The state-level list is empty — no row has a `regionCode`.
- [x] A row corrupted to stale text is repaired by re-seeding, in place,
      without a second row appearing.
- [x] No two seeded entries collide on the natural key.
- [x] `flutter analyze` clean, all tests green.

---

## Open question for #36

**102 and 108 are both ambulance lines and our source does not distinguish
them.** In practice 102 and 108 are run differently in different states. Both
are seeded with the label "Ambulance" because inventing a distinction would be
exactly the kind of confident-sounding guess this whole feature exists to
avoid. Resolve it when the flagged numbers are verified, from state portals,
not from memory.
