# Issue #56 — backup and restore

*Not in the original backlog. Added 13 Sep, after the signing bug made the
gap impossible to ignore.*

## Why this exists

This app's entire value is a directory typed by hand before a trip. Its only
copy lives on one handset. Until 13 Sep every new build was signed with a
different key, so installing one meant uninstalling the last — **which
deletes the database**. That was happening silently, build after build.

Stable signing fixes the recurring case. It does nothing for the general one:
a dropped phone in Meghalaya has exactly the same effect as an uninstall, and
there is no network to have synced anything to.

## What is in a backup, and what is deliberately not

The distinction is the one `clearTripCache` already draws: **what you typed
versus what came off the network.**

| Table | In | Why |
|---|---|---|
| Trips, Stops, Legs | ✓ | Typed. |
| Contacts | ✓ | The point of the app. |
| ImportBatches | ✓ | So an import can still be rolled back after a restore. |
| CallLogs | ✓ | Your record of what you called and when. |
| ChecklistItems | ✓ | Includes everything you edited by hand. |
| Travellers, Expenses, ExpenseSplits | ✓ | The shared ledger. |
| TimelineEntries, TrustedContacts | ✓ | Typed, when those features land. |
| AppSettings | partly | See the key, below. |
| Pois, PoiContacts | ✗ | Came off Overpass. Re-downloadable. |
| WeatherSnapshots | ✗ | Stale by definition. Re-downloadable. |
| MapTiles | ✗ | Tens of megabytes, and re-downloadable. |
| Legs' route cache | ✗ | Same reason as the POIs it came with. |
| **EmergencyHelplines** | **✗** | **See below. This one is not about size.** |

### Emergency numbers are never restored from a file

They are seeded by the app, from sources recorded in `sourceNote`, and the
whole trust model rests on that. **A file must never be able to put a number
on the emergency screen.** A backup that carried them would be a way to inject
one — by editing a JSON file, or by sending somebody a "backup" to restore.

They are re-seeded from the app's own data on first launch anyway, so nothing
is lost by excluding them. There is nothing to weigh here.

### The map key is not in the backup

`AppSettings` restores except for the MapTiler key. A backup is a file people
mail to themselves and copy between phones, and a key sitting in it travels
further than anybody intends. Re-entering it is one paste.

## The confirmation question

**A restore brings back `callConfirmed` and the trust tier exactly as they
were. This is a deliberate exception to the import invariant.**

Everywhere else in this app, bulk entry forces `userEntered` and unconfirmed —
`insertBatch` does it in the DAO so no caller can get it wrong. That rule
exists because *a number in a spreadsheet is an unverified number*.

A backup is not a spreadsheet. It is this app's own record of work you did:
you called that homestay, it answered, you marked it. Dropping that on restore
would force somebody to re-call twenty places after replacing a phone, which
would make backups useless and push people to not take them — a worse outcome
for safety than the risk it avoids.

But the file is editable JSON, and it could come from anyone. So the restore
screen **states what it is about to trust**, including how many numbers the
file claims were confirmed, and says plainly: restore only a backup this app
made, from your own phone.

No pretence of cryptographic protection. Any signature would need its key in
the APK, which protects nothing and would only make the claim look stronger
than it is.

## Restore replaces; it does not merge

One mode. Everything the backup covers is wiped and rewritten in one
transaction. Merging would mean duplicate detection across nine tables with
foreign keys between them, and a half-merged database is worse than either
outcome.

The confirmation says exactly what is on the phone now and what will replace
it. A restore that finds a newer trip already there is the case worth being
careful about, so the counts are shown side by side.

## Format

Plain JSON, one array per table, using Drift's own `toJson`. Readable by a
person, which matters: if a future version of this app cannot open the file,
somebody can still read their own phone numbers out of it in a text editor.
That is the real guarantee.

```json
{
  "format": "safarsathi.backup",
  "formatVersion": 1,
  "schemaVersion": 4,
  "createdAt": "2026-09-13T18:20:00.000Z",
  "tables": { "trips": [ … ], "contacts": [ … ] }
}
```

A file from a **newer** schema is refused, because columns this build has
never heard of cannot be restored faithfully and a silent partial restore is
the worst possible outcome. A file from an **older** one is accepted: every
migration in this project has been additive, so the missing columns take their
defaults.
