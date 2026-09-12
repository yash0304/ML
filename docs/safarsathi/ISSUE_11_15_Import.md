# Issues #11–#15 — Bulk contact import

**Size:** M+M+M+M+S · **Depends on:** #2, #5, #8 · **Spec:** PROJECT_RUNDOWN §4.5, SCREENS.md §6

One document for all five, because they are one flow and splitting the guide
would mean five copies of the same context. Each issue has its own acceptance
block at the bottom.

## The shape of the flow

```
More tab
  └─ Import contacts
       #11  pick a file  →  parse  →  (pick a sheet, if the workbook has several)
       #12  map columns  →  auto-matched, user can correct
       #13  validate     →  preview, three states, per-row deselect
       #14  commit       →  one transaction, one ImportBatch
       #14  history      →  roll the whole batch back
```

Everything before "commit" is in memory. Nothing touches the database until
the user presses the commit button, and then it all lands or none of it does.

## The invariant this whole feature is built around

**Import cannot produce a confirmed contact.** Not by mapping a column called
`verified`, not by a future caller passing the flag, not by any path. A number
in a spreadsheet is an unverified number. This is enforced in
`ContactsDao.insertBatch`, which is the single write surface, and pinned by a
test that fails if the guard is removed.

The import screens must also **say** this, in words, on the preview. Someone
importing 60 numbers should not have to infer the trust model from a dot.

---

## #11 — Sheet parser

**File:** `lib/features/import/data/sheet_parser.dart`

Parse **bytes**, not paths. The parser then has no dependency on
`file_picker`, no platform channel, and no filesystem, which means the whole
of #11 is testable in a plain Dart test with a string literal.

Types:

- `SheetTable { name, headers, rows, sourceRowNumbers }`
- `ParsedWorkbook { fileName, sheets }`

**`sourceRowNumbers` is the point of this issue.** Every downstream error
message says "row 14", and row 14 must mean what the user sees in Excel — the
1-based line including the header, before blank rows were dropped. Losing that
index turns every error message into a scavenger hunt.

Rules:

- Rows are padded or truncated to the header width, so a short row is missing
  cells rather than throwing.
- Entirely blank rows are dropped, and dropping them must not renumber
  anything after them.
- A workbook with no non-empty sheet is an error with a sentence, not an
  empty list that silently imports nothing.
- Numbers arriving as doubles from XLSX render as `9876543210`, never
  `9.87654321E9`. Excel stores a phone number typed without a leading `+` as
  a float, and this is the single most likely way a real file breaks.

## #12 — Column mapping

**File:** `lib/features/import/data/column_mapping.dart`, screen
`presentation/column_mapping_screen.dart`

`ImportField`: name, phone, category, note, stopName, isEmergency, whatsapp.
Name and phone are required; the rest are optional.

`autoMatchColumns(headers)` compares on a squashed key — lowercased, with
everything that is not a letter or digit removed — so `Phone Number`,
`phone_number` and `PHONE-NUMBER` all match the same alias. Each field carries
a list of aliases including the Hinglish ones a real sheet actually uses
(`mobile`, `contact`, `no`, `number`).

A column is never claimed twice: first field to match an unclaimed column wins,
in declaration order.

## #13 — Validation and preview

**Files:** `data/import_validation.dart`, `presentation/import_preview_screen.dart`

`RowState`: `ready`, `warning`, `skip`.

- `skip` — no name, or no number. There is nothing to save.
- `warning` — saves fine, but the user should look: the number did not
  normalise, or it is already in the diary, or it repeats earlier in the same
  file.
- `ready` — nothing to say.

Rules:

- A duplicate is detected against the existing diary **and against earlier
  rows of the same file**. A sheet that lists the same homestay on two stops
  is the common case, not an edge case.
- A row that fails to normalise still imports, carrying the raw text. The
  diary shows an amber dot on it anyway, so nothing is lost and the user keeps
  the only record they have of that number.
- Skipped rows are deselected by default and cannot be selected — there is
  genuinely nothing to write.
- Warning rows are **selected** by default. A warning is information, not a
  veto.
- The preview states colour and **stripe**, so severity reads without hue.
  Red is not used here; red belongs to emergency.

## #14 — Commit and rollback

**Files:** `data/import_commit.dart`, `presentation/import_history_screen.dart`

Commit calls the existing `ContactsDao.insertBatch`, which already wraps the
`ImportBatch` insert and the contact inserts in one transaction and already
forces the tier down. #14 adds:

- `rowsImported` / `rowsSkipped` recorded on the batch, so history is honest
  about what a file actually did.
- `watchImportBatches(tripId)` on the DAO.
- A history screen listing every batch with its file name, sheet, counts and
  date, and a rollback that deletes the batch's contacts and the batch row in
  one transaction.

**Rollback deletes contacts the user may have since confirmed.** So it asks
first, and the confirmation names the count.

## #15 — Stop-name fuzzy matching

**File:** `data/stop_matcher.dart`

Exact match on the squashed key first. Then Levenshtein distance with a
threshold that scales with the length of the name, so `Cherrapunji` matches
`Cherrapunjee` but `Shillong` does not match `Silchar`.

**An unmatched stop name is not an error.** The row imports as a trip-level
contact and the preview says which stop it landed on, or that it landed on
none. Refusing to import a good number because a place name was spelled
differently would be the feature working against its own purpose.

---

## Acceptance

**#11** — [x] CSV and XLSX both parse to the same shape. [x] Row numbers
survive blank-row removal. [x] A float phone number renders as digits.
[x] Short rows pad rather than throw. [x] An empty workbook errors with a
sentence.

**#12** — [x] Aliases match across case, spacing and punctuation. [x] No
column claimed twice. [x] The user can remap and clear any field. [x] Name and
phone are enforced before the preview opens.

**#13** — [x] A file with a blank name, a bad number and a duplicate flags
each correctly and imports the rest. [x] In-file duplicates detected.
[x] Skipped rows cannot be selected. [x] Warning rows are selected by default.

**#14** — [x] Import 7 rows, all land unconfirmed. [x] Counts recorded.
[x] Rollback removes exactly that batch. [x] **A confirmed row cannot be
imported** — pinned by a test that fails without the guard.

**#15** — [x] Exact and near matches resolve. [x] A wrong-but-similar name
does not match. [x] An unmatched name still imports, trip-level.
