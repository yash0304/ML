# Issue #10 — the multi-add screen

*The last unbuilt feature before October. Depends on #8.*

## What it is for

Sitting at home on WiFi with a booking sheet or a WhatsApp thread open, typing
in eight numbers: homestay, driver, guide, the hospital in Shillong, the one in
Sohra, a fuel stop, two friends. Today that is eight round trips through the
full entry form — open, fill six fields, save, back, open again.

Multi-add is the same eight numbers as one screen of thin rows.

**It is not a replacement for the entry form.** The form owns everything a
single contact might need: the stop it belongs to, a note, WhatsApp, the
confirmation. Multi-add owns exactly two things — a name and a number — plus a
category, because sorting eleven entries afterwards is worse than tapping a
chip while you type. Anything else is edited later, on the entry that now
exists.

## The invariant, restated

Rows land through `ContactsDao.insertBatch`, the same function file import
uses, and for the same reason: it **forces** `userEntered` and
`callConfirmed = false` on every row whatever the caller passes. There is no
argument this screen could get wrong.

That matters more here than for a file import. Typing eight numbers quickly
feels like work completed, and it is exactly the moment somebody might assume
they are done. The screen says so once, plainly, above the button.

A second path that re-implemented the guard is how the invariant erodes; that
bug has already been found once in this project.

## Recorded as a batch, and therefore undoable

Reusing `insertBatch` means every multi-add session is an `ImportBatch`, so it
appears in the import history and can be rolled back wholesale — which is worth
having, because the most likely mistake is typing ten rows against the wrong
trip.

The history screen currently says "imported" and "Undo import". For a typed
batch it says **"added"** and **"Undo these entries"**. The batch is marked by
its `fileName` being `typedBatchLabel`, exposed as `ImportBatchSummary.wasTyped`
rather than compared as a string at each call site.

## The interactions

**The sheet grows itself.** Typing into the last row appends a blank one. There
is no "add row" button to find, and no arbitrary cap — the backlog says five to
ten, but nothing enforces it.

**Category inherits downward.** A new row takes the category of the row above.
Most bulk sessions are one kind at a time — "here are my three hotels" — so
setting it once and having it stick is the fast path, and overriding one row
costs a tap.

**Incomplete rows are skipped, not errors.** A row with a name and no number,
or the trailing blank, simply does not save. The button counts what will:
"Save 6 entries". Rows that will be skipped say so quietly.

**A bad number is a warning, never a block** — as everywhere else in this app.
`PhoneNormaliser` warns; the row still saves. Refusing would lose the only
record somebody has of a number they read off a signboard.

**Duplicates warn twice over.** Against the diary, and against the sheet
itself — typing the same driver twice off a booking confirmation is easy, and
the second one is the row you are looking at.

**Leaving with typed rows asks first.** Eight numbers typed and lost to a back
gesture is the same harm the swipe actions were designed against at #45.

## What this does not do

- No stop, note, or WhatsApp per row. That is the entry form's job.
- No paste-a-block parsing. That is what the CSV import is for, and guessing
  at pasted text is how a phone number becomes a note.
- No per-row confirmation. Nothing typed is ever confirmed.
