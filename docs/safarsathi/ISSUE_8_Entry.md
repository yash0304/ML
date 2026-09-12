# Issue #8 — Add / edit entry

**Size:** M · **Depends on:** #6 · **Blocks:** #9, #10, #13

A ruled diary page as a form: label in the margin voice, value on the line,
a hairline under each. SCREENS.md §5.

One screen serves both new and edit. The diary's FAB opens it empty; a
long-press on an entry opens it filled.

---

## Three rules about what the form refuses to do

### It will not save a nameless or numberless entry

Those are the only two hard requirements. Everything else is optional.

### It will not block on a bad number

An unreadable or incomplete number is a **warning**, not an error. The user
may be halfway through typing, holding a number with an extension, or copying
something odd off a signboard. The amber dot already says the number is
unverified; refusing the save would lose the only record they have of it.

When the parse fails, `phoneE164` stays null and the raw value is stored
alone — no confident-looking wrong value in the normalised column.

### It will not block on a duplicate

It says which entry already holds the number and lets you save anyway. Two
entries for one number is a legitimate thing to want: the homestay's landline
and the owner's mobile can be the same line on different days.

Editing an entry is not a duplicate of itself.

---

## The one rule about what it insists on

**Changing the digits of a confirmed entry clears the confirmation.**

A confirmation means *"I called THIS number and it worked."* Change the
digits and that is no longer true, so the confirmation goes with them: back
to `userEntered`, `callConfirmed = false`, `confirmedAt` cleared, and the
entry reappears in the readiness count.

The form says so before you save, and the tier explainer at the bottom
switches from "Stays confirmed" to a warning. Silently keeping a green tick
against a number nobody has dialled is exactly the failure the trust system
exists to prevent.

Changing only the name, note, category or stop keeps the confirmation.

---

## Normalisation

`phone_numbers_parser`, pure Dart, no platform channel, works with no signal.

The country to parse against comes from the caller rather than being fixed to
India, because a European trip crosses borders mid-itinerary. Today the app
passes `IsoCode.IN`; wiring it to the current stop's `countryCode` is a
one-line change once trip structure exists at #16.

---

## Acceptance criteria

All verified. `flutter analyze` clean, 118 tests green.

- [x] A nameless or numberless entry is refused, with a plain reason.
- [x] Both the raw and the normalised number are stored.
- [x] An unreadable number warns and still saves, with `phoneE164` null.
- [x] A duplicate warns and still saves, naming the entry that holds it.
- [x] Editing an entry is not flagged as a duplicate of itself.
- [x] **A new entry always lands unconfirmed**, whatever else is true.
- [x] Editing the name keeps a confirmation; editing the digits clears it,
      and the form says so first.
- [x] An empty note is stored as nothing, not as blank text.
- [x] A stop can be chosen; whole-trip is the default.
- [x] The category order matches the diary's thumb index.
- [ ] Looks right on a phone. Owed with everything else.

## Note for #9

There is still no read-only entry screen — SCREENS.md §2. Long-press goes
straight to the form, which is enough to edit but is not where the confirm
stamp belongs. Build §2 at #9 and move the long-press there, with Edit as an
action on it.
