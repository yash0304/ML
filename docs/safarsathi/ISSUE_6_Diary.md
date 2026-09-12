# Issue #6 — Diary screen

**Size:** M · **Depends on:** #3, #5 · **Blocks:** #7, #8, #9, #43, #49

The first screen worth looking at. Built from SCREENS.md §1 and the prototype.

**The drafted `code/dialer_screen.dart` and `DIALER_RETRO_PATCH.md` are both
stale.** They predate the copy-first decision and the diary metaphor — the
drafted screen dials on tap, has tabs instead of a thumb index, and uses the
v1 token API. Read them for the trust-tier rendering and nothing else.

---

## Scope

**In:** the screen renders real rows from `ContactsDao.watchContacts`, with
the ruled entries, numbered margin, category thumb index, readiness banner,
search, empty state and page footer.

**Out:** what a tap actually does. Copy, the toast, `OPEN DIALER`, call and
chat are #7. The screen exposes an `onCopy` callback and #7 fills it in.
Tapping an entry does nothing visible until then, which is expected.

Also out: the bottom navigation. It belongs to a shell above this screen and
there is only one screen to navigate between, so it arrives when a second one
does.

---

## Structure

```
app bar        trip kicker · "Diary" · scope line · stop-scope toggle
search         name, number or note
banner         only when something is unconfirmed
─────────────────────────────────────────────
margin  entries                    thumb index
 01     name + trust dot                 ALL
        NUMBER, larger                   STAY
        meta line                        MOVE
 02     …                                FOOD
─────────────────────────────────────────────
page footer    PAGE 1 OF 1 · 6 ENTRIES
```

- **Margin** — 30px, entry number in stencil, right border a doubled hairline
  the way a notebook margin is ruled.
- **The number is set larger than the meta line** and in tabular figures. In
  a diary the number is the content.
- **Thumb index** — vertical stencil labels, scrollable, selected tab filled
  `signal`. #49 narrows it to the categories the trip actually uses.
- **Trust dot** — `cautionMark`, 7px, beside the name. Unchanged from
  DESIGN.md §4 and not negotiable.

## Dev seeding

`DiaryScreen` needs a trip, and trip CRUD is #16. A debug-only seeder creates
one demo trip with a handful of contacts **when the database has no trip at
all**, so the app renders something on a device.

Its numbers are deliberately, obviously fake — `+91 90000 000xx` — so nobody
can mistake placeholder data for a real number. It is deleted at #16.

## The smoke screen goes

`lib/features/dev/smoke_screen.dart` was written at #1 to prove the tokens and
fonts work on a device. The diary replaces it as `home`.

---

## What this issue cost, and what it taught

Most of the effort went into widget tests, not the screen. Three things bit,
and all three are now written down so they bite once:

1. **A widget test cannot close a Drift database.** `close()` awaits work the
   fake clock never advances, so the test hangs until the runner is killed —
   no error, no timeout, just silence. This is what made the first attempts
   look like the screen was broken.
2. **Cancelling a Drift query stream schedules zero-duration cleanup timers**
   during disposal, and the framework's pending-timer check runs *after* user
   teardowns, so no amount of pumping in `addTearDown` clears them.
3. **`pumpAndSettle` can never settle** on a screen whose loading state is a
   `CircularProgressIndicator`, because it animates forever.

The fix for all three was one change that improves the design: **the screen
takes streams rather than the DAO.** `DiaryScreen` receives a
`watchContacts` function and an `unconfirmedCount` stream; `ReadinessBanner`
receives a stream. Widget tests drive them with in-memory fakes and no
database at all, and the DAO's real behaviour stays covered against real
SQLite in its own 19 tests.

A separate lesson worth not repeating: `pkill -f "flutter test"` matches the
shell running the command and kills it, which looks exactly like a crash.

## Acceptance criteria

All verified. `flutter analyze` clean, 74 tests green in about three seconds.

- [x] Entries render, ordered pinned → confirmed → alphabetical.
- [x] **The amber dot appears on `userEntered` and `communityOsm` rows and
      not on `userVerified` or `verifiedNational` ones, in both themes.**
- [x] The margin numbers run 01, 02, 03 down the page.
- [x] The number is set larger than the meta line, in tabular figures.
- [x] Search filters live as you type, and clearing restores the page.
- [x] Tapping a thumb index tab filters by category; ALL restores.
- [x] The readiness banner appears only when something is unconfirmed, and
      reads singular for one.
- [x] The empty state differs for "no entries yet" and "nothing matches".
- [x] The stop-scope toggle appears only when there is a current stop, and
      keeps trip-wide contacts when on.
- [x] Renders without overflow at 400×800, in both themes.
- [x] `flutter analyze` clean, all tests green.
- [ ] Looks right on a real phone. Still owed, along with the fonts.
