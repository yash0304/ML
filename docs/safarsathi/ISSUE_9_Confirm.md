# Issue #9 — The entry screen and the confirmation stamp

**Size:** M · **Depends on:** #7, #8 · **Blocks:** #20, #43, #50

Two parts. The screen SCREENS.md §2 describes, which does not exist yet, and
the one animation the whole app is allowed.

---

## Why this issue matters more than it looks

`userEntered → userVerified` is the single action the entire trust system
depends on. Everything else — the amber dot, the readiness count, the
blocking checklist items at #20, the pre-departure check — reads the boolean
this action sets.

Today that boolean can only be set by a test. There is no way for a user to
confirm a number at all, which means every number in the app is permanently
unverified and the readiness banner can never reach zero.

## 1. The entry screen

A read-only page for one contact. The diary's long-press goes here; Edit
opens the form built at #8.

- **The number is the headline** — 27pt Courier Prime, selectable, so a long
  press selects it natively as well as through the copy button.
- Provenance sits directly under it and changes with the tier:
  *"Typed by you · not confirmed"* with the amber dot, or *"Confirmed by you,
  11 Sep · you reached this number"* in signal green.
- **`COPY NUMBER`** full width, then Dialer / Call / Chat as equals.
- `RECORD`: attached stop, note, source, and last action — labelled that way
  rather than "last called", because a copy counts too.
- `CONFIRM`: the explanation, the button, and the stamp.

## 2. The stamp

`StampBadge` already exists in `core/widgets/retro.dart` and does the right
thing: it animates only on the `false → true` transition, fires the haptic at
the moment of contact rather than at animation start, and renders statically
on a rebuild so scrolling past a confirmed entry is silent and still.

The beat, from DESIGN_VISUAL_v2.md §5.3:

| Time | What happens |
|---|---|
| 0ms | The button locks. |
| 0–380 | The stamp rotates in from −8° to −3° on a spring, ink settling to 0.85. |
| 190 | Medium haptic, at contact. |
| 0–140 | The provenance line turns green. |
| 0–140 | The diary's readiness count rolls down, via its own stream. |
| 380+ | Zero frames scheduled. |

## The rule the button enforces

**"Only after you have actually called it."**

The screen says it plainly and the app cannot check it. That is the honest
position: the app cannot know whether a call connected, so it asks the user
to assert it and makes the assertion deliberate — a labelled button on a
detail screen, not a swipe or a long-press that could happen by accident.

Un-confirming is also possible, because a number that worked in October may
not work in November.

---

## Acceptance criteria

All verified. `flutter analyze` clean, 140 tests green.

- [x] Long-pressing a diary entry opens the entry screen.
- [x] The number is the headline, selectable, in the typewriter face.
- [x] Provenance reads correctly for all five tiers, with the amber dot only
      when untrusted.
- [x] Copy, dialer, call and chat all work from here and are logged.
- [x] A failed action says what went wrong rather than nothing.
- [x] Confirming lands the stamp, turns the provenance green, and promotes
      the tier.
- [x] The stamp animates on the transition and renders statically on a
      rebuild, so scrolling past a confirmed entry is silent and still.
- [x] Un-confirming is possible and restores the unconfirmed line.
- [x] Government tiers get no confirm section at all.
- [x] Goldens of both states.
- [ ] Looks and feels right on a phone — the haptic in particular, which no
      test can check.

## What this issue found

**Clearing a confirmation left the line still reading "Confirmed by you".**
The screen computed its tier from the stored value, which is still
`userVerified` until the write lands and the row is re-read. It now mirrors
`ContactsDao.markConfirmed` exactly: clearing drops to `userEntered`.

**The record rows were rendering centred**, because the column wrapping them
defaulted to centre alignment and each row shrank to its content, turning the
hairlines into short stubs instead of rules across the page. Caught by the
golden, invisible to every widget test.
