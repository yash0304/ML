# Issue #50 — Emergency screen

**Size:** M · **Depends on:** #4, #7, #53

**The screen the whole trust system exists to protect.** Read
DESIGN_VISUAL_v2.md §0 before touching it.

---

## It is the exempt screen

No grain. No ruled paper. No thumb index. No stamps. No ticket edges. No
swipe actions. Plain rows, red badges, and provenance under every line.

Everywhere else in the app, ephemera carries category and place. Here it
carries nothing, because anything a user could mistake for a verification
mark would defeat the tiering.

## Here the tap calls

Copy-first is a planning workflow. In an emergency a copy-and-paste dance is
a liability, so:

- **Tapping the row places the call**, with `heavyImpact`. Calling 112 should
  feel heavier than calling your homestay.
- Copy is demoted to a secondary icon, for the case where you are reading a
  number out to somebody else.

## Two sections, never interleaved

**`OFFICIAL HELPLINES`** — the bundled Tier 1 numbers seeded at #4, filtered
to the countries this trip touches. Every one shows its government source in
the subtitle. The user can see where the number came from and judge it.

**`YOUR LOCAL CONTACTS`** — trip contacts the user marked emergency-relevant:
the homestay owner who can call a local ambulance faster than 108 can find
the village. These keep their trust dot.

A visible header separates them. A bundled government short code and a number
somebody typed must never sit in the same list.

## The absence is stated

State and union territory helplines are missing on purpose, and the screen
says so rather than leaving a silent gap. Guessed coverage is worse than
none — #37.

## Acceptance criteria

- [ ] Bundled helplines render with their badge, label and source.
- [ ] Trip emergency contacts render in a separate headed section.
- [ ] **Tapping a row calls**, and fires the heavy haptic.
- [ ] Copy is present but secondary, and is logged.
- [ ] No grain, no stamp, no ticket, no milestone anywhere on the screen.
- [ ] The state-list absence is explained on screen.
- [ ] `flutter analyze` clean, tests green, plus a golden.
