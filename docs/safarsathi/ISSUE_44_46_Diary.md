# Issues #44, #45, #46 — the three interactions the diary was missing

*DESIGN_VISUAL_v2.md §0 first. DIALER_RETRO_PATCH.md edits 7, 8 and 11.*

Most of the retro pass landed as the screens were written: grain, stencil,
`PressScale`, the hazard stripe and the rolling count on the readiness banner,
the amber dot, the category thumb index. Three things never did, and all three
are interactions rather than paint, which is why they slipped — a static
golden cannot notice a missing gesture.

| Issue | What was missing |
|---|---|
| #44 | The confirmation stamp never landed on a row. `StampBadge` existed and was used only by the empty state. |
| #45 | No swipe actions at all. |
| #46 | No over-scroll cache stamp. |

---

## #44 — the stamp lands once

`StampBadge` animates in `didUpdateWidget`, on the `false → true` transition
only, and fires `Haptics.confirm()` at the halfway point rather than at the
start — the haptic is meant to coincide with contact, not with the wind-up. A
badge built fresh with `landed: true` starts at controller value 1: still,
silent, already stamped.

That is the whole design, and it has **one failure mode**, which is element
reuse. `ListView.builder` recycles elements. Without keys, scrolling can hand
the element that was showing a confirmed contact to an unconfirmed one and
back, and `didUpdateWidget` would read that as a confirmation: a stamp
animating and a haptic firing for a row nobody touched, while scrolling.

So every row gets `ValueKey(contact.id)`. The backlog's acceptance line —
"scroll past it afterwards and nothing animates or buzzes" — is exactly this
bug, and it is pinned by a test that scrolls a list of confirmed contacts and
counts the haptic channel's messages.

**The amber dot cross-fades rather than vanishing.** Both markers are about
the same fact, and swapping one for the other in a single frame reads as a
glitch; 140ms of cross-fade reads as one thing becoming another.

## #45 — swipe, springing back

`Dismissible` with `confirmDismiss` returning `false` on both sides. The swipe
performs its action and the row springs back; nothing is ever removed by a
gesture. Removing a phone number by accident, offline, is unrecoverable in the
way that matters — you are on a road and it is gone.

- Right: **call**, on `c.signal`.
- Left: **pin / unpin**, on `c.stone`.
- `Haptics.light()` once as the threshold is crossed, not on every pixel.

**The whole-row tap still copies the number.** DECISIONS.md 2026-09-11
requires the commonest action to need no aiming. Swipe adds reach; it does not
take the easy target away.

## #46 — over-scroll shows what is cached

Pull-to-refresh would be a lie in an app that makes no network call on the
road: the spinner would promise something the app is built never to do. The
same gesture reveals **when this trip was last downloaded** instead, which is
the honest answer to what a person pulling down actually wants to know.

Revealed by over-scroll, faded in proportion to it, gone when released. It
needs no state of its own beyond the scroll offset.

---

## Also ticked, having been built along the way

Verified rather than written this round: **#41** (`motion.dart` — durations,
the spring, `Motion.d()` reduce-motion collapse, `Haptics`, `PressScale`),
**#42** (all seven `retro.dart` primitives), **#43** (the ten other dialer
patch edits) and **#49** (the category thumb index). Each now has a test or a
golden that would fail if it regressed, which is what "done" has meant in this
project.

**#48**, the exemption test, is the one that matters most and it was already
green: the emergency subtree contains no `StampBadge`, `TicketCard`,
`MilestoneMarker`, `HazardStripe` or `GrainOverlay`, and the trust dot renders
at 7px in `cautionMark`. This round it gains the case it was missing — that
adding a stamp to the diary did not leak one into the emergency tab.

## What this does not do

- No dismissal. Ever. See above.
- No pull-to-refresh, for the reason above.
- Nothing new on the emergency tab. That is the point of #48.
