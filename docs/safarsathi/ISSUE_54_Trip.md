# Issue #54 — Trip screen

**Size:** M · **Depends on:** #53 · **Spec:** SCREENS.md §4, DESIGN_VISUAL_v2.md §2–§4

## Why this issue exists

After #53 the app had five tabs and one of them had content. The trip is the
thing every other screen hangs off — the diary is scoped to it, the money is
split between its travellers, the emergency screen reads its state — and until
you can see it, none of the rest has a frame to sit in.

This issue is **read-only on purpose**. Creating and editing a trip is #16.
What #54 delivers is the answer to "where am I in this trip, and what is the
next leg", which is the question the app exists to answer at a dhaba at 3pm.

## What it must show

1. **The next leg as a milestone marker.** The Indian kilometre stone is the
   one piece of road furniture every driver reads without thinking. Origin on
   the cap, destination and distance on the body.
2. **A readiness banner** when something is unconfirmed, reusing the diary's
   widget rather than growing a second one.
3. **One ticket card per stop**, in arrival order, with the current stop marked.

## Build steps

1. `lib/features/trips/data/trip_summary.dart` — a `watchTripSummary(db,
   tripId, currentStopId:)` returning a stream of a plain summary record.
   The screen takes the **stream**, never the DAO (the constraint established
   at #6: a widget test cannot close a Drift database without hanging).
2. `lib/features/trips/presentation/trip_screen.dart` — layout only. No
   queries in the widget tree.
3. Wire it into `AppShell` destination 2.

## The one rule that is easy to get wrong

**The milestone cap is muted when the leg has never been synced.** A distance
painted in full ink is a claim that the number came from somewhere. A leg with
`lastSyncedAt == null` has a distance the user typed or the app guessed, and
the screen must not dress that up as surveyed. This is the trust tiering
invariant applied to geometry instead of phone numbers — same rule, different
table.

## Acceptance criteria

- [x] `flutter analyze` clean.
- [x] The screen renders from a stream; no DAO reaches the widget layer.
- [x] Cap is muted for an unsynced leg, full ink for a synced one.
- [x] Golden `trip.png` committed and re-rendered from bundled fonts.
- [x] Stops appear in arrival order with the current one marked.
- [x] Nothing on this screen writes to the database.
