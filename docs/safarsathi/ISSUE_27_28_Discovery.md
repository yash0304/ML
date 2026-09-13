# Issues #27, #28 — What is along the way

**Size:** M + S · **Depends on:** #24, #25 · **Spec:** SCREENS.md §11

The corridor has been downloaded since #21–#23 and no screen has ever shown it
to a person. This is the last gap between what the app knows and what it tells
you.

---

## The ordering rule

**Distance along the route, not distance from you.**

"Coming up in 12 km" is the useful sentence while moving. "0.2 km away" is a
map's answer to a question nobody asked in a car: it treats the road as a
plane, and a place 200 m away across a gorge is an hour of driving.

Every place therefore renders on its own **milestone marker**, carrying the
kilometres along the leg. That is the same idiom the Trip screen uses for the
next leg, and it is the right one: a milestone is a thing you pass.

## The trust rule, again, at its weakest source

**A number from OpenStreetMap keeps the amber dot, always.** `CorridorSync`
already forces `communityOsm` at the write, and the detail screen has to say
what that means in words: *from open map data, nobody has checked it.*

### Where this deviates from the backlog, on purpose

The backlog says "save as contact (lands as `userEntered`)". **It lands as
`communityOsm` instead.**

Both tiers carry the amber dot, so the trust outcome is identical. What
differs is provenance, and the entry screen reads it out loud:

| Tier | What the entry screen says |
|---|---|
| `userEntered` | Typed by you · not confirmed |
| `communityOsm` | From open map data · nobody has checked it |

Saving an OSM number as `userEntered` would make the app tell you that you
typed a number a stranger put in a public wiki. In an app whose entire premise
is knowing where a number came from, that is the one thing it must not do.

The number becomes `userVerified` the moment you call it and say so, exactly
as any other number does. That path is unchanged.

## Deep-linking out

For reviews and photos, hand off to Google Maps rather than trying to hold
them. That costs nothing, needs no key, and is honest about what an offline
app can carry.

**The link is offered, never followed automatically**, and the screen says it
needs signal — a dead tap on a mountain road with no explanation is worse than
no button.

---

## Acceptance

**#27** — [ ] Places listed in route order with kilometres along the leg.
[ ] Category chips filter. [ ] A leg with nothing downloaded says so rather
than looking empty. [ ] A place carrying an OSM number shows the amber dot in
the list.

**#28** — [ ] Distance off route is shown as well as along it. [ ] The OSM
number renders with its unverified marker and its provenance in words.
[ ] Copy works and is logged. [ ] Saving lands the contact as `communityOsm`,
never confirmed. [ ] The Google Maps link says it needs signal.
