# Issues #51 and #52 — the two detail screens

*SCREENS.md §10 and §11. Written before the code, as the others were.*

These are the last two screens the design set out. Both are assembly over
data that already exists — no new table, no new client, no new migration.
That makes them the cheapest issues left and also the easiest to get subtly
wrong, because assembly is where a number quietly starts meaning something
other than its label.

---

## #51 — Stop

### What it holds

| Section | Source | Note |
|---|---|---|
| `WEATHER` | `weatherSnapshots` for this stop | with the staleness stamp |
| `WHAT IS HERE` | contacts, checklist items, corridor places | counts with chevrons |
| `WHAT HAPPENS HERE` | `stops.activityTags` | **editable**, and it rebuilds the checklist |
| `CACHED HERE` | `legs.lastSyncedAt` | how you decide what to delete |

### The staleness correction

SCREENS.md §10 says the forecast turns `caution` **past three days**. Issue
#26 shipped seven. Seven was laxer than the app's own specification and, in
Meghalaya in October, wrong on the facts: a week-old forecast there is a
different season. The bands are now:

| Band | Age |
|---|---|
| fresh | under 1 day |
| ageing | 1 to 3 days |
| stale | 3 days and over |

The wording moved with them — "Over a week old" became "More than three days
old" — and the two test files that pinned the old numbers were updated. This
is a correction to shipped behaviour, not new work, so it is logged in
DECISIONS.md under its own date.

### Nearby places

A stop has no corridor of its own; a corridor belongs to a leg. So "places on
the roads either side" is the union over every leg that touches this stop,
arriving or leaving. The label says exactly that rather than "nearby", which
would imply a radius the app never computes.

### Tags rebuild the checklist

The chips are editable **because they drive the packing list**, and the screen
says so in a sentence underneath. That sentence is a promise, so `onTags`
actually calls `regeneratePackList` and then `syncReadinessChecklist`. Hand
edits survive, because #29 keyed generated items on `generatorKey` rather than
on the label.

### The diary chevron

The count says "diary entries **here**". Tapping it therefore opens the diary
already narrowed to this stop — `DiaryScreen` gained a `startStopScoped` flag
for it. Opening the whole-trip list would answer a different question from
the one the count asked.

---

## #52 — Leg

### Rome2Rio, reduced

Rome2Rio was one of the five apps this project set out to absorb and the one
that could not be: it needs live operator schedule databases. What survives is
what a person was told, typed in by them. The screen **says so** — "Typed by
you. There is no timetable to look up" — rather than presenting an empty form
that implies a lookup which will never happen.

### On the road

Corridor places, **ordered by distance along the route, not by distance from
you**. "Coming up in 12 km" is the useful sentence while moving; "0.2 km away"
treats the road as a plane, and a place 200 m off across a Khasi gorge is an
hour of driving.

Five inline, then a count and `SEE ALL` into the full discovery screen (#27).
Five is enough to be worth reading at a glance and few enough not to bury the
cache section.

Anything carrying a number from OpenStreetMap keeps the amber dot. This is the
trust invariant, and it does not soften because the row is small.

### Empty is two different things

`This leg has not been downloaded yet` (caution) and `Nothing tagged along
this road in OpenStreetMap` (muted) are not the same state, and collapsing
them would tell someone their road is empty when in fact nobody has looked.

### Tiles are not per-leg

`CACHED FOR THIS LEG` reports route and place count but **not** a tile size,
and says why: adjacent legs share the ground between them, so a per-leg tile
figure would double-count. The total lives in Settings, which is where the
delete button is anyway.

---

## Two streams, not one

`watchLegDiscovery` and `watchLegTransport` are separate because the halves
change for different reasons — one when a person edits a field, the other
when a download lands. Merging them would rebuild the corridor list on every
keystroke in the form.

## What this does not do

- No map on either screen. The trip map (#24) is its own surface.
- No per-stop tile figure, for the reason above.
- No editing of transport in place; the leg form (#18) already exists and is
  one tap away.

---

## Afterwards: the map key

Shipped in the same round, out of order, because it turned out to be blocking
the first user rather than a future one.

The MapTiler key had exactly one way in — `--dart-define` at build time, from a
gitignored file locally or a GitHub secret in CI. Neither is available to
somebody holding an APK that CI produced without the secret configured, which
is precisely where Yash was: he had a key, the app had nowhere to put it, and
the map screens correctly said maps were off while offering no way to turn them
on.

**Settings → Map key** now takes one. It is stored in `AppSettings` like every
other setting, it overrides the build-time key when both exist, and the map and
sync screens resolve the provider when they open rather than holding one in a
field — so a key saved in Settings works without restarting.

Two things this deliberately does not change:

- **The key still never enters the repository.** That was the actual
  constraint, and it holds. A test still asserts the test build carries no key.
- **The cache id still does not depend on the key.** A rotated key must not
  orphan a downloaded trip.

The key is not a password. It identifies the account to MapTiler and is visible
to whoever holds the phone, as it is in every mobile map SDK. The control that
matters is restricting it to the app's package name in the MapTiler dashboard,
and RUNNING.md says so twice.
