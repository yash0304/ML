# Issues #16–#20 — Trip structure

**Size:** M+S+S+S+M · **Depends on:** #2, #6, #9 · **Spec:** SCREENS.md §4, §10, §11

One document, because these five are one chain: you cannot generate legs
without stops, scope the diary without a real current stop, or check readiness
without knowing which stops are overnight.

## What this block changes about the app

Every screen so far reads real data off a trip **nobody can create or edit**.
This block replaces `DemoTrip` with the user's actual itinerary. After it, the
demo seed is a convenience rather than the only way in.

```
#16  trips list → trip editor → stop editor      (create, edit, reorder, delete)
#17  stops change  →  legs regenerate            (cached route data preserved)
#18  leg editor                                   (mode, times, booked, note)
#19  the diary's trip and current stop are real
#20  overnight stops generate blocking checklist items
```

---

## #16 — Trip and stop CRUD

**Files:** `features/trips/data/trip_editor.dart`,
`presentation/trip_list_screen.dart`, `trip_form_screen.dart`,
`stop_form_screen.dart`, `stop_list_screen.dart`

### The rule that shapes the data model

**A place may appear twice.** The Meghalaya itinerary is Shillong →
Cherrapunji → Shillong → Dawki, and the two Shillong rows are different
stops with different dates, different contacts and different nights. Nothing
may key a stop by its name.

This is why `sequenceOrder` exists and why the acceptance test builds the real
itinerary and asserts two Shillong rows coexist with distinct ids.

### Reordering

`sequenceOrder` is renumbered densely — 1, 2, 3 — on every change, inside a
transaction. A sparse or gapped ordering works until two stops end up sharing
a number, and then the itinerary silently scrambles.

### Deleting

Deleting a stop cascades to its POIs, checklist items and legs, and **sets its
contacts' `stopId` to null rather than deleting them**. That is already in the
schema (`KeyAction.setNull` on `Contacts.stopId`) and it is the right rule: a
number you have confirmed does not stop being a real number because you
dropped the stop from the plan.

The confirmation says exactly that, and names the count.

### Nights

`nights` is derived from arrival and departure when both are set, and typed
otherwise. **A stop with `nights > 0` is an overnight stop**, which is what
#20 keys the readiness check off.

## #17 — Leg auto-generation

**File:** `features/trips/data/leg_generator.dart`

A leg exists between every pair of consecutive stops. The whole of #17 is one
function: given the trip's stops in order and its existing legs, produce the
inserts, updates and deletes that make the legs match.

### The one rule worth the issue

**A leg whose from/to pair is unchanged keeps its row, and therefore its
cached route, distance and `lastSyncedAt`.** Regenerating by deleting every
leg and inserting fresh ones is four lines shorter and throws away the only
data in the app that needed a network connection to obtain. On a trip where
the user reorders the last two stops, every earlier leg must come through
untouched.

Matching is by the (fromStopId, toStopId) pair, not by sequence position.

## #18 — Transport leg details

**File:** `presentation/leg_form_screen.dart`

Mode, planned departure and arrival, booked flag, note. All typed.

There is no live schedule lookup and there never will be. The app is offline
at runtime; a Rome2Rio-style fetch is exactly the dependency this project
exists to avoid. What it can do is hold what the user already knows.

## #19 — Dialer trip scoping

The diary currently takes its trip and current stop from `DemoTrip`. After
#16 the app has a real active trip, so:

- `Trips.isActive` marks the one the app opens on. Exactly one may be active;
  activating a trip deactivates the rest, in a transaction.
- The **current stop** is derived from today's date against the stops'
  arrival and departure dates, falling back to the first stop when the trip
  has not started and the last when it is over.
- The diary's stop-scope toggle uses that, not a hardcoded id.

**Deriving the current stop from the date is a decision with a failure mode:**
a trip whose dates were never filled in has no current stop. The fallback has
to be a stop rather than null, or the toggle disappears on exactly the trips
most likely to be planned in a hurry.

## #20 — Pre-departure readiness

**File:** `features/trips/data/readiness.dart`

For every **overnight** stop, if it has an accommodation contact that is not
confirmed, generate a blocking checklist item: *"Call and confirm <stop>
accommodation number."*

- Items are `isGenerated: true`, `isBlocking: true`, and carry `contactId`.
- **Regeneration never discards a user's edit.** `isUserEdited` exists for
  this; an item the user has touched survives regeneration unchanged.
- Confirming the contact closes the item. The item is a view of the contact's
  state, not a second copy of it.
- A stop with no accommodation contact at all generates a different item:
  *"No accommodation number for <stop>."* Absence is the more dangerous case
  and it must not be silent.

The trip reads **not ready** while any blocking item is open.

---

## Acceptance

**#16** — [ ] Build the full Meghalaya itinerary; two Shillong rows coexist
with distinct ids. [ ] Reorder renumbers densely. [ ] Deleting a stop keeps
its contacts, unattached. [ ] Nights derive from dates when both are set.

**#17** — [ ] Legs match consecutive pairs after every change. [ ] A leg whose
pair is unchanged keeps its cached route and `lastSyncedAt`. [ ] Removing a
middle stop joins its neighbours.

**#18** — [ ] Mode, times, booked and note round-trip. [ ] Nothing on the
screen fetches anything.

**#19** — [ ] Exactly one trip is active. [ ] The current stop follows the
date. [ ] A trip with no dates still has a current stop.

**#20** — [ ] A trip with one unconfirmed homestay reads not-ready; confirm
it, reads ready. [ ] A stop with no accommodation number generates its own
item. [ ] A user-edited item survives regeneration.
