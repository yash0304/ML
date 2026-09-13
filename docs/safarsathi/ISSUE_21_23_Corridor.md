# Issues #21, #22, #23 — The corridor

**Size:** M + M + M · **Depends on:** #2, #17 · **Spec:** PROJECT_RUNDOWN §3, §6

The first three of the eight offline-map issues. These are the **data layer**:
geometry, a route, and a set of places along it. No map is drawn here — that
is #24, and it is blocked on a decision only Yash can make.

---

## The thing that has to be settled before any of this: the network

Every earlier decision in this project said the app makes **no network call**.
The release manifest declares no permissions at all, and DECISIONS.md logs
that as the clearest possible proof of the offline claim: the release build
cannot reach the network even if a future line of code tried.

Offline maps make that literally impossible. Routes, places and tiles have to
come from somewhere, once, before you leave.

**So the claim changes, and it has to change honestly.** The app is offline
**at runtime**, not offline absolutely. It reaches the network only while you
are sitting on WiFi setting a trip up, only when you press a button that says
so, and never once you are moving. The manifest gets `INTERNET` back, and
every sync surface says in words what it is about to fetch and from where.

Anything weaker than that — a background refresh, a silent retry on the road,
a "just checking for updates" — breaks the promise the whole app is built on.

## #21 — Overpass client

**Files:** `features/discovery/data/overpass_client.dart`, `poi_category.dart`

Queries OpenStreetMap's Overpass API by bounding box and category, and parses
the result into POI drafts.

### The rule that matters

**A phone number from OSM is `communityOsm` and can never be anything else.**
It is a number a stranger typed into a public wiki. It may be a decade old. It
gets the amber dot, it is never pre-confirmed, and saving one into the diary
lands it as `userEntered` — the same tier as something typed by hand.

This is the trust invariant reaching a new source, and the new source is the
least trustworthy one the app has.

### Politeness

Overpass is free infrastructure paid for by volunteers. The client:

- sends a real `User-Agent` naming the app,
- refuses a bounding box beyond a sane area, because a query that big is a
  bug and it costs someone else money,
- runs **once per leg at setup**, never on a timer, never on the road.

## #22 — Route polyline

**Files:** `features/discovery/data/osrm_client.dart`, `polyline.dart`

One OSRM call per leg at setup. Stores the encoded polyline and the distance.

`polyline.dart` is the Google encoded-polyline codec, written out. It is forty
lines of bit-shifting that has to be exactly right, and pulling a package in
for it would mean a dependency on something this app can verify itself.

**OSRM's default precision is 5; its v5 API returns 6 when asked.** Getting
that wrong scales the entire route by ten and puts Shillong in the Bay of
Bengal, so the precision is an explicit argument and both are tested.

## #23 — Corridor

**File:** `features/discovery/data/corridor.dart`

Given a decoded polyline and a buffer in kilometres:

- the **bounding box** to query, expanded by the buffer,
- each POI's **distance off the route**, as the shortest distance to any
  segment,
- each POI's **distance along the route**, which is what powers "coming up in
  12 km".

### Two pieces of geometry that are easy to get wrong

**Longitude degrees shrink with latitude.** Expanding a bbox by 3 km means
adding 3/111 degrees of latitude but 3/(111·cos φ) degrees of longitude. At
Shillong's 25.6° that is a 10% difference; ignore it and the corridor is
narrower than advertised on the east–west axis, and the places you were
promised are missing with no error.

**Distance to a route is distance to a SEGMENT, not to a vertex.** A road
running dead straight for 40 km has two vertices. Measuring to the nearer
vertex would report a dhaba halfway along as 20 km off the route when it is
on it.

---

## Acceptance

**#21** — [ ] A query is built for a bbox and a category set. [ ] A response
parses to POIs with names and coordinates. [ ] `phone` and `contact:phone` are
extracted. [ ] **Every extracted number is `communityOsm`.** [ ] An oversized
bbox is refused. [ ] A malformed response fails with a sentence, not a crash.

**#22** — [ ] Polylines round-trip at precision 5 and 6. [ ] A known OSRM
string decodes to known coordinates. [ ] A route response yields a polyline
and a distance. [ ] An empty route is handled.

**#23** — [ ] A bbox expands correctly in both axes, accounting for latitude.
[ ] Distance off route measures to the segment, not the vertex. [ ] Distance
along route is cumulative and monotonic. [ ] A point beyond the corridor is
excluded.
