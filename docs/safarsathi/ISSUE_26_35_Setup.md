# Stop coordinates, #26 weather, #35 settings

**Depends on:** #16, #21–#23 · **Blocked by:** nothing

Everything here works while #24 waits on a tile-provider decision. None of it
needs a map.

---

## Stop coordinates — the gap that blocks the corridor

`CorridorSync` needs `Stops.lat` and `Stops.lon`, and **nothing in the app
sets them.** The stop form has no map picker and will not have one until #24,
so the corridor code is currently unreachable on a real trip.

Two ways in, both at setup:

1. **Look the place up.** Nominatim, OSM's own geocoder, over the stop's name.
2. **Type it in.** A plain latitude and longitude field, for when the lookup
   is wrong or the place has no name a geocoder knows.

### Nominatim's usage policy is strict and this app will obey it

- **A real `User-Agent` identifying the app.** An anonymous client is blocked.
- **At most one request per second**, enforced by the client, not by hope.
- **No bulk or automated use.** This is one lookup, when a person taps a
  button, for a place they typed.

Breaking any of these gets an IP banned, and it would be entirely deserved.

### The rule

**A looked-up coordinate is shown before it is saved.** The geocoder returns
its best guess at a display name; the user confirms it is the right Shillong.
Silently accepting the first result is how a trip ends up routed to a village
in Karnataka with the same name.

## #26 — Weather snapshot

**Source:** Open-Meteo. Free, no API key, no account, and its terms cover
non-commercial use. That matters here beyond cost: **it does not block on
anything Yash has to sign up for.**

Fetched per stop for the trip's dates, at setup, and then frozen.

### The rule that is the whole issue

**A snapshot must never look current.** `cachedAt` is not nullable in the
schema for exactly this reason, and the UI states the age in words —
"forecast taken 5 days ago" — wherever a temperature appears. A five-day-old
forecast shown as today's weather is worse than no forecast, because someone
packs on it.

Staleness has three bands: fresh under two days, ageing under a week, and
**stale beyond that, which reads in `caution`**.

## #35 — Settings, call history, cache

**Settings**

- **Theme override.** System, day, or lamp. A phone in a pocket does not know
  it is night in a valley.
- **Corridor width**, the default for new legs.
- Nothing else. Every option is a thing that can be wrong.

**Call history** — what the diary already logs, read back. Copies count as
actions, because the dial happens in the Android dialer after a paste and the
app never sees it.

**Cache** — what each trip is holding: places, routes, weather. Clearing is
per trip and says what goes.

---

## Acceptance

**Coordinates** — [ ] A lookup returns candidates with display names.
[ ] Nothing is saved until the user picks one. [ ] Manual entry accepts and
validates a lat/lon pair. [ ] The client rate-limits itself to 1 req/sec.
[ ] The User-Agent names the app.

**#26** — [ ] A forecast parses per day for a date range. [ ] `cachedAt` is
always set. [ ] Age reads in words. [ ] Beyond a week it renders in caution.
[ ] Re-fetching replaces rather than duplicating.

**#35** — [ ] The theme override survives a restart. [ ] Call history reads
back, newest first. [ ] Cache counts are real. [ ] Clearing a trip's cache
removes exactly that trip's.
