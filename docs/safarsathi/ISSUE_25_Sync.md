# Issue #25 — The sync orchestrator

**Size:** L (split) · **Depends on:** #23, #24 · **Spec:** PROJECT_RUNDOWN §6

One screen that downloads everything a trip needs, in one press, with per-item
progress and an honest account of what failed.

---

## Why this exists

Every piece already works. Routes come from #22, places from #21, tiles from
#24, weather from #26. What was missing is that the user has to visit **three
separate screens** and know to, in the right order, before leaving WiFi.

That is not a feature gap, it is the difference between a trip that is
prepared and one that is not.

## No Riverpod

The backlog says this screen "likely justifies introducing Riverpod." It does
not, and it is not being added.

What this screen holds is a list of tasks and an index into it, for the
lifetime of one route. A `StatefulWidget` over a stream does that. Introducing
a state-management library for one screen would mean a dependency, a second
way of doing things beside the `StreamBuilder` every other screen uses, and a
migration decision for the twelve screens that already work without it.

If a later feature needs shared cross-screen state, revisit then. **The
backlog line is superseded, not forgotten.**

## Ordering, and why it is not arbitrary

```
per leg:    route  →  places  →  tiles
per stop:   weather
```

- **Places need the route.** The corridor is a band around the polyline; with
  no polyline the box falls back to the straight line, which in the Meghalaya
  hills is a different valley.
- **Tiles need the route** for the same reason, and they are last because they
  are by far the largest download. A failure there should not cost the small,
  useful things.
- **Weather needs neither.** It depends only on a stop having coordinates.

## One failure does not stop the run

This differs deliberately from the tile downloader, where a rejected key
aborts everything. Here each task is independent and a failure is **recorded
and stepped over**:

- one leg with no coordinates does not stop the other three,
- Overpass being busy does not cost you the tiles,
- a stop outside the forecast range does not cost you the route.

The screen then **lists what failed and why**, in words. A sync that silently
does 60% is worse than one that says which 40% is missing.

## Resumable, by construction

Nothing here tracks its own progress. Every underlying step already skips work
already done: tiles check the disk, routes and places overwrite cheaply,
weather replaces. Pressing download again after a failure re-runs the whole
plan and only the missing parts cost anything.

That is simpler than a resume cursor and it cannot get out of step with what
is actually on disk.

---

## Acceptance

- [ ] One press downloads routes, places, tiles and weather for a whole trip.
- [ ] Progress names the item being worked on, not just a percentage.
- [ ] A failing task is recorded and the run continues.
- [ ] Failures are listed afterwards, each with its reason.
- [ ] `lastSyncedAt` is set per leg by the underlying step, not by this one.
- [ ] Re-running after a partial failure fetches only what is missing.
- [ ] A trip with no coordinates says so rather than reporting success.
- [ ] No new state-management dependency.
