# HANDOFF — SafarSathi — 2026-09-13 (after #51 and #52, the detail screens)

> Overwrite this file at the end of every session. It must let a cold model
> (any model) resume in under 2 minutes.

## Where we are

- **Issues #1–#9, #11–#29, #31, #32, #35, #50–#55 are done except for the
  device check.** Analyses clean, **661 tests** in about fifty seconds.
- Backlog position: 38 / 56 done.
- Every screen SCREENS.md specifies now exists. What remains is polish,
  background services, and one thing that cannot be done from this container.

## THE LIVE PROBLEM: Yash cannot enter the MapTiler key

**This is the next thing to fix.** Yash reported on 13 Sep, with a screenshot
of Settings, that there is nowhere in the app to put his MapTiler key, so map
download does nothing and Settings reads "Nothing downloaded".

He is right, and it is a real gap. The key is currently **build-time only** —
`String.fromEnvironment('MAPTILER_KEY')`, supplied by `--dart-define`. Locally
that comes from a gitignored `maptiler.json`; in CI from a GitHub Actions
secret named `MAPTILER_KEY` which **has not been added to the repository**. An
APK built without it runs fine and disables the map with a sentence, which is
exactly what he is seeing.

Two ways out, and they are not exclusive:

1. **The zero-code one:** Yash adds `MAPTILER_KEY` under the repository's
   Settings → Secrets and variables → Actions → New repository secret, then
   re-runs the APK workflow. The workflow already reads it and already prints
   whether maps ended up enabled.
2. **The one he actually expected:** a field in Settings where the key is
   pasted at runtime and stored in `AppSettings`. This also honours the
   original constraint — the key never touches the repository — and it works
   for any build, including ones he did not produce. `MapTilerRaster` already
   takes `apiKey` as a constructor argument, so the provider abstraction is
   ready for it; what is not ready is `activeTileProvider`, which is a
   `const` resolved at compile time. That constant is the thing to change.

Do not paste a key into any file. He has it; he has never shared it, correctly.

## Blocked on nothing else — but #36 could not be done here

**#36, verifying 1930, 1078, 1033 and 104, was attempted and abandoned.** This
container's egress proxy blocks `.gov.in` entirely: `cybercrime.gov.in`,
`ndma.gov.in` and `www.ndma.gov.in` all refused.

The project rule is absolute — no emergency number ships without a
government-domain source recorded in `sourceNote`. A search-engine summary is
not that. **Those four numbers remain unseeded and must stay unseeded until
someone reads a `.gov.in` page directly.**

## What #51 and #52 added

Two detail screens, both assembly over data that already existed. No new
table, no migration, no client. See `ISSUE_51_52_Detail.md`.

- **Stop** (`stop_detail_screen.dart`, `stop_detail.dart`): the forecast with
  its staleness stamp, counts with chevrons, editable activity tags that
  really do rebuild the packing list, and what is cached for the roads either
  side. Reached by tapping a stop in the itinerary; the stop form is now one
  pencil further in.
- **Leg** (`leg_detail_screen.dart`): typed transport with the sentence
  saying it is typed, then the corridor's first five places on a milestone
  rail ordered by kilometres along the route, then `SEE ALL`.

**A correction shipped with them.** SCREENS.md §10 has always put the weather
caution boundary at three days; #26 shipped seven. The bands are now fresh
under a day, ageing one to three, stale at three and over, and the wording
moved with them. Logged in DECISIONS.md.

## The map, and its key

**MapTiler**, chosen by Yash, behind a `MapTileProvider` interface. Nothing
outside `features/map/data/tile_provider.dart` names it; swapping to Stadia is
a new implementation and one line in `activeTileProvider`.

The tile cache is **written here, not FMTC** — FMTC stores tiles in ObjectBox,
a second native database engine beside SQLite. Tiles are files under
`<documents>/tiles/<provider>/<z>/<x>/<y>.png`, indexed in the `MapTiles`
table.

**`OfflineTileProvider` has no HTTP client at all.** A tile that was not
downloaded renders as a transparent pixel. That absence is the guarantee.

## The shape of the code, in one paragraph

Drift over SQLite is the only source of truth, at **schemaVersion 4**.
Migration steps run in ascending order, each additive; **no shipped migration
is ever edited**. There is no repository layer and no state-management
library: DAOs expose streams, screens take streams rather than DAOs, and
`lib/app.dart` is the composition root that wires one to the other. All
spacing, colour and type comes from `AppTokens`. Money is integer paise.

**The Drift trap this project has been bitten by four times now:** a stream
fires only for the tables its own query touches. Anything that joins or counts
across tables needs
`db.customSelect('SELECT 1', readsFrom: {...}).watch()` as a tick.

## How to work on it

```
cd safarsathi
/opt/flutter/bin/flutter analyze
/opt/flutter/bin/flutter test
/opt/flutter/bin/flutter test --update-goldens test/golden_test.dart
```

**Look at the goldens.** Thirty-three images under `test/goldens/`. They have
caught roughly a dozen bugs no other test could — "1 legs", "roughly about
24 MB", three shares that did not add up to the total, and this round a
milestone rail that broke into disconnected segments at every row boundary.

## Still unverified on hardware

The three fonts loading, haptics firing, whether `tel:` with an empty path
opens the Android dialer, the night palette at 2am, and — still — **no map
tile has ever actually been fetched.** Every test fakes the HTTP. That last
one is now blocked on the key problem above.

## What is left

- **The key problem above.** First.
- **#10** multi-add for contacts.
- **#41–#49**, nine small visual polish issues.
- **#36**, needs a browser Yash controls.
- **#30, #33, #34** background services — not before October.
