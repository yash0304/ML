# HANDOFF — SafarSathi — 2026-09-13 (after the polish pass, #41–#49)

> Overwrite this file at the end of every session. It must let a cold model
> (any model) resume in under 2 minutes.

## Where we are

- **Issues #1–#9, #11–#29, #31, #32, #35, #41–#46, #48–#55 are done except
  for the device check.** Analyses clean, **698 tests** in about fifty
  seconds.
- Backlog position: 46 / 56 done.
- Every screen SCREENS.md specifies exists, and the retro pass over them is
  finished. What remains is #10, one audit, background services, and one
  thing that cannot be done from this container.

## The MapTiler key — fixed, and Yash needs to do one thing

Yash reported on 13 Sep, with a screenshot of Settings, that there was nowhere
in the app to put his MapTiler key, so map download did nothing. He was right:
the key was **build-time only**, and `--dart-define` cannot help someone who
did not produce the build.

**There is now a field: Settings → Map key.** Paste, Save, then More → Map to
download. It is stored in the app's own database, it overrides any key baked
in at build time, and tiles already on disk survive a key change because the
cache id never depended on the key.

**What Yash still has to do:** paste the key into that field on the phone,
and — separately, and importantly — restrict the key to this app's package
name in the MapTiler dashboard. Optionally add `MAPTILER_KEY` as a repository
secret so CI builds ship with maps already on; the workflow already reads it.

Do not paste a key into any file. He has it and has never shared it, correctly.

## Blocked on nothing else — but #36 could not be done here

**#36, verifying 1930, 1078, 1033 and 104, was attempted and abandoned.** This
container's egress proxy blocks `.gov.in` entirely: `cybercrime.gov.in`,
`ndma.gov.in` and `www.ndma.gov.in` all refused.

The project rule is absolute — no emergency number ships without a
government-domain source recorded in `sourceNote`. A search-engine summary is
not that. **Those four numbers remain unseeded and must stay unseeded until
someone reads a `.gov.in` page directly.**

## What the polish pass added

Most of the retro idiom had landed as the screens were written. Three things
never had, and all three were interactions, which is why they slipped — a
static golden cannot see a missing gesture. See `ISSUE_44_46_Diary.md`.

- **#44** the confirmation stamp now lands on a diary row, once, on the
  `false → true` transition.
- **#45** swipe right to call, left to pin, both springing back.
- **#46** over-scroll reveals when the trip was last downloaded.

**Two real bugs came out of it, both worth knowing about.**

`DIALER_RETRO_PATCH.md` edit 8 says `if (confirmed) StampBadge(landed:
confirmed)`. That can never animate — the widget only exists in the landed
state, so it never sees the transition it exists to animate. `StampBadge` now
collapses to zero width instead and is mounted unconditionally.

The over-scroll stamp first read `metrics.pixels`, which works on iOS and
does nothing on Android: clamping physics never lets pixels go negative. It
reads `OverscrollNotification` now.

**#41, #42, #43, #48 and #49 were built along the way and never ticked.**
Each now carries the verification the backlog asked for — reduce-motion
collapsing durations while haptics survive, the perforation path at four
widths, and #48 extended to prove the new stamp and swipe did not leak onto
the emergency tab.

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

## The map

**MapTiler**, chosen by Yash, behind a `MapTileProvider` interface. Nothing
outside `features/map/data/tile_provider.dart` names it; swapping to Stadia is
a new implementation and one line there. `tileProviderFor(typedKey)` is what
the app calls — the key comes from Settings, falling back to the build.

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
tile has ever actually been fetched.** Every test fakes the HTTP. That one is
now unblocked — it needs Yash to paste his key and press download.

## What is left

- **#10** multi-add for contacts — the last unbuilt feature before October.
- **#47** night theme audit. Deliberately NOT ticked: the numbers check out
  (grain drops to 2%, the emergency red has no glow) and both themes render
  in tests, but the backlog asks for contrast checked *on a real device* and
  nobody has done that. Do not tick it from a container.
- **#36**, needs a browser Yash controls.
- **#30, #33, #34** background services — not before October.
