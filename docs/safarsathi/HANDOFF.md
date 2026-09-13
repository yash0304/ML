# HANDOFF — SafarSathi — 2026-09-13 (after #25, the sync orchestrator)

> Overwrite this file at the end of every session. It must let a cold model
> (any model) resume in under 2 minutes.

## Where we are

- **Issues #1–#9, #11–#26, #29, #31, #32, #35, #50, #53, #54, #55 are done
  except for the device check.** Analyses clean, **605 tests** in about fifty
  seconds on Flutter 3.47.3 / Dart 3.13.3.
- Backlog position: 34 / 56 done.
- **One press downloads everything.** More → "Download everything" runs every
  leg's route and places, every stop's weather, and the whole map, with
  per-item progress and a list of anything that failed.
- Database at **schemaVersion 4**. Steps ascend, each additive, none edited.

## Blocked on nothing — but #36 could not be done here

**#36, verifying 1930, 1078, 1033 and 104, was attempted and abandoned.** This
container's egress proxy blocks `.gov.in` entirely: `cybercrime.gov.in`,
`ndma.gov.in` and `www.ndma.gov.in` all refused.

The project rule is absolute — no emergency number ships without a
government-domain source recorded in `sourceNote`. A search-engine summary is
not that. **Those four numbers remain unseeded and must stay unseeded until
someone reads a `.gov.in` page directly.** Yash can do it from a browser in
five minutes; a future session on a different network may be able to.

## The map, and its key

**MapTiler**, chosen by Yash, behind a `MapTileProvider` interface. Nothing
outside `features/map/data/tile_provider.dart` names it; swapping to Stadia is
a new implementation and one line in `activeTileProvider`.

**The key is never committed.** `String.fromEnvironment('MAPTILER_KEY')`,
filled from `--dart-define`: a gitignored `maptiler.json` locally, a GitHub
Actions secret in CI. A build with no key runs and disables the map with a
sentence. See RUNNING.md.

The tile cache is **written here, not FMTC** — FMTC stores tiles in ObjectBox,
a second native database engine beside SQLite. Tiles are files under
`<documents>/tiles/<provider>/<z>/<x>/<y>.png`, indexed in the `MapTiles`
table.

**`OfflineTileProvider` has no HTTP client at all.** A tile that was not
downloaded renders as a transparent pixel. That absence is the guarantee.

## BLOCKED: the tile provider

**#24 cannot start until Yash picks MapTiler or Stadia and creates an account.**
Both have a free tier that covers one person's trips. The standard OSM tile
server is explicitly not an option — its usage policy forbids bulk downloading,
which is the entire feature. See PROJECT_RUNDOWN §6.

Whoever picks this up: ask, do not guess, and do not ship a key in the repo.

## The corridor, in one paragraph

`Corridor` takes a decoded route and a buffer in kilometres. It yields the
bounding box to query, and for any point, how far off the route it sits and how
far along. `OsrmClient` fetches one route per leg at setup; `OverpassClient`
asks OpenStreetMap what is in the box. `CorridorSync.syncLeg` runs all three
and writes the result in **one transaction at the end**, so a failure halfway
leaves the leg exactly as it was. Every phone number that arrives this way is
`communityOsm` and cannot be anything else.

## The checklist and the ledger, in one paragraph

The checklist is generated from stop tags and nights, with the blocking items
from #20 sitting above the pack list in the same screen. Its one promise is
that **your edits survive regeneration** — including a removal, which marks
the item edited-and-done rather than deleting a row the generator would simply
rewrite. The ledger takes expenses with even or hand-set splits, and **will
not save a split that does not sum to the amount**. Every figure is integer
paise, and `formatRupees` shows those paise whenever they exist.

## Trip structure, in one paragraph

A trip has ordered stops; legs exist between consecutive stops and are never
created by hand. Editing lives in the itinerary screen (More tab, or the EDIT
rule on the Trip tab's Stops header), which is separate from the read-only
Trip tab on purpose. The rule that shapes the data model: **a place may appear
twice**, so nothing keys a stop by name. The rule that earns issue #17: **a
leg whose from/to pair is unchanged keeps its row**, and therefore the cached
route that cost a network connection to obtain.

## The import flow, in one paragraph

`More → Import from a sheet`. The file picker lives in `ImportFlow` and
nowhere else; everything under it takes plain values, which is why parsing,
matching and validation are all testable with no platform channel. Parse →
choose a sheet if the workbook has several → map columns → preview → commit.
Nothing touches the database until commit, and then it all lands or none of it
does.

## Type

Three faces, three jobs, changed at Yash's request on 2026-09-12 because
Inter made the app look generated:

- **Jost** — signage. A Futura revival; Futura is what was painted on enamel
  road signs and milestone caps from the sixties to the eighties.
- **Archivo** — words.
- **Courier Prime** — numbers. A typewriter, and monospaced, so the column
  alignment the diary depends on is structural rather than a font feature.

The rule when a number could go either way: a number you might dial, count or
compare is typed; a number painted onto an object is signage.

## Running it on a phone

`RUNNING.md` has the full procedure, the four things only hardware can
settle, and what to do when each fails.

## Look at the app without a phone

```
flutter test --update-goldens test/golden_test.dart
```

Writes `test/goldens/*.png` — twenty-eight images now: the diary in both themes, empty
and mid-copy; the entry form; an entry unconfirmed and confirmed; the
emergency screen; the trip screen; the money screen. Rendered from the real
widget tree with the bundled fonts and the SDK icon font loaded.

**This is the fastest way to see a change, and the highest-value tool in the
project.** It has caught six bugs no widget test could see: a shadow painting
the margin rule as a grey column, the FAB sitting over the thumb index, the
toast hidden under the FAB, record rows centring instead of stretching, and
the missing rupee glyph. A widget test asserts the text is in the tree; only
an image shows you it rendered as a box.

## Prototypes

- Screens, eleven, live: https://claude.ai/code/artifact/548c02bc-d0ef-439d-abbc-7a5909a95d14
- Visual system: https://claude.ai/code/artifact/0d81dd67-d217-4056-81b3-33e0bb889e95

Contact numbers in them are illustrative placeholders. The helplines are the
real Tier-1 codes with their government sources.

## What is verified, and what is not

| Checked | How |
|---|---|
| Compiles, analyses clean | `flutter analyze` — no issues |
| Palette contrast, both themes, both grounds | test, WCAG ratios computed |
| Tabular figures on every numeric style | test |
| The `wght` axis pinned on every style | test |
| No drop shadow in either theme | test |
| Layout at 400×800 and in night palette | widget test, no exceptions |
| All 16 tables exist | test against `sqlite_master` |
| Foreign keys on, cascades work | test — deletes a trip, checks orphans |
| No contact can arrive confirmed | test |
| Double-seeding a helpline is rejected | test — makes #4 idempotent |
| POI is stop XOR leg | test — both-null and both-set rejected |
| A 3-way split of ₹3,200.11 loses nothing | test |
| Seeding twice changes nothing, ids intact | test |
| 1930, 1078, 1033, 104 never reach the DB | test |
| Every seeded number carries a source | test |
| The state helpline list ships empty | test |
| Diary ordering: pinned, confirmed, alphabetical | test |
| Filters compose; stop scope keeps trip-wide | test |
| **Import cannot produce a confirmed contact** | test |
| A failed batch leaves nothing behind | test |
| A copy is logged like any other action | test |
| The trust dot renders on untrusted tiers only | widget test, both themes |
| Diary margin numbers, page footer, empty states | widget test |
| Search, thumb index, stop-scope toggle | widget test |
| The screen fits 400×800 in both themes | widget test |
| Copy prefers E.164, falls back to raw | test |
| Every action logs; a failed one does not | test |
| WhatsApp strips non-digits | test |
| A failed launch says so instead of nothing | test |
| The toast shows, offers the dialer, expires | widget test |
| E.164 normalisation, India and Germany | test |
| A bad number warns but still saves | test |
| A duplicate warns but never blocks | test |
| **Editing digits clears a confirmation** | test |
| A new entry always lands unconfirmed | test |
| Provenance reads right on all five tiers | widget test |
| Confirming lands the stamp, clears the dot | widget test |
| Un-confirming restores the unconfirmed line | widget test |
| Government tiers get no confirm section | widget test |
| Even splits sum back to the total, n = 1..12 | test |
| Simplified payments sum back to the balances | test |
| Rupee amounts use Indian grouping | test |
| **The rupee sign renders, not a tofu box** | test on the glyph |
| Settle-up sits above the ledger | widget test |
| The milestone cap mutes on an unsynced leg | widget test |
| Five tabs switch and keep their state | widget test |
| SOS is red only while active | widget test |
| The emergency screen carries no grain or stamps | widget test |
| CSV and XLSX parse to the same shape | test |
| Row numbers survive blank-row removal | test |
| **An Excel float phone number renders as digits** | test |
| A short row pads; a long one truncates | test |
| Header aliases match real-world spellings | test |
| `no` inside `notes` does not steal `phone` | test |
| No column is claimed by two fields | test |
| The mixed file flags each problem, imports the rest | test |
| In-file duplicates are caught | test |
| A skipped row cannot be selected | test |
| A blank cell never flags an emergency number | test |
| Seven rows import as seven unconfirmed contacts | test on a real DB |
| **Import still cannot produce a confirmed contact** | test, re-proven by removing the guard |
| Rollback removes exactly that batch | test |
| Rollback takes confirmed contacts with it | test |
| History counts what survives, not what landed | test |
| `Puri` does not match `Pune` | test |
| An unmatched stop still imports, trip-wide | test |
| **Two Shillong rows coexist with distinct ids** | test on the real itinerary |
| Reordering renumbers densely | test |
| Deleting a stop keeps its contacts, unattached | test |
| Deleting a trip takes everything under it | test |
| **A reorder preserves an earlier leg's cached route** | test on a real DB |
| A repeated pair gets its own leg row | test |
| Removing a middle stop joins its neighbours | test |
| Exactly one trip is ever active | test |
| The current stop follows today's date | test |
| **A dateless trip still has a current stop** | test |
| Nights derive from dates; reversed is zero | test |
| **Unconfirmed blocks; confirming clears** | test |
| **An absent number blocks, with different words** | test |
| A pass-through stop never blocks | test |
| A user-edited checklist item survives regeneration | test |
| The readiness panel uses no emergency red | widget test |
| combineLatest2 waits for both, closes on both | test |
| Tags and nights produce pack items | test |
| A count scales with its own tag's nights | test |
| **An edited label survives regeneration** | test |
| **A removed generated item stays removed** | test |
| Ticking is not an edit | test |
| **A pre-v2 row is adopted, not duplicated** | test |
| `generator_key` exists on the table | test against PRAGMA |
| Rupee text parses to exact paise | test |
| More than two decimals truncate | test |
| Nonsense parses to null, never zero | test |
| **An unbalanced split cannot be saved** | test, form and editor |
| **A traveller in the ledger cannot be removed** | test |
| The Meghalaya ledger settles in two payments | test |
| A cycle collapses to nothing | test |
| Paise are shown, never rounded away | test |
| The money summary updates on a new traveller | test |
| Haversine matches known distances | test |
| **Distance measures to the segment, not the vertex** | test |
| **Box padding widens more in longitude than latitude** | test |
| Padding is really the distance asked for | test |
| Google's canonical polyline decodes | test |
| Polylines round-trip at precision 5 and 6 | test |
| The wrong precision is off by ten | test |
| An oversized Overpass box is refused | test |
| An unnamed OSM place is dropped | test |
| A hotel with a restaurant reads as a hotel | test |
| **Every OSM number lands as `communityOsm`** | test on a real DB |
| A semicolon list becomes several numbers | test |
| Re-syncing replaces rather than accumulating | test |
| **A failed sync leaves the leg exactly as it was** | test |
| OSRM coordinates go in lon,lat order | test |
| Places outside the corridor are excluded | test |
| **The geocoder rate-limits itself to 1/sec** | test, injected clock |
| A lookup shows its candidates before saving | widget test |
| An out-of-range coordinate is refused, not clamped | test |
| A stop with no coordinates says what that costs | widget test |
| Forecast days parse from a date range | test |
| **Every day of one fetch shares its cachedAt** | test |
| **Staleness bands, and the age in words** | test |
| Beyond a week renders in caution | widget test |
| Re-fetching weather replaces rather than duplicating | test |
| The theme override reports the chosen mode | widget test |
| A cache row states what is actually held | widget test |
| Clearing says what survives it | widget test |
| Call history shows copies as actions | widget test |
| `app_settings` has `key` as its primary key | test against PRAGMA |
| A setting upserts rather than duplicating | test |
| Tile maths matches a hand-worked coordinate | test |
| **Latitude clamps at the Mercator limit** | test |
| Tile count grows as 4^z | test |
| Counting matches generating | test |
| **No key disables the map and says why** | test + widget test |
| **No real key is compiled into the test build** | test |
| Attribution is on every drawn map | widget test |
| A missing tile is null, never an error | test |
| Two providers cannot see each other's tiles | test |
| **Clearing removes the files, not just the index** | test |
| The estimate is shown and fetches nothing | test + widget test |
| **A second download fetches nothing** | test |
| A rejected key stops the run; a missing tile does not | test |
| Adjacent legs do not double-count shared terrain | test |
| Legs without coordinates are named, not skipped | widget test |
| `map_tiles` identity is unique per provider | test |
| One corridor task per leg, one weather per stop | test |
| **Route and places are one task, not two** | test |
| Every corridor runs before the map | test |
| **One failure does not stop the run** | test |
| A failure carries the task it belongs to | test |
| **Re-running fetches only what is missing** | test |
| Progress names the item being worked on | widget test |
| Failures are listed with their reason | widget test |
| Skipped legs and stops are named | widget test |
| Counts read as singular for one | widget test |
| No failure renders in emergency red | widget test |

**Not verified, and not verifiable without a phone:**

- **Whether the fonts actually load.** The widget test harness substitutes its
  own font, so no automated check will ever catch a family-name mismatch.
  If the stencil labels do not look condensed on device, `pubspec.yaml` and
  `app_tokens.dart` disagree.
- Whether the haptics fire.
- Whether the night palette is pleasant at 2am, as opposed to merely passing.
- Whether the grain reads as texture or as dirt.

## Testing rules learned at #6 — do not relearn these

1. **A widget test cannot close a Drift database.** `close()` awaits work the
   fake clock never advances; the test hangs until the runner is killed, with
   no error and no timeout.
2. **Cancelling a Drift query stream schedules zero-duration cleanup timers**
   during disposal, and the pending-timer check runs after user teardowns, so
   pumping in `addTearDown` cannot clear them.
3. **`pumpAndSettle` never settles** on a screen whose loading state is a
   `CircularProgressIndicator`.
4. A filter change swaps in a new stream that delivers on a microtask, so an
   interaction needs **two** pumps.

The answer to all of it: **screens take streams, not DAOs.** Widget tests use
in-memory fakes with no database. Keep it that way.

Also: `pkill -f "flutter test"` matches the shell running the command and
kills it, which looks exactly like a crash.

## What issue #5 found

**The drafted `insertBatch` could import a confirmed contact.** It passed the
caller's companion straight through, so a screen building a verified row would
have landed it verified — and the readiness check, the amber dot and the
blocking checklist items all read that flag. Now forced in the DAO, which is
the single write surface. A test proves the hole exists without the guard.

Also: the drafted ordering comment claimed recency the code never did, and
should not — `lastCalledAt` is null for never-called contacts, so ordering by
it would sink exactly the numbers the readiness system wants surfaced.

## What issue #2 found

**Code generation silently produced a schema with no foreign keys at all.**
At `drift_dev` 2.31 under `analyzer` 10, every `references(...)` was
discarded; `build_runner` reported success and wrote 20 outputs, with only a
vague warning about a class name scrolling past. Upgrading drift to 2.35
fixed it, which cascaded into bumping `drift_flutter` and
`sqlite3_flutter_libs`.

The tests now assert against `sqlite_master`, not against generated Dart.
Reading a generator's output to check the generator is circular.

## What the first compile found

Each of these was silent and would have cost a session later:

1. **The repo root `.gitignore` is a Python one; its `lib/` rule swallowed the
   whole Dart source tree.** The first commit reported success and contained
   no code. Fixed with `!lib/` at the top of `safarsathi/.gitignore` — there
   is a comment there saying not to remove it.
2. **Google Fonts no longer ships static instances** of Inter or Archivo
   Narrow. Both are bundled as variable fonts; every style pins `wght` via
   `FontVariation` or the app renders at one weight with no warning.
3. **`muted` failed AA.** `#6B7670` measures 4.41:1 on paper and 3.94:1 on
   stone against a documented 4.9:1 — the design doc's arithmetic was wrong.
   Now `#5F6963`, asserted by test on both grounds.
4. Two `clamp` calls returning `num` where `double` was required, and a
   `CustomPaint` with an infinite size. Caught by reading, before the SDK
   finished downloading.
5. The 8-bit grain tile was 15 KB against a 4 KB budget; 2-bit is 3.2 KB.

## Next action (this line starts the next session)

**#27 and #28, the discovery screen and POI detail.** The corridor data has
been downloaded since #21–#23 and no screen has ever shown it to a person.
That is the last big gap between what the app knows and what it tells you: a
list of what is coming up along the current leg, ordered by distance along the
route, each with "in 12 km".

Both specs are written: SCREENS.md §5 and the backlog entries. #28 must render
an OSM phone number **with its unverified marker**, and saving one lands it as
`userEntered` — the tier rules already enforce that in `CorridorSync`, but the
screen has to say it.

After that, #51 and #52 (stop and leg detail), then #10 (multi-add), then the
visual polish pass #41–#49.

**#36 needs a browser Yash controls.** See the section above.

**#30 (GPS timeline) and #33/#34 (check-in escalation) involve background
services.** Neither before October.

Still owed on hardware, none of it confirmed in words yet:

1. Do the three fonts load, or is everything on a silent fallback?
2. Are the haptics felt, especially the heavy one on the emergency screen?
3. Does `tel:` with an empty path open the Android dialer, or error?
4. Is the night palette pleasant at 2am, as opposed to merely compliant?

**Also unverified:** no map tile has ever actually been fetched. Every test
fakes the HTTP. The first real download is the first proof the MapTiler URL
shape and key are right.

## Note on the environment

Flutter is **not** installed in a fresh container. Getting it took a 1.5 GB
download. If a future session needs to compile, budget for that or work on a
machine that already has it.

## Open questions / waiting on Yash

1. **Project name.** "SafarSathi" is still a placeholder.
2. **Dedicated repo.** `docs/safarsathi/` and `safarsathi/` are parked in
   `yash0304/ml` and move together.
3. **Map tile provider.** MapTiler or Stadia. Needed before #24.
4. **Trip target.** Meghalaya is 1–5 October. Contacts-only v0.1 is plausible;
   the offline sync milestone is not.
5. **Archivo Narrow** — look at it on the phone. It is the one purely
   aesthetic choice in the system.
6. **`tel:` with an empty path** — confirm it opens the Android dialer.
   Fallback is `ACTION_DIAL` over a platform channel.
7. **Thumb index overflow.** Eleven categories do not fit a phone edge.
8. **Checklist regeneration** must not discard manual edits. The
   `isUserEdited` column exists for this; the rule is decided at #29.
9. **`sqlite3_flutter_libs` resolves to `0.6.0+eol`.** The `+eol` marker
   suggests the package is being retired, probably folded into `sqlite3`.
   Worth ten minutes reading its changelog before the build depends on it
   for a year. Not urgent — it works, and it is what `drift_flutter` 0.3.1
   requires.

## Decided in chat but check DECISIONS.md logged them

Empty. All decisions across this session are in DECISIONS.md.
