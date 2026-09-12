# SafarSathi — Backlog v0.1

Checked box = done. Work top to bottom; dependencies are noted where they
bind. Never more than 2–3 issues in progress at once.

Sizing assumes ~5–10 hrs/week. **S** ≈ one sitting, **M** ≈ two or three,
**L** ≈ split it before starting.

Before starting any issue, write `docs/ISSUE_<N>_<Name>.md` first — the
step-by-step build guide with acceptance criteria. Writing the guide before
the code is what makes an issue resumable mid-way by a fresh chat.

---

## Milestone 1 — Foundation (make it run)

- [x] **#1 Flutter project scaffold** — S — *analyze clean, 11 tests green; device check still owed*
  Create project, folder structure (`core/`, `features/`), add
  dependencies: drift, drift_flutter, sqlite3_flutter_libs, path_provider,
  url_launcher, file_picker, csv, excel, libphonenumber_plugin.
  Verify: `flutter run` shows a blank app on device.

- [x] **#2 Drift database wiring** — M — *16 tables, 27 foreign keys, 9 schema tests; device check owed*
  `app_database.dart` with all tables from the drafted schema files
  (Trips, Stops, Legs, Pois, ChecklistItems, WeatherSnapshots,
  TimelineEntries, Expenses, ExpenseSplits, Contacts, ImportBatches,
  CallLogs, EmergencyHelplines, PoiContacts, TrustedContacts).
  Run `build_runner`, confirm generated code compiles.
  Verify: DB file created on device, tables present.
  *Depends on #1.*

- [x] **#3 Design tokens + theme** — M — *done as part of #1; device check owed* — *was S; grew with the v2 visual system*
  `app_tokens.dart` v2 in place: `AppColors` as a `ThemeExtension` with the
  full day and night sets, type roles, geometry scale, and `AppTokens.theme()`
  wiring both. Bundle **Inter and Archivo Narrow** (both SIL OFL). Add the
  128×128 grain tile at `assets/texture/grain_128.png`, under 4 KB.
  Splashes off app-wide (`NoSplash.splashFactory`); no shadow on any theme.
  Verify: a throwaway screen renders the palette in both themes, tabular
  figures align on a number, and switching the system theme swaps cleanly.
  *Depends on #1. See DESIGN_VISUAL_v2.md §2–§4.*

- [x] **#4 Emergency helpline seeding** — S — *14 rows, 4 flagged and skipped, 9 tests*
  Seed Tier 1 national numbers on first launch. **Must be idempotent** —
  re-running after reinstall or migration must not duplicate rows.
  **Do not seed the four `needsVerification` numbers** (1930, 1078, 1033,
  104) until they are confirmed against a `.gov.in` source.
  Verify: query returns the expected set; run seeding twice, count unchanged.
  *Depends on #2.*

---

## Milestone 2 — Contacts and dialer (the first usable thing)

- [x] **#5 ContactsDao** — M — *19 tests; found and closed an import hole in the invariant*
  Drop in the drafted DAO. Write unit tests for `watchContacts` ordering
  (pinned → confirmed → alphabetical), filter composition, and
  `watchUnconfirmedCount`.
  Verify: tests green against an in-memory DB.
  *Depends on #2.*

- [x] **#6 Diary screen** — M — *28 widget tests; screens now take streams, not DAOs*
  The diary: ruled entries, numbered margin, readiness banner, search.
  Wire routes. Seed a few fake contacts by hand to see it render.
  The number renders larger than the meta line — it is the point of the row.
  Verify: list renders, filters work, amber dot appears on `userEntered` rows
  and not on `userVerified` ones.
  *Depends on #3, #5.*

- [x] **#7 Copy, dialer, call, chat** — M — *17 tests; golden harness added, found 2 layout bugs*
  **Tap copies the E.164 number to the clipboard.** Toast above the nav bar
  with the number and an `OPEN DIALER` button that launches the platform
  dialer with an empty field. `url_launcher` for `tel:`, `sms:`, `wa.me` as
  explicit secondary actions. Log every action to CallLogs, **including
  `copy`**. Handle the no-app-available case gracefully.
  Verify: on a real device, tap copies and the number pastes into the Android
  dialer; `tel:` with no path opens the dialer rather than erroring. If it
  does error, fall back to `ACTION_DIAL` over a platform channel and log a
  decision about it.
  *Depends on #6.*

- [x] **#8 Add / edit contact screen** — M — *22 tests; editing digits now clears a confirmation*
  Single contact form. Category picker from the fixed vocabulary. E.164
  normalisation on save, storing both raw and normalised. Duplicate warning
  (not a block) when the E.164 already exists in the trip.
  Verify: add a contact, it appears in the dialer with an amber dot.
  *Depends on #6.*

- [x] **#9 Mark-as-confirmed flow** — M — *was S; includes the entry screen. 18 tests*
  The long-press sheet action that promotes `userEntered → userVerified`.
  Readiness banner updates live.
  Verify: confirm a contact, amber dot clears, banner count drops.
  *Depends on #8.*

---

- [ ] **#49 Category thumb index** — S
  The vertical index down the right edge of the diary. Selected tab filled.
  Needs a query returning only the categories the trip actually uses —
  eleven do not fit an edge.
  *Depends on #6. See SCREENS.md open question 1.*

- [ ] **#50 Emergency screen** — M — *see SCREENS.md §3*
  The exempt screen. Tap calls, heavy haptic, copy demoted to a secondary
  icon. Bundled helplines and trip contacts in separate headed sections.
  Provenance under every bundled line.
  *Depends on #4, #7.*

---

## Milestone 2.5 — Visual system (Milestone idiom)

Sequenced here on purpose: after the dialer exists and renders, before bulk
import multiplies the number of screens that would have to be retrofitted.
Read `DESIGN_VISUAL_v2.md` before starting any of these, §0 first.

- [ ] **#41 Motion + haptics core** — S
  `motion.dart`: durations, curves, the spring, `Motion.d()` reduce-motion
  collapse, the `Haptics` facade, and `PressScale`.
  Verify: every tappable scales and buzzes; turning on reduce-motion zeroes
  the animations and leaves the haptics working.
  *Depends on #3.*

- [ ] **#42 Retro primitives** — M
  `retro.dart`: `StencilLabel`, `TicketCard` with the perforation clipper,
  `MilestoneMarker`, `StampBadge`, `HazardStripe`, `GrainOverlay`,
  `RollingDigits`. Widget tests for the perforation path at three widths.
  Verify: a gallery screen renders all seven in both themes.
  *Depends on #41.*

- [ ] **#43 Dialer retro pass** — M
  Work `DIALER_RETRO_PATCH.md` top to bottom, all eleven edits.
  Verify: its acceptance list, every line.
  *Depends on #6, #42.*

- [ ] **#44 Confirmation stamp interaction** — S
  The stamp lands once on the `false → true` transition, haptic at contact,
  amber dot cross-fades out, readiness count rolls down.
  Verify: confirm a contact and watch all four happen in one 380ms beat;
  scroll past it afterwards and nothing animates or buzzes.
  *Depends on #9, #43.*

- [ ] **#45 Row swipe actions** — S
  Right to call, left to pin, both springing back rather than dismissing.
  Haptic on threshold crossing. Whole-row tap still dials.
  *Depends on #43.*

- [ ] **#46 Over-scroll cache stamp** — S
  Replaces pull-to-refresh, which would be a lie in an offline app. Shows
  `lastSyncedAt` and cache size.
  *Depends on #42. Real data arrives with #25; until then show the seeded
  placeholder and mark it as such.*

- [ ] **#47 Night theme audit** — S
  Walk every built screen in `AppColors.night`. Check contrast on real
  devices, not just on the numbers in the doc. Grain drops to 2%. The
  emergency red gets no glow treatment.
  *Depends on #43.*

- [ ] **#48 Safety-surface exemption test** — S
  **Golden test that fails if the emergency tab ever grows ephemera.**
  Asserts the emergency subtree contains no `StampBadge`, `TicketCard`,
  `MilestoneMarker` or `GrainOverlay`, and that the trust dot renders with
  `cautionMark` at 7px on `userEntered` rows.
  This is the test that keeps a future session from decorating the one
  surface that must not be decorated. Keep it green.
  *Depends on #43.*

---

## Milestone 3 — Bulk import

- [ ] **#10 Multi-add screen** — S
  Repeatable inline rows for adding 5–10 contacts without reopening a form.
  *Depends on #8.*

- [ ] **#11 Sheet parser** — M
  `file_picker` → CSV via `csv`, XLSX via `excel`. Multi-sheet XLSX shows a
  sheet selector. Return a normalised row list with original row indices
  preserved for error reporting.
  Verify: parse the template CSV and a hand-made XLSX, both yield 7 rows.
  *Depends on #2.*

- [ ] **#12 Column mapping screen** — M
  Auto-match headers against expected columns, let the user remap
  mismatches, remember the mapping for the session.
  *Depends on #11.*

- [ ] **#13 Validation + preview** — M
  E.164 normalisation, duplicate detection against existing contacts,
  missing-name and invalid-number flagging. Preview list with green / amber
  / red row states and per-row deselect.
  Verify: feed a file with a blank name, a bad number and a duplicate —
  each flags correctly and the rest still import.
  *Depends on #12.*

- [ ] **#14 Import commit + rollback** — M
  Single transaction: create `ImportBatch`, insert contacts with
  `importBatchId`, all rows as `userEntered` / `callConfirmed = false`.
  Import history screen with wholesale rollback.
  **Import must not be able to produce a confirmed contact.** Pin this with
  a test.
  Verify: import 7 rows, all show amber dots; roll back, all disappear.
  *Depends on #13.*

- [ ] **#15 Stop-name fuzzy matching** — S
  Match the `stop_name` column against the trip's Stops; unmatched rows
  still import as trip-level contacts.
  *Depends on #14.*

---

## Milestone 4 — Trip structure

- [ ] **#16 Trip + Stop CRUD** — M
  Create a trip, add ordered stops, reorder by drag, set arrival/departure
  dates and activity tags. Handle repeat visits (Shillong twice) correctly.
  Verify: build the full Meghalaya itinerary; two Shillong rows coexist.
  *Depends on #2.*

- [ ] **#17 Leg auto-generation** — S
  Legs derive from consecutive stops. Reordering stops regenerates legs
  without orphaning cached data where the pair is unchanged.
  *Depends on #16.*

- [ ] **#18 Transport leg details** — S
  Mode, planned departure/arrival, notes. Manual entry (no Rome2Rio-style
  live fetch — see PROJECT_RUNDOWN §3).
  *Depends on #17.*

- [ ] **#19 Dialer trip scoping** — S
  Wire the real trip and current stop into the dialer, replacing the
  hardcoded values from #6. Stop-scope toggle uses the live current stop.
  *Depends on #16, #6.*

- [ ] **#20 Pre-departure readiness check** — M
  Auto-generate a blocking checklist item per overnight stop: *"Call and
  confirm <stop> accommodation number."* Trip shows not-ready until each is
  confirmed.
  Verify: trip with one unconfirmed homestay reads not-ready; confirm it,
  reads ready.
  *Depends on #9, #16.*

---

## Milestone 5 — Offline sync

- [ ] **#21 Overpass client** — M
  Query by bbox and category. Parse into Pois. Extract `phone` /
  `contact:phone` tags into PoiContacts as `communityOsm`.
  Verify: query a small Shillong bbox, get plausible results.
  *Depends on #2.*

- [ ] **#22 Route polyline fetch** — M
  OSRM call per leg at setup time, store encoded polyline. **This is the
  only live routing call and it happens once, on WiFi.**
  *Depends on #17.*

- [ ] **#23 Corridor buffering** — M
  Buffer the polyline (default 3 km, per-leg adjustable), derive the query
  bbox, compute each POI's distance from the route.
  *Depends on #21, #22.*

- [ ] **#24 Map tile caching** — L — *split before starting*
  `flutter_map` + FMTC. Region download at a defined zoom range.
  **Use MapTiler or Stadia, not the standard OSM tile server** — see
  PROJECT_RUNDOWN §6. Needs an API key and a size estimate shown to the
  user before download.
  *Depends on #23.*

- [ ] **#25 Sync orchestrator** — L — *split before starting*
  "Download all" across every leg, per-leg progress, resumable on failure,
  `lastSyncedAt` per leg, re-sync while still online. This is the screen
  that likely justifies introducing Riverpod.
  *Depends on #23, #24.*

- [ ] **#26 Weather snapshot** — M
  Fetch per stop for trip dates at setup. Store with `cachedAt` and
  **display staleness prominently** — a five-day-old forecast must not look
  current.
  *Depends on #16.*

---

## Milestone 6 — En-route discovery

- [ ] **#27 Route discovery screen** — M
  Map + list of corridor POIs, category filter chips, ordered by distance
  along route with a "coming up in X km" treatment.
  *Depends on #24, #25.*

- [ ] **#28 POI detail** — S
  Details, distance off route, OSM contact number rendered **with an
  unverified marker**, "save as contact" action (lands as `userEntered`),
  deep-link out to Google Maps for reviews and photos.
  *Depends on #27.*

---

## Milestone 7 — Remaining feature set

- [ ] **#29 Checklist generation** — M — *see SCREENS.md §7*
  Activity-tagged pack lists from stop tags + nights + cached forecast.
  Generated items display the tags that produced them. **Manual edits must
  survive a regeneration** — needs a "user touched this" flag per item.
  Blocking items come from #20, rendered in the same list.
  *Depends on #16, #20.*

- [ ] **#30 Timeline / GPS logging** — L — *split before starting* — *see §9*
  Background location service, Polarsteps-style. The rail is the road: stops
  render as milestone caps, notes and photos as plain dots. **The logging
  toggle states its battery cost on screen.**
  *Depends on #16.*

- [ ] **#31 Expenses + splits** — M — *see SCREENS.md §8*
  Local ledger, named travellers with no accounts and no sync. Multi-currency
  with a manual rate snapshot shown with its date. Balances positive in
  signal, negative in muted, **never red**.
  *Depends on #16.*

- [ ] **#32 Simplify-debts algorithm** — M — with unit tests — *see §8*
  Collapse the IOU graph to the fewest settlement payments. Pure local maths.
  Settlement is **recorded, never executed** — no payment integration.
  Test with the three-person Meghalaya ledger and with a cycle.
  *Depends on #31.*

- [ ] **#51 Stop detail screen** — M — *see SCREENS.md §10*
  Weather snapshot with the staleness treatment, what-is-here counts,
  editable activity tags, per-stop cache sizes.
  *Depends on #16, #26.*

- [ ] **#52 Leg detail screen** — M — *see SCREENS.md §11*
  Typed transport details, and the corridor list ordered by distance along
  the route with each place on its own milestone.
  *Depends on #18, #23.*
- [ ] **#33 Trusted contacts + check-in** — M — SMS intent, no server relay
- [ ] **#34 Check-in escalation** — M — WorkManager timer, alert on ETA + buffer
- [ ] **#35 Settings + cache management** — S — per-trip cache size, clear, retention

---

## Milestone 8 — International

- [ ] **#36 Verify the four flagged numbers** — S
  Confirm 1930, 1078, 1033, 104 against government sources. Seed the ones
  that check out; drop the ones that don't.
  **Do this before any release, not before the Europe trip.**

- [ ] **#37 State-level helpline curation** — L — *split by region*
  Manual collection from 28 state + 8 UT portals, with provenance recorded
  per number. Ship progressively; partial coverage is fine, guessed
  coverage is not.

- [ ] **#38 EU emergency data** — S
  112 is EU-wide. Add country-specific supplements plus Indian embassy
  contacts per country.
  *Depends on #4.*

- [ ] **#39 Multi-currency expenses** — S
  Manual exchange-rate snapshot per country at setup.
  *Depends on #31.*

- [ ] **#40 Offline phrasebook** — M
  Per-country basic phrases. Lower priority than everything above.

---

## Parked (revisit only if the constraint changes)

- Google Places integration — costs scale badly for corridor queries
- Live transit schedules (Rome2Rio-style) — impossible offline
- Cloud sync / multi-device — would reintroduce a backend
- Self-hosted Overpass — only needed at real user scale
