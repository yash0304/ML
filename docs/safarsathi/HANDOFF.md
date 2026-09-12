# HANDOFF — SafarSathi — 2026-09-12 (after issues #50, #53, #54, #55)

> Overwrite this file at the end of every session. It must let a cold model
> (any model) resume in under 2 minutes.

## Where we are

- **Issues #1–#9, #50, #53, #54, #55 are done except for the device check.**
  The project analyses clean and passes **165 tests** in about nine seconds on
  Flutter 3.47.3 / Dart 3.13.3. It has never run on a phone.
- Backlog position: 13 / 55 done. **Four of the five tabs are live.** Diary,
  Trip, Money and SOS all render real data off the demo trip; **More** is
  still a `NotBuiltYet` placeholder (import, history, settings — #11–#15, #35).
- **An APK exists.** GitHub Actions builds a release APK on every push to
  `claude/offline-retro-modern-app-design-r2n4ju`; the workflow runs
  `flutter analyze` and the full suite first, so any APK that exists passed
  everything. Download it from the run's `safarsathi-apk` artifact.
- **The contacts half of the app is whole.** Add an entry, edit it, copy its
  number, open the dialer, and confirm it — which clears the amber dot and
  drops the readiness count. That loop is the thing the whole project was
  designed around.
- The app opens with no trip and a **"Create demo trip"** button, which seeds
  three stops, two legs (the second deliberately unsynced, so the milestone
  cap renders muted), seven contacts, three travellers and five split
  expenses. Every seeded number is `+91 90000 000xx` and obviously fake.

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

Writes `test/goldens/*.png` — ten images now: the diary in both themes, empty
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

**Install the APK and use it.** Thirteen issues are done and not one line has
executed on hardware. Four things no test can check are stacked up:

1. Do the three fonts actually load? If the signage is not geometric and the
   numbers are not a typewriter, the family names in `pubspec.yaml` and
   `app_tokens.dart` disagree and everything since #1 sits on a silent
   fallback.
2. Are the haptics felt, especially the heavy one on the emergency screen?
3. Does `tel:` with an empty path open the Android dialer, or error? The whole
   copy-first workflow rests on this. If it errors, fall back to `ACTION_DIAL`
   over a platform channel and log a decision.
4. Is the night palette pleasant at 2am, as opposed to merely compliant?

Tap **Create demo trip** on first launch or every screen is empty.

**Then the only stub left is More** — bulk import (#11–#15), call history
(#35), settings. If October is close, #11–#15 is the highest-value block:
Yash's real contacts live in a sheet, and typing them in one at a time is not
a plan.

After that, #16 (trip editing) is what turns the demo trip into his trip.

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
