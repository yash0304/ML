# HANDOFF — SafarSathi — 2026-09-11 (second session block)

> Overwrite this file at the end of every session. It must let a cold model
> (any model) resume in under 2 minutes.

## Where we are

- Current issue: **#1 Flutter project scaffold — not started.** Nothing has
  been compiled or run.
- Backlog position: 0 / 50 issues done.
- Two design passes are complete. Screens are specified. The next thing that
  happens is code.

## Prototypes

- Screens: https://claude.ai/code/artifact/548c02bc-d0ef-439d-abbc-7a5909a95d14
- Visual system: https://claude.ai/code/artifact/0d81dd67-d217-4056-81b3-33e0bb889e95

Both are interactive. Numbers shown in them are illustrative placeholders
except the Tier-1 helplines, which are real.

## What happened this session

- **The dialling model changed.** Yash dials by pasting into the Android
  dialer, so tapping an entry now copies the number instead of dialling.
  The copy toast carries an `OPEN DIALER` button. This reverses an earlier
  decision; both lines are in DECISIONS.md.
- **Contacts became a diary.** Ruled lines, numbered margin, category thumb
  index down the right edge, page count at the foot. The retro idiom now
  does real work instead of decorating a list.
- Wrote `SCREENS.md` — six specifications: Diary, Entry, Emergency, Trip,
  New entry, Import preview.
- Logged eight more decisions. One supersedes the tap-to-dial rule.
- Renamed backlog #6 to the Diary screen, grew #7 from S to M for the copy
  workflow, added #49 (thumb index) and #50 (emergency screen).

## In-flight state

- Files touched: `SCREENS.md` (new), `DECISIONS.md`, `BACKLOG_v0.1.md`,
  `README.md`, this file.
- The three drafted Dart files are unchanged from the previous block and
  **still reflect tap-to-dial**. `DIALER_RETRO_PATCH.md` edits 6 and 7 need
  rewriting for copy-first before anyone follows them — do that as the first
  step of #7, not now.
- Last known good: all committed on
  `claude/offline-retro-modern-app-design-r2n4ju`.

## Next action (this line starts the next session)

**Write `docs/ISSUE_1_Scaffold.md`, then do backlog #1 — create the Flutter
project and get a blank app running on a device.** Then #2 (Drift wiring),
#3 (both themes, both fonts, grain tile), #5 (DAO + tests), #6 (the diary).

## Open questions / waiting on Yash

1. **Project name.** "SafarSathi" is still a placeholder.
2. **Dedicated repo.** Docs are parked in `yash0304/ml`.
3. **Map tile provider.** MapTiler or Stadia. Needed before #24.
4. **Trip target.** Meghalaya is 1–5 October. A contacts-only v0.1 for that
   trip is plausible; the offline sync milestone is not.
5. **Archivo Narrow.** Look at it on a real phone before #3 bundles it.
6. **Grain tile.** 128×128, under 4 KB. Nobody has made it yet.
7. **`tel:` with an empty path** — confirm on a real device that it opens the
   Android dialer. Fallback is `ACTION_DIAL` over a platform channel, which
   would be the first platform-specific code in the project.
8. **Thumb index overflow.** Eleven categories do not fit a phone edge.
   Preference is to show only the categories the trip actually uses.

## Decided in chat but check DECISIONS.md logged them

Empty. All twenty-two decisions across both blocks are in DECISIONS.md.
