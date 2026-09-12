# SafarSathi — docs

Offline-first travel companion for multi-stop road trips. Pre-code: the
schema and one screen are drafted, nothing has been compiled or run.

Parked in this repo under `docs/safarsathi/` until SafarSathi gets its own.
Nothing here relates to the ML toolkit at the repo root.

## Read in this order

| File | What it is |
|---|---|
| `PROJECT_RUNDOWN.md` | The full record of the design conversation. Read once. |
| `DESIGN.md` | Architecture, data model, offline strategy, **trust tiers (§4)**. |
| `DESIGN_VISUAL_v2.md` | The "Milestone" visual and interaction system. Supersedes DESIGN.md §5 only. |
| `SCREENS.md` | Six screen specifications, tight enough to build from. Start here for what the app looks like. |
| `DECISIONS.md` | Dated decision log, newest first. Never delete a superseded line. |
| `BACKLOG_v0.1.md` | 48 issues with checkboxes. Checked = done. |
| `HANDOFF.md` | Session-to-session baton. Read this first if you are resuming. |
| `RUNNING.md` | How to run it on a phone, and the four things only hardware can settle. |
| `DIALER_RETRO_PATCH.md` | Eleven exact edits taking the drafted dialer to v2. |

## The two things a new session must not break

1. **Trust tiering.** An imported spreadsheet row is not a verified number.
   PROJECT_RUNDOWN §4.4 and DESIGN.md §4.
2. **The exemption rule.** Ephemera never carries trust; the emergency screen
   takes no retro treatment at all. DESIGN_VISUAL_v2.md §0.
3. **Copy is the primary action**, everywhere except the emergency screen,
   where the tap calls. SCREENS.md §0 and §3.

## Code

`code/` holds drafted Dart. **None of it has been compiled.** Expect import
paths, generated Drift classes and `build_runner` output to need fixing on
first contact.

| File | State |
|---|---|
| `app_tokens.dart` | v2. Two themes, type roles, geometry, `ThemeData`. |
| `motion.dart` | v2. Durations, curves, haptics, `PressScale`. |
| `retro.dart` | v2. The seven retro primitives. |
| `dialer_screen.dart` | v1, unpatched. Apply `DIALER_RETRO_PATCH.md`. |
| `contacts_dao.dart` | v1, unchanged. |
| `app_tokens_v1_superseded.dart` | Kept for reference only. Do not import. |
