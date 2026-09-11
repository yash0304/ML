# HANDOFF — SafarSathi — 2026-09-11

> Overwrite this file at the end of every session. It must let a cold model
> (any model) resume in under 2 minutes.

## Where we are

- Current issue: **#1 Flutter project scaffold — not started.** Nothing has
  been compiled or run. The project still does not exist as a Flutter app.
- Backlog position: 0 / 48 issues done. Milestone 2.5 (#41–#48) is new this
  session and adds the visual system work.
- This session produced **design only**, no application code that runs.

## What happened this session

- Yash asked for an offline, highly interactive, retro-yet-modern design.
  Four scoping calls were made: the retro idiom applies everywhere **except**
  safety surfaces; the idiom is Indian road ephemera; interaction is tactile
  and cheap rather than full-motion; docs live at `docs/safarsathi/` in the
  `yash0304/ml` repo until SafarSathi gets its own.
- Wrote `DESIGN_VISUAL_v2.md` — the "Milestone" visual and interaction system.
  It supersedes DESIGN.md §5 only. Everything else in DESIGN.md, and all of
  §4, still outranks it.
- Wrote three drafted Dart files: `code/app_tokens.dart` (v2, two themes),
  `code/motion.dart`, `code/retro.dart`.
- Wrote `DIALER_RETRO_PATCH.md` — eleven exact edits to bring the drafted
  dialer to v2, with an acceptance list.
- Logged fourteen decisions in DECISIONS.md. Three of them supersede earlier
  lines: the pure-white surface, the single-typeface rule, and the single
  `caution` token.
- Amended backlog #3 (S → M) and added Milestone 2.5, issues #41–#48.

## In-flight state

- Files touched: everything under `docs/safarsathi/`. All docs complete.
  All three Dart files **drafted, never compiled** — same caveat as every
  other drafted file in this project.
- `code/dialer_screen.dart`, `code/contacts_dao.dart` and
  `code/app_tokens_v1_superseded.dart` are carried over untouched.
- Last known good: all committed on
  `claude/offline-retro-modern-app-design-r2n4ju`.

## Next action (this line starts the next session)

**Write `docs/ISSUE_1_Scaffold.md`, then do backlog #1 — create the Flutter
project and get a blank app running on a device.** Nothing else can start
until the project exists. Then #2 (Drift wiring) and #3 (tokens + both
themes + two fonts + grain tile).

Do not start Milestone 2.5 before #6. The visual system is deliberately
sequenced after the dialer renders, so there is something real to apply it to.

## Open questions / waiting on Yash

Carried from the previous handoff, all still open:

1. **Project name.** "SafarSathi" is still a placeholder.
2. **Dedicated repo.** The docs are parked in `yash0304/ml` for now. Needs a
   folder under `C:\Users\Yash\Desktop\Github\` and a real repo.
3. **Map tile provider.** MapTiler or Stadia. Needed before #24.
4. **Trip target.** Meghalaya is 1–5 October. Milestones 1–4 plus 2.5 are
   plausible by then at 5–10 hrs/week; Milestone 5 is not. Worth deciding
   whether v0.1 for that trip is contacts-only.

New this session:

5. **Archivo Narrow.** Look at it on a real phone before #3 bundles it. It is
   the one choice in v2 that is a matter of taste rather than measurement,
   and swapping it later means touching every stencil style.
6. **Grain tile.** Needs generating, 128×128, under 4 KB. Trivial, but it is
   an asset nobody has made yet.

## Decided in chat but check DECISIONS.md logged them

Empty. All fourteen decisions are in DECISIONS.md.
