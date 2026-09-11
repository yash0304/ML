# HANDOFF — SafarSathi — 2026-09-11 (end of session)

> Overwrite this file at the end of every session. It must let a cold model
> (any model) resume in under 2 minutes.

## Where we are

- **Issue #1 is done except for the device check.** The Flutter project at
  `safarsathi/` analyses clean and passes 11 tests on Flutter 3.47.3 /
  Dart 3.13.3. It has never run on a phone.
- Backlog position: 1 / 52 done. Next is **#2, Drift database wiring.**
- Design is complete and prototyped: eleven screens, both themes, full
  decision log.

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

**Not verified, and not verifiable without a phone:**

- **Whether the fonts actually load.** The widget test harness substitutes its
  own font, so no automated check will ever catch a family-name mismatch.
  If the stencil labels do not look condensed on device, `pubspec.yaml` and
  `app_tokens.dart` disagree.
- Whether the haptics fire.
- Whether the night palette is pleasant at 2am, as opposed to merely passing.
- Whether the grain reads as texture or as dirt.

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

**Write `docs/ISSUE_2_Database.md`, then do backlog #2 — Drift wiring.** All
fifteen tables from the drafted schema files, `build_runner` run, generated
code compiling, DB file created on device. Then #3 is already largely done by
#1, so confirm and tick it; then #5 (ContactsDao plus ordering tests) and #6
(the diary screen).

Before anything else, run `flutter run` on the phone and settle the four
unverified items above. If the fonts are wrong, fix that first — everything
after is built on it.

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
8. **Checklist regeneration** must not discard manual edits — needs a
   "user touched this" flag, decided at #29.

## Decided in chat but check DECISIONS.md logged them

Empty. All decisions across this session are in DECISIONS.md.
