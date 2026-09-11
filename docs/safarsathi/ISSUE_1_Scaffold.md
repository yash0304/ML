# Issue #1 — Flutter project scaffold

**Size:** S · **Depends on:** nothing · **Blocks:** everything

Nothing in this project has ever been compiled. The point of this issue is not
features — it is to get a Flutter project that builds, analyses clean, and runs
on Yash's phone, so that from #2 onward every drafted file can be verified
rather than assumed.

---

## Where the project lives

`safarsathi/` at the root of the `yash0304/ml` repo, beside `docs/safarsathi/`.
Both move together when SafarSathi gets its own repo. The app is not
documentation, so it does not belong under `docs/`.

---

## Prerequisites on Yash's machine

| Thing | Why |
|---|---|
| Flutter stable 3.47 or newer | The SDK this scaffold was generated against |
| Android SDK + platform tools | `flutter doctor` must pass the Android section |
| A phone with USB debugging on | The only real verification for #1 |

`flutter doctor` does not need to be green on iOS, web or desktop. Android
alone is enough for this project.

---

## Steps

### 1. Create the project

```bash
cd <repo root>
flutter create --org com.yashmodi --project-name safarsathi \
  --platforms android,ios safarsathi
```

`--platforms android,ios` keeps the web, Linux, macOS and Windows runners out
of the tree. They are dead weight for an app whose whole premise is a phone
with no signal.

### 2. Folder structure

```
safarsathi/lib/
├── main.dart
├── app.dart                    # MaterialApp, themes, routes
├── core/
│   ├── theme/
│   │   ├── app_tokens.dart     # from docs/safarsathi/code/
│   │   └── motion.dart         # from docs/safarsathi/code/
│   ├── widgets/
│   │   └── retro.dart          # from docs/safarsathi/code/
│   └── database/               # #2 fills this
└── features/
    ├── contacts/
    │   ├── data/
    │   └── presentation/
    ├── trips/
    └── emergency/
```

`core/` is anything more than one feature uses. `features/` is one folder per
domain, each with `data/` and `presentation/`. There is deliberately **no
repository layer** between DAO and widget — add one when a second consumer of
the same data appears, not before. See DECISIONS.md.

### 3. Dependencies

```yaml
dependencies:
  drift:                    # local database — the only source of truth
  drift_flutter:            # opens the DB in the app's documents directory
  sqlite3_flutter_libs:     # bundles SQLite so behaviour matches across OEMs
  path_provider:
  url_launcher:             # tel:, sms:, wa.me — no dialer permission needed
  file_picker:              # sheet import
  csv:
  excel:
  phone_numbers_parser:     # E.164 normalisation, pure Dart

dev_dependencies:
  drift_dev:
  build_runner:
  flutter_lints:
```

**Note on the phone number package.** The backlog said
`libphonenumber_plugin`, which wraps the native library over a platform
channel. `phone_numbers_parser` is pure Dart, needs no channel, and runs
offline with no platform dependency — a better fit for this app. If it is
swapped, log the decision.

### 4. Fonts and assets

Bundle **Inter** and **Archivo Narrow**, both SIL OFL, from Google Fonts.
Download the static weights rather than using the `google_fonts` package,
which fetches at runtime — unacceptable in an app defined by having no network.

```
safarsathi/assets/
├── fonts/
│   ├── Inter-Regular.ttf      Inter-Medium.ttf      Inter-SemiBold.ttf
│   └── ArchivoNarrow-Bold.ttf
└── texture/
    └── grain_128.png          # 128×128, under 4 KB, 3% noise
```

Declare both families in `pubspec.yaml` under `fonts:`, matching the family
names `Inter` and `ArchivoNarrow` that `app_tokens.dart` expects. A mismatch
here fails silently into the system font, which is the single most likely way
to get a scaffold that "works" but looks wrong.

### 5. Lints

Keep `flutter_lints` and add `prefer_const_constructors` plus
`require_trailing_commas`. Do not disable `avoid_print`.

### 6. Theme wiring

`app.dart` builds a `MaterialApp` with `theme: AppTokens.light`,
`darkTheme: AppTokens.dark`, and `themeMode: ThemeMode.system`. A manual
override lands in Settings at #35 — a phone in a pocket does not know it is
night in a valley.

### 7. Smoke screen

A throwaway screen that renders one of each: a `StencilLabel`, a `TicketCard`,
a `MilestoneMarker`, a `StampBadge`, a `HazardStripe`, a phone number in
`numberStyle`, and a `PressScale` button. It exists to prove the tokens, both
fonts, the grain tile and the haptics all actually work on a device, and it is
deleted at #6 when the diary replaces it.

### 8. Run it

```bash
cd safarsathi
flutter pub get
flutter analyze
flutter test
flutter run
```

---

## Acceptance criteria

- [ ] `flutter analyze` reports **no issues**.
- [ ] `flutter test` passes.
- [ ] `flutter run` installs and opens on a real Android phone.
- [ ] The smoke screen renders all seven elements without overflow.
- [ ] **Both fonts are actually loading.** The stencil labels must look
      condensed. If they look like the body font, the family name in
      `pubspec.yaml` does not match `app_tokens.dart` and everything after this
      will be built on a silent fallback.
- [ ] Switching the phone to dark mode swaps the app to the Lamp palette, and
      nothing becomes unreadable.
- [ ] Pressing the smoke-screen button produces a scale response **and a
      haptic you can feel**.
- [ ] The grain is visible as texture when you look for it and invisible when
      you do not.

---

## Gotchas

- **A missing font fails silently.** Check it by eye before moving on.
- `sqlite3_flutter_libs` adds several MB to the APK. That is the correct
  trade: it removes a class of OEM-specific SQLite differences that would
  otherwise appear only on somebody else's phone.
- Do not add a state management library. StreamBuilder over Drift streams is
  the decision until #25.
- Do not commit `safarsathi/build/` or `.dart_tool/`.
