# Running SafarSathi on a phone

Everything so far has been verified by tests and by rendered images. Nothing
has executed on hardware. This is how to change that.

---

## What you need on the machine

| Thing | Notes |
|---|---|
| **Flutter stable 3.47 or newer** | The project is built against 3.47.3 / Dart 3.13.3, and `pubspec.yaml` requires Dart `^3.9.0`. |
| **Android Studio** | Only for the Android SDK and platform tools it installs. You do not have to write code in it. |
| **An Android phone with USB debugging on** | Settings → About phone → tap Build number seven times → Developer options → USB debugging. |

`flutter doctor` has to pass the Flutter and Android sections. It does **not**
need to pass iOS, web, Chrome or Visual Studio — none of them matter here.

```
flutter doctor
flutter doctor --android-licenses     # accept them all
```

## Get the code

The project lives in the `ml` repo for now, on the working branch, in the
`safarsathi/` folder.

```
cd C:\Users\Yash\Desktop\Github
git clone https://github.com/yash0304/ML.git safarsathi-repo
cd safarsathi-repo
git checkout claude/offline-retro-modern-app-design-r2n4ju
cd safarsathi
```

If it is already cloned, `git pull` on that branch instead.

## Run it

```
flutter pub get
flutter run
```

Plug the phone in first and accept the "Allow USB debugging?" prompt on its
screen. `flutter devices` should list it.

**The first build takes several minutes** — Gradle downloads a lot on the
first run and says very little while it does. Later builds are seconds.

Once it is running, `r` hot-reloads and `R` restarts.

## What you should see

A debug build seeds a demo trip with six entries the first time it opens,
because trip creation does not exist yet (#16). **Its numbers are deliberately
fake — +91 90000 000xx — so nothing here can be mistaken for a real number.**

---

## The four things only a phone can settle

Everything else is covered by 140 tests and by the goldens in
`safarsathi/test/goldens/`. These four are not, and cannot be.

### 1. Do the three fonts actually load?

The single most important check, because **no test can catch this**. The
widget test harness substitutes its own font, so a mismatch between
`pubspec.yaml` and `app_tokens.dart` passes every test and produces an app
that silently renders in the system face.

Compare against `test/goldens/diary_paper.png`:

- Section labels — `MEGHALAYA`, `PAGE 1 OF 1`, the index tabs — should be
  **geometric and slightly condensed** (Jost). Circular O, single-storey a.
- Names and notes should be a **plain newspaper grotesque** (Archivo).
- **Every phone number should look typed** — slab-ended, evenly spaced,
  unmistakably a typewriter (Courier Prime).

If the numbers do not look typed, stop and fix that before anything else.
Everything built since issue 1 sits on top of it.

### 2. Are the haptics felt?

- Tap a diary entry → light tick as it copies.
- Flip a thumb-index tab → selection tick.
- **Mark an entry confirmed → a medium thump, at the moment the stamp lands,
  not when the animation starts.** That timing is the whole point.

### 3. Does the dialer open?

This is the assumption the entire copy-first workflow rests on.

- Tap an entry → the toast appears → tap **OPEN DIALER**.
- The Android dialer should open **with an empty field**, ready to paste.
- Long-press the field and paste. The number should be there.

If instead you get *"No dialer app on this phone"*, then `tel:` with an empty
path does not work on your Android version. Fall back to an `ACTION_DIAL`
intent over a platform channel — the first platform-specific code in the
project — and log a decision about it.

Also try **Call** on the entry screen, which uses `tel:` with the number.

### 4. Is the night palette pleasant at 2am?

Switch the phone to dark mode. Contrast is asserted by test in both themes,
but a test cannot tell you whether it is comfortable in a dark room, which is
the actual use case.

---

## If something goes wrong

**Gradle fails on the first build.** Almost always a missing SDK licence. Run
`flutter doctor --android-licenses` and accept everything, then
`flutter clean` and try again.

**The phone does not appear in `flutter devices`.** Change the USB mode from
"Charging" to "File transfer" on the phone. On Windows you may also need the
OEM USB driver.

**Gradle fails on an AAR metadata check.** A plugin is compiling against an
older Android API than another plugin demands. Upgrade the named plugin; this
happened with `file_picker` 8, which compiled against 34 while
`flutter_plugin_android_lifecycle` required 36. Note that `flutter analyze`
and the test suite both pass regardless, because neither touches Gradle —
only a real build catches it.

**Actions do nothing.** Check `android/app/src/main/AndroidManifest.xml` still
has its `<queries>` block. On Android 11 and later an app cannot see which
other apps handle `tel:`, `sms:` or `https:` unless it declares them there,
and `url_launcher` just returns false. This was missing until 2026-09-12.

**The app opens empty.** The demo seed only runs in debug builds and only when
the database has no trip at all. Uninstall and reinstall to reseed.

---

## Getting an APK without installing anything

**This is the shortest path from a commit to something on your phone.**

`.github/workflows/safarsathi-apk.yml` builds a release APK on every push to
the working branch. GitHub's runners already have the Android SDK, which the
container this project is developed in does not — `dl.google.com` is blocked
by its egress policy, so no APK can be built there.

To get the file:

1. Open the repo's **Actions** tab on GitHub.
2. Click the newest **SafarSathi APK** run.
3. Download the **safarsathi-apk** artifact at the bottom. It is a zip with
   `app-release.apk` inside.
4. Copy the APK to the phone and open it. Android will ask you to allow
   installing from that source; that permission is per-app and you can revoke
   it afterwards.

The workflow runs `flutter analyze` and the full test suite before it builds,
so a run that produces an APK is one that passed everything.

**It is signed with Android's debug keys**, which is the Flutter template's
default. That is fine for a test build and not fine for anything you would
publish.

### On first open

A release build has no trip, because trip creation is #16. The app opens on a
**Create demo trip** button that fills the diary with six placeholder entries.
Their numbers are deliberately fake.

### Building one locally instead

If you do have the toolchain:

```
flutter build apk --release
```

The file lands at `build/app/outputs/flutter-apk/app-release.apk`.
