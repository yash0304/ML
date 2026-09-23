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


## Getting the APK onto a phone

**https://github.com/yash0304/ML/releases/download/latest-apk/safarsathi.apk**

That link always points at the newest build and never changes. Open it on the
phone and install.

The Actions artifact still exists, but it is a zip that has to be found inside
a workflow run and extracted — which is a poor way to move a file onto a
handset. The release is the same APK, one tap.

The release notes on each build say two things worth reading before you
install: whether it was signed with the real key (if not, installing needs the
current app uninstalled, which deletes the database) and whether maps are on.

## The signing key — read this before installing a second build

**Android refuses to upgrade an app whose signing key changed.** It says
"App not installed as package conflicts with an existing package", and the
only way through is to uninstall the current copy — **which deletes the
database: every contact, every expense, the whole trip.**

Until 13 Sep this project signed release builds with `signingConfigs.debug`,
the Flutter template's default. On your own machine that is harmless, because
the debug keystore is generated once and reused. On a CI runner there is no
debug keystore, so Gradle **makes a new random one on every run** — which
meant every APK this project ever produced was signed differently from the
last, and each one could only be installed by throwing away the data in the
one before it.

### Make the key once

**In Android Studio** (no command line, and any project will do — a keystore
belongs to you, not to a project):

1. Build → **Generate Signed App Bundle / APK…**
2. Pick **APK**, then Next
3. Click **Create new…**
4. Fill in the New Key Store dialog:
   - **Key store path** — browse somewhere permanent, name it `safarsathi.jks`
   - **Password** and confirm — write it down somewhere real
   - **Alias**: `safarsathi`
   - **Key password** — the same one is fine
   - **Validity**: 25 years or more
   - **First and Last Name** — the only certificate field that matters
5. OK. The `.jks` is written at that point.
6. **Cancel the rest of the wizard.** Nothing needs building; the key was the
   whole errand.

**Or from a terminal**, if you have a JDK on the path:

```
keytool -genkey -v -keystore safarsathi.jks -keyalg RSA -keysize 2048 \
        -validity 10000 -alias safarsathi
```

Keep `safarsathi.jks` somewhere you will still have it in five years. **If it
is lost, no future build can ever upgrade an installed copy again** — the only
way back is an uninstall, and the data goes with it. It is gitignored, along
with `android/key.properties` and anything else ending `.jks` or `.keystore`.

### Locally

Create `safarsathi/android/key.properties`:

```
storeFile=/absolute/path/to/safarsathi.jks
storePassword=…
keyAlias=safarsathi
keyPassword=…
```

Builds without that file still work. They fall back to debug signing and warn
in the Gradle log, so a contributor without the keystore is not blocked.

### In CI

Four repository secrets, under **Settings → Secrets and variables → Actions**:

| Secret | Value |
|---|---|
| `ANDROID_KEYSTORE_BASE64` | the keystore as base64 — see below |
| `ANDROID_KEYSTORE_PASSWORD` | the store password |
| `ANDROID_KEY_ALIAS` | `safarsathi` |
| `ANDROID_KEY_PASSWORD` | the key password |

The workflow prints which signing it used, so a missing secret is visible in
the build log rather than only on the phone.

**Turning the keystore into base64.** On Linux or macOS:

```
base64 -w0 safarsathi.jks
```

**On Windows**, `base64` does not exist. In PowerShell:

```powershell
[Convert]::ToBase64String([IO.File]::ReadAllBytes("C:\path\to\safarsathi.jks")) | Set-Clipboard
```

That puts the whole string on the clipboard, ready to paste straight into the
secret. It is one very long line with no newlines, which is what the workflow
expects — a wrapped version will not decode.

### One last uninstall

The build that first carries the real key is signed differently from whatever
is on the phone now, so Android refuses it with the same "package conflicts"
message — **the current copy has to be uninstalled once**. Every build after
it goes over the top and keeps the data.

Uninstalling deletes the database and the downloaded map tiles with it, so
that one time, in this order:

1. **More → Backup and restore → With the map.** That writes a zip holding
   the data *and* the tiles. The plain backup above it leaves the tiles out,
   and re-downloading them needs WiFi and patience.
2. **Share the file off the phone** — Drive, a chat to yourself, anywhere but
   the app's own storage, which the uninstall takes too.
3. Uninstall, install the new APK, restore the file.

If the installed build is old enough not to offer "With the map", take the
plain backup and plan to re-download the map afterwards.

### "Package conflicts" after the release key was already set up

Every build since 18 Sep 2026 is signed with the same key:

```
CN=safar, OU=sathi · created 18 Sep 2026 11:46 GMT
SHA-256 CAF8E28961F14D1C3F656CDA0D6077F6424910C523401D1B32AC10A058A04F43
```

The build log's "Report how this build was signed" step prints the
certificate actually inside each APK. If it matches the line above, the new
APK is fine and the conflict is with **what is on the phone** — almost always
an older `safarsathi (n).apk` from before that date, tapped from Downloads by
mistake or installed at some point. Before anything else:

1. **Do not uninstall.** That deletes the diary and the downloaded map.
2. Files → Downloads → delete every `safarsathi*.apk`.
3. Download the latest build again and open it from the browser's own
   "download complete" notice, not from the file manager.

Each build now carries its own version, `0.1.<run number>`, shown in the
release notes and in Settings → Apps → SafarSathi — so "which build is on
this phone?" has an answer.

If it still conflicts, the installed copy itself is signed with some other
key and the only way forward is an uninstall — so first take **More → Backup
and restore → With the map**, move the file off the phone, and restore it
after installing.

## The MapTiler key

Maps need a key. **It is never committed.** A build without one still runs —
the map screens disable themselves and say so, which is the designed
behaviour, not a failure.

There are three ways to supply one and the repository is none of them.

### On the phone, in the app — the one most people want

**Settings → Map key.** Paste the key, tap Save, then More → Map to download.

This is the only route that works on an APK somebody else built. It needs no
secret, no rebuild and no access to CI; the key is stored in the app's own
database on that phone and is sent to MapTiler only while a map is actually
downloading. A key typed here **overrides** one baked in at build time, so it
is also how you rotate a key without producing a new build.

Tiles already on disk survive a key change: the on-disk cache is keyed on the
provider and style, never on the key.

### On your own machine

Create `safarsathi/maptiler.json`, which is gitignored:

```json
{ "MAPTILER_KEY": "your key" }
```

Then:

```
flutter run --dart-define-from-file=maptiler.json
flutter build apk --release --dart-define-from-file=maptiler.json
```

`maptiler.example.json` is committed as a template. Copy it, do not edit it.

### In CI

Add a repository secret named `MAPTILER_KEY` under **Settings → Secrets and
variables → Actions**. The workflow passes it through `--dart-define` and
prints whether maps ended up enabled, so a missing secret is visible in the
build log rather than only on the phone.

### Restrict the key

Whether it is baked in by `--dart-define` or typed into Settings, the key ends
up on the phone, as it does with every mobile map SDK. It is not a secret from
whoever holds the handset; what these routes avoid is the key living in git
history forever.

**The control that actually matters is in the MapTiler dashboard: restrict the
key to this app's package name.** Do that now rather than later.

---

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
