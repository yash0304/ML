# DECISIONS — SafarSathi

Newest first. Format: `YYYY-MM-DD — [AREA] Decision. (Why.)`
Never delete a superseded decision — add a new dated line above it.

---

2026-09-18 — [BACKUP] **The full backup is written and read as a stream, never assembled in memory, and it leaves the phone by a share sheet rather than the save dialog.** (`FilePicker.saveFile` takes the whole file as a `Uint8List`; a 350 MB tile cache passed that way is how a backup becomes a crash on the phone that most needed one. `ZipFileEncoder` writes entry by entry to disk, `InputFileStream` reads it back the same way, and `share_plus` hands the finished file over by path. Tiles are STORED, not deflated — they are already-compressed PNG and WebP, so deflating spends minutes of phone CPU to save nothing.)

2026-09-18 — [BACKUP] After a full restore the tile index is rebuilt by walking the directory, not trusted from the archive. (A truncated or partly-extracted archive would otherwise leave the cache screen reporting thousands of tiles over a directory that does not have them — the worst possible answer to "what have I got offline".)

2026-09-18 — [BACKUP] A full restore puts the database back first and the tiles second. (If the tiles fail halfway — the phone fills up, the file is damaged — the trip, the numbers and the checklist are already back, and the map is the half that can be fetched again. The other order loses the irreplaceable part to protect the replaceable one.)

2026-09-18 — [MAP] **Tiles are fetched as WebP, and the reader accepts both WebP and PNG.** (MapTiler serves the same tiles either way and WebP is roughly a third the size — a Meghalaya corridor at 346 MB becomes something nearer 110 MB, which is the difference between a re-download somebody does and one they put off. Verified against the live endpoint before shipping. The format is deliberately NOT part of the provider id and the reader tries both extensions, because folding it in would have stranded every tile already on disk and turned a working offline map blank a fortnight before the trip — the exact cost the change exists to avoid.)

2026-09-18 — [UI] A leg with no kilometres now says which of its two stops has no location, rather than showing an empty space. (From the phone: five legs, and the two touching Sohrra were blank. Both were blocked on the same thing — that stop had never been located — and the screen said neither which stop nor that it mattered. "Not downloaded yet" and "Sohrra has no location yet" need different things done about them, so they read differently: the second is in caution, because only the user can clear it.)

2026-09-18 — [BRAND] The launcher icon is rail, road and air on one line, and it does NOT use the app's green. (The mark is seen beside thirty other icons on a home screen, so it is allowed to shout where the app itself stays quiet: violet falling to magenta across the diagonal, marks in the app's own paper so the two still belong together. The merge is the line itself — sleepers on the left half make it a railway, the clean right half makes it a road, and the plane lifts off the end. Drawn by `tool/make_icon.py` at 4x and downsampled, so it is reproducible and every density comes from one source rather than five hand-edited files.)

2026-09-18 — [BRAND] The home-screen label is "SafarSathi", not "safarsathi". (The Flutter template lowercases it from the project directory name.)

2026-09-14 — [DEMO] The demo trip seeds one traveller — you — and no companions. (It used to seed Ankit and Priya so the settle-up had something to settle, which meant anybody opening the app for the first time found two strangers in their ledger and had to work out they were not real. Adding a traveller is one tap on Money → Travellers, and doing it yourself is a better way to learn that screen than finding it pre-filled.)

2026-09-14 — [TEST] `EntryScreen` takes an injectable clock. (Confirming a contact stamped `DateTime.now()` into the golden image, so `entry_confirmed.png` failed on every day after the one it was rendered on. A golden that depends on the calendar is a golden that cries wolf, and the project already froze time this way on the stop screen.)

2026-09-13 — [BACKUP] **A restore brings back `callConfirmed` and the trust tier exactly as they were. This is a deliberate, reasoned exception to the rule that bulk entry never produces a confirmed contact.** (That rule exists because a number in a spreadsheet is unverified, and `insertBatch` enforces it in the DAO so no caller can get it wrong. A backup is not a spreadsheet — it is this app's own record of calls the user made. Dropping confirmations on restore would force re-calling twenty places after replacing a phone, which would make backups useless and push people not to take them: a worse outcome for safety than the risk it avoids. The file is editable JSON and could come from anyone, so the restore screen states how many numbers the file claims were confirmed and says to restore only a backup this app made. No pretence of cryptographic protection — any signature would need its key in the APK, which protects nothing and would only make the claim look stronger than it is.)

2026-09-13 — [BACKUP] **Emergency helplines are never carried in a backup.** (They are seeded by the app from sources recorded in `sourceNote`, and the whole trust model rests on that. A backup that carried them would be a way to put a number on the emergency screen by editing a JSON file, or by sending somebody a "backup" to restore. They re-seed on first launch, so excluding them costs nothing. There was nothing to weigh here.)

2026-09-13 — [BACKUP] A backup carries what you typed and not what came off the network — the same line `clearTripCache` already draws. (POIs, forecasts, tiles and the legs' route cache are all re-downloadable, and tiles alone would make the file tens of megabytes. A restored leg reads as not downloaded, which is the truth once the places that came with it are gone.)

2026-09-13 — [BACKUP] The MapTiler key is excluded from the backup. (A backup is a file people mail to themselves and copy between handsets; a key inside one travels further than anybody intends. Re-entering it is a single paste. A restore also never clears a key already on the receiving phone.)

2026-09-13 — [BACKUP] Restore replaces; it never merges. (Merging would mean duplicate detection across nine tables with foreign keys between them, and a half-merged database is worse than either outcome. The confirmation shows what is on the phone now beside what the file holds, because a restore landing on top of a newer trip is the case worth being careful about.)

2026-09-13 — [BACKUP] A backup from a newer schema is refused outright; an older one is accepted. (Columns this build has never heard of cannot be restored faithfully, and a silent partial restore is the worst outcome available. Older is safe because every migration in this project has been additive.)

2026-09-13 — [BACKUP] The format is plain indented JSON rather than anything compact. (If a future version of this app cannot open the file, somebody can still read their own phone numbers out of it in a text editor. For a backup, that is the real guarantee.)

2026-09-13 — [ANDROID] **Release builds are signed with a real keystore supplied by secret, not with `signingConfigs.debug`.** (The Flutter template's default is harmless on a developer machine, where the debug keystore is generated once and reused, and silently destructive from CI, where no debug keystore exists so Gradle makes a new random one every run. Android refuses to upgrade an app whose signature changed, so every APK this project produced could only be installed by uninstalling the previous one — which deletes the database. Yash hit it as "App not installed as package conflicts with an existing package" on the sixth build, and it had almost certainly been silently costing him his data on the five before it. A build with no keystore still succeeds and still falls back to debug signing, but both Gradle and the workflow now say so out loud.)

2026-09-13 — [ANDROID] The CI signing step is guarded in the shell, not by a step-level `if` on the secrets context. (That context is not dependably available to `if` expressions, and a silently-skipped signing step is the exact failure the change exists to prevent.)

2026-09-13 — [SYNC] **Thrown objects are classified into a sentence before they reach the sync screen; `TripSync._readable` is replaced by `describeSyncError`.** (`_readable` carried a comment saying it turned an exception "into something a person can act on" and then truncated the string to 160 characters. On Yash's phone that rendered seven rows of `ClientException with SocketException: Failed host lookup: 'router.project-osrm.org' (OS Error: No address associated with hostname, errno = 7), uri=https://r…`, each cut off mid-URL. A comment describing behaviour the code does not have is worse than no comment: it stops anybody looking.)

2026-09-13 — [SYNC] A run where every task failed never says "Everything else went through". (It said exactly that under seven failures out of seven. `summariseSyncFailures` now leads with one sentence naming the actual cause and the actual fix, and the per-task rows sit under it — seven repetitions of one DNS failure is one fact rendered seven times.)

2026-09-13 — [SYNC] A failed DNS lookup and a dropped connection are reported differently. (`Failed host lookup` means retrying on the same network cannot help; `Software caused connection abort` means it can. Both appeared on the same screen and the app treated them identically. `SyncError.needsDifferentNetwork` is the distinction.)

2026-09-13 — [SYNC] A missing map key is never reported as a network problem. (It arrives in the same failure list as the DNS errors, and telling somebody to check their WiFi about it sends them to entirely the wrong place. The summary points at Settings → Map key instead.)

2026-09-13 — [NET] Every outbound request now carries a client-side timeout; there were none. (`http.get` with no timeout waits as long as the socket stays open, which on a WiFi that has stopped forwarding packets is forever — and sync runs its tasks in sequence, so one hung request stalls the whole download behind it while the button still says "Downloading…". Overpass gets 90s, deliberately longer than the 60s server-side timeout in its own query, so a client cutoff never kills a query that was about to return.)

2026-09-13 — [ANDROID] The manifest's `<queries>` comment no longer claims the release build ships without INTERNET. (It said so directly underneath the `uses-permission` line that grants it — left over from before #21 narrowed the offline claim. Two paragraphs in one file asserting opposite things is how somebody later trusts the wrong one.)

2026-09-13 — [CONTACTS] **Multi-add writes through `ContactsDao.insertBatch`, the same function file import uses, rather than its own insert loop.** (`insertBatch` forces `userEntered` and `callConfirmed = false` on every row whatever the caller passes, so there is no argument the screen could get wrong. A second write path that re-implemented the guard is exactly how the invariant erodes — and that guard has already caught one real bug in this project. It also means a typed session is an `ImportBatch`, so it appears in the history and can be rolled back wholesale, which matters because the likeliest mistake is typing ten rows against the wrong trip.)

2026-09-13 — [CONTACTS] The import history says "added" and "Undo these entries" for a typed batch, via `ImportBatchSummary.wasTyped`. (Reusing the batch machinery must not make the app tell somebody it "imported" numbers they sat and typed. The check lives on the summary so no call site compares the label string.)

2026-09-13 — [CONTACTS] A new multi-add row inherits the category of the row above it. (Most bulk sessions are one kind at a time — "here are my three hotels". Setting it once and having it stick is the fast path; overriding a row costs one tap. Nothing else on the sheet carries downward.)

2026-09-13 — [CONTACTS] Multi-add takes a name, a number and a category, and nothing else. (Stop, note and WhatsApp belong to the entry form, which owns one contact properly. A sheet is for getting digits down; the details are edited later on an entry that now exists. Declined: parsing a pasted block of text — guessing at that is how a phone number becomes a note, and the CSV import already exists for bulk.)

2026-09-13 — [UI] Multi-add's fields are transparent with a hairline rule, not the app's filled `stone` inputs. (The shared `inputDecorationTheme` is right for the entry form's three or four fields and wrong for eight stacked pairs, which merge into one grey block and lose the ruled-notebook idiom the diary is built on.)

2026-09-13 — [UI] **`StampBadge` is mounted unconditionally and collapses to zero width until it lands; DIALER_RETRO_PATCH.md edit 8, which mounts it only when already confirmed, is wrong and superseded.** (`if (confirmed) StampBadge(landed: confirmed)` reads correctly and can never animate: the widget only ever exists in the landed state, so it never sees the false → true transition that fires the stamp and the haptic. The patch document has spelled it that way since before there was code. The badge now owns its own space via `Align(widthFactor:)`, which is what makes unconditional mounting free.)

2026-09-13 — [UI] Every diary row carries `ValueKey(contact.id)`. (`ListView` recycles elements. Without a key a confirmed row's element gets handed an unconfirmed contact and back, `StampBadge` reads that as a confirmation, and the phone stamps and buzzes at somebody who is only scrolling. Pinned by a test that flings a list of thirty rows and asserts the haptic channel stayed silent.)

2026-09-13 — [UI] Both trust markers leave the tree rather than fading to zero opacity. (An invisible widget still has semantics. A dot at opacity 0 keeps announcing "Not confirmed yet" on a confirmed contact, and a stamp at opacity 0 announces "Confirmed" on one nobody has confirmed. Same lie, both directions.)

2026-09-13 — [UI] Swipe never dismisses: `confirmDismiss` returns false on both sides and the row springs back. (Losing a phone number by accident, on a road, offline, is not recoverable in the way that matters. Swipe adds reach; the whole-row tap keeps the commonest action needing no aiming, per 2026-09-11.)

2026-09-13 — [UI] The over-scroll cache stamp reads `OverscrollNotification`, not `metrics.pixels`. (Android's clamping physics never lets `pixels` go negative — it reports the excess through the notification and paints a glow. Reading pixels alone would have shipped a feature that worked in iOS simulators and did nothing at all on the phone this app is for.)

2026-09-13 — [UI] Over-scroll shows what is cached instead of a pull-to-refresh spinner. (The spinner would promise the one thing the app is built never to do. A date and a count is the honest answer to what somebody pulling down actually wants to know.)

2026-09-13 — [MAP] **The MapTiler key can be typed into Settings and stored in the app's database, and a typed key overrides the build-time one.** (The build-time-only route was not enough and the gap was real: Yash had a key and an APK off CI with no repository secret configured, and therefore nowhere to put it — maps were simply off with no way in. `--dart-define` cannot help someone who did not produce the build. The key still never enters the repository, which was the actual constraint; it now has two ways in rather than one. It is not a password — it identifies the account to MapTiler and is visible to whoever holds the phone, exactly as in every mobile map SDK — so the control that matters is still restricting it to the app's package name in the MapTiler dashboard.)

2026-09-13 — [MAP] The tile cache id depends on provider and style, never on the key. (A key can be rotated; a downloaded trip must survive that. Tiles already on disk are still found after the key changes.)

2026-09-13 — [MAP] The provider is resolved when the map or sync screen opens, not held in a field. (So a key typed into Settings works immediately, without restarting the app — which is the first thing anyone tries after saving one.)

2026-09-13 — [WEATHER] **The staleness caution boundary is three days, not seven. #26 shipped seven and that is corrected here.** (SCREENS.md §10 has always said "under three days it is muted; past three days it turns caution". Seven was laxer than the app's own specification, and wrong on the facts for the trip it was built for: an October forecast for Meghalaya is a different season after five days. Bands are now fresh under a day, ageing one to three, stale at three and over, and the wording moved with them — "Over a week old" became "More than three days old". A stale forecast that looks current is the failure mode the whole screen exists to prevent, so the laxer number was the one thing it could not afford.)

2026-09-13 — [TRIPS] Editing activity tags on the stop screen calls `regeneratePackList`, and the screen says out loud that it does. (The chips are editable *because* they drive the checklist. A row of chips that silently rebuilds a list elsewhere is a surprise; a row of chips that claims to and does not is a lie. Hand edits survive because #29 keys generated items on `generatorKey` rather than on the label.)

2026-09-13 — [TRIPS] The stop screen's "diary entries" chevron opens the diary already narrowed to that stop. (`DiaryScreen` gained a `startStopScoped` flag for it. The count said "here"; opening the whole-trip list would answer a different question from the one the count asked.)

2026-09-13 — [TRIPS] "Places on the roads either side" is the union over every leg touching the stop, and the label says so. (A stop has no corridor of its own — a corridor belongs to a leg. Calling it "nearby" would imply a radius the app never computes.)

2026-09-13 — [TRIPS] The leg screen reports no per-leg tile figure, and says why. (Adjacent legs share the ground between them, so a per-leg number would double-count. The honest total lives in Settings, which is where the delete button is anyway.)

2026-09-13 — [TRIPS] `watchLegTransport` is a separate stream from `watchLegDiscovery`. (The two halves of a leg change for different reasons — one when a person edits a field, the other when a download lands. Merging them would rebuild the corridor list on every keystroke in the form.)

2026-09-13 — [TRIPS] Tapping a stop in the itinerary opens the stop, not the form; the form is one pencil further in. (The itinerary is still the planning surface, but a stop now has somewhere to be. `ItineraryScreen.onOpen` is optional, so the older behaviour is what happens when nothing is wired to it.)

2026-09-13 — [DISCOVERY] **Saving a corridor place lands the contact as `communityOsm`, NOT `userEntered`. The backlog line specifying `userEntered` is wrong and is superseded.** (Both tiers carry the amber dot, so the trust outcome is identical — but the entry screen reads provenance out loud: `userEntered` says "Typed by you · not confirmed" and `communityOsm` says "From open map data · nobody has checked it". Saving an OSM number as `userEntered` would make the app tell the user they typed a number a stranger put in a public wiki. In an app whose entire premise is knowing where a number came from, that is the one thing it must not do. The path out is unchanged: call it, mark it confirmed, and it becomes `userVerified` like anything else.)

2026-09-13 — [DISCOVERY] Corridor places are ordered by distance ALONG the route, never by distance from the user. ("Coming up in 12 km" is the useful sentence while moving. "0.2 km away" treats the road as a plane, and a place 200 m off across a gorge is an hour of driving. Each place renders on a milestone marker, the same idiom the Trip screen uses — a milestone is a thing you pass.)

2026-09-13 — [DISCOVERY] The category chips offer only the categories the leg actually contains. (A chip that matches nothing is a filter that only disappoints. `LegDiscovery.categoriesPresent` derives them from the data.)

2026-09-13 — [DISCOVERY] An unsynced leg and a genuinely empty one read differently. ("This leg has not been downloaded yet" sends you to the sync screen; "OpenStreetMap simply has nothing tagged here" tells you to stop looking. Confusing them sends the user to the wrong place.)

2026-09-13 — [DISCOVERY] The Google Maps handoff says it needs signal BEFORE the tap, and it is the only action in the app that does. (Reviews and photos are not something an offline app can carry, and deep-linking out costs nothing and needs no key. A dead tap on a mountain road with no explanation is worse than no button.)

2026-09-13 — [DISCOVERY] The detail screen quotes the provenance in the exact words the entry screen uses. (Two screens describing the same tier differently is how a trust system stops being believed.)

2026-09-13 — [ARCH] **Riverpod is not being introduced, and the backlog line saying #25 would justify it is superseded.** (What the sync screen holds is a plan and an index into it, for the lifetime of one route — a `StatefulWidget` over a stream does that. A state-management library here would mean a dependency, a second way of doing things beside the `StreamBuilder` every other screen uses, and a migration decision for the twelve screens already working without one. Revisit if a later feature needs genuinely shared cross-screen state; superseded, not forgotten.)

2026-09-13 — [SYNC] Route and places are ONE task, not two. (`CorridorSync.syncLeg` routes the leg and queries its box in a single call and writes both in one transaction. Two progress rows would report progress corresponding to no work — the first draft did exactly that, with a no-op branch to make the second row "succeed".)

2026-09-13 — [SYNC] Every leg's corridor runs before the map does. (Tiles derive their box from the route polyline; with no polyline the box falls back to the straight line between stops, which in the Meghalaya hills is a different valley. Tiles are also last because they are by far the largest download, and failing there should not cost the small useful things.)

2026-09-13 — [SYNC] **One failure does not stop the run.** Each task is independent; a failure is recorded, stepped over, and listed afterwards with its reason. (Deliberately different from the tile downloader, where a rejected key aborts everything because it affects every remaining tile. Here Overpass being busy must not cost the map, and a stop outside the forecast range must not cost the route. A sync that silently does 60% is worse than one that says which 40% is missing.)

2026-09-13 — [SYNC] Resumability is by construction, not by a cursor. (Every underlying step already skips work already done: tiles check the disk, routes and places overwrite cheaply, weather replaces. Re-running the whole plan costs only the missing parts, and it cannot get out of step with what is actually on disk the way a stored position can.)

2026-09-13 — [SYNC] Progress names the item, not only a percentage. ("Route and what is along it · Shillong → Cherrapunji" is something a person can wait through. "43%" is not.)

2026-09-13 — [SYNC] Only the tiles are size-estimated, and the estimate says so. (A route is a few kilobytes, a place list tens, a forecast less. Itemising three negligible things is noise; "about 24 MB, nearly all of it map" is the true shape of the download.)

2026-09-13 — [UI] Counts on the sync screen are pluralised. ("1 legs" on the last screen someone sees before leaving reads as sloppiness, and a one-leg trip is common. Caught by rendering the golden.)

2026-09-13 — [MAP] **MapTiler is the tile provider.** Chosen by Yash, who holds the account. (Resolves the open question that had blocked #24 since the milestone was written. The standard OSM tile server was never an option: its usage policy forbids the bulk downloading that is the entire feature.)

2026-09-13 — [MAP] The provider sits behind a `MapTileProvider` interface and nothing outside `tile_provider.dart` names MapTiler. Attribution is part of the interface, not an afterthought. (Swapping to Stadia later is a new implementation and one line. Attribution is in the interface because every provider's terms require it and a map rendering without it is a licence violation — a new implementation should not be able to compile without supplying one.)

2026-09-13 — [MAP] The API key arrives through `String.fromEnvironment`, filled by `--dart-define`: a gitignored `maptiler.json` locally, a GitHub Actions secret in CI, and nothing at all otherwise. **A build with no key runs, disables the map, and says so.** (A blank grey rectangle is indistinguishable from a bug. The key is a compile-time constant so it is baked into the binary, as with every mobile map SDK — that is not a secret from whoever holds the APK. What this protects is the key never entering git history. The control that actually matters is restricting the key to the app's package name in the MapTiler dashboard, and that is written into the docs rather than assumed.)

2026-09-13 — [MAP] **Not flutter_map_tile_caching.** The tile cache is written here instead. (FMTC stores tiles in ObjectBox — a second native database engine beside SQLite. This app's stated invariant is that SQLite is the only source of truth; a cache is not worth another native build to break and another migration story. Slippy-map arithmetic is forty lines, tiles are files on disk, and the index is one Drift table. The same trade this project already took for Levenshtein, the polyline codec and `combineLatest2`.)

2026-09-13 — [MAP] **The rendering path has no HTTP client at all.** Not a fallback, not a timeout, not a retry. A tile that was not downloaded renders as a transparent pixel. (The absence IS the guarantee. Any code path that could reach the network is one that will, on a mountain road, with no signal, while the user waits.)

2026-09-13 — [MAP] Zoom 12 to 15, and the tile count is shown before anything downloads. (Tile count grows as 4^z: the same corridor at zoom 18 is over a hundred times the tiles of zoom 14, for detail nobody reads at a dhaba. The estimate says it is an estimate, because a wrong guess about size on a metered connection is not forgiven.)

2026-09-13 — [MAP] Downloads are serial with a small delay, and resumable by construction — a tile already on disk is skipped. A rejected key or a rate-limit stops the whole run; any other single-tile failure does not. (A hundred parallel requests is how a free tier gets blocked. A key problem affects every remaining tile, so continuing would be thousands of guaranteed failures against someone else's quota; a missing tile is just ocean.)

2026-09-13 — [MAP] A trip's tiles are deduplicated across legs before counting or downloading. (Adjacent legs share the terrain around the stop they meet at, so counting per box charges the user twice. Using one box around the whole trip would be worse still — for a loop it is mostly ground nobody drives through.)

2026-09-13 — [MAP] Clearing the tile cache deletes the FILES, not only the index. (Clearing the index alone would leave hundreds of megabytes on the phone reporting as zero, which is the worst possible answer to "clear the cache".)

2026-09-13 — [SCHEMA] schemaVersion goes to 4, adding the `MapTiles` index. Its unique key is (provider, z, x, y). (Two providers' tiles must never be mistaken for each other and the same tile must never be counted twice. Found while writing it: `insertOnConflictUpdate` defaults to conflicting on the PRIMARY key, which an autoIncrement insert never supplies — so re-writing a tile threw a constraint error instead of updating. The conflict target is now the unique key, explicitly.)

2026-09-13 — [MAP] `MapDownloadScreen` converts its usage stream to broadcast before subscribing. (The usage row lives in a ListView, whose children are disposed and rebuilt freely; a single-subscription stream throws on the second subscribe. Drift's own streams are broadcast so this never bit in production, but a screen that only works with one kind of stream is a trap for the next caller.)

2026-09-13 — [SCHEMA] schemaVersion goes to 3, adding the `AppSettings` key-value table. Migration steps now run in ascending version order, and a comment says why. (The v2 and v3 steps happen to be independent, but a step written as `from < 3` above one written as `from < 2` would silently misbehave the moment a later step assumed an earlier one's column — and only on the phones that had been sitting on the older version. Caught by reading, before it could matter.)

2026-09-13 — [SETTINGS] Settings live in the database, not in shared preferences. (So there is still exactly one place this app keeps state and exactly one thing to back up. There are three keys, and every option is a thing that can be wrong.)

2026-09-13 — [SETTINGS] Clearing a trip's cache removes only what came off the network — places, routes, forecasts — and the dialog says so. (Contacts, expenses, the itinerary and the checklist are the user's own work. Conflating the two is how a "free up space" button destroys an afternoon of typing.)

2026-09-13 — [SETTINGS] Call history counts a COPY as an action. (The dial happens in the Android dialer after a paste, so the app never sees the call itself. Recording only calls would show an empty history to someone who used the app all day.)

2026-09-13 — [WEATHER] Open-Meteo, not a provider needing an account. (Free, no API key, non-commercial use covered. Beyond cost that matters for one reason: it is the only network source in this app that blocks on nothing anybody has to sign up for, which is why #26 could ship while #24 waits on a tile provider.)

2026-09-13 — [WEATHER] Every day of one fetch carries the SAME `cachedAt`, stamped once for the whole call. (Per-row timestamps drift by milliseconds and read as though some days were fresher than others, which is precisely the confusion the column exists to prevent.)

2026-09-13 — [WEATHER] Staleness has three bands — fresh under two days, ageing under a week, stale beyond — and the age is stated IN WORDS beside every stop. Stale renders in caution and mutes its own numbers. (A five-day-old forecast shown as today's weather is worse than no forecast, because somebody packs on it. "Taken 5 days ago" is a fact a person can weigh; a timestamp is not, and a bare temperature is a lie by omission.)

2026-09-13 — [GEO] Stop coordinates are set by a Nominatim lookup that SHOWS ITS CANDIDATES, or by typing a lat/lon pair. Nothing is saved until the user picks. (Two real places are called Shillong, 2,500 km apart. Silently accepting the first result is how a trip ends up routed to a village in Karnataka. The typed option exists because a good share of the homestays this app is for have no name any geocoder knows.)

2026-09-13 — [GEO] The geocoding client rate-limits ITSELF to one request per second and sends a real User-Agent, because Nominatim's usage policy requires both. (Enforced in the client rather than hoped for at the call sites. Breaking either gets an IP banned, and it would be deserved. Pinned by a test with an injected clock.)

2026-09-13 — [GEO] An out-of-range typed coordinate is refused, never clamped. (Clamping would place a stop somewhere real and wrong, which is worse than an empty field.)

2026-09-13 — [UI] A stop with no coordinates says on the form what that costs: the legs either side cannot be downloaded and nothing along them will be found. (It is the quietest possible failure — everything else about the stop looks complete — so the form states it in caution rather than leaving a blank field.)

2026-09-13 — [ARCH] **The release manifest declares INTERNET again. This supersedes the 2026-09-12 decision that it would declare no permissions at all.** (Offline maps make the absolute version of the claim impossible: routes, places and tiles have to come from somewhere, once, before you leave. So the claim narrows and has to be stated honestly — the app is offline WHILE YOU ARE MOVING, not offline absolutely. It reaches the network only on a screen the user opened, after a button saying what it will fetch and from whom, and never once the trip has started. There is still no other permission and no background service, precisely so that stays checkable. The rationale is written into the manifest itself, where the next person to read it will be standing.)

2026-09-13 — [DISCOVERY] A phone number from OpenStreetMap is `communityOsm` and can never be anything else; saving one into the diary lands it as `userEntered`. (The trust invariant reaching the least trustworthy source the app has. It is a number a stranger typed into a public wiki and it may be a decade old. The tier is forced in `CorridorSync` at the single write point, the same way import forces it in the DAO.)

2026-09-13 — [DISCOVERY] Overpass queries refuse a bounding box beyond 15,000 km², send a real User-Agent, and run once per leg at setup. (It is free infrastructure paid for by volunteers. A query bigger than that is a bug, and it costs someone else money. An anonymous flood is how a free service gets an IP blocked.)

2026-09-13 — [DISCOVERY] An OSM place with no name is dropped. ("Unnamed fuel station, 14 km ahead" tells you nothing you can act on or ask a local for.)

2026-09-13 — [DISCOVERY] Category matching is by declaration order and the more specific tag wins, so a hotel with a restaurant reads as somewhere to sleep. (`tourism=hotel` says what a place IS; `amenity=restaurant`, which half of them also carry, says what it also does. Backwards — which is how it was first written — every guest house on the route files under Food, and at 8pm that is the wrong answer. Caught by a test whose comment asserted the behaviour the code did not have.)

2026-09-13 — [DISCOVERY] Re-syncing a leg replaces its places wholesale rather than merging. (A re-sync is the user asking for what is there now. Merging would leave places that have since closed, with no way to tell which.)

2026-09-13 — [DISCOVERY] A leg's whole sync is one transaction written at the end. (A failure halfway would otherwise leave the leg half-synced with a `lastSyncedAt` implying otherwise, which is worse than not synced at all. Pinned by a test that fails the Overpass call after the route succeeds.)

2026-09-13 — [DISCOVERY] The sync screen states what it will fetch before it fetches anything, and the estimate uses straight-line distance while saying it underestimates. (The real route is precisely the thing not yet downloaded. An app that just starts pulling data on someone's hotel WiFi is an app they stop trusting the moment they notice.)

2026-09-13 — [GEO] Bounding-box padding widens longitude by 1/cos φ, using whichever edge is furthest from the equator. (Three kilometres is 3/111 degrees of latitude but 3/(111·cos φ) degrees of longitude. At Shillong's 25.6° that is a 10% difference; ignoring it makes the corridor narrower than advertised east–west and the promised places are simply missing, with no error anywhere.)

2026-09-13 — [GEO] Distance from a point to a route is distance to a SEGMENT, never to the nearest vertex. (A road running dead straight for 40 km has two vertices. Measuring to the nearer one would report a dhaba halfway along it as 20 km off the route when it is sitting on it.)

2026-09-13 — [GEO] The encoded-polyline codec is written out, and precision is an explicit argument on both encode and decode. (Forty lines of bit-shifting over a format unchanged since 2005; a package would mean trusting somebody else's forty lines and carrying their release cycle. Precision matters because OSRM's older API returns 5 and its v5 API returns 6 on request: decoding one as the other scales the whole route by ten and puts Shillong in the Bay of Bengal, which reads as a broken map rather than a wrong number. The request pins `geometries=polyline` rather than letting the server choose.)

2026-09-13 — [GEO] OSRM takes coordinates in LON,LAT order, and the URL builder is tested for it. (It follows GeoJSON, not the lat-first convention the rest of this app uses. Swapping them does not error — it silently routes somewhere else entirely.)

2026-09-13 — [SCHEMA] schemaVersion goes to 2, adding `ChecklistItems.generatorKey`. Version 1's step is untouched. (The first migration this project has needed. The rule stands: a shipped migration is never edited, each version's step is additive and stays as written. Rows written before v2 carry a null key and are adopted by label once, then keyed from then on — pinned by a test that simulates exactly what is already on the phone.)

2026-09-13 — [CHECKLIST] A generated item is keyed by the RULE that produced it, never by its label. (Matching by label looked fine until someone renamed an item: the generator then found no row for its rule and inserted a second copy beside the user's. Caught by the acceptance test for the one rule this issue exists to enforce, which is the test earning its place — the feature's headline promise was quietly broken by its own implementation.)

2026-09-13 — [CHECKLIST] A manual edit survives regeneration, and a REMOVED generated item stays removed by being marked edited-and-done rather than deleted. (Deleting the row would simply bring it back on the next run. "I do not need leech socks" has to survive, and this is the only way it can without a separate tombstone table.)

2026-09-13 — [CHECKLIST] Ticking an item is NOT an edit. (Everyone ticks things off. Treating that as taking ownership would freeze the entire list against regeneration the first time someone packed a toothbrush.)

2026-09-13 — [CHECKLIST] The generator is not run automatically on every stop edit; it runs when the checklist is opened, and on an explicit rebuild. (Regeneration is cheap but it is also the moment edits could be lost, so it happens while the user is looking at the list rather than invisibly behind a drag on another screen. The rebuild button also lets them test the promise on purpose.)

2026-09-13 — [CHECKLIST] Item counts scale with the nights carrying that TAG, not with the whole trip. (Five nights in a city and two trekking means two pairs of trekking socks, not seven. A generated list that is obviously wrong about one thing gets ignored about everything.)

2026-09-13 — [CHECKLIST] No weather lookup at generation time, and the screen says so. (The forecast is a cached snapshot that may not exist. A checklist you cannot produce without one is a checklist you cannot produce at a dhaba.)

2026-09-13 — [MONEY] `formatRupees` shows paise whenever they exist and never rounds them away. Supersedes the rounding introduced at #55. (Rounding read better right up until the split screen put three shares of a ₹3,200.11 taxi beside their total: ₹1,067 three times against ₹3,200, under a line claiming the shares added up exactly. They did, in paise. A ledger that rounds is a ledger that looks wrong at exactly the moment someone checks it. Found by rendering the golden.)

2026-09-13 — [MONEY] `parseRupees` truncates beyond two decimals rather than rounding, and returns null on nonsense rather than zero. (Rounding 340.567 to 340.57 invents a paisa nobody spent and the ledger is out by it forever. Recording zero for unparseable text would put a free taxi in the ledger.)

2026-09-13 — [MONEY] The expense form cannot save an unbalanced split, and the editor asserts it again. (Shares that do not sum to the amount put the ledger permanently out by the difference and nothing downstream would ever notice. The form is a convenience; the editor is the guarantee.)

2026-09-13 — [MONEY] Shares follow the amount until the user takes them over by hand, and an existing uneven split opens already in hand-set mode. (Recomputing an even split on open would silently rewrite a deliberate arrangement — someone who paid more of a room does not want it averaged away by opening the screen.)

2026-09-13 — [MONEY] Deleting a traveller who appears in the ledger is REFUSED with the count, never cascaded. (The schema cascades their splits, which would quietly change everyone else's balance: the expense keeps its total but loses a share, so the payer is suddenly owed more than they are.)

2026-09-13 — [MONEY] Editing an expense replaces its splits wholesale rather than diffing them. (A split whose traveller was removed from the expense has to go, and the unique key on (expense, traveller) makes a partial update fiddly for no gain.)

2026-09-13 — [MONEY] `watchMoneySummary` names its real dependency set with a `readsFrom` tick. (Same bug the trip summary had at #16: it watched only `expenses`, so adding a traveller or editing a split left the screen stale. Invisible until #31 made either possible.)

2026-09-13 — [TRIP] A place may appear more than once, and nothing keys a stop by its name. (The Meghalaya itinerary is Shillong → Cherrapunji → Shillong → Dawki, and the two Shillong rows are different stops with different dates, contacts and nights. This is why `sequenceOrder` exists, why `addStop` appends rather than upserting, and why the acceptance test builds the real itinerary and asserts the two rows have distinct ids. The stop form says it out loud too, because it looks like a mistake.)

2026-09-13 — [TRIP] `sequenceOrder` is renumbered densely — 1, 2, 3 — inside a transaction after every change. (A sparse ordering works right up until two stops share a number, and then the itinerary scrambles in a way that is very hard to read back off the screen.)

2026-09-13 — [TRIP] Deleting a stop keeps its contacts, unattached. Deleting a TRIP takes them. (The schema already said this with `setNull` on `Contacts.stopId`; the confirmation dialog now says it in words and names the count. A number you have dialled and confirmed does not stop being a real number because you dropped the stop from the plan. A trip is a different matter: it is the container everything hangs off.)

2026-09-13 — [TRIP] Nights derive from the dates when both are set, and are typed otherwise. A reversed range is zero, never negative. (Two dates plus a contradicting night count is a question with an obvious answer, and asking the user to keep them in sync by hand is asking them to maintain the app's data model.)

2026-09-13 — [TRIP] A leg whose (from, to) pair is unchanged KEEPS ITS ROW, and therefore its cached polyline, distance and `lastSyncedAt`. Matching is by pair, never by sequence position. (Regenerating by deleting every leg and inserting fresh ones is four lines shorter and throws away the only data in this app that needed a network connection to obtain. Reordering the last two stops of a ten-stop trip has to leave the first eight legs untouched, and a test proves it does.)

2026-09-13 — [TRIP] A repeated pair consumes existing leg rows one at a time rather than matching by lookup. (Shillong → Cherrapunji → Shillong → Cherrapunji is a real itinerary. A single lookup per pair would reuse one row twice and silently drop a leg.)

2026-09-13 — [TRIP] Legs cannot be created directly. They exist because two stops are consecutive, and the leg list says so on its empty state. (A leg the user made by hand would have no pair to match on, so the generator would delete it on the next reorder — a bug that would look like the app losing data.)

2026-09-13 — [TRIP] Transport details are typed and there is no schedule lookup, now or ever. (A Rome2Rio-style fetch is precisely the runtime network dependency this project exists to avoid. What the app can do is hold what the user was told, where they can read it with no signal.)

2026-09-13 — [TRIP] Exactly one trip is active, and activation is a transaction that clears the rest. (The diary scopes to it, the money splits within it, the emergency screen reads its stops. Two active trips would make all three ambiguous. `ensureActiveTrip` promotes the newest at startup, because every trip predating #16 is stored with the flag unset.)

2026-09-13 — [TRIP] The current stop is derived from today's date, and THE FALLBACK IS A STOP, NEVER NULL. (Before the trip: the first stop. After it, or with no dates at all: the last one reached. A trip whose dates were never filled in is exactly the trip planned in a hurry, and returning null there would make the diary's stop-scope toggle vanish on the itineraries most likely to need it. The boundary day belongs to the stop that is leaving, not the one arriving: you are still there until you go.)

2026-09-13 — [TRIP] `watchTripSummary` and `watchActiveTripContext` name their real dependency set with a `readsFrom` tick query. (A Drift stream only fires for the tables its own query touches. Watching `trips` alone looked correct for as long as nothing could edit a stop; the moment #16 shipped, adding a stop would have left the Trip screen stale with no error anywhere. Found by reading, not by a test — a stale stream fails no assertion.)

2026-09-13 — [READINESS] Only OVERNIGHT stops block departure, and an ABSENT accommodation number blocks as loudly as an unconfirmed one, with different words. (A lunch stop with no number is an inconvenience; a homestay with no number at 9pm in a valley is a night in the car. Absence is the more dangerous of the two and the case apps usually say nothing at all about, so it reads "NOTHING TO CALL" rather than being silently ready.)

2026-09-13 — [READINESS] One confirmed number clears a stop, however many unconfirmed ones sit beside it. (A second number for the same homestay is not a second problem. Blocking on every row would make the check unclearable for anyone who imported a sheet.)

2026-09-13 — [READINESS] A generated checklist item the user has edited survives regeneration untouched, and is not silently ticked off either. (`isUserEdited` exists for exactly this. #20 is the first generator to run, so the rule is settled here rather than at #29: their wording wins over ours, always.)

2026-09-13 — [READINESS] An item that stops blocking is marked done, not deleted. (Something that got handled should stay visible. A list that empties itself gives no sense that anything was accomplished.)

2026-09-13 — [READINESS] The pre-departure block uses `cautionMark`, never emergency red, and a test asserts no red on the panel. (Red means emergency and nothing else. A number you have not called yet is a thing to do before you leave, and spending the emergency colour on it blunts red on the one screen where it has to carry weight.)

2026-09-13 — [ARCH] `combineLatest2` is written out in thirty lines rather than adding rxdart. (The app needs exactly this and nothing else from the reactive-extensions world. It waits for both sources before emitting, so a readiness check built from stops alone cannot flash "not ready" for one frame and then correct itself — a screen that lies briefly is worse than one that is briefly empty.)

2026-09-13 — [UI] The Trip tab stays read only; editing lives behind an EDIT rule on the Stops header and in the More tab. (The thing you look at on the road should not be a thing you can knock out of shape with a stray tap while driving.)

2026-09-13 — [UI] `ReorderableListView.onReorderItem`, not `onReorder`. (The older callback reports the destination index as it would be before the item is removed, so every downward drag lands one row too high unless the caller subtracts one. The newer one has already done that arithmetic.)

2026-09-12 — [IMPORT] The sheet parser takes BYTES, not a path. (No `file_picker` import, no `dart:io`, no platform channel, so the whole of #11 is testable in a plain Dart test with a string literal. It also survives the file picker being replaced, and it is the only thing that works on Android anyway: a picked file usually lives behind a `content://` URI with no readable filesystem path.)

2026-09-12 — [IMPORT] Every row carries the 1-based line number it occupied in the original file, header included, and blank rows do not renumber what follows. (Every warning downstream says "row 14", and row 14 has to mean what the user sees in Excel. Dropping blank rows and reindexing turns each message into a scavenger hunt through a sixty-line sheet.)

2026-09-12 — [IMPORT] A cell that arrives as a whole-number double renders as digits, not scientific notation. (Excel stores a phone number typed without a leading `+` as a float, so `9876543210` comes back as `9.87654321E9`, normalises to nothing, and looks like a corrupt file. This is the single most likely way a real sheet of Indian mobile numbers breaks, and it is invisible until someone imports one.)

2026-09-12 — [IMPORT] Column auto-matching resolves exact alias hits across all fields BEFORE any containment guessing. (Otherwise `no` matches inside `notes` and steals the column `phone` wanted. Pinned by a test with headers `Name, Notes, Number`.)

2026-09-12 — [IMPORT] A column is never claimed by two fields; the first match in enum order wins, and reassigning a column takes it away from whoever held it. (A sheet with both `Phone` and `WhatsApp Number` otherwise maps one column twice and the user gets a duplicate they never asked for.)

2026-09-12 — [IMPORT] A row that fails E.164 normalisation still imports, carrying the raw text. (The diary already shows an amber dot on it, so nothing is claimed that is not true, and refusing it would lose the only record the user has of that number. Same reasoning as the entry form at #8: a bad number warns, it never blocks.)

2026-09-12 — [IMPORT] Warning rows arrive SELECTED; only rows with nothing to save are deselected, and those cannot be turned on. (A warning is information, not a veto. Making the user re-tick fifty duplicate warnings would train them to stop reading warnings, which is the opposite of the point.)

2026-09-12 — [IMPORT] Duplicates are detected against the diary AND against earlier rows of the same file. (A sheet listing the same homestay under two stops is the common case, not an edge case.)

2026-09-12 — [IMPORT] A blank cell never marks a contact as an emergency number. Only an explicit yes does — `y`, `yes`, `true`, `1`, `haan`, `x`, `sos`. (An emergency flag set by accident puts a wrong number on the one screen that has to be right. Anything unrecognised reads as no.)

2026-09-12 — [IMPORT] Preview severity reads as a 4px stripe as well as a colour, and RED IS NOT USED. (Colour alone fails in bright sun and for a colourblind reader. Red belongs to the emergency tab; a duplicate row is not an emergency, and spending the colour here devalues it where it counts.)

2026-09-12 — [IMPORT] Rollback asks first and the question names how many contacts will go, including any since confirmed. (An undo that silently removes nine numbers is indistinguishable from a bug. The history screen also tracks how many of a batch survive, so it never offers to remove rows already deleted one by one.)

2026-09-12 — [IMPORT] The template is a header row copied to the clipboard, not a downloaded file. (A release build cannot write to shared storage without a permission this app deliberately does not ask for. Pasting a line into row 1 of a new sheet solves the same problem with nothing to grant, and auto-matching means most people never need it.)

2026-09-12 — [IMPORT] An unmatched stop name is not an error: the row imports as a trip-level contact and the preview says which stop it landed on, or that it landed on none. (Refusing a good phone number because a place was spelled differently would be the feature working against its own purpose. The number is what you need at 9pm outside a locked homestay; the stop association is a convenience.)

2026-09-12 — [IMPORT] Fuzzy stop-matching tolerance scales with name length: nothing under five characters, then roughly a fifth of the name, capped at three edits. (A fixed threshold is wrong at both ends. Two edits makes `Puri` match `Pune`, 1,500 km apart; one edit makes `Cherrapunji` miss `Cherrapunjee`. Levenshtein is written out in fifteen lines rather than pulled from a package, because it runs against a handful of names and the app should carry no dependency it does not need.)

2026-09-12 — [IMPORT] The import trust guard was re-proven by removing it. (Deleting the tier override in `insertBatch` makes the test report a confirmed contact arriving through import, so the test is doing work rather than passing vacuously. Worth repeating whenever that method is touched: a guard nobody has watched fail is a guard nobody knows is there.)

2026-09-12 — [MONEY] `numberStyle` and `badgeStyle` carry `fontFamilyFallback: [Archivo]`. (Courier Prime is a 1950s typewriter design and has no rupee glyph — ₹ was only adopted in 2010 — so every amount on the money screen rendered as a tofu box. The fallback supplies the sign while the digits still come from Courier. Found by rendering the golden; no widget test can see a missing glyph, because the text is present in the tree either way.)

2026-09-12 — [MONEY] A negative balance is never red. Negative renders muted, positive in signal green. (Red in this app means emergency and nothing else. Owing your brother-in-law ₹400 is not an emergency, and spending the emergency colour on a settled debt devalues it on the one screen where it has to carry weight.)

2026-09-12 — [MONEY] The app records a settlement, it never executes one. No payment integration, now or later. (It would reintroduce a network dependency and a compliance surface for something people already do with cash or UPI in thirty seconds. The value here is knowing the number, not moving it.)

2026-09-12 — [MONEY] Settle-up sits ABOVE the ledger, not behind a tab. (It is the only part of a shared ledger anyone acts on. Most apps bury it; putting it first is the whole reason to have this screen rather than a spreadsheet.)

2026-09-12 — [MONEY] Debts are simplified greedily — largest debt to largest credit. (The general minimum-payments problem is NP-hard, and the greedy result is not provably minimal for every graph. It is optimal for the shapes a five-person trip produces, never emits more payments than there are people, and is exact: the payments always sum back to the balances. Exactness is the property that matters; minimality is a nicety.)

2026-09-12 — [MONEY] Even splits hand out the remainder one minor unit at a time rather than rounding each share. (A three-way split of ₹100 has to still be ₹100 when you add it back up. Rounding each share independently loses or invents a paisa, and a ledger that does not balance is worse than no ledger.)

2026-09-12 — [TRIP] The milestone cap is muted when the leg has never been synced. (A distance painted in full ink claims the number was surveyed. A leg with a null `lastSyncedAt` has a distance the user typed or the app guessed. This is the trust tiering invariant applied to geometry instead of phone numbers — same rule, different table.)

2026-09-12 — [TRIP] The trip screen is read-only. Creating and editing a trip stays at #16. (The question the app has to answer at a dhaba at 3pm is "where am I and what is the next leg", not "let me restructure my itinerary". Shipping the read path first makes four of five tabs live without waiting on the editor.)

2026-09-12 — [NAV] Five tabs — Diary, Trip, Money, SOS, More — over an `IndexedStack`, so each tab keeps its scroll position and filter state. (A rebuild-on-switch would drop the diary's category filter every time someone checked a balance. The cost is that all five trees stay alive; at this size that is cheaper than the state plumbing needed to restore them.)

2026-09-12 — [NAV] SOS renders in emergency red only while it is the active tab. (A permanently red item in the bar becomes wallpaper within a day, and then the colour means nothing at the moment it has to mean something.)

2026-09-12 — [BUILD] `file_picker` upgraded from ^8.3.7 to ^12.3.0. (The first CI build failed: file_picker 8 compiles against Android API 34, while `flutter_plugin_android_lifecycle` now requires everything depending on it to compile against 36 or later. Nothing in the app uses file_picker yet — it arrives at #11 for sheet import — so the jump carries no API risk today, and leaving it would have blocked every Android build. Found only by attempting a real build; `flutter analyze` and the whole test suite pass either way, because none of it touches Gradle.)

2026-09-12 — [BUILD] APKs are built by a GitHub Actions workflow, not locally. (The development container cannot reach `dl.google.com`, so it has no Android SDK and cannot produce one. GitHub's runners already have it, and the workflow runs `flutter analyze` and the full suite before building, so any APK that exists is one that passed everything.)

2026-09-12 — [DEV] The demo trip can be created on demand in any build, not only seeded automatically in debug. (A release APK has no trip and no way to make one until #16, so it would open to a blank screen. The app now offers a "Create demo trip" button instead, which is honest about what it is and keeps an installable build useful.)

2026-09-12 — [ARCH] Package visibility declared in the Android manifest for `tel:`, `sms:` and `https:`. (On Android 11 and later an app cannot see which other apps handle an intent unless it declares them, so `url_launcher` returns false and every action fails silently. Flutter's template ships only a PROCESS_TEXT query, so this was missing — and the first device test would have blamed the empty-path `tel:` theory for a package-visibility problem. Found by reading the manifest before writing the run instructions, not by any test: nothing in a widget test touches the manifest.)

2026-09-12 — [ARCH] The release manifest declares NO permissions at all, INTERNET included. Only the debug manifest has it, for hot reload. (The app makes no network call at runtime, so it should not ask for the ability to. It is also the clearest possible proof of the offline claim: the release build cannot reach the network even if a future line of code tried.)

2026-09-12 — [CONTACTS] Confirming lives on a labelled button on the entry screen, never on a swipe or a long-press. (It is the single action the whole trust system depends on, and the app cannot check it — it has no way to know whether a call connected. So it asks the user to assert it and makes the assertion deliberate. A gesture that could fire by accident would silently promote an unverified number.)

2026-09-12 — [CONTACTS] Clearing a confirmation drops the tier back to `userEntered`, in the UI as well as in the DAO. (The entry screen computed its tier from the stored value, so after clearing it still read "Confirmed by you". Found by a test. The screen now mirrors `markConfirmed` exactly.)

2026-09-12 — [CONTACTS] Confirmation is offered only on tiers a user can vouch for. `verifiedNational` and `verifiedState` get no confirm section at all. (Those are authoritative by provenance, not because anyone dialled them. Asking the user to confirm 112 would imply their assertion is what makes it true.)

2026-09-12 — [CONTACTS] Un-confirming is supported and says why: a number that worked in October may not work in November. (A one-way promotion would leave stale confirmations with no way back, which is the same failure as a fabricated number wearing a slower disguise.)

2026-09-12 — [UI] The entry screen labels its usage row "Last action", not "Last called". (A copy counts, since the dial happens in the Android dialer after a paste. Calling it "last called" would overstate what the app actually observed.)

2026-09-12 — [UI] Typefaces replaced: **Jost** for signage, **Archivo** for words, **Courier Prime** for numbers. Supersedes the Inter + Archivo Narrow pairing chosen on 2026-09-11. (Yash's call: the app looked AI-generated, and he is right — Inter is the house face of every dashboard shipped since 2018 and reads as a default rather than a choice. The replacements are period-rooted rather than merely different: Jost is a Futura revival, and Futura is the geometric sans painted onto enamel road signs and milestone caps from the sixties through the eighties; Archivo revives the mid-century newspaper grotesques; Courier Prime is a typewriter, which is what typed numbers into a diary. All three are SIL OFL and together are 924 KB, slightly less than the two they replaced.)

2026-09-12 — [UI] Numbers are set in the typewriter face, not the body face. The rule: a number you might dial, count or compare is TYPED — Courier Prime; a number painted onto an object is SIGNAGE — Jost. (So the emergency badge reads as typed while a milestone's km reads as painted, though both are digits. Courier is monospaced, so phone numbers align down the column by construction rather than by asking for a font feature — the alignment the whole diary depends on is now structural.)

2026-09-12 — [UI] The diary's thumb index uses `ContactCategory.shortLabels`, separate from the full labels used everywhere else. (Jost is wider than the condensed face it replaced, and TRANSPORT and PHARMACY clipped when set vertically on a phone edge. A phone edge has room for about six characters.)

2026-09-12 — [UI] The search field is set in the word face, not the number face. (It takes a name as often as a number, and a typewriter face made every hint look like a serial number.)

2026-09-12 — [CONTACTS] Changing the digits of a confirmed entry clears its confirmation: back to `userEntered`, `callConfirmed` false, `confirmedAt` null. Changing the name, note, category or stop does not. (A confirmation means "I called THIS number and it worked". Change the digits and that is no longer true. Silently keeping a green tick against a number nobody has dialled is exactly the failure the trust system exists to prevent. The form warns before saving, and a test pins it.)

2026-09-12 — [CONTACTS] An unreadable or incomplete number is a warning, never a block. `phoneE164` stays null and the raw value is stored alone. (The user may be halfway through typing, holding a number with an extension, or copying something off a signboard. The amber dot already says the number is unverified; refusing the save would lose their only record of it. Storing a guessed E.164 for a number that did not validate would be worse — a confident-looking wrong value in the column used for dialling and dedupe.)

2026-09-12 — [CONTACTS] A duplicate number warns and names the entry that already holds it, but never blocks. (Two entries for one number is legitimate: a homestay's landline and its owner's mobile can be the same line on different days. Editing an entry is not treated as a duplicate of itself.)

2026-09-12 — [DATA] The country to normalise against is passed in by the caller rather than fixed to India. (A European trip crosses borders mid-itinerary and `countryCode` already lives on Stop for exactly this reason. The app passes IsoCode.IN today; wiring it to the current stop is one line once trip structure exists at #16.)

2026-09-12 — [UI] `ContactCategory.pickerOrder` is the single order categories are offered in, shared by the diary's thumb index and the entry form. (Declaration order leads with hospital; on a road trip you reach for stay, transport and food far more often. More importantly the two places a user picks a category must agree, or the muscle memory breaks.)

2026-09-12 — [TEST] Golden images are generated from the real widget tree with the bundled fonts loaded, via `test/golden_test.dart`. (It is the only way to see the app without a phone, and it immediately found two bugs no other test could: the doubled margin rule was painting as a solid grey column, and the New entry button was sitting on top of the thumb index. Material's icon font ships with the SDK rather than the app, so the harness loads it too or every icon renders as a tofu box and the image lies about what the user sees.)

2026-09-12 — [UI] The doubled margin rule is drawn as two sibling hairlines, not as a border plus a `boxShadow`. (A shadow spreads behind the whole box and fills it, which is what the first golden showed. Cosmetic, but it made the diary look like a table rather than a notebook.)

2026-09-12 — [ARCH] `ContactActions` takes an injectable launcher and clipboard. (Platform intents cannot run in a test, and the alternative is leaving the most failure-prone code in the app untested. Injecting them means the digits, the schemes, the log rows and every failure path are covered without a device.)

2026-09-12 — [CONTACTS] A launch that fails throws `ActionFailure` with a sentence, and the screen shows it in the toast. Some devices throw instead of returning false, so both are caught. (A silent no-op on a phone that cannot place a call is the worst possible outcome for this app. A failed action is also never logged as if it happened.)

2026-09-12 — [CONTACTS] WhatsApp links strip every non-digit before building the `wa.me` URL. (A leading + or any spacing breaks the link silently.)

2026-09-12 — [UI] The copy toast sits above the New entry button rather than beside it. (The first golden had the button covering the Open dialer action, which is the entire point of the toast.)

2026-09-12 — [UI] The toast is hand-drawn, not a `SnackBar`. (The number has to read in tabular figures and the action has to be a stencil mark; bending Material's snackbar theming to that costs more than drawing it.)

2026-09-12 — [ARCH] Screens take streams, not DAOs. `DiaryScreen` receives a `Stream<List<Contact>> Function(ContactFilter)` and a `Stream<int>`; `ReadinessBanner` receives a `Stream<int>`. The composition root wires the DAO in. (Forced by a real constraint and better structure anyway. A widget test CANNOT close a Drift database — `close()` awaits work the fake clock never advances, so the test hangs until the runner is killed — and cancelling a query stream leaves zero-duration cleanup timers the framework then reports as pending, after user teardowns have already run. Passing streams removes Drift from widget tests entirely. The DAO's real ordering, filtering and search stay covered against real SQLite in `contacts_dao_test.dart`, so nothing is lost: the screen's job is to render what arrives and say what it wants next.)

2026-09-12 — [TEST] Never call `pumpAndSettle` on a screen whose loading state is a `CircularProgressIndicator`. (It animates forever, so the call can never settle and the test hangs for its full ten-minute timeout. Use bounded pumps. A filter change swaps in a new stream that delivers on a microtask, so an interaction needs two pumps, not one.)

2026-09-12 — [UI] The diary replaces the smoke screen as `home`. Bottom navigation is deferred until a second screen exists. (Nav belongs to a shell above the screen, and there is nothing yet to navigate between. Building the bar now would mean four dead destinations.)

2026-09-12 — [DEV] A debug-only demo trip is seeded when the database holds no trip at all, so the diary has something to render before trip CRUD exists at #16. Its numbers are deliberately, obviously fake — +91 90000 000xx. (Placeholder data must never be mistakable for a number someone might actually dial. The file is deleted at #16.)

2026-09-12 — [CONTACTS] `insertBatch` forces every imported row to `userEntered` / `callConfirmed = false` / `confirmedAt = null`, overriding whatever the caller passes. (The drafted DAO passed the caller's companion straight through, so a screen that built a confirmed row would have imported it confirmed — hollowing out the invariant the whole trust system rests on. Enforcing it at the DAO makes it structural: the DAO is the single write surface, so a future caller that forgets cannot get it wrong. A test proves the failure exists without the guard.)

2026-09-12 — [CONTACTS] The diary feed is ordered pinned, then confirmed, then alphabetically, and deliberately NOT by recency. (The drafted DAO's comment claimed recency but its code never did it. Keeping it alphabetical for two reasons: a diary's page order has to be stable or the margin numbers shift under you; and `lastCalledAt` is null for anything never called, which SQLite sorts as smallest, so a DESC recency term would sink every never-called number to the bottom of its group — exactly the category the readiness system exists to surface. Recency belongs in the call log, not the page order.)

2026-09-12 — [CONTACTS] `lastCalledAt` means "last outbound action of any kind", including `copy`. (Since the dial happens in the Android dialer after a paste, a copy is the closest thing the app observes to a call. The entry screen should label the field that way rather than "Last called".)

2026-09-12 — [ARCH] `app_database.dart` re-exports `tables.dart`. (So a DAO can declare `@DriftAccessor(tables: [...])` with a single import, which is what let the drafted DAO drop in with no changes to its body beyond the invariant fix.)

2026-09-11 — [CONTACTS] Helpline seeding upserts on the natural key (country + number + service) rather than insert-or-ignore. (Both are idempotent, but insert-or-ignore would strand anyone who already installed the app on an old label or sourceNote if a later version corrects one — and these are bundled reference data, not user rows. The seeder never touches `id`, so anything referencing a helpline survives a re-seed. Pinned by a test that corrupts a row and re-seeds.)

2026-09-11 — [CONTACTS] The four flagged numbers stay in the seed file marked `needsVerification` and are skipped by the seeder, rather than being deleted from it. (So a future session can see they were considered and rejected rather than overlooked. #36 verifies them against a .gov.in source and flips the flag; nothing else should. A test asserts all four are absent from the database and present in the file.)

2026-09-11 — [CONTACTS] 102 and 108 both seed with the plain label "Ambulance" because PROJECT_RUNDOWN §4.2 does not distinguish them. (They are run differently in different states, but inventing a distinction would be exactly the confident-sounding guess this whole feature exists to avoid. Resolve at #36, from state portals, not from memory.)

2026-09-11 — [ARCH] The database is constructed in `main()` and passed down through the widget tree, not reached for through a global or a service locator. (Consistent with having no repository layer: DAO to widget, until a second consumer of the same data appears. It also makes every widget test able to substitute an in-memory database with no ceremony.)

2026-09-11 — [CONTACTS] Seeding is awaited before `runApp`. (A fresh install must never render an empty emergency screen, not even for one frame. Seeding fourteen rows costs nothing.)

2026-09-11 — [ARCH] drift and drift_dev pinned to ^2.35.0, drift_flutter to ^0.3.1, sqlite3_flutter_libs to ^0.6.0. (At drift_dev 2.31 under analyzer 10, code generation SILENTLY DROPPED EVERY FOREIGN KEY, unique key and table constraint from the generated schema, emitting only a vague "this parameter should be a simple class name" warning that build_runner reported as success. The upgrade pulls analyzer 13 and generates all 27 references correctly. The schema tests added at #2 assert against sqlite_master rather than against generated Dart, so this class of silent failure cannot recur unnoticed.)

2026-09-11 — [DATA] `PRAGMA foreign_keys = ON` runs in `beforeOpen`. (SQLite defaults it OFF. Without it every ON DELETE CASCADE in the schema is inert and deleting a trip leaves orphaned stops, contacts and expenses behind silently. Pinned by a test.)

2026-09-11 — [DATA] Money is stored as integer minor units — paise, cents — never as a double. (Floating point accumulates rounding error across a three-way split and this ledger has to balance exactly. A test splits ₹3,200.11 three ways and asserts not one paisa goes missing.)

2026-09-11 — [DATA] `Travellers` table added; expense splits reference a person row rather than a free-text name. (Not in the original data model because the expense feature was described before it was designed. A name per split would make balances unjoinable. No accounts, no sync, no server — just named people on a trip.)

2026-09-11 — [DATA] POI stop-XOR-leg exclusivity is a table CHECK constraint, not a convention. (A POI attached to neither answers no query; one attached to both answers two queries wrongly. DESIGN.md §2 called for the rule, so the database should be the thing that holds it.)

2026-09-11 — [DATA] EmergencyHelplines carries a unique key on country + number + service type. (This is what makes the first-launch seeding at #4 idempotent rather than merely careful. Seeding runs again after a reinstall or a migration.)

2026-09-11 — [DATA] `WeatherSnapshots.cachedAt` is non-nullable. (A snapshot without an age is a forecast pretending to be current, which is the exact failure the snapshot model exists to prevent.)

2026-09-11 — [DATA] `CallLogs.action` takes `copy` as a first-class value alongside call, dialer, sms and whatsapp. (The dial now happens in the Android dialer after a paste, so without logging copies the recents ordering and the record of who was actually reached would rot the moment the workflow changed.)

2026-09-11 — [UI] `muted` darkened from #6B7670 to #5F6963. (The original measured 4.41:1 on paper and 3.94:1 on the raised surface, both failing AA, on the token that carries every phone number, caption and provenance line. DESIGN_VISUAL_v2 claimed 4.9:1 — the arithmetic was done by hand and was wrong. Found by the first test run, not by eye.)

2026-09-11 — [UI] Contrast is asserted by a test over both grounds, `paper` and `stone`, in both themes, rather than stated in a document. (Checking only against the base ground is how a failing value got written down as passing. Chips, the search field and category avatars all sit on `stone`, where every ratio is roughly 0.6 lower.)

2026-09-11 — [UI] Bottom navigation grows to five items: Diary, Trip, Money, SOS, More. Checklist, timeline, stop detail and leg detail hang off Trip rather than taking a sixth slot. (Five is the Material maximum. Money earns top level because expenses are entered daily; the checklist is used hard before departure and the timeline logs itself, so both are destinations rather than tabs.)

2026-09-11 — [CHECKLIST] Unconfirmed numbers attached to an overnight stop are rendered as blocking checklist items, in the same list as the pack items. (This is where the trust system surfaces for the user. "Call the homestay" belongs beside "pack leech socks" because both are things that must be done before leaving signal, and it gives the readiness check a home instead of a banner.)

2026-09-11 — [CHECKLIST] Every generated item displays the activity tags that produced it. (A generated list nobody understands gets ignored. Showing that leech socks came from trek + rain + caves makes the list arguable, which is what makes it trusted.)

2026-09-11 — [MONEY] Balances render positive in signal and negative in muted, never in red. (Owing money is not an emergency. Red stays rationed to the emergency surface.)

2026-09-11 — [MONEY] Settle-up is shown above the ledger. (The question people actually have is who owes whom, not what was spent. The ledger is the audit trail, not the headline.)

2026-09-11 — [MONEY] Settlement is recorded, never executed. The app has no payment integration and never will. (It would reintroduce a network dependency and a compliance surface for something people do with cash or UPI in thirty seconds.)

2026-09-11 — [TIMELINE] The GPS logging toggle states its battery cost on the screen, roughly 4% a day. (It is the one feature in the app that genuinely drains the phone. Burying that in settings would be dishonest in a product whose premise is surviving days without a charger.)

2026-09-11 — [WEATHER] The staleness stamp is muted under three days and turns caution past three, with a sentence spelling out what it means. (A five-day-old forecast presented as current is the exact failure this snapshot model exists to avoid — Windy's pattern is only honest if the age is louder than the data.)

2026-09-11 — [TRANSPORT] Leg transport details are typed by the user and the screen says so in words. (There is no live schedule lookup and there never will be offline. Implying one through a blank field that looks fetchable would be worse than an empty form.)

2026-09-11 — [ARCH] The Flutter project lives at `safarsathi/` in the repo root, beside `docs/safarsathi/`. (The app is not documentation and does not belong under `docs/`. Both folders move together when SafarSathi gets its own repo.)

2026-09-11 — [ARCH] Platform folders limited to android and ios; web, Linux, macOS and Windows runners are not generated. (Dead weight for an app whose whole premise is a phone with no signal.)

2026-09-11 — [UI] Inter and Archivo Narrow are bundled as variable fonts, with weights selected through `FontVariation` on the `wght` axis in every text style. (Google Fonts no longer publishes static instances for either family. `fontWeight` alone does not reliably move the axis, so the whole app would silently render at one weight. A test fails if any style stops pinning it.)

2026-09-11 — [UI] Fonts are bundled as assets rather than pulled with the `google_fonts` package. (That package fetches at runtime, which is unacceptable in an app defined by having no network.)

2026-09-11 — [UI] Grain tile is 128×128 two-bit greyscale, 3.2 KB. (Eight-bit noise compressed to 15 KB, nearly four times the budget in DESIGN_VISUAL_v2 §4 — random data barely compresses. Four grey levels are indistinguishable at 3% opacity.)

2026-09-11 — [DATA] `phone_numbers_parser` replaces `libphonenumber_plugin` for E.164 normalisation. (Pure Dart, no platform channel, works offline. The plugin wraps a native library over a channel, adding a platform dependency for what is string work.)

2026-09-11 — [UI] Tapping a diary entry copies the number to the clipboard; it no longer dials. Supersedes the same-day decision that the whole dialer row is the tap target and it dials. (Yash dials by pasting into the Android dialer, so copy is his actual workflow. Direct call stays as an explicit button and as a swipe-right, so nothing is lost.)

2026-09-11 — [UI] The copy toast carries an OPEN DIALER action that launches the platform dialer with an empty field, making the workflow copy → open → paste in two taps. (Neither action needs a dialer permission. Verify on a real device that a `tel:` with no path opens the dialer rather than erroring; the fallback is ACTION_DIAL over a platform channel.)

2026-09-11 — [UI] The emergency screen keeps tap-to-call as its primary action with a heavy haptic; copy is demoted to a secondary icon there. (Copy-first is a planning workflow. A copy-and-paste dance at the wrong moment is a liability, and this is the screen where that matters.)

2026-09-11 — [UI] Contacts are presented as a pocket diary: ruled lines, a numbered margin, a category thumb index down the right edge, and a page count at the foot. (Yash asked for a real contact-diary system. The index is the retro form doing actual work — flipping to "Stay" one-handed in a village — rather than decoration, and the margin numbers give you something to say out loud when reading a number to someone.)

2026-09-11 — [UI] The number is set larger than the name in every diary row and is the headline on the entry screen, at 27pt tabular with `user-select: all`. (In a diary the number is the content. Now that the app's job is to hand the number to the system dialer, legibility and selectability of the digits outrank filing.)

2026-09-11 — [UI] Four-item bottom navigation: Diary, Trip, SOS, More. The SOS item renders in emergency red only while active. (Red stays rationed — the one place it leaves the emergency screen, it is pointing at it.)

2026-09-11 — [CONTACTS] Every copy action is written to CallLogs with action 'copy'. (Recents ordering and the record of who you actually reached must not degrade just because the dial now happens outside the app.)

2026-09-11 — [UI] Import preview encodes row severity as a stripe as well as a colour, and uses muted rather than red for a skipped row. (State must read without depending on hue, and red belongs to emergency.)

2026-09-11 — [UI] Visual system v2 adopted, codename "Milestone": a retro-modern Indian road-ephemera idiom across the app, with safety surfaces exempt. Supersedes DESIGN.md §5 only. (Yash asked for retro yet modern and highly interactive; confining ephemera to non-safety surfaces buys character without touching the trust signal that DESIGN.md §4 exists to protect.)

2026-09-11 — [UI] Exemption rule: ephemera may carry category, provenance, place and delight, never trust. The emergency tab and the contact-row trust markers take no retro treatment at all. (A stamp or badge a user could mistake for a verification mark would defeat the entire tier system.)

2026-09-11 — [UI] Base surface moves from #FFFFFF to warm paper #FAF7F0; raised surfaces to #EFEAE0. (Supersedes the pure-white surface in app_tokens v1. Warm paper is the single highest-leverage retro move and costs nothing — ink still reads 15:1 on it.)

2026-09-11 — [UI] No drop shadows anywhere. Separation is a 1px rule plus a ground shift; the FAB and modal sheets take a 1px ink border instead. (Print has no drop shadows, and a border reads as a cut edge, which is the correct metaphor and survives both themes without a shadow colour.)

2026-09-11 — [UI] Second typeface admitted: Archivo Narrow, restricted to stencil moments — section headers, milestone numerals, stamp marks, emergency badges, tab labels. Body text stays Inter. (Supersedes the single-family rule in DESIGN.md §5. A retro direction without a voice is a beige repaint; confining the second face to labels keeps every running sentence one family. Archivo Narrow is a grotesque like Inter, so the pair reads as one voice at two volumes rather than a collision.)

2026-09-11 — [UI] `caution` splits into two tokens: `caution` #A35F10 for text and icons, `cautionMark` #C77B1E for the 7px trust dot and the hazard stripe. (The original #C77B1E measures 3.1:1 on warm paper and fails AA as text; the dot is a graphic held to the 3:1 floor and still needs to read amber at seven pixels.)

2026-09-11 — [UI] Night theme added — warm charcoal #14120E, not neutral black. (This app's defining use is a village at 11pm with no signal; a day-only theme is a functional gap, not a style gap. Neutral black would be cheaper on OLED and would throw away the paper premise.)

2026-09-11 — [UI] Hazard stripe and grain texture are each rationed to one use — the readiness banner's left edge, and the paper ground below all text. (Same reasoning as red: a motif used everywhere means nothing anywhere.)

2026-09-11 — [MOTION] Motion budget: nothing animates at rest. No ambient motion, no parallax, no animated route drawing, no particles. (The app runs for days between charges with no signal; an idle animation is a battery cost with no user attached to it.)

2026-09-11 — [MOTION] Haptic vocabulary fixed to seven events, never on scroll, never on a stream rebuild, never when data merely arrives. Haptics are not suppressed by reduce-motion. (Feedback, not animation — a user who turned off motion still needs to feel that a call was placed.)

2026-09-11 — [MOTION] Material ink splashes disabled app-wide; every tappable uses a 0.97 press-scale plus a haptic. (A splash spreads outward from a finger; a press-scale reads as paper being pushed, which is the right metaphor for this idiom.)

2026-09-11 — [MOTION] "Mark as confirmed" gets the app's one signature animation, a stamp landing over 380ms with the haptic fired at contact rather than at animation start. (It is the single action the whole trust system depends on; making it feel consequential is design work, not decoration.)

2026-09-11 — [MOTION] Pull-to-refresh replaced by an over-scroll cache stamp showing `lastSyncedAt` and cache size. (There is no network at runtime, so a refresh gesture would be a lie; the same gesture reveals the one thing an offline user actually wants to know.)

2026-09-11 — [DOCS] SafarSathi docs live at `docs/safarsathi/` inside the `yash0304/ml` repo until a dedicated repo exists. (Yash's call this session. The repo is the memory — parking them in a namespaced folder beats leaving a design system in a chat window.)

2026-09-11 — [UI] Category is carried by a monochrome icon avatar, not a coloured chip. (Colour is reserved entirely for trust tier; two colour systems competing in one row makes neither readable.)

2026-09-11 — [UI] Red appears on the emergency tab only. (If red also marks validation errors or deletes, it stops meaning emergency.)

2026-09-11 — [UI] Phone numbers render with tabular figures. (Digits align down the column so a number is read at a glance rather than parsed.)

2026-09-11 — [UI] Palette anchored on Indian highway milestone markers: green = official/confirmed, amber = caution/unverified, red = emergency. (On the road these hues already carry meaning; borrowing that vocabulary beats inventing an arbitrary brand palette.)

2026-09-11 — [UI] Readiness banner renders only when unconfirmed contacts exist. (A permanent banner becomes wallpaper and stops being read.)

2026-09-11 — [UI] Whole dialer row is the tap target and it dials. (Used one-handed, in poor light, sometimes under stress — the commonest action should need no aiming.)

2026-09-11 — [ARCH] No state management library yet; StreamBuilder over Drift streams. Introduce Riverpod at the sync orchestrator (#25). (Drift streams are already reactive; adding a library now is ceremony. DAO layer won't need to change later.)

2026-09-11 — [ARCH] No repository layer between DAO and widget. (Add it when a second consumer of the same data appears, not before.)

2026-09-11 — [ARCH] Sync layer writes to SQLite then disappears; no widget may make a network call. (Keeps the offline constraint structurally enforced rather than merely intended.)

2026-09-11 — [DATA] Contacts store both phoneRaw and phoneE164. (Raw is displayed because reformatting someone's own input is confusing; E164 is for dialing and dedupe, and normalisation can fail on bad input.)

2026-09-11 — [DATA] POI attaches to either a Stop or a Leg, never both. ("What's in Shillong" and "what's on the road to Shillong" need different radii and orderings; merging makes both queries wrong.)

2026-09-11 — [DATA] countryCode lives on Stop, not Trip. (A European trip crosses borders mid-itinerary; emergency numbers, currency and language switch at that boundary.)

2026-09-11 — [DATA] Stops model occurrences, not unique places. (Meghalaya visits Shillong twice, South India visits Ooty and Bangalore twice — two rows is the correct representation, not a bug.)

2026-09-11 — [CONTACTS] Bulk import always lands rows as userEntered / callConfirmed=false, enforced by test. (A number in a spreadsheet is still an unverified number; letting import satisfy the readiness check would hollow out the whole feature.)

2026-09-11 — [CONTACTS] Trust tier must be visible in the list row, never behind a tap. (Prevents the core failure mode: dialing a stale OSM-tagged hospital number at night believing it is authoritative.)

2026-09-11 — [CONTACTS] Individual hospital, pharmacy, homestay and restaurant numbers will never be generated or bundled. Sourced only from OSM tags (marked unverified) or user entry. (No verified open dataset exists; a fabricated ambulance number is worse than no number.)

2026-09-11 — [CONTACTS] State-level helpline list ships empty until manually curated from government portals, with provenance per number. (No authoritative combined dataset exists across 28 states + 8 UTs; guessed coverage is worse than none.)

2026-09-11 — [CONTACTS] 1930, 1078, 1033 and 104 flagged needsVerification and excluded from seeding. (Widely cited on aggregator sites but not confirmed from a .gov.in source; 104 is state-operated and not live everywhere.)

2026-09-11 — [CONTACTS] No emergency number ships without a government-domain source, recorded in sourceNote and shown in the UI. (Yash explicitly asked that this data be genuine; provenance in the UI lets the user judge for themselves.)

2026-09-11 — [SYNC] Map tiles from MapTiler or Stadia, not the standard OSM tile server. (The standard server discourages bulk offline caching; both alternatives permit it on their free tiers.)

2026-09-11 — [SYNC] Overpass (OSM) chosen over Google Places for POI data. (Places is $32/1k basic, $35/1k once any ratings field is included — one field silently upgrades the whole call's SKU — against a non-pooling 5,000/1,000 monthly free tier. Corridor queries multiply calls by category × segment. Deep-link to Google Maps for richer data at zero cost.)

2026-09-11 — [SYNC] Weather is a snapshot cached at setup, displayed with staleness. (Windy's offline pattern; a five-day-old forecast must not look current.)

2026-09-11 — [SCOPE] Rome2Rio-style live multimodal routing dropped; transport legs are manually entered. (It depends on live operator schedule databases, which the offline constraint rules out.)

2026-09-11 — [SCOPE] Check-in alerts delivered by native SMS intent, not a server relay. (Works with zero data, only cell signal — and avoids reintroducing a backend.)

2026-09-11 — [ARCH] App is completely offline at runtime. No FastAPI backend. Network activity happens once, at trip setup, on WiFi. (Yash's explicit requirement; supersedes the Flutter + FastAPI architecture proposed earlier in the same session.)
