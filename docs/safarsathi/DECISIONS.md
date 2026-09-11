# DECISIONS — SafarSathi

Newest first. Format: `YYYY-MM-DD — [AREA] Decision. (Why.)`
Never delete a superseded decision — add a new dated line above it.

---

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
