# SafarSathi — Project Rundown

> Working name. Swap it if you have something better.
> This document is the full record of the design conversation that created
> this project. Read it once; everything after that lives in DECISIONS.md,
> BACKLOG_v0.1.md and HANDOFF.md.

**Status:** pre-code. Schema and one screen drafted, nothing built or run yet.
**Created:** 2026-09-11

---

## 1. What this is

An offline-first travel companion for multi-stop road trips, built around a
problem that ordinary travel apps ignore: **the moments when you have no
signal and need a number that actually works.**

Two real trips drove the design:

| Trip | Route |
|---|---|
| Meghalaya (Oct) | Guwahati → Shillong → Cherrapunji → Kongthong → Shillong |
| South India (Nov) | Bangalore → Ooty → Coimbatore → Trivandrum → Munnar → Ooty → Bangalore |
| Europe (future) | Germany + neighbouring countries, multi-country single trip |

Note the Meghalaya route visits Shillong twice and the South India route
visits both Ooty and Bangalore twice. **Repeat visits are a first-class
case, not an edge case** — this is why the schema models Stops as ordered
occurrences rather than unique places.

---

## 2. The hard constraint

**Completely offline at runtime.** There is no backend. No FastAPI service,
no sync server, no live API call once the trip starts.

The only network activity happens at **trip setup, while on WiFi**: one
batch download that caches everything the trip will need. After that the
app is a local SQLite database and a map renderer.

This constraint is not negotiable and it shapes every decision below. When
in doubt: if a feature needs the network mid-trip, it is out of scope or it
gets redesigned into a pre-cached snapshot.

---

## 3. Feature set and where each idea came from

Yash researched five apps. What was borrowed from each:

**PackPoint** — activity-tagged packing checklists (trip length, weather,
planned activities). Borrowed as: locally-stored checklist templates keyed
to activity tags on each Stop. No weather API needed at generation time —
user picks conditions manually, or the cached weather snapshot fills it in.

**Polarsteps** — automatic background GPS path logging that builds a trip
timeline without manual check-ins. Borrowed nearly whole: GPS needs no
internet, so this works perfectly offline.

**Windy** — its offline mode pattern: download forecast for a selected
region *before* losing connection, then it's static. Borrowed as the
weather snapshot model. Not live weather — a frozen snapshot per stop.

**Rome2Rio** — multimodal A→B route planning. **Mostly not borrowable**;
it depends on live operator schedule databases. Reduced to: user manually
logs planned transport legs (mode, departure, arrival).

**Splitwise** — offline expense entry and the "simplify debts" algorithm
that collapses many IOUs into the fewest settlement payments. Borrowed
almost 1:1, since it's just local math over a ledger.

Plus two features that came from the original problem framing:
- **En-route discovery** — POIs (restaurants, viewpoints, fuel, hospitals,
  ATMs) along the *corridor between* two stops, not just at each endpoint.
- **Safety check-in** — arrival confirmation to a trusted contact, with
  escalation if no check-in by ETA + buffer. Delivered by native SMS intent,
  **not** a server relay, so it works with zero data.

---

## 4. The contacts and dialer system — read this carefully

This is the part with the most design thinking behind it, and the part
where getting it wrong is actually dangerous.

### 4.1 The request

Yash asked for all of India's emergency numbers bundled: state-wise
emergency contacts, hospitals, women helpline, nearby medical shops
(Apollo/local chemist), police, ambulance, fire, plus local
guesthouse/hotel/restaurant owner numbers for on-the-ground assistance.

He also explicitly asked to **verify the data was genuine and not
fabricated.** That instinct is correct and it drove the whole design.

### 4.2 What is actually obtainable

**Tier 1 — safe to bundle.** National short codes, verifiable from
government sources:

| Number | Service | Source |
|---|---|---|
| 112 | Single national emergency (ERSS) | 112.gov.in, MHA |
| 100 / 101 / 102 / 108 | Police / Fire / Ambulance (legacy, still active) | widely active alongside 112 |
| 181 | Women Helpline | india.gov.in helpline directory |
| 14490 | NCW 24×7 Women Helpline | ncw.gov.in |
| 1091 | Anti-Obscene Calls Cell | india.gov.in |
| 1098 | Child Helpline | india.gov.in |
| 1363 | Tourist Helpline | india.gov.in |
| 139 | Railway security / medical | india.gov.in |
| 14567 / 14456 / 14433 | Senior citizens / Disabilities / NHRC | india.gov.in |

**Four numbers are flagged `needsVerification`** in the seed file —
1930 (cyber crime), 1078 (NDMA disaster), 1033 (highway accident),
104 (health). They are widely cited on aggregator sites but were not
confirmed from a `.gov.in` source. **Verify before shipping.** 104 in
particular is state-operated and not live everywhere.

**Tier 2 — state helplines.** These exist, but there is no authoritative
combined machine-readable dataset. Populating them means manually visiting
each of 28 state + 8 UT government portals. The seed list is deliberately
**left empty**. Filling it with guesses would be worse than shipping it
empty.

**Tier 3 — individual hospitals, pharmacies, homestay owners, restaurants.
NOT OBTAINABLE as verified data, and must never be generated.** There are
thousands of such numbers, they change constantly, and no open verified
dataset covers them. A fabricated ambulance number at 2am in Kongthong is
worse than no number at all.

### 4.3 How Tier 3 is solved instead

Two honest sources:

1. **OSM `phone` / `contact:phone` tags**, pulled during the offline sync.
   Real data, but community-contributed and sparse. Must always render
   with an unverified marker.
2. **User-entered and user-confirmed contacts.** The homestay owner you
   actually called. This is the number that matters.

### 4.4 The trust tier system — the core invariant

```
verifiedNational  Govt short code. Authoritative.        -> plain
verifiedState     State govt published.                  -> plain
userVerified      User called it, it worked.             -> plain
userEntered       Typed or imported. Not confirmed.      -> amber dot
communityOsm      From OSM tags. Always unverified.      -> amber dot
```

**An imported spreadsheet row is NOT verified.** Import sets
`callConfirmed = false`. The user must actually call the number and mark it
confirmed, which promotes it to `userVerified`. This is enforced by a
pre-departure readiness check: the trip does not read as ready while
overnight-stop contacts remain unconfirmed.

**If you change one thing in this codebase, do not change this.** The whole
value of the emergency feature is that the user can tell at a glance which
numbers are guaranteed to work.

### 4.5 Bulk import

Three entry paths: single add, multi-add (repeatable inline rows), and
sheet import (CSV/XLSX, single or multiple sheets).

Import flow: pick file → parse → column-mapping screen with header
auto-match → validation pass (E.164 normalisation, duplicate detection,
invalid-row flagging) → preview with green/amber/red row states → commit
as one transaction tied to an `ImportBatch` so a bad file rolls back whole.

`stop_name` column is fuzzy-matched against the trip's Stops; unmatched
rows still import as trip-level contacts.

A CSV template with example rows exists at `assets/contact_import_template.csv`.
**The sample rows are illustrative placeholders, not real numbers.**

---

## 5. Data model

```
Trip
 └── Stop (ordered; same place may appear twice)
      └── Leg (connects consecutive Stops)
           └── Poi (corridor POIs)
      └── Poi (stop-local POIs)
      └── WeatherSnapshot
      └── ChecklistItem
 └── Contact ──── ImportBatch
 └── CallLog
 └── TimelineEntry
 └── Expense ──── ExpenseSplit
 └── TrustedContact
EmergencyHelpline  (bundled reference data, not trip-scoped)
```

Key decisions:
- `countryCode` lives on **Stop**, not Trip — a European trip crosses
  borders mid-itinerary.
- POIs attach to **either** a Stop or a Leg, never both. "What's in
  Shillong" and "what's on the road to Shillong" are different queries with
  different radii.
- Contacts store **both** `phoneRaw` (what the user typed, shown back
  verbatim) and `phoneE164` (normalised, for dialing and dedupe).

---

## 6. Offline sync flow

Triggered once per trip, on WiFi, before departure:

1. User enters the full ordered itinerary (5 stops = 4 legs).
2. "Download all" loops every leg: fetch route polyline, buffer the
   corridor (default 3 km), query Overpass for POIs by category, download
   map tiles for the corridor bbox.
3. Weather snapshot per stop for the trip dates.
4. Checklist generated per stop from activity tags.
5. Per-leg progress and sync timestamp stored, so the user can see what's
   cached and re-sync if still online.

**Data source decision: Overpass (OSM), not Google Places.** Google Places
Nearby/Text Search is $32/1k calls (Pro tier) and $35/1k once any ratings
field is included, which silently upgrades the whole call's SKU. Free tier
is 5,000 Pro / 1,000 Enterprise calls monthly, non-pooling, non-rolling.
For a route-corridor product doing category × segment queries, that burns
fast. Overpass is free. For richer data, deep-link out to the Google Maps
app instead — costs nothing.

**Two gotchas:** the standard OSM tile server discourages bulk offline
caching — use MapTiler or Stadia, whose free tiers permit it. And the
public Overpass instance is fine at personal scale but would need
self-hosting if this ever gets real users.

---

## 7. Stack

- **Flutter** (Dart) — Yash's existing stack, shared with VividVault
- **Drift** over SQLite — local database, the only persistence layer
- **flutter_map** + **flutter_map_tile_caching** — offline map rendering
- **url_launcher** — tel:, sms:, wa.me intents (no dialer permission needed)
- **file_picker** + **csv** + **excel** — sheet import, all pure Dart
- **libphonenumber_plugin** — E.164 normalisation
- **No backend.** No FastAPI. If one is ever added it must remain optional.

---

## 8. What is drafted vs what is not

**Drafted (not compiled, not run):**
- `database_schema.dart` — Trip/Stop/Leg/Poi/Checklist/Weather/Timeline/Expense
- `contacts_dialer_schema.dart` — Contacts, ImportBatches, CallLogs
- `contacts_schema_addendum.dart` — tiered helpline tables
- `emergency_contacts.dart` — Tier 1 seed data with provenance notes
- `contacts_dao.dart` — the DAO the dialer reads from
- `dialer_screen.dart` — the dialer widget
- `app_tokens.dart` — colour and type tokens
- `contact_import_template.csv` — import format

**Not started:** Flutter project scaffold, migrations, import parser,
offline sync orchestration, map screen, checklist, expenses, timeline,
check-in, everything else.

**Nothing has been built or verified.** Treat all drafted code as a
starting point that has never seen a compiler.
