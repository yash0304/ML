# SafarSathi — Design Document

Companion to PROJECT_RUNDOWN.md. That one says *what* was decided; this one
says *how it is built and why it looks the way it does*.

---

## 1. Architectural position

```
┌─────────────────────────────────────────────┐
│  Presentation   Flutter widgets             │
│                 StreamBuilder over DAOs     │
├─────────────────────────────────────────────┤
│  Data           Drift DAOs (one per domain) │
│                 ContactsDao, TripsDao, ...  │
├─────────────────────────────────────────────┤
│  Persistence    SQLite via Drift            │
│                 THE ONLY SOURCE OF TRUTH    │
├─────────────────────────────────────────────┤
│  Sync (setup)   Overpass │ tiles │ weather  │
│                 RUNS ONCE, ON WIFI, ONLY    │
└─────────────────────────────────────────────┘
```

There is no repository layer between DAO and widget yet. That is deliberate
— it can be introduced when a second consumer of the same data appears, not
before. Adding it now would be ceremony.

**The sync layer writes into SQLite and then disappears.** Nothing in the
presentation layer may ever call it directly, and no widget may make a
network call. If you find yourself writing `http` inside a screen, stop.

### State management

Deliberately absent for now. The dialer uses `StreamBuilder` directly over
Drift streams, which are already reactive. When a screen needs cross-screen
state (the sync orchestrator will), introduce Riverpod then — the DAO layer
won't need to change. See DECISIONS.md 2026-09-11.

---

## 2. Data model rationale

### Stop as occurrence, not place

The Meghalaya route visits Shillong twice; the South India route visits Ooty
and Bangalore twice each. Modelling Stops as unique places would force
awkward special-casing for arrival dates, checklists and contacts.

So: **each visit is its own Stop row** with its own `sequenceOrder`,
`arrivalDate`, `activityTags` and attached POIs. Two Shillong rows is the
correct representation, not a bug.

### Leg as first-class entity

The original problem was "what's between A and B", so the connection needs
to hold data: transport mode, planned times, route polyline, corridor radius,
and its own cached POIs. A Leg is not just a pointer between two Stops.

### POI attaches to Stop XOR Leg

`Poi.stopId` and `Poi.legId` are both nullable; exactly one is set.
Stop-attached POIs answer "what's in Shillong" (small radius, dense).
Leg-attached POIs answer "what's on the road to Shillong" (corridor buffer,
sparse, ordered by distance along route). Merging them would make both
queries wrong.

### countryCode on Stop

A German trip crosses into Austria mid-itinerary. Emergency numbers,
currency and language all switch at that boundary, not at trip boundary.

### phoneRaw and phoneE164 both stored

`phoneRaw` is displayed — the user typed it, they recognise it, and
reformatting someone's own input is disrespectful and confusing.
`phoneE164` is used for dialing and duplicate detection. Normalisation can
fail (bad input, unknown country); the raw value must survive that.

---

## 3. Offline strategy

Three categories of data, three strategies:

| Data | Strategy |
|---|---|
| Trip, stops, legs, contacts, checklists, expenses, timeline | Native local. Created on-device, never needs network. |
| POIs, map tiles, weather, route polylines | Cached snapshot. Fetched once at setup, static thereafter. |
| Emergency helplines | Bundled with the app binary. Seeded on first launch. |

**Staleness is surfaced, not hidden.** Every cached item carries a
`lastSyncedAt` / `cachedAt`. Weather in particular degrades fast — a
five-day-old forecast should be labelled as such, not presented as current.

**Seeding must be idempotent.** First-launch seeding of emergency helplines
will run again after a reinstall or a migration; it must not duplicate rows.

---

## 4. The trust tier system

The single most important design decision in the app.

```
TIER              MEANING                        TREATMENT
verifiedNational  Govt short code                plain, no marker
verifiedState     State govt published           plain, no marker
userVerified      User called it, it worked      plain, no marker
userEntered       Typed or imported              amber dot
communityOsm      From OSM tags                  amber dot
```

### Rules

1. A number's tier is visible without tapping. The amber dot sits next to
   the name in the list row.
2. `userEntered → userVerified` happens only through the explicit "mark as
   confirmed" action, which the UI labels *"Only after you have actually
   called it."*
3. Bulk import cannot bypass this. Imported rows land as `userEntered`.
4. The emergency tab renders bundled helplines and user contacts in
   **separate sections with a visible header**. They are never interleaved.
5. Every bundled helpline shows its `sourceNote` in the subtitle. The user
   can see where the number came from.

### Why this matters more than it looks

The failure mode this prevents: user glances at the emergency screen at
night in a village with no signal, dials the first hospital number they
see, and it's a five-year-old OSM tag that rings a disconnected line. The
amber dot is the difference between a tool and a liability.

---

## 5. Visual design

> **SUPERSEDED 2026-09-11 by `DESIGN_VISUAL_v2.md` ("Milestone").**
> Read this section for the reasoning, which still holds, then take the
> tokens and rules from v2. Three things changed: the ground is warm
> paper rather than pure white, a second typeface is admitted for
> stencil moments only, and `caution` splits into a text token and a
> graphics token. Everything else in this document, and all of §4,
> outranks v2.

### Palette rationale

Anchored on Indian highway milestone markers — white stone with a coloured
cap. On the road, hue already carries meaning, so the app borrows that
vocabulary rather than inventing an arbitrary brand palette.

```
surface        #FFFFFF   base
surfaceRaised  #F1F4F1   chips, avatars, search field
ink            #1A211C   primary text
muted          #6B7670   secondary text, numbers, inactive icons

signal         #1F6B4A   official, confirmed, primary action
signalSoft     #DCEBE3   selected chip background

caution        #C77B1E   unverified, unconfirmed
cautionSoft    #FBF0DD   readiness banner background

emergency      #B32B23   emergency tab ONLY
emergencySoft  #FAE3E1
```

**Red is rationed.** It appears on the emergency tab and nowhere else. The
moment red starts marking validation errors or delete buttons, it stops
meaning emergency and the emergency tab loses its urgency.

### Typography

One family (Inter) throughout. No display/body split — this is a utility,
not a brochure, and a second typeface would be decoration.

**Phone numbers use tabular figures.** This is the one genuinely
subject-driven type decision: digits align down the column, so a number can
be read at a glance rather than parsed character by character. Applied to
`numberStyle` and to the emergency tab's number badges.

### Layout principles

The dialer is used one-handed, often in poor light, sometimes under stress.
That drives four rules:

1. **Tap target is the whole row, and it dials.** The most common action
   needs no aiming. Secondary actions (WhatsApp, SMS) are explicit icons;
   destructive and organisational actions are behind a long-press sheet.
2. **Stop scoping is one toggle.** Standing in Kongthong, you should not
   scroll past Bangalore contacts. The toggle in the app bar filters to the
   current stop while keeping trip-wide contacts (your driver isn't tied to
   one stop, but you still need him).
3. **The readiness banner only appears when there is something unconfirmed.**
   A permanent banner becomes wallpaper and stops being read.
4. **Empty states give direction.** "No contacts saved for this trip yet"
   plus an import button, not a shrug.

### What was deliberately avoided

No card-per-contact layout — cards waste vertical space and a dense list
scans faster. No gradient headers. No coloured category chips; category is
carried by a monochrome icon avatar so colour stays reserved for tier
meaning, which is the information that actually matters here.

---

## 6. Screen inventory

| Screen | Status | Purpose |
|---|---|---|
| Dialer | drafted | Trip-scoped contact directory + emergency tab |
| Add / Edit contact | not started | Single contact entry |
| Multi-add | not started | Repeatable inline rows |
| Sheet import | not started | Pick → map columns → validate → preview → commit |
| Import history | not started | Past batches, rollback |
| Home / Active trip | not started | Current leg, ETA, quick actions |
| Itinerary builder | not started | Ordered stop list, reorder, add legs |
| Offline sync | not started | Per-leg download progress |
| Route discovery | not started | Map + list of corridor POIs |
| POI detail | not started | Details, distance off route, save as contact |
| Checklist | not started | Per-stop, activity-tagged |
| Timeline | not started | Auto-logged GPS trail + notes/photos |
| Expenses | not started | Local ledger + simplify-debts |
| Check-in | not started | Arrival confirm, SMS to trusted contact |
| Settings | not started | Data retention, cache management |

---

## 7. Non-negotiables

Carried from Yash's existing project conventions, plus ones specific here:

- Never edit a shipped migration. No destructive migration fallback in
  release builds.
- Offline-first: network is a setup-time enhancement, never a runtime
  dependency.
- All UI values from `AppTokens`. No hard-coded colours or sizes in screens.
- Seeding and bootstrap logic must be idempotent.
- **No emergency number ships without a government-domain source.** Anything
  unverified carries `needsVerification: true` and does not reach the UI.
- **Trust tier must remain visible in the list row.** Not behind a tap.
- Decisions of consequence go into DECISIONS.md the day they are made.
