# SafarSathi — Visual & Interaction System v2
## Codename: **Milestone**

Supersedes DESIGN.md §5 (Visual design) only. Everything else in DESIGN.md —
architecture, data model, offline strategy, and **all of §4 (trust tiers)** —
stands unchanged and outranks this document wherever they touch.

**Brief:** offline-first, highly interactive, retro yet modern.
**Constraint that shapes the answer:** the app is used one-handed, in poor
light, sometimes under stress, on a phone that may not see a charger for two
days. Retro has to survive that. Interaction has to be free.

---

## 0. The exemption rule — read this before anything else

> **Ephemera may carry category, provenance, place and delight.
> Ephemera may never carry trust.**

Trust is carried by exactly two things, and they do not change: the amber dot
beside an unconfirmed name, and plain unstyled type on everything verified.
No stamp, texture, badge, colour or animation may be introduced that a user
could mistake for a verification mark.

**Exempt surfaces — the Milestone idiom does not apply:**

| Surface | Treatment |
|---|---|
| Contact row trust markers | Exactly as DESIGN.md §4. Amber dot, nothing else. |
| Emergency tab, in full | Current austerity. No grain, no stamps, no ticket edges, no motion beyond a press state. |
| Readiness banner text | Plain. The hazard stripe on its left edge is the only ephemera permitted, and it is decoration on an already-amber card. |

A golden test pins this (backlog #48). If a future session adds a stamp to the
emergency tab, the test fails and it should.

---

## 1. Where the idiom comes from

The palette already borrowed from **Indian highway milestone markers** — white
stone, coloured cap, black stencil. v2 takes the rest of the object and the
things that travel with it.

| Source object | What it becomes |
|---|---|
| Kilometre stone — white slab, coloured cap band, stencilled distance | Screen headers; the "coming up in 14 km" marker on route discovery |
| Rubber stamp on a railway ticket or permit book | The confirmation mark. The app's one signature moment. |
| Bus / railway ticket stub — perforated edge, punched hole, ruled fields | Trip cards, leg cards, expense rows, import batches |
| Highway signage stencil — condensed, uppercase, ink-on-enamel | Section headers, milestone numerals, emergency badges |
| The paper itself — warm, toothed, never bleached white | The app's ground |

**What keeps it modern rather than costume:**

1. Every retro form is a geometric primitive — a notch, a perforation, a band,
   a 1px rule. No illustration, no photographic texture, no image assets beyond
   one 4 KB grain tile.
2. Zero skeuomorphism. No bevels, no inner shadows, no faux-paper curl, no
   drop shadows anywhere in the app.
3. Modern spacing and touch targets. 48dp minimum, a real type scale, generous
   line height. Retro objects were cramped; this is not.
4. Texture is held at 3%. It is felt, not seen.

The test for any new element: *would this have been printed, stamped or
stencilled, and is it drawable in under 40 lines of `CustomPainter`?* If no to
either, it does not belong.

---

## 2. Colour

Two complete sets. Night is not a nice-to-have — the defining use case of this
app is a village at 11pm with no signal, and a day-only theme is a functional
gap.

### 2.1 Day — "Paper"

```
                                          on paper   on stone
paper          #FAF7F0   ground                 —          —
stone          #EFEAE0   raised surfaces        —          —
rule           #D8D1C2   hairlines, perforations
ink            #191E1A   primary text        15.80      14.10
muted          #5F6963   secondary, numbers   5.32       4.75

signal         #1F6B4A   official, confirmed  6.02       5.37
signalSoft     #DCEBE3   selected chip, stamp ink bed

caution        #A35F10   caution TEXT, icons  4.67          —
cautionMark    #C77B1E   7px dot, hazard      3.13          —   graphics only
cautionSoft    #F6EBD6   readiness banner bed

emergency      #B32B23   emergency tab ONLY   5.97          —
emergencySoft  #F7DEDB
```

**These ratios are measured by a test, not estimated.** An earlier draft of
this document put `muted` at `#6B7670` and claimed 4.9:1. It actually measures
**4.41:1 on paper and 3.94:1 on stone** — a fail on the token that carries
every phone number, caption and provenance line in the app. The arithmetic was
done by hand and was wrong. `muted` is now `#5F6963`, and
`test/scaffold_test.dart` asserts every pair on both grounds so the next wrong
guess fails the build instead of shipping.

Four deliberate changes from v1, each logged in DECISIONS.md:

- **Ground moves off pure white.** This is the single highest-leverage retro
  move in the whole system and it costs nothing — ink on paper still reads
  15:1. Every other change follows from it.
- **`caution` splits in two.** The original `#C77B1E` measures 3.1:1 on warm
  paper, which fails AA for text. Caution *text and icons* now use `#A35F10`
  at 4.7:1; the 7px dot and the hazard stripe keep the brighter ochre, because
  they are graphics held to the 3:1 non-text floor and need to read amber at
  seven pixels.
- **Soft tints warmed** so they sit on paper rather than floating on it.
- **`muted` darkened** from `#6B7670` to `#5F6963` after measurement, see above.

`signal`, `emergency` and the hue meanings are untouched. Green still means
go and confirmed, amber still means unverified, **red still appears on the
emergency tab and nowhere else.**

### 2.2 Night — "Lamp"

Warm charcoal, not neutral black. Black would be cheaper on OLED and would
also throw away the whole paper premise; the warmth is what keeps it the same
app after dark.

```
paper          #14120E   ground
stone          #1F1C17   raised
rule           #332E26   hairlines
ink            #EDE6D7   primary text                  15.1:1
muted          #9A9287   secondary text                 6.1:1

signal         #4FBF8B   official, confirmed            8.2:1
signalSoft     #17301F

caution        #E0A040   caution text and icons         8.3:1
cautionMark    #E0A040   dot and stripe — same value, the ground carries it
cautionSoft    #33260F

emergency      #FF6B5E   emergency tab ONLY             6.7:1
emergencySoft  #3A1512
```

Night rules: grain drops to 2%; the emergency tab's red gets *no* glow or
bloom treatment; the theme follows the system setting with a manual override
in Settings, because a phone in a pocket does not know it is night in a valley.

### 2.3 Elevation

**There are no drop shadows in this app.** Print does not have them, and every
Material default that adds one is overridden. Separation is carried by a 1px
`rule` line and a ground shift from `paper` to `stone`.

The two elements that genuinely need to float — the FAB and modal bottom
sheets — take a **1px `ink` border** instead. It reads as a cut edge, which is
the correct metaphor, and it survives both themes without a shadow colour.

---

## 3. Typography

### 3.1 Two families, sharply divided

DESIGN.md §5 said one family and called a second one decoration. That was
right for a pure-utility brief and is wrong for this one — a retro direction
without a voice is just a beige repaint. The rule that keeps it honest:

> **Archivo Narrow speaks. Inter reads.**
> Anything a user has to read as a sentence is Inter. Anything stencilled
> onto an object — a label, a number on a marker, a stamp — is Archivo Narrow.

| Role | Family | Spec | Where |
|---|---|---|---|
| `stencilStyle` | Archivo Narrow 700 | 13 / 1.0, +1.2 tracking, UPPERCASE | Section headers, field labels, tab labels |
| `milestoneStyle` | Archivo Narrow 700 | 28 / 1.0, tabular | The km numeral on a milestone marker |
| `stampStyle` | Archivo Narrow 700 | 11 / 1.0, +1.6 tracking, UPPERCASE | Stamp marks: CONFIRMED, CACHED, IMPORTED |
| `badgeStyle` | Archivo Narrow 700 | 14 / 1.0, tabular | Emergency number badges |
| `titleStyle` | Inter 600 | 16 / 1.3 | Screen and row titles |
| `rowTitleStyle` | Inter 500 | 15 / 1.3 | Contact names |
| `numberStyle` | Inter 400 | 14, **tabular figures** | Every phone number. Unchanged from v1. |
| `captionStyle` | Inter 400 | 12.5 / 1.35 | Notes, provenance, staleness |

Both faces are SIL OFL and bundle offline. Archivo Narrow is a grotesque like
Inter, so the pair reads as one voice at two volumes rather than as a collision
— which is exactly what a poster face like Bebas would have caused.

### 3.2 Tabular figures stay the point

v1's best type decision, kept and extended: every digit the user might compare
down a column gets `FontFeature.tabularFigures()`. Phone numbers, emergency
badges, milestone distances, expense amounts, km readings. This is also what
makes the digit-roll animation in §5.3 possible without the row width jittering.

---

## 4. Geometry, texture, spacing

```
radiusSharp   0    tickets, stamps, milestone caps, hazard stripes — paper is cut
radiusSoft    4    chips, category avatars, search field, sheets
hairline      1    every rule, every border
gutter        16   screen side padding, never less
spacing       4 · 8 · 12 · 16 · 24 · 32
tapTarget     48   minimum, always
```

**Grain.** One 128×128 tiled PNG, under 4 KB, drawn at 3% opacity (2% at
night) over `paper` only. Three hard rules: never above text, never on the
emergency tab, never animated.

**Perforation.** Semicircular notches, r = 4, pitch 10, painted along a ticket
edge by `PerforationPainter`. It is the app's divider motif and it replaces
`Divider` on ticket-shaped objects only — dense contact lists keep a plain
hairline, because a perforated list would be noise.

**Hazard stripe.** 6px, 45° `cautionMark` bars on `cautionSoft`. Appears on
exactly one element: the left edge of the readiness banner. Rationed the same
way red is.

---

## 5. Interaction — "tactile, cheap"

The governing constraint:

> **Nothing animates at rest.**
> If the user is not touching the screen and no state has just changed, zero
> frames are scheduled. An offline trip app runs for days between charges; an
> idle animation is a battery cost with no user attached to it.

That rules out ambient motion, looping textures, animated map layers and
parallax. What is left — and what actually makes an app feel alive — is
response: every touch answers immediately, in the hand as well as on screen.

### 5.1 Motion constants

```
instant  90ms    press-in
quick   140ms    chip swap, cross-fade, dot clear
base    220ms    route transition, list reorder
sheet   280ms    modal sheet, spring
stamp   380ms    the confirmation stamp, spring with overshoot

standard  Curves.easeOutCubic
spring    SpringDescription(mass: 1, stiffness: 380, damping: 26)
press     scale 0.97 in over `instant`, back over `quick`, uniform on every tappable
```

`MediaQuery.disableAnimations` collapses every duration to zero, with one
exception: the stamp becomes a `quick` cross-fade rather than vanishing, so
the user still gets a visible confirmation. **Haptics fire regardless of the
reduce-motion setting** — they are feedback, not animation.

### 5.2 Haptic vocabulary

Fixed, and small enough to stay meaningful.

| Event | Feedback |
|---|---|
| Chip, tab or filter change | `selectionClick` |
| Row press-in; sheet open | `lightImpact` |
| Pin toggle; swipe threshold crossed | `lightImpact` |
| **Confirmation stamp lands**; import commit | `mediumImpact` |
| **Dialing an emergency number** | `heavyImpact` |
| Destructive confirm — import rollback, delete | `heavyImpact` |
| Validation error on an import row | two `lightImpact`, 80ms apart |

Never on scroll. Never on a stream rebuild. Never when data merely arrives.

### 5.3 The six interactions worth building

**1. The Confirmation Stamp — the signature moment.**
`userEntered → userVerified` is the single action the entire trust system
depends on. It currently closes a sheet and silently changes a boolean. It
should feel like a stamp landing in a permit book.

On tap in the long-press sheet: the sheet springs shut; the row scales
0.98 → 1.00; a `CONFIRMED` stamp rotates in from −8° to −3° with a spring
overshoot over `stamp`, its ink settling from 0 → 0.85 opacity; `mediumImpact`
fires at the instant of contact, not at animation start; the amber dot
cross-fades out over `quick`; the readiness banner's count rolls down a digit.
Total 380ms, runs once, then the screen is still again.

This is where the interaction budget is spent. Everything else is restraint.

**2. Row swipe actions.** Swipe right reveals a `signal` ticket panel — call.
Swipe left reveals a `stone` panel — pin. `lightImpact` when the threshold is
crossed, so the commit point is felt rather than guessed. The whole-row tap
still dials, exactly as DECISIONS.md requires; swipe adds reach without taking
anything away.

**3. Digit roll.** The readiness count, expense totals and km figures animate
by rolling digits over `quick`. Cheap, legible, and only possible because the
numbers are already tabular.

**4. Over-scroll cache stamp.** There is no network at runtime, so
pull-to-refresh would be a lie. The same gesture instead drags a stamp into
view at the top of the list: `CACHED 11 SEP · 4 LEGS · 62 MB`. A dead gesture
turned into the one piece of information an offline user actually wants.

**5. Shared-element sheet.** The category avatar flies from its row into the
detail sheet header over `base`. Confirms which row was opened, which matters
on a dense list.

**6. Milestone sticky headers.** On route discovery, the list is grouped by
distance along the leg and each group header is a milestone marker with the km
stencilled on its cap. Headers stick and swap as you scroll. No parallax.

---

## 6. Components

| Component | Form | Notes |
|---|---|---|
| `MilestoneHeader` | Stone slab, rounded top, coloured cap band, stencilled numeral | Screen headers; route-discovery group headers |
| `TicketCard` | Perforated edge, ruled fields, punched hole | Trips, legs, expenses, import batches |
| `StampBadge` | Rotated, ink-bled, uppercase stencil | `CONFIRMED` · `CACHED` · `IMPORTED` · `ROLLED BACK` |
| `StencilLabel` | Uppercase Archivo Narrow, +1.2 tracking, hairline under | Every section header |
| `HazardStripe` | 6px 45° bars | Readiness banner only |
| `ChipRail` | Stone chip, hairline, ink fill when selected | Category filters |
| `PressScale` | 0.97 scale + haptic | Wraps every tappable in the app |
| `RollingDigits` | Tabular digit roll | Counts and totals |
| `GrainOverlay` | 3% tiled noise | Root-level, below all text |
| `ContactRow` | **Structurally unchanged** | New ground and swipe only. Trust dot untouched. |
| `EmergencyRow` | **Wholly exempt** | No ephemera of any kind. |

Code for the first nine is in `code/app_tokens.dart`, `code/motion.dart` and
`code/retro.dart`. None of it has been compiled — same caveat as every other
drafted file in this project.

---

## 7. Screen-by-screen

| Screen | Idiom applied |
|---|---|
| Home / Active trip | Milestone header for the current leg; ticket card per upcoming stop; cache stamp with `lastSyncedAt` |
| Dialer — contacts tab | Paper ground, stencil section headers, chip rail, swipe actions, stamp on confirm. Trust dot untouched. |
| Dialer — emergency tab | **Exempt.** Unchanged from the drafted screen but for the paper ground. |
| Itinerary builder | Ticket card per stop, perforation between them; drag handle is a punched hole |
| Offline sync | Per-leg ticket with a progress rule that fills left to right; `CACHED` stamp lands per leg on completion |
| Route discovery | Milestone group headers by km along the leg; POI rows carry an unverified dot, same rules as contacts |
| POI detail | Ticket card; OSM number in `numberStyle` with the amber dot and provenance caption |
| Checklist | Ruled paper lines; check is a stencil tick, not a Material checkbox |
| Expenses | Ticket stub per entry; totals in tabular Archivo Narrow; settlement summary as a stamped receipt |
| Timeline | Vertical rule as the road, stops as milestone caps along it |
| Sheet import | Preview rows keep green/amber/red states from BACKLOG #13. `IMPORTED` stamp on commit, `ROLLED BACK` on undo. |

---

## 8. Accessibility and performance

- Every colour pair above is measured **by a test that runs on every build**,
  not by hand. Text meets 4.5:1 on both `paper` and `stone`, graphics meet
  3:1, in both themes. `cautionMark` is graphics-only precisely because it
  does not clear the text bar.
- The stamp carries a text label, not just a graphic, and is announced to
  screen readers as "Confirmed". The amber dot keeps its existing tooltip.
- Grain never sits above text. Contrast is measured on the flat ground, and
  3% noise does not move it meaningfully, but nothing depends on that.
- `disableAnimations` is honoured everywhere; haptics are not suppressed by it.
- Budget: one 4 KB grain tile, two font families, no other image assets, no
  continuously scheduled frames, target 60fps on a 2019 mid-range Android.

---

## 9. What this deliberately does not do

- **No dark-mode-only neon, no CRT, no scanlines.** Retro here means printed
  paper, not a screen pretending to be older hardware.
- **No parallax, no animated route drawing, no particles.** All ruled out by
  the motion budget, and all of them look worse on a 3-year-old phone.
- **No coloured category chips.** DECISIONS.md 2026-09-11 still holds: colour
  is reserved for trust meaning. Category is a monochrome icon, now on stone.
- **No card-per-contact.** The dense list still scans faster. Ticket cards are
  for objects you consider one at a time — a trip, a leg, an expense — not for
  a list you scan under stress.
- **No second accent hue.** The palette has three meanings and it keeps three.
