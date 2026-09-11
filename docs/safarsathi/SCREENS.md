# SafarSathi — Screen specifications v0.2

Prototype: https://claude.ai/code/artifact/548c02bc-d0ef-439d-abbc-7a5909a95d14

Eleven screens, specified tightly enough to build from. Tokens and motion come
from `DESIGN_VISUAL_v2.md`; the exemption rule in its §0 governs all of them.

**The workflow this design serves:** Yash finds a number, taps it, it lands on
the clipboard, he opens the Android dialer and pastes. Direct dialling stays
available but is no longer the default. This reverses an earlier decision and
is logged in DECISIONS.md.

---

## 0. Navigation

Four-item bottom bar on every screen, stencil labels, 9.5pt, icon above.

| Item | Screen |
|---|---|
| Diary | The contact diary. The app's home. |
| Trip | Active trip, next leg, stops. Checklist, timeline, stop and leg detail hang off it. |
| Money | Ledger, balances and settle-up. Top level because it is entered daily. |
| SOS | Emergency. Renders in `emergency` red **only while active**. |
| More | Import, history, settings, cache. |

Five items, which is the Material maximum. Checklist and timeline sit under
Trip rather than taking a sixth slot: the checklist is used hard before
departure and at each pack-up, the timeline logs itself and is mostly read.

Red on the SOS item while active is the one place red leaves the emergency
screen, and it is still pointing at it.

---

## 1. Diary

The main screen. A pocket contact diary, not a contact list.

**Structure, top to bottom:** app bar with trip kicker, title and scope line ·
search field · readiness banner (only when something is unconfirmed) · the
diary body · page footer.

**The diary body** is a row: a 30px ruled margin, the entries, and a 30px
category thumb index pinned to the right edge.

- **Margin** — entry number, stencil 9.5pt, tabular, muted. Right border is a
  doubled hairline (border plus a 2px offset box-shadow), the way a notebook
  margin is ruled. The number gives you something to say out loud when reading
  a number to somebody.
- **Entry** — name with the trust dot, then the number at 15pt tabular, then a
  meta line. **The number is set larger than the meta and is the point of the
  row.** Trailing: a copy affordance, icon over a stencil `COPY` label.
- **Thumb index** — vertical stencil labels, `writing-mode: vertical-rl`,
  stone ground, selected tab filled `signal` with `paper` text. Categories:
  All, Stay, Move, Food, Med, Fuel, and the rest of the vocabulary below the
  fold. This is how you flip to "Stay" one-handed standing in a village.
- **Page footer** — `PAGE 1 OF 2 · 6 OF 11 ENTRIES`, stencil, tabular.

**Interaction**

| Gesture | Result |
|---|---|
| Tap anywhere on the entry | Copy the E.164 number. Light haptic. Toast. Log to CallLogs as `copy`. |
| Long press | Open the entry screen. |
| Swipe right | Call directly (`tel:` intent). |
| Swipe left | Pin to top. |

**The toast** sits above the nav bar, ink ground, paper text: the word
`Copied`, the number on its own line in tabular figures, and an `OPEN DIALER`
button. It holds 3.2 seconds. The button launches the platform dialer with an
empty field (`tel:` with no path), so the whole workflow is copy, open, paste.
Neither action needs a dialer permission.

---

## 2. Entry

One contact, as a diary page.

- App bar: back chevron, `ENTRY 03 · STAY` kicker, name as title.
- **The number is the headline** — 27pt stencil, tabular, `user-select: all`
  so a long press also selects it natively.
- Provenance line directly under it, in `caution` with the dot when
  unconfirmed, in `signal` when confirmed: *"Typed by you, 9 Sep · not
  confirmed"* → *"Confirmed by you, 11 Sep · you reached this number"*.
- **`COPY NUMBER`** full-width primary button.
- Under it, three equal ghost buttons: `DIALER`, `CALL`, `CHAT`.
- `RECORD` section, ruled fields: attached stop, note, source, last called.
- `CONFIRM` section: the explanation, the button, and the stamp landing to its
  left when pressed. This is the one animated moment in the app — see
  DESIGN_VISUAL_v2.md §5.3.

---

## 3. Emergency

**The exempt screen.** No grain, no ruled paper, no thumb index, no stamps, no
ticket edges, no swipe actions.

- App bar says plainly: *"Tap the number to call. No copying step."*
- `OFFICIAL HELPLINES` — number badge, label, and the government source under
  every line. Trailing: a muted copy button and a red call button.
- **Here the tap calls**, with `heavyImpact`. Copy is demoted to a secondary
  icon. Copy-first is a planning workflow; a copy-and-paste dance at the wrong
  moment is a liability.
- `YOUR LOCAL CONTACTS` — the trip contacts the user marked emergency-relevant.
  Separate section, visible header, never interleaved with the bundled lines.
- Footer states why state-level helplines are absent rather than leaving a
  silent gap.

---

## 4. Trip

- App bar: day counter, trip name, shape of the trip.
- `NEXT LEG` — a milestone marker beside the leg description. **The cap colour
  carries cache state**: `signal` when the leg is downloaded, `muted` when it
  is not. Never red.
- Readiness banner, same component as the diary.
- `STOPS` — one ticket card per stop, perforated bottom edge as the divider.
  Fields: nights, diary count, arrival, and a cache or weather line.
- **Staleness is stated, never hidden.** "Cached 6 d ago" on weather is the
  entire point of a snapshot model.

---

## 5. New entry

A ruled diary page as a form. Label in the stencil margin voice, value on the
line, hairline under each field.

- Fields: name, number, category chips, attach-to stop, note, reachable-on.
- **The duplicate warning does not block.** It says the number is already in
  the diary under another name and lets you save anyway.
- A `HOW IT WILL BE SAVED` block states the tier in words: unconfirmed, with
  an amber dot, until you call it and say so.

---

## 6. Import preview

- Header states the counts: rows read, ready, and needing attention.
- Each row carries a **severity stripe as well as a colour**, so state reads
  without depending on hue: `signal` for clean, `cautionMark` for a warning,
  `muted` for a skipped row. Red is not used here — it belongs to emergency.
- Warnings name the specific problem: already in your diary, no country code,
  no number on row 5.
- A `WHAT IMPORT DOES` block restates the invariant: every imported row lands
  unconfirmed, and the whole batch rolls back in one action.


---

## 7. Checklist — PackPoint

Activity-tagged pack list, generated from each stop's tags plus the number of
nights plus the cached forecast. No weather lookup at generation time.

- Progress rule under the app bar: a 3px hairline filled in `signal`.
- **`BEFORE YOU LEAVE SIGNAL`** — the blocking section. This is where the
  trust system surfaces: every unconfirmed number attached to an overnight
  stop becomes a blocking item, carrying the amber dot and a `BLOCKING`
  stencil tag. **The trip does not read ready while any of these are open.**
  "Call the homestay" therefore sits in the same list as "pack leech socks",
  which is the correct place for it.
- **`PACK`** — items with a quantity on the right and, underneath, the tags
  that produced them. You can see why the list thinks you need leech socks.
- The tick is a stencil mark in an 18px box, not a Material checkbox.
  Checked items go muted and struck through.
- A `WHY THESE` block explains the generation inputs and promises that manual
  edits survive a regeneration.

## 8. Money — Splitwise

- Total spent as the headline, per-person share under it.
- **`BALANCES` on a ticket stub** — one row per traveller, positive in
  `signal`, negative in `muted`. Never red: owing money is not an emergency.
- **`SETTLE UP` comes before the ledger**, because the question people
  actually have is who owes whom, not what was spent. Shows the simplified
  payments and states the saving: "two payments instead of five".
- **`LEDGER`** — one row per expense: what, who paid, how it split, when, and
  the amount right-aligned in stencil tabular.
- Multi-currency uses a manual rate snapshot taken at setup and always shown
  with its date. There is no live rate offline and the screen never implies
  there is.

## 9. Timeline — Polarsteps

- A logging toggle that **states the battery cost on the screen** — about 4%
  a day — rather than burying it in settings. It is the one feature here that
  genuinely drains the phone.
- **The vertical rail is the road.** Arrivals at stops are small milestone
  caps on it; notes and photos are plain dots. Structure and annotation read
  differently at a glance.
- Each arrival carries distance from the previous stop and time taken.
- GPS needs no signal, so this keeps building in a gorge with no bars. That is
  why it survived the offline constraint intact.

## 10. Stop — Windy's snapshot pattern

- `WEATHER` opens with a staleness stamp. **Under three days it is muted;
  past three days it turns `caution` and a sentence spells out what that
  means.** A stale forecast that looks current is the failure mode this whole
  screen is designed against.
- Three day rows: date, condition, min–max, rainfall, all tabular.
- `WHAT IS HERE` — diary entries, checklist, cached places, each with a count
  and a chevron.
- `ACTIVITY TAGS` are editable here **because they drive the checklist**. The
  link between the two is made visible rather than left implied.
- `CACHED HERE` — tile size and last sync date, which is also how you decide
  what to delete when the phone fills up.

## 11. Leg — Rome2Rio, reduced

- `TRANSPORT` is a ruled form: mode, departs, arrives, booked, note. **Typed
  by you, and the screen says so** instead of implying a lookup that cannot
  happen offline.
- **`ON THE ROAD`** — corridor places ordered by distance along the route,
  each on its own small milestone marker. "Coming up in 12 km" is more useful
  while moving than "0.2 km away" on a map.
- Anything carrying a number from open map data keeps the amber dot. Nothing
  community-contributed is ever presented as verified.
- `CACHED FOR THIS LEG` — route line, corridor place count, tile size.

---

## What each researched app contributed

| App | Screen | Borrowed |
|---|---|---|
| PackPoint | 07 | Activity-tagged pack lists. Extended: unconfirmed numbers become blocking items. |
| Splitwise | 08 | Shared ledger and simplify-debts. Local maths, so it works whole. |
| Polarsteps | 09 | Automatic GPS trail. Borrowed whole — GPS needs no signal. |
| Windy | 10 | The offline snapshot pattern, not live weather. |
| Rome2Rio | 11 | Mostly not borrowable. Reduced to typed transport legs plus the corridor. |

---

## Open questions

1. **Thumb index overflow.** Eleven categories do not fit a 620px edge. Either
   the index scrolls with the page (current prototype) or it collapses to the
   six categories the trip actually uses. The second is better and needs a
   query that returns categories in use.
2. **Copy feedback on a locked phone.** The toast is useless if the user copies
   and the screen sleeps before they reach the dialer. Worth checking whether
   the clipboard survives, and whether the `OPEN DIALER` button should fire
   automatically on a long press instead.
3. **Checklist regeneration.** Editing a generated item then changing an
   activity tag must not silently discard the edit. Needs a "user touched
   this" flag per item, decided at #29.
4. **Who is on the trip.** The ledger assumes named travellers with no
   accounts and no sync. Splits are local rows; settling is something people
   do with cash or UPI outside the app, and the app only records it.
5. **`tel:` with an empty path.** Confirm on a real device that it opens the
   Android dialer rather than erroring. Fallback is `ACTION_DIAL` via a
   platform channel, which would be the first platform-specific code in the
   project.
