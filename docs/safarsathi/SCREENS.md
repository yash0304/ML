# SafarSathi — Screen specifications v0.1

Prototype: https://claude.ai/code/artifact/548c02bc-d0ef-439d-abbc-7a5909a95d14

Six screens, specified tightly enough to build from. Tokens and motion come
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
| Trip | Active trip, next leg, stops. |
| SOS | Emergency. Renders in `emergency` red **only while active**. |
| More | Import, history, settings, cache. |

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

## Open questions

1. **Thumb index overflow.** Eleven categories do not fit a 620px edge. Either
   the index scrolls with the page (current prototype) or it collapses to the
   six categories the trip actually uses. The second is better and needs a
   query that returns categories in use.
2. **Copy feedback on a locked phone.** The toast is useless if the user copies
   and the screen sleeps before they reach the dialer. Worth checking whether
   the clipboard survives, and whether the `OPEN DIALER` button should fire
   automatically on a long press instead.
3. **`tel:` with an empty path.** Confirm on a real device that it opens the
   Android dialer rather than erroring. Fallback is `ACTION_DIAL` via a
   platform channel, which would be the first platform-specific code in the
   project.
