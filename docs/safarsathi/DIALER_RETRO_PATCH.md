# Dialer — applying the Milestone idiom

`code/dialer_screen.dart` is carried over unchanged from the pre-code drafts.
This is the exact edit list to bring it to v2. Work top to bottom; each edit is
independent, so a session can stop halfway and the file still compiles.

**Before you start, read DESIGN_VISUAL_v2.md §0.** Two of these edits are
*removals* of things you might otherwise be tempted to decorate.

---

### 1. Read colours from the theme, not from statics

`AppTokens.signal` and friends are gone — there are two themes now. At the top
of every `build` and every widget that paints:

```dart
final c = AppTokens.of(context);
```

Then `AppTokens.signal` → `c.signal`, `AppTokens.surface` → `c.paper`,
`AppTokens.surfaceRaised` → `c.stone`, `AppTokens.muted` → `c.muted`.
Type styles stay static but no longer carry colour, so every `Text` now needs
`style: AppTokens.rowTitleStyle.copyWith(color: c.ink)`.

### 2. Wrap the body in grain

```dart
body: GrainOverlay(
  child: TabBarView(controller: _tabs, children: [...]),
),
```

Grain sits below everything. It must not appear behind the emergency tab —
`_buildEmergencyTab` returns its own `Container(color: c.paper)` to mask it.

### 3. Stencil the section headers

`_sectionHeader(String)` is replaced wholesale by `StencilLabel(text)`.
Delete the local helper.

### 4. Stencil the tab labels

```dart
tabs: [
  Tab(child: Text('CONTACTS', style: AppTokens.stencilStyle)),
  Tab(child: Text('EMERGENCY', style: AppTokens.stencilStyle)),
],
```

### 5. Hazard-stripe the readiness banner, and roll its count

The banner keeps its `cautionSoft` bed and its plain Inter text. Two changes:
a `HazardStripe()` as the leading child of the `Row`, inside the clip; and the
count becomes `RollingDigits`, so confirming a contact visibly decrements it.

The stripe is the only ephemera permitted on this element. Do not add a stamp,
a ticket edge or an icon beyond the existing `error_outline`.

### 6. Replace `InkWell` with `PressScale` in `_ContactRow`

Splashes are off app-wide. `PressScale` gives the scale and the haptic:

```dart
PressScale(
  onTap: onCall,
  onLongPress: onLongPress,
  child: Padding(...),   // row body unchanged
)
```

### 7. Swipe actions on `_ContactRow`

Wrap the `PressScale` in a `Dismissible` with `confirmDismiss` returning
`false` on both sides — the swipe performs its action and springs back rather
than removing the row.

- `background` (swipe right): `c.signal` panel, phone icon, label `CALL`.
- `secondaryBackground` (swipe left): `c.stone` panel, pin icon, label
  `PIN` / `UNPIN`.
- Fire `Haptics.light()` once when the threshold is crossed.

The whole-row tap still dials. DECISIONS.md 2026-09-11 requires that the
commonest action needs no aiming; swipe adds reach without taking it away.

### 8. The confirmation stamp

In `_showContactSheet`, the "mark as confirmed" `ListTile` currently calls
`markConfirmed` and pops. Keep exactly that, and let the row do the rest: the
Drift stream pushes the updated `Contact`, so `_ContactRow` rebuilds with
`callConfirmed == true`. Add to the row, positioned over the trailing icons:

```dart
if (contact.callConfirmed)
  StampBadge(label: 'Confirmed', landed: contact.callConfirmed),
```

`StampBadge` animates only on the false → true transition and fires
`Haptics.confirm()` at the moment of contact, so a scroll past an
already-confirmed row is silent and still.

### 9. Emergency tab — two removals

- `heavyImpact` on dial. `_dialRaw` gets `Haptics.grave()` before launching.
  Calling 112 should feel heavier than calling your homestay.
- **Nothing else changes.** No stamps, no ticket cards, no grain, no
  stencil beyond the section header from edit 3, no swipe actions. The number
  badge keeps `AppTokens.badgeStyle` on `c.emergencySoft` and that is the
  entire treatment.

### 10. Trust markers — do not touch

The amber dot in `_ContactRow` stays exactly as drafted, with one token
change: `AppTokens.caution` → `c.cautionMark`, because it is a 7px graphic
rather than text. Its size, position, tooltip and the `tier.isTrusted`
condition are unchanged.

### 11. Over-scroll cache stamp

Replace the contacts `ListView.separated` scroll physics with
`AlwaysScrollableScrollPhysics` and put a `SliverToBoxAdapter` above it
holding a `StampBadge(label: 'Cached 11 Sep · 4 legs')` revealed by
over-scroll. There is no network at runtime, so a pull-to-refresh spinner
would be a lie; the same gesture shows what is cached instead.

---

## Acceptance

- Amber dot appears on `userEntered` rows and not on `userVerified` ones,
  in both themes. Unchanged from backlog #6.
- Confirming a contact stamps the row once, clears the dot, decrements the
  banner, and fires one medium haptic.
- Scrolling a list of confirmed contacts fires no haptics and schedules no
  animation frames.
- The emergency tab contains no `StampBadge`, no `TicketCard`, no
  `GrainOverlay` and no `MilestoneMarker`. Pinned by the golden test in
  backlog #48.
- Every screen passes contrast in `AppColors.night` as well as
  `AppColors.day`.
