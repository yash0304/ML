# #47 — Night theme audit

**Backlog line:** Walk every built screen in `AppColors.night`. Check contrast
on real devices, not just on the numbers in the doc. Grain drops to 2%. The
emergency red gets no glow treatment.

## What was already true

- The night tokens pass AA as text on paper and stone (ink 15.1, muted
  6.1, signal 8.2, caution 8.3, emergency 6.7 on paper). `scaffold_test`
  pins them.
- Grain is 2% at night (`AppColors.night.grainOpacity`, pinned in
  `retro_primitives_test`).
- Nothing in the app draws a shadow or a blur, so the emergency red has no
  glow anywhere. No screen hard-codes a colour; every colour comes from
  `AppTokens.of(context)`.

## What the walk found

1. **Material's purple baseline leaked through.** The theme set five colour
   roles (surface, onSurface, primary, onPrimary, error) and left the rest
   on Material 3's default purple seed. Dialogs (32 of them), date and time
   pickers, the popup menu, the dropdown and the switch track drew in cool
   lavender greys. At night that was `#2B2930` under warm charcoal. Every
   role is now mapped onto the palette in both themes. Dialogs get the
   bottom sheet's treatment: paper with a 1 px ink border, no elevation
   tint.
2. **Form errors were red.** Material paints `errorText` in
   `colorScheme.error`, which was the emergency red: the "add a stop on the
   way" dialog put red outside the SOS tab. `error` is now caution amber.
3. **Hints were drawn in the hairline colour.** Placeholders such as
   "Taxi, Shillong to Cherrapunji" and "3200.11" were in `rule`: 1.4:1 at
   night and 1.3:1 on stone by day, so nearly invisible. They are now in
   `muted` (5.5 on stone at night, 4.8 by day). This touched 14 places
   across 10 screens. The itinerary's drag handle and the backup screen's
   arrow move from `rule` to `muted` too, to clear the 3:1 floor for graphics.

## Tests

- `test/golden_test.dart`: 14 screens are now also shot at night as
  `<name>_lamp.png`: emergency, trip, money, more, itinerary, checklist,
  expense form, settings, weather, discovery, place detail, stop detail,
  leg detail and sync. Each night shot also checks that every piece of text
  on screen is drawn in a night palette colour, which catches any default
  that slips through.
- `test/scaffold_test.dart`: every Material colour role, in both themes,
  is a palette colour. There is no surface tint, `error` is not the
  emergency red, and the snackbar action meets AA on its ground.

## Still to do by hand

"On real devices": the numbers and goldens cannot judge an OLED panel at
minimum brightness. Open the Trip page, the SOS tab and a form at night on
the phone, and report anything hard to read.
