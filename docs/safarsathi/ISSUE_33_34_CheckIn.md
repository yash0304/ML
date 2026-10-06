# #33 Trusted contacts + check-in, #34 Check-in escalation

**Backlog lines:** #33 — SMS intent, no server relay. #34 — WorkManager
timer, alert on ETA + buffer.

## The rule both share

Nothing is sent by the app. A check-in, like an SOS, is a text written for
the person and opened in the phone's own SMS app; they press Send. A text
needs one bar and no data. No server relays anything (DECISIONS
2026-09-11). The trusted people are the SOS list: one list, for every trip.

## #33 — Check in

- **Trip page → Check in.** It opens on the stop today's leg arrives at,
  falling back to the stop the trip says you are at. Every stop is a chip.
- **The message:** "Reached Sohra safely — 17:20, 2 Oct.", then where I am
  (GPS, optional, waits at most 12 s), the map link, and the stay chosen at
  that stop.
- **Recording:** a check-in is recorded as an `arrival` in TimelineEntries
  when the message is opened. The screen says the app cannot see whether
  Send was pressed.

Acceptance:
- [x] No trusted people: the screen says where to choose them.
- [x] Texting someone opens SMS with the message and records the arrival.
- [x] Location off is never asked for.

## #34 — If nobody checks in

- **When:** for a leg with a planned arrival, a local notification at
  arrival + buffer if that stop has no check-in by then: "Reached Sohra?
  Tap to check in." A second at + 2 × buffer says that home has not heard.
  The buffer is the trusted contacts' `escalateAfterMinutes`, default 120.
- **Nothing is sent by itself.** The notification opens the check-in screen.
  An escalation that texts people automatically would fire on a phone left
  in a bag, and it would need SMS permission this app does not hold.
- **Scheduling:** the app schedules when a leg is saved and when it opens,
  and cancels a stop's reminders once that stop is checked in. No
  WorkManager: an exact-time local notification does the same job with no
  background worker.
- **Permission:** Android 13+ asks for notifications. It asks only when you
  turn reminders on, never at launch.

Acceptance:
- [ ] Which reminders are due is a pure function, pinned by tests.
- [ ] A check-in cancels its stop's reminders.
- [ ] Reminders off schedules nothing.
