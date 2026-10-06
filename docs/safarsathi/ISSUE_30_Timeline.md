# #30 — Timeline and route log

**Backlog line:** Background location service, Polarsteps-style. The rail is
the road: stops render as milestone caps, notes and photos as plain dots.
**The logging toggle states its battery cost on screen.** *Split before
starting.*

## The split

- **30a — the timeline itself.** A screen that tells the trip as it went:
  each day, the arrivals (check-ins from #33) as milestone caps, and notes
  and photos as plain dots. Add a note, with a photo if you like, from the
  screen. It works with logging off, because check-ins and notes are enough.
- **30b — the route log.** While switched on, the phone's GPS position is
  stored as `fix` rows, thinned to one per 150 m or 10 minutes. Each arrival
  shows the distance driven since the last one. The trip map draws the
  track.

## Location: no background permission, still

DECISIONS (23 Sep) says location is used "only while the app is open; no
background location". Route logging keeps that. It is something the person
starts, and while it runs a notification says "SafarSathi is logging your
route". Android treats that as a foreground service: location in use, not in
the background. So no ACCESS_BACKGROUND_LOCATION is asked for. It keeps
logging with the screen off, and stops when switched off or when the app is
closed from Recents. The switch says it costs about 4% of the battery a day,
and more on a long drive with the screen off.

## Photos

Picked through the system photo picker or taken with the camera app; no
storage or camera permission. They are copied into the app's own folder and
stay on the phone. **They are not in the JSON backup**, and the screen says
so. The timeline rows are.

## Acceptance

- [x] Days in order. Arrivals are caps, notes and photos are dots, and fixes
      are not shown one by one.
- [x] Distance since the previous arrival comes from the logged track when
      there is one, else the straight line, labelled as such.
- [x] Thinning is a pure function, pinned by tests.
- [x] The logging switch states the battery cost; turning it on asks for
      location only then.
- [x] The trip map draws the logged track.
