# Issue #53 — Navigation shell

**Size:** S · **Depends on:** #6 · **Blocks:** #50, #54, #55

Deferred at #6 because there was only one screen and a bar with four dead
destinations is worse than no bar. There are now enough screens for it to
carry its weight.

Five items, the Material maximum, from SCREENS.md §0:

| Item | Screen |
|---|---|
| Diary | The contact diary. |
| Trip | Active trip, next leg, stops. |
| Money | Ledger and settle-up. Top level because it is entered daily. |
| SOS | Emergency. **Renders in `emergency` red only while active.** |
| More | Import, history, settings. A stub for now. |

Red on the SOS item while active is the one place red leaves the emergency
screen, and it is still pointing at it.

The shell owns the database and the trip, and hands each screen what it
needs. It keeps state per tab with an `IndexedStack`, so switching away from
a half-typed search and back does not lose it.

## Acceptance criteria

- [ ] Five destinations, the active one marked.
- [ ] SOS is red when active, muted otherwise; nothing else is ever red.
- [ ] Switching tabs and back preserves each screen's state.
- [ ] The bar sits above the system gesture inset.
- [ ] `flutter analyze` clean, tests green.
