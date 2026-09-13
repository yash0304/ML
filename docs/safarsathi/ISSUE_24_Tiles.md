# Issue #24 — Map tiles, offline

**Size:** L (split) · **Depends on:** #23 · **Provider:** MapTiler (chosen by Yash, 2026-09-13)

The map. Downloaded on WiFi at trip setup, stored on the phone, and then read
from disk with the network unreachable.

---

## The API key never enters the repository

Loaded through `String.fromEnvironment`, which Flutter fills from
`--dart-define` at build time. Three ways it arrives, none of them a file in
git:

| Where | How |
|---|---|
| Local dev | `--dart-define-from-file=maptiler.json`, gitignored |
| CI | a GitHub Actions secret, passed as `--dart-define` |
| Nowhere | the app runs, and every map surface says the key is missing |

**A missing key must not crash and must not be silent.** It disables the map
and says so in words, because a blank grey rectangle is indistinguishable
from a bug.

`String.fromEnvironment` is a compile-time constant, so the key is baked into
the binary. That is what every mobile map SDK does and it is not a secret from
whoever holds the APK. What it protects against is the key sitting in a public
git history forever. **Restrict the key to this app's package name in the
MapTiler dashboard** — that is the control that actually matters.

## The provider stays behind an interface

`MapTileProvider` describes what any raster tile source has to supply: a URL
template, a zoom range, an attribution string, and tile dimensions.
`MapTilerRaster` implements it. Nothing outside that one file mentions
MapTiler.

Swapping to Stadia later is a new implementation and one line where the
provider is chosen. **Attribution is part of the interface, not an
afterthought** — every provider's terms require it, and a map that renders
without it is a licence violation, not a styling choice.

## Why not flutter_map_tile_caching

FMTC is the obvious answer and this project is not using it.

It stores tiles in ObjectBox, which is a **second native database engine**
alongside SQLite. This app's stated invariant is that SQLite is the only
source of truth; adding a second engine for a cache means another native
build to break, another migration story, and another thing to explain.

The alternative is small enough to own: slippy-map tile arithmetic is about
forty lines, tiles are files on disk, and an index of what has been downloaded
is one more Drift table. That is the same trade this project already took for
Levenshtein, the polyline codec, and `combineLatest2`.

## Slippy-map arithmetic

The Web Mercator scheme every raster provider uses.

```
x = floor((lon + 180) / 360 · 2^z)
y = floor((1 − ln(tan φ + sec φ) / π) / 2 · 2^z)
```

Two things that are easy to get wrong:

- **Tile count grows as 4^z.** Zoom 14 over a corridor is a few hundred tiles;
  zoom 18 over the same corridor is tens of thousands. The zoom range is
  capped and **the count is shown to the user before anything downloads.**
- **Web Mercator is undefined beyond ±85.0511°.** Latitude clamps there, or
  the tangent blows up and y goes to infinity.

## Storage

Tiles are files: `<app documents>/tiles/<provider>/<z>/<x>/<y>.png`. The Drift
table `MapTiles` indexes them by (provider, z, x, y) with a byte count and a
fetch date, which is what makes the cache screen able to say a real number
rather than a guess.

**Rendering reads only from disk.** The cache-backed tile provider has no HTTP
client at all — not a fallback, not a timeout, nothing. A tile that was not
downloaded renders as blank. That is the guarantee the whole app rests on, and
the only way to be sure of it is for the code path to be absent.

## Downloading

- Zoom range **12 to 15** by default. 12 shows the region, 15 shows the
  streets of a town. Anything finer multiplies the count by four for detail
  nobody reads at a dhaba.
- **Resumable**: a tile already on disk is skipped, so a failed download
  restarts where it stopped rather than from nothing.
- **Serial, with a small delay.** A hundred parallel requests is how a free
  tier gets rate-limited. This runs once, on WiFi, and a minute is fine.
- Progress reported per tile so the screen can show it.

---

## Acceptance

- [ ] A missing key disables the map with a sentence, and never crashes.
- [ ] No key, template or secret appears in any tracked file.
- [ ] Tile maths matches known values for a known coordinate.
- [ ] Latitude clamps at the Mercator limit.
- [ ] The tile count is shown before download and matches what arrives.
- [ ] A second download of the same region fetches nothing.
- [ ] The renderer has no network path at all.
- [ ] Attribution renders on every map.
- [ ] The cache screen counts tiles and bytes; clearing removes the files too.
