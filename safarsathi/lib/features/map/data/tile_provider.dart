// lib/features/map/data/tile_provider.dart
//
// What any raster tile source has to supply — issue #24.
//
// MapTiler is what this app uses today. NOTHING OUTSIDE THIS FILE NAMES IT.
// Swapping to Stadia, Thunderforest or a self-hosted server later is a new
// implementation of this interface and one line in `activeTileProvider`.
//
// THE API KEY NEVER ENTERS THE REPOSITORY. There are two ways in and the
// repository is neither of them:
//
//   1. Typed into Settings on the phone, stored in the app's own database.
//      This is the one that works for a build somebody else produced, which
//      turned out to be the case that mattered — an APK off CI with no secret
//      configured has no way to be given a key otherwise.
//   2. `--dart-define=MAPTILER_KEY=…` at build time, read here through
//      `String.fromEnvironment`: a gitignored JSON file locally, a GitHub
//      secret in CI.
//
// The typed one wins when both are present. With neither, the map disables
// itself and says so, because a blank grey rectangle is indistinguishable
// from a bug.

/// A raster tile source.
abstract class MapTileProvider {
  /// Stable key used in the on-disk cache path and the tile index, so two
  /// providers' tiles can never be mistaken for each other.
  String get id;

  /// Shown to the user wherever the source is named.
  String get label;

  /// True when the provider has everything it needs to serve a tile. False
  /// means no key.
  bool get isConfigured;

  /// Why it is not configured, in words a person can act on.
  String get configurationHint;

  /// LEGALLY REQUIRED, on every map, by every provider's terms. Part of the
  /// interface rather than an afterthought, so a new implementation cannot
  /// forget it: a map rendering without attribution is a licence violation,
  /// not a styling choice.
  String get attribution;

  /// How many pixels across the fetched image is.
  ///
  /// NOT the same thing as [gridSize], and conflating the two is what made a
  /// complete download render blank. See [gridSize].
  int get tileSize;

  /// How much of the slippy grid one tile covers, in logical pixels.
  ///
  /// THIS IS 256 FOR THE STANDARD XYZ SCHEME AND HAS NOTHING TO DO WITH HOW
  /// MANY PIXELS THE IMAGE HAS. A retina tile is 512 pixels of detail over
  /// the same 256-pixel square of ground; the extra pixels are density, not
  /// coverage.
  ///
  /// The renderer divides the map's pixel bounds by this to work out which
  /// x and y to ask for. Handing it the image size instead halves every
  /// index, so it asks for z12 tiles using z11 numbering — coordinates
  /// pointing at ground nobody downloaded, which renders as nothing while
  /// routes and markers, positioned by the projection rather than by tiles,
  /// keep drawing correctly. That combination reads as a styling bug and
  /// cost a day.
  int get gridSize;

  int get minZoom;
  int get maxZoom;

  /// The image format tiles are fetched and stored in, without the dot.
  ///
  /// Part of the interface because the cache is files on disk and the store
  /// has to know what to name them.
  String get format;

  /// The URL for one tile. Throws when unconfigured rather than returning a
  /// URL that will 403 — a failure at the point of the mistake beats a
  /// thousand failed requests.
  String urlFor(int z, int x, int y);
}

class TileProviderNotConfigured implements Exception {
  final String message;
  const TileProviderNotConfigured(this.message);
  @override
  String toString() => message;
}

/// MapTiler's raster tiles.
///
/// The key is a compile-time constant, so it ends up in the binary. That is
/// what every mobile map SDK does and it is not a secret from whoever holds
/// the APK; what this protects against is the key sitting in a public git
/// history forever. The control that actually matters is restricting the key
/// to this app's package name in the MapTiler dashboard.
class MapTilerRaster implements MapTileProvider {
  /// Typed into Settings, or supplied at build time. Empty when neither.
  final String apiKey;

  /// MapTiler's style id. `outdoor-v2` carries terrain shading and trails,
  /// which is the right map for a road trip through hills; `streets-v2` is
  /// the flatter city alternative.
  final String style;

  /// WEBP, NOT PNG. MapTiler serves the same tiles either way, and WebP is
  /// roughly a third the size at the same visual quality — 346 MB of a
  /// Meghalaya corridor becomes something closer to 110 MB. That matters more
  /// than it sounds: it is the difference between a re-download somebody will
  /// do and one they will put off.
  ///
  /// Tiles already on disk as PNG keep working; see `TileStore.formats`.
  @override
  final String format;

  const MapTilerRaster({
    this.apiKey = buildTimeKey,
    this.style = 'outdoor-v2',
    this.format = 'webp',
  });

  /// What `--dart-define` left in the binary, or the empty string.
  static const buildTimeKey = String.fromEnvironment('MAPTILER_KEY');

  /// Deliberately NOT including the format: the cache is the same ground
  /// whichever way the bytes were encoded, and folding the format in here
  /// would strand every tile downloaded before the switch.
  @override
  String get id => 'maptiler-$style';

  @override
  String get label => 'MapTiler';

  @override
  bool get isConfigured => apiKey.isNotEmpty;

  @override
  String get configurationHint =>
      'No MapTiler key yet, so maps are off. Settings → Map key takes one; '
      'maptiler.com gives you a free one.';

  @override
  String get attribution => '© MapTiler © OpenStreetMap contributors';

  /// `@2x` in the URL asks MapTiler for the retina rendering: the same tile,
  /// twice the pixels.
  @override
  int get tileSize => 512;

  /// Still 256. The `@2x` tiles cover exactly the same ground as the plain
  /// ones — they are the standard XYZ grid at double density.
  @override
  int get gridSize => 256;

  /// 1 to 20 is what MapTiler serves. What this app actually downloads is a
  /// much narrower band; see `TileDownloader`.
  @override
  int get minZoom => 1;

  @override
  int get maxZoom => 20;

  @override
  String urlFor(int z, int x, int y) {
    if (!isConfigured) {
      throw const TileProviderNotConfigured('No MapTiler key.');
    }
    return 'https://api.maptiler.com/maps/$style/$z/$x/$y@2x.$format'
        '?key=$apiKey';
  }
}

/// A provider that serves nothing, for tests and for a build with no key.
class NoTileProvider implements MapTileProvider {
  const NoTileProvider();

  @override
  String get id => 'none';
  @override
  String get label => 'No map provider';
  @override
  bool get isConfigured => false;
  @override
  String get configurationHint => 'No map provider is configured.';
  @override
  String get attribution => '';
  @override
  int get tileSize => 256;
  @override
  int get gridSize => 256;
  @override
  int get minZoom => 0;
  @override
  int get maxZoom => 0;
  @override
  String get format => 'png';
  @override
  String urlFor(int z, int x, int y) =>
      throw const TileProviderNotConfigured('No map provider is configured.');
}

/// THE ONE LINE THAT PICKS A PROVIDER. Everything else in the app goes
/// through [tileProviderFor].
///
/// The build-time-only view of it, which is what a test sees and what the app
/// falls back to when nothing has been typed into Settings.
const MapTileProvider activeTileProvider = MapTilerRaster();

/// The provider to actually use, given whatever key Settings is holding.
///
/// The typed key wins over the build-time one. The cache id does not depend on
/// the key, so tiles already on disk are still found after the key changes —
/// which matters, because a key can be rotated and a downloaded trip must
/// survive that.
MapTileProvider tileProviderFor(String? typedKey) {
  final key = (typedKey ?? '').trim();
  if (key.isNotEmpty) return MapTilerRaster(apiKey: key);
  return activeTileProvider;
}
