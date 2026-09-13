// lib/features/map/data/tile_downloader.dart
//
// Pulling a region's tiles down at setup — issue #24.
//
// Runs once, on WiFi, behind a button that has already told the user how many
// tiles and roughly how many megabytes. Never on a timer, never on the road.

import 'dart:typed_data';

import 'package:http/http.dart' as http;

import '../../discovery/data/geo.dart';
import 'tile_math.dart';
import 'tile_provider.dart';
import 'tile_store.dart';

/// The band this app actually downloads.
///
/// 12 shows a region and its roads; 15 shows the streets of a town. Anything
/// finer multiplies the count by four for detail nobody reads at a dhaba, and
/// anything coarser is a map you cannot navigate by.
const defaultMinZoom = 12;
const defaultMaxZoom = 15;

class TilePlan {
  final int total;

  /// Already on disk from an earlier run, so a failed download resumes rather
  /// than starting over.
  final int alreadyHave;

  final int minZoom;
  final int maxZoom;

  const TilePlan({
    required this.total,
    required this.alreadyHave,
    required this.minZoom,
    required this.maxZoom,
  });

  int get toFetch => total - alreadyHave;
  int get estimatedBytes => toFetch * averageTileBytes;
  String get estimatedSize => describeBytes(estimatedBytes);
  bool get isComplete => toFetch == 0;
}

class TileProgress {
  final int done;
  final int total;
  final int failed;

  const TileProgress({
    required this.done,
    required this.total,
    this.failed = 0,
  });

  double get fraction => total == 0 ? 1 : done / total;
}

class TileDownloadException implements Exception {
  final String message;
  const TileDownloadException(this.message);
  @override
  String toString() => message;
}

class TileDownloader {
  final TileStore store;
  final MapTileProvider provider;

  /// Injected so tests never reach the network.
  final Future<Uint8List?> Function(String url) fetch;

  /// A small gap between requests. A hundred parallel fetches is how a free
  /// tier gets rate-limited; this runs once on WiFi and a minute is fine.
  final Duration delay;
  final Future<void> Function(Duration) sleep;

  TileDownloader({
    required this.store,
    required this.provider,
    Future<Uint8List?> Function(String url)? fetch,
    this.delay = const Duration(milliseconds: 40),
    Future<void> Function(Duration)? sleep,
  }) : fetch = fetch ?? _fetchOverHttp,
       sleep = sleep ?? Future.delayed;

  static Future<Uint8List?> _fetchOverHttp(String url) async {
    final response = await http.get(Uri.parse(url));
    if (response.statusCode == 200) return response.bodyBytes;
    if (response.statusCode == 401 || response.statusCode == 403) {
      throw const TileDownloadException(
        'The map provider rejected this build\'s key. Check it is valid and '
        'that it allows this app.',
      );
    }
    if (response.statusCode == 429) {
      throw const TileDownloadException(
        'The map provider is rate-limiting this download. Wait a few minutes '
        'and resume — tiles already fetched are kept.',
      );
    }
    // A missing tile — ocean, or beyond the style's coverage — is normal and
    // not an error. It simply will not render.
    return null;
  }

  /// What downloading [box] would involve, without fetching anything.
  Future<TilePlan> plan(
    BoundingBox box, {
    int minZoom = defaultMinZoom,
    int maxZoom = defaultMaxZoom,
  }) async {
    final tiles = tilesForBox(box, minZoom: minZoom, maxZoom: maxZoom);
    var have = 0;
    for (final tile in tiles) {
      if (await store.has(provider.id, tile)) have++;
    }
    return TilePlan(
      total: tiles.length,
      alreadyHave: have,
      minZoom: minZoom,
      maxZoom: maxZoom,
    );
  }

  /// Downloads every tile covering [box] that is not already on disk.
  ///
  /// Resumable by construction: a tile present on disk is skipped, so a
  /// download interrupted at 80% restarts at 80% rather than at nothing.
  Stream<TileProgress> download(
    BoundingBox box, {
    int minZoom = defaultMinZoom,
    int maxZoom = defaultMaxZoom,
  }) => downloadTiles(tilesForBox(box, minZoom: minZoom, maxZoom: maxZoom));

  /// Downloads an explicit tile list.
  ///
  /// Taken as a list rather than a box so a whole trip can be deduplicated
  /// first: adjacent legs share the terrain between their stops, and the
  /// rectangle enclosing a loop is mostly ground nobody drives through.
  Stream<TileProgress> downloadTiles(List<TileCoordinate> tiles) async* {
    if (!provider.isConfigured) {
      throw TileDownloadException(provider.configurationHint);
    }

    var done = 0;
    var failed = 0;

    yield TileProgress(done: 0, total: tiles.length);

    for (final tile in tiles) {
      if (await store.has(provider.id, tile)) {
        done++;
        yield TileProgress(done: done, total: tiles.length, failed: failed);
        continue;
      }

      try {
        final bytes = await fetch(provider.urlFor(tile.z, tile.x, tile.y));
        if (bytes != null && bytes.isNotEmpty) {
          await store.write(provider.id, tile, bytes);
        } else {
          failed++;
        }
      } on TileDownloadException {
        // A key or rate-limit problem affects every remaining tile, so
        // stopping is the only useful response. What is on disk stays.
        rethrow;
      } on Object {
        // One tile failing for any other reason is not the download failing.
        failed++;
      }

      done++;
      yield TileProgress(done: done, total: tiles.length, failed: failed);
      if (delay > Duration.zero) await sleep(delay);
    }
  }
}
