// lib/features/map/presentation/offline_tile_provider.dart
//
// Rendering from disk, and only from disk — issue #24.
//
// THERE IS NO HTTP CLIENT IN THIS FILE. Not a fallback, not a timeout, not a
// "try the network if the tile is missing". A tile that was not downloaded
// renders as nothing.
//
// That absence IS the guarantee. Any code path that could reach out is a code
// path that will, on a mountain road, with no signal, while the user waits.

import 'dart:ui' as ui;

import 'package:flutter/foundation.dart';
import 'package:flutter/widgets.dart';
import 'package:flutter_map/flutter_map.dart';

import '../data/tile_math.dart';
import '../data/tile_store.dart';

/// Serves flutter_map from the local tile cache.
class OfflineTileProvider extends TileProvider {
  final TileStore store;
  final String providerId;

  OfflineTileProvider({required this.store, required this.providerId});

  @override
  ImageProvider getImage(TileCoordinates coordinates, TileLayer options) =>
      _CachedTileImage(
        store: store,
        providerId: providerId,
        tile: TileCoordinate(coordinates.z, coordinates.x, coordinates.y),
      );
}

/// An [ImageProvider] backed by a file the downloader put there.
class _CachedTileImage extends ImageProvider<_CachedTileImage> {
  final TileStore store;
  final String providerId;
  final TileCoordinate tile;

  const _CachedTileImage({
    required this.store,
    required this.providerId,
    required this.tile,
  });

  @override
  Future<_CachedTileImage> obtainKey(ImageConfiguration configuration) =>
      SynchronousFuture<_CachedTileImage>(this);

  @override
  ImageStreamCompleter loadImage(
    _CachedTileImage key,
    ImageDecoderCallback decode,
  ) => MultiFrameImageStreamCompleter(
    codec: _load(key, decode),
    scale: 1.0,
    debugLabel: '$providerId/${key.tile}',
  );

  Future<ui.Codec> _load(
    _CachedTileImage key,
    ImageDecoderCallback decode,
  ) async {
    final bytes = await key.store.read(key.providerId, key.tile);
    if (bytes == null || bytes.isEmpty) {
      // Not an error. A tile outside the downloaded region simply is not
      // there, and the map shows the paper underneath.
      return decode(
        await ui.ImmutableBuffer.fromUint8List(_transparentPixel),
      );
    }
    return decode(await ui.ImmutableBuffer.fromUint8List(bytes));
  }

  @override
  bool operator ==(Object other) =>
      other is _CachedTileImage &&
      other.providerId == providerId &&
      other.tile == tile;

  @override
  int get hashCode => Object.hash(providerId, tile);
}

/// A 1×1 transparent PNG, for a tile that was never downloaded.
final Uint8List _transparentPixel = Uint8List.fromList(const [
  0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, //
  0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
  0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
  0x08, 0x06, 0x00, 0x00, 0x00, 0x1F, 0x15, 0xC4,
  0x89, 0x00, 0x00, 0x00, 0x0A, 0x49, 0x44, 0x41,
  0x54, 0x78, 0x9C, 0x63, 0x00, 0x01, 0x00, 0x00,
  0x05, 0x00, 0x01, 0x0D, 0x0A, 0x2D, 0xB4, 0x00,
  0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE,
  0x42, 0x60, 0x82,
]);
