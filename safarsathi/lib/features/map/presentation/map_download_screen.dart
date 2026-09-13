// lib/features/map/presentation/map_download_screen.dart
//
// The screen that says what it is about to fetch, then fetches it — issue #24.
//
// THE ESTIMATE COMES FIRST AND IT IS A REAL NUMBER. An app that starts pulling
// data on someone's hotel WiFi is an app they stop trusting the moment they
// notice. Nothing here runs until a button is pressed.

import 'package:flutter/material.dart';

import '../../../core/theme/app_tokens.dart';
import '../../../core/theme/motion.dart';
import '../../../core/widgets/retro.dart';
import '../data/map_download.dart';
import '../data/tile_downloader.dart';
import '../data/tile_math.dart';
import '../data/tile_provider.dart';

class MapDownloadScreen extends StatefulWidget {
  final MapTileProvider provider;
  final Future<MapEstimate> Function() estimate;
  final Stream<TileProgress> Function() download;

  /// Live cache usage, so the number on screen is the real one after a clear.
  final Stream<({int count, int bytes})> usage;
  final Future<void> Function() onClear;

  const MapDownloadScreen({
    super.key,
    required this.provider,
    required this.estimate,
    required this.download,
    required this.usage,
    required this.onClear,
  });

  @override
  State<MapDownloadScreen> createState() => _MapDownloadScreenState();
}

class _MapDownloadScreenState extends State<MapDownloadScreen> {
  late Future<MapEstimate> _estimate = widget.estimate();

  /// Held as a broadcast stream, subscribed once.
  ///
  /// The usage row lives inside a ListView, whose children are disposed and
  /// rebuilt freely. A single-subscription stream throws "already listened to"
  /// the second time that happens. Drift's own streams are broadcast so this
  /// never bit in production, but a screen that only works with one kind of
  /// stream is a trap for the next caller.
  late final Stream<({int count, int bytes})> _usage = widget.usage
      .asBroadcastStream();

  TileProgress? _progress;
  String? _error;
  bool _running = false;

  Future<void> _run() async {
    if (_running) return;
    setState(() {
      _running = true;
      _error = null;
    });

    try {
      await for (final progress in widget.download()) {
        if (!mounted) return;
        setState(() => _progress = progress);
      }
      if (!mounted) return;
      Haptics.confirm();
      setState(() => _estimate = widget.estimate());
    } on Object catch (e) {
      if (!mounted) return;
      Haptics.reject();
      // What is already on disk stays; a resumed download picks up there.
      setState(() => _error = '$e');
    } finally {
      if (mounted) setState(() => _running = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);

    return Scaffold(
      backgroundColor: c.paper,
      appBar: AppBar(
        title: const Text('Offline map'),
        backgroundColor: c.paper,
        foregroundColor: c.ink,
        elevation: 0,
      ),
      body: ListView(
        padding: const EdgeInsets.only(bottom: 96),
        children: [
          if (!widget.provider.isConfigured)
            Padding(
              padding: const EdgeInsets.fromLTRB(
                AppTokens.gutter,
                AppTokens.s16,
                AppTokens.gutter,
                0,
              ),
              child: Text(
                widget.provider.configurationHint,
                style: AppTokens.captionStyle.copyWith(color: c.cautionMark),
              ),
            ),

          const StencilLabel('What this would download'),
          FutureBuilder<MapEstimate>(
            future: _estimate,
            builder: (context, snap) {
              final estimate = snap.data;
              if (estimate == null) {
                return Padding(
                  padding: const EdgeInsets.symmetric(
                    horizontal: AppTokens.gutter,
                  ),
                  child: Text(
                    'Working it out…',
                    style: AppTokens.captionStyle.copyWith(color: c.muted),
                  ),
                );
              }
              return _Estimate(estimate: estimate);
            },
          ),

          if (_progress != null) ...[
            const StencilLabel('Progress'),
            _Progress(progress: _progress!),
          ],

          if (_error != null)
            Padding(
              padding: const EdgeInsets.fromLTRB(
                AppTokens.gutter,
                AppTokens.s16,
                AppTokens.gutter,
                0,
              ),
              child: Text(
                '$_error\n\nWhat downloaded so far is kept. Press download '
                'again to carry on from there.',
                style: AppTokens.captionStyle.copyWith(color: c.cautionMark),
              ),
            ),

          const StencilLabel('On this phone'),
          StreamBuilder<({int count, int bytes})>(
            stream: _usage,
            builder: (context, snap) {
              final usage = snap.data;
              return Padding(
                padding: const EdgeInsets.symmetric(
                  horizontal: AppTokens.gutter,
                ),
                child: Row(
                  children: [
                    Expanded(
                      child: Text(
                        usage == null || usage.count == 0
                            ? 'No map tiles stored.'
                            : '${usage.count} tiles · '
                                  '${describeBytes(usage.bytes)}',
                        style: AppTokens.captionStyle.copyWith(color: c.muted),
                      ),
                    ),
                    if (usage != null && usage.count > 0)
                      GestureDetector(
                        onTap: () async {
                          Haptics.grave();
                          await widget.onClear();
                          if (mounted) {
                            setState(() => _estimate = widget.estimate());
                          }
                        },
                        child: Text(
                          'CLEAR',
                          style: AppTokens.stencilStyle.copyWith(
                            fontSize: 9.5,
                            color: c.muted,
                          ),
                        ),
                      ),
                  ],
                ),
              );
            },
          ),

          const StencilLabel('How this works'),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: AppTokens.gutter),
            child: Text(
              'Tiles are fetched once, here, on WiFi, and then read from this '
              'phone. The map never asks the network again — not as a '
              'fallback, not on a timer. Anywhere you did not download shows '
              'as blank rather than as a spinner.\n\n'
              'Zoom $defaultMinZoom to $defaultMaxZoom: enough to see the '
              'region and the streets of a town. Finer than that multiplies '
              'the download by four for detail nobody reads at a dhaba.\n\n'
              'Map data from ${widget.provider.label} and OpenStreetMap.',
              style: AppTokens.captionStyle.copyWith(color: c.muted),
            ),
          ),
        ],
      ),
      bottomNavigationBar: SafeArea(
        minimum: const EdgeInsets.all(AppTokens.s16),
        child: FutureBuilder<MapEstimate>(
          future: _estimate,
          builder: (context, snap) {
            final estimate = snap.data;
            final ready =
                widget.provider.isConfigured &&
                !_running &&
                estimate != null &&
                estimate.toFetch > 0;

            return PressScale(
              onTap: ready ? _run : null,
              child: Container(
                height: 48,
                alignment: Alignment.center,
                decoration: BoxDecoration(
                  color: ready ? c.signal : c.stone,
                  border: Border.all(color: ready ? c.ink : c.rule),
                  borderRadius: BorderRadius.circular(AppTokens.radiusSoft),
                ),
                child: Text(
                  _running
                      ? 'Downloading…'
                      : !widget.provider.isConfigured
                      ? 'No map provider'
                      : estimate == null
                      ? 'Working it out…'
                      : estimate.hasNothingToDo
                      ? 'Nothing to download yet'
                      : estimate.isComplete
                      ? 'Already downloaded'
                      : 'Download ${estimate.toFetch} tiles '
                            '(${estimate.estimatedSize})',
                  style: AppTokens.stencilStyle.copyWith(
                    fontSize: 11.5,
                    color: ready ? c.paper : c.muted,
                  ),
                ),
              ),
            );
          },
        ),
      ),
    );
  }
}

class _Estimate extends StatelessWidget {
  final MapEstimate estimate;
  const _Estimate({required this.estimate});

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);

    if (estimate.hasNothingToDo) {
      return Padding(
        padding: const EdgeInsets.symmetric(horizontal: AppTokens.gutter),
        child: Text(
          estimate.legCount == 0
              ? 'This trip has no legs yet. Add at least two stops.'
              : 'None of the legs have coordinates yet. Set them on each '
                    'stop first — without them there is nothing to cover.',
          style: AppTokens.captionStyle.copyWith(color: c.cautionMark),
        ),
      );
    }

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: AppTokens.gutter),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              _Figure(label: 'Tiles', value: '${estimate.tileCount}'),
              _Figure(label: 'Have', value: '${estimate.alreadyHave}'),
              _Figure(label: 'To fetch', value: '${estimate.toFetch}'),
              _Figure(label: 'About', value: estimate.estimatedSize),
            ],
          ),
          const SizedBox(height: AppTokens.s8),
          Text(
            // Said out loud, because a wrong guess about size on a metered
            // connection is the kind of thing people do not forgive.
            'The size is an estimate — a tile of empty hillside is small and '
            'a town is not.',
            style: AppTokens.captionStyle.copyWith(color: c.muted),
          ),
          if (estimate.legsWithoutCoordinates > 0)
            Padding(
              padding: const EdgeInsets.only(top: AppTokens.s8),
              child: Text(
                '${estimate.legsWithoutCoordinates} '
                '${estimate.legsWithoutCoordinates == 1 ? 'leg is' : 'legs are'}'
                ' skipped for want of coordinates on their stops.',
                style: AppTokens.captionStyle.copyWith(color: c.cautionMark),
              ),
            ),
        ],
      ),
    );
  }
}

class _Figure extends StatelessWidget {
  final String label;
  final String value;
  const _Figure({required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    return Expanded(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            value,
            style: AppTokens.numberStyle.copyWith(color: c.ink, fontSize: 17),
          ),
          Text(
            label.toUpperCase(),
            style: AppTokens.stencilStyle.copyWith(fontSize: 9, color: c.muted),
          ),
        ],
      ),
    );
  }
}

class _Progress extends StatelessWidget {
  final TileProgress progress;
  const _Progress({required this.progress});

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: AppTokens.gutter),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          SizedBox(
            height: 3,
            child: Row(
              children: [
                Expanded(
                  flex: (progress.fraction * 1000).round(),
                  child: Container(color: c.signal),
                ),
                Expanded(
                  flex: 1000 - (progress.fraction * 1000).round(),
                  child: Container(color: c.rule),
                ),
              ],
            ),
          ),
          const SizedBox(height: AppTokens.s8),
          Text(
            '${progress.done} of ${progress.total}'
            '${progress.failed > 0 ? ' · ${progress.failed} not available' : ''}',
            style: AppTokens.numberStyle.copyWith(color: c.ink),
          ),
        ],
      ),
    );
  }
}
