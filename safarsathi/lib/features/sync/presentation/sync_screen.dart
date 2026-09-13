// lib/features/sync/presentation/sync_screen.dart
//
// "Download everything" — issue #25.
//
// A StatefulWidget over a stream. No state-management library: what this holds
// is a plan and a position in it, for the lifetime of one route.

import 'package:flutter/material.dart';

import '../../../core/theme/app_tokens.dart';
import '../../../core/theme/motion.dart';
import '../../../core/widgets/retro.dart';
import '../data/sync_error.dart';
import '../data/trip_sync.dart';

class SyncScreen extends StatefulWidget {
  final Future<TripSyncPlan> Function() plan;
  final Stream<SyncProgress> Function() run;

  /// Rough size, phrased in words rather than a false precision.
  final Future<String> Function() estimateSize;

  const SyncScreen({
    super.key,
    required this.plan,
    required this.run,
    required this.estimateSize,
  });

  @override
  State<SyncScreen> createState() => _SyncScreenState();
}

class _SyncScreenState extends State<SyncScreen> {
  late Future<TripSyncPlan> _plan = widget.plan();
  late Future<String> _size = widget.estimateSize();

  SyncProgress? _progress;
  bool _running = false;

  Future<void> _start() async {
    if (_running) return;
    setState(() => _running = true);

    try {
      await for (final progress in widget.run()) {
        if (!mounted) return;
        setState(() => _progress = progress);
      }
      if (!mounted) return;
      // Confirm even with failures: something got done, and the list below
      // says what did not.
      Haptics.confirm();
      setState(() {
        _plan = widget.plan();
        _size = widget.estimateSize();
      });
    } finally {
      if (mounted) setState(() => _running = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    final progress = _progress;

    return Scaffold(
      backgroundColor: c.paper,
      appBar: AppBar(
        title: const Text('Get ready to leave'),
        backgroundColor: c.paper,
        foregroundColor: c.ink,
        elevation: 0,
      ),
      body: ListView(
        padding: const EdgeInsets.only(bottom: 96),
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(
              AppTokens.gutter,
              AppTokens.s8,
              AppTokens.gutter,
              0,
            ),
            child: Text(
              'Do this on WiFi before you go. It fetches the route for every '
              'leg, what is along it, the map, and the forecast — and then '
              'the app never asks the network again.',
              style: AppTokens.captionStyle.copyWith(color: c.muted),
            ),
          ),

          const StencilLabel('What it will do'),
          FutureBuilder<TripSyncPlan>(
            future: _plan,
            builder: (context, snap) {
              final plan = snap.data;
              if (plan == null) {
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
              return _Plan(plan: plan, size: _size);
            },
          ),

          if (progress != null) ...[
            const StencilLabel('Progress'),
            _Progress(progress: progress),
          ],

          if (progress != null && progress.hadFailures) ...[
            const StencilLabel('What did not work'),
            // Said ONCE, above the list, because seven rows repeating the
            // same DNS failure is one fact rendered seven times.
            Builder(
              builder: (context) {
                final summary = summariseSyncFailures(
                  [for (final f in progress.failures) f.error],
                  total: progress.total,
                );
                if (summary == null) return const SizedBox.shrink();
                final offline = progress.failures.every(
                  (f) => f.error.needsDifferentNetwork,
                );
                return Padding(
                  padding: const EdgeInsets.fromLTRB(
                    AppTokens.gutter,
                    0,
                    AppTokens.gutter,
                    AppTokens.s8,
                  ),
                  child: Text(
                    summary,
                    style: AppTokens.captionStyle.copyWith(
                      color: offline ? c.cautionMark : c.muted,
                    ),
                  ),
                );
              },
            ),
            for (final failure in progress.failures)
              _FailureRow(failure: failure),
          ],

          if (progress != null &&
              progress.isDone &&
              !progress.hadFailures) ...[
            Padding(
              padding: const EdgeInsets.fromLTRB(
                AppTokens.gutter,
                AppTokens.s16,
                AppTokens.gutter,
                0,
              ),
              child: Row(
                children: [
                  Icon(Icons.check, size: 16, color: c.signal),
                  const SizedBox(width: AppTokens.s8),
                  Expanded(
                    child: Text(
                      'Everything downloaded. You can turn the data off now.',
                      style: AppTokens.captionStyle.copyWith(color: c.signal),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ],
      ),
      bottomNavigationBar: SafeArea(
        minimum: const EdgeInsets.all(AppTokens.s16),
        child: FutureBuilder<TripSyncPlan>(
          future: _plan,
          builder: (context, snap) {
            final plan = snap.data;
            final ready = !_running && plan != null && !plan.isEmpty;

            return PressScale(
              onTap: ready ? _start : null,
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
                      : plan == null
                      ? 'Working it out…'
                      : plan.isEmpty
                      ? 'Nothing to download yet'
                      : 'Download everything (${plan.total} steps)',
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

class _Plan extends StatelessWidget {
  final TripSyncPlan plan;
  final Future<String> size;

  const _Plan({required this.plan, required this.size});

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);

    if (plan.isEmpty) {
      return Padding(
        padding: const EdgeInsets.symmetric(horizontal: AppTokens.gutter),
        child: Text(
          plan.legsWithoutCoordinates > 0 ||
                  plan.stopsWithoutCoordinates > 0
              ? 'None of this trip has coordinates yet. Open each stop and '
                    'find it on the map first — without that there is nothing '
                    'to download.'
              : 'This trip has no stops yet.',
          style: AppTokens.captionStyle.copyWith(color: c.cautionMark),
        ),
      );
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        _Line(
          label: 'Routes and places',
          value: _count(plan.countOf(SyncKind.corridor), 'leg'),
        ),
        _Line(
          label: 'Weather',
          value: _count(plan.countOf(SyncKind.weather), 'stop'),
        ),
        _Line(
          label: 'Map',
          value: plan.countOf(SyncKind.tiles) > 0
              ? 'the whole corridor'
              : 'nothing to cover',
        ),
        Padding(
          padding: const EdgeInsets.fromLTRB(
            AppTokens.gutter,
            AppTokens.s12,
            AppTokens.gutter,
            0,
          ),
          child: FutureBuilder<String>(
            future: size,
            builder: (context, snap) => Text(
              // The estimate phrases itself — "about 24 MB, nearly all of it
              // map" — so prefixing "Roughly" produced "Roughly about".
              snap.data == null
                  ? 'Working out the size…'
                  : '${snap.data![0].toUpperCase()}${snap.data!.substring(1)}.',
              style: AppTokens.captionStyle.copyWith(color: c.muted),
            ),
          ),
        ),
        if (plan.legsWithoutCoordinates > 0 ||
            plan.stopsWithoutCoordinates > 0)
          Padding(
            padding: const EdgeInsets.fromLTRB(
              AppTokens.gutter,
              AppTokens.s8,
              AppTokens.gutter,
              0,
            ),
            child: Text(
              // Named, never silent. A sync that quietly covers less than the
              // user expects is worse than one that refuses.
              '${[
                if (plan.legsWithoutCoordinates > 0)
                  '${plan.legsWithoutCoordinates} '
                      '${plan.legsWithoutCoordinates == 1 ? 'leg' : 'legs'}',
                if (plan.stopsWithoutCoordinates > 0)
                  '${plan.stopsWithoutCoordinates} '
                      '${plan.stopsWithoutCoordinates == 1 ? 'stop' : 'stops'}',
              ].join(' and ')} will be skipped for want of coordinates.',
              style: AppTokens.captionStyle.copyWith(color: c.cautionMark),
            ),
          ),
      ],
    );
  }
}

/// "1 leg", "3 legs". A trip with one of anything is common enough that
/// "1 legs" would be read as sloppiness on the screen people see last before
/// leaving.
String _count(int n, String noun) => '$n $noun${n == 1 ? '' : 's'}';

class _Line extends StatelessWidget {
  final String label;
  final String value;
  const _Line({required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    return Container(
      padding: const EdgeInsets.symmetric(
        horizontal: AppTokens.gutter,
        vertical: AppTokens.s12,
      ),
      decoration: BoxDecoration(
        border: Border(
          bottom: BorderSide(color: c.rule, width: AppTokens.hairline),
        ),
      ),
      child: Row(
        children: [
          Expanded(
            child: Text(
              label,
              style: AppTokens.rowTitleStyle.copyWith(color: c.ink),
            ),
          ),
          Text(
            value,
            style: AppTokens.captionStyle.copyWith(color: c.muted),
          ),
        ],
      ),
    );
  }
}

class _Progress extends StatelessWidget {
  final SyncProgress progress;
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
          Row(
            children: [
              Text(
                '${progress.done} of ${progress.total}',
                style: AppTokens.numberStyle.copyWith(color: c.ink),
              ),
              const SizedBox(width: AppTokens.s12),
              Expanded(
                child: Text(
                  // The item, not just a percentage. "Route and what is along
                  // it · Shillong → Cherrapunji" is something a person can
                  // wait through; "43%" is not.
                  progress.current?.label ?? 'Finished',
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: AppTokens.captionStyle.copyWith(color: c.muted),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class _FailureRow extends StatelessWidget {
  final SyncFailure failure;
  const _FailureRow({required this.failure});

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    return Container(
      padding: const EdgeInsets.symmetric(
        horizontal: AppTokens.gutter,
        vertical: AppTokens.s12,
      ),
      decoration: BoxDecoration(
        border: Border(
          bottom: BorderSide(color: c.rule, width: AppTokens.hairline),
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            failure.task.label,
            style: AppTokens.rowTitleStyle.copyWith(color: c.ink),
          ),
          const SizedBox(height: 2),
          Text(
            failure.reason,
            // Amber, not red. A leg that did not download is a thing to
            // retry, not an emergency.
            style: AppTokens.captionStyle.copyWith(color: c.cautionMark),
          ),
        ],
      ),
    );
  }
}
