// lib/features/trips/presentation/timeline_screen.dart
//
// The trip as it went — #30. SCREENS.md §9: the vertical rail is the road;
// arrivals are milestone caps on it, notes and photos plain dots. Structure
// and annotation read differently at a glance.

import 'package:flutter/material.dart';

import '../../../core/database/app_database.dart';
import '../../../core/theme/app_tokens.dart';
import '../../../core/widgets/retro.dart';
import '../data/timeline.dart';

class TimelineScreen extends StatefulWidget {
  final Stream<List<TimelineDay>> timeline;

  /// Whether the route log is on, and the switch. The switch returns a
  /// sentence when it could not do what was asked.
  final Stream<bool>? logging;
  final Future<String?> Function(bool on)? onLogging;

  final VoidCallback? onAddNote;
  final Future<void> Function(TimelineEntry note)? onDelete;

  /// Draws a stored photo. Injected so tests need no files.
  final Widget Function(String path) photo;

  const TimelineScreen({
    super.key,
    required this.timeline,
    required this.photo,
    this.logging,
    this.onLogging,
    this.onAddNote,
    this.onDelete,
  });

  @override
  State<TimelineScreen> createState() => _TimelineScreenState();
}

class _TimelineScreenState extends State<TimelineScreen> {
  String? _loggingProblem;

  static const _months = [
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
  ];

  static String _time(DateTime t) =>
      '${t.hour.toString().padLeft(2, '0')}:'
      '${t.minute.toString().padLeft(2, '0')}';

  static String _duration(Duration d) {
    final h = d.inHours;
    final m = d.inMinutes % 60;
    if (h == 0) return '$m min';
    return m == 0 ? '$h h' : '$h h $m min';
  }

  Future<void> _confirmDelete(TimelineEntry note) async {
    final yes = await showDialog<bool>(
      context: context,
      builder: (d) => AlertDialog(
        title: const Text('Delete this note?'),
        content: const Text('Its photos are removed from the timeline too.'),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(d).pop(false),
            child: const Text('Keep'),
          ),
          TextButton(
            key: const Key('timeline-delete-confirm'),
            onPressed: () => Navigator.of(d).pop(true),
            child: const Text('Delete'),
          ),
        ],
      ),
    );
    if (yes == true) await widget.onDelete?.call(note);
  }

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    final caption = AppTokens.captionStyle.copyWith(color: c.muted);

    return Scaffold(
      backgroundColor: c.paper,
      appBar: AppBar(
        title: const Text('Timeline'),
        backgroundColor: c.paper,
        foregroundColor: c.ink,
        elevation: 0,
      ),
      floatingActionButton: widget.onAddNote == null
          ? null
          : FloatingActionButton.extended(
              key: const Key('timeline-add-note'),
              onPressed: widget.onAddNote,
              backgroundColor: c.signal,
              foregroundColor: c.paper,
              icon: const Icon(Icons.edit_note, size: 20),
              label: const Text('Add a note'),
            ),
      body: StreamBuilder<List<TimelineDay>>(
        stream: widget.timeline,
        builder: (context, snap) {
          final days = snap.data ?? const <TimelineDay>[];
          return ListView(
            padding: const EdgeInsets.only(bottom: 96),
            children: [
              if (widget.logging != null && widget.onLogging != null)
                StreamBuilder<bool>(
                  stream: widget.logging,
                  builder: (context, on) => SwitchListTile(
                    key: const Key('timeline-logging'),
                    value: on.data ?? false,
                    onChanged: (v) async {
                      final problem = await widget.onLogging!(v);
                      if (mounted) setState(() => _loggingProblem = problem);
                    },
                    title: const Text('Log my route'),
                    subtitle: Text(
                      _loggingProblem ??
                          // THE BATTERY COST, ON THE SCREEN (SCREENS.md §9).
                          // It is the one feature here that drains the phone.
                          'Uses the GPS while on — about 4% of the battery a '
                              'day, more on a long drive. Needs no signal. A '
                              'notification shows while it runs; closing the '
                              'app from Recents stops it.',
                      style: AppTokens.captionStyle.copyWith(
                        color: _loggingProblem == null ? c.muted : c.caution,
                      ),
                    ),
                  ),
                ),
              if (days.isEmpty)
                Padding(
                  padding: const EdgeInsets.all(AppTokens.gutter),
                  child: Text(
                    'Nothing yet. Check in when you arrive (Trip page → Check '
                    'in) and add notes here. With the route log on, the road '
                    'you drive is measured too.',
                    style: caption,
                  ),
                ),
              for (final d in days) ...[
                StencilLabel(
                  '${d.day.day} ${_months[d.day.month - 1]}'
                  '${d.loggedKm >= 0.1 ? ' · ${d.loggedKm.toStringAsFixed(d.loggedKm < 10 ? 1 : 0)} km logged' : ''}',
                ),
                for (final (i, item) in d.items.indexed)
                  _RailRow(
                    first: i == 0,
                    last: i == d.items.length - 1,
                    cap: item is TimelineArrival,
                    child: item is TimelineArrival
                        ? _arrival(c, item)
                        : _note(c, item as TimelineEntry),
                  ),
              ],
              Padding(
                padding: const EdgeInsets.fromLTRB(
                  AppTokens.gutter,
                  AppTokens.s24,
                  AppTokens.gutter,
                  0,
                ),
                child: Text(
                  'Photos stay on this phone and are not in the backup file; '
                  'the timeline itself is.',
                  style: caption,
                ),
              ),
            ],
          );
        },
      ),
    );
  }

  Widget _arrival(AppColors c, TimelineArrival a) {
    final parts = [
      if (a.km != null)
        a.straightLine
            ? '${a.km!.round()} km in a straight line'
            : '${a.km!.toStringAsFixed(a.km! < 10 ? 1 : 0)} km driven',
      if (a.since != null) _duration(a.since!),
    ];
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'Reached ${a.stopName} · ${_time(a.entry.occurredAt)}',
          style: AppTokens.rowTitleStyle.copyWith(color: c.ink),
        ),
        if (parts.isNotEmpty)
          Text(
            '${parts.join(', ')} since the last arrival',
            style: AppTokens.captionStyle.copyWith(color: c.muted),
          ),
      ],
    );
  }

  Widget _note(AppColors c, TimelineEntry n) {
    final photos = photosOf(n);
    return GestureDetector(
      onLongPress: widget.onDelete == null ? null : () => _confirmDelete(n),
      behavior: HitTestBehavior.opaque,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          if (n.body != null)
            Text(n.body!, style: AppTokens.rowTitleStyle.copyWith(color: c.ink)),
          if (photos.isNotEmpty)
            Padding(
              padding: const EdgeInsets.only(top: AppTokens.s4),
              child: Wrap(
                spacing: AppTokens.s4,
                runSpacing: AppTokens.s4,
                children: [
                  for (final p in photos)
                    ClipRRect(
                      borderRadius: BorderRadius.circular(
                        AppTokens.radiusSoft,
                      ),
                      child: SizedBox(
                        width: 84,
                        height: 84,
                        child: widget.photo(p),
                      ),
                    ),
                ],
              ),
            ),
          Text(
            _time(n.occurredAt),
            style: AppTokens.captionStyle.copyWith(color: c.muted),
          ),
        ],
      ),
    );
  }
}

/// One stop on the rail: the line through it, a cap for an arrival or a dot
/// for a note, and what it says beside it.
class _RailRow extends StatelessWidget {
  final bool first;
  final bool last;
  final bool cap;
  final Widget child;

  const _RailRow({
    required this.first,
    required this.last,
    required this.cap,
    required this.child,
  });

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    return IntrinsicHeight(
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          SizedBox(
            width: 48,
            child: Stack(
              alignment: Alignment.topCenter,
              children: [
                Positioned(
                  top: first ? 14 : 0,
                  bottom: last ? null : 0,
                  height: last ? 14 : null,
                  child: Container(width: 2, color: c.rule),
                ),
                Padding(
                  padding: const EdgeInsets.only(top: 8),
                  child: cap
                      ? Container(
                          key: const Key('rail-cap'),
                          width: 16,
                          height: 12,
                          decoration: BoxDecoration(
                            color: c.signal,
                            borderRadius: const BorderRadius.vertical(
                              top: Radius.circular(8),
                            ),
                            border: Border.all(color: c.ink),
                          ),
                        )
                      : Container(
                          key: const Key('rail-dot'),
                          margin: const EdgeInsets.only(top: 2),
                          width: 8,
                          height: 8,
                          decoration: BoxDecoration(
                            color: c.muted,
                            shape: BoxShape.circle,
                          ),
                        ),
                ),
              ],
            ),
          ),
          Expanded(
            child: Padding(
              padding: const EdgeInsets.fromLTRB(
                0,
                AppTokens.s4,
                AppTokens.gutter,
                AppTokens.s12,
              ),
              child: child,
            ),
          ),
        ],
      ),
    );
  }
}
