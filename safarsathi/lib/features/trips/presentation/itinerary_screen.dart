// lib/features/trips/presentation/itinerary_screen.dart
//
// The editable itinerary — issue #16.
//
// The Trip tab stays read-only and stays the thing you look at on the road.
// This is the planning surface: drag to reorder, tap to edit, add at the end.

import 'package:flutter/material.dart';

import '../../../core/database/app_database.dart';
import '../../../core/theme/app_tokens.dart';
import '../../../core/theme/motion.dart';
import '../data/trip_editor.dart';

class ItineraryScreen extends StatelessWidget {
  final String tripName;
  final Stream<List<Stop>> stops;
  final Future<void> Function(int from, int to) onReorder;
  final void Function(Stop stop) onEdit;
  final VoidCallback onAdd;
  final VoidCallback? onEditTrip;

  /// The stop detail screen (#51). When it is wired, tapping a row opens the
  /// stop rather than the form — the form is then one pencil further in.
  /// Without it the row still opens the form, which is what this screen did
  /// before the detail screen existed.
  final void Function(Stop stop)? onOpen;

  const ItineraryScreen({
    super.key,
    required this.tripName,
    required this.stops,
    required this.onReorder,
    required this.onEdit,
    required this.onAdd,
    this.onEditTrip,
    this.onOpen,
  });

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);

    return Scaffold(
      backgroundColor: c.paper,
      appBar: AppBar(
        title: Text(tripName),
        backgroundColor: c.paper,
        foregroundColor: c.ink,
        elevation: 0,
        actions: [
          if (onEditTrip != null)
            IconButton(
              onPressed: onEditTrip,
              icon: const Icon(Icons.edit_outlined),
              color: c.muted,
            ),
        ],
      ),
      body: StreamBuilder<List<Stop>>(
        stream: stops,
        builder: (context, snap) {
          final list = snap.data;
          if (list == null) return const SizedBox();
          if (list.isEmpty) return const _NoStops();

          return ReorderableListView.builder(
            padding: const EdgeInsets.only(bottom: 96),
            itemCount: list.length,
            // onReorderItem, not onReorder. The older callback reports the
            // destination index as it would be BEFORE the item is removed, so
            // every downward move lands one row too high unless the caller
            // subtracts one. This one has already done that arithmetic.
            onReorderItem: (from, to) {
              Haptics.light();
              onReorder(from, to);
            },
            itemBuilder: (context, i) => _StopRow(
              key: ValueKey(list[i].id),
              stop: list[i],
              index: i,
              onTap: () => (onOpen ?? onEdit)(list[i]),
            ),
          );
        },
      ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: onAdd,
        backgroundColor: c.signal,
        foregroundColor: c.paper,
        icon: const Icon(Icons.add, size: 18),
        label: Text(
          'Add stop',
          style: AppTokens.stencilStyle.copyWith(
            fontSize: 10.5,
            color: c.paper,
          ),
        ),
      ),
    );
  }
}

class _StopRow extends StatelessWidget {
  final Stop stop;
  final int index;
  final VoidCallback onTap;

  const _StopRow({
    super.key,
    required this.stop,
    required this.index,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    final tags = parseTags(stop.activityTags);
    final a = stop.arrivalDate;

    return Container(
      decoration: BoxDecoration(
        color: c.paper,
        border: Border(
          bottom: BorderSide(color: c.rule, width: AppTokens.hairline),
        ),
      ),
      child: InkWell(
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.symmetric(
            horizontal: AppTokens.gutter,
            vertical: AppTokens.s12,
          ),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              SizedBox(
                width: 26,
                child: Text(
                  '${stop.sequenceOrder}',
                  style: AppTokens.numberStyle.copyWith(color: c.muted),
                ),
              ),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      stop.name,
                      style: AppTokens.rowTitleStyle.copyWith(color: c.ink),
                    ),
                    const SizedBox(height: 2),
                    Row(
                      children: [
                        if (a != null) ...[
                          Text(
                            '${a.day.toString().padLeft(2, '0')}/'
                            '${a.month.toString().padLeft(2, '0')}',
                            style: AppTokens.numberStyle.copyWith(
                              color: c.muted,
                              fontSize: 11.5,
                            ),
                          ),
                          const SizedBox(width: AppTokens.s8),
                        ],
                        Text(
                          stop.nights == 0
                              ? 'passing through'
                              : '${stop.nights} '
                                    '${stop.nights == 1 ? 'night' : 'nights'}',
                          style: AppTokens.captionStyle.copyWith(
                            color: stop.nights == 0 ? c.muted : c.ink,
                          ),
                        ),
                      ],
                    ),
                    if (tags.isNotEmpty)
                      Padding(
                        padding: const EdgeInsets.only(top: 4),
                        child: Text(
                          tags.join(' · '),
                          style: AppTokens.stencilStyle.copyWith(
                            fontSize: 9,
                            color: c.muted,
                          ),
                        ),
                      ),
                  ],
                ),
              ),
              ReorderableDragStartListener(
                index: index,
                child: Padding(
                  padding: const EdgeInsets.only(left: AppTokens.s8),
                  child: Icon(Icons.drag_handle, size: 20, color: c.rule),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _NoStops extends StatelessWidget {
  const _NoStops();

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(AppTokens.s32),
        child: Text(
          'No stops yet. Add them in the order you will reach them; the legs '
          'between them are worked out for you.',
          textAlign: TextAlign.center,
          style: AppTokens.captionStyle.copyWith(color: c.muted),
        ),
      ),
    );
  }
}
