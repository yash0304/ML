// lib/features/trips/presentation/trip_list_screen.dart
//
// Every trip, and which one the app is currently on — issue #16 / #19.
//
// EXACTLY ONE TRIP IS ACTIVE. The diary scopes to it, the money splits within
// it, the emergency screen reads its stops. Switching is a deliberate act with
// a visible marker, not a filter that can be left in an ambiguous state.

import 'package:flutter/material.dart';

import '../../../core/database/app_database.dart';
import '../../../core/theme/app_tokens.dart';
import '../../../core/theme/motion.dart';
import '../../../core/widgets/retro.dart';

class TripListScreen extends StatelessWidget {
  final Stream<List<Trip>> trips;
  final Future<void> Function(int tripId) onActivate;
  final void Function(Trip trip) onOpen;
  final Future<void> Function(Trip trip) onDelete;
  final VoidCallback onCreate;

  const TripListScreen({
    super.key,
    required this.trips,
    required this.onActivate,
    required this.onOpen,
    required this.onDelete,
    required this.onCreate,
  });

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);

    return Scaffold(
      backgroundColor: c.paper,
      appBar: AppBar(
        title: const Text('Trips'),
        backgroundColor: c.paper,
        foregroundColor: c.ink,
        elevation: 0,
      ),
      body: StreamBuilder<List<Trip>>(
        stream: trips,
        builder: (context, snap) {
          final list = snap.data;
          if (list == null) return const SizedBox();
          if (list.isEmpty) {
            return Center(
              child: Padding(
                padding: const EdgeInsets.all(AppTokens.s32),
                child: Text(
                  'No trips yet. A trip is a name and a list of stops; '
                  'everything else in the app hangs off it.',
                  textAlign: TextAlign.center,
                  style: AppTokens.captionStyle.copyWith(color: c.muted),
                ),
              ),
            );
          }
          return ListView.separated(
            padding: const EdgeInsets.all(AppTokens.gutter),
            itemCount: list.length,
            separatorBuilder: (_, _) => const SizedBox(height: AppTokens.s12),
            itemBuilder: (context, i) => _TripCard(
              trip: list[i],
              onOpen: () => onOpen(list[i]),
              onActivate: () => onActivate(list[i].id),
              onDelete: () => _confirmDelete(context, list[i]),
            ),
          );
        },
      ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: onCreate,
        backgroundColor: c.signal,
        foregroundColor: c.paper,
        icon: const Icon(Icons.add, size: 18),
        label: Text(
          'New trip',
          style: AppTokens.stencilStyle.copyWith(
            fontSize: 10.5,
            color: c.paper,
          ),
        ),
      ),
    );
  }

  Future<void> _confirmDelete(BuildContext context, Trip trip) async {
    final c = AppTokens.of(context);
    final ok = await showDialog<bool>(
      context: context,
      builder: (dialogContext) => AlertDialog(
        backgroundColor: c.paper,
        title: Text(
          'Delete ${trip.name}?',
          style: AppTokens.titleStyle.copyWith(color: c.ink, fontSize: 18),
        ),
        content: Text(
          // Unlike deleting a stop, this one really does take the contacts.
          'This removes the trip and everything under it: every stop, every '
          'contact in its diary, every expense. It cannot be undone.',
          style: AppTokens.captionStyle.copyWith(color: c.muted),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(dialogContext).pop(false),
            child: Text('Keep', style: TextStyle(color: c.muted)),
          ),
          TextButton(
            onPressed: () => Navigator.of(dialogContext).pop(true),
            child: Text('Delete', style: TextStyle(color: c.emergency)),
          ),
        ],
      ),
    );
    if (ok == true) {
      Haptics.grave();
      await onDelete(trip);
    }
  }
}

class _TripCard extends StatelessWidget {
  final Trip trip;
  final VoidCallback onOpen;
  final VoidCallback onActivate;
  final VoidCallback onDelete;

  const _TripCard({
    required this.trip,
    required this.onOpen,
    required this.onActivate,
    required this.onDelete,
  });

  String _dates() {
    String f(DateTime d) =>
        '${d.day.toString().padLeft(2, '0')}/'
        '${d.month.toString().padLeft(2, '0')}';
    final s = trip.startDate;
    final e = trip.endDate;
    if (s == null && e == null) return 'No dates set';
    if (s != null && e != null) return '${f(s)} — ${f(e)} ${e.year}';
    return f((s ?? e)!);
  }

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    return TicketCard(
      onTap: onOpen,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Row(
            children: [
              Expanded(
                child: Text(
                  trip.name,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: AppTokens.titleStyle.copyWith(
                    color: c.ink,
                    fontSize: 18,
                  ),
                ),
              ),
              if (trip.isActive)
                Text(
                  'ON THIS TRIP',
                  style: AppTokens.stencilStyle.copyWith(
                    fontSize: 9,
                    color: c.signal,
                  ),
                ),
            ],
          ),
          const SizedBox(height: AppTokens.s4),
          Text(
            _dates(),
            style: AppTokens.numberStyle.copyWith(
              color: c.muted,
              fontSize: 11.5,
            ),
          ),
          const SizedBox(height: AppTokens.s12),
          Row(
            children: [
              if (!trip.isActive)
                PressScale(
                  onTap: onActivate,
                  child: Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: AppTokens.s12,
                      vertical: AppTokens.s8,
                    ),
                    decoration: BoxDecoration(
                      border: Border.all(color: c.ink),
                      borderRadius: BorderRadius.circular(
                        AppTokens.radiusSoft,
                      ),
                    ),
                    child: Text(
                      'Switch to this',
                      style: AppTokens.stencilStyle.copyWith(
                        fontSize: 10,
                        color: c.ink,
                      ),
                    ),
                  ),
                ),
              const Spacer(),
              GestureDetector(
                onTap: onDelete,
                child: Icon(Icons.delete_outline, size: 20, color: c.muted),
              ),
            ],
          ),
        ],
      ),
    );
  }
}
