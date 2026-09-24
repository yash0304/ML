// lib/features/discovery/presentation/discovery_screen.dart
//
// What is along this leg — issue #27. SCREENS.md §11, `ON THE ROAD`.
//
// ORDERED BY DISTANCE ALONG THE ROUTE, never by distance from you. "Coming up
// in 12 km" is the useful sentence while moving; "0.2 km away" treats the road
// as a plane, and a place 200 m off across a gorge is an hour of driving.

import 'package:flutter/material.dart';

import '../../../core/theme/app_tokens.dart';
import '../../../core/theme/motion.dart';
import '../../../core/widgets/retro.dart';
import '../data/place_details.dart'
    show foodLine, servesJain, servesVeg;
import '../data/discovery.dart';
import '../data/poi_category.dart' show placeCategoryLabel;

class DiscoveryScreen extends StatefulWidget {
  final Stream<LegDiscovery> discovery;
  final void Function(CorridorPlace place) onOpen;

  const DiscoveryScreen({
    super.key,
    required this.discovery,
    required this.onOpen,
  });

  @override
  State<DiscoveryScreen> createState() => _DiscoveryScreenState();
}

class _DiscoveryScreenState extends State<DiscoveryScreen> {
  String? _category;

  /// Diet filters, on top of the category. Offered only when some place on
  /// the leg says so — see the chip rule below.
  bool _veg = false;
  bool _jain = false;

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);

    return Scaffold(
      backgroundColor: c.paper,
      appBar: AppBar(
        title: const Text('On the road'),
        backgroundColor: c.paper,
        foregroundColor: c.ink,
        elevation: 0,
      ),
      body: StreamBuilder<LegDiscovery>(
        stream: widget.discovery,
        builder: (context, snap) {
          final leg = snap.data;
          if (leg == null) return const SizedBox();

          if (leg.places.isEmpty) {
            return _Empty(synced: leg.isSynced);
          }

          // A chip nobody can match is a filter that only disappoints, so the
          // row offers only the categories this leg actually has.
          final categories = leg.categoriesPresent;
          // The same rule for diets: a Veg or Jain chip appears only if some
          // place on this leg says it serves that. On most Khasi hill roads
          // no place is tagged Jain, and the honest result is no Jain chip —
          // not one that always comes back empty.
          final anyVeg = leg.places.any((p) => servesVeg(p.tags));
          final anyJain = leg.places.any((p) => servesJain(p.tags));
          final shown = [
            for (final p in leg.places)
              if ((_category == null || p.category == _category) &&
                  (!_veg || servesVeg(p.tags)) &&
                  (!_jain || servesJain(p.tags)))
                p,
          ];

          return ListView(
            padding: const EdgeInsets.only(bottom: AppTokens.s32),
            children: [
              Padding(
                padding: const EdgeInsets.fromLTRB(
                  AppTokens.gutter,
                  AppTokens.s8,
                  AppTokens.gutter,
                  0,
                ),
                child: Text(
                  '${leg.fromName} → ${leg.toName}'
                  '${leg.distanceKm == null ? '' : ' · ${leg.distanceKm!.round()} km'}',
                  style: AppTokens.captionStyle.copyWith(color: c.muted),
                ),
              ),

              const StencilLabel('Filter'),
              Padding(
                padding: const EdgeInsets.symmetric(
                  horizontal: AppTokens.gutter,
                ),
                child: Wrap(
                  spacing: AppTokens.s8,
                  runSpacing: AppTokens.s8,
                  children: [
                    _Chip(
                      label: 'All',
                      on: _category == null,
                      onTap: () => setState(() => _category = null),
                    ),
                    for (final key in categories)
                      _Chip(
                        label: placeCategoryLabel(key),
                        on: _category == key,
                        onTap: () => setState(
                          () => _category = _category == key ? null : key,
                        ),
                      ),
                    if (anyVeg)
                      _Chip(
                        label: 'Veg',
                        on: _veg,
                        onTap: () => setState(() => _veg = !_veg),
                      ),
                    if (anyJain)
                      _Chip(
                        label: 'Jain',
                        on: _jain,
                        onTap: () => setState(() => _jain = !_jain),
                      ),
                  ],
                ),
              ),

              const StencilLabel('Coming up'),
              for (final place in shown)
                _PlaceRow(
                  place: place,
                  planned: leg.plannedOsmIds.contains(place.osmId),
                  onTap: () => widget.onOpen(place),
                ),

              if (shown.isEmpty)
                Padding(
                  padding: const EdgeInsets.symmetric(
                    horizontal: AppTokens.gutter,
                  ),
                  child: Text(
                    'Nothing on this leg matches those filters.',
                    style: AppTokens.captionStyle.copyWith(color: c.muted),
                  ),
                ),

              const StencilLabel('Where this came from'),
              Padding(
                padding: const EdgeInsets.symmetric(
                  horizontal: AppTokens.gutter,
                ),
                child: Text(
                  'OpenStreetMap, downloaded once before you left. Distances '
                  'are along the road, not straight lines — something 200 m '
                  'away across a gorge is an hour of driving.\n\n'
                  'Any number here was typed into a public map by a stranger. '
                  'It carries the amber dot until you call it yourself.',
                  style: AppTokens.captionStyle.copyWith(color: c.muted),
                ),
              ),
            ],
          );
        },
      ),
    );
  }
}

class _PlaceRow extends StatelessWidget {
  final CorridorPlace place;
  final VoidCallback onTap;

  /// In the leg's plan: marked, so the list reads as a plan and not only
  /// as a catalogue.
  final bool planned;

  const _PlaceRow({
    required this.place,
    required this.onTap,
    this.planned = false,
  });

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);

    return PressScale(
      onTap: onTap,
      child: Container(
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
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // A milestone, because a milestone is a thing you pass. Same
            // idiom the Trip screen uses for the next leg.
            MilestoneMarker(
              numeral: '${place.alongRouteKm.round()}',
              place: placeCategoryLabel(place.category),
            ),
            const SizedBox(width: AppTokens.s16),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const SizedBox(height: AppTokens.s8),
                  Row(
                    children: [
                      Expanded(
                        child: Text(
                          place.name,
                          maxLines: 2,
                          overflow: TextOverflow.ellipsis,
                          style: AppTokens.rowTitleStyle.copyWith(
                            color: c.ink,
                          ),
                        ),
                      ),
                      // THE AMBER DOT, on anything carrying a number from
                      // open map data. Nothing community-contributed is ever
                      // presented as verified.
                      if (place.hasPhone)
                        Container(
                          width: 7,
                          height: 7,
                          margin: const EdgeInsets.only(left: AppTokens.s8),
                          decoration: BoxDecoration(
                            color: c.cautionMark,
                            shape: BoxShape.circle,
                          ),
                        ),
                    ],
                  ),
                  if (planned)
                    Text(
                      'IN THE PLAN',
                      style: AppTokens.stencilStyle.copyWith(
                        fontSize: 9,
                        color: c.signal,
                      ),
                    ),
                  if (foodLine(place.tags) != null)
                    Padding(
                      padding: const EdgeInsets.only(top: 2),
                      child: Text(
                        foodLine(place.tags)!,
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: AppTokens.captionStyle.copyWith(color: c.ink),
                      ),
                    ),
                  const SizedBox(height: 4),
                  Text(
                    place.offRouteKm < 0.2
                        ? 'on the road'
                        : '${place.offRouteKm.toStringAsFixed(1)} km off the '
                              'road',
                    style: AppTokens.captionStyle.copyWith(color: c.muted),
                  ),
                  if (place.hasPhone)
                    Padding(
                      padding: const EdgeInsets.only(top: 2),
                      child: Text(
                        place.phones.first.phoneRaw,
                        style: AppTokens.numberStyle.copyWith(color: c.ink),
                      ),
                    ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _Empty extends StatelessWidget {
  final bool synced;
  const _Empty({required this.synced});

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(AppTokens.s32),
        child: Text(
          // Two different reasons for an empty list, and confusing them
          // sends the user to the wrong screen.
          synced
              ? 'Nothing was found along this leg. OpenStreetMap simply has '
                    'nothing tagged here — it happens on quiet roads.'
              : 'This leg has not been downloaded yet. Do that on WiFi from '
                    'More, then Download everything.',
          textAlign: TextAlign.center,
          style: AppTokens.captionStyle.copyWith(color: c.muted),
        ),
      ),
    );
  }
}

class _Chip extends StatelessWidget {
  final String label;
  final bool on;
  final VoidCallback onTap;
  const _Chip({required this.label, required this.on, required this.onTap});

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    return PressScale(
      onTap: () {
        Haptics.select();
        onTap();
      },
      child: Container(
        padding: const EdgeInsets.symmetric(
          horizontal: AppTokens.s12,
          vertical: AppTokens.s8,
        ),
        decoration: BoxDecoration(
          color: on ? c.signal : Colors.transparent,
          border: Border.all(color: on ? c.ink : c.rule),
          borderRadius: BorderRadius.circular(AppTokens.radiusSoft),
        ),
        child: Text(
          label.toUpperCase(),
          style: AppTokens.stencilStyle.copyWith(
            fontSize: 10,
            color: on ? c.paper : c.muted,
          ),
        ),
      ),
    );
  }
}
