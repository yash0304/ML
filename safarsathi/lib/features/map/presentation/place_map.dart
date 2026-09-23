// lib/features/map/presentation/place_map.dart
//
// One diary entry on the offline map, and how far you are from it.
//
// This is the part that works with no signal: the downloaded tiles, the pin,
// and the phone's own GPS dot. Turn-by-turn directions are Google Maps' job
// and need a route server, so the entry page hands off for those and says
// so. What this adds is the answer to "which way, and how far?" in a valley
// with no bars.

import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart' as fm;
import 'package:latlong2/latlong.dart' as ll;

import '../../../core/theme/app_tokens.dart';
import '../../contacts/data/place_location.dart' show describeDistance;
import '../../discovery/data/geo.dart';
import '../data/here.dart';
import '../data/tile_downloader.dart' show defaultMinZoom, defaultMaxZoom;
import '../data/tile_math.dart';
import '../data/tile_provider.dart';
import '../data/tile_store.dart';
import 'trip_map.dart';

/// Past this, you and the place are not worth one frame: zooming out to fit
/// both would leave the band anything was downloaded for, and a blank map
/// says nothing. The distance is still given.
const _fitWithinMetres = 25000.0;

class PlaceMap extends StatefulWidget {
  final MapTileProvider provider;
  final TileStore store;
  final LatLng place;
  final LocationSource? location;
  final double height;

  const PlaceMap({
    super.key,
    required this.provider,
    required this.store,
    required this.place,
    this.location,
    this.height = 220,
  });

  @override
  State<PlaceMap> createState() => _PlaceMapState();
}

class _PlaceMapState extends State<PlaceMap> {
  final _controller = fm.MapController();

  /// The most detailed level downloaded at this spot, or null for none.
  /// Cached: a place's coverage does not change while its page is open.
  late final Future<int?> _zoom = _bestZoom();

  /// Set once [_zoom] resolves: whether a map is actually on screen.
  bool _drawn = false;

  HereFix? _me;
  String? _message;
  bool _locating = false;

  Future<int?> _bestZoom() async {
    // From one below the top: 14 shows a village's lanes and still the road
    // in; 15 is a single street.
    for (var z = defaultMaxZoom - 1; z >= defaultMinZoom; z--) {
      if (await widget.store.has(
        widget.provider.id,
        tileFor(widget.place, z),
      )) {
        _drawn = true;
        return z;
      }
    }
    return null;
  }

  Future<void> _locate() async {
    final source = widget.location;
    if (source == null || _locating) return;
    setState(() {
      _locating = true;
      _message = null;
    });

    final state = await source.check(ask: true);
    if (state != HereState.locating && state != HereState.found) {
      if (!mounted) return;
      setState(() {
        _locating = false;
        _message = switch (state) {
          HereState.serviceOff => 'Location is switched off on the phone.',
          HereState.blocked =>
            'Location is blocked for this app in the phone\'s Settings.',
          _ => 'Location was not allowed.',
        };
      });
      return;
    }

    final fix = await source.once();
    if (!mounted) return;
    setState(() {
      _locating = false;
      _me = fix;
      if (fix == null) {
        _message = 'No position yet. In a valley the GPS can take a minute.';
      }
    });
    if (fix == null) return;

    // No map drawn — the download does not reach here — means no camera to
    // move. The distance line still answers the question.
    final metres = haversineMetres(fix.at, widget.place);
    if (_drawn && metres <= _fitWithinMetres) {
      _controller.fitCamera(
        fm.CameraFit.bounds(
          bounds: fm.LatLngBounds(
            ll.LatLng(fix.at.lat, fix.at.lon),
            ll.LatLng(widget.place.lat, widget.place.lon),
          ),
          padding: const EdgeInsets.all(48),
          maxZoom: defaultMaxZoom.toDouble(),
          minZoom: (defaultMinZoom - 1).toDouble(),
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    final me = _me;

    return FutureBuilder<int?>(
      future: _zoom,
      builder: (context, snap) {
        if (snap.connectionState != ConnectionState.done) {
          return SizedBox(height: widget.height);
        }
        final zoom = snap.data;

        return Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            if (zoom == null)
              // Not the trip map's "nothing downloaded": the map may be
              // complete and simply not reach this far off the route.
              Padding(
                padding: const EdgeInsets.symmetric(
                  horizontal: AppTokens.gutter,
                ),
                child: Text(
                  'The downloaded map does not reach this place, so it '
                  'cannot be drawn here. Directions below still work with '
                  'signal.',
                  style: AppTokens.captionStyle.copyWith(color: c.muted),
                ),
              )
            else
              TripMap(
                provider: widget.provider,
                store: widget.store,
                place: widget.place,
                initialZoom: zoom,
                height: widget.height,
                me: me,
                controller: _controller,
              ),
            if (widget.location != null)
              Padding(
                padding: const EdgeInsets.fromLTRB(
                  AppTokens.gutter,
                  AppTokens.s8,
                  AppTokens.gutter,
                  0,
                ),
                child: me == null
                    ? InkWell(
                        key: const Key('place-map-locate'),
                        onTap: _locating ? null : _locate,
                        child: Padding(
                          padding: const EdgeInsets.symmetric(
                            vertical: AppTokens.s4,
                          ),
                          child: Row(
                            children: [
                              Icon(
                                Icons.my_location,
                                size: 16,
                                color: c.signal,
                              ),
                              const SizedBox(width: AppTokens.s8),
                              Expanded(
                                child: Text(
                                  _locating
                                      ? 'Finding where you are…'
                                      : _message ??
                                            'How far am I? Uses GPS, no '
                                                'signal needed',
                                  style: AppTokens.captionStyle.copyWith(
                                    color: _message == null
                                        ? c.signal
                                        : c.caution,
                                  ),
                                ),
                              ),
                            ],
                          ),
                        ),
                      )
                    : Text(
                        // Straight line, said out loud: 3 km across a gorge
                        // can be an hour by road.
                        '${describeDistance(haversineMetres(me.at, widget.place))} '
                        'from you in a straight line · ${describeFix(me)}',
                        key: const Key('place-map-distance'),
                        style: AppTokens.captionStyle.copyWith(color: c.ink),
                      ),
              ),
          ],
        );
      },
    );
  }
}
