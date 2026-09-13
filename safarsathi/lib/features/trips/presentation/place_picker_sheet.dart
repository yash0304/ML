// lib/features/trips/presentation/place_picker_sheet.dart
//
// Giving a stop coordinates, without a map.
//
// The map picker arrives at #24. Until then a stop needs a latitude and a
// longitude for the corridor to route between, and there are two honest ways
// to get them: look the name up, or type them in.
//
// NOTHING IS SAVED UNTIL THE USER PICKS A CANDIDATE. The geocoder returns its
// best guess with a display name; silently accepting the first result is how a
// trip ends up routed to a village in Karnataka with the same name as the town
// you meant.

import 'package:flutter/material.dart';

import '../../../core/theme/app_tokens.dart';
import '../../../core/theme/motion.dart';
import '../../../core/widgets/retro.dart';
import '../../discovery/data/geo.dart';
import '../../discovery/data/geocoder.dart';

class PlacePickerSheet extends StatefulWidget {
  final String initialQuery;
  final String countryCode;
  final LatLng? current;

  /// Injected so the widget test never reaches the network.
  final Future<List<GeocodeResult>> Function(String query, String country)
  search;

  const PlacePickerSheet({
    super.key,
    required this.initialQuery,
    required this.search,
    this.countryCode = 'IN',
    this.current,
  });

  @override
  State<PlacePickerSheet> createState() => _PlacePickerSheetState();
}

class _PlacePickerSheetState extends State<PlacePickerSheet> {
  late final _query = TextEditingController(text: widget.initialQuery);
  late final _manual = TextEditingController(
    text: widget.current == null ? '' : formatLatLon(widget.current!),
  );

  List<GeocodeResult>? _results;
  String? _error;
  bool _searching = false;

  @override
  void dispose() {
    _query.dispose();
    _manual.dispose();
    super.dispose();
  }

  Future<void> _run() async {
    if (_searching) return;
    setState(() {
      _searching = true;
      _error = null;
    });
    try {
      final results = await widget.search(
        _query.text.trim(),
        widget.countryCode,
      );
      if (!mounted) return;
      setState(() {
        _results = results;
        if (results.isEmpty) {
          _error =
              'Nothing found for that name. Try adding the district, or '
              'type the coordinates below.';
        }
      });
    } on Object catch (e) {
      if (!mounted) return;
      Haptics.reject();
      setState(() => _error = '$e');
    } finally {
      if (mounted) setState(() => _searching = false);
    }
  }

  void _take(LatLng location) {
    Haptics.confirm();
    Navigator.of(context).pop(location);
  }

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    final manual = parseLatLon(_manual.text);

    return Scaffold(
      backgroundColor: c.paper,
      appBar: AppBar(
        title: const Text('Where is this?'),
        backgroundColor: c.paper,
        foregroundColor: c.ink,
        elevation: 0,
      ),
      body: ListView(
        padding: const EdgeInsets.only(bottom: AppTokens.s32),
        children: [
          const StencilLabel('Look it up'),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: AppTokens.gutter),
            child: Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: _query,
                    textInputAction: TextInputAction.search,
                    style: AppTokens.rowTitleStyle.copyWith(color: c.ink),
                    decoration: InputDecoration(
                      hintText: 'Cherrapunji, Meghalaya',
                      hintStyle: AppTokens.rowTitleStyle.copyWith(
                        color: c.rule,
                      ),
                      border: UnderlineInputBorder(
                        borderSide: BorderSide(color: c.rule),
                      ),
                    ),
                    onSubmitted: (_) => _run(),
                  ),
                ),
                const SizedBox(width: AppTokens.s12),
                PressScale(
                  onTap: _searching ? null : _run,
                  child: Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: AppTokens.s16,
                      vertical: AppTokens.s12,
                    ),
                    decoration: BoxDecoration(
                      color: _searching ? c.stone : c.signal,
                      border: Border.all(color: _searching ? c.rule : c.ink),
                      borderRadius: BorderRadius.circular(
                        AppTokens.radiusSoft,
                      ),
                    ),
                    child: Text(
                      _searching ? 'Looking' : 'Search',
                      style: AppTokens.stencilStyle.copyWith(
                        fontSize: 10,
                        color: _searching ? c.muted : c.paper,
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ),

          if (_error != null)
            Padding(
              padding: const EdgeInsets.fromLTRB(
                AppTokens.gutter,
                AppTokens.s12,
                AppTokens.gutter,
                0,
              ),
              child: Text(
                _error!,
                style: AppTokens.captionStyle.copyWith(color: c.cautionMark),
              ),
            ),

          for (final result in _results ?? const <GeocodeResult>[])
            _ResultRow(result: result, onTap: () => _take(result.location)),

          const StencilLabel('Or type the coordinates'),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: AppTokens.gutter),
            child: TextField(
              controller: _manual,
              style: AppTokens.numberStyle.copyWith(color: c.ink),
              decoration: InputDecoration(
                hintText: '25.5788, 91.8933',
                hintStyle: AppTokens.numberStyle.copyWith(color: c.rule),
                border: UnderlineInputBorder(
                  borderSide: BorderSide(color: c.rule),
                ),
              ),
              onChanged: (_) => setState(() {}),
            ),
          ),
          Padding(
            padding: const EdgeInsets.fromLTRB(
              AppTokens.gutter,
              AppTokens.s8,
              AppTokens.gutter,
              0,
            ),
            child: Text(
              'For a homestay no map knows about. Long-press it in Google '
              'Maps and copy what appears.',
              style: AppTokens.captionStyle.copyWith(color: c.muted),
            ),
          ),
          if (manual != null)
            Padding(
              padding: const EdgeInsets.fromLTRB(
                AppTokens.gutter,
                AppTokens.s16,
                AppTokens.gutter,
                0,
              ),
              child: PressScale(
                onTap: () => _take(manual),
                child: Container(
                  height: 44,
                  alignment: Alignment.center,
                  decoration: BoxDecoration(
                    color: c.signal,
                    border: Border.all(color: c.ink),
                    borderRadius: BorderRadius.circular(AppTokens.radiusSoft),
                  ),
                  child: Text(
                    'Use these coordinates',
                    style: AppTokens.stencilStyle.copyWith(
                      fontSize: 10.5,
                      color: c.paper,
                    ),
                  ),
                ),
              ),
            ),

          const StencilLabel('About this lookup'),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: AppTokens.gutter),
            child: Text(
              'Names are looked up against OpenStreetMap, and that is the '
              'only moment this screen uses the network. Do it on WiFi '
              'before you leave.',
              style: AppTokens.captionStyle.copyWith(color: c.muted),
            ),
          ),
        ],
      ),
    );
  }
}

class _ResultRow extends StatelessWidget {
  final GeocodeResult result;
  final VoidCallback onTap;

  const _ResultRow({required this.result, required this.onTap});

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
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Expanded(
                  child: Text(
                    result.shortName,
                    style: AppTokens.rowTitleStyle.copyWith(color: c.ink),
                  ),
                ),
                if (result.kind != null)
                  Text(
                    result.kind!.toUpperCase(),
                    style: AppTokens.stencilStyle.copyWith(
                      fontSize: 9,
                      color: c.muted,
                    ),
                  ),
              ],
            ),
            const SizedBox(height: 2),
            // The full name, because this is the line that tells two places
            // with the same name apart.
            Text(
              result.displayName,
              style: AppTokens.captionStyle.copyWith(color: c.muted),
            ),
            const SizedBox(height: 2),
            Text(
              formatLatLon(result.location),
              style: AppTokens.numberStyle.copyWith(
                color: c.muted,
                fontSize: 11,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
