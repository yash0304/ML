// lib/features/settings/presentation/settings_screen.dart
//
// Issue #35. Three settings, the call history, and what each trip is holding.

import 'package:flutter/material.dart';

import '../../../core/theme/app_tokens.dart';
import '../../../core/theme/motion.dart';
import '../../../core/widgets/retro.dart';
import '../data/settings.dart';

class SettingsScreen extends StatelessWidget {
  final Stream<ThemeMode> themeMode;
  final Future<void> Function(ThemeMode) onThemeMode;
  final Stream<double> corridorKm;
  final Future<void> Function(double) onCorridorKm;
  final Stream<List<CacheSummary>> caches;
  final Future<void> Function(int tripId) onClearCache;
  final VoidCallback onCallHistory;

  /// The map provider's key, as typed on this phone. Empty means none.
  final Stream<String>? mapKey;
  final Future<void> Function(String)? onMapKey;

  /// True when the build itself carries a key. Then a typed one is a
  /// replacement rather than the only way in, and the copy has to say so.
  final bool hasBuildKey;

  /// Named so the section can say whose key it wants.
  final String mapProviderLabel;

  const SettingsScreen({
    super.key,
    required this.themeMode,
    required this.onThemeMode,
    required this.corridorKm,
    required this.onCorridorKm,
    required this.caches,
    required this.onClearCache,
    required this.onCallHistory,
    this.mapKey,
    this.onMapKey,
    this.hasBuildKey = false,
    this.mapProviderLabel = 'MapTiler',
  });

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);

    return Scaffold(
      backgroundColor: c.paper,
      appBar: AppBar(
        title: const Text('Settings'),
        backgroundColor: c.paper,
        foregroundColor: c.ink,
        elevation: 0,
      ),
      body: ListView(
        padding: const EdgeInsets.only(bottom: AppTokens.s32),
        children: [
          const StencilLabel('Look'),
          StreamBuilder<ThemeMode>(
            stream: themeMode,
            builder: (context, snap) {
              final mode = snap.data ?? ThemeMode.system;
              return Padding(
                padding: const EdgeInsets.symmetric(
                  horizontal: AppTokens.gutter,
                ),
                child: Wrap(
                  spacing: AppTokens.s8,
                  children: [
                    for (final option in ThemeMode.values)
                      _Chip(
                        label: switch (option) {
                          ThemeMode.system => 'Follow phone',
                          ThemeMode.light => 'Paper',
                          ThemeMode.dark => 'Lamp',
                        },
                        on: mode == option,
                        onTap: () {
                          Haptics.select();
                          onThemeMode(option);
                        },
                      ),
                  ],
                ),
              );
            },
          ),
          Padding(
            padding: const EdgeInsets.fromLTRB(
              AppTokens.gutter,
              AppTokens.s8,
              AppTokens.gutter,
              0,
            ),
            child: Text(
              'A phone in a pocket does not know it is night in a valley.',
              style: AppTokens.captionStyle.copyWith(color: c.muted),
            ),
          ),

          const StencilLabel('Corridor width'),
          StreamBuilder<double>(
            stream: corridorKm,
            builder: (context, snap) {
              final km = snap.data ?? 3.0;
              return Padding(
                padding: const EdgeInsets.symmetric(
                  horizontal: AppTokens.gutter,
                ),
                child: Wrap(
                  spacing: AppTokens.s8,
                  children: [
                    for (final option in const [1.0, 3.0, 5.0, 10.0])
                      _Chip(
                        label: '${option.round()} km',
                        on: km == option,
                        onTap: () {
                          Haptics.select();
                          onCorridorKm(option);
                        },
                      ),
                  ],
                ),
              );
            },
          ),
          Padding(
            padding: const EdgeInsets.fromLTRB(
              AppTokens.gutter,
              AppTokens.s8,
              AppTokens.gutter,
              0,
            ),
            child: Text(
              'How far either side of the road to look for places. Wider '
              'finds more and takes longer to download; anything you would '
              'have to leave the route for an hour to reach is not really on '
              'your way.',
              style: AppTokens.captionStyle.copyWith(color: c.muted),
            ),
          ),

          const StencilLabel('History'),
          PressScale(
            onTap: onCallHistory,
            child: Container(
              padding: const EdgeInsets.symmetric(
                horizontal: AppTokens.gutter,
                vertical: AppTokens.s16,
              ),
              decoration: BoxDecoration(
                border: Border(
                  bottom: BorderSide(color: c.rule, width: AppTokens.hairline),
                ),
              ),
              child: Row(
                children: [
                  Icon(Icons.history, size: 20, color: c.ink),
                  const SizedBox(width: AppTokens.s16),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'What you have called',
                          style: AppTokens.rowTitleStyle.copyWith(color: c.ink),
                        ),
                        const SizedBox(height: 2),
                        Text(
                          'Copies count too — the dial happens in the phone '
                          'dialer, so the app never sees the call itself.',
                          style: AppTokens.captionStyle.copyWith(
                            color: c.muted,
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ),

          if (mapKey != null && onMapKey != null) ...[
            const StencilLabel('Map key'),
            StreamBuilder<String>(
              stream: mapKey,
              builder: (context, snap) => _MapKeyField(
                // Keyed on what is stored, so the field is rebuilt from
                // scratch when the saved key changes underneath it rather
                // than holding a stale draft.
                key: ValueKey(snap.data ?? ''),
                saved: snap.data ?? '',
                provider: mapProviderLabel,
                hasBuildKey: hasBuildKey,
                onSave: onMapKey!,
              ),
            ),
          ],

          const StencilLabel('Downloaded'),
          StreamBuilder<List<CacheSummary>>(
            stream: caches,
            builder: (context, snap) {
              final list = snap.data;
              if (list == null) return const SizedBox();
              if (list.isEmpty) {
                return Padding(
                  padding: const EdgeInsets.symmetric(
                    horizontal: AppTokens.gutter,
                  ),
                  child: Text(
                    'No trips yet.',
                    style: AppTokens.captionStyle.copyWith(color: c.muted),
                  ),
                );
              }
              return Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  for (final cache in list)
                    _CacheRow(
                      cache: cache,
                      onClear: () => _confirmClear(context, cache),
                    ),
                  Padding(
                    padding: const EdgeInsets.fromLTRB(
                      AppTokens.gutter,
                      AppTokens.s12,
                      AppTokens.gutter,
                      0,
                    ),
                    child: Text(
                      'Clearing removes only what came off the network. Your '
                      'contacts, expenses, itinerary and checklist are yours '
                      'and stay put.',
                      style: AppTokens.captionStyle.copyWith(color: c.muted),
                    ),
                  ),
                ],
              );
            },
          ),
        ],
      ),
    );
  }

  Future<void> _confirmClear(BuildContext context, CacheSummary cache) async {
    final c = AppTokens.of(context);
    final ok = await showDialog<bool>(
      context: context,
      builder: (dialogContext) => AlertDialog(
        backgroundColor: c.paper,
        title: Text(
          'Clear downloaded data?',
          style: AppTokens.titleStyle.copyWith(color: c.ink, fontSize: 18),
        ),
        content: Text(
          'Removes ${cache.poiCount} places, '
          '${cache.routedLegCount} routes and ${cache.weatherDayCount} days '
          'of forecast for ${cache.tripName}.\n\n'
          'You will need WiFi to get them back.',
          style: AppTokens.captionStyle.copyWith(color: c.muted),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(dialogContext).pop(false),
            child: Text('Keep', style: TextStyle(color: c.muted)),
          ),
          TextButton(
            onPressed: () => Navigator.of(dialogContext).pop(true),
            child: Text('Clear', style: TextStyle(color: c.emergency)),
          ),
        ],
      ),
    );
    if (ok == true) {
      Haptics.grave();
      await onClearCache(cache.tripId);
    }
  }
}

class _CacheRow extends StatelessWidget {
  final CacheSummary cache;
  final VoidCallback onClear;

  const _CacheRow({required this.cache, required this.onClear});

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
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  cache.tripName,
                  style: AppTokens.rowTitleStyle.copyWith(color: c.ink),
                ),
                const SizedBox(height: 2),
                Text(
                  cache.isEmpty
                      ? 'Nothing downloaded'
                      : '${cache.poiCount} places · '
                            '${cache.routedLegCount}/${cache.legCount} legs '
                            'routed · ${cache.weatherDayCount} forecast days',
                  style: AppTokens.captionStyle.copyWith(color: c.muted),
                ),
              ],
            ),
          ),
          if (!cache.isEmpty)
            GestureDetector(
              onTap: onClear,
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
  }
}

class CallHistoryScreen extends StatelessWidget {
  final Stream<List<CallLogEntry>> history;
  const CallHistoryScreen({super.key, required this.history});

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    return Scaffold(
      backgroundColor: c.paper,
      appBar: AppBar(
        title: const Text('What you have called'),
        backgroundColor: c.paper,
        foregroundColor: c.ink,
        elevation: 0,
      ),
      body: StreamBuilder<List<CallLogEntry>>(
        stream: history,
        builder: (context, snap) {
          final list = snap.data;
          if (list == null) return const SizedBox();
          if (list.isEmpty) {
            return Center(
              child: Padding(
                padding: const EdgeInsets.all(AppTokens.s32),
                child: Text(
                  'Nothing yet. Copying a number counts, so this fills up as '
                  'soon as you start using the diary.',
                  textAlign: TextAlign.center,
                  style: AppTokens.captionStyle.copyWith(color: c.muted),
                ),
              ),
            );
          }
          return ListView.builder(
            itemCount: list.length,
            itemBuilder: (context, i) => _HistoryRow(entry: list[i]),
          );
        },
      ),
    );
  }
}

class _HistoryRow extends StatelessWidget {
  final CallLogEntry entry;
  const _HistoryRow({required this.entry});

  static const _labels = {
    'copy': 'Copied',
    'dialer': 'Opened the dialer',
    'call': 'Called',
    'sms': 'Texted',
    'whatsapp': 'WhatsApp',
  };

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    final at = entry.occurredAt;

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
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  entry.contactName,
                  style: AppTokens.rowTitleStyle.copyWith(color: c.ink),
                ),
                const SizedBox(height: 2),
                Text(
                  entry.phoneRaw,
                  style: AppTokens.numberStyle.copyWith(color: c.muted),
                ),
                const SizedBox(height: 2),
                Text(
                  (_labels[entry.action] ?? entry.action).toUpperCase(),
                  style: AppTokens.stencilStyle.copyWith(
                    fontSize: 9,
                    color: c.muted,
                  ),
                ),
              ],
            ),
          ),
          Text(
            '${at.day.toString().padLeft(2, '0')}/'
            '${at.month.toString().padLeft(2, '0')}  '
            '${at.hour.toString().padLeft(2, '0')}:'
            '${at.minute.toString().padLeft(2, '0')}',
            style: AppTokens.numberStyle.copyWith(
              color: c.muted,
              fontSize: 11.5,
            ),
          ),
        ],
      ),
    );
  }
}

/// Where the map key is typed in.
///
/// This exists because the build-time route was not enough: an APK produced by
/// CI with no secret configured has no way of being given a key, which is
/// exactly the position the first user of this app was in.
class _MapKeyField extends StatefulWidget {
  final String saved;
  final String provider;
  final bool hasBuildKey;
  final Future<void> Function(String) onSave;

  const _MapKeyField({
    super.key,
    required this.saved,
    required this.provider,
    required this.hasBuildKey,
    required this.onSave,
  });

  @override
  State<_MapKeyField> createState() => _MapKeyFieldState();
}

class _MapKeyFieldState extends State<_MapKeyField> {
  late final _controller = TextEditingController(text: widget.saved);
  bool _hidden = true;
  bool _justSaved = false;

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  bool get _changed => _controller.text.trim() != widget.saved;

  Future<void> _save() async {
    Haptics.light();
    await widget.onSave(_controller.text.trim());
    if (mounted) setState(() => _justSaved = true);
  }

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    final active = widget.saved.isNotEmpty;

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: AppTokens.gutter),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Expanded(
                child: TextField(
                  controller: _controller,
                  obscureText: _hidden && active,
                  autocorrect: false,
                  enableSuggestions: false,
                  onChanged: (_) => setState(() => _justSaved = false),
                  style: AppTokens.numberStyle.copyWith(color: c.ink),
                  decoration: InputDecoration(
                    isDense: true,
                    hintText: 'Paste your ${widget.provider} key',
                    hintStyle: AppTokens.captionStyle.copyWith(color: c.muted),
                    enabledBorder: UnderlineInputBorder(
                      borderSide: BorderSide(color: c.rule),
                    ),
                    focusedBorder: UnderlineInputBorder(
                      borderSide: BorderSide(color: c.signal),
                    ),
                  ),
                ),
              ),
              if (active)
                IconButton(
                  onPressed: () => setState(() => _hidden = !_hidden),
                  color: c.muted,
                  icon: Icon(
                    _hidden ? Icons.visibility_outlined : Icons.visibility_off_outlined,
                    size: 18,
                  ),
                ),
            ],
          ),
          const SizedBox(height: AppTokens.s8),
          Row(
            children: [
              PressScale(
                onTap: _changed ? _save : null,
                child: Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: AppTokens.s16,
                    vertical: AppTokens.s8,
                  ),
                  decoration: BoxDecoration(
                    color: _changed ? c.signal : Colors.transparent,
                    border: Border.all(color: _changed ? c.ink : c.rule),
                    borderRadius: BorderRadius.circular(AppTokens.radiusSoft),
                  ),
                  child: Text(
                    // Uppercase, like every other stencil control in the app.
                    'SAVE',
                    style: AppTokens.stencilStyle.copyWith(
                      fontSize: 10.5,
                      color: _changed ? c.paper : c.muted,
                    ),
                  ),
                ),
              ),
              const SizedBox(width: AppTokens.s12),
              Expanded(
                child: Text(
                  _justSaved && !_changed
                      ? 'Saved. Download the map from More → Map.'
                      : active
                      ? 'A key is set. Maps can download.'
                      : widget.hasBuildKey
                      ? 'This build already carries a key. Anything typed '
                            'here replaces it.'
                      : 'No key yet, so maps stay off.',
                  style: AppTokens.captionStyle.copyWith(
                    color: active || widget.hasBuildKey
                        ? c.muted
                        : c.cautionMark,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: AppTokens.s8),
          Text(
            // Said plainly, because "API key" means nothing to most people and
            // a key pasted into the wrong app is a real cost.
            'A free ${widget.provider} account gives you one. It stays on this '
            'phone, in this app, and is only ever sent to ${widget.provider} '
            'while a map is downloading — never on the road.',
            style: AppTokens.captionStyle.copyWith(color: c.muted),
          ),
        ],
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
      onTap: onTap,
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
