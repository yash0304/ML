// lib/features/trips/presentation/stop_form_screen.dart
//
// Adding and editing one stop — issue #16.
//
// The form takes a draft and hands back a draft. It never touches the
// database, which keeps it testable and keeps the "a place may appear twice"
// rule where it belongs: in the editor, which appends rather than upserting
// by name.

import 'package:flutter/material.dart';

import '../../../core/theme/app_tokens.dart';
import '../../../core/theme/motion.dart';
import '../../../core/widgets/retro.dart';
import '../../discovery/data/geo.dart';
import '../../discovery/data/geocoder.dart' show formatLatLon;
import '../data/trip_editor.dart';

class StopFormScreen extends StatefulWidget {
  final StopDraft? existing;
  final Future<void> Function(StopDraft draft) onSave;

  /// Null for a new stop. Provided when editing, so the screen can offer to
  /// delete and say what deleting costs.
  final Future<int> Function()? contactsHere;
  final Future<void> Function()? onDelete;

  /// Opens the place picker and returns the chosen coordinates, or null.
  /// Optional so the golden harness and the widget tests render without one.
  final Future<LatLng?> Function(String name, LatLng? current)? onPickPlace;

  const StopFormScreen({
    super.key,
    required this.onSave,
    this.existing,
    this.contactsHere,
    this.onDelete,
    this.onPickPlace,
  });

  @override
  State<StopFormScreen> createState() => _StopFormScreenState();
}

class _StopFormScreenState extends State<StopFormScreen> {
  late StopDraft _draft =
      widget.existing ?? const StopDraft(name: '', countryCode: 'IN');
  late final _name = TextEditingController(text: _draft.name);
  late final _note = TextEditingController(text: _draft.note ?? '');
  late final _nights = TextEditingController(text: '${_draft.nights}');
  bool _saving = false;

  bool get _isEdit => widget.existing?.id != null;
  bool get _datesDriveNights =>
      _draft.arrivalDate != null && _draft.departureDate != null;

  @override
  void dispose() {
    _name.dispose();
    _note.dispose();
    _nights.dispose();
    super.dispose();
  }

  Future<void> _pickDate({required bool arrival}) async {
    final initial =
        (arrival ? _draft.arrivalDate : _draft.departureDate) ??
        _draft.arrivalDate ??
        DateTime.now();
    final picked = await showDatePicker(
      context: context,
      initialDate: initial,
      firstDate: DateTime(DateTime.now().year - 1),
      lastDate: DateTime(DateTime.now().year + 5),
    );
    if (picked == null) return;
    setState(() {
      _draft = arrival
          ? _draft.copyWith(arrivalDate: picked)
          : _draft.copyWith(departureDate: picked);
      if (_datesDriveNights) {
        _nights.text = '${_draft.effectiveNights}';
      }
    });
    Haptics.select();
  }

  void _clearDate({required bool arrival}) {
    setState(() {
      _draft = arrival
          ? _draft.copyWith(arrivalDate: null)
          : _draft.copyWith(departureDate: null);
    });
  }

  void _toggleTag(String tag) {
    final tags = List.of(_draft.activityTags);
    tags.contains(tag) ? tags.remove(tag) : tags.add(tag);
    setState(() => _draft = _draft.copyWith(activityTags: tags));
    Haptics.select();
  }

  Future<void> _save() async {
    if (_saving) return;
    final name = _name.text.trim();
    if (name.isEmpty) {
      Haptics.reject();
      return;
    }
    setState(() => _saving = true);
    final draft = _draft.copyWith(
      name: name,
      note: _note.text.trim().isEmpty ? null : _note.text.trim(),
      nights: _datesDriveNights
          ? _draft.effectiveNights
          : int.tryParse(_nights.text.trim()) ?? 0,
    );
    await widget.onSave(draft);
    if (mounted) setState(() => _saving = false);
  }

  Future<void> _confirmDelete() async {
    final count = await widget.contactsHere?.call() ?? 0;
    if (!mounted) return;
    final c = AppTokens.of(context);

    final ok = await showDialog<bool>(
      context: context,
      builder: (dialogContext) => AlertDialog(
        backgroundColor: c.paper,
        title: Text(
          'Remove ${_draft.name}?',
          style: AppTokens.titleStyle.copyWith(color: c.ink, fontSize: 18),
        ),
        content: Text(
          count == 0
              ? 'The legs either side of it will join up.'
              : 'The legs either side will join up. '
                    '$count ${count == 1 ? 'contact stays' : 'contacts stay'} '
                    'in your diary, no longer attached to a stop — a number '
                    'you have confirmed is still a real number.',
          style: AppTokens.captionStyle.copyWith(color: c.muted),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(dialogContext).pop(false),
            child: Text('Keep', style: TextStyle(color: c.muted)),
          ),
          TextButton(
            onPressed: () => Navigator.of(dialogContext).pop(true),
            child: Text('Remove', style: TextStyle(color: c.emergency)),
          ),
        ],
      ),
    );
    if (ok == true) {
      Haptics.grave();
      await widget.onDelete?.call();
    }
  }

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);

    return Scaffold(
      backgroundColor: c.paper,
      appBar: AppBar(
        title: Text(_isEdit ? 'Edit stop' : 'Add a stop'),
        backgroundColor: c.paper,
        foregroundColor: c.ink,
        elevation: 0,
        actions: [
          if (_isEdit && widget.onDelete != null)
            IconButton(
              onPressed: _confirmDelete,
              icon: const Icon(Icons.delete_outline),
              color: c.muted,
            ),
        ],
      ),
      body: ListView(
        padding: const EdgeInsets.only(bottom: 96),
        children: [
          const StencilLabel('Place'),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: AppTokens.gutter),
            child: TextField(
              controller: _name,
              autofocus: !_isEdit,
              textCapitalization: TextCapitalization.words,
              style: AppTokens.rowTitleStyle.copyWith(color: c.ink),
              decoration: InputDecoration(
                hintText: 'Shillong',
                hintStyle: AppTokens.rowTitleStyle.copyWith(color: c.rule),
                border: UnderlineInputBorder(
                  borderSide: BorderSide(color: c.rule),
                ),
              ),
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
              // The rule, said out loud, because it looks like a mistake.
              'The same place can appear more than once. Shillong on the way '
              'out and Shillong on the way back are two stops.',
              style: AppTokens.captionStyle.copyWith(color: c.muted),
            ),
          ),

          const StencilLabel('Dates'),
          _DateRow(
            label: 'Arrive',
            value: _draft.arrivalDate,
            onPick: () => _pickDate(arrival: true),
            onClear: _draft.arrivalDate == null
                ? null
                : () => _clearDate(arrival: true),
          ),
          _DateRow(
            label: 'Leave',
            value: _draft.departureDate,
            onPick: () => _pickDate(arrival: false),
            onClear: _draft.departureDate == null
                ? null
                : () => _clearDate(arrival: false),
          ),

          const StencilLabel('Nights'),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: AppTokens.gutter),
            child: Row(
              children: [
                SizedBox(
                  width: 64,
                  child: TextField(
                    controller: _nights,
                    enabled: !_datesDriveNights,
                    keyboardType: TextInputType.number,
                    style: AppTokens.numberStyle.copyWith(
                      color: _datesDriveNights ? c.muted : c.ink,
                      fontSize: 18,
                    ),
                    decoration: InputDecoration(
                      isDense: true,
                      border: UnderlineInputBorder(
                        borderSide: BorderSide(color: c.rule),
                      ),
                    ),
                    onChanged: (v) => setState(
                      () => _draft = _draft.copyWith(
                        nights: int.tryParse(v.trim()) ?? 0,
                      ),
                    ),
                  ),
                ),
                const SizedBox(width: AppTokens.s16),
                Expanded(
                  child: Text(
                    _datesDriveNights
                        ? 'Worked out from the dates above.'
                        : 'A stop with a night on it is one the app will '
                              'block departure over until you have a '
                              'confirmed number for it.',
                    style: AppTokens.captionStyle.copyWith(color: c.muted),
                  ),
                ),
              ],
            ),
          ),

          const StencilLabel('What happens here'),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: AppTokens.gutter),
            child: Wrap(
              spacing: AppTokens.s8,
              runSpacing: AppTokens.s8,
              children: [
                for (final tag in knownActivityTags)
                  _TagChip(
                    label: tag,
                    on: _draft.activityTags.contains(tag),
                    onTap: () => _toggleTag(tag),
                  ),
              ],
            ),
          ),

          if (widget.onPickPlace != null) ...[
            const StencilLabel('Where it is'),
            _CoordinateRow(
              draft: _draft,
              onPick: () async {
                final picked = await widget.onPickPlace!(
                  _name.text.trim().isEmpty ? _draft.name : _name.text.trim(),
                  _draft.hasCoordinates
                      ? LatLng(_draft.lat!, _draft.lon!)
                      : null,
                );
                if (picked == null) return;
                setState(
                  () => _draft = _draft.copyWith(
                    lat: picked.lat,
                    lon: picked.lon,
                  ),
                );
              },
              onClear: () =>
                  setState(() => _draft = _draft.copyWith(lat: null, lon: null)),
            ),
          ],

          const StencilLabel('Note'),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: AppTokens.gutter),
            child: TextField(
              controller: _note,
              // Was three. A note holding what is at a stop — the pharmacy
              // that shuts at five, the hospital two villages over — runs
              // longer than that, and editing it through a three-line
              // window is how it stays unwritten.
              maxLines: 12,
              minLines: 2,
              style: AppTokens.captionStyle.copyWith(color: c.ink),
              decoration: InputDecoration(
                hintText: 'Blue gate past the church',
                hintStyle: AppTokens.captionStyle.copyWith(color: c.rule),
                border: OutlineInputBorder(
                  borderSide: BorderSide(color: c.rule),
                ),
              ),
            ),
          ),
          const SizedBox(height: AppTokens.s24),
        ],
      ),
      bottomNavigationBar: SafeArea(
        minimum: const EdgeInsets.all(AppTokens.s16),
        child: PressScale(
          onTap: _saving ? null : _save,
          child: Container(
            height: 48,
            alignment: Alignment.center,
            decoration: BoxDecoration(
              color: c.signal,
              border: Border.all(color: c.ink),
              borderRadius: BorderRadius.circular(AppTokens.radiusSoft),
            ),
            child: Text(
              _isEdit ? 'Save stop' : 'Add stop',
              style: AppTokens.stencilStyle.copyWith(
                fontSize: 11.5,
                color: c.paper,
              ),
            ),
          ),
        ),
      ),
    );
  }
}

class _DateRow extends StatelessWidget {
  final String label;
  final DateTime? value;
  final VoidCallback onPick;
  final VoidCallback? onClear;

  const _DateRow({
    required this.label,
    required this.value,
    required this.onPick,
    this.onClear,
  });

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    final d = value;
    return PressScale(
      onTap: onPick,
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
          children: [
            SizedBox(
              width: 72,
              child: Text(
                label,
                style: AppTokens.rowTitleStyle.copyWith(color: c.ink),
              ),
            ),
            Expanded(
              child: Text(
                d == null
                    ? 'Not set'
                    : '${d.day.toString().padLeft(2, '0')}/'
                          '${d.month.toString().padLeft(2, '0')}/${d.year}',
                style: d == null
                    ? AppTokens.captionStyle.copyWith(color: c.muted)
                    : AppTokens.numberStyle.copyWith(color: c.ink),
              ),
            ),
            if (onClear != null)
              GestureDetector(
                onTap: onClear,
                child: Icon(Icons.close, size: 18, color: c.muted),
              ),
          ],
        ),
      ),
    );
  }
}

class _TagChip extends StatelessWidget {
  final String label;
  final bool on;
  final VoidCallback onTap;

  const _TagChip({required this.label, required this.on, required this.onTap});

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

/// The coordinate line. Says plainly what is missing and why it matters,
/// because a stop with no coordinates silently excludes its leg from every
/// route and every place the corridor would have found.
class _CoordinateRow extends StatelessWidget {
  final StopDraft draft;
  final VoidCallback onPick;
  final VoidCallback onClear;

  const _CoordinateRow({
    required this.draft,
    required this.onPick,
    required this.onClear,
  });

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    final has = draft.hasCoordinates;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        PressScale(
          onTap: onPick,
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
              children: [
                Icon(
                  has ? Icons.place_outlined : Icons.search,
                  size: 18,
                  color: has ? c.ink : c.cautionMark,
                ),
                const SizedBox(width: AppTokens.s12),
                Expanded(
                  child: Text(
                    has
                        ? formatLatLon(LatLng(draft.lat!, draft.lon!))
                        : 'Not set — find it',
                    style: has
                        ? AppTokens.numberStyle.copyWith(color: c.ink)
                        : AppTokens.rowTitleStyle.copyWith(
                            color: c.cautionMark,
                          ),
                  ),
                ),
                if (has)
                  GestureDetector(
                    onTap: onClear,
                    child: Icon(Icons.close, size: 18, color: c.muted),
                  ),
              ],
            ),
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
            has
                ? 'Used to download the route and what is along it.'
                : 'Without this, the leg into and out of this stop cannot be '
                      'downloaded, and nothing along it will be found.',
            style: AppTokens.captionStyle.copyWith(
              color: has ? c.muted : c.cautionMark,
            ),
          ),
        ),
      ],
    );
  }
}
