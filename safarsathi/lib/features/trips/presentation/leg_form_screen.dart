// lib/features/trips/presentation/leg_form_screen.dart
//
// Transport details for one leg — issue #18.
//
// EVERYTHING HERE IS TYPED. There is no live schedule lookup and there never
// will be: a Rome2Rio-style fetch is exactly the runtime network dependency
// this project exists to avoid. What the app can do is hold what the user
// already knows, and have it in their hand with no signal.

import 'package:flutter/material.dart';

import '../../../core/theme/app_tokens.dart';
import '../../../core/theme/motion.dart';
import '../../../core/widgets/retro.dart';

const legModes = [
  'Car',
  'Taxi',
  'Shared sumo',
  'Bus',
  'Train',
  'Flight',
  'Ferry',
  'Walk',
];

class LegFormScreen extends StatefulWidget {
  final String fromName;
  final String toName;
  final String? mode;
  final DateTime? plannedDeparture;
  final DateTime? plannedArrival;
  final bool isBooked;
  final String? note;
  final double? distanceKm;

  final Future<void> Function({
    String? mode,
    DateTime? plannedDeparture,
    DateTime? plannedArrival,
    required bool isBooked,
    String? note,
  })
  onSave;

  const LegFormScreen({
    super.key,
    required this.fromName,
    required this.toName,
    required this.onSave,
    this.mode,
    this.plannedDeparture,
    this.plannedArrival,
    this.isBooked = false,
    this.note,
    this.distanceKm,
  });

  @override
  State<LegFormScreen> createState() => _LegFormScreenState();
}

class _LegFormScreenState extends State<LegFormScreen> {
  late String? _mode = widget.mode;
  late DateTime? _departure = widget.plannedDeparture;
  late DateTime? _arrival = widget.plannedArrival;
  late bool _booked = widget.isBooked;
  late final _note = TextEditingController(text: widget.note ?? '');
  bool _saving = false;

  @override
  void dispose() {
    _note.dispose();
    super.dispose();
  }

  Future<void> _pick({required bool departure}) async {
    final base = (departure ? _departure : _arrival) ?? DateTime.now();
    final date = await showDatePicker(
      context: context,
      initialDate: base,
      firstDate: DateTime(DateTime.now().year - 1),
      lastDate: DateTime(DateTime.now().year + 5),
    );
    if (date == null || !mounted) return;
    final time = await showTimePicker(
      context: context,
      initialTime: TimeOfDay.fromDateTime(base),
    );
    if (time == null) return;
    final value = DateTime(
      date.year,
      date.month,
      date.day,
      time.hour,
      time.minute,
    );
    setState(() => departure ? _departure = value : _arrival = value);
    Haptics.select();
  }

  Future<void> _save() async {
    if (_saving) return;
    setState(() => _saving = true);
    await widget.onSave(
      mode: _mode,
      plannedDeparture: _departure,
      plannedArrival: _arrival,
      isBooked: _booked,
      note: _note.text.trim().isEmpty ? null : _note.text.trim(),
    );
    if (mounted) setState(() => _saving = false);
  }

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);

    return Scaffold(
      backgroundColor: c.paper,
      appBar: AppBar(
        title: const Text('Getting there'),
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
            child: Row(
              children: [
                Expanded(
                  child: Text(
                    '${widget.fromName} → ${widget.toName}',
                    style: AppTokens.titleStyle.copyWith(
                      color: c.ink,
                      fontSize: 18,
                    ),
                  ),
                ),
                if (widget.distanceKm != null)
                  Text(
                    '${widget.distanceKm!.round()} km',
                    style: AppTokens.numberStyle.copyWith(color: c.muted),
                  ),
              ],
            ),
          ),

          const StencilLabel('How'),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: AppTokens.gutter),
            child: Wrap(
              spacing: AppTokens.s8,
              runSpacing: AppTokens.s8,
              children: [
                for (final m in legModes)
                  _Chip(
                    label: m,
                    on: _mode == m,
                    onTap: () {
                      setState(() => _mode = _mode == m ? null : m);
                      Haptics.select();
                    },
                  ),
              ],
            ),
          ),

          const StencilLabel('When'),
          _TimeRow(
            label: 'Leave',
            value: _departure,
            onTap: () => _pick(departure: true),
            onClear: _departure == null
                ? null
                : () => setState(() => _departure = null),
          ),
          _TimeRow(
            label: 'Arrive',
            value: _arrival,
            onTap: () => _pick(departure: false),
            onClear: _arrival == null
                ? null
                : () => setState(() => _arrival = null),
          ),
          Padding(
            padding: const EdgeInsets.fromLTRB(
              AppTokens.gutter,
              AppTokens.s8,
              AppTokens.gutter,
              0,
            ),
            child: Text(
              'Typed, not looked up. The app makes no network call, so these '
              'are the times you were told, held where you can read them with '
              'no signal.',
              style: AppTokens.captionStyle.copyWith(color: c.muted),
            ),
          ),

          const StencilLabel('Booked'),
          PressScale(
            onTap: () {
              setState(() => _booked = !_booked);
              Haptics.light();
            },
            child: Container(
              padding: const EdgeInsets.symmetric(
                horizontal: AppTokens.gutter,
                vertical: AppTokens.s12,
              ),
              child: Row(
                children: [
                  Container(
                    width: 18,
                    height: 18,
                    alignment: Alignment.center,
                    decoration: BoxDecoration(
                      border: Border.all(color: c.ink),
                      color: _booked ? c.signal : Colors.transparent,
                    ),
                    child: _booked
                        ? Icon(Icons.check, size: 13, color: c.paper)
                        : const SizedBox.shrink(),
                  ),
                  const SizedBox(width: AppTokens.s12),
                  Text(
                    _booked ? 'Booked' : 'Not booked yet',
                    style: AppTokens.rowTitleStyle.copyWith(color: c.ink),
                  ),
                ],
              ),
            ),
          ),

          const StencilLabel('Note'),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: AppTokens.gutter),
            child: TextField(
              controller: _note,
              maxLines: 3,
              minLines: 1,
              style: AppTokens.captionStyle.copyWith(color: c.ink),
              decoration: InputDecoration(
                hintText: 'Sumo stand behind Police Bazar, leaves when full',
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
              'Save',
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

class _TimeRow extends StatelessWidget {
  final String label;
  final DateTime? value;
  final VoidCallback onTap;
  final VoidCallback? onClear;

  const _TimeRow({
    required this.label,
    required this.value,
    required this.onTap,
    this.onClear,
  });

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    final d = value;
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
                          '${d.month.toString().padLeft(2, '0')}  '
                          '${d.hour.toString().padLeft(2, '0')}:'
                          '${d.minute.toString().padLeft(2, '0')}',
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
