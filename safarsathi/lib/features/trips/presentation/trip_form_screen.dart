// lib/features/trips/presentation/trip_form_screen.dart
//
// Creating and renaming a trip — issue #16.

import 'package:flutter/material.dart';

import '../../../core/theme/app_tokens.dart';
import '../../../core/theme/motion.dart';
import '../../../core/widgets/retro.dart';

class TripFormScreen extends StatefulWidget {
  final String? initialName;
  final DateTime? initialStart;
  final DateTime? initialEnd;
  final Future<void> Function(String name, DateTime? start, DateTime? end)
  onSave;

  const TripFormScreen({
    super.key,
    required this.onSave,
    this.initialName,
    this.initialStart,
    this.initialEnd,
  });

  @override
  State<TripFormScreen> createState() => _TripFormScreenState();
}

class _TripFormScreenState extends State<TripFormScreen> {
  late final _name = TextEditingController(text: widget.initialName ?? '');
  late DateTime? _start = widget.initialStart;
  late DateTime? _end = widget.initialEnd;
  bool _saving = false;

  bool get _isEdit => widget.initialName != null;

  @override
  void dispose() {
    _name.dispose();
    super.dispose();
  }

  Future<void> _pick({required bool start}) async {
    final picked = await showDatePicker(
      context: context,
      initialDate: (start ? _start : _end) ?? _start ?? DateTime.now(),
      firstDate: DateTime(DateTime.now().year - 1),
      lastDate: DateTime(DateTime.now().year + 5),
    );
    if (picked == null) return;
    setState(() => start ? _start = picked : _end = picked);
    Haptics.select();
  }

  Future<void> _save() async {
    final name = _name.text.trim();
    if (name.isEmpty || _saving) {
      if (name.isEmpty) Haptics.reject();
      return;
    }
    setState(() => _saving = true);
    await widget.onSave(name, _start, _end);
    if (mounted) setState(() => _saving = false);
  }

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    return Scaffold(
      backgroundColor: c.paper,
      appBar: AppBar(
        title: Text(_isEdit ? 'Edit trip' : 'New trip'),
        backgroundColor: c.paper,
        foregroundColor: c.ink,
        elevation: 0,
      ),
      body: ListView(
        children: [
          const StencilLabel('Name'),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: AppTokens.gutter),
            child: TextField(
              controller: _name,
              autofocus: !_isEdit,
              textCapitalization: TextCapitalization.words,
              style: AppTokens.titleStyle.copyWith(color: c.ink, fontSize: 20),
              decoration: InputDecoration(
                hintText: 'Meghalaya, October',
                hintStyle: AppTokens.titleStyle.copyWith(
                  color: c.rule,
                  fontSize: 20,
                ),
                border: UnderlineInputBorder(
                  borderSide: BorderSide(color: c.rule),
                ),
              ),
            ),
          ),
          const StencilLabel('Dates'),
          _Row(
            label: 'From',
            value: _start,
            onTap: () => _pick(start: true),
            onClear: _start == null ? null : () => setState(() => _start = null),
          ),
          _Row(
            label: 'To',
            value: _end,
            onTap: () => _pick(start: false),
            onClear: _end == null ? null : () => setState(() => _end = null),
          ),
          Padding(
            padding: const EdgeInsets.fromLTRB(
              AppTokens.gutter,
              AppTokens.s16,
              AppTokens.gutter,
              0,
            ),
            child: Text(
              'Dates are optional. Without them the app cannot work out which '
              'stop you are at, so it shows the first one until you say '
              'otherwise.',
              style: AppTokens.captionStyle.copyWith(color: c.muted),
            ),
          ),
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
              _isEdit ? 'Save trip' : 'Create trip',
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

class _Row extends StatelessWidget {
  final String label;
  final DateTime? value;
  final VoidCallback onTap;
  final VoidCallback? onClear;

  const _Row({
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
