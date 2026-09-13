// lib/features/money/presentation/expense_form_screen.dart
//
// Adding and editing an expense — issue #31.
//
// The screen's one job beyond collecting fields is to make an unbalanced
// split impossible to save. Shares that do not sum to the amount put the
// ledger permanently out by the difference, and nothing downstream would ever
// notice.

import 'package:flutter/material.dart';

import '../../../core/database/app_database.dart';
import '../../../core/theme/app_tokens.dart';
import '../../../core/theme/motion.dart';
import '../../../core/widgets/retro.dart';
import '../data/expense_editor.dart';
import '../data/settlement.dart';

class ExpenseFormScreen extends StatefulWidget {
  final List<Traveller> travellers;
  final ExpenseDraft? existing;
  final Future<void> Function(ExpenseDraft draft) onSave;
  final Future<void> Function()? onDelete;

  const ExpenseFormScreen({
    super.key,
    required this.travellers,
    required this.onSave,
    this.existing,
    this.onDelete,
  });

  @override
  State<ExpenseFormScreen> createState() => _ExpenseFormScreenState();
}

class _ExpenseFormScreenState extends State<ExpenseFormScreen> {
  late final _description = TextEditingController(
    text: widget.existing?.description ?? '',
  );
  late final _amount = TextEditingController(
    text: widget.existing == null
        ? ''
        : rupeesFieldText(widget.existing!.amountMinor),
  );

  late int _paidById =
      widget.existing?.paidById ?? widget.travellers.first.id;
  late DateTime _spentAt = widget.existing?.spentAt ?? DateTime.now();

  /// Who the expense is split between.
  late final Set<int> _between = widget.existing == null
      ? {for (final t in widget.travellers) t.id}
      : widget.existing!.shares.keys.toSet();

  /// Set only once the user takes the shares over by hand. Until then they
  /// follow the amount, which is what people expect: change the total, the
  /// even split changes with it.
  Map<int, int>? _customShares;

  bool _saving = false;

  bool get _isEdit => widget.existing?.id != null;
  int get _amountMinor => parseRupees(_amount.text) ?? 0;

  Map<int, int> get _shares {
    final custom = _customShares;
    if (custom != null) {
      return {
        for (final id in _between) id: custom[id] ?? 0,
      };
    }
    return evenSplit(_amountMinor, _between.toList());
  }

  int get _shareTotal => _shares.values.fold(0, (a, b) => a + b);
  int get _outBy => _amountMinor - _shareTotal;

  bool get _canSave =>
      _description.text.trim().isNotEmpty &&
      _amountMinor > 0 &&
      _between.isNotEmpty &&
      _outBy == 0;

  @override
  void initState() {
    super.initState();
    if (widget.existing != null) {
      // An existing expense may hold shares that are not an even split, and
      // reverting them to even on open would silently rewrite someone's
      // deliberate arrangement.
      final even = evenSplit(
        widget.existing!.amountMinor,
        widget.existing!.shares.keys.toList(),
      );
      final same = even.entries.every(
        (e) => widget.existing!.shares[e.key] == e.value,
      );
      if (!same) _customShares = Map.of(widget.existing!.shares);
    }
  }

  @override
  void dispose() {
    _description.dispose();
    _amount.dispose();
    super.dispose();
  }

  void _toggle(int id) {
    setState(() {
      if (_between.contains(id)) {
        _between.remove(id);
        _customShares?.remove(id);
      } else {
        _between.add(id);
        // A traveller added to a custom split starts at zero rather than
        // guessing, so the imbalance is visible and deliberate.
        _customShares?[id] = 0;
      }
    });
    Haptics.select();
  }

  void _takeOverShares() {
    setState(() => _customShares = Map.of(_shares));
    Haptics.light();
  }

  void _backToEven() {
    setState(() => _customShares = null);
    Haptics.light();
  }

  Future<void> _pickDate() async {
    final picked = await showDatePicker(
      context: context,
      initialDate: _spentAt,
      firstDate: DateTime(DateTime.now().year - 2),
      lastDate: DateTime(DateTime.now().year + 2),
    );
    if (picked != null) setState(() => _spentAt = picked);
  }

  Future<void> _save() async {
    if (!_canSave || _saving) {
      Haptics.reject();
      return;
    }
    setState(() => _saving = true);
    await widget.onSave(
      ExpenseDraft(
        id: widget.existing?.id,
        description: _description.text.trim(),
        amountMinor: _amountMinor,
        paidById: _paidById,
        shares: _shares,
        spentAt: _spentAt,
        stopId: widget.existing?.stopId,
        category: widget.existing?.category,
        currency: widget.existing?.currency ?? 'INR',
        rateToBase: widget.existing?.rateToBase ?? 1.0,
        rateCapturedAt: widget.existing?.rateCapturedAt,
      ),
    );
    if (mounted) setState(() => _saving = false);
  }

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);

    return Scaffold(
      backgroundColor: c.paper,
      appBar: AppBar(
        title: Text(_isEdit ? 'Edit expense' : 'Add an expense'),
        backgroundColor: c.paper,
        foregroundColor: c.ink,
        elevation: 0,
        actions: [
          if (_isEdit && widget.onDelete != null)
            IconButton(
              onPressed: widget.onDelete,
              icon: const Icon(Icons.delete_outline),
              color: c.muted,
            ),
        ],
      ),
      body: ListView(
        padding: const EdgeInsets.only(bottom: 96),
        children: [
          const StencilLabel('What for'),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: AppTokens.gutter),
            child: TextField(
              controller: _description,
              autofocus: !_isEdit,
              textCapitalization: TextCapitalization.sentences,
              style: AppTokens.rowTitleStyle.copyWith(color: c.ink),
              decoration: InputDecoration(
                hintText: 'Taxi, Shillong to Cherrapunji',
                hintStyle: AppTokens.rowTitleStyle.copyWith(color: c.rule),
                border: UnderlineInputBorder(
                  borderSide: BorderSide(color: c.rule),
                ),
              ),
              onChanged: (_) => setState(() {}),
            ),
          ),

          const StencilLabel('How much'),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: AppTokens.gutter),
            child: TextField(
              controller: _amount,
              keyboardType: const TextInputType.numberWithOptions(
                decimal: true,
              ),
              style: AppTokens.numberStyle.copyWith(color: c.ink, fontSize: 22),
              decoration: InputDecoration(
                prefixText: '₹ ',
                prefixStyle: AppTokens.numberStyle.copyWith(
                  color: c.muted,
                  fontSize: 22,
                ),
                hintText: '3200.11',
                hintStyle: AppTokens.numberStyle.copyWith(
                  color: c.rule,
                  fontSize: 22,
                ),
                border: UnderlineInputBorder(
                  borderSide: BorderSide(color: c.rule),
                ),
              ),
              onChanged: (_) => setState(() {}),
            ),
          ),

          const StencilLabel('Who paid'),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: AppTokens.gutter),
            child: Wrap(
              spacing: AppTokens.s8,
              runSpacing: AppTokens.s8,
              children: [
                for (final t in widget.travellers)
                  _Chip(
                    label: t.name,
                    on: _paidById == t.id,
                    onTap: () {
                      setState(() => _paidById = t.id);
                      Haptics.select();
                    },
                  ),
              ],
            ),
          ),

          _splitHeader(c),
          for (final t in widget.travellers)
            _ShareRow(
              traveller: t,
              included: _between.contains(t.id),
              shareMinor: _shares[t.id] ?? 0,
              editable: _customShares != null,
              onToggle: () => _toggle(t.id),
              onChanged: (minor) => setState(() {
                _customShares = {...?_customShares, t.id: minor};
              }),
            ),
          _balanceLine(c),

          const StencilLabel('When'),
          PressScale(
            onTap: _pickDate,
            child: Container(
              padding: const EdgeInsets.symmetric(
                horizontal: AppTokens.gutter,
                vertical: AppTokens.s12,
              ),
              child: Row(
                children: [
                  Icon(Icons.calendar_today_outlined, size: 16, color: c.muted),
                  const SizedBox(width: AppTokens.s12),
                  Text(
                    '${_spentAt.day.toString().padLeft(2, '0')}/'
                    '${_spentAt.month.toString().padLeft(2, '0')}/'
                    '${_spentAt.year}',
                    style: AppTokens.numberStyle.copyWith(color: c.ink),
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: AppTokens.s24),
        ],
      ),
      bottomNavigationBar: SafeArea(
        minimum: const EdgeInsets.all(AppTokens.s16),
        child: PressScale(
          onTap: _canSave && !_saving ? _save : null,
          child: Container(
            height: 48,
            alignment: Alignment.center,
            decoration: BoxDecoration(
              color: _canSave ? c.signal : c.stone,
              border: Border.all(color: _canSave ? c.ink : c.rule),
              borderRadius: BorderRadius.circular(AppTokens.radiusSoft),
            ),
            child: Text(
              _canSave
                  ? (_isEdit ? 'Save expense' : 'Add expense')
                  : 'Not ready yet',
              style: AppTokens.stencilStyle.copyWith(
                fontSize: 11.5,
                color: _canSave ? c.paper : c.muted,
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _splitHeader(AppColors c) => Row(
    children: [
      const Expanded(child: StencilLabel('Split between')),
      Padding(
        padding: const EdgeInsets.only(right: AppTokens.gutter),
        child: GestureDetector(
          onTap: _customShares == null ? _takeOverShares : _backToEven,
          child: Text(
            _customShares == null ? 'SET BY HAND' : 'BACK TO EVEN',
            style: AppTokens.stencilStyle.copyWith(
              fontSize: 9.5,
              color: c.signal,
            ),
          ),
        ),
      ),
    ],
  );

  /// The line that makes the rule visible: what the shares add up to, and how
  /// far off the amount they are.
  Widget _balanceLine(AppColors c) {
    final even = _outBy == 0;
    return Padding(
      padding: const EdgeInsets.fromLTRB(
        AppTokens.gutter,
        AppTokens.s12,
        AppTokens.gutter,
        0,
      ),
      child: Row(
        children: [
          Expanded(
            child: Text(
              _between.isEmpty
                  ? 'Nobody is sharing this yet.'
                  : even
                  ? 'Shares add up exactly.'
                  : _outBy > 0
                  ? '${formatRupees(_outBy)} still to allocate.'
                  : '${formatRupees(-_outBy)} over the amount.',
              style: AppTokens.captionStyle.copyWith(
                // Amber, not red. An unfinished split is not an emergency.
                color: even ? c.signal : c.cautionMark,
              ),
            ),
          ),
          Text(
            formatRupees(_shareTotal),
            style: AppTokens.numberStyle.copyWith(
              color: even ? c.ink : c.cautionMark,
            ),
          ),
        ],
      ),
    );
  }
}

class _ShareRow extends StatefulWidget {
  final Traveller traveller;
  final bool included;
  final int shareMinor;
  final bool editable;
  final VoidCallback onToggle;
  final ValueChanged<int> onChanged;

  const _ShareRow({
    required this.traveller,
    required this.included,
    required this.shareMinor,
    required this.editable,
    required this.onToggle,
    required this.onChanged,
  });

  @override
  State<_ShareRow> createState() => _ShareRowState();
}

class _ShareRowState extends State<_ShareRow> {
  late final _controller = TextEditingController(
    text: rupeesFieldText(widget.shareMinor),
  );

  @override
  void didUpdateWidget(_ShareRow old) {
    super.didUpdateWidget(old);
    // Only push the computed value down while the user is NOT driving the
    // field, or every keystroke fights the even split.
    if (!widget.editable && widget.shareMinor != old.shareMinor) {
      _controller.text = rupeesFieldText(widget.shareMinor);
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);

    return GestureDetector(
      onTap: widget.onToggle,
      behavior: HitTestBehavior.opaque,
      child: Container(
        padding: const EdgeInsets.symmetric(
          horizontal: AppTokens.gutter,
          vertical: AppTokens.s8,
        ),
        decoration: BoxDecoration(
          border: Border(
            bottom: BorderSide(color: c.rule, width: AppTokens.hairline),
          ),
        ),
        child: Row(
          children: [
            Container(
              width: 18,
              height: 18,
              alignment: Alignment.center,
              decoration: BoxDecoration(
                border: Border.all(color: widget.included ? c.ink : c.rule),
                color: widget.included ? c.signal : Colors.transparent,
              ),
              child: widget.included
                  ? Icon(Icons.check, size: 13, color: c.paper)
                  : const SizedBox.shrink(),
            ),
            const SizedBox(width: AppTokens.s12),
            Expanded(
              child: Text(
                widget.traveller.name,
                style: AppTokens.rowTitleStyle.copyWith(
                  color: widget.included ? c.ink : c.muted,
                ),
              ),
            ),
            if (widget.included)
              SizedBox(
                width: 96,
                child: widget.editable
                    ? TextField(
                        controller: _controller,
                        textAlign: TextAlign.right,
                        keyboardType: const TextInputType.numberWithOptions(
                          decimal: true,
                        ),
                        style: AppTokens.numberStyle.copyWith(color: c.ink),
                        decoration: InputDecoration(
                          isDense: true,
                          prefixText: '₹',
                          prefixStyle: AppTokens.numberStyle.copyWith(
                            color: c.muted,
                          ),
                          border: UnderlineInputBorder(
                            borderSide: BorderSide(color: c.rule),
                          ),
                        ),
                        onChanged: (v) => widget.onChanged(parseRupees(v) ?? 0),
                      )
                    : Text(
                        formatRupees(widget.shareMinor),
                        textAlign: TextAlign.right,
                        style: AppTokens.numberStyle.copyWith(color: c.ink),
                      ),
              ),
          ],
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
