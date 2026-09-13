// lib/features/contacts/presentation/multi_add_screen.dart
//
// Several contacts at once — issue #10.
//
// The entry form owns one contact properly: its stop, its note, WhatsApp, the
// confirmation. This owns exactly two fields and does them eight times,
// because sitting at home with a booking sheet is a different job from
// recording one number carefully.

import 'package:flutter/material.dart';

import '../../../core/theme/app_tokens.dart';
import '../../../core/theme/motion.dart';
import '../data/contacts_dao.dart';
import '../data/multi_add.dart';

class MultiAddScreen extends StatefulWidget {
  /// Normalises a row and looks its number up in the diary. Injected so the
  /// screen can be rendered and tested without a database.
  final Future<MultiAddRow> Function(MultiAddRow) check;

  /// Writes the ready rows. Returns what actually landed.
  final Future<MultiAddResult> Function(MultiAddSheet) onSave;

  const MultiAddScreen({super.key, required this.check, required this.onSave});

  @override
  State<MultiAddScreen> createState() => _MultiAddScreenState();
}

class _MultiAddScreenState extends State<MultiAddScreen> {
  MultiAddSheet _sheet = MultiAddSheet.empty();
  final _controllers = <int, (TextEditingController, TextEditingController)>{};
  bool _saving = false;

  @override
  void dispose() {
    for (final pair in _controllers.values) {
      pair.$1.dispose();
      pair.$2.dispose();
    }
    super.dispose();
  }

  (TextEditingController, TextEditingController) _controllersFor(int i) =>
      _controllers.putIfAbsent(
        i,
        () => (
          TextEditingController(text: _sheet.rows[i].name),
          TextEditingController(text: _sheet.rows[i].phoneRaw),
        ),
      );

  void _replace(int i, MultiAddRow row) {
    final rows = List.of(_sheet.rows);
    rows[i] = row;
    setState(() => _sheet = MultiAddSheet(rows).settled());
  }

  Future<void> _edit(int i, {String? name, String? phone}) async {
    final current = _sheet.rows[i];
    var next = current.copyWith(name: name, phoneRaw: phone);
    _replace(i, next);

    // Only the number needs checking, and only after the row exists.
    if (phone != null) {
      next = await widget.check(next);
      if (!mounted) return;
      // The row may have moved or been typed into again while we waited.
      final at = _sheet.rows.length > i ? _sheet.rows[i] : null;
      if (at == null || at.phoneRaw != next.phoneRaw) return;
      _replace(i, next);
    }
  }

  void _remove(int i) {
    Haptics.light();
    final rows = List.of(_sheet.rows)..removeAt(i);
    // The controllers are keyed by index, so removing a row shifts every
    // later row's text onto the wrong controller unless they are rebuilt.
    for (final pair in _controllers.values) {
      pair.$1.dispose();
      pair.$2.dispose();
    }
    _controllers.clear();
    setState(() => _sheet = MultiAddSheet(rows).settled());
  }

  Future<void> _save() async {
    if (!_sheet.canSave || _saving) return;
    setState(() => _saving = true);
    final result = await widget.onSave(_sheet);
    if (!mounted) return;
    Navigator.of(context).pop(result);
  }

  /// Eight numbers typed and lost to a back gesture is the harm the swipe
  /// actions were designed against at #45.
  Future<bool> _confirmLeave() async {
    if (!_sheet.hasAnything || _saving) return true;
    final c = AppTokens.of(context);
    final ok = await showDialog<bool>(
      context: context,
      builder: (dialogContext) => AlertDialog(
        backgroundColor: c.paper,
        title: Text(
          'Throw these away?',
          style: AppTokens.titleStyle.copyWith(color: c.ink),
        ),
        content: Text(
          _sheet.readyCount == 0
              ? 'Nothing here is finished yet, and leaving loses what you '
                    'have typed.'
              : '${_sheet.readyCount} '
                    '${_sheet.readyCount == 1 ? "entry is" : "entries are"} '
                    'ready to save. Leaving loses them.',
          style: AppTokens.captionStyle.copyWith(color: c.muted),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(dialogContext).pop(false),
            child: Text('Keep typing', style: TextStyle(color: c.muted)),
          ),
          TextButton(
            onPressed: () => Navigator.of(dialogContext).pop(true),
            child: Text('Throw away', style: TextStyle(color: c.emergency)),
          ),
        ],
      ),
    );
    return ok ?? false;
  }

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);

    return PopScope(
      canPop: !_sheet.hasAnything,
      onPopInvokedWithResult: (didPop, _) async {
        if (didPop) return;
        if (await _confirmLeave() && mounted) {
          if (context.mounted) Navigator.of(context).pop();
        }
      },
      child: Scaffold(
        backgroundColor: c.paper,
        appBar: AppBar(
          title: const Text('Add several'),
          backgroundColor: c.paper,
          foregroundColor: c.ink,
          elevation: 0,
        ),
        body: SafeArea(
          child: Column(
            children: [
              Expanded(
                child: ListView.builder(
                  padding: const EdgeInsets.only(bottom: AppTokens.s16),
                  itemCount: _sheet.rows.length + 1,
                  itemBuilder: (context, i) {
                    if (i == _sheet.rows.length) return const _Preamble();
                    final pair = _controllersFor(i);
                    return _Row(
                      key: ValueKey('row-$i'),
                      index: i,
                      row: _sheet.rows[i],
                      nameController: pair.$1,
                      phoneController: pair.$2,
                      onName: (v) => _edit(i, name: v),
                      onPhone: (v) => _edit(i, phone: v),
                      onCategory: (category) => _replace(
                        i,
                        _sheet.rows[i].copyWith(category: category),
                      ),
                      onRemove: _sheet.rows[i].isBlank ? null : () => _remove(i),
                    );
                  },
                ),
              ),
              _Footer(sheet: _sheet, saving: _saving, onSave: _save),
            ],
          ),
        ),
      ),
    );
  }
}

/// Said once, at the bottom of the rows, where somebody who has just typed
/// eight numbers will read it before reaching the button.
class _Preamble extends StatelessWidget {
  const _Preamble();

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    return Padding(
      padding: const EdgeInsets.fromLTRB(
        AppTokens.gutter,
        AppTokens.s16,
        AppTokens.gutter,
        0,
      ),
      child: Text(
        // TYPING EIGHT NUMBERS QUICKLY FEELS LIKE WORK COMPLETED. It is not:
        // none of these has been dialled.
        'Everything here saves unconfirmed, with the amber dot. Call each one '
        'before you leave signal and mark it confirmed then — that is the '
        'only thing that clears it.',
        style: AppTokens.captionStyle.copyWith(color: c.muted),
      ),
    );
  }
}

/// A line on a ruled sheet, not a field in a form.
///
/// The app's shared `inputDecorationTheme` fills every field with `stone`,
/// which is right for the three or four fields of the entry form and wrong
/// for eight stacked pairs here — they merge into one grey block and lose the
/// notebook the diary is built on. Transparent with a hairline under each
/// line instead.
InputDecoration _ruled(AppColors c, String hint, TextStyle style) =>
    InputDecoration(
      isDense: true,
      filled: false,
      hintText: hint,
      hintStyle: style.copyWith(color: c.rule),
      contentPadding: const EdgeInsets.symmetric(vertical: 6),
      enabledBorder: UnderlineInputBorder(
        borderSide: BorderSide(color: c.rule, width: AppTokens.hairline),
      ),
      focusedBorder: UnderlineInputBorder(
        borderSide: BorderSide(color: c.signal),
      ),
    );

class _Row extends StatelessWidget {
  final int index;
  final MultiAddRow row;
  final TextEditingController nameController;
  final TextEditingController phoneController;
  final ValueChanged<String> onName;
  final ValueChanged<String> onPhone;
  final ValueChanged<String> onCategory;
  final VoidCallback? onRemove;

  const _Row({
    super.key,
    required this.index,
    required this.row,
    required this.nameController,
    required this.phoneController,
    required this.onName,
    required this.onPhone,
    required this.onCategory,
    this.onRemove,
  });

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);

    return Container(
      padding: const EdgeInsets.fromLTRB(
        AppTokens.gutter,
        AppTokens.s12,
        AppTokens.s8,
        AppTokens.s12,
      ),
      decoration: BoxDecoration(
        border: Border(
          bottom: BorderSide(color: c.rule, width: AppTokens.hairline),
        ),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 26,
            child: Padding(
              padding: const EdgeInsets.only(top: 10),
              child: Text(
                '${index + 1}'.padLeft(2, '0'),
                style: AppTokens.numberStyle.copyWith(
                  color: c.muted,
                  fontSize: 12,
                ),
              ),
            ),
          ),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                TextField(
                  controller: nameController,
                  onChanged: onName,
                  textCapitalization: TextCapitalization.words,
                  style: AppTokens.rowTitleStyle.copyWith(color: c.ink),
                  decoration: _ruled(c, 'Name', AppTokens.rowTitleStyle),
                ),
                TextField(
                  controller: phoneController,
                  onChanged: onPhone,
                  keyboardType: TextInputType.phone,
                  style: AppTokens.numberStyle.copyWith(
                    fontSize: 15,
                    color: c.ink,
                  ),
                  decoration: _ruled(
                    c,
                    'Number',
                    AppTokens.numberStyle.copyWith(fontSize: 15),
                  ),
                ),
                const SizedBox(height: AppTokens.s4),
                Row(
                  children: [
                    _CategoryChip(value: row.category, onPick: onCategory),
                    const SizedBox(width: AppTokens.s8),
                    Expanded(child: _RowNote(row: row)),
                  ],
                ),
              ],
            ),
          ),
          SizedBox(
            width: 34,
            child: onRemove == null
                ? const SizedBox()
                : IconButton(
                    onPressed: onRemove,
                    icon: const Icon(Icons.close, size: 16),
                    color: c.muted,
                    tooltip: 'Remove this row',
                  ),
          ),
        ],
      ),
    );
  }
}

/// Whatever this row has to say about itself: unfinished, an odd-looking
/// number, or a number already in the diary. Never more than one thing —
/// three stacked notes on a row this small is noise, and the most serious
/// one is the one worth reading.
class _RowNote extends StatelessWidget {
  final MultiAddRow row;
  const _RowNote({required this.row});

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);

    final (text, tone) = switch (row) {
      _ when row.isBlank => (null, c.muted),
      _ when row.duplicateInSheet => ('Already on this sheet', c.cautionMark),
      _ when row.duplicateOf != null => (
        'Already in the diary as ${row.duplicateOf}',
        c.cautionMark,
      ),
      _ when row.isPartial => (row.missing, c.muted),
      _ when row.warning != null => (row.warning, c.cautionMark),
      _ => (null, c.muted),
    };

    if (text == null) return const SizedBox();
    return Text(
      text,
      maxLines: 2,
      overflow: TextOverflow.ellipsis,
      style: AppTokens.captionStyle.copyWith(fontSize: 11.5, color: tone),
    );
  }
}

class _CategoryChip extends StatelessWidget {
  final String value;
  final ValueChanged<String> onPick;
  const _CategoryChip({required this.value, required this.onPick});

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);

    return PopupMenuButton<String>(
      onSelected: (picked) {
        Haptics.select();
        onPick(picked);
      },
      color: c.paper,
      tooltip: 'Category',
      itemBuilder: (context) => [
        for (final category in ContactCategory.pickerOrder)
          PopupMenuItem(
            value: category,
            child: Text(
              ContactCategory.labels[category] ?? category,
              style: AppTokens.rowTitleStyle.copyWith(
                color: category == value ? c.signal : c.ink,
              ),
            ),
          ),
      ],
      child: Container(
        padding: const EdgeInsets.symmetric(
          horizontal: AppTokens.s8,
          vertical: 4,
        ),
        decoration: BoxDecoration(
          color: c.stone,
          border: Border.all(color: c.rule),
          borderRadius: BorderRadius.circular(AppTokens.radiusSoft),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(
              (ContactCategory.shortLabels[value] ?? value).toUpperCase(),
              style: AppTokens.stencilStyle.copyWith(
                fontSize: 9,
                color: c.muted,
              ),
            ),
            const SizedBox(width: 2),
            Icon(Icons.arrow_drop_down, size: 14, color: c.muted),
          ],
        ),
      ),
    );
  }
}

class _Footer extends StatelessWidget {
  final MultiAddSheet sheet;
  final bool saving;
  final VoidCallback onSave;

  const _Footer({
    required this.sheet,
    required this.saving,
    required this.onSave,
  });

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    final n = sheet.readyCount;
    final skipped = sheet.partialCount;

    return Container(
      padding: const EdgeInsets.all(AppTokens.gutter),
      decoration: BoxDecoration(
        color: c.paper,
        border: Border(top: BorderSide(color: c.rule, width: AppTokens.hairline)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          if (skipped > 0)
            Padding(
              padding: const EdgeInsets.only(bottom: AppTokens.s8),
              child: Text(
                // Skipped, not rejected. Half a row is not an error.
                '$skipped unfinished ${skipped == 1 ? "row" : "rows"} will be '
                'left behind.',
                style: AppTokens.captionStyle.copyWith(color: c.muted),
              ),
            ),
          PressScale(
            onTap: n == 0 || saving ? null : onSave,
            feedback: Haptics.confirm,
            child: Container(
              padding: const EdgeInsets.symmetric(vertical: AppTokens.s12),
              alignment: Alignment.center,
              decoration: BoxDecoration(
                color: n == 0 || saving ? c.stone : c.signal,
                border: Border.all(color: n == 0 || saving ? c.rule : c.ink),
                borderRadius: BorderRadius.circular(AppTokens.radiusSoft),
              ),
              child: Text(
                n == 0
                    ? 'Nothing to save yet'
                    : 'Save $n ${n == 1 ? "entry" : "entries"}',
                style: AppTokens.stencilStyle.copyWith(
                  fontSize: 11.5,
                  color: n == 0 || saving ? c.muted : c.paper,
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
