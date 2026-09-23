// lib/features/import/presentation/import_preview_screen.dart
//
// Step 3 of import: see exactly what will land, then decide.
//
// SEVERITY READS AS A STRIPE AS WELL AS A COLOUR (SCREENS.md §6), so state
// survives a colourblind reader and a phone in bright sun. Red is not used
// here at all — red belongs to the emergency tab, and a duplicate row is not
// an emergency.

import 'package:flutter/material.dart';

import '../../../core/theme/app_tokens.dart';
import '../../../core/theme/motion.dart';
import '../../../core/widgets/retro.dart';
import '../data/import_validation.dart';

class ImportPreviewScreen extends StatefulWidget {
  final ImportPreview preview;
  final String fileName;
  final String? sheetName;

  /// Commits the selected rows. Returns how many landed.
  final Future<int> Function(ImportPreview preview) onCommit;

  const ImportPreviewScreen({
    super.key,
    required this.preview,
    required this.fileName,
    required this.onCommit,
    this.sheetName,
  });

  @override
  State<ImportPreviewScreen> createState() => _ImportPreviewScreenState();
}

class _ImportPreviewScreenState extends State<ImportPreviewScreen> {
  late final List<ValidatedRow> _rows = List.of(widget.preview.rows);
  bool _busy = false;

  ImportPreview get _current => ImportPreview(_rows);

  void _toggle(int index) {
    final row = _rows[index];
    if (!row.canSelect) {
      Haptics.reject();
      return;
    }
    setState(() => _rows[index] = row.copyWith(selected: !row.selected));
    Haptics.select();
  }

  Future<void> _commit() async {
    if (_busy) return;
    setState(() => _busy = true);
    try {
      await widget.onCommit(_current);
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    final p = _current;
    final selected = p.selected;
    final adding = p.toImport.length;
    final placing = p.toPlace.length;

    return Scaffold(
      backgroundColor: c.paper,
      appBar: AppBar(
        title: const Text('Before importing'),
        backgroundColor: c.paper,
        foregroundColor: c.ink,
        elevation: 0,
      ),
      body: ListView(
        padding: const EdgeInsets.only(bottom: 96),
        children: [
          _Counts(preview: p),
          const StencilLabel('Rows'),
          for (var i = 0; i < _rows.length; i++)
            _RowTile(row: _rows[i], onTap: () => _toggle(i)),
          const StencilLabel('What import does'),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: AppTokens.gutter),
            child: Text(
              'Every row lands unconfirmed, with an amber dot, however it was '
              'spelled in the sheet. A number in a spreadsheet is still a '
              'number nobody has dialled. Call it, then mark it confirmed.\n\n'
              'The whole batch can be undone in one action from Import '
              'history.'
              '${placing == 0 ? '' : '\n\nA location added to an entry '
                  'already in your diary only fills in where there was none, '
                  'and stays if the batch is undone.'}',
              style: AppTokens.captionStyle.copyWith(color: c.muted),
            ),
          ),
          const SizedBox(height: AppTokens.s24),
        ],
      ),
      bottomNavigationBar: SafeArea(
        minimum: const EdgeInsets.all(AppTokens.s16),
        child: PressScale(
          onTap: selected == 0 || _busy ? null : _commit,
          child: Container(
            height: 48,
            alignment: Alignment.center,
            decoration: BoxDecoration(
              color: selected == 0 ? c.stone : c.signal,
              border: Border.all(color: selected == 0 ? c.rule : c.ink),
              borderRadius: BorderRadius.circular(AppTokens.radiusSoft),
            ),
            child: Text(
              _buttonLabel(adding, placing),
              style: AppTokens.stencilStyle.copyWith(
                fontSize: 11.5,
                color: selected == 0 ? c.muted : c.paper,
              ),
            ),
          ),
        ),
      ),
    );
  }
}

String _buttonLabel(int adding, int placing) {
  final add = 'Import $adding ${adding == 1 ? 'contact' : 'contacts'}';
  final place = '$placing ${placing == 1 ? 'location' : 'locations'}';
  if (adding == 0 && placing == 0) return 'Nothing selected';
  if (placing == 0) return add;
  if (adding == 0) return 'Add $place';
  return '$add + $place';
}

class _Counts extends StatelessWidget {
  final ImportPreview preview;
  const _Counts({required this.preview});

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    return Padding(
      padding: const EdgeInsets.fromLTRB(
        AppTokens.gutter,
        AppTokens.s8,
        AppTokens.gutter,
        0,
      ),
      child: Row(
        children: [
          _Count(label: 'Read', value: preview.total, color: c.ink),
          _Count(label: 'Ready', value: preview.ready, color: c.signal),
          _Count(
            label: 'Check',
            value: preview.warnings,
            color: c.cautionMark,
          ),
          _Count(label: 'Skipped', value: preview.skipped, color: c.muted),
        ],
      ),
    );
  }
}

class _Count extends StatelessWidget {
  final String label;
  final int value;
  final Color color;
  const _Count({
    required this.label,
    required this.value,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    return Expanded(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('$value', style: AppTokens.numberStyle.copyWith(
            color: color,
            fontSize: 20,
          )),
          Text(
            label.toUpperCase(),
            style: AppTokens.stencilStyle.copyWith(
              color: c.muted,
              fontSize: 9,
            ),
          ),
        ],
      ),
    );
  }
}

class _RowTile extends StatelessWidget {
  final ValidatedRow row;
  final VoidCallback onTap;

  const _RowTile({required this.row, required this.onTap});

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    final stripe = switch (row.state) {
      RowState.ready => c.signal,
      RowState.warning => c.cautionMark,
      RowState.skip => c.muted,
    };
    final dimmed = !row.selected;

    return GestureDetector(
      onTap: onTap,
      behavior: HitTestBehavior.opaque,
      child: Container(
        decoration: BoxDecoration(
          border: Border(
            bottom: BorderSide(color: c.rule, width: AppTokens.hairline),
          ),
        ),
        child: IntrinsicHeight(
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // The stripe. State has to read without depending on hue.
              Container(width: 4, color: dimmed ? c.rule : stripe),
              Expanded(
                child: Padding(
                  padding: const EdgeInsets.fromLTRB(
                    AppTokens.s12,
                    AppTokens.s12,
                    AppTokens.gutter,
                    AppTokens.s12,
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          SizedBox(
                            width: 26,
                            child: Text(
                              '${row.sourceRow}',
                              style: AppTokens.numberStyle.copyWith(
                                color: c.muted,
                                fontSize: 11,
                              ),
                            ),
                          ),
                          Expanded(
                            child: Text(
                              row.name.isEmpty ? '(no name)' : row.name,
                              maxLines: 1,
                              overflow: TextOverflow.ellipsis,
                              style: AppTokens.rowTitleStyle.copyWith(
                                color: dimmed ? c.muted : c.ink,
                                decoration: row.state == RowState.skip
                                    ? TextDecoration.lineThrough
                                    : null,
                              ),
                            ),
                          ),
                          _Tick(on: row.selected, enabled: row.canSelect),
                        ],
                      ),
                      if (row.phoneRaw.isNotEmpty)
                        Padding(
                          padding: const EdgeInsets.only(left: 26, top: 2),
                          child: Text(
                            row.phoneRaw,
                            style: AppTokens.numberStyle.copyWith(
                              color: dimmed ? c.muted : c.ink,
                            ),
                          ),
                        ),
                      if (row.stopName != null)
                        Padding(
                          padding: const EdgeInsets.only(left: 26, top: 2),
                          child: Text(
                            row.stopName!,
                            style: AppTokens.captionStyle.copyWith(
                              color: c.muted,
                            ),
                          ),
                        ),
                      for (final m in row.messages)
                        Padding(
                          padding: const EdgeInsets.only(left: 26, top: 2),
                          child: Text(
                            m,
                            style: AppTokens.captionStyle.copyWith(
                              color: row.state == RowState.skip
                                  ? c.muted
                                  : c.cautionMark,
                            ),
                          ),
                        ),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

/// A stencil tick in a box, matching the checklist mark rather than a
/// Material checkbox.
class _Tick extends StatelessWidget {
  final bool on;
  final bool enabled;
  const _Tick({required this.on, required this.enabled});

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    return Container(
      width: 18,
      height: 18,
      alignment: Alignment.center,
      decoration: BoxDecoration(
        border: Border.all(color: enabled ? c.ink : c.rule),
        color: on ? c.signal : Colors.transparent,
      ),
      child: on
          ? Icon(Icons.check, size: 13, color: c.paper)
          : const SizedBox.shrink(),
    );
  }
}
