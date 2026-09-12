// lib/features/import/presentation/column_mapping_screen.dart
//
// Step 2 of import: confirm what each column means.
//
// Auto-matching does the work; this screen exists so the user can see what
// the app guessed and correct it. Showing the guess is the point — an import
// that silently maps the wrong column produces sixty wrong contacts and no
// clue where they came from.

import 'package:flutter/material.dart';

import '../../../core/theme/app_tokens.dart';
import '../../../core/theme/motion.dart';
import '../../../core/widgets/retro.dart';
import '../data/column_mapping.dart';
import '../data/sheet_parser.dart';

class ColumnMappingScreen extends StatefulWidget {
  final SheetTable table;
  final ColumnMapping initial;

  /// Called with the confirmed mapping. The caller moves to the preview.
  final void Function(ColumnMapping mapping) onContinue;

  const ColumnMappingScreen({
    super.key,
    required this.table,
    required this.initial,
    required this.onContinue,
  });

  @override
  State<ColumnMappingScreen> createState() => _ColumnMappingScreenState();
}

class _ColumnMappingScreenState extends State<ColumnMappingScreen> {
  late final ColumnMapping _mapping = Map.of(widget.initial);

  void _set(ImportField field, int? column) {
    setState(() {
      if (column == null) {
        _mapping.remove(field);
      } else {
        // A column belongs to one field. Reassigning it takes it away from
        // whoever held it, rather than quietly duplicating the data.
        _mapping.removeWhere((_, c) => c == column);
        _mapping[field] = column;
      }
    });
    Haptics.select();
  }

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    final missing = missingRequired(_mapping);
    final ready = missing.isEmpty;

    return Scaffold(
      backgroundColor: c.paper,
      appBar: AppBar(
        title: const Text('Match the columns'),
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
            child: Text(
              '${widget.table.name} · ${widget.table.rowCount} rows. '
              'We have guessed what each column holds. Change anything that '
              'looks wrong.',
              style: AppTokens.captionStyle.copyWith(color: c.muted),
            ),
          ),
          const StencilLabel('Columns'),
          for (final field in ImportField.values)
            _FieldRow(
              field: field,
              headers: widget.table.headers,
              selected: _mapping[field],
              sample: _sampleFor(_mapping[field]),
              onChanged: (col) => _set(field, col),
            ),
          if (!ready)
            Padding(
              padding: const EdgeInsets.fromLTRB(
                AppTokens.gutter,
                AppTokens.s16,
                AppTokens.gutter,
                0,
              ),
              child: Text(
                'Still need: ${missing.map((f) => f.label).join(' and ')}. '
                'A contact is a name and a number; the rest is optional.',
                style: AppTokens.captionStyle.copyWith(color: c.cautionMark),
              ),
            ),
        ],
      ),
      bottomNavigationBar: SafeArea(
        minimum: const EdgeInsets.all(AppTokens.s16),
        child: PressScale(
          onTap: ready ? () => widget.onContinue(Map.of(_mapping)) : null,
          child: Container(
            height: 48,
            alignment: Alignment.center,
            decoration: BoxDecoration(
              color: ready ? c.signal : c.stone,
              border: Border.all(color: ready ? c.ink : c.rule),
              borderRadius: BorderRadius.circular(AppTokens.radiusSoft),
            ),
            child: Text(
              'See what will be imported',
              style: AppTokens.stencilStyle.copyWith(
                fontSize: 11.5,
                color: ready ? c.paper : c.muted,
              ),
            ),
          ),
        ),
      ),
    );
  }

  /// The first non-empty value in a column, so the user can tell at a glance
  /// whether the guess is right without opening the file again.
  String? _sampleFor(int? column) {
    if (column == null) return null;
    for (final row in widget.table.rows) {
      if (column < row.length && row[column].isNotEmpty) return row[column];
    }
    return null;
  }
}

class _FieldRow extends StatelessWidget {
  final ImportField field;
  final List<String> headers;
  final int? selected;
  final String? sample;
  final ValueChanged<int?> onChanged;

  const _FieldRow({
    required this.field,
    required this.headers,
    required this.selected,
    required this.sample,
    required this.onChanged,
  });

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    return Container(
      padding: const EdgeInsets.symmetric(
        horizontal: AppTokens.gutter,
        vertical: AppTokens.s12,
      ),
      decoration: BoxDecoration(
        border: Border(bottom: BorderSide(color: c.rule, width: AppTokens.hairline)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 92,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  field.label,
                  style: AppTokens.rowTitleStyle.copyWith(color: c.ink),
                ),
                if (field.isRequired)
                  Text(
                    'required',
                    style: AppTokens.captionStyle.copyWith(
                      color: c.muted,
                      fontSize: 10.5,
                    ),
                  ),
              ],
            ),
          ),
          const SizedBox(width: AppTokens.s12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                DropdownButtonFormField<int?>(
                  initialValue: selected,
                  isDense: true,
                  isExpanded: true,
                  decoration: InputDecoration(
                    isDense: true,
                    contentPadding: const EdgeInsets.symmetric(
                      horizontal: AppTokens.s8,
                      vertical: AppTokens.s8,
                    ),
                    border: OutlineInputBorder(
                      borderSide: BorderSide(color: c.rule),
                      borderRadius: BorderRadius.circular(AppTokens.radiusSoft),
                    ),
                    enabledBorder: OutlineInputBorder(
                      borderSide: BorderSide(color: c.rule),
                      borderRadius: BorderRadius.circular(AppTokens.radiusSoft),
                    ),
                  ),
                  style: AppTokens.captionStyle.copyWith(color: c.ink),
                  items: [
                    DropdownMenuItem<int?>(
                      value: null,
                      child: Text(
                        field.isRequired ? 'Not matched' : 'Not in this sheet',
                        style: AppTokens.captionStyle.copyWith(color: c.muted),
                      ),
                    ),
                    for (var i = 0; i < headers.length; i++)
                      DropdownMenuItem<int?>(
                        value: i,
                        child: Text(
                          headers[i].isEmpty ? 'Column ${i + 1}' : headers[i],
                          overflow: TextOverflow.ellipsis,
                        ),
                      ),
                  ],
                  onChanged: onChanged,
                ),
                if (sample != null) ...[
                  const SizedBox(height: AppTokens.s4),
                  Text(
                    'e.g. $sample',
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: AppTokens.captionStyle.copyWith(
                      color: c.muted,
                      fontSize: 11,
                    ),
                  ),
                ],
              ],
            ),
          ),
        ],
      ),
    );
  }
}
