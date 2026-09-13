// lib/features/checklist/presentation/checklist_screen.dart
//
// SCREENS.md §7. The pack list, with the pre-departure block at the top of it.
//
// Putting "call the homestay" in the same list as "pack leech socks" is the
// point, not a compromise. Both are things that have to happen before you
// leave, and one of them is the reason this app exists.

import 'package:flutter/material.dart';

import '../../../core/database/app_database.dart';
import '../../../core/theme/app_tokens.dart';
import '../../../core/theme/motion.dart';
import '../../../core/widgets/retro.dart';
import '../data/checklist_dao.dart';

class ChecklistScreen extends StatelessWidget {
  final Stream<ChecklistView> checklist;
  final Future<void> Function(ChecklistItem item, bool done) onToggle;
  final Future<void> Function(ChecklistItem item) onRemove;
  final Future<void> Function(ChecklistItem item, String label, String? qty)
  onEdit;
  final Future<void> Function(String label) onAdd;

  /// Rerun the generator. Offered rather than automatic, so the promise that
  /// edits survive is something the user can test on purpose.
  final Future<void> Function()? onRegenerate;

  const ChecklistScreen({
    super.key,
    required this.checklist,
    required this.onToggle,
    required this.onRemove,
    required this.onEdit,
    required this.onAdd,
    this.onRegenerate,
  });

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);

    return StreamBuilder<ChecklistView>(
      stream: checklist,
      builder: (context, snap) {
        final view = snap.data;

        return Scaffold(
          backgroundColor: c.paper,
          appBar: AppBar(
            title: const Text('Checklist'),
            backgroundColor: c.paper,
            foregroundColor: c.ink,
            elevation: 0,
            actions: [
              if (onRegenerate != null)
                IconButton(
                  onPressed: onRegenerate,
                  icon: const Icon(Icons.refresh),
                  color: c.muted,
                  tooltip: 'Rebuild from the itinerary',
                ),
            ],
            bottom: PreferredSize(
              preferredSize: const Size.fromHeight(3),
              child: _Progress(value: view?.progress ?? 0),
            ),
          ),
          body: view == null
              ? const SizedBox()
              : ListView(
                  padding: const EdgeInsets.only(bottom: 96),
                  children: [
                    if (view.blocking.isNotEmpty) ...[
                      const StencilLabel('Before you leave signal'),
                      for (final item in view.blocking)
                        _Row(
                          item: item,
                          blocking: true,
                          onToggle: (v) => onToggle(item, v),
                          onRemove: () => onRemove(item),
                          onEdit: (l, q) => onEdit(item, l, q),
                        ),
                      Padding(
                        padding: const EdgeInsets.fromLTRB(
                          AppTokens.gutter,
                          AppTokens.s8,
                          AppTokens.gutter,
                          0,
                        ),
                        child: Text(
                          view.isReady
                              ? 'Nothing blocking. The trip reads ready.'
                              : 'The trip does not read ready while any of '
                                    'these are open.',
                          style: AppTokens.captionStyle.copyWith(
                            color: view.isReady ? c.signal : c.cautionMark,
                          ),
                        ),
                      ),
                    ],

                    const StencilLabel('Pack'),
                    if (view.pack.isEmpty)
                      Padding(
                        padding: const EdgeInsets.symmetric(
                          horizontal: AppTokens.gutter,
                        ),
                        child: Text(
                          'Nothing yet. Tag your stops with what happens '
                          'there — trek, caves, rain, homestay — and the list '
                          'builds itself.',
                          style: AppTokens.captionStyle.copyWith(
                            color: c.muted,
                          ),
                        ),
                      ),
                    for (final item in view.pack)
                      _Row(
                        item: item,
                        onToggle: (v) => onToggle(item, v),
                        onRemove: () => onRemove(item),
                        onEdit: (l, q) => onEdit(item, l, q),
                      ),

                    const StencilLabel('Why these'),
                    Padding(
                      padding: const EdgeInsets.symmetric(
                        horizontal: AppTokens.gutter,
                      ),
                      child: Text(
                        'Built from the tags on your stops and how many '
                        'nights you are there. No weather is looked up — the '
                        'app makes no network call.\n\n'
                        'Anything you change or remove stays changed. '
                        'Rebuilding from the itinerary will not undo it.',
                        style: AppTokens.captionStyle.copyWith(color: c.muted),
                      ),
                    ),
                    const SizedBox(height: AppTokens.s24),
                  ],
                ),
          floatingActionButton: FloatingActionButton.extended(
            onPressed: () => _addDialog(context),
            backgroundColor: c.signal,
            foregroundColor: c.paper,
            icon: const Icon(Icons.add, size: 18),
            label: Text(
              'Add',
              style: AppTokens.stencilStyle.copyWith(
                fontSize: 10.5,
                color: c.paper,
              ),
            ),
          ),
        );
      },
    );
  }

  Future<void> _addDialog(BuildContext context) async {
    final c = AppTokens.of(context);
    final controller = TextEditingController();
    final label = await showDialog<String>(
      context: context,
      builder: (dialogContext) => AlertDialog(
        backgroundColor: c.paper,
        title: Text(
          'Add to the list',
          style: AppTokens.titleStyle.copyWith(color: c.ink, fontSize: 18),
        ),
        content: TextField(
          controller: controller,
          autofocus: true,
          style: AppTokens.rowTitleStyle.copyWith(color: c.ink),
          decoration: InputDecoration(
            hintText: 'Spare specs',
            hintStyle: AppTokens.rowTitleStyle.copyWith(color: c.rule),
          ),
          onSubmitted: (v) => Navigator.of(dialogContext).pop(v.trim()),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(dialogContext).pop(),
            child: Text('Cancel', style: TextStyle(color: c.muted)),
          ),
          TextButton(
            onPressed: () =>
                Navigator.of(dialogContext).pop(controller.text.trim()),
            child: Text('Add', style: TextStyle(color: c.signal)),
          ),
        ],
      ),
    );
    controller.dispose();
    if (label != null && label.isNotEmpty) await onAdd(label);
  }
}

/// A 3px hairline under the app bar, filled in signal. DESIGN_VISUAL_v2 §4:
/// the progress rule, not a Material progress bar.
class _Progress extends StatelessWidget {
  final double value;
  const _Progress({required this.value});

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    return SizedBox(
      height: 3,
      child: Row(
        children: [
          Expanded(
            flex: (value.clamp(0.0, 1.0) * 1000).round(),
            child: Container(color: c.signal),
          ),
          Expanded(
            flex: 1000 - (value.clamp(0.0, 1.0) * 1000).round(),
            child: Container(color: c.rule),
          ),
        ],
      ),
    );
  }
}

class _Row extends StatelessWidget {
  final ChecklistItem item;
  final bool blocking;
  final ValueChanged<bool> onToggle;
  final VoidCallback onRemove;
  final void Function(String label, String? quantity) onEdit;

  const _Row({
    required this.item,
    required this.onToggle,
    required this.onRemove,
    required this.onEdit,
    this.blocking = false,
  });

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    final tags = item.sourceTags
        .split(',')
        .where((t) => t.trim().isNotEmpty && t != 'readiness')
        .toList();

    return GestureDetector(
      onTap: () {
        Haptics.select();
        onToggle(!item.isDone);
      },
      onLongPress: () => _editDialog(context),
      behavior: HitTestBehavior.opaque,
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
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // A stencil mark in an 18px box, never a Material checkbox.
            Container(
              width: 18,
              height: 18,
              margin: const EdgeInsets.only(top: 1),
              alignment: Alignment.center,
              decoration: BoxDecoration(
                border: Border.all(color: item.isDone ? c.signal : c.ink),
                color: item.isDone ? c.signal : Colors.transparent,
              ),
              child: item.isDone
                  ? Icon(Icons.check, size: 13, color: c.paper)
                  : const SizedBox.shrink(),
            ),
            const SizedBox(width: AppTokens.s12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    item.label,
                    style: AppTokens.rowTitleStyle.copyWith(
                      color: item.isDone ? c.muted : c.ink,
                      decoration: item.isDone
                          ? TextDecoration.lineThrough
                          : null,
                    ),
                  ),
                  if (blocking && !item.isDone)
                    Padding(
                      padding: const EdgeInsets.only(top: 2),
                      child: Text(
                        'BLOCKING',
                        style: AppTokens.stencilStyle.copyWith(
                          fontSize: 9,
                          color: c.cautionMark,
                        ),
                      ),
                    ),
                  if (tags.isNotEmpty)
                    Padding(
                      padding: const EdgeInsets.only(top: 2),
                      child: Text(
                        tags.join(' · '),
                        style: AppTokens.stencilStyle.copyWith(
                          fontSize: 9,
                          color: c.muted,
                        ),
                      ),
                    ),
                  if (item.isUserEdited && !blocking)
                    Padding(
                      padding: const EdgeInsets.only(top: 2),
                      child: Text(
                        'YOURS',
                        style: AppTokens.stencilStyle.copyWith(
                          fontSize: 9,
                          color: c.muted,
                        ),
                      ),
                    ),
                ],
              ),
            ),
            if (item.quantity != null)
              Padding(
                padding: const EdgeInsets.only(left: AppTokens.s8),
                child: Text(
                  item.quantity!,
                  style: AppTokens.numberStyle.copyWith(
                    color: item.isDone ? c.muted : c.ink,
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }

  Future<void> _editDialog(BuildContext context) async {
    final c = AppTokens.of(context);
    final label = TextEditingController(text: item.label);
    final qty = TextEditingController(text: item.quantity ?? '');

    final result = await showDialog<String>(
      context: context,
      builder: (dialogContext) => AlertDialog(
        backgroundColor: c.paper,
        title: Text(
          'Edit',
          style: AppTokens.titleStyle.copyWith(color: c.ink, fontSize: 18),
        ),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            TextField(
              controller: label,
              autofocus: true,
              style: AppTokens.rowTitleStyle.copyWith(color: c.ink),
            ),
            const SizedBox(height: AppTokens.s8),
            TextField(
              controller: qty,
              style: AppTokens.numberStyle.copyWith(color: c.ink),
              decoration: InputDecoration(
                labelText: 'How many',
                labelStyle: AppTokens.captionStyle.copyWith(color: c.muted),
              ),
            ),
            const SizedBox(height: AppTokens.s12),
            Text(
              'Your version stays, whatever the generator thinks later.',
              style: AppTokens.captionStyle.copyWith(color: c.muted),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(dialogContext).pop('remove'),
            child: Text('Remove', style: TextStyle(color: c.emergency)),
          ),
          TextButton(
            onPressed: () => Navigator.of(dialogContext).pop(),
            child: Text('Cancel', style: TextStyle(color: c.muted)),
          ),
          TextButton(
            onPressed: () => Navigator.of(dialogContext).pop('save'),
            child: Text('Save', style: TextStyle(color: c.signal)),
          ),
        ],
      ),
    );

    if (result == 'remove') {
      Haptics.grave();
      onRemove();
    } else if (result == 'save' && label.text.trim().isNotEmpty) {
      onEdit(
        label.text.trim(),
        qty.text.trim().isEmpty ? null : qty.text.trim(),
      );
    }
    label.dispose();
    qty.dispose();
  }
}
