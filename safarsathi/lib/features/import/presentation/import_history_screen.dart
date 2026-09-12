// lib/features/import/presentation/import_history_screen.dart
//
// Every batch this trip has imported, and the undo.
//
// A batch is an object you consider one at a time, so it gets a ticket stub —
// which is also literally what it is: a record of a transaction that happened.

import 'package:flutter/material.dart';

import '../../../core/theme/app_tokens.dart';
import '../../../core/theme/motion.dart';
import '../../../core/widgets/retro.dart';
import '../data/import_commit.dart';

class ImportHistoryScreen extends StatelessWidget {
  final Stream<List<ImportBatchSummary>> batches;
  final Future<void> Function(int batchId) onRollback;

  const ImportHistoryScreen({
    super.key,
    required this.batches,
    required this.onRollback,
  });

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    return Scaffold(
      backgroundColor: c.paper,
      appBar: AppBar(
        title: const Text('Import history'),
        backgroundColor: c.paper,
        foregroundColor: c.ink,
        elevation: 0,
      ),
      body: StreamBuilder<List<ImportBatchSummary>>(
        stream: batches,
        builder: (context, snap) {
          final list = snap.data;
          if (list == null) return const SizedBox();
          if (list.isEmpty) {
            return Center(
              child: Padding(
                padding: const EdgeInsets.all(AppTokens.s32),
                child: Text(
                  'Nothing imported yet. When you bring a sheet in, it will '
                  'be listed here so you can undo the whole thing in one '
                  'action.',
                  textAlign: TextAlign.center,
                  style: AppTokens.captionStyle.copyWith(color: c.muted),
                ),
              ),
            );
          }
          return ListView.separated(
            padding: const EdgeInsets.all(AppTokens.gutter),
            itemCount: list.length,
            separatorBuilder: (_, _) => const SizedBox(height: AppTokens.s12),
            itemBuilder: (context, i) =>
                _BatchCard(batch: list[i], onRollback: onRollback),
          );
        },
      ),
    );
  }
}

class _BatchCard extends StatelessWidget {
  final ImportBatchSummary batch;
  final Future<void> Function(int batchId) onRollback;

  const _BatchCard({required this.batch, required this.onRollback});

  Future<void> _confirm(BuildContext context) async {
    final c = AppTokens.of(context);
    // ROLLBACK DELETES CONTACTS THE USER MAY HAVE SINCE CONFIRMED, so it asks
    // first and the question names the count. A silent undo of nine numbers
    // is indistinguishable from a bug.
    final ok = await showDialog<bool>(
      context: context,
      builder: (dialogContext) => AlertDialog(
        backgroundColor: c.paper,
        title: Text(
          'Undo this import?',
          style: AppTokens.titleStyle.copyWith(color: c.ink, fontSize: 18),
        ),
        content: Text(
          batch.stillPresent == 0
              ? 'None of this batch is left in the diary. This just clears '
                    'the record of it.'
              : 'This removes ${batch.stillPresent} '
                    '${batch.stillPresent == 1 ? 'contact' : 'contacts'} from '
                    'your diary, including any you have since confirmed. It '
                    'cannot be undone.',
          style: AppTokens.captionStyle.copyWith(color: c.muted),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(dialogContext).pop(false),
            child: Text('Keep', style: TextStyle(color: c.muted)),
          ),
          TextButton(
            onPressed: () => Navigator.of(dialogContext).pop(true),
            child: Text('Undo import', style: TextStyle(color: c.emergency)),
          ),
        ],
      ),
    );
    if (ok == true) {
      Haptics.grave();
      await onRollback(batch.id);
    }
  }

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    final d = batch.importedAt;
    final when =
        '${d.day.toString().padLeft(2, '0')}/'
        '${d.month.toString().padLeft(2, '0')}/${d.year}';

    return TicketCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Text(
            batch.fileName,
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
            style: AppTokens.rowTitleStyle.copyWith(color: c.ink),
          ),
          if (batch.sheetName != null)
            Text(
              'Sheet: ${batch.sheetName}',
              style: AppTokens.captionStyle.copyWith(color: c.muted),
            ),
          const SizedBox(height: AppTokens.s8),
          Row(
            children: [
              Text(
                '${batch.rowsImported}',
                style: AppTokens.numberStyle.copyWith(color: c.ink),
              ),
              const SizedBox(width: AppTokens.s4),
              Text(
                'imported',
                style: AppTokens.captionStyle.copyWith(color: c.muted),
              ),
              if (batch.rowsSkipped > 0) ...[
                const SizedBox(width: AppTokens.s12),
                Text(
                  '${batch.rowsSkipped}',
                  style: AppTokens.numberStyle.copyWith(color: c.muted),
                ),
                const SizedBox(width: AppTokens.s4),
                Text(
                  'skipped',
                  style: AppTokens.captionStyle.copyWith(color: c.muted),
                ),
              ],
              const Spacer(),
              Text(
                when,
                style: AppTokens.numberStyle.copyWith(
                  color: c.muted,
                  fontSize: 11.5,
                ),
              ),
            ],
          ),
          if (batch.stillPresent != batch.rowsImported)
            Padding(
              padding: const EdgeInsets.only(top: AppTokens.s4),
              child: Text(
                '${batch.stillPresent} still in the diary.',
                style: AppTokens.captionStyle.copyWith(color: c.muted),
              ),
            ),
          const SizedBox(height: AppTokens.s12),
          Align(
            alignment: Alignment.centerLeft,
            child: PressScale(
              onTap: () => _confirm(context),
              child: Container(
                padding: const EdgeInsets.symmetric(
                  horizontal: AppTokens.s12,
                  vertical: AppTokens.s8,
                ),
                decoration: BoxDecoration(
                  border: Border.all(color: c.rule),
                  borderRadius: BorderRadius.circular(AppTokens.radiusSoft),
                ),
                child: Text(
                  'Undo import',
                  style: AppTokens.stencilStyle.copyWith(
                    fontSize: 10,
                    color: c.ink,
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
