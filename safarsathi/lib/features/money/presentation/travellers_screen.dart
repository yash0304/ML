// lib/features/money/presentation/travellers_screen.dart
//
// Who is on the trip — issue #31.
//
// Named people, nothing more. No accounts, no invitations, no sync, no server.
// Splitwise's whole value at a dhaba with no signal is knowing the number; the
// account system is what stops you getting it.

import 'package:flutter/material.dart';

import '../../../core/database/app_database.dart';
import '../../../core/theme/app_tokens.dart';
import '../../../core/theme/motion.dart';
import '../../../core/widgets/retro.dart';
import '../data/expense_editor.dart';

class TravellersScreen extends StatelessWidget {
  final Stream<List<Traveller>> travellers;
  final Future<void> Function(String name) onAdd;
  final Future<void> Function(Traveller t, String name) onRename;

  /// Throws [TravellerInUse] when the person appears in the ledger. The screen
  /// turns that into a sentence rather than letting it crash.
  final Future<void> Function(Traveller t) onDelete;

  const TravellersScreen({
    super.key,
    required this.travellers,
    required this.onAdd,
    required this.onRename,
    required this.onDelete,
  });

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);

    return Scaffold(
      backgroundColor: c.paper,
      appBar: AppBar(
        title: const Text('Who is on this trip'),
        backgroundColor: c.paper,
        foregroundColor: c.ink,
        elevation: 0,
      ),
      body: StreamBuilder<List<Traveller>>(
        stream: travellers,
        builder: (context, snap) {
          final list = snap.data;
          if (list == null) return const SizedBox();

          return ListView(
            padding: const EdgeInsets.only(bottom: 96),
            children: [
              if (list.isEmpty)
                Padding(
                  padding: const EdgeInsets.all(AppTokens.gutter),
                  child: Text(
                    'Add everyone sharing costs. Just names — there are no '
                    'accounts to set up and nothing is sent anywhere.',
                    style: AppTokens.captionStyle.copyWith(color: c.muted),
                  ),
                ),
              for (final t in list)
                _Row(
                  traveller: t,
                  onRename: (name) => onRename(t, name),
                  onDelete: () => _delete(context, t),
                ),
              const StencilLabel('How this works'),
              Padding(
                padding: const EdgeInsets.symmetric(
                  horizontal: AppTokens.gutter,
                ),
                child: Text(
                  'Everything stays on this phone. Nobody else gets a copy, '
                  'nobody is invited, nothing syncs.\n\n'
                  'Someone who appears in the ledger cannot be removed — '
                  'taking them out would change what everyone else owes.',
                  style: AppTokens.captionStyle.copyWith(color: c.muted),
                ),
              ),
              const SizedBox(height: AppTokens.s24),
            ],
          );
        },
      ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: () => _add(context),
        backgroundColor: c.signal,
        foregroundColor: c.paper,
        icon: const Icon(Icons.person_add_alt, size: 18),
        label: Text(
          'Add someone',
          style: AppTokens.stencilStyle.copyWith(
            fontSize: 10.5,
            color: c.paper,
          ),
        ),
      ),
    );
  }

  Future<void> _add(BuildContext context) async {
    final name = await _nameDialog(context, title: 'Add someone');
    if (name != null && name.isNotEmpty) await onAdd(name);
  }

  Future<void> _delete(BuildContext context, Traveller t) async {
    final messenger = ScaffoldMessenger.of(context);
    final c = AppTokens.of(context);
    try {
      await onDelete(t);
      Haptics.light();
    } on TravellerInUse catch (e) {
      // Refused, with the number. Cascading would quietly change everyone
      // else's balance: the expense keeps its total but loses a share.
      Haptics.reject();
      messenger.showSnackBar(
        SnackBar(
          backgroundColor: c.ink,
          content: Text(
            '${e.name} is in ${e.expenseCount} '
            '${e.expenseCount == 1 ? 'expense' : 'expenses'}. Remove or '
            'reassign those first.',
            style: AppTokens.captionStyle.copyWith(color: c.paper),
          ),
        ),
      );
    }
  }
}

Future<String?> _nameDialog(
  BuildContext context, {
  required String title,
  String initial = '',
}) async {
  final c = AppTokens.of(context);
  final controller = TextEditingController(text: initial);
  final name = await showDialog<String>(
    context: context,
    builder: (dialogContext) => AlertDialog(
      backgroundColor: c.paper,
      title: Text(
        title,
        style: AppTokens.titleStyle.copyWith(color: c.ink, fontSize: 18),
      ),
      content: TextField(
        controller: controller,
        autofocus: true,
        textCapitalization: TextCapitalization.words,
        style: AppTokens.rowTitleStyle.copyWith(color: c.ink),
        decoration: InputDecoration(
          hintText: 'Priya',
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
          child: Text('Save', style: TextStyle(color: c.signal)),
        ),
      ],
    ),
  );
  controller.dispose();
  return name;
}

class _Row extends StatelessWidget {
  final Traveller traveller;
  final ValueChanged<String> onRename;
  final VoidCallback onDelete;

  const _Row({
    required this.traveller,
    required this.onRename,
    required this.onDelete,
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
        border: Border(
          bottom: BorderSide(color: c.rule, width: AppTokens.hairline),
        ),
      ),
      child: Row(
        children: [
          Expanded(
            child: GestureDetector(
              onTap: () async {
                final name = await _nameDialog(
                  context,
                  title: 'Rename',
                  initial: traveller.name,
                );
                if (name != null && name.isNotEmpty) onRename(name);
              },
              behavior: HitTestBehavior.opaque,
              child: Text(
                traveller.name,
                style: AppTokens.rowTitleStyle.copyWith(color: c.ink),
              ),
            ),
          ),
          GestureDetector(
            onTap: onDelete,
            child: Icon(Icons.close, size: 18, color: c.muted),
          ),
        ],
      ),
    );
  }
}
