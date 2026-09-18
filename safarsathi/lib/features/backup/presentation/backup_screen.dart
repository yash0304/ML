// lib/features/backup/presentation/backup_screen.dart
//
// Issue #56. Two buttons and a great deal of care about the second one.

import 'package:flutter/material.dart';

import '../../../core/theme/app_tokens.dart';
import '../../../core/theme/motion.dart';
import '../../../core/widgets/retro.dart';
import '../../map/data/tile_math.dart' show describeBytes;
import '../data/backup.dart';
import '../data/full_backup.dart';

class BackupScreen extends StatefulWidget {
  /// Writes the backup wherever the user chooses. Returns what to say about
  /// it, or null if they cancelled the save dialog.
  final Future<String?> Function() onExport;

  /// The same, plus every downloaded tile. Null when the app cannot offer it.
  final Future<String?> Function()? onExportFull;

  /// What a full backup would carry, for the label on its button.
  final Future<FullBackupPlan> Function()? onPlanFull;

  /// Lets the user pick a file and parses it. Null if they cancelled.
  final Future<BackupContents?> Function() onPick;

  /// What is on the phone right now, for the side-by-side.
  final Future<BackupContents> Function() onCurrent;

  final Future<void> Function(BackupContents) onRestore;

  const BackupScreen({
    super.key,
    required this.onExport,
    this.onExportFull,
    this.onPlanFull,
    required this.onPick,
    required this.onCurrent,
    required this.onRestore,
  });

  @override
  State<BackupScreen> createState() => _BackupScreenState();
}

class _BackupScreenState extends State<BackupScreen> {
  bool _busy = false;
  String? _message;
  String? _problem;
  FullBackupPlan? _plan;

  BackupContents? _picked;
  BackupContents? _current;

  @override
  void initState() {
    super.initState();
    // Asked once, on open. The button cannot honestly say how big the file
    // will be without it, and "Save a full backup" with no size is the kind
    // of button people press and then regret.
    widget.onPlanFull?.call().then((plan) {
      if (mounted) setState(() => _plan = plan);
    });
  }

  Future<void> _run(Future<void> Function() body) async {
    setState(() {
      _busy = true;
      _message = null;
      _problem = null;
    });
    try {
      await body();
    } on BackupException catch (e) {
      if (mounted) setState(() => _problem = e.message);
    } on Object catch (e) {
      if (mounted) setState(() => _problem = '$e');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _export() => _run(() async {
    final where = await widget.onExport();
    if (!mounted || where == null) return;
    Haptics.confirm();
    setState(() => _message = where);
  });

  Future<void> _exportFull() => _run(() async {
    final where = await widget.onExportFull!();
    if (!mounted || where == null) return;
    Haptics.confirm();
    setState(() => _message = where);
  });

  Future<void> _pick() => _run(() async {
    final picked = await widget.onPick();
    if (!mounted || picked == null) return;
    final current = await widget.onCurrent();
    if (!mounted) return;
    setState(() {
      _picked = picked;
      _current = current;
    });
  });

  Future<void> _restore() async {
    final backup = _picked;
    if (backup == null) return;
    if (!await _confirm(backup)) return;

    await _run(() async {
      await widget.onRestore(backup);
      if (!mounted) return;
      Haptics.confirm();
      setState(() {
        _picked = null;
        _current = null;
        _message = 'Restored. Everything from that backup is back.';
      });
    });
  }

  Future<bool> _confirm(BackupContents backup) async {
    final c = AppTokens.of(context);
    final now = _current;
    final losing = now != null && !now.isEmpty;

    final ok = await showDialog<bool>(
      context: context,
      builder: (dialogContext) => AlertDialog(
        backgroundColor: c.paper,
        title: Text(
          losing ? 'Replace what is here?' : 'Restore this backup?',
          style: AppTokens.titleStyle.copyWith(color: c.ink),
        ),
        content: Text(
          losing
              ? 'This phone has ${now.trips} '
                    '${now.trips == 1 ? "trip" : "trips"} and '
                    '${now.contacts} '
                    '${now.contacts == 1 ? "number" : "numbers"} in it now. '
                    'All of it goes, and the backup takes its place. '
                    'There is no undo.'
              : 'Nothing here will be lost — the app is empty.',
          style: AppTokens.captionStyle.copyWith(color: c.muted),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(dialogContext).pop(false),
            child: Text('Keep what is here', style: TextStyle(color: c.muted)),
          ),
          TextButton(
            onPressed: () => Navigator.of(dialogContext).pop(true),
            child: Text(
              losing ? 'Replace it' : 'Restore',
              style: TextStyle(color: losing ? c.emergency : c.signal),
            ),
          ),
        ],
      ),
    );
    return ok ?? false;
  }

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    final picked = _picked;

    return Scaffold(
      backgroundColor: c.paper,
      appBar: AppBar(
        title: const Text('Backup'),
        backgroundColor: c.paper,
        foregroundColor: c.ink,
        elevation: 0,
      ),
      body: ListView(
        padding: const EdgeInsets.only(bottom: AppTokens.s32),
        children: [
          const StencilLabel('Save a copy'),
          const _Note(
            // The reason, said plainly, because a backup nobody takes is the
            // same as no backup at all.
            'Everything you typed — the trip, the numbers, the checklist, the '
            'money — as one file. A dropped phone on the road does the same '
            'thing to this app as reinstalling it, and there is no server '
            'holding a second copy.',
          ),
          _Button(
            label: 'Save a backup',
            primary: true,
            busy: _busy,
            onTap: _export,
          ),
          const _Note(
            'Places, forecasts and map tiles are left out — they came off the '
            'network and can come off it again. Emergency numbers are left '
            'out too: those only ever come from the app itself, never from a '
            'file.',
            muted: true,
          ),

          if (widget.onExportFull != null) ...[
            const StencilLabel('With the map'),
            _Note(
              _plan == null
                  ? 'Working out what is downloaded…'
                  : !_plan!.hasTiles
                  ? 'Nothing is downloaded yet, so this would hold the same '
                        'thing as the file above. Download a map first.'
                  : 'The same, plus the ${_plan!.tileCount} map tiles already '
                        'on this phone — about '
                        '${describeBytes(_plan!.tileBytes)}. Bigger and slower '
                        'to write, and the only version that puts a working '
                        'offline map onto a new phone without WiFi.',
            ),
            _Button(
              label: 'Save a full backup',
              primary: false,
              busy: _busy || _plan?.hasTiles != true,
              onTap: _exportFull,
            ),
          ],

          const StencilLabel('Bring one back'),
          _Button(
            label: picked == null ? 'Choose a backup file' : 'Choose another',
            primary: false,
            busy: _busy,
            onTap: _pick,
          ),

          if (picked != null) ...[
            _Summary(backup: picked, current: _current),
            _Button(
              label: 'Restore this backup',
              primary: true,
              busy: _busy,
              onTap: _restore,
            ),
          ],

          if (_message != null)
            Padding(
              padding: const EdgeInsets.fromLTRB(
                AppTokens.gutter,
                AppTokens.s16,
                AppTokens.gutter,
                0,
              ),
              child: Text(
                _message!,
                style: AppTokens.captionStyle.copyWith(color: c.signal),
              ),
            ),

          if (_problem != null)
            Padding(
              padding: const EdgeInsets.fromLTRB(
                AppTokens.gutter,
                AppTokens.s16,
                AppTokens.gutter,
                0,
              ),
              child: Text(
                _problem!,
                style: AppTokens.captionStyle.copyWith(color: c.cautionMark),
              ),
            ),
        ],
      ),
    );
  }
}

/// What the chosen file holds, against what is on the phone.
class _Summary extends StatelessWidget {
  final BackupContents backup;
  final BackupContents? current;

  const _Summary({required this.backup, this.current});

  static String _when(DateTime d) {
    final local = d.toLocal();
    String two(int n) => n.toString().padLeft(2, '0');
    return '${two(local.day)}/${two(local.month)}/${local.year} '
        '${two(local.hour)}:${two(local.minute)}';
  }

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    final now = current;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Padding(
          padding: const EdgeInsets.fromLTRB(
            AppTokens.gutter,
            AppTokens.s12,
            AppTokens.gutter,
            AppTokens.s8,
          ),
          child: Text(
            'Taken ${_when(backup.createdAt)}',
            style: AppTokens.captionStyle.copyWith(color: c.muted),
          ),
        ),
        _Row(label: 'Trips', backup: backup.trips, current: now?.trips),
        _Row(label: 'Stops', backup: backup.stops, current: now?.stops),
        _Row(label: 'Numbers', backup: backup.contacts, current: now?.contacts),
        _Row(
          label: 'Checklist items',
          backup: backup.checklistItems,
          current: now?.checklistItems,
        ),
        _Row(
          label: 'Expenses',
          backup: backup.expenses,
          current: now?.expenses,
        ),
        if (backup.carriesMap)
          _Row(label: 'Map tiles', backup: backup.tileCount),

        Padding(
          padding: const EdgeInsets.fromLTRB(
            AppTokens.gutter,
            AppTokens.s12,
            AppTokens.gutter,
            AppTokens.s16,
          ),
          child: Text(
            // THE ONE THING A RESTORE TAKES ON TRUST. Everywhere else in this
            // app, anything arriving in bulk is forced to unconfirmed. A
            // backup is the app's own record of calls you made, so it comes
            // back as it was — and the screen says so rather than letting it
            // happen quietly.
            backup.confirmedContacts == 0
                ? 'Nothing in this backup is marked as confirmed.'
                : 'This file says ${backup.confirmedContacts} of those '
                      '${backup.confirmedContacts == 1 ? "numbers was" : "numbers were"} '
                      'confirmed by calling them, and they will come back that '
                      'way. Restore only a backup this app made, from your '
                      'own phone.',
            style: AppTokens.captionStyle.copyWith(
              color: backup.confirmedContacts == 0 ? c.muted : c.cautionMark,
            ),
          ),
        ),
      ],
    );
  }
}

class _Row extends StatelessWidget {
  final String label;
  final int backup;
  final int? current;

  const _Row({required this.label, required this.backup, this.current});

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
            child: Text(
              label,
              style: AppTokens.rowTitleStyle.copyWith(color: c.ink),
            ),
          ),
          if (current != null) ...[
            Text(
              '$current',
              style: AppTokens.numberStyle.copyWith(color: c.muted),
            ),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: AppTokens.s8),
              child: Icon(Icons.arrow_forward, size: 13, color: c.rule),
            ),
          ],
          Text(
            '$backup',
            style: AppTokens.numberStyle.copyWith(color: c.ink),
          ),
        ],
      ),
    );
  }
}

class _Note extends StatelessWidget {
  final String text;
  final bool muted;
  const _Note(this.text, {this.muted = false});

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    return Padding(
      padding: const EdgeInsets.fromLTRB(
        AppTokens.gutter,
        0,
        AppTokens.gutter,
        AppTokens.s12,
      ),
      child: Text(
        text,
        style: AppTokens.captionStyle.copyWith(color: c.muted),
      ),
    );
  }
}

class _Button extends StatelessWidget {
  final String label;
  final bool primary;
  final bool busy;
  final VoidCallback onTap;

  const _Button({
    required this.label,
    required this.primary,
    required this.busy,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    return Padding(
      padding: const EdgeInsets.fromLTRB(
        AppTokens.gutter,
        0,
        AppTokens.gutter,
        AppTokens.s8,
      ),
      child: PressScale(
        onTap: busy ? null : onTap,
        child: Container(
          width: double.infinity,
          padding: const EdgeInsets.symmetric(vertical: AppTokens.s12),
          alignment: Alignment.center,
          decoration: BoxDecoration(
            color: busy
                ? c.stone
                : primary
                ? c.signal
                : Colors.transparent,
            border: Border.all(color: busy ? c.rule : c.ink),
            borderRadius: BorderRadius.circular(AppTokens.radiusSoft),
          ),
          child: Text(
            // Uppercase, like every other stencil control in the app.
            (busy ? 'Working…' : label).toUpperCase(),
            style: AppTokens.stencilStyle.copyWith(
              fontSize: 11,
              color: busy
                  ? c.muted
                  : primary
                  ? c.paper
                  : c.ink,
            ),
          ),
        ),
      ),
    );
  }
}
