// lib/features/emergency/presentation/sos_panel.dart
//
// "Text my location" — the top of the SOS tab. See data/sos.dart for why it
// is SMS, why it never sends by itself, and why it never waits on the GPS.

import 'dart:async';

import 'package:flutter/material.dart';

import '../../../core/database/app_database.dart';
import '../../../core/theme/app_tokens.dart';
import '../../../core/widgets/retro.dart';
import '../../contacts/data/phone_contact_picker.dart';
import '../../map/data/here.dart';
import '../data/sos.dart';

class SosPanel extends StatefulWidget {
  final Stream<List<TrustedContact>> trusted;
  final LocationSource location;

  /// The stop the trip plan says you are at, for when there is no fix.
  final Future<String?> Function() nearStop;

  /// Today's vehicle and driver, when a leg is planned today.
  final Future<String?> Function()? ride;

  /// Opens the SMS app. False when there is none to open.
  final Future<bool> Function(Uri sms) openSms;

  /// The share sheet, for WhatsApp and the rest.
  final Future<void> Function(String text) share;

  /// Choosing someone: the phone's contact picker, then saving them.
  final Future<PickedContact?> Function() pickPerson;
  final Future<void> Function(String name, String phone) addPerson;
  final Future<void> Function(TrustedContact person) removePerson;

  /// How long to wait for a fix before sending without one. Short: this is
  /// an emergency, and the person can always skip it sooner.
  final Duration wait;

  const SosPanel({
    super.key,
    required this.trusted,
    required this.location,
    required this.nearStop,
    this.ride,
    required this.openSms,
    required this.share,
    required this.pickPerson,
    required this.addPerson,
    required this.removePerson,
    this.wait = const Duration(seconds: 12),
  });

  @override
  State<SosPanel> createState() => _SosPanelState();
}

class _SosPanelState extends State<SosPanel> {
  /// Who the text is being prepared for; null when idle. "share" for the
  /// share sheet.
  Object? _preparing;
  Completer<void>? _skip;

  void _say(String text) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(text)));
  }

  Future<String> _compose() async {
    HereFix? fix;
    // Asking is right here: the person has just pressed an SOS button.
    final state = await widget.location.check(ask: true);
    if (state == HereState.locating || state == HereState.found) {
      final skip = _skip = Completer<void>();
      fix = await Future.any<HereFix?>([
        widget.location.once(timeout: widget.wait),
        skip.future.then((_) => null),
      ]);
    }
    return sosMessage(
      fix: fix,
      nearStop: await widget.nearStop(),
      ride: await widget.ride?.call(),
    );
  }

  Future<void> _send(TrustedContact person) async {
    if (_preparing != null) return;
    setState(() => _preparing = person.id);
    try {
      final text = await _compose();
      final opened = await widget.openSms(smsUri(person.phoneE164, text));
      // No SMS app is rare, but the message must still get out somehow.
      if (!opened) await widget.share(text);
    } finally {
      if (mounted) setState(() => _preparing = null);
    }
  }

  Future<void> _shareAnotherWay() async {
    if (_preparing != null) return;
    setState(() => _preparing = 'share');
    try {
      await widget.share(await _compose());
    } finally {
      if (mounted) setState(() => _preparing = null);
    }
  }

  Future<void> _add() async {
    final PickedContact? picked;
    try {
      picked = await widget.pickPerson();
    } on ContactPickException catch (e) {
      _say(e.message);
      return;
    }
    if (picked == null) return;
    try {
      await widget.addPerson(picked.name, picked.number);
    } on TrustedException catch (e) {
      _say(e.message);
    }
  }

  Future<void> _remove(TrustedContact person) async {
    final yes = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: Text('Stop texting ${person.name}?'),
        content: const Text('They will no longer get your SOS text.'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text('Keep'),
          ),
          TextButton(
            onPressed: () => Navigator.pop(context, true),
            child: const Text('Remove'),
          ),
        ],
      ),
    );
    if (yes == true) await widget.removePerson(person);
  }

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    return StreamBuilder<List<TrustedContact>>(
      stream: widget.trusted,
      builder: (context, snap) {
        final people = snap.data ?? const <TrustedContact>[];
        return Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const StencilLabel('Text my location'),
            Padding(
              padding: const EdgeInsets.symmetric(
                horizontal: AppTokens.gutter,
              ),
              child: Text(
                'Writes a text saying where you are and opens your SMS app — '
                'you press Send. A text gets through on one bar, with no '
                'mobile data.',
                style: AppTokens.captionStyle.copyWith(color: c.muted),
              ),
            ),
            const SizedBox(height: AppTokens.s8),
            for (final person in people)
              _PersonRow(
                person: person,
                preparing: _preparing == person.id,
                onTap: () => _send(person),
                onLongPress: () => _remove(person),
              ),
            if (_preparing != null)
              _Waiting(onSkip: () => _skip?.complete()),
            _ActionRow(
              key: const Key('sos-add'),
              icon: Icons.person_add_alt_1_outlined,
              text: people.isEmpty
                  ? 'Choose who gets your SOS text'
                  : 'Add someone',
              onTap: _add,
            ),
            _ActionRow(
              key: const Key('sos-share'),
              icon: Icons.ios_share,
              text: 'Send it another way (WhatsApp…)',
              onTap: _shareAnotherWay,
            ),
          ],
        );
      },
    );
  }
}

class _PersonRow extends StatelessWidget {
  final TrustedContact person;
  final bool preparing;
  final VoidCallback onTap;
  final VoidCallback onLongPress;

  const _PersonRow({
    required this.person,
    required this.preparing,
    required this.onTap,
    required this.onLongPress,
  });

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    return InkWell(
      onTap: onTap,
      onLongPress: onLongPress,
      child: Container(
        margin: const EdgeInsets.fromLTRB(
          AppTokens.gutter,
          AppTokens.s4,
          AppTokens.gutter,
          AppTokens.s4,
        ),
        padding: const EdgeInsets.all(AppTokens.s12),
        decoration: BoxDecoration(
          border: Border.all(color: c.emergency, width: 1.4),
          borderRadius: BorderRadius.circular(AppTokens.radiusSoft),
        ),
        child: Row(
          children: [
            Icon(Icons.sms_outlined, color: c.emergency),
            const SizedBox(width: AppTokens.s12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Text ${person.name}',
                    style: AppTokens.rowTitleStyle.copyWith(color: c.ink),
                  ),
                  Text(
                    person.phoneE164,
                    style: AppTokens.captionStyle.copyWith(color: c.muted),
                  ),
                ],
              ),
            ),
            if (preparing)
              SizedBox(
                width: 18,
                height: 18,
                child: CircularProgressIndicator(
                  strokeWidth: 2,
                  color: c.emergency,
                ),
              ),
          ],
        ),
      ),
    );
  }
}

class _Waiting extends StatelessWidget {
  final VoidCallback onSkip;
  const _Waiting({required this.onSkip});

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    return Padding(
      padding: const EdgeInsets.symmetric(
        horizontal: AppTokens.gutter,
        vertical: AppTokens.s4,
      ),
      child: Row(
        children: [
          Expanded(
            child: Text(
              'Getting your location…',
              style: AppTokens.captionStyle.copyWith(color: c.muted),
            ),
          ),
          TextButton(
            key: const Key('sos-skip'),
            onPressed: onSkip,
            child: const Text('Send without it'),
          ),
        ],
      ),
    );
  }
}

class _ActionRow extends StatelessWidget {
  final IconData icon;
  final String text;
  final VoidCallback onTap;
  const _ActionRow({
    super.key,
    required this.icon,
    required this.text,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    return InkWell(
      onTap: onTap,
      child: Padding(
        padding: const EdgeInsets.symmetric(
          horizontal: AppTokens.gutter,
          vertical: AppTokens.s12,
        ),
        child: Row(
          children: [
            Icon(icon, size: 18, color: c.signal),
            const SizedBox(width: AppTokens.s8),
            Expanded(
              child: Text(
                text,
                style: AppTokens.rowTitleStyle.copyWith(color: c.signal),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
