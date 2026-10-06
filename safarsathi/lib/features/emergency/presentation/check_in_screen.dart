// lib/features/emergency/presentation/check_in_screen.dart
//
// "Reached Sohra safely" — #33. Same people, same road as the SOS text: the
// phone's SMS app, opened with the message written, and the person presses
// Send.

import 'dart:async';

import 'package:flutter/material.dart';

import '../../../core/database/app_database.dart';
import '../../../core/theme/app_tokens.dart';
import '../../../core/theme/motion.dart';
import '../../../core/widgets/retro.dart';
import '../../map/data/here.dart';
import '../data/check_in.dart';
import '../data/sos.dart' show smsUri;

typedef CheckInStop = ({int id, String name});

class CheckInScreen extends StatefulWidget {
  final List<CheckInStop> stops;
  final int? initialStopId;
  final Stream<List<TrustedContact>> trusted;
  final LocationSource location;

  /// "Alpha Guest House, +91 89748 04455" — the stay chosen at that stop.
  final Future<String?> Function(int stopId) stayAt;

  final Future<bool> Function(Uri sms) openSms;
  final Future<void> Function(String text) share;

  /// Records the check-in once a message has been opened to send.
  final Future<void> Function(CheckInStop stop, HereFix? fix) onRecorded;

  final Duration wait;
  final DateTime Function() clock;

  const CheckInScreen({
    super.key,
    required this.stops,
    required this.trusted,
    required this.location,
    required this.stayAt,
    required this.openSms,
    required this.share,
    required this.onRecorded,
    this.initialStopId,
    this.wait = const Duration(seconds: 12),
    this.clock = DateTime.now,
  });

  @override
  State<CheckInScreen> createState() => _CheckInScreenState();
}

class _CheckInScreenState extends State<CheckInScreen> {
  late int? _stopId =
      widget.initialStopId ??
      (widget.stops.isEmpty ? null : widget.stops.first.id);
  bool _withLocation = true;
  bool _busy = false;
  String? _done;

  CheckInStop? get _stop {
    for (final s in widget.stops) {
      if (s.id == _stopId) return s;
    }
    return null;
  }

  Future<(String, HereFix?)> _compose(CheckInStop stop) async {
    HereFix? fix;
    if (_withLocation) {
      final state = await widget.location.check(ask: true);
      if (state == HereState.locating || state == HereState.found) {
        fix = await widget.location.once(timeout: widget.wait);
      }
    }
    final text = checkInMessage(
      stopName: stop.name,
      at: widget.clock(),
      fix: fix,
      stay: await widget.stayAt(stop.id),
      now: widget.clock(),
    );
    return (text, fix);
  }

  Future<void> _send({TrustedContact? to}) async {
    final stop = _stop;
    if (stop == null || _busy) return;
    setState(() => _busy = true);
    try {
      final (text, fix) = await _compose(stop);
      var opened = false;
      if (to != null) opened = await widget.openSms(smsUri(to.phoneE164, text));
      if (!opened) {
        await widget.share(text);
        opened = true;
      }
      await widget.onRecorded(stop, fix);
      Haptics.confirm();
      if (mounted) {
        setState(
          () => _done = 'Marked as checked in at ${stop.name}. The app cannot '
              'see whether you pressed Send — it counts the message opened.',
        );
      }
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    final caption = AppTokens.captionStyle.copyWith(color: c.muted);
    final stop = _stop;

    return Scaffold(
      backgroundColor: c.paper,
      appBar: AppBar(
        title: const Text('Check in'),
        backgroundColor: c.paper,
        foregroundColor: c.ink,
        elevation: 0,
      ),
      body: StreamBuilder<List<TrustedContact>>(
        stream: widget.trusted,
        builder: (context, snap) {
          final people = snap.data ?? const <TrustedContact>[];
          return ListView(
            padding: const EdgeInsets.only(bottom: AppTokens.s32),
            children: [
              Padding(
                padding: const EdgeInsets.fromLTRB(
                  AppTokens.gutter,
                  AppTokens.s8,
                  AppTokens.gutter,
                  0,
                ),
                child: Text(
                  'Tell home you have arrived. The text is written for you '
                  'and your SMS app opens — you press Send. One bar is '
                  'enough; no mobile data needed.',
                  style: caption,
                ),
              ),
              const StencilLabel('Reached'),
              Padding(
                padding: const EdgeInsets.symmetric(
                  horizontal: AppTokens.gutter,
                ),
                child: Wrap(
                  spacing: AppTokens.s8,
                  runSpacing: AppTokens.s8,
                  children: [
                    for (final s in widget.stops)
                      ChoiceChip(
                        key: Key('checkin-stop-${s.id}'),
                        label: Text(s.name),
                        selected: s.id == _stopId,
                        onSelected: (_) => setState(() {
                          _stopId = s.id;
                          _done = null;
                        }),
                      ),
                  ],
                ),
              ),
              SwitchListTile(
                key: const Key('checkin-location'),
                value: _withLocation,
                onChanged: (v) => setState(() => _withLocation = v),
                title: const Text('Include where I am'),
                subtitle: Text(
                  'From the GPS, which needs no signal. Waits up to '
                  '${widget.wait.inSeconds} s for a fix, then sends without.',
                  style: caption,
                ),
              ),
              const StencilLabel('Send to'),
              if (people.isEmpty)
                Padding(
                  padding: const EdgeInsets.symmetric(
                    horizontal: AppTokens.gutter,
                  ),
                  child: Text(
                    'No one to tell yet. Choose people on the SOS tab — the '
                    'same people get both your check-ins and an SOS.',
                    style: AppTokens.captionStyle.copyWith(
                      color: c.cautionMark,
                    ),
                  ),
                ),
              for (final p in people)
                ListTile(
                  key: Key('checkin-to-${p.id}'),
                  enabled: stop != null && !_busy,
                  leading: Icon(Icons.sms_outlined, color: c.signal),
                  title: Text('Text ${p.name}'),
                  subtitle: Text(p.phoneE164, style: caption),
                  onTap: () => _send(to: p),
                ),
              ListTile(
                key: const Key('checkin-share'),
                enabled: stop != null && !_busy,
                leading: Icon(Icons.ios_share, color: c.muted),
                title: const Text('Send it another way (WhatsApp…)'),
                onTap: () => _send(),
              ),
              if (_busy)
                Padding(
                  padding: const EdgeInsets.all(AppTokens.gutter),
                  child: Text('Writing the message…', style: caption),
                ),
              if (_done != null)
                Padding(
                  padding: const EdgeInsets.all(AppTokens.gutter),
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Icon(Icons.check_circle, size: 18, color: c.signal),
                      const SizedBox(width: AppTokens.s8),
                      Expanded(
                        child: Text(
                          _done!,
                          key: const Key('checkin-done'),
                          style: AppTokens.captionStyle.copyWith(
                            color: c.signal,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
            ],
          );
        },
      ),
    );
  }
}
