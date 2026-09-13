// lib/features/contacts/data/contact_actions.dart
//
// What a tap on a diary entry actually does.
//
// COPY IS THE PRIMARY ACTION. Yash dials by pasting into the Android dialer,
// so the app's job is to put the right digits on the clipboard and get out of
// the way. Direct dialling stays available as an explicit choice.
//
// Every action is logged, copy included. Since the dial now happens outside
// the app, a copy is the closest thing we observe to a call — without logging
// it, the record of who was reached rots.

import 'package:flutter/services.dart';
import 'package:url_launcher/url_launcher.dart';

import '../../../core/database/app_database.dart';
import '../../../core/theme/motion.dart';
import 'contacts_dao.dart';

/// Action strings written to `CallLogs.action`.
class ContactAction {
  static const copy = 'copy';
  static const dialer = 'dialer';
  static const call = 'call';
  static const sms = 'sms';
  static const whatsapp = 'whatsapp';
}

/// Why an action did not happen, so the UI can say something true.
class ActionFailure implements Exception {
  final String message;
  const ActionFailure(this.message);

  @override
  String toString() => message;
}

/// Launches platform intents and records what was done.
///
/// Injectable so tests can drive it without a platform: pass a [launch] that
/// records instead of launching.
class ContactActions {
  final ContactsDao dao;
  final int? tripId;

  /// Defaults to `url_launcher`. Returns false when nothing can handle it.
  final Future<bool> Function(Uri) launch;

  /// Defaults to the system clipboard.
  final Future<void> Function(String) copyToClipboard;

  ContactActions({
    required this.dao,
    this.tripId,
    Future<bool> Function(Uri)? launch,
    Future<void> Function(String)? copyToClipboard,
  }) : launch = launch ?? _launchExternal,
       copyToClipboard = copyToClipboard ?? _setClipboard;

  static Future<bool> _launchExternal(Uri uri) =>
      launchUrl(uri, mode: LaunchMode.externalApplication);

  static Future<void> _setClipboard(String text) =>
      Clipboard.setData(ClipboardData(text: text));

  /// The number to hand to the dialer: normalised if we have it, raw if not.
  ///
  /// Raw is what the user typed and recognises; E.164 is what dials reliably
  /// across borders. Normalisation can fail on bad input, so raw is the
  /// fallback rather than the other way round.
  static String dialable(Contact c) => c.phoneE164 ?? c.phoneRaw;

  /// Copies the number and logs it. Returns what landed on the clipboard.
  Future<String> copy(Contact c) async {
    final number = dialable(c);
    await copyToClipboard(number);
    Haptics.light();
    await _log(c, ContactAction.copy);
    return number;
  }

  /// Opens the platform dialer with no number in it, so the user can paste.
  ///
  /// Needs no permission. Whether `tel:` with an empty path opens the Android
  /// dialer or errors is UNVERIFIED on hardware — see ISSUE_7_Actions.md. On
  /// failure this reports it rather than doing nothing.
  Future<void> openDialer({Contact? forContact}) async {
    final ok = await launch(Uri(scheme: 'tel'));
    if (!ok) {
      throw const ActionFailure('No dialer app on this phone.');
    }
    if (forContact != null) {
      await _log(forContact, ContactAction.dialer);
    }
  }

  /// Places a call to a number that is not a contact — a bundled helpline.
  ///
  /// Heavy haptic, because this is the emergency path. Nothing is written to
  /// CallLogs: those rows reference a contact, and a helpline is not one.
  Future<void> callNumber(String number) async {
    Haptics.grave();
    await _launchOrThrow(
      Uri(scheme: 'tel', path: number),
      'No app on this phone can place a call.',
    );
  }

  /// Opens the phone's dialer on a number that is not a contact.
  ///
  /// Same empty-path trick as `openDialer`: the number goes on the clipboard
  /// and the dialer opens ready to paste, because that is how this app was
  /// designed to be used.
  Future<void> openDialerFor(String number) async {
    await copyToClipboard(number);
    Haptics.light();
    await _launchOrThrow(
      Uri(scheme: 'tel'),
      'No dialer app on this phone.',
    );
  }

  /// Hands a place off to a maps app, for reviews and photos this app does
  /// not carry.
  ///
  /// THIS IS THE ONE ACTION IN THE APP THAT NEEDS SIGNAL, and every screen
  /// offering it says so before the tap rather than after.
  Future<void> openMaps(String url) async {
    Haptics.light();
    await _launchOrThrow(
      Uri.parse(url),
      'No app on this phone can open a map link.',
    );
  }

  /// Copies a number that is not a contact.
  Future<String> copyNumber(String number) async {
    await copyToClipboard(number);
    Haptics.light();
    return number;
  }

  Future<void> call(Contact c) async {
    Haptics.light();
    await _launchOrThrow(
      Uri(scheme: 'tel', path: dialable(c)),
      'No app on this phone can place a call.',
    );
    await _log(c, ContactAction.call);
  }

  Future<void> sms(Contact c) async {
    await _launchOrThrow(
      Uri(scheme: 'sms', path: dialable(c)),
      'No messaging app on this phone.',
    );
    await _log(c, ContactAction.sms);
  }

  Future<void> whatsapp(Contact c) async {
    // wa.me takes digits only — a leading + or any spacing breaks the link.
    final digits = dialable(c).replaceAll(RegExp(r'[^0-9]'), '');
    if (digits.isEmpty) {
      throw const ActionFailure('That entry has no number to message.');
    }
    await _launchOrThrow(
      Uri.parse('https://wa.me/$digits'),
      'No app on this phone can open WhatsApp.',
    );
    await _log(c, ContactAction.whatsapp);
  }

  Future<void> _launchOrThrow(Uri uri, String message) async {
    bool ok;
    try {
      ok = await launch(uri);
    } on Object {
      // Some devices throw rather than returning false. Either way the user
      // needs to be told, because a silent no-op is the worst outcome here.
      throw ActionFailure(message);
    }
    if (!ok) throw ActionFailure(message);
  }

  Future<void> _log(Contact c, String action) =>
      dao.logCall(contactId: c.id, tripId: tripId, action: action);
}
