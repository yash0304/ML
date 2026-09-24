// lib/features/emergency/data/sos.dart
//
// A text that says where you are, to the people you chose, ready to send.
//
// SMS, NOT DATA. A text gets through on one bar with no mobile data — the
// exact situation on the Dawki road — where WhatsApp and every app that
// needs the internet do not. So the main path opens the phone's own SMS app
// with the message written; the person presses Send. The app never sends
// anything by itself: an SOS that fires without being seen is an SOS that
// fires by accident.
//
// NEVER BLOCKED ON LOCATION. No fix, a refused permission, GPS switched off:
// the text still goes, saying the location is not available and naming the
// stop on the plan instead. Help that waits for a satellite is not help.

import 'package:drift/drift.dart';

import '../../../core/database/app_database.dart';
import '../../contacts/data/phone_normaliser.dart';
import '../../map/data/here.dart';

/// The message. Pure, so every case is pinned by a test.
///
/// [ride] is today's vehicle and driver when a leg is planned today —
/// "Taxi ML 05 A 1234 · driver Anthony, +91 98560 12345". On a road with no
/// signal it is often the fastest way anyone can find you: the driver may
/// have a bar where you do not, and a registration can be asked about.
String sosMessage({
  HereFix? fix,
  String? nearStop,
  String? ride,
  DateTime? now,
}) {
  final out = StringBuffer('SOS — I need help.\n');

  if (fix != null) {
    final lat = fix.at.lat.toStringAsFixed(5);
    final lon = fix.at.lon.toStringAsFixed(5);
    out
      ..writeln('Location: $lat, $lon (${describeFix(fix, now: now)})')
      ..writeln('Map: https://maps.google.com/?q=$lat,$lon');
  } else {
    out.writeln('Location: not available from my phone right now.');
  }
  if (nearStop != null) {
    out.writeln(
      fix == null
          ? 'I should be at or near $nearStop (on our trip plan).'
          : 'Near $nearStop on our trip plan.',
    );
  }
  if (ride != null) out.writeln('Travelling: $ride');
  out.write('Sent from SafarSathi.');
  return out.toString();
}

/// The phone's SMS app, opened on [phone] with [body] written in.
///
/// Built by hand rather than with `queryParameters`, which encodes spaces as
/// "+" — and several SMS apps show those plus signs literally, turning
/// "I need help" into "I+need+help" in the one message that must be clear.
Uri smsUri(String phone, String body) => Uri.parse(
  'sms:${Uri.encodeComponent(phone)}?body=${Uri.encodeComponent(body)}',
);

/// The people who get the SOS text. For every trip: family does not change
/// between holidays.
Stream<List<TrustedContact>> watchTrusted(AppDatabase db) =>
    (db.select(db.trustedContacts)
          ..where((t) => t.tripId.isNull())
          ..orderBy([(t) => OrderingTerm(expression: t.id)]))
        .watch();

class TrustedException implements Exception {
  final String message;
  const TrustedException(this.message);
  @override
  String toString() => message;
}

/// Adds [name] at [phone]. Refuses a number that will not normalise — an
/// SOS to a number that cannot be dialled is worse than none, because the
/// person believes it was sent — and a number already on the list.
Future<void> addTrusted(AppDatabase db, String name, String phone) async {
  final normalised = PhoneNormaliser.normalise(phone);
  final e164 = normalised.e164;
  if (e164 == null) {
    throw TrustedException(
      '"$phone" could not be read as a phone number, so it cannot receive '
      'an SOS text.',
    );
  }
  final existing = await (db.select(db.trustedContacts)
        ..where((t) => t.phoneE164.equals(e164) & t.tripId.isNull()))
      .get();
  if (existing.isNotEmpty) {
    throw TrustedException('${existing.first.name} is already on the list.');
  }
  await db.into(db.trustedContacts).insert(
    TrustedContactsCompanion.insert(
      name: name.trim().isEmpty ? e164 : name.trim(),
      phoneE164: e164,
    ),
  );
}

Future<void> removeTrusted(AppDatabase db, int id) =>
    (db.delete(db.trustedContacts)..where((t) => t.id.equals(id))).go();
