// lib/features/contacts/data/entry_draft.dart
//
// What the entry form produces, and how it reaches the database.
//
// A plain value object rather than a drift companion, so the form has no
// opinion about persistence and can be tested without one.

import 'package:drift/drift.dart';

import '../../../core/database/app_database.dart';
import 'contacts_dao.dart';

class EntryDraft {
  /// Null for a new entry.
  final int? id;

  final String name;
  final String phoneRaw;
  final String? phoneE164;
  final String category;
  final int? stopId;
  final String? note;
  final bool hasWhatsapp;

  /// Where the place is. Both or neither; null clears a saved position.
  final double? lat;
  final double? lon;

  /// Set when an already-confirmed entry has had its digits changed.
  ///
  /// A confirmation means "I called THIS number and it worked". Change the
  /// digits and that is no longer true, so the confirmation has to go with
  /// them. See DECISIONS.md.
  final bool resetConfirmation;

  const EntryDraft({
    this.id,
    required this.name,
    required this.phoneRaw,
    this.phoneE164,
    this.category = ContactCategory.other,
    this.stopId,
    this.note,
    this.hasWhatsapp = false,
    this.lat,
    this.lon,
    this.resetConfirmation = false,
  });

  bool get isNew => id == null;
}

/// Writes a draft. New entries always land unconfirmed; edits keep their
/// tier unless the number itself changed.
Future<int> saveEntry(ContactsDao dao, EntryDraft draft, {int? tripId}) async {
  final note = (draft.note?.trim().isEmpty ?? true) ? null : draft.note!.trim();

  if (draft.isNew) {
    return dao.insertContact(
      ContactsCompanion.insert(
        name: draft.name.trim(),
        phoneRaw: draft.phoneRaw.trim(),
        phoneE164: Value(draft.phoneE164),
        tripId: Value(tripId),
        stopId: Value(draft.stopId),
        category: Value(draft.category),
        note: Value(note),
        hasWhatsapp: Value(draft.hasWhatsapp),
        lat: Value(draft.lat),
        lon: Value(draft.lon),
        // Never anything else. Typing a number does not make it work.
        tier: Value(ContactTier.userEntered.name),
        callConfirmed: const Value(false),
      ),
    );
  }

  await (dao.update(dao.contacts)..where((c) => c.id.equals(draft.id!))).write(
    ContactsCompanion(
      name: Value(draft.name.trim()),
      phoneRaw: Value(draft.phoneRaw.trim()),
      phoneE164: Value(draft.phoneE164),
      stopId: Value(draft.stopId),
      category: Value(draft.category),
      note: Value(note),
      hasWhatsapp: Value(draft.hasWhatsapp),
      // The form starts from the saved position, so writing it back is a
      // no-op unless the person changed or cleared it.
      lat: Value(draft.lat),
      lon: Value(draft.lon),
      // Absent leaves the columns alone; present clears the confirmation.
      callConfirmed: draft.resetConfirmation
          ? const Value(false)
          : const Value.absent(),
      confirmedAt: draft.resetConfirmation
          ? const Value(null)
          : const Value.absent(),
      tier: draft.resetConfirmation
          ? Value(ContactTier.userEntered.name)
          : const Value.absent(),
    ),
  );
  return draft.id!;
}
