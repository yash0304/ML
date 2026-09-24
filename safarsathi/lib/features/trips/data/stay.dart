// lib/features/trips/data/stay.dart
//
// Where you actually sleep at a stop — as opposed to every guest house the
// diary happens to hold there.
//
// A stop collects accommodation numbers from several places: a sheet of
// options, a backup, the booking you made. Before this, the app treated all
// of them as "tonight" and showed whichever sorted first, so Bramhome Guest
// House appeared as the Shillong bed for somebody not staying there. The
// choice is now the person's, and until they make it the app says "not
// decided" rather than picking one.

import 'package:drift/drift.dart';

import '../../../core/database/app_database.dart';
import '../../contacts/data/contacts_dao.dart' show ContactCategory;
import 'tonight.dart' show staysAt;

/// The stay chosen for [stop], or null when none is chosen or the chosen
/// entry has since been deleted. An id that matches nothing is "not
/// decided", never an error: see Stops.stayContactId.
Contact? chosenStay(Stop stop, List<Contact> diary) {
  final id = stop.stayContactId;
  if (id == null) return null;
  for (final c in diary) {
    if (c.id == id) return c;
  }
  return null;
}

/// What can be chosen at [stopId]: the accommodation numbers saved there,
/// confirmed first.
List<Contact> stayOptions(List<Contact> diary, int stopId) =>
    staysAt(diary, stopId);

/// Records where you sleep at [stopId]. Null clears it back to not decided.
Future<void> setStay(AppDatabase db, int stopId, int? contactId) =>
    (db.update(db.stops)..where((s) => s.id.equals(stopId))).write(
      StopsCompanion(stayContactId: Value(contactId)),
    );

/// Makes a newly typed stay THE stay, when it is the first place to stay the
/// person has saved at a stop they sleep at and nothing is chosen yet.
///
/// Typed, not imported: a sheet of options is exactly what must not choose
/// itself. A person adding "our homestay" to an empty stop has chosen it, and
/// making them say so twice would be busywork. Returns whether it did.
Future<bool> adoptFirstStay(AppDatabase db, int contactId) async {
  final c = await (db.select(
    db.contacts,
  )..where((x) => x.id.equals(contactId))).getSingleOrNull();
  if (c == null || c.stopId == null || c.importBatchId != null) return false;
  if (c.category != ContactCategory.accommodation) return false;

  final stop = await (db.select(
    db.stops,
  )..where((s) => s.id.equals(c.stopId!))).getSingleOrNull();
  if (stop == null || stop.nights <= 0 || stop.stayContactId != null) {
    return false;
  }

  final others = await (db.select(db.contacts)..where(
        (x) =>
            x.stopId.equals(stop.id) &
            x.category.equals(ContactCategory.accommodation) &
            x.id.equals(contactId).not(),
      ))
      .get();
  if (others.isNotEmpty) return false;

  await setStay(db, stop.id, contactId);
  return true;
}
