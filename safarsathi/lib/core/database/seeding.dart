// lib/core/database/seeding.dart
//
// Bootstrap of bundled reference data. MUST BE IDEMPOTENT — it runs again
// after a reinstall and after every migration, and it must not duplicate
// rows or disturb ids.

import 'package:drift/drift.dart';

import '../../features/emergency/data/emergency_seed.dart';
import 'app_database.dart';

/// Writes bundled reference data into the database.
///
/// Safe to call on every launch. Returns the number of rows written, which is
/// the full seed count each time — an upsert that changes nothing still
/// reports the row.
Future<int> seedReferenceData(AppDatabase db) async {
  return db.transaction(() async {
    var written = 0;
    for (final row in emergencySeed) {
      // Flagged numbers never reach the database. Issue #36 verifies them
      // against a .gov.in source and flips the flag; nothing else should.
      if (row.needsVerification) continue;

      await db
          .into(db.emergencyHelplines)
          .insert(
            EmergencyHelplinesCompanion.insert(
              countryCode: row.countryCode,
              serviceType: row.serviceType,
              label: row.label,
              number: row.number,
              sourceNote: row.sourceNote,
              sourceUrl: Value(row.sourceUrl),
            ),
            // Upsert on the NATURAL key, not the primary key.
            //
            // Insert-or-ignore would also be idempotent, but it would strand
            // anyone who already installed the app on an old label or an old
            // sourceNote if a later version corrects one. These are bundled
            // reference data, not user rows, so a correction should reach
            // them. `id` is never touched, so anything referencing a row
            // survives a re-seed.
            onConflict: DoUpdate(
              (_) => EmergencyHelplinesCompanion.custom(
                label: Variable(row.label),
                sourceNote: Variable(row.sourceNote),
                sourceUrl: Variable(row.sourceUrl),
              ),
              target: [
                db.emergencyHelplines.countryCode,
                db.emergencyHelplines.number,
                db.emergencyHelplines.serviceType,
              ],
            ),
          );
      written++;
    }
    return written;
  });
}
