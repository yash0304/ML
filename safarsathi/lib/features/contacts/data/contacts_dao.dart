// lib/features/contacts/data/contacts_dao.dart
//
// Drift DAO — the single read/write surface for contact data.
// The diary never touches the database directly; it watches these streams.
//
// OFFLINE: every method here is local SQLite. No network path exists.

import 'package:drift/drift.dart';
import '../../../core/database/app_database.dart';

part 'contacts_dao.g.dart';

/// Trust tier. Drives visual treatment in the diary — see DESIGN.md §4.
/// Stored as a string in the DB so tiers can be added without a migration.
enum ContactTier {
  verifiedNational, // Govt short codes. Authoritative.
  verifiedState, // State govt published.
  userVerified, // User called it and it worked.
  userEntered, // User typed/imported it. Not yet confirmed.
  communityOsm; // From OSM tags. Always unverified.

  static ContactTier parse(String raw) => ContactTier.values.firstWhere(
    (t) => t.name == raw,
    orElse: () => ContactTier.userEntered,
  );

  /// True when the number is known-good and can be shown without a caution mark.
  bool get isTrusted =>
      this == verifiedNational || this == verifiedState || this == userVerified;
}

class ContactCategory {
  static const emergency = 'emergency';
  static const hospital = 'hospital';
  static const pharmacy = 'pharmacy';
  static const accommodation = 'accommodation';
  static const restaurant = 'restaurant';
  static const transport = 'transport';
  static const guide = 'guide';
  static const fuel = 'fuel';
  static const localContact = 'localContact';
  static const embassy = 'embassy';
  static const other = 'other';

  static const all = [
    emergency,
    hospital,
    pharmacy,
    accommodation,
    restaurant,
    transport,
    guide,
    fuel,
    localContact,
    embassy,
    other,
  ];

  /// The order categories are offered in, everywhere the user picks one.
  ///
  /// Not `all`, which is declaration order and leads with hospital. On a road
  /// trip you reach for stay, transport and food far more often, and the
  /// thumb index and the form must agree or the muscle memory breaks.
  /// `emergency` is absent: those live on their own screen.
  static const pickerOrder = [
    accommodation,
    transport,
    restaurant,
    hospital,
    pharmacy,
    fuel,
    guide,
    localContact,
    embassy,
    other,
  ];

  static const labels = <String, String>{
    emergency: 'Emergency',
    hospital: 'Hospital',
    pharmacy: 'Pharmacy',
    accommodation: 'Stay',
    restaurant: 'Food',
    transport: 'Transport',
    guide: 'Guide',
    fuel: 'Fuel',
    localContact: 'Local',
    embassy: 'Embassy',
    other: 'Other',
  };
}

/// Filter state the dialer passes down. Kept as a value object so the
/// screen can rebuild its stream on any change without bespoke plumbing.
class ContactFilter {
  final int? tripId;
  final int? stopId; // non-null => "current stop only" is on
  final String? category; // null => all categories
  final String searchTerm;

  const ContactFilter({
    this.tripId,
    this.stopId,
    this.category,
    this.searchTerm = '',
  });

  ContactFilter copyWith({
    int? tripId,
    int? stopId,
    String? category,
    String? searchTerm,
    bool clearStop = false,
    bool clearCategory = false,
  }) {
    return ContactFilter(
      tripId: tripId ?? this.tripId,
      stopId: clearStop ? null : (stopId ?? this.stopId),
      category: clearCategory ? null : (category ?? this.category),
      searchTerm: searchTerm ?? this.searchTerm,
    );
  }
}

@DriftAccessor(tables: [Contacts, CallLogs, EmergencyHelplines, ImportBatches])
class ContactsDao extends DatabaseAccessor<AppDatabase>
    with _$ContactsDaoMixin {
  ContactsDao(super.db);

  // ---------------------------------------------------------------------
  // READS
  // ---------------------------------------------------------------------

  /// Main diary feed. Ordering is deliberate: pinned first, then the
  /// contacts you've actually confirmed, then everything else by name.
  ///
  /// Deliberately NOT by recency. A diary's page order has to be stable or
  /// the margin numbers shift under you. And `lastCalledAt` is null for
  /// anything never called, which SQLite sorts as smallest — a DESC recency
  /// term would sink every never-called number to the bottom of its group,
  /// which is exactly the category the readiness system wants in front of
  /// you. See DECISIONS.md.
  Stream<List<Contact>> watchContacts(ContactFilter filter) {
    final q = select(contacts)..where((c) => c.isEmergency.equals(false));

    if (filter.tripId != null) {
      q.where((c) => c.tripId.equals(filter.tripId!));
    }
    if (filter.stopId != null) {
      // Stop-scoped view still includes trip-wide contacts (stopId null),
      // because your driver isn't tied to one stop but you still need him.
      q.where((c) => c.stopId.equals(filter.stopId!) | c.stopId.isNull());
    }
    if (filter.category != null) {
      q.where((c) => c.category.equals(filter.category!));
    }
    if (filter.searchTerm.trim().isNotEmpty) {
      final term = '%${filter.searchTerm.trim()}%';
      q.where(
        (c) =>
            c.name.like(term) |
            c.phoneRaw.like(term) |
            c.phoneE164.like(term) |
            c.note.like(term) |
            c.category.like(term),
      );
    }

    q.orderBy([
      (c) => OrderingTerm(expression: c.isPinned, mode: OrderingMode.desc),
      (c) => OrderingTerm(expression: c.callConfirmed, mode: OrderingMode.desc),
      (c) => OrderingTerm(expression: c.name, mode: OrderingMode.asc),
    ]);

    return q.watch();
  }

  /// Bundled emergency numbers for the countries this trip touches.
  /// Separate stream, separate tab — never merged into the main list.
  Stream<List<EmergencyHelpline>> watchEmergencyHelplines(
    List<String> countryCodes,
  ) {
    final q = select(emergencyHelplines)
      ..where((e) => e.countryCode.isIn(countryCodes))
      ..orderBy([(e) => OrderingTerm(expression: e.serviceType)]);
    return q.watch();
  }

  /// Contacts the user marked emergency-relevant for this specific trip
  /// (the homestay owner who can call a local ambulance faster than 108).
  Stream<List<Contact>> watchTripEmergencyContacts(int tripId) {
    final q = select(contacts)
      ..where((c) => c.tripId.equals(tripId) & c.isEmergency.equals(true))
      ..orderBy([(c) => OrderingTerm(expression: c.name)]);
    return q.watch();
  }

  /// Count of unconfirmed contacts — powers the pre-departure readiness badge.
  Stream<int> watchUnconfirmedCount(int tripId) {
    final countExp = contacts.id.count();
    final q = selectOnly(contacts)
      ..addColumns([countExp])
      ..where(
        contacts.tripId.equals(tripId) & contacts.callConfirmed.equals(false),
      );
    return q.map((row) => row.read(countExp) ?? 0).watchSingle();
  }

  Future<Contact?> findByE164(String e164, {int? tripId}) {
    final q = select(contacts)..where((c) => c.phoneE164.equals(e164));
    if (tripId != null) q.where((c) => c.tripId.equals(tripId));
    return q.getSingleOrNull();
  }

  // ---------------------------------------------------------------------
  // WRITES
  // ---------------------------------------------------------------------

  Future<int> insertContact(ContactsCompanion entry) =>
      into(contacts).insert(entry);

  /// Bulk insert for sheet import. Single transaction so a bad file
  /// rolls back whole rather than leaving half a spreadsheet in the DB.
  ///
  /// EVERY ROW LANDS UNCONFIRMED, whatever the caller passes. A number in a
  /// spreadsheet is still an unverified number, and letting import satisfy
  /// the readiness check would hollow out the whole trust system.
  ///
  /// This is enforced here rather than at the import screen on purpose: the
  /// DAO is the single write surface, so a future caller that forgets cannot
  /// get it wrong. Pinned by a test.
  Future<void> insertBatch(
    List<ContactsCompanion> entries,
    ImportBatchesCompanion batch,
  ) async {
    await transaction(() async {
      final batchId = await into(importBatches).insert(batch);
      await this.batch((b) {
        b.insertAll(
          contacts,
          entries
              .map(
                (e) => e.copyWith(
                  importBatchId: Value(batchId),
                  tier: Value(ContactTier.userEntered.name),
                  callConfirmed: const Value(false),
                  confirmedAt: const Value(null),
                ),
              )
              .toList(),
        );
      });
    });
  }

  /// Undo an import wholesale.
  Future<void> rollbackImport(int batchId) async {
    await transaction(() async {
      await (delete(
        contacts,
      )..where((c) => c.importBatchId.equals(batchId))).go();
      await (delete(importBatches)..where((b) => b.id.equals(batchId))).go();
    });
  }

  Future<void> updateContact(Contact contact) =>
      update(contacts).replace(contact);

  Future<void> deleteContact(int id) =>
      (delete(contacts)..where((c) => c.id.equals(id))).go();

  Future<void> togglePin(int id, bool pinned) =>
      (update(contacts)..where((c) => c.id.equals(id))).write(
        ContactsCompanion(isPinned: Value(pinned)),
      );

  /// The confirmation action. This is what clears the pre-departure block.
  Future<void> markConfirmed(int id, {bool confirmed = true}) =>
      (update(contacts)..where((c) => c.id.equals(id))).write(
        ContactsCompanion(
          callConfirmed: Value(confirmed),
          confirmedAt: Value(confirmed ? DateTime.now() : null),
          tier: Value(
            confirmed
                ? ContactTier.userVerified.name
                : ContactTier.userEntered.name,
          ),
        ),
      );

  /// Called on every outbound action, including `copy`.
  ///
  /// Since the dial happens in the Android dialer after a paste, a copy is
  /// the closest thing the app sees to a call. That makes `lastCalledAt`
  /// really "last outbound action of any kind" — the entry screen should
  /// label it that way rather than "Last called".
  Future<void> logCall({
    required int contactId,
    int? tripId,
    required String action,
  }) async {
    await transaction(() async {
      await into(callLogs).insert(
        CallLogsCompanion.insert(
          contactId: contactId,
          tripId: Value(tripId),
          action: action,
        ),
      );
      final row = await (select(
        contacts,
      )..where((c) => c.id.equals(contactId))).getSingleOrNull();
      if (row != null) {
        await (update(contacts)..where((c) => c.id.equals(contactId))).write(
          ContactsCompanion(
            lastCalledAt: Value(DateTime.now()),
            callCount: Value(row.callCount + 1),
          ),
        );
      }
    });
  }
}
