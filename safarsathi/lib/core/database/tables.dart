// lib/core/database/tables.dart
//
// Every table in the app. SQLite is the only source of truth — there is no
// backend and no runtime network call, so if something is not here it does
// not exist.
//
// Reconstructed from PROJECT_RUNDOWN §5, DESIGN.md §2 and SCREENS.md. The
// contacts tables take their column names from contacts_dao.dart, which is
// dropped in unchanged at #5 — if a name here disagrees with that file, that
// file wins.

import 'package:drift/drift.dart';

// ---------------------------------------------------------------------------
// TRIP STRUCTURE
// ---------------------------------------------------------------------------

class Trips extends Table {
  IntColumn get id => integer().autoIncrement()();
  TextColumn get name => text().withLength(min: 1, max: 120)();
  DateTimeColumn get startDate => dateTime().nullable()();
  DateTimeColumn get endDate => dateTime().nullable()();

  /// Base currency for the ledger. Per-expense currency can differ; the rate
  /// is a manual snapshot taken at setup.
  TextColumn get baseCurrency =>
      text().withLength(max: 3).withDefault(const Constant('INR'))();

  BoolColumn get isActive => boolean().withDefault(const Constant(false))();
  DateTimeColumn get createdAt => dateTime().withDefault(currentDateAndTime)();
}

/// A Stop is an OCCURRENCE, not a place.
///
/// The Meghalaya route visits Shillong twice and the South India route visits
/// Ooty and Bangalore twice each. Two Shillong rows is the correct
/// representation, not a bug — each visit has its own dates, activity tags,
/// checklist and contacts.
class Stops extends Table {
  IntColumn get id => integer().autoIncrement()();
  IntColumn get tripId =>
      integer().references(Trips, #id, onDelete: KeyAction.cascade)();
  TextColumn get name => text()();
  IntColumn get sequenceOrder => integer()();

  /// Country lives on the STOP, not the Trip. A German trip crosses into
  /// Austria mid-itinerary, and emergency numbers, currency and language all
  /// switch at that boundary rather than at the trip boundary.
  TextColumn get countryCode => text().withLength(max: 2)();

  DateTimeColumn get arrivalDate => dateTime().nullable()();
  DateTimeColumn get departureDate => dateTime().nullable()();
  IntColumn get nights => integer().withDefault(const Constant(0))();

  /// Comma-separated: trek, caves, rain, homestay. Drives checklist
  /// generation at #29.
  TextColumn get activityTags => text().withDefault(const Constant(''))();

  RealColumn get lat => real().nullable()();
  RealColumn get lon => real().nullable()();
  TextColumn get note => text().nullable()();
}

/// A Leg is a first-class entity, not a pointer between two Stops. The
/// original problem was "what is between A and B", so the connection has to
/// hold data: mode, planned times, route line, corridor width, its own POIs.
class Legs extends Table {
  IntColumn get id => integer().autoIncrement()();
  IntColumn get tripId =>
      integer().references(Trips, #id, onDelete: KeyAction.cascade)();
  IntColumn get fromStopId =>
      integer().references(Stops, #id, onDelete: KeyAction.cascade)();
  IntColumn get toStopId =>
      integer().references(Stops, #id, onDelete: KeyAction.cascade)();
  IntColumn get sequenceOrder => integer()();

  /// Typed by the user. There is no live schedule lookup and there never will
  /// be offline — see DECISIONS.md.
  TextColumn get mode => text().nullable()();
  DateTimeColumn get plannedDeparture => dateTime().nullable()();
  DateTimeColumn get plannedArrival => dateTime().nullable()();
  BoolColumn get isBooked => boolean().withDefault(const Constant(false))();
  TextColumn get note => text().nullable()();

  /// Encoded polyline. The only live routing call in the app, made once at
  /// setup on WiFi.
  TextColumn get routePolyline => text().nullable()();
  RealColumn get distanceKm => real().nullable()();
  RealColumn get corridorKm => real().withDefault(const Constant(3.0))();
  DateTimeColumn get lastSyncedAt => dateTime().nullable()();
}

// ---------------------------------------------------------------------------
// CACHED PLACES
// ---------------------------------------------------------------------------

/// A POI attaches to a Stop XOR a Leg.
///
/// Stop-attached answers "what is in Kongthong" — small radius, dense.
/// Leg-attached answers "what is on the road to Sohra" — corridor buffer,
/// sparse, ordered by distance along the route. Merging them makes both
/// queries wrong, so the exclusivity is a database constraint rather than a
/// convention someone can forget.
class Pois extends Table {
  IntColumn get id => integer().autoIncrement()();
  IntColumn get tripId =>
      integer().references(Trips, #id, onDelete: KeyAction.cascade)();
  IntColumn get stopId => integer().nullable().references(
    Stops,
    #id,
    onDelete: KeyAction.cascade,
  )();
  IntColumn get legId =>
      integer().nullable().references(Legs, #id, onDelete: KeyAction.cascade)();

  TextColumn get name => text()();
  TextColumn get category => text()();
  RealColumn get lat => real()();
  RealColumn get lon => real()();

  /// How far along the leg this sits, and how far off the line. Powers the
  /// "coming up in 12 km" ordering on the leg screen.
  RealColumn get distanceAlongRouteKm => real().nullable()();
  RealColumn get distanceOffRouteKm => real().nullable()();

  TextColumn get osmId => text().nullable()();
  TextColumn get rawTags => text().nullable()();
  DateTimeColumn get cachedAt => dateTime().withDefault(currentDateAndTime)();

  // Drift reads this statically, so it must be a const list literal.
  @override
  List<String> get customConstraints => const [
    'CHECK ((stop_id IS NULL) <> (leg_id IS NULL))',
  ];
}

/// Phone numbers found on map tags. Always unverified, with no path in the
/// app that promotes one without the user saving it as a contact and calling
/// it themselves.
class PoiContacts extends Table {
  IntColumn get id => integer().autoIncrement()();
  IntColumn get poiId =>
      integer().references(Pois, #id, onDelete: KeyAction.cascade)();
  TextColumn get phoneRaw => text()();
  TextColumn get phoneE164 => text().nullable()();
  TextColumn get tier => text().withDefault(const Constant('communityOsm'))();

  /// Which OSM key it came from: `phone` or `contact:phone`.
  TextColumn get sourceTag => text().nullable()();
}

// ---------------------------------------------------------------------------
// THE DIARY
// ---------------------------------------------------------------------------

/// One sheet import, so a bad file rolls back whole rather than leaving half
/// a spreadsheet in the database.
class ImportBatches extends Table {
  IntColumn get id => integer().autoIncrement()();
  IntColumn get tripId => integer().nullable().references(
    Trips,
    #id,
    onDelete: KeyAction.cascade,
  )();
  TextColumn get fileName => text()();
  TextColumn get sheetName => text().nullable()();
  IntColumn get rowsImported => integer().withDefault(const Constant(0))();
  IntColumn get rowsSkipped => integer().withDefault(const Constant(0))();
  DateTimeColumn get importedAt => dateTime().withDefault(currentDateAndTime)();
}

class Contacts extends Table {
  IntColumn get id => integer().autoIncrement()();
  IntColumn get tripId => integer().nullable().references(
    Trips,
    #id,
    onDelete: KeyAction.cascade,
  )();
  IntColumn get stopId => integer().nullable().references(
    Stops,
    #id,
    onDelete: KeyAction.setNull,
  )();

  TextColumn get name => text()();

  /// What the user typed, shown back verbatim. Reformatting someone's own
  /// input is confusing, so the raw value is what the diary displays.
  TextColumn get phoneRaw => text()();

  /// Normalised, for dialing and duplicate detection. Nullable because
  /// normalisation genuinely fails on bad input and the raw value must
  /// survive that.
  TextColumn get phoneE164 => text().nullable()();

  TextColumn get note => text().nullable()();
  TextColumn get category => text().withDefault(const Constant('other'))();

  /// THE CORE INVARIANT. Defaults to userEntered. Nothing may reach this
  /// table already verified — not an import, not a POI save, not a form.
  TextColumn get tier => text().withDefault(const Constant('userEntered'))();
  BoolColumn get callConfirmed =>
      boolean().withDefault(const Constant(false))();
  DateTimeColumn get confirmedAt => dateTime().nullable()();

  BoolColumn get isPinned => boolean().withDefault(const Constant(false))();
  BoolColumn get isEmergency => boolean().withDefault(const Constant(false))();
  BoolColumn get hasWhatsapp => boolean().withDefault(const Constant(false))();

  DateTimeColumn get lastCalledAt => dateTime().nullable()();
  IntColumn get callCount => integer().withDefault(const Constant(0))();

  IntColumn get importBatchId => integer().nullable().references(
    ImportBatches,
    #id,
    onDelete: KeyAction.setNull,
  )();
  DateTimeColumn get createdAt => dateTime().withDefault(currentDateAndTime)();

  /// Where the place is, when the source said (v5).
  ///
  /// Lets a number be put on a leg at its real distance along the road —
  /// "Pynursla hospital, 38 km" — rather than only under a stop. Nullable
  /// and never guessed: a contact typed by hand has no position, and one
  /// invented from its stop's coordinates would sit at the wrong kilometre
  /// on every leg it touched.
  RealColumn get lat => real().nullable()();
  RealColumn get lon => real().nullable()();
}

/// Every outbound action, including `copy`.
///
/// Copy is first-class because the dial now happens in the Android dialer
/// after a paste. Without logging it, recents ordering and the record of who
/// was actually reached would rot the moment the workflow changed.
class CallLogs extends Table {
  IntColumn get id => integer().autoIncrement()();
  IntColumn get contactId =>
      integer().references(Contacts, #id, onDelete: KeyAction.cascade)();
  IntColumn get tripId => integer().nullable().references(
    Trips,
    #id,
    onDelete: KeyAction.cascade,
  )();

  /// copy | call | dialer | sms | whatsapp
  TextColumn get action => text()();
  DateTimeColumn get occurredAt => dateTime().withDefault(currentDateAndTime)();
}

/// Bundled reference data, not trip-scoped.
///
/// The unique key is what makes first-launch seeding idempotent. Seeding runs
/// again after a reinstall or a migration and must not duplicate rows.
class EmergencyHelplines extends Table {
  IntColumn get id => integer().autoIncrement()();
  TextColumn get countryCode => text().withLength(max: 2)();

  /// State or union territory. Null means national.
  TextColumn get regionCode => text().nullable()();

  TextColumn get serviceType => text()();
  TextColumn get label => text()();
  TextColumn get number => text()();

  /// Shown in the UI under every bundled number, always. The user can see
  /// where it came from and judge for themselves.
  TextColumn get sourceNote => text()();
  TextColumn get sourceUrl => text().nullable()();

  /// Flagged numbers do not reach the UI. 1930, 1078, 1033 and 104 are
  /// widely cited but were not confirmed from a .gov.in source.
  BoolColumn get needsVerification =>
      boolean().withDefault(const Constant(false))();

  TextColumn get tier =>
      text().withDefault(const Constant('verifiedNational'))();

  @override
  List<Set<Column<Object>>> get uniqueKeys => [
    {countryCode, number, serviceType},
  ];
}

// ---------------------------------------------------------------------------
// CHECKLIST
// ---------------------------------------------------------------------------

/// Pack items and blocking readiness items live in ONE list.
///
/// "Call and confirm the Kongthong homestay number" belongs beside "pack
/// leech socks" because both are things that must happen before leaving
/// signal. This is where the trust system surfaces for the user.
class ChecklistItems extends Table {
  IntColumn get id => integer().autoIncrement()();
  IntColumn get tripId =>
      integer().references(Trips, #id, onDelete: KeyAction.cascade)();
  IntColumn get stopId => integer().nullable().references(
    Stops,
    #id,
    onDelete: KeyAction.cascade,
  )();

  TextColumn get label => text()();
  TextColumn get quantity => text().nullable()();

  /// The tags that produced this item, shown under it. A generated list
  /// nobody understands gets ignored.
  TextColumn get sourceTags => text().withDefault(const Constant(''))();

  BoolColumn get isDone => boolean().withDefault(const Constant(false))();

  /// The trip does not read ready while any blocking item is open.
  BoolColumn get isBlocking => boolean().withDefault(const Constant(false))();

  /// Set on blocking items generated from an unconfirmed number.
  IntColumn get contactId => integer().nullable().references(
    Contacts,
    #id,
    onDelete: KeyAction.cascade,
  )();

  BoolColumn get isGenerated => boolean().withDefault(const Constant(true))();

  /// The generator rule that produced this item, stable across renames.
  ///
  /// Matching a generated item by its LABEL looked fine until someone renamed
  /// one: the generator then found no row for its rule and inserted a second
  /// copy alongside the user's. Null for items the user wrote themselves and
  /// for blocking items, which are keyed by stop.
  TextColumn get generatorKey => text().nullable()();

  /// Set the moment a user edits a generated item, so regeneration cannot
  /// silently discard their change.
  BoolColumn get isUserEdited => boolean().withDefault(const Constant(false))();

  IntColumn get sortOrder => integer().withDefault(const Constant(0))();
}

// ---------------------------------------------------------------------------
// WEATHER
// ---------------------------------------------------------------------------

class WeatherSnapshots extends Table {
  IntColumn get id => integer().autoIncrement()();
  IntColumn get stopId =>
      integer().references(Stops, #id, onDelete: KeyAction.cascade)();
  DateTimeColumn get forDate => dateTime()();
  TextColumn get condition => text()();
  RealColumn get tempMinC => real().nullable()();
  RealColumn get tempMaxC => real().nullable()();
  RealColumn get rainMm => real().nullable()();

  /// NOT nullable, deliberately. A snapshot without an age is a forecast
  /// pretending to be current, which is the exact failure this table exists
  /// to prevent.
  DateTimeColumn get cachedAt => dateTime()();

  @override
  List<Set<Column<Object>>> get uniqueKeys => [
    {stopId, forDate},
  ];
}

// ---------------------------------------------------------------------------
// MONEY
// ---------------------------------------------------------------------------

/// Named people on the trip. No accounts, no sync, no server.
///
/// Not in the original data model — the expense feature was described before
/// it was designed. Splits have to point at a person, and a free-text name
/// per split would make balances unjoinable.
class Travellers extends Table {
  IntColumn get id => integer().autoIncrement()();
  IntColumn get tripId =>
      integer().references(Trips, #id, onDelete: KeyAction.cascade)();
  TextColumn get name => text()();
  BoolColumn get isSelf => boolean().withDefault(const Constant(false))();
}

class Expenses extends Table {
  IntColumn get id => integer().autoIncrement()();
  IntColumn get tripId =>
      integer().references(Trips, #id, onDelete: KeyAction.cascade)();
  IntColumn get stopId => integer().nullable().references(
    Stops,
    #id,
    onDelete: KeyAction.setNull,
  )();

  TextColumn get description => text()();

  /// MINOR UNITS, as an integer. Paise, not rupees. Floating point
  /// accumulates rounding error across a three-way split and this ledger has
  /// to balance exactly.
  IntColumn get amountMinor => integer()();

  TextColumn get currency =>
      text().withLength(max: 3).withDefault(const Constant('INR'))();

  /// Manual snapshot taken at setup. There is no live rate offline, and the
  /// UI always shows this alongside its capture date.
  RealColumn get rateToBase => real().withDefault(const Constant(1.0))();
  DateTimeColumn get rateCapturedAt => dateTime().nullable()();

  IntColumn get paidById =>
      integer().references(Travellers, #id, onDelete: KeyAction.cascade)();
  TextColumn get category => text().nullable()();
  DateTimeColumn get spentAt => dateTime().withDefault(currentDateAndTime)();
}

class ExpenseSplits extends Table {
  IntColumn get id => integer().autoIncrement()();
  IntColumn get expenseId =>
      integer().references(Expenses, #id, onDelete: KeyAction.cascade)();
  IntColumn get travellerId =>
      integer().references(Travellers, #id, onDelete: KeyAction.cascade)();

  /// Minor units again. The shares of one expense must sum to its
  /// `amountMinor` exactly — #32 asserts this.
  IntColumn get shareMinor => integer()();

  @override
  List<Set<Column<Object>>> get uniqueKeys => [
    {expenseId, travellerId},
  ];
}

// ---------------------------------------------------------------------------
// TIMELINE AND CHECK-IN
// ---------------------------------------------------------------------------

class TimelineEntries extends Table {
  IntColumn get id => integer().autoIncrement()();
  IntColumn get tripId =>
      integer().references(Trips, #id, onDelete: KeyAction.cascade)();
  IntColumn get stopId => integer().nullable().references(
    Stops,
    #id,
    onDelete: KeyAction.setNull,
  )();

  /// arrival | departure | note | photo | fix
  TextColumn get kind => text()();
  TextColumn get title => text().nullable()();
  TextColumn get body => text().nullable()();

  RealColumn get lat => real().nullable()();
  RealColumn get lon => real().nullable()();
  RealColumn get accuracyM => real().nullable()();

  /// Newline-separated local file paths. Photos stay on the device.
  TextColumn get photoPaths => text().nullable()();

  DateTimeColumn get occurredAt => dateTime()();
}

/// Check-in recipients. Delivery is by native SMS intent, not a server relay,
/// so it works with zero data and only cell signal.
class TrustedContacts extends Table {
  IntColumn get id => integer().autoIncrement()();
  IntColumn get tripId => integer().nullable().references(
    Trips,
    #id,
    onDelete: KeyAction.cascade,
  )();
  TextColumn get name => text()();
  TextColumn get phoneE164 => text()();
  BoolColumn get notifyOnArrival =>
      boolean().withDefault(const Constant(true))();
  BoolColumn get escalate => boolean().withDefault(const Constant(true))();
  IntColumn get escalateAfterMinutes =>
      integer().withDefault(const Constant(120))();
}

/// Key-value settings — issue #35.
///
/// In the database rather than shared preferences, so there is still exactly
/// one place this app keeps state and exactly one thing to back up. There are
/// three keys and every one of them is a thing that can be wrong.
class AppSettings extends Table {
  TextColumn get key => text()();
  TextColumn get value => text()();

  @override
  Set<Column<Object>> get primaryKey => {key};
}

/// What map tiles are on this phone — issue #24.
///
/// The tiles themselves are files under the app's documents directory; this
/// indexes them so the cache screen can state a real number rather than
/// walking a directory tree, and so a re-download knows what to skip.
///
/// Deliberately NOT flutter_map_tile_caching, which stores tiles in ObjectBox
/// — a second native database engine beside SQLite. A cache is not worth
/// another build to break and another migration story. See DECISIONS.md.
class MapTiles extends Table {
  IntColumn get id => integer().autoIncrement()();

  /// Which provider served it. Two providers' tiles must never be mistaken
  /// for each other, so this is part of the identity and part of the path.
  TextColumn get provider => text()();

  IntColumn get z => integer()();
  IntColumn get x => integer()();
  IntColumn get y => integer()();

  IntColumn get bytes => integer()();
  DateTimeColumn get fetchedAt => dateTime().withDefault(currentDateAndTime)();

  @override
  List<Set<Column<Object>>> get uniqueKeys => [
    {provider, z, x, y},
  ];
}
