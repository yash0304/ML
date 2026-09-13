// lib/features/trips/data/trip_editor.dart
//
// Creating and editing trips and stops — issue #16, and the trip-scoping half
// of #19.
//
// THE RULE THAT SHAPES EVERYTHING HERE: a place may appear twice. The
// Meghalaya itinerary is Shillong → Cherrapunji → Shillong → Dawki, and the
// two Shillong rows are different stops with different dates, different
// contacts and different nights. NOTHING may key a stop by its name.

import 'package:drift/drift.dart';

import '../../../core/database/app_database.dart';
import 'leg_generator.dart';

/// A stop as the editor screens pass it around.
class StopDraft {
  final int? id;
  final String name;
  final String countryCode;
  final DateTime? arrivalDate;
  final DateTime? departureDate;
  final int nights;
  final List<String> activityTags;
  final String? note;

  /// Where the place actually is. Null until someone looks it up or types it.
  ///
  /// The corridor cannot route a leg whose ends have no coordinates, so this
  /// is the field that decides whether offline maps work on a real trip.
  final double? lat;
  final double? lon;

  const StopDraft({
    required this.name,
    this.id,
    this.countryCode = 'IN',
    this.arrivalDate,
    this.departureDate,
    this.nights = 0,
    this.activityTags = const [],
    this.note,
    this.lat,
    this.lon,
  });

  bool get hasCoordinates => lat != null && lon != null;

  /// A stop the user sleeps at. This is what the readiness check at #20 keys
  /// off: a lunch stop with no number is an inconvenience, a homestay with no
  /// number is a night in the car.
  bool get isOvernight => effectiveNights > 0;

  /// Derived from the dates when both are set, typed otherwise.
  ///
  /// Derivation wins because two dates and a contradicting night count is a
  /// question with an obvious answer, and asking the user to keep them in
  /// sync by hand is asking them to maintain the app's data model.
  int get effectiveNights {
    final a = arrivalDate;
    final d = departureDate;
    if (a == null || d == null) return nights;
    final span = DateTime(
      d.year,
      d.month,
      d.day,
    ).difference(DateTime(a.year, a.month, a.day)).inDays;
    return span < 0 ? 0 : span;
  }

  StopDraft copyWith({
    String? name,
    String? countryCode,
    Object? arrivalDate = _unset,
    Object? departureDate = _unset,
    int? nights,
    List<String>? activityTags,
    Object? note = _unset,
    Object? lat = _unset,
    Object? lon = _unset,
  }) => StopDraft(
    id: id,
    name: name ?? this.name,
    countryCode: countryCode ?? this.countryCode,
    lat: lat == _unset ? this.lat : lat as double?,
    lon: lon == _unset ? this.lon : lon as double?,
    arrivalDate: arrivalDate == _unset
        ? this.arrivalDate
        : arrivalDate as DateTime?,
    departureDate: departureDate == _unset
        ? this.departureDate
        : departureDate as DateTime?,
    nights: nights ?? this.nights,
    activityTags: activityTags ?? this.activityTags,
    note: note == _unset ? this.note : note as String?,
  );

  static StopDraft fromRow(Stop s) => StopDraft(
    id: s.id,
    name: s.name,
    countryCode: s.countryCode,
    arrivalDate: s.arrivalDate,
    departureDate: s.departureDate,
    nights: s.nights,
    activityTags: parseTags(s.activityTags),
    note: s.note,
    lat: s.lat,
    lon: s.lon,
  );
}

const _unset = Object();

/// Tags are stored comma-separated, which is a deliberate simplification: a
/// join table for four words per stop is machinery with no payoff. Parsing
/// tolerates spacing and empty entries so a user typing "trek, caves," gets
/// what they meant.
List<String> parseTags(String raw) => [
  for (final t in raw.split(','))
    if (t.trim().isNotEmpty) t.trim(),
];

String encodeTags(List<String> tags) =>
    tags.map((t) => t.trim()).where((t) => t.isNotEmpty).join(',');

/// The tags offered in the editor. Free text is allowed too; these are the
/// ones the checklist generator at #29 knows how to act on.
const knownActivityTags = [
  'trek',
  'caves',
  'rain',
  'homestay',
  'camping',
  'beach',
  'cold',
  'city',
  'drive',
];

class TripEditor {
  final AppDatabase db;
  const TripEditor(this.db);

  // -- trips ---------------------------------------------------------------

  Stream<List<Trip>> watchTrips() =>
      (db.select(db.trips)..orderBy([
            (t) => OrderingTerm(
              expression: t.createdAt,
              mode: OrderingMode.desc,
            ),
          ]))
          .watch();

  Stream<Trip?> watchActiveTrip() =>
      (db.select(db.trips)..where((t) => t.isActive.equals(true)))
          .watchSingleOrNull();

  Future<int> createTrip({
    required String name,
    DateTime? startDate,
    DateTime? endDate,
    bool makeActive = true,
  }) async {
    return db.transaction(() async {
      final id = await db
          .into(db.trips)
          .insert(
            TripsCompanion.insert(
              name: name,
              startDate: Value(startDate),
              endDate: Value(endDate),
            ),
          );
      if (makeActive) await _activate(id);
      return id;
    });
  }

  Future<void> updateTrip(
    int tripId, {
    required String name,
    DateTime? startDate,
    DateTime? endDate,
  }) => (db.update(db.trips)..where((t) => t.id.equals(tripId))).write(
    TripsCompanion(
      name: Value(name),
      startDate: Value(startDate),
      endDate: Value(endDate),
    ),
  );

  /// EXACTLY ONE TRIP IS ACTIVE. The app opens on it, the diary scopes to it,
  /// the emergency screen reads its stops. Two active trips would make every
  /// one of those ambiguous, so activation is a transaction that clears the
  /// rest rather than a flag anyone can set.
  Future<void> setActiveTrip(int tripId) =>
      db.transaction(() => _activate(tripId));

  Future<void> _activate(int tripId) async {
    await db.update(db.trips).write(const TripsCompanion(isActive: Value(false)));
    await (db.update(db.trips)..where((t) => t.id.equals(tripId))).write(
      const TripsCompanion(isActive: Value(true)),
    );
  }

  /// Deleting a trip cascades to everything under it. The caller confirms.
  Future<void> deleteTrip(int tripId) async {
    await db.transaction(() async {
      final wasActive = await (db.select(
        db.trips,
      )..where((t) => t.id.equals(tripId))).getSingleOrNull();
      await (db.delete(db.trips)..where((t) => t.id.equals(tripId))).go();

      // Leaving no trip active would open the app on nothing. Promote the
      // most recent survivor instead.
      if (wasActive?.isActive ?? false) {
        final next =
            await (db.select(db.trips)
                  ..orderBy([
                    (t) => OrderingTerm(
                      expression: t.createdAt,
                      mode: OrderingMode.desc,
                    ),
                  ])
                  ..limit(1))
                .getSingleOrNull();
        if (next != null) await _activate(next.id);
      }
    });
  }

  // -- stops ---------------------------------------------------------------

  Stream<List<Stop>> watchStops(int tripId) =>
      (db.select(db.stops)
            ..where((s) => s.tripId.equals(tripId))
            ..orderBy([(s) => OrderingTerm(expression: s.sequenceOrder)]))
          .watch();

  Future<List<Stop>> stopsOf(int tripId) =>
      (db.select(db.stops)
            ..where((s) => s.tripId.equals(tripId))
            ..orderBy([(s) => OrderingTerm(expression: s.sequenceOrder)]))
          .get();

  /// Appends a stop and regenerates legs.
  ///
  /// Note there is no check for a duplicate name, and there must not be:
  /// Shillong appearing twice is a correct itinerary, not a mistake to catch.
  Future<int> addStop(int tripId, StopDraft draft) async {
    return db.transaction(() async {
      final existing = await stopsOf(tripId);
      final id = await db
          .into(db.stops)
          .insert(_companion(tripId, draft, existing.length + 1));
      await regenerateLegs(db, tripId);
      return id;
    });
  }

  Future<void> updateStop(int tripId, StopDraft draft) async {
    final id = draft.id;
    if (id == null) throw ArgumentError('updateStop needs a saved stop');
    await (db.update(db.stops)..where((s) => s.id.equals(id))).write(
      StopsCompanion(
        name: Value(draft.name),
        countryCode: Value(draft.countryCode),
        arrivalDate: Value(draft.arrivalDate),
        departureDate: Value(draft.departureDate),
        nights: Value(draft.effectiveNights),
        activityTags: Value(encodeTags(draft.activityTags)),
        note: Value(draft.note),
        lat: Value(draft.lat),
        lon: Value(draft.lon),
      ),
    );
  }

  /// How many contacts would be left unattached by deleting this stop.
  ///
  /// The confirmation names this number. A contact is NOT deleted with its
  /// stop — the schema sets its stopId to null — because a number you have
  /// dialled and confirmed does not stop being a real number because you
  /// dropped the stop from the plan.
  Future<int> contactsAt(int stopId) => (db.selectOnly(db.contacts)
        ..addColumns([db.contacts.id.count()])
        ..where(db.contacts.stopId.equals(stopId)))
      .map((r) => r.read(db.contacts.id.count()) ?? 0)
      .getSingle();

  Future<void> deleteStop(int tripId, int stopId) async {
    await db.transaction(() async {
      await (db.delete(db.stops)..where((s) => s.id.equals(stopId))).go();
      await _renumber(tripId);
      await regenerateLegs(db, tripId);
    });
  }

  /// Moves the stop at [from] to [to], as a drag-to-reorder list reports it.
  Future<void> reorderStops(int tripId, int from, int to) async {
    await db.transaction(() async {
      final stops = await stopsOf(tripId);
      if (from < 0 || from >= stops.length) return;
      final moved = stops.removeAt(from);
      stops.insert(to.clamp(0, stops.length), moved);

      for (var i = 0; i < stops.length; i++) {
        await (db.update(db.stops)..where((s) => s.id.equals(stops[i].id)))
            .write(StopsCompanion(sequenceOrder: Value(i + 1)));
      }
      await regenerateLegs(db, tripId);
    });
  }

  /// Renumbers densely — 1, 2, 3 — after any change.
  ///
  /// A sparse ordering works right up until two stops share a number, and then
  /// the itinerary scrambles in a way that is very hard to read back from the
  /// screen.
  Future<void> _renumber(int tripId) async {
    final stops = await stopsOf(tripId);
    for (var i = 0; i < stops.length; i++) {
      if (stops[i].sequenceOrder == i + 1) continue;
      await (db.update(db.stops)..where((s) => s.id.equals(stops[i].id)))
          .write(StopsCompanion(sequenceOrder: Value(i + 1)));
    }
  }

  StopsCompanion _companion(int tripId, StopDraft d, int order) =>
      StopsCompanion.insert(
        tripId: tripId,
        name: d.name,
        sequenceOrder: order,
        countryCode: d.countryCode,
        arrivalDate: Value(d.arrivalDate),
        departureDate: Value(d.departureDate),
        nights: Value(d.effectiveNights),
        activityTags: Value(encodeTags(d.activityTags)),
        note: Value(d.note),
        lat: Value(d.lat),
        lon: Value(d.lon),
      );

  // -- legs ----------------------------------------------------------------

  Future<void> updateLeg(
    int legId, {
    String? mode,
    DateTime? plannedDeparture,
    DateTime? plannedArrival,
    required bool isBooked,
    String? note,
  }) => (db.update(db.legs)..where((l) => l.id.equals(legId))).write(
    LegsCompanion(
      mode: Value(mode),
      plannedDeparture: Value(plannedDeparture),
      plannedArrival: Value(plannedArrival),
      isBooked: Value(isBooked),
      note: Value(note),
    ),
  );
}

/// Which stop the app considers "here", derived from today's date.
///
/// THE FALLBACK IS A STOP, NEVER NULL. A trip whose dates were never filled in
/// is exactly the trip planned in a hurry, and returning null there would make
/// the diary's stop-scope toggle vanish on the itineraries most likely to need
/// it. Before the trip: the first stop. After it, or with no dates at all: the
/// last one reached.
Stop? currentStopOf(List<Stop> stops, {DateTime? now}) {
  if (stops.isEmpty) return null;
  final today = _dayOf(now ?? DateTime.now());

  for (final s in stops) {
    final a = s.arrivalDate;
    if (a == null) continue;
    final from = _dayOf(a);
    final to = _dayOf(s.departureDate ?? a);
    if (!today.isBefore(from) && !today.isAfter(to)) return s;
  }

  final dated = [for (final s in stops) if (s.arrivalDate != null) s];
  if (dated.isEmpty) return stops.first;

  if (today.isBefore(_dayOf(dated.first.arrivalDate!))) return stops.first;

  // Past every arrival date, or in a gap between stops: the last one whose
  // arrival has already happened.
  Stop? reached;
  for (final s in dated) {
    if (!today.isBefore(_dayOf(s.arrivalDate!))) reached = s;
  }
  return reached ?? stops.first;
}

DateTime _dayOf(DateTime d) => DateTime(d.year, d.month, d.day);

/// The trip the app is on, plus where it thinks you are — issue #19.
///
/// This replaces `DemoTrip`, which was a placeholder holding a hardcoded
/// current stop. Everything downstream (the diary's stop scope, the trip
/// screen's milestone, the emergency screen's place label) reads from here.
class ActiveTripContext {
  final int tripId;
  final String name;
  final int? currentStopId;
  final String? currentStopName;
  final String countryCode;

  const ActiveTripContext({
    required this.tripId,
    required this.name,
    this.currentStopId,
    this.currentStopName,
    this.countryCode = 'IN',
  });
}

/// Live context for whichever trip is active, or null when there is no trip.
///
/// Watches stops as well as trips, so adding a stop or changing its dates
/// moves the current stop without a restart.
Stream<ActiveTripContext?> watchActiveTripContext(AppDatabase db) {
  final tick = db
      .customSelect('SELECT 1', readsFrom: {db.trips, db.stops})
      .watch();

  return tick.asyncMap((_) async {
    final trip = await (db.select(
      db.trips,
    )..where((t) => t.isActive.equals(true))).getSingleOrNull();
    if (trip == null) return null;

    final stops =
        await (db.select(db.stops)
              ..where((s) => s.tripId.equals(trip.id))
              ..orderBy([(s) => OrderingTerm(expression: s.sequenceOrder)]))
            .get();

    final here = currentStopOf(stops);
    return ActiveTripContext(
      tripId: trip.id,
      name: trip.name,
      currentStopId: here?.id,
      currentStopName: here?.name,
      // Country lives on the stop, not the trip: a European itinerary crosses
      // a border mid-trip and the emergency numbers change with it.
      countryCode: here?.countryCode ?? 'IN',
    );
  });
}

/// Makes sure something is active before the app opens.
///
/// A database can hold trips with none flagged active — every trip predating
/// #16 is in exactly that state, and so is the survivor of a delete that
/// raced. Promoting the newest is better than opening on nothing.
Future<void> ensureActiveTrip(AppDatabase db) async {
  final active = await (db.select(
    db.trips,
  )..where((t) => t.isActive.equals(true))).getSingleOrNull();
  if (active != null) return;

  final newest =
      await (db.select(db.trips)
            ..orderBy([
              (t) => OrderingTerm(
                expression: t.createdAt,
                mode: OrderingMode.desc,
              ),
            ])
            ..limit(1))
          .getSingleOrNull();
  if (newest != null) await TripEditor(db).setActiveTrip(newest.id);
}
