// lib/features/trips/data/planned_stops.dart
//
// Stops you mean to make on the way: the viewpoint above the gorge, the
// waterfall, the dhaba for lunch. Between two towns there is a day of road,
// and a plan that names only the towns tells the people at home nothing
// about where you will be at two in the afternoon.

import 'package:drift/drift.dart';

import '../../../core/database/app_database.dart';
import '../../discovery/data/corridor.dart';
import '../../discovery/data/discovery.dart' show CorridorPlace;
import '../../discovery/data/geo.dart';

/// A planned stop with where it falls on its leg's road, when that can be
/// worked out.
class PlannedOnTheWay {
  final PlannedStop stop;

  /// Kilometres from the start of the leg. Null for a stop with no position,
  /// or a leg with no line to measure along.
  final double? alongRouteKm;

  const PlannedOnTheWay(this.stop, {this.alongRouteKm});
}

/// In road order: placed stops by kilometre, then the rest in the order they
/// were added. A typed stop with no position is still a stop — it goes last
/// rather than being left out.
List<PlannedOnTheWay> orderPlanned(
  List<PlannedStop> rows,
  Corridor? corridor,
) {
  final placed = <PlannedOnTheWay>[];
  final unplaced = <PlannedOnTheWay>[];
  for (final r in rows) {
    final at = r.lat == null || r.lon == null || corridor == null
        ? null
        : corridor.locate(LatLng(r.lat!, r.lon!));
    if (at == null) {
      unplaced.add(PlannedOnTheWay(r));
    } else {
      placed.add(PlannedOnTheWay(r, alongRouteKm: at.alongRouteKm));
    }
  }
  placed.sort((a, b) => a.alongRouteKm!.compareTo(b.alongRouteKm!));
  unplaced.sort((a, b) => a.stop.createdAt.compareTo(b.stop.createdAt));
  return [...placed, ...unplaced];
}

/// Adds a stop typed by name, with a position if the person had one.
Future<int> planStop(
  AppDatabase db, {
  required int tripId,
  required int legId,
  required String name,
  double? lat,
  double? lon,
  String? note,
}) => db.into(db.plannedStops).insert(
  PlannedStopsCompanion.insert(
    tripId: tripId,
    legId: Value(legId),
    name: name.trim(),
    lat: Value(lat),
    lon: Value(lon),
    note: Value(note == null || note.trim().isEmpty ? null : note.trim()),
  ),
);

/// Adds a downloaded place. Once per leg: planning it twice from two taps
/// is not two stops.
Future<void> planPlace(
  AppDatabase db, {
  required int tripId,
  required int legId,
  required CorridorPlace place,
}) async {
  if (place.osmId != null &&
      await isPlacePlanned(db, legId: legId, osmId: place.osmId!)) {
    return;
  }
  await db.into(db.plannedStops).insert(
    PlannedStopsCompanion.insert(
      tripId: tripId,
      legId: Value(legId),
      name: place.name,
      category: Value(place.category),
      lat: Value(place.lat),
      lon: Value(place.lon),
      osmId: Value(place.osmId),
    ),
  );
}

Future<bool> isPlacePlanned(
  AppDatabase db, {
  required int legId,
  required String osmId,
}) async =>
    await (db.select(db.plannedStops)..where(
          (p) => p.legId.equals(legId) & p.osmId.equals(osmId),
        ))
        .getSingleOrNull() !=
    null;

/// Takes a downloaded place back out of the leg's plan.
Future<void> unplanPlace(
  AppDatabase db, {
  required int legId,
  required String osmId,
}) =>
    (db.delete(db.plannedStops)
          ..where((p) => p.legId.equals(legId) & p.osmId.equals(osmId)))
        .go();

Future<void> unplanStop(AppDatabase db, int id) =>
    (db.delete(db.plannedStops)..where((p) => p.id.equals(id))).go();
