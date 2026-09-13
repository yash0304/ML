import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show Clipboard, ClipboardData;
import 'package:drift/drift.dart' show OrderingTerm;
import 'package:phone_numbers_parser/phone_numbers_parser.dart' show IsoCode;

import 'core/database/app_database.dart';
import 'core/theme/app_tokens.dart';
import 'core/theme/motion.dart';
import 'core/widgets/app_shell.dart';
import 'features/contacts/data/contact_actions.dart';
import 'features/contacts/data/entry_draft.dart';
import 'features/contacts/presentation/diary_screen.dart';
import 'features/contacts/presentation/entry_form_screen.dart';
import 'features/contacts/presentation/entry_screen.dart';
import 'features/dev/dev_seed.dart';
import 'features/emergency/presentation/emergency_screen.dart';
import 'features/import/data/import_commit.dart';
import 'features/import/presentation/import_flow.dart';
import 'features/import/presentation/import_history_screen.dart';
import 'features/import/presentation/more_screen.dart';
import 'features/money/data/money_summary.dart';
import 'features/money/presentation/money_screen.dart';
import 'features/trips/data/readiness.dart';
import 'features/trips/data/trip_editor.dart';
import 'features/trips/data/trip_summary.dart';
import 'features/trips/presentation/itinerary_screen.dart';
import 'features/trips/presentation/leg_form_screen.dart';
import 'features/trips/presentation/leg_list_screen.dart';
import 'features/trips/presentation/stop_form_screen.dart';
import 'features/trips/presentation/trip_form_screen.dart';
import 'features/trips/presentation/trip_list_screen.dart';
import 'features/trips/presentation/trip_screen.dart';

class SafarSathiApp extends StatelessWidget {
  /// Passed down rather than reached for globally. There is no repository
  /// layer and no service locator yet — DAO to widget, until a second
  /// consumer of the same data appears.
  final AppDatabase db;

  const SafarSathiApp({super.key, required this.db});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'SafarSathi',
      debugShowCheckedModeBanner: false,
      theme: AppTokens.light,
      darkTheme: AppTokens.dark,
      // Follows the system for now. A manual override lands in Settings at
      // backlog #35 — a phone in a pocket does not know it is night in a
      // valley.
      themeMode: ThemeMode.system,
      home: _Home(db: db),
    );
  }
}

/// Opens on whichever trip is active, and follows it live.
///
/// Before #16 this held a `DemoTrip` fetched once, with a hardcoded current
/// stop. Now the trip, its stops and the stop the app thinks you are at all
/// come off a stream, so creating a trip or dragging a stop updates every tab
/// without a restart.
class _Home extends StatefulWidget {
  final AppDatabase db;
  const _Home({required this.db});

  @override
  State<_Home> createState() => _HomeState();
}

class _HomeState extends State<_Home> {
  late final TripEditor _editor = TripEditor(widget.db);
  late final Future<void> _ready = _bootstrap();

  Future<void> _bootstrap() async {
    // A debug build seeds a demo so there is something to look at; a release
    // build opens on the "no trip" screen and offers to make one.
    await ensureDemoTrip(widget.db);
    await ensureActiveTrip(widget.db);
  }

  // -- trips ---------------------------------------------------------------

  Future<void> _openTrips(BuildContext context) => Navigator.of(context).push(
    MaterialPageRoute<void>(
      builder: (_) => TripListScreen(
        trips: _editor.watchTrips(),
        onActivate: _editor.setActiveTrip,
        onCreate: () => _openTripForm(context),
        onDelete: (trip) => _editor.deleteTrip(trip.id),
        onOpen: (trip) async {
          await _editor.setActiveTrip(trip.id);
          if (context.mounted) _openItinerary(context, trip.id, trip.name);
        },
      ),
    ),
  );

  Future<void> _openTripForm(BuildContext context, {Trip? existing}) =>
      Navigator.of(context).push(
        MaterialPageRoute<void>(
          builder: (formContext) => TripFormScreen(
            initialName: existing?.name,
            initialStart: existing?.startDate,
            initialEnd: existing?.endDate,
            onSave: (name, start, end) async {
              if (existing == null) {
                await _editor.createTrip(
                  name: name,
                  startDate: start,
                  endDate: end,
                );
              } else {
                await _editor.updateTrip(
                  existing.id,
                  name: name,
                  startDate: start,
                  endDate: end,
                );
              }
              if (formContext.mounted) Navigator.of(formContext).pop();
            },
          ),
        ),
      );

  // -- stops ---------------------------------------------------------------

  Future<void> _openItinerary(
    BuildContext context,
    int tripId,
    String tripName,
  ) => Navigator.of(context).push(
    MaterialPageRoute<void>(
      builder: (itineraryContext) => ItineraryScreen(
        tripName: tripName,
        stops: _editor.watchStops(tripId),
        onReorder: (from, to) => _editor.reorderStops(tripId, from, to),
        onAdd: () => _openStopForm(itineraryContext, tripId),
        onEdit: (stop) => _openStopForm(itineraryContext, tripId, stop: stop),
        onEditTrip: () async {
          final trip = await (widget.db.select(
            widget.db.trips,
          )..where((t) => t.id.equals(tripId))).getSingleOrNull();
          if (trip != null && itineraryContext.mounted) {
            await _openTripForm(itineraryContext, existing: trip);
          }
        },
      ),
    ),
  );

  Future<void> _openStopForm(
    BuildContext context,
    int tripId, {
    Stop? stop,
  }) => Navigator.of(context).push(
    MaterialPageRoute<void>(
      builder: (formContext) => StopFormScreen(
        existing: stop == null ? null : StopDraft.fromRow(stop),
        contactsHere: stop == null ? null : () => _editor.contactsAt(stop.id),
        onDelete: stop == null
            ? null
            : () async {
                await _editor.deleteStop(tripId, stop.id);
                await syncReadinessChecklist(widget.db, tripId);
                if (formContext.mounted) Navigator.of(formContext).pop();
              },
        onSave: (draft) async {
          if (draft.id == null) {
            await _editor.addStop(tripId, draft);
          } else {
            await _editor.updateStop(tripId, draft);
          }
          // A stop gaining or losing a night changes what blocks departure.
          await syncReadinessChecklist(widget.db, tripId);
          if (formContext.mounted) Navigator.of(formContext).pop();
        },
      ),
    ),
  );

  Future<void> _openLegForm(BuildContext context, int legId) async {
    final db = widget.db;
    final leg = await (db.select(
      db.legs,
    )..where((l) => l.id.equals(legId))).getSingleOrNull();
    if (leg == null || !context.mounted) return;

    final stops = await (db.select(
      db.stops,
    )..where((s) => s.id.isIn([leg.fromStopId, leg.toStopId]))).get();
    final byId = {for (final s in stops) s.id: s.name};
    if (!context.mounted) return;

    await Navigator.of(context).push(
      MaterialPageRoute<void>(
        builder: (formContext) => LegFormScreen(
          fromName: byId[leg.fromStopId] ?? '—',
          toName: byId[leg.toStopId] ?? '—',
          mode: leg.mode,
          plannedDeparture: leg.plannedDeparture,
          plannedArrival: leg.plannedArrival,
          isBooked: leg.isBooked,
          note: leg.note,
          distanceKm: leg.distanceKm,
          onSave:
              ({
                mode,
                plannedDeparture,
                plannedArrival,
                required isBooked,
                note,
              }) async {
                await _editor.updateLeg(
                  legId,
                  mode: mode,
                  plannedDeparture: plannedDeparture,
                  plannedArrival: plannedArrival,
                  isBooked: isBooked,
                  note: note,
                );
                if (formContext.mounted) Navigator.of(formContext).pop();
              },
        ),
      ),
    );
  }

  // -- contacts ------------------------------------------------------------

  /// The read-only entry screen. Long-press in the diary opens it.
  Future<void> _openEntry(
    BuildContext context,
    ActiveTripContext trip,
    Contact contact,
    ContactActions actions,
  ) async {
    final db = widget.db;
    String? stopName;
    if (contact.stopId != null) {
      final stop = await (db.select(
        db.stops,
      )..where((s) => s.id.equals(contact.stopId!))).getSingleOrNull();
      stopName = stop?.name;
    }
    if (!context.mounted) return;

    await Navigator.of(context).push(
      MaterialPageRoute<void>(
        builder: (_) => EntryScreen(
          contact: contact,
          stopName: stopName,
          onCopy: actions.copy,
          onOpenDialer: actions.openDialer,
          onCall: actions.call,
          onChat: actions.whatsapp,
          onConfirm: (c, {required confirmed}) async {
            await db.contactsDao.markConfirmed(c.id, confirmed: confirmed);
            // Confirming a homestay number is what clears a blocking item.
            await syncReadinessChecklist(db, trip.tripId);
          },
          onEdit: (c) => _openForm(context, trip, existing: c),
        ),
      ),
    );
  }

  /// The entry form, for a new entry or an existing one.
  Future<void> _openForm(
    BuildContext context,
    ActiveTripContext trip, {
    Contact? existing,
  }) async {
    final db = widget.db;
    final stops =
        await (db.select(db.stops)
              ..where((s) => s.tripId.equals(trip.tripId))
              ..orderBy([(s) => OrderingTerm(expression: s.sequenceOrder)]))
            .get();
    if (!context.mounted) return;

    await Navigator.of(context).push(
      MaterialPageRoute<void>(
        builder: (_) => EntryFormScreen(
          existing: existing,
          stops: [for (final s in stops) StopOption(s.id, s.name)],
          // A European trip crosses borders mid-itinerary, so the country to
          // normalise against comes from the stop, not the trip.
          country: IsoCode.IN,
          findDuplicate: (e164) =>
              db.contactsDao.findByE164(e164, tripId: trip.tripId),
          onSave: (draft) async {
            await saveEntry(db.contactsDao, draft, tripId: trip.tripId);
            await syncReadinessChecklist(db, trip.tripId);
          },
        ),
      ),
    );
  }

  /// Puts a header row on the clipboard rather than writing a file.
  ///
  /// A release build cannot write to shared storage without a permission this
  /// app deliberately does not ask for, and pasting a line into a new sheet
  /// solves the same problem with nothing to grant.
  Future<void> _copyTemplate(BuildContext context) async {
    final c = AppTokens.of(context);
    await Clipboard.setData(const ClipboardData(text: importTemplateHeader));
    if (!context.mounted) return;
    Haptics.light();
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        backgroundColor: c.ink,
        content: Text(
          'Header row copied. Paste it into row 1 of a new sheet.',
          style: AppTokens.captionStyle.copyWith(color: c.paper),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);

    return FutureBuilder<void>(
      future: _ready,
      builder: (context, bootstrap) {
        if (bootstrap.connectionState != ConnectionState.done) {
          return Scaffold(backgroundColor: c.paper, body: const SizedBox());
        }

        return StreamBuilder<ActiveTripContext?>(
          stream: watchActiveTripContext(widget.db),
          builder: (context, snap) {
            if (snap.connectionState == ConnectionState.waiting) {
              return Scaffold(backgroundColor: c.paper, body: const SizedBox());
            }
            final trip = snap.data;
            if (trip == null) {
              return _NoTrip(
                onCreate: () => _openTripForm(context),
                onDemo: () async {
                  // createDemoTrip does not set the active flag, so without
                  // this the trip would land in the database and the screen
                  // would sit here looking broken.
                  await createDemoTrip(widget.db);
                  await ensureActiveTrip(widget.db);
                },
              );
            }
            return _shell(context, trip);
          },
        );
      },
    );
  }

  Widget _shell(BuildContext context, ActiveTripContext trip) {
    final db = widget.db;
    final dao = db.contactsDao;
    final actions = ContactActions(dao: dao, tripId: trip.tripId);
    final unconfirmed = dao.watchUnconfirmedCount(trip.tripId);

    return AppShell(
      destinations: [
        ShellDestination(
          label: 'Diary',
          icon: Icons.menu_book_outlined,
          screen: DiaryScreen(
            watchContacts: dao.watchContacts,
            unconfirmedCount: unconfirmed,
            tripId: trip.tripId,
            tripName: trip.name,
            // Both real now, derived from today's date against the stops.
            currentStopId: trip.currentStopId,
            currentStopName: trip.currentStopName,
            onCopy: actions.copy,
            onOpenDialer: actions.openDialer,
            onAdd: () => _openForm(context, trip),
            onOpen: (contact) => _openEntry(context, trip, contact, actions),
          ),
        ),
        ShellDestination(
          label: 'Trip',
          icon: Icons.route_outlined,
          screen: TripScreen(
            trip: watchTripSummary(db, trip.tripId),
            unconfirmedCount: unconfirmed,
            readiness: watchReadiness(db, trip.tripId),
            onEditItinerary: () =>
                _openItinerary(context, trip.tripId, trip.name),
          ),
        ),
        ShellDestination(
          label: 'Money',
          icon: Icons.currency_rupee,
          screen: MoneyScreen(summary: watchMoneySummary(db, trip.tripId)),
        ),
        ShellDestination(
          label: 'SOS',
          icon: Icons.emergency_outlined,
          emergency: true,
          screen: EmergencyScreen(
            helplines: dao.watchEmergencyHelplines([trip.countryCode]),
            localContacts: dao.watchTripEmergencyContacts(trip.tripId),
            placeLabel: trip.currentStopName == null
                ? 'India'
                : 'India · ${trip.currentStopName}',
            onCall: actions.callNumber,
            onCopy: actions.copyNumber,
          ),
        ),
        ShellDestination(
          label: 'More',
          icon: Icons.more_horiz,
          screen: MoreScreen(
            contactCount: dao.watchContactCount(trip.tripId),
            onTrips: () => _openTrips(context),
            onItinerary: () => _openItinerary(context, trip.tripId, trip.name),
            onLegs: () => _openLegs(context, trip.tripId),
            onImport: () async {
              await ImportFlow(db: db, tripId: trip.tripId).start(context);
              await syncReadinessChecklist(db, trip.tripId);
            },
            onHistory: () => Navigator.of(context).push(
              MaterialPageRoute<void>(
                builder: (_) => ImportHistoryScreen(
                  batches: watchImportBatches(db, trip.tripId),
                  onRollback: (id) async {
                    await dao.rollbackImport(id);
                    await syncReadinessChecklist(db, trip.tripId);
                  },
                ),
              ),
            ),
            onTemplate: () => _copyTemplate(context),
          ),
        ),
      ],
    );
  }

  Future<void> _openLegs(BuildContext context, int tripId) =>
      Navigator.of(context).push(
        MaterialPageRoute<void>(
          builder: (legsContext) => LegListScreen(
            legs: watchLegSummaries(widget.db, tripId),
            onOpen: (legId) => _openLegForm(legsContext, legId),
          ),
        ),
      );
}

/// Shown when the database holds no trip at all.
class _NoTrip extends StatelessWidget {
  final VoidCallback onCreate;
  final Future<void> Function() onDemo;

  const _NoTrip({required this.onCreate, required this.onDemo});

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    return Scaffold(
      backgroundColor: c.paper,
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(AppTokens.s32),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(
                'No trip yet',
                style: AppTokens.titleStyle.copyWith(color: c.ink),
              ),
              const SizedBox(height: AppTokens.s8),
              Text(
                'A trip is a name and a list of stops. Everything else in the '
                'app hangs off it: the diary is scoped to it, the money splits '
                'within it, the emergency screen reads its stops.',
                textAlign: TextAlign.center,
                style: AppTokens.captionStyle.copyWith(color: c.muted),
              ),
              const SizedBox(height: AppTokens.s24),
              PressScale(
                onTap: onCreate,
                child: Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: AppTokens.s24,
                    vertical: AppTokens.s12,
                  ),
                  decoration: BoxDecoration(
                    color: c.signal,
                    border: Border.all(color: c.ink),
                    borderRadius: BorderRadius.circular(AppTokens.radiusSoft),
                  ),
                  child: Text(
                    'Start a trip',
                    style: AppTokens.stencilStyle.copyWith(
                      fontSize: 11.5,
                      color: c.paper,
                    ),
                  ),
                ),
              ),
              const SizedBox(height: AppTokens.s16),
              GestureDetector(
                onTap: onDemo,
                child: Text(
                  'Or fill it with a demo trip to look around',
                  style: AppTokens.captionStyle.copyWith(
                    color: c.muted,
                    decoration: TextDecoration.underline,
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
