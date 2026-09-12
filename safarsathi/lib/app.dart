import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show Clipboard, ClipboardData;

import 'core/database/app_database.dart';
import 'core/theme/app_tokens.dart';
import 'core/theme/motion.dart';
import 'core/widgets/app_shell.dart';
import 'features/emergency/presentation/emergency_screen.dart';
import 'features/money/data/money_summary.dart';
import 'features/money/presentation/money_screen.dart';
import 'features/trips/data/trip_summary.dart';
import 'features/trips/presentation/trip_screen.dart';
import 'package:drift/drift.dart' show OrderingTerm;
import 'package:phone_numbers_parser/phone_numbers_parser.dart' show IsoCode;

import 'features/contacts/data/contact_actions.dart';
import 'features/contacts/data/entry_draft.dart';
import 'features/contacts/presentation/diary_screen.dart';
import 'features/contacts/presentation/entry_form_screen.dart';
import 'features/contacts/presentation/entry_screen.dart';
import 'features/dev/dev_seed.dart';
import 'features/import/data/import_commit.dart';
import 'features/import/presentation/import_flow.dart';
import 'features/import/presentation/import_history_screen.dart';
import 'features/import/presentation/more_screen.dart';

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

/// Opens the diary on whatever trip exists. Trip selection is #16; until
/// then a debug build seeds a demo trip so there is something to render.
class _Home extends StatefulWidget {
  final AppDatabase db;
  const _Home({required this.db});

  @override
  State<_Home> createState() => _HomeState();
}

class _HomeState extends State<_Home> {
  late Future<DemoTrip?> _trip = ensureDemoTrip(widget.db);

  Future<void> _makeDemoTrip() async {
    final created = createDemoTrip(widget.db);
    setState(() => _trip = created.then<DemoTrip?>((t) => t));
    await created;
  }

  /// The read-only entry screen. Long-press in the diary opens it.
  Future<void> _openEntry(
    BuildContext context,
    DemoTrip trip,
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
          onConfirm: (c, {required confirmed}) =>
              db.contactsDao.markConfirmed(c.id, confirmed: confirmed),
          onEdit: (c) => _openForm(context, trip, existing: c),
        ),
      ),
    );
  }

  /// The entry form, for a new entry or an existing one.
  Future<void> _openForm(
    BuildContext context,
    DemoTrip trip, {
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
          onSave: (draft) =>
              saveEntry(db.contactsDao, draft, tripId: trip.tripId),
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
    return FutureBuilder<DemoTrip?>(
      future: _trip,
      builder: (context, snap) {
        if (snap.connectionState != ConnectionState.done) {
          return Scaffold(backgroundColor: c.paper, body: const SizedBox());
        }
        final trip = snap.data;
        if (trip == null) {
          // A release build has no trip and no way to make one until #16.
          return _NoTrip(onCreate: _makeDemoTrip);
        }
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
                currentStopId: trip.currentStopId,
                currentStopName: trip.currentStopName,
                onCopy: actions.copy,
                onOpenDialer: actions.openDialer,
                onAdd: () => _openForm(context, trip),
                onOpen: (contact) =>
                    _openEntry(context, trip, contact, actions),
              ),
            ),
            ShellDestination(
              label: 'Trip',
              icon: Icons.route_outlined,
              screen: TripScreen(
                trip: watchTripSummary(
                  db,
                  trip.tripId,
                  currentStopId: trip.currentStopId,
                ),
                unconfirmedCount: unconfirmed,
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
                helplines: dao.watchEmergencyHelplines(const ['IN']),
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
                onImport: () =>
                    ImportFlow(db: db, tripId: trip.tripId).start(context),
                onHistory: () => Navigator.of(context).push(
                  MaterialPageRoute<void>(
                    builder: (_) => ImportHistoryScreen(
                      batches: watchImportBatches(db, trip.tripId),
                      onRollback: dao.rollbackImport,
                    ),
                  ),
                ),
                onTemplate: () => _copyTemplate(context),
              ),
            ),
          ],
        );
      },
    );
  }
}

/// Shown when the database holds no trip at all. Real trip creation is #16;
/// until then this offers the demo so an installed build has something in it.
class _NoTrip extends StatelessWidget {
  final Future<void> Function() onCreate;
  const _NoTrip({required this.onCreate});

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
                'Building a trip properly comes later. For now this fills the '
                'diary with a few placeholder entries so there is something '
                'to look at. Their numbers are deliberately fake.',
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
                    'Create demo trip',
                    style: AppTokens.stencilStyle.copyWith(
                      fontSize: 11.5,
                      color: c.paper,
                    ),
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
