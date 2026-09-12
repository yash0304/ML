import 'package:flutter/material.dart';

import 'core/database/app_database.dart';
import 'core/theme/app_tokens.dart';
import 'features/contacts/data/contact_actions.dart';
import 'features/contacts/presentation/diary_screen.dart';
import 'features/dev/dev_seed.dart';

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
  late final Future<DemoTrip?> _trip = ensureDemoTrip(widget.db);

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    return FutureBuilder<DemoTrip?>(
      future: _trip,
      builder: (context, snap) {
        if (!snap.hasData) {
          return Scaffold(backgroundColor: c.paper, body: const SizedBox());
        }
        final trip = snap.data!;
        final dao = widget.db.contactsDao;
        final actions = ContactActions(dao: dao, tripId: trip.tripId);
        return DiaryScreen(
          watchContacts: dao.watchContacts,
          unconfirmedCount: dao.watchUnconfirmedCount(trip.tripId),
          tripId: trip.tripId,
          tripName: trip.name,
          currentStopId: trip.currentStopId,
          currentStopName: trip.currentStopName,
          onCopy: actions.copy,
          onOpenDialer: actions.openDialer,
          // The entry screen is #8.
        );
      },
    );
  }
}
