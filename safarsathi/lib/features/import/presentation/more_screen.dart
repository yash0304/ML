// lib/features/import/presentation/more_screen.dart
//
// The fifth tab. Import, history, and the honest list of what is not built.
//
// Naming what is missing is deliberate. A settings screen with three items and
// no explanation reads as an unfinished app; the same screen saying which
// issue each absent thing belongs to reads as a plan.

import 'package:flutter/material.dart';

import '../../../core/theme/app_tokens.dart';
import '../../../core/theme/motion.dart';
import '../../../core/widgets/retro.dart';

class MoreScreen extends StatelessWidget {
  final VoidCallback onTrips;
  final VoidCallback onItinerary;
  final VoidCallback onChecklist;
  final VoidCallback onTravellers;
  final VoidCallback onWeather;
  final VoidCallback onMap;

  final VoidCallback onSync;
  final VoidCallback onSettings;
  final VoidCallback onImport;
  final VoidCallback onHistory;
  final VoidCallback onMultiAdd;

  /// One number from the phone's own contacts app, into the entry form.
  final VoidCallback onPickFromPhone;
  final VoidCallback onBackup;
  final VoidCallback onTemplate;

  /// Live count of contacts on this trip, so the tab says something true
  /// about the diary rather than being a menu of verbs.
  final Stream<int> contactCount;

  const MoreScreen({
    super.key,
    required this.onTrips,
    required this.onItinerary,
    required this.onChecklist,
    required this.onTravellers,
    required this.onWeather,
    required this.onMap,
    required this.onSync,
    required this.onSettings,
    required this.onImport,
    required this.onHistory,
    required this.onMultiAdd,
    required this.onPickFromPhone,
    required this.onBackup,
    required this.onTemplate,
    required this.contactCount,
  });

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    return Scaffold(
      backgroundColor: c.paper,
      body: SafeArea(
        child: ListView(
          children: [
            Padding(
              padding: const EdgeInsets.fromLTRB(
                AppTokens.gutter,
                AppTokens.s24,
                AppTokens.gutter,
                0,
              ),
              child: Text(
                'More',
                style: AppTokens.titleStyle.copyWith(color: c.ink),
              ),
            ),
            const StencilLabel('Before you leave'),
            _Item(
              icon: Icons.download_for_offline_outlined,
              title: 'Download everything',
              subtitle:
                  'Routes, places, map and forecast for the whole trip, in '
                  'one go, on WiFi. Then the app never asks the network '
                  'again.',
              onTap: onSync,
            ),
            const StencilLabel('Trip'),
            _Item(
              icon: Icons.luggage_outlined,
              title: 'Trips',
              subtitle:
                  'Switch between trips, or start a new one. The app is '
                  'always on exactly one.',
              onTap: onTrips,
            ),
            _Item(
              icon: Icons.list_alt_outlined,
              title: 'Edit the itinerary',
              subtitle:
                  'Add stops, drag to reorder, set dates and nights. The same '
                  'place can appear twice.',
              onTap: onItinerary,
            ),
            _Item(
              icon: Icons.checklist_outlined,
              title: 'Packing checklist',
              subtitle:
                  'Built from your stop tags and nights. Anything you change '
                  'stays changed.',
              onTap: onChecklist,
            ),
            _Item(
              icon: Icons.group_outlined,
              title: 'Who is on this trip',
              subtitle: 'Names only. No accounts, nothing sent anywhere.',
              onTap: onTravellers,
            ),
            _Item(
              icon: Icons.download_for_offline_outlined,
              title: 'Download the map',
              subtitle:
                  'Fetch the tiles for this trip on WiFi, and see how much '
                  'room they take.',
              onTap: onMap,
            ),
            _Item(
              icon: Icons.cloud_outlined,
              title: 'Weather',
              subtitle:
                  'A forecast downloaded once and frozen. The app always '
                  'says how old it is.',
              onTap: onWeather,
            ),
            const StencilLabel('Contacts'),
            _Item(
              icon: Icons.contacts_outlined,
              title: 'From my phone\'s contacts',
              subtitle:
                  'Pick one saved number — the driver, the homestay — and '
                  'file it under a stop. The app sees only the one you pick.',
              onTap: onPickFromPhone,
            ),
            _Item(
              icon: Icons.playlist_add,
              title: 'Add several at once',
              subtitle:
                  'A name and a number per line, for when you have a booking '
                  'sheet open and eight of them to get down.',
              onTap: onMultiAdd,
            ),
            _Item(
              icon: Icons.upload_file_outlined,
              title: 'Import from a sheet',
              subtitle:
                  'CSV or Excel. Columns are matched for you, and you see '
                  'every row before anything is saved.',
              onTap: onImport,
            ),
            _Item(
              icon: Icons.history,
              title: 'Import history',
              subtitle:
                  'Undo a whole import, or a whole batch you typed, in one '
                  'action.',
              onTap: onHistory,
            ),
            _Item(
              icon: Icons.table_chart_outlined,
              title: 'Column names we recognise',
              subtitle:
                  'Copy a header row to start a sheet from. Your own headers '
                  'will probably match anyway.',
              onTap: onTemplate,
            ),
            const StencilLabel('Keeping it'),
            _Item(
              icon: Icons.save_alt,
              title: 'Backup and restore',
              subtitle:
                  'Everything you typed, as one file. Nothing here is on a '
                  'server, so this is the only second copy there is.',
              onTap: onBackup,
            ),

            const StencilLabel('This trip'),
            StreamBuilder<int>(
              stream: contactCount,
              builder: (context, snap) => Padding(
                padding: const EdgeInsets.symmetric(
                  horizontal: AppTokens.gutter,
                ),
                child: Row(
                  children: [
                    Text(
                      '${snap.data ?? 0}',
                      style: AppTokens.numberStyle.copyWith(
                        color: c.ink,
                        fontSize: 20,
                      ),
                    ),
                    const SizedBox(width: AppTokens.s8),
                    Text(
                      'contacts in the diary',
                      style: AppTokens.captionStyle.copyWith(color: c.muted),
                    ),
                  ],
                ),
              ),
            ),
            const StencilLabel('App'),
            _Item(
              icon: Icons.settings_outlined,
              title: 'Settings',
              subtitle:
                  'Theme, corridor width, call history, and what each trip '
                  'has downloaded.',
              onTap: onSettings,
            ),
            const StencilLabel('Not built yet'),
            Padding(
              padding: const EdgeInsets.symmetric(
                horizontal: AppTokens.gutter,
              ),
              child: Text(
                // THIS USED TO SAY THE APP MAKES NO NETWORK CALL AND THE
                // RELEASE BUILD DOES NOT EVEN ASK FOR PERMISSION TO. Both were
                // false from the day maps, routes and weather were added: the
                // manifest holds INTERNET and "Download everything" uses it. It
                // also listed the leg screen as unbuilt, long after it was.
                // A screen whose job is to be honest about the app has to be.
                'The GPS timeline (#30). Trusted-contact check-ins (#33, '
                '#34).\n\n'
                'The app uses the internet only when you ask it to, on a '
                'screen that says what it will fetch: Download everything, '
                'Download the map, and Weather. On the road it asks for '
                'nothing — the map, routes, places and forecasts are read '
                'from this phone.',
                style: AppTokens.captionStyle.copyWith(color: c.muted),
              ),
            ),
            const SizedBox(height: AppTokens.s32),
          ],
        ),
      ),
    );
  }
}

class _Item extends StatelessWidget {
  final IconData icon;
  final String title;
  final String subtitle;
  final VoidCallback onTap;

  const _Item({
    required this.icon,
    required this.title,
    required this.subtitle,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    return PressScale(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(
          horizontal: AppTokens.gutter,
          vertical: AppTokens.s16,
        ),
        decoration: BoxDecoration(
          border: Border(
            bottom: BorderSide(color: c.rule, width: AppTokens.hairline),
          ),
        ),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Icon(icon, size: 20, color: c.ink),
            const SizedBox(width: AppTokens.s16),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    title,
                    style: AppTokens.rowTitleStyle.copyWith(color: c.ink),
                  ),
                  const SizedBox(height: 2),
                  Text(
                    subtitle,
                    style: AppTokens.captionStyle.copyWith(color: c.muted),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
