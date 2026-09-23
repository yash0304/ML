import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';
import 'dart:convert';
import 'dart:typed_data';
import 'dart:io';

import 'package:flutter/services.dart' show Clipboard, ClipboardData;
import 'package:file_picker/file_picker.dart';
import 'package:path_provider/path_provider.dart';
import 'package:drift/drift.dart' show OrderingTerm;
import 'package:phone_numbers_parser/phone_numbers_parser.dart' show IsoCode;

import 'core/database/app_database.dart';
import 'core/theme/app_tokens.dart';
import 'core/theme/motion.dart';
import 'core/widgets/app_shell.dart';
import 'features/contacts/data/contacts_dao.dart' show ContactCategory;
import 'features/contacts/data/phone_contact_picker.dart';
import 'features/contacts/data/contact_actions.dart';
import 'features/contacts/data/entry_draft.dart';
import 'features/contacts/presentation/diary_screen.dart';
import 'features/contacts/presentation/entry_form_screen.dart';
import 'features/contacts/presentation/entry_screen.dart';
import 'package:share_plus/share_plus.dart';

import 'features/backup/data/backup.dart';
import 'features/backup/data/full_backup.dart';
import 'features/backup/presentation/backup_screen.dart';
import 'features/contacts/data/multi_add.dart';
import 'features/contacts/presentation/multi_add_screen.dart';
import 'features/checklist/data/checklist_dao.dart';
import 'features/checklist/data/checklist_generator.dart';
import 'features/checklist/presentation/checklist_screen.dart';
import 'features/emergency/data/sos.dart';
import 'features/emergency/presentation/sos_panel.dart';
import 'features/emergency/presentation/emergency_screen.dart';
import 'features/import/data/import_commit.dart';
import 'features/import/presentation/import_flow.dart';
import 'features/import/presentation/import_history_screen.dart';
import 'features/import/presentation/more_screen.dart';
import 'features/discovery/data/corridor_sync.dart';
import 'features/discovery/data/discovery.dart';
import 'features/discovery/data/geo.dart';
import 'features/discovery/data/geocoder.dart';
import 'features/discovery/presentation/discovery_screen.dart';
import 'features/discovery/presentation/poi_detail_screen.dart';
import 'features/map/data/here.dart';
import 'features/map/data/map_download.dart';
import 'features/map/data/tile_downloader.dart';
import 'features/map/data/tile_provider.dart';
import 'features/map/data/tile_store.dart';
import 'features/map/presentation/map_download_screen.dart';
import 'features/map/presentation/trip_map_screen.dart';
import 'features/money/data/expense_editor.dart';
import 'features/money/data/money_summary.dart';
import 'features/money/presentation/expense_form_screen.dart';
import 'features/money/presentation/travellers_screen.dart';
import 'features/money/presentation/money_screen.dart';
import 'features/trips/data/trip_health.dart';
import 'features/trips/data/plan_message.dart';
import 'features/trips/data/tonight.dart';
import 'features/trips/data/readiness.dart';
import 'features/trips/data/trip_editor.dart';
import 'features/trips/data/stop_detail.dart';
import 'features/trips/data/trip_summary.dart';
import 'features/settings/data/settings.dart';
import 'features/sync/data/trip_sync.dart';
import 'features/sync/presentation/sync_screen.dart';
import 'features/settings/presentation/settings_screen.dart';
import 'features/trips/presentation/itinerary_screen.dart';
import 'features/trips/presentation/place_picker_sheet.dart';
import 'features/weather/data/weather_sync.dart';
import 'features/weather/presentation/weather_screen.dart';
import 'features/trips/presentation/leg_detail_screen.dart';
import 'features/trips/presentation/leg_form_screen.dart';
import 'features/trips/presentation/leg_list_screen.dart';
import 'features/trips/presentation/stop_detail_screen.dart';
import 'features/trips/presentation/stop_form_screen.dart';
import 'features/trips/presentation/trip_form_screen.dart';
import 'features/trips/presentation/trip_list_screen.dart';
import 'features/trips/presentation/trip_screen.dart';

class SafarSathiApp extends StatelessWidget {
  /// Passed down rather than reached for globally. There is no repository
  /// layer and no service locator yet — DAO to widget, until a second
  /// consumer of the same data appears.
  final AppDatabase db;

  final ThemeMode themeMode;

  const SafarSathiApp({
    super.key,
    required this.db,
    this.themeMode = ThemeMode.system,
  });

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'SafarSathi',
      debugShowCheckedModeBanner: false,
      theme: AppTokens.light,
      darkTheme: AppTokens.dark,
      themeMode: themeMode,
      home: _Home(db: db),
    );
  }
}

/// Wraps the app so the theme override from Settings (#35) reaches
/// `MaterialApp.themeMode`. A phone in a pocket does not know it is night in a
/// valley, so following the system is the default rather than the only option.
class SafarSathiRoot extends StatelessWidget {
  final AppDatabase db;
  const SafarSathiRoot({super.key, required this.db});

  @override
  Widget build(BuildContext context) {
    return StreamBuilder<ThemeMode>(
      stream: SettingsRepository(db).watchThemeMode(),
      builder: (context, snap) =>
          SafarSathiApp(db: db, themeMode: snap.data ?? ThemeMode.system),
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
  late final ExpenseEditor _money = ExpenseEditor(widget.db);
  late final ChecklistDao _checklist = ChecklistDao(widget.db);
  late final SettingsRepository _settings = SettingsRepository(widget.db);
  late final WeatherSync _weather = WeatherSync(db: widget.db);
  late final Geocoder _geocoder = Geocoder();

  /// Where downloaded tiles live. Resolved once at startup, because
  /// `getApplicationDocumentsDirectory` is a platform call and the map screen
  /// should not await one every time it rebuilds.
  TileStore? _tiles;
  late final Future<void> _ready = _bootstrap();

  Future<void> _bootstrap() async {
    // No seeding, in any build: an empty database opens on "No trip yet".
    await ensureActiveTrip(widget.db);

    final documents = await getApplicationDocumentsDirectory();
    _tiles = TileStore(
      db: widget.db,
      root: Directory('${documents.path}/tiles'),
    );
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
        onOpen: (stop) =>
            _openStopDetail(itineraryContext, tripId, tripName, stop),
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

  /// One stop — issue #51.
  ///
  /// The stop row itself is passed only for the title and the edit hop; every
  /// number on the screen arrives on the stream, so an edit made from here is
  /// reflected without popping back.
  Future<void> _openStopDetail(
    BuildContext context,
    int tripId,
    String tripName,
    Stop stop,
  ) => Navigator.of(context).push(
    MaterialPageRoute<void>(
      builder: (detailContext) => StopDetailScreen(
        detail: watchStopDetail(widget.db, stop.id),
        onEdit: () async {
          // Re-read rather than reusing the row this route was opened with:
          // by now the tags may have been edited on this very screen, and the
          // form would open showing the old ones.
          final fresh = await (widget.db.select(
            widget.db.stops,
          )..where((s) => s.id.equals(stop.id))).getSingleOrNull();
          if (fresh != null && detailContext.mounted) {
            await _openStopForm(detailContext, tripId, stop: fresh);
          }
        },
        onOpenDiary: () => Navigator.of(detailContext).push(
          MaterialPageRoute<void>(
            builder: (diaryContext) {
              final actions = ContactActions(
                dao: widget.db.contactsDao,
                tripId: tripId,
              );
              return DiaryScreen(
                watchContacts: widget.db.contactsDao.watchContacts,
                unconfirmedCount: widget.db.contactsDao.watchUnconfirmedCount(
                  tripId,
                ),
                tripId: tripId,
                tripName: tripName,
                currentStopId: stop.id,
                currentStopName: stop.name,
                // The count on the stop screen said "here", so the list has
                // to open saying the same thing.
                startStopScoped: true,
                onCopy: actions.copy,
                onOpenDialer: actions.openDialer,
                onOpen: (contact) async {
                  final trip = await watchActiveTripContext(
                    widget.db,
                  ).first;
                  if (trip != null && diaryContext.mounted) {
                    await _openEntry(diaryContext, trip, contact, actions);
                  }
                },
              );
            },
          ),
        ),
        onOpenChecklist: () => _openChecklist(detailContext, tripId),
        onTags: (tags) async {
          final fresh = await (widget.db.select(
            widget.db.stops,
          )..where((s) => s.id.equals(stop.id))).getSingleOrNull();
          if (fresh == null) return;
          await _editor.updateStop(
            tripId,
            StopDraft.fromRow(fresh).copyWith(activityTags: tags),
          );
          // THE LINK THE SCREEN CLAIMS. Editing tags here has to actually
          // rebuild the packing list, or the sentence under the chips is a
          // lie. Hand edits survive: regeneratePackList keys on the
          // generator, not the label.
          await regeneratePackList(widget.db, tripId);
          await syncReadinessChecklist(widget.db, tripId);
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
        onPickPlace: (name, current) => _pickPlace(formContext, name, current),
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

  /// The place picker (stop coordinates). The geocoder lives here so the sheet
  /// itself never reaches the network in a test.
  Future<LatLng?> _pickPlace(
    BuildContext context,
    String name,
    LatLng? current,
  ) => Navigator.of(context).push<LatLng>(
    MaterialPageRoute<LatLng>(
      builder: (_) => PlacePickerSheet(
        initialQuery: name,
        current: current,
        search: (query, country) =>
            _geocoder.search(query, countryCode: country),
      ),
    ),
  );

  /// Everything a trip needs, in one press (#25).
  ///
  /// The other three screens still exist for anyone who wants one piece; this
  /// is the one that means you cannot leave having forgotten a step.
  // -- discovery -----------------------------------------------------------

  Future<void> _openDiscovery(
    BuildContext context,
    int tripId,
    int legId,
  ) => Navigator.of(context).push(
    MaterialPageRoute<void>(
      builder: (discoveryContext) => DiscoveryScreen(
        discovery: watchLegDiscovery(widget.db, legId),
        onOpen: (place) => _openPoi(discoveryContext, tripId, place),
      ),
    ),
  );

  Future<void> _openPoi(
    BuildContext context,
    int tripId,
    CorridorPlace place,
  ) async {
    final db = widget.db;
    final actions = ContactActions(dao: db.contactsDao, tripId: tripId);

    // Already in the diary? Matched on the normalised number, so the same
    // place saved from two legs is recognised once.
    var saved = false;
    for (final phone in place.phones) {
      final e164 = phone.phoneE164;
      if (e164 == null) continue;
      if (await db.contactsDao.findByE164(e164, tripId: tripId) != null) {
        saved = true;
        break;
      }
    }
    if (!context.mounted) return;

    await Navigator.of(context).push(
      MaterialPageRoute<void>(
        builder: (poiContext) => PoiDetailScreen(
          place: place,
          alreadySaved: saved,
          onCopy: (phone) => actions.copyNumber(phone.phoneRaw),
          onOpenDialer: (phone) => actions.openDialerFor(phone.phoneRaw),
          onOpenMaps: () => actions.openMaps(googleMapsUrl(place)),
          onSave: (phone) async {
            await savePlaceAsContact(
              db,
              tripId: tripId,
              place: place,
              phone: phone,
            );
            if (!poiContext.mounted) return;
            final c = AppTokens.of(poiContext);
            Haptics.confirm();
            Navigator.of(poiContext).pop();
            ScaffoldMessenger.of(poiContext).showSnackBar(
              SnackBar(
                backgroundColor: c.ink,
                content: Text(
                  'Saved. It still reads "from open map data" until you call '
                  'it and say so.',
                  style: AppTokens.captionStyle.copyWith(color: c.paper),
                ),
              ),
            );
          },
        ),
      ),
    );
  }

  Future<void> _openSync(BuildContext context, int tripId) async {
    final store = _tiles;
    if (store == null) return;

    // Resolved here, at the moment the screen opens, rather than held in a
    // field: the key can be typed into Settings and used without restarting.
    final provider = tileProviderFor(await _settings.readMapTilerKey());
    if (!context.mounted) return;

    final map = MapDownload(
      db: widget.db,
      downloader: TileDownloader(store: store, provider: provider),
    );
    final sync = TripSync(
      db: widget.db,
      corridor: CorridorSync(db: widget.db),
      weather: _weather,
      map: map,
    );

    await Navigator.of(context).push(
      MaterialPageRoute<void>(
        builder: (_) => SyncScreen(
          plan: () => sync.plan(tripId),
          run: () => sync.run(tripId),
          estimateSize: () => estimateSyncSize(map, tripId),
        ),
      ),
    );
  }

  /// Looking at the downloaded map.
  ///
  /// Separate from the download screen on purpose: until now the only map
  /// surface in the app was the one that fetches tiles and counts them, so
  /// there was no way to see what had been fetched.
  Future<void> _viewMap(BuildContext context, int tripId) async {
    final store = _tiles;
    if (store == null) return;

    final provider = tileProviderFor(await _settings.readMapTilerKey());
    if (!context.mounted) return;

    await Navigator.of(context).push(
      MaterialPageRoute<void>(
        builder: (mapContext) => TripMapScreen(
          provider: provider,
          store: store,
          location: const DeviceLocation(),
          load: () =>
              readTripMap(widget.db, tripId, providerId: provider.id),
          onDownload: () => _openMap(mapContext, tripId),
        ),
      ),
    );
  }

  Future<void> _openMap(BuildContext context, int tripId) async {
    final store = _tiles;
    if (store == null) return;

    final provider = tileProviderFor(await _settings.readMapTilerKey());
    if (!context.mounted) return;

    final download = MapDownload(
      db: widget.db,
      downloader: TileDownloader(store: store, provider: provider),
    );

    await Navigator.of(context).push(
      MaterialPageRoute<void>(
        builder: (_) => MapDownloadScreen(
          provider: provider,
          estimate: () => download.estimate(tripId),
          download: () => download.download(tripId),
          usage: store.watchUsage(provider.id),
          onClear: () => store.clear(provider.id),
        ),
      ),
    );
  }

  Future<void> _openWeather(BuildContext context, int tripId) =>
      Navigator.of(context).push(
        MaterialPageRoute<void>(
          builder: (weatherContext) => WeatherScreen(
            weather: _weather.watchTripWeather(tripId),
            onRefresh: () => _fetchWeather(weatherContext, tripId),
          ),
        ),
      );

  /// Fetches every stop's forecast. Reports what failed rather than stopping
  /// at the first stop with no coordinates.
  Future<void> _fetchWeather(BuildContext context, int tripId) async {
    final messenger = ScaffoldMessenger.of(context);
    final c = AppTokens.of(context);
    final stops = await _editor.stopsOf(tripId);

    var done = 0;
    var skipped = 0;
    for (final stop in stops) {
      try {
        await _weather.syncStop(stop);
        done++;
      } on Object {
        skipped++;
      }
    }
    if (!context.mounted) return;

    Haptics.light();
    messenger.showSnackBar(
      SnackBar(
        backgroundColor: c.ink,
        content: Text(
          skipped == 0
              ? 'Forecast downloaded for $done '
                    '${done == 1 ? 'stop' : 'stops'}.'
              : 'Forecast downloaded for $done. $skipped skipped — those '
                    'stops need coordinates first.',
          style: AppTokens.captionStyle.copyWith(color: c.paper),
        ),
      ),
    );
  }

  Future<void> _openSettings(BuildContext context, int tripId) =>
      Navigator.of(context).push(
        MaterialPageRoute<void>(
          builder: (settingsContext) => SettingsScreen(
            themeMode: _settings.watchThemeMode(),
            onThemeMode: _settings.setThemeMode,
            corridorKm: _settings.watchCorridorKm(),
            onCorridorKm: _settings.setCorridorKm,
            caches: watchCacheSummaries(widget.db),
            onClearCache: (id) => clearTripCache(widget.db, id),
            mapKey: _settings.watchMapTilerKey(),
            onMapKey: _settings.setMapTilerKey,
            hasBuildKey: MapTilerRaster.buildTimeKey.isNotEmpty,
            mapProviderLabel: activeTileProvider.label,
            onCallHistory: () => Navigator.of(settingsContext).push(
              MaterialPageRoute<void>(
                builder: (_) => CallHistoryScreen(
                  history: watchCallHistory(widget.db, tripId),
                ),
              ),
            ),
          ),
        ),
      );

  /// One leg — issue #52.
  Future<void> _openLegDetail(
    BuildContext context,
    int tripId,
    int legId,
  ) => Navigator.of(context).push(
    MaterialPageRoute<void>(
      builder: (detailContext) => LegDetailScreen(
        discovery: watchLegDiscovery(widget.db, legId),
        transport: watchLegTransport(widget.db, legId),
        onEditTransport: () => _openLegForm(detailContext, legId),
        onSeeAll: () => _openDiscovery(detailContext, tripId, legId),
        onOpenPlace: (place) => _openPoi(detailContext, tripId, place),
        // The diary's own entry screen, so a number reached from the road is
        // called, copied and confirmed exactly as it would be from the diary
        // — one set of actions, one trust state, not a second copy of either.
        onOpenContact: (contact) async {
          final trip = await watchActiveTripContext(widget.db).first;
          if (trip == null || !detailContext.mounted) return;
          await _openEntry(
            detailContext,
            trip,
            contact,
            ContactActions(dao: widget.db.contactsDao, tripId: trip.tripId),
          );
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
        builder: (entryContext) => EntryScreen(
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
          onEdit: (c) async {
            final deleted = await _openForm(entryContext, trip, existing: c);
            // Gone: close the entry too, rather than show a contact that no
            // longer exists and would dial if tapped.
            if (deleted && entryContext.mounted) {
              Navigator.of(entryContext).pop();
            }
          },
        ),
      ),
    );
  }

  /// The entry form, for a new entry or an existing one. True when the entry
  /// was deleted from it.
  Future<bool> _openForm(
    BuildContext context,
    ActiveTripContext trip, {
    Contact? existing,
    bool pickOnOpen = false,
    int? initialStopId,
    String? initialCategory,
  }) async {
    final db = widget.db;
    final stops =
        await (db.select(db.stops)
              ..where((s) => s.tripId.equals(trip.tripId))
              ..orderBy([(s) => OrderingTerm(expression: s.sequenceOrder)]))
            .get();
    if (!context.mounted) return false;

    final deleted = await Navigator.of(context).push<bool>(
      MaterialPageRoute<bool>(
        builder: (_) => EntryFormScreen(
          existing: existing,
          pickFromPhone: const PhoneContactPicker().pick,
          pickOnOpen: pickOnOpen,
          initialStopId: initialStopId,
          initialCategory: initialCategory,
          onDelete: existing == null
              ? null
              : () async {
                  await db.contactsDao.deleteContact(existing.id);
                  // A deleted homestay may be what a readiness item was
                  // waiting on; the list has to know it is gone.
                  await syncReadinessChecklist(db, trip.tripId);
                },
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
    return deleted == true;
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
              return _NoTrip(onCreate: () => _openTripForm(context));
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
          label: 'Trip',
          icon: Icons.route_outlined,
          screen: TripScreen(
            trip: watchTripSummary(db, trip.tripId),
            unconfirmedCount: unconfirmed,
            readiness: watchReadiness(db, trip.tripId),
            onEditItinerary: () =>
                _openItinerary(context, trip.tripId, trip.name),
            onViewMap: () => _viewMap(context, trip.tripId),
            onLegs: () => _openLegs(context, trip.tripId),
            tonight: watchTonight(db, trip.tripId),
            onOpenContact: (contact) =>
                _openEntry(context, trip, contact, actions),
            health: watchTripHealth(db, trip.tripId),
            onRemoveSamples: () async {
              await removeSamples(db, trip.tripId);
              await syncReadinessChecklist(db, trip.tripId);
            },
            onRemoveDuplicates: () async {
              await removeDuplicates(db, trip.tripId);
              await syncReadinessChecklist(db, trip.tripId);
            },
            onSharePlan: () async {
              // Plain text through the share sheet: WhatsApp, SMS, email —
              // whichever the person at home actually reads.
              final text = await buildPlanMessage(db, trip.tripId);
              await SharePlus.instance.share(
                ShareParams(text: text, subject: '${trip.name} — the plan'),
              );
            },
            onAddStay: (stopId) => _openForm(
              context,
              trip,
              initialStopId: stopId,
              initialCategory: ContactCategory.accommodation,
            ),
          ),
        ),
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
            onTogglePin: (contact, pinned) =>
                dao.togglePin(contact.id, pinned),
            cacheStamp: watchCacheStamp(db, trip.tripId),
          ),
        ),
        ShellDestination(
          label: 'Money',
          icon: Icons.currency_rupee,
          screen: MoneyScreen(
            summary: watchMoneySummary(db, trip.tripId),
            onAdd: () => _openExpense(context, trip.tripId),
            onOpen: (id) => _openExpense(context, trip.tripId, expenseId: id),
            onTravellers: () => _openTravellers(context, trip.tripId),
          ),
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
            sosPanel: SosPanel(
              trusted: watchTrusted(db),
              location: const DeviceLocation(),
              nearStop: () async => trip.currentStopName,
              openSms: (uri) =>
                  launchUrl(uri, mode: LaunchMode.externalApplication),
              share: (text) => SharePlus.instance.share(
                ShareParams(text: text, subject: 'SOS'),
              ),
              pickPerson: const PhoneContactPicker().pick,
              addPerson: (name, phone) => addTrusted(db, name, phone),
              removePerson: (person) => removeTrusted(db, person.id),
            ),
          ),
        ),
        ShellDestination(
          label: 'More',
          icon: Icons.more_horiz,
          screen: MoreScreen(
            contactCount: dao.watchContactCount(trip.tripId),
            onTrips: () => _openTrips(context),
            onItinerary: () => _openItinerary(context, trip.tripId, trip.name),
            onChecklist: () => _openChecklist(context, trip.tripId),
            onTravellers: () => _openTravellers(context, trip.tripId),
            onWeather: () => _openWeather(context, trip.tripId),
            onMap: () => _openMap(context, trip.tripId),
            onSync: () => _openSync(context, trip.tripId),
            onSettings: () => _openSettings(context, trip.tripId),
            onMultiAdd: () => _openMultiAdd(context, trip.tripId),
            onPickFromPhone: () => _openForm(context, trip, pickOnOpen: true),
            onBackup: () => _openBackup(context),
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

  // -- money ---------------------------------------------------------------

  Future<void> _openTravellers(BuildContext context, int tripId) =>
      Navigator.of(context).push(
        MaterialPageRoute<void>(
          builder: (_) => TravellersScreen(
            travellers: _money.watchTravellers(tripId),
            onAdd: (name) => _money.addTraveller(tripId, name),
            onRename: (t, name) => _money.renameTraveller(t.id, name),
            onDelete: _money.deleteTraveller,
          ),
        ),
      );

  Future<void> _openExpense(
    BuildContext context,
    int tripId, {
    int? expenseId,
  }) async {
    final db = widget.db;
    final travellers = await _money.travellersOf(tripId);

    if (travellers.isEmpty) {
      // An expense needs someone to have paid it, so there is nothing useful
      // the form could show. Send them to the place that fixes it.
      if (!context.mounted) return;
      await _openTravellers(context, tripId);
      return;
    }

    ExpenseDraft? existing;
    if (expenseId != null) {
      final row = await (db.select(
        db.expenses,
      )..where((e) => e.id.equals(expenseId))).getSingleOrNull();
      if (row != null) {
        existing = ExpenseDraft(
          id: row.id,
          description: row.description,
          amountMinor: row.amountMinor,
          paidById: row.paidById,
          shares: await _money.sharesOf(row.id),
          spentAt: row.spentAt,
          category: row.category,
          stopId: row.stopId,
          currency: row.currency,
          rateToBase: row.rateToBase,
          rateCapturedAt: row.rateCapturedAt,
        );
      }
    }
    if (!context.mounted) return;

    await Navigator.of(context).push(
      MaterialPageRoute<void>(
        builder: (formContext) => ExpenseFormScreen(
          travellers: travellers,
          existing: existing,
          onDelete: expenseId == null
              ? null
              : () async {
                  await _money.deleteExpense(expenseId);
                  if (formContext.mounted) Navigator.of(formContext).pop();
                },
          onSave: (draft) async {
            await _money.saveExpense(tripId, draft);
            if (formContext.mounted) Navigator.of(formContext).pop();
          },
        ),
      ),
    );
  }

  // -- checklist -------------------------------------------------------------

  Future<void> _openChecklist(BuildContext context, int tripId) async {
    final db = widget.db;
    // Build it on first open rather than on every stop edit: regeneration is
    // cheap but it is also the moment edits could be lost, so it happens when
    // the user is looking at the list.
    await regeneratePackList(db, tripId);
    await syncReadinessChecklist(db, tripId);
    if (!context.mounted) return;

    await Navigator.of(context).push(
      MaterialPageRoute<void>(
        builder: (_) => ChecklistScreen(
          checklist: watchChecklist(db, tripId),
          onToggle: (item, done) => _checklist.setDone(item.id, done),
          onRemove: _checklist.remove,
          onEdit: (item, label, qty) =>
              _checklist.edit(item.id, label: label, quantity: qty),
          onAdd: (label) => _checklist.addOwn(tripId, label),
          onRegenerate: () async {
            await regeneratePackList(db, tripId);
            await syncReadinessChecklist(db, tripId);
          },
        ),
      ),
    );
  }

  /// The archive a restore is about to unpack, when a zip was picked.
  ///
  /// Held here rather than on the screen because it is a file on disk that
  /// has to be cleaned up whichever way the screen is left.
  File? _pendingArchive;

  Future<void> _clearPendingArchive() async {
    final archive = _pendingArchive;
    _pendingArchive = null;
    if (archive != null && archive.existsSync()) await archive.delete();
  }

  /// Backup and restore — issue #56.
  Future<void> _openBackup(BuildContext context) => Navigator.of(context).push(
    MaterialPageRoute<void>(
      builder: (_) => BackupScreen(
        onExport: () async {
          final json = await exportBackup(widget.db);
          final saved = await FilePicker.saveFile(
            fileName: backupFileName(),
            bytes: Uint8List.fromList(utf8.encode(json)),
            mimeType: 'application/json',
            dialogTitle: 'Save your SafarSathi backup',
          );
          if (saved == null) return null;
          return 'Saved. Keep it somewhere that is not this phone — '
              'that is the whole point of it.';
        },
        onPick: () async {
          final file = await FilePicker.pickFile(
            type: FileType.custom,
            allowedExtensions: const ['json', 'zip'],
            dialogTitle: 'Pick a SafarSathi backup',
          );
          if (file == null) return null;

          // Whatever was picked last time is no longer what will be
          // restored, and a stale archive here would silently write the
          // wrong map back.
          await _clearPendingArchive();

          if ((file.extension ?? '').toLowerCase() != 'zip') {
            // Bytes, never a path: on Android a picked file usually lives
            // behind a content:// URI with no readable filesystem path.
            return readBackup(utf8.decode(await file.readAsBytes()));
          }

          // STREAMED TO DISK, NOT READ INTO MEMORY. A full archive carries
          // every downloaded tile, and reading that as one list is how a
          // restore turns into a crash on the phone that needed it most.
          final temp = await getTemporaryDirectory();
          final copy = File('${temp.path}/restore-${file.name}');
          final sink = copy.openWrite();
          await sink.addStream(file.readAsByteStream());
          await sink.close();

          _pendingArchive = copy;
          return (await readFullBackup(copy)).database;
        },
        onPlanFull: () => planFullBackup(widget.db),
        onExportFull: () async {
          final store = _tiles;
          if (store == null) return null;

          // Written to the app's own cache first, then handed out by path.
          // The save dialog takes bytes, and a 350 MB tile cache as one
          // Uint8List is how you turn a backup into a crash.
          final temp = await getTemporaryDirectory();
          final archive = File('${temp.path}/${fullBackupFileName()}');
          await writeFullBackup(
            widget.db,
            destination: archive,
            tileRoot: store.root,
          );

          final result = await SharePlus.instance.share(
            ShareParams(
              files: [XFile(archive.path)],
              fileNameOverrides: [archive.uri.pathSegments.last],
              subject: 'SafarSathi backup, with the map',
            ),
          );
          // The temp copy is the app's, not the user's; whatever they chose
          // has its own copy by now.
          if (archive.existsSync()) await archive.delete();

          if (result.status == ShareResultStatus.dismissed) return null;
          return 'Saved, map included. Keep it somewhere that is not this '
              'phone — that is the whole point of it.';
        },
        onCurrent: () => currentContents(widget.db),
        onRestore: (backup) async {
          final archive = _pendingArchive;
          final store = _tiles;
          if (archive != null && store != null) {
            await restoreFullBackup(
              widget.db,
              archive,
              tileRoot: store.root,
            );
          } else {
            await restoreBackup(widget.db, backup);
          }
          await _clearPendingArchive();
          // A restored database may hold a different active trip, or none.
          await ensureActiveTrip(widget.db);
        },
      ),
    ),
  );

  /// Several contacts at once — issue #10.
  Future<void> _openMultiAdd(BuildContext context, int tripId) async {
    final db = widget.db;
    final dao = db.contactsDao;

    final result = await Navigator.of(context).push<MultiAddResult>(
      MaterialPageRoute<MultiAddResult>(
        builder: (_) => MultiAddScreen(
          check: (row) => checkRow(
            row,
            findDuplicate: (e164) => dao.findByE164(e164, tripId: tripId),
          ),
          onSave: (sheet) =>
              commitMultiAdd(db, tripId: tripId, sheet: sheet),
        ),
      ),
    );

    if (result != null && result.added > 0) {
      // New unconfirmed numbers change what blocks departure.
      await syncReadinessChecklist(db, tripId);
    }
  }

  Future<void> _openLegs(BuildContext context, int tripId) =>
      Navigator.of(context).push(
        MaterialPageRoute<void>(
          builder: (legsContext) => LegListScreen(
            legs: watchLegSummaries(widget.db, tripId),
            onOpen: (legId) => _openLegDetail(legsContext, tripId, legId),
            onDiscover: (legId) =>
                _openDiscovery(legsContext, tripId, legId),
          ),
        ),
      );
}

/// Shown when the database holds no trip at all.
class _NoTrip extends StatelessWidget {
  final VoidCallback onCreate;

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
            ],
          ),
        ),
      ),
    );
  }
}
