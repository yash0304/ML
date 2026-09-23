// test/stop_leg_detail_test.dart — issues #51 and #52.
//
// The two detail screens are assembly over data that already exists, which is
// exactly where a number quietly starts meaning something other than its
// label. These tests pin the labels to the numbers.

import 'package:drift/drift.dart' hide isNull, isNotNull;
import 'package:drift/native.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:safarsathi/core/database/app_database.dart';
import 'package:safarsathi/core/theme/app_tokens.dart';
import 'package:safarsathi/features/discovery/data/discovery.dart';
import 'package:safarsathi/features/weather/data/weather_client.dart';
import 'package:safarsathi/features/trips/data/stop_detail.dart';
import 'package:safarsathi/features/trips/data/trip_editor.dart';
import 'package:safarsathi/features/trips/presentation/leg_detail_screen.dart';
import 'package:safarsathi/features/trips/presentation/stop_detail_screen.dart';

/// Both screens are long lists and most assertions sit below a phone's fold.
void useTallSurface(WidgetTester tester) {
  tester.view.physicalSize = const Size(420, 2400);
  tester.view.devicePixelRatio = 1.0;
  addTearDown(tester.view.reset);
}

Widget wrap(Widget child) => MaterialApp(theme: AppTokens.light, home: child);

final _cached = DateTime(2025, 10, 1, 8);
final _now = DateTime(2025, 10, 1, 20);

WeatherSnapshot _day(
  int id,
  DateTime date,
  String condition, {
  DateTime? cachedAt,
  double? rain,
}) => WeatherSnapshot(
  id: id,
  stopId: 1,
  forDate: date,
  condition: condition,
  tempMinC: 17,
  tempMaxC: 24,
  rainMm: rain,
  cachedAt: cachedAt ?? _cached,
);

Stop _stop({String tags = 'trek,caves', int nights = 2, String? note}) => Stop(
  id: 1,
  tripId: 1,
  name: 'Cherrapunji',
  sequenceOrder: 2,
  nights: nights,
  countryCode: 'IN',
  activityTags: tags,
  arrivalDate: DateTime(2025, 10, 2),
  lat: 25.27,
  lon: 91.72,
  note: note,
);

StopDetail _detail({
  List<WeatherSnapshot>? weather,
  DateTime? cachedAt,
  int diary = 4,
  int unconfirmed = 2,
  int checklistCount = 12,
  int checklistDone = 5,
  int places = 9,
  DateTime? synced,
  String tags = 'trek,caves',
  String? note,
}) => StopDetail(
  stop: _stop(tags: tags, note: note),
  weather: weather ?? [_day(1, DateTime(2025, 10, 2), 'Heavy rain', rain: 41)],
  weatherCachedAt: cachedAt ?? _cached,
  diaryCount: diary,
  unconfirmedCount: unconfirmed,
  checklistCount: checklistCount,
  checklistDone: checklistDone,
  nearbyPlaceCount: places,
  lastSyncedAt: synced,
);

CorridorPlace _place(
  int id,
  String name,
  double along, {
  String category = 'fuel',
  bool phone = false,
}) => CorridorPlace(
  id: id,
  name: name,
  category: category,
  lat: 25.3,
  lon: 91.7,
  alongRouteKm: along,
  offRouteKm: 0.4,
  phones: phone
      ? [
          PoiContact(
            id: id,
            poiId: id,
            phoneRaw: '03637 000000',
            phoneE164: '+913637000000',
            tier: 'communityOsm',
          ),
        ]
      : const [],
);

LegDiscovery _leg({
  List<CorridorPlace>? places,
  DateTime? synced,
  double? distanceKm = 54,
}) => LegDiscovery(
  legId: 1,
  fromName: 'Shillong',
  toName: 'Cherrapunji',
  distanceKm: distanceKm,
  lastSyncedAt: synced,
  places: places ?? const [],
);

void main() {
  group('#51 — what the stop screen holds', () {
    testWidgets('a fresh forecast carries its age and no warning', (
      tester,
    ) async {
      useTallSurface(tester);
      await tester.pumpWidget(
        wrap(
          StopDetailScreen(
            detail: Stream.value(_detail()),
            now: _now,
          ),
        ),
      );
      await tester.pump();

      expect(find.text('TAKEN 12 HOURS AGO'), findsOneWidget);
      expect(find.textContaining('More than three days old'), findsNothing);
      expect(find.text('Heavy rain'), findsOneWidget);
      expect(find.text('17–24°'), findsOneWidget);
      expect(find.text('41'), findsOneWidget);
    });

    testWidgets('PAST THREE DAYS the forecast says so in words', (
      tester,
    ) async {
      useTallSurface(tester);
      final old = _now.subtract(const Duration(days: 4));
      await tester.pumpWidget(
        wrap(
          StopDetailScreen(
            detail: Stream.value(
              _detail(
                cachedAt: old,
                weather: [_day(1, DateTime(2025, 10, 2), 'Rain', cachedAt: old)],
              ),
            ),
            now: _now,
          ),
        ),
      );
      await tester.pump();

      // The number alone is not the point — the sentence is, because a
      // stale forecast that looks current is what this screen exists against.
      expect(find.textContaining('More than three days old'), findsOneWidget);
      expect(find.text('TAKEN 4 DAYS AGO'), findsOneWidget);
    });

    testWidgets('three days exactly is already stale, not ageing', (
      tester,
    ) async {
      // SCREENS.md §10 draws the line at three days. Off-by-one here means a
      // seventy-two-hour-old forecast renders as if it were current.
      final at = _now.subtract(const Duration(days: 3));
      expect(_detail(cachedAt: at).weatherStaleness(now: _now), Staleness.stale);
      expect(
        _detail(
          cachedAt: _now.subtract(const Duration(days: 2, hours: 23)),
        ).weatherStaleness(now: _now),
        Staleness.ageing,
      );
    });

    testWidgets('nothing downloaded reads as nothing downloaded', (
      tester,
    ) async {
      useTallSurface(tester);
      await tester.pumpWidget(
        wrap(
          StopDetailScreen(
            detail: Stream.value(
              StopDetail(
                stop: _stop(),
                weather: const [],
                diaryCount: 0,
                unconfirmedCount: 0,
                checklistCount: 0,
                checklistDone: 0,
                nearbyPlaceCount: 0,
              ),
            ),
            now: _now,
          ),
        ),
      );
      await tester.pump();

      expect(
        find.text('No forecast downloaded for this stop.'),
        findsOneWidget,
      );
      expect(
        find.textContaining('have not been\ndownloaded yet.'),
        findsNothing,
      );
      expect(find.textContaining('not been'), findsOneWidget);
    });

    testWidgets('the unconfirmed count rides the diary row', (tester) async {
      useTallSurface(tester);
      await tester.pumpWidget(
        wrap(
          StopDetailScreen(
            detail: Stream.value(_detail(diary: 4, unconfirmed: 2)),
            now: _now,
            onOpenDiary: () {},
          ),
        ),
      );
      await tester.pump();

      expect(find.text('Diary entries'), findsOneWidget);
      expect(find.text('4'), findsOneWidget);
      expect(find.text('2 unconfirmed'), findsOneWidget);
      // "5 of 12", never a bare "5" that could be read as five items.
      expect(find.text('5 of 12'), findsOneWidget);
    });

    testWidgets('a stop with nothing unconfirmed says nothing about it', (
      tester,
    ) async {
      useTallSurface(tester);
      await tester.pumpWidget(
        wrap(
          StopDetailScreen(
            detail: Stream.value(_detail(diary: 4, unconfirmed: 0)),
            now: _now,
          ),
        ),
      );
      await tester.pump();
      expect(find.textContaining('unconfirmed'), findsNothing);
    });

    testWidgets('tapping a tag hands back the whole new set', (tester) async {
      useTallSurface(tester);
      List<String>? handed;
      await tester.pumpWidget(
        wrap(
          StopDetailScreen(
            detail: Stream.value(_detail(tags: 'trek')),
            now: _now,
            onTags: (tags) async => handed = tags,
          ),
        ),
      );
      await tester.pump();

      await tester.tap(
        find.descendant(of: find.byType(Wrap), matching: find.text('CAVES')),
      );
      await tester.pump();
      expect(handed, ['trek', 'caves']);

      await tester.tap(
        find.descendant(of: find.byType(Wrap), matching: find.text('TREK')),
      );
      await tester.pump();
      // Still from the stream's ['trek'], because the screen is not the
      // source of truth for its own tags — the database is.
      expect(handed, isEmpty);
    });

    testWidgets('THE SENTENCE THAT LINKS TAGS TO THE CHECKLIST is present', (
      tester,
    ) async {
      useTallSurface(tester);
      await tester.pumpWidget(
        wrap(StopDetailScreen(detail: Stream.value(_detail()), now: _now)),
      );
      await tester.pump();
      // Nobody would guess a row of chips rebuilds the packing list.
      expect(
        find.textContaining('These build the packing checklist'),
        findsOneWidget,
      );
    });

    testWidgets('an undownloaded stop flags its cache in caution', (
      tester,
    ) async {
      useTallSurface(tester);
      await tester.pumpWidget(
        wrap(
          StopDetailScreen(
            detail: Stream.value(_detail(synced: null)),
            now: _now,
          ),
        ),
      );
      await tester.pump();
      expect(
        find.textContaining('have not been downloaded yet'),
        findsOneWidget,
      );

      await tester.pumpWidget(
        wrap(
          StopDetailScreen(
            detail: Stream.value(_detail(synced: DateTime(2025, 9, 28))),
            now: _now,
          ),
        ),
      );
      await tester.pump();
      expect(
        find.textContaining('downloaded 28/09/2025, with 9 places'),
        findsOneWidget,
      );
    });
  });

  group('#52 — what the leg screen holds', () {
    testWidgets('an empty transport form says nothing is recorded', (
      tester,
    ) async {
      useTallSurface(tester);
      await tester.pumpWidget(
        wrap(
          LegDetailScreen(
            discovery: Stream.value(_leg()),
            transport: Stream.value(const LegTransport()),
          ),
        ),
      );
      await tester.pump();

      expect(find.text('Nothing recorded for this leg yet.'), findsOneWidget);
      // AND SAYS WHY IT IS EMPTY, rather than implying a lookup.
      expect(find.textContaining('Typed by you'), findsOneWidget);
      expect(find.text('Shillong → Cherrapunji'), findsOneWidget);
    });

    testWidgets('typed transport renders every field', (tester) async {
      useTallSurface(tester);
      await tester.pumpWidget(
        wrap(
          LegDetailScreen(
            discovery: Stream.value(_leg()),
            transport: Stream.value(
              LegTransport(
                mode: 'Shared sumo',
                plannedDeparture: DateTime(2025, 10, 2, 7, 30),
                plannedArrival: DateTime(2025, 10, 2, 9, 45),
                isBooked: false,
                note: 'Bara Bazar stand, ask for the Sohra counter.',
              ),
            ),
          ),
        ),
      );
      // Twice: the discovery stream resolves the scaffold, and the transport
      // stream inside it resolves on the frame after that.
      await tester.pump();
      await tester.pump();

      expect(find.text('Shared sumo'), findsOneWidget);
      expect(find.text('02/10  07:30'), findsOneWidget);
      expect(find.text('02/10  09:45'), findsOneWidget);
      expect(find.text('Not yet'), findsOneWidget);
      expect(find.textContaining('Bara Bazar'), findsOneWidget);
    });

    testWidgets('places are ORDERED ALONG THE ROUTE and capped at five', (
      tester,
    ) async {
      useTallSurface(tester);
      await tester.pumpWidget(
        wrap(
          LegDetailScreen(
            discovery: Stream.value(
              _leg(
                synced: DateTime(2025, 9, 30),
                places: [
                  for (var i = 0; i < 7; i++)
                    _place(i + 1, 'Place ${i + 1}', (i + 1) * 6),
                ],
              ),
            ),
            transport: Stream.value(const LegTransport()),
          ),
        ),
      );
      await tester.pump();

      expect(find.text('6 km'), findsOneWidget);
      expect(find.text('30 km'), findsOneWidget);
      expect(find.text('Place 5'), findsOneWidget);
      expect(find.text('Place 6'), findsNothing);
      expect(find.text('2 more further along.'), findsOneWidget);
    });

    testWidgets('AN OSM NUMBER KEEPS THE AMBER DOT', (tester) async {
      useTallSurface(tester);
      await tester.pumpWidget(
        wrap(
          LegDetailScreen(
            discovery: Stream.value(
              _leg(
                synced: DateTime(2025, 9, 30),
                places: [
                  _place(1, 'Duwan Sing petrol pump', 12, phone: true),
                  _place(2, 'Roadside dhaba', 20, category: 'food'),
                ],
              ),
            ),
            transport: Stream.value(const LegTransport()),
          ),
        ),
      );
      await tester.pump();

      // One place has a number, one does not, so exactly one dot. Nothing
      // community-contributed is ever presented as verified.
      final dots = tester.widgetList<Container>(find.byType(Container)).where((
        w,
      ) {
        final d = w.decoration;
        return d is BoxDecoration && d.shape == BoxShape.circle;
      });
      expect(dots.length, 1);
    });

    testWidgets('NOT DOWNLOADED and NOTHING THERE are different sentences', (
      tester,
    ) async {
      useTallSurface(tester);
      await tester.pumpWidget(
        wrap(
          LegDetailScreen(
            discovery: Stream.value(_leg(synced: null)),
            transport: Stream.value(const LegTransport()),
          ),
        ),
      );
      await tester.pump();
      expect(
        find.text('This leg has not been downloaded yet.'),
        findsOneWidget,
      );

      await tester.pumpWidget(
        wrap(
          LegDetailScreen(
            discovery: Stream.value(_leg(synced: DateTime(2025, 9, 30))),
            transport: Stream.value(const LegTransport()),
          ),
        ),
      );
      await tester.pump();
      expect(
        find.text('Nothing tagged along this road in OpenStreetMap.'),
        findsOneWidget,
      );
    });

    testWidgets('the cache section reports no per-leg tile figure', (
      tester,
    ) async {
      useTallSurface(tester);
      await tester.pumpWidget(
        wrap(
          LegDetailScreen(
            discovery: Stream.value(
              _leg(synced: DateTime(2025, 9, 30), places: [_place(1, 'A', 3)]),
            ),
            transport: Stream.value(const LegTransport()),
          ),
        ),
      );
      await tester.pump();

      expect(find.text('54 km'), findsOneWidget);
      expect(find.text('30/09/2025'), findsOneWidget);
      // Adjacent legs share ground; a per-leg tile figure would double-count.
      expect(find.textContaining('counted for the whole trip'), findsOneWidget);
    });

    testWidgets('SEE ALL only appears when there is something to see', (
      tester,
    ) async {
      useTallSurface(tester);
      await tester.pumpWidget(
        wrap(
          LegDetailScreen(
            discovery: Stream.value(_leg(synced: DateTime(2025, 9, 30))),
            transport: Stream.value(const LegTransport()),
            onSeeAll: () {},
          ),
        ),
      );
      await tester.pump();
      expect(find.text('SEE ALL'), findsNothing);
    });
  });

  group('#51 — the query behind the screen', () {
    late AppDatabase db;
    late TripEditor editor;

    setUp(() {
      db = AppDatabase(NativeDatabase.memory());
      editor = TripEditor(db);
    });
    tearDown(() => db.close());

    test('counts come from the right tables', () async {
      final tripId = await editor.createTrip(name: 'Meghalaya');
      final a = await editor.addStop(
        tripId,
        const StopDraft(name: 'Shillong', nights: 2),
      );
      final b = await editor.addStop(
        tripId,
        const StopDraft(name: 'Cherrapunji', nights: 1),
      );
      final legs = await (db.select(
        db.legs,
      )..where((l) => l.tripId.equals(tripId))).get();
      expect(legs, hasLength(1));

      await db
          .into(db.contacts)
          .insert(
            ContactsCompanion.insert(
              tripId: Value(tripId),
              stopId: Value(b),
              name: 'Sohra homestay',
              phoneRaw: '03637 000000',
              tier: const Value('userEntered'),
            ),
          );
      await db
          .into(db.weatherSnapshots)
          .insert(
            WeatherSnapshotsCompanion.insert(
              stopId: b,
              forDate: DateTime(2025, 10, 3),
              condition: 'Heavy rain',
              cachedAt: _cached,
            ),
          );

      final detail = await watchStopDetail(db, b).first;
      expect(detail.stop.name, 'Cherrapunji');
      expect(detail.diaryCount, 1);
      // Imported or typed is never confirmed, so this stop still blocks.
      expect(detail.unconfirmedCount, 1);
      expect(detail.weather, hasLength(1));
      expect(detail.weatherCachedAt, _cached);
      // The leg between a and b touches b, and has not been synced.
      expect(detail.lastSyncedAt, isNull);
      expect(detail.nearbyPlaceCount, 0);
      expect(a, isNot(b));
    });

    test('the stream fires when a table it only counts changes', () async {
      // The Drift trap this project has now been bitten by three times: a
      // stream fires only for the tables its own query touches, and this
      // query touches six.
      final tripId = await editor.createTrip(name: 'Meghalaya');
      final stopId = await editor.addStop(
        tripId,
        const StopDraft(name: 'Shillong', nights: 1),
      );

      final seen = <int>[];
      final sub = watchStopDetail(
        db,
        stopId,
      ).listen((d) => seen.add(d.diaryCount));
      await Future<void>.delayed(const Duration(milliseconds: 60));

      await db
          .into(db.contacts)
          .insert(
            ContactsCompanion.insert(
              tripId: Value(tripId),
              stopId: Value(stopId),
              name: 'Police Bazar chemist',
              phoneRaw: '0364 000000',
              tier: const Value('userEntered'),
            ),
          );
      await Future<void>.delayed(const Duration(milliseconds: 60));
      await sub.cancel();

      expect(seen.last, 1);
      expect(seen.length, greaterThan(1));
    });

    test('transport reads straight off the leg row', () async {
      final tripId = await editor.createTrip(name: 'Meghalaya');
      await editor.addStop(tripId, const StopDraft(name: 'Shillong'));
      await editor.addStop(tripId, const StopDraft(name: 'Cherrapunji'));
      final leg = (await (db.select(
        db.legs,
      )..where((l) => l.tripId.equals(tripId))).get()).single;

      await editor.updateLeg(leg.id, mode: 'Shared sumo', isBooked: true);
      final t = await watchLegTransport(db, leg.id).first;
      expect(t.mode, 'Shared sumo');
      expect(t.isBooked, isTrue);
      expect(t.isEmpty, isFalse);
    });
  });

  group('the stop note is readable, not just writable', () {
    // THE STOP FORM HAS HAD A NOTE BOX SINCE #16 AND NOTHING RENDERED IT.
    // Whatever was typed went into the database and out of reach — the same
    // shape of bug as a map screen wired to nothing. Found while putting the
    // Meghalaya places-with-no-number list into stop notes, which would have
    // been typed in and then invisible.

    testWidgets('a note the user wrote is shown', (tester) async {
      useTallSurface(tester);
      await tester.pumpWidget(
        wrap(
          StopDetailScreen(
            detail: Stream.value(
              _detail(note: 'Erica Pharmacy, Dukan Rd. No phone — walk in.'),
            ),
            now: _now,
          ),
        ),
      );
      await tester.pump();

      expect(
        find.text('Erica Pharmacy, Dukan Rd. No phone — walk in.'),
        findsOneWidget,
      );
    });

    testWidgets('a long note is not truncated', (tester) async {
      useTallSurface(tester);
      final long = List.generate(8, (i) => 'Line $i of what is here').join('\n');
      await tester.pumpWidget(
        wrap(
          StopDetailScreen(detail: Stream.value(_detail(note: long)), now: _now),
        ),
      );
      await tester.pump();

      final text = tester.widget<Text>(find.text(long));
      expect(
        text.maxLines,
        isNull,
        reason: 'a hospital three villages away, truncated at one line, is '
            'worse than not written down',
      );
    });

    testWidgets('no note renders nothing at all', (tester) async {
      useTallSurface(tester);
      await tester.pumpWidget(
        wrap(StopDetailScreen(detail: Stream.value(_detail()), now: _now)),
      );
      await tester.pump();

      // Not an empty box, not a heading with nothing under it.
      expect(find.byType(SizedBox), findsWidgets);
      expect(find.textContaining('Line 0 of'), findsNothing);
    });

    testWidgets('a note of only whitespace is treated as absent', (
      tester,
    ) async {
      useTallSurface(tester);
      await tester.pumpWidget(
        wrap(
          StopDetailScreen(
            detail: Stream.value(_detail(note: '   \n  ')),
            now: _now,
          ),
        ),
      );
      await tester.pump();

      expect(find.text('   \n  '), findsNothing);
    });
  });

  group('your numbers on the leg', () {
    Contact diary(
      int id,
      String name, {
      String category = 'hospital',
      String? note,
      bool confirmed = false,
    }) => Contact(
      id: id,
      name: name,
      phoneRaw: '+91 90000 0000$id',
      category: category,
      tier: 'userEntered',
      callConfirmed: confirmed,
      isPinned: false,
      isEmergency: false,
      hasWhatsapp: false,
      callCount: 0,
      createdAt: DateTime(2026, 9, 22),
      note: note,
    );

    LegDiscovery legWith({
      List<LegContact> onTheWay = const [],
      List<Contact> atDestination = const [],
      bool canPlace = true,
      int unplaced = 0,
      bool anyPlaced = false,
      List<NearbyHelp> nearestHelp = const [],
    }) => LegDiscovery(
      legId: 1,
      fromName: 'Shillong',
      toName: 'Dawki',
      places: const [],
      onTheWay: onTheWay,
      atDestination: atDestination,
      canPlace: canPlace,
      unplaced: unplaced,
      anyPlaced: anyPlaced,
      nearestHelp: nearestHelp,
    );

    Future<void> pump(WidgetTester tester, LegDiscovery leg,
        {void Function(Contact)? onOpen}) async {
      useTallSurface(tester);
      await tester.pumpWidget(
        wrap(
          LegDetailScreen(
            discovery: Stream.value(leg),
            transport: Stream.value(const LegTransport()),
            onOpenContact: onOpen,
          ),
        ),
      );
      await tester.pump();
    }

    testWidgets('a number on the way shows its km, number and note', (
      tester,
    ) async {
      await pump(
        tester,
        legWith(
          onTheWay: [
            LegContact(
              contact: diary(
                1,
                'Pynursla Sub-Divisional Hospital',
                note: '24x7. Govt source: East Khasi Hills District',
              ),
              alongRouteKm: 38.4,
              offRouteKm: 0.3,
            ),
          ],
        ),
      );

      expect(find.text('YOUR NUMBERS ON THE WAY'), findsOneWidget);
      expect(find.text('38 km'), findsOneWidget);
      expect(find.text('Pynursla Sub-Divisional Hospital'), findsOneWidget);
      expect(find.text('HOSPITAL · +91 90000 00001'), findsOneWidget);
      expect(
        find.text('24x7. Govt source: East Khasi Hills District'),
        findsOneWidget,
      );
    });

    testWidgets('tapping a number opens that diary entry', (tester) async {
      Contact? opened;
      final hospital = diary(4, 'Dawki PHC');
      await pump(
        tester,
        legWith(atDestination: [hospital]),
        onOpen: (c) => opened = c,
      );
      await tester.tap(find.text('Dawki PHC'));
      expect(opened?.id, 4);
    });

    testWidgets('an unconfirmed number carries the amber dot, a confirmed '
        'one does not', (tester) async {
      await pump(
        tester,
        legWith(
          atDestination: [
            diary(1, 'Not called yet'),
            diary(2, 'Called and confirmed', confirmed: true),
          ],
        ),
      );
      // Inside a list each row's semantics merge into one node, so the dot
      // is announced as part of its row — "Not called yet, … Not confirmed
      // yet" — which is what a screen reader should say. One row says it;
      // the confirmed row must not.
      final flagged = find.bySemanticsLabel(RegExp('Not confirmed yet'));
      expect(flagged, findsOneWidget);
      expect(
        tester.getSemantics(flagged).label,
        contains('Not called yet'),
      );
    });

    testWidgets('EMPTY BECAUSE NOTHING HAS A LOCATION says so, and says '
        'what to do', (tester) async {
      // The state of every contact imported before coordinates were read.
      // "None of your numbers lie along this road" would be false.
      await pump(tester, legWith(unplaced: 12));
      expect(find.textContaining('12 of your numbers have no location'),
          findsOneWidget);
      expect(find.textContaining('Latitude and Longitude'), findsOneWidget);
      expect(find.textContaining('None of your numbers lie'), findsNothing);
    });

    testWidgets('empty because the leg cannot be measured says that instead',
        (tester) async {
      await pump(tester, legWith(canPlace: false, unplaced: 3));
      expect(find.textContaining('nothing can be placed along it yet'),
          findsOneWidget);
    });

    testWidgets('genuinely empty is the only case that says nothing is there',
        (tester) async {
      await pump(tester, legWith());
      expect(find.text('None of your numbers lie along this road.'),
          findsOneWidget);
    });

    testWidgets('the destination list stops at eight and points at the diary',
        (tester) async {
      await pump(
        tester,
        legWith(atDestination: [
          for (var i = 1; i <= 11; i++) diary(i, 'Place $i'),
        ]),
      );
      expect(find.text('YOUR NUMBERS AT DAWKI'), findsOneWidget);
      expect(find.text('Place 8'), findsOneWidget);
      expect(find.text('Place 9'), findsNothing);
      expect(find.text('3 more at Dawki — all of them are in the diary.'),
          findsOneWidget);
    });

    testWidgets('once anything is placed, the re-import hint stops', (
      tester,
    ) async {
      await pump(tester, legWith(unplaced: 4, anyPlaced: true));
      expect(find.textContaining('no location'), findsNothing);
      expect(find.text('None of your numbers lie along this road.'),
          findsOneWidget);
    });

    testWidgets('nearest help names the stop and says the km are straight-line',
        (tester) async {
      await pump(
        tester,
        legWith(nearestHelp: [
          NearbyHelp(
            contact: diary(7, 'MK Pharmacy', category: 'pharmacy'),
            straightLineKm: 13.2,
          ),
        ]),
      );
      expect(find.text('NEAREST HELP TO DAWKI'), findsOneWidget);
      expect(find.textContaining('straight-line'), findsOneWidget);
      expect(find.text('PHARMACY · +91 90000 00007 · 13 KM AWAY'),
          findsOneWidget);
    });

    testWidgets('no nearest-help section when there is none', (tester) async {
      await pump(tester, legWith());
      expect(find.textContaining('NEAREST HELP'), findsNothing);
    });

    testWidgets('an empty destination says so by name', (tester) async {
      await pump(tester, legWith());
      expect(find.text('Nothing in your diary at Dawki yet.'), findsOneWidget);
    });
  });
}
