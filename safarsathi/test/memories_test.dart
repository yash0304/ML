// test/memories_test.dart — the Memories tab, the trip's photos as an album.

import 'package:drift/native.dart';
import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:safarsathi/core/database/app_database.dart';
import 'package:safarsathi/core/theme/app_tokens.dart';
import 'package:safarsathi/core/widgets/app_shell.dart';
import 'package:safarsathi/features/memories/data/memories.dart';
import 'package:safarsathi/features/memories/presentation/add_memories_screen.dart';
import 'package:safarsathi/features/memories/presentation/memories_screen.dart';
import 'package:safarsathi/features/trips/data/timeline.dart';
import 'package:safarsathi/features/trips/data/trip_editor.dart';

TimelineEntry note(
  int id,
  DateTime at, {
  int? stopId,
  String? body,
  String? photos,
  String kind = TimelineKind.note,
}) => TimelineEntry(
  id: id,
  tripId: 1,
  stopId: stopId,
  kind: kind,
  body: body,
  photoPaths: photos,
  occurredAt: at,
);

Stop stop(int id, String name, {DateTime? arrives, int nights = 1}) => Stop(
  id: id,
  tripId: 1,
  name: name,
  sequenceOrder: id,
  countryCode: 'IN',
  nights: nights,
  activityTags: '',
  arrivalDate: arrives,
);

Widget box(String path) => ColoredBox(
  color: const Color(0xFF888888),
  child: Text(path.split('/').last),
);

void main() {
  group('the album', () {
    final stops = {
      1: stop(1, 'Shillong'),
      2: stop(2, 'Cherrapunji'),
      3: stop(3, 'Dawki'),
    };

    test('every photo on its own, by day, oldest first, with its places', () {
      final days = buildMemories([
        note(3, DateTime(2026, 10, 3, 15), stopId: 3, photos: '/p/e.jpg'),
        note(1, DateTime(2026, 10, 2, 9), stopId: 2,
            photos: '/p/a.jpg\n/p/b.jpg', body: 'Root bridge'),
        note(2, DateTime(2026, 10, 2, 16), stopId: 2, photos: '/p/c.jpg'),
        note(4, DateTime(2026, 10, 2, 18), stopId: 1, photos: '/p/d.jpg'),
        // Words alone are a note, not a memory.
        note(5, DateTime(2026, 10, 2, 19), body: 'Rain all evening'),
        // A route-log point never is.
        note(6, DateTime(2026, 10, 2, 20), kind: TimelineKind.fix),
      ], stops);

      expect(days, hasLength(2));
      expect(days[0].day, DateTime(2026, 10, 2));
      expect(days[0].photos.map((m) => m.path), [
        '/p/a.jpg',
        '/p/b.jpg',
        '/p/c.jpg',
        '/p/d.jpg',
      ]);
      expect(days[0].places, ['Cherrapunji', 'Shillong']);
      expect(days[0].photos.first.caption, 'Root bridge');
      expect(days[0].photos[1].caption, 'Root bridge');
      expect(days[1].places, ['Dawki']);
    });

    test('a photo filed nowhere has no place, and the day still shows', () {
      final days = buildMemories([
        note(1, DateTime(2026, 10, 4, 9), photos: '/p/a.jpg'),
      ], stops);
      expect(days.single.places, isEmpty);
      expect(days.single.photos.single.place, isNull);
    });
  });

  group('when photos added are filed', () {
    final sohra = stop(2, 'Cherrapunji', arrives: DateTime(2026, 10, 2));

    test('during the stay: now', () {
      final now = DateTime(2026, 10, 2, 17, 40);
      expect(memoryDate(sohra, now), now);
      // The night there runs into the next morning.
      final morning = DateTime(2026, 10, 3, 8);
      expect(memoryDate(sohra, morning), morning);
    });

    test('afterwards, at home: the day the trip reached that place', () {
      expect(
        memoryDate(sohra, DateTime(2026, 10, 6, 21)),
        DateTime(2026, 10, 2, 12),
      );
    });

    test('a stop with no date, or none chosen: now', () {
      final now = DateTime(2026, 10, 6, 21);
      expect(memoryDate(stop(1, 'Shillong'), now), now);
      expect(memoryDate(null, now), now);
    });
  });

  group('stored', () {
    late AppDatabase db;
    late int tripId;
    setUp(() async {
      db = AppDatabase(NativeDatabase.memory());
      tripId = await TripEditor(db).createTrip(name: 'Meghalaya');
    });
    tearDown(() => db.close());

    Future<List<MemoryDay>> album() => watchMemories(db, tripId).first;

    test('photos added show in the album and on the Timeline', () async {
      await addTimelineNote(db, tripId: tripId, text: 'Living root bridge',
          photoPaths: ['/p/a.jpg', '/p/b.jpg'], at: DateTime(2026, 10, 2));
      await addTimelineNote(db, tripId: tripId, text: 'Just words');
      final days = await album();
      expect(days.single.photos, hasLength(2));
      final timeline = await watchTimeline(db, tripId).first;
      expect(timeline.expand((d) => d.items), hasLength(2));
    });

    test('deleting one photo keeps the others and the words', () async {
      await addTimelineNote(db, tripId: tripId, text: 'Root bridge',
          photoPaths: ['/p/a.jpg', '/p/b.jpg']);
      final first = (await album()).single.photos.first;
      await removeMemory(db, first);
      final left = (await album()).single.photos.single;
      expect(left.path, '/p/b.jpg');
      expect(left.caption, 'Root bridge');
    });

    test('the last photo of a note with words leaves the words', () async {
      await addTimelineNote(db, tripId: tripId, text: 'Root bridge',
          photoPaths: ['/p/a.jpg']);
      await removeMemory(db, (await album()).single.photos.single);
      expect(await album(), isEmpty);
      final rows = await db.select(db.timelineEntries).get();
      expect(rows.single.body, 'Root bridge');
      expect(rows.single.photoPaths, isNull);
    });

    test('the last photo of a note without words removes it', () async {
      await addTimelineNote(db, tripId: tripId, text: '',
          photoPaths: ['/p/a.jpg']);
      await removeMemory(db, (await album()).single.photos.single);
      expect(await db.select(db.timelineEntries).get(), isEmpty);
    });

    test('words can be changed, and cleared', () async {
      final id = await addTimelineNote(db, tripId: tripId, text: 'Bridge',
          photoPaths: ['/p/a.jpg']);
      await setMemoryCaption(db, id, '  Double-decker root bridge ');
      expect((await album()).single.photos.single.caption,
          'Double-decker root bridge');
      await setMemoryCaption(db, id, '   ');
      expect((await album()).single.photos.single.caption, isNull);
    });
  });

  group('the tab', () {
    Future<void> pump(WidgetTester t, Widget home) async {
      t.view.physicalSize = const Size(400, 900);
      t.view.devicePixelRatio = 1;
      addTearDown(t.view.reset);
      await t.pumpWidget(MaterialApp(theme: AppTokens.light, home: home));
      await t.pump();
    }

    MemoriesScreen screen(
      List<MemoryDay> days, {
      VoidCallback? onAdd,
      List<Memory>? deleted,
      List<Memory>? shared,
      List<(int, String)>? captions,
    }) => MemoriesScreen(
      tripName: 'Meghalaya',
      memories: Stream.value(days),
      onAdd: onAdd ?? () {},
      thumb: box,
      photo: box,
      onShare: (m) async => shared?.add(m),
      onDelete: (m) async => deleted?.add(m),
      onCaption: (id, text) async => captions?.add((id, text)),
    );

    final days = buildMemories([
      note(1, DateTime(2026, 10, 2, 9), stopId: 2,
          photos: '/p/a.jpg\n/p/b.jpg', body: 'Root bridge'),
      note(2, DateTime(2026, 10, 3, 15), stopId: 3, photos: '/p/c.jpg'),
    ], {2: stop(2, 'Cherrapunji'), 3: stop(3, 'Dawki')});

    testWidgets('empty: says what it is for', (t) async {
      var added = 0;
      await pump(t, screen(const [], onAdd: () => added++));
      expect(find.text('No photos yet'), findsOneWidget);
      expect(find.textContaining('Keep a few photos'), findsOneWidget);
      await t.tap(find.byKey(const Key('memories-add')));
      expect(added, 1);
    });

    testWidgets('the album: counts, days with places, a tile per photo',
        (t) async {
      await pump(t, screen(days));
      expect(find.text('3 photos · 2 days'), findsOneWidget);
      expect(find.text('2 OCT · CHERRAPUNJI'), findsOneWidget);
      expect(find.text('3 OCT · DAWKI'), findsOneWidget);
      expect(find.byKey(const Key('memory-/p/b.jpg')), findsOneWidget);
      expect(find.textContaining('not in the backup'), findsOneWidget);
    });

    testWidgets('a photo opens whole, with its words, and swipes on',
        (t) async {
      await pump(t, screen(days));
      await t.tap(find.byKey(const Key('memory-/p/b.jpg')));
      await t.pumpAndSettle();
      expect(find.text('2 OF 3'), findsNothing); // the stencil does not shout
      expect(find.text('2 of 3'), findsOneWidget);
      expect(find.text('Root bridge'), findsOneWidget);
      expect(find.textContaining('Cherrapunji'), findsOneWidget);
      await t.fling(find.byType(PageView), const Offset(-300, 0), 1000);
      await t.pumpAndSettle();
      expect(find.text('3 of 3'), findsOneWidget);
      expect(find.text('Add a few words'), findsOneWidget);
    });

    testWidgets('share, change the words, delete after asking', (t) async {
      final deleted = <Memory>[];
      final shared = <Memory>[];
      final captions = <(int, String)>[];
      await pump(t, screen(days,
          deleted: deleted, shared: shared, captions: captions));
      await t.tap(find.byKey(const Key('memory-/p/a.jpg')));
      await t.pumpAndSettle();

      await t.tap(find.byKey(const Key('memory-share')));
      expect(shared.single.path, '/p/a.jpg');

      await t.tap(find.byKey(const Key('memory-caption')));
      await t.pumpAndSettle();
      expect(find.text('Shown under every photo added with this one'),
          findsOneWidget);
      await t.enterText(
          find.byKey(const Key('memory-caption-field')), 'Umshiang bridge');
      await t.tap(find.byKey(const Key('memory-caption-save')));
      await t.pumpAndSettle();
      expect(captions.single, (1, 'Umshiang bridge'));
      expect(find.text('Umshiang bridge'), findsOneWidget);

      await t.tap(find.byKey(const Key('memory-delete')));
      await t.pumpAndSettle();
      expect(find.textContaining('gallery'), findsOneWidget);
      await t.tap(find.byKey(const Key('memory-delete-confirm')));
      await t.pumpAndSettle();
      expect(deleted.single.path, '/p/a.jpg');
      expect(find.text('1 of 2'), findsOneWidget);
    });
  });

  group('adding', () {
    final stops = [
      stop(1, 'Shillong'),
      stop(2, 'Cherrapunji'),
    ];

    Future<void> pump(WidgetTester t, Widget screen) async {
      t.view.physicalSize = const Size(400, 900);
      t.view.devicePixelRatio = 1;
      addTearDown(t.view.reset);
      await t.pumpWidget(MaterialApp(
        theme: AppTokens.light,
        home: Builder(
          builder: (context) => Scaffold(
            body: TextButton(
              onPressed: () => Navigator.of(context).push(
                MaterialPageRoute<void>(builder: (_) => screen),
              ),
              child: const Text('open'),
            ),
          ),
        ),
      ));
      await t.tap(find.text('open'));
      await t.pumpAndSettle();
    }

    testWidgets('choose several, file under a place, add words, keep',
        (t) async {
      final saved = <(List<String>, String, int?)>[];
      final discarded = <String>[];
      await pump(t, AddMemoriesScreen(
        stops: stops,
        initialStopId: 1,
        pickMany: () async => ['/p/1.jpg', '/p/2.jpg', '/p/3.jpg'],
        takeOne: () async => '/p/cam.jpg',
        thumb: box,
        onDiscard: (p) async => discarded.addAll(p),
        onSave: (photos, words, stopId) async =>
            saved.add((photos, words, stopId)),
      ));
      expect(find.text('Choose a photo first'), findsOneWidget);
      await t.tap(find.byKey(const Key('memory-save')));
      await t.pump();
      expect(saved, isEmpty);

      await t.tap(find.byKey(const Key('memory-choose')));
      await t.pump();
      await t.tap(find.byKey(const Key('memory-camera')));
      await t.pump();
      expect(find.text('Keep these 4 photos'), findsOneWidget);

      await t.tap(find.byKey(const Key('memory-remove-/p/2.jpg')));
      await t.pump();
      expect(discarded, ['/p/2.jpg']);
      expect(find.text('Keep these 3 photos'), findsOneWidget);

      await t.tap(find.byKey(const Key('memory-stop-2')));
      await t.enterText(find.byKey(const Key('memory-words')), 'Nohkalikai');
      await t.tap(find.byKey(const Key('memory-save')));
      await t.pumpAndSettle();
      expect(saved.single.$1, ['/p/1.jpg', '/p/3.jpg', '/p/cam.jpg']);
      expect(saved.single.$2, 'Nohkalikai');
      expect(saved.single.$3, 2);
      // Saved copies are kept.
      expect(discarded, ['/p/2.jpg']);
      expect(find.text('open'), findsOneWidget);
    });

    testWidgets('opens the picker at once; leaving deletes the copies',
        (t) async {
      final discarded = <String>[];
      var picks = 0;
      await pump(t, AddMemoriesScreen(
        stops: stops,
        pickOnOpen: true,
        pickMany: () async {
          picks++;
          return ['/p/1.jpg'];
        },
        takeOne: () async => null,
        thumb: box,
        onDiscard: (p) async => discarded.addAll(p),
        onSave: (_, _, _) async {},
      ));
      expect(picks, 1);
      expect(find.text('Keep this photo'), findsOneWidget);
      await t.pageBack();
      await t.pumpAndSettle();
      expect(discarded, ['/p/1.jpg']);
    });
  });

  testWidgets('six tabs fit a small phone, labels whole', (t) async {
    t.view.physicalSize = const Size(320, 640);
    t.view.devicePixelRatio = 1;
    addTearDown(t.view.reset);
    await t.pumpWidget(MaterialApp(
      theme: AppTokens.light,
      home: AppShell(destinations: [
        for (final (label, icon) in [
          ('Trip', Icons.route_outlined),
          ('Diary', Icons.menu_book_outlined),
          ('Money', Icons.currency_rupee),
          ('SOS', Icons.emergency_outlined),
          ('Memories', Icons.photo_library_outlined),
          ('More', Icons.more_horiz),
        ])
          ShellDestination(
            label: label,
            icon: icon,
            emergency: label == 'SOS',
            screen: Text(label),
          ),
      ]),
    ));
    expect(t.takeException(), isNull);
    final label = t.renderObject<RenderParagraph>(find.text('MEMORIES'));
    expect(label.didExceedMaxLines, isFalse);
    expect(label.size.width, lessThanOrEqualTo(320 / 6));
  });
}
