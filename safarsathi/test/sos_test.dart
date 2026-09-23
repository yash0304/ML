// test/sos_test.dart — "Text my location".

import 'dart:async';

import 'package:drift/native.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:safarsathi/core/database/app_database.dart';
import 'package:safarsathi/core/theme/app_tokens.dart';
import 'package:safarsathi/features/contacts/data/phone_contact_picker.dart';
import 'package:safarsathi/features/discovery/data/geo.dart';
import 'package:safarsathi/features/emergency/data/sos.dart';
import 'package:safarsathi/features/emergency/presentation/sos_panel.dart';
import 'package:safarsathi/features/map/data/here.dart';

class ScriptedLocation implements LocationSource {
  HereState state;
  Future<HereFix?> Function() fix;
  final asks = <bool>[];
  ScriptedLocation(this.state, this.fix);

  @override
  Future<HereState> check({bool ask = false}) async {
    asks.add(ask);
    return state;
  }

  @override
  Future<HereFix?> once({Duration timeout = const Duration(seconds: 20)}) =>
      fix();

  @override
  Stream<HereFix> watch() => const Stream.empty();
  @override
  Future<void> openAppSettings() async {}
  @override
  Future<void> openLocationSettings() async {}
}

final fixAtSohra = HereFix(
  at: const LatLng(25.27183, 91.73295),
  accuracyM: 12,
  time: DateTime(2026, 10, 3, 17, 0),
);

void main() {
  group('the message', () {
    test('with a fix: coordinates, how good the fix is, a map link, the stop',
        () {
      final msg = sosMessage(
        fix: fixAtSohra,
        nearStop: 'Sohrra',
        now: DateTime(2026, 10, 3, 17, 0, 20),
      );
      expect(msg, '''
SOS — I need help.
Location: 25.27183, 91.73295 (±12 m, 20 s ago)
Map: https://maps.google.com/?q=25.27183,91.73295
Near Sohrra on our trip plan.
Sent from SafarSathi.''');
    });

    test('WITHOUT A FIX IT STILL SAYS SOMETHING USEFUL', () {
      final msg = sosMessage(nearStop: 'Sohrra');
      expect(msg, contains('not available from my phone right now'));
      expect(msg, contains('I should be at or near Sohrra'));
      expect(msg, startsWith('SOS — I need help.'));
    });

    test('with neither fix nor stop it is still a call for help', () {
      expect(sosMessage(), startsWith('SOS — I need help.'));
    });
  });

  group('the SMS link', () {
    test('spaces are %20, never "+", which some SMS apps show literally', () {
      final uri = smsUri('+919800000001', 'I need help\nMap: x?q=1,2');
      final s = uri.toString();
      expect(s, startsWith('sms:%2B919800000001?body='));
      expect(s, contains('I%20need%20help'));
      expect(s, isNot(contains('I+need')));
      expect(s, contains('%0A'));
      // And it decodes back to exactly the message.
      expect(Uri.decodeComponent(s.split('body=').last),
          'I need help\nMap: x?q=1,2');
    });
  });

  group('who gets it', () {
    late AppDatabase db;
    setUp(() => db = AppDatabase(NativeDatabase.memory()));
    tearDown(() => db.close());

    test('added from a phone-style number, stored normalised', () async {
      await addTrusted(db, 'Mummy', '098 250 12345');
      final people = await watchTrusted(db).first;
      expect(people.single.name, 'Mummy');
      expect(people.single.phoneE164, '+919825012345');
      expect(people.single.tripId, isNull,
          reason: 'family does not change between trips');
    });

    test('A NUMBER THAT CANNOT BE DIALLED IS REFUSED', () async {
      // An SOS to it would be believed sent and reach nobody.
      expect(() => addTrusted(db, 'Typo', '12'),
          throwsA(isA<TrustedException>()));
    });

    test('the same number twice is refused by name', () async {
      await addTrusted(db, 'Mummy', '+91 98250 12345');
      expect(
        () => addTrusted(db, 'Mom', '098250 12345'),
        throwsA(isA<TrustedException>()
            .having((e) => e.message, 'message', contains('Mummy'))),
      );
    });

    test('removing takes them off', () async {
      await addTrusted(db, 'Mummy', '+91 98250 12345');
      final id = (await watchTrusted(db).first).single.id;
      await removeTrusted(db, id);
      expect(await watchTrusted(db).first, isEmpty);
    });
  });

  group('the panel', () {
    late List<Uri> smsOpened;
    late List<String> shared;
    late List<(String, String)> added;
    late List<int> removed;
    var smsWorks = true;

    const mummy = TrustedContact(
      id: 1,
      name: 'Mummy',
      phoneE164: '+919825012345',
      notifyOnArrival: false,
      escalate: false,
      escalateAfterMinutes: 30,
    );

    Future<void> pump(
      WidgetTester tester,
      LocationSource location, {
      List<TrustedContact> people = const [],
      Future<PickedContact?> Function()? pick,
      Future<void> Function(String, String)? add,
    }) async {
      smsOpened = [];
      shared = [];
      added = [];
      removed = [];
      tester.view.physicalSize = const Size(420, 900);
      tester.view.devicePixelRatio = 1.0;
      addTearDown(tester.view.reset);
      await tester.pumpWidget(MaterialApp(
        theme: AppTokens.light,
        home: Scaffold(
          body: SosPanel(
            trusted: Stream.value(people),
            location: location,
            nearStop: () async => 'Sohrra',
            openSms: (uri) async {
              smsOpened.add(uri);
              return smsWorks;
            },
            share: (text) async => shared.add(text),
            pickPerson: pick ?? () async => null,
            addPerson: add ?? (n, p) async => added.add((n, p)),
            removePerson: (p) async => removed.add(p.id),
            wait: const Duration(seconds: 5),
          ),
        ),
      ));
      await tester.pump();
    }

    String bodyOf(Uri uri) =>
        Uri.decodeComponent(uri.toString().split('body=').last);

    testWidgets('with nobody chosen, it asks you to choose', (tester) async {
      await pump(tester, ScriptedLocation(HereState.found, () async => null));
      expect(find.text('Choose who gets your SOS text'), findsOneWidget);
    });

    testWidgets('a tap asks for location, then opens the SMS app written',
        (tester) async {
      final loc = ScriptedLocation(HereState.locating, () async => fixAtSohra);
      smsWorks = true;
      await pump(tester, loc, people: [mummy]);
      await tester.tap(find.text('Text Mummy'));
      await tester.pumpAndSettle();

      expect(loc.asks, [true]);
      expect(smsOpened, hasLength(1));
      expect(smsOpened.single.toString(), startsWith('sms:%2B919825012345'));
      expect(bodyOf(smsOpened.single), contains('25.27183, 91.73295'));
      expect(shared, isEmpty);
    });

    testWidgets('LOCATION REFUSED: THE TEXT STILL GOES, SAYING SO',
        (tester) async {
      final loc = ScriptedLocation(HereState.denied, () async => fixAtSohra);
      await pump(tester, loc, people: [mummy]);
      await tester.tap(find.text('Text Mummy'));
      await tester.pumpAndSettle();

      final body = bodyOf(smsOpened.single);
      expect(body, contains('not available'));
      expect(body, contains('Sohrra'));
    });

    testWidgets('"SEND WITHOUT IT" DOES NOT WAIT FOR THE GPS', (tester) async {
      final never = Completer<HereFix?>();
      final loc = ScriptedLocation(HereState.locating, () => never.future);
      await pump(tester, loc, people: [mummy]);
      await tester.tap(find.text('Text Mummy'));
      await tester.pump();
      expect(find.text('Getting your location…'), findsOneWidget);

      await tester.tap(find.byKey(const Key('sos-skip')));
      await tester.pumpAndSettle();
      expect(smsOpened, hasLength(1));
      expect(bodyOf(smsOpened.single), contains('not available'));
    });

    testWidgets('no SMS app: the share sheet gets the same message',
        (tester) async {
      smsWorks = false;
      final loc = ScriptedLocation(HereState.locating, () async => fixAtSohra);
      await pump(tester, loc, people: [mummy]);
      await tester.tap(find.text('Text Mummy'));
      await tester.pumpAndSettle();
      expect(shared.single, contains('25.27183'));
      smsWorks = true;
    });

    testWidgets('"another way" shares the message without anyone chosen',
        (tester) async {
      final loc = ScriptedLocation(HereState.locating, () async => fixAtSohra);
      await pump(tester, loc);
      await tester.tap(find.byKey(const Key('sos-share')));
      await tester.pumpAndSettle();
      expect(shared.single, startsWith('SOS — I need help.'));
    });

    testWidgets('choosing someone goes through the phone\'s picker',
        (tester) async {
      await pump(
        tester,
        ScriptedLocation(HereState.found, () async => null),
        pick: () async =>
            const PickedContact(name: 'Mummy', number: '+91 98250 12345'),
      );
      await tester.tap(find.byKey(const Key('sos-add')));
      await tester.pumpAndSettle();
      expect(added.single, ('Mummy', '+91 98250 12345'));
    });

    testWidgets('a number that cannot be saved is said', (tester) async {
      await pump(
        tester,
        ScriptedLocation(HereState.found, () async => null),
        pick: () async => const PickedContact(name: 'Typo', number: '12'),
        add: (n, p) async =>
            throw const TrustedException('"12" could not be read'),
      );
      await tester.tap(find.byKey(const Key('sos-add')));
      await tester.pumpAndSettle();
      expect(find.text('"12" could not be read'), findsOneWidget);
    });

    testWidgets('a long press asks, then removes', (tester) async {
      await pump(tester, ScriptedLocation(HereState.found, () async => null),
          people: [mummy]);
      await tester.longPress(find.text('Text Mummy'));
      await tester.pumpAndSettle();
      expect(find.text('Stop texting Mummy?'), findsOneWidget);
      await tester.tap(find.text('Remove'));
      await tester.pumpAndSettle();
      expect(removed, [1]);
    });
  });
}
