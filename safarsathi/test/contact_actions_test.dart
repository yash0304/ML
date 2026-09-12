// Issue #7 acceptance.
//
// The launcher and the clipboard are injected, so these run without a
// platform. What is checked is what the user gets: the right digits, the
// right scheme, a log row every time, and a message rather than silence when
// nothing can handle the intent.

import 'package:drift/drift.dart' hide isNull, isNotNull;
import 'package:drift/native.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:safarsathi/core/database/app_database.dart';
import 'package:safarsathi/features/contacts/data/contact_actions.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  late AppDatabase db;
  late int tripId;
  late List<Uri> launched;
  late List<String> clipboard;
  late bool launchSucceeds;
  late bool launchThrows;

  ContactActions makeActions() => ContactActions(
    dao: db.contactsDao,
    tripId: tripId,
    launch: (uri) async {
      if (launchThrows) throw StateError('platform blew up');
      launched.add(uri);
      return launchSucceeds;
    },
    copyToClipboard: (text) async => clipboard.add(text),
  );

  setUp(() async {
    db = AppDatabase(NativeDatabase.memory());
    tripId = await db
        .into(db.trips)
        .insert(TripsCompanion.insert(name: 'Meghalaya'));
    launched = [];
    clipboard = [];
    launchSucceeds = true;
    launchThrows = false;
  });

  tearDown(() async => db.close());

  Future<Contact> add({String? e164, String raw = '+91 90000 00001'}) async {
    final id = await db
        .into(db.contacts)
        .insert(
          ContactsCompanion.insert(
            name: 'Kongthong homestay',
            phoneRaw: raw,
            phoneE164: Value(e164),
            tripId: Value(tripId),
          ),
        );
    return (db.select(db.contacts)..where((c) => c.id.equals(id))).getSingle();
  }

  Future<List<String>> loggedActions() async =>
      (await db.select(db.callLogs).get()).map((l) => l.action).toList();

  group('copy', () {
    test('prefers the normalised number', () async {
      final c = await add(e164: '+919000000001', raw: '+91 90000 00001');
      final copied = await makeActions().copy(c);

      expect(copied, '+919000000001');
      expect(clipboard, ['+919000000001']);
    });

    test(
      'falls back to what the user typed when normalisation failed',
      () async {
        // phoneE164 is nullable precisely because normalisation can fail on bad
        // input, and the raw value has to survive that.
        final c = await add(raw: '0364 900 0005');
        expect(await makeActions().copy(c), '0364 900 0005');
      },
    );

    test('is logged, because the dial now happens outside the app', () async {
      final c = await add();
      await makeActions().copy(c);

      expect(await loggedActions(), [ContactAction.copy]);
      final row = await (db.select(
        db.contacts,
      )..where((x) => x.id.equals(c.id))).getSingle();
      expect(row.callCount, 1);
    });
  });

  group('open dialer', () {
    test('launches tel: with no number in it', () async {
      await makeActions().openDialer();
      expect(launched.single.scheme, 'tel');
      expect(launched.single.path, isEmpty);
    });

    test('says so when there is no dialer', () async {
      launchSucceeds = false;
      await expectLater(
        makeActions().openDialer(),
        throwsA(isA<ActionFailure>()),
      );
    });

    test('logs against a contact when one is named', () async {
      final c = await add();
      await makeActions().openDialer(forContact: c);
      expect(await loggedActions(), [ContactAction.dialer]);
    });
  });

  group('call, sms and chat', () {
    test('call launches tel: with the number', () async {
      final c = await add(e164: '+919000000001');
      await makeActions().call(c);

      expect(launched.single.scheme, 'tel');
      expect(launched.single.path, '+919000000001');
      expect(await loggedActions(), [ContactAction.call]);
    });

    test('sms launches the messaging scheme', () async {
      final c = await add(e164: '+919000000001');
      await makeActions().sms(c);

      expect(launched.single.scheme, 'sms');
      expect(await loggedActions(), [ContactAction.sms]);
    });

    test('whatsapp strips every non-digit', () async {
      // wa.me takes digits only; a leading + or any spacing breaks the link.
      final c = await add(raw: '+91 90000 00001');
      await makeActions().whatsapp(c);

      expect(launched.single.toString(), 'https://wa.me/919000000001');
      expect(await loggedActions(), [ContactAction.whatsapp]);
    });

    test('whatsapp refuses an entry with no digits', () async {
      final c = await add(raw: 'ask at reception');
      await expectLater(
        makeActions().whatsapp(c),
        throwsA(isA<ActionFailure>()),
      );
    });
  });

  group('failure is never silent', () {
    test('a refused launch reports what could not be opened', () async {
      launchSucceeds = false;
      final c = await add();
      await expectLater(makeActions().call(c), throwsA(isA<ActionFailure>()));
    });

    test('a launcher that throws is reported the same way', () async {
      // Some devices throw rather than returning false.
      launchThrows = true;
      final c = await add();
      await expectLater(makeActions().sms(c), throwsA(isA<ActionFailure>()));
    });

    test('a failed action is not logged as if it happened', () async {
      launchSucceeds = false;
      final c = await add();
      await makeActions().call(c).catchError((_) {});
      expect(await loggedActions(), isEmpty);
    });
  });
}
