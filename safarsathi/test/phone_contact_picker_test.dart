// test/phone_contact_picker_test.dart
//
// Picking one number from the phone's own contacts. The native half is the
// system picker (MainActivity.kt); these pin the Dart half against a mocked
// platform channel, and the form's behaviour around it.

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:safarsathi/core/theme/app_tokens.dart';
import 'package:safarsathi/features/contacts/data/entry_draft.dart';
import 'package:safarsathi/features/contacts/data/phone_contact_picker.dart';
import 'package:safarsathi/features/contacts/presentation/entry_form_screen.dart';

import 'entry_form_test.dart' show existingEntry;

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('the channel', () {
    void answer(Future<Object?> Function(MethodCall) handler) {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(PhoneContactPicker.channel, handler);
      addTearDown(
        () => TestDefaultBinaryMessengerBinding
            .instance
            .defaultBinaryMessenger
            .setMockMethodCallHandler(PhoneContactPicker.channel, null),
      );
    }

    test('a pick comes back as a name and a number', () async {
      String? method;
      answer((call) async {
        method = call.method;
        return {'name': ' Bah Kyn (driver) ', 'number': ' +91 98631 00000 '};
      });

      final picked = await const PhoneContactPicker().pick();
      expect(method, 'pickPhone');
      expect(picked?.name, 'Bah Kyn (driver)');
      expect(picked?.number, '+91 98631 00000');
    });

    test('backing out of the picker is null, not an error', () async {
      answer((_) async => null);
      expect(await const PhoneContactPicker().pick(), isNull);
    });

    test('a contact with no number is said, not silently dropped', () async {
      answer((_) async => {'name': 'Somebody', 'number': ''});
      expect(
        const PhoneContactPicker().pick(),
        throwsA(isA<ContactPickException>()
            .having((e) => e.message, 'message', contains('no phone number'))),
      );
    });

    test('a platform failure carries the native sentence', () async {
      answer((_) async => throw PlatformException(
            code: 'unavailable',
            message: 'This phone has no contacts app to pick from.',
          ));
      expect(
        const PhoneContactPicker().pick(),
        throwsA(isA<ContactPickException>().having(
          (e) => e.message,
          'message',
          'This phone has no contacts app to pick from.',
        )),
      );
    });

    test('no native half at all is a sentence, not a crash', () async {
      // Nothing registered on the channel: what a test or desktop run sees.
      expect(
        const PhoneContactPicker().pick(),
        throwsA(isA<ContactPickException>()),
      );
    });
  });

  group('the form', () {
    late List<EntryDraft> saved;

    Future<void> pumpForm(
      WidgetTester tester, {
      required Future<PickedContact?> Function() pick,
      bool pickOnOpen = false,
      bool editing = false,
    }) async {
      saved = [];
      tester.view.physicalSize = const Size(420, 1400);
      tester.view.devicePixelRatio = 1.0;
      addTearDown(tester.view.reset);
      await tester.pumpWidget(
        MaterialApp(
          theme: AppTokens.light,
          home: EntryFormScreen(
            existing: editing ? existingEntry() : null,
            pickFromPhone: pick,
            pickOnOpen: pickOnOpen,
            onSave: (d) async => saved.add(d),
          ),
        ),
      );
      await tester.pump();
    }

    String fieldText(WidgetTester tester, String key) => tester
        .widget<TextField>(
          find.descendant(of: find.byKey(Key(key)), matching: find.byType(TextField)),
        )
        .controller!
        .text;

    testWidgets('picking fills the name and the number', (tester) async {
      await pumpForm(
        tester,
        pick: () async =>
            const PickedContact(name: 'Bah Kyn', number: '+91 98631 00000'),
      );
      await tester.tap(find.byKey(const Key('pick-from-phone')));
      await tester.pumpAndSettle();

      expect(fieldText(tester, 'field-name'), 'Bah Kyn');
      expect(fieldText(tester, 'field-number'), '+91 98631 00000');
    });

    testWidgets('A PICKED NUMBER SAVES AS AN UNCONFIRMED ENTRY LIKE ANY OTHER',
        (tester) async {
      // Being in the phone's contacts says nothing about whether a number
      // still reaches the right person in Meghalaya. It goes through the
      // same form and lands with the same amber dot.
      await pumpForm(
        tester,
        pick: () async =>
            const PickedContact(name: 'Bah Kyn', number: '+91 98631 00000'),
      );
      await tester.tap(find.byKey(const Key('pick-from-phone')));
      await tester.pumpAndSettle();
      await tester.tap(find.text('Save to diary'));
      await tester.pumpAndSettle();

      expect(saved.single.name, 'Bah Kyn');
      expect(saved.single.phoneE164, '+919863100000');
      expect(saved.single.resetConfirmation, isFalse);
    });

    testWidgets('backing out leaves what was typed alone', (tester) async {
      await pumpForm(tester, pick: () async => null);
      await tester.enterText(
        find.descendant(
          of: find.byKey(const Key('field-name')),
          matching: find.byType(TextField),
        ),
        'Typed first',
      );
      await tester.tap(find.byKey(const Key('pick-from-phone')));
      await tester.pumpAndSettle();
      expect(fieldText(tester, 'field-name'), 'Typed first');
    });

    testWidgets('a failure is shown, not swallowed', (tester) async {
      await pumpForm(
        tester,
        pick: () async =>
            throw const ContactPickException('That contact has no phone number.'),
      );
      await tester.tap(find.byKey(const Key('pick-from-phone')));
      await tester.pump();
      expect(find.text('That contact has no phone number.'), findsOneWidget);
    });

    testWidgets('pickOnOpen opens the picker without a tap', (tester) async {
      var opened = 0;
      await pumpForm(
        tester,
        pickOnOpen: true,
        pick: () async {
          opened++;
          return null;
        },
      );
      await tester.pump();
      expect(opened, 1);
    });

    testWidgets('editing an existing entry does not offer it', (tester) async {
      // Replacing a confirmed number from the address book would quietly
      // swap the digits the confirmation was for.
      await pumpForm(tester, editing: true, pick: () async => null);
      expect(find.byKey(const Key('pick-from-phone')), findsNothing);
    });
  });
}
