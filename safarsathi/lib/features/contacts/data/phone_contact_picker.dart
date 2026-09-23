// lib/features/contacts/data/phone_contact_picker.dart
//
// One number from the phone's own contacts, chosen by the user.
//
// THE SYSTEM PICKER, NOT THE ADDRESS BOOK. MainActivity.kt starts Android's
// own contact picker; the user chooses one number and this app gets a
// one-time grant to read that row. No READ_CONTACTS permission, no list of
// everyone the user knows ever passing through this app. Asked for so that
// a driver or homestay already saved on the phone does not have to be typed
// in twice — which is all it needs to do.

import 'package:flutter/services.dart';

class PickedContact {
  final String name;
  final String number;
  const PickedContact({required this.name, required this.number});
}

/// Why a pick could not happen, in a sentence for a snack bar.
class ContactPickException implements Exception {
  final String message;
  const ContactPickException(this.message);
  @override
  String toString() => message;
}

class PhoneContactPicker {
  static const channel = MethodChannel('safarsathi/contacts');

  const PhoneContactPicker();

  /// The chosen contact, or null if the user backed out.
  Future<PickedContact?> pick() async {
    final Map<String, dynamic>? result;
    try {
      result = await channel.invokeMapMethod<String, dynamic>('pickPhone');
    } on MissingPluginException {
      // A build without the native half — a test, a desktop run.
      throw const ContactPickException(
        'Picking from the phone\'s contacts is not available here.',
      );
    } on PlatformException catch (e) {
      throw ContactPickException(
        e.message ?? 'The phone\'s contacts could not be opened.',
      );
    }

    if (result == null) return null;
    final name = (result['name'] as String? ?? '').trim();
    final number = (result['number'] as String? ?? '').trim();

    // A contact with a name and no number cannot go in a phone diary. The
    // picker is filtered to phone numbers, so this is a corrupt row rather
    // than a normal case — and it is said, not silently dropped.
    if (number.isEmpty) {
      throw const ContactPickException('That contact has no phone number.');
    }
    return PickedContact(name: name, number: number);
  }
}
