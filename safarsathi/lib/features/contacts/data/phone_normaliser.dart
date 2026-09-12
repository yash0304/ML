// lib/features/contacts/data/phone_normaliser.dart
//
// E.164 normalisation, done in pure Dart with no platform channel so it works
// with no signal and no network.
//
// BOTH FORMS ARE KEPT. `phoneRaw` is what the user typed and what the diary
// shows back — reformatting someone's own input is confusing. `phoneE164` is
// what gets dialled and what duplicate detection compares. Normalisation
// genuinely fails on bad input, and when it does the raw value has to survive.

import 'package:phone_numbers_parser/phone_numbers_parser.dart';

class NormalisedPhone {
  /// Exactly what the user typed, trimmed. Never discarded.
  final String raw;

  /// Set only when the number parsed AND validated. A half-typed number gets
  /// null here rather than a confident-looking wrong value.
  final String? e164;

  /// Null when there is nothing to say. A sentence when the number looks
  /// wrong — which warns but never blocks a save.
  final String? warning;

  const NormalisedPhone({required this.raw, this.e164, this.warning});

  bool get isEmpty => raw.isEmpty;
  bool get normalised => e164 != null;
}

class PhoneNormaliser {
  PhoneNormaliser._();

  /// Parses [input] as a number dialled in [country].
  ///
  /// An unreadable or implausible number is a WARNING, not an error. The user
  /// may be halfway through typing, or holding a number with an extension, or
  /// copying something odd off a signboard. The amber dot already says the
  /// number is unverified; refusing to save it would lose the only record
  /// they have of it.
  static NormalisedPhone normalise(
    String input, {
    IsoCode country = IsoCode.IN,
  }) {
    final raw = input.trim();
    if (raw.isEmpty) {
      return const NormalisedPhone(raw: '');
    }

    try {
      final parsed = PhoneNumber.parse(raw, destinationCountry: country);
      if (!parsed.isValid()) {
        return NormalisedPhone(
          raw: raw,
          warning: 'That does not look like a complete number. Saved as typed.',
        );
      }
      return NormalisedPhone(raw: raw, e164: parsed.international);
    } on Object {
      return NormalisedPhone(
        raw: raw,
        warning: 'Could not read that as a phone number. Saved as typed.',
      );
    }
  }
}
