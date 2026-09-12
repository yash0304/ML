// Issue #8 — E.164 normalisation.
//
// The governing rule: a number that will not parse is a WARNING, never a
// block. The user may be halfway through typing, holding an extension, or
// copying something odd off a signboard. The amber dot already says the
// number is unverified; refusing the save would lose their only record of it.

import 'package:flutter_test/flutter_test.dart';
import 'package:phone_numbers_parser/phone_numbers_parser.dart';

import 'package:safarsathi/features/contacts/data/phone_normaliser.dart';

void main() {
  test('an Indian mobile typed locally normalises', () {
    final r = PhoneNormaliser.normalise('98560 41122');
    expect(r.e164, '+919856041122');
    expect(r.warning, isNull);
    expect(r.raw, '98560 41122', reason: 'the raw value is never discarded');
  });

  test('an already-international number survives round trip', () {
    expect(PhoneNormaliser.normalise('+91 98560 41122').e164, '+919856041122');
  });

  test('punctuation and spacing do not matter', () {
    for (final input in [
      '+91-98560-41122',
      '(+91) 98560 41122',
      '+91.98560.41122',
    ]) {
      expect(
        PhoneNormaliser.normalise(input).e164,
        '+919856041122',
        reason: input,
      );
    }
  });

  test('the country comes from the caller, so Europe works mid-trip', () {
    final r = PhoneNormaliser.normalise('030 12345678', country: IsoCode.DE);
    expect(r.e164, startsWith('+49'));
  });

  test('an incomplete number warns and is still kept', () {
    final r = PhoneNormaliser.normalise('9856');
    expect(r.e164, isNull, reason: 'no confident-looking wrong value');
    expect(r.warning, isNotNull);
    expect(r.raw, '9856');
  });

  test('something that is not a number at all warns rather than throwing', () {
    final r = PhoneNormaliser.normalise('ask at reception');
    expect(r.e164, isNull);
    expect(r.warning, isNotNull);
  });

  test('empty is empty, with nothing to complain about', () {
    final r = PhoneNormaliser.normalise('   ');
    expect(r.isEmpty, isTrue);
    expect(r.warning, isNull);
    expect(r.e164, isNull);
  });

  test('a landline with an area code normalises', () {
    final r = PhoneNormaliser.normalise('0364 2224100');
    expect(r.e164, isNotNull);
    expect(r.e164, startsWith('+91'));
  });
}
