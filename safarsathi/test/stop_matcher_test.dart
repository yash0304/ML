// test/stop_matcher_test.dart — issue #15

import 'package:flutter_test/flutter_test.dart';
import 'package:safarsathi/features/import/data/stop_matcher.dart';

void main() {
  const stops = [
    StopCandidate(1, 'Shillong'),
    StopCandidate(2, 'Cherrapunji'),
    StopCandidate(3, 'Dawki'),
  ];

  test('an exact name matches', () {
    expect(matchStop('Shillong', stops).stopId, 1);
  });

  test('case and punctuation do not matter', () {
    expect(matchStop('  cherra-punji ', stops).stopId, 2);
  });

  test('a near spelling matches', () {
    expect(matchStop('Cherrapunjee', stops).stopId, 2);
    expect(matchStop('Shilong', stops).stopId, 1);
  });

  test('a different place with a similar name does NOT match', () {
    // The whole risk of fuzzy matching is filing a contact under the wrong
    // town. Silchar is a real, different place 200 km from Shillong.
    expect(matchStop('Silchar', stops).stopId, isNull);
    expect(matchStop('Silchar', stops).unmatched, isTrue);
  });

  test('short names are matched strictly', () {
    // Four characters or fewer get no tolerance at all: Puri and Pune are one
    // edit apart and 1,500 km apart.
    const short = [StopCandidate(9, 'Puri')];
    expect(matchStop('Pune', short).stopId, isNull);
    expect(matchStop('Puri', short).stopId, 9);
  });

  test('a qualified trip name still matches a bare sheet name', () {
    const qualified = [StopCandidate(4, 'Shillong, Meghalaya')];
    expect(matchStop('Shillong', qualified).stopId, 4);
  });

  test('an unmatched name still yields a usable result', () {
    // THE RULE: an unmatched stop is not an error. The row imports trip-wide.
    final m = matchStop('Guwahati', stops);
    expect(m.stopId, isNull);
    expect(m.unmatched, isTrue);
    expect(m.stopName, 'Guwahati');
  });

  test('an empty cell is not an unmatched stop', () {
    expect(matchStop('', stops).unmatched, isFalse);
    expect(matchStop('   ', stops).stopId, isNull);
  });

  test('a trip with no stops reports unmatched rather than crashing', () {
    expect(matchStop('Shillong', const []).unmatched, isTrue);
  });

  test('levenshtein counts edits', () {
    expect(levenshtein('kitten', 'sitting'), 3);
    expect(levenshtein('same', 'same'), 0);
    expect(levenshtein('', 'abc'), 3);
  });
}
