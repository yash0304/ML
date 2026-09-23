// test/sun_test.dart
//
// Sunrise and sunset, offline — checked against astral, an independent
// implementation, rather than against the same formula written twice. The
// expected clock times below came from astral with Asia/Kolkata, and are
// compared as Indian time because that is how they will be read.

import 'package:flutter_test/flutter_test.dart';
import 'package:safarsathi/core/util/sun.dart';

const ist = Duration(hours: 5, minutes: 30);

int minutesOf(String hhmm) {
  final p = hhmm.split(':');
  return int.parse(p[0]) * 60 + int.parse(p[1]);
}

void main() {
  group('against astral, within two minutes', () {
    const cases = [
      ('Shillong', 25.5764, 91.8838, (2026, 10, 1), '05:14', '17:09'),
      ('Sohra', 25.2718, 91.7327, (2026, 10, 3), '05:16', '17:07'),
      ('Dawki', 25.1849, 92.0223, (2026, 10, 5), '05:15', '17:04'),
      // Far from the trip, so the formula is not only right in October in
      // one corner of the country.
      ('Guwahati, midsummer', 26.1779, 91.7528, (2026, 6, 21), '04:31', '18:17'),
      ('Delhi, midwinter', 28.6139, 77.209, (2026, 12, 21), '07:09', '17:28'),
      ('Kanyakumari, equinox', 8.0883, 77.5385, (2026, 3, 20), '06:24', '18:30'),
    ];

    for (final (name, lat, lon, (y, m, d), rise, set) in cases) {
      test(name, () {
        final sun = sunTimes(lat, lon, DateTime(y, m, d));
        final gotRise = minutesOf(clockTime(sun.sunrise!, offset: ist));
        final gotSet = minutesOf(clockTime(sun.sunset!, offset: ist));
        expect(gotRise, closeTo(minutesOf(rise), 2), reason: 'sunrise');
        expect(gotSet, closeTo(minutesOf(set), 2), reason: 'sunset');
      });
    }
  });

  test('THE SUNRISE IS ON THE DAY ASKED FOR, IN INDIAN TIME', () {
    // 05:14 in Shillong on 1 October is 23:44 UTC on 30 September. A formula
    // that pinned itself to UTC days would hand back the next morning's.
    final sun = sunTimes(25.5764, 91.8838, DateTime(2026, 10, 1));
    final local = sun.sunrise!.add(ist);
    expect(local.day, 1);
    expect(local.month, 10);
  });

  test('the time of day passed in is ignored', () {
    final a = sunTimes(25.57, 91.88, DateTime(2026, 10, 1, 0, 1));
    final b = sunTimes(25.57, 91.88, DateTime(2026, 10, 1, 23, 59));
    expect(a.sunset, b.sunset);
  });

  test('where the sun does not set, it says so rather than inventing one', () {
    // Svalbard in June.
    final sun = sunTimes(78.22, 15.65, DateTime(2026, 6, 21));
    expect(sun.sunrise, isNull);
    expect(sun.sunset, isNull);
  });

  group('the warning', () {
    final sun = sunTimes(25.2718, 91.7327, DateTime(2026, 10, 3)); // ~17:07

    DateTime at(int h, int m) =>
        DateTime.utc(2026, 10, 3, h, m).subtract(ist);

    test('comfortably in daylight says nothing', () {
      expect(daylightWarning(at(11, 0), sun, verb: 'Arrives'), isNull);
    });

    test('after sunset says how long after, and that it is dark', () {
      final w = daylightWarning(at(18, 0), sun, verb: 'Arrives');
      expect(w, startsWith('Arrives '));
      expect(w, contains('after sunset — in the dark'));
    });

    test('within half an hour of sunset is cutting it close', () {
      final w = daylightWarning(at(16, 50), sun, verb: 'Arrives');
      expect(w, contains('only'));
      expect(w, contains('before sunset'));
    });

    test('before sunrise is dark too', () {
      final w = daylightWarning(at(4, 30), sun, verb: 'Leaves');
      expect(w, contains('before sunrise — in the dark'));
    });

    test('long spans read as hours', () {
      final w = daylightWarning(at(19, 22), sun, verb: 'Arrives');
      expect(w, contains(' h '));
    });
  });
}
