// test/sync_error_test.dart
//
// The exact strings Yash's phone produced on 13 Sep are pinned here by name.
// The screen showed them raw and truncated mid-URL; the point of this file is
// that it can never do that again.

import 'dart:async';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;

import 'package:safarsathi/features/map/data/tile_downloader.dart';
import 'package:safarsathi/features/sync/data/sync_error.dart';

void main() {
  group('what the phone actually threw', () {
    test('A FAILED DNS LOOKUP READS AS NO INTERNET', () {
      // Verbatim from the device, including errno 7.
      final e = describeSyncError(
        "ClientException with SocketException: Failed host lookup: "
        "'router.project-osrm.org' (OS Error: No address associated with "
        "hostname, errno = 7), uri=https://router.project-osrm.org/route/v1",
      );

      expect(e.kind, SyncErrorKind.offline);
      expect(e.message, 'No internet. The phone could not look up the '
          'address at all.');
      // Not a word of the original leaks into what the person reads.
      expect(e.message, isNot(contains('SocketException')));
      expect(e.message, isNot(contains('errno')));
      expect(e.message, isNot(contains('uri=')));
      expect(e.needsDifferentNetwork, isTrue);
    });

    test('open-meteo failing the same way says the same thing', () {
      final e = describeSyncError(
        "ClientException with SocketException: Failed host lookup: "
        "'api.open-meteo.com' (OS Error: No address associated with hostname, "
        "errno = 7), uri=https://api.open-meteo.com/v1/forecast",
      );
      expect(e.kind, SyncErrorKind.offline);
    });

    test('a dropped connection is NOT reported as being offline', () {
      // Also verbatim. This one resolved and then died, which is a different
      // problem with a different answer — retrying can work.
      final e = describeSyncError(
        'ClientException: Software caused connection abort, '
        'uri=https://overpass-api.de/api/interpreter',
      );
      expect(e.kind, SyncErrorKind.dropped);
      expect(e.needsDifferentNetwork, isFalse);
      expect(e.message, 'The connection dropped partway through.');
    });
  });

  group('the other kinds', () {
    test('a timeout says the connection is slow, not absent', () {
      expect(
        describeSyncError(TimeoutException('x')).kind,
        SyncErrorKind.slow,
      );
    });

    test('a server error keeps its number and blames the server', () {
      final e = describeSyncError('OpenStreetMap returned 503.');
      expect(e.kind, SyncErrorKind.refused);
      expect(e.message, contains('503'));
      expect(e.message, contains('Not something this phone did'));
    });

    test('a 4xx is the request being turned down, not a server fault', () {
      final e = describeSyncError('The routing service returned 400.');
      expect(e.kind, SyncErrorKind.refused);
      expect(e.message, isNot(contains('Not something this phone did')));
    });

    test('A MISSING MAP KEY IS NOT A NETWORK PROBLEM', () {
      // Sending somebody to check their WiFi about a missing key would send
      // them to entirely the wrong place.
      final e = describeSyncError(
        const TileDownloadException('No MapTiler key.'),
      );
      expect(e.kind, SyncErrorKind.notConfigured);
      expect(e.message, 'No MapTiler key.');
    });

    test('a bare SocketException still lands somewhere useful', () {
      final e = describeSyncError(const SocketException('boom'));
      expect(e.kind, SyncErrorKind.dropped);
    });

    test('a ClientException with nothing recognisable in it', () {
      final e = describeSyncError(http.ClientException('odd'));
      expect(e.kind, SyncErrorKind.dropped);
    });

    test('something unexpected is truncated, but only as a last resort', () {
      final long = 'z' * 400;
      final e = describeSyncError(long);
      expect(e.kind, SyncErrorKind.unknown);
      expect(e.message.length, lessThan(150));
      expect(e.detail, long);
    });
  });

  group('the summary line', () {
    SyncError offline() => describeSyncError(
      'SocketException: Failed host lookup: (errno = 7)',
    );

    test('EVERYTHING FAILING OFFLINE NEVER SAYS EVERYTHING ELSE WENT THROUGH', () {
      // The bug in the screenshot: seven failures out of seven, under a line
      // claiming the rest had succeeded.
      final summary = summariseSyncFailures(
        [for (var i = 0; i < 7; i++) offline()],
        total: 7,
      );
      expect(summary, isNotNull);
      expect(summary, isNot(contains('Everything else went through')));
      expect(summary, contains('no working internet'));
      expect(summary, contains('nothing downloaded'));
    });

    test('some failing offline keeps what arrived', () {
      final summary = summariseSyncFailures([offline()], total: 7);
      expect(summary, contains('What arrived before that is kept'));
    });

    test('a mixed run that failed entirely still does not claim success', () {
      final summary = summariseSyncFailures(
        [offline(), describeSyncError('returned 500')],
        total: 2,
      );
      expect(summary, contains('Nothing downloaded'));
      expect(summary, isNot(contains('Everything else went through')));
    });

    test('a partial mixed run is the one case that may say it', () {
      final summary = summariseSyncFailures(
        [offline(), describeSyncError('returned 500')],
        total: 9,
      );
      expect(summary, contains('Everything else went through'));
    });

    test('only the map failing sends the user to the key, not the WiFi', () {
      final summary = summariseSyncFailures(
        [describeSyncError(const TileDownloadException('No MapTiler key.'))],
        total: 4,
      );
      expect(summary, contains('Settings → Map key'));
    });

    test('no failures, nothing to say', () {
      expect(summariseSyncFailures(const [], total: 4), isNull);
    });
  });
}
