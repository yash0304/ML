// lib/features/sync/data/sync_error.dart
//
// Turning a thrown object into a sentence somebody standing in a hotel lobby
// can act on.
//
// This exists because the first version did not. `TripSync._readable` carried
// a comment saying it turned an exception "into something a person can act
// on" and then truncated the string to 160 characters, so the sync screen
// showed:
//
//   ClientException with SocketException: Failed host lookup:
//   'router.project-osrm.org' (OS Error: No address associated with
//   hostname, errno = 7), uri=https://r...
//
// Seven times over, each one cut off mid-URL. Nothing in that tells a person
// what to do, and the one thing it does say — that the phone could not reach
// the internet at all — is the thing it says least clearly.

import 'dart:async';
import 'dart:io';

import 'package:http/http.dart' as http;

import '../../map/data/tile_downloader.dart';
import '../../map/data/tile_provider.dart';

enum SyncErrorKind {
  /// Nothing resolved. The phone has no working internet.
  offline,

  /// It resolved, then the connection died partway.
  dropped,

  /// It answered, eventually, but not soon enough.
  slow,

  /// It answered with a refusal.
  refused,

  /// No map key. Not a network problem at all.
  notConfigured,

  /// Genuinely unexpected.
  unknown,
}

class SyncError {
  final SyncErrorKind kind;

  /// One sentence, addressed to the person holding the phone.
  final String message;

  /// The original text, kept for the rare case somebody needs it. Never the
  /// first thing shown.
  final String detail;

  const SyncError({
    required this.kind,
    required this.message,
    required this.detail,
  });

  /// True when trying again on the same network cannot possibly help.
  bool get needsDifferentNetwork => kind == SyncErrorKind.offline;
}

/// What to say about [error].
SyncError describeSyncError(Object error) {
  final detail = '$error';
  final text = detail.toLowerCase();

  // A missing key is not a network failure, and telling somebody to check
  // their WiFi about it would send them to the wrong place entirely.
  if (error is TileProviderNotConfigured || error is TileDownloadException) {
    final message = error is TileDownloadException
        ? error.message
        : '$error';
    if (text.contains('key')) {
      return SyncError(
        kind: SyncErrorKind.notConfigured,
        message: message,
        detail: detail,
      );
    }
  }

  if (error is TimeoutException ||
      text.contains('timeoutexception') ||
      text.contains('timed out')) {
    return SyncError(
      kind: SyncErrorKind.slow,
      message: 'Gave up waiting. The connection is there but too slow to '
          'finish this.',
      detail: detail,
    );
  }

  // DNS. On Android this is errno 7, EAI_NONAME, and it is what every request
  // returns when the phone has no usable connection — the WiFi can be
  // associated and the icon full while the network behind it is dead.
  //
  // Matched on the message as well as the type because `package:http` wraps
  // the SocketException in a ClientException on some platforms, and then the
  // type is gone and only the text survives.
  if (text.contains('failed host lookup') ||
      text.contains('no address associated with hostname') ||
      text.contains('nodename nor servname')) {
    return SyncError(
      kind: SyncErrorKind.offline,
      message: 'No internet. The phone could not look up the address at all.',
      detail: detail,
    );
  }

  if (text.contains('connection refused') ||
      text.contains('network is unreachable')) {
    return SyncError(
      kind: SyncErrorKind.offline,
      message: 'No internet. Nothing on this network answered.',
      detail: detail,
    );
  }

  if (text.contains('connection abort') ||
      text.contains('connection reset') ||
      text.contains('connection closed') ||
      text.contains('connection terminated')) {
    return SyncError(
      kind: SyncErrorKind.dropped,
      message: 'The connection dropped partway through.',
      detail: detail,
    );
  }

  // An HTTP status the client turned into an exception. The number is the
  // useful part and it is worth keeping.
  final status = RegExp(r'\b(4\d\d|5\d\d)\b').firstMatch(detail);
  if (status != null) {
    final code = status.group(1)!;
    return SyncError(
      kind: SyncErrorKind.refused,
      message: code.startsWith('5')
          ? 'The server had a problem ($code). Not something this phone did.'
          : 'The server turned the request down ($code).',
      detail: detail,
    );
  }

  if (error is SocketException || error is http.ClientException) {
    return SyncError(
      kind: SyncErrorKind.dropped,
      message: 'The connection failed partway through.',
      detail: detail,
    );
  }

  return SyncError(
    kind: SyncErrorKind.unknown,
    message: detail.length > 140 ? '${detail.substring(0, 137)}…' : detail,
    detail: detail,
  );
}

/// The one thing to say when a whole run failed the same way.
///
/// Seven rows repeating the same DNS error is not seven pieces of
/// information. If every task failed because the phone is offline, that is
/// one fact and it belongs at the top, said once.
String? summariseSyncFailures(List<SyncError> errors, {required int total}) {
  if (errors.isEmpty) return null;

  final everything = errors.length >= total;
  final kinds = {for (final e in errors) e.kind};

  if (kinds.length == 1 && kinds.single == SyncErrorKind.offline) {
    return everything
        ? 'This phone has no working internet, so nothing downloaded. '
              'Connect to WiFi and press download again — anything that '
              'does arrive is kept.'
        : 'The connection went away partway through. What arrived before '
              'that is kept; press download again on WiFi for the rest.';
  }

  if (kinds.length == 1 && kinds.single == SyncErrorKind.notConfigured) {
    return 'Settings → Map key is empty, so the maps were skipped. '
        'Everything else here does not need it.';
  }

  if (everything) {
    return 'Nothing downloaded. Press download again once you are on a '
        'network that works — whatever arrives is kept.';
  }

  return 'Everything else went through. Pressing download again retries only '
      'these — whatever already arrived is kept.';
}
