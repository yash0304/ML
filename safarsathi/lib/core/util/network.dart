// lib/core/util/network.dart
//
// How long to wait before giving up on a request.
//
// Every one of these was missing. `http.get` with no timeout waits as long as
// the socket stays open, which on a hotel WiFi that has stopped forwarding
// packets is forever — and the sync runs its tasks in sequence, so one hung
// request stalls the whole download behind it with the screen still cheerfully
// saying "Downloading…". A failure a person can see beats a wait they cannot.

class NetTimeouts {
  NetTimeouts._();

  /// One map tile. Hundreds of these run in a row, so a slow one has to fail
  /// fast or the estimate stops meaning anything.
  static const tile = Duration(seconds: 20);

  /// One forecast. Small response, no excuse for being slow.
  static const weather = Duration(seconds: 30);

  /// One route. Larger, but still one request.
  static const route = Duration(seconds: 30);

  /// One Overpass query.
  ///
  /// LONGER THAN OVERPASS'S OWN SERVER-SIDE TIMEOUT ON PURPOSE. The query
  /// carries `[timeout:60]`, so a client timeout at or under sixty seconds
  /// would cut off queries that were about to return, and would do it most
  /// often on exactly the big corridor queries that matter.
  static const places = Duration(seconds: 90);
}
