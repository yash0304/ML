// lib/core/database/watch_tables.dart
//
// "Tell me when any of these tables changes" — the tick every screen stream
// is built on.
//
// WHY THIS EXISTS: every screen used to tick on
// `customSelect('SELECT 1', readsFrom: tables).watch()`. Drift shares one
// active stream between all queries with the same SQL and variables, and
// `readsFrom` is NOT part of that key (StreamKey in drift's
// stream_queries.dart). So every 'SELECT 1' in the app shared whichever was
// registered first — usually the active-trip stream, which watches trips and
// stops — and fired only when those changed. Found on the phone: an expense
// saved and the Money tab did not move until the app was left and reopened,
// because reopening subscribes afresh and a fresh subscription always fetches
// once. Tests never saw it: each ran one stream at a time.
//
// The fix is to make the SQL itself differ by dependency set. The comment is
// ignored by SQLite and is exactly what drift keys on.

import 'package:drift/drift.dart';

/// Emits once on listen, then whenever any of [tables] is written.
Stream<void> watchTables(
  GeneratedDatabase db,
  Set<ResultSetImplementation> tables,
) {
  final names = [for (final t in tables) t.entityName]..sort();
  return db
      .customSelect('SELECT 1 /* ${names.join(',')} */', readsFrom: tables)
      .watch()
      .map((_) {});
}
