// SafarSathi — offline-first travel companion.
//
// Nothing in this app may make a network call at runtime. The only network
// activity happens once, at trip setup, on WiFi. See docs/safarsathi/DESIGN.md.

import 'package:flutter/material.dart';

import 'app.dart';
import 'core/database/app_database.dart';
import 'core/database/seeding.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  final db = AppDatabase();

  // Awaited before the first frame, so the emergency screen is never briefly
  // empty on a fresh install. Idempotent — it runs on every launch.
  await seedReferenceData(db);

  runApp(SafarSathiRoot(db: db));
}
