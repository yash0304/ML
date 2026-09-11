// SafarSathi — offline-first travel companion.
//
// Nothing in this app may make a network call at runtime. The only network
// activity happens once, at trip setup, on WiFi. See docs/safarsathi/DESIGN.md.

import 'package:flutter/material.dart';

import 'app.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const SafarSathiApp());
}
