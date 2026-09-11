import 'package:flutter/material.dart';

import 'core/theme/app_tokens.dart';
import 'features/dev/smoke_screen.dart';

class SafarSathiApp extends StatelessWidget {
  const SafarSathiApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'SafarSathi',
      debugShowCheckedModeBanner: false,
      theme: AppTokens.light,
      darkTheme: AppTokens.dark,
      // Follows the system for now. A manual override lands in Settings at
      // backlog #35 — a phone in a pocket does not know it is night in a
      // valley.
      themeMode: ThemeMode.system,
      home: const SmokeScreen(),
    );
  }
}
