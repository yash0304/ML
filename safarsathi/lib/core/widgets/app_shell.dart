// lib/core/widgets/app_shell.dart
//
// The five-item bottom bar from SCREENS.md §0. Deferred at #6 because there
// was only one screen and a bar with four dead destinations is worse than no
// bar at all.
//
// RED LEAVES THE EMERGENCY SCREEN EXACTLY ONCE: the SOS item is emergency red
// while it is the active tab, and muted otherwise. Even then it is pointing
// at the emergency screen.

import 'package:flutter/material.dart';

import '../theme/app_tokens.dart';
import '../theme/motion.dart';

class ShellDestination {
  final String label;
  final IconData icon;
  final Widget screen;

  /// Only the emergency destination sets this.
  final bool emergency;

  const ShellDestination({
    required this.label,
    required this.icon,
    required this.screen,
    this.emergency = false,
  });
}

class AppShell extends StatefulWidget {
  final List<ShellDestination> destinations;
  final int initialIndex;

  const AppShell({
    super.key,
    required this.destinations,
    this.initialIndex = 0,
  });

  @override
  State<AppShell> createState() => _AppShellState();
}

class _AppShellState extends State<AppShell> {
  late int _index = widget.initialIndex;

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);

    return Scaffold(
      backgroundColor: c.paper,
      // IndexedStack rather than swapping the child, so switching away from a
      // half-typed search and back does not lose it.
      body: IndexedStack(
        index: _index,
        children: [for (final d in widget.destinations) d.screen],
      ),
      bottomNavigationBar: Container(
        decoration: BoxDecoration(
          color: c.paper,
          border: Border(
            top: BorderSide(color: c.rule, width: AppTokens.hairline),
          ),
        ),
        child: SafeArea(
          top: false,
          child: Row(
            children: [
              for (var i = 0; i < widget.destinations.length; i++)
                Expanded(
                  child: _Tab(
                    destination: widget.destinations[i],
                    active: i == _index,
                    onTap: () {
                      if (i == _index) return;
                      setState(() => _index = i);
                    },
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }
}

class _Tab extends StatelessWidget {
  final ShellDestination destination;
  final bool active;
  final VoidCallback onTap;

  const _Tab({
    required this.destination,
    required this.active,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    final tone = !active
        ? c.muted
        : destination.emergency
        ? c.emergency
        : c.ink;

    return Semantics(
      button: true,
      selected: active,
      label: destination.label,
      child: PressScale(
        onTap: onTap,
        feedback: Haptics.select,
        child: Container(
          padding: const EdgeInsets.only(top: 8, bottom: 9),
          decoration: BoxDecoration(
            border: Border(
              top: BorderSide(
                color: active ? tone : Colors.transparent,
                width: 2,
              ),
            ),
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(destination.icon, size: 19, color: tone),
              const SizedBox(height: 3),
              Text(
                destination.label.toUpperCase(),
                style: AppTokens.stencilStyle.copyWith(
                  fontSize: 9,
                  letterSpacing: 1.1,
                  color: tone,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

/// A destination that has not been built yet. Honest rather than empty.
class NotBuiltYet extends StatelessWidget {
  final String title;
  final String what;
  final String issue;

  const NotBuiltYet({
    super.key,
    required this.title,
    required this.what,
    required this.issue,
  });

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    return Scaffold(
      backgroundColor: c.paper,
      body: SafeArea(
        child: Center(
          child: Padding(
            padding: const EdgeInsets.all(AppTokens.s32),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(title, style: AppTokens.titleStyle.copyWith(color: c.ink)),
                const SizedBox(height: AppTokens.s8),
                Text(
                  what,
                  textAlign: TextAlign.center,
                  style: AppTokens.captionStyle.copyWith(color: c.muted),
                ),
                const SizedBox(height: AppTokens.s16),
                Text(
                  issue.toUpperCase(),
                  style: AppTokens.stencilStyle.copyWith(
                    fontSize: 10,
                    color: c.muted,
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
