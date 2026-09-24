// lib/features/trips/presentation/planned_stop_dialog.dart
//
// A stop on the way, typed: "Mawkdok Dympep viewpoint", "lunch at Pynursla".
// The downloaded places cover what OpenStreetMap knows; this covers what a
// driver or a friend told you about.

import 'package:flutter/material.dart';

import '../../../core/theme/app_tokens.dart';
import '../../contacts/data/place_location.dart';

typedef PlannedStopInput = ({
  String name,
  double? lat,
  double? lon,
  String? note,
});

Future<PlannedStopInput?> showPlannedStopDialog(
  BuildContext context, {
  required String legName,
}) => showDialog<PlannedStopInput>(
  context: context,
  builder: (_) => PlannedStopDialog(legName: legName),
);

class PlannedStopDialog extends StatefulWidget {
  final String legName;
  const PlannedStopDialog({super.key, required this.legName});

  @override
  State<PlannedStopDialog> createState() => _PlannedStopDialogState();
}

class _PlannedStopDialogState extends State<PlannedStopDialog> {
  final _name = TextEditingController();
  final _location = TextEditingController();
  final _note = TextEditingController();
  String? _nameError;
  String? _locationError;

  @override
  void dispose() {
    _name.dispose();
    _location.dispose();
    _note.dispose();
    super.dispose();
  }

  void _add() {
    final name = _name.text.trim();
    final place = parseLocation(_location.text);
    setState(() {
      _nameError = name.isEmpty ? 'A stop needs a name.' : null;
      _locationError = place.problem;
    });
    if (name.isEmpty || place.problem != null) return;
    Navigator.of(context).pop((
      name: name,
      lat: place.at?.lat,
      lon: place.at?.lon,
      note: _note.text.trim().isEmpty ? null : _note.text.trim(),
    ));
  }

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    InputDecoration deco(String label, String? hint, String? error) =>
        InputDecoration(labelText: label, hintText: hint, errorText: error);

    return AlertDialog(
      backgroundColor: c.paper,
      title: Text('A stop on ${widget.legName}'),
      content: SingleChildScrollView(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            TextField(
              key: const Key('planned-name'),
              controller: _name,
              autofocus: true,
              textCapitalization: TextCapitalization.words,
              decoration: deco('Name', 'Mawkdok Dympep viewpoint', _nameError),
            ),
            TextField(
              key: const Key('planned-location'),
              controller: _location,
              decoration: deco(
                'Location (optional)',
                'Paste from Google Maps',
                _locationError,
              ),
            ),
            TextField(
              key: const Key('planned-note'),
              controller: _note,
              decoration: deco('Note (optional)', 'Lunch, 30 min', null),
            ),
            const SizedBox(height: AppTokens.s12),
            Text(
              'With a location it sits at its kilometre on the road and '
              'gets directions. Without one it is still in the plan.',
              style: AppTokens.captionStyle.copyWith(color: c.muted),
            ),
          ],
        ),
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.of(context).pop(),
          child: const Text('Cancel'),
        ),
        TextButton(
          key: const Key('planned-add'),
          onPressed: _add,
          child: const Text('Add to the plan'),
        ),
      ],
    );
  }
}
