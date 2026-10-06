// lib/features/trips/presentation/timeline_note_screen.dart
//
// A note on the timeline, with photos if you like — #30.

import 'package:flutter/material.dart';

import '../../../core/theme/app_tokens.dart';
import '../../../core/theme/motion.dart';
import '../../../core/widgets/retro.dart';

class TimelineNoteScreen extends StatefulWidget {
  /// Picks one photo and returns where it was copied, or null. [camera]
  /// opens the camera app; otherwise the system photo picker.
  final Future<String?> Function({required bool camera})? pickPhoto;

  final Future<void> Function(String text, List<String> photos) onSave;

  final Widget Function(String path) photo;

  const TimelineNoteScreen({
    super.key,
    required this.onSave,
    required this.photo,
    this.pickPhoto,
  });

  @override
  State<TimelineNoteScreen> createState() => _TimelineNoteScreenState();
}

class _TimelineNoteScreenState extends State<TimelineNoteScreen> {
  final _text = TextEditingController();
  final _photos = <String>[];
  bool _saving = false;

  @override
  void dispose() {
    _text.dispose();
    super.dispose();
  }

  bool get _canSave => _text.text.trim().isNotEmpty || _photos.isNotEmpty;

  Future<void> _add({required bool camera}) async {
    final path = await widget.pickPhoto!(camera: camera);
    if (path != null && mounted) setState(() => _photos.add(path));
  }

  Future<void> _save() async {
    if (!_canSave || _saving) {
      Haptics.reject();
      return;
    }
    setState(() => _saving = true);
    await widget.onSave(_text.text, List.of(_photos));
    Haptics.confirm();
    if (mounted) Navigator.of(context).maybePop();
  }

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    return Scaffold(
      backgroundColor: c.paper,
      appBar: AppBar(
        title: const Text('A note'),
        backgroundColor: c.paper,
        foregroundColor: c.ink,
        elevation: 0,
      ),
      body: ListView(
        padding: const EdgeInsets.only(bottom: 96),
        children: [
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: AppTokens.gutter),
            child: TextField(
              key: const Key('note-text'),
              controller: _text,
              autofocus: true,
              minLines: 3,
              maxLines: 8,
              textCapitalization: TextCapitalization.sentences,
              onChanged: (_) => setState(() {}),
              decoration: const InputDecoration(
                hintText: 'The fog lifted at Nohkalikai at 4 pm',
              ),
            ),
          ),
          if (widget.pickPhoto != null) ...[
            const StencilLabel('Photos'),
            Padding(
              padding: const EdgeInsets.symmetric(
                horizontal: AppTokens.gutter,
              ),
              child: Wrap(
                spacing: AppTokens.s8,
                runSpacing: AppTokens.s8,
                children: [
                  for (final p in _photos)
                    ClipRRect(
                      borderRadius: BorderRadius.circular(
                        AppTokens.radiusSoft,
                      ),
                      child: SizedBox(width: 84, height: 84, child: widget.photo(p)),
                    ),
                  OutlinedButton.icon(
                    key: const Key('note-photo-gallery'),
                    onPressed: () => _add(camera: false),
                    icon: const Icon(Icons.photo_library_outlined),
                    label: const Text('From photos'),
                  ),
                  OutlinedButton.icon(
                    key: const Key('note-photo-camera'),
                    onPressed: () => _add(camera: true),
                    icon: const Icon(Icons.photo_camera_outlined),
                    label: const Text('Take one'),
                  ),
                ],
              ),
            ),
            Padding(
              padding: const EdgeInsets.fromLTRB(
                AppTokens.gutter,
                AppTokens.s8,
                AppTokens.gutter,
                0,
              ),
              child: Text(
                'A copy is kept in the app, on this phone. No photo permission '
                'is asked for: the phone\'s own picker hands over only what '
                'you choose.',
                style: AppTokens.captionStyle.copyWith(color: c.muted),
              ),
            ),
          ],
        ],
      ),
      bottomNavigationBar: SafeArea(
        minimum: const EdgeInsets.all(AppTokens.s16),
        child: PressScale(
          key: const Key('note-save'),
          onTap: _save,
          child: Container(
            height: 48,
            alignment: Alignment.center,
            decoration: BoxDecoration(
              color: _canSave ? c.signal : c.stone,
              border: Border.all(color: _canSave ? c.ink : c.rule),
              borderRadius: BorderRadius.circular(AppTokens.radiusSoft),
            ),
            child: Text(
              'Add to the timeline',
              style: AppTokens.stencilStyle.copyWith(
                fontSize: 11.5,
                color: _canSave ? c.paper : c.muted,
              ),
            ),
          ),
        ),
      ),
    );
  }
}
