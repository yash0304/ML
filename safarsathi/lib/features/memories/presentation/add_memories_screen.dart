// lib/features/memories/presentation/add_memories_screen.dart
//
// A few photos into Memories: choose them, say where, add a few words.

import 'package:flutter/material.dart';

import '../../../core/database/app_database.dart';
import '../../../core/theme/app_tokens.dart';
import '../../../core/theme/motion.dart';
import '../../../core/widgets/retro.dart';

class AddMemoriesScreen extends StatefulWidget {
  final List<Stop> stops;

  /// Where the trip is now; chosen to begin with.
  final int? initialStopId;

  /// Several from the phone's photo picker, or one from the camera app.
  /// Each returns the copies made in the app's folder.
  final Future<List<String>> Function() pickMany;
  final Future<String?> Function() takeOne;

  final Future<void> Function(List<String> photos, String words, int? stopId)
  onSave;

  /// Copies that were picked and then not saved, to be deleted.
  final Future<void> Function(List<String> photos) onDiscard;

  final Widget Function(String path) thumb;

  /// Open the phone's picker straight away, since that is what the person
  /// came for.
  final bool pickOnOpen;

  const AddMemoriesScreen({
    super.key,
    required this.stops,
    required this.pickMany,
    required this.takeOne,
    required this.onSave,
    required this.onDiscard,
    required this.thumb,
    this.initialStopId,
    this.pickOnOpen = false,
  });

  @override
  State<AddMemoriesScreen> createState() => _AddMemoriesScreenState();
}

class _AddMemoriesScreenState extends State<AddMemoriesScreen> {
  final _photos = <String>[];
  final _words = TextEditingController();
  late int? _stopId = widget.initialStopId;
  bool _saved = false;
  bool _saving = false;

  @override
  void initState() {
    super.initState();
    if (widget.pickOnOpen) {
      WidgetsBinding.instance.addPostFrameCallback((_) => _choose());
    }
  }

  @override
  void dispose() {
    _words.dispose();
    // Left without saving: the copies would sit in the app's folder with
    // nothing pointing at them.
    if (!_saved && _photos.isNotEmpty) widget.onDiscard(List.of(_photos));
    super.dispose();
  }

  Future<void> _choose() async {
    final picked = await widget.pickMany();
    if (picked.isNotEmpty && mounted) setState(() => _photos.addAll(picked));
  }

  Future<void> _take() async {
    final p = await widget.takeOne();
    if (p != null && mounted) setState(() => _photos.add(p));
  }

  Future<void> _remove(String p) async {
    setState(() => _photos.remove(p));
    await widget.onDiscard([p]);
  }

  Future<void> _save() async {
    if (_photos.isEmpty || _saving) {
      Haptics.reject();
      return;
    }
    setState(() => _saving = true);
    await widget.onSave(List.of(_photos), _words.text, _stopId);
    _saved = true;
    Haptics.confirm();
    if (mounted) Navigator.of(context).maybePop();
  }

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    final n = _photos.length;
    return Scaffold(
      backgroundColor: c.paper,
      appBar: AppBar(
        title: const Text('Add to Memories'),
        backgroundColor: c.paper,
        foregroundColor: c.ink,
        elevation: 0,
      ),
      body: ListView(
        padding: const EdgeInsets.only(bottom: 96),
        children: [
          const StencilLabel('Photos'),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: AppTokens.gutter),
            child: Wrap(
              spacing: AppTokens.s8,
              runSpacing: AppTokens.s8,
              children: [
                for (final p in _photos)
                  SizedBox(
                    width: 96,
                    height: 96,
                    child: Stack(
                      fit: StackFit.expand,
                      children: [
                        ClipRRect(
                          borderRadius: BorderRadius.circular(
                            AppTokens.radiusSoft,
                          ),
                          child: widget.thumb(p),
                        ),
                        Positioned(
                          top: 2,
                          right: 2,
                          child: GestureDetector(
                            key: Key('memory-remove-$p'),
                            onTap: () => _remove(p),
                            child: Container(
                              padding: const EdgeInsets.all(2),
                              decoration: BoxDecoration(
                                color: c.paper,
                                shape: BoxShape.circle,
                                border: Border.all(color: c.ink),
                              ),
                              child: Icon(Icons.close, size: 14, color: c.ink),
                            ),
                          ),
                        ),
                      ],
                    ),
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
            child: Wrap(
              spacing: AppTokens.s8,
              runSpacing: AppTokens.s8,
              children: [
                OutlinedButton.icon(
                  key: const Key('memory-choose'),
                  onPressed: _choose,
                  icon: const Icon(Icons.photo_library_outlined),
                  label: Text(n == 0 ? 'Choose photos' : 'Choose more'),
                ),
                OutlinedButton.icon(
                  key: const Key('memory-camera'),
                  onPressed: _take,
                  icon: const Icon(Icons.photo_camera_outlined),
                  label: const Text('Take one'),
                ),
              ],
            ),
          ),
          if (widget.stops.isNotEmpty) ...[
            const StencilLabel('Where'),
            Padding(
              padding: const EdgeInsets.symmetric(
                horizontal: AppTokens.gutter,
              ),
              child: Wrap(
                spacing: AppTokens.s8,
                runSpacing: AppTokens.s8,
                children: [
                  for (final s in widget.stops)
                    ChoiceChip(
                      key: Key('memory-stop-${s.id}'),
                      label: Text(s.name),
                      selected: _stopId == s.id,
                      onSelected: (on) =>
                          setState(() => _stopId = on ? s.id : null),
                    ),
                ],
              ),
            ),
          ],
          const StencilLabel('A few words'),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: AppTokens.gutter),
            child: TextField(
              key: const Key('memory-words'),
              controller: _words,
              minLines: 1,
              maxLines: 4,
              textCapitalization: TextCapitalization.sentences,
              decoration: const InputDecoration(
                hintText: 'Double-decker root bridge, after 3,000 steps',
              ),
            ),
          ),
          Padding(
            padding: const EdgeInsets.fromLTRB(
              AppTokens.gutter,
              AppTokens.s12,
              AppTokens.gutter,
              0,
            ),
            child: Text(
              'Added after the trip, they are filed under the day you reached '
              'that place. A copy is kept in the app, on this phone; the '
              'phone\'s own picker hands over only what you choose, so no '
              'photo permission is asked for.',
              style: AppTokens.captionStyle.copyWith(color: c.muted),
            ),
          ),
        ],
      ),
      bottomNavigationBar: SafeArea(
        minimum: const EdgeInsets.all(AppTokens.s16),
        child: PressScale(
          key: const Key('memory-save'),
          onTap: _save,
          child: Container(
            height: 48,
            alignment: Alignment.center,
            decoration: BoxDecoration(
              color: n > 0 ? c.signal : c.stone,
              border: Border.all(color: n > 0 ? c.ink : c.rule),
              borderRadius: BorderRadius.circular(AppTokens.radiusSoft),
            ),
            child: Text(
              n == 0
                  ? 'Choose a photo first'
                  : 'Keep ${n == 1 ? 'this photo' : 'these $n photos'}',
              style: AppTokens.stencilStyle.copyWith(
                fontSize: 11.5,
                color: n > 0 ? c.paper : c.muted,
              ),
            ),
          ),
        ),
      ),
    );
  }
}
