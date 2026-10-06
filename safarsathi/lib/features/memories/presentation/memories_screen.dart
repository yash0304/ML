// lib/features/memories/presentation/memories_screen.dart
//
// The Memories tab: the trip's photos as an album, by day and place.
// Tap one to see it whole; swipe through the rest from there.

import 'package:flutter/material.dart';

import '../../../core/theme/app_tokens.dart';
import '../../../core/theme/motion.dart';
import '../../../core/widgets/retro.dart';
import '../data/memories.dart';

const _months = [
  'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
];

String _date(DateTime d) => '${d.day} ${_months[d.month - 1]}';

String _time(DateTime t) =>
    '${t.hour.toString().padLeft(2, '0')}:'
    '${t.minute.toString().padLeft(2, '0')}';

class MemoriesScreen extends StatelessWidget {
  final String tripName;
  final Stream<List<MemoryDay>> memories;
  final VoidCallback onAdd;

  /// Draws a stored photo small, for the grid, and whole, for the viewer.
  /// Injected so tests need no files.
  final Widget Function(String path) thumb;
  final Widget Function(String path) photo;

  final Future<void> Function(Memory m) onShare;
  final Future<void> Function(Memory m) onDelete;
  final Future<void> Function(int entryId, String text) onCaption;

  const MemoriesScreen({
    super.key,
    required this.tripName,
    required this.memories,
    required this.onAdd,
    required this.thumb,
    required this.photo,
    required this.onShare,
    required this.onDelete,
    required this.onCaption,
  });

  void _open(BuildContext context, List<Memory> all, int index) {
    Haptics.select();
    Navigator.of(context).push(
      MaterialPageRoute<void>(
        builder: (_) => MemoryViewer(
          memories: all,
          initialIndex: index,
          photo: photo,
          onShare: onShare,
          onDelete: onDelete,
          onCaption: onCaption,
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    final caption = AppTokens.captionStyle.copyWith(color: c.muted);
    return Scaffold(
      backgroundColor: c.paper,
      floatingActionButton: FloatingActionButton.extended(
        key: const Key('memories-add'),
        onPressed: onAdd,
        icon: const Icon(Icons.add_photo_alternate_outlined, size: 20),
        label: const Text('Add photos'),
      ),
      body: SafeArea(
        child: StreamBuilder<List<MemoryDay>>(
          stream: memories,
          builder: (context, snap) {
            final days = snap.data ?? const <MemoryDay>[];
            final all = [for (final d in days) ...d.photos];
            return ListView(
              padding: const EdgeInsets.only(bottom: 96),
              children: [
                Padding(
                  padding: const EdgeInsets.fromLTRB(
                    AppTokens.gutter,
                    AppTokens.s24,
                    AppTokens.gutter,
                    0,
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        tripName.toUpperCase(),
                        style: AppTokens.stencilStyle.copyWith(
                          color: c.muted,
                        ),
                      ),
                      Text(
                        'Memories',
                        style: AppTokens.titleStyle.copyWith(color: c.ink),
                      ),
                      Text(
                        all.isEmpty
                            ? 'No photos yet'
                            : '${all.length} '
                                  '${all.length == 1 ? 'photo' : 'photos'}'
                                  ' · ${days.length} '
                                  '${days.length == 1 ? 'day' : 'days'}',
                        key: const Key('memories-count'),
                        style: caption,
                      ),
                    ],
                  ),
                ),
                if (snap.hasData && all.isEmpty)
                  Padding(
                    padding: const EdgeInsets.all(AppTokens.gutter),
                    child: Text(
                      'Keep a few photos of the trip here: the root bridge, '
                      'the fog at Nohkalikai, the people you met. Tap Add '
                      'photos, choose them, and say where they were taken. '
                      'They show on the Timeline too.',
                      style: caption,
                    ),
                  ),
                for (final d in days) ...[
                  StencilLabel(
                    [_date(d.day), ...d.places].join(' · '),
                  ),
                  Padding(
                    padding: const EdgeInsets.symmetric(
                      horizontal: AppTokens.gutter,
                    ),
                    child: GridView.count(
                      crossAxisCount: 3,
                      mainAxisSpacing: AppTokens.s4,
                      crossAxisSpacing: AppTokens.s4,
                      shrinkWrap: true,
                      physics: const NeverScrollableScrollPhysics(),
                      children: [
                        for (final m in d.photos)
                          PressScale(
                            key: Key('memory-${m.path}'),
                            onTap: () => _open(context, all, all.indexOf(m)),
                            child: ClipRRect(
                              borderRadius: BorderRadius.circular(
                                AppTokens.radiusSoft,
                              ),
                              child: ColoredBox(
                                color: c.stone,
                                child: thumb(m.path),
                              ),
                            ),
                          ),
                      ],
                    ),
                  ),
                ],
                if (all.isNotEmpty)
                  Padding(
                    padding: const EdgeInsets.fromLTRB(
                      AppTokens.gutter,
                      AppTokens.s24,
                      AppTokens.gutter,
                      0,
                    ),
                    child: Text(
                      'Kept on this phone only, and not in the backup file. '
                      'To keep one somewhere else, open it and tap Share.',
                      style: caption,
                    ),
                  ),
              ],
            );
          },
        ),
      ),
    );
  }
}

/// One photo whole, pinch to zoom, swipe for the next.
class MemoryViewer extends StatefulWidget {
  final List<Memory> memories;
  final int initialIndex;
  final Widget Function(String path) photo;
  final Future<void> Function(Memory m) onShare;
  final Future<void> Function(Memory m) onDelete;
  final Future<void> Function(int entryId, String text) onCaption;

  const MemoryViewer({
    super.key,
    required this.memories,
    required this.initialIndex,
    required this.photo,
    required this.onShare,
    required this.onDelete,
    required this.onCaption,
  });

  @override
  State<MemoryViewer> createState() => _MemoryViewerState();
}

class _MemoryViewerState extends State<MemoryViewer> {
  late final List<Memory> _memories = List.of(widget.memories);
  late int _index = widget.initialIndex;
  late final _pages = PageController(initialPage: widget.initialIndex);

  /// Captions changed here, by note, so the screen shows the new words
  /// without waiting to be reopened.
  final _captions = <int, String?>{};

  @override
  void dispose() {
    _pages.dispose();
    super.dispose();
  }

  Memory get _current => _memories[_index];

  String? _captionOf(Memory m) =>
      _captions.containsKey(m.entry.id) ? _captions[m.entry.id] : m.caption;

  Future<void> _delete() async {
    final m = _current;
    final yes = await showDialog<bool>(
      context: context,
      builder: (d) => AlertDialog(
        title: const Text('Delete this photo?'),
        content: const Text(
          'It is removed from Memories and the Timeline, and its copy in the '
          'app is deleted. The original in your phone\'s gallery, if it came '
          'from there, is not touched.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(d).pop(false),
            child: const Text('Keep'),
          ),
          TextButton(
            key: const Key('memory-delete-confirm'),
            onPressed: () => Navigator.of(d).pop(true),
            child: const Text('Delete'),
          ),
        ],
      ),
    );
    if (yes != true) return;
    await widget.onDelete(m);
    Haptics.confirm();
    if (!mounted) return;
    if (_memories.length == 1) {
      Navigator.of(context).maybePop();
      return;
    }
    setState(() {
      _memories.removeAt(_index);
      if (_index >= _memories.length) _index = _memories.length - 1;
    });
  }

  Future<void> _editCaption() async {
    final m = _current;
    final saved = await showDialog<String>(
      context: context,
      builder: (_) => _CaptionDialog(
        initial: _captionOf(m) ?? '',
        shared: (m.entry.photoPaths ?? '').split('\n').length > 1,
      ),
    );
    if (saved == null) return;
    await widget.onCaption(m.entry.id, saved);
    if (mounted) {
      setState(
        () => _captions[m.entry.id] = saved.trim().isEmpty
            ? null
            : saved.trim(),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    final m = _current;
    final words = _captionOf(m);
    return Scaffold(
      backgroundColor: c.paper,
      appBar: AppBar(
        backgroundColor: c.paper,
        foregroundColor: c.ink,
        elevation: 0,
        title: Text(
          '${_index + 1} of ${_memories.length}',
          style: AppTokens.stencilStyle.copyWith(color: c.muted),
        ),
        actions: [
          IconButton(
            key: const Key('memory-share'),
            tooltip: 'Share or save elsewhere',
            icon: const Icon(Icons.ios_share),
            onPressed: () => widget.onShare(m),
          ),
          IconButton(
            key: const Key('memory-delete'),
            tooltip: 'Delete',
            icon: const Icon(Icons.delete_outline),
            onPressed: _delete,
          ),
        ],
      ),
      body: Column(
        children: [
          Expanded(
            child: PageView.builder(
              controller: _pages,
              itemCount: _memories.length,
              onPageChanged: (i) => setState(() => _index = i),
              itemBuilder: (_, i) => InteractiveViewer(
                maxScale: 4,
                child: Center(child: widget.photo(_memories[i].path)),
              ),
            ),
          ),
          Container(
            width: double.infinity,
            padding: const EdgeInsets.fromLTRB(
              AppTokens.gutter,
              AppTokens.s12,
              AppTokens.gutter,
              AppTokens.s16,
            ),
            decoration: BoxDecoration(
              border: Border(
                top: BorderSide(color: c.rule, width: AppTokens.hairline),
              ),
            ),
            child: SafeArea(
              top: false,
              child: PressScale(
                key: const Key('memory-caption'),
                onTap: _editCaption,
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      words ?? 'Add a few words',
                      style: AppTokens.rowTitleStyle.copyWith(
                        color: words == null ? c.muted : c.ink,
                      ),
                    ),
                    const SizedBox(height: 2),
                    Text(
                      [
                        _date(m.at),
                        _time(m.at),
                        if (m.place != null) m.place!,
                      ].join(' · '),
                      style: AppTokens.captionStyle.copyWith(color: c.muted),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

/// Owns its text controller, so it outlives the dialog's closing animation.
class _CaptionDialog extends StatefulWidget {
  final String initial;

  /// The words are the note's, shown under every photo added with it.
  final bool shared;

  const _CaptionDialog({required this.initial, required this.shared});

  @override
  State<_CaptionDialog> createState() => _CaptionDialogState();
}

class _CaptionDialogState extends State<_CaptionDialog> {
  late final _text = TextEditingController(text: widget.initial);

  @override
  void dispose() {
    _text.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => AlertDialog(
    title: const Text('A few words'),
    content: TextField(
      key: const Key('memory-caption-field'),
      controller: _text,
      autofocus: true,
      minLines: 1,
      maxLines: 4,
      textCapitalization: TextCapitalization.sentences,
      decoration: InputDecoration(
        helperText: widget.shared
            ? 'Shown under every photo added with this one'
            : null,
      ),
    ),
    actions: [
      TextButton(
        onPressed: () => Navigator.of(context).pop(),
        child: const Text('Cancel'),
      ),
      TextButton(
        key: const Key('memory-caption-save'),
        onPressed: () => Navigator.of(context).pop(_text.text),
        child: const Text('Save'),
      ),
    ],
  );
}
