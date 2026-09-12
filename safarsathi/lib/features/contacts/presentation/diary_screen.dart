// lib/features/contacts/presentation/diary_screen.dart
//
// The diary. Used one-handed, often in poor light, sometimes under stress.
//
// State: StreamBuilder over the DAO, deliberately no state-management
// dependency yet. Drift streams are already reactive; Riverpod arrives at the
// sync orchestrator (#25) and the DAO layer will not need to change.
//
// What a tap DOES is #7. This screen exposes onCopy and onOpen and leaves
// them to the caller.

import 'dart:async';

import 'package:flutter/material.dart';

import '../../../core/database/app_database.dart';
import '../../../core/theme/app_tokens.dart';
import '../../../core/widgets/retro.dart';
import '../data/contacts_dao.dart';
import 'diary_widgets.dart';

class DiaryScreen extends StatefulWidget {
  /// Streams rather than the DAO itself.
  ///
  /// The screen's job is to render whatever arrives and to say what it wants
  /// to see next; where the rows come from is the composition root's problem.
  /// It also means the screen can be tested without a database — which
  /// matters more than it sounds, because a widget test cannot close a Drift
  /// database or flush its stream-cleanup timers.
  final Stream<List<Contact>> Function(ContactFilter) watchContacts;
  final Stream<int> unconfirmedCount;

  final int tripId;
  final String tripName;
  final int? currentStopId;
  final String? currentStopName;

  /// Copies the number and returns what landed on the clipboard, which the
  /// screen then shows in the toast. Throwing an [Object] whose `toString`
  /// reads as a sentence surfaces as an error toast.
  final Future<String> Function(Contact)? onCopy;

  /// Opens the platform dialer with no number, for the paste.
  final Future<void> Function()? onOpenDialer;

  /// Wired at #8 — the entry screen.
  final void Function(Contact)? onOpen;
  final VoidCallback? onAdd;

  const DiaryScreen({
    super.key,
    required this.watchContacts,
    required this.unconfirmedCount,
    required this.tripId,
    required this.tripName,
    this.currentStopId,
    this.currentStopName,
    this.onCopy,
    this.onOpenDialer,
    this.onOpen,
    this.onAdd,
  });

  @override
  State<DiaryScreen> createState() => _DiaryScreenState();
}

class _DiaryScreenState extends State<DiaryScreen> {
  final _searchController = TextEditingController();
  late ContactFilter _filter = ContactFilter(tripId: widget.tripId);
  bool _stopScoped = false;

  String? _toastNumber;
  String? _toastMessage;
  Timer? _toastTimer;

  @override
  void dispose() {
    _toastTimer?.cancel();
    _searchController.dispose();
    super.dispose();
  }

  void _showToast({String? number, String? message}) {
    _toastTimer?.cancel();
    setState(() {
      _toastNumber = number;
      _toastMessage = message;
    });
    _toastTimer = Timer(const Duration(milliseconds: 3200), () {
      if (mounted) {
        setState(() {
          _toastNumber = null;
          _toastMessage = null;
        });
      }
    });
  }

  Future<void> _copy(Contact contact) async {
    final handler = widget.onCopy;
    if (handler == null) return;
    try {
      _showToast(number: await handler(contact));
    } on Object catch (e) {
      _showToast(message: e.toString());
    }
  }

  Future<void> _openDialer() async {
    final handler = widget.onOpenDialer;
    if (handler == null) return;
    try {
      await handler();
      _toastTimer?.cancel();
      if (mounted) {
        setState(() {
          _toastNumber = null;
          _toastMessage = null;
        });
      }
    } on Object catch (e) {
      _showToast(message: e.toString());
    }
  }

  void _setFilter(ContactFilter next) => setState(() => _filter = next);

  void _toggleStopScope() {
    _stopScoped = !_stopScoped;
    _setFilter(
      _stopScoped
          ? _filter.copyWith(stopId: widget.currentStopId)
          : _filter.copyWith(clearStop: true),
    );
  }

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);

    return Scaffold(
      backgroundColor: c.paper,
      // Nudged clear of the thumb index, which owns the right edge.
      floatingActionButton: widget.onAdd == null
          ? null
          : Padding(
              padding: const EdgeInsets.only(right: 34),
              child: FloatingActionButton.extended(
                onPressed: widget.onAdd,
                icon: const Icon(Icons.add),
                label: const Text('New entry', style: AppTokens.stencilStyle),
              ),
            ),
      body: GrainOverlay(
        child: SafeArea(
          child: Stack(
            children: [
              Column(
                children: [
                  _buildAppBar(c),
                  _buildSearch(c),
                  ReadinessBanner(unconfirmedCount: widget.unconfirmedCount),
                  Expanded(child: _buildPages(c)),
                ],
              ),
              if (_toastNumber != null || _toastMessage != null)
                Positioned(
                  left: AppTokens.s12,
                  right: 42,
                  // Above the New entry button, which would otherwise cover
                  // the Open dialer action — the whole point of the toast.
                  bottom: 76,
                  child: CopyToast(
                    number: _toastNumber,
                    message: _toastMessage,
                    onOpenDialer: widget.onOpenDialer == null
                        ? null
                        : _openDialer,
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }

  // --------------------------------------------------------------- app bar

  Widget _buildAppBar(AppColors c) {
    final scope = _stopScoped && widget.currentStopName != null
        ? '${widget.currentStopName} only'
        : 'Whole trip';

    return Padding(
      padding: const EdgeInsets.fromLTRB(
        AppTokens.gutter,
        AppTokens.s12,
        AppTokens.s8,
        0,
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  widget.tripName.toUpperCase(),
                  style: AppTokens.stencilStyle.copyWith(
                    fontSize: 10,
                    letterSpacing: 1.6,
                    color: c.muted,
                  ),
                  overflow: TextOverflow.ellipsis,
                ),
                const SizedBox(height: 2),
                Text(
                  'Diary',
                  style: AppTokens.titleStyle.copyWith(color: c.ink),
                ),
                Text(
                  widget.currentStopName == null
                      ? scope
                      : '${widget.currentStopName} · $scope',
                  style: AppTokens.captionStyle.copyWith(color: c.muted),
                  overflow: TextOverflow.ellipsis,
                ),
              ],
            ),
          ),
          if (widget.currentStopId != null)
            IconButton(
              tooltip: _stopScoped ? 'Showing this stop' : 'Showing whole trip',
              onPressed: _toggleStopScope,
              icon: Icon(
                _stopScoped ? Icons.place : Icons.travel_explore,
                color: _stopScoped ? c.signal : c.muted,
              ),
            ),
        ],
      ),
    );
  }

  // ---------------------------------------------------------------- search

  Widget _buildSearch(AppColors c) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(
        AppTokens.gutter,
        AppTokens.s8,
        AppTokens.gutter,
        AppTokens.s4,
      ),
      child: TextField(
        controller: _searchController,
        onChanged: (v) => _setFilter(_filter.copyWith(searchTerm: v)),
        style: AppTokens.rowTitleStyle.copyWith(color: c.ink),
        decoration: InputDecoration(
          hintText: 'Search name, number or note',
          hintStyle: AppTokens.rowTitleStyle.copyWith(color: c.muted),
          prefixIcon: Icon(Icons.search, size: 18, color: c.muted),
          prefixIconConstraints: const BoxConstraints(minWidth: 38),
          contentPadding: const EdgeInsets.symmetric(
            vertical: AppTokens.s8,
            horizontal: AppTokens.s8,
          ),
          suffixIcon: _searchController.text.isEmpty
              ? null
              : IconButton(
                  icon: Icon(Icons.close, size: 17, color: c.muted),
                  onPressed: () {
                    _searchController.clear();
                    _setFilter(_filter.copyWith(searchTerm: ''));
                  },
                ),
        ),
      ),
    );
  }

  // ----------------------------------------------------------------- pages

  Widget _buildPages(AppColors c) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Expanded(
          child: StreamBuilder<List<Contact>>(
            stream: widget.watchContacts(_filter),
            builder: (context, snap) {
              if (!snap.hasData) {
                return Center(
                  child: SizedBox(
                    width: 22,
                    height: 22,
                    child: CircularProgressIndicator(
                      strokeWidth: 2,
                      color: c.muted,
                    ),
                  ),
                );
              }
              final items = snap.data!;
              if (items.isEmpty) {
                return DiaryEmptyState(
                  searching:
                      _filter.searchTerm.isNotEmpty || _filter.category != null,
                  onAdd: widget.onAdd,
                );
              }
              return ListView.builder(
                padding: const EdgeInsets.only(bottom: 96),
                itemCount: items.length + 1,
                itemBuilder: (context, i) {
                  if (i == items.length) {
                    return _PageFooter(count: items.length);
                  }
                  return DiaryEntry(
                    contact: items[i],
                    lineNumber: i + 1,
                    onCopy: widget.onCopy == null
                        ? null
                        : () => _copy(items[i]),
                    onOpen: widget.onOpen == null
                        ? null
                        : () => widget.onOpen!(items[i]),
                  );
                },
              );
            },
          ),
        ),
        CategoryIndex(
          selected: _filter.category,
          onSelect: (category) => _setFilter(
            category == null
                ? _filter.copyWith(clearCategory: true)
                : _filter.copyWith(category: category),
          ),
        ),
      ],
    );
  }
}

class _PageFooter extends StatelessWidget {
  final int count;
  const _PageFooter({required this.count});

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    final style = AppTokens.stencilStyle.copyWith(
      fontSize: 9.5,
      letterSpacing: 1.3,
      color: c.muted,
    );
    return Padding(
      padding: const EdgeInsets.fromLTRB(
        AppTokens.s12,
        AppTokens.s12,
        AppTokens.s12,
        AppTokens.s16,
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text('Page 1 of 1', style: style),
          Text('$count ${count == 1 ? "entry" : "entries"}', style: style),
        ],
      ),
    );
  }
}
