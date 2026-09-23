// lib/features/contacts/presentation/entry_screen.dart
//
// One entry, as a diary page. SCREENS.md §2.
//
// THIS SCREEN OWNS THE MOST IMPORTANT ACTION IN THE APP. Promoting a number
// from userEntered to userVerified is what the amber dot, the readiness
// count and the blocking checklist items all read. The app cannot know
// whether a call connected, so it asks the user to assert it — and makes the
// assertion deliberate: a labelled button on a detail screen, never a swipe
// or a long-press that could happen by accident.

import 'package:flutter/material.dart';

import '../../../core/database/app_database.dart';
import '../../../core/theme/app_tokens.dart';
import '../../../core/theme/motion.dart';
import '../../../core/widgets/retro.dart';
import '../../discovery/data/geo.dart' show LatLng;
import '../data/contacts_dao.dart';
import '../data/place_location.dart';
import 'diary_widgets.dart';

class EntryScreen extends StatefulWidget {
  final Contact contact;

  /// Where this entry is attached, resolved by the caller.
  final String? stopName;

  /// Copies and returns what landed on the clipboard.
  final Future<String> Function(Contact)? onCopy;
  final Future<void> Function()? onOpenDialer;
  final Future<void> Function(Contact)? onCall;
  final Future<void> Function(Contact)? onChat;

  /// Promotes or demotes the tier. Returns when the write has landed.
  final Future<void> Function(Contact, {required bool confirmed})? onConfirm;

  final void Function(Contact)? onEdit;

  /// The offline map for a placed entry. Built by the caller, which owns the
  /// tiles; null draws no map.
  final Widget Function(BuildContext context, LatLng place)? placeMap;

  /// Opens a Google Maps link. Null hides the directions button.
  final Future<void> Function(String url)? onOpenMaps;

  /// Reads the time a confirmation happened. Injectable because a golden that
  /// confirms a contact would otherwise bake today's date into the image and
  /// fail on every later day — which it duly did.
  final DateTime Function() clock;

  const EntryScreen({
    super.key,
    required this.contact,
    this.stopName,
    this.onCopy,
    this.onOpenDialer,
    this.onCall,
    this.onChat,
    this.onConfirm,
    this.onEdit,
    this.placeMap,
    this.onOpenMaps,
    this.clock = DateTime.now,
  });

  @override
  State<EntryScreen> createState() => _EntryScreenState();
}

class _EntryScreenState extends State<EntryScreen> {
  late bool _confirmed = widget.contact.callConfirmed;
  late DateTime? _confirmedAt = widget.contact.confirmedAt;
  bool _working = false;
  String? _toast;

  ContactTier get _tier {
    final stored = ContactTier.parse(widget.contact.tier);
    if (_confirmed) return ContactTier.userVerified;
    // Clearing a user confirmation drops back to userEntered, mirroring
    // ContactsDao.markConfirmed. Without this the line would still read
    // "Confirmed by you" after the user had just cleared it.
    if (stored == ContactTier.userVerified) return ContactTier.userEntered;
    return stored;
  }

  /// Government tiers are authoritative by provenance, not by anyone having
  /// dialled them, so there is nothing here for the user to assert.
  bool get _userCanConfirm {
    final stored = ContactTier.parse(widget.contact.tier);
    return stored != ContactTier.verifiedNational &&
        stored != ContactTier.verifiedState;
  }

  Future<void> _run(Future<void> Function() action) async {
    try {
      await action();
    } on Object catch (e) {
      if (mounted) setState(() => _toast = e.toString());
    }
  }

  Future<void> _copy() async {
    final handler = widget.onCopy;
    if (handler == null) return;
    await _run(() async {
      final number = await handler(widget.contact);
      if (mounted) setState(() => _toast = 'Copied  $number');
    });
  }

  Future<void> _toggleConfirmed() async {
    final handler = widget.onConfirm;
    if (handler == null || _working) return;
    final next = !_confirmed;
    setState(() => _working = true);
    await _run(() => handler(widget.contact, confirmed: next));
    if (!mounted) return;
    setState(() {
      _working = false;
      _confirmed = next;
      _confirmedAt = next ? widget.clock() : null;
    });
  }

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    final contact = widget.contact;

    return Scaffold(
      backgroundColor: c.paper,
      body: GrainOverlay(
        child: SafeArea(
          child: Stack(
            children: [
              ListView(
                padding: const EdgeInsets.only(bottom: AppTokens.s32),
                children: [
                  _header(c, contact),
                  _number(c, contact),
                  _primaryActions(c, contact),
                  const StencilLabel('Where it is'),
                  _where(c, contact),
                  const StencilLabel('Record'),
                  _record(c, contact),
                  if (_userCanConfirm) ...[
                    const StencilLabel('Confirm'),
                    _confirmSection(c),
                  ],
                ],
              ),
              if (_toast != null)
                Positioned(
                  left: AppTokens.s12,
                  right: AppTokens.s12,
                  bottom: AppTokens.s12,
                  child: PressScale(
                    onTap: () => setState(() => _toast = null),
                    child: CopyToast(
                      number: _toast!.startsWith('Copied')
                          ? _toast!.substring(7).trim()
                          : null,
                      message: _toast!.startsWith('Copied') ? null : _toast,
                      onOpenDialer: widget.onOpenDialer == null
                          ? null
                          : () => _run(() async {
                              await widget.onOpenDialer!();
                              if (mounted) setState(() => _toast = null);
                            }),
                    ),
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _header(AppColors c, Contact contact) {
    final category =
        ContactCategory.labels[contact.category] ?? contact.category;
    return Padding(
      padding: const EdgeInsets.fromLTRB(
        AppTokens.s8,
        AppTokens.s12,
        AppTokens.gutter,
        0,
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          IconButton(
            onPressed: () => Navigator.of(context).maybePop(),
            icon: Icon(Icons.arrow_back, color: c.muted),
            tooltip: 'Back',
          ),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  category.toUpperCase(),
                  style: AppTokens.stencilStyle.copyWith(
                    fontSize: 10,
                    letterSpacing: 1.6,
                    color: c.muted,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  contact.name,
                  style: AppTokens.titleStyle.copyWith(color: c.ink),
                ),
              ],
            ),
          ),
          if (widget.onEdit != null)
            PressScale(
              onTap: () => widget.onEdit!(contact),
              child: Padding(
                padding: const EdgeInsets.only(top: AppTokens.s8),
                child: Text(
                  'Edit',
                  style: AppTokens.stencilStyle.copyWith(
                    fontSize: 10.5,
                    color: c.signal,
                  ),
                ),
              ),
            ),
        ],
      ),
    );
  }

  Widget _number(AppColors c, Contact contact) {
    final trusted = _tier.isTrusted;
    return Padding(
      padding: const EdgeInsets.fromLTRB(
        AppTokens.gutter,
        AppTokens.s16,
        AppTokens.gutter,
        0,
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Selectable so a long press picks it up natively too, not only
          // through the copy button.
          SelectableText(
            contact.phoneRaw,
            style: AppTokens.numberStyle.copyWith(fontSize: 25, color: c.ink),
          ),
          const SizedBox(height: AppTokens.s8),
          Row(
            children: [
              if (!trusted)
                Container(
                  width: 7,
                  height: 7,
                  margin: const EdgeInsets.only(right: AppTokens.s8),
                  decoration: BoxDecoration(
                    color: c.cautionMark,
                    shape: BoxShape.circle,
                  ),
                ),
              Expanded(
                child: Text(
                  _provenance(contact),
                  style: AppTokens.captionStyle.copyWith(
                    color: trusted ? c.signal : c.caution,
                  ),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  String _provenance(Contact contact) {
    switch (_tier) {
      case ContactTier.verifiedNational:
        return 'Government short code. Authoritative.';
      case ContactTier.verifiedState:
        return 'Published by the state government.';
      case ContactTier.userVerified:
        final when = _confirmedAt;
        return when == null
            ? 'Confirmed by you · you reached this number'
            : 'Confirmed by you, ${_date(when)} · you reached this number';
      case ContactTier.communityOsm:
        return 'From open map data · nobody has checked it';
      case ContactTier.userEntered:
        return 'Typed by you · not confirmed';
    }
  }

  static String _date(DateTime d) {
    const months = [
      'Jan',
      'Feb',
      'Mar',
      'Apr',
      'May',
      'Jun',
      'Jul',
      'Aug',
      'Sep',
      'Oct',
      'Nov',
      'Dec',
    ];
    return '${d.day} ${months[d.month - 1]}';
  }

  Widget _primaryActions(AppColors c, Contact contact) {
    return Column(
      children: [
        Padding(
          padding: const EdgeInsets.fromLTRB(
            AppTokens.gutter,
            AppTokens.s16,
            AppTokens.gutter,
            0,
          ),
          child: _Button(
            label: 'Copy number',
            icon: Icons.copy_rounded,
            onTap: widget.onCopy == null ? null : _copy,
          ),
        ),
        Padding(
          padding: const EdgeInsets.fromLTRB(
            AppTokens.gutter,
            AppTokens.s8,
            AppTokens.gutter,
            0,
          ),
          child: Row(
            children: [
              Expanded(
                child: _Button(
                  label: 'Dialer',
                  icon: Icons.dialpad,
                  ghost: true,
                  onTap: widget.onOpenDialer == null
                      ? null
                      : () => _run(widget.onOpenDialer!),
                ),
              ),
              const SizedBox(width: AppTokens.s8),
              Expanded(
                child: _Button(
                  label: 'Call',
                  icon: Icons.call,
                  ghost: true,
                  onTap: widget.onCall == null
                      ? null
                      : () => _run(() => widget.onCall!(contact)),
                ),
              ),
              const SizedBox(width: AppTokens.s8),
              Expanded(
                child: _Button(
                  label: 'Chat',
                  icon: Icons.chat_bubble_outline,
                  ghost: true,
                  onTap: widget.onChat == null
                      ? null
                      : () => _run(() => widget.onChat!(contact)),
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }

  LatLng? _placeOf(Contact contact) =>
      contact.lat == null || contact.lon == null
      ? null
      : LatLng(contact.lat!, contact.lon!);

  Widget _where(AppColors c, Contact contact) {
    final place = _placeOf(contact);
    final open = widget.onOpenMaps;
    final caption = AppTokens.captionStyle.copyWith(color: c.muted);
    const pad = EdgeInsets.symmetric(horizontal: AppTokens.gutter);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        if (place != null && widget.placeMap != null)
          // Keyed on the position, so a location changed in Edit draws a
          // new map rather than the old one's cached coverage.
          KeyedSubtree(
            key: ValueKey('${place.lat},${place.lon}'),
            child: widget.placeMap!(context, place),
          ),
        if (place == null)
          Padding(
            padding: pad,
            child: Text(
              'No location saved for this number. Tap Edit and paste it from '
              'Google Maps to see it on the map here.',
              style: caption,
            ),
          ),
        if (open != null) ...[
          Padding(
            padding: const EdgeInsets.fromLTRB(
              AppTokens.gutter,
              AppTokens.s12,
              AppTokens.gutter,
              0,
            ),
            child: _Button(
              key: const Key('entry-directions'),
              label: place == null ? 'Search in Google Maps' : 'Directions',
              icon: place == null ? Icons.search : Icons.directions,
              ghost: true,
              onTap: () => _run(
                () => open(
                  place == null
                      ? mapsSearchUrl(contact.name, widget.stopName)
                      : directionsUrl(place),
                ),
              ),
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
              // Said before the tap, not after a dead one on a hill road.
              place == null
                  ? 'Needs signal. A search by name — check it found the '
                        'right place before you set off.'
                  : 'Opens Google Maps, starting from where you are. Needs '
                        'signal — unless you saved this area as an offline '
                        'map in Google Maps, which then gives driving '
                        'directions with none.',
              style: caption,
            ),
          ),
        ],
      ],
    );
  }

  Widget _record(AppColors c, Contact contact) {
    return Column(
      // Without stretch each row shrinks to its content and the rules render
      // as short centred stubs instead of ruling the page.
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        _Row(label: 'Attached to', value: widget.stopName ?? 'Whole trip'),
        _Row(
          label: 'Note',
          value: (contact.note?.trim().isEmpty ?? true)
              ? null
              : contact.note!.trim(),
          placeholder: 'None',
        ),
        _Row(
          label: 'Source',
          value: contact.importBatchId != null
              ? 'Imported from a sheet'
              : 'Typed by you',
        ),
        // Not "last called": a copy counts, because the dial happens in the
        // Android dialer after a paste.
        _Row(
          label: 'Last action',
          value: contact.lastCalledAt == null
              ? null
              : '${_date(contact.lastCalledAt!)} · '
                    '${contact.callCount} in total',
          placeholder: 'Never used',
          last: true,
        ),
      ],
    );
  }

  Widget _confirmSection(AppColors c) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(
        AppTokens.gutter,
        0,
        AppTokens.gutter,
        0,
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Expanded(
                child: Text(
                  _confirmed
                      ? 'Confirmed. Clear it if the number stops working — '
                            'one that worked in October may not in November.'
                      : 'Only after you have actually called it. The app '
                            'cannot tell whether a call connected, so this '
                            'is yours to say.',
                  style: AppTokens.captionStyle.copyWith(color: c.muted),
                ),
              ),
              const SizedBox(width: AppTokens.s12),
              StampBadge(label: 'Confirmed', landed: _confirmed),
            ],
          ),
          const SizedBox(height: AppTokens.s12),
          _Button(
            label: _confirmed ? 'Clear confirmation' : 'Mark confirmed',
            ghost: _confirmed,
            onTap: widget.onConfirm == null ? null : _toggleConfirmed,
          ),
        ],
      ),
    );
  }
}

// ---------------------------------------------------------------------------

class _Row extends StatelessWidget {
  final String label;
  final String? value;
  final String placeholder;
  final bool last;

  const _Row({
    required this.label,
    this.value,
    this.placeholder = '—',
    this.last = false,
  });

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: AppTokens.gutter),
      padding: const EdgeInsets.only(top: AppTokens.s12, bottom: AppTokens.s8),
      decoration: BoxDecoration(
        border: last
            ? null
            : Border(
                bottom: BorderSide(color: c.rule, width: AppTokens.hairline),
              ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            label.toUpperCase(),
            style: AppTokens.stencilStyle.copyWith(
              fontSize: 9.5,
              color: c.muted,
            ),
          ),
          const SizedBox(height: AppTokens.s4),
          Text(
            value ?? placeholder,
            style: AppTokens.rowTitleStyle.copyWith(
              color: value == null ? c.muted : c.ink,
            ),
          ),
        ],
      ),
    );
  }
}

class _Button extends StatelessWidget {
  final String label;
  final IconData? icon;
  final bool ghost;
  final VoidCallback? onTap;

  const _Button({
    super.key,
    required this.label,
    this.icon,
    this.ghost = false,
    this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    final enabled = onTap != null;
    final foreground = ghost ? c.ink : c.paper;

    return PressScale(
      onTap: onTap,
      feedback: ghost ? Haptics.light : Haptics.confirm,
      child: Opacity(
        opacity: enabled ? 1 : 0.4,
        child: Container(
          padding: const EdgeInsets.symmetric(vertical: AppTokens.s12),
          decoration: BoxDecoration(
            color: ghost ? null : c.signal,
            border: Border.all(color: ghost ? c.rule : c.ink),
            borderRadius: BorderRadius.circular(AppTokens.radiusSoft),
          ),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              if (icon != null) ...[
                Icon(icon, size: 16, color: foreground),
                const SizedBox(width: 6),
              ],
              Flexible(
                child: Text(
                  label,
                  overflow: TextOverflow.ellipsis,
                  style: AppTokens.stencilStyle.copyWith(
                    fontSize: 10.5,
                    color: foreground,
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
