// lib/features/contacts/presentation/dialer_screen.dart
//
// The dialer. Used one-handed, often in poor light, sometimes under stress.
// Every design choice here serves glanceability — see DESIGN.md.
//
// TRUST TIERING IS THE POINT. A number from a spreadsheet and 112 must never
// look the same. Verified = plain. Unverified = amber dot. No exceptions.
//
// State: StreamBuilder over the DAO, deliberately no state-management
// dependency yet (see DECISIONS.md 2026-09-11). Swap to Riverpod later
// without touching the DAO.

import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';
import '../data/contacts_dao.dart';
import '../../../core/database/app_database.dart';
import '../../../core/theme/app_tokens.dart';

class DialerScreen extends StatefulWidget {
  final ContactsDao dao;
  final int tripId;
  final String tripName;
  final int? currentStopId;
  final String? currentStopName;
  final List<String> tripCountryCodes;

  const DialerScreen({
    super.key,
    required this.dao,
    required this.tripId,
    required this.tripName,
    required this.tripCountryCodes,
    this.currentStopId,
    this.currentStopName,
  });

  @override
  State<DialerScreen> createState() => _DialerScreenState();
}

class _DialerScreenState extends State<DialerScreen>
    with SingleTickerProviderStateMixin {
  late final TabController _tabs;
  final _searchController = TextEditingController();
  late ContactFilter _filter;
  bool _stopScoped = false;

  @override
  void initState() {
    super.initState();
    _tabs = TabController(length: 2, vsync: this);
    _filter = ContactFilter(tripId: widget.tripId);
  }

  @override
  void dispose() {
    _tabs.dispose();
    _searchController.dispose();
    super.dispose();
  }

  void _setFilter(ContactFilter next) => setState(() => _filter = next);

  void _toggleStopScope() {
    _stopScoped = !_stopScoped;
    _setFilter(_stopScoped
        ? _filter.copyWith(stopId: widget.currentStopId)
        : _filter.copyWith(clearStop: true));
  }

  // -----------------------------------------------------------------------
  // ACTIONS — all via platform intents. No dialer permission required.
  // -----------------------------------------------------------------------

  Future<void> _dial(Contact c) async {
    final number = c.phoneE164 ?? c.phoneRaw;
    await _launch(Uri(scheme: 'tel', path: number));
    await widget.dao
        .logCall(contactId: c.id, tripId: widget.tripId, action: 'call');
  }

  Future<void> _dialRaw(String number) =>
      _launch(Uri(scheme: 'tel', path: number));

  Future<void> _whatsapp(Contact c) async {
    final digits =
        (c.phoneE164 ?? c.phoneRaw).replaceAll(RegExp(r'[^0-9]'), '');
    await _launch(Uri.parse('https://wa.me/$digits'));
    await widget.dao
        .logCall(contactId: c.id, tripId: widget.tripId, action: 'whatsapp');
  }

  Future<void> _sms(Contact c) async {
    await _launch(Uri(scheme: 'sms', path: c.phoneE164 ?? c.phoneRaw));
    await widget.dao
        .logCall(contactId: c.id, tripId: widget.tripId, action: 'sms');
  }

  Future<void> _launch(Uri uri) async {
    if (!await launchUrl(uri, mode: LaunchMode.externalApplication)) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('No app on this phone can open that.')),
      );
    }
  }

  // -----------------------------------------------------------------------
  // BUILD
  // -----------------------------------------------------------------------

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTokens.surface,
      appBar: AppBar(
        backgroundColor: AppTokens.surface,
        elevation: 0,
        titleSpacing: 16,
        title: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(widget.tripName,
                style: AppTokens.titleStyle, overflow: TextOverflow.ellipsis),
            if (widget.currentStopName != null)
              Text(widget.currentStopName!, style: AppTokens.captionStyle),
          ],
        ),
        actions: [
          if (widget.currentStopId != null)
            IconButton(
              tooltip: _stopScoped ? 'Showing this stop' : 'Showing whole trip',
              onPressed: _toggleStopScope,
              icon: Icon(
                _stopScoped ? Icons.place : Icons.travel_explore,
                color: _stopScoped ? AppTokens.signal : AppTokens.muted,
              ),
            ),
        ],
        bottom: TabBar(
          controller: _tabs,
          labelColor: AppTokens.ink,
          unselectedLabelColor: AppTokens.muted,
          indicatorColor: AppTokens.signal,
          tabs: const [
            Tab(text: 'Contacts'),
            Tab(text: 'Emergency'),
          ],
        ),
      ),
      body: TabBarView(
        controller: _tabs,
        children: [_buildContactsTab(), _buildEmergencyTab()],
      ),
      floatingActionButton: FloatingActionButton.extended(
        backgroundColor: AppTokens.signal,
        onPressed: () => Navigator.pushNamed(context, '/contacts/add',
            arguments: widget.tripId),
        icon: const Icon(Icons.add),
        label: const Text('Add contact'),
      ),
    );
  }

  // ---------------------------------------------------------------- contacts

  Widget _buildContactsTab() {
    return Column(
      children: [
        _buildSearchBar(),
        _buildCategoryChips(),
        _buildReadinessBanner(),
        Expanded(
          child: StreamBuilder<List<Contact>>(
            stream: widget.dao.watchContacts(_filter),
            builder: (context, snap) {
              if (!snap.hasData) {
                return const Center(child: CircularProgressIndicator());
              }
              final items = snap.data!;
              if (items.isEmpty) return _buildEmptyState();
              return ListView.separated(
                padding: const EdgeInsets.only(bottom: 96),
                itemCount: items.length,
                separatorBuilder: (_, __) =>
                    const Divider(height: 1, indent: 72),
                itemBuilder: (context, i) => _ContactRow(
                  contact: items[i],
                  onCall: () => _dial(items[i]),
                  onWhatsapp: () => _whatsapp(items[i]),
                  onSms: () => _sms(items[i]),
                  onLongPress: () => _showContactSheet(items[i]),
                ),
              );
            },
          ),
        ),
      ],
    );
  }

  Widget _buildSearchBar() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 8, 16, 4),
      child: TextField(
        controller: _searchController,
        onChanged: (v) => _setFilter(_filter.copyWith(searchTerm: v)),
        decoration: InputDecoration(
          hintText: 'Search name, number or note',
          prefixIcon: const Icon(Icons.search, size: 20),
          isDense: true,
          filled: true,
          fillColor: AppTokens.surfaceRaised,
          border: OutlineInputBorder(
            borderRadius: BorderRadius.circular(10),
            borderSide: BorderSide.none,
          ),
          suffixIcon: _searchController.text.isEmpty
              ? null
              : IconButton(
                  icon: const Icon(Icons.close, size: 18),
                  onPressed: () {
                    _searchController.clear();
                    _setFilter(_filter.copyWith(searchTerm: ''));
                  },
                ),
        ),
      ),
    );
  }

  Widget _buildCategoryChips() {
    return SizedBox(
      height: 44,
      child: ListView(
        scrollDirection: Axis.horizontal,
        padding: const EdgeInsets.symmetric(horizontal: 12),
        children: [
          _chip('All', _filter.category == null,
              () => _setFilter(_filter.copyWith(clearCategory: true))),
          ...ContactCategory.all
              .where((c) => c != ContactCategory.emergency)
              .map((c) => _chip(
                    ContactCategory.labels[c] ?? c,
                    _filter.category == c,
                    () => _setFilter(_filter.copyWith(category: c)),
                  )),
        ],
      ),
    );
  }

  Widget _chip(String label, bool selected, VoidCallback onTap) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 6),
      child: ChoiceChip(
        label: Text(label),
        selected: selected,
        onSelected: (_) => onTap(),
        showCheckmark: false,
        selectedColor: AppTokens.signalSoft,
        backgroundColor: AppTokens.surfaceRaised,
        side: BorderSide.none,
        labelStyle: TextStyle(
          fontSize: 13,
          color: selected ? AppTokens.signal : AppTokens.muted,
          fontWeight: selected ? FontWeight.w600 : FontWeight.w500,
        ),
      ),
    );
  }

  /// Pre-departure readiness. Only shows when something is unconfirmed —
  /// an always-present banner becomes wallpaper and stops being read.
  Widget _buildReadinessBanner() {
    return StreamBuilder<int>(
      stream: widget.dao.watchUnconfirmedCount(widget.tripId),
      builder: (context, snap) {
        final n = snap.data ?? 0;
        if (n == 0) return const SizedBox.shrink();
        return Container(
          width: double.infinity,
          margin: const EdgeInsets.fromLTRB(16, 4, 16, 8),
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
          decoration: BoxDecoration(
            color: AppTokens.cautionSoft,
            borderRadius: BorderRadius.circular(8),
          ),
          child: Row(
            children: [
              Icon(Icons.error_outline, size: 18, color: AppTokens.caution),
              const SizedBox(width: 10),
              Expanded(
                child: Text(
                  '$n ${n == 1 ? "number" : "numbers"} not confirmed yet. '
                  'Call before you leave signal.',
                  style: TextStyle(fontSize: 13, color: AppTokens.ink),
                ),
              ),
            ],
          ),
        );
      },
    );
  }

  Widget _buildEmptyState() {
    final searching = _filter.searchTerm.isNotEmpty;
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(32),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(searching ? Icons.search_off : Icons.contact_phone_outlined,
                size: 40, color: AppTokens.muted),
            const SizedBox(height: 16),
            Text(
              searching
                  ? 'Nothing matches that.'
                  : 'No contacts saved for this trip yet.',
              style: AppTokens.titleStyle,
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 8),
            Text(
              searching
                  ? 'Try a shorter search, or clear the category filter.'
                  : 'Add your homestay, driver and guide numbers here, '
                      'or import a sheet.',
              style: AppTokens.captionStyle,
              textAlign: TextAlign.center,
            ),
            if (!searching) ...[
              const SizedBox(height: 20),
              OutlinedButton.icon(
                onPressed: () =>
                    Navigator.pushNamed(context, '/contacts/import'),
                icon: const Icon(Icons.upload_file, size: 18),
                label: const Text('Import from sheet'),
              ),
            ],
          ],
        ),
      ),
    );
  }

  // --------------------------------------------------------------- emergency

  Widget _buildEmergencyTab() {
    return ListView(
      padding: const EdgeInsets.only(bottom: 96),
      children: [
        StreamBuilder<List<EmergencyHelpline>>(
          stream: widget.dao.watchEmergencyHelplines(widget.tripCountryCodes),
          builder: (context, snap) {
            final lines = snap.data ?? const [];
            if (lines.isEmpty) return const SizedBox.shrink();
            return Column(
              children: [
                _sectionHeader('Official helplines'),
                ...lines.map((h) => _EmergencyRow(
                      label: h.label,
                      number: h.number,
                      sourceNote: h.sourceNote,
                      onCall: () => _dialRaw(h.number),
                    )),
              ],
            );
          },
        ),
        StreamBuilder<List<Contact>>(
          stream: widget.dao.watchTripEmergencyContacts(widget.tripId),
          builder: (context, snap) {
            final items = snap.data ?? const [];
            if (items.isEmpty) return const SizedBox.shrink();
            return Column(
              children: [
                _sectionHeader('Your local contacts'),
                ...items.map((c) => _ContactRow(
                      contact: c,
                      onCall: () => _dial(c),
                      onWhatsapp: () => _whatsapp(c),
                      onSms: () => _sms(c),
                      onLongPress: () => _showContactSheet(c),
                    )),
              ],
            );
          },
        ),
      ],
    );
  }

  Widget _sectionHeader(String text) => Padding(
        padding: const EdgeInsets.fromLTRB(16, 20, 16, 8),
        child: Text(text, style: AppTokens.sectionStyle),
      );

  // ------------------------------------------------------------ detail sheet

  void _showContactSheet(Contact c) {
    showModalBottomSheet(
      context: context,
      backgroundColor: AppTokens.surface,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(16)),
      ),
      builder: (_) => SafeArea(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const SizedBox(height: 12),
            ListTile(
              title: Text(c.name, style: AppTokens.titleStyle),
              subtitle: Text(c.phoneRaw, style: AppTokens.numberStyle),
            ),
            if (c.note != null && c.note!.isNotEmpty)
              Padding(
                padding: const EdgeInsets.fromLTRB(16, 0, 16, 12),
                child: Align(
                  alignment: Alignment.centerLeft,
                  child: Text(c.note!, style: AppTokens.captionStyle),
                ),
              ),
            const Divider(height: 1),
            ListTile(
              leading: Icon(
                c.callConfirmed ? Icons.verified : Icons.check_circle_outline,
                color: c.callConfirmed ? AppTokens.signal : AppTokens.muted,
              ),
              title: Text(c.callConfirmed
                  ? 'Confirmed — you reached this number'
                  : 'Mark as confirmed'),
              subtitle: c.callConfirmed
                  ? null
                  : const Text('Only after you have actually called it'),
              onTap: () {
                widget.dao.markConfirmed(c.id, confirmed: !c.callConfirmed);
                Navigator.pop(context);
              },
            ),
            ListTile(
              leading: Icon(c.isPinned ? Icons.push_pin : Icons.push_pin_outlined),
              title: Text(c.isPinned ? 'Unpin' : 'Pin to top'),
              onTap: () {
                widget.dao.togglePin(c.id, !c.isPinned);
                Navigator.pop(context);
              },
            ),
            ListTile(
              leading: const Icon(Icons.edit_outlined),
              title: const Text('Edit'),
              onTap: () {
                Navigator.pop(context);
                Navigator.pushNamed(context, '/contacts/edit', arguments: c.id);
              },
            ),
            const SizedBox(height: 8),
          ],
        ),
      ),
    );
  }
}

// =========================================================================
// ROW WIDGETS
// =========================================================================

class _ContactRow extends StatelessWidget {
  final Contact contact;
  final VoidCallback onCall;
  final VoidCallback onWhatsapp;
  final VoidCallback onSms;
  final VoidCallback onLongPress;

  const _ContactRow({
    required this.contact,
    required this.onCall,
    required this.onWhatsapp,
    required this.onSms,
    required this.onLongPress,
  });

  @override
  Widget build(BuildContext context) {
    final tier = ContactTier.parse(contact.tier);
    final trusted = tier.isTrusted;

    return InkWell(
      onTap: onCall,
      onLongPress: onLongPress,
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            _CategoryAvatar(category: contact.category),
            const SizedBox(width: 16),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      if (contact.isPinned)
                        Padding(
                          padding: const EdgeInsets.only(right: 5),
                          child: Icon(Icons.push_pin,
                              size: 13, color: AppTokens.muted),
                        ),
                      Flexible(
                        child: Text(contact.name,
                            style: AppTokens.rowTitleStyle,
                            overflow: TextOverflow.ellipsis),
                      ),
                      // The whole point of the tiering. Unconfirmed numbers
                      // carry a caution mark; confirmed ones carry nothing.
                      if (!trusted)
                        Padding(
                          padding: const EdgeInsets.only(left: 6),
                          child: Tooltip(
                            message: 'Not confirmed yet',
                            child: Container(
                              width: 7,
                              height: 7,
                              decoration: BoxDecoration(
                                color: AppTokens.caution,
                                shape: BoxShape.circle,
                              ),
                            ),
                          ),
                        ),
                    ],
                  ),
                  const SizedBox(height: 2),
                  Text(contact.phoneRaw, style: AppTokens.numberStyle),
                  if (contact.note != null && contact.note!.isNotEmpty)
                    Padding(
                      padding: const EdgeInsets.only(top: 2),
                      child: Text(contact.note!,
                          style: AppTokens.captionStyle,
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis),
                    ),
                ],
              ),
            ),
            if (contact.hasWhatsapp)
              IconButton(
                onPressed: onWhatsapp,
                icon: const Icon(Icons.chat_bubble_outline, size: 20),
                color: AppTokens.muted,
                visualDensity: VisualDensity.compact,
              ),
            IconButton(
              onPressed: onSms,
              icon: const Icon(Icons.sms_outlined, size: 20),
              color: AppTokens.muted,
              visualDensity: VisualDensity.compact,
            ),
            IconButton(
              onPressed: onCall,
              icon: const Icon(Icons.call, size: 22),
              color: AppTokens.signal,
              visualDensity: VisualDensity.compact,
            ),
          ],
        ),
      ),
    );
  }
}

class _EmergencyRow extends StatelessWidget {
  final String label;
  final String number;
  final String sourceNote;
  final VoidCallback onCall;

  const _EmergencyRow({
    required this.label,
    required this.number,
    required this.sourceNote,
    required this.onCall,
  });

  @override
  Widget build(BuildContext context) {
    return ListTile(
      onTap: onCall,
      leading: Container(
        width: 44,
        height: 44,
        alignment: Alignment.center,
        decoration: BoxDecoration(
          color: AppTokens.emergencySoft,
          borderRadius: BorderRadius.circular(10),
        ),
        child: Text(number,
            style: TextStyle(
              fontSize: number.length > 4 ? 11 : 14,
              fontWeight: FontWeight.w700,
              color: AppTokens.emergency,
              fontFeatures: const [FontFeature.tabularFigures()],
            )),
      ),
      title: Text(label, style: AppTokens.rowTitleStyle),
      subtitle: Text(sourceNote,
          style: AppTokens.captionStyle,
          maxLines: 1,
          overflow: TextOverflow.ellipsis),
      trailing: Icon(Icons.call, color: AppTokens.emergency),
    );
  }
}

class _CategoryAvatar extends StatelessWidget {
  final String category;
  const _CategoryAvatar({required this.category});

  static const _icons = <String, IconData>{
    ContactCategory.hospital: Icons.local_hospital_outlined,
    ContactCategory.pharmacy: Icons.medication_outlined,
    ContactCategory.accommodation: Icons.hotel_outlined,
    ContactCategory.restaurant: Icons.restaurant_outlined,
    ContactCategory.transport: Icons.directions_car_outlined,
    ContactCategory.guide: Icons.hiking_outlined,
    ContactCategory.fuel: Icons.local_gas_station_outlined,
    ContactCategory.localContact: Icons.person_outline,
    ContactCategory.embassy: Icons.account_balance_outlined,
    ContactCategory.emergency: Icons.emergency_outlined,
    ContactCategory.other: Icons.more_horiz,
  };

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 40,
      height: 40,
      decoration: BoxDecoration(
        color: AppTokens.surfaceRaised,
        borderRadius: BorderRadius.circular(10),
      ),
      child: Icon(_icons[category] ?? Icons.more_horiz,
          size: 20, color: AppTokens.muted),
    );
  }
}
