// lib/features/contacts/presentation/entry_form_screen.dart
//
// A ruled diary page as a form: label in the margin voice, value on the line,
// a hairline under each. See SCREENS.md §5.
//
// Takes callbacks rather than a DAO, same as the diary — it can be driven and
// tested without a database.

import 'package:flutter/material.dart';
import 'package:phone_numbers_parser/phone_numbers_parser.dart';

import '../../../core/database/app_database.dart';
import '../../../core/theme/app_tokens.dart';
import '../../../core/theme/motion.dart';
import '../../../core/widgets/retro.dart';
import '../data/phone_contact_picker.dart';
import '../data/contacts_dao.dart';
import '../data/entry_draft.dart';
import '../data/phone_normaliser.dart';

/// A stop the entry can be attached to.
class StopOption {
  final int id;
  final String name;
  const StopOption(this.id, this.name);
}

class EntryFormScreen extends StatefulWidget {
  /// Null for a new entry.
  final Contact? existing;

  final List<StopOption> stops;

  /// Used to read a local number. Comes from the current stop's country, so
  /// a European trip normalises correctly mid-itinerary.
  final IsoCode country;

  /// Returns an existing entry with the same normalised number, if any.
  final Future<Contact?> Function(String e164)? findDuplicate;

  final Future<void> Function(EntryDraft) onSave;

  /// Fills name and number from the phone's own contacts app. Null hides the
  /// option — editing an existing entry, or a build with no picker.
  final Future<PickedContact?> Function()? pickFromPhone;

  /// Opens the picker as soon as the form appears, for the "From my phone's
  /// contacts" entry on More, where picking is the whole point of coming in.
  final bool pickOnOpen;

  /// Where a new entry starts, when the screen that opened the form already
  /// knows — "No stay for Mawlynnong yet · add one" should not make you pick
  /// Mawlynnong and Stay again. Ignored when editing.
  final int? initialStopId;
  final String? initialCategory;

  const EntryFormScreen({
    super.key,
    this.existing,
    this.stops = const [],
    this.country = IsoCode.IN,
    this.findDuplicate,
    required this.onSave,
    this.pickFromPhone,
    this.pickOnOpen = false,
    this.initialStopId,
    this.initialCategory,
  });

  @override
  State<EntryFormScreen> createState() => _EntryFormScreenState();
}

class _EntryFormScreenState extends State<EntryFormScreen> {
  late final TextEditingController _name;
  late final TextEditingController _phone;
  late final TextEditingController _note;
  late String _category;
  late int? _stopId;
  late bool _hasWhatsapp;

  NormalisedPhone _normalised = const NormalisedPhone(raw: '');
  Contact? _duplicate;
  String? _nameError;
  String? _phoneError;
  bool _saving = false;

  bool get _isEdit => widget.existing != null;

  @override
  void initState() {
    super.initState();
    final e = widget.existing;
    _name = TextEditingController(text: e?.name ?? '');
    _phone = TextEditingController(text: e?.phoneRaw ?? '');
    _note = TextEditingController(text: e?.note ?? '');
    _category = e?.category ?? widget.initialCategory ?? ContactCategory.other;
    _stopId = e == null ? widget.initialStopId : e.stopId;
    _hasWhatsapp = e?.hasWhatsapp ?? false;
    if (e != null) _renormalise(e.phoneRaw);
    if (widget.pickOnOpen && _canPick) {
      WidgetsBinding.instance.addPostFrameCallback((_) => _pick());
    }
  }

  bool get _canPick => widget.pickFromPhone != null && !_isEdit;

  Future<void> _pick() async {
    final PickedContact? picked;
    try {
      picked = await widget.pickFromPhone!();
    } on ContactPickException catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context)
          .showSnackBar(SnackBar(content: Text(e.message)));
      return;
    }
    // Backed out of the picker: leave whatever was already typed alone.
    if (picked == null || !mounted) return;

    setState(() {
      // The phone's name is kept as it is. Renaming "Driver Bah Kyn" to
      // something tidier is the user's call, made in the field below.
      if (picked!.name.isNotEmpty) _name.text = picked.name;
      _phone.text = picked.number;
      _nameError = null;
      _phoneError = null;
    });
    await _renormalise(picked.number);
  }

  @override
  void dispose() {
    _name.dispose();
    _phone.dispose();
    _note.dispose();
    super.dispose();
  }

  /// True when an already-confirmed entry has had its digits changed.
  bool get _confirmationAtRisk {
    final e = widget.existing;
    if (e == null || !e.callConfirmed) return false;
    return _phone.text.trim() != e.phoneRaw.trim();
  }

  Future<void> _renormalise(String value) async {
    final result = PhoneNormaliser.normalise(value, country: widget.country);
    Contact? duplicate;
    final find = widget.findDuplicate;
    if (find != null && result.e164 != null) {
      final hit = await find(result.e164!);
      // Editing an entry is not a duplicate of itself.
      if (hit != null && hit.id != widget.existing?.id) duplicate = hit;
    }
    if (!mounted) return;
    setState(() {
      _normalised = result;
      _duplicate = duplicate;
      if (value.trim().isNotEmpty) _phoneError = null;
    });
  }

  Future<void> _save() async {
    final name = _name.text.trim();
    final phone = _phone.text.trim();
    setState(() {
      _nameError = name.isEmpty ? 'An entry needs a name.' : null;
      _phoneError = phone.isEmpty ? 'An entry needs a number.' : null;
    });
    if (name.isEmpty || phone.isEmpty) return;

    setState(() => _saving = true);
    Haptics.confirm();
    await widget.onSave(
      EntryDraft(
        id: widget.existing?.id,
        name: name,
        phoneRaw: phone,
        phoneE164: _normalised.e164,
        category: _category,
        stopId: _stopId,
        note: _note.text,
        hasWhatsapp: _hasWhatsapp,
        resetConfirmation: _confirmationAtRisk,
      ),
    );
    if (mounted) Navigator.of(context).maybePop();
  }

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);

    return Scaffold(
      backgroundColor: c.paper,
      body: GrainOverlay(
        child: SafeArea(
          child: Column(
            children: [
              _header(c),
              Expanded(
                child: ListView(
                  padding: const EdgeInsets.only(bottom: AppTokens.s24),
                  children: [
                    const StencilLabel('The number'),
                    if (_canPick)
                      _PickRow(
                        key: const Key('pick-from-phone'),
                        onTap: _pick,
                      ),
                    _TextField(
                      key: const Key('field-name'),
                      label: 'Name',
                      controller: _name,
                      hint: 'Who or what this is',
                      error: _nameError,
                      onChanged: (_) => setState(() => _nameError = null),
                    ),
                    _TextField(
                      key: const Key('field-number'),
                      label: 'Number',
                      controller: _phone,
                      hint: 'As you would dial it',
                      keyboardType: TextInputType.phone,
                      error: _phoneError,
                      big: true,
                      onChanged: _renormalise,
                    ),
                    if (_normalised.warning != null)
                      _Notice(text: _normalised.warning!, tone: c.caution),
                    if (_duplicate != null)
                      _Notice(
                        text:
                            'Already in your diary as "${_duplicate!.name}". '
                            'Saving anyway is fine.',
                        tone: c.caution,
                      ),
                    if (_confirmationAtRisk)
                      _Notice(
                        text:
                            'You confirmed the old number. Changing the '
                            'digits clears that, because the confirmation '
                            'was for the number you actually called.',
                        tone: c.caution,
                      ),
                    _categoryField(c),
                    if (widget.stops.isNotEmpty) _stopField(c),
                    _TextField(
                      key: const Key('field-note'),
                      label: 'Note',
                      controller: _note,
                      hint: 'Optional — what this number is for',
                    ),
                    _whatsappField(c),
                    const StencilLabel('How it will be saved'),
                    _tierExplainer(c),
                    Padding(
                      padding: const EdgeInsets.fromLTRB(
                        AppTokens.gutter,
                        AppTokens.s16,
                        AppTokens.gutter,
                        0,
                      ),
                      child: _SaveButton(
                        label: _isEdit ? 'Save changes' : 'Save to diary',
                        busy: _saving,
                        onTap: _save,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _header(AppColors c) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(
        AppTokens.s8,
        AppTokens.s12,
        AppTokens.gutter,
        0,
      ),
      child: Row(
        children: [
          IconButton(
            onPressed: () => Navigator.of(context).maybePop(),
            icon: Icon(Icons.arrow_back, color: c.muted),
            tooltip: 'Back',
          ),
          Expanded(
            child: Text(
              _isEdit ? 'Edit entry' : 'New entry',
              style: AppTokens.titleStyle.copyWith(color: c.ink),
            ),
          ),
        ],
      ),
    );
  }

  Widget _categoryField(AppColors c) {
    return _Field(
      label: 'Category',
      child: Wrap(
        spacing: 6,
        runSpacing: 6,
        children: [
          for (final key in ContactCategory.pickerOrder)
            _Chip(
              label: ContactCategory.labels[key] ?? key,
              active: _category == key,
              onTap: () => setState(() => _category = key),
            ),
        ],
      ),
    );
  }

  Widget _stopField(AppColors c) {
    return _Field(
      label: 'Attach to',
      child: Wrap(
        spacing: 6,
        runSpacing: 6,
        children: [
          _Chip(
            label: 'Whole trip',
            active: _stopId == null,
            onTap: () => setState(() => _stopId = null),
          ),
          for (final stop in widget.stops)
            _Chip(
              label: stop.name,
              active: _stopId == stop.id,
              onTap: () => setState(() => _stopId = stop.id),
            ),
        ],
      ),
    );
  }

  Widget _whatsappField(AppColors c) {
    return _Field(
      label: 'Also reachable on',
      child: Row(
        children: [
          _Chip(
            label: 'WhatsApp',
            active: _hasWhatsapp,
            onTap: () => setState(() => _hasWhatsapp = !_hasWhatsapp),
          ),
        ],
      ),
    );
  }

  Widget _tierExplainer(AppColors c) {
    final confirmedAndUnchanged =
        _isEdit && widget.existing!.callConfirmed && !_confirmationAtRisk;

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: AppTokens.gutter),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            width: 7,
            height: 7,
            margin: const EdgeInsets.only(top: 6, right: AppTokens.s8),
            decoration: BoxDecoration(
              color: confirmedAndUnchanged ? c.signal : c.cautionMark,
              shape: BoxShape.circle,
            ),
          ),
          Expanded(
            child: Text(
              confirmedAndUnchanged
                  ? 'Stays confirmed. You called this number and it worked.'
                  : 'As unconfirmed, with an amber dot. It becomes confirmed '
                        'only after you call it and say so. Typing it here '
                        'does not make it work.',
              style: AppTokens.captionStyle.copyWith(color: c.muted),
            ),
          ),
        ],
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Ruled form parts
// ---------------------------------------------------------------------------

class _Field extends StatelessWidget {
  final String label;
  final Widget child;
  const _Field({required this.label, required this.child});

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: AppTokens.gutter),
      padding: const EdgeInsets.only(top: AppTokens.s12, bottom: AppTokens.s8),
      decoration: BoxDecoration(
        border: Border(
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
          child,
        ],
      ),
    );
  }
}

class _TextField extends StatelessWidget {
  final String label;
  final TextEditingController controller;
  final String? hint;
  final String? error;
  final bool big;
  final TextInputType? keyboardType;
  final ValueChanged<String>? onChanged;

  const _TextField({
    super.key,
    required this.label,
    required this.controller,
    this.hint,
    this.error,
    this.big = false,
    this.keyboardType,
    this.onChanged,
  });

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    return _Field(
      label: label,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          TextField(
            controller: controller,
            keyboardType: keyboardType,
            onChanged: onChanged,
            style: big
                ? AppTokens.numberStyle.copyWith(fontSize: 17, color: c.ink)
                : AppTokens.rowTitleStyle.copyWith(color: c.ink),
            decoration: InputDecoration(
              isDense: true,
              filled: false,
              border: InputBorder.none,
              enabledBorder: InputBorder.none,
              focusedBorder: InputBorder.none,
              contentPadding: EdgeInsets.zero,
              hintText: hint,
              hintStyle: AppTokens.captionStyle.copyWith(
                fontSize: 14,
                color: c.muted,
              ),
            ),
          ),
          if (error != null)
            Padding(
              padding: const EdgeInsets.only(top: AppTokens.s4),
              child: Text(
                error!,
                style: AppTokens.captionStyle.copyWith(color: c.caution),
              ),
            ),
        ],
      ),
    );
  }
}

/// Tells you something and gets out of the way. Never blocks a save.
class _Notice extends StatelessWidget {
  final String text;
  final Color tone;
  const _Notice({required this.text, required this.tone});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(
        AppTokens.gutter,
        AppTokens.s8,
        AppTokens.gutter,
        0,
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(Icons.error_outline, size: 15, color: tone),
          const SizedBox(width: AppTokens.s8),
          Expanded(
            child: Text(
              text,
              style: AppTokens.captionStyle.copyWith(color: tone),
            ),
          ),
        ],
      ),
    );
  }
}

class _Chip extends StatelessWidget {
  final String label;
  final bool active;
  final VoidCallback onTap;
  const _Chip({required this.label, required this.active, required this.onTap});

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    return PressScale(
      onTap: onTap,
      feedback: Haptics.select,
      child: Container(
        padding: const EdgeInsets.symmetric(
          horizontal: AppTokens.s8,
          vertical: 5,
        ),
        decoration: BoxDecoration(
          color: active ? c.signalSoft : c.stone,
          border: Border.all(color: active ? c.signal : c.rule),
          borderRadius: BorderRadius.circular(AppTokens.radiusSoft),
        ),
        child: Text(
          label.toUpperCase(),
          style: AppTokens.stencilStyle.copyWith(
            fontSize: 9.5,
            color: active ? c.signal : c.muted,
          ),
        ),
      ),
    );
  }
}

class _SaveButton extends StatelessWidget {
  final String label;
  final bool busy;
  final VoidCallback onTap;
  const _SaveButton({
    required this.label,
    required this.busy,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    return PressScale(
      onTap: busy ? null : onTap,
      feedback: Haptics.confirm,
      child: Container(
        width: double.infinity,
        padding: const EdgeInsets.symmetric(vertical: AppTokens.s12),
        alignment: Alignment.center,
        decoration: BoxDecoration(
          color: busy ? c.muted : c.signal,
          border: Border.all(color: c.ink),
          borderRadius: BorderRadius.circular(AppTokens.radiusSoft),
        ),
        child: Text(
          label,
          style: AppTokens.stencilStyle.copyWith(
            fontSize: 11.5,
            color: c.paper,
          ),
        ),
      ),
    );
  }
}

/// "From my phone's contacts" — fills the two fields below it.
class _PickRow extends StatelessWidget {
  final VoidCallback onTap;
  const _PickRow({super.key, required this.onTap});

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    return InkWell(
      onTap: onTap,
      child: Padding(
        padding: const EdgeInsets.fromLTRB(
          AppTokens.gutter,
          AppTokens.s4,
          AppTokens.gutter,
          AppTokens.s12,
        ),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Padding(
              padding: const EdgeInsets.only(top: 2),
              child: Icon(Icons.contacts_outlined, size: 18, color: c.signal),
            ),
            const SizedBox(width: AppTokens.s8),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'From my phone\'s contacts',
                    style: AppTokens.rowTitleStyle.copyWith(color: c.signal),
                  ),
                  Text(
                    // Said on the button, because it is the first thing a
                    // careful person wonders before tapping it. Under the
                    // label, not beside it: beside it ran 21 px off a phone.
                    'Picks one number. The app reads nothing else.',
                    style: AppTokens.captionStyle.copyWith(color: c.muted),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
