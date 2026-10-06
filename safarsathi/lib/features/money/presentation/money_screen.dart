// lib/features/money/presentation/money_screen.dart
//
// The shared ledger. SCREENS.md §8.
//
// SETTLE UP COMES BEFORE THE LEDGER, because the question people actually
// have is who owes whom, not what was spent. The ledger is the audit trail,
// not the headline.
//
// Balances render positive in signal and negative in muted, NEVER in red.
// Owing money is not an emergency.

import 'package:flutter/material.dart';

import '../../../core/theme/motion.dart';
import '../../../core/theme/app_tokens.dart';
import '../../../core/widgets/retro.dart';
import '../data/currency.dart';
import '../data/expense_tags.dart';
import '../data/money_summary.dart';
import '../data/settlement.dart';

class MoneyScreen extends StatelessWidget {
  final Stream<MoneySummary> summary;
  final String? selfName;

  /// Adding an expense (#31). Optional so the golden harness and the widget
  /// tests can render the screen without wiring an editor.
  final VoidCallback? onAdd;

  /// Opening one to edit it.
  final void Function(int expenseId)? onOpen;

  /// Managing who is on the trip. Also the way out of the empty state, since
  /// an expense needs at least one traveller to be paid by.
  final VoidCallback? onTravellers;

  /// Writes the ledger to a spreadsheet file and offers to share or save it.
  /// Null hides the button.
  final Future<void> Function()? onExport;

  /// Tags the untagged expenses from what they say; returns how many. Null
  /// hides the offer.
  final Future<int> Function()? onTagUntagged;

  /// The trip's currencies and rates. Null hides the link.
  final VoidCallback? onCurrencies;

  const MoneyScreen({
    super.key,
    required this.summary,
    this.selfName,
    this.onAdd,
    this.onOpen,
    this.onTravellers,
    this.onExport,
    this.onTagUntagged,
    this.onCurrencies,
  });

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);

    return Scaffold(
      backgroundColor: c.paper,
      floatingActionButton: onAdd == null
          ? null
          : FloatingActionButton.extended(
              onPressed: onAdd,
              backgroundColor: c.signal,
              foregroundColor: c.paper,
              icon: const Icon(Icons.add, size: 18),
              label: Text(
                'Add expense',
                style: AppTokens.stencilStyle.copyWith(
                  fontSize: 10.5,
                  color: c.paper,
                ),
              ),
            ),
      body: GrainOverlay(
        child: SafeArea(
          bottom: false,
          child: StreamBuilder<MoneySummary>(
            stream: summary,
            builder: (context, snap) {
              final data = snap.data;
              if (data == null) return const SizedBox();
              if (data.ledger.isEmpty) return _empty(c, data);

              return ListView(
                padding: const EdgeInsets.only(bottom: AppTokens.s24),
                children: [
                  _header(c, data),
                  const StencilLabel('Balances'),
                  _balances(c, data),
                  const StencilLabel('Settle up'),
                  ..._settlements(c, data),
                  _settleNote(c, data),
                  // WHERE IT WENT, then the lines themselves. One widget,
                  // because tapping a tag filters the ledger under it.
                  _TagsAndLedger(
                    data: data,
                    onOpen: onOpen,
                    onTagUntagged: onTagUntagged,
                    row: (e) => _ledgerRow(c, e),
                  ),
                  _currencyNote(c),
                ],
              );
            },
          ),
        ),
      ),
    );
  }

  Widget _empty(AppColors c, MoneySummary data) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(AppTokens.s32),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(
              'Nothing spent yet.',
              style: AppTokens.titleStyle.copyWith(color: c.ink),
            ),
            const SizedBox(height: AppTokens.s8),
            Text(
              data.travellerCount == 0
                  ? 'Add whoever is sharing costs first. Just names — there '
                        'are no accounts and nothing is sent anywhere.'
                  : 'Expenses are local rows and the settling is local maths, '
                        'so none of this needs a signal.',
              textAlign: TextAlign.center,
              style: AppTokens.captionStyle.copyWith(color: c.muted),
            ),
            if (data.travellerCount == 0 && onTravellers != null) ...[
              const SizedBox(height: AppTokens.s24),
              PressScale(
                onTap: onTravellers,
                child: Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: AppTokens.s24,
                    vertical: AppTokens.s12,
                  ),
                  decoration: BoxDecoration(
                    color: c.signal,
                    border: Border.all(color: c.ink),
                    borderRadius: BorderRadius.circular(AppTokens.radiusSoft),
                  ),
                  child: Text(
                    'Add travellers',
                    style: AppTokens.stencilStyle.copyWith(
                      fontSize: 11.5,
                      color: c.paper,
                    ),
                  ),
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _header(AppColors c, MoneySummary data) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(
        AppTokens.gutter,
        AppTokens.s12,
        AppTokens.gutter,
        0,
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Expanded(
                child: Text(
                  '${data.travellerCount} travelling'.toUpperCase(),
                  style: AppTokens.stencilStyle.copyWith(
                    fontSize: 10,
                    letterSpacing: 1.6,
                    color: c.muted,
                  ),
                ),
              ),
              if (onCurrencies != null)
                Padding(
                  padding: const EdgeInsets.only(right: AppTokens.s16),
                  child: PressScale(
                    key: const Key('money-currencies'),
                    onTap: onCurrencies,
                    child: Text(
                      'CURRENCIES',
                      style: AppTokens.stencilStyle.copyWith(
                        fontSize: 10,
                        color: c.signal,
                      ),
                    ),
                  ),
                ),
              if (onExport != null)
                PressScale(
                  key: const Key('money-export'),
                  onTap: onExport,
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(Icons.download_outlined, size: 15, color: c.signal),
                      const SizedBox(width: 4),
                      Text(
                        'EXPORT',
                        style: AppTokens.stencilStyle.copyWith(
                          fontSize: 10,
                          color: c.signal,
                        ),
                      ),
                    ],
                  ),
                ),
            ],
          ),
          const SizedBox(height: AppTokens.s4),
          Text(
            formatRupees(data.totalMinor),
            style: AppTokens.numberStyle.copyWith(fontSize: 26, color: c.ink),
          ),
          Text(
            'Spent so far · ${formatRupees(data.perHeadMinor)} each',
            style: AppTokens.captionStyle.copyWith(color: c.muted),
          ),
        ],
      ),
    );
  }

  Widget _balances(AppColors c, MoneySummary data) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: AppTokens.gutter),
      child: TicketCard(
        child: Column(
          children: [
            for (var i = 0; i < data.balances.length; i++)
              Container(
                padding: const EdgeInsets.symmetric(vertical: AppTokens.s8),
                decoration: BoxDecoration(
                  border: i == data.balances.length - 1
                      ? null
                      : Border(
                          bottom: BorderSide(
                            color: c.rule,
                            width: AppTokens.hairline,
                          ),
                        ),
                ),
                child: Row(
                  children: [
                    Expanded(
                      child: Text(
                        data.balances[i].name,
                        style: AppTokens.rowTitleStyle.copyWith(color: c.ink),
                      ),
                    ),
                    Text(
                      formatRupees(data.balances[i].netMinor, signed: true),
                      style: AppTokens.numberStyle.copyWith(
                        fontSize: 14,
                        // Never red. Owing money is not an emergency.
                        color: data.balances[i].netMinor > 0
                            ? c.signal
                            : c.muted,
                      ),
                    ),
                  ],
                ),
              ),
          ],
        ),
      ),
    );
  }

  List<Widget> _settlements(AppColors c, MoneySummary data) {
    if (data.settlements.isEmpty) {
      return [
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: AppTokens.gutter),
          child: Text(
            'Everyone is square.',
            style: AppTokens.rowTitleStyle.copyWith(color: c.ink),
          ),
        ),
      ];
    }
    return [
      for (final s in data.settlements)
        Container(
          margin: const EdgeInsets.symmetric(horizontal: AppTokens.gutter),
          padding: const EdgeInsets.symmetric(vertical: AppTokens.s12),
          decoration: BoxDecoration(
            border: Border(
              bottom: BorderSide(color: c.rule, width: AppTokens.hairline),
            ),
          ),
          child: Row(
            children: [
              Text(
                s.fromName,
                style: AppTokens.rowTitleStyle.copyWith(color: c.ink),
              ),
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: AppTokens.s8),
                child: Text(
                  '→',
                  style: AppTokens.rowTitleStyle.copyWith(color: c.muted),
                ),
              ),
              Expanded(
                child: Text(
                  s.toName,
                  style: AppTokens.rowTitleStyle.copyWith(color: c.ink),
                ),
              ),
              Text(
                formatRupees(s.amountMinor),
                style: AppTokens.numberStyle.copyWith(
                  fontSize: 14,
                  color: c.ink,
                ),
              ),
            ],
          ),
        ),
    ];
  }

  Widget _settleNote(AppColors c, MoneySummary data) {
    final saved = data.paymentsSaved;
    return Padding(
      padding: const EdgeInsets.fromLTRB(
        AppTokens.gutter,
        AppTokens.s8,
        AppTokens.gutter,
        0,
      ),
      child: Text(
        saved > 0
            ? '${data.settlements.length} '
                  '${data.settlements.length == 1 ? "payment" : "payments"} '
                  'instead of ${data.settlements.length + saved}. Worked out '
                  'on the phone, so it needs no signal.'
            : 'Worked out on the phone, so it needs no signal.',
        style: AppTokens.captionStyle.copyWith(color: c.muted),
      ),
    );
  }

  Widget _ledgerRow(AppColors c, LedgerEntry e) {
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: AppTokens.gutter),
      padding: const EdgeInsets.symmetric(vertical: AppTokens.s8),
      decoration: BoxDecoration(
        border: Border(
          bottom: BorderSide(color: c.rule, width: AppTokens.hairline),
        ),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  e.description,
                  style: AppTokens.rowTitleStyle.copyWith(color: c.ink),
                ),
                Text(
                  '${tagLabel(e.tag)} · ${e.paidByName} paid · split '
                  '${e.splitCount} ${e.splitCount == 1 ? "way" : "ways"} · '
                  '${_date(e.spentAt)}',
                  style: AppTokens.captionStyle.copyWith(
                    fontSize: 11.5,
                    color: c.muted,
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(width: AppTokens.s8),
          Column(
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [
              Text(
                formatRupees(e.amountMinor),
                style: AppTokens.numberStyle.copyWith(
                  fontSize: 14,
                  color: c.ink,
                ),
              ),
              // Spent abroad: what it was, under what it counts as.
              if (e.isForeign && e.originalMinor != null)
                Text(
                  formatMoney(e.originalMinor!, e.currency),
                  style: AppTokens.captionStyle.copyWith(
                    fontSize: 11.5,
                    color: c.muted,
                  ),
                ),
            ],
          ),
        ],
      ),
    );
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

  Widget _currencyNote(AppColors c) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(
        AppTokens.gutter,
        AppTokens.s24,
        AppTokens.gutter,
        0,
      ),
      child: Text(
        'The app records a settlement, it never makes one. A trip that crosses '
        'a border uses an exchange rate you save at setup, shown with the date '
        'you saved it — there is no live rate offline.',
        style: AppTokens.captionStyle.copyWith(color: c.muted),
      ),
    );
  }
}

/// Where the money went, by tag, and the ledger under it. Tapping a tag shows
/// only its lines; tapping it again shows them all.
class _TagsAndLedger extends StatefulWidget {
  final MoneySummary data;
  final void Function(int expenseId)? onOpen;
  final Widget Function(LedgerEntry e) row;
  final Future<int> Function()? onTagUntagged;

  const _TagsAndLedger({
    required this.data,
    required this.row,
    this.onOpen,
    this.onTagUntagged,
  });

  @override
  State<_TagsAndLedger> createState() => _TagsAndLedgerState();
}

class _TagsAndLedgerState extends State<_TagsAndLedger> {
  /// The tag filtered on. A record so "untagged" (null) is a filter too.
  ({String? tag})? _only;

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    final data = widget.data;
    final tags = data.byTag;
    final only = _only;
    final shown = only == null
        ? data.ledger
        : [
            for (final e in data.ledger)
              if (e.tag == only.tag) e,
          ];

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        const StencilLabel('Where it went'),
        for (final t in tags)
          InkWell(
            key: Key('tag-total-${t.tag ?? 'none'}'),
            onTap: () => setState(
              () => _only = only != null && only.tag == t.tag
                  ? null
                  : (tag: t.tag),
            ),
            child: Container(
              margin: const EdgeInsets.symmetric(horizontal: AppTokens.gutter),
              padding: const EdgeInsets.symmetric(vertical: AppTokens.s8),
              decoration: BoxDecoration(
                border: Border(
                  bottom: BorderSide(color: c.rule, width: AppTokens.hairline),
                ),
              ),
              child: Row(
                children: [
                  Expanded(
                    child: Text(
                      '${tagLabel(t.tag)} · ${t.count}',
                      style: AppTokens.rowTitleStyle.copyWith(
                        color: only != null && only.tag == t.tag
                            ? c.signal
                            : c.ink,
                      ),
                    ),
                  ),
                  Text(
                    formatRupees(t.totalMinor),
                    style: AppTokens.numberStyle.copyWith(
                      fontSize: 14,
                      color: c.ink,
                    ),
                  ),
                ],
              ),
            ),
          ),
        if (widget.onTagUntagged != null && tags.any((t) => t.tag == null))
          InkWell(
            key: const Key('tag-untagged'),
            onTap: () async {
              final messenger = ScaffoldMessenger.of(context);
              final untagged = tags.firstWhere((t) => t.tag == null).count;
              final done = await widget.onTagUntagged!();
              messenger.showSnackBar(
                SnackBar(
                  content: Text(
                    done == untagged
                        ? 'Tagged all $done from what they say.'
                        : 'Tagged $done of $untagged. The rest need a tag '
                              'picked by hand — open each one.',
                  ),
                ),
              );
            },
            child: Padding(
              padding: const EdgeInsets.fromLTRB(
                AppTokens.gutter,
                AppTokens.s8,
                AppTokens.gutter,
                0,
              ),
              child: Text(
                'Tag the untagged ones from what they say',
                style: AppTokens.captionStyle.copyWith(color: c.signal),
              ),
            ),
          ),
        Row(
          children: [
            Expanded(
              child: StencilLabel(
                only == null ? 'Ledger' : 'Ledger · ${tagLabel(only.tag)}',
              ),
            ),
            if (only != null)
              Padding(
                padding: const EdgeInsets.only(
                  right: AppTokens.gutter,
                  top: AppTokens.s16,
                ),
                child: PressScale(
                  key: const Key('ledger-show-all'),
                  onTap: () => setState(() => _only = null),
                  child: Text(
                    'SHOW ALL',
                    style: AppTokens.stencilStyle.copyWith(
                      fontSize: 10,
                      color: c.signal,
                    ),
                  ),
                ),
              ),
          ],
        ),
        for (final e in shown)
          widget.onOpen == null
              ? widget.row(e)
              : GestureDetector(
                  onTap: () => widget.onOpen!(e.id),
                  behavior: HitTestBehavior.opaque,
                  child: widget.row(e),
                ),
      ],
    );
  }
}
