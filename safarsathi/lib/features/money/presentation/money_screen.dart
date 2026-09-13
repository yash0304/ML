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

  const MoneyScreen({
    super.key,
    required this.summary,
    this.selfName,
    this.onAdd,
    this.onOpen,
    this.onTravellers,
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
                  const StencilLabel('Ledger'),
                  for (final e in data.ledger)
                    onOpen == null
                        ? _ledgerRow(c, e)
                        : GestureDetector(
                            onTap: () => onOpen!(e.id),
                            behavior: HitTestBehavior.opaque,
                            child: _ledgerRow(c, e),
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
          Text(
            '${data.travellerCount} travelling'.toUpperCase(),
            style: AppTokens.stencilStyle.copyWith(
              fontSize: 10,
              letterSpacing: 1.6,
              color: c.muted,
            ),
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
                  '${e.paidByName} paid · split ${e.splitCount} '
                  '${e.splitCount == 1 ? "way" : "ways"} · ${_date(e.spentAt)}',
                  style: AppTokens.captionStyle.copyWith(
                    fontSize: 11.5,
                    color: c.muted,
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(width: AppTokens.s8),
          Text(
            formatRupees(e.amountMinor),
            style: AppTokens.numberStyle.copyWith(fontSize: 14, color: c.ink),
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
