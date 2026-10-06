// lib/features/money/presentation/currencies_screen.dart
//
// The trip's currencies and the rates you typed — #39.
//
// No live rate, said plainly. The rate is the one the person was given (the
// money changer at Dawki, the bank app before leaving), shown with the day
// it was typed. Expenses keep the rate they were saved with.

import 'package:flutter/material.dart';

import '../../../core/database/app_database.dart';
import '../../../core/theme/app_tokens.dart';
import '../../../core/theme/motion.dart';
import '../../../core/widgets/retro.dart';
import '../data/currency.dart';

class CurrenciesScreen extends StatefulWidget {
  final Stream<List<CurrencyRate>> rates;
  final Future<void> Function(String code, double rate) onSave;
  final Future<void> Function(CurrencyRate rate) onDelete;

  const CurrenciesScreen({
    super.key,
    required this.rates,
    required this.onSave,
    required this.onDelete,
  });

  @override
  State<CurrenciesScreen> createState() => _CurrenciesScreenState();
}

class _CurrenciesScreenState extends State<CurrenciesScreen> {
  final _code = TextEditingController();
  final _rate = TextEditingController();
  String? _error;

  @override
  void dispose() {
    _code.dispose();
    _rate.dispose();
    super.dispose();
  }

  Future<void> _save() async {
    final code = parseCurrencyCode(_code.text);
    final rate = parseRate(_rate.text);
    setState(() {
      _error = code == null
          ? 'A currency is three letters: EUR, BDT, BTN.'
          : code == baseCurrency
          ? 'Rupees are what everything is counted in; they need no rate.'
          : rate == null
          ? 'How many rupees for one $code? A number like 92.40.'
          : null;
    });
    if (_error != null) {
      Haptics.reject();
      return;
    }
    await widget.onSave(code!, rate!);
    Haptics.confirm();
    _code.clear();
    _rate.clear();
    if (mounted) FocusScope.of(context).unfocus();
  }

  Future<void> _delete(CurrencyRate r) async {
    final ok = await showDialog<bool>(
      context: context,
      builder: (d) => AlertDialog(
        title: Text('Remove ${r.code}?'),
        content: const Text(
          'Expenses already saved in it keep their amount and the rate they '
          'were saved with. New ones can no longer be entered in it.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(d).pop(false),
            child: const Text('Keep'),
          ),
          TextButton(
            key: const Key('currency-remove-confirm'),
            onPressed: () => Navigator.of(d).pop(true),
            child: const Text('Remove'),
          ),
        ],
      ),
    );
    if (ok == true) await widget.onDelete(r);
  }

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    final caption = AppTokens.captionStyle.copyWith(color: c.muted);

    return Scaffold(
      backgroundColor: c.paper,
      appBar: AppBar(
        title: const Text('Currencies'),
        backgroundColor: c.paper,
        foregroundColor: c.ink,
        elevation: 0,
      ),
      body: StreamBuilder<List<CurrencyRate>>(
        stream: widget.rates,
        builder: (context, snap) {
          final rates = snap.data ?? const <CurrencyRate>[];
          return ListView(
            padding: const EdgeInsets.only(bottom: AppTokens.s32),
            children: [
              Padding(
                padding: const EdgeInsets.fromLTRB(
                  AppTokens.gutter,
                  AppTokens.s8,
                  AppTokens.gutter,
                  0,
                ),
                child: Text(
                  'Everything is counted in rupees. For money spent in another '
                  'currency, save the rate you were actually given — there is '
                  'no live rate offline. An expense keeps the rate it was '
                  'saved with, so changing a rate here only affects expenses '
                  'saved after.',
                  style: caption,
                ),
              ),
              const StencilLabel('Saved rates'),
              if (rates.isEmpty)
                Padding(
                  padding: const EdgeInsets.symmetric(
                    horizontal: AppTokens.gutter,
                  ),
                  child: Text('Only rupees so far.', style: caption),
                ),
              for (final r in rates)
                Container(
                  margin: const EdgeInsets.symmetric(
                    horizontal: AppTokens.gutter,
                  ),
                  padding: const EdgeInsets.symmetric(vertical: AppTokens.s8),
                  decoration: BoxDecoration(
                    border: Border(
                      bottom: BorderSide(
                        color: c.rule,
                        width: AppTokens.hairline,
                      ),
                    ),
                  ),
                  child: Row(
                    children: [
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              r.code,
                              style: AppTokens.rowTitleStyle.copyWith(
                                color: c.ink,
                              ),
                            ),
                            Text(describeRate(r), style: caption),
                          ],
                        ),
                      ),
                      IconButton(
                        key: Key('currency-remove-${r.code}'),
                        tooltip: 'Remove ${r.code}',
                        onPressed: () => _delete(r),
                        icon: Icon(Icons.close, size: 18, color: c.muted),
                      ),
                    ],
                  ),
                ),
              const StencilLabel('Add or update a rate'),
              Padding(
                padding: const EdgeInsets.symmetric(
                  horizontal: AppTokens.gutter,
                ),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    SizedBox(
                      width: 90,
                      child: TextField(
                        key: const Key('currency-code'),
                        controller: _code,
                        textCapitalization: TextCapitalization.characters,
                        maxLength: 3,
                        decoration: const InputDecoration(
                          labelText: 'Code',
                          hintText: 'EUR',
                          counterText: '',
                        ),
                      ),
                    ),
                    const SizedBox(width: AppTokens.s12),
                    Expanded(
                      child: TextField(
                        key: const Key('currency-rate'),
                        controller: _rate,
                        keyboardType: const TextInputType.numberWithOptions(
                          decimal: true,
                        ),
                        decoration: const InputDecoration(
                          labelText: 'Rupees for one unit',
                          hintText: '92.40',
                          prefixText: '₹ ',
                        ),
                      ),
                    ),
                  ],
                ),
              ),
              if (_error != null)
                Padding(
                  padding: const EdgeInsets.fromLTRB(
                    AppTokens.gutter,
                    AppTokens.s8,
                    AppTokens.gutter,
                    0,
                  ),
                  child: Text(
                    _error!,
                    style: AppTokens.captionStyle.copyWith(color: c.caution),
                  ),
                ),
              Padding(
                padding: const EdgeInsets.fromLTRB(
                  AppTokens.gutter,
                  AppTokens.s16,
                  AppTokens.gutter,
                  0,
                ),
                child: PressScale(
                  key: const Key('currency-save'),
                  onTap: _save,
                  child: Container(
                    height: 44,
                    alignment: Alignment.center,
                    decoration: BoxDecoration(
                      color: c.signal,
                      border: Border.all(color: c.ink),
                      borderRadius: BorderRadius.circular(
                        AppTokens.radiusSoft,
                      ),
                    ),
                    child: Text(
                      'Save rate',
                      style: AppTokens.stencilStyle.copyWith(
                        fontSize: 11,
                        color: c.paper,
                      ),
                    ),
                  ),
                ),
              ),
            ],
          );
        },
      ),
    );
  }
}
