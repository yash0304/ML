// lib/features/phrasebook/presentation/phrasebook_screen.dart
//
// A few phrases, offline — #40. Languages for the trip's countries first;
// tap a phrase to show it full screen, large, to the person you are asking.

import 'package:flutter/material.dart';

import '../../../core/theme/app_tokens.dart';
import '../../../core/theme/motion.dart';
import '../../../core/widgets/retro.dart';
import '../data/phrases.dart';

class PhrasebookScreen extends StatefulWidget {
  /// Country codes of this trip's stops, so their languages come first.
  final Set<String> countries;

  const PhrasebookScreen({super.key, this.countries = const {}});

  @override
  State<PhrasebookScreen> createState() => _PhrasebookScreenState();
}

class _PhrasebookScreenState extends State<PhrasebookScreen> {
  late final List<Language> _languages = languagesFor(widget.countries);
  late Language _language = _languages.first;

  void _show(Phrase p) {
    Haptics.select();
    Navigator.of(context).push(
      MaterialPageRoute<void>(
        fullscreenDialog: true,
        builder: (_) => ShowPhraseScreen(phrase: p, language: _language),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    final caption = AppTokens.captionStyle.copyWith(color: c.muted);
    return Scaffold(
      backgroundColor: c.paper,
      appBar: AppBar(
        title: const Text('Phrasebook'),
        backgroundColor: c.paper,
        foregroundColor: c.ink,
        elevation: 0,
      ),
      body: ListView(
        padding: const EdgeInsets.only(bottom: AppTokens.s32),
        children: [
          SizedBox(
            height: 56,
            child: ListView(
              scrollDirection: Axis.horizontal,
              padding: const EdgeInsets.symmetric(
                horizontal: AppTokens.gutter,
                vertical: AppTokens.s8,
              ),
              children: [
                for (final l in _languages)
                  Padding(
                    padding: const EdgeInsets.only(right: AppTokens.s8),
                    child: ChoiceChip(
                      key: Key('lang-${l.code}'),
                      label: Text(l.name),
                      selected: l == _language,
                      onSelected: (_) => setState(() => _language = l),
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
            child: Text(
              '${_language.native} · ${_language.phraseCount} phrases. '
              'Written for SafarSathi, not checked by a native speaker. Tap '
              'a phrase to show it large.',
              key: const Key('phrasebook-note'),
              style: caption,
            ),
          ),
          if (_language.note != null)
            Padding(
              padding: const EdgeInsets.fromLTRB(
                AppTokens.gutter,
                AppTokens.s8,
                AppTokens.gutter,
                0,
              ),
              child: Text(_language.note!, style: caption),
            ),
          for (final g in _language.groups) ...[
            StencilLabel(g.title),
            for (final p in g.phrases) _PhraseRow(phrase: p, onTap: () => _show(p)),
          ],
          Padding(
            padding: const EdgeInsets.fromLTRB(
              AppTokens.gutter,
              AppTokens.s24,
              AppTokens.gutter,
              0,
            ),
            child: Text(
              'Emergency numbers are on the SOS tab, not here.',
              style: caption,
            ),
          ),
        ],
      ),
    );
  }
}

class _PhraseRow extends StatelessWidget {
  final Phrase phrase;
  final VoidCallback onTap;

  const _PhraseRow({required this.phrase, required this.onTap});

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    return PressScale(
      key: Key('phrase-${phrase.english}'),
      onTap: onTap,
      child: Container(
        width: double.infinity,
        padding: const EdgeInsets.symmetric(
          horizontal: AppTokens.gutter,
          vertical: AppTokens.s12,
        ),
        decoration: BoxDecoration(
          border: Border(
            bottom: BorderSide(color: c.rule, width: AppTokens.hairline),
          ),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              phrase.english,
              style: AppTokens.captionStyle.copyWith(color: c.muted),
            ),
            const SizedBox(height: 2),
            Text(
              phrase.text,
              style: AppTokens.rowTitleStyle.copyWith(color: c.ink),
            ),
            if (phrase.say != null)
              Text(
                phrase.say!,
                style: AppTokens.rowTitleStyle.copyWith(
                  fontWeight: FontWeight.w400,
                  color: c.ink,
                  fontStyle: FontStyle.italic,
                ),
              ),
            if (phrase.note != null)
              Text(
                phrase.note!,
                style: AppTokens.captionStyle.copyWith(color: c.muted),
              ),
          ],
        ),
      ),
    );
  }
}

/// One phrase, as large as the screen allows, to hold up to someone.
class ShowPhraseScreen extends StatelessWidget {
  final Phrase phrase;
  final Language language;

  const ShowPhraseScreen({
    super.key,
    required this.phrase,
    required this.language,
  });

  @override
  Widget build(BuildContext context) {
    final c = AppTokens.of(context);
    return Scaffold(
      backgroundColor: c.paper,
      appBar: AppBar(
        backgroundColor: c.paper,
        foregroundColor: c.ink,
        elevation: 0,
        title: Text(language.name),
      ),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(AppTokens.gutter),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Expanded(
                child: Center(
                  child: FittedBox(
                    fit: BoxFit.scaleDown,
                    child: ConstrainedBox(
                      constraints: BoxConstraints(
                        maxWidth: MediaQuery.sizeOf(context).width -
                            2 * AppTokens.gutter,
                      ),
                      child: Text(
                        phrase.text,
                        key: const Key('show-phrase'),
                        textAlign: TextAlign.center,
                        style: AppTokens.titleStyle.copyWith(
                          color: c.ink,
                          fontSize: 44,
                          height: 1.25,
                        ),
                      ),
                    ),
                  ),
                ),
              ),
              if (phrase.say != null)
                Text(
                  phrase.say!,
                  textAlign: TextAlign.center,
                  style: AppTokens.rowTitleStyle.copyWith(
                  fontWeight: FontWeight.w400,
                    color: c.ink,
                    fontStyle: FontStyle.italic,
                  ),
                ),
              const SizedBox(height: AppTokens.s8),
              Text(
                phrase.english,
                textAlign: TextAlign.center,
                style: AppTokens.captionStyle.copyWith(color: c.muted),
              ),
              const SizedBox(height: AppTokens.s24),
            ],
          ),
        ),
      ),
    );
  }
}
