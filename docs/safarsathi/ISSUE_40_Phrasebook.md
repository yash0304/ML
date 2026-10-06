# #40 — Offline phrasebook

**Backlog line:** Per-country basic phrases. Lower priority than everything above.

## Why

On the Dawki side of Meghalaya, a Bangladeshi shop; in Austria, a pharmacy.
A few phrases, readable or showable with no signal, cover the moments that
matter: help, a doctor, the toilet, the bill, and vegetarian food (with "no
onion, no garlic" for Jain food).

## Build

1. **Data** (`lib/features/phrasebook/data/phrases.dart`): a const list of
   languages, each with groups (Basics, Emergency, Getting around, Food) of
   phrases. A phrase has the English, the text in the language's own script,
   a romanised "say" line for scripts the reader may not read (Hindi,
   Bengali), and an optional note ("gayi if a woman is speaking").
2. **Languages:** Hindi and Khasi (IN), Bengali (BD, IN), German (DE, AT,
   CH), French (FR, BE, CH), Italian (IT, CH), Spanish (ES). Khasi carries
   only Khublei and Kumno, with a note that most people in Meghalaya also
   speak English or Hindi.
3. **Order:** `languagesFor(countries)` puts the languages of the trip's
   stop countries first; the rest keep the list's order.
4. **Screen** (More → Phrasebook): language chips, the note "Written for
   SafarSathi, not checked by a native speaker", then the groups. Tapping a
   phrase opens it full screen at 44 pt, with the say line and English under it.

## Rules

- **No phone numbers.** Emergency numbers are on the SOS tab under its rule
  that every number shows its government source. A test fails on any run of
  three or more digits in a phrase.
- Nothing is downloaded and nothing is spoken aloud; there is no
  translation service.

## Tests

`test/phrasebook_test.dart`: ordering by country, unique codes, a say line on
every Hindi and Bengali phrase, no digits, vegetarian/Jain/help phrases in
every full language, and the screen (opens on the trip's language, switches
language, shows a phrase large, draws in the night theme).
