// lib/features/phrasebook/data/phrases.dart
//
// A few phrases per language, offline — #40.
//
// WRITTEN FOR SAFARSATHI, NOT CHECKED BY A NATIVE SPEAKER, and the screen
// says so. They are the plain, common forms a phrase guide would give; the
// "Show" button exists because pointing at the screen works when
// pronunciation does not. Emergency NUMBERS are deliberately not here: they
// live on the SOS tab under their own rules about sources.
//
// Khasi carries only the two words visitors to Meghalaya are reliably taught.
// Anything more would be a guess, and most people there also speak English or
// Hindi.

class Phrase {
  final String english;

  /// What to say or show, in the language's own script.
  final String text;

  /// How to say it, for scripts the reader may not read. Null for languages
  /// written in the Latin alphabet.
  final String? say;

  /// A note on use: "(gayi, if a woman is speaking)".
  final String? note;

  const Phrase(this.english, this.text, {this.say, this.note});
}

class PhraseGroup {
  final String title;
  final List<Phrase> phrases;
  const PhraseGroup(this.title, this.phrases);
}

class Language {
  final String code;
  final String name;

  /// Its own name for itself, for the list.
  final String native;

  /// ISO country codes where a trip will want it first.
  final Set<String> countries;

  final List<PhraseGroup> groups;
  final String? note;

  const Language({
    required this.code,
    required this.name,
    required this.native,
    required this.countries,
    required this.groups,
    this.note,
  });

  int get phraseCount => groups.fold(0, (n, g) => n + g.phrases.length);
}

const languages = <Language>[
  Language(
    code: 'hi',
    name: 'Hindi',
    native: 'हिन्दी',
    countries: {'IN'},
    groups: [
      PhraseGroup('Basics', [
        Phrase('Hello', 'नमस्ते', say: 'Namaste'),
        Phrase('Thank you', 'धन्यवाद', say: 'Dhanyavaad'),
        Phrase('Please', 'कृपया', say: 'Kripya'),
        Phrase('Yes / No', 'हाँ / नहीं', say: 'Haan / Nahin'),
        Phrase('Sorry', 'माफ़ कीजिए', say: 'Maaf kijiye'),
        Phrase('I don\'t understand', 'मुझे समझ नहीं आया',
            say: 'Mujhe samajh nahin aaya'),
        Phrase('Do you speak English?', 'क्या आप अंग्रेज़ी बोलते हैं?',
            say: 'Kya aap angrezi bolte hain?'),
      ]),
      PhraseGroup('Emergency', [
        Phrase('Help!', 'बचाओ!', say: 'Bachao!'),
        Phrase('Call an ambulance', 'एम्बुलेंस बुलाइए',
            say: 'Ambulance bulaiye'),
        Phrase('Call the police', 'पुलिस को बुलाइए', say: 'Police ko bulaiye'),
        Phrase('I need a doctor', 'मुझे डॉक्टर चाहिए',
            say: 'Mujhe doctor chahiye'),
        Phrase('Where is the hospital?', 'अस्पताल कहाँ है?',
            say: 'Aspataal kahan hai?'),
        Phrase('I am lost', 'मैं रास्ता भूल गया हूँ',
            say: 'Main raasta bhool gaya hoon',
            note: '"gayi" instead of "gaya" if a woman is speaking'),
      ]),
      PhraseGroup('Getting around', [
        Phrase('Where is the toilet?', 'टॉयलेट कहाँ है?',
            say: 'Toilet kahan hai?'),
        Phrase('How much is this?', 'यह कितने का है?', say: 'Yeh kitne ka hai?'),
        Phrase('The bill, please', 'बिल दीजिए', say: 'Bill dijiye'),
        Phrase('Water', 'पानी', say: 'Paani'),
      ]),
      PhraseGroup('Food', [
        Phrase('I am vegetarian', 'मैं शाकाहारी हूँ', say: 'Main shakahari hoon'),
        Phrase('No meat, no fish, no egg', 'मांस, मछली, अंडा नहीं',
            say: 'Maans, machhli, anda nahin'),
        Phrase('No onion, no garlic', 'प्याज़ और लहसुन नहीं',
            say: 'Pyaaz aur lahsun nahin', note: 'For Jain food'),
      ]),
    ],
  ),
  Language(
    code: 'kha',
    name: 'Khasi',
    native: 'Ka Ktien Khasi',
    countries: {'IN'},
    note: 'Only the two words visitors are reliably taught. Most people in '
        'Meghalaya also speak English or Hindi.',
    groups: [
      PhraseGroup('Basics', [
        Phrase('Thank you', 'Khublei',
            note: 'Also used for hello and goodbye'),
        Phrase('Hello (literally "how?")', 'Kumno'),
      ]),
    ],
  ),
  Language(
    code: 'bn',
    name: 'Bengali',
    native: 'বাংলা',
    countries: {'BD', 'IN'},
    note: 'Bangladesh — across the Dawki–Tamabil border — and West Bengal.',
    groups: [
      PhraseGroup('Basics', [
        Phrase('Hello', 'নমস্কার / আসসালামু আলাইকুম',
            say: 'Nomoshkar / Assalamu alaikum'),
        Phrase('Thank you', 'ধন্যবাদ', say: 'Dhonnobad'),
        Phrase('Please', 'দয়া করে', say: 'Doya kore'),
        Phrase('Yes / No', 'হ্যাঁ / না', say: 'Hyan / Na'),
        Phrase('Sorry', 'দুঃখিত', say: 'Dukkhito'),
        Phrase('I don\'t understand', 'আমি বুঝতে পারছি না',
            say: 'Ami bujhte parchhi na'),
        Phrase('Do you speak English?', 'আপনি কি ইংরেজি বলেন?',
            say: 'Apni ki ingreji bolen?'),
      ]),
      PhraseGroup('Emergency', [
        Phrase('Help!', 'বাঁচাও!', say: 'Bachao!'),
        Phrase('Call an ambulance', 'অ্যাম্বুলেন্স ডাকুন',
            say: 'Ambulance dakun'),
        Phrase('Call the police', 'পুলিশ ডাকুন', say: 'Pulish dakun'),
        Phrase('I need a doctor', 'আমার ডাক্তার দরকার',
            say: 'Amar daktar dorkar'),
        Phrase('Where is the hospital?', 'হাসপাতাল কোথায়?',
            say: 'Hashpatal kothay?'),
      ]),
      PhraseGroup('Getting around', [
        Phrase('Where is the toilet?', 'টয়লেট কোথায়?', say: 'Toilet kothay?'),
        Phrase('How much is this?', 'এটার দাম কত?', say: 'Etar dam koto?'),
        Phrase('Water', 'পানি', say: 'Pani'),
      ]),
      PhraseGroup('Food', [
        Phrase('I am vegetarian', 'আমি নিরামিষাশী', say: 'Ami niramishashi'),
        Phrase('No meat, no fish, no egg', 'মাংস, মাছ, ডিম না',
            say: 'Mangsho, machh, dim na'),
        Phrase('No onion, no garlic', 'পেঁয়াজ, রসুন না',
            say: 'Peyaj, roshun na', note: 'For Jain food'),
      ]),
    ],
  ),
  Language(
    code: 'de',
    name: 'German',
    native: 'Deutsch',
    countries: {'DE', 'AT', 'CH'},
    groups: [
      PhraseGroup('Basics', [
        Phrase('Hello', 'Hallo / Guten Tag'),
        Phrase('Thank you', 'Danke'),
        Phrase('Please', 'Bitte'),
        Phrase('Yes / No', 'Ja / Nein'),
        Phrase('Sorry', 'Entschuldigung'),
        Phrase('I don\'t understand', 'Ich verstehe nicht'),
        Phrase('Do you speak English?', 'Sprechen Sie Englisch?'),
      ]),
      PhraseGroup('Emergency', [
        Phrase('Help!', 'Hilfe!'),
        Phrase('Call an ambulance', 'Rufen Sie einen Krankenwagen!'),
        Phrase('Call the police', 'Rufen Sie die Polizei!'),
        Phrase('I need a doctor', 'Ich brauche einen Arzt'),
        Phrase('Where is the hospital?', 'Wo ist das Krankenhaus?'),
        Phrase('I am lost', 'Ich habe mich verlaufen'),
      ]),
      PhraseGroup('Getting around', [
        Phrase('Where is the toilet?', 'Wo ist die Toilette?'),
        Phrase('How much is this?', 'Wie viel kostet das?'),
        Phrase('The bill, please', 'Die Rechnung, bitte'),
        Phrase('Water', 'Wasser'),
      ]),
      PhraseGroup('Food', [
        Phrase('I am vegetarian', 'Ich bin Vegetarier',
            note: '"Vegetarierin" if a woman is speaking'),
        Phrase('No meat, no fish, no egg', 'Kein Fleisch, kein Fisch, keine Eier'),
        Phrase('No onion, no garlic', 'Keine Zwiebeln, kein Knoblauch',
            note: 'For Jain food'),
      ]),
    ],
  ),
  Language(
    code: 'fr',
    name: 'French',
    native: 'Français',
    countries: {'FR', 'BE', 'CH'},
    groups: [
      PhraseGroup('Basics', [
        Phrase('Hello', 'Bonjour'),
        Phrase('Thank you', 'Merci'),
        Phrase('Please', 'S\'il vous plaît'),
        Phrase('Yes / No', 'Oui / Non'),
        Phrase('Sorry', 'Pardon'),
        Phrase('I don\'t understand', 'Je ne comprends pas'),
        Phrase('Do you speak English?', 'Parlez-vous anglais ?'),
      ]),
      PhraseGroup('Emergency', [
        Phrase('Help!', 'Au secours !'),
        Phrase('Call an ambulance', 'Appelez une ambulance !'),
        Phrase('Call the police', 'Appelez la police !'),
        Phrase('I need a doctor', 'J\'ai besoin d\'un médecin'),
        Phrase('Where is the hospital?', 'Où est l\'hôpital ?'),
        Phrase('I am lost', 'Je suis perdu',
            note: '"perdue" if a woman is speaking'),
      ]),
      PhraseGroup('Getting around', [
        Phrase('Where is the toilet?', 'Où sont les toilettes ?'),
        Phrase('How much is this?', 'Combien ça coûte ?'),
        Phrase('The bill, please', 'L\'addition, s\'il vous plaît'),
        Phrase('Water', 'De l\'eau'),
      ]),
      PhraseGroup('Food', [
        Phrase('I am vegetarian', 'Je suis végétarien',
            note: '"végétarienne" if a woman is speaking'),
        Phrase('No meat, no fish, no egg',
            'Pas de viande, pas de poisson, pas d\'œufs'),
        Phrase('No onion, no garlic', 'Pas d\'oignon, pas d\'ail',
            note: 'For Jain food'),
      ]),
    ],
  ),
  Language(
    code: 'it',
    name: 'Italian',
    native: 'Italiano',
    countries: {'IT', 'CH'},
    groups: [
      PhraseGroup('Basics', [
        Phrase('Hello', 'Buongiorno / Ciao'),
        Phrase('Thank you', 'Grazie'),
        Phrase('Please', 'Per favore'),
        Phrase('Yes / No', 'Sì / No'),
        Phrase('Sorry', 'Scusi'),
        Phrase('I don\'t understand', 'Non capisco'),
        Phrase('Do you speak English?', 'Parla inglese?'),
      ]),
      PhraseGroup('Emergency', [
        Phrase('Help!', 'Aiuto!'),
        Phrase('Call an ambulance', 'Chiami un\'ambulanza!'),
        Phrase('Call the police', 'Chiami la polizia!'),
        Phrase('I need a doctor', 'Ho bisogno di un medico'),
        Phrase('Where is the hospital?', 'Dov\'è l\'ospedale?'),
        Phrase('I am lost', 'Mi sono perso',
            note: '"persa" if a woman is speaking'),
      ]),
      PhraseGroup('Getting around', [
        Phrase('Where is the toilet?', 'Dov\'è il bagno?'),
        Phrase('How much is this?', 'Quanto costa?'),
        Phrase('The bill, please', 'Il conto, per favore'),
        Phrase('Water', 'Acqua'),
      ]),
      PhraseGroup('Food', [
        Phrase('I am vegetarian', 'Sono vegetariano',
            note: '"vegetariana" if a woman is speaking'),
        Phrase('No meat, no fish, no egg', 'Niente carne, niente pesce, niente uova'),
        Phrase('No onion, no garlic', 'Niente cipolla, niente aglio',
            note: 'For Jain food'),
      ]),
    ],
  ),
  Language(
    code: 'es',
    name: 'Spanish',
    native: 'Español',
    countries: {'ES'},
    groups: [
      PhraseGroup('Basics', [
        Phrase('Hello', 'Hola'),
        Phrase('Thank you', 'Gracias'),
        Phrase('Please', 'Por favor'),
        Phrase('Yes / No', 'Sí / No'),
        Phrase('Sorry', 'Perdón'),
        Phrase('I don\'t understand', 'No entiendo'),
        Phrase('Do you speak English?', '¿Habla inglés?'),
      ]),
      PhraseGroup('Emergency', [
        Phrase('Help!', '¡Socorro!'),
        Phrase('Call an ambulance', '¡Llame a una ambulancia!'),
        Phrase('Call the police', '¡Llame a la policía!'),
        Phrase('I need a doctor', 'Necesito un médico'),
        Phrase('Where is the hospital?', '¿Dónde está el hospital?'),
        Phrase('I am lost', 'Estoy perdido',
            note: '"perdida" if a woman is speaking'),
      ]),
      PhraseGroup('Getting around', [
        Phrase('Where is the toilet?', '¿Dónde está el baño?'),
        Phrase('How much is this?', '¿Cuánto cuesta?'),
        Phrase('The bill, please', 'La cuenta, por favor'),
        Phrase('Water', 'Agua'),
      ]),
      PhraseGroup('Food', [
        Phrase('I am vegetarian', 'Soy vegetariano',
            note: '"vegetariana" if a woman is speaking'),
        Phrase('No meat, no fish, no egg', 'Sin carne, sin pescado, sin huevo'),
        Phrase('No onion, no garlic', 'Sin cebolla, sin ajo',
            note: 'For Jain food'),
      ]),
    ],
  ),
];

/// Languages with the ones for [countries] first — a trip in Meghalaya
/// opens on Hindi and Khasi, one in Austria on German. Otherwise the list's
/// own order.
List<Language> languagesFor(Set<String> countries) {
  final first = [
    for (final l in languages)
      if (l.countries.intersection(countries).isNotEmpty) l,
  ];
  return [
    ...first,
    for (final l in languages)
      if (!first.contains(l)) l,
  ];
}
