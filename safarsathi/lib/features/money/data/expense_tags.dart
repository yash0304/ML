// lib/features/money/data/expense_tags.dart
//
// What a spend was for — food, the stay, the taxi — so the ledger can say
// where the money went and not only how much went.
//
// Stored in Expenses.category, a column that existed from the first schema
// and that no screen ever wrote. A fixed list rather than free text: "food",
// "Food" and "lunch" would be three tags in a total, and the point of a tag
// is to add up.

/// Tag key → what it reads as. Order is the order the chips appear in.
const expenseTags = <String, String>{
  'food': 'Food',
  'stay': 'Stay',
  'transport': 'Transport',
  'fuel': 'Fuel',
  'entry': 'Entry & tickets',
  'guide': 'Guides & tips',
  'shopping': 'Shopping',
  'other': 'Other',
};

const untaggedLabel = 'Untagged';

String tagLabel(String? key) =>
    key == null ? untaggedLabel : expenseTags[key] ?? key;

/// A first guess from what was typed, offered — never saved unasked. The
/// person sees the chip lit and can change it; nothing is tagged behind
/// their back.
String? guessTag(String description) {
  final d = description.toLowerCase();
  bool has(List<String> words) => words.any(d.contains);
  if (has(['petrol', 'diesel', 'fuel'])) return 'fuel';
  if (has(['taxi', 'cab', 'sumo', 'bus', 'auto', 'train', 'flight',
      'driver', 'boat', 'toll', 'parking'])) {
    return 'transport';
  }
  if (has(['hotel', 'homestay', 'guest house', 'guesthouse', 'room',
      'stay', 'camp', 'tent'])) {
    return 'stay';
  }
  if (has(['ticket', 'entry', 'entrance', 'permit', 'fee'])) return 'entry';
  if (has(['guide', 'tip'])) return 'guide';
  if (has(['lunch', 'dinner', 'breakfast', 'tea', 'chai', 'coffee', 'food',
      'snack', 'momo', 'thali', 'dhaba', 'restaurant', 'water', 'meal'])) {
    return 'food';
  }
  if (has(['shop', 'souvenir', 'gift', 'market', 'shawl'])) return 'shopping';
  return null;
}
