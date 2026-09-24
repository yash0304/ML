// lib/features/discovery/data/poi_category.dart
//
// Which OSM tags the app asks for, and what it calls them.
//
// OpenStreetMap's tagging is vast and inconsistent. This is a deliberately
// small slice: the things a driver actually stops for between two towns.

import '../../contacts/data/contacts_dao.dart' show ContactCategory;

/// A category as the app knows it, with the OSM filters that find it.
class PoiCategory {
  final String key;
  final String label;

  /// Overpass tag filters, e.g. `amenity=fuel`. Any match qualifies.
  final List<String> filters;

  const PoiCategory({
    required this.key,
    required this.label,
    required this.filters,
  });
}

/// DECLARATION ORDER IS MATCH ORDER, and the rule is that the more specific
/// tag wins. A hotel with a restaurant is a hotel: `tourism=hotel` is a
/// stronger statement about what the place IS than `amenity=restaurant`, which
/// half of them also carry. Getting this backwards files every guest house on
/// the route under Food, and at 8pm that is the wrong answer.
const poiCategories = <PoiCategory>[
  PoiCategory(
    key: 'fuel',
    label: 'Fuel',
    filters: ['amenity=fuel'],
  ),
  PoiCategory(
    key: 'hospital',
    label: 'Medical',
    filters: ['amenity=hospital', 'amenity=clinic', 'amenity=doctors'],
  ),
  PoiCategory(
    key: 'pharmacy',
    label: 'Chemist',
    filters: ['amenity=pharmacy'],
  ),
  PoiCategory(
    key: 'accommodation',
    label: 'Stay',
    filters: [
      'tourism=hotel',
      'tourism=guest_house',
      'tourism=hostel',
      'tourism=chalet',
    ],
  ),
  PoiCategory(
    key: 'restaurant',
    label: 'Food',
    filters: ['amenity=restaurant', 'amenity=cafe', 'amenity=fast_food'],
  ),
  PoiCategory(
    key: 'atm',
    label: 'Cash',
    filters: ['amenity=atm', 'amenity=bank'],
  ),
  PoiCategory(
    key: 'toilets',
    label: 'Toilets',
    filters: ['amenity=toilets'],
  ),
  // SIGHTS: what a Meghalaya road is driven for. Waterfalls and caves are
  // tagged apart from viewpoints in OSM (Nohkalikai is waterway=waterfall,
  // Mawsmai natural=cave_entrance), so all four are asked for.
  PoiCategory(
    key: 'viewpoint',
    label: 'Sights',
    filters: [
      'tourism=viewpoint',
      'tourism=attraction',
      'waterway=waterfall',
      'natural=cave_entrance',
    ],
  ),
  PoiCategory(
    key: 'police',
    label: 'Police',
    filters: ['amenity=police'],
  ),
  PoiCategory(
    key: 'repair',
    label: 'Repairs',
    filters: ['shop=car_repair', 'shop=tyres', 'shop=motorcycle_repair'],
  ),
];

/// Categories a trip fetches unless the user narrows it. Deliberately not all
/// of them: every extra category is another query against free infrastructure,
/// and nobody plans a route around public toilets.
const defaultPoiCategoryKeys = [
  'fuel',
  'hospital',
  'pharmacy',
  'accommodation',
  'restaurant',
  'atm',
  'repair',
  // Asked for on the phone: viewpoints between stops, to plan and send
  // home. Last, so a slow query has already fetched the help it needs.
  'viewpoint',
];

/// What a place's category reads as on screen: the diary's word where the
/// diary has one, else this file's ("Sights", "Cash").
String placeCategoryLabel(String key) =>
    ContactCategory.labels[key] ?? poiCategoryByKey(key)?.label ?? key;

PoiCategory? poiCategoryByKey(String key) {
  for (final c in poiCategories) {
    if (c.key == key) return c;
  }
  return null;
}

/// Works out which app category an OSM element belongs to, from its tags.
///
/// First match in declaration order wins, so a hotel with a restaurant reads
/// as a place to sleep rather than a place to eat — which is what you were
/// looking for when you queried at 8pm.
String? categoryForTags(Map<String, String> tags) {
  for (final category in poiCategories) {
    for (final filter in category.filters) {
      final parts = filter.split('=');
      if (parts.length != 2) continue;
      if (tags[parts[0]] == parts[1]) return category.key;
    }
  }
  return null;
}
