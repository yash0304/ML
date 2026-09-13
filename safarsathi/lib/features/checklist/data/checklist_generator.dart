// lib/features/checklist/data/checklist_generator.dart
//
// The pack list — issue #29.
//
// Three inputs, all already in the database: a stop's activity tags, its
// nights, and how many stops the trip has. NO WEATHER LOOKUP AT GENERATION
// TIME. The forecast is a cached snapshot that may not exist, and a checklist
// that cannot be produced without one is a checklist you cannot produce at a
// dhaba.
//
// THE RULE THAT IS THE WHOLE ISSUE: a manual edit survives regeneration.
// Change a label, change a quantity, or delete an item, and running the
// generator again must not undo it.

import 'package:drift/drift.dart';

import '../../../core/database/app_database.dart';
import '../../trips/data/trip_editor.dart';

/// One thing to pack, before it reaches the database.
class PackItem {
  /// Stable identity for this rule, unchanged by anything the user does to
  /// the label. Matching by label meant a rename produced a duplicate.
  final String key;

  final String label;

  /// Null when a count adds nothing — you do not pack "1 passport".
  final String? quantity;

  /// The tags that produced it, shown under the item. A generated list nobody
  /// understands gets ignored, and then the one item that mattered is ignored
  /// with it.
  final Set<String> sourceTags;

  const PackItem({
    required this.key,
    required this.label,
    this.quantity,
    this.sourceTags = const {},
  });
}

/// A rule: this tag wants these things.
class _Rule {
  final String label;

  /// Given the nights across every stop carrying this tag, how many.
  /// Null for things you pack one of whatever happens.
  final String? Function(int nights)? count;

  const _Rule(this.label, {this.count});
}

String? _perNight(int nights) => '$nights';
String? _perNightPlusOne(int nights) => '${nights + 1}';

/// What each activity tag asks for.
///
/// Declared as data rather than code so a new tag is one list entry, and so
/// the whole rule set can be read at a glance and argued with. These are
/// opinionated about Meghalaya in October because that is the trip this app
/// was built for; they are a starting point, not a truth.
const _rules = <String, List<_Rule>>{
  'trek': [
    _Rule('Walking boots'),
    _Rule('Blister plasters'),
    _Rule('Trekking socks', count: _perNight),
    _Rule('Leech socks'),
    _Rule('Water bottle'),
  ],
  'caves': [
    _Rule('Headtorch'),
    _Rule('Spare batteries'),
    _Rule('Old shoes you can soak'),
    _Rule('Change of clothes'),
  ],
  'rain': [
    _Rule('Poncho or rain jacket'),
    _Rule('Dry bag for the phone'),
    _Rule('Quick-dry trousers'),
    _Rule('Spare plastic bags'),
  ],
  'homestay': [
    _Rule('Torch, for the walk to the loo'),
    _Rule('Own towel'),
    _Rule('Small gift for the host'),
    _Rule('Cash — no card machine'),
  ],
  'camping': [
    _Rule('Sleeping bag'),
    _Rule('Mat'),
    _Rule('Headtorch'),
    _Rule('Matches, in a dry bag'),
  ],
  'beach': [_Rule('Sunscreen'), _Rule('Flip-flops'), _Rule('Swimwear')],
  'cold': [
    _Rule('Thermals'),
    _Rule('Gloves'),
    _Rule('Woollen cap'),
    _Rule('Lip balm'),
  ],
  'city': [_Rule('Phone charger'), _Rule('Power bank')],
  'drive': [
    _Rule('Driving licence'),
    _Rule('Vehicle papers'),
    _Rule('Phone mount'),
    _Rule('Offline music'),
  ],
};

/// Packed whatever the trip is.
const _always = <_Rule>[
  _Rule('Changes of clothes', count: _perNightPlusOne),
  _Rule('Toothbrush and paste'),
  _Rule('Any medicines you take'),
  _Rule('Phone charger and cable'),
  _Rule('ID'),
];

/// Works out the pack list from stops alone. Pure, so the rules are arguable
/// in a test rather than only observable on a phone.
List<PackItem> generatePackList(List<Stop> stops) {
  // Nights per tag, so "trekking socks" scales with the trekking, not with
  // the whole trip.
  final nightsByTag = <String, int>{};
  var totalNights = 0;

  for (final stop in stops) {
    totalNights += stop.nights;
    for (final tag in parseTags(stop.activityTags)) {
      nightsByTag[tag] = (nightsByTag[tag] ?? 0) + stop.nights;
    }
  }

  // An item asked for by two tags is packed once and credits both, which is
  // the point of showing the tags: you can see why a headtorch is on the list
  // twice over and pack one.
  final merged = <String, PackItem>{};

  void add(_Rule rule, int nights, String? tag) {
    final existing = merged[rule.label];
    final tags = {...?existing?.sourceTags, ?tag};
    merged[rule.label] = PackItem(
      key: rule.label,
      label: rule.label,
      quantity: rule.count?.call(nights) ?? existing?.quantity,
      sourceTags: tags,
    );
  }

  for (final rule in _always) {
    add(rule, totalNights, null);
  }
  for (final entry in nightsByTag.entries) {
    for (final rule in _rules[entry.key] ?? const <_Rule>[]) {
      add(rule, entry.value, entry.key);
    }
  }

  return merged.values.toList();
}

/// Writes the pack list into the checklist, leaving anything the user touched
/// exactly as they left it.
///
/// Deleting needs the same care as editing. A deleted generated item would
/// simply reappear on the next run, so a delete marks the row edited AND done
/// rather than removing it. That is the only way "I do not need leech socks"
/// survives a regeneration without a separate tombstone table.
Future<void> regeneratePackList(AppDatabase db, int tripId) async {
  final stops =
      await (db.select(db.stops)
            ..where((s) => s.tripId.equals(tripId))
            ..orderBy([(s) => OrderingTerm(expression: s.sequenceOrder)]))
          .get();

  final wanted = generatePackList(stops);

  await db.transaction(() async {
    final existing =
        await (db.select(db.checklistItems)..where(
              (i) => i.tripId.equals(tripId) & i.isBlocking.equals(false),
            ))
            .get();

    // KEYED BY RULE, NOT BY LABEL. Rows written before v2 have a null key, so
    // they are adopted by label once and carry a key from then on.
    final byKey = <String, ChecklistItem>{};
    for (final e in existing) {
      final key = e.generatorKey;
      if (key != null) {
        byKey[key] = e;
      } else if (e.isGenerated && !byKey.containsKey(e.label)) {
        byKey[e.label] = e;
      }
    }

    var order = 0;
    final claimed = <int>{};

    for (final item in wanted) {
      order++;
      final match = byKey[item.key];

      if (match == null) {
        await db
            .into(db.checklistItems)
            .insert(
              ChecklistItemsCompanion.insert(
                tripId: tripId,
                label: item.label,
                generatorKey: Value(item.key),
                quantity: Value(item.quantity),
                sourceTags: Value(item.sourceTags.join(',')),
                sortOrder: Value(order),
              ),
            );
        continue;
      }

      claimed.add(match.id);

      // Their wording and their count win over ours, always. The key is still
      // stamped on, so the next run recognises the row they renamed.
      if (match.isUserEdited) {
        if (match.generatorKey == null) {
          await (db.update(
            db.checklistItems,
          )..where((i) => i.id.equals(match.id))).write(
            ChecklistItemsCompanion(generatorKey: Value(item.key)),
          );
        }
        continue;
      }

      await (db.update(
        db.checklistItems,
      )..where((i) => i.id.equals(match.id))).write(
        ChecklistItemsCompanion(
          label: Value(item.label),
          generatorKey: Value(item.key),
          quantity: Value(item.quantity),
          sourceTags: Value(item.sourceTags.join(',')),
          sortOrder: Value(order),
        ),
      );
    }

    // Items the rules no longer ask for. A generated one goes; one the user
    // wrote or edited stays, because they put it there on purpose.
    for (final e in existing) {
      if (claimed.contains(e.id)) continue;
      if (e.isUserEdited || !e.isGenerated) continue;
      await (db.delete(
        db.checklistItems,
      )..where((i) => i.id.equals(e.id))).go();
    }
  });
}
