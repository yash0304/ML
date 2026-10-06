# #58 — Memories tab

**Asked for (6 Oct, after the Meghalaya trip):** "It should have memories tab
where few of the photos can be saved as well."

## Why

The Timeline (#30) already takes photos with notes, but as part of the day's
record, one note at a time. After a trip you want the photos together: a
few of the best, by place, to look back at and to share.

## Build

1. **No new table.** A memory is a `TimelineEntries` note with
   `photoPaths`. `buildMemories(entries, stops)` flattens them into one item
   per photo, grouped by day, oldest first. Each day lists its places in
   order without repeats. Route-log fixes and notes without photos are left
   out.
2. **Tab:** Memories is sixth, after SOS (Trip, Diary, Money, SOS,
   Memories, More), so SOS stays fourth.
3. **Album:** the header shows the trip, "N photos · N days", then a stencil
   label per day ("2 OCT · CHERRAPUNJI") over a three-column grid. The grid
   decodes images at 360 px.
4. **Add photos:** the system picker opens at once and allows several
   (`pickMultiImage`); **Take one** opens the camera. Each photo can be
   removed before saving. Place chips default to the current stop, and
   there is a field for a few words. **Keep these N photos** saves. Copies
   abandoned by leaving or removing are deleted from the app's folder.
5. **Date:** while at the stop, now. Afterwards, noon on the day the trip
   arrived there (`memoryDate`), so photos added at home keep the trip's order.
6. **Viewer:** a PageView across the whole album, pinch to zoom, "2 of 7",
   the words, date, time and place. Tap the words to edit them; when several
   photos share the note, the box says the words show under all of them.
   **Share** passes the file with its words and place. **Delete** asks
   first, says the gallery original is not touched, removes the photo from
   its note, deletes the note if nothing is left, and deletes the file.

## Not in scope

- Photos in the backup file. They would make it hundreds of MB; the
  Timeline has the same rule, and both screens say so.
- Reading the date from a photo's EXIF data. This would need a new
  dependency; filing under the stop's arrival day is close enough for an
  album.

## Tests

`test/memories_test.dart`:
- grouping and places;
- the three date cases;
- removing a photo: one of several, the last photo with words, the last
  photo without words;
- editing and clearing the words;
- the tab: empty state, counts, the viewer and swipe, share, words and
  delete;
- the add screen: choose, camera, remove, place, save, and discarding on
  leaving;
- six tab labels fitting whole at 320 dp.

Goldens: `memories.png`, `memories_lamp.png`.
