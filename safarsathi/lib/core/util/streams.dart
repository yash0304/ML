// lib/core/util/streams.dart
//
// One stream combinator, written out rather than pulled in.
//
// The app needs exactly this and nothing else from the reactive-extensions
// world, and adding rxdart for it would mean a dependency, a mental model and
// a migration risk in exchange for thirty lines.

import 'dart:async';

/// Emits whenever either source emits, once BOTH have emitted at least once.
///
/// The "both have emitted" rule matters here: a readiness check built from
/// stops alone, before contacts arrive, would flash "not ready" on every trip
/// for one frame and then correct itself. A screen that lies briefly is worse
/// than a screen that is briefly empty.
///
/// LISTENABLE MORE THAN ONCE. Each listener gets its own pair of
/// subscriptions. This was a single-subscription controller, and a screen
/// built in a ListView disposes a section scrolled off the top and builds it
/// again on the way back: the second listen threw "Stream has already been
/// listened to" and the Trip page came back blank — reported as "whenever I
/// scroll down and go back up it is blank". The sources must allow several
/// listeners too; every Drift stream does.
Stream<R> combineLatest2<A, B, R>(
  Stream<A> a,
  Stream<B> b,
  R Function(A, B) combine,
) => Stream.multi((controller) {
  A? latestA;
  B? latestB;
  var hasA = false;
  var hasB = false;
  var doneA = false;
  var doneB = false;

  void emit() {
    if (hasA && hasB) controller.add(combine(latestA as A, latestB as B));
  }

  void closeIfDone() {
    if (doneA && doneB) controller.close();
  }

  final subA = a.listen(
    (value) {
      latestA = value;
      hasA = true;
      emit();
    },
    onError: controller.addError,
    onDone: () {
      doneA = true;
      closeIfDone();
    },
  );
  final subB = b.listen(
    (value) {
      latestB = value;
      hasB = true;
      emit();
    },
    onError: controller.addError,
    onDone: () {
      doneB = true;
      closeIfDone();
    },
  );

  // Both are cancelled even if the first throws, or a database stream is
  // left open behind a disposed screen.
  controller.onCancel = () => Future.wait([subA.cancel(), subB.cancel()]);
});
