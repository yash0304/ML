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
Stream<R> combineLatest2<A, B, R>(
  Stream<A> a,
  Stream<B> b,
  R Function(A, B) combine,
) {
  late StreamController<R> controller;
  StreamSubscription<A>? subA;
  StreamSubscription<B>? subB;

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

  controller = StreamController<R>(
    onListen: () {
      subA = a.listen(
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
      subB = b.listen(
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
    },
    onCancel: () async {
      // Both subscriptions are cancelled even if the first throws, or a
      // database stream is left open behind a disposed screen.
      await Future.wait([
        if (subA != null) subA!.cancel(),
        if (subB != null) subB!.cancel(),
      ]);
    },
  );

  return controller.stream;
}
