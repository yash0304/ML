// test/streams_test.dart
//
// Thirty lines of combinator instead of a reactive-extensions dependency, so
// the thirty lines had better be right.

import 'dart:async';

import 'package:flutter_test/flutter_test.dart';
import 'package:safarsathi/core/util/streams.dart';

void main() {
  test('waits for BOTH sources before emitting anything', () async {
    final a = StreamController<int>();
    final b = StreamController<int>();
    final seen = <String>[];
    final sub = combineLatest2(a.stream, b.stream, (int x, int y) => '$x:$y')
        .listen(seen.add);

    a.add(1);
    await Future<void>.delayed(Duration.zero);
    // A readiness check built from stops alone would flash "not ready" on
    // every trip for one frame and then correct itself.
    expect(seen, isEmpty);

    b.add(2);
    await Future<void>.delayed(Duration.zero);
    expect(seen, ['1:2']);

    await sub.cancel();
    await a.close();
    await b.close();
  });

  test('re-emits on either side, holding the other', () async {
    final a = StreamController<int>();
    final b = StreamController<int>();
    final seen = <String>[];
    final sub = combineLatest2(a.stream, b.stream, (int x, int y) => '$x:$y')
        .listen(seen.add);

    a.add(1);
    b.add(2);
    a.add(3);
    b.add(4);
    await Future<void>.delayed(Duration.zero);

    expect(seen, ['1:2', '3:2', '3:4']);
    await sub.cancel();
    await a.close();
    await b.close();
  });

  test('closes only when both sources are done', () async {
    final a = StreamController<int>();
    final b = StreamController<int>();
    var closed = false;
    final sub = combineLatest2(
      a.stream,
      b.stream,
      (int x, int y) => x + y,
    ).listen((_) {}, onDone: () => closed = true);

    a.add(1);
    b.add(2);
    await a.close();
    await Future<void>.delayed(Duration.zero);
    expect(closed, isFalse);

    await b.close();
    await Future<void>.delayed(Duration.zero);
    expect(closed, isTrue);
    await sub.cancel();
  });

  test('an error on either side reaches the listener', () async {
    final a = StreamController<int>();
    final b = StreamController<int>();
    Object? caught;
    final sub = combineLatest2(
      a.stream,
      b.stream,
      (int x, int y) => x + y,
    ).listen((_) {}, onError: (Object e) => caught = e);

    a.addError('boom');
    await Future<void>.delayed(Duration.zero);
    expect(caught, 'boom');

    await sub.cancel();
    await a.close();
    await b.close();
  });

  test('cancelling unsubscribes from both sources', () async {
    final a = StreamController<int>();
    final b = StreamController<int>();
    final sub = combineLatest2(
      a.stream,
      b.stream,
      (int x, int y) => x + y,
    ).listen((_) {});

    await sub.cancel();
    expect(a.hasListener, isFalse);
    expect(b.hasListener, isFalse);
    await a.close();
    await b.close();
  });
}
