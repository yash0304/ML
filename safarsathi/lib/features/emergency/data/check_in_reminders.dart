// lib/features/emergency/data/check_in_reminders.dart
//
// "Reached Sohra?" — #34, when nobody has checked in.
//
// At a leg's planned arrival plus a buffer, if that stop has no check-in,
// the phone reminds its owner. A second reminder at twice the buffer says
// home has not heard. That is the whole escalation: a notification that
// opens the check-in screen. NOTHING IS SENT BY ITSELF — an escalation that
// texted people automatically would fire from a phone left in a bag, and it
// would need an SMS permission this app does not hold and will not ask for.
//
// No server, no WorkManager: an inexact local notification scheduled on the
// phone does the same job with no background worker, and survives a reboot.

import 'package:drift/drift.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:timezone/data/latest.dart' as tzdata;
import 'package:timezone/timezone.dart' as tz;

import '../../../core/database/app_database.dart';

/// How long after a planned arrival before the first reminder, when no
/// trusted person says otherwise. Two hours: a Meghalaya road is late by an
/// hour as a matter of course.
const defaultReminderBufferMinutes = 120;

class CheckInReminder {
  /// Stable per leg and step, so rescheduling replaces rather than piles up.
  final int id;
  final DateTime at;
  final int stopId;
  final String stopName;

  /// The second, stronger reminder.
  final bool escalation;

  const CheckInReminder({
    required this.id,
    required this.at,
    required this.stopId,
    required this.stopName,
    required this.escalation,
  });

  String get title => escalation
      ? 'Still no check-in from $stopName'
      : 'Reached $stopName?';

  String get body => escalation
      ? 'Home has not heard from you. Tap to text them that you are safe.'
      : 'Your plan had you arriving a while ago. Tap to tell home.';

  /// What the notification carries back when tapped.
  String get payload => 'checkin:$stopId';

  @override
  bool operator ==(Object other) =>
      other is CheckInReminder &&
      other.id == id &&
      other.at == at &&
      other.stopId == stopId &&
      other.escalation == escalation;

  @override
  int get hashCode => Object.hash(id, at, stopId, escalation);

  @override
  String toString() => 'CheckInReminder($id, $at, $stopName, '
      '${escalation ? 'escalation' : 'first'})';
}

/// The reminders that should exist right now. Pure.
///
/// A leg with no planned arrival has no reminder: there is no time to be
/// late against, and guessing one would nag. A check-in at the destination
/// made after the leg set off counts — an earlier one is from a previous
/// visit (Shillong is often both first and last). Times already past are
/// dropped; a reminder for yesterday helps nobody.
List<CheckInReminder> checkInReminders({
  required List<Leg> legs,
  required Map<int, String> stopNames,
  required Map<int, DateTime> checkIns,
  required DateTime now,
  int bufferMinutes = defaultReminderBufferMinutes,
}) {
  final out = <CheckInReminder>[];
  for (final leg in legs) {
    final arrival = leg.plannedArrival;
    if (arrival == null) continue;
    final since =
        leg.plannedDeparture ?? arrival.subtract(const Duration(days: 1));
    final last = checkIns[leg.toStopId];
    if (last != null && last.isAfter(since)) continue;

    final name = stopNames[leg.toStopId] ?? 'your stop';
    for (final step in [1, 2]) {
      final at = arrival.add(Duration(minutes: bufferMinutes * step));
      if (!at.isAfter(now)) continue;
      out.add(
        CheckInReminder(
          id: leg.id * 2 + (step - 1),
          at: at,
          stopId: leg.toStopId,
          stopName: name,
          escalation: step == 2,
        ),
      );
    }
  }
  out.sort((a, b) => a.at.compareTo(b.at));
  return out;
}

/// The buffer: the shortest any trusted person asked for, else the default.
int reminderBuffer(List<TrustedContact> trusted) {
  final asked = [
    for (final t in trusted)
      if (t.escalate && t.escalateAfterMinutes > 0) t.escalateAfterMinutes,
  ];
  if (asked.isEmpty) return defaultReminderBufferMinutes;
  return asked.reduce((a, b) => a < b ? a : b);
}

/// Where reminders go. Abstract so tests can watch what would be scheduled.
abstract class ReminderScheduler {
  /// Replaces every scheduled check-in reminder with [reminders].
  Future<void> replaceAll(List<CheckInReminder> reminders);

  /// Asks to post notifications. True when allowed.
  Future<bool> requestPermission();

  /// The stop of a reminder that opened the app, if one did.
  Future<int?> launchedForStop();
}

int? stopFromPayload(String? payload) {
  if (payload == null || !payload.startsWith('checkin:')) return null;
  return int.tryParse(payload.substring('checkin:'.length));
}

class DeviceReminderScheduler implements ReminderScheduler {
  DeviceReminderScheduler({this.onTapped});

  /// Called with the stop when a reminder is tapped while the app runs.
  final void Function(int stopId)? onTapped;

  final _plugin = FlutterLocalNotificationsPlugin();
  bool _ready = false;

  static const _channel = AndroidNotificationDetails(
    'check_in',
    'Check-in reminders',
    channelDescription:
        'A reminder to tell home you have arrived, after a leg\'s planned '
        'arrival. Nothing is sent without you.',
    importance: Importance.high,
    priority: Priority.high,
  );

  Future<void> _init() async {
    if (_ready) return;
    tzdata.initializeTimeZones();
    await _plugin.initialize(
      settings: const InitializationSettings(
        android: AndroidInitializationSettings('@mipmap/ic_launcher'),
      ),
      onDidReceiveNotificationResponse: (r) {
        final stop = stopFromPayload(r.payload);
        if (stop != null) onTapped?.call(stop);
      },
    );
    _ready = true;
  }

  @override
  Future<void> replaceAll(List<CheckInReminder> reminders) async {
    await _init();
    // The app posts no other notifications, so clearing all of them clears
    // exactly the stale reminders and nothing else.
    await _plugin.cancelAll();
    for (final r in reminders) {
      try {
        await _plugin.zonedSchedule(
          id: r.id,
          title: r.title,
          body: r.body,
          // An instant, not a wall-clock time: UTC is right wherever the
          // phone thinks it is, Shillong or Salzburg.
          scheduledDate: tz.TZDateTime.from(r.at.toUtc(), tz.UTC),
          notificationDetails: const NotificationDetails(android: _channel),
          androidScheduleMode: AndroidScheduleMode.inexactAllowWhileIdle,
          payload: r.payload,
        );
      } on Object catch (e) {
        // One bad reminder must not stop the rest being scheduled.
        debugPrint('Check-in reminder ${r.id} not scheduled: $e');
      }
    }
  }

  @override
  Future<bool> requestPermission() async {
    await _init();
    final android = _plugin
        .resolvePlatformSpecificImplementation<
          AndroidFlutterLocalNotificationsPlugin
        >();
    return await android?.requestNotificationsPermission() ?? true;
  }

  @override
  Future<int?> launchedForStop() async {
    await _init();
    final details = await _plugin.getNotificationAppLaunchDetails();
    if (details?.didNotificationLaunchApp != true) return null;
    return stopFromPayload(details!.notificationResponse?.payload);
  }
}

/// What should be scheduled for the active trip, read from the database.
/// Empty when reminders are off or there is no active trip.
Future<List<CheckInReminder>> remindersDue(
  AppDatabase db, {
  required bool enabled,
  DateTime? now,
}) async {
  if (!enabled) return const [];
  final trip = await (db.select(
    db.trips,
  )..where((t) => t.isActive.equals(true))).getSingleOrNull();
  if (trip == null) return const [];

  final legs = await (db.select(
    db.legs,
  )..where((l) => l.tripId.equals(trip.id))).get();
  final stops = await (db.select(
    db.stops,
  )..where((s) => s.tripId.equals(trip.id))).get();
  final arrivals = await (db.select(db.timelineEntries)..where(
        (t) => t.tripId.equals(trip.id) & t.kind.equals('arrival'),
      ))
      .get();
  final trusted = await (db.select(
    db.trustedContacts,
  )..where((t) => t.tripId.isNull() | t.tripId.equals(trip.id))).get();

  final checkIns = <int, DateTime>{};
  for (final a in arrivals) {
    final s = a.stopId;
    if (s == null) continue;
    final prev = checkIns[s];
    if (prev == null || a.occurredAt.isAfter(prev)) checkIns[s] = a.occurredAt;
  }

  return checkInReminders(
    legs: legs,
    stopNames: {for (final s in stops) s.id: s.name},
    checkIns: checkIns,
    now: now ?? DateTime.now(),
    bufferMinutes: reminderBuffer(trusted),
  );
}
