// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'contacts_dao.dart';

// ignore_for_file: type=lint
mixin _$ContactsDaoMixin on DatabaseAccessor<AppDatabase> {
  $TripsTable get trips => attachedDatabase.trips;
  $StopsTable get stops => attachedDatabase.stops;
  $ImportBatchesTable get importBatches => attachedDatabase.importBatches;
  $ContactsTable get contacts => attachedDatabase.contacts;
  $CallLogsTable get callLogs => attachedDatabase.callLogs;
  $EmergencyHelplinesTable get emergencyHelplines =>
      attachedDatabase.emergencyHelplines;
  ContactsDaoManager get managers => ContactsDaoManager(this);
}

class ContactsDaoManager {
  final _$ContactsDaoMixin _db;
  ContactsDaoManager(this._db);
  $$TripsTableTableManager get trips =>
      $$TripsTableTableManager(_db.attachedDatabase, _db.trips);
  $$StopsTableTableManager get stops =>
      $$StopsTableTableManager(_db.attachedDatabase, _db.stops);
  $$ImportBatchesTableTableManager get importBatches =>
      $$ImportBatchesTableTableManager(_db.attachedDatabase, _db.importBatches);
  $$ContactsTableTableManager get contacts =>
      $$ContactsTableTableManager(_db.attachedDatabase, _db.contacts);
  $$CallLogsTableTableManager get callLogs =>
      $$CallLogsTableTableManager(_db.attachedDatabase, _db.callLogs);
  $$EmergencyHelplinesTableTableManager get emergencyHelplines =>
      $$EmergencyHelplinesTableTableManager(
        _db.attachedDatabase,
        _db.emergencyHelplines,
      );
}
