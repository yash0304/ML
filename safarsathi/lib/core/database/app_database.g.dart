// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'app_database.dart';

// ignore_for_file: type=lint
class $TripsTable extends Trips with TableInfo<$TripsTable, Trip> {
  @override
  final GeneratedDatabase attachedDatabase;
  final String? _alias;
  $TripsTable(this.attachedDatabase, [this._alias]);
  static const VerificationMeta _idMeta = const VerificationMeta('id');
  @override
  late final GeneratedColumn<int> id = GeneratedColumn<int>(
    'id',
    aliasedName,
    false,
    hasAutoIncrement: true,
    type: DriftSqlType.int,
    requiredDuringInsert: false,
    defaultConstraints: GeneratedColumn.constraintIsAlways(
      'PRIMARY KEY AUTOINCREMENT',
    ),
  );
  static const VerificationMeta _nameMeta = const VerificationMeta('name');
  @override
  late final GeneratedColumn<String> name = GeneratedColumn<String>(
    'name',
    aliasedName,
    false,
    additionalChecks: GeneratedColumn.checkTextLength(
      minTextLength: 1,
      maxTextLength: 120,
    ),
    type: DriftSqlType.string,
    requiredDuringInsert: true,
  );
  static const VerificationMeta _startDateMeta = const VerificationMeta(
    'startDate',
  );
  @override
  late final GeneratedColumn<DateTime> startDate = GeneratedColumn<DateTime>(
    'start_date',
    aliasedName,
    true,
    type: DriftSqlType.dateTime,
    requiredDuringInsert: false,
  );
  static const VerificationMeta _endDateMeta = const VerificationMeta(
    'endDate',
  );
  @override
  late final GeneratedColumn<DateTime> endDate = GeneratedColumn<DateTime>(
    'end_date',
    aliasedName,
    true,
    type: DriftSqlType.dateTime,
    requiredDuringInsert: false,
  );
  static const VerificationMeta _baseCurrencyMeta = const VerificationMeta(
    'baseCurrency',
  );
  @override
  late final GeneratedColumn<String> baseCurrency = GeneratedColumn<String>(
    'base_currency',
    aliasedName,
    false,
    additionalChecks: GeneratedColumn.checkTextLength(maxTextLength: 3),
    type: DriftSqlType.string,
    requiredDuringInsert: false,
    defaultValue: const Constant('INR'),
  );
  static const VerificationMeta _isActiveMeta = const VerificationMeta(
    'isActive',
  );
  @override
  late final GeneratedColumn<bool> isActive = GeneratedColumn<bool>(
    'is_active',
    aliasedName,
    false,
    type: DriftSqlType.bool,
    requiredDuringInsert: false,
    defaultConstraints: GeneratedColumn.constraintIsAlways(
      'CHECK ("is_active" IN (0, 1))',
    ),
    defaultValue: const Constant(false),
  );
  static const VerificationMeta _createdAtMeta = const VerificationMeta(
    'createdAt',
  );
  @override
  late final GeneratedColumn<DateTime> createdAt = GeneratedColumn<DateTime>(
    'created_at',
    aliasedName,
    false,
    type: DriftSqlType.dateTime,
    requiredDuringInsert: false,
    defaultValue: currentDateAndTime,
  );
  @override
  List<GeneratedColumn> get $columns => [
    id,
    name,
    startDate,
    endDate,
    baseCurrency,
    isActive,
    createdAt,
  ];
  @override
  String get aliasedName => _alias ?? actualTableName;
  @override
  String get actualTableName => $name;
  static const String $name = 'trips';
  @override
  VerificationContext validateIntegrity(
    Insertable<Trip> instance, {
    bool isInserting = false,
  }) {
    final context = VerificationContext();
    final data = instance.toColumns(true);
    if (data.containsKey('id')) {
      context.handle(_idMeta, id.isAcceptableOrUnknown(data['id']!, _idMeta));
    }
    if (data.containsKey('name')) {
      context.handle(
        _nameMeta,
        name.isAcceptableOrUnknown(data['name']!, _nameMeta),
      );
    } else if (isInserting) {
      context.missing(_nameMeta);
    }
    if (data.containsKey('start_date')) {
      context.handle(
        _startDateMeta,
        startDate.isAcceptableOrUnknown(data['start_date']!, _startDateMeta),
      );
    }
    if (data.containsKey('end_date')) {
      context.handle(
        _endDateMeta,
        endDate.isAcceptableOrUnknown(data['end_date']!, _endDateMeta),
      );
    }
    if (data.containsKey('base_currency')) {
      context.handle(
        _baseCurrencyMeta,
        baseCurrency.isAcceptableOrUnknown(
          data['base_currency']!,
          _baseCurrencyMeta,
        ),
      );
    }
    if (data.containsKey('is_active')) {
      context.handle(
        _isActiveMeta,
        isActive.isAcceptableOrUnknown(data['is_active']!, _isActiveMeta),
      );
    }
    if (data.containsKey('created_at')) {
      context.handle(
        _createdAtMeta,
        createdAt.isAcceptableOrUnknown(data['created_at']!, _createdAtMeta),
      );
    }
    return context;
  }

  @override
  Set<GeneratedColumn> get $primaryKey => {id};
  @override
  Trip map(Map<String, dynamic> data, {String? tablePrefix}) {
    final effectivePrefix = tablePrefix != null ? '$tablePrefix.' : '';
    return Trip(
      id: attachedDatabase.typeMapping.read(
        DriftSqlType.int,
        data['${effectivePrefix}id'],
      )!,
      name: attachedDatabase.typeMapping.read(
        DriftSqlType.string,
        data['${effectivePrefix}name'],
      )!,
      startDate: attachedDatabase.typeMapping.read(
        DriftSqlType.dateTime,
        data['${effectivePrefix}start_date'],
      ),
      endDate: attachedDatabase.typeMapping.read(
        DriftSqlType.dateTime,
        data['${effectivePrefix}end_date'],
      ),
      baseCurrency: attachedDatabase.typeMapping.read(
        DriftSqlType.string,
        data['${effectivePrefix}base_currency'],
      )!,
      isActive: attachedDatabase.typeMapping.read(
        DriftSqlType.bool,
        data['${effectivePrefix}is_active'],
      )!,
      createdAt: attachedDatabase.typeMapping.read(
        DriftSqlType.dateTime,
        data['${effectivePrefix}created_at'],
      )!,
    );
  }

  @override
  $TripsTable createAlias(String alias) {
    return $TripsTable(attachedDatabase, alias);
  }
}

class Trip extends DataClass implements Insertable<Trip> {
  final int id;
  final String name;
  final DateTime? startDate;
  final DateTime? endDate;

  /// Base currency for the ledger. Per-expense currency can differ; the rate
  /// is a manual snapshot taken at setup.
  final String baseCurrency;
  final bool isActive;
  final DateTime createdAt;
  const Trip({
    required this.id,
    required this.name,
    this.startDate,
    this.endDate,
    required this.baseCurrency,
    required this.isActive,
    required this.createdAt,
  });
  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    map['id'] = Variable<int>(id);
    map['name'] = Variable<String>(name);
    if (!nullToAbsent || startDate != null) {
      map['start_date'] = Variable<DateTime>(startDate);
    }
    if (!nullToAbsent || endDate != null) {
      map['end_date'] = Variable<DateTime>(endDate);
    }
    map['base_currency'] = Variable<String>(baseCurrency);
    map['is_active'] = Variable<bool>(isActive);
    map['created_at'] = Variable<DateTime>(createdAt);
    return map;
  }

  TripsCompanion toCompanion(bool nullToAbsent) {
    return TripsCompanion(
      id: Value(id),
      name: Value(name),
      startDate: startDate == null && nullToAbsent
          ? const Value.absent()
          : Value(startDate),
      endDate: endDate == null && nullToAbsent
          ? const Value.absent()
          : Value(endDate),
      baseCurrency: Value(baseCurrency),
      isActive: Value(isActive),
      createdAt: Value(createdAt),
    );
  }

  factory Trip.fromJson(
    Map<String, dynamic> json, {
    ValueSerializer? serializer,
  }) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return Trip(
      id: serializer.fromJson<int>(json['id']),
      name: serializer.fromJson<String>(json['name']),
      startDate: serializer.fromJson<DateTime?>(json['startDate']),
      endDate: serializer.fromJson<DateTime?>(json['endDate']),
      baseCurrency: serializer.fromJson<String>(json['baseCurrency']),
      isActive: serializer.fromJson<bool>(json['isActive']),
      createdAt: serializer.fromJson<DateTime>(json['createdAt']),
    );
  }
  @override
  Map<String, dynamic> toJson({ValueSerializer? serializer}) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return <String, dynamic>{
      'id': serializer.toJson<int>(id),
      'name': serializer.toJson<String>(name),
      'startDate': serializer.toJson<DateTime?>(startDate),
      'endDate': serializer.toJson<DateTime?>(endDate),
      'baseCurrency': serializer.toJson<String>(baseCurrency),
      'isActive': serializer.toJson<bool>(isActive),
      'createdAt': serializer.toJson<DateTime>(createdAt),
    };
  }

  Trip copyWith({
    int? id,
    String? name,
    Value<DateTime?> startDate = const Value.absent(),
    Value<DateTime?> endDate = const Value.absent(),
    String? baseCurrency,
    bool? isActive,
    DateTime? createdAt,
  }) => Trip(
    id: id ?? this.id,
    name: name ?? this.name,
    startDate: startDate.present ? startDate.value : this.startDate,
    endDate: endDate.present ? endDate.value : this.endDate,
    baseCurrency: baseCurrency ?? this.baseCurrency,
    isActive: isActive ?? this.isActive,
    createdAt: createdAt ?? this.createdAt,
  );
  Trip copyWithCompanion(TripsCompanion data) {
    return Trip(
      id: data.id.present ? data.id.value : this.id,
      name: data.name.present ? data.name.value : this.name,
      startDate: data.startDate.present ? data.startDate.value : this.startDate,
      endDate: data.endDate.present ? data.endDate.value : this.endDate,
      baseCurrency: data.baseCurrency.present
          ? data.baseCurrency.value
          : this.baseCurrency,
      isActive: data.isActive.present ? data.isActive.value : this.isActive,
      createdAt: data.createdAt.present ? data.createdAt.value : this.createdAt,
    );
  }

  @override
  String toString() {
    return (StringBuffer('Trip(')
          ..write('id: $id, ')
          ..write('name: $name, ')
          ..write('startDate: $startDate, ')
          ..write('endDate: $endDate, ')
          ..write('baseCurrency: $baseCurrency, ')
          ..write('isActive: $isActive, ')
          ..write('createdAt: $createdAt')
          ..write(')'))
        .toString();
  }

  @override
  int get hashCode => Object.hash(
    id,
    name,
    startDate,
    endDate,
    baseCurrency,
    isActive,
    createdAt,
  );
  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      (other is Trip &&
          other.id == this.id &&
          other.name == this.name &&
          other.startDate == this.startDate &&
          other.endDate == this.endDate &&
          other.baseCurrency == this.baseCurrency &&
          other.isActive == this.isActive &&
          other.createdAt == this.createdAt);
}

class TripsCompanion extends UpdateCompanion<Trip> {
  final Value<int> id;
  final Value<String> name;
  final Value<DateTime?> startDate;
  final Value<DateTime?> endDate;
  final Value<String> baseCurrency;
  final Value<bool> isActive;
  final Value<DateTime> createdAt;
  const TripsCompanion({
    this.id = const Value.absent(),
    this.name = const Value.absent(),
    this.startDate = const Value.absent(),
    this.endDate = const Value.absent(),
    this.baseCurrency = const Value.absent(),
    this.isActive = const Value.absent(),
    this.createdAt = const Value.absent(),
  });
  TripsCompanion.insert({
    this.id = const Value.absent(),
    required String name,
    this.startDate = const Value.absent(),
    this.endDate = const Value.absent(),
    this.baseCurrency = const Value.absent(),
    this.isActive = const Value.absent(),
    this.createdAt = const Value.absent(),
  }) : name = Value(name);
  static Insertable<Trip> custom({
    Expression<int>? id,
    Expression<String>? name,
    Expression<DateTime>? startDate,
    Expression<DateTime>? endDate,
    Expression<String>? baseCurrency,
    Expression<bool>? isActive,
    Expression<DateTime>? createdAt,
  }) {
    return RawValuesInsertable({
      if (id != null) 'id': id,
      if (name != null) 'name': name,
      if (startDate != null) 'start_date': startDate,
      if (endDate != null) 'end_date': endDate,
      if (baseCurrency != null) 'base_currency': baseCurrency,
      if (isActive != null) 'is_active': isActive,
      if (createdAt != null) 'created_at': createdAt,
    });
  }

  TripsCompanion copyWith({
    Value<int>? id,
    Value<String>? name,
    Value<DateTime?>? startDate,
    Value<DateTime?>? endDate,
    Value<String>? baseCurrency,
    Value<bool>? isActive,
    Value<DateTime>? createdAt,
  }) {
    return TripsCompanion(
      id: id ?? this.id,
      name: name ?? this.name,
      startDate: startDate ?? this.startDate,
      endDate: endDate ?? this.endDate,
      baseCurrency: baseCurrency ?? this.baseCurrency,
      isActive: isActive ?? this.isActive,
      createdAt: createdAt ?? this.createdAt,
    );
  }

  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    if (id.present) {
      map['id'] = Variable<int>(id.value);
    }
    if (name.present) {
      map['name'] = Variable<String>(name.value);
    }
    if (startDate.present) {
      map['start_date'] = Variable<DateTime>(startDate.value);
    }
    if (endDate.present) {
      map['end_date'] = Variable<DateTime>(endDate.value);
    }
    if (baseCurrency.present) {
      map['base_currency'] = Variable<String>(baseCurrency.value);
    }
    if (isActive.present) {
      map['is_active'] = Variable<bool>(isActive.value);
    }
    if (createdAt.present) {
      map['created_at'] = Variable<DateTime>(createdAt.value);
    }
    return map;
  }

  @override
  String toString() {
    return (StringBuffer('TripsCompanion(')
          ..write('id: $id, ')
          ..write('name: $name, ')
          ..write('startDate: $startDate, ')
          ..write('endDate: $endDate, ')
          ..write('baseCurrency: $baseCurrency, ')
          ..write('isActive: $isActive, ')
          ..write('createdAt: $createdAt')
          ..write(')'))
        .toString();
  }
}

class $StopsTable extends Stops with TableInfo<$StopsTable, Stop> {
  @override
  final GeneratedDatabase attachedDatabase;
  final String? _alias;
  $StopsTable(this.attachedDatabase, [this._alias]);
  static const VerificationMeta _idMeta = const VerificationMeta('id');
  @override
  late final GeneratedColumn<int> id = GeneratedColumn<int>(
    'id',
    aliasedName,
    false,
    hasAutoIncrement: true,
    type: DriftSqlType.int,
    requiredDuringInsert: false,
    defaultConstraints: GeneratedColumn.constraintIsAlways(
      'PRIMARY KEY AUTOINCREMENT',
    ),
  );
  static const VerificationMeta _tripIdMeta = const VerificationMeta('tripId');
  @override
  late final GeneratedColumn<int> tripId = GeneratedColumn<int>(
    'trip_id',
    aliasedName,
    false,
    type: DriftSqlType.int,
    requiredDuringInsert: true,
    defaultConstraints: GeneratedColumn.constraintIsAlways(
      'REFERENCES trips (id) ON DELETE CASCADE',
    ),
  );
  static const VerificationMeta _nameMeta = const VerificationMeta('name');
  @override
  late final GeneratedColumn<String> name = GeneratedColumn<String>(
    'name',
    aliasedName,
    false,
    type: DriftSqlType.string,
    requiredDuringInsert: true,
  );
  static const VerificationMeta _sequenceOrderMeta = const VerificationMeta(
    'sequenceOrder',
  );
  @override
  late final GeneratedColumn<int> sequenceOrder = GeneratedColumn<int>(
    'sequence_order',
    aliasedName,
    false,
    type: DriftSqlType.int,
    requiredDuringInsert: true,
  );
  static const VerificationMeta _countryCodeMeta = const VerificationMeta(
    'countryCode',
  );
  @override
  late final GeneratedColumn<String> countryCode = GeneratedColumn<String>(
    'country_code',
    aliasedName,
    false,
    additionalChecks: GeneratedColumn.checkTextLength(maxTextLength: 2),
    type: DriftSqlType.string,
    requiredDuringInsert: true,
  );
  static const VerificationMeta _arrivalDateMeta = const VerificationMeta(
    'arrivalDate',
  );
  @override
  late final GeneratedColumn<DateTime> arrivalDate = GeneratedColumn<DateTime>(
    'arrival_date',
    aliasedName,
    true,
    type: DriftSqlType.dateTime,
    requiredDuringInsert: false,
  );
  static const VerificationMeta _departureDateMeta = const VerificationMeta(
    'departureDate',
  );
  @override
  late final GeneratedColumn<DateTime> departureDate =
      GeneratedColumn<DateTime>(
        'departure_date',
        aliasedName,
        true,
        type: DriftSqlType.dateTime,
        requiredDuringInsert: false,
      );
  static const VerificationMeta _nightsMeta = const VerificationMeta('nights');
  @override
  late final GeneratedColumn<int> nights = GeneratedColumn<int>(
    'nights',
    aliasedName,
    false,
    type: DriftSqlType.int,
    requiredDuringInsert: false,
    defaultValue: const Constant(0),
  );
  static const VerificationMeta _activityTagsMeta = const VerificationMeta(
    'activityTags',
  );
  @override
  late final GeneratedColumn<String> activityTags = GeneratedColumn<String>(
    'activity_tags',
    aliasedName,
    false,
    type: DriftSqlType.string,
    requiredDuringInsert: false,
    defaultValue: const Constant(''),
  );
  static const VerificationMeta _latMeta = const VerificationMeta('lat');
  @override
  late final GeneratedColumn<double> lat = GeneratedColumn<double>(
    'lat',
    aliasedName,
    true,
    type: DriftSqlType.double,
    requiredDuringInsert: false,
  );
  static const VerificationMeta _lonMeta = const VerificationMeta('lon');
  @override
  late final GeneratedColumn<double> lon = GeneratedColumn<double>(
    'lon',
    aliasedName,
    true,
    type: DriftSqlType.double,
    requiredDuringInsert: false,
  );
  static const VerificationMeta _noteMeta = const VerificationMeta('note');
  @override
  late final GeneratedColumn<String> note = GeneratedColumn<String>(
    'note',
    aliasedName,
    true,
    type: DriftSqlType.string,
    requiredDuringInsert: false,
  );
  @override
  List<GeneratedColumn> get $columns => [
    id,
    tripId,
    name,
    sequenceOrder,
    countryCode,
    arrivalDate,
    departureDate,
    nights,
    activityTags,
    lat,
    lon,
    note,
  ];
  @override
  String get aliasedName => _alias ?? actualTableName;
  @override
  String get actualTableName => $name;
  static const String $name = 'stops';
  @override
  VerificationContext validateIntegrity(
    Insertable<Stop> instance, {
    bool isInserting = false,
  }) {
    final context = VerificationContext();
    final data = instance.toColumns(true);
    if (data.containsKey('id')) {
      context.handle(_idMeta, id.isAcceptableOrUnknown(data['id']!, _idMeta));
    }
    if (data.containsKey('trip_id')) {
      context.handle(
        _tripIdMeta,
        tripId.isAcceptableOrUnknown(data['trip_id']!, _tripIdMeta),
      );
    } else if (isInserting) {
      context.missing(_tripIdMeta);
    }
    if (data.containsKey('name')) {
      context.handle(
        _nameMeta,
        name.isAcceptableOrUnknown(data['name']!, _nameMeta),
      );
    } else if (isInserting) {
      context.missing(_nameMeta);
    }
    if (data.containsKey('sequence_order')) {
      context.handle(
        _sequenceOrderMeta,
        sequenceOrder.isAcceptableOrUnknown(
          data['sequence_order']!,
          _sequenceOrderMeta,
        ),
      );
    } else if (isInserting) {
      context.missing(_sequenceOrderMeta);
    }
    if (data.containsKey('country_code')) {
      context.handle(
        _countryCodeMeta,
        countryCode.isAcceptableOrUnknown(
          data['country_code']!,
          _countryCodeMeta,
        ),
      );
    } else if (isInserting) {
      context.missing(_countryCodeMeta);
    }
    if (data.containsKey('arrival_date')) {
      context.handle(
        _arrivalDateMeta,
        arrivalDate.isAcceptableOrUnknown(
          data['arrival_date']!,
          _arrivalDateMeta,
        ),
      );
    }
    if (data.containsKey('departure_date')) {
      context.handle(
        _departureDateMeta,
        departureDate.isAcceptableOrUnknown(
          data['departure_date']!,
          _departureDateMeta,
        ),
      );
    }
    if (data.containsKey('nights')) {
      context.handle(
        _nightsMeta,
        nights.isAcceptableOrUnknown(data['nights']!, _nightsMeta),
      );
    }
    if (data.containsKey('activity_tags')) {
      context.handle(
        _activityTagsMeta,
        activityTags.isAcceptableOrUnknown(
          data['activity_tags']!,
          _activityTagsMeta,
        ),
      );
    }
    if (data.containsKey('lat')) {
      context.handle(
        _latMeta,
        lat.isAcceptableOrUnknown(data['lat']!, _latMeta),
      );
    }
    if (data.containsKey('lon')) {
      context.handle(
        _lonMeta,
        lon.isAcceptableOrUnknown(data['lon']!, _lonMeta),
      );
    }
    if (data.containsKey('note')) {
      context.handle(
        _noteMeta,
        note.isAcceptableOrUnknown(data['note']!, _noteMeta),
      );
    }
    return context;
  }

  @override
  Set<GeneratedColumn> get $primaryKey => {id};
  @override
  Stop map(Map<String, dynamic> data, {String? tablePrefix}) {
    final effectivePrefix = tablePrefix != null ? '$tablePrefix.' : '';
    return Stop(
      id: attachedDatabase.typeMapping.read(
        DriftSqlType.int,
        data['${effectivePrefix}id'],
      )!,
      tripId: attachedDatabase.typeMapping.read(
        DriftSqlType.int,
        data['${effectivePrefix}trip_id'],
      )!,
      name: attachedDatabase.typeMapping.read(
        DriftSqlType.string,
        data['${effectivePrefix}name'],
      )!,
      sequenceOrder: attachedDatabase.typeMapping.read(
        DriftSqlType.int,
        data['${effectivePrefix}sequence_order'],
      )!,
      countryCode: attachedDatabase.typeMapping.read(
        DriftSqlType.string,
        data['${effectivePrefix}country_code'],
      )!,
      arrivalDate: attachedDatabase.typeMapping.read(
        DriftSqlType.dateTime,
        data['${effectivePrefix}arrival_date'],
      ),
      departureDate: attachedDatabase.typeMapping.read(
        DriftSqlType.dateTime,
        data['${effectivePrefix}departure_date'],
      ),
      nights: attachedDatabase.typeMapping.read(
        DriftSqlType.int,
        data['${effectivePrefix}nights'],
      )!,
      activityTags: attachedDatabase.typeMapping.read(
        DriftSqlType.string,
        data['${effectivePrefix}activity_tags'],
      )!,
      lat: attachedDatabase.typeMapping.read(
        DriftSqlType.double,
        data['${effectivePrefix}lat'],
      ),
      lon: attachedDatabase.typeMapping.read(
        DriftSqlType.double,
        data['${effectivePrefix}lon'],
      ),
      note: attachedDatabase.typeMapping.read(
        DriftSqlType.string,
        data['${effectivePrefix}note'],
      ),
    );
  }

  @override
  $StopsTable createAlias(String alias) {
    return $StopsTable(attachedDatabase, alias);
  }
}

class Stop extends DataClass implements Insertable<Stop> {
  final int id;
  final int tripId;
  final String name;
  final int sequenceOrder;

  /// Country lives on the STOP, not the Trip. A German trip crosses into
  /// Austria mid-itinerary, and emergency numbers, currency and language all
  /// switch at that boundary rather than at the trip boundary.
  final String countryCode;
  final DateTime? arrivalDate;
  final DateTime? departureDate;
  final int nights;

  /// Comma-separated: trek, caves, rain, homestay. Drives checklist
  /// generation at #29.
  final String activityTags;
  final double? lat;
  final double? lon;
  final String? note;
  const Stop({
    required this.id,
    required this.tripId,
    required this.name,
    required this.sequenceOrder,
    required this.countryCode,
    this.arrivalDate,
    this.departureDate,
    required this.nights,
    required this.activityTags,
    this.lat,
    this.lon,
    this.note,
  });
  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    map['id'] = Variable<int>(id);
    map['trip_id'] = Variable<int>(tripId);
    map['name'] = Variable<String>(name);
    map['sequence_order'] = Variable<int>(sequenceOrder);
    map['country_code'] = Variable<String>(countryCode);
    if (!nullToAbsent || arrivalDate != null) {
      map['arrival_date'] = Variable<DateTime>(arrivalDate);
    }
    if (!nullToAbsent || departureDate != null) {
      map['departure_date'] = Variable<DateTime>(departureDate);
    }
    map['nights'] = Variable<int>(nights);
    map['activity_tags'] = Variable<String>(activityTags);
    if (!nullToAbsent || lat != null) {
      map['lat'] = Variable<double>(lat);
    }
    if (!nullToAbsent || lon != null) {
      map['lon'] = Variable<double>(lon);
    }
    if (!nullToAbsent || note != null) {
      map['note'] = Variable<String>(note);
    }
    return map;
  }

  StopsCompanion toCompanion(bool nullToAbsent) {
    return StopsCompanion(
      id: Value(id),
      tripId: Value(tripId),
      name: Value(name),
      sequenceOrder: Value(sequenceOrder),
      countryCode: Value(countryCode),
      arrivalDate: arrivalDate == null && nullToAbsent
          ? const Value.absent()
          : Value(arrivalDate),
      departureDate: departureDate == null && nullToAbsent
          ? const Value.absent()
          : Value(departureDate),
      nights: Value(nights),
      activityTags: Value(activityTags),
      lat: lat == null && nullToAbsent ? const Value.absent() : Value(lat),
      lon: lon == null && nullToAbsent ? const Value.absent() : Value(lon),
      note: note == null && nullToAbsent ? const Value.absent() : Value(note),
    );
  }

  factory Stop.fromJson(
    Map<String, dynamic> json, {
    ValueSerializer? serializer,
  }) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return Stop(
      id: serializer.fromJson<int>(json['id']),
      tripId: serializer.fromJson<int>(json['tripId']),
      name: serializer.fromJson<String>(json['name']),
      sequenceOrder: serializer.fromJson<int>(json['sequenceOrder']),
      countryCode: serializer.fromJson<String>(json['countryCode']),
      arrivalDate: serializer.fromJson<DateTime?>(json['arrivalDate']),
      departureDate: serializer.fromJson<DateTime?>(json['departureDate']),
      nights: serializer.fromJson<int>(json['nights']),
      activityTags: serializer.fromJson<String>(json['activityTags']),
      lat: serializer.fromJson<double?>(json['lat']),
      lon: serializer.fromJson<double?>(json['lon']),
      note: serializer.fromJson<String?>(json['note']),
    );
  }
  @override
  Map<String, dynamic> toJson({ValueSerializer? serializer}) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return <String, dynamic>{
      'id': serializer.toJson<int>(id),
      'tripId': serializer.toJson<int>(tripId),
      'name': serializer.toJson<String>(name),
      'sequenceOrder': serializer.toJson<int>(sequenceOrder),
      'countryCode': serializer.toJson<String>(countryCode),
      'arrivalDate': serializer.toJson<DateTime?>(arrivalDate),
      'departureDate': serializer.toJson<DateTime?>(departureDate),
      'nights': serializer.toJson<int>(nights),
      'activityTags': serializer.toJson<String>(activityTags),
      'lat': serializer.toJson<double?>(lat),
      'lon': serializer.toJson<double?>(lon),
      'note': serializer.toJson<String?>(note),
    };
  }

  Stop copyWith({
    int? id,
    int? tripId,
    String? name,
    int? sequenceOrder,
    String? countryCode,
    Value<DateTime?> arrivalDate = const Value.absent(),
    Value<DateTime?> departureDate = const Value.absent(),
    int? nights,
    String? activityTags,
    Value<double?> lat = const Value.absent(),
    Value<double?> lon = const Value.absent(),
    Value<String?> note = const Value.absent(),
  }) => Stop(
    id: id ?? this.id,
    tripId: tripId ?? this.tripId,
    name: name ?? this.name,
    sequenceOrder: sequenceOrder ?? this.sequenceOrder,
    countryCode: countryCode ?? this.countryCode,
    arrivalDate: arrivalDate.present ? arrivalDate.value : this.arrivalDate,
    departureDate: departureDate.present
        ? departureDate.value
        : this.departureDate,
    nights: nights ?? this.nights,
    activityTags: activityTags ?? this.activityTags,
    lat: lat.present ? lat.value : this.lat,
    lon: lon.present ? lon.value : this.lon,
    note: note.present ? note.value : this.note,
  );
  Stop copyWithCompanion(StopsCompanion data) {
    return Stop(
      id: data.id.present ? data.id.value : this.id,
      tripId: data.tripId.present ? data.tripId.value : this.tripId,
      name: data.name.present ? data.name.value : this.name,
      sequenceOrder: data.sequenceOrder.present
          ? data.sequenceOrder.value
          : this.sequenceOrder,
      countryCode: data.countryCode.present
          ? data.countryCode.value
          : this.countryCode,
      arrivalDate: data.arrivalDate.present
          ? data.arrivalDate.value
          : this.arrivalDate,
      departureDate: data.departureDate.present
          ? data.departureDate.value
          : this.departureDate,
      nights: data.nights.present ? data.nights.value : this.nights,
      activityTags: data.activityTags.present
          ? data.activityTags.value
          : this.activityTags,
      lat: data.lat.present ? data.lat.value : this.lat,
      lon: data.lon.present ? data.lon.value : this.lon,
      note: data.note.present ? data.note.value : this.note,
    );
  }

  @override
  String toString() {
    return (StringBuffer('Stop(')
          ..write('id: $id, ')
          ..write('tripId: $tripId, ')
          ..write('name: $name, ')
          ..write('sequenceOrder: $sequenceOrder, ')
          ..write('countryCode: $countryCode, ')
          ..write('arrivalDate: $arrivalDate, ')
          ..write('departureDate: $departureDate, ')
          ..write('nights: $nights, ')
          ..write('activityTags: $activityTags, ')
          ..write('lat: $lat, ')
          ..write('lon: $lon, ')
          ..write('note: $note')
          ..write(')'))
        .toString();
  }

  @override
  int get hashCode => Object.hash(
    id,
    tripId,
    name,
    sequenceOrder,
    countryCode,
    arrivalDate,
    departureDate,
    nights,
    activityTags,
    lat,
    lon,
    note,
  );
  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      (other is Stop &&
          other.id == this.id &&
          other.tripId == this.tripId &&
          other.name == this.name &&
          other.sequenceOrder == this.sequenceOrder &&
          other.countryCode == this.countryCode &&
          other.arrivalDate == this.arrivalDate &&
          other.departureDate == this.departureDate &&
          other.nights == this.nights &&
          other.activityTags == this.activityTags &&
          other.lat == this.lat &&
          other.lon == this.lon &&
          other.note == this.note);
}

class StopsCompanion extends UpdateCompanion<Stop> {
  final Value<int> id;
  final Value<int> tripId;
  final Value<String> name;
  final Value<int> sequenceOrder;
  final Value<String> countryCode;
  final Value<DateTime?> arrivalDate;
  final Value<DateTime?> departureDate;
  final Value<int> nights;
  final Value<String> activityTags;
  final Value<double?> lat;
  final Value<double?> lon;
  final Value<String?> note;
  const StopsCompanion({
    this.id = const Value.absent(),
    this.tripId = const Value.absent(),
    this.name = const Value.absent(),
    this.sequenceOrder = const Value.absent(),
    this.countryCode = const Value.absent(),
    this.arrivalDate = const Value.absent(),
    this.departureDate = const Value.absent(),
    this.nights = const Value.absent(),
    this.activityTags = const Value.absent(),
    this.lat = const Value.absent(),
    this.lon = const Value.absent(),
    this.note = const Value.absent(),
  });
  StopsCompanion.insert({
    this.id = const Value.absent(),
    required int tripId,
    required String name,
    required int sequenceOrder,
    required String countryCode,
    this.arrivalDate = const Value.absent(),
    this.departureDate = const Value.absent(),
    this.nights = const Value.absent(),
    this.activityTags = const Value.absent(),
    this.lat = const Value.absent(),
    this.lon = const Value.absent(),
    this.note = const Value.absent(),
  }) : tripId = Value(tripId),
       name = Value(name),
       sequenceOrder = Value(sequenceOrder),
       countryCode = Value(countryCode);
  static Insertable<Stop> custom({
    Expression<int>? id,
    Expression<int>? tripId,
    Expression<String>? name,
    Expression<int>? sequenceOrder,
    Expression<String>? countryCode,
    Expression<DateTime>? arrivalDate,
    Expression<DateTime>? departureDate,
    Expression<int>? nights,
    Expression<String>? activityTags,
    Expression<double>? lat,
    Expression<double>? lon,
    Expression<String>? note,
  }) {
    return RawValuesInsertable({
      if (id != null) 'id': id,
      if (tripId != null) 'trip_id': tripId,
      if (name != null) 'name': name,
      if (sequenceOrder != null) 'sequence_order': sequenceOrder,
      if (countryCode != null) 'country_code': countryCode,
      if (arrivalDate != null) 'arrival_date': arrivalDate,
      if (departureDate != null) 'departure_date': departureDate,
      if (nights != null) 'nights': nights,
      if (activityTags != null) 'activity_tags': activityTags,
      if (lat != null) 'lat': lat,
      if (lon != null) 'lon': lon,
      if (note != null) 'note': note,
    });
  }

  StopsCompanion copyWith({
    Value<int>? id,
    Value<int>? tripId,
    Value<String>? name,
    Value<int>? sequenceOrder,
    Value<String>? countryCode,
    Value<DateTime?>? arrivalDate,
    Value<DateTime?>? departureDate,
    Value<int>? nights,
    Value<String>? activityTags,
    Value<double?>? lat,
    Value<double?>? lon,
    Value<String?>? note,
  }) {
    return StopsCompanion(
      id: id ?? this.id,
      tripId: tripId ?? this.tripId,
      name: name ?? this.name,
      sequenceOrder: sequenceOrder ?? this.sequenceOrder,
      countryCode: countryCode ?? this.countryCode,
      arrivalDate: arrivalDate ?? this.arrivalDate,
      departureDate: departureDate ?? this.departureDate,
      nights: nights ?? this.nights,
      activityTags: activityTags ?? this.activityTags,
      lat: lat ?? this.lat,
      lon: lon ?? this.lon,
      note: note ?? this.note,
    );
  }

  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    if (id.present) {
      map['id'] = Variable<int>(id.value);
    }
    if (tripId.present) {
      map['trip_id'] = Variable<int>(tripId.value);
    }
    if (name.present) {
      map['name'] = Variable<String>(name.value);
    }
    if (sequenceOrder.present) {
      map['sequence_order'] = Variable<int>(sequenceOrder.value);
    }
    if (countryCode.present) {
      map['country_code'] = Variable<String>(countryCode.value);
    }
    if (arrivalDate.present) {
      map['arrival_date'] = Variable<DateTime>(arrivalDate.value);
    }
    if (departureDate.present) {
      map['departure_date'] = Variable<DateTime>(departureDate.value);
    }
    if (nights.present) {
      map['nights'] = Variable<int>(nights.value);
    }
    if (activityTags.present) {
      map['activity_tags'] = Variable<String>(activityTags.value);
    }
    if (lat.present) {
      map['lat'] = Variable<double>(lat.value);
    }
    if (lon.present) {
      map['lon'] = Variable<double>(lon.value);
    }
    if (note.present) {
      map['note'] = Variable<String>(note.value);
    }
    return map;
  }

  @override
  String toString() {
    return (StringBuffer('StopsCompanion(')
          ..write('id: $id, ')
          ..write('tripId: $tripId, ')
          ..write('name: $name, ')
          ..write('sequenceOrder: $sequenceOrder, ')
          ..write('countryCode: $countryCode, ')
          ..write('arrivalDate: $arrivalDate, ')
          ..write('departureDate: $departureDate, ')
          ..write('nights: $nights, ')
          ..write('activityTags: $activityTags, ')
          ..write('lat: $lat, ')
          ..write('lon: $lon, ')
          ..write('note: $note')
          ..write(')'))
        .toString();
  }
}

class $LegsTable extends Legs with TableInfo<$LegsTable, Leg> {
  @override
  final GeneratedDatabase attachedDatabase;
  final String? _alias;
  $LegsTable(this.attachedDatabase, [this._alias]);
  static const VerificationMeta _idMeta = const VerificationMeta('id');
  @override
  late final GeneratedColumn<int> id = GeneratedColumn<int>(
    'id',
    aliasedName,
    false,
    hasAutoIncrement: true,
    type: DriftSqlType.int,
    requiredDuringInsert: false,
    defaultConstraints: GeneratedColumn.constraintIsAlways(
      'PRIMARY KEY AUTOINCREMENT',
    ),
  );
  static const VerificationMeta _tripIdMeta = const VerificationMeta('tripId');
  @override
  late final GeneratedColumn<int> tripId = GeneratedColumn<int>(
    'trip_id',
    aliasedName,
    false,
    type: DriftSqlType.int,
    requiredDuringInsert: true,
    defaultConstraints: GeneratedColumn.constraintIsAlways(
      'REFERENCES trips (id) ON DELETE CASCADE',
    ),
  );
  static const VerificationMeta _fromStopIdMeta = const VerificationMeta(
    'fromStopId',
  );
  @override
  late final GeneratedColumn<int> fromStopId = GeneratedColumn<int>(
    'from_stop_id',
    aliasedName,
    false,
    type: DriftSqlType.int,
    requiredDuringInsert: true,
    defaultConstraints: GeneratedColumn.constraintIsAlways(
      'REFERENCES stops (id) ON DELETE CASCADE',
    ),
  );
  static const VerificationMeta _toStopIdMeta = const VerificationMeta(
    'toStopId',
  );
  @override
  late final GeneratedColumn<int> toStopId = GeneratedColumn<int>(
    'to_stop_id',
    aliasedName,
    false,
    type: DriftSqlType.int,
    requiredDuringInsert: true,
    defaultConstraints: GeneratedColumn.constraintIsAlways(
      'REFERENCES stops (id) ON DELETE CASCADE',
    ),
  );
  static const VerificationMeta _sequenceOrderMeta = const VerificationMeta(
    'sequenceOrder',
  );
  @override
  late final GeneratedColumn<int> sequenceOrder = GeneratedColumn<int>(
    'sequence_order',
    aliasedName,
    false,
    type: DriftSqlType.int,
    requiredDuringInsert: true,
  );
  static const VerificationMeta _modeMeta = const VerificationMeta('mode');
  @override
  late final GeneratedColumn<String> mode = GeneratedColumn<String>(
    'mode',
    aliasedName,
    true,
    type: DriftSqlType.string,
    requiredDuringInsert: false,
  );
  static const VerificationMeta _plannedDepartureMeta = const VerificationMeta(
    'plannedDeparture',
  );
  @override
  late final GeneratedColumn<DateTime> plannedDeparture =
      GeneratedColumn<DateTime>(
        'planned_departure',
        aliasedName,
        true,
        type: DriftSqlType.dateTime,
        requiredDuringInsert: false,
      );
  static const VerificationMeta _plannedArrivalMeta = const VerificationMeta(
    'plannedArrival',
  );
  @override
  late final GeneratedColumn<DateTime> plannedArrival =
      GeneratedColumn<DateTime>(
        'planned_arrival',
        aliasedName,
        true,
        type: DriftSqlType.dateTime,
        requiredDuringInsert: false,
      );
  static const VerificationMeta _isBookedMeta = const VerificationMeta(
    'isBooked',
  );
  @override
  late final GeneratedColumn<bool> isBooked = GeneratedColumn<bool>(
    'is_booked',
    aliasedName,
    false,
    type: DriftSqlType.bool,
    requiredDuringInsert: false,
    defaultConstraints: GeneratedColumn.constraintIsAlways(
      'CHECK ("is_booked" IN (0, 1))',
    ),
    defaultValue: const Constant(false),
  );
  static const VerificationMeta _noteMeta = const VerificationMeta('note');
  @override
  late final GeneratedColumn<String> note = GeneratedColumn<String>(
    'note',
    aliasedName,
    true,
    type: DriftSqlType.string,
    requiredDuringInsert: false,
  );
  static const VerificationMeta _routePolylineMeta = const VerificationMeta(
    'routePolyline',
  );
  @override
  late final GeneratedColumn<String> routePolyline = GeneratedColumn<String>(
    'route_polyline',
    aliasedName,
    true,
    type: DriftSqlType.string,
    requiredDuringInsert: false,
  );
  static const VerificationMeta _distanceKmMeta = const VerificationMeta(
    'distanceKm',
  );
  @override
  late final GeneratedColumn<double> distanceKm = GeneratedColumn<double>(
    'distance_km',
    aliasedName,
    true,
    type: DriftSqlType.double,
    requiredDuringInsert: false,
  );
  static const VerificationMeta _corridorKmMeta = const VerificationMeta(
    'corridorKm',
  );
  @override
  late final GeneratedColumn<double> corridorKm = GeneratedColumn<double>(
    'corridor_km',
    aliasedName,
    false,
    type: DriftSqlType.double,
    requiredDuringInsert: false,
    defaultValue: const Constant(3.0),
  );
  static const VerificationMeta _lastSyncedAtMeta = const VerificationMeta(
    'lastSyncedAt',
  );
  @override
  late final GeneratedColumn<DateTime> lastSyncedAt = GeneratedColumn<DateTime>(
    'last_synced_at',
    aliasedName,
    true,
    type: DriftSqlType.dateTime,
    requiredDuringInsert: false,
  );
  @override
  List<GeneratedColumn> get $columns => [
    id,
    tripId,
    fromStopId,
    toStopId,
    sequenceOrder,
    mode,
    plannedDeparture,
    plannedArrival,
    isBooked,
    note,
    routePolyline,
    distanceKm,
    corridorKm,
    lastSyncedAt,
  ];
  @override
  String get aliasedName => _alias ?? actualTableName;
  @override
  String get actualTableName => $name;
  static const String $name = 'legs';
  @override
  VerificationContext validateIntegrity(
    Insertable<Leg> instance, {
    bool isInserting = false,
  }) {
    final context = VerificationContext();
    final data = instance.toColumns(true);
    if (data.containsKey('id')) {
      context.handle(_idMeta, id.isAcceptableOrUnknown(data['id']!, _idMeta));
    }
    if (data.containsKey('trip_id')) {
      context.handle(
        _tripIdMeta,
        tripId.isAcceptableOrUnknown(data['trip_id']!, _tripIdMeta),
      );
    } else if (isInserting) {
      context.missing(_tripIdMeta);
    }
    if (data.containsKey('from_stop_id')) {
      context.handle(
        _fromStopIdMeta,
        fromStopId.isAcceptableOrUnknown(
          data['from_stop_id']!,
          _fromStopIdMeta,
        ),
      );
    } else if (isInserting) {
      context.missing(_fromStopIdMeta);
    }
    if (data.containsKey('to_stop_id')) {
      context.handle(
        _toStopIdMeta,
        toStopId.isAcceptableOrUnknown(data['to_stop_id']!, _toStopIdMeta),
      );
    } else if (isInserting) {
      context.missing(_toStopIdMeta);
    }
    if (data.containsKey('sequence_order')) {
      context.handle(
        _sequenceOrderMeta,
        sequenceOrder.isAcceptableOrUnknown(
          data['sequence_order']!,
          _sequenceOrderMeta,
        ),
      );
    } else if (isInserting) {
      context.missing(_sequenceOrderMeta);
    }
    if (data.containsKey('mode')) {
      context.handle(
        _modeMeta,
        mode.isAcceptableOrUnknown(data['mode']!, _modeMeta),
      );
    }
    if (data.containsKey('planned_departure')) {
      context.handle(
        _plannedDepartureMeta,
        plannedDeparture.isAcceptableOrUnknown(
          data['planned_departure']!,
          _plannedDepartureMeta,
        ),
      );
    }
    if (data.containsKey('planned_arrival')) {
      context.handle(
        _plannedArrivalMeta,
        plannedArrival.isAcceptableOrUnknown(
          data['planned_arrival']!,
          _plannedArrivalMeta,
        ),
      );
    }
    if (data.containsKey('is_booked')) {
      context.handle(
        _isBookedMeta,
        isBooked.isAcceptableOrUnknown(data['is_booked']!, _isBookedMeta),
      );
    }
    if (data.containsKey('note')) {
      context.handle(
        _noteMeta,
        note.isAcceptableOrUnknown(data['note']!, _noteMeta),
      );
    }
    if (data.containsKey('route_polyline')) {
      context.handle(
        _routePolylineMeta,
        routePolyline.isAcceptableOrUnknown(
          data['route_polyline']!,
          _routePolylineMeta,
        ),
      );
    }
    if (data.containsKey('distance_km')) {
      context.handle(
        _distanceKmMeta,
        distanceKm.isAcceptableOrUnknown(data['distance_km']!, _distanceKmMeta),
      );
    }
    if (data.containsKey('corridor_km')) {
      context.handle(
        _corridorKmMeta,
        corridorKm.isAcceptableOrUnknown(data['corridor_km']!, _corridorKmMeta),
      );
    }
    if (data.containsKey('last_synced_at')) {
      context.handle(
        _lastSyncedAtMeta,
        lastSyncedAt.isAcceptableOrUnknown(
          data['last_synced_at']!,
          _lastSyncedAtMeta,
        ),
      );
    }
    return context;
  }

  @override
  Set<GeneratedColumn> get $primaryKey => {id};
  @override
  Leg map(Map<String, dynamic> data, {String? tablePrefix}) {
    final effectivePrefix = tablePrefix != null ? '$tablePrefix.' : '';
    return Leg(
      id: attachedDatabase.typeMapping.read(
        DriftSqlType.int,
        data['${effectivePrefix}id'],
      )!,
      tripId: attachedDatabase.typeMapping.read(
        DriftSqlType.int,
        data['${effectivePrefix}trip_id'],
      )!,
      fromStopId: attachedDatabase.typeMapping.read(
        DriftSqlType.int,
        data['${effectivePrefix}from_stop_id'],
      )!,
      toStopId: attachedDatabase.typeMapping.read(
        DriftSqlType.int,
        data['${effectivePrefix}to_stop_id'],
      )!,
      sequenceOrder: attachedDatabase.typeMapping.read(
        DriftSqlType.int,
        data['${effectivePrefix}sequence_order'],
      )!,
      mode: attachedDatabase.typeMapping.read(
        DriftSqlType.string,
        data['${effectivePrefix}mode'],
      ),
      plannedDeparture: attachedDatabase.typeMapping.read(
        DriftSqlType.dateTime,
        data['${effectivePrefix}planned_departure'],
      ),
      plannedArrival: attachedDatabase.typeMapping.read(
        DriftSqlType.dateTime,
        data['${effectivePrefix}planned_arrival'],
      ),
      isBooked: attachedDatabase.typeMapping.read(
        DriftSqlType.bool,
        data['${effectivePrefix}is_booked'],
      )!,
      note: attachedDatabase.typeMapping.read(
        DriftSqlType.string,
        data['${effectivePrefix}note'],
      ),
      routePolyline: attachedDatabase.typeMapping.read(
        DriftSqlType.string,
        data['${effectivePrefix}route_polyline'],
      ),
      distanceKm: attachedDatabase.typeMapping.read(
        DriftSqlType.double,
        data['${effectivePrefix}distance_km'],
      ),
      corridorKm: attachedDatabase.typeMapping.read(
        DriftSqlType.double,
        data['${effectivePrefix}corridor_km'],
      )!,
      lastSyncedAt: attachedDatabase.typeMapping.read(
        DriftSqlType.dateTime,
        data['${effectivePrefix}last_synced_at'],
      ),
    );
  }

  @override
  $LegsTable createAlias(String alias) {
    return $LegsTable(attachedDatabase, alias);
  }
}

class Leg extends DataClass implements Insertable<Leg> {
  final int id;
  final int tripId;
  final int fromStopId;
  final int toStopId;
  final int sequenceOrder;

  /// Typed by the user. There is no live schedule lookup and there never will
  /// be offline — see DECISIONS.md.
  final String? mode;
  final DateTime? plannedDeparture;
  final DateTime? plannedArrival;
  final bool isBooked;
  final String? note;

  /// Encoded polyline. The only live routing call in the app, made once at
  /// setup on WiFi.
  final String? routePolyline;
  final double? distanceKm;
  final double corridorKm;
  final DateTime? lastSyncedAt;
  const Leg({
    required this.id,
    required this.tripId,
    required this.fromStopId,
    required this.toStopId,
    required this.sequenceOrder,
    this.mode,
    this.plannedDeparture,
    this.plannedArrival,
    required this.isBooked,
    this.note,
    this.routePolyline,
    this.distanceKm,
    required this.corridorKm,
    this.lastSyncedAt,
  });
  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    map['id'] = Variable<int>(id);
    map['trip_id'] = Variable<int>(tripId);
    map['from_stop_id'] = Variable<int>(fromStopId);
    map['to_stop_id'] = Variable<int>(toStopId);
    map['sequence_order'] = Variable<int>(sequenceOrder);
    if (!nullToAbsent || mode != null) {
      map['mode'] = Variable<String>(mode);
    }
    if (!nullToAbsent || plannedDeparture != null) {
      map['planned_departure'] = Variable<DateTime>(plannedDeparture);
    }
    if (!nullToAbsent || plannedArrival != null) {
      map['planned_arrival'] = Variable<DateTime>(plannedArrival);
    }
    map['is_booked'] = Variable<bool>(isBooked);
    if (!nullToAbsent || note != null) {
      map['note'] = Variable<String>(note);
    }
    if (!nullToAbsent || routePolyline != null) {
      map['route_polyline'] = Variable<String>(routePolyline);
    }
    if (!nullToAbsent || distanceKm != null) {
      map['distance_km'] = Variable<double>(distanceKm);
    }
    map['corridor_km'] = Variable<double>(corridorKm);
    if (!nullToAbsent || lastSyncedAt != null) {
      map['last_synced_at'] = Variable<DateTime>(lastSyncedAt);
    }
    return map;
  }

  LegsCompanion toCompanion(bool nullToAbsent) {
    return LegsCompanion(
      id: Value(id),
      tripId: Value(tripId),
      fromStopId: Value(fromStopId),
      toStopId: Value(toStopId),
      sequenceOrder: Value(sequenceOrder),
      mode: mode == null && nullToAbsent ? const Value.absent() : Value(mode),
      plannedDeparture: plannedDeparture == null && nullToAbsent
          ? const Value.absent()
          : Value(plannedDeparture),
      plannedArrival: plannedArrival == null && nullToAbsent
          ? const Value.absent()
          : Value(plannedArrival),
      isBooked: Value(isBooked),
      note: note == null && nullToAbsent ? const Value.absent() : Value(note),
      routePolyline: routePolyline == null && nullToAbsent
          ? const Value.absent()
          : Value(routePolyline),
      distanceKm: distanceKm == null && nullToAbsent
          ? const Value.absent()
          : Value(distanceKm),
      corridorKm: Value(corridorKm),
      lastSyncedAt: lastSyncedAt == null && nullToAbsent
          ? const Value.absent()
          : Value(lastSyncedAt),
    );
  }

  factory Leg.fromJson(
    Map<String, dynamic> json, {
    ValueSerializer? serializer,
  }) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return Leg(
      id: serializer.fromJson<int>(json['id']),
      tripId: serializer.fromJson<int>(json['tripId']),
      fromStopId: serializer.fromJson<int>(json['fromStopId']),
      toStopId: serializer.fromJson<int>(json['toStopId']),
      sequenceOrder: serializer.fromJson<int>(json['sequenceOrder']),
      mode: serializer.fromJson<String?>(json['mode']),
      plannedDeparture: serializer.fromJson<DateTime?>(
        json['plannedDeparture'],
      ),
      plannedArrival: serializer.fromJson<DateTime?>(json['plannedArrival']),
      isBooked: serializer.fromJson<bool>(json['isBooked']),
      note: serializer.fromJson<String?>(json['note']),
      routePolyline: serializer.fromJson<String?>(json['routePolyline']),
      distanceKm: serializer.fromJson<double?>(json['distanceKm']),
      corridorKm: serializer.fromJson<double>(json['corridorKm']),
      lastSyncedAt: serializer.fromJson<DateTime?>(json['lastSyncedAt']),
    );
  }
  @override
  Map<String, dynamic> toJson({ValueSerializer? serializer}) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return <String, dynamic>{
      'id': serializer.toJson<int>(id),
      'tripId': serializer.toJson<int>(tripId),
      'fromStopId': serializer.toJson<int>(fromStopId),
      'toStopId': serializer.toJson<int>(toStopId),
      'sequenceOrder': serializer.toJson<int>(sequenceOrder),
      'mode': serializer.toJson<String?>(mode),
      'plannedDeparture': serializer.toJson<DateTime?>(plannedDeparture),
      'plannedArrival': serializer.toJson<DateTime?>(plannedArrival),
      'isBooked': serializer.toJson<bool>(isBooked),
      'note': serializer.toJson<String?>(note),
      'routePolyline': serializer.toJson<String?>(routePolyline),
      'distanceKm': serializer.toJson<double?>(distanceKm),
      'corridorKm': serializer.toJson<double>(corridorKm),
      'lastSyncedAt': serializer.toJson<DateTime?>(lastSyncedAt),
    };
  }

  Leg copyWith({
    int? id,
    int? tripId,
    int? fromStopId,
    int? toStopId,
    int? sequenceOrder,
    Value<String?> mode = const Value.absent(),
    Value<DateTime?> plannedDeparture = const Value.absent(),
    Value<DateTime?> plannedArrival = const Value.absent(),
    bool? isBooked,
    Value<String?> note = const Value.absent(),
    Value<String?> routePolyline = const Value.absent(),
    Value<double?> distanceKm = const Value.absent(),
    double? corridorKm,
    Value<DateTime?> lastSyncedAt = const Value.absent(),
  }) => Leg(
    id: id ?? this.id,
    tripId: tripId ?? this.tripId,
    fromStopId: fromStopId ?? this.fromStopId,
    toStopId: toStopId ?? this.toStopId,
    sequenceOrder: sequenceOrder ?? this.sequenceOrder,
    mode: mode.present ? mode.value : this.mode,
    plannedDeparture: plannedDeparture.present
        ? plannedDeparture.value
        : this.plannedDeparture,
    plannedArrival: plannedArrival.present
        ? plannedArrival.value
        : this.plannedArrival,
    isBooked: isBooked ?? this.isBooked,
    note: note.present ? note.value : this.note,
    routePolyline: routePolyline.present
        ? routePolyline.value
        : this.routePolyline,
    distanceKm: distanceKm.present ? distanceKm.value : this.distanceKm,
    corridorKm: corridorKm ?? this.corridorKm,
    lastSyncedAt: lastSyncedAt.present ? lastSyncedAt.value : this.lastSyncedAt,
  );
  Leg copyWithCompanion(LegsCompanion data) {
    return Leg(
      id: data.id.present ? data.id.value : this.id,
      tripId: data.tripId.present ? data.tripId.value : this.tripId,
      fromStopId: data.fromStopId.present
          ? data.fromStopId.value
          : this.fromStopId,
      toStopId: data.toStopId.present ? data.toStopId.value : this.toStopId,
      sequenceOrder: data.sequenceOrder.present
          ? data.sequenceOrder.value
          : this.sequenceOrder,
      mode: data.mode.present ? data.mode.value : this.mode,
      plannedDeparture: data.plannedDeparture.present
          ? data.plannedDeparture.value
          : this.plannedDeparture,
      plannedArrival: data.plannedArrival.present
          ? data.plannedArrival.value
          : this.plannedArrival,
      isBooked: data.isBooked.present ? data.isBooked.value : this.isBooked,
      note: data.note.present ? data.note.value : this.note,
      routePolyline: data.routePolyline.present
          ? data.routePolyline.value
          : this.routePolyline,
      distanceKm: data.distanceKm.present
          ? data.distanceKm.value
          : this.distanceKm,
      corridorKm: data.corridorKm.present
          ? data.corridorKm.value
          : this.corridorKm,
      lastSyncedAt: data.lastSyncedAt.present
          ? data.lastSyncedAt.value
          : this.lastSyncedAt,
    );
  }

  @override
  String toString() {
    return (StringBuffer('Leg(')
          ..write('id: $id, ')
          ..write('tripId: $tripId, ')
          ..write('fromStopId: $fromStopId, ')
          ..write('toStopId: $toStopId, ')
          ..write('sequenceOrder: $sequenceOrder, ')
          ..write('mode: $mode, ')
          ..write('plannedDeparture: $plannedDeparture, ')
          ..write('plannedArrival: $plannedArrival, ')
          ..write('isBooked: $isBooked, ')
          ..write('note: $note, ')
          ..write('routePolyline: $routePolyline, ')
          ..write('distanceKm: $distanceKm, ')
          ..write('corridorKm: $corridorKm, ')
          ..write('lastSyncedAt: $lastSyncedAt')
          ..write(')'))
        .toString();
  }

  @override
  int get hashCode => Object.hash(
    id,
    tripId,
    fromStopId,
    toStopId,
    sequenceOrder,
    mode,
    plannedDeparture,
    plannedArrival,
    isBooked,
    note,
    routePolyline,
    distanceKm,
    corridorKm,
    lastSyncedAt,
  );
  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      (other is Leg &&
          other.id == this.id &&
          other.tripId == this.tripId &&
          other.fromStopId == this.fromStopId &&
          other.toStopId == this.toStopId &&
          other.sequenceOrder == this.sequenceOrder &&
          other.mode == this.mode &&
          other.plannedDeparture == this.plannedDeparture &&
          other.plannedArrival == this.plannedArrival &&
          other.isBooked == this.isBooked &&
          other.note == this.note &&
          other.routePolyline == this.routePolyline &&
          other.distanceKm == this.distanceKm &&
          other.corridorKm == this.corridorKm &&
          other.lastSyncedAt == this.lastSyncedAt);
}

class LegsCompanion extends UpdateCompanion<Leg> {
  final Value<int> id;
  final Value<int> tripId;
  final Value<int> fromStopId;
  final Value<int> toStopId;
  final Value<int> sequenceOrder;
  final Value<String?> mode;
  final Value<DateTime?> plannedDeparture;
  final Value<DateTime?> plannedArrival;
  final Value<bool> isBooked;
  final Value<String?> note;
  final Value<String?> routePolyline;
  final Value<double?> distanceKm;
  final Value<double> corridorKm;
  final Value<DateTime?> lastSyncedAt;
  const LegsCompanion({
    this.id = const Value.absent(),
    this.tripId = const Value.absent(),
    this.fromStopId = const Value.absent(),
    this.toStopId = const Value.absent(),
    this.sequenceOrder = const Value.absent(),
    this.mode = const Value.absent(),
    this.plannedDeparture = const Value.absent(),
    this.plannedArrival = const Value.absent(),
    this.isBooked = const Value.absent(),
    this.note = const Value.absent(),
    this.routePolyline = const Value.absent(),
    this.distanceKm = const Value.absent(),
    this.corridorKm = const Value.absent(),
    this.lastSyncedAt = const Value.absent(),
  });
  LegsCompanion.insert({
    this.id = const Value.absent(),
    required int tripId,
    required int fromStopId,
    required int toStopId,
    required int sequenceOrder,
    this.mode = const Value.absent(),
    this.plannedDeparture = const Value.absent(),
    this.plannedArrival = const Value.absent(),
    this.isBooked = const Value.absent(),
    this.note = const Value.absent(),
    this.routePolyline = const Value.absent(),
    this.distanceKm = const Value.absent(),
    this.corridorKm = const Value.absent(),
    this.lastSyncedAt = const Value.absent(),
  }) : tripId = Value(tripId),
       fromStopId = Value(fromStopId),
       toStopId = Value(toStopId),
       sequenceOrder = Value(sequenceOrder);
  static Insertable<Leg> custom({
    Expression<int>? id,
    Expression<int>? tripId,
    Expression<int>? fromStopId,
    Expression<int>? toStopId,
    Expression<int>? sequenceOrder,
    Expression<String>? mode,
    Expression<DateTime>? plannedDeparture,
    Expression<DateTime>? plannedArrival,
    Expression<bool>? isBooked,
    Expression<String>? note,
    Expression<String>? routePolyline,
    Expression<double>? distanceKm,
    Expression<double>? corridorKm,
    Expression<DateTime>? lastSyncedAt,
  }) {
    return RawValuesInsertable({
      if (id != null) 'id': id,
      if (tripId != null) 'trip_id': tripId,
      if (fromStopId != null) 'from_stop_id': fromStopId,
      if (toStopId != null) 'to_stop_id': toStopId,
      if (sequenceOrder != null) 'sequence_order': sequenceOrder,
      if (mode != null) 'mode': mode,
      if (plannedDeparture != null) 'planned_departure': plannedDeparture,
      if (plannedArrival != null) 'planned_arrival': plannedArrival,
      if (isBooked != null) 'is_booked': isBooked,
      if (note != null) 'note': note,
      if (routePolyline != null) 'route_polyline': routePolyline,
      if (distanceKm != null) 'distance_km': distanceKm,
      if (corridorKm != null) 'corridor_km': corridorKm,
      if (lastSyncedAt != null) 'last_synced_at': lastSyncedAt,
    });
  }

  LegsCompanion copyWith({
    Value<int>? id,
    Value<int>? tripId,
    Value<int>? fromStopId,
    Value<int>? toStopId,
    Value<int>? sequenceOrder,
    Value<String?>? mode,
    Value<DateTime?>? plannedDeparture,
    Value<DateTime?>? plannedArrival,
    Value<bool>? isBooked,
    Value<String?>? note,
    Value<String?>? routePolyline,
    Value<double?>? distanceKm,
    Value<double>? corridorKm,
    Value<DateTime?>? lastSyncedAt,
  }) {
    return LegsCompanion(
      id: id ?? this.id,
      tripId: tripId ?? this.tripId,
      fromStopId: fromStopId ?? this.fromStopId,
      toStopId: toStopId ?? this.toStopId,
      sequenceOrder: sequenceOrder ?? this.sequenceOrder,
      mode: mode ?? this.mode,
      plannedDeparture: plannedDeparture ?? this.plannedDeparture,
      plannedArrival: plannedArrival ?? this.plannedArrival,
      isBooked: isBooked ?? this.isBooked,
      note: note ?? this.note,
      routePolyline: routePolyline ?? this.routePolyline,
      distanceKm: distanceKm ?? this.distanceKm,
      corridorKm: corridorKm ?? this.corridorKm,
      lastSyncedAt: lastSyncedAt ?? this.lastSyncedAt,
    );
  }

  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    if (id.present) {
      map['id'] = Variable<int>(id.value);
    }
    if (tripId.present) {
      map['trip_id'] = Variable<int>(tripId.value);
    }
    if (fromStopId.present) {
      map['from_stop_id'] = Variable<int>(fromStopId.value);
    }
    if (toStopId.present) {
      map['to_stop_id'] = Variable<int>(toStopId.value);
    }
    if (sequenceOrder.present) {
      map['sequence_order'] = Variable<int>(sequenceOrder.value);
    }
    if (mode.present) {
      map['mode'] = Variable<String>(mode.value);
    }
    if (plannedDeparture.present) {
      map['planned_departure'] = Variable<DateTime>(plannedDeparture.value);
    }
    if (plannedArrival.present) {
      map['planned_arrival'] = Variable<DateTime>(plannedArrival.value);
    }
    if (isBooked.present) {
      map['is_booked'] = Variable<bool>(isBooked.value);
    }
    if (note.present) {
      map['note'] = Variable<String>(note.value);
    }
    if (routePolyline.present) {
      map['route_polyline'] = Variable<String>(routePolyline.value);
    }
    if (distanceKm.present) {
      map['distance_km'] = Variable<double>(distanceKm.value);
    }
    if (corridorKm.present) {
      map['corridor_km'] = Variable<double>(corridorKm.value);
    }
    if (lastSyncedAt.present) {
      map['last_synced_at'] = Variable<DateTime>(lastSyncedAt.value);
    }
    return map;
  }

  @override
  String toString() {
    return (StringBuffer('LegsCompanion(')
          ..write('id: $id, ')
          ..write('tripId: $tripId, ')
          ..write('fromStopId: $fromStopId, ')
          ..write('toStopId: $toStopId, ')
          ..write('sequenceOrder: $sequenceOrder, ')
          ..write('mode: $mode, ')
          ..write('plannedDeparture: $plannedDeparture, ')
          ..write('plannedArrival: $plannedArrival, ')
          ..write('isBooked: $isBooked, ')
          ..write('note: $note, ')
          ..write('routePolyline: $routePolyline, ')
          ..write('distanceKm: $distanceKm, ')
          ..write('corridorKm: $corridorKm, ')
          ..write('lastSyncedAt: $lastSyncedAt')
          ..write(')'))
        .toString();
  }
}

class $PoisTable extends Pois with TableInfo<$PoisTable, Poi> {
  @override
  final GeneratedDatabase attachedDatabase;
  final String? _alias;
  $PoisTable(this.attachedDatabase, [this._alias]);
  static const VerificationMeta _idMeta = const VerificationMeta('id');
  @override
  late final GeneratedColumn<int> id = GeneratedColumn<int>(
    'id',
    aliasedName,
    false,
    hasAutoIncrement: true,
    type: DriftSqlType.int,
    requiredDuringInsert: false,
    defaultConstraints: GeneratedColumn.constraintIsAlways(
      'PRIMARY KEY AUTOINCREMENT',
    ),
  );
  static const VerificationMeta _tripIdMeta = const VerificationMeta('tripId');
  @override
  late final GeneratedColumn<int> tripId = GeneratedColumn<int>(
    'trip_id',
    aliasedName,
    false,
    type: DriftSqlType.int,
    requiredDuringInsert: true,
    defaultConstraints: GeneratedColumn.constraintIsAlways(
      'REFERENCES trips (id) ON DELETE CASCADE',
    ),
  );
  static const VerificationMeta _stopIdMeta = const VerificationMeta('stopId');
  @override
  late final GeneratedColumn<int> stopId = GeneratedColumn<int>(
    'stop_id',
    aliasedName,
    true,
    type: DriftSqlType.int,
    requiredDuringInsert: false,
    defaultConstraints: GeneratedColumn.constraintIsAlways(
      'REFERENCES stops (id) ON DELETE CASCADE',
    ),
  );
  static const VerificationMeta _legIdMeta = const VerificationMeta('legId');
  @override
  late final GeneratedColumn<int> legId = GeneratedColumn<int>(
    'leg_id',
    aliasedName,
    true,
    type: DriftSqlType.int,
    requiredDuringInsert: false,
    defaultConstraints: GeneratedColumn.constraintIsAlways(
      'REFERENCES legs (id) ON DELETE CASCADE',
    ),
  );
  static const VerificationMeta _nameMeta = const VerificationMeta('name');
  @override
  late final GeneratedColumn<String> name = GeneratedColumn<String>(
    'name',
    aliasedName,
    false,
    type: DriftSqlType.string,
    requiredDuringInsert: true,
  );
  static const VerificationMeta _categoryMeta = const VerificationMeta(
    'category',
  );
  @override
  late final GeneratedColumn<String> category = GeneratedColumn<String>(
    'category',
    aliasedName,
    false,
    type: DriftSqlType.string,
    requiredDuringInsert: true,
  );
  static const VerificationMeta _latMeta = const VerificationMeta('lat');
  @override
  late final GeneratedColumn<double> lat = GeneratedColumn<double>(
    'lat',
    aliasedName,
    false,
    type: DriftSqlType.double,
    requiredDuringInsert: true,
  );
  static const VerificationMeta _lonMeta = const VerificationMeta('lon');
  @override
  late final GeneratedColumn<double> lon = GeneratedColumn<double>(
    'lon',
    aliasedName,
    false,
    type: DriftSqlType.double,
    requiredDuringInsert: true,
  );
  static const VerificationMeta _distanceAlongRouteKmMeta =
      const VerificationMeta('distanceAlongRouteKm');
  @override
  late final GeneratedColumn<double> distanceAlongRouteKm =
      GeneratedColumn<double>(
        'distance_along_route_km',
        aliasedName,
        true,
        type: DriftSqlType.double,
        requiredDuringInsert: false,
      );
  static const VerificationMeta _distanceOffRouteKmMeta =
      const VerificationMeta('distanceOffRouteKm');
  @override
  late final GeneratedColumn<double> distanceOffRouteKm =
      GeneratedColumn<double>(
        'distance_off_route_km',
        aliasedName,
        true,
        type: DriftSqlType.double,
        requiredDuringInsert: false,
      );
  static const VerificationMeta _osmIdMeta = const VerificationMeta('osmId');
  @override
  late final GeneratedColumn<String> osmId = GeneratedColumn<String>(
    'osm_id',
    aliasedName,
    true,
    type: DriftSqlType.string,
    requiredDuringInsert: false,
  );
  static const VerificationMeta _rawTagsMeta = const VerificationMeta(
    'rawTags',
  );
  @override
  late final GeneratedColumn<String> rawTags = GeneratedColumn<String>(
    'raw_tags',
    aliasedName,
    true,
    type: DriftSqlType.string,
    requiredDuringInsert: false,
  );
  static const VerificationMeta _cachedAtMeta = const VerificationMeta(
    'cachedAt',
  );
  @override
  late final GeneratedColumn<DateTime> cachedAt = GeneratedColumn<DateTime>(
    'cached_at',
    aliasedName,
    false,
    type: DriftSqlType.dateTime,
    requiredDuringInsert: false,
    defaultValue: currentDateAndTime,
  );
  @override
  List<GeneratedColumn> get $columns => [
    id,
    tripId,
    stopId,
    legId,
    name,
    category,
    lat,
    lon,
    distanceAlongRouteKm,
    distanceOffRouteKm,
    osmId,
    rawTags,
    cachedAt,
  ];
  @override
  String get aliasedName => _alias ?? actualTableName;
  @override
  String get actualTableName => $name;
  static const String $name = 'pois';
  @override
  VerificationContext validateIntegrity(
    Insertable<Poi> instance, {
    bool isInserting = false,
  }) {
    final context = VerificationContext();
    final data = instance.toColumns(true);
    if (data.containsKey('id')) {
      context.handle(_idMeta, id.isAcceptableOrUnknown(data['id']!, _idMeta));
    }
    if (data.containsKey('trip_id')) {
      context.handle(
        _tripIdMeta,
        tripId.isAcceptableOrUnknown(data['trip_id']!, _tripIdMeta),
      );
    } else if (isInserting) {
      context.missing(_tripIdMeta);
    }
    if (data.containsKey('stop_id')) {
      context.handle(
        _stopIdMeta,
        stopId.isAcceptableOrUnknown(data['stop_id']!, _stopIdMeta),
      );
    }
    if (data.containsKey('leg_id')) {
      context.handle(
        _legIdMeta,
        legId.isAcceptableOrUnknown(data['leg_id']!, _legIdMeta),
      );
    }
    if (data.containsKey('name')) {
      context.handle(
        _nameMeta,
        name.isAcceptableOrUnknown(data['name']!, _nameMeta),
      );
    } else if (isInserting) {
      context.missing(_nameMeta);
    }
    if (data.containsKey('category')) {
      context.handle(
        _categoryMeta,
        category.isAcceptableOrUnknown(data['category']!, _categoryMeta),
      );
    } else if (isInserting) {
      context.missing(_categoryMeta);
    }
    if (data.containsKey('lat')) {
      context.handle(
        _latMeta,
        lat.isAcceptableOrUnknown(data['lat']!, _latMeta),
      );
    } else if (isInserting) {
      context.missing(_latMeta);
    }
    if (data.containsKey('lon')) {
      context.handle(
        _lonMeta,
        lon.isAcceptableOrUnknown(data['lon']!, _lonMeta),
      );
    } else if (isInserting) {
      context.missing(_lonMeta);
    }
    if (data.containsKey('distance_along_route_km')) {
      context.handle(
        _distanceAlongRouteKmMeta,
        distanceAlongRouteKm.isAcceptableOrUnknown(
          data['distance_along_route_km']!,
          _distanceAlongRouteKmMeta,
        ),
      );
    }
    if (data.containsKey('distance_off_route_km')) {
      context.handle(
        _distanceOffRouteKmMeta,
        distanceOffRouteKm.isAcceptableOrUnknown(
          data['distance_off_route_km']!,
          _distanceOffRouteKmMeta,
        ),
      );
    }
    if (data.containsKey('osm_id')) {
      context.handle(
        _osmIdMeta,
        osmId.isAcceptableOrUnknown(data['osm_id']!, _osmIdMeta),
      );
    }
    if (data.containsKey('raw_tags')) {
      context.handle(
        _rawTagsMeta,
        rawTags.isAcceptableOrUnknown(data['raw_tags']!, _rawTagsMeta),
      );
    }
    if (data.containsKey('cached_at')) {
      context.handle(
        _cachedAtMeta,
        cachedAt.isAcceptableOrUnknown(data['cached_at']!, _cachedAtMeta),
      );
    }
    return context;
  }

  @override
  Set<GeneratedColumn> get $primaryKey => {id};
  @override
  Poi map(Map<String, dynamic> data, {String? tablePrefix}) {
    final effectivePrefix = tablePrefix != null ? '$tablePrefix.' : '';
    return Poi(
      id: attachedDatabase.typeMapping.read(
        DriftSqlType.int,
        data['${effectivePrefix}id'],
      )!,
      tripId: attachedDatabase.typeMapping.read(
        DriftSqlType.int,
        data['${effectivePrefix}trip_id'],
      )!,
      stopId: attachedDatabase.typeMapping.read(
        DriftSqlType.int,
        data['${effectivePrefix}stop_id'],
      ),
      legId: attachedDatabase.typeMapping.read(
        DriftSqlType.int,
        data['${effectivePrefix}leg_id'],
      ),
      name: attachedDatabase.typeMapping.read(
        DriftSqlType.string,
        data['${effectivePrefix}name'],
      )!,
      category: attachedDatabase.typeMapping.read(
        DriftSqlType.string,
        data['${effectivePrefix}category'],
      )!,
      lat: attachedDatabase.typeMapping.read(
        DriftSqlType.double,
        data['${effectivePrefix}lat'],
      )!,
      lon: attachedDatabase.typeMapping.read(
        DriftSqlType.double,
        data['${effectivePrefix}lon'],
      )!,
      distanceAlongRouteKm: attachedDatabase.typeMapping.read(
        DriftSqlType.double,
        data['${effectivePrefix}distance_along_route_km'],
      ),
      distanceOffRouteKm: attachedDatabase.typeMapping.read(
        DriftSqlType.double,
        data['${effectivePrefix}distance_off_route_km'],
      ),
      osmId: attachedDatabase.typeMapping.read(
        DriftSqlType.string,
        data['${effectivePrefix}osm_id'],
      ),
      rawTags: attachedDatabase.typeMapping.read(
        DriftSqlType.string,
        data['${effectivePrefix}raw_tags'],
      ),
      cachedAt: attachedDatabase.typeMapping.read(
        DriftSqlType.dateTime,
        data['${effectivePrefix}cached_at'],
      )!,
    );
  }

  @override
  $PoisTable createAlias(String alias) {
    return $PoisTable(attachedDatabase, alias);
  }
}

class Poi extends DataClass implements Insertable<Poi> {
  final int id;
  final int tripId;
  final int? stopId;
  final int? legId;
  final String name;
  final String category;
  final double lat;
  final double lon;

  /// How far along the leg this sits, and how far off the line. Powers the
  /// "coming up in 12 km" ordering on the leg screen.
  final double? distanceAlongRouteKm;
  final double? distanceOffRouteKm;
  final String? osmId;
  final String? rawTags;
  final DateTime cachedAt;
  const Poi({
    required this.id,
    required this.tripId,
    this.stopId,
    this.legId,
    required this.name,
    required this.category,
    required this.lat,
    required this.lon,
    this.distanceAlongRouteKm,
    this.distanceOffRouteKm,
    this.osmId,
    this.rawTags,
    required this.cachedAt,
  });
  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    map['id'] = Variable<int>(id);
    map['trip_id'] = Variable<int>(tripId);
    if (!nullToAbsent || stopId != null) {
      map['stop_id'] = Variable<int>(stopId);
    }
    if (!nullToAbsent || legId != null) {
      map['leg_id'] = Variable<int>(legId);
    }
    map['name'] = Variable<String>(name);
    map['category'] = Variable<String>(category);
    map['lat'] = Variable<double>(lat);
    map['lon'] = Variable<double>(lon);
    if (!nullToAbsent || distanceAlongRouteKm != null) {
      map['distance_along_route_km'] = Variable<double>(distanceAlongRouteKm);
    }
    if (!nullToAbsent || distanceOffRouteKm != null) {
      map['distance_off_route_km'] = Variable<double>(distanceOffRouteKm);
    }
    if (!nullToAbsent || osmId != null) {
      map['osm_id'] = Variable<String>(osmId);
    }
    if (!nullToAbsent || rawTags != null) {
      map['raw_tags'] = Variable<String>(rawTags);
    }
    map['cached_at'] = Variable<DateTime>(cachedAt);
    return map;
  }

  PoisCompanion toCompanion(bool nullToAbsent) {
    return PoisCompanion(
      id: Value(id),
      tripId: Value(tripId),
      stopId: stopId == null && nullToAbsent
          ? const Value.absent()
          : Value(stopId),
      legId: legId == null && nullToAbsent
          ? const Value.absent()
          : Value(legId),
      name: Value(name),
      category: Value(category),
      lat: Value(lat),
      lon: Value(lon),
      distanceAlongRouteKm: distanceAlongRouteKm == null && nullToAbsent
          ? const Value.absent()
          : Value(distanceAlongRouteKm),
      distanceOffRouteKm: distanceOffRouteKm == null && nullToAbsent
          ? const Value.absent()
          : Value(distanceOffRouteKm),
      osmId: osmId == null && nullToAbsent
          ? const Value.absent()
          : Value(osmId),
      rawTags: rawTags == null && nullToAbsent
          ? const Value.absent()
          : Value(rawTags),
      cachedAt: Value(cachedAt),
    );
  }

  factory Poi.fromJson(
    Map<String, dynamic> json, {
    ValueSerializer? serializer,
  }) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return Poi(
      id: serializer.fromJson<int>(json['id']),
      tripId: serializer.fromJson<int>(json['tripId']),
      stopId: serializer.fromJson<int?>(json['stopId']),
      legId: serializer.fromJson<int?>(json['legId']),
      name: serializer.fromJson<String>(json['name']),
      category: serializer.fromJson<String>(json['category']),
      lat: serializer.fromJson<double>(json['lat']),
      lon: serializer.fromJson<double>(json['lon']),
      distanceAlongRouteKm: serializer.fromJson<double?>(
        json['distanceAlongRouteKm'],
      ),
      distanceOffRouteKm: serializer.fromJson<double?>(
        json['distanceOffRouteKm'],
      ),
      osmId: serializer.fromJson<String?>(json['osmId']),
      rawTags: serializer.fromJson<String?>(json['rawTags']),
      cachedAt: serializer.fromJson<DateTime>(json['cachedAt']),
    );
  }
  @override
  Map<String, dynamic> toJson({ValueSerializer? serializer}) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return <String, dynamic>{
      'id': serializer.toJson<int>(id),
      'tripId': serializer.toJson<int>(tripId),
      'stopId': serializer.toJson<int?>(stopId),
      'legId': serializer.toJson<int?>(legId),
      'name': serializer.toJson<String>(name),
      'category': serializer.toJson<String>(category),
      'lat': serializer.toJson<double>(lat),
      'lon': serializer.toJson<double>(lon),
      'distanceAlongRouteKm': serializer.toJson<double?>(distanceAlongRouteKm),
      'distanceOffRouteKm': serializer.toJson<double?>(distanceOffRouteKm),
      'osmId': serializer.toJson<String?>(osmId),
      'rawTags': serializer.toJson<String?>(rawTags),
      'cachedAt': serializer.toJson<DateTime>(cachedAt),
    };
  }

  Poi copyWith({
    int? id,
    int? tripId,
    Value<int?> stopId = const Value.absent(),
    Value<int?> legId = const Value.absent(),
    String? name,
    String? category,
    double? lat,
    double? lon,
    Value<double?> distanceAlongRouteKm = const Value.absent(),
    Value<double?> distanceOffRouteKm = const Value.absent(),
    Value<String?> osmId = const Value.absent(),
    Value<String?> rawTags = const Value.absent(),
    DateTime? cachedAt,
  }) => Poi(
    id: id ?? this.id,
    tripId: tripId ?? this.tripId,
    stopId: stopId.present ? stopId.value : this.stopId,
    legId: legId.present ? legId.value : this.legId,
    name: name ?? this.name,
    category: category ?? this.category,
    lat: lat ?? this.lat,
    lon: lon ?? this.lon,
    distanceAlongRouteKm: distanceAlongRouteKm.present
        ? distanceAlongRouteKm.value
        : this.distanceAlongRouteKm,
    distanceOffRouteKm: distanceOffRouteKm.present
        ? distanceOffRouteKm.value
        : this.distanceOffRouteKm,
    osmId: osmId.present ? osmId.value : this.osmId,
    rawTags: rawTags.present ? rawTags.value : this.rawTags,
    cachedAt: cachedAt ?? this.cachedAt,
  );
  Poi copyWithCompanion(PoisCompanion data) {
    return Poi(
      id: data.id.present ? data.id.value : this.id,
      tripId: data.tripId.present ? data.tripId.value : this.tripId,
      stopId: data.stopId.present ? data.stopId.value : this.stopId,
      legId: data.legId.present ? data.legId.value : this.legId,
      name: data.name.present ? data.name.value : this.name,
      category: data.category.present ? data.category.value : this.category,
      lat: data.lat.present ? data.lat.value : this.lat,
      lon: data.lon.present ? data.lon.value : this.lon,
      distanceAlongRouteKm: data.distanceAlongRouteKm.present
          ? data.distanceAlongRouteKm.value
          : this.distanceAlongRouteKm,
      distanceOffRouteKm: data.distanceOffRouteKm.present
          ? data.distanceOffRouteKm.value
          : this.distanceOffRouteKm,
      osmId: data.osmId.present ? data.osmId.value : this.osmId,
      rawTags: data.rawTags.present ? data.rawTags.value : this.rawTags,
      cachedAt: data.cachedAt.present ? data.cachedAt.value : this.cachedAt,
    );
  }

  @override
  String toString() {
    return (StringBuffer('Poi(')
          ..write('id: $id, ')
          ..write('tripId: $tripId, ')
          ..write('stopId: $stopId, ')
          ..write('legId: $legId, ')
          ..write('name: $name, ')
          ..write('category: $category, ')
          ..write('lat: $lat, ')
          ..write('lon: $lon, ')
          ..write('distanceAlongRouteKm: $distanceAlongRouteKm, ')
          ..write('distanceOffRouteKm: $distanceOffRouteKm, ')
          ..write('osmId: $osmId, ')
          ..write('rawTags: $rawTags, ')
          ..write('cachedAt: $cachedAt')
          ..write(')'))
        .toString();
  }

  @override
  int get hashCode => Object.hash(
    id,
    tripId,
    stopId,
    legId,
    name,
    category,
    lat,
    lon,
    distanceAlongRouteKm,
    distanceOffRouteKm,
    osmId,
    rawTags,
    cachedAt,
  );
  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      (other is Poi &&
          other.id == this.id &&
          other.tripId == this.tripId &&
          other.stopId == this.stopId &&
          other.legId == this.legId &&
          other.name == this.name &&
          other.category == this.category &&
          other.lat == this.lat &&
          other.lon == this.lon &&
          other.distanceAlongRouteKm == this.distanceAlongRouteKm &&
          other.distanceOffRouteKm == this.distanceOffRouteKm &&
          other.osmId == this.osmId &&
          other.rawTags == this.rawTags &&
          other.cachedAt == this.cachedAt);
}

class PoisCompanion extends UpdateCompanion<Poi> {
  final Value<int> id;
  final Value<int> tripId;
  final Value<int?> stopId;
  final Value<int?> legId;
  final Value<String> name;
  final Value<String> category;
  final Value<double> lat;
  final Value<double> lon;
  final Value<double?> distanceAlongRouteKm;
  final Value<double?> distanceOffRouteKm;
  final Value<String?> osmId;
  final Value<String?> rawTags;
  final Value<DateTime> cachedAt;
  const PoisCompanion({
    this.id = const Value.absent(),
    this.tripId = const Value.absent(),
    this.stopId = const Value.absent(),
    this.legId = const Value.absent(),
    this.name = const Value.absent(),
    this.category = const Value.absent(),
    this.lat = const Value.absent(),
    this.lon = const Value.absent(),
    this.distanceAlongRouteKm = const Value.absent(),
    this.distanceOffRouteKm = const Value.absent(),
    this.osmId = const Value.absent(),
    this.rawTags = const Value.absent(),
    this.cachedAt = const Value.absent(),
  });
  PoisCompanion.insert({
    this.id = const Value.absent(),
    required int tripId,
    this.stopId = const Value.absent(),
    this.legId = const Value.absent(),
    required String name,
    required String category,
    required double lat,
    required double lon,
    this.distanceAlongRouteKm = const Value.absent(),
    this.distanceOffRouteKm = const Value.absent(),
    this.osmId = const Value.absent(),
    this.rawTags = const Value.absent(),
    this.cachedAt = const Value.absent(),
  }) : tripId = Value(tripId),
       name = Value(name),
       category = Value(category),
       lat = Value(lat),
       lon = Value(lon);
  static Insertable<Poi> custom({
    Expression<int>? id,
    Expression<int>? tripId,
    Expression<int>? stopId,
    Expression<int>? legId,
    Expression<String>? name,
    Expression<String>? category,
    Expression<double>? lat,
    Expression<double>? lon,
    Expression<double>? distanceAlongRouteKm,
    Expression<double>? distanceOffRouteKm,
    Expression<String>? osmId,
    Expression<String>? rawTags,
    Expression<DateTime>? cachedAt,
  }) {
    return RawValuesInsertable({
      if (id != null) 'id': id,
      if (tripId != null) 'trip_id': tripId,
      if (stopId != null) 'stop_id': stopId,
      if (legId != null) 'leg_id': legId,
      if (name != null) 'name': name,
      if (category != null) 'category': category,
      if (lat != null) 'lat': lat,
      if (lon != null) 'lon': lon,
      if (distanceAlongRouteKm != null)
        'distance_along_route_km': distanceAlongRouteKm,
      if (distanceOffRouteKm != null)
        'distance_off_route_km': distanceOffRouteKm,
      if (osmId != null) 'osm_id': osmId,
      if (rawTags != null) 'raw_tags': rawTags,
      if (cachedAt != null) 'cached_at': cachedAt,
    });
  }

  PoisCompanion copyWith({
    Value<int>? id,
    Value<int>? tripId,
    Value<int?>? stopId,
    Value<int?>? legId,
    Value<String>? name,
    Value<String>? category,
    Value<double>? lat,
    Value<double>? lon,
    Value<double?>? distanceAlongRouteKm,
    Value<double?>? distanceOffRouteKm,
    Value<String?>? osmId,
    Value<String?>? rawTags,
    Value<DateTime>? cachedAt,
  }) {
    return PoisCompanion(
      id: id ?? this.id,
      tripId: tripId ?? this.tripId,
      stopId: stopId ?? this.stopId,
      legId: legId ?? this.legId,
      name: name ?? this.name,
      category: category ?? this.category,
      lat: lat ?? this.lat,
      lon: lon ?? this.lon,
      distanceAlongRouteKm: distanceAlongRouteKm ?? this.distanceAlongRouteKm,
      distanceOffRouteKm: distanceOffRouteKm ?? this.distanceOffRouteKm,
      osmId: osmId ?? this.osmId,
      rawTags: rawTags ?? this.rawTags,
      cachedAt: cachedAt ?? this.cachedAt,
    );
  }

  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    if (id.present) {
      map['id'] = Variable<int>(id.value);
    }
    if (tripId.present) {
      map['trip_id'] = Variable<int>(tripId.value);
    }
    if (stopId.present) {
      map['stop_id'] = Variable<int>(stopId.value);
    }
    if (legId.present) {
      map['leg_id'] = Variable<int>(legId.value);
    }
    if (name.present) {
      map['name'] = Variable<String>(name.value);
    }
    if (category.present) {
      map['category'] = Variable<String>(category.value);
    }
    if (lat.present) {
      map['lat'] = Variable<double>(lat.value);
    }
    if (lon.present) {
      map['lon'] = Variable<double>(lon.value);
    }
    if (distanceAlongRouteKm.present) {
      map['distance_along_route_km'] = Variable<double>(
        distanceAlongRouteKm.value,
      );
    }
    if (distanceOffRouteKm.present) {
      map['distance_off_route_km'] = Variable<double>(distanceOffRouteKm.value);
    }
    if (osmId.present) {
      map['osm_id'] = Variable<String>(osmId.value);
    }
    if (rawTags.present) {
      map['raw_tags'] = Variable<String>(rawTags.value);
    }
    if (cachedAt.present) {
      map['cached_at'] = Variable<DateTime>(cachedAt.value);
    }
    return map;
  }

  @override
  String toString() {
    return (StringBuffer('PoisCompanion(')
          ..write('id: $id, ')
          ..write('tripId: $tripId, ')
          ..write('stopId: $stopId, ')
          ..write('legId: $legId, ')
          ..write('name: $name, ')
          ..write('category: $category, ')
          ..write('lat: $lat, ')
          ..write('lon: $lon, ')
          ..write('distanceAlongRouteKm: $distanceAlongRouteKm, ')
          ..write('distanceOffRouteKm: $distanceOffRouteKm, ')
          ..write('osmId: $osmId, ')
          ..write('rawTags: $rawTags, ')
          ..write('cachedAt: $cachedAt')
          ..write(')'))
        .toString();
  }
}

class $PoiContactsTable extends PoiContacts
    with TableInfo<$PoiContactsTable, PoiContact> {
  @override
  final GeneratedDatabase attachedDatabase;
  final String? _alias;
  $PoiContactsTable(this.attachedDatabase, [this._alias]);
  static const VerificationMeta _idMeta = const VerificationMeta('id');
  @override
  late final GeneratedColumn<int> id = GeneratedColumn<int>(
    'id',
    aliasedName,
    false,
    hasAutoIncrement: true,
    type: DriftSqlType.int,
    requiredDuringInsert: false,
    defaultConstraints: GeneratedColumn.constraintIsAlways(
      'PRIMARY KEY AUTOINCREMENT',
    ),
  );
  static const VerificationMeta _poiIdMeta = const VerificationMeta('poiId');
  @override
  late final GeneratedColumn<int> poiId = GeneratedColumn<int>(
    'poi_id',
    aliasedName,
    false,
    type: DriftSqlType.int,
    requiredDuringInsert: true,
    defaultConstraints: GeneratedColumn.constraintIsAlways(
      'REFERENCES pois (id) ON DELETE CASCADE',
    ),
  );
  static const VerificationMeta _phoneRawMeta = const VerificationMeta(
    'phoneRaw',
  );
  @override
  late final GeneratedColumn<String> phoneRaw = GeneratedColumn<String>(
    'phone_raw',
    aliasedName,
    false,
    type: DriftSqlType.string,
    requiredDuringInsert: true,
  );
  static const VerificationMeta _phoneE164Meta = const VerificationMeta(
    'phoneE164',
  );
  @override
  late final GeneratedColumn<String> phoneE164 = GeneratedColumn<String>(
    'phone_e164',
    aliasedName,
    true,
    type: DriftSqlType.string,
    requiredDuringInsert: false,
  );
  static const VerificationMeta _tierMeta = const VerificationMeta('tier');
  @override
  late final GeneratedColumn<String> tier = GeneratedColumn<String>(
    'tier',
    aliasedName,
    false,
    type: DriftSqlType.string,
    requiredDuringInsert: false,
    defaultValue: const Constant('communityOsm'),
  );
  static const VerificationMeta _sourceTagMeta = const VerificationMeta(
    'sourceTag',
  );
  @override
  late final GeneratedColumn<String> sourceTag = GeneratedColumn<String>(
    'source_tag',
    aliasedName,
    true,
    type: DriftSqlType.string,
    requiredDuringInsert: false,
  );
  @override
  List<GeneratedColumn> get $columns => [
    id,
    poiId,
    phoneRaw,
    phoneE164,
    tier,
    sourceTag,
  ];
  @override
  String get aliasedName => _alias ?? actualTableName;
  @override
  String get actualTableName => $name;
  static const String $name = 'poi_contacts';
  @override
  VerificationContext validateIntegrity(
    Insertable<PoiContact> instance, {
    bool isInserting = false,
  }) {
    final context = VerificationContext();
    final data = instance.toColumns(true);
    if (data.containsKey('id')) {
      context.handle(_idMeta, id.isAcceptableOrUnknown(data['id']!, _idMeta));
    }
    if (data.containsKey('poi_id')) {
      context.handle(
        _poiIdMeta,
        poiId.isAcceptableOrUnknown(data['poi_id']!, _poiIdMeta),
      );
    } else if (isInserting) {
      context.missing(_poiIdMeta);
    }
    if (data.containsKey('phone_raw')) {
      context.handle(
        _phoneRawMeta,
        phoneRaw.isAcceptableOrUnknown(data['phone_raw']!, _phoneRawMeta),
      );
    } else if (isInserting) {
      context.missing(_phoneRawMeta);
    }
    if (data.containsKey('phone_e164')) {
      context.handle(
        _phoneE164Meta,
        phoneE164.isAcceptableOrUnknown(data['phone_e164']!, _phoneE164Meta),
      );
    }
    if (data.containsKey('tier')) {
      context.handle(
        _tierMeta,
        tier.isAcceptableOrUnknown(data['tier']!, _tierMeta),
      );
    }
    if (data.containsKey('source_tag')) {
      context.handle(
        _sourceTagMeta,
        sourceTag.isAcceptableOrUnknown(data['source_tag']!, _sourceTagMeta),
      );
    }
    return context;
  }

  @override
  Set<GeneratedColumn> get $primaryKey => {id};
  @override
  PoiContact map(Map<String, dynamic> data, {String? tablePrefix}) {
    final effectivePrefix = tablePrefix != null ? '$tablePrefix.' : '';
    return PoiContact(
      id: attachedDatabase.typeMapping.read(
        DriftSqlType.int,
        data['${effectivePrefix}id'],
      )!,
      poiId: attachedDatabase.typeMapping.read(
        DriftSqlType.int,
        data['${effectivePrefix}poi_id'],
      )!,
      phoneRaw: attachedDatabase.typeMapping.read(
        DriftSqlType.string,
        data['${effectivePrefix}phone_raw'],
      )!,
      phoneE164: attachedDatabase.typeMapping.read(
        DriftSqlType.string,
        data['${effectivePrefix}phone_e164'],
      ),
      tier: attachedDatabase.typeMapping.read(
        DriftSqlType.string,
        data['${effectivePrefix}tier'],
      )!,
      sourceTag: attachedDatabase.typeMapping.read(
        DriftSqlType.string,
        data['${effectivePrefix}source_tag'],
      ),
    );
  }

  @override
  $PoiContactsTable createAlias(String alias) {
    return $PoiContactsTable(attachedDatabase, alias);
  }
}

class PoiContact extends DataClass implements Insertable<PoiContact> {
  final int id;
  final int poiId;
  final String phoneRaw;
  final String? phoneE164;
  final String tier;

  /// Which OSM key it came from: `phone` or `contact:phone`.
  final String? sourceTag;
  const PoiContact({
    required this.id,
    required this.poiId,
    required this.phoneRaw,
    this.phoneE164,
    required this.tier,
    this.sourceTag,
  });
  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    map['id'] = Variable<int>(id);
    map['poi_id'] = Variable<int>(poiId);
    map['phone_raw'] = Variable<String>(phoneRaw);
    if (!nullToAbsent || phoneE164 != null) {
      map['phone_e164'] = Variable<String>(phoneE164);
    }
    map['tier'] = Variable<String>(tier);
    if (!nullToAbsent || sourceTag != null) {
      map['source_tag'] = Variable<String>(sourceTag);
    }
    return map;
  }

  PoiContactsCompanion toCompanion(bool nullToAbsent) {
    return PoiContactsCompanion(
      id: Value(id),
      poiId: Value(poiId),
      phoneRaw: Value(phoneRaw),
      phoneE164: phoneE164 == null && nullToAbsent
          ? const Value.absent()
          : Value(phoneE164),
      tier: Value(tier),
      sourceTag: sourceTag == null && nullToAbsent
          ? const Value.absent()
          : Value(sourceTag),
    );
  }

  factory PoiContact.fromJson(
    Map<String, dynamic> json, {
    ValueSerializer? serializer,
  }) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return PoiContact(
      id: serializer.fromJson<int>(json['id']),
      poiId: serializer.fromJson<int>(json['poiId']),
      phoneRaw: serializer.fromJson<String>(json['phoneRaw']),
      phoneE164: serializer.fromJson<String?>(json['phoneE164']),
      tier: serializer.fromJson<String>(json['tier']),
      sourceTag: serializer.fromJson<String?>(json['sourceTag']),
    );
  }
  @override
  Map<String, dynamic> toJson({ValueSerializer? serializer}) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return <String, dynamic>{
      'id': serializer.toJson<int>(id),
      'poiId': serializer.toJson<int>(poiId),
      'phoneRaw': serializer.toJson<String>(phoneRaw),
      'phoneE164': serializer.toJson<String?>(phoneE164),
      'tier': serializer.toJson<String>(tier),
      'sourceTag': serializer.toJson<String?>(sourceTag),
    };
  }

  PoiContact copyWith({
    int? id,
    int? poiId,
    String? phoneRaw,
    Value<String?> phoneE164 = const Value.absent(),
    String? tier,
    Value<String?> sourceTag = const Value.absent(),
  }) => PoiContact(
    id: id ?? this.id,
    poiId: poiId ?? this.poiId,
    phoneRaw: phoneRaw ?? this.phoneRaw,
    phoneE164: phoneE164.present ? phoneE164.value : this.phoneE164,
    tier: tier ?? this.tier,
    sourceTag: sourceTag.present ? sourceTag.value : this.sourceTag,
  );
  PoiContact copyWithCompanion(PoiContactsCompanion data) {
    return PoiContact(
      id: data.id.present ? data.id.value : this.id,
      poiId: data.poiId.present ? data.poiId.value : this.poiId,
      phoneRaw: data.phoneRaw.present ? data.phoneRaw.value : this.phoneRaw,
      phoneE164: data.phoneE164.present ? data.phoneE164.value : this.phoneE164,
      tier: data.tier.present ? data.tier.value : this.tier,
      sourceTag: data.sourceTag.present ? data.sourceTag.value : this.sourceTag,
    );
  }

  @override
  String toString() {
    return (StringBuffer('PoiContact(')
          ..write('id: $id, ')
          ..write('poiId: $poiId, ')
          ..write('phoneRaw: $phoneRaw, ')
          ..write('phoneE164: $phoneE164, ')
          ..write('tier: $tier, ')
          ..write('sourceTag: $sourceTag')
          ..write(')'))
        .toString();
  }

  @override
  int get hashCode =>
      Object.hash(id, poiId, phoneRaw, phoneE164, tier, sourceTag);
  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      (other is PoiContact &&
          other.id == this.id &&
          other.poiId == this.poiId &&
          other.phoneRaw == this.phoneRaw &&
          other.phoneE164 == this.phoneE164 &&
          other.tier == this.tier &&
          other.sourceTag == this.sourceTag);
}

class PoiContactsCompanion extends UpdateCompanion<PoiContact> {
  final Value<int> id;
  final Value<int> poiId;
  final Value<String> phoneRaw;
  final Value<String?> phoneE164;
  final Value<String> tier;
  final Value<String?> sourceTag;
  const PoiContactsCompanion({
    this.id = const Value.absent(),
    this.poiId = const Value.absent(),
    this.phoneRaw = const Value.absent(),
    this.phoneE164 = const Value.absent(),
    this.tier = const Value.absent(),
    this.sourceTag = const Value.absent(),
  });
  PoiContactsCompanion.insert({
    this.id = const Value.absent(),
    required int poiId,
    required String phoneRaw,
    this.phoneE164 = const Value.absent(),
    this.tier = const Value.absent(),
    this.sourceTag = const Value.absent(),
  }) : poiId = Value(poiId),
       phoneRaw = Value(phoneRaw);
  static Insertable<PoiContact> custom({
    Expression<int>? id,
    Expression<int>? poiId,
    Expression<String>? phoneRaw,
    Expression<String>? phoneE164,
    Expression<String>? tier,
    Expression<String>? sourceTag,
  }) {
    return RawValuesInsertable({
      if (id != null) 'id': id,
      if (poiId != null) 'poi_id': poiId,
      if (phoneRaw != null) 'phone_raw': phoneRaw,
      if (phoneE164 != null) 'phone_e164': phoneE164,
      if (tier != null) 'tier': tier,
      if (sourceTag != null) 'source_tag': sourceTag,
    });
  }

  PoiContactsCompanion copyWith({
    Value<int>? id,
    Value<int>? poiId,
    Value<String>? phoneRaw,
    Value<String?>? phoneE164,
    Value<String>? tier,
    Value<String?>? sourceTag,
  }) {
    return PoiContactsCompanion(
      id: id ?? this.id,
      poiId: poiId ?? this.poiId,
      phoneRaw: phoneRaw ?? this.phoneRaw,
      phoneE164: phoneE164 ?? this.phoneE164,
      tier: tier ?? this.tier,
      sourceTag: sourceTag ?? this.sourceTag,
    );
  }

  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    if (id.present) {
      map['id'] = Variable<int>(id.value);
    }
    if (poiId.present) {
      map['poi_id'] = Variable<int>(poiId.value);
    }
    if (phoneRaw.present) {
      map['phone_raw'] = Variable<String>(phoneRaw.value);
    }
    if (phoneE164.present) {
      map['phone_e164'] = Variable<String>(phoneE164.value);
    }
    if (tier.present) {
      map['tier'] = Variable<String>(tier.value);
    }
    if (sourceTag.present) {
      map['source_tag'] = Variable<String>(sourceTag.value);
    }
    return map;
  }

  @override
  String toString() {
    return (StringBuffer('PoiContactsCompanion(')
          ..write('id: $id, ')
          ..write('poiId: $poiId, ')
          ..write('phoneRaw: $phoneRaw, ')
          ..write('phoneE164: $phoneE164, ')
          ..write('tier: $tier, ')
          ..write('sourceTag: $sourceTag')
          ..write(')'))
        .toString();
  }
}

class $ImportBatchesTable extends ImportBatches
    with TableInfo<$ImportBatchesTable, ImportBatche> {
  @override
  final GeneratedDatabase attachedDatabase;
  final String? _alias;
  $ImportBatchesTable(this.attachedDatabase, [this._alias]);
  static const VerificationMeta _idMeta = const VerificationMeta('id');
  @override
  late final GeneratedColumn<int> id = GeneratedColumn<int>(
    'id',
    aliasedName,
    false,
    hasAutoIncrement: true,
    type: DriftSqlType.int,
    requiredDuringInsert: false,
    defaultConstraints: GeneratedColumn.constraintIsAlways(
      'PRIMARY KEY AUTOINCREMENT',
    ),
  );
  static const VerificationMeta _tripIdMeta = const VerificationMeta('tripId');
  @override
  late final GeneratedColumn<int> tripId = GeneratedColumn<int>(
    'trip_id',
    aliasedName,
    true,
    type: DriftSqlType.int,
    requiredDuringInsert: false,
    defaultConstraints: GeneratedColumn.constraintIsAlways(
      'REFERENCES trips (id) ON DELETE CASCADE',
    ),
  );
  static const VerificationMeta _fileNameMeta = const VerificationMeta(
    'fileName',
  );
  @override
  late final GeneratedColumn<String> fileName = GeneratedColumn<String>(
    'file_name',
    aliasedName,
    false,
    type: DriftSqlType.string,
    requiredDuringInsert: true,
  );
  static const VerificationMeta _sheetNameMeta = const VerificationMeta(
    'sheetName',
  );
  @override
  late final GeneratedColumn<String> sheetName = GeneratedColumn<String>(
    'sheet_name',
    aliasedName,
    true,
    type: DriftSqlType.string,
    requiredDuringInsert: false,
  );
  static const VerificationMeta _rowsImportedMeta = const VerificationMeta(
    'rowsImported',
  );
  @override
  late final GeneratedColumn<int> rowsImported = GeneratedColumn<int>(
    'rows_imported',
    aliasedName,
    false,
    type: DriftSqlType.int,
    requiredDuringInsert: false,
    defaultValue: const Constant(0),
  );
  static const VerificationMeta _rowsSkippedMeta = const VerificationMeta(
    'rowsSkipped',
  );
  @override
  late final GeneratedColumn<int> rowsSkipped = GeneratedColumn<int>(
    'rows_skipped',
    aliasedName,
    false,
    type: DriftSqlType.int,
    requiredDuringInsert: false,
    defaultValue: const Constant(0),
  );
  static const VerificationMeta _importedAtMeta = const VerificationMeta(
    'importedAt',
  );
  @override
  late final GeneratedColumn<DateTime> importedAt = GeneratedColumn<DateTime>(
    'imported_at',
    aliasedName,
    false,
    type: DriftSqlType.dateTime,
    requiredDuringInsert: false,
    defaultValue: currentDateAndTime,
  );
  @override
  List<GeneratedColumn> get $columns => [
    id,
    tripId,
    fileName,
    sheetName,
    rowsImported,
    rowsSkipped,
    importedAt,
  ];
  @override
  String get aliasedName => _alias ?? actualTableName;
  @override
  String get actualTableName => $name;
  static const String $name = 'import_batches';
  @override
  VerificationContext validateIntegrity(
    Insertable<ImportBatche> instance, {
    bool isInserting = false,
  }) {
    final context = VerificationContext();
    final data = instance.toColumns(true);
    if (data.containsKey('id')) {
      context.handle(_idMeta, id.isAcceptableOrUnknown(data['id']!, _idMeta));
    }
    if (data.containsKey('trip_id')) {
      context.handle(
        _tripIdMeta,
        tripId.isAcceptableOrUnknown(data['trip_id']!, _tripIdMeta),
      );
    }
    if (data.containsKey('file_name')) {
      context.handle(
        _fileNameMeta,
        fileName.isAcceptableOrUnknown(data['file_name']!, _fileNameMeta),
      );
    } else if (isInserting) {
      context.missing(_fileNameMeta);
    }
    if (data.containsKey('sheet_name')) {
      context.handle(
        _sheetNameMeta,
        sheetName.isAcceptableOrUnknown(data['sheet_name']!, _sheetNameMeta),
      );
    }
    if (data.containsKey('rows_imported')) {
      context.handle(
        _rowsImportedMeta,
        rowsImported.isAcceptableOrUnknown(
          data['rows_imported']!,
          _rowsImportedMeta,
        ),
      );
    }
    if (data.containsKey('rows_skipped')) {
      context.handle(
        _rowsSkippedMeta,
        rowsSkipped.isAcceptableOrUnknown(
          data['rows_skipped']!,
          _rowsSkippedMeta,
        ),
      );
    }
    if (data.containsKey('imported_at')) {
      context.handle(
        _importedAtMeta,
        importedAt.isAcceptableOrUnknown(data['imported_at']!, _importedAtMeta),
      );
    }
    return context;
  }

  @override
  Set<GeneratedColumn> get $primaryKey => {id};
  @override
  ImportBatche map(Map<String, dynamic> data, {String? tablePrefix}) {
    final effectivePrefix = tablePrefix != null ? '$tablePrefix.' : '';
    return ImportBatche(
      id: attachedDatabase.typeMapping.read(
        DriftSqlType.int,
        data['${effectivePrefix}id'],
      )!,
      tripId: attachedDatabase.typeMapping.read(
        DriftSqlType.int,
        data['${effectivePrefix}trip_id'],
      ),
      fileName: attachedDatabase.typeMapping.read(
        DriftSqlType.string,
        data['${effectivePrefix}file_name'],
      )!,
      sheetName: attachedDatabase.typeMapping.read(
        DriftSqlType.string,
        data['${effectivePrefix}sheet_name'],
      ),
      rowsImported: attachedDatabase.typeMapping.read(
        DriftSqlType.int,
        data['${effectivePrefix}rows_imported'],
      )!,
      rowsSkipped: attachedDatabase.typeMapping.read(
        DriftSqlType.int,
        data['${effectivePrefix}rows_skipped'],
      )!,
      importedAt: attachedDatabase.typeMapping.read(
        DriftSqlType.dateTime,
        data['${effectivePrefix}imported_at'],
      )!,
    );
  }

  @override
  $ImportBatchesTable createAlias(String alias) {
    return $ImportBatchesTable(attachedDatabase, alias);
  }
}

class ImportBatche extends DataClass implements Insertable<ImportBatche> {
  final int id;
  final int? tripId;
  final String fileName;
  final String? sheetName;
  final int rowsImported;
  final int rowsSkipped;
  final DateTime importedAt;
  const ImportBatche({
    required this.id,
    this.tripId,
    required this.fileName,
    this.sheetName,
    required this.rowsImported,
    required this.rowsSkipped,
    required this.importedAt,
  });
  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    map['id'] = Variable<int>(id);
    if (!nullToAbsent || tripId != null) {
      map['trip_id'] = Variable<int>(tripId);
    }
    map['file_name'] = Variable<String>(fileName);
    if (!nullToAbsent || sheetName != null) {
      map['sheet_name'] = Variable<String>(sheetName);
    }
    map['rows_imported'] = Variable<int>(rowsImported);
    map['rows_skipped'] = Variable<int>(rowsSkipped);
    map['imported_at'] = Variable<DateTime>(importedAt);
    return map;
  }

  ImportBatchesCompanion toCompanion(bool nullToAbsent) {
    return ImportBatchesCompanion(
      id: Value(id),
      tripId: tripId == null && nullToAbsent
          ? const Value.absent()
          : Value(tripId),
      fileName: Value(fileName),
      sheetName: sheetName == null && nullToAbsent
          ? const Value.absent()
          : Value(sheetName),
      rowsImported: Value(rowsImported),
      rowsSkipped: Value(rowsSkipped),
      importedAt: Value(importedAt),
    );
  }

  factory ImportBatche.fromJson(
    Map<String, dynamic> json, {
    ValueSerializer? serializer,
  }) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return ImportBatche(
      id: serializer.fromJson<int>(json['id']),
      tripId: serializer.fromJson<int?>(json['tripId']),
      fileName: serializer.fromJson<String>(json['fileName']),
      sheetName: serializer.fromJson<String?>(json['sheetName']),
      rowsImported: serializer.fromJson<int>(json['rowsImported']),
      rowsSkipped: serializer.fromJson<int>(json['rowsSkipped']),
      importedAt: serializer.fromJson<DateTime>(json['importedAt']),
    );
  }
  @override
  Map<String, dynamic> toJson({ValueSerializer? serializer}) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return <String, dynamic>{
      'id': serializer.toJson<int>(id),
      'tripId': serializer.toJson<int?>(tripId),
      'fileName': serializer.toJson<String>(fileName),
      'sheetName': serializer.toJson<String?>(sheetName),
      'rowsImported': serializer.toJson<int>(rowsImported),
      'rowsSkipped': serializer.toJson<int>(rowsSkipped),
      'importedAt': serializer.toJson<DateTime>(importedAt),
    };
  }

  ImportBatche copyWith({
    int? id,
    Value<int?> tripId = const Value.absent(),
    String? fileName,
    Value<String?> sheetName = const Value.absent(),
    int? rowsImported,
    int? rowsSkipped,
    DateTime? importedAt,
  }) => ImportBatche(
    id: id ?? this.id,
    tripId: tripId.present ? tripId.value : this.tripId,
    fileName: fileName ?? this.fileName,
    sheetName: sheetName.present ? sheetName.value : this.sheetName,
    rowsImported: rowsImported ?? this.rowsImported,
    rowsSkipped: rowsSkipped ?? this.rowsSkipped,
    importedAt: importedAt ?? this.importedAt,
  );
  ImportBatche copyWithCompanion(ImportBatchesCompanion data) {
    return ImportBatche(
      id: data.id.present ? data.id.value : this.id,
      tripId: data.tripId.present ? data.tripId.value : this.tripId,
      fileName: data.fileName.present ? data.fileName.value : this.fileName,
      sheetName: data.sheetName.present ? data.sheetName.value : this.sheetName,
      rowsImported: data.rowsImported.present
          ? data.rowsImported.value
          : this.rowsImported,
      rowsSkipped: data.rowsSkipped.present
          ? data.rowsSkipped.value
          : this.rowsSkipped,
      importedAt: data.importedAt.present
          ? data.importedAt.value
          : this.importedAt,
    );
  }

  @override
  String toString() {
    return (StringBuffer('ImportBatche(')
          ..write('id: $id, ')
          ..write('tripId: $tripId, ')
          ..write('fileName: $fileName, ')
          ..write('sheetName: $sheetName, ')
          ..write('rowsImported: $rowsImported, ')
          ..write('rowsSkipped: $rowsSkipped, ')
          ..write('importedAt: $importedAt')
          ..write(')'))
        .toString();
  }

  @override
  int get hashCode => Object.hash(
    id,
    tripId,
    fileName,
    sheetName,
    rowsImported,
    rowsSkipped,
    importedAt,
  );
  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      (other is ImportBatche &&
          other.id == this.id &&
          other.tripId == this.tripId &&
          other.fileName == this.fileName &&
          other.sheetName == this.sheetName &&
          other.rowsImported == this.rowsImported &&
          other.rowsSkipped == this.rowsSkipped &&
          other.importedAt == this.importedAt);
}

class ImportBatchesCompanion extends UpdateCompanion<ImportBatche> {
  final Value<int> id;
  final Value<int?> tripId;
  final Value<String> fileName;
  final Value<String?> sheetName;
  final Value<int> rowsImported;
  final Value<int> rowsSkipped;
  final Value<DateTime> importedAt;
  const ImportBatchesCompanion({
    this.id = const Value.absent(),
    this.tripId = const Value.absent(),
    this.fileName = const Value.absent(),
    this.sheetName = const Value.absent(),
    this.rowsImported = const Value.absent(),
    this.rowsSkipped = const Value.absent(),
    this.importedAt = const Value.absent(),
  });
  ImportBatchesCompanion.insert({
    this.id = const Value.absent(),
    this.tripId = const Value.absent(),
    required String fileName,
    this.sheetName = const Value.absent(),
    this.rowsImported = const Value.absent(),
    this.rowsSkipped = const Value.absent(),
    this.importedAt = const Value.absent(),
  }) : fileName = Value(fileName);
  static Insertable<ImportBatche> custom({
    Expression<int>? id,
    Expression<int>? tripId,
    Expression<String>? fileName,
    Expression<String>? sheetName,
    Expression<int>? rowsImported,
    Expression<int>? rowsSkipped,
    Expression<DateTime>? importedAt,
  }) {
    return RawValuesInsertable({
      if (id != null) 'id': id,
      if (tripId != null) 'trip_id': tripId,
      if (fileName != null) 'file_name': fileName,
      if (sheetName != null) 'sheet_name': sheetName,
      if (rowsImported != null) 'rows_imported': rowsImported,
      if (rowsSkipped != null) 'rows_skipped': rowsSkipped,
      if (importedAt != null) 'imported_at': importedAt,
    });
  }

  ImportBatchesCompanion copyWith({
    Value<int>? id,
    Value<int?>? tripId,
    Value<String>? fileName,
    Value<String?>? sheetName,
    Value<int>? rowsImported,
    Value<int>? rowsSkipped,
    Value<DateTime>? importedAt,
  }) {
    return ImportBatchesCompanion(
      id: id ?? this.id,
      tripId: tripId ?? this.tripId,
      fileName: fileName ?? this.fileName,
      sheetName: sheetName ?? this.sheetName,
      rowsImported: rowsImported ?? this.rowsImported,
      rowsSkipped: rowsSkipped ?? this.rowsSkipped,
      importedAt: importedAt ?? this.importedAt,
    );
  }

  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    if (id.present) {
      map['id'] = Variable<int>(id.value);
    }
    if (tripId.present) {
      map['trip_id'] = Variable<int>(tripId.value);
    }
    if (fileName.present) {
      map['file_name'] = Variable<String>(fileName.value);
    }
    if (sheetName.present) {
      map['sheet_name'] = Variable<String>(sheetName.value);
    }
    if (rowsImported.present) {
      map['rows_imported'] = Variable<int>(rowsImported.value);
    }
    if (rowsSkipped.present) {
      map['rows_skipped'] = Variable<int>(rowsSkipped.value);
    }
    if (importedAt.present) {
      map['imported_at'] = Variable<DateTime>(importedAt.value);
    }
    return map;
  }

  @override
  String toString() {
    return (StringBuffer('ImportBatchesCompanion(')
          ..write('id: $id, ')
          ..write('tripId: $tripId, ')
          ..write('fileName: $fileName, ')
          ..write('sheetName: $sheetName, ')
          ..write('rowsImported: $rowsImported, ')
          ..write('rowsSkipped: $rowsSkipped, ')
          ..write('importedAt: $importedAt')
          ..write(')'))
        .toString();
  }
}

class $ContactsTable extends Contacts with TableInfo<$ContactsTable, Contact> {
  @override
  final GeneratedDatabase attachedDatabase;
  final String? _alias;
  $ContactsTable(this.attachedDatabase, [this._alias]);
  static const VerificationMeta _idMeta = const VerificationMeta('id');
  @override
  late final GeneratedColumn<int> id = GeneratedColumn<int>(
    'id',
    aliasedName,
    false,
    hasAutoIncrement: true,
    type: DriftSqlType.int,
    requiredDuringInsert: false,
    defaultConstraints: GeneratedColumn.constraintIsAlways(
      'PRIMARY KEY AUTOINCREMENT',
    ),
  );
  static const VerificationMeta _tripIdMeta = const VerificationMeta('tripId');
  @override
  late final GeneratedColumn<int> tripId = GeneratedColumn<int>(
    'trip_id',
    aliasedName,
    true,
    type: DriftSqlType.int,
    requiredDuringInsert: false,
    defaultConstraints: GeneratedColumn.constraintIsAlways(
      'REFERENCES trips (id) ON DELETE CASCADE',
    ),
  );
  static const VerificationMeta _stopIdMeta = const VerificationMeta('stopId');
  @override
  late final GeneratedColumn<int> stopId = GeneratedColumn<int>(
    'stop_id',
    aliasedName,
    true,
    type: DriftSqlType.int,
    requiredDuringInsert: false,
    defaultConstraints: GeneratedColumn.constraintIsAlways(
      'REFERENCES stops (id) ON DELETE SET NULL',
    ),
  );
  static const VerificationMeta _nameMeta = const VerificationMeta('name');
  @override
  late final GeneratedColumn<String> name = GeneratedColumn<String>(
    'name',
    aliasedName,
    false,
    type: DriftSqlType.string,
    requiredDuringInsert: true,
  );
  static const VerificationMeta _phoneRawMeta = const VerificationMeta(
    'phoneRaw',
  );
  @override
  late final GeneratedColumn<String> phoneRaw = GeneratedColumn<String>(
    'phone_raw',
    aliasedName,
    false,
    type: DriftSqlType.string,
    requiredDuringInsert: true,
  );
  static const VerificationMeta _phoneE164Meta = const VerificationMeta(
    'phoneE164',
  );
  @override
  late final GeneratedColumn<String> phoneE164 = GeneratedColumn<String>(
    'phone_e164',
    aliasedName,
    true,
    type: DriftSqlType.string,
    requiredDuringInsert: false,
  );
  static const VerificationMeta _noteMeta = const VerificationMeta('note');
  @override
  late final GeneratedColumn<String> note = GeneratedColumn<String>(
    'note',
    aliasedName,
    true,
    type: DriftSqlType.string,
    requiredDuringInsert: false,
  );
  static const VerificationMeta _categoryMeta = const VerificationMeta(
    'category',
  );
  @override
  late final GeneratedColumn<String> category = GeneratedColumn<String>(
    'category',
    aliasedName,
    false,
    type: DriftSqlType.string,
    requiredDuringInsert: false,
    defaultValue: const Constant('other'),
  );
  static const VerificationMeta _tierMeta = const VerificationMeta('tier');
  @override
  late final GeneratedColumn<String> tier = GeneratedColumn<String>(
    'tier',
    aliasedName,
    false,
    type: DriftSqlType.string,
    requiredDuringInsert: false,
    defaultValue: const Constant('userEntered'),
  );
  static const VerificationMeta _callConfirmedMeta = const VerificationMeta(
    'callConfirmed',
  );
  @override
  late final GeneratedColumn<bool> callConfirmed = GeneratedColumn<bool>(
    'call_confirmed',
    aliasedName,
    false,
    type: DriftSqlType.bool,
    requiredDuringInsert: false,
    defaultConstraints: GeneratedColumn.constraintIsAlways(
      'CHECK ("call_confirmed" IN (0, 1))',
    ),
    defaultValue: const Constant(false),
  );
  static const VerificationMeta _confirmedAtMeta = const VerificationMeta(
    'confirmedAt',
  );
  @override
  late final GeneratedColumn<DateTime> confirmedAt = GeneratedColumn<DateTime>(
    'confirmed_at',
    aliasedName,
    true,
    type: DriftSqlType.dateTime,
    requiredDuringInsert: false,
  );
  static const VerificationMeta _isPinnedMeta = const VerificationMeta(
    'isPinned',
  );
  @override
  late final GeneratedColumn<bool> isPinned = GeneratedColumn<bool>(
    'is_pinned',
    aliasedName,
    false,
    type: DriftSqlType.bool,
    requiredDuringInsert: false,
    defaultConstraints: GeneratedColumn.constraintIsAlways(
      'CHECK ("is_pinned" IN (0, 1))',
    ),
    defaultValue: const Constant(false),
  );
  static const VerificationMeta _isEmergencyMeta = const VerificationMeta(
    'isEmergency',
  );
  @override
  late final GeneratedColumn<bool> isEmergency = GeneratedColumn<bool>(
    'is_emergency',
    aliasedName,
    false,
    type: DriftSqlType.bool,
    requiredDuringInsert: false,
    defaultConstraints: GeneratedColumn.constraintIsAlways(
      'CHECK ("is_emergency" IN (0, 1))',
    ),
    defaultValue: const Constant(false),
  );
  static const VerificationMeta _hasWhatsappMeta = const VerificationMeta(
    'hasWhatsapp',
  );
  @override
  late final GeneratedColumn<bool> hasWhatsapp = GeneratedColumn<bool>(
    'has_whatsapp',
    aliasedName,
    false,
    type: DriftSqlType.bool,
    requiredDuringInsert: false,
    defaultConstraints: GeneratedColumn.constraintIsAlways(
      'CHECK ("has_whatsapp" IN (0, 1))',
    ),
    defaultValue: const Constant(false),
  );
  static const VerificationMeta _lastCalledAtMeta = const VerificationMeta(
    'lastCalledAt',
  );
  @override
  late final GeneratedColumn<DateTime> lastCalledAt = GeneratedColumn<DateTime>(
    'last_called_at',
    aliasedName,
    true,
    type: DriftSqlType.dateTime,
    requiredDuringInsert: false,
  );
  static const VerificationMeta _callCountMeta = const VerificationMeta(
    'callCount',
  );
  @override
  late final GeneratedColumn<int> callCount = GeneratedColumn<int>(
    'call_count',
    aliasedName,
    false,
    type: DriftSqlType.int,
    requiredDuringInsert: false,
    defaultValue: const Constant(0),
  );
  static const VerificationMeta _importBatchIdMeta = const VerificationMeta(
    'importBatchId',
  );
  @override
  late final GeneratedColumn<int> importBatchId = GeneratedColumn<int>(
    'import_batch_id',
    aliasedName,
    true,
    type: DriftSqlType.int,
    requiredDuringInsert: false,
    defaultConstraints: GeneratedColumn.constraintIsAlways(
      'REFERENCES import_batches (id) ON DELETE SET NULL',
    ),
  );
  static const VerificationMeta _createdAtMeta = const VerificationMeta(
    'createdAt',
  );
  @override
  late final GeneratedColumn<DateTime> createdAt = GeneratedColumn<DateTime>(
    'created_at',
    aliasedName,
    false,
    type: DriftSqlType.dateTime,
    requiredDuringInsert: false,
    defaultValue: currentDateAndTime,
  );
  @override
  List<GeneratedColumn> get $columns => [
    id,
    tripId,
    stopId,
    name,
    phoneRaw,
    phoneE164,
    note,
    category,
    tier,
    callConfirmed,
    confirmedAt,
    isPinned,
    isEmergency,
    hasWhatsapp,
    lastCalledAt,
    callCount,
    importBatchId,
    createdAt,
  ];
  @override
  String get aliasedName => _alias ?? actualTableName;
  @override
  String get actualTableName => $name;
  static const String $name = 'contacts';
  @override
  VerificationContext validateIntegrity(
    Insertable<Contact> instance, {
    bool isInserting = false,
  }) {
    final context = VerificationContext();
    final data = instance.toColumns(true);
    if (data.containsKey('id')) {
      context.handle(_idMeta, id.isAcceptableOrUnknown(data['id']!, _idMeta));
    }
    if (data.containsKey('trip_id')) {
      context.handle(
        _tripIdMeta,
        tripId.isAcceptableOrUnknown(data['trip_id']!, _tripIdMeta),
      );
    }
    if (data.containsKey('stop_id')) {
      context.handle(
        _stopIdMeta,
        stopId.isAcceptableOrUnknown(data['stop_id']!, _stopIdMeta),
      );
    }
    if (data.containsKey('name')) {
      context.handle(
        _nameMeta,
        name.isAcceptableOrUnknown(data['name']!, _nameMeta),
      );
    } else if (isInserting) {
      context.missing(_nameMeta);
    }
    if (data.containsKey('phone_raw')) {
      context.handle(
        _phoneRawMeta,
        phoneRaw.isAcceptableOrUnknown(data['phone_raw']!, _phoneRawMeta),
      );
    } else if (isInserting) {
      context.missing(_phoneRawMeta);
    }
    if (data.containsKey('phone_e164')) {
      context.handle(
        _phoneE164Meta,
        phoneE164.isAcceptableOrUnknown(data['phone_e164']!, _phoneE164Meta),
      );
    }
    if (data.containsKey('note')) {
      context.handle(
        _noteMeta,
        note.isAcceptableOrUnknown(data['note']!, _noteMeta),
      );
    }
    if (data.containsKey('category')) {
      context.handle(
        _categoryMeta,
        category.isAcceptableOrUnknown(data['category']!, _categoryMeta),
      );
    }
    if (data.containsKey('tier')) {
      context.handle(
        _tierMeta,
        tier.isAcceptableOrUnknown(data['tier']!, _tierMeta),
      );
    }
    if (data.containsKey('call_confirmed')) {
      context.handle(
        _callConfirmedMeta,
        callConfirmed.isAcceptableOrUnknown(
          data['call_confirmed']!,
          _callConfirmedMeta,
        ),
      );
    }
    if (data.containsKey('confirmed_at')) {
      context.handle(
        _confirmedAtMeta,
        confirmedAt.isAcceptableOrUnknown(
          data['confirmed_at']!,
          _confirmedAtMeta,
        ),
      );
    }
    if (data.containsKey('is_pinned')) {
      context.handle(
        _isPinnedMeta,
        isPinned.isAcceptableOrUnknown(data['is_pinned']!, _isPinnedMeta),
      );
    }
    if (data.containsKey('is_emergency')) {
      context.handle(
        _isEmergencyMeta,
        isEmergency.isAcceptableOrUnknown(
          data['is_emergency']!,
          _isEmergencyMeta,
        ),
      );
    }
    if (data.containsKey('has_whatsapp')) {
      context.handle(
        _hasWhatsappMeta,
        hasWhatsapp.isAcceptableOrUnknown(
          data['has_whatsapp']!,
          _hasWhatsappMeta,
        ),
      );
    }
    if (data.containsKey('last_called_at')) {
      context.handle(
        _lastCalledAtMeta,
        lastCalledAt.isAcceptableOrUnknown(
          data['last_called_at']!,
          _lastCalledAtMeta,
        ),
      );
    }
    if (data.containsKey('call_count')) {
      context.handle(
        _callCountMeta,
        callCount.isAcceptableOrUnknown(data['call_count']!, _callCountMeta),
      );
    }
    if (data.containsKey('import_batch_id')) {
      context.handle(
        _importBatchIdMeta,
        importBatchId.isAcceptableOrUnknown(
          data['import_batch_id']!,
          _importBatchIdMeta,
        ),
      );
    }
    if (data.containsKey('created_at')) {
      context.handle(
        _createdAtMeta,
        createdAt.isAcceptableOrUnknown(data['created_at']!, _createdAtMeta),
      );
    }
    return context;
  }

  @override
  Set<GeneratedColumn> get $primaryKey => {id};
  @override
  Contact map(Map<String, dynamic> data, {String? tablePrefix}) {
    final effectivePrefix = tablePrefix != null ? '$tablePrefix.' : '';
    return Contact(
      id: attachedDatabase.typeMapping.read(
        DriftSqlType.int,
        data['${effectivePrefix}id'],
      )!,
      tripId: attachedDatabase.typeMapping.read(
        DriftSqlType.int,
        data['${effectivePrefix}trip_id'],
      ),
      stopId: attachedDatabase.typeMapping.read(
        DriftSqlType.int,
        data['${effectivePrefix}stop_id'],
      ),
      name: attachedDatabase.typeMapping.read(
        DriftSqlType.string,
        data['${effectivePrefix}name'],
      )!,
      phoneRaw: attachedDatabase.typeMapping.read(
        DriftSqlType.string,
        data['${effectivePrefix}phone_raw'],
      )!,
      phoneE164: attachedDatabase.typeMapping.read(
        DriftSqlType.string,
        data['${effectivePrefix}phone_e164'],
      ),
      note: attachedDatabase.typeMapping.read(
        DriftSqlType.string,
        data['${effectivePrefix}note'],
      ),
      category: attachedDatabase.typeMapping.read(
        DriftSqlType.string,
        data['${effectivePrefix}category'],
      )!,
      tier: attachedDatabase.typeMapping.read(
        DriftSqlType.string,
        data['${effectivePrefix}tier'],
      )!,
      callConfirmed: attachedDatabase.typeMapping.read(
        DriftSqlType.bool,
        data['${effectivePrefix}call_confirmed'],
      )!,
      confirmedAt: attachedDatabase.typeMapping.read(
        DriftSqlType.dateTime,
        data['${effectivePrefix}confirmed_at'],
      ),
      isPinned: attachedDatabase.typeMapping.read(
        DriftSqlType.bool,
        data['${effectivePrefix}is_pinned'],
      )!,
      isEmergency: attachedDatabase.typeMapping.read(
        DriftSqlType.bool,
        data['${effectivePrefix}is_emergency'],
      )!,
      hasWhatsapp: attachedDatabase.typeMapping.read(
        DriftSqlType.bool,
        data['${effectivePrefix}has_whatsapp'],
      )!,
      lastCalledAt: attachedDatabase.typeMapping.read(
        DriftSqlType.dateTime,
        data['${effectivePrefix}last_called_at'],
      ),
      callCount: attachedDatabase.typeMapping.read(
        DriftSqlType.int,
        data['${effectivePrefix}call_count'],
      )!,
      importBatchId: attachedDatabase.typeMapping.read(
        DriftSqlType.int,
        data['${effectivePrefix}import_batch_id'],
      ),
      createdAt: attachedDatabase.typeMapping.read(
        DriftSqlType.dateTime,
        data['${effectivePrefix}created_at'],
      )!,
    );
  }

  @override
  $ContactsTable createAlias(String alias) {
    return $ContactsTable(attachedDatabase, alias);
  }
}

class Contact extends DataClass implements Insertable<Contact> {
  final int id;
  final int? tripId;
  final int? stopId;
  final String name;

  /// What the user typed, shown back verbatim. Reformatting someone's own
  /// input is confusing, so the raw value is what the diary displays.
  final String phoneRaw;

  /// Normalised, for dialing and duplicate detection. Nullable because
  /// normalisation genuinely fails on bad input and the raw value must
  /// survive that.
  final String? phoneE164;
  final String? note;
  final String category;

  /// THE CORE INVARIANT. Defaults to userEntered. Nothing may reach this
  /// table already verified — not an import, not a POI save, not a form.
  final String tier;
  final bool callConfirmed;
  final DateTime? confirmedAt;
  final bool isPinned;
  final bool isEmergency;
  final bool hasWhatsapp;
  final DateTime? lastCalledAt;
  final int callCount;
  final int? importBatchId;
  final DateTime createdAt;
  const Contact({
    required this.id,
    this.tripId,
    this.stopId,
    required this.name,
    required this.phoneRaw,
    this.phoneE164,
    this.note,
    required this.category,
    required this.tier,
    required this.callConfirmed,
    this.confirmedAt,
    required this.isPinned,
    required this.isEmergency,
    required this.hasWhatsapp,
    this.lastCalledAt,
    required this.callCount,
    this.importBatchId,
    required this.createdAt,
  });
  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    map['id'] = Variable<int>(id);
    if (!nullToAbsent || tripId != null) {
      map['trip_id'] = Variable<int>(tripId);
    }
    if (!nullToAbsent || stopId != null) {
      map['stop_id'] = Variable<int>(stopId);
    }
    map['name'] = Variable<String>(name);
    map['phone_raw'] = Variable<String>(phoneRaw);
    if (!nullToAbsent || phoneE164 != null) {
      map['phone_e164'] = Variable<String>(phoneE164);
    }
    if (!nullToAbsent || note != null) {
      map['note'] = Variable<String>(note);
    }
    map['category'] = Variable<String>(category);
    map['tier'] = Variable<String>(tier);
    map['call_confirmed'] = Variable<bool>(callConfirmed);
    if (!nullToAbsent || confirmedAt != null) {
      map['confirmed_at'] = Variable<DateTime>(confirmedAt);
    }
    map['is_pinned'] = Variable<bool>(isPinned);
    map['is_emergency'] = Variable<bool>(isEmergency);
    map['has_whatsapp'] = Variable<bool>(hasWhatsapp);
    if (!nullToAbsent || lastCalledAt != null) {
      map['last_called_at'] = Variable<DateTime>(lastCalledAt);
    }
    map['call_count'] = Variable<int>(callCount);
    if (!nullToAbsent || importBatchId != null) {
      map['import_batch_id'] = Variable<int>(importBatchId);
    }
    map['created_at'] = Variable<DateTime>(createdAt);
    return map;
  }

  ContactsCompanion toCompanion(bool nullToAbsent) {
    return ContactsCompanion(
      id: Value(id),
      tripId: tripId == null && nullToAbsent
          ? const Value.absent()
          : Value(tripId),
      stopId: stopId == null && nullToAbsent
          ? const Value.absent()
          : Value(stopId),
      name: Value(name),
      phoneRaw: Value(phoneRaw),
      phoneE164: phoneE164 == null && nullToAbsent
          ? const Value.absent()
          : Value(phoneE164),
      note: note == null && nullToAbsent ? const Value.absent() : Value(note),
      category: Value(category),
      tier: Value(tier),
      callConfirmed: Value(callConfirmed),
      confirmedAt: confirmedAt == null && nullToAbsent
          ? const Value.absent()
          : Value(confirmedAt),
      isPinned: Value(isPinned),
      isEmergency: Value(isEmergency),
      hasWhatsapp: Value(hasWhatsapp),
      lastCalledAt: lastCalledAt == null && nullToAbsent
          ? const Value.absent()
          : Value(lastCalledAt),
      callCount: Value(callCount),
      importBatchId: importBatchId == null && nullToAbsent
          ? const Value.absent()
          : Value(importBatchId),
      createdAt: Value(createdAt),
    );
  }

  factory Contact.fromJson(
    Map<String, dynamic> json, {
    ValueSerializer? serializer,
  }) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return Contact(
      id: serializer.fromJson<int>(json['id']),
      tripId: serializer.fromJson<int?>(json['tripId']),
      stopId: serializer.fromJson<int?>(json['stopId']),
      name: serializer.fromJson<String>(json['name']),
      phoneRaw: serializer.fromJson<String>(json['phoneRaw']),
      phoneE164: serializer.fromJson<String?>(json['phoneE164']),
      note: serializer.fromJson<String?>(json['note']),
      category: serializer.fromJson<String>(json['category']),
      tier: serializer.fromJson<String>(json['tier']),
      callConfirmed: serializer.fromJson<bool>(json['callConfirmed']),
      confirmedAt: serializer.fromJson<DateTime?>(json['confirmedAt']),
      isPinned: serializer.fromJson<bool>(json['isPinned']),
      isEmergency: serializer.fromJson<bool>(json['isEmergency']),
      hasWhatsapp: serializer.fromJson<bool>(json['hasWhatsapp']),
      lastCalledAt: serializer.fromJson<DateTime?>(json['lastCalledAt']),
      callCount: serializer.fromJson<int>(json['callCount']),
      importBatchId: serializer.fromJson<int?>(json['importBatchId']),
      createdAt: serializer.fromJson<DateTime>(json['createdAt']),
    );
  }
  @override
  Map<String, dynamic> toJson({ValueSerializer? serializer}) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return <String, dynamic>{
      'id': serializer.toJson<int>(id),
      'tripId': serializer.toJson<int?>(tripId),
      'stopId': serializer.toJson<int?>(stopId),
      'name': serializer.toJson<String>(name),
      'phoneRaw': serializer.toJson<String>(phoneRaw),
      'phoneE164': serializer.toJson<String?>(phoneE164),
      'note': serializer.toJson<String?>(note),
      'category': serializer.toJson<String>(category),
      'tier': serializer.toJson<String>(tier),
      'callConfirmed': serializer.toJson<bool>(callConfirmed),
      'confirmedAt': serializer.toJson<DateTime?>(confirmedAt),
      'isPinned': serializer.toJson<bool>(isPinned),
      'isEmergency': serializer.toJson<bool>(isEmergency),
      'hasWhatsapp': serializer.toJson<bool>(hasWhatsapp),
      'lastCalledAt': serializer.toJson<DateTime?>(lastCalledAt),
      'callCount': serializer.toJson<int>(callCount),
      'importBatchId': serializer.toJson<int?>(importBatchId),
      'createdAt': serializer.toJson<DateTime>(createdAt),
    };
  }

  Contact copyWith({
    int? id,
    Value<int?> tripId = const Value.absent(),
    Value<int?> stopId = const Value.absent(),
    String? name,
    String? phoneRaw,
    Value<String?> phoneE164 = const Value.absent(),
    Value<String?> note = const Value.absent(),
    String? category,
    String? tier,
    bool? callConfirmed,
    Value<DateTime?> confirmedAt = const Value.absent(),
    bool? isPinned,
    bool? isEmergency,
    bool? hasWhatsapp,
    Value<DateTime?> lastCalledAt = const Value.absent(),
    int? callCount,
    Value<int?> importBatchId = const Value.absent(),
    DateTime? createdAt,
  }) => Contact(
    id: id ?? this.id,
    tripId: tripId.present ? tripId.value : this.tripId,
    stopId: stopId.present ? stopId.value : this.stopId,
    name: name ?? this.name,
    phoneRaw: phoneRaw ?? this.phoneRaw,
    phoneE164: phoneE164.present ? phoneE164.value : this.phoneE164,
    note: note.present ? note.value : this.note,
    category: category ?? this.category,
    tier: tier ?? this.tier,
    callConfirmed: callConfirmed ?? this.callConfirmed,
    confirmedAt: confirmedAt.present ? confirmedAt.value : this.confirmedAt,
    isPinned: isPinned ?? this.isPinned,
    isEmergency: isEmergency ?? this.isEmergency,
    hasWhatsapp: hasWhatsapp ?? this.hasWhatsapp,
    lastCalledAt: lastCalledAt.present ? lastCalledAt.value : this.lastCalledAt,
    callCount: callCount ?? this.callCount,
    importBatchId: importBatchId.present
        ? importBatchId.value
        : this.importBatchId,
    createdAt: createdAt ?? this.createdAt,
  );
  Contact copyWithCompanion(ContactsCompanion data) {
    return Contact(
      id: data.id.present ? data.id.value : this.id,
      tripId: data.tripId.present ? data.tripId.value : this.tripId,
      stopId: data.stopId.present ? data.stopId.value : this.stopId,
      name: data.name.present ? data.name.value : this.name,
      phoneRaw: data.phoneRaw.present ? data.phoneRaw.value : this.phoneRaw,
      phoneE164: data.phoneE164.present ? data.phoneE164.value : this.phoneE164,
      note: data.note.present ? data.note.value : this.note,
      category: data.category.present ? data.category.value : this.category,
      tier: data.tier.present ? data.tier.value : this.tier,
      callConfirmed: data.callConfirmed.present
          ? data.callConfirmed.value
          : this.callConfirmed,
      confirmedAt: data.confirmedAt.present
          ? data.confirmedAt.value
          : this.confirmedAt,
      isPinned: data.isPinned.present ? data.isPinned.value : this.isPinned,
      isEmergency: data.isEmergency.present
          ? data.isEmergency.value
          : this.isEmergency,
      hasWhatsapp: data.hasWhatsapp.present
          ? data.hasWhatsapp.value
          : this.hasWhatsapp,
      lastCalledAt: data.lastCalledAt.present
          ? data.lastCalledAt.value
          : this.lastCalledAt,
      callCount: data.callCount.present ? data.callCount.value : this.callCount,
      importBatchId: data.importBatchId.present
          ? data.importBatchId.value
          : this.importBatchId,
      createdAt: data.createdAt.present ? data.createdAt.value : this.createdAt,
    );
  }

  @override
  String toString() {
    return (StringBuffer('Contact(')
          ..write('id: $id, ')
          ..write('tripId: $tripId, ')
          ..write('stopId: $stopId, ')
          ..write('name: $name, ')
          ..write('phoneRaw: $phoneRaw, ')
          ..write('phoneE164: $phoneE164, ')
          ..write('note: $note, ')
          ..write('category: $category, ')
          ..write('tier: $tier, ')
          ..write('callConfirmed: $callConfirmed, ')
          ..write('confirmedAt: $confirmedAt, ')
          ..write('isPinned: $isPinned, ')
          ..write('isEmergency: $isEmergency, ')
          ..write('hasWhatsapp: $hasWhatsapp, ')
          ..write('lastCalledAt: $lastCalledAt, ')
          ..write('callCount: $callCount, ')
          ..write('importBatchId: $importBatchId, ')
          ..write('createdAt: $createdAt')
          ..write(')'))
        .toString();
  }

  @override
  int get hashCode => Object.hash(
    id,
    tripId,
    stopId,
    name,
    phoneRaw,
    phoneE164,
    note,
    category,
    tier,
    callConfirmed,
    confirmedAt,
    isPinned,
    isEmergency,
    hasWhatsapp,
    lastCalledAt,
    callCount,
    importBatchId,
    createdAt,
  );
  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      (other is Contact &&
          other.id == this.id &&
          other.tripId == this.tripId &&
          other.stopId == this.stopId &&
          other.name == this.name &&
          other.phoneRaw == this.phoneRaw &&
          other.phoneE164 == this.phoneE164 &&
          other.note == this.note &&
          other.category == this.category &&
          other.tier == this.tier &&
          other.callConfirmed == this.callConfirmed &&
          other.confirmedAt == this.confirmedAt &&
          other.isPinned == this.isPinned &&
          other.isEmergency == this.isEmergency &&
          other.hasWhatsapp == this.hasWhatsapp &&
          other.lastCalledAt == this.lastCalledAt &&
          other.callCount == this.callCount &&
          other.importBatchId == this.importBatchId &&
          other.createdAt == this.createdAt);
}

class ContactsCompanion extends UpdateCompanion<Contact> {
  final Value<int> id;
  final Value<int?> tripId;
  final Value<int?> stopId;
  final Value<String> name;
  final Value<String> phoneRaw;
  final Value<String?> phoneE164;
  final Value<String?> note;
  final Value<String> category;
  final Value<String> tier;
  final Value<bool> callConfirmed;
  final Value<DateTime?> confirmedAt;
  final Value<bool> isPinned;
  final Value<bool> isEmergency;
  final Value<bool> hasWhatsapp;
  final Value<DateTime?> lastCalledAt;
  final Value<int> callCount;
  final Value<int?> importBatchId;
  final Value<DateTime> createdAt;
  const ContactsCompanion({
    this.id = const Value.absent(),
    this.tripId = const Value.absent(),
    this.stopId = const Value.absent(),
    this.name = const Value.absent(),
    this.phoneRaw = const Value.absent(),
    this.phoneE164 = const Value.absent(),
    this.note = const Value.absent(),
    this.category = const Value.absent(),
    this.tier = const Value.absent(),
    this.callConfirmed = const Value.absent(),
    this.confirmedAt = const Value.absent(),
    this.isPinned = const Value.absent(),
    this.isEmergency = const Value.absent(),
    this.hasWhatsapp = const Value.absent(),
    this.lastCalledAt = const Value.absent(),
    this.callCount = const Value.absent(),
    this.importBatchId = const Value.absent(),
    this.createdAt = const Value.absent(),
  });
  ContactsCompanion.insert({
    this.id = const Value.absent(),
    this.tripId = const Value.absent(),
    this.stopId = const Value.absent(),
    required String name,
    required String phoneRaw,
    this.phoneE164 = const Value.absent(),
    this.note = const Value.absent(),
    this.category = const Value.absent(),
    this.tier = const Value.absent(),
    this.callConfirmed = const Value.absent(),
    this.confirmedAt = const Value.absent(),
    this.isPinned = const Value.absent(),
    this.isEmergency = const Value.absent(),
    this.hasWhatsapp = const Value.absent(),
    this.lastCalledAt = const Value.absent(),
    this.callCount = const Value.absent(),
    this.importBatchId = const Value.absent(),
    this.createdAt = const Value.absent(),
  }) : name = Value(name),
       phoneRaw = Value(phoneRaw);
  static Insertable<Contact> custom({
    Expression<int>? id,
    Expression<int>? tripId,
    Expression<int>? stopId,
    Expression<String>? name,
    Expression<String>? phoneRaw,
    Expression<String>? phoneE164,
    Expression<String>? note,
    Expression<String>? category,
    Expression<String>? tier,
    Expression<bool>? callConfirmed,
    Expression<DateTime>? confirmedAt,
    Expression<bool>? isPinned,
    Expression<bool>? isEmergency,
    Expression<bool>? hasWhatsapp,
    Expression<DateTime>? lastCalledAt,
    Expression<int>? callCount,
    Expression<int>? importBatchId,
    Expression<DateTime>? createdAt,
  }) {
    return RawValuesInsertable({
      if (id != null) 'id': id,
      if (tripId != null) 'trip_id': tripId,
      if (stopId != null) 'stop_id': stopId,
      if (name != null) 'name': name,
      if (phoneRaw != null) 'phone_raw': phoneRaw,
      if (phoneE164 != null) 'phone_e164': phoneE164,
      if (note != null) 'note': note,
      if (category != null) 'category': category,
      if (tier != null) 'tier': tier,
      if (callConfirmed != null) 'call_confirmed': callConfirmed,
      if (confirmedAt != null) 'confirmed_at': confirmedAt,
      if (isPinned != null) 'is_pinned': isPinned,
      if (isEmergency != null) 'is_emergency': isEmergency,
      if (hasWhatsapp != null) 'has_whatsapp': hasWhatsapp,
      if (lastCalledAt != null) 'last_called_at': lastCalledAt,
      if (callCount != null) 'call_count': callCount,
      if (importBatchId != null) 'import_batch_id': importBatchId,
      if (createdAt != null) 'created_at': createdAt,
    });
  }

  ContactsCompanion copyWith({
    Value<int>? id,
    Value<int?>? tripId,
    Value<int?>? stopId,
    Value<String>? name,
    Value<String>? phoneRaw,
    Value<String?>? phoneE164,
    Value<String?>? note,
    Value<String>? category,
    Value<String>? tier,
    Value<bool>? callConfirmed,
    Value<DateTime?>? confirmedAt,
    Value<bool>? isPinned,
    Value<bool>? isEmergency,
    Value<bool>? hasWhatsapp,
    Value<DateTime?>? lastCalledAt,
    Value<int>? callCount,
    Value<int?>? importBatchId,
    Value<DateTime>? createdAt,
  }) {
    return ContactsCompanion(
      id: id ?? this.id,
      tripId: tripId ?? this.tripId,
      stopId: stopId ?? this.stopId,
      name: name ?? this.name,
      phoneRaw: phoneRaw ?? this.phoneRaw,
      phoneE164: phoneE164 ?? this.phoneE164,
      note: note ?? this.note,
      category: category ?? this.category,
      tier: tier ?? this.tier,
      callConfirmed: callConfirmed ?? this.callConfirmed,
      confirmedAt: confirmedAt ?? this.confirmedAt,
      isPinned: isPinned ?? this.isPinned,
      isEmergency: isEmergency ?? this.isEmergency,
      hasWhatsapp: hasWhatsapp ?? this.hasWhatsapp,
      lastCalledAt: lastCalledAt ?? this.lastCalledAt,
      callCount: callCount ?? this.callCount,
      importBatchId: importBatchId ?? this.importBatchId,
      createdAt: createdAt ?? this.createdAt,
    );
  }

  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    if (id.present) {
      map['id'] = Variable<int>(id.value);
    }
    if (tripId.present) {
      map['trip_id'] = Variable<int>(tripId.value);
    }
    if (stopId.present) {
      map['stop_id'] = Variable<int>(stopId.value);
    }
    if (name.present) {
      map['name'] = Variable<String>(name.value);
    }
    if (phoneRaw.present) {
      map['phone_raw'] = Variable<String>(phoneRaw.value);
    }
    if (phoneE164.present) {
      map['phone_e164'] = Variable<String>(phoneE164.value);
    }
    if (note.present) {
      map['note'] = Variable<String>(note.value);
    }
    if (category.present) {
      map['category'] = Variable<String>(category.value);
    }
    if (tier.present) {
      map['tier'] = Variable<String>(tier.value);
    }
    if (callConfirmed.present) {
      map['call_confirmed'] = Variable<bool>(callConfirmed.value);
    }
    if (confirmedAt.present) {
      map['confirmed_at'] = Variable<DateTime>(confirmedAt.value);
    }
    if (isPinned.present) {
      map['is_pinned'] = Variable<bool>(isPinned.value);
    }
    if (isEmergency.present) {
      map['is_emergency'] = Variable<bool>(isEmergency.value);
    }
    if (hasWhatsapp.present) {
      map['has_whatsapp'] = Variable<bool>(hasWhatsapp.value);
    }
    if (lastCalledAt.present) {
      map['last_called_at'] = Variable<DateTime>(lastCalledAt.value);
    }
    if (callCount.present) {
      map['call_count'] = Variable<int>(callCount.value);
    }
    if (importBatchId.present) {
      map['import_batch_id'] = Variable<int>(importBatchId.value);
    }
    if (createdAt.present) {
      map['created_at'] = Variable<DateTime>(createdAt.value);
    }
    return map;
  }

  @override
  String toString() {
    return (StringBuffer('ContactsCompanion(')
          ..write('id: $id, ')
          ..write('tripId: $tripId, ')
          ..write('stopId: $stopId, ')
          ..write('name: $name, ')
          ..write('phoneRaw: $phoneRaw, ')
          ..write('phoneE164: $phoneE164, ')
          ..write('note: $note, ')
          ..write('category: $category, ')
          ..write('tier: $tier, ')
          ..write('callConfirmed: $callConfirmed, ')
          ..write('confirmedAt: $confirmedAt, ')
          ..write('isPinned: $isPinned, ')
          ..write('isEmergency: $isEmergency, ')
          ..write('hasWhatsapp: $hasWhatsapp, ')
          ..write('lastCalledAt: $lastCalledAt, ')
          ..write('callCount: $callCount, ')
          ..write('importBatchId: $importBatchId, ')
          ..write('createdAt: $createdAt')
          ..write(')'))
        .toString();
  }
}

class $CallLogsTable extends CallLogs with TableInfo<$CallLogsTable, CallLog> {
  @override
  final GeneratedDatabase attachedDatabase;
  final String? _alias;
  $CallLogsTable(this.attachedDatabase, [this._alias]);
  static const VerificationMeta _idMeta = const VerificationMeta('id');
  @override
  late final GeneratedColumn<int> id = GeneratedColumn<int>(
    'id',
    aliasedName,
    false,
    hasAutoIncrement: true,
    type: DriftSqlType.int,
    requiredDuringInsert: false,
    defaultConstraints: GeneratedColumn.constraintIsAlways(
      'PRIMARY KEY AUTOINCREMENT',
    ),
  );
  static const VerificationMeta _contactIdMeta = const VerificationMeta(
    'contactId',
  );
  @override
  late final GeneratedColumn<int> contactId = GeneratedColumn<int>(
    'contact_id',
    aliasedName,
    false,
    type: DriftSqlType.int,
    requiredDuringInsert: true,
    defaultConstraints: GeneratedColumn.constraintIsAlways(
      'REFERENCES contacts (id) ON DELETE CASCADE',
    ),
  );
  static const VerificationMeta _tripIdMeta = const VerificationMeta('tripId');
  @override
  late final GeneratedColumn<int> tripId = GeneratedColumn<int>(
    'trip_id',
    aliasedName,
    true,
    type: DriftSqlType.int,
    requiredDuringInsert: false,
    defaultConstraints: GeneratedColumn.constraintIsAlways(
      'REFERENCES trips (id) ON DELETE CASCADE',
    ),
  );
  static const VerificationMeta _actionMeta = const VerificationMeta('action');
  @override
  late final GeneratedColumn<String> action = GeneratedColumn<String>(
    'action',
    aliasedName,
    false,
    type: DriftSqlType.string,
    requiredDuringInsert: true,
  );
  static const VerificationMeta _occurredAtMeta = const VerificationMeta(
    'occurredAt',
  );
  @override
  late final GeneratedColumn<DateTime> occurredAt = GeneratedColumn<DateTime>(
    'occurred_at',
    aliasedName,
    false,
    type: DriftSqlType.dateTime,
    requiredDuringInsert: false,
    defaultValue: currentDateAndTime,
  );
  @override
  List<GeneratedColumn> get $columns => [
    id,
    contactId,
    tripId,
    action,
    occurredAt,
  ];
  @override
  String get aliasedName => _alias ?? actualTableName;
  @override
  String get actualTableName => $name;
  static const String $name = 'call_logs';
  @override
  VerificationContext validateIntegrity(
    Insertable<CallLog> instance, {
    bool isInserting = false,
  }) {
    final context = VerificationContext();
    final data = instance.toColumns(true);
    if (data.containsKey('id')) {
      context.handle(_idMeta, id.isAcceptableOrUnknown(data['id']!, _idMeta));
    }
    if (data.containsKey('contact_id')) {
      context.handle(
        _contactIdMeta,
        contactId.isAcceptableOrUnknown(data['contact_id']!, _contactIdMeta),
      );
    } else if (isInserting) {
      context.missing(_contactIdMeta);
    }
    if (data.containsKey('trip_id')) {
      context.handle(
        _tripIdMeta,
        tripId.isAcceptableOrUnknown(data['trip_id']!, _tripIdMeta),
      );
    }
    if (data.containsKey('action')) {
      context.handle(
        _actionMeta,
        action.isAcceptableOrUnknown(data['action']!, _actionMeta),
      );
    } else if (isInserting) {
      context.missing(_actionMeta);
    }
    if (data.containsKey('occurred_at')) {
      context.handle(
        _occurredAtMeta,
        occurredAt.isAcceptableOrUnknown(data['occurred_at']!, _occurredAtMeta),
      );
    }
    return context;
  }

  @override
  Set<GeneratedColumn> get $primaryKey => {id};
  @override
  CallLog map(Map<String, dynamic> data, {String? tablePrefix}) {
    final effectivePrefix = tablePrefix != null ? '$tablePrefix.' : '';
    return CallLog(
      id: attachedDatabase.typeMapping.read(
        DriftSqlType.int,
        data['${effectivePrefix}id'],
      )!,
      contactId: attachedDatabase.typeMapping.read(
        DriftSqlType.int,
        data['${effectivePrefix}contact_id'],
      )!,
      tripId: attachedDatabase.typeMapping.read(
        DriftSqlType.int,
        data['${effectivePrefix}trip_id'],
      ),
      action: attachedDatabase.typeMapping.read(
        DriftSqlType.string,
        data['${effectivePrefix}action'],
      )!,
      occurredAt: attachedDatabase.typeMapping.read(
        DriftSqlType.dateTime,
        data['${effectivePrefix}occurred_at'],
      )!,
    );
  }

  @override
  $CallLogsTable createAlias(String alias) {
    return $CallLogsTable(attachedDatabase, alias);
  }
}

class CallLog extends DataClass implements Insertable<CallLog> {
  final int id;
  final int contactId;
  final int? tripId;

  /// copy | call | dialer | sms | whatsapp
  final String action;
  final DateTime occurredAt;
  const CallLog({
    required this.id,
    required this.contactId,
    this.tripId,
    required this.action,
    required this.occurredAt,
  });
  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    map['id'] = Variable<int>(id);
    map['contact_id'] = Variable<int>(contactId);
    if (!nullToAbsent || tripId != null) {
      map['trip_id'] = Variable<int>(tripId);
    }
    map['action'] = Variable<String>(action);
    map['occurred_at'] = Variable<DateTime>(occurredAt);
    return map;
  }

  CallLogsCompanion toCompanion(bool nullToAbsent) {
    return CallLogsCompanion(
      id: Value(id),
      contactId: Value(contactId),
      tripId: tripId == null && nullToAbsent
          ? const Value.absent()
          : Value(tripId),
      action: Value(action),
      occurredAt: Value(occurredAt),
    );
  }

  factory CallLog.fromJson(
    Map<String, dynamic> json, {
    ValueSerializer? serializer,
  }) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return CallLog(
      id: serializer.fromJson<int>(json['id']),
      contactId: serializer.fromJson<int>(json['contactId']),
      tripId: serializer.fromJson<int?>(json['tripId']),
      action: serializer.fromJson<String>(json['action']),
      occurredAt: serializer.fromJson<DateTime>(json['occurredAt']),
    );
  }
  @override
  Map<String, dynamic> toJson({ValueSerializer? serializer}) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return <String, dynamic>{
      'id': serializer.toJson<int>(id),
      'contactId': serializer.toJson<int>(contactId),
      'tripId': serializer.toJson<int?>(tripId),
      'action': serializer.toJson<String>(action),
      'occurredAt': serializer.toJson<DateTime>(occurredAt),
    };
  }

  CallLog copyWith({
    int? id,
    int? contactId,
    Value<int?> tripId = const Value.absent(),
    String? action,
    DateTime? occurredAt,
  }) => CallLog(
    id: id ?? this.id,
    contactId: contactId ?? this.contactId,
    tripId: tripId.present ? tripId.value : this.tripId,
    action: action ?? this.action,
    occurredAt: occurredAt ?? this.occurredAt,
  );
  CallLog copyWithCompanion(CallLogsCompanion data) {
    return CallLog(
      id: data.id.present ? data.id.value : this.id,
      contactId: data.contactId.present ? data.contactId.value : this.contactId,
      tripId: data.tripId.present ? data.tripId.value : this.tripId,
      action: data.action.present ? data.action.value : this.action,
      occurredAt: data.occurredAt.present
          ? data.occurredAt.value
          : this.occurredAt,
    );
  }

  @override
  String toString() {
    return (StringBuffer('CallLog(')
          ..write('id: $id, ')
          ..write('contactId: $contactId, ')
          ..write('tripId: $tripId, ')
          ..write('action: $action, ')
          ..write('occurredAt: $occurredAt')
          ..write(')'))
        .toString();
  }

  @override
  int get hashCode => Object.hash(id, contactId, tripId, action, occurredAt);
  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      (other is CallLog &&
          other.id == this.id &&
          other.contactId == this.contactId &&
          other.tripId == this.tripId &&
          other.action == this.action &&
          other.occurredAt == this.occurredAt);
}

class CallLogsCompanion extends UpdateCompanion<CallLog> {
  final Value<int> id;
  final Value<int> contactId;
  final Value<int?> tripId;
  final Value<String> action;
  final Value<DateTime> occurredAt;
  const CallLogsCompanion({
    this.id = const Value.absent(),
    this.contactId = const Value.absent(),
    this.tripId = const Value.absent(),
    this.action = const Value.absent(),
    this.occurredAt = const Value.absent(),
  });
  CallLogsCompanion.insert({
    this.id = const Value.absent(),
    required int contactId,
    this.tripId = const Value.absent(),
    required String action,
    this.occurredAt = const Value.absent(),
  }) : contactId = Value(contactId),
       action = Value(action);
  static Insertable<CallLog> custom({
    Expression<int>? id,
    Expression<int>? contactId,
    Expression<int>? tripId,
    Expression<String>? action,
    Expression<DateTime>? occurredAt,
  }) {
    return RawValuesInsertable({
      if (id != null) 'id': id,
      if (contactId != null) 'contact_id': contactId,
      if (tripId != null) 'trip_id': tripId,
      if (action != null) 'action': action,
      if (occurredAt != null) 'occurred_at': occurredAt,
    });
  }

  CallLogsCompanion copyWith({
    Value<int>? id,
    Value<int>? contactId,
    Value<int?>? tripId,
    Value<String>? action,
    Value<DateTime>? occurredAt,
  }) {
    return CallLogsCompanion(
      id: id ?? this.id,
      contactId: contactId ?? this.contactId,
      tripId: tripId ?? this.tripId,
      action: action ?? this.action,
      occurredAt: occurredAt ?? this.occurredAt,
    );
  }

  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    if (id.present) {
      map['id'] = Variable<int>(id.value);
    }
    if (contactId.present) {
      map['contact_id'] = Variable<int>(contactId.value);
    }
    if (tripId.present) {
      map['trip_id'] = Variable<int>(tripId.value);
    }
    if (action.present) {
      map['action'] = Variable<String>(action.value);
    }
    if (occurredAt.present) {
      map['occurred_at'] = Variable<DateTime>(occurredAt.value);
    }
    return map;
  }

  @override
  String toString() {
    return (StringBuffer('CallLogsCompanion(')
          ..write('id: $id, ')
          ..write('contactId: $contactId, ')
          ..write('tripId: $tripId, ')
          ..write('action: $action, ')
          ..write('occurredAt: $occurredAt')
          ..write(')'))
        .toString();
  }
}

class $EmergencyHelplinesTable extends EmergencyHelplines
    with TableInfo<$EmergencyHelplinesTable, EmergencyHelpline> {
  @override
  final GeneratedDatabase attachedDatabase;
  final String? _alias;
  $EmergencyHelplinesTable(this.attachedDatabase, [this._alias]);
  static const VerificationMeta _idMeta = const VerificationMeta('id');
  @override
  late final GeneratedColumn<int> id = GeneratedColumn<int>(
    'id',
    aliasedName,
    false,
    hasAutoIncrement: true,
    type: DriftSqlType.int,
    requiredDuringInsert: false,
    defaultConstraints: GeneratedColumn.constraintIsAlways(
      'PRIMARY KEY AUTOINCREMENT',
    ),
  );
  static const VerificationMeta _countryCodeMeta = const VerificationMeta(
    'countryCode',
  );
  @override
  late final GeneratedColumn<String> countryCode = GeneratedColumn<String>(
    'country_code',
    aliasedName,
    false,
    additionalChecks: GeneratedColumn.checkTextLength(maxTextLength: 2),
    type: DriftSqlType.string,
    requiredDuringInsert: true,
  );
  static const VerificationMeta _regionCodeMeta = const VerificationMeta(
    'regionCode',
  );
  @override
  late final GeneratedColumn<String> regionCode = GeneratedColumn<String>(
    'region_code',
    aliasedName,
    true,
    type: DriftSqlType.string,
    requiredDuringInsert: false,
  );
  static const VerificationMeta _serviceTypeMeta = const VerificationMeta(
    'serviceType',
  );
  @override
  late final GeneratedColumn<String> serviceType = GeneratedColumn<String>(
    'service_type',
    aliasedName,
    false,
    type: DriftSqlType.string,
    requiredDuringInsert: true,
  );
  static const VerificationMeta _labelMeta = const VerificationMeta('label');
  @override
  late final GeneratedColumn<String> label = GeneratedColumn<String>(
    'label',
    aliasedName,
    false,
    type: DriftSqlType.string,
    requiredDuringInsert: true,
  );
  static const VerificationMeta _numberMeta = const VerificationMeta('number');
  @override
  late final GeneratedColumn<String> number = GeneratedColumn<String>(
    'number',
    aliasedName,
    false,
    type: DriftSqlType.string,
    requiredDuringInsert: true,
  );
  static const VerificationMeta _sourceNoteMeta = const VerificationMeta(
    'sourceNote',
  );
  @override
  late final GeneratedColumn<String> sourceNote = GeneratedColumn<String>(
    'source_note',
    aliasedName,
    false,
    type: DriftSqlType.string,
    requiredDuringInsert: true,
  );
  static const VerificationMeta _sourceUrlMeta = const VerificationMeta(
    'sourceUrl',
  );
  @override
  late final GeneratedColumn<String> sourceUrl = GeneratedColumn<String>(
    'source_url',
    aliasedName,
    true,
    type: DriftSqlType.string,
    requiredDuringInsert: false,
  );
  static const VerificationMeta _needsVerificationMeta = const VerificationMeta(
    'needsVerification',
  );
  @override
  late final GeneratedColumn<bool> needsVerification = GeneratedColumn<bool>(
    'needs_verification',
    aliasedName,
    false,
    type: DriftSqlType.bool,
    requiredDuringInsert: false,
    defaultConstraints: GeneratedColumn.constraintIsAlways(
      'CHECK ("needs_verification" IN (0, 1))',
    ),
    defaultValue: const Constant(false),
  );
  static const VerificationMeta _tierMeta = const VerificationMeta('tier');
  @override
  late final GeneratedColumn<String> tier = GeneratedColumn<String>(
    'tier',
    aliasedName,
    false,
    type: DriftSqlType.string,
    requiredDuringInsert: false,
    defaultValue: const Constant('verifiedNational'),
  );
  @override
  List<GeneratedColumn> get $columns => [
    id,
    countryCode,
    regionCode,
    serviceType,
    label,
    number,
    sourceNote,
    sourceUrl,
    needsVerification,
    tier,
  ];
  @override
  String get aliasedName => _alias ?? actualTableName;
  @override
  String get actualTableName => $name;
  static const String $name = 'emergency_helplines';
  @override
  VerificationContext validateIntegrity(
    Insertable<EmergencyHelpline> instance, {
    bool isInserting = false,
  }) {
    final context = VerificationContext();
    final data = instance.toColumns(true);
    if (data.containsKey('id')) {
      context.handle(_idMeta, id.isAcceptableOrUnknown(data['id']!, _idMeta));
    }
    if (data.containsKey('country_code')) {
      context.handle(
        _countryCodeMeta,
        countryCode.isAcceptableOrUnknown(
          data['country_code']!,
          _countryCodeMeta,
        ),
      );
    } else if (isInserting) {
      context.missing(_countryCodeMeta);
    }
    if (data.containsKey('region_code')) {
      context.handle(
        _regionCodeMeta,
        regionCode.isAcceptableOrUnknown(data['region_code']!, _regionCodeMeta),
      );
    }
    if (data.containsKey('service_type')) {
      context.handle(
        _serviceTypeMeta,
        serviceType.isAcceptableOrUnknown(
          data['service_type']!,
          _serviceTypeMeta,
        ),
      );
    } else if (isInserting) {
      context.missing(_serviceTypeMeta);
    }
    if (data.containsKey('label')) {
      context.handle(
        _labelMeta,
        label.isAcceptableOrUnknown(data['label']!, _labelMeta),
      );
    } else if (isInserting) {
      context.missing(_labelMeta);
    }
    if (data.containsKey('number')) {
      context.handle(
        _numberMeta,
        number.isAcceptableOrUnknown(data['number']!, _numberMeta),
      );
    } else if (isInserting) {
      context.missing(_numberMeta);
    }
    if (data.containsKey('source_note')) {
      context.handle(
        _sourceNoteMeta,
        sourceNote.isAcceptableOrUnknown(data['source_note']!, _sourceNoteMeta),
      );
    } else if (isInserting) {
      context.missing(_sourceNoteMeta);
    }
    if (data.containsKey('source_url')) {
      context.handle(
        _sourceUrlMeta,
        sourceUrl.isAcceptableOrUnknown(data['source_url']!, _sourceUrlMeta),
      );
    }
    if (data.containsKey('needs_verification')) {
      context.handle(
        _needsVerificationMeta,
        needsVerification.isAcceptableOrUnknown(
          data['needs_verification']!,
          _needsVerificationMeta,
        ),
      );
    }
    if (data.containsKey('tier')) {
      context.handle(
        _tierMeta,
        tier.isAcceptableOrUnknown(data['tier']!, _tierMeta),
      );
    }
    return context;
  }

  @override
  Set<GeneratedColumn> get $primaryKey => {id};
  @override
  List<Set<GeneratedColumn>> get uniqueKeys => [
    {countryCode, number, serviceType},
  ];
  @override
  EmergencyHelpline map(Map<String, dynamic> data, {String? tablePrefix}) {
    final effectivePrefix = tablePrefix != null ? '$tablePrefix.' : '';
    return EmergencyHelpline(
      id: attachedDatabase.typeMapping.read(
        DriftSqlType.int,
        data['${effectivePrefix}id'],
      )!,
      countryCode: attachedDatabase.typeMapping.read(
        DriftSqlType.string,
        data['${effectivePrefix}country_code'],
      )!,
      regionCode: attachedDatabase.typeMapping.read(
        DriftSqlType.string,
        data['${effectivePrefix}region_code'],
      ),
      serviceType: attachedDatabase.typeMapping.read(
        DriftSqlType.string,
        data['${effectivePrefix}service_type'],
      )!,
      label: attachedDatabase.typeMapping.read(
        DriftSqlType.string,
        data['${effectivePrefix}label'],
      )!,
      number: attachedDatabase.typeMapping.read(
        DriftSqlType.string,
        data['${effectivePrefix}number'],
      )!,
      sourceNote: attachedDatabase.typeMapping.read(
        DriftSqlType.string,
        data['${effectivePrefix}source_note'],
      )!,
      sourceUrl: attachedDatabase.typeMapping.read(
        DriftSqlType.string,
        data['${effectivePrefix}source_url'],
      ),
      needsVerification: attachedDatabase.typeMapping.read(
        DriftSqlType.bool,
        data['${effectivePrefix}needs_verification'],
      )!,
      tier: attachedDatabase.typeMapping.read(
        DriftSqlType.string,
        data['${effectivePrefix}tier'],
      )!,
    );
  }

  @override
  $EmergencyHelplinesTable createAlias(String alias) {
    return $EmergencyHelplinesTable(attachedDatabase, alias);
  }
}

class EmergencyHelpline extends DataClass
    implements Insertable<EmergencyHelpline> {
  final int id;
  final String countryCode;

  /// State or union territory. Null means national.
  final String? regionCode;
  final String serviceType;
  final String label;
  final String number;

  /// Shown in the UI under every bundled number, always. The user can see
  /// where it came from and judge for themselves.
  final String sourceNote;
  final String? sourceUrl;

  /// Flagged numbers do not reach the UI. 1930, 1078, 1033 and 104 are
  /// widely cited but were not confirmed from a .gov.in source.
  final bool needsVerification;
  final String tier;
  const EmergencyHelpline({
    required this.id,
    required this.countryCode,
    this.regionCode,
    required this.serviceType,
    required this.label,
    required this.number,
    required this.sourceNote,
    this.sourceUrl,
    required this.needsVerification,
    required this.tier,
  });
  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    map['id'] = Variable<int>(id);
    map['country_code'] = Variable<String>(countryCode);
    if (!nullToAbsent || regionCode != null) {
      map['region_code'] = Variable<String>(regionCode);
    }
    map['service_type'] = Variable<String>(serviceType);
    map['label'] = Variable<String>(label);
    map['number'] = Variable<String>(number);
    map['source_note'] = Variable<String>(sourceNote);
    if (!nullToAbsent || sourceUrl != null) {
      map['source_url'] = Variable<String>(sourceUrl);
    }
    map['needs_verification'] = Variable<bool>(needsVerification);
    map['tier'] = Variable<String>(tier);
    return map;
  }

  EmergencyHelplinesCompanion toCompanion(bool nullToAbsent) {
    return EmergencyHelplinesCompanion(
      id: Value(id),
      countryCode: Value(countryCode),
      regionCode: regionCode == null && nullToAbsent
          ? const Value.absent()
          : Value(regionCode),
      serviceType: Value(serviceType),
      label: Value(label),
      number: Value(number),
      sourceNote: Value(sourceNote),
      sourceUrl: sourceUrl == null && nullToAbsent
          ? const Value.absent()
          : Value(sourceUrl),
      needsVerification: Value(needsVerification),
      tier: Value(tier),
    );
  }

  factory EmergencyHelpline.fromJson(
    Map<String, dynamic> json, {
    ValueSerializer? serializer,
  }) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return EmergencyHelpline(
      id: serializer.fromJson<int>(json['id']),
      countryCode: serializer.fromJson<String>(json['countryCode']),
      regionCode: serializer.fromJson<String?>(json['regionCode']),
      serviceType: serializer.fromJson<String>(json['serviceType']),
      label: serializer.fromJson<String>(json['label']),
      number: serializer.fromJson<String>(json['number']),
      sourceNote: serializer.fromJson<String>(json['sourceNote']),
      sourceUrl: serializer.fromJson<String?>(json['sourceUrl']),
      needsVerification: serializer.fromJson<bool>(json['needsVerification']),
      tier: serializer.fromJson<String>(json['tier']),
    );
  }
  @override
  Map<String, dynamic> toJson({ValueSerializer? serializer}) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return <String, dynamic>{
      'id': serializer.toJson<int>(id),
      'countryCode': serializer.toJson<String>(countryCode),
      'regionCode': serializer.toJson<String?>(regionCode),
      'serviceType': serializer.toJson<String>(serviceType),
      'label': serializer.toJson<String>(label),
      'number': serializer.toJson<String>(number),
      'sourceNote': serializer.toJson<String>(sourceNote),
      'sourceUrl': serializer.toJson<String?>(sourceUrl),
      'needsVerification': serializer.toJson<bool>(needsVerification),
      'tier': serializer.toJson<String>(tier),
    };
  }

  EmergencyHelpline copyWith({
    int? id,
    String? countryCode,
    Value<String?> regionCode = const Value.absent(),
    String? serviceType,
    String? label,
    String? number,
    String? sourceNote,
    Value<String?> sourceUrl = const Value.absent(),
    bool? needsVerification,
    String? tier,
  }) => EmergencyHelpline(
    id: id ?? this.id,
    countryCode: countryCode ?? this.countryCode,
    regionCode: regionCode.present ? regionCode.value : this.regionCode,
    serviceType: serviceType ?? this.serviceType,
    label: label ?? this.label,
    number: number ?? this.number,
    sourceNote: sourceNote ?? this.sourceNote,
    sourceUrl: sourceUrl.present ? sourceUrl.value : this.sourceUrl,
    needsVerification: needsVerification ?? this.needsVerification,
    tier: tier ?? this.tier,
  );
  EmergencyHelpline copyWithCompanion(EmergencyHelplinesCompanion data) {
    return EmergencyHelpline(
      id: data.id.present ? data.id.value : this.id,
      countryCode: data.countryCode.present
          ? data.countryCode.value
          : this.countryCode,
      regionCode: data.regionCode.present
          ? data.regionCode.value
          : this.regionCode,
      serviceType: data.serviceType.present
          ? data.serviceType.value
          : this.serviceType,
      label: data.label.present ? data.label.value : this.label,
      number: data.number.present ? data.number.value : this.number,
      sourceNote: data.sourceNote.present
          ? data.sourceNote.value
          : this.sourceNote,
      sourceUrl: data.sourceUrl.present ? data.sourceUrl.value : this.sourceUrl,
      needsVerification: data.needsVerification.present
          ? data.needsVerification.value
          : this.needsVerification,
      tier: data.tier.present ? data.tier.value : this.tier,
    );
  }

  @override
  String toString() {
    return (StringBuffer('EmergencyHelpline(')
          ..write('id: $id, ')
          ..write('countryCode: $countryCode, ')
          ..write('regionCode: $regionCode, ')
          ..write('serviceType: $serviceType, ')
          ..write('label: $label, ')
          ..write('number: $number, ')
          ..write('sourceNote: $sourceNote, ')
          ..write('sourceUrl: $sourceUrl, ')
          ..write('needsVerification: $needsVerification, ')
          ..write('tier: $tier')
          ..write(')'))
        .toString();
  }

  @override
  int get hashCode => Object.hash(
    id,
    countryCode,
    regionCode,
    serviceType,
    label,
    number,
    sourceNote,
    sourceUrl,
    needsVerification,
    tier,
  );
  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      (other is EmergencyHelpline &&
          other.id == this.id &&
          other.countryCode == this.countryCode &&
          other.regionCode == this.regionCode &&
          other.serviceType == this.serviceType &&
          other.label == this.label &&
          other.number == this.number &&
          other.sourceNote == this.sourceNote &&
          other.sourceUrl == this.sourceUrl &&
          other.needsVerification == this.needsVerification &&
          other.tier == this.tier);
}

class EmergencyHelplinesCompanion extends UpdateCompanion<EmergencyHelpline> {
  final Value<int> id;
  final Value<String> countryCode;
  final Value<String?> regionCode;
  final Value<String> serviceType;
  final Value<String> label;
  final Value<String> number;
  final Value<String> sourceNote;
  final Value<String?> sourceUrl;
  final Value<bool> needsVerification;
  final Value<String> tier;
  const EmergencyHelplinesCompanion({
    this.id = const Value.absent(),
    this.countryCode = const Value.absent(),
    this.regionCode = const Value.absent(),
    this.serviceType = const Value.absent(),
    this.label = const Value.absent(),
    this.number = const Value.absent(),
    this.sourceNote = const Value.absent(),
    this.sourceUrl = const Value.absent(),
    this.needsVerification = const Value.absent(),
    this.tier = const Value.absent(),
  });
  EmergencyHelplinesCompanion.insert({
    this.id = const Value.absent(),
    required String countryCode,
    this.regionCode = const Value.absent(),
    required String serviceType,
    required String label,
    required String number,
    required String sourceNote,
    this.sourceUrl = const Value.absent(),
    this.needsVerification = const Value.absent(),
    this.tier = const Value.absent(),
  }) : countryCode = Value(countryCode),
       serviceType = Value(serviceType),
       label = Value(label),
       number = Value(number),
       sourceNote = Value(sourceNote);
  static Insertable<EmergencyHelpline> custom({
    Expression<int>? id,
    Expression<String>? countryCode,
    Expression<String>? regionCode,
    Expression<String>? serviceType,
    Expression<String>? label,
    Expression<String>? number,
    Expression<String>? sourceNote,
    Expression<String>? sourceUrl,
    Expression<bool>? needsVerification,
    Expression<String>? tier,
  }) {
    return RawValuesInsertable({
      if (id != null) 'id': id,
      if (countryCode != null) 'country_code': countryCode,
      if (regionCode != null) 'region_code': regionCode,
      if (serviceType != null) 'service_type': serviceType,
      if (label != null) 'label': label,
      if (number != null) 'number': number,
      if (sourceNote != null) 'source_note': sourceNote,
      if (sourceUrl != null) 'source_url': sourceUrl,
      if (needsVerification != null) 'needs_verification': needsVerification,
      if (tier != null) 'tier': tier,
    });
  }

  EmergencyHelplinesCompanion copyWith({
    Value<int>? id,
    Value<String>? countryCode,
    Value<String?>? regionCode,
    Value<String>? serviceType,
    Value<String>? label,
    Value<String>? number,
    Value<String>? sourceNote,
    Value<String?>? sourceUrl,
    Value<bool>? needsVerification,
    Value<String>? tier,
  }) {
    return EmergencyHelplinesCompanion(
      id: id ?? this.id,
      countryCode: countryCode ?? this.countryCode,
      regionCode: regionCode ?? this.regionCode,
      serviceType: serviceType ?? this.serviceType,
      label: label ?? this.label,
      number: number ?? this.number,
      sourceNote: sourceNote ?? this.sourceNote,
      sourceUrl: sourceUrl ?? this.sourceUrl,
      needsVerification: needsVerification ?? this.needsVerification,
      tier: tier ?? this.tier,
    );
  }

  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    if (id.present) {
      map['id'] = Variable<int>(id.value);
    }
    if (countryCode.present) {
      map['country_code'] = Variable<String>(countryCode.value);
    }
    if (regionCode.present) {
      map['region_code'] = Variable<String>(regionCode.value);
    }
    if (serviceType.present) {
      map['service_type'] = Variable<String>(serviceType.value);
    }
    if (label.present) {
      map['label'] = Variable<String>(label.value);
    }
    if (number.present) {
      map['number'] = Variable<String>(number.value);
    }
    if (sourceNote.present) {
      map['source_note'] = Variable<String>(sourceNote.value);
    }
    if (sourceUrl.present) {
      map['source_url'] = Variable<String>(sourceUrl.value);
    }
    if (needsVerification.present) {
      map['needs_verification'] = Variable<bool>(needsVerification.value);
    }
    if (tier.present) {
      map['tier'] = Variable<String>(tier.value);
    }
    return map;
  }

  @override
  String toString() {
    return (StringBuffer('EmergencyHelplinesCompanion(')
          ..write('id: $id, ')
          ..write('countryCode: $countryCode, ')
          ..write('regionCode: $regionCode, ')
          ..write('serviceType: $serviceType, ')
          ..write('label: $label, ')
          ..write('number: $number, ')
          ..write('sourceNote: $sourceNote, ')
          ..write('sourceUrl: $sourceUrl, ')
          ..write('needsVerification: $needsVerification, ')
          ..write('tier: $tier')
          ..write(')'))
        .toString();
  }
}

class $ChecklistItemsTable extends ChecklistItems
    with TableInfo<$ChecklistItemsTable, ChecklistItem> {
  @override
  final GeneratedDatabase attachedDatabase;
  final String? _alias;
  $ChecklistItemsTable(this.attachedDatabase, [this._alias]);
  static const VerificationMeta _idMeta = const VerificationMeta('id');
  @override
  late final GeneratedColumn<int> id = GeneratedColumn<int>(
    'id',
    aliasedName,
    false,
    hasAutoIncrement: true,
    type: DriftSqlType.int,
    requiredDuringInsert: false,
    defaultConstraints: GeneratedColumn.constraintIsAlways(
      'PRIMARY KEY AUTOINCREMENT',
    ),
  );
  static const VerificationMeta _tripIdMeta = const VerificationMeta('tripId');
  @override
  late final GeneratedColumn<int> tripId = GeneratedColumn<int>(
    'trip_id',
    aliasedName,
    false,
    type: DriftSqlType.int,
    requiredDuringInsert: true,
    defaultConstraints: GeneratedColumn.constraintIsAlways(
      'REFERENCES trips (id) ON DELETE CASCADE',
    ),
  );
  static const VerificationMeta _stopIdMeta = const VerificationMeta('stopId');
  @override
  late final GeneratedColumn<int> stopId = GeneratedColumn<int>(
    'stop_id',
    aliasedName,
    true,
    type: DriftSqlType.int,
    requiredDuringInsert: false,
    defaultConstraints: GeneratedColumn.constraintIsAlways(
      'REFERENCES stops (id) ON DELETE CASCADE',
    ),
  );
  static const VerificationMeta _labelMeta = const VerificationMeta('label');
  @override
  late final GeneratedColumn<String> label = GeneratedColumn<String>(
    'label',
    aliasedName,
    false,
    type: DriftSqlType.string,
    requiredDuringInsert: true,
  );
  static const VerificationMeta _quantityMeta = const VerificationMeta(
    'quantity',
  );
  @override
  late final GeneratedColumn<String> quantity = GeneratedColumn<String>(
    'quantity',
    aliasedName,
    true,
    type: DriftSqlType.string,
    requiredDuringInsert: false,
  );
  static const VerificationMeta _sourceTagsMeta = const VerificationMeta(
    'sourceTags',
  );
  @override
  late final GeneratedColumn<String> sourceTags = GeneratedColumn<String>(
    'source_tags',
    aliasedName,
    false,
    type: DriftSqlType.string,
    requiredDuringInsert: false,
    defaultValue: const Constant(''),
  );
  static const VerificationMeta _isDoneMeta = const VerificationMeta('isDone');
  @override
  late final GeneratedColumn<bool> isDone = GeneratedColumn<bool>(
    'is_done',
    aliasedName,
    false,
    type: DriftSqlType.bool,
    requiredDuringInsert: false,
    defaultConstraints: GeneratedColumn.constraintIsAlways(
      'CHECK ("is_done" IN (0, 1))',
    ),
    defaultValue: const Constant(false),
  );
  static const VerificationMeta _isBlockingMeta = const VerificationMeta(
    'isBlocking',
  );
  @override
  late final GeneratedColumn<bool> isBlocking = GeneratedColumn<bool>(
    'is_blocking',
    aliasedName,
    false,
    type: DriftSqlType.bool,
    requiredDuringInsert: false,
    defaultConstraints: GeneratedColumn.constraintIsAlways(
      'CHECK ("is_blocking" IN (0, 1))',
    ),
    defaultValue: const Constant(false),
  );
  static const VerificationMeta _contactIdMeta = const VerificationMeta(
    'contactId',
  );
  @override
  late final GeneratedColumn<int> contactId = GeneratedColumn<int>(
    'contact_id',
    aliasedName,
    true,
    type: DriftSqlType.int,
    requiredDuringInsert: false,
    defaultConstraints: GeneratedColumn.constraintIsAlways(
      'REFERENCES contacts (id) ON DELETE CASCADE',
    ),
  );
  static const VerificationMeta _isGeneratedMeta = const VerificationMeta(
    'isGenerated',
  );
  @override
  late final GeneratedColumn<bool> isGenerated = GeneratedColumn<bool>(
    'is_generated',
    aliasedName,
    false,
    type: DriftSqlType.bool,
    requiredDuringInsert: false,
    defaultConstraints: GeneratedColumn.constraintIsAlways(
      'CHECK ("is_generated" IN (0, 1))',
    ),
    defaultValue: const Constant(true),
  );
  static const VerificationMeta _generatorKeyMeta = const VerificationMeta(
    'generatorKey',
  );
  @override
  late final GeneratedColumn<String> generatorKey = GeneratedColumn<String>(
    'generator_key',
    aliasedName,
    true,
    type: DriftSqlType.string,
    requiredDuringInsert: false,
  );
  static const VerificationMeta _isUserEditedMeta = const VerificationMeta(
    'isUserEdited',
  );
  @override
  late final GeneratedColumn<bool> isUserEdited = GeneratedColumn<bool>(
    'is_user_edited',
    aliasedName,
    false,
    type: DriftSqlType.bool,
    requiredDuringInsert: false,
    defaultConstraints: GeneratedColumn.constraintIsAlways(
      'CHECK ("is_user_edited" IN (0, 1))',
    ),
    defaultValue: const Constant(false),
  );
  static const VerificationMeta _sortOrderMeta = const VerificationMeta(
    'sortOrder',
  );
  @override
  late final GeneratedColumn<int> sortOrder = GeneratedColumn<int>(
    'sort_order',
    aliasedName,
    false,
    type: DriftSqlType.int,
    requiredDuringInsert: false,
    defaultValue: const Constant(0),
  );
  @override
  List<GeneratedColumn> get $columns => [
    id,
    tripId,
    stopId,
    label,
    quantity,
    sourceTags,
    isDone,
    isBlocking,
    contactId,
    isGenerated,
    generatorKey,
    isUserEdited,
    sortOrder,
  ];
  @override
  String get aliasedName => _alias ?? actualTableName;
  @override
  String get actualTableName => $name;
  static const String $name = 'checklist_items';
  @override
  VerificationContext validateIntegrity(
    Insertable<ChecklistItem> instance, {
    bool isInserting = false,
  }) {
    final context = VerificationContext();
    final data = instance.toColumns(true);
    if (data.containsKey('id')) {
      context.handle(_idMeta, id.isAcceptableOrUnknown(data['id']!, _idMeta));
    }
    if (data.containsKey('trip_id')) {
      context.handle(
        _tripIdMeta,
        tripId.isAcceptableOrUnknown(data['trip_id']!, _tripIdMeta),
      );
    } else if (isInserting) {
      context.missing(_tripIdMeta);
    }
    if (data.containsKey('stop_id')) {
      context.handle(
        _stopIdMeta,
        stopId.isAcceptableOrUnknown(data['stop_id']!, _stopIdMeta),
      );
    }
    if (data.containsKey('label')) {
      context.handle(
        _labelMeta,
        label.isAcceptableOrUnknown(data['label']!, _labelMeta),
      );
    } else if (isInserting) {
      context.missing(_labelMeta);
    }
    if (data.containsKey('quantity')) {
      context.handle(
        _quantityMeta,
        quantity.isAcceptableOrUnknown(data['quantity']!, _quantityMeta),
      );
    }
    if (data.containsKey('source_tags')) {
      context.handle(
        _sourceTagsMeta,
        sourceTags.isAcceptableOrUnknown(data['source_tags']!, _sourceTagsMeta),
      );
    }
    if (data.containsKey('is_done')) {
      context.handle(
        _isDoneMeta,
        isDone.isAcceptableOrUnknown(data['is_done']!, _isDoneMeta),
      );
    }
    if (data.containsKey('is_blocking')) {
      context.handle(
        _isBlockingMeta,
        isBlocking.isAcceptableOrUnknown(data['is_blocking']!, _isBlockingMeta),
      );
    }
    if (data.containsKey('contact_id')) {
      context.handle(
        _contactIdMeta,
        contactId.isAcceptableOrUnknown(data['contact_id']!, _contactIdMeta),
      );
    }
    if (data.containsKey('is_generated')) {
      context.handle(
        _isGeneratedMeta,
        isGenerated.isAcceptableOrUnknown(
          data['is_generated']!,
          _isGeneratedMeta,
        ),
      );
    }
    if (data.containsKey('generator_key')) {
      context.handle(
        _generatorKeyMeta,
        generatorKey.isAcceptableOrUnknown(
          data['generator_key']!,
          _generatorKeyMeta,
        ),
      );
    }
    if (data.containsKey('is_user_edited')) {
      context.handle(
        _isUserEditedMeta,
        isUserEdited.isAcceptableOrUnknown(
          data['is_user_edited']!,
          _isUserEditedMeta,
        ),
      );
    }
    if (data.containsKey('sort_order')) {
      context.handle(
        _sortOrderMeta,
        sortOrder.isAcceptableOrUnknown(data['sort_order']!, _sortOrderMeta),
      );
    }
    return context;
  }

  @override
  Set<GeneratedColumn> get $primaryKey => {id};
  @override
  ChecklistItem map(Map<String, dynamic> data, {String? tablePrefix}) {
    final effectivePrefix = tablePrefix != null ? '$tablePrefix.' : '';
    return ChecklistItem(
      id: attachedDatabase.typeMapping.read(
        DriftSqlType.int,
        data['${effectivePrefix}id'],
      )!,
      tripId: attachedDatabase.typeMapping.read(
        DriftSqlType.int,
        data['${effectivePrefix}trip_id'],
      )!,
      stopId: attachedDatabase.typeMapping.read(
        DriftSqlType.int,
        data['${effectivePrefix}stop_id'],
      ),
      label: attachedDatabase.typeMapping.read(
        DriftSqlType.string,
        data['${effectivePrefix}label'],
      )!,
      quantity: attachedDatabase.typeMapping.read(
        DriftSqlType.string,
        data['${effectivePrefix}quantity'],
      ),
      sourceTags: attachedDatabase.typeMapping.read(
        DriftSqlType.string,
        data['${effectivePrefix}source_tags'],
      )!,
      isDone: attachedDatabase.typeMapping.read(
        DriftSqlType.bool,
        data['${effectivePrefix}is_done'],
      )!,
      isBlocking: attachedDatabase.typeMapping.read(
        DriftSqlType.bool,
        data['${effectivePrefix}is_blocking'],
      )!,
      contactId: attachedDatabase.typeMapping.read(
        DriftSqlType.int,
        data['${effectivePrefix}contact_id'],
      ),
      isGenerated: attachedDatabase.typeMapping.read(
        DriftSqlType.bool,
        data['${effectivePrefix}is_generated'],
      )!,
      generatorKey: attachedDatabase.typeMapping.read(
        DriftSqlType.string,
        data['${effectivePrefix}generator_key'],
      ),
      isUserEdited: attachedDatabase.typeMapping.read(
        DriftSqlType.bool,
        data['${effectivePrefix}is_user_edited'],
      )!,
      sortOrder: attachedDatabase.typeMapping.read(
        DriftSqlType.int,
        data['${effectivePrefix}sort_order'],
      )!,
    );
  }

  @override
  $ChecklistItemsTable createAlias(String alias) {
    return $ChecklistItemsTable(attachedDatabase, alias);
  }
}

class ChecklistItem extends DataClass implements Insertable<ChecklistItem> {
  final int id;
  final int tripId;
  final int? stopId;
  final String label;
  final String? quantity;

  /// The tags that produced this item, shown under it. A generated list
  /// nobody understands gets ignored.
  final String sourceTags;
  final bool isDone;

  /// The trip does not read ready while any blocking item is open.
  final bool isBlocking;

  /// Set on blocking items generated from an unconfirmed number.
  final int? contactId;
  final bool isGenerated;

  /// The generator rule that produced this item, stable across renames.
  ///
  /// Matching a generated item by its LABEL looked fine until someone renamed
  /// one: the generator then found no row for its rule and inserted a second
  /// copy alongside the user's. Null for items the user wrote themselves and
  /// for blocking items, which are keyed by stop.
  final String? generatorKey;

  /// Set the moment a user edits a generated item, so regeneration cannot
  /// silently discard their change.
  final bool isUserEdited;
  final int sortOrder;
  const ChecklistItem({
    required this.id,
    required this.tripId,
    this.stopId,
    required this.label,
    this.quantity,
    required this.sourceTags,
    required this.isDone,
    required this.isBlocking,
    this.contactId,
    required this.isGenerated,
    this.generatorKey,
    required this.isUserEdited,
    required this.sortOrder,
  });
  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    map['id'] = Variable<int>(id);
    map['trip_id'] = Variable<int>(tripId);
    if (!nullToAbsent || stopId != null) {
      map['stop_id'] = Variable<int>(stopId);
    }
    map['label'] = Variable<String>(label);
    if (!nullToAbsent || quantity != null) {
      map['quantity'] = Variable<String>(quantity);
    }
    map['source_tags'] = Variable<String>(sourceTags);
    map['is_done'] = Variable<bool>(isDone);
    map['is_blocking'] = Variable<bool>(isBlocking);
    if (!nullToAbsent || contactId != null) {
      map['contact_id'] = Variable<int>(contactId);
    }
    map['is_generated'] = Variable<bool>(isGenerated);
    if (!nullToAbsent || generatorKey != null) {
      map['generator_key'] = Variable<String>(generatorKey);
    }
    map['is_user_edited'] = Variable<bool>(isUserEdited);
    map['sort_order'] = Variable<int>(sortOrder);
    return map;
  }

  ChecklistItemsCompanion toCompanion(bool nullToAbsent) {
    return ChecklistItemsCompanion(
      id: Value(id),
      tripId: Value(tripId),
      stopId: stopId == null && nullToAbsent
          ? const Value.absent()
          : Value(stopId),
      label: Value(label),
      quantity: quantity == null && nullToAbsent
          ? const Value.absent()
          : Value(quantity),
      sourceTags: Value(sourceTags),
      isDone: Value(isDone),
      isBlocking: Value(isBlocking),
      contactId: contactId == null && nullToAbsent
          ? const Value.absent()
          : Value(contactId),
      isGenerated: Value(isGenerated),
      generatorKey: generatorKey == null && nullToAbsent
          ? const Value.absent()
          : Value(generatorKey),
      isUserEdited: Value(isUserEdited),
      sortOrder: Value(sortOrder),
    );
  }

  factory ChecklistItem.fromJson(
    Map<String, dynamic> json, {
    ValueSerializer? serializer,
  }) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return ChecklistItem(
      id: serializer.fromJson<int>(json['id']),
      tripId: serializer.fromJson<int>(json['tripId']),
      stopId: serializer.fromJson<int?>(json['stopId']),
      label: serializer.fromJson<String>(json['label']),
      quantity: serializer.fromJson<String?>(json['quantity']),
      sourceTags: serializer.fromJson<String>(json['sourceTags']),
      isDone: serializer.fromJson<bool>(json['isDone']),
      isBlocking: serializer.fromJson<bool>(json['isBlocking']),
      contactId: serializer.fromJson<int?>(json['contactId']),
      isGenerated: serializer.fromJson<bool>(json['isGenerated']),
      generatorKey: serializer.fromJson<String?>(json['generatorKey']),
      isUserEdited: serializer.fromJson<bool>(json['isUserEdited']),
      sortOrder: serializer.fromJson<int>(json['sortOrder']),
    );
  }
  @override
  Map<String, dynamic> toJson({ValueSerializer? serializer}) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return <String, dynamic>{
      'id': serializer.toJson<int>(id),
      'tripId': serializer.toJson<int>(tripId),
      'stopId': serializer.toJson<int?>(stopId),
      'label': serializer.toJson<String>(label),
      'quantity': serializer.toJson<String?>(quantity),
      'sourceTags': serializer.toJson<String>(sourceTags),
      'isDone': serializer.toJson<bool>(isDone),
      'isBlocking': serializer.toJson<bool>(isBlocking),
      'contactId': serializer.toJson<int?>(contactId),
      'isGenerated': serializer.toJson<bool>(isGenerated),
      'generatorKey': serializer.toJson<String?>(generatorKey),
      'isUserEdited': serializer.toJson<bool>(isUserEdited),
      'sortOrder': serializer.toJson<int>(sortOrder),
    };
  }

  ChecklistItem copyWith({
    int? id,
    int? tripId,
    Value<int?> stopId = const Value.absent(),
    String? label,
    Value<String?> quantity = const Value.absent(),
    String? sourceTags,
    bool? isDone,
    bool? isBlocking,
    Value<int?> contactId = const Value.absent(),
    bool? isGenerated,
    Value<String?> generatorKey = const Value.absent(),
    bool? isUserEdited,
    int? sortOrder,
  }) => ChecklistItem(
    id: id ?? this.id,
    tripId: tripId ?? this.tripId,
    stopId: stopId.present ? stopId.value : this.stopId,
    label: label ?? this.label,
    quantity: quantity.present ? quantity.value : this.quantity,
    sourceTags: sourceTags ?? this.sourceTags,
    isDone: isDone ?? this.isDone,
    isBlocking: isBlocking ?? this.isBlocking,
    contactId: contactId.present ? contactId.value : this.contactId,
    isGenerated: isGenerated ?? this.isGenerated,
    generatorKey: generatorKey.present ? generatorKey.value : this.generatorKey,
    isUserEdited: isUserEdited ?? this.isUserEdited,
    sortOrder: sortOrder ?? this.sortOrder,
  );
  ChecklistItem copyWithCompanion(ChecklistItemsCompanion data) {
    return ChecklistItem(
      id: data.id.present ? data.id.value : this.id,
      tripId: data.tripId.present ? data.tripId.value : this.tripId,
      stopId: data.stopId.present ? data.stopId.value : this.stopId,
      label: data.label.present ? data.label.value : this.label,
      quantity: data.quantity.present ? data.quantity.value : this.quantity,
      sourceTags: data.sourceTags.present
          ? data.sourceTags.value
          : this.sourceTags,
      isDone: data.isDone.present ? data.isDone.value : this.isDone,
      isBlocking: data.isBlocking.present
          ? data.isBlocking.value
          : this.isBlocking,
      contactId: data.contactId.present ? data.contactId.value : this.contactId,
      isGenerated: data.isGenerated.present
          ? data.isGenerated.value
          : this.isGenerated,
      generatorKey: data.generatorKey.present
          ? data.generatorKey.value
          : this.generatorKey,
      isUserEdited: data.isUserEdited.present
          ? data.isUserEdited.value
          : this.isUserEdited,
      sortOrder: data.sortOrder.present ? data.sortOrder.value : this.sortOrder,
    );
  }

  @override
  String toString() {
    return (StringBuffer('ChecklistItem(')
          ..write('id: $id, ')
          ..write('tripId: $tripId, ')
          ..write('stopId: $stopId, ')
          ..write('label: $label, ')
          ..write('quantity: $quantity, ')
          ..write('sourceTags: $sourceTags, ')
          ..write('isDone: $isDone, ')
          ..write('isBlocking: $isBlocking, ')
          ..write('contactId: $contactId, ')
          ..write('isGenerated: $isGenerated, ')
          ..write('generatorKey: $generatorKey, ')
          ..write('isUserEdited: $isUserEdited, ')
          ..write('sortOrder: $sortOrder')
          ..write(')'))
        .toString();
  }

  @override
  int get hashCode => Object.hash(
    id,
    tripId,
    stopId,
    label,
    quantity,
    sourceTags,
    isDone,
    isBlocking,
    contactId,
    isGenerated,
    generatorKey,
    isUserEdited,
    sortOrder,
  );
  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      (other is ChecklistItem &&
          other.id == this.id &&
          other.tripId == this.tripId &&
          other.stopId == this.stopId &&
          other.label == this.label &&
          other.quantity == this.quantity &&
          other.sourceTags == this.sourceTags &&
          other.isDone == this.isDone &&
          other.isBlocking == this.isBlocking &&
          other.contactId == this.contactId &&
          other.isGenerated == this.isGenerated &&
          other.generatorKey == this.generatorKey &&
          other.isUserEdited == this.isUserEdited &&
          other.sortOrder == this.sortOrder);
}

class ChecklistItemsCompanion extends UpdateCompanion<ChecklistItem> {
  final Value<int> id;
  final Value<int> tripId;
  final Value<int?> stopId;
  final Value<String> label;
  final Value<String?> quantity;
  final Value<String> sourceTags;
  final Value<bool> isDone;
  final Value<bool> isBlocking;
  final Value<int?> contactId;
  final Value<bool> isGenerated;
  final Value<String?> generatorKey;
  final Value<bool> isUserEdited;
  final Value<int> sortOrder;
  const ChecklistItemsCompanion({
    this.id = const Value.absent(),
    this.tripId = const Value.absent(),
    this.stopId = const Value.absent(),
    this.label = const Value.absent(),
    this.quantity = const Value.absent(),
    this.sourceTags = const Value.absent(),
    this.isDone = const Value.absent(),
    this.isBlocking = const Value.absent(),
    this.contactId = const Value.absent(),
    this.isGenerated = const Value.absent(),
    this.generatorKey = const Value.absent(),
    this.isUserEdited = const Value.absent(),
    this.sortOrder = const Value.absent(),
  });
  ChecklistItemsCompanion.insert({
    this.id = const Value.absent(),
    required int tripId,
    this.stopId = const Value.absent(),
    required String label,
    this.quantity = const Value.absent(),
    this.sourceTags = const Value.absent(),
    this.isDone = const Value.absent(),
    this.isBlocking = const Value.absent(),
    this.contactId = const Value.absent(),
    this.isGenerated = const Value.absent(),
    this.generatorKey = const Value.absent(),
    this.isUserEdited = const Value.absent(),
    this.sortOrder = const Value.absent(),
  }) : tripId = Value(tripId),
       label = Value(label);
  static Insertable<ChecklistItem> custom({
    Expression<int>? id,
    Expression<int>? tripId,
    Expression<int>? stopId,
    Expression<String>? label,
    Expression<String>? quantity,
    Expression<String>? sourceTags,
    Expression<bool>? isDone,
    Expression<bool>? isBlocking,
    Expression<int>? contactId,
    Expression<bool>? isGenerated,
    Expression<String>? generatorKey,
    Expression<bool>? isUserEdited,
    Expression<int>? sortOrder,
  }) {
    return RawValuesInsertable({
      if (id != null) 'id': id,
      if (tripId != null) 'trip_id': tripId,
      if (stopId != null) 'stop_id': stopId,
      if (label != null) 'label': label,
      if (quantity != null) 'quantity': quantity,
      if (sourceTags != null) 'source_tags': sourceTags,
      if (isDone != null) 'is_done': isDone,
      if (isBlocking != null) 'is_blocking': isBlocking,
      if (contactId != null) 'contact_id': contactId,
      if (isGenerated != null) 'is_generated': isGenerated,
      if (generatorKey != null) 'generator_key': generatorKey,
      if (isUserEdited != null) 'is_user_edited': isUserEdited,
      if (sortOrder != null) 'sort_order': sortOrder,
    });
  }

  ChecklistItemsCompanion copyWith({
    Value<int>? id,
    Value<int>? tripId,
    Value<int?>? stopId,
    Value<String>? label,
    Value<String?>? quantity,
    Value<String>? sourceTags,
    Value<bool>? isDone,
    Value<bool>? isBlocking,
    Value<int?>? contactId,
    Value<bool>? isGenerated,
    Value<String?>? generatorKey,
    Value<bool>? isUserEdited,
    Value<int>? sortOrder,
  }) {
    return ChecklistItemsCompanion(
      id: id ?? this.id,
      tripId: tripId ?? this.tripId,
      stopId: stopId ?? this.stopId,
      label: label ?? this.label,
      quantity: quantity ?? this.quantity,
      sourceTags: sourceTags ?? this.sourceTags,
      isDone: isDone ?? this.isDone,
      isBlocking: isBlocking ?? this.isBlocking,
      contactId: contactId ?? this.contactId,
      isGenerated: isGenerated ?? this.isGenerated,
      generatorKey: generatorKey ?? this.generatorKey,
      isUserEdited: isUserEdited ?? this.isUserEdited,
      sortOrder: sortOrder ?? this.sortOrder,
    );
  }

  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    if (id.present) {
      map['id'] = Variable<int>(id.value);
    }
    if (tripId.present) {
      map['trip_id'] = Variable<int>(tripId.value);
    }
    if (stopId.present) {
      map['stop_id'] = Variable<int>(stopId.value);
    }
    if (label.present) {
      map['label'] = Variable<String>(label.value);
    }
    if (quantity.present) {
      map['quantity'] = Variable<String>(quantity.value);
    }
    if (sourceTags.present) {
      map['source_tags'] = Variable<String>(sourceTags.value);
    }
    if (isDone.present) {
      map['is_done'] = Variable<bool>(isDone.value);
    }
    if (isBlocking.present) {
      map['is_blocking'] = Variable<bool>(isBlocking.value);
    }
    if (contactId.present) {
      map['contact_id'] = Variable<int>(contactId.value);
    }
    if (isGenerated.present) {
      map['is_generated'] = Variable<bool>(isGenerated.value);
    }
    if (generatorKey.present) {
      map['generator_key'] = Variable<String>(generatorKey.value);
    }
    if (isUserEdited.present) {
      map['is_user_edited'] = Variable<bool>(isUserEdited.value);
    }
    if (sortOrder.present) {
      map['sort_order'] = Variable<int>(sortOrder.value);
    }
    return map;
  }

  @override
  String toString() {
    return (StringBuffer('ChecklistItemsCompanion(')
          ..write('id: $id, ')
          ..write('tripId: $tripId, ')
          ..write('stopId: $stopId, ')
          ..write('label: $label, ')
          ..write('quantity: $quantity, ')
          ..write('sourceTags: $sourceTags, ')
          ..write('isDone: $isDone, ')
          ..write('isBlocking: $isBlocking, ')
          ..write('contactId: $contactId, ')
          ..write('isGenerated: $isGenerated, ')
          ..write('generatorKey: $generatorKey, ')
          ..write('isUserEdited: $isUserEdited, ')
          ..write('sortOrder: $sortOrder')
          ..write(')'))
        .toString();
  }
}

class $WeatherSnapshotsTable extends WeatherSnapshots
    with TableInfo<$WeatherSnapshotsTable, WeatherSnapshot> {
  @override
  final GeneratedDatabase attachedDatabase;
  final String? _alias;
  $WeatherSnapshotsTable(this.attachedDatabase, [this._alias]);
  static const VerificationMeta _idMeta = const VerificationMeta('id');
  @override
  late final GeneratedColumn<int> id = GeneratedColumn<int>(
    'id',
    aliasedName,
    false,
    hasAutoIncrement: true,
    type: DriftSqlType.int,
    requiredDuringInsert: false,
    defaultConstraints: GeneratedColumn.constraintIsAlways(
      'PRIMARY KEY AUTOINCREMENT',
    ),
  );
  static const VerificationMeta _stopIdMeta = const VerificationMeta('stopId');
  @override
  late final GeneratedColumn<int> stopId = GeneratedColumn<int>(
    'stop_id',
    aliasedName,
    false,
    type: DriftSqlType.int,
    requiredDuringInsert: true,
    defaultConstraints: GeneratedColumn.constraintIsAlways(
      'REFERENCES stops (id) ON DELETE CASCADE',
    ),
  );
  static const VerificationMeta _forDateMeta = const VerificationMeta(
    'forDate',
  );
  @override
  late final GeneratedColumn<DateTime> forDate = GeneratedColumn<DateTime>(
    'for_date',
    aliasedName,
    false,
    type: DriftSqlType.dateTime,
    requiredDuringInsert: true,
  );
  static const VerificationMeta _conditionMeta = const VerificationMeta(
    'condition',
  );
  @override
  late final GeneratedColumn<String> condition = GeneratedColumn<String>(
    'condition',
    aliasedName,
    false,
    type: DriftSqlType.string,
    requiredDuringInsert: true,
  );
  static const VerificationMeta _tempMinCMeta = const VerificationMeta(
    'tempMinC',
  );
  @override
  late final GeneratedColumn<double> tempMinC = GeneratedColumn<double>(
    'temp_min_c',
    aliasedName,
    true,
    type: DriftSqlType.double,
    requiredDuringInsert: false,
  );
  static const VerificationMeta _tempMaxCMeta = const VerificationMeta(
    'tempMaxC',
  );
  @override
  late final GeneratedColumn<double> tempMaxC = GeneratedColumn<double>(
    'temp_max_c',
    aliasedName,
    true,
    type: DriftSqlType.double,
    requiredDuringInsert: false,
  );
  static const VerificationMeta _rainMmMeta = const VerificationMeta('rainMm');
  @override
  late final GeneratedColumn<double> rainMm = GeneratedColumn<double>(
    'rain_mm',
    aliasedName,
    true,
    type: DriftSqlType.double,
    requiredDuringInsert: false,
  );
  static const VerificationMeta _cachedAtMeta = const VerificationMeta(
    'cachedAt',
  );
  @override
  late final GeneratedColumn<DateTime> cachedAt = GeneratedColumn<DateTime>(
    'cached_at',
    aliasedName,
    false,
    type: DriftSqlType.dateTime,
    requiredDuringInsert: true,
  );
  @override
  List<GeneratedColumn> get $columns => [
    id,
    stopId,
    forDate,
    condition,
    tempMinC,
    tempMaxC,
    rainMm,
    cachedAt,
  ];
  @override
  String get aliasedName => _alias ?? actualTableName;
  @override
  String get actualTableName => $name;
  static const String $name = 'weather_snapshots';
  @override
  VerificationContext validateIntegrity(
    Insertable<WeatherSnapshot> instance, {
    bool isInserting = false,
  }) {
    final context = VerificationContext();
    final data = instance.toColumns(true);
    if (data.containsKey('id')) {
      context.handle(_idMeta, id.isAcceptableOrUnknown(data['id']!, _idMeta));
    }
    if (data.containsKey('stop_id')) {
      context.handle(
        _stopIdMeta,
        stopId.isAcceptableOrUnknown(data['stop_id']!, _stopIdMeta),
      );
    } else if (isInserting) {
      context.missing(_stopIdMeta);
    }
    if (data.containsKey('for_date')) {
      context.handle(
        _forDateMeta,
        forDate.isAcceptableOrUnknown(data['for_date']!, _forDateMeta),
      );
    } else if (isInserting) {
      context.missing(_forDateMeta);
    }
    if (data.containsKey('condition')) {
      context.handle(
        _conditionMeta,
        condition.isAcceptableOrUnknown(data['condition']!, _conditionMeta),
      );
    } else if (isInserting) {
      context.missing(_conditionMeta);
    }
    if (data.containsKey('temp_min_c')) {
      context.handle(
        _tempMinCMeta,
        tempMinC.isAcceptableOrUnknown(data['temp_min_c']!, _tempMinCMeta),
      );
    }
    if (data.containsKey('temp_max_c')) {
      context.handle(
        _tempMaxCMeta,
        tempMaxC.isAcceptableOrUnknown(data['temp_max_c']!, _tempMaxCMeta),
      );
    }
    if (data.containsKey('rain_mm')) {
      context.handle(
        _rainMmMeta,
        rainMm.isAcceptableOrUnknown(data['rain_mm']!, _rainMmMeta),
      );
    }
    if (data.containsKey('cached_at')) {
      context.handle(
        _cachedAtMeta,
        cachedAt.isAcceptableOrUnknown(data['cached_at']!, _cachedAtMeta),
      );
    } else if (isInserting) {
      context.missing(_cachedAtMeta);
    }
    return context;
  }

  @override
  Set<GeneratedColumn> get $primaryKey => {id};
  @override
  List<Set<GeneratedColumn>> get uniqueKeys => [
    {stopId, forDate},
  ];
  @override
  WeatherSnapshot map(Map<String, dynamic> data, {String? tablePrefix}) {
    final effectivePrefix = tablePrefix != null ? '$tablePrefix.' : '';
    return WeatherSnapshot(
      id: attachedDatabase.typeMapping.read(
        DriftSqlType.int,
        data['${effectivePrefix}id'],
      )!,
      stopId: attachedDatabase.typeMapping.read(
        DriftSqlType.int,
        data['${effectivePrefix}stop_id'],
      )!,
      forDate: attachedDatabase.typeMapping.read(
        DriftSqlType.dateTime,
        data['${effectivePrefix}for_date'],
      )!,
      condition: attachedDatabase.typeMapping.read(
        DriftSqlType.string,
        data['${effectivePrefix}condition'],
      )!,
      tempMinC: attachedDatabase.typeMapping.read(
        DriftSqlType.double,
        data['${effectivePrefix}temp_min_c'],
      ),
      tempMaxC: attachedDatabase.typeMapping.read(
        DriftSqlType.double,
        data['${effectivePrefix}temp_max_c'],
      ),
      rainMm: attachedDatabase.typeMapping.read(
        DriftSqlType.double,
        data['${effectivePrefix}rain_mm'],
      ),
      cachedAt: attachedDatabase.typeMapping.read(
        DriftSqlType.dateTime,
        data['${effectivePrefix}cached_at'],
      )!,
    );
  }

  @override
  $WeatherSnapshotsTable createAlias(String alias) {
    return $WeatherSnapshotsTable(attachedDatabase, alias);
  }
}

class WeatherSnapshot extends DataClass implements Insertable<WeatherSnapshot> {
  final int id;
  final int stopId;
  final DateTime forDate;
  final String condition;
  final double? tempMinC;
  final double? tempMaxC;
  final double? rainMm;

  /// NOT nullable, deliberately. A snapshot without an age is a forecast
  /// pretending to be current, which is the exact failure this table exists
  /// to prevent.
  final DateTime cachedAt;
  const WeatherSnapshot({
    required this.id,
    required this.stopId,
    required this.forDate,
    required this.condition,
    this.tempMinC,
    this.tempMaxC,
    this.rainMm,
    required this.cachedAt,
  });
  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    map['id'] = Variable<int>(id);
    map['stop_id'] = Variable<int>(stopId);
    map['for_date'] = Variable<DateTime>(forDate);
    map['condition'] = Variable<String>(condition);
    if (!nullToAbsent || tempMinC != null) {
      map['temp_min_c'] = Variable<double>(tempMinC);
    }
    if (!nullToAbsent || tempMaxC != null) {
      map['temp_max_c'] = Variable<double>(tempMaxC);
    }
    if (!nullToAbsent || rainMm != null) {
      map['rain_mm'] = Variable<double>(rainMm);
    }
    map['cached_at'] = Variable<DateTime>(cachedAt);
    return map;
  }

  WeatherSnapshotsCompanion toCompanion(bool nullToAbsent) {
    return WeatherSnapshotsCompanion(
      id: Value(id),
      stopId: Value(stopId),
      forDate: Value(forDate),
      condition: Value(condition),
      tempMinC: tempMinC == null && nullToAbsent
          ? const Value.absent()
          : Value(tempMinC),
      tempMaxC: tempMaxC == null && nullToAbsent
          ? const Value.absent()
          : Value(tempMaxC),
      rainMm: rainMm == null && nullToAbsent
          ? const Value.absent()
          : Value(rainMm),
      cachedAt: Value(cachedAt),
    );
  }

  factory WeatherSnapshot.fromJson(
    Map<String, dynamic> json, {
    ValueSerializer? serializer,
  }) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return WeatherSnapshot(
      id: serializer.fromJson<int>(json['id']),
      stopId: serializer.fromJson<int>(json['stopId']),
      forDate: serializer.fromJson<DateTime>(json['forDate']),
      condition: serializer.fromJson<String>(json['condition']),
      tempMinC: serializer.fromJson<double?>(json['tempMinC']),
      tempMaxC: serializer.fromJson<double?>(json['tempMaxC']),
      rainMm: serializer.fromJson<double?>(json['rainMm']),
      cachedAt: serializer.fromJson<DateTime>(json['cachedAt']),
    );
  }
  @override
  Map<String, dynamic> toJson({ValueSerializer? serializer}) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return <String, dynamic>{
      'id': serializer.toJson<int>(id),
      'stopId': serializer.toJson<int>(stopId),
      'forDate': serializer.toJson<DateTime>(forDate),
      'condition': serializer.toJson<String>(condition),
      'tempMinC': serializer.toJson<double?>(tempMinC),
      'tempMaxC': serializer.toJson<double?>(tempMaxC),
      'rainMm': serializer.toJson<double?>(rainMm),
      'cachedAt': serializer.toJson<DateTime>(cachedAt),
    };
  }

  WeatherSnapshot copyWith({
    int? id,
    int? stopId,
    DateTime? forDate,
    String? condition,
    Value<double?> tempMinC = const Value.absent(),
    Value<double?> tempMaxC = const Value.absent(),
    Value<double?> rainMm = const Value.absent(),
    DateTime? cachedAt,
  }) => WeatherSnapshot(
    id: id ?? this.id,
    stopId: stopId ?? this.stopId,
    forDate: forDate ?? this.forDate,
    condition: condition ?? this.condition,
    tempMinC: tempMinC.present ? tempMinC.value : this.tempMinC,
    tempMaxC: tempMaxC.present ? tempMaxC.value : this.tempMaxC,
    rainMm: rainMm.present ? rainMm.value : this.rainMm,
    cachedAt: cachedAt ?? this.cachedAt,
  );
  WeatherSnapshot copyWithCompanion(WeatherSnapshotsCompanion data) {
    return WeatherSnapshot(
      id: data.id.present ? data.id.value : this.id,
      stopId: data.stopId.present ? data.stopId.value : this.stopId,
      forDate: data.forDate.present ? data.forDate.value : this.forDate,
      condition: data.condition.present ? data.condition.value : this.condition,
      tempMinC: data.tempMinC.present ? data.tempMinC.value : this.tempMinC,
      tempMaxC: data.tempMaxC.present ? data.tempMaxC.value : this.tempMaxC,
      rainMm: data.rainMm.present ? data.rainMm.value : this.rainMm,
      cachedAt: data.cachedAt.present ? data.cachedAt.value : this.cachedAt,
    );
  }

  @override
  String toString() {
    return (StringBuffer('WeatherSnapshot(')
          ..write('id: $id, ')
          ..write('stopId: $stopId, ')
          ..write('forDate: $forDate, ')
          ..write('condition: $condition, ')
          ..write('tempMinC: $tempMinC, ')
          ..write('tempMaxC: $tempMaxC, ')
          ..write('rainMm: $rainMm, ')
          ..write('cachedAt: $cachedAt')
          ..write(')'))
        .toString();
  }

  @override
  int get hashCode => Object.hash(
    id,
    stopId,
    forDate,
    condition,
    tempMinC,
    tempMaxC,
    rainMm,
    cachedAt,
  );
  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      (other is WeatherSnapshot &&
          other.id == this.id &&
          other.stopId == this.stopId &&
          other.forDate == this.forDate &&
          other.condition == this.condition &&
          other.tempMinC == this.tempMinC &&
          other.tempMaxC == this.tempMaxC &&
          other.rainMm == this.rainMm &&
          other.cachedAt == this.cachedAt);
}

class WeatherSnapshotsCompanion extends UpdateCompanion<WeatherSnapshot> {
  final Value<int> id;
  final Value<int> stopId;
  final Value<DateTime> forDate;
  final Value<String> condition;
  final Value<double?> tempMinC;
  final Value<double?> tempMaxC;
  final Value<double?> rainMm;
  final Value<DateTime> cachedAt;
  const WeatherSnapshotsCompanion({
    this.id = const Value.absent(),
    this.stopId = const Value.absent(),
    this.forDate = const Value.absent(),
    this.condition = const Value.absent(),
    this.tempMinC = const Value.absent(),
    this.tempMaxC = const Value.absent(),
    this.rainMm = const Value.absent(),
    this.cachedAt = const Value.absent(),
  });
  WeatherSnapshotsCompanion.insert({
    this.id = const Value.absent(),
    required int stopId,
    required DateTime forDate,
    required String condition,
    this.tempMinC = const Value.absent(),
    this.tempMaxC = const Value.absent(),
    this.rainMm = const Value.absent(),
    required DateTime cachedAt,
  }) : stopId = Value(stopId),
       forDate = Value(forDate),
       condition = Value(condition),
       cachedAt = Value(cachedAt);
  static Insertable<WeatherSnapshot> custom({
    Expression<int>? id,
    Expression<int>? stopId,
    Expression<DateTime>? forDate,
    Expression<String>? condition,
    Expression<double>? tempMinC,
    Expression<double>? tempMaxC,
    Expression<double>? rainMm,
    Expression<DateTime>? cachedAt,
  }) {
    return RawValuesInsertable({
      if (id != null) 'id': id,
      if (stopId != null) 'stop_id': stopId,
      if (forDate != null) 'for_date': forDate,
      if (condition != null) 'condition': condition,
      if (tempMinC != null) 'temp_min_c': tempMinC,
      if (tempMaxC != null) 'temp_max_c': tempMaxC,
      if (rainMm != null) 'rain_mm': rainMm,
      if (cachedAt != null) 'cached_at': cachedAt,
    });
  }

  WeatherSnapshotsCompanion copyWith({
    Value<int>? id,
    Value<int>? stopId,
    Value<DateTime>? forDate,
    Value<String>? condition,
    Value<double?>? tempMinC,
    Value<double?>? tempMaxC,
    Value<double?>? rainMm,
    Value<DateTime>? cachedAt,
  }) {
    return WeatherSnapshotsCompanion(
      id: id ?? this.id,
      stopId: stopId ?? this.stopId,
      forDate: forDate ?? this.forDate,
      condition: condition ?? this.condition,
      tempMinC: tempMinC ?? this.tempMinC,
      tempMaxC: tempMaxC ?? this.tempMaxC,
      rainMm: rainMm ?? this.rainMm,
      cachedAt: cachedAt ?? this.cachedAt,
    );
  }

  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    if (id.present) {
      map['id'] = Variable<int>(id.value);
    }
    if (stopId.present) {
      map['stop_id'] = Variable<int>(stopId.value);
    }
    if (forDate.present) {
      map['for_date'] = Variable<DateTime>(forDate.value);
    }
    if (condition.present) {
      map['condition'] = Variable<String>(condition.value);
    }
    if (tempMinC.present) {
      map['temp_min_c'] = Variable<double>(tempMinC.value);
    }
    if (tempMaxC.present) {
      map['temp_max_c'] = Variable<double>(tempMaxC.value);
    }
    if (rainMm.present) {
      map['rain_mm'] = Variable<double>(rainMm.value);
    }
    if (cachedAt.present) {
      map['cached_at'] = Variable<DateTime>(cachedAt.value);
    }
    return map;
  }

  @override
  String toString() {
    return (StringBuffer('WeatherSnapshotsCompanion(')
          ..write('id: $id, ')
          ..write('stopId: $stopId, ')
          ..write('forDate: $forDate, ')
          ..write('condition: $condition, ')
          ..write('tempMinC: $tempMinC, ')
          ..write('tempMaxC: $tempMaxC, ')
          ..write('rainMm: $rainMm, ')
          ..write('cachedAt: $cachedAt')
          ..write(')'))
        .toString();
  }
}

class $TravellersTable extends Travellers
    with TableInfo<$TravellersTable, Traveller> {
  @override
  final GeneratedDatabase attachedDatabase;
  final String? _alias;
  $TravellersTable(this.attachedDatabase, [this._alias]);
  static const VerificationMeta _idMeta = const VerificationMeta('id');
  @override
  late final GeneratedColumn<int> id = GeneratedColumn<int>(
    'id',
    aliasedName,
    false,
    hasAutoIncrement: true,
    type: DriftSqlType.int,
    requiredDuringInsert: false,
    defaultConstraints: GeneratedColumn.constraintIsAlways(
      'PRIMARY KEY AUTOINCREMENT',
    ),
  );
  static const VerificationMeta _tripIdMeta = const VerificationMeta('tripId');
  @override
  late final GeneratedColumn<int> tripId = GeneratedColumn<int>(
    'trip_id',
    aliasedName,
    false,
    type: DriftSqlType.int,
    requiredDuringInsert: true,
    defaultConstraints: GeneratedColumn.constraintIsAlways(
      'REFERENCES trips (id) ON DELETE CASCADE',
    ),
  );
  static const VerificationMeta _nameMeta = const VerificationMeta('name');
  @override
  late final GeneratedColumn<String> name = GeneratedColumn<String>(
    'name',
    aliasedName,
    false,
    type: DriftSqlType.string,
    requiredDuringInsert: true,
  );
  static const VerificationMeta _isSelfMeta = const VerificationMeta('isSelf');
  @override
  late final GeneratedColumn<bool> isSelf = GeneratedColumn<bool>(
    'is_self',
    aliasedName,
    false,
    type: DriftSqlType.bool,
    requiredDuringInsert: false,
    defaultConstraints: GeneratedColumn.constraintIsAlways(
      'CHECK ("is_self" IN (0, 1))',
    ),
    defaultValue: const Constant(false),
  );
  @override
  List<GeneratedColumn> get $columns => [id, tripId, name, isSelf];
  @override
  String get aliasedName => _alias ?? actualTableName;
  @override
  String get actualTableName => $name;
  static const String $name = 'travellers';
  @override
  VerificationContext validateIntegrity(
    Insertable<Traveller> instance, {
    bool isInserting = false,
  }) {
    final context = VerificationContext();
    final data = instance.toColumns(true);
    if (data.containsKey('id')) {
      context.handle(_idMeta, id.isAcceptableOrUnknown(data['id']!, _idMeta));
    }
    if (data.containsKey('trip_id')) {
      context.handle(
        _tripIdMeta,
        tripId.isAcceptableOrUnknown(data['trip_id']!, _tripIdMeta),
      );
    } else if (isInserting) {
      context.missing(_tripIdMeta);
    }
    if (data.containsKey('name')) {
      context.handle(
        _nameMeta,
        name.isAcceptableOrUnknown(data['name']!, _nameMeta),
      );
    } else if (isInserting) {
      context.missing(_nameMeta);
    }
    if (data.containsKey('is_self')) {
      context.handle(
        _isSelfMeta,
        isSelf.isAcceptableOrUnknown(data['is_self']!, _isSelfMeta),
      );
    }
    return context;
  }

  @override
  Set<GeneratedColumn> get $primaryKey => {id};
  @override
  Traveller map(Map<String, dynamic> data, {String? tablePrefix}) {
    final effectivePrefix = tablePrefix != null ? '$tablePrefix.' : '';
    return Traveller(
      id: attachedDatabase.typeMapping.read(
        DriftSqlType.int,
        data['${effectivePrefix}id'],
      )!,
      tripId: attachedDatabase.typeMapping.read(
        DriftSqlType.int,
        data['${effectivePrefix}trip_id'],
      )!,
      name: attachedDatabase.typeMapping.read(
        DriftSqlType.string,
        data['${effectivePrefix}name'],
      )!,
      isSelf: attachedDatabase.typeMapping.read(
        DriftSqlType.bool,
        data['${effectivePrefix}is_self'],
      )!,
    );
  }

  @override
  $TravellersTable createAlias(String alias) {
    return $TravellersTable(attachedDatabase, alias);
  }
}

class Traveller extends DataClass implements Insertable<Traveller> {
  final int id;
  final int tripId;
  final String name;
  final bool isSelf;
  const Traveller({
    required this.id,
    required this.tripId,
    required this.name,
    required this.isSelf,
  });
  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    map['id'] = Variable<int>(id);
    map['trip_id'] = Variable<int>(tripId);
    map['name'] = Variable<String>(name);
    map['is_self'] = Variable<bool>(isSelf);
    return map;
  }

  TravellersCompanion toCompanion(bool nullToAbsent) {
    return TravellersCompanion(
      id: Value(id),
      tripId: Value(tripId),
      name: Value(name),
      isSelf: Value(isSelf),
    );
  }

  factory Traveller.fromJson(
    Map<String, dynamic> json, {
    ValueSerializer? serializer,
  }) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return Traveller(
      id: serializer.fromJson<int>(json['id']),
      tripId: serializer.fromJson<int>(json['tripId']),
      name: serializer.fromJson<String>(json['name']),
      isSelf: serializer.fromJson<bool>(json['isSelf']),
    );
  }
  @override
  Map<String, dynamic> toJson({ValueSerializer? serializer}) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return <String, dynamic>{
      'id': serializer.toJson<int>(id),
      'tripId': serializer.toJson<int>(tripId),
      'name': serializer.toJson<String>(name),
      'isSelf': serializer.toJson<bool>(isSelf),
    };
  }

  Traveller copyWith({int? id, int? tripId, String? name, bool? isSelf}) =>
      Traveller(
        id: id ?? this.id,
        tripId: tripId ?? this.tripId,
        name: name ?? this.name,
        isSelf: isSelf ?? this.isSelf,
      );
  Traveller copyWithCompanion(TravellersCompanion data) {
    return Traveller(
      id: data.id.present ? data.id.value : this.id,
      tripId: data.tripId.present ? data.tripId.value : this.tripId,
      name: data.name.present ? data.name.value : this.name,
      isSelf: data.isSelf.present ? data.isSelf.value : this.isSelf,
    );
  }

  @override
  String toString() {
    return (StringBuffer('Traveller(')
          ..write('id: $id, ')
          ..write('tripId: $tripId, ')
          ..write('name: $name, ')
          ..write('isSelf: $isSelf')
          ..write(')'))
        .toString();
  }

  @override
  int get hashCode => Object.hash(id, tripId, name, isSelf);
  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      (other is Traveller &&
          other.id == this.id &&
          other.tripId == this.tripId &&
          other.name == this.name &&
          other.isSelf == this.isSelf);
}

class TravellersCompanion extends UpdateCompanion<Traveller> {
  final Value<int> id;
  final Value<int> tripId;
  final Value<String> name;
  final Value<bool> isSelf;
  const TravellersCompanion({
    this.id = const Value.absent(),
    this.tripId = const Value.absent(),
    this.name = const Value.absent(),
    this.isSelf = const Value.absent(),
  });
  TravellersCompanion.insert({
    this.id = const Value.absent(),
    required int tripId,
    required String name,
    this.isSelf = const Value.absent(),
  }) : tripId = Value(tripId),
       name = Value(name);
  static Insertable<Traveller> custom({
    Expression<int>? id,
    Expression<int>? tripId,
    Expression<String>? name,
    Expression<bool>? isSelf,
  }) {
    return RawValuesInsertable({
      if (id != null) 'id': id,
      if (tripId != null) 'trip_id': tripId,
      if (name != null) 'name': name,
      if (isSelf != null) 'is_self': isSelf,
    });
  }

  TravellersCompanion copyWith({
    Value<int>? id,
    Value<int>? tripId,
    Value<String>? name,
    Value<bool>? isSelf,
  }) {
    return TravellersCompanion(
      id: id ?? this.id,
      tripId: tripId ?? this.tripId,
      name: name ?? this.name,
      isSelf: isSelf ?? this.isSelf,
    );
  }

  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    if (id.present) {
      map['id'] = Variable<int>(id.value);
    }
    if (tripId.present) {
      map['trip_id'] = Variable<int>(tripId.value);
    }
    if (name.present) {
      map['name'] = Variable<String>(name.value);
    }
    if (isSelf.present) {
      map['is_self'] = Variable<bool>(isSelf.value);
    }
    return map;
  }

  @override
  String toString() {
    return (StringBuffer('TravellersCompanion(')
          ..write('id: $id, ')
          ..write('tripId: $tripId, ')
          ..write('name: $name, ')
          ..write('isSelf: $isSelf')
          ..write(')'))
        .toString();
  }
}

class $ExpensesTable extends Expenses with TableInfo<$ExpensesTable, Expense> {
  @override
  final GeneratedDatabase attachedDatabase;
  final String? _alias;
  $ExpensesTable(this.attachedDatabase, [this._alias]);
  static const VerificationMeta _idMeta = const VerificationMeta('id');
  @override
  late final GeneratedColumn<int> id = GeneratedColumn<int>(
    'id',
    aliasedName,
    false,
    hasAutoIncrement: true,
    type: DriftSqlType.int,
    requiredDuringInsert: false,
    defaultConstraints: GeneratedColumn.constraintIsAlways(
      'PRIMARY KEY AUTOINCREMENT',
    ),
  );
  static const VerificationMeta _tripIdMeta = const VerificationMeta('tripId');
  @override
  late final GeneratedColumn<int> tripId = GeneratedColumn<int>(
    'trip_id',
    aliasedName,
    false,
    type: DriftSqlType.int,
    requiredDuringInsert: true,
    defaultConstraints: GeneratedColumn.constraintIsAlways(
      'REFERENCES trips (id) ON DELETE CASCADE',
    ),
  );
  static const VerificationMeta _stopIdMeta = const VerificationMeta('stopId');
  @override
  late final GeneratedColumn<int> stopId = GeneratedColumn<int>(
    'stop_id',
    aliasedName,
    true,
    type: DriftSqlType.int,
    requiredDuringInsert: false,
    defaultConstraints: GeneratedColumn.constraintIsAlways(
      'REFERENCES stops (id) ON DELETE SET NULL',
    ),
  );
  static const VerificationMeta _descriptionMeta = const VerificationMeta(
    'description',
  );
  @override
  late final GeneratedColumn<String> description = GeneratedColumn<String>(
    'description',
    aliasedName,
    false,
    type: DriftSqlType.string,
    requiredDuringInsert: true,
  );
  static const VerificationMeta _amountMinorMeta = const VerificationMeta(
    'amountMinor',
  );
  @override
  late final GeneratedColumn<int> amountMinor = GeneratedColumn<int>(
    'amount_minor',
    aliasedName,
    false,
    type: DriftSqlType.int,
    requiredDuringInsert: true,
  );
  static const VerificationMeta _currencyMeta = const VerificationMeta(
    'currency',
  );
  @override
  late final GeneratedColumn<String> currency = GeneratedColumn<String>(
    'currency',
    aliasedName,
    false,
    additionalChecks: GeneratedColumn.checkTextLength(maxTextLength: 3),
    type: DriftSqlType.string,
    requiredDuringInsert: false,
    defaultValue: const Constant('INR'),
  );
  static const VerificationMeta _rateToBaseMeta = const VerificationMeta(
    'rateToBase',
  );
  @override
  late final GeneratedColumn<double> rateToBase = GeneratedColumn<double>(
    'rate_to_base',
    aliasedName,
    false,
    type: DriftSqlType.double,
    requiredDuringInsert: false,
    defaultValue: const Constant(1.0),
  );
  static const VerificationMeta _rateCapturedAtMeta = const VerificationMeta(
    'rateCapturedAt',
  );
  @override
  late final GeneratedColumn<DateTime> rateCapturedAt =
      GeneratedColumn<DateTime>(
        'rate_captured_at',
        aliasedName,
        true,
        type: DriftSqlType.dateTime,
        requiredDuringInsert: false,
      );
  static const VerificationMeta _paidByIdMeta = const VerificationMeta(
    'paidById',
  );
  @override
  late final GeneratedColumn<int> paidById = GeneratedColumn<int>(
    'paid_by_id',
    aliasedName,
    false,
    type: DriftSqlType.int,
    requiredDuringInsert: true,
    defaultConstraints: GeneratedColumn.constraintIsAlways(
      'REFERENCES travellers (id) ON DELETE CASCADE',
    ),
  );
  static const VerificationMeta _categoryMeta = const VerificationMeta(
    'category',
  );
  @override
  late final GeneratedColumn<String> category = GeneratedColumn<String>(
    'category',
    aliasedName,
    true,
    type: DriftSqlType.string,
    requiredDuringInsert: false,
  );
  static const VerificationMeta _spentAtMeta = const VerificationMeta(
    'spentAt',
  );
  @override
  late final GeneratedColumn<DateTime> spentAt = GeneratedColumn<DateTime>(
    'spent_at',
    aliasedName,
    false,
    type: DriftSqlType.dateTime,
    requiredDuringInsert: false,
    defaultValue: currentDateAndTime,
  );
  @override
  List<GeneratedColumn> get $columns => [
    id,
    tripId,
    stopId,
    description,
    amountMinor,
    currency,
    rateToBase,
    rateCapturedAt,
    paidById,
    category,
    spentAt,
  ];
  @override
  String get aliasedName => _alias ?? actualTableName;
  @override
  String get actualTableName => $name;
  static const String $name = 'expenses';
  @override
  VerificationContext validateIntegrity(
    Insertable<Expense> instance, {
    bool isInserting = false,
  }) {
    final context = VerificationContext();
    final data = instance.toColumns(true);
    if (data.containsKey('id')) {
      context.handle(_idMeta, id.isAcceptableOrUnknown(data['id']!, _idMeta));
    }
    if (data.containsKey('trip_id')) {
      context.handle(
        _tripIdMeta,
        tripId.isAcceptableOrUnknown(data['trip_id']!, _tripIdMeta),
      );
    } else if (isInserting) {
      context.missing(_tripIdMeta);
    }
    if (data.containsKey('stop_id')) {
      context.handle(
        _stopIdMeta,
        stopId.isAcceptableOrUnknown(data['stop_id']!, _stopIdMeta),
      );
    }
    if (data.containsKey('description')) {
      context.handle(
        _descriptionMeta,
        description.isAcceptableOrUnknown(
          data['description']!,
          _descriptionMeta,
        ),
      );
    } else if (isInserting) {
      context.missing(_descriptionMeta);
    }
    if (data.containsKey('amount_minor')) {
      context.handle(
        _amountMinorMeta,
        amountMinor.isAcceptableOrUnknown(
          data['amount_minor']!,
          _amountMinorMeta,
        ),
      );
    } else if (isInserting) {
      context.missing(_amountMinorMeta);
    }
    if (data.containsKey('currency')) {
      context.handle(
        _currencyMeta,
        currency.isAcceptableOrUnknown(data['currency']!, _currencyMeta),
      );
    }
    if (data.containsKey('rate_to_base')) {
      context.handle(
        _rateToBaseMeta,
        rateToBase.isAcceptableOrUnknown(
          data['rate_to_base']!,
          _rateToBaseMeta,
        ),
      );
    }
    if (data.containsKey('rate_captured_at')) {
      context.handle(
        _rateCapturedAtMeta,
        rateCapturedAt.isAcceptableOrUnknown(
          data['rate_captured_at']!,
          _rateCapturedAtMeta,
        ),
      );
    }
    if (data.containsKey('paid_by_id')) {
      context.handle(
        _paidByIdMeta,
        paidById.isAcceptableOrUnknown(data['paid_by_id']!, _paidByIdMeta),
      );
    } else if (isInserting) {
      context.missing(_paidByIdMeta);
    }
    if (data.containsKey('category')) {
      context.handle(
        _categoryMeta,
        category.isAcceptableOrUnknown(data['category']!, _categoryMeta),
      );
    }
    if (data.containsKey('spent_at')) {
      context.handle(
        _spentAtMeta,
        spentAt.isAcceptableOrUnknown(data['spent_at']!, _spentAtMeta),
      );
    }
    return context;
  }

  @override
  Set<GeneratedColumn> get $primaryKey => {id};
  @override
  Expense map(Map<String, dynamic> data, {String? tablePrefix}) {
    final effectivePrefix = tablePrefix != null ? '$tablePrefix.' : '';
    return Expense(
      id: attachedDatabase.typeMapping.read(
        DriftSqlType.int,
        data['${effectivePrefix}id'],
      )!,
      tripId: attachedDatabase.typeMapping.read(
        DriftSqlType.int,
        data['${effectivePrefix}trip_id'],
      )!,
      stopId: attachedDatabase.typeMapping.read(
        DriftSqlType.int,
        data['${effectivePrefix}stop_id'],
      ),
      description: attachedDatabase.typeMapping.read(
        DriftSqlType.string,
        data['${effectivePrefix}description'],
      )!,
      amountMinor: attachedDatabase.typeMapping.read(
        DriftSqlType.int,
        data['${effectivePrefix}amount_minor'],
      )!,
      currency: attachedDatabase.typeMapping.read(
        DriftSqlType.string,
        data['${effectivePrefix}currency'],
      )!,
      rateToBase: attachedDatabase.typeMapping.read(
        DriftSqlType.double,
        data['${effectivePrefix}rate_to_base'],
      )!,
      rateCapturedAt: attachedDatabase.typeMapping.read(
        DriftSqlType.dateTime,
        data['${effectivePrefix}rate_captured_at'],
      ),
      paidById: attachedDatabase.typeMapping.read(
        DriftSqlType.int,
        data['${effectivePrefix}paid_by_id'],
      )!,
      category: attachedDatabase.typeMapping.read(
        DriftSqlType.string,
        data['${effectivePrefix}category'],
      ),
      spentAt: attachedDatabase.typeMapping.read(
        DriftSqlType.dateTime,
        data['${effectivePrefix}spent_at'],
      )!,
    );
  }

  @override
  $ExpensesTable createAlias(String alias) {
    return $ExpensesTable(attachedDatabase, alias);
  }
}

class Expense extends DataClass implements Insertable<Expense> {
  final int id;
  final int tripId;
  final int? stopId;
  final String description;

  /// MINOR UNITS, as an integer. Paise, not rupees. Floating point
  /// accumulates rounding error across a three-way split and this ledger has
  /// to balance exactly.
  final int amountMinor;
  final String currency;

  /// Manual snapshot taken at setup. There is no live rate offline, and the
  /// UI always shows this alongside its capture date.
  final double rateToBase;
  final DateTime? rateCapturedAt;
  final int paidById;
  final String? category;
  final DateTime spentAt;
  const Expense({
    required this.id,
    required this.tripId,
    this.stopId,
    required this.description,
    required this.amountMinor,
    required this.currency,
    required this.rateToBase,
    this.rateCapturedAt,
    required this.paidById,
    this.category,
    required this.spentAt,
  });
  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    map['id'] = Variable<int>(id);
    map['trip_id'] = Variable<int>(tripId);
    if (!nullToAbsent || stopId != null) {
      map['stop_id'] = Variable<int>(stopId);
    }
    map['description'] = Variable<String>(description);
    map['amount_minor'] = Variable<int>(amountMinor);
    map['currency'] = Variable<String>(currency);
    map['rate_to_base'] = Variable<double>(rateToBase);
    if (!nullToAbsent || rateCapturedAt != null) {
      map['rate_captured_at'] = Variable<DateTime>(rateCapturedAt);
    }
    map['paid_by_id'] = Variable<int>(paidById);
    if (!nullToAbsent || category != null) {
      map['category'] = Variable<String>(category);
    }
    map['spent_at'] = Variable<DateTime>(spentAt);
    return map;
  }

  ExpensesCompanion toCompanion(bool nullToAbsent) {
    return ExpensesCompanion(
      id: Value(id),
      tripId: Value(tripId),
      stopId: stopId == null && nullToAbsent
          ? const Value.absent()
          : Value(stopId),
      description: Value(description),
      amountMinor: Value(amountMinor),
      currency: Value(currency),
      rateToBase: Value(rateToBase),
      rateCapturedAt: rateCapturedAt == null && nullToAbsent
          ? const Value.absent()
          : Value(rateCapturedAt),
      paidById: Value(paidById),
      category: category == null && nullToAbsent
          ? const Value.absent()
          : Value(category),
      spentAt: Value(spentAt),
    );
  }

  factory Expense.fromJson(
    Map<String, dynamic> json, {
    ValueSerializer? serializer,
  }) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return Expense(
      id: serializer.fromJson<int>(json['id']),
      tripId: serializer.fromJson<int>(json['tripId']),
      stopId: serializer.fromJson<int?>(json['stopId']),
      description: serializer.fromJson<String>(json['description']),
      amountMinor: serializer.fromJson<int>(json['amountMinor']),
      currency: serializer.fromJson<String>(json['currency']),
      rateToBase: serializer.fromJson<double>(json['rateToBase']),
      rateCapturedAt: serializer.fromJson<DateTime?>(json['rateCapturedAt']),
      paidById: serializer.fromJson<int>(json['paidById']),
      category: serializer.fromJson<String?>(json['category']),
      spentAt: serializer.fromJson<DateTime>(json['spentAt']),
    );
  }
  @override
  Map<String, dynamic> toJson({ValueSerializer? serializer}) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return <String, dynamic>{
      'id': serializer.toJson<int>(id),
      'tripId': serializer.toJson<int>(tripId),
      'stopId': serializer.toJson<int?>(stopId),
      'description': serializer.toJson<String>(description),
      'amountMinor': serializer.toJson<int>(amountMinor),
      'currency': serializer.toJson<String>(currency),
      'rateToBase': serializer.toJson<double>(rateToBase),
      'rateCapturedAt': serializer.toJson<DateTime?>(rateCapturedAt),
      'paidById': serializer.toJson<int>(paidById),
      'category': serializer.toJson<String?>(category),
      'spentAt': serializer.toJson<DateTime>(spentAt),
    };
  }

  Expense copyWith({
    int? id,
    int? tripId,
    Value<int?> stopId = const Value.absent(),
    String? description,
    int? amountMinor,
    String? currency,
    double? rateToBase,
    Value<DateTime?> rateCapturedAt = const Value.absent(),
    int? paidById,
    Value<String?> category = const Value.absent(),
    DateTime? spentAt,
  }) => Expense(
    id: id ?? this.id,
    tripId: tripId ?? this.tripId,
    stopId: stopId.present ? stopId.value : this.stopId,
    description: description ?? this.description,
    amountMinor: amountMinor ?? this.amountMinor,
    currency: currency ?? this.currency,
    rateToBase: rateToBase ?? this.rateToBase,
    rateCapturedAt: rateCapturedAt.present
        ? rateCapturedAt.value
        : this.rateCapturedAt,
    paidById: paidById ?? this.paidById,
    category: category.present ? category.value : this.category,
    spentAt: spentAt ?? this.spentAt,
  );
  Expense copyWithCompanion(ExpensesCompanion data) {
    return Expense(
      id: data.id.present ? data.id.value : this.id,
      tripId: data.tripId.present ? data.tripId.value : this.tripId,
      stopId: data.stopId.present ? data.stopId.value : this.stopId,
      description: data.description.present
          ? data.description.value
          : this.description,
      amountMinor: data.amountMinor.present
          ? data.amountMinor.value
          : this.amountMinor,
      currency: data.currency.present ? data.currency.value : this.currency,
      rateToBase: data.rateToBase.present
          ? data.rateToBase.value
          : this.rateToBase,
      rateCapturedAt: data.rateCapturedAt.present
          ? data.rateCapturedAt.value
          : this.rateCapturedAt,
      paidById: data.paidById.present ? data.paidById.value : this.paidById,
      category: data.category.present ? data.category.value : this.category,
      spentAt: data.spentAt.present ? data.spentAt.value : this.spentAt,
    );
  }

  @override
  String toString() {
    return (StringBuffer('Expense(')
          ..write('id: $id, ')
          ..write('tripId: $tripId, ')
          ..write('stopId: $stopId, ')
          ..write('description: $description, ')
          ..write('amountMinor: $amountMinor, ')
          ..write('currency: $currency, ')
          ..write('rateToBase: $rateToBase, ')
          ..write('rateCapturedAt: $rateCapturedAt, ')
          ..write('paidById: $paidById, ')
          ..write('category: $category, ')
          ..write('spentAt: $spentAt')
          ..write(')'))
        .toString();
  }

  @override
  int get hashCode => Object.hash(
    id,
    tripId,
    stopId,
    description,
    amountMinor,
    currency,
    rateToBase,
    rateCapturedAt,
    paidById,
    category,
    spentAt,
  );
  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      (other is Expense &&
          other.id == this.id &&
          other.tripId == this.tripId &&
          other.stopId == this.stopId &&
          other.description == this.description &&
          other.amountMinor == this.amountMinor &&
          other.currency == this.currency &&
          other.rateToBase == this.rateToBase &&
          other.rateCapturedAt == this.rateCapturedAt &&
          other.paidById == this.paidById &&
          other.category == this.category &&
          other.spentAt == this.spentAt);
}

class ExpensesCompanion extends UpdateCompanion<Expense> {
  final Value<int> id;
  final Value<int> tripId;
  final Value<int?> stopId;
  final Value<String> description;
  final Value<int> amountMinor;
  final Value<String> currency;
  final Value<double> rateToBase;
  final Value<DateTime?> rateCapturedAt;
  final Value<int> paidById;
  final Value<String?> category;
  final Value<DateTime> spentAt;
  const ExpensesCompanion({
    this.id = const Value.absent(),
    this.tripId = const Value.absent(),
    this.stopId = const Value.absent(),
    this.description = const Value.absent(),
    this.amountMinor = const Value.absent(),
    this.currency = const Value.absent(),
    this.rateToBase = const Value.absent(),
    this.rateCapturedAt = const Value.absent(),
    this.paidById = const Value.absent(),
    this.category = const Value.absent(),
    this.spentAt = const Value.absent(),
  });
  ExpensesCompanion.insert({
    this.id = const Value.absent(),
    required int tripId,
    this.stopId = const Value.absent(),
    required String description,
    required int amountMinor,
    this.currency = const Value.absent(),
    this.rateToBase = const Value.absent(),
    this.rateCapturedAt = const Value.absent(),
    required int paidById,
    this.category = const Value.absent(),
    this.spentAt = const Value.absent(),
  }) : tripId = Value(tripId),
       description = Value(description),
       amountMinor = Value(amountMinor),
       paidById = Value(paidById);
  static Insertable<Expense> custom({
    Expression<int>? id,
    Expression<int>? tripId,
    Expression<int>? stopId,
    Expression<String>? description,
    Expression<int>? amountMinor,
    Expression<String>? currency,
    Expression<double>? rateToBase,
    Expression<DateTime>? rateCapturedAt,
    Expression<int>? paidById,
    Expression<String>? category,
    Expression<DateTime>? spentAt,
  }) {
    return RawValuesInsertable({
      if (id != null) 'id': id,
      if (tripId != null) 'trip_id': tripId,
      if (stopId != null) 'stop_id': stopId,
      if (description != null) 'description': description,
      if (amountMinor != null) 'amount_minor': amountMinor,
      if (currency != null) 'currency': currency,
      if (rateToBase != null) 'rate_to_base': rateToBase,
      if (rateCapturedAt != null) 'rate_captured_at': rateCapturedAt,
      if (paidById != null) 'paid_by_id': paidById,
      if (category != null) 'category': category,
      if (spentAt != null) 'spent_at': spentAt,
    });
  }

  ExpensesCompanion copyWith({
    Value<int>? id,
    Value<int>? tripId,
    Value<int?>? stopId,
    Value<String>? description,
    Value<int>? amountMinor,
    Value<String>? currency,
    Value<double>? rateToBase,
    Value<DateTime?>? rateCapturedAt,
    Value<int>? paidById,
    Value<String?>? category,
    Value<DateTime>? spentAt,
  }) {
    return ExpensesCompanion(
      id: id ?? this.id,
      tripId: tripId ?? this.tripId,
      stopId: stopId ?? this.stopId,
      description: description ?? this.description,
      amountMinor: amountMinor ?? this.amountMinor,
      currency: currency ?? this.currency,
      rateToBase: rateToBase ?? this.rateToBase,
      rateCapturedAt: rateCapturedAt ?? this.rateCapturedAt,
      paidById: paidById ?? this.paidById,
      category: category ?? this.category,
      spentAt: spentAt ?? this.spentAt,
    );
  }

  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    if (id.present) {
      map['id'] = Variable<int>(id.value);
    }
    if (tripId.present) {
      map['trip_id'] = Variable<int>(tripId.value);
    }
    if (stopId.present) {
      map['stop_id'] = Variable<int>(stopId.value);
    }
    if (description.present) {
      map['description'] = Variable<String>(description.value);
    }
    if (amountMinor.present) {
      map['amount_minor'] = Variable<int>(amountMinor.value);
    }
    if (currency.present) {
      map['currency'] = Variable<String>(currency.value);
    }
    if (rateToBase.present) {
      map['rate_to_base'] = Variable<double>(rateToBase.value);
    }
    if (rateCapturedAt.present) {
      map['rate_captured_at'] = Variable<DateTime>(rateCapturedAt.value);
    }
    if (paidById.present) {
      map['paid_by_id'] = Variable<int>(paidById.value);
    }
    if (category.present) {
      map['category'] = Variable<String>(category.value);
    }
    if (spentAt.present) {
      map['spent_at'] = Variable<DateTime>(spentAt.value);
    }
    return map;
  }

  @override
  String toString() {
    return (StringBuffer('ExpensesCompanion(')
          ..write('id: $id, ')
          ..write('tripId: $tripId, ')
          ..write('stopId: $stopId, ')
          ..write('description: $description, ')
          ..write('amountMinor: $amountMinor, ')
          ..write('currency: $currency, ')
          ..write('rateToBase: $rateToBase, ')
          ..write('rateCapturedAt: $rateCapturedAt, ')
          ..write('paidById: $paidById, ')
          ..write('category: $category, ')
          ..write('spentAt: $spentAt')
          ..write(')'))
        .toString();
  }
}

class $ExpenseSplitsTable extends ExpenseSplits
    with TableInfo<$ExpenseSplitsTable, ExpenseSplit> {
  @override
  final GeneratedDatabase attachedDatabase;
  final String? _alias;
  $ExpenseSplitsTable(this.attachedDatabase, [this._alias]);
  static const VerificationMeta _idMeta = const VerificationMeta('id');
  @override
  late final GeneratedColumn<int> id = GeneratedColumn<int>(
    'id',
    aliasedName,
    false,
    hasAutoIncrement: true,
    type: DriftSqlType.int,
    requiredDuringInsert: false,
    defaultConstraints: GeneratedColumn.constraintIsAlways(
      'PRIMARY KEY AUTOINCREMENT',
    ),
  );
  static const VerificationMeta _expenseIdMeta = const VerificationMeta(
    'expenseId',
  );
  @override
  late final GeneratedColumn<int> expenseId = GeneratedColumn<int>(
    'expense_id',
    aliasedName,
    false,
    type: DriftSqlType.int,
    requiredDuringInsert: true,
    defaultConstraints: GeneratedColumn.constraintIsAlways(
      'REFERENCES expenses (id) ON DELETE CASCADE',
    ),
  );
  static const VerificationMeta _travellerIdMeta = const VerificationMeta(
    'travellerId',
  );
  @override
  late final GeneratedColumn<int> travellerId = GeneratedColumn<int>(
    'traveller_id',
    aliasedName,
    false,
    type: DriftSqlType.int,
    requiredDuringInsert: true,
    defaultConstraints: GeneratedColumn.constraintIsAlways(
      'REFERENCES travellers (id) ON DELETE CASCADE',
    ),
  );
  static const VerificationMeta _shareMinorMeta = const VerificationMeta(
    'shareMinor',
  );
  @override
  late final GeneratedColumn<int> shareMinor = GeneratedColumn<int>(
    'share_minor',
    aliasedName,
    false,
    type: DriftSqlType.int,
    requiredDuringInsert: true,
  );
  @override
  List<GeneratedColumn> get $columns => [
    id,
    expenseId,
    travellerId,
    shareMinor,
  ];
  @override
  String get aliasedName => _alias ?? actualTableName;
  @override
  String get actualTableName => $name;
  static const String $name = 'expense_splits';
  @override
  VerificationContext validateIntegrity(
    Insertable<ExpenseSplit> instance, {
    bool isInserting = false,
  }) {
    final context = VerificationContext();
    final data = instance.toColumns(true);
    if (data.containsKey('id')) {
      context.handle(_idMeta, id.isAcceptableOrUnknown(data['id']!, _idMeta));
    }
    if (data.containsKey('expense_id')) {
      context.handle(
        _expenseIdMeta,
        expenseId.isAcceptableOrUnknown(data['expense_id']!, _expenseIdMeta),
      );
    } else if (isInserting) {
      context.missing(_expenseIdMeta);
    }
    if (data.containsKey('traveller_id')) {
      context.handle(
        _travellerIdMeta,
        travellerId.isAcceptableOrUnknown(
          data['traveller_id']!,
          _travellerIdMeta,
        ),
      );
    } else if (isInserting) {
      context.missing(_travellerIdMeta);
    }
    if (data.containsKey('share_minor')) {
      context.handle(
        _shareMinorMeta,
        shareMinor.isAcceptableOrUnknown(data['share_minor']!, _shareMinorMeta),
      );
    } else if (isInserting) {
      context.missing(_shareMinorMeta);
    }
    return context;
  }

  @override
  Set<GeneratedColumn> get $primaryKey => {id};
  @override
  List<Set<GeneratedColumn>> get uniqueKeys => [
    {expenseId, travellerId},
  ];
  @override
  ExpenseSplit map(Map<String, dynamic> data, {String? tablePrefix}) {
    final effectivePrefix = tablePrefix != null ? '$tablePrefix.' : '';
    return ExpenseSplit(
      id: attachedDatabase.typeMapping.read(
        DriftSqlType.int,
        data['${effectivePrefix}id'],
      )!,
      expenseId: attachedDatabase.typeMapping.read(
        DriftSqlType.int,
        data['${effectivePrefix}expense_id'],
      )!,
      travellerId: attachedDatabase.typeMapping.read(
        DriftSqlType.int,
        data['${effectivePrefix}traveller_id'],
      )!,
      shareMinor: attachedDatabase.typeMapping.read(
        DriftSqlType.int,
        data['${effectivePrefix}share_minor'],
      )!,
    );
  }

  @override
  $ExpenseSplitsTable createAlias(String alias) {
    return $ExpenseSplitsTable(attachedDatabase, alias);
  }
}

class ExpenseSplit extends DataClass implements Insertable<ExpenseSplit> {
  final int id;
  final int expenseId;
  final int travellerId;

  /// Minor units again. The shares of one expense must sum to its
  /// `amountMinor` exactly — #32 asserts this.
  final int shareMinor;
  const ExpenseSplit({
    required this.id,
    required this.expenseId,
    required this.travellerId,
    required this.shareMinor,
  });
  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    map['id'] = Variable<int>(id);
    map['expense_id'] = Variable<int>(expenseId);
    map['traveller_id'] = Variable<int>(travellerId);
    map['share_minor'] = Variable<int>(shareMinor);
    return map;
  }

  ExpenseSplitsCompanion toCompanion(bool nullToAbsent) {
    return ExpenseSplitsCompanion(
      id: Value(id),
      expenseId: Value(expenseId),
      travellerId: Value(travellerId),
      shareMinor: Value(shareMinor),
    );
  }

  factory ExpenseSplit.fromJson(
    Map<String, dynamic> json, {
    ValueSerializer? serializer,
  }) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return ExpenseSplit(
      id: serializer.fromJson<int>(json['id']),
      expenseId: serializer.fromJson<int>(json['expenseId']),
      travellerId: serializer.fromJson<int>(json['travellerId']),
      shareMinor: serializer.fromJson<int>(json['shareMinor']),
    );
  }
  @override
  Map<String, dynamic> toJson({ValueSerializer? serializer}) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return <String, dynamic>{
      'id': serializer.toJson<int>(id),
      'expenseId': serializer.toJson<int>(expenseId),
      'travellerId': serializer.toJson<int>(travellerId),
      'shareMinor': serializer.toJson<int>(shareMinor),
    };
  }

  ExpenseSplit copyWith({
    int? id,
    int? expenseId,
    int? travellerId,
    int? shareMinor,
  }) => ExpenseSplit(
    id: id ?? this.id,
    expenseId: expenseId ?? this.expenseId,
    travellerId: travellerId ?? this.travellerId,
    shareMinor: shareMinor ?? this.shareMinor,
  );
  ExpenseSplit copyWithCompanion(ExpenseSplitsCompanion data) {
    return ExpenseSplit(
      id: data.id.present ? data.id.value : this.id,
      expenseId: data.expenseId.present ? data.expenseId.value : this.expenseId,
      travellerId: data.travellerId.present
          ? data.travellerId.value
          : this.travellerId,
      shareMinor: data.shareMinor.present
          ? data.shareMinor.value
          : this.shareMinor,
    );
  }

  @override
  String toString() {
    return (StringBuffer('ExpenseSplit(')
          ..write('id: $id, ')
          ..write('expenseId: $expenseId, ')
          ..write('travellerId: $travellerId, ')
          ..write('shareMinor: $shareMinor')
          ..write(')'))
        .toString();
  }

  @override
  int get hashCode => Object.hash(id, expenseId, travellerId, shareMinor);
  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      (other is ExpenseSplit &&
          other.id == this.id &&
          other.expenseId == this.expenseId &&
          other.travellerId == this.travellerId &&
          other.shareMinor == this.shareMinor);
}

class ExpenseSplitsCompanion extends UpdateCompanion<ExpenseSplit> {
  final Value<int> id;
  final Value<int> expenseId;
  final Value<int> travellerId;
  final Value<int> shareMinor;
  const ExpenseSplitsCompanion({
    this.id = const Value.absent(),
    this.expenseId = const Value.absent(),
    this.travellerId = const Value.absent(),
    this.shareMinor = const Value.absent(),
  });
  ExpenseSplitsCompanion.insert({
    this.id = const Value.absent(),
    required int expenseId,
    required int travellerId,
    required int shareMinor,
  }) : expenseId = Value(expenseId),
       travellerId = Value(travellerId),
       shareMinor = Value(shareMinor);
  static Insertable<ExpenseSplit> custom({
    Expression<int>? id,
    Expression<int>? expenseId,
    Expression<int>? travellerId,
    Expression<int>? shareMinor,
  }) {
    return RawValuesInsertable({
      if (id != null) 'id': id,
      if (expenseId != null) 'expense_id': expenseId,
      if (travellerId != null) 'traveller_id': travellerId,
      if (shareMinor != null) 'share_minor': shareMinor,
    });
  }

  ExpenseSplitsCompanion copyWith({
    Value<int>? id,
    Value<int>? expenseId,
    Value<int>? travellerId,
    Value<int>? shareMinor,
  }) {
    return ExpenseSplitsCompanion(
      id: id ?? this.id,
      expenseId: expenseId ?? this.expenseId,
      travellerId: travellerId ?? this.travellerId,
      shareMinor: shareMinor ?? this.shareMinor,
    );
  }

  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    if (id.present) {
      map['id'] = Variable<int>(id.value);
    }
    if (expenseId.present) {
      map['expense_id'] = Variable<int>(expenseId.value);
    }
    if (travellerId.present) {
      map['traveller_id'] = Variable<int>(travellerId.value);
    }
    if (shareMinor.present) {
      map['share_minor'] = Variable<int>(shareMinor.value);
    }
    return map;
  }

  @override
  String toString() {
    return (StringBuffer('ExpenseSplitsCompanion(')
          ..write('id: $id, ')
          ..write('expenseId: $expenseId, ')
          ..write('travellerId: $travellerId, ')
          ..write('shareMinor: $shareMinor')
          ..write(')'))
        .toString();
  }
}

class $TimelineEntriesTable extends TimelineEntries
    with TableInfo<$TimelineEntriesTable, TimelineEntry> {
  @override
  final GeneratedDatabase attachedDatabase;
  final String? _alias;
  $TimelineEntriesTable(this.attachedDatabase, [this._alias]);
  static const VerificationMeta _idMeta = const VerificationMeta('id');
  @override
  late final GeneratedColumn<int> id = GeneratedColumn<int>(
    'id',
    aliasedName,
    false,
    hasAutoIncrement: true,
    type: DriftSqlType.int,
    requiredDuringInsert: false,
    defaultConstraints: GeneratedColumn.constraintIsAlways(
      'PRIMARY KEY AUTOINCREMENT',
    ),
  );
  static const VerificationMeta _tripIdMeta = const VerificationMeta('tripId');
  @override
  late final GeneratedColumn<int> tripId = GeneratedColumn<int>(
    'trip_id',
    aliasedName,
    false,
    type: DriftSqlType.int,
    requiredDuringInsert: true,
    defaultConstraints: GeneratedColumn.constraintIsAlways(
      'REFERENCES trips (id) ON DELETE CASCADE',
    ),
  );
  static const VerificationMeta _stopIdMeta = const VerificationMeta('stopId');
  @override
  late final GeneratedColumn<int> stopId = GeneratedColumn<int>(
    'stop_id',
    aliasedName,
    true,
    type: DriftSqlType.int,
    requiredDuringInsert: false,
    defaultConstraints: GeneratedColumn.constraintIsAlways(
      'REFERENCES stops (id) ON DELETE SET NULL',
    ),
  );
  static const VerificationMeta _kindMeta = const VerificationMeta('kind');
  @override
  late final GeneratedColumn<String> kind = GeneratedColumn<String>(
    'kind',
    aliasedName,
    false,
    type: DriftSqlType.string,
    requiredDuringInsert: true,
  );
  static const VerificationMeta _titleMeta = const VerificationMeta('title');
  @override
  late final GeneratedColumn<String> title = GeneratedColumn<String>(
    'title',
    aliasedName,
    true,
    type: DriftSqlType.string,
    requiredDuringInsert: false,
  );
  static const VerificationMeta _bodyMeta = const VerificationMeta('body');
  @override
  late final GeneratedColumn<String> body = GeneratedColumn<String>(
    'body',
    aliasedName,
    true,
    type: DriftSqlType.string,
    requiredDuringInsert: false,
  );
  static const VerificationMeta _latMeta = const VerificationMeta('lat');
  @override
  late final GeneratedColumn<double> lat = GeneratedColumn<double>(
    'lat',
    aliasedName,
    true,
    type: DriftSqlType.double,
    requiredDuringInsert: false,
  );
  static const VerificationMeta _lonMeta = const VerificationMeta('lon');
  @override
  late final GeneratedColumn<double> lon = GeneratedColumn<double>(
    'lon',
    aliasedName,
    true,
    type: DriftSqlType.double,
    requiredDuringInsert: false,
  );
  static const VerificationMeta _accuracyMMeta = const VerificationMeta(
    'accuracyM',
  );
  @override
  late final GeneratedColumn<double> accuracyM = GeneratedColumn<double>(
    'accuracy_m',
    aliasedName,
    true,
    type: DriftSqlType.double,
    requiredDuringInsert: false,
  );
  static const VerificationMeta _photoPathsMeta = const VerificationMeta(
    'photoPaths',
  );
  @override
  late final GeneratedColumn<String> photoPaths = GeneratedColumn<String>(
    'photo_paths',
    aliasedName,
    true,
    type: DriftSqlType.string,
    requiredDuringInsert: false,
  );
  static const VerificationMeta _occurredAtMeta = const VerificationMeta(
    'occurredAt',
  );
  @override
  late final GeneratedColumn<DateTime> occurredAt = GeneratedColumn<DateTime>(
    'occurred_at',
    aliasedName,
    false,
    type: DriftSqlType.dateTime,
    requiredDuringInsert: true,
  );
  @override
  List<GeneratedColumn> get $columns => [
    id,
    tripId,
    stopId,
    kind,
    title,
    body,
    lat,
    lon,
    accuracyM,
    photoPaths,
    occurredAt,
  ];
  @override
  String get aliasedName => _alias ?? actualTableName;
  @override
  String get actualTableName => $name;
  static const String $name = 'timeline_entries';
  @override
  VerificationContext validateIntegrity(
    Insertable<TimelineEntry> instance, {
    bool isInserting = false,
  }) {
    final context = VerificationContext();
    final data = instance.toColumns(true);
    if (data.containsKey('id')) {
      context.handle(_idMeta, id.isAcceptableOrUnknown(data['id']!, _idMeta));
    }
    if (data.containsKey('trip_id')) {
      context.handle(
        _tripIdMeta,
        tripId.isAcceptableOrUnknown(data['trip_id']!, _tripIdMeta),
      );
    } else if (isInserting) {
      context.missing(_tripIdMeta);
    }
    if (data.containsKey('stop_id')) {
      context.handle(
        _stopIdMeta,
        stopId.isAcceptableOrUnknown(data['stop_id']!, _stopIdMeta),
      );
    }
    if (data.containsKey('kind')) {
      context.handle(
        _kindMeta,
        kind.isAcceptableOrUnknown(data['kind']!, _kindMeta),
      );
    } else if (isInserting) {
      context.missing(_kindMeta);
    }
    if (data.containsKey('title')) {
      context.handle(
        _titleMeta,
        title.isAcceptableOrUnknown(data['title']!, _titleMeta),
      );
    }
    if (data.containsKey('body')) {
      context.handle(
        _bodyMeta,
        body.isAcceptableOrUnknown(data['body']!, _bodyMeta),
      );
    }
    if (data.containsKey('lat')) {
      context.handle(
        _latMeta,
        lat.isAcceptableOrUnknown(data['lat']!, _latMeta),
      );
    }
    if (data.containsKey('lon')) {
      context.handle(
        _lonMeta,
        lon.isAcceptableOrUnknown(data['lon']!, _lonMeta),
      );
    }
    if (data.containsKey('accuracy_m')) {
      context.handle(
        _accuracyMMeta,
        accuracyM.isAcceptableOrUnknown(data['accuracy_m']!, _accuracyMMeta),
      );
    }
    if (data.containsKey('photo_paths')) {
      context.handle(
        _photoPathsMeta,
        photoPaths.isAcceptableOrUnknown(data['photo_paths']!, _photoPathsMeta),
      );
    }
    if (data.containsKey('occurred_at')) {
      context.handle(
        _occurredAtMeta,
        occurredAt.isAcceptableOrUnknown(data['occurred_at']!, _occurredAtMeta),
      );
    } else if (isInserting) {
      context.missing(_occurredAtMeta);
    }
    return context;
  }

  @override
  Set<GeneratedColumn> get $primaryKey => {id};
  @override
  TimelineEntry map(Map<String, dynamic> data, {String? tablePrefix}) {
    final effectivePrefix = tablePrefix != null ? '$tablePrefix.' : '';
    return TimelineEntry(
      id: attachedDatabase.typeMapping.read(
        DriftSqlType.int,
        data['${effectivePrefix}id'],
      )!,
      tripId: attachedDatabase.typeMapping.read(
        DriftSqlType.int,
        data['${effectivePrefix}trip_id'],
      )!,
      stopId: attachedDatabase.typeMapping.read(
        DriftSqlType.int,
        data['${effectivePrefix}stop_id'],
      ),
      kind: attachedDatabase.typeMapping.read(
        DriftSqlType.string,
        data['${effectivePrefix}kind'],
      )!,
      title: attachedDatabase.typeMapping.read(
        DriftSqlType.string,
        data['${effectivePrefix}title'],
      ),
      body: attachedDatabase.typeMapping.read(
        DriftSqlType.string,
        data['${effectivePrefix}body'],
      ),
      lat: attachedDatabase.typeMapping.read(
        DriftSqlType.double,
        data['${effectivePrefix}lat'],
      ),
      lon: attachedDatabase.typeMapping.read(
        DriftSqlType.double,
        data['${effectivePrefix}lon'],
      ),
      accuracyM: attachedDatabase.typeMapping.read(
        DriftSqlType.double,
        data['${effectivePrefix}accuracy_m'],
      ),
      photoPaths: attachedDatabase.typeMapping.read(
        DriftSqlType.string,
        data['${effectivePrefix}photo_paths'],
      ),
      occurredAt: attachedDatabase.typeMapping.read(
        DriftSqlType.dateTime,
        data['${effectivePrefix}occurred_at'],
      )!,
    );
  }

  @override
  $TimelineEntriesTable createAlias(String alias) {
    return $TimelineEntriesTable(attachedDatabase, alias);
  }
}

class TimelineEntry extends DataClass implements Insertable<TimelineEntry> {
  final int id;
  final int tripId;
  final int? stopId;

  /// arrival | departure | note | photo | fix
  final String kind;
  final String? title;
  final String? body;
  final double? lat;
  final double? lon;
  final double? accuracyM;

  /// Newline-separated local file paths. Photos stay on the device.
  final String? photoPaths;
  final DateTime occurredAt;
  const TimelineEntry({
    required this.id,
    required this.tripId,
    this.stopId,
    required this.kind,
    this.title,
    this.body,
    this.lat,
    this.lon,
    this.accuracyM,
    this.photoPaths,
    required this.occurredAt,
  });
  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    map['id'] = Variable<int>(id);
    map['trip_id'] = Variable<int>(tripId);
    if (!nullToAbsent || stopId != null) {
      map['stop_id'] = Variable<int>(stopId);
    }
    map['kind'] = Variable<String>(kind);
    if (!nullToAbsent || title != null) {
      map['title'] = Variable<String>(title);
    }
    if (!nullToAbsent || body != null) {
      map['body'] = Variable<String>(body);
    }
    if (!nullToAbsent || lat != null) {
      map['lat'] = Variable<double>(lat);
    }
    if (!nullToAbsent || lon != null) {
      map['lon'] = Variable<double>(lon);
    }
    if (!nullToAbsent || accuracyM != null) {
      map['accuracy_m'] = Variable<double>(accuracyM);
    }
    if (!nullToAbsent || photoPaths != null) {
      map['photo_paths'] = Variable<String>(photoPaths);
    }
    map['occurred_at'] = Variable<DateTime>(occurredAt);
    return map;
  }

  TimelineEntriesCompanion toCompanion(bool nullToAbsent) {
    return TimelineEntriesCompanion(
      id: Value(id),
      tripId: Value(tripId),
      stopId: stopId == null && nullToAbsent
          ? const Value.absent()
          : Value(stopId),
      kind: Value(kind),
      title: title == null && nullToAbsent
          ? const Value.absent()
          : Value(title),
      body: body == null && nullToAbsent ? const Value.absent() : Value(body),
      lat: lat == null && nullToAbsent ? const Value.absent() : Value(lat),
      lon: lon == null && nullToAbsent ? const Value.absent() : Value(lon),
      accuracyM: accuracyM == null && nullToAbsent
          ? const Value.absent()
          : Value(accuracyM),
      photoPaths: photoPaths == null && nullToAbsent
          ? const Value.absent()
          : Value(photoPaths),
      occurredAt: Value(occurredAt),
    );
  }

  factory TimelineEntry.fromJson(
    Map<String, dynamic> json, {
    ValueSerializer? serializer,
  }) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return TimelineEntry(
      id: serializer.fromJson<int>(json['id']),
      tripId: serializer.fromJson<int>(json['tripId']),
      stopId: serializer.fromJson<int?>(json['stopId']),
      kind: serializer.fromJson<String>(json['kind']),
      title: serializer.fromJson<String?>(json['title']),
      body: serializer.fromJson<String?>(json['body']),
      lat: serializer.fromJson<double?>(json['lat']),
      lon: serializer.fromJson<double?>(json['lon']),
      accuracyM: serializer.fromJson<double?>(json['accuracyM']),
      photoPaths: serializer.fromJson<String?>(json['photoPaths']),
      occurredAt: serializer.fromJson<DateTime>(json['occurredAt']),
    );
  }
  @override
  Map<String, dynamic> toJson({ValueSerializer? serializer}) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return <String, dynamic>{
      'id': serializer.toJson<int>(id),
      'tripId': serializer.toJson<int>(tripId),
      'stopId': serializer.toJson<int?>(stopId),
      'kind': serializer.toJson<String>(kind),
      'title': serializer.toJson<String?>(title),
      'body': serializer.toJson<String?>(body),
      'lat': serializer.toJson<double?>(lat),
      'lon': serializer.toJson<double?>(lon),
      'accuracyM': serializer.toJson<double?>(accuracyM),
      'photoPaths': serializer.toJson<String?>(photoPaths),
      'occurredAt': serializer.toJson<DateTime>(occurredAt),
    };
  }

  TimelineEntry copyWith({
    int? id,
    int? tripId,
    Value<int?> stopId = const Value.absent(),
    String? kind,
    Value<String?> title = const Value.absent(),
    Value<String?> body = const Value.absent(),
    Value<double?> lat = const Value.absent(),
    Value<double?> lon = const Value.absent(),
    Value<double?> accuracyM = const Value.absent(),
    Value<String?> photoPaths = const Value.absent(),
    DateTime? occurredAt,
  }) => TimelineEntry(
    id: id ?? this.id,
    tripId: tripId ?? this.tripId,
    stopId: stopId.present ? stopId.value : this.stopId,
    kind: kind ?? this.kind,
    title: title.present ? title.value : this.title,
    body: body.present ? body.value : this.body,
    lat: lat.present ? lat.value : this.lat,
    lon: lon.present ? lon.value : this.lon,
    accuracyM: accuracyM.present ? accuracyM.value : this.accuracyM,
    photoPaths: photoPaths.present ? photoPaths.value : this.photoPaths,
    occurredAt: occurredAt ?? this.occurredAt,
  );
  TimelineEntry copyWithCompanion(TimelineEntriesCompanion data) {
    return TimelineEntry(
      id: data.id.present ? data.id.value : this.id,
      tripId: data.tripId.present ? data.tripId.value : this.tripId,
      stopId: data.stopId.present ? data.stopId.value : this.stopId,
      kind: data.kind.present ? data.kind.value : this.kind,
      title: data.title.present ? data.title.value : this.title,
      body: data.body.present ? data.body.value : this.body,
      lat: data.lat.present ? data.lat.value : this.lat,
      lon: data.lon.present ? data.lon.value : this.lon,
      accuracyM: data.accuracyM.present ? data.accuracyM.value : this.accuracyM,
      photoPaths: data.photoPaths.present
          ? data.photoPaths.value
          : this.photoPaths,
      occurredAt: data.occurredAt.present
          ? data.occurredAt.value
          : this.occurredAt,
    );
  }

  @override
  String toString() {
    return (StringBuffer('TimelineEntry(')
          ..write('id: $id, ')
          ..write('tripId: $tripId, ')
          ..write('stopId: $stopId, ')
          ..write('kind: $kind, ')
          ..write('title: $title, ')
          ..write('body: $body, ')
          ..write('lat: $lat, ')
          ..write('lon: $lon, ')
          ..write('accuracyM: $accuracyM, ')
          ..write('photoPaths: $photoPaths, ')
          ..write('occurredAt: $occurredAt')
          ..write(')'))
        .toString();
  }

  @override
  int get hashCode => Object.hash(
    id,
    tripId,
    stopId,
    kind,
    title,
    body,
    lat,
    lon,
    accuracyM,
    photoPaths,
    occurredAt,
  );
  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      (other is TimelineEntry &&
          other.id == this.id &&
          other.tripId == this.tripId &&
          other.stopId == this.stopId &&
          other.kind == this.kind &&
          other.title == this.title &&
          other.body == this.body &&
          other.lat == this.lat &&
          other.lon == this.lon &&
          other.accuracyM == this.accuracyM &&
          other.photoPaths == this.photoPaths &&
          other.occurredAt == this.occurredAt);
}

class TimelineEntriesCompanion extends UpdateCompanion<TimelineEntry> {
  final Value<int> id;
  final Value<int> tripId;
  final Value<int?> stopId;
  final Value<String> kind;
  final Value<String?> title;
  final Value<String?> body;
  final Value<double?> lat;
  final Value<double?> lon;
  final Value<double?> accuracyM;
  final Value<String?> photoPaths;
  final Value<DateTime> occurredAt;
  const TimelineEntriesCompanion({
    this.id = const Value.absent(),
    this.tripId = const Value.absent(),
    this.stopId = const Value.absent(),
    this.kind = const Value.absent(),
    this.title = const Value.absent(),
    this.body = const Value.absent(),
    this.lat = const Value.absent(),
    this.lon = const Value.absent(),
    this.accuracyM = const Value.absent(),
    this.photoPaths = const Value.absent(),
    this.occurredAt = const Value.absent(),
  });
  TimelineEntriesCompanion.insert({
    this.id = const Value.absent(),
    required int tripId,
    this.stopId = const Value.absent(),
    required String kind,
    this.title = const Value.absent(),
    this.body = const Value.absent(),
    this.lat = const Value.absent(),
    this.lon = const Value.absent(),
    this.accuracyM = const Value.absent(),
    this.photoPaths = const Value.absent(),
    required DateTime occurredAt,
  }) : tripId = Value(tripId),
       kind = Value(kind),
       occurredAt = Value(occurredAt);
  static Insertable<TimelineEntry> custom({
    Expression<int>? id,
    Expression<int>? tripId,
    Expression<int>? stopId,
    Expression<String>? kind,
    Expression<String>? title,
    Expression<String>? body,
    Expression<double>? lat,
    Expression<double>? lon,
    Expression<double>? accuracyM,
    Expression<String>? photoPaths,
    Expression<DateTime>? occurredAt,
  }) {
    return RawValuesInsertable({
      if (id != null) 'id': id,
      if (tripId != null) 'trip_id': tripId,
      if (stopId != null) 'stop_id': stopId,
      if (kind != null) 'kind': kind,
      if (title != null) 'title': title,
      if (body != null) 'body': body,
      if (lat != null) 'lat': lat,
      if (lon != null) 'lon': lon,
      if (accuracyM != null) 'accuracy_m': accuracyM,
      if (photoPaths != null) 'photo_paths': photoPaths,
      if (occurredAt != null) 'occurred_at': occurredAt,
    });
  }

  TimelineEntriesCompanion copyWith({
    Value<int>? id,
    Value<int>? tripId,
    Value<int?>? stopId,
    Value<String>? kind,
    Value<String?>? title,
    Value<String?>? body,
    Value<double?>? lat,
    Value<double?>? lon,
    Value<double?>? accuracyM,
    Value<String?>? photoPaths,
    Value<DateTime>? occurredAt,
  }) {
    return TimelineEntriesCompanion(
      id: id ?? this.id,
      tripId: tripId ?? this.tripId,
      stopId: stopId ?? this.stopId,
      kind: kind ?? this.kind,
      title: title ?? this.title,
      body: body ?? this.body,
      lat: lat ?? this.lat,
      lon: lon ?? this.lon,
      accuracyM: accuracyM ?? this.accuracyM,
      photoPaths: photoPaths ?? this.photoPaths,
      occurredAt: occurredAt ?? this.occurredAt,
    );
  }

  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    if (id.present) {
      map['id'] = Variable<int>(id.value);
    }
    if (tripId.present) {
      map['trip_id'] = Variable<int>(tripId.value);
    }
    if (stopId.present) {
      map['stop_id'] = Variable<int>(stopId.value);
    }
    if (kind.present) {
      map['kind'] = Variable<String>(kind.value);
    }
    if (title.present) {
      map['title'] = Variable<String>(title.value);
    }
    if (body.present) {
      map['body'] = Variable<String>(body.value);
    }
    if (lat.present) {
      map['lat'] = Variable<double>(lat.value);
    }
    if (lon.present) {
      map['lon'] = Variable<double>(lon.value);
    }
    if (accuracyM.present) {
      map['accuracy_m'] = Variable<double>(accuracyM.value);
    }
    if (photoPaths.present) {
      map['photo_paths'] = Variable<String>(photoPaths.value);
    }
    if (occurredAt.present) {
      map['occurred_at'] = Variable<DateTime>(occurredAt.value);
    }
    return map;
  }

  @override
  String toString() {
    return (StringBuffer('TimelineEntriesCompanion(')
          ..write('id: $id, ')
          ..write('tripId: $tripId, ')
          ..write('stopId: $stopId, ')
          ..write('kind: $kind, ')
          ..write('title: $title, ')
          ..write('body: $body, ')
          ..write('lat: $lat, ')
          ..write('lon: $lon, ')
          ..write('accuracyM: $accuracyM, ')
          ..write('photoPaths: $photoPaths, ')
          ..write('occurredAt: $occurredAt')
          ..write(')'))
        .toString();
  }
}

class $TrustedContactsTable extends TrustedContacts
    with TableInfo<$TrustedContactsTable, TrustedContact> {
  @override
  final GeneratedDatabase attachedDatabase;
  final String? _alias;
  $TrustedContactsTable(this.attachedDatabase, [this._alias]);
  static const VerificationMeta _idMeta = const VerificationMeta('id');
  @override
  late final GeneratedColumn<int> id = GeneratedColumn<int>(
    'id',
    aliasedName,
    false,
    hasAutoIncrement: true,
    type: DriftSqlType.int,
    requiredDuringInsert: false,
    defaultConstraints: GeneratedColumn.constraintIsAlways(
      'PRIMARY KEY AUTOINCREMENT',
    ),
  );
  static const VerificationMeta _tripIdMeta = const VerificationMeta('tripId');
  @override
  late final GeneratedColumn<int> tripId = GeneratedColumn<int>(
    'trip_id',
    aliasedName,
    true,
    type: DriftSqlType.int,
    requiredDuringInsert: false,
    defaultConstraints: GeneratedColumn.constraintIsAlways(
      'REFERENCES trips (id) ON DELETE CASCADE',
    ),
  );
  static const VerificationMeta _nameMeta = const VerificationMeta('name');
  @override
  late final GeneratedColumn<String> name = GeneratedColumn<String>(
    'name',
    aliasedName,
    false,
    type: DriftSqlType.string,
    requiredDuringInsert: true,
  );
  static const VerificationMeta _phoneE164Meta = const VerificationMeta(
    'phoneE164',
  );
  @override
  late final GeneratedColumn<String> phoneE164 = GeneratedColumn<String>(
    'phone_e164',
    aliasedName,
    false,
    type: DriftSqlType.string,
    requiredDuringInsert: true,
  );
  static const VerificationMeta _notifyOnArrivalMeta = const VerificationMeta(
    'notifyOnArrival',
  );
  @override
  late final GeneratedColumn<bool> notifyOnArrival = GeneratedColumn<bool>(
    'notify_on_arrival',
    aliasedName,
    false,
    type: DriftSqlType.bool,
    requiredDuringInsert: false,
    defaultConstraints: GeneratedColumn.constraintIsAlways(
      'CHECK ("notify_on_arrival" IN (0, 1))',
    ),
    defaultValue: const Constant(true),
  );
  static const VerificationMeta _escalateMeta = const VerificationMeta(
    'escalate',
  );
  @override
  late final GeneratedColumn<bool> escalate = GeneratedColumn<bool>(
    'escalate',
    aliasedName,
    false,
    type: DriftSqlType.bool,
    requiredDuringInsert: false,
    defaultConstraints: GeneratedColumn.constraintIsAlways(
      'CHECK ("escalate" IN (0, 1))',
    ),
    defaultValue: const Constant(true),
  );
  static const VerificationMeta _escalateAfterMinutesMeta =
      const VerificationMeta('escalateAfterMinutes');
  @override
  late final GeneratedColumn<int> escalateAfterMinutes = GeneratedColumn<int>(
    'escalate_after_minutes',
    aliasedName,
    false,
    type: DriftSqlType.int,
    requiredDuringInsert: false,
    defaultValue: const Constant(120),
  );
  @override
  List<GeneratedColumn> get $columns => [
    id,
    tripId,
    name,
    phoneE164,
    notifyOnArrival,
    escalate,
    escalateAfterMinutes,
  ];
  @override
  String get aliasedName => _alias ?? actualTableName;
  @override
  String get actualTableName => $name;
  static const String $name = 'trusted_contacts';
  @override
  VerificationContext validateIntegrity(
    Insertable<TrustedContact> instance, {
    bool isInserting = false,
  }) {
    final context = VerificationContext();
    final data = instance.toColumns(true);
    if (data.containsKey('id')) {
      context.handle(_idMeta, id.isAcceptableOrUnknown(data['id']!, _idMeta));
    }
    if (data.containsKey('trip_id')) {
      context.handle(
        _tripIdMeta,
        tripId.isAcceptableOrUnknown(data['trip_id']!, _tripIdMeta),
      );
    }
    if (data.containsKey('name')) {
      context.handle(
        _nameMeta,
        name.isAcceptableOrUnknown(data['name']!, _nameMeta),
      );
    } else if (isInserting) {
      context.missing(_nameMeta);
    }
    if (data.containsKey('phone_e164')) {
      context.handle(
        _phoneE164Meta,
        phoneE164.isAcceptableOrUnknown(data['phone_e164']!, _phoneE164Meta),
      );
    } else if (isInserting) {
      context.missing(_phoneE164Meta);
    }
    if (data.containsKey('notify_on_arrival')) {
      context.handle(
        _notifyOnArrivalMeta,
        notifyOnArrival.isAcceptableOrUnknown(
          data['notify_on_arrival']!,
          _notifyOnArrivalMeta,
        ),
      );
    }
    if (data.containsKey('escalate')) {
      context.handle(
        _escalateMeta,
        escalate.isAcceptableOrUnknown(data['escalate']!, _escalateMeta),
      );
    }
    if (data.containsKey('escalate_after_minutes')) {
      context.handle(
        _escalateAfterMinutesMeta,
        escalateAfterMinutes.isAcceptableOrUnknown(
          data['escalate_after_minutes']!,
          _escalateAfterMinutesMeta,
        ),
      );
    }
    return context;
  }

  @override
  Set<GeneratedColumn> get $primaryKey => {id};
  @override
  TrustedContact map(Map<String, dynamic> data, {String? tablePrefix}) {
    final effectivePrefix = tablePrefix != null ? '$tablePrefix.' : '';
    return TrustedContact(
      id: attachedDatabase.typeMapping.read(
        DriftSqlType.int,
        data['${effectivePrefix}id'],
      )!,
      tripId: attachedDatabase.typeMapping.read(
        DriftSqlType.int,
        data['${effectivePrefix}trip_id'],
      ),
      name: attachedDatabase.typeMapping.read(
        DriftSqlType.string,
        data['${effectivePrefix}name'],
      )!,
      phoneE164: attachedDatabase.typeMapping.read(
        DriftSqlType.string,
        data['${effectivePrefix}phone_e164'],
      )!,
      notifyOnArrival: attachedDatabase.typeMapping.read(
        DriftSqlType.bool,
        data['${effectivePrefix}notify_on_arrival'],
      )!,
      escalate: attachedDatabase.typeMapping.read(
        DriftSqlType.bool,
        data['${effectivePrefix}escalate'],
      )!,
      escalateAfterMinutes: attachedDatabase.typeMapping.read(
        DriftSqlType.int,
        data['${effectivePrefix}escalate_after_minutes'],
      )!,
    );
  }

  @override
  $TrustedContactsTable createAlias(String alias) {
    return $TrustedContactsTable(attachedDatabase, alias);
  }
}

class TrustedContact extends DataClass implements Insertable<TrustedContact> {
  final int id;
  final int? tripId;
  final String name;
  final String phoneE164;
  final bool notifyOnArrival;
  final bool escalate;
  final int escalateAfterMinutes;
  const TrustedContact({
    required this.id,
    this.tripId,
    required this.name,
    required this.phoneE164,
    required this.notifyOnArrival,
    required this.escalate,
    required this.escalateAfterMinutes,
  });
  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    map['id'] = Variable<int>(id);
    if (!nullToAbsent || tripId != null) {
      map['trip_id'] = Variable<int>(tripId);
    }
    map['name'] = Variable<String>(name);
    map['phone_e164'] = Variable<String>(phoneE164);
    map['notify_on_arrival'] = Variable<bool>(notifyOnArrival);
    map['escalate'] = Variable<bool>(escalate);
    map['escalate_after_minutes'] = Variable<int>(escalateAfterMinutes);
    return map;
  }

  TrustedContactsCompanion toCompanion(bool nullToAbsent) {
    return TrustedContactsCompanion(
      id: Value(id),
      tripId: tripId == null && nullToAbsent
          ? const Value.absent()
          : Value(tripId),
      name: Value(name),
      phoneE164: Value(phoneE164),
      notifyOnArrival: Value(notifyOnArrival),
      escalate: Value(escalate),
      escalateAfterMinutes: Value(escalateAfterMinutes),
    );
  }

  factory TrustedContact.fromJson(
    Map<String, dynamic> json, {
    ValueSerializer? serializer,
  }) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return TrustedContact(
      id: serializer.fromJson<int>(json['id']),
      tripId: serializer.fromJson<int?>(json['tripId']),
      name: serializer.fromJson<String>(json['name']),
      phoneE164: serializer.fromJson<String>(json['phoneE164']),
      notifyOnArrival: serializer.fromJson<bool>(json['notifyOnArrival']),
      escalate: serializer.fromJson<bool>(json['escalate']),
      escalateAfterMinutes: serializer.fromJson<int>(
        json['escalateAfterMinutes'],
      ),
    );
  }
  @override
  Map<String, dynamic> toJson({ValueSerializer? serializer}) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return <String, dynamic>{
      'id': serializer.toJson<int>(id),
      'tripId': serializer.toJson<int?>(tripId),
      'name': serializer.toJson<String>(name),
      'phoneE164': serializer.toJson<String>(phoneE164),
      'notifyOnArrival': serializer.toJson<bool>(notifyOnArrival),
      'escalate': serializer.toJson<bool>(escalate),
      'escalateAfterMinutes': serializer.toJson<int>(escalateAfterMinutes),
    };
  }

  TrustedContact copyWith({
    int? id,
    Value<int?> tripId = const Value.absent(),
    String? name,
    String? phoneE164,
    bool? notifyOnArrival,
    bool? escalate,
    int? escalateAfterMinutes,
  }) => TrustedContact(
    id: id ?? this.id,
    tripId: tripId.present ? tripId.value : this.tripId,
    name: name ?? this.name,
    phoneE164: phoneE164 ?? this.phoneE164,
    notifyOnArrival: notifyOnArrival ?? this.notifyOnArrival,
    escalate: escalate ?? this.escalate,
    escalateAfterMinutes: escalateAfterMinutes ?? this.escalateAfterMinutes,
  );
  TrustedContact copyWithCompanion(TrustedContactsCompanion data) {
    return TrustedContact(
      id: data.id.present ? data.id.value : this.id,
      tripId: data.tripId.present ? data.tripId.value : this.tripId,
      name: data.name.present ? data.name.value : this.name,
      phoneE164: data.phoneE164.present ? data.phoneE164.value : this.phoneE164,
      notifyOnArrival: data.notifyOnArrival.present
          ? data.notifyOnArrival.value
          : this.notifyOnArrival,
      escalate: data.escalate.present ? data.escalate.value : this.escalate,
      escalateAfterMinutes: data.escalateAfterMinutes.present
          ? data.escalateAfterMinutes.value
          : this.escalateAfterMinutes,
    );
  }

  @override
  String toString() {
    return (StringBuffer('TrustedContact(')
          ..write('id: $id, ')
          ..write('tripId: $tripId, ')
          ..write('name: $name, ')
          ..write('phoneE164: $phoneE164, ')
          ..write('notifyOnArrival: $notifyOnArrival, ')
          ..write('escalate: $escalate, ')
          ..write('escalateAfterMinutes: $escalateAfterMinutes')
          ..write(')'))
        .toString();
  }

  @override
  int get hashCode => Object.hash(
    id,
    tripId,
    name,
    phoneE164,
    notifyOnArrival,
    escalate,
    escalateAfterMinutes,
  );
  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      (other is TrustedContact &&
          other.id == this.id &&
          other.tripId == this.tripId &&
          other.name == this.name &&
          other.phoneE164 == this.phoneE164 &&
          other.notifyOnArrival == this.notifyOnArrival &&
          other.escalate == this.escalate &&
          other.escalateAfterMinutes == this.escalateAfterMinutes);
}

class TrustedContactsCompanion extends UpdateCompanion<TrustedContact> {
  final Value<int> id;
  final Value<int?> tripId;
  final Value<String> name;
  final Value<String> phoneE164;
  final Value<bool> notifyOnArrival;
  final Value<bool> escalate;
  final Value<int> escalateAfterMinutes;
  const TrustedContactsCompanion({
    this.id = const Value.absent(),
    this.tripId = const Value.absent(),
    this.name = const Value.absent(),
    this.phoneE164 = const Value.absent(),
    this.notifyOnArrival = const Value.absent(),
    this.escalate = const Value.absent(),
    this.escalateAfterMinutes = const Value.absent(),
  });
  TrustedContactsCompanion.insert({
    this.id = const Value.absent(),
    this.tripId = const Value.absent(),
    required String name,
    required String phoneE164,
    this.notifyOnArrival = const Value.absent(),
    this.escalate = const Value.absent(),
    this.escalateAfterMinutes = const Value.absent(),
  }) : name = Value(name),
       phoneE164 = Value(phoneE164);
  static Insertable<TrustedContact> custom({
    Expression<int>? id,
    Expression<int>? tripId,
    Expression<String>? name,
    Expression<String>? phoneE164,
    Expression<bool>? notifyOnArrival,
    Expression<bool>? escalate,
    Expression<int>? escalateAfterMinutes,
  }) {
    return RawValuesInsertable({
      if (id != null) 'id': id,
      if (tripId != null) 'trip_id': tripId,
      if (name != null) 'name': name,
      if (phoneE164 != null) 'phone_e164': phoneE164,
      if (notifyOnArrival != null) 'notify_on_arrival': notifyOnArrival,
      if (escalate != null) 'escalate': escalate,
      if (escalateAfterMinutes != null)
        'escalate_after_minutes': escalateAfterMinutes,
    });
  }

  TrustedContactsCompanion copyWith({
    Value<int>? id,
    Value<int?>? tripId,
    Value<String>? name,
    Value<String>? phoneE164,
    Value<bool>? notifyOnArrival,
    Value<bool>? escalate,
    Value<int>? escalateAfterMinutes,
  }) {
    return TrustedContactsCompanion(
      id: id ?? this.id,
      tripId: tripId ?? this.tripId,
      name: name ?? this.name,
      phoneE164: phoneE164 ?? this.phoneE164,
      notifyOnArrival: notifyOnArrival ?? this.notifyOnArrival,
      escalate: escalate ?? this.escalate,
      escalateAfterMinutes: escalateAfterMinutes ?? this.escalateAfterMinutes,
    );
  }

  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    if (id.present) {
      map['id'] = Variable<int>(id.value);
    }
    if (tripId.present) {
      map['trip_id'] = Variable<int>(tripId.value);
    }
    if (name.present) {
      map['name'] = Variable<String>(name.value);
    }
    if (phoneE164.present) {
      map['phone_e164'] = Variable<String>(phoneE164.value);
    }
    if (notifyOnArrival.present) {
      map['notify_on_arrival'] = Variable<bool>(notifyOnArrival.value);
    }
    if (escalate.present) {
      map['escalate'] = Variable<bool>(escalate.value);
    }
    if (escalateAfterMinutes.present) {
      map['escalate_after_minutes'] = Variable<int>(escalateAfterMinutes.value);
    }
    return map;
  }

  @override
  String toString() {
    return (StringBuffer('TrustedContactsCompanion(')
          ..write('id: $id, ')
          ..write('tripId: $tripId, ')
          ..write('name: $name, ')
          ..write('phoneE164: $phoneE164, ')
          ..write('notifyOnArrival: $notifyOnArrival, ')
          ..write('escalate: $escalate, ')
          ..write('escalateAfterMinutes: $escalateAfterMinutes')
          ..write(')'))
        .toString();
  }
}

abstract class _$AppDatabase extends GeneratedDatabase {
  _$AppDatabase(QueryExecutor e) : super(e);
  $AppDatabaseManager get managers => $AppDatabaseManager(this);
  late final $TripsTable trips = $TripsTable(this);
  late final $StopsTable stops = $StopsTable(this);
  late final $LegsTable legs = $LegsTable(this);
  late final $PoisTable pois = $PoisTable(this);
  late final $PoiContactsTable poiContacts = $PoiContactsTable(this);
  late final $ImportBatchesTable importBatches = $ImportBatchesTable(this);
  late final $ContactsTable contacts = $ContactsTable(this);
  late final $CallLogsTable callLogs = $CallLogsTable(this);
  late final $EmergencyHelplinesTable emergencyHelplines =
      $EmergencyHelplinesTable(this);
  late final $ChecklistItemsTable checklistItems = $ChecklistItemsTable(this);
  late final $WeatherSnapshotsTable weatherSnapshots = $WeatherSnapshotsTable(
    this,
  );
  late final $TravellersTable travellers = $TravellersTable(this);
  late final $ExpensesTable expenses = $ExpensesTable(this);
  late final $ExpenseSplitsTable expenseSplits = $ExpenseSplitsTable(this);
  late final $TimelineEntriesTable timelineEntries = $TimelineEntriesTable(
    this,
  );
  late final $TrustedContactsTable trustedContacts = $TrustedContactsTable(
    this,
  );
  late final ContactsDao contactsDao = ContactsDao(this as AppDatabase);
  @override
  Iterable<TableInfo<Table, Object?>> get allTables =>
      allSchemaEntities.whereType<TableInfo<Table, Object?>>();
  @override
  List<DatabaseSchemaEntity> get allSchemaEntities => [
    trips,
    stops,
    legs,
    pois,
    poiContacts,
    importBatches,
    contacts,
    callLogs,
    emergencyHelplines,
    checklistItems,
    weatherSnapshots,
    travellers,
    expenses,
    expenseSplits,
    timelineEntries,
    trustedContacts,
  ];
  @override
  StreamQueryUpdateRules get streamUpdateRules => const StreamQueryUpdateRules([
    WritePropagation(
      on: TableUpdateQuery.onTableName(
        'trips',
        limitUpdateKind: UpdateKind.delete,
      ),
      result: [TableUpdate('stops', kind: UpdateKind.delete)],
    ),
    WritePropagation(
      on: TableUpdateQuery.onTableName(
        'trips',
        limitUpdateKind: UpdateKind.delete,
      ),
      result: [TableUpdate('legs', kind: UpdateKind.delete)],
    ),
    WritePropagation(
      on: TableUpdateQuery.onTableName(
        'stops',
        limitUpdateKind: UpdateKind.delete,
      ),
      result: [TableUpdate('legs', kind: UpdateKind.delete)],
    ),
    WritePropagation(
      on: TableUpdateQuery.onTableName(
        'stops',
        limitUpdateKind: UpdateKind.delete,
      ),
      result: [TableUpdate('legs', kind: UpdateKind.delete)],
    ),
    WritePropagation(
      on: TableUpdateQuery.onTableName(
        'trips',
        limitUpdateKind: UpdateKind.delete,
      ),
      result: [TableUpdate('pois', kind: UpdateKind.delete)],
    ),
    WritePropagation(
      on: TableUpdateQuery.onTableName(
        'stops',
        limitUpdateKind: UpdateKind.delete,
      ),
      result: [TableUpdate('pois', kind: UpdateKind.delete)],
    ),
    WritePropagation(
      on: TableUpdateQuery.onTableName(
        'legs',
        limitUpdateKind: UpdateKind.delete,
      ),
      result: [TableUpdate('pois', kind: UpdateKind.delete)],
    ),
    WritePropagation(
      on: TableUpdateQuery.onTableName(
        'pois',
        limitUpdateKind: UpdateKind.delete,
      ),
      result: [TableUpdate('poi_contacts', kind: UpdateKind.delete)],
    ),
    WritePropagation(
      on: TableUpdateQuery.onTableName(
        'trips',
        limitUpdateKind: UpdateKind.delete,
      ),
      result: [TableUpdate('import_batches', kind: UpdateKind.delete)],
    ),
    WritePropagation(
      on: TableUpdateQuery.onTableName(
        'trips',
        limitUpdateKind: UpdateKind.delete,
      ),
      result: [TableUpdate('contacts', kind: UpdateKind.delete)],
    ),
    WritePropagation(
      on: TableUpdateQuery.onTableName(
        'stops',
        limitUpdateKind: UpdateKind.delete,
      ),
      result: [TableUpdate('contacts', kind: UpdateKind.update)],
    ),
    WritePropagation(
      on: TableUpdateQuery.onTableName(
        'import_batches',
        limitUpdateKind: UpdateKind.delete,
      ),
      result: [TableUpdate('contacts', kind: UpdateKind.update)],
    ),
    WritePropagation(
      on: TableUpdateQuery.onTableName(
        'contacts',
        limitUpdateKind: UpdateKind.delete,
      ),
      result: [TableUpdate('call_logs', kind: UpdateKind.delete)],
    ),
    WritePropagation(
      on: TableUpdateQuery.onTableName(
        'trips',
        limitUpdateKind: UpdateKind.delete,
      ),
      result: [TableUpdate('call_logs', kind: UpdateKind.delete)],
    ),
    WritePropagation(
      on: TableUpdateQuery.onTableName(
        'trips',
        limitUpdateKind: UpdateKind.delete,
      ),
      result: [TableUpdate('checklist_items', kind: UpdateKind.delete)],
    ),
    WritePropagation(
      on: TableUpdateQuery.onTableName(
        'stops',
        limitUpdateKind: UpdateKind.delete,
      ),
      result: [TableUpdate('checklist_items', kind: UpdateKind.delete)],
    ),
    WritePropagation(
      on: TableUpdateQuery.onTableName(
        'contacts',
        limitUpdateKind: UpdateKind.delete,
      ),
      result: [TableUpdate('checklist_items', kind: UpdateKind.delete)],
    ),
    WritePropagation(
      on: TableUpdateQuery.onTableName(
        'stops',
        limitUpdateKind: UpdateKind.delete,
      ),
      result: [TableUpdate('weather_snapshots', kind: UpdateKind.delete)],
    ),
    WritePropagation(
      on: TableUpdateQuery.onTableName(
        'trips',
        limitUpdateKind: UpdateKind.delete,
      ),
      result: [TableUpdate('travellers', kind: UpdateKind.delete)],
    ),
    WritePropagation(
      on: TableUpdateQuery.onTableName(
        'trips',
        limitUpdateKind: UpdateKind.delete,
      ),
      result: [TableUpdate('expenses', kind: UpdateKind.delete)],
    ),
    WritePropagation(
      on: TableUpdateQuery.onTableName(
        'stops',
        limitUpdateKind: UpdateKind.delete,
      ),
      result: [TableUpdate('expenses', kind: UpdateKind.update)],
    ),
    WritePropagation(
      on: TableUpdateQuery.onTableName(
        'travellers',
        limitUpdateKind: UpdateKind.delete,
      ),
      result: [TableUpdate('expenses', kind: UpdateKind.delete)],
    ),
    WritePropagation(
      on: TableUpdateQuery.onTableName(
        'expenses',
        limitUpdateKind: UpdateKind.delete,
      ),
      result: [TableUpdate('expense_splits', kind: UpdateKind.delete)],
    ),
    WritePropagation(
      on: TableUpdateQuery.onTableName(
        'travellers',
        limitUpdateKind: UpdateKind.delete,
      ),
      result: [TableUpdate('expense_splits', kind: UpdateKind.delete)],
    ),
    WritePropagation(
      on: TableUpdateQuery.onTableName(
        'trips',
        limitUpdateKind: UpdateKind.delete,
      ),
      result: [TableUpdate('timeline_entries', kind: UpdateKind.delete)],
    ),
    WritePropagation(
      on: TableUpdateQuery.onTableName(
        'stops',
        limitUpdateKind: UpdateKind.delete,
      ),
      result: [TableUpdate('timeline_entries', kind: UpdateKind.update)],
    ),
    WritePropagation(
      on: TableUpdateQuery.onTableName(
        'trips',
        limitUpdateKind: UpdateKind.delete,
      ),
      result: [TableUpdate('trusted_contacts', kind: UpdateKind.delete)],
    ),
  ]);
}

typedef $$TripsTableCreateCompanionBuilder =
    TripsCompanion Function({
      Value<int> id,
      required String name,
      Value<DateTime?> startDate,
      Value<DateTime?> endDate,
      Value<String> baseCurrency,
      Value<bool> isActive,
      Value<DateTime> createdAt,
    });
typedef $$TripsTableUpdateCompanionBuilder =
    TripsCompanion Function({
      Value<int> id,
      Value<String> name,
      Value<DateTime?> startDate,
      Value<DateTime?> endDate,
      Value<String> baseCurrency,
      Value<bool> isActive,
      Value<DateTime> createdAt,
    });

final class $$TripsTableReferences
    extends BaseReferences<_$AppDatabase, $TripsTable, Trip> {
  $$TripsTableReferences(super.$_db, super.$_table, super.$_typedResult);

  static MultiTypedResultKey<$StopsTable, List<Stop>> _stopsRefsTable(
    _$AppDatabase db,
  ) => MultiTypedResultKey.fromTable(
    db.stops,
    aliasName: 'trips__id__stops__trip_id',
  );

  $$StopsTableProcessedTableManager get stopsRefs {
    final manager = $$StopsTableTableManager(
      $_db,
      $_db.stops,
    ).filter((f) => f.tripId.id.sqlEquals($_itemColumn<int>('id')!));

    final cache = $_typedResult.readTableOrNull(_stopsRefsTable($_db));
    return ProcessedTableManager(
      manager.$state.copyWith(prefetchedData: cache),
    );
  }

  static MultiTypedResultKey<$LegsTable, List<Leg>> _legsRefsTable(
    _$AppDatabase db,
  ) => MultiTypedResultKey.fromTable(
    db.legs,
    aliasName: 'trips__id__legs__trip_id',
  );

  $$LegsTableProcessedTableManager get legsRefs {
    final manager = $$LegsTableTableManager(
      $_db,
      $_db.legs,
    ).filter((f) => f.tripId.id.sqlEquals($_itemColumn<int>('id')!));

    final cache = $_typedResult.readTableOrNull(_legsRefsTable($_db));
    return ProcessedTableManager(
      manager.$state.copyWith(prefetchedData: cache),
    );
  }

  static MultiTypedResultKey<$PoisTable, List<Poi>> _poisRefsTable(
    _$AppDatabase db,
  ) => MultiTypedResultKey.fromTable(
    db.pois,
    aliasName: 'trips__id__pois__trip_id',
  );

  $$PoisTableProcessedTableManager get poisRefs {
    final manager = $$PoisTableTableManager(
      $_db,
      $_db.pois,
    ).filter((f) => f.tripId.id.sqlEquals($_itemColumn<int>('id')!));

    final cache = $_typedResult.readTableOrNull(_poisRefsTable($_db));
    return ProcessedTableManager(
      manager.$state.copyWith(prefetchedData: cache),
    );
  }

  static MultiTypedResultKey<$ImportBatchesTable, List<ImportBatche>>
  _importBatchesRefsTable(_$AppDatabase db) => MultiTypedResultKey.fromTable(
    db.importBatches,
    aliasName: 'trips__id__import_batches__trip_id',
  );

  $$ImportBatchesTableProcessedTableManager get importBatchesRefs {
    final manager = $$ImportBatchesTableTableManager(
      $_db,
      $_db.importBatches,
    ).filter((f) => f.tripId.id.sqlEquals($_itemColumn<int>('id')!));

    final cache = $_typedResult.readTableOrNull(_importBatchesRefsTable($_db));
    return ProcessedTableManager(
      manager.$state.copyWith(prefetchedData: cache),
    );
  }

  static MultiTypedResultKey<$ContactsTable, List<Contact>> _contactsRefsTable(
    _$AppDatabase db,
  ) => MultiTypedResultKey.fromTable(
    db.contacts,
    aliasName: 'trips__id__contacts__trip_id',
  );

  $$ContactsTableProcessedTableManager get contactsRefs {
    final manager = $$ContactsTableTableManager(
      $_db,
      $_db.contacts,
    ).filter((f) => f.tripId.id.sqlEquals($_itemColumn<int>('id')!));

    final cache = $_typedResult.readTableOrNull(_contactsRefsTable($_db));
    return ProcessedTableManager(
      manager.$state.copyWith(prefetchedData: cache),
    );
  }

  static MultiTypedResultKey<$CallLogsTable, List<CallLog>> _callLogsRefsTable(
    _$AppDatabase db,
  ) => MultiTypedResultKey.fromTable(
    db.callLogs,
    aliasName: 'trips__id__call_logs__trip_id',
  );

  $$CallLogsTableProcessedTableManager get callLogsRefs {
    final manager = $$CallLogsTableTableManager(
      $_db,
      $_db.callLogs,
    ).filter((f) => f.tripId.id.sqlEquals($_itemColumn<int>('id')!));

    final cache = $_typedResult.readTableOrNull(_callLogsRefsTable($_db));
    return ProcessedTableManager(
      manager.$state.copyWith(prefetchedData: cache),
    );
  }

  static MultiTypedResultKey<$ChecklistItemsTable, List<ChecklistItem>>
  _checklistItemsRefsTable(_$AppDatabase db) => MultiTypedResultKey.fromTable(
    db.checklistItems,
    aliasName: 'trips__id__checklist_items__trip_id',
  );

  $$ChecklistItemsTableProcessedTableManager get checklistItemsRefs {
    final manager = $$ChecklistItemsTableTableManager(
      $_db,
      $_db.checklistItems,
    ).filter((f) => f.tripId.id.sqlEquals($_itemColumn<int>('id')!));

    final cache = $_typedResult.readTableOrNull(_checklistItemsRefsTable($_db));
    return ProcessedTableManager(
      manager.$state.copyWith(prefetchedData: cache),
    );
  }

  static MultiTypedResultKey<$TravellersTable, List<Traveller>>
  _travellersRefsTable(_$AppDatabase db) => MultiTypedResultKey.fromTable(
    db.travellers,
    aliasName: 'trips__id__travellers__trip_id',
  );

  $$TravellersTableProcessedTableManager get travellersRefs {
    final manager = $$TravellersTableTableManager(
      $_db,
      $_db.travellers,
    ).filter((f) => f.tripId.id.sqlEquals($_itemColumn<int>('id')!));

    final cache = $_typedResult.readTableOrNull(_travellersRefsTable($_db));
    return ProcessedTableManager(
      manager.$state.copyWith(prefetchedData: cache),
    );
  }

  static MultiTypedResultKey<$ExpensesTable, List<Expense>> _expensesRefsTable(
    _$AppDatabase db,
  ) => MultiTypedResultKey.fromTable(
    db.expenses,
    aliasName: 'trips__id__expenses__trip_id',
  );

  $$ExpensesTableProcessedTableManager get expensesRefs {
    final manager = $$ExpensesTableTableManager(
      $_db,
      $_db.expenses,
    ).filter((f) => f.tripId.id.sqlEquals($_itemColumn<int>('id')!));

    final cache = $_typedResult.readTableOrNull(_expensesRefsTable($_db));
    return ProcessedTableManager(
      manager.$state.copyWith(prefetchedData: cache),
    );
  }

  static MultiTypedResultKey<$TimelineEntriesTable, List<TimelineEntry>>
  _timelineEntriesRefsTable(_$AppDatabase db) => MultiTypedResultKey.fromTable(
    db.timelineEntries,
    aliasName: 'trips__id__timeline_entries__trip_id',
  );

  $$TimelineEntriesTableProcessedTableManager get timelineEntriesRefs {
    final manager = $$TimelineEntriesTableTableManager(
      $_db,
      $_db.timelineEntries,
    ).filter((f) => f.tripId.id.sqlEquals($_itemColumn<int>('id')!));

    final cache = $_typedResult.readTableOrNull(
      _timelineEntriesRefsTable($_db),
    );
    return ProcessedTableManager(
      manager.$state.copyWith(prefetchedData: cache),
    );
  }

  static MultiTypedResultKey<$TrustedContactsTable, List<TrustedContact>>
  _trustedContactsRefsTable(_$AppDatabase db) => MultiTypedResultKey.fromTable(
    db.trustedContacts,
    aliasName: 'trips__id__trusted_contacts__trip_id',
  );

  $$TrustedContactsTableProcessedTableManager get trustedContactsRefs {
    final manager = $$TrustedContactsTableTableManager(
      $_db,
      $_db.trustedContacts,
    ).filter((f) => f.tripId.id.sqlEquals($_itemColumn<int>('id')!));

    final cache = $_typedResult.readTableOrNull(
      _trustedContactsRefsTable($_db),
    );
    return ProcessedTableManager(
      manager.$state.copyWith(prefetchedData: cache),
    );
  }
}

class $$TripsTableFilterComposer extends Composer<_$AppDatabase, $TripsTable> {
  $$TripsTableFilterComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  ColumnFilters<int> get id => $composableBuilder(
    column: $table.id,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<String> get name => $composableBuilder(
    column: $table.name,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<DateTime> get startDate => $composableBuilder(
    column: $table.startDate,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<DateTime> get endDate => $composableBuilder(
    column: $table.endDate,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<String> get baseCurrency => $composableBuilder(
    column: $table.baseCurrency,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<bool> get isActive => $composableBuilder(
    column: $table.isActive,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<DateTime> get createdAt => $composableBuilder(
    column: $table.createdAt,
    builder: (column) => ColumnFilters(column),
  );

  Expression<bool> stopsRefs(
    Expression<bool> Function($$StopsTableFilterComposer f) f,
  ) {
    final $$StopsTableFilterComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.id,
      referencedTable: $db.stops,
      getReferencedColumn: (t) => t.tripId,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$StopsTableFilterComposer(
            $db: $db,
            $table: $db.stops,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return f(composer);
  }

  Expression<bool> legsRefs(
    Expression<bool> Function($$LegsTableFilterComposer f) f,
  ) {
    final $$LegsTableFilterComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.id,
      referencedTable: $db.legs,
      getReferencedColumn: (t) => t.tripId,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$LegsTableFilterComposer(
            $db: $db,
            $table: $db.legs,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return f(composer);
  }

  Expression<bool> poisRefs(
    Expression<bool> Function($$PoisTableFilterComposer f) f,
  ) {
    final $$PoisTableFilterComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.id,
      referencedTable: $db.pois,
      getReferencedColumn: (t) => t.tripId,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$PoisTableFilterComposer(
            $db: $db,
            $table: $db.pois,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return f(composer);
  }

  Expression<bool> importBatchesRefs(
    Expression<bool> Function($$ImportBatchesTableFilterComposer f) f,
  ) {
    final $$ImportBatchesTableFilterComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.id,
      referencedTable: $db.importBatches,
      getReferencedColumn: (t) => t.tripId,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$ImportBatchesTableFilterComposer(
            $db: $db,
            $table: $db.importBatches,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return f(composer);
  }

  Expression<bool> contactsRefs(
    Expression<bool> Function($$ContactsTableFilterComposer f) f,
  ) {
    final $$ContactsTableFilterComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.id,
      referencedTable: $db.contacts,
      getReferencedColumn: (t) => t.tripId,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$ContactsTableFilterComposer(
            $db: $db,
            $table: $db.contacts,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return f(composer);
  }

  Expression<bool> callLogsRefs(
    Expression<bool> Function($$CallLogsTableFilterComposer f) f,
  ) {
    final $$CallLogsTableFilterComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.id,
      referencedTable: $db.callLogs,
      getReferencedColumn: (t) => t.tripId,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$CallLogsTableFilterComposer(
            $db: $db,
            $table: $db.callLogs,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return f(composer);
  }

  Expression<bool> checklistItemsRefs(
    Expression<bool> Function($$ChecklistItemsTableFilterComposer f) f,
  ) {
    final $$ChecklistItemsTableFilterComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.id,
      referencedTable: $db.checklistItems,
      getReferencedColumn: (t) => t.tripId,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$ChecklistItemsTableFilterComposer(
            $db: $db,
            $table: $db.checklistItems,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return f(composer);
  }

  Expression<bool> travellersRefs(
    Expression<bool> Function($$TravellersTableFilterComposer f) f,
  ) {
    final $$TravellersTableFilterComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.id,
      referencedTable: $db.travellers,
      getReferencedColumn: (t) => t.tripId,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$TravellersTableFilterComposer(
            $db: $db,
            $table: $db.travellers,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return f(composer);
  }

  Expression<bool> expensesRefs(
    Expression<bool> Function($$ExpensesTableFilterComposer f) f,
  ) {
    final $$ExpensesTableFilterComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.id,
      referencedTable: $db.expenses,
      getReferencedColumn: (t) => t.tripId,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$ExpensesTableFilterComposer(
            $db: $db,
            $table: $db.expenses,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return f(composer);
  }

  Expression<bool> timelineEntriesRefs(
    Expression<bool> Function($$TimelineEntriesTableFilterComposer f) f,
  ) {
    final $$TimelineEntriesTableFilterComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.id,
      referencedTable: $db.timelineEntries,
      getReferencedColumn: (t) => t.tripId,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$TimelineEntriesTableFilterComposer(
            $db: $db,
            $table: $db.timelineEntries,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return f(composer);
  }

  Expression<bool> trustedContactsRefs(
    Expression<bool> Function($$TrustedContactsTableFilterComposer f) f,
  ) {
    final $$TrustedContactsTableFilterComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.id,
      referencedTable: $db.trustedContacts,
      getReferencedColumn: (t) => t.tripId,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$TrustedContactsTableFilterComposer(
            $db: $db,
            $table: $db.trustedContacts,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return f(composer);
  }
}

class $$TripsTableOrderingComposer
    extends Composer<_$AppDatabase, $TripsTable> {
  $$TripsTableOrderingComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  ColumnOrderings<int> get id => $composableBuilder(
    column: $table.id,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<String> get name => $composableBuilder(
    column: $table.name,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<DateTime> get startDate => $composableBuilder(
    column: $table.startDate,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<DateTime> get endDate => $composableBuilder(
    column: $table.endDate,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<String> get baseCurrency => $composableBuilder(
    column: $table.baseCurrency,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<bool> get isActive => $composableBuilder(
    column: $table.isActive,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<DateTime> get createdAt => $composableBuilder(
    column: $table.createdAt,
    builder: (column) => ColumnOrderings(column),
  );
}

class $$TripsTableAnnotationComposer
    extends Composer<_$AppDatabase, $TripsTable> {
  $$TripsTableAnnotationComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  GeneratedColumn<int> get id =>
      $composableBuilder(column: $table.id, builder: (column) => column);

  GeneratedColumn<String> get name =>
      $composableBuilder(column: $table.name, builder: (column) => column);

  GeneratedColumn<DateTime> get startDate =>
      $composableBuilder(column: $table.startDate, builder: (column) => column);

  GeneratedColumn<DateTime> get endDate =>
      $composableBuilder(column: $table.endDate, builder: (column) => column);

  GeneratedColumn<String> get baseCurrency => $composableBuilder(
    column: $table.baseCurrency,
    builder: (column) => column,
  );

  GeneratedColumn<bool> get isActive =>
      $composableBuilder(column: $table.isActive, builder: (column) => column);

  GeneratedColumn<DateTime> get createdAt =>
      $composableBuilder(column: $table.createdAt, builder: (column) => column);

  Expression<T> stopsRefs<T extends Object>(
    Expression<T> Function($$StopsTableAnnotationComposer a) f,
  ) {
    final $$StopsTableAnnotationComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.id,
      referencedTable: $db.stops,
      getReferencedColumn: (t) => t.tripId,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$StopsTableAnnotationComposer(
            $db: $db,
            $table: $db.stops,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return f(composer);
  }

  Expression<T> legsRefs<T extends Object>(
    Expression<T> Function($$LegsTableAnnotationComposer a) f,
  ) {
    final $$LegsTableAnnotationComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.id,
      referencedTable: $db.legs,
      getReferencedColumn: (t) => t.tripId,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$LegsTableAnnotationComposer(
            $db: $db,
            $table: $db.legs,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return f(composer);
  }

  Expression<T> poisRefs<T extends Object>(
    Expression<T> Function($$PoisTableAnnotationComposer a) f,
  ) {
    final $$PoisTableAnnotationComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.id,
      referencedTable: $db.pois,
      getReferencedColumn: (t) => t.tripId,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$PoisTableAnnotationComposer(
            $db: $db,
            $table: $db.pois,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return f(composer);
  }

  Expression<T> importBatchesRefs<T extends Object>(
    Expression<T> Function($$ImportBatchesTableAnnotationComposer a) f,
  ) {
    final $$ImportBatchesTableAnnotationComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.id,
      referencedTable: $db.importBatches,
      getReferencedColumn: (t) => t.tripId,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$ImportBatchesTableAnnotationComposer(
            $db: $db,
            $table: $db.importBatches,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return f(composer);
  }

  Expression<T> contactsRefs<T extends Object>(
    Expression<T> Function($$ContactsTableAnnotationComposer a) f,
  ) {
    final $$ContactsTableAnnotationComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.id,
      referencedTable: $db.contacts,
      getReferencedColumn: (t) => t.tripId,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$ContactsTableAnnotationComposer(
            $db: $db,
            $table: $db.contacts,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return f(composer);
  }

  Expression<T> callLogsRefs<T extends Object>(
    Expression<T> Function($$CallLogsTableAnnotationComposer a) f,
  ) {
    final $$CallLogsTableAnnotationComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.id,
      referencedTable: $db.callLogs,
      getReferencedColumn: (t) => t.tripId,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$CallLogsTableAnnotationComposer(
            $db: $db,
            $table: $db.callLogs,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return f(composer);
  }

  Expression<T> checklistItemsRefs<T extends Object>(
    Expression<T> Function($$ChecklistItemsTableAnnotationComposer a) f,
  ) {
    final $$ChecklistItemsTableAnnotationComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.id,
      referencedTable: $db.checklistItems,
      getReferencedColumn: (t) => t.tripId,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$ChecklistItemsTableAnnotationComposer(
            $db: $db,
            $table: $db.checklistItems,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return f(composer);
  }

  Expression<T> travellersRefs<T extends Object>(
    Expression<T> Function($$TravellersTableAnnotationComposer a) f,
  ) {
    final $$TravellersTableAnnotationComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.id,
      referencedTable: $db.travellers,
      getReferencedColumn: (t) => t.tripId,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$TravellersTableAnnotationComposer(
            $db: $db,
            $table: $db.travellers,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return f(composer);
  }

  Expression<T> expensesRefs<T extends Object>(
    Expression<T> Function($$ExpensesTableAnnotationComposer a) f,
  ) {
    final $$ExpensesTableAnnotationComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.id,
      referencedTable: $db.expenses,
      getReferencedColumn: (t) => t.tripId,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$ExpensesTableAnnotationComposer(
            $db: $db,
            $table: $db.expenses,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return f(composer);
  }

  Expression<T> timelineEntriesRefs<T extends Object>(
    Expression<T> Function($$TimelineEntriesTableAnnotationComposer a) f,
  ) {
    final $$TimelineEntriesTableAnnotationComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.id,
      referencedTable: $db.timelineEntries,
      getReferencedColumn: (t) => t.tripId,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$TimelineEntriesTableAnnotationComposer(
            $db: $db,
            $table: $db.timelineEntries,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return f(composer);
  }

  Expression<T> trustedContactsRefs<T extends Object>(
    Expression<T> Function($$TrustedContactsTableAnnotationComposer a) f,
  ) {
    final $$TrustedContactsTableAnnotationComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.id,
      referencedTable: $db.trustedContacts,
      getReferencedColumn: (t) => t.tripId,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$TrustedContactsTableAnnotationComposer(
            $db: $db,
            $table: $db.trustedContacts,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return f(composer);
  }
}

class $$TripsTableTableManager
    extends
        RootTableManager<
          _$AppDatabase,
          $TripsTable,
          Trip,
          $$TripsTableFilterComposer,
          $$TripsTableOrderingComposer,
          $$TripsTableAnnotationComposer,
          $$TripsTableCreateCompanionBuilder,
          $$TripsTableUpdateCompanionBuilder,
          (Trip, $$TripsTableReferences),
          Trip,
          PrefetchHooks Function({
            bool stopsRefs,
            bool legsRefs,
            bool poisRefs,
            bool importBatchesRefs,
            bool contactsRefs,
            bool callLogsRefs,
            bool checklistItemsRefs,
            bool travellersRefs,
            bool expensesRefs,
            bool timelineEntriesRefs,
            bool trustedContactsRefs,
          })
        > {
  $$TripsTableTableManager(_$AppDatabase db, $TripsTable table)
    : super(
        TableManagerState(
          db: db,
          table: table,
          createFilteringComposer: () =>
              $$TripsTableFilterComposer($db: db, $table: table),
          createOrderingComposer: () =>
              $$TripsTableOrderingComposer($db: db, $table: table),
          createComputedFieldComposer: () =>
              $$TripsTableAnnotationComposer($db: db, $table: table),
          updateCompanionCallback:
              ({
                Value<int> id = const Value.absent(),
                Value<String> name = const Value.absent(),
                Value<DateTime?> startDate = const Value.absent(),
                Value<DateTime?> endDate = const Value.absent(),
                Value<String> baseCurrency = const Value.absent(),
                Value<bool> isActive = const Value.absent(),
                Value<DateTime> createdAt = const Value.absent(),
              }) => TripsCompanion(
                id: id,
                name: name,
                startDate: startDate,
                endDate: endDate,
                baseCurrency: baseCurrency,
                isActive: isActive,
                createdAt: createdAt,
              ),
          createCompanionCallback:
              ({
                Value<int> id = const Value.absent(),
                required String name,
                Value<DateTime?> startDate = const Value.absent(),
                Value<DateTime?> endDate = const Value.absent(),
                Value<String> baseCurrency = const Value.absent(),
                Value<bool> isActive = const Value.absent(),
                Value<DateTime> createdAt = const Value.absent(),
              }) => TripsCompanion.insert(
                id: id,
                name: name,
                startDate: startDate,
                endDate: endDate,
                baseCurrency: baseCurrency,
                isActive: isActive,
                createdAt: createdAt,
              ),
          withReferenceMapper: (p0) => p0
              .map(
                (e) => (
                  e.readTable<$TripsTable, Trip>(table),
                  $$TripsTableReferences(db, table, e),
                ),
              )
              .toList(),
          prefetchHooksCallback:
              ({
                stopsRefs = false,
                legsRefs = false,
                poisRefs = false,
                importBatchesRefs = false,
                contactsRefs = false,
                callLogsRefs = false,
                checklistItemsRefs = false,
                travellersRefs = false,
                expensesRefs = false,
                timelineEntriesRefs = false,
                trustedContactsRefs = false,
              }) {
                return PrefetchHooks(
                  db: db,
                  explicitlyWatchedTables: [
                    if (stopsRefs) db.stops,
                    if (legsRefs) db.legs,
                    if (poisRefs) db.pois,
                    if (importBatchesRefs) db.importBatches,
                    if (contactsRefs) db.contacts,
                    if (callLogsRefs) db.callLogs,
                    if (checklistItemsRefs) db.checklistItems,
                    if (travellersRefs) db.travellers,
                    if (expensesRefs) db.expenses,
                    if (timelineEntriesRefs) db.timelineEntries,
                    if (trustedContactsRefs) db.trustedContacts,
                  ],
                  addJoins: null,
                  getPrefetchedDataCallback: (items) async {
                    return [
                      if (stopsRefs)
                        await $_getPrefetchedData<Trip, $TripsTable, Stop>(
                          currentTable: table,
                          referencedTable: $$TripsTableReferences
                              ._stopsRefsTable(db),
                          managerFromTypedResult: (p0) =>
                              $$TripsTableReferences(db, table, p0).stopsRefs,
                          referencedItemsForCurrentItem:
                              (item, referencedItems) => referencedItems.where(
                                (e) => e.tripId == item.id,
                              ),
                          typedResults: items,
                        ),
                      if (legsRefs)
                        await $_getPrefetchedData<Trip, $TripsTable, Leg>(
                          currentTable: table,
                          referencedTable: $$TripsTableReferences
                              ._legsRefsTable(db),
                          managerFromTypedResult: (p0) =>
                              $$TripsTableReferences(db, table, p0).legsRefs,
                          referencedItemsForCurrentItem:
                              (item, referencedItems) => referencedItems.where(
                                (e) => e.tripId == item.id,
                              ),
                          typedResults: items,
                        ),
                      if (poisRefs)
                        await $_getPrefetchedData<Trip, $TripsTable, Poi>(
                          currentTable: table,
                          referencedTable: $$TripsTableReferences
                              ._poisRefsTable(db),
                          managerFromTypedResult: (p0) =>
                              $$TripsTableReferences(db, table, p0).poisRefs,
                          referencedItemsForCurrentItem:
                              (item, referencedItems) => referencedItems.where(
                                (e) => e.tripId == item.id,
                              ),
                          typedResults: items,
                        ),
                      if (importBatchesRefs)
                        await $_getPrefetchedData<
                          Trip,
                          $TripsTable,
                          ImportBatche
                        >(
                          currentTable: table,
                          referencedTable: $$TripsTableReferences
                              ._importBatchesRefsTable(db),
                          managerFromTypedResult: (p0) =>
                              $$TripsTableReferences(
                                db,
                                table,
                                p0,
                              ).importBatchesRefs,
                          referencedItemsForCurrentItem:
                              (item, referencedItems) => referencedItems.where(
                                (e) => e.tripId == item.id,
                              ),
                          typedResults: items,
                        ),
                      if (contactsRefs)
                        await $_getPrefetchedData<Trip, $TripsTable, Contact>(
                          currentTable: table,
                          referencedTable: $$TripsTableReferences
                              ._contactsRefsTable(db),
                          managerFromTypedResult: (p0) =>
                              $$TripsTableReferences(
                                db,
                                table,
                                p0,
                              ).contactsRefs,
                          referencedItemsForCurrentItem:
                              (item, referencedItems) => referencedItems.where(
                                (e) => e.tripId == item.id,
                              ),
                          typedResults: items,
                        ),
                      if (callLogsRefs)
                        await $_getPrefetchedData<Trip, $TripsTable, CallLog>(
                          currentTable: table,
                          referencedTable: $$TripsTableReferences
                              ._callLogsRefsTable(db),
                          managerFromTypedResult: (p0) =>
                              $$TripsTableReferences(
                                db,
                                table,
                                p0,
                              ).callLogsRefs,
                          referencedItemsForCurrentItem:
                              (item, referencedItems) => referencedItems.where(
                                (e) => e.tripId == item.id,
                              ),
                          typedResults: items,
                        ),
                      if (checklistItemsRefs)
                        await $_getPrefetchedData<
                          Trip,
                          $TripsTable,
                          ChecklistItem
                        >(
                          currentTable: table,
                          referencedTable: $$TripsTableReferences
                              ._checklistItemsRefsTable(db),
                          managerFromTypedResult: (p0) =>
                              $$TripsTableReferences(
                                db,
                                table,
                                p0,
                              ).checklistItemsRefs,
                          referencedItemsForCurrentItem:
                              (item, referencedItems) => referencedItems.where(
                                (e) => e.tripId == item.id,
                              ),
                          typedResults: items,
                        ),
                      if (travellersRefs)
                        await $_getPrefetchedData<Trip, $TripsTable, Traveller>(
                          currentTable: table,
                          referencedTable: $$TripsTableReferences
                              ._travellersRefsTable(db),
                          managerFromTypedResult: (p0) =>
                              $$TripsTableReferences(
                                db,
                                table,
                                p0,
                              ).travellersRefs,
                          referencedItemsForCurrentItem:
                              (item, referencedItems) => referencedItems.where(
                                (e) => e.tripId == item.id,
                              ),
                          typedResults: items,
                        ),
                      if (expensesRefs)
                        await $_getPrefetchedData<Trip, $TripsTable, Expense>(
                          currentTable: table,
                          referencedTable: $$TripsTableReferences
                              ._expensesRefsTable(db),
                          managerFromTypedResult: (p0) =>
                              $$TripsTableReferences(
                                db,
                                table,
                                p0,
                              ).expensesRefs,
                          referencedItemsForCurrentItem:
                              (item, referencedItems) => referencedItems.where(
                                (e) => e.tripId == item.id,
                              ),
                          typedResults: items,
                        ),
                      if (timelineEntriesRefs)
                        await $_getPrefetchedData<
                          Trip,
                          $TripsTable,
                          TimelineEntry
                        >(
                          currentTable: table,
                          referencedTable: $$TripsTableReferences
                              ._timelineEntriesRefsTable(db),
                          managerFromTypedResult: (p0) =>
                              $$TripsTableReferences(
                                db,
                                table,
                                p0,
                              ).timelineEntriesRefs,
                          referencedItemsForCurrentItem:
                              (item, referencedItems) => referencedItems.where(
                                (e) => e.tripId == item.id,
                              ),
                          typedResults: items,
                        ),
                      if (trustedContactsRefs)
                        await $_getPrefetchedData<
                          Trip,
                          $TripsTable,
                          TrustedContact
                        >(
                          currentTable: table,
                          referencedTable: $$TripsTableReferences
                              ._trustedContactsRefsTable(db),
                          managerFromTypedResult: (p0) =>
                              $$TripsTableReferences(
                                db,
                                table,
                                p0,
                              ).trustedContactsRefs,
                          referencedItemsForCurrentItem:
                              (item, referencedItems) => referencedItems.where(
                                (e) => e.tripId == item.id,
                              ),
                          typedResults: items,
                        ),
                    ];
                  },
                );
              },
        ),
      );
}

typedef $$TripsTableProcessedTableManager =
    ProcessedTableManager<
      _$AppDatabase,
      $TripsTable,
      Trip,
      $$TripsTableFilterComposer,
      $$TripsTableOrderingComposer,
      $$TripsTableAnnotationComposer,
      $$TripsTableCreateCompanionBuilder,
      $$TripsTableUpdateCompanionBuilder,
      (Trip, $$TripsTableReferences),
      Trip,
      PrefetchHooks Function({
        bool stopsRefs,
        bool legsRefs,
        bool poisRefs,
        bool importBatchesRefs,
        bool contactsRefs,
        bool callLogsRefs,
        bool checklistItemsRefs,
        bool travellersRefs,
        bool expensesRefs,
        bool timelineEntriesRefs,
        bool trustedContactsRefs,
      })
    >;
typedef $$StopsTableCreateCompanionBuilder =
    StopsCompanion Function({
      Value<int> id,
      required int tripId,
      required String name,
      required int sequenceOrder,
      required String countryCode,
      Value<DateTime?> arrivalDate,
      Value<DateTime?> departureDate,
      Value<int> nights,
      Value<String> activityTags,
      Value<double?> lat,
      Value<double?> lon,
      Value<String?> note,
    });
typedef $$StopsTableUpdateCompanionBuilder =
    StopsCompanion Function({
      Value<int> id,
      Value<int> tripId,
      Value<String> name,
      Value<int> sequenceOrder,
      Value<String> countryCode,
      Value<DateTime?> arrivalDate,
      Value<DateTime?> departureDate,
      Value<int> nights,
      Value<String> activityTags,
      Value<double?> lat,
      Value<double?> lon,
      Value<String?> note,
    });

final class $$StopsTableReferences
    extends BaseReferences<_$AppDatabase, $StopsTable, Stop> {
  $$StopsTableReferences(super.$_db, super.$_table, super.$_typedResult);

  static $TripsTable _tripIdTable(_$AppDatabase db) =>
      db.trips.createAlias('stops__trip_id__trips__id');

  $$TripsTableProcessedTableManager get tripId {
    final $_column = $_itemColumn<int>('trip_id')!;

    final manager = $$TripsTableTableManager(
      $_db,
      $_db.trips,
    ).filter((f) => f.id.sqlEquals($_column));
    final item = $_typedResult.readTableOrNull(_tripIdTable($_db));
    if (item == null) return manager;
    return ProcessedTableManager(
      manager.$state.copyWith(prefetchedData: [item]),
    );
  }

  static MultiTypedResultKey<$PoisTable, List<Poi>> _poisRefsTable(
    _$AppDatabase db,
  ) => MultiTypedResultKey.fromTable(
    db.pois,
    aliasName: 'stops__id__pois__stop_id',
  );

  $$PoisTableProcessedTableManager get poisRefs {
    final manager = $$PoisTableTableManager(
      $_db,
      $_db.pois,
    ).filter((f) => f.stopId.id.sqlEquals($_itemColumn<int>('id')!));

    final cache = $_typedResult.readTableOrNull(_poisRefsTable($_db));
    return ProcessedTableManager(
      manager.$state.copyWith(prefetchedData: cache),
    );
  }

  static MultiTypedResultKey<$ContactsTable, List<Contact>> _contactsRefsTable(
    _$AppDatabase db,
  ) => MultiTypedResultKey.fromTable(
    db.contacts,
    aliasName: 'stops__id__contacts__stop_id',
  );

  $$ContactsTableProcessedTableManager get contactsRefs {
    final manager = $$ContactsTableTableManager(
      $_db,
      $_db.contacts,
    ).filter((f) => f.stopId.id.sqlEquals($_itemColumn<int>('id')!));

    final cache = $_typedResult.readTableOrNull(_contactsRefsTable($_db));
    return ProcessedTableManager(
      manager.$state.copyWith(prefetchedData: cache),
    );
  }

  static MultiTypedResultKey<$ChecklistItemsTable, List<ChecklistItem>>
  _checklistItemsRefsTable(_$AppDatabase db) => MultiTypedResultKey.fromTable(
    db.checklistItems,
    aliasName: 'stops__id__checklist_items__stop_id',
  );

  $$ChecklistItemsTableProcessedTableManager get checklistItemsRefs {
    final manager = $$ChecklistItemsTableTableManager(
      $_db,
      $_db.checklistItems,
    ).filter((f) => f.stopId.id.sqlEquals($_itemColumn<int>('id')!));

    final cache = $_typedResult.readTableOrNull(_checklistItemsRefsTable($_db));
    return ProcessedTableManager(
      manager.$state.copyWith(prefetchedData: cache),
    );
  }

  static MultiTypedResultKey<$WeatherSnapshotsTable, List<WeatherSnapshot>>
  _weatherSnapshotsRefsTable(_$AppDatabase db) => MultiTypedResultKey.fromTable(
    db.weatherSnapshots,
    aliasName: 'stops__id__weather_snapshots__stop_id',
  );

  $$WeatherSnapshotsTableProcessedTableManager get weatherSnapshotsRefs {
    final manager = $$WeatherSnapshotsTableTableManager(
      $_db,
      $_db.weatherSnapshots,
    ).filter((f) => f.stopId.id.sqlEquals($_itemColumn<int>('id')!));

    final cache = $_typedResult.readTableOrNull(
      _weatherSnapshotsRefsTable($_db),
    );
    return ProcessedTableManager(
      manager.$state.copyWith(prefetchedData: cache),
    );
  }

  static MultiTypedResultKey<$ExpensesTable, List<Expense>> _expensesRefsTable(
    _$AppDatabase db,
  ) => MultiTypedResultKey.fromTable(
    db.expenses,
    aliasName: 'stops__id__expenses__stop_id',
  );

  $$ExpensesTableProcessedTableManager get expensesRefs {
    final manager = $$ExpensesTableTableManager(
      $_db,
      $_db.expenses,
    ).filter((f) => f.stopId.id.sqlEquals($_itemColumn<int>('id')!));

    final cache = $_typedResult.readTableOrNull(_expensesRefsTable($_db));
    return ProcessedTableManager(
      manager.$state.copyWith(prefetchedData: cache),
    );
  }

  static MultiTypedResultKey<$TimelineEntriesTable, List<TimelineEntry>>
  _timelineEntriesRefsTable(_$AppDatabase db) => MultiTypedResultKey.fromTable(
    db.timelineEntries,
    aliasName: 'stops__id__timeline_entries__stop_id',
  );

  $$TimelineEntriesTableProcessedTableManager get timelineEntriesRefs {
    final manager = $$TimelineEntriesTableTableManager(
      $_db,
      $_db.timelineEntries,
    ).filter((f) => f.stopId.id.sqlEquals($_itemColumn<int>('id')!));

    final cache = $_typedResult.readTableOrNull(
      _timelineEntriesRefsTable($_db),
    );
    return ProcessedTableManager(
      manager.$state.copyWith(prefetchedData: cache),
    );
  }
}

class $$StopsTableFilterComposer extends Composer<_$AppDatabase, $StopsTable> {
  $$StopsTableFilterComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  ColumnFilters<int> get id => $composableBuilder(
    column: $table.id,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<String> get name => $composableBuilder(
    column: $table.name,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<int> get sequenceOrder => $composableBuilder(
    column: $table.sequenceOrder,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<String> get countryCode => $composableBuilder(
    column: $table.countryCode,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<DateTime> get arrivalDate => $composableBuilder(
    column: $table.arrivalDate,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<DateTime> get departureDate => $composableBuilder(
    column: $table.departureDate,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<int> get nights => $composableBuilder(
    column: $table.nights,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<String> get activityTags => $composableBuilder(
    column: $table.activityTags,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<double> get lat => $composableBuilder(
    column: $table.lat,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<double> get lon => $composableBuilder(
    column: $table.lon,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<String> get note => $composableBuilder(
    column: $table.note,
    builder: (column) => ColumnFilters(column),
  );

  $$TripsTableFilterComposer get tripId {
    final $$TripsTableFilterComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.tripId,
      referencedTable: $db.trips,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$TripsTableFilterComposer(
            $db: $db,
            $table: $db.trips,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }

  Expression<bool> poisRefs(
    Expression<bool> Function($$PoisTableFilterComposer f) f,
  ) {
    final $$PoisTableFilterComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.id,
      referencedTable: $db.pois,
      getReferencedColumn: (t) => t.stopId,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$PoisTableFilterComposer(
            $db: $db,
            $table: $db.pois,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return f(composer);
  }

  Expression<bool> contactsRefs(
    Expression<bool> Function($$ContactsTableFilterComposer f) f,
  ) {
    final $$ContactsTableFilterComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.id,
      referencedTable: $db.contacts,
      getReferencedColumn: (t) => t.stopId,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$ContactsTableFilterComposer(
            $db: $db,
            $table: $db.contacts,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return f(composer);
  }

  Expression<bool> checklistItemsRefs(
    Expression<bool> Function($$ChecklistItemsTableFilterComposer f) f,
  ) {
    final $$ChecklistItemsTableFilterComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.id,
      referencedTable: $db.checklistItems,
      getReferencedColumn: (t) => t.stopId,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$ChecklistItemsTableFilterComposer(
            $db: $db,
            $table: $db.checklistItems,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return f(composer);
  }

  Expression<bool> weatherSnapshotsRefs(
    Expression<bool> Function($$WeatherSnapshotsTableFilterComposer f) f,
  ) {
    final $$WeatherSnapshotsTableFilterComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.id,
      referencedTable: $db.weatherSnapshots,
      getReferencedColumn: (t) => t.stopId,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$WeatherSnapshotsTableFilterComposer(
            $db: $db,
            $table: $db.weatherSnapshots,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return f(composer);
  }

  Expression<bool> expensesRefs(
    Expression<bool> Function($$ExpensesTableFilterComposer f) f,
  ) {
    final $$ExpensesTableFilterComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.id,
      referencedTable: $db.expenses,
      getReferencedColumn: (t) => t.stopId,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$ExpensesTableFilterComposer(
            $db: $db,
            $table: $db.expenses,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return f(composer);
  }

  Expression<bool> timelineEntriesRefs(
    Expression<bool> Function($$TimelineEntriesTableFilterComposer f) f,
  ) {
    final $$TimelineEntriesTableFilterComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.id,
      referencedTable: $db.timelineEntries,
      getReferencedColumn: (t) => t.stopId,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$TimelineEntriesTableFilterComposer(
            $db: $db,
            $table: $db.timelineEntries,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return f(composer);
  }
}

class $$StopsTableOrderingComposer
    extends Composer<_$AppDatabase, $StopsTable> {
  $$StopsTableOrderingComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  ColumnOrderings<int> get id => $composableBuilder(
    column: $table.id,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<String> get name => $composableBuilder(
    column: $table.name,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<int> get sequenceOrder => $composableBuilder(
    column: $table.sequenceOrder,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<String> get countryCode => $composableBuilder(
    column: $table.countryCode,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<DateTime> get arrivalDate => $composableBuilder(
    column: $table.arrivalDate,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<DateTime> get departureDate => $composableBuilder(
    column: $table.departureDate,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<int> get nights => $composableBuilder(
    column: $table.nights,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<String> get activityTags => $composableBuilder(
    column: $table.activityTags,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<double> get lat => $composableBuilder(
    column: $table.lat,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<double> get lon => $composableBuilder(
    column: $table.lon,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<String> get note => $composableBuilder(
    column: $table.note,
    builder: (column) => ColumnOrderings(column),
  );

  $$TripsTableOrderingComposer get tripId {
    final $$TripsTableOrderingComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.tripId,
      referencedTable: $db.trips,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$TripsTableOrderingComposer(
            $db: $db,
            $table: $db.trips,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }
}

class $$StopsTableAnnotationComposer
    extends Composer<_$AppDatabase, $StopsTable> {
  $$StopsTableAnnotationComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  GeneratedColumn<int> get id =>
      $composableBuilder(column: $table.id, builder: (column) => column);

  GeneratedColumn<String> get name =>
      $composableBuilder(column: $table.name, builder: (column) => column);

  GeneratedColumn<int> get sequenceOrder => $composableBuilder(
    column: $table.sequenceOrder,
    builder: (column) => column,
  );

  GeneratedColumn<String> get countryCode => $composableBuilder(
    column: $table.countryCode,
    builder: (column) => column,
  );

  GeneratedColumn<DateTime> get arrivalDate => $composableBuilder(
    column: $table.arrivalDate,
    builder: (column) => column,
  );

  GeneratedColumn<DateTime> get departureDate => $composableBuilder(
    column: $table.departureDate,
    builder: (column) => column,
  );

  GeneratedColumn<int> get nights =>
      $composableBuilder(column: $table.nights, builder: (column) => column);

  GeneratedColumn<String> get activityTags => $composableBuilder(
    column: $table.activityTags,
    builder: (column) => column,
  );

  GeneratedColumn<double> get lat =>
      $composableBuilder(column: $table.lat, builder: (column) => column);

  GeneratedColumn<double> get lon =>
      $composableBuilder(column: $table.lon, builder: (column) => column);

  GeneratedColumn<String> get note =>
      $composableBuilder(column: $table.note, builder: (column) => column);

  $$TripsTableAnnotationComposer get tripId {
    final $$TripsTableAnnotationComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.tripId,
      referencedTable: $db.trips,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$TripsTableAnnotationComposer(
            $db: $db,
            $table: $db.trips,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }

  Expression<T> poisRefs<T extends Object>(
    Expression<T> Function($$PoisTableAnnotationComposer a) f,
  ) {
    final $$PoisTableAnnotationComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.id,
      referencedTable: $db.pois,
      getReferencedColumn: (t) => t.stopId,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$PoisTableAnnotationComposer(
            $db: $db,
            $table: $db.pois,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return f(composer);
  }

  Expression<T> contactsRefs<T extends Object>(
    Expression<T> Function($$ContactsTableAnnotationComposer a) f,
  ) {
    final $$ContactsTableAnnotationComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.id,
      referencedTable: $db.contacts,
      getReferencedColumn: (t) => t.stopId,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$ContactsTableAnnotationComposer(
            $db: $db,
            $table: $db.contacts,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return f(composer);
  }

  Expression<T> checklistItemsRefs<T extends Object>(
    Expression<T> Function($$ChecklistItemsTableAnnotationComposer a) f,
  ) {
    final $$ChecklistItemsTableAnnotationComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.id,
      referencedTable: $db.checklistItems,
      getReferencedColumn: (t) => t.stopId,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$ChecklistItemsTableAnnotationComposer(
            $db: $db,
            $table: $db.checklistItems,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return f(composer);
  }

  Expression<T> weatherSnapshotsRefs<T extends Object>(
    Expression<T> Function($$WeatherSnapshotsTableAnnotationComposer a) f,
  ) {
    final $$WeatherSnapshotsTableAnnotationComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.id,
      referencedTable: $db.weatherSnapshots,
      getReferencedColumn: (t) => t.stopId,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$WeatherSnapshotsTableAnnotationComposer(
            $db: $db,
            $table: $db.weatherSnapshots,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return f(composer);
  }

  Expression<T> expensesRefs<T extends Object>(
    Expression<T> Function($$ExpensesTableAnnotationComposer a) f,
  ) {
    final $$ExpensesTableAnnotationComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.id,
      referencedTable: $db.expenses,
      getReferencedColumn: (t) => t.stopId,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$ExpensesTableAnnotationComposer(
            $db: $db,
            $table: $db.expenses,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return f(composer);
  }

  Expression<T> timelineEntriesRefs<T extends Object>(
    Expression<T> Function($$TimelineEntriesTableAnnotationComposer a) f,
  ) {
    final $$TimelineEntriesTableAnnotationComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.id,
      referencedTable: $db.timelineEntries,
      getReferencedColumn: (t) => t.stopId,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$TimelineEntriesTableAnnotationComposer(
            $db: $db,
            $table: $db.timelineEntries,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return f(composer);
  }
}

class $$StopsTableTableManager
    extends
        RootTableManager<
          _$AppDatabase,
          $StopsTable,
          Stop,
          $$StopsTableFilterComposer,
          $$StopsTableOrderingComposer,
          $$StopsTableAnnotationComposer,
          $$StopsTableCreateCompanionBuilder,
          $$StopsTableUpdateCompanionBuilder,
          (Stop, $$StopsTableReferences),
          Stop,
          PrefetchHooks Function({
            bool tripId,
            bool poisRefs,
            bool contactsRefs,
            bool checklistItemsRefs,
            bool weatherSnapshotsRefs,
            bool expensesRefs,
            bool timelineEntriesRefs,
          })
        > {
  $$StopsTableTableManager(_$AppDatabase db, $StopsTable table)
    : super(
        TableManagerState(
          db: db,
          table: table,
          createFilteringComposer: () =>
              $$StopsTableFilterComposer($db: db, $table: table),
          createOrderingComposer: () =>
              $$StopsTableOrderingComposer($db: db, $table: table),
          createComputedFieldComposer: () =>
              $$StopsTableAnnotationComposer($db: db, $table: table),
          updateCompanionCallback:
              ({
                Value<int> id = const Value.absent(),
                Value<int> tripId = const Value.absent(),
                Value<String> name = const Value.absent(),
                Value<int> sequenceOrder = const Value.absent(),
                Value<String> countryCode = const Value.absent(),
                Value<DateTime?> arrivalDate = const Value.absent(),
                Value<DateTime?> departureDate = const Value.absent(),
                Value<int> nights = const Value.absent(),
                Value<String> activityTags = const Value.absent(),
                Value<double?> lat = const Value.absent(),
                Value<double?> lon = const Value.absent(),
                Value<String?> note = const Value.absent(),
              }) => StopsCompanion(
                id: id,
                tripId: tripId,
                name: name,
                sequenceOrder: sequenceOrder,
                countryCode: countryCode,
                arrivalDate: arrivalDate,
                departureDate: departureDate,
                nights: nights,
                activityTags: activityTags,
                lat: lat,
                lon: lon,
                note: note,
              ),
          createCompanionCallback:
              ({
                Value<int> id = const Value.absent(),
                required int tripId,
                required String name,
                required int sequenceOrder,
                required String countryCode,
                Value<DateTime?> arrivalDate = const Value.absent(),
                Value<DateTime?> departureDate = const Value.absent(),
                Value<int> nights = const Value.absent(),
                Value<String> activityTags = const Value.absent(),
                Value<double?> lat = const Value.absent(),
                Value<double?> lon = const Value.absent(),
                Value<String?> note = const Value.absent(),
              }) => StopsCompanion.insert(
                id: id,
                tripId: tripId,
                name: name,
                sequenceOrder: sequenceOrder,
                countryCode: countryCode,
                arrivalDate: arrivalDate,
                departureDate: departureDate,
                nights: nights,
                activityTags: activityTags,
                lat: lat,
                lon: lon,
                note: note,
              ),
          withReferenceMapper: (p0) => p0
              .map(
                (e) => (
                  e.readTable<$StopsTable, Stop>(table),
                  $$StopsTableReferences(db, table, e),
                ),
              )
              .toList(),
          prefetchHooksCallback:
              ({
                tripId = false,
                poisRefs = false,
                contactsRefs = false,
                checklistItemsRefs = false,
                weatherSnapshotsRefs = false,
                expensesRefs = false,
                timelineEntriesRefs = false,
              }) {
                return PrefetchHooks(
                  db: db,
                  explicitlyWatchedTables: [
                    if (poisRefs) db.pois,
                    if (contactsRefs) db.contacts,
                    if (checklistItemsRefs) db.checklistItems,
                    if (weatherSnapshotsRefs) db.weatherSnapshots,
                    if (expensesRefs) db.expenses,
                    if (timelineEntriesRefs) db.timelineEntries,
                  ],
                  addJoins:
                      <
                        T extends TableManagerState<
                          dynamic,
                          dynamic,
                          dynamic,
                          dynamic,
                          dynamic,
                          dynamic,
                          dynamic,
                          dynamic,
                          dynamic,
                          dynamic,
                          dynamic
                        >
                      >(state) {
                        if (tripId) {
                          state =
                              state.withJoin(
                                    currentTable: table,
                                    currentColumn: table.tripId,
                                    referencedTable: $$StopsTableReferences
                                        ._tripIdTable(db),
                                    referencedColumn: $$StopsTableReferences
                                        ._tripIdTable(db)
                                        .id,
                                  )
                                  as T;
                        }

                        return state;
                      },
                  getPrefetchedDataCallback: (items) async {
                    return [
                      if (poisRefs)
                        await $_getPrefetchedData<Stop, $StopsTable, Poi>(
                          currentTable: table,
                          referencedTable: $$StopsTableReferences
                              ._poisRefsTable(db),
                          managerFromTypedResult: (p0) =>
                              $$StopsTableReferences(db, table, p0).poisRefs,
                          referencedItemsForCurrentItem:
                              (item, referencedItems) => referencedItems.where(
                                (e) => e.stopId == item.id,
                              ),
                          typedResults: items,
                        ),
                      if (contactsRefs)
                        await $_getPrefetchedData<Stop, $StopsTable, Contact>(
                          currentTable: table,
                          referencedTable: $$StopsTableReferences
                              ._contactsRefsTable(db),
                          managerFromTypedResult: (p0) =>
                              $$StopsTableReferences(
                                db,
                                table,
                                p0,
                              ).contactsRefs,
                          referencedItemsForCurrentItem:
                              (item, referencedItems) => referencedItems.where(
                                (e) => e.stopId == item.id,
                              ),
                          typedResults: items,
                        ),
                      if (checklistItemsRefs)
                        await $_getPrefetchedData<
                          Stop,
                          $StopsTable,
                          ChecklistItem
                        >(
                          currentTable: table,
                          referencedTable: $$StopsTableReferences
                              ._checklistItemsRefsTable(db),
                          managerFromTypedResult: (p0) =>
                              $$StopsTableReferences(
                                db,
                                table,
                                p0,
                              ).checklistItemsRefs,
                          referencedItemsForCurrentItem:
                              (item, referencedItems) => referencedItems.where(
                                (e) => e.stopId == item.id,
                              ),
                          typedResults: items,
                        ),
                      if (weatherSnapshotsRefs)
                        await $_getPrefetchedData<
                          Stop,
                          $StopsTable,
                          WeatherSnapshot
                        >(
                          currentTable: table,
                          referencedTable: $$StopsTableReferences
                              ._weatherSnapshotsRefsTable(db),
                          managerFromTypedResult: (p0) =>
                              $$StopsTableReferences(
                                db,
                                table,
                                p0,
                              ).weatherSnapshotsRefs,
                          referencedItemsForCurrentItem:
                              (item, referencedItems) => referencedItems.where(
                                (e) => e.stopId == item.id,
                              ),
                          typedResults: items,
                        ),
                      if (expensesRefs)
                        await $_getPrefetchedData<Stop, $StopsTable, Expense>(
                          currentTable: table,
                          referencedTable: $$StopsTableReferences
                              ._expensesRefsTable(db),
                          managerFromTypedResult: (p0) =>
                              $$StopsTableReferences(
                                db,
                                table,
                                p0,
                              ).expensesRefs,
                          referencedItemsForCurrentItem:
                              (item, referencedItems) => referencedItems.where(
                                (e) => e.stopId == item.id,
                              ),
                          typedResults: items,
                        ),
                      if (timelineEntriesRefs)
                        await $_getPrefetchedData<
                          Stop,
                          $StopsTable,
                          TimelineEntry
                        >(
                          currentTable: table,
                          referencedTable: $$StopsTableReferences
                              ._timelineEntriesRefsTable(db),
                          managerFromTypedResult: (p0) =>
                              $$StopsTableReferences(
                                db,
                                table,
                                p0,
                              ).timelineEntriesRefs,
                          referencedItemsForCurrentItem:
                              (item, referencedItems) => referencedItems.where(
                                (e) => e.stopId == item.id,
                              ),
                          typedResults: items,
                        ),
                    ];
                  },
                );
              },
        ),
      );
}

typedef $$StopsTableProcessedTableManager =
    ProcessedTableManager<
      _$AppDatabase,
      $StopsTable,
      Stop,
      $$StopsTableFilterComposer,
      $$StopsTableOrderingComposer,
      $$StopsTableAnnotationComposer,
      $$StopsTableCreateCompanionBuilder,
      $$StopsTableUpdateCompanionBuilder,
      (Stop, $$StopsTableReferences),
      Stop,
      PrefetchHooks Function({
        bool tripId,
        bool poisRefs,
        bool contactsRefs,
        bool checklistItemsRefs,
        bool weatherSnapshotsRefs,
        bool expensesRefs,
        bool timelineEntriesRefs,
      })
    >;
typedef $$LegsTableCreateCompanionBuilder =
    LegsCompanion Function({
      Value<int> id,
      required int tripId,
      required int fromStopId,
      required int toStopId,
      required int sequenceOrder,
      Value<String?> mode,
      Value<DateTime?> plannedDeparture,
      Value<DateTime?> plannedArrival,
      Value<bool> isBooked,
      Value<String?> note,
      Value<String?> routePolyline,
      Value<double?> distanceKm,
      Value<double> corridorKm,
      Value<DateTime?> lastSyncedAt,
    });
typedef $$LegsTableUpdateCompanionBuilder =
    LegsCompanion Function({
      Value<int> id,
      Value<int> tripId,
      Value<int> fromStopId,
      Value<int> toStopId,
      Value<int> sequenceOrder,
      Value<String?> mode,
      Value<DateTime?> plannedDeparture,
      Value<DateTime?> plannedArrival,
      Value<bool> isBooked,
      Value<String?> note,
      Value<String?> routePolyline,
      Value<double?> distanceKm,
      Value<double> corridorKm,
      Value<DateTime?> lastSyncedAt,
    });

final class $$LegsTableReferences
    extends BaseReferences<_$AppDatabase, $LegsTable, Leg> {
  $$LegsTableReferences(super.$_db, super.$_table, super.$_typedResult);

  static $TripsTable _tripIdTable(_$AppDatabase db) =>
      db.trips.createAlias('legs__trip_id__trips__id');

  $$TripsTableProcessedTableManager get tripId {
    final $_column = $_itemColumn<int>('trip_id')!;

    final manager = $$TripsTableTableManager(
      $_db,
      $_db.trips,
    ).filter((f) => f.id.sqlEquals($_column));
    final item = $_typedResult.readTableOrNull(_tripIdTable($_db));
    if (item == null) return manager;
    return ProcessedTableManager(
      manager.$state.copyWith(prefetchedData: [item]),
    );
  }

  static $StopsTable _fromStopIdTable(_$AppDatabase db) =>
      db.stops.createAlias('legs__from_stop_id__stops__id');

  $$StopsTableProcessedTableManager get fromStopId {
    final $_column = $_itemColumn<int>('from_stop_id')!;

    final manager = $$StopsTableTableManager(
      $_db,
      $_db.stops,
    ).filter((f) => f.id.sqlEquals($_column));
    final item = $_typedResult.readTableOrNull(_fromStopIdTable($_db));
    if (item == null) return manager;
    return ProcessedTableManager(
      manager.$state.copyWith(prefetchedData: [item]),
    );
  }

  static $StopsTable _toStopIdTable(_$AppDatabase db) =>
      db.stops.createAlias('legs__to_stop_id__stops__id');

  $$StopsTableProcessedTableManager get toStopId {
    final $_column = $_itemColumn<int>('to_stop_id')!;

    final manager = $$StopsTableTableManager(
      $_db,
      $_db.stops,
    ).filter((f) => f.id.sqlEquals($_column));
    final item = $_typedResult.readTableOrNull(_toStopIdTable($_db));
    if (item == null) return manager;
    return ProcessedTableManager(
      manager.$state.copyWith(prefetchedData: [item]),
    );
  }

  static MultiTypedResultKey<$PoisTable, List<Poi>> _poisRefsTable(
    _$AppDatabase db,
  ) => MultiTypedResultKey.fromTable(
    db.pois,
    aliasName: 'legs__id__pois__leg_id',
  );

  $$PoisTableProcessedTableManager get poisRefs {
    final manager = $$PoisTableTableManager(
      $_db,
      $_db.pois,
    ).filter((f) => f.legId.id.sqlEquals($_itemColumn<int>('id')!));

    final cache = $_typedResult.readTableOrNull(_poisRefsTable($_db));
    return ProcessedTableManager(
      manager.$state.copyWith(prefetchedData: cache),
    );
  }
}

class $$LegsTableFilterComposer extends Composer<_$AppDatabase, $LegsTable> {
  $$LegsTableFilterComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  ColumnFilters<int> get id => $composableBuilder(
    column: $table.id,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<int> get sequenceOrder => $composableBuilder(
    column: $table.sequenceOrder,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<String> get mode => $composableBuilder(
    column: $table.mode,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<DateTime> get plannedDeparture => $composableBuilder(
    column: $table.plannedDeparture,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<DateTime> get plannedArrival => $composableBuilder(
    column: $table.plannedArrival,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<bool> get isBooked => $composableBuilder(
    column: $table.isBooked,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<String> get note => $composableBuilder(
    column: $table.note,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<String> get routePolyline => $composableBuilder(
    column: $table.routePolyline,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<double> get distanceKm => $composableBuilder(
    column: $table.distanceKm,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<double> get corridorKm => $composableBuilder(
    column: $table.corridorKm,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<DateTime> get lastSyncedAt => $composableBuilder(
    column: $table.lastSyncedAt,
    builder: (column) => ColumnFilters(column),
  );

  $$TripsTableFilterComposer get tripId {
    final $$TripsTableFilterComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.tripId,
      referencedTable: $db.trips,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$TripsTableFilterComposer(
            $db: $db,
            $table: $db.trips,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }

  $$StopsTableFilterComposer get fromStopId {
    final $$StopsTableFilterComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.fromStopId,
      referencedTable: $db.stops,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$StopsTableFilterComposer(
            $db: $db,
            $table: $db.stops,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }

  $$StopsTableFilterComposer get toStopId {
    final $$StopsTableFilterComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.toStopId,
      referencedTable: $db.stops,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$StopsTableFilterComposer(
            $db: $db,
            $table: $db.stops,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }

  Expression<bool> poisRefs(
    Expression<bool> Function($$PoisTableFilterComposer f) f,
  ) {
    final $$PoisTableFilterComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.id,
      referencedTable: $db.pois,
      getReferencedColumn: (t) => t.legId,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$PoisTableFilterComposer(
            $db: $db,
            $table: $db.pois,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return f(composer);
  }
}

class $$LegsTableOrderingComposer extends Composer<_$AppDatabase, $LegsTable> {
  $$LegsTableOrderingComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  ColumnOrderings<int> get id => $composableBuilder(
    column: $table.id,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<int> get sequenceOrder => $composableBuilder(
    column: $table.sequenceOrder,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<String> get mode => $composableBuilder(
    column: $table.mode,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<DateTime> get plannedDeparture => $composableBuilder(
    column: $table.plannedDeparture,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<DateTime> get plannedArrival => $composableBuilder(
    column: $table.plannedArrival,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<bool> get isBooked => $composableBuilder(
    column: $table.isBooked,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<String> get note => $composableBuilder(
    column: $table.note,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<String> get routePolyline => $composableBuilder(
    column: $table.routePolyline,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<double> get distanceKm => $composableBuilder(
    column: $table.distanceKm,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<double> get corridorKm => $composableBuilder(
    column: $table.corridorKm,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<DateTime> get lastSyncedAt => $composableBuilder(
    column: $table.lastSyncedAt,
    builder: (column) => ColumnOrderings(column),
  );

  $$TripsTableOrderingComposer get tripId {
    final $$TripsTableOrderingComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.tripId,
      referencedTable: $db.trips,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$TripsTableOrderingComposer(
            $db: $db,
            $table: $db.trips,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }

  $$StopsTableOrderingComposer get fromStopId {
    final $$StopsTableOrderingComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.fromStopId,
      referencedTable: $db.stops,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$StopsTableOrderingComposer(
            $db: $db,
            $table: $db.stops,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }

  $$StopsTableOrderingComposer get toStopId {
    final $$StopsTableOrderingComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.toStopId,
      referencedTable: $db.stops,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$StopsTableOrderingComposer(
            $db: $db,
            $table: $db.stops,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }
}

class $$LegsTableAnnotationComposer
    extends Composer<_$AppDatabase, $LegsTable> {
  $$LegsTableAnnotationComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  GeneratedColumn<int> get id =>
      $composableBuilder(column: $table.id, builder: (column) => column);

  GeneratedColumn<int> get sequenceOrder => $composableBuilder(
    column: $table.sequenceOrder,
    builder: (column) => column,
  );

  GeneratedColumn<String> get mode =>
      $composableBuilder(column: $table.mode, builder: (column) => column);

  GeneratedColumn<DateTime> get plannedDeparture => $composableBuilder(
    column: $table.plannedDeparture,
    builder: (column) => column,
  );

  GeneratedColumn<DateTime> get plannedArrival => $composableBuilder(
    column: $table.plannedArrival,
    builder: (column) => column,
  );

  GeneratedColumn<bool> get isBooked =>
      $composableBuilder(column: $table.isBooked, builder: (column) => column);

  GeneratedColumn<String> get note =>
      $composableBuilder(column: $table.note, builder: (column) => column);

  GeneratedColumn<String> get routePolyline => $composableBuilder(
    column: $table.routePolyline,
    builder: (column) => column,
  );

  GeneratedColumn<double> get distanceKm => $composableBuilder(
    column: $table.distanceKm,
    builder: (column) => column,
  );

  GeneratedColumn<double> get corridorKm => $composableBuilder(
    column: $table.corridorKm,
    builder: (column) => column,
  );

  GeneratedColumn<DateTime> get lastSyncedAt => $composableBuilder(
    column: $table.lastSyncedAt,
    builder: (column) => column,
  );

  $$TripsTableAnnotationComposer get tripId {
    final $$TripsTableAnnotationComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.tripId,
      referencedTable: $db.trips,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$TripsTableAnnotationComposer(
            $db: $db,
            $table: $db.trips,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }

  $$StopsTableAnnotationComposer get fromStopId {
    final $$StopsTableAnnotationComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.fromStopId,
      referencedTable: $db.stops,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$StopsTableAnnotationComposer(
            $db: $db,
            $table: $db.stops,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }

  $$StopsTableAnnotationComposer get toStopId {
    final $$StopsTableAnnotationComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.toStopId,
      referencedTable: $db.stops,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$StopsTableAnnotationComposer(
            $db: $db,
            $table: $db.stops,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }

  Expression<T> poisRefs<T extends Object>(
    Expression<T> Function($$PoisTableAnnotationComposer a) f,
  ) {
    final $$PoisTableAnnotationComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.id,
      referencedTable: $db.pois,
      getReferencedColumn: (t) => t.legId,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$PoisTableAnnotationComposer(
            $db: $db,
            $table: $db.pois,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return f(composer);
  }
}

class $$LegsTableTableManager
    extends
        RootTableManager<
          _$AppDatabase,
          $LegsTable,
          Leg,
          $$LegsTableFilterComposer,
          $$LegsTableOrderingComposer,
          $$LegsTableAnnotationComposer,
          $$LegsTableCreateCompanionBuilder,
          $$LegsTableUpdateCompanionBuilder,
          (Leg, $$LegsTableReferences),
          Leg,
          PrefetchHooks Function({
            bool tripId,
            bool fromStopId,
            bool toStopId,
            bool poisRefs,
          })
        > {
  $$LegsTableTableManager(_$AppDatabase db, $LegsTable table)
    : super(
        TableManagerState(
          db: db,
          table: table,
          createFilteringComposer: () =>
              $$LegsTableFilterComposer($db: db, $table: table),
          createOrderingComposer: () =>
              $$LegsTableOrderingComposer($db: db, $table: table),
          createComputedFieldComposer: () =>
              $$LegsTableAnnotationComposer($db: db, $table: table),
          updateCompanionCallback:
              ({
                Value<int> id = const Value.absent(),
                Value<int> tripId = const Value.absent(),
                Value<int> fromStopId = const Value.absent(),
                Value<int> toStopId = const Value.absent(),
                Value<int> sequenceOrder = const Value.absent(),
                Value<String?> mode = const Value.absent(),
                Value<DateTime?> plannedDeparture = const Value.absent(),
                Value<DateTime?> plannedArrival = const Value.absent(),
                Value<bool> isBooked = const Value.absent(),
                Value<String?> note = const Value.absent(),
                Value<String?> routePolyline = const Value.absent(),
                Value<double?> distanceKm = const Value.absent(),
                Value<double> corridorKm = const Value.absent(),
                Value<DateTime?> lastSyncedAt = const Value.absent(),
              }) => LegsCompanion(
                id: id,
                tripId: tripId,
                fromStopId: fromStopId,
                toStopId: toStopId,
                sequenceOrder: sequenceOrder,
                mode: mode,
                plannedDeparture: plannedDeparture,
                plannedArrival: plannedArrival,
                isBooked: isBooked,
                note: note,
                routePolyline: routePolyline,
                distanceKm: distanceKm,
                corridorKm: corridorKm,
                lastSyncedAt: lastSyncedAt,
              ),
          createCompanionCallback:
              ({
                Value<int> id = const Value.absent(),
                required int tripId,
                required int fromStopId,
                required int toStopId,
                required int sequenceOrder,
                Value<String?> mode = const Value.absent(),
                Value<DateTime?> plannedDeparture = const Value.absent(),
                Value<DateTime?> plannedArrival = const Value.absent(),
                Value<bool> isBooked = const Value.absent(),
                Value<String?> note = const Value.absent(),
                Value<String?> routePolyline = const Value.absent(),
                Value<double?> distanceKm = const Value.absent(),
                Value<double> corridorKm = const Value.absent(),
                Value<DateTime?> lastSyncedAt = const Value.absent(),
              }) => LegsCompanion.insert(
                id: id,
                tripId: tripId,
                fromStopId: fromStopId,
                toStopId: toStopId,
                sequenceOrder: sequenceOrder,
                mode: mode,
                plannedDeparture: plannedDeparture,
                plannedArrival: plannedArrival,
                isBooked: isBooked,
                note: note,
                routePolyline: routePolyline,
                distanceKm: distanceKm,
                corridorKm: corridorKm,
                lastSyncedAt: lastSyncedAt,
              ),
          withReferenceMapper: (p0) => p0
              .map(
                (e) => (
                  e.readTable<$LegsTable, Leg>(table),
                  $$LegsTableReferences(db, table, e),
                ),
              )
              .toList(),
          prefetchHooksCallback:
              ({
                tripId = false,
                fromStopId = false,
                toStopId = false,
                poisRefs = false,
              }) {
                return PrefetchHooks(
                  db: db,
                  explicitlyWatchedTables: [if (poisRefs) db.pois],
                  addJoins:
                      <
                        T extends TableManagerState<
                          dynamic,
                          dynamic,
                          dynamic,
                          dynamic,
                          dynamic,
                          dynamic,
                          dynamic,
                          dynamic,
                          dynamic,
                          dynamic,
                          dynamic
                        >
                      >(state) {
                        if (tripId) {
                          state =
                              state.withJoin(
                                    currentTable: table,
                                    currentColumn: table.tripId,
                                    referencedTable: $$LegsTableReferences
                                        ._tripIdTable(db),
                                    referencedColumn: $$LegsTableReferences
                                        ._tripIdTable(db)
                                        .id,
                                  )
                                  as T;
                        }
                        if (fromStopId) {
                          state =
                              state.withJoin(
                                    currentTable: table,
                                    currentColumn: table.fromStopId,
                                    referencedTable: $$LegsTableReferences
                                        ._fromStopIdTable(db),
                                    referencedColumn: $$LegsTableReferences
                                        ._fromStopIdTable(db)
                                        .id,
                                  )
                                  as T;
                        }
                        if (toStopId) {
                          state =
                              state.withJoin(
                                    currentTable: table,
                                    currentColumn: table.toStopId,
                                    referencedTable: $$LegsTableReferences
                                        ._toStopIdTable(db),
                                    referencedColumn: $$LegsTableReferences
                                        ._toStopIdTable(db)
                                        .id,
                                  )
                                  as T;
                        }

                        return state;
                      },
                  getPrefetchedDataCallback: (items) async {
                    return [
                      if (poisRefs)
                        await $_getPrefetchedData<Leg, $LegsTable, Poi>(
                          currentTable: table,
                          referencedTable: $$LegsTableReferences._poisRefsTable(
                            db,
                          ),
                          managerFromTypedResult: (p0) =>
                              $$LegsTableReferences(db, table, p0).poisRefs,
                          referencedItemsForCurrentItem:
                              (item, referencedItems) => referencedItems.where(
                                (e) => e.legId == item.id,
                              ),
                          typedResults: items,
                        ),
                    ];
                  },
                );
              },
        ),
      );
}

typedef $$LegsTableProcessedTableManager =
    ProcessedTableManager<
      _$AppDatabase,
      $LegsTable,
      Leg,
      $$LegsTableFilterComposer,
      $$LegsTableOrderingComposer,
      $$LegsTableAnnotationComposer,
      $$LegsTableCreateCompanionBuilder,
      $$LegsTableUpdateCompanionBuilder,
      (Leg, $$LegsTableReferences),
      Leg,
      PrefetchHooks Function({
        bool tripId,
        bool fromStopId,
        bool toStopId,
        bool poisRefs,
      })
    >;
typedef $$PoisTableCreateCompanionBuilder =
    PoisCompanion Function({
      Value<int> id,
      required int tripId,
      Value<int?> stopId,
      Value<int?> legId,
      required String name,
      required String category,
      required double lat,
      required double lon,
      Value<double?> distanceAlongRouteKm,
      Value<double?> distanceOffRouteKm,
      Value<String?> osmId,
      Value<String?> rawTags,
      Value<DateTime> cachedAt,
    });
typedef $$PoisTableUpdateCompanionBuilder =
    PoisCompanion Function({
      Value<int> id,
      Value<int> tripId,
      Value<int?> stopId,
      Value<int?> legId,
      Value<String> name,
      Value<String> category,
      Value<double> lat,
      Value<double> lon,
      Value<double?> distanceAlongRouteKm,
      Value<double?> distanceOffRouteKm,
      Value<String?> osmId,
      Value<String?> rawTags,
      Value<DateTime> cachedAt,
    });

final class $$PoisTableReferences
    extends BaseReferences<_$AppDatabase, $PoisTable, Poi> {
  $$PoisTableReferences(super.$_db, super.$_table, super.$_typedResult);

  static $TripsTable _tripIdTable(_$AppDatabase db) =>
      db.trips.createAlias('pois__trip_id__trips__id');

  $$TripsTableProcessedTableManager get tripId {
    final $_column = $_itemColumn<int>('trip_id')!;

    final manager = $$TripsTableTableManager(
      $_db,
      $_db.trips,
    ).filter((f) => f.id.sqlEquals($_column));
    final item = $_typedResult.readTableOrNull(_tripIdTable($_db));
    if (item == null) return manager;
    return ProcessedTableManager(
      manager.$state.copyWith(prefetchedData: [item]),
    );
  }

  static $StopsTable _stopIdTable(_$AppDatabase db) =>
      db.stops.createAlias('pois__stop_id__stops__id');

  $$StopsTableProcessedTableManager? get stopId {
    final $_column = $_itemColumn<int>('stop_id');
    if ($_column == null) return null;
    final manager = $$StopsTableTableManager(
      $_db,
      $_db.stops,
    ).filter((f) => f.id.sqlEquals($_column));
    final item = $_typedResult.readTableOrNull(_stopIdTable($_db));
    if (item == null) return manager;
    return ProcessedTableManager(
      manager.$state.copyWith(prefetchedData: [item]),
    );
  }

  static $LegsTable _legIdTable(_$AppDatabase db) =>
      db.legs.createAlias('pois__leg_id__legs__id');

  $$LegsTableProcessedTableManager? get legId {
    final $_column = $_itemColumn<int>('leg_id');
    if ($_column == null) return null;
    final manager = $$LegsTableTableManager(
      $_db,
      $_db.legs,
    ).filter((f) => f.id.sqlEquals($_column));
    final item = $_typedResult.readTableOrNull(_legIdTable($_db));
    if (item == null) return manager;
    return ProcessedTableManager(
      manager.$state.copyWith(prefetchedData: [item]),
    );
  }

  static MultiTypedResultKey<$PoiContactsTable, List<PoiContact>>
  _poiContactsRefsTable(_$AppDatabase db) => MultiTypedResultKey.fromTable(
    db.poiContacts,
    aliasName: 'pois__id__poi_contacts__poi_id',
  );

  $$PoiContactsTableProcessedTableManager get poiContactsRefs {
    final manager = $$PoiContactsTableTableManager(
      $_db,
      $_db.poiContacts,
    ).filter((f) => f.poiId.id.sqlEquals($_itemColumn<int>('id')!));

    final cache = $_typedResult.readTableOrNull(_poiContactsRefsTable($_db));
    return ProcessedTableManager(
      manager.$state.copyWith(prefetchedData: cache),
    );
  }
}

class $$PoisTableFilterComposer extends Composer<_$AppDatabase, $PoisTable> {
  $$PoisTableFilterComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  ColumnFilters<int> get id => $composableBuilder(
    column: $table.id,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<String> get name => $composableBuilder(
    column: $table.name,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<String> get category => $composableBuilder(
    column: $table.category,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<double> get lat => $composableBuilder(
    column: $table.lat,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<double> get lon => $composableBuilder(
    column: $table.lon,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<double> get distanceAlongRouteKm => $composableBuilder(
    column: $table.distanceAlongRouteKm,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<double> get distanceOffRouteKm => $composableBuilder(
    column: $table.distanceOffRouteKm,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<String> get osmId => $composableBuilder(
    column: $table.osmId,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<String> get rawTags => $composableBuilder(
    column: $table.rawTags,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<DateTime> get cachedAt => $composableBuilder(
    column: $table.cachedAt,
    builder: (column) => ColumnFilters(column),
  );

  $$TripsTableFilterComposer get tripId {
    final $$TripsTableFilterComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.tripId,
      referencedTable: $db.trips,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$TripsTableFilterComposer(
            $db: $db,
            $table: $db.trips,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }

  $$StopsTableFilterComposer get stopId {
    final $$StopsTableFilterComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.stopId,
      referencedTable: $db.stops,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$StopsTableFilterComposer(
            $db: $db,
            $table: $db.stops,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }

  $$LegsTableFilterComposer get legId {
    final $$LegsTableFilterComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.legId,
      referencedTable: $db.legs,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$LegsTableFilterComposer(
            $db: $db,
            $table: $db.legs,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }

  Expression<bool> poiContactsRefs(
    Expression<bool> Function($$PoiContactsTableFilterComposer f) f,
  ) {
    final $$PoiContactsTableFilterComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.id,
      referencedTable: $db.poiContacts,
      getReferencedColumn: (t) => t.poiId,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$PoiContactsTableFilterComposer(
            $db: $db,
            $table: $db.poiContacts,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return f(composer);
  }
}

class $$PoisTableOrderingComposer extends Composer<_$AppDatabase, $PoisTable> {
  $$PoisTableOrderingComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  ColumnOrderings<int> get id => $composableBuilder(
    column: $table.id,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<String> get name => $composableBuilder(
    column: $table.name,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<String> get category => $composableBuilder(
    column: $table.category,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<double> get lat => $composableBuilder(
    column: $table.lat,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<double> get lon => $composableBuilder(
    column: $table.lon,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<double> get distanceAlongRouteKm => $composableBuilder(
    column: $table.distanceAlongRouteKm,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<double> get distanceOffRouteKm => $composableBuilder(
    column: $table.distanceOffRouteKm,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<String> get osmId => $composableBuilder(
    column: $table.osmId,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<String> get rawTags => $composableBuilder(
    column: $table.rawTags,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<DateTime> get cachedAt => $composableBuilder(
    column: $table.cachedAt,
    builder: (column) => ColumnOrderings(column),
  );

  $$TripsTableOrderingComposer get tripId {
    final $$TripsTableOrderingComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.tripId,
      referencedTable: $db.trips,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$TripsTableOrderingComposer(
            $db: $db,
            $table: $db.trips,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }

  $$StopsTableOrderingComposer get stopId {
    final $$StopsTableOrderingComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.stopId,
      referencedTable: $db.stops,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$StopsTableOrderingComposer(
            $db: $db,
            $table: $db.stops,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }

  $$LegsTableOrderingComposer get legId {
    final $$LegsTableOrderingComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.legId,
      referencedTable: $db.legs,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$LegsTableOrderingComposer(
            $db: $db,
            $table: $db.legs,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }
}

class $$PoisTableAnnotationComposer
    extends Composer<_$AppDatabase, $PoisTable> {
  $$PoisTableAnnotationComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  GeneratedColumn<int> get id =>
      $composableBuilder(column: $table.id, builder: (column) => column);

  GeneratedColumn<String> get name =>
      $composableBuilder(column: $table.name, builder: (column) => column);

  GeneratedColumn<String> get category =>
      $composableBuilder(column: $table.category, builder: (column) => column);

  GeneratedColumn<double> get lat =>
      $composableBuilder(column: $table.lat, builder: (column) => column);

  GeneratedColumn<double> get lon =>
      $composableBuilder(column: $table.lon, builder: (column) => column);

  GeneratedColumn<double> get distanceAlongRouteKm => $composableBuilder(
    column: $table.distanceAlongRouteKm,
    builder: (column) => column,
  );

  GeneratedColumn<double> get distanceOffRouteKm => $composableBuilder(
    column: $table.distanceOffRouteKm,
    builder: (column) => column,
  );

  GeneratedColumn<String> get osmId =>
      $composableBuilder(column: $table.osmId, builder: (column) => column);

  GeneratedColumn<String> get rawTags =>
      $composableBuilder(column: $table.rawTags, builder: (column) => column);

  GeneratedColumn<DateTime> get cachedAt =>
      $composableBuilder(column: $table.cachedAt, builder: (column) => column);

  $$TripsTableAnnotationComposer get tripId {
    final $$TripsTableAnnotationComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.tripId,
      referencedTable: $db.trips,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$TripsTableAnnotationComposer(
            $db: $db,
            $table: $db.trips,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }

  $$StopsTableAnnotationComposer get stopId {
    final $$StopsTableAnnotationComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.stopId,
      referencedTable: $db.stops,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$StopsTableAnnotationComposer(
            $db: $db,
            $table: $db.stops,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }

  $$LegsTableAnnotationComposer get legId {
    final $$LegsTableAnnotationComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.legId,
      referencedTable: $db.legs,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$LegsTableAnnotationComposer(
            $db: $db,
            $table: $db.legs,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }

  Expression<T> poiContactsRefs<T extends Object>(
    Expression<T> Function($$PoiContactsTableAnnotationComposer a) f,
  ) {
    final $$PoiContactsTableAnnotationComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.id,
      referencedTable: $db.poiContacts,
      getReferencedColumn: (t) => t.poiId,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$PoiContactsTableAnnotationComposer(
            $db: $db,
            $table: $db.poiContacts,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return f(composer);
  }
}

class $$PoisTableTableManager
    extends
        RootTableManager<
          _$AppDatabase,
          $PoisTable,
          Poi,
          $$PoisTableFilterComposer,
          $$PoisTableOrderingComposer,
          $$PoisTableAnnotationComposer,
          $$PoisTableCreateCompanionBuilder,
          $$PoisTableUpdateCompanionBuilder,
          (Poi, $$PoisTableReferences),
          Poi,
          PrefetchHooks Function({
            bool tripId,
            bool stopId,
            bool legId,
            bool poiContactsRefs,
          })
        > {
  $$PoisTableTableManager(_$AppDatabase db, $PoisTable table)
    : super(
        TableManagerState(
          db: db,
          table: table,
          createFilteringComposer: () =>
              $$PoisTableFilterComposer($db: db, $table: table),
          createOrderingComposer: () =>
              $$PoisTableOrderingComposer($db: db, $table: table),
          createComputedFieldComposer: () =>
              $$PoisTableAnnotationComposer($db: db, $table: table),
          updateCompanionCallback:
              ({
                Value<int> id = const Value.absent(),
                Value<int> tripId = const Value.absent(),
                Value<int?> stopId = const Value.absent(),
                Value<int?> legId = const Value.absent(),
                Value<String> name = const Value.absent(),
                Value<String> category = const Value.absent(),
                Value<double> lat = const Value.absent(),
                Value<double> lon = const Value.absent(),
                Value<double?> distanceAlongRouteKm = const Value.absent(),
                Value<double?> distanceOffRouteKm = const Value.absent(),
                Value<String?> osmId = const Value.absent(),
                Value<String?> rawTags = const Value.absent(),
                Value<DateTime> cachedAt = const Value.absent(),
              }) => PoisCompanion(
                id: id,
                tripId: tripId,
                stopId: stopId,
                legId: legId,
                name: name,
                category: category,
                lat: lat,
                lon: lon,
                distanceAlongRouteKm: distanceAlongRouteKm,
                distanceOffRouteKm: distanceOffRouteKm,
                osmId: osmId,
                rawTags: rawTags,
                cachedAt: cachedAt,
              ),
          createCompanionCallback:
              ({
                Value<int> id = const Value.absent(),
                required int tripId,
                Value<int?> stopId = const Value.absent(),
                Value<int?> legId = const Value.absent(),
                required String name,
                required String category,
                required double lat,
                required double lon,
                Value<double?> distanceAlongRouteKm = const Value.absent(),
                Value<double?> distanceOffRouteKm = const Value.absent(),
                Value<String?> osmId = const Value.absent(),
                Value<String?> rawTags = const Value.absent(),
                Value<DateTime> cachedAt = const Value.absent(),
              }) => PoisCompanion.insert(
                id: id,
                tripId: tripId,
                stopId: stopId,
                legId: legId,
                name: name,
                category: category,
                lat: lat,
                lon: lon,
                distanceAlongRouteKm: distanceAlongRouteKm,
                distanceOffRouteKm: distanceOffRouteKm,
                osmId: osmId,
                rawTags: rawTags,
                cachedAt: cachedAt,
              ),
          withReferenceMapper: (p0) => p0
              .map(
                (e) => (
                  e.readTable<$PoisTable, Poi>(table),
                  $$PoisTableReferences(db, table, e),
                ),
              )
              .toList(),
          prefetchHooksCallback:
              ({
                tripId = false,
                stopId = false,
                legId = false,
                poiContactsRefs = false,
              }) {
                return PrefetchHooks(
                  db: db,
                  explicitlyWatchedTables: [
                    if (poiContactsRefs) db.poiContacts,
                  ],
                  addJoins:
                      <
                        T extends TableManagerState<
                          dynamic,
                          dynamic,
                          dynamic,
                          dynamic,
                          dynamic,
                          dynamic,
                          dynamic,
                          dynamic,
                          dynamic,
                          dynamic,
                          dynamic
                        >
                      >(state) {
                        if (tripId) {
                          state =
                              state.withJoin(
                                    currentTable: table,
                                    currentColumn: table.tripId,
                                    referencedTable: $$PoisTableReferences
                                        ._tripIdTable(db),
                                    referencedColumn: $$PoisTableReferences
                                        ._tripIdTable(db)
                                        .id,
                                  )
                                  as T;
                        }
                        if (stopId) {
                          state =
                              state.withJoin(
                                    currentTable: table,
                                    currentColumn: table.stopId,
                                    referencedTable: $$PoisTableReferences
                                        ._stopIdTable(db),
                                    referencedColumn: $$PoisTableReferences
                                        ._stopIdTable(db)
                                        .id,
                                  )
                                  as T;
                        }
                        if (legId) {
                          state =
                              state.withJoin(
                                    currentTable: table,
                                    currentColumn: table.legId,
                                    referencedTable: $$PoisTableReferences
                                        ._legIdTable(db),
                                    referencedColumn: $$PoisTableReferences
                                        ._legIdTable(db)
                                        .id,
                                  )
                                  as T;
                        }

                        return state;
                      },
                  getPrefetchedDataCallback: (items) async {
                    return [
                      if (poiContactsRefs)
                        await $_getPrefetchedData<Poi, $PoisTable, PoiContact>(
                          currentTable: table,
                          referencedTable: $$PoisTableReferences
                              ._poiContactsRefsTable(db),
                          managerFromTypedResult: (p0) => $$PoisTableReferences(
                            db,
                            table,
                            p0,
                          ).poiContactsRefs,
                          referencedItemsForCurrentItem:
                              (item, referencedItems) => referencedItems.where(
                                (e) => e.poiId == item.id,
                              ),
                          typedResults: items,
                        ),
                    ];
                  },
                );
              },
        ),
      );
}

typedef $$PoisTableProcessedTableManager =
    ProcessedTableManager<
      _$AppDatabase,
      $PoisTable,
      Poi,
      $$PoisTableFilterComposer,
      $$PoisTableOrderingComposer,
      $$PoisTableAnnotationComposer,
      $$PoisTableCreateCompanionBuilder,
      $$PoisTableUpdateCompanionBuilder,
      (Poi, $$PoisTableReferences),
      Poi,
      PrefetchHooks Function({
        bool tripId,
        bool stopId,
        bool legId,
        bool poiContactsRefs,
      })
    >;
typedef $$PoiContactsTableCreateCompanionBuilder =
    PoiContactsCompanion Function({
      Value<int> id,
      required int poiId,
      required String phoneRaw,
      Value<String?> phoneE164,
      Value<String> tier,
      Value<String?> sourceTag,
    });
typedef $$PoiContactsTableUpdateCompanionBuilder =
    PoiContactsCompanion Function({
      Value<int> id,
      Value<int> poiId,
      Value<String> phoneRaw,
      Value<String?> phoneE164,
      Value<String> tier,
      Value<String?> sourceTag,
    });

final class $$PoiContactsTableReferences
    extends BaseReferences<_$AppDatabase, $PoiContactsTable, PoiContact> {
  $$PoiContactsTableReferences(super.$_db, super.$_table, super.$_typedResult);

  static $PoisTable _poiIdTable(_$AppDatabase db) =>
      db.pois.createAlias('poi_contacts__poi_id__pois__id');

  $$PoisTableProcessedTableManager get poiId {
    final $_column = $_itemColumn<int>('poi_id')!;

    final manager = $$PoisTableTableManager(
      $_db,
      $_db.pois,
    ).filter((f) => f.id.sqlEquals($_column));
    final item = $_typedResult.readTableOrNull(_poiIdTable($_db));
    if (item == null) return manager;
    return ProcessedTableManager(
      manager.$state.copyWith(prefetchedData: [item]),
    );
  }
}

class $$PoiContactsTableFilterComposer
    extends Composer<_$AppDatabase, $PoiContactsTable> {
  $$PoiContactsTableFilterComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  ColumnFilters<int> get id => $composableBuilder(
    column: $table.id,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<String> get phoneRaw => $composableBuilder(
    column: $table.phoneRaw,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<String> get phoneE164 => $composableBuilder(
    column: $table.phoneE164,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<String> get tier => $composableBuilder(
    column: $table.tier,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<String> get sourceTag => $composableBuilder(
    column: $table.sourceTag,
    builder: (column) => ColumnFilters(column),
  );

  $$PoisTableFilterComposer get poiId {
    final $$PoisTableFilterComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.poiId,
      referencedTable: $db.pois,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$PoisTableFilterComposer(
            $db: $db,
            $table: $db.pois,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }
}

class $$PoiContactsTableOrderingComposer
    extends Composer<_$AppDatabase, $PoiContactsTable> {
  $$PoiContactsTableOrderingComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  ColumnOrderings<int> get id => $composableBuilder(
    column: $table.id,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<String> get phoneRaw => $composableBuilder(
    column: $table.phoneRaw,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<String> get phoneE164 => $composableBuilder(
    column: $table.phoneE164,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<String> get tier => $composableBuilder(
    column: $table.tier,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<String> get sourceTag => $composableBuilder(
    column: $table.sourceTag,
    builder: (column) => ColumnOrderings(column),
  );

  $$PoisTableOrderingComposer get poiId {
    final $$PoisTableOrderingComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.poiId,
      referencedTable: $db.pois,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$PoisTableOrderingComposer(
            $db: $db,
            $table: $db.pois,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }
}

class $$PoiContactsTableAnnotationComposer
    extends Composer<_$AppDatabase, $PoiContactsTable> {
  $$PoiContactsTableAnnotationComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  GeneratedColumn<int> get id =>
      $composableBuilder(column: $table.id, builder: (column) => column);

  GeneratedColumn<String> get phoneRaw =>
      $composableBuilder(column: $table.phoneRaw, builder: (column) => column);

  GeneratedColumn<String> get phoneE164 =>
      $composableBuilder(column: $table.phoneE164, builder: (column) => column);

  GeneratedColumn<String> get tier =>
      $composableBuilder(column: $table.tier, builder: (column) => column);

  GeneratedColumn<String> get sourceTag =>
      $composableBuilder(column: $table.sourceTag, builder: (column) => column);

  $$PoisTableAnnotationComposer get poiId {
    final $$PoisTableAnnotationComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.poiId,
      referencedTable: $db.pois,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$PoisTableAnnotationComposer(
            $db: $db,
            $table: $db.pois,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }
}

class $$PoiContactsTableTableManager
    extends
        RootTableManager<
          _$AppDatabase,
          $PoiContactsTable,
          PoiContact,
          $$PoiContactsTableFilterComposer,
          $$PoiContactsTableOrderingComposer,
          $$PoiContactsTableAnnotationComposer,
          $$PoiContactsTableCreateCompanionBuilder,
          $$PoiContactsTableUpdateCompanionBuilder,
          (PoiContact, $$PoiContactsTableReferences),
          PoiContact,
          PrefetchHooks Function({bool poiId})
        > {
  $$PoiContactsTableTableManager(_$AppDatabase db, $PoiContactsTable table)
    : super(
        TableManagerState(
          db: db,
          table: table,
          createFilteringComposer: () =>
              $$PoiContactsTableFilterComposer($db: db, $table: table),
          createOrderingComposer: () =>
              $$PoiContactsTableOrderingComposer($db: db, $table: table),
          createComputedFieldComposer: () =>
              $$PoiContactsTableAnnotationComposer($db: db, $table: table),
          updateCompanionCallback:
              ({
                Value<int> id = const Value.absent(),
                Value<int> poiId = const Value.absent(),
                Value<String> phoneRaw = const Value.absent(),
                Value<String?> phoneE164 = const Value.absent(),
                Value<String> tier = const Value.absent(),
                Value<String?> sourceTag = const Value.absent(),
              }) => PoiContactsCompanion(
                id: id,
                poiId: poiId,
                phoneRaw: phoneRaw,
                phoneE164: phoneE164,
                tier: tier,
                sourceTag: sourceTag,
              ),
          createCompanionCallback:
              ({
                Value<int> id = const Value.absent(),
                required int poiId,
                required String phoneRaw,
                Value<String?> phoneE164 = const Value.absent(),
                Value<String> tier = const Value.absent(),
                Value<String?> sourceTag = const Value.absent(),
              }) => PoiContactsCompanion.insert(
                id: id,
                poiId: poiId,
                phoneRaw: phoneRaw,
                phoneE164: phoneE164,
                tier: tier,
                sourceTag: sourceTag,
              ),
          withReferenceMapper: (p0) => p0
              .map(
                (e) => (
                  e.readTable<$PoiContactsTable, PoiContact>(table),
                  $$PoiContactsTableReferences(db, table, e),
                ),
              )
              .toList(),
          prefetchHooksCallback: ({poiId = false}) {
            return PrefetchHooks(
              db: db,
              explicitlyWatchedTables: [],
              addJoins:
                  <
                    T extends TableManagerState<
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic
                    >
                  >(state) {
                    if (poiId) {
                      state =
                          state.withJoin(
                                currentTable: table,
                                currentColumn: table.poiId,
                                referencedTable: $$PoiContactsTableReferences
                                    ._poiIdTable(db),
                                referencedColumn: $$PoiContactsTableReferences
                                    ._poiIdTable(db)
                                    .id,
                              )
                              as T;
                    }

                    return state;
                  },
              getPrefetchedDataCallback: (items) async {
                return [];
              },
            );
          },
        ),
      );
}

typedef $$PoiContactsTableProcessedTableManager =
    ProcessedTableManager<
      _$AppDatabase,
      $PoiContactsTable,
      PoiContact,
      $$PoiContactsTableFilterComposer,
      $$PoiContactsTableOrderingComposer,
      $$PoiContactsTableAnnotationComposer,
      $$PoiContactsTableCreateCompanionBuilder,
      $$PoiContactsTableUpdateCompanionBuilder,
      (PoiContact, $$PoiContactsTableReferences),
      PoiContact,
      PrefetchHooks Function({bool poiId})
    >;
typedef $$ImportBatchesTableCreateCompanionBuilder =
    ImportBatchesCompanion Function({
      Value<int> id,
      Value<int?> tripId,
      required String fileName,
      Value<String?> sheetName,
      Value<int> rowsImported,
      Value<int> rowsSkipped,
      Value<DateTime> importedAt,
    });
typedef $$ImportBatchesTableUpdateCompanionBuilder =
    ImportBatchesCompanion Function({
      Value<int> id,
      Value<int?> tripId,
      Value<String> fileName,
      Value<String?> sheetName,
      Value<int> rowsImported,
      Value<int> rowsSkipped,
      Value<DateTime> importedAt,
    });

final class $$ImportBatchesTableReferences
    extends BaseReferences<_$AppDatabase, $ImportBatchesTable, ImportBatche> {
  $$ImportBatchesTableReferences(
    super.$_db,
    super.$_table,
    super.$_typedResult,
  );

  static $TripsTable _tripIdTable(_$AppDatabase db) =>
      db.trips.createAlias('import_batches__trip_id__trips__id');

  $$TripsTableProcessedTableManager? get tripId {
    final $_column = $_itemColumn<int>('trip_id');
    if ($_column == null) return null;
    final manager = $$TripsTableTableManager(
      $_db,
      $_db.trips,
    ).filter((f) => f.id.sqlEquals($_column));
    final item = $_typedResult.readTableOrNull(_tripIdTable($_db));
    if (item == null) return manager;
    return ProcessedTableManager(
      manager.$state.copyWith(prefetchedData: [item]),
    );
  }

  static MultiTypedResultKey<$ContactsTable, List<Contact>> _contactsRefsTable(
    _$AppDatabase db,
  ) => MultiTypedResultKey.fromTable(
    db.contacts,
    aliasName: 'import_batches__id__contacts__import_batch_id',
  );

  $$ContactsTableProcessedTableManager get contactsRefs {
    final manager = $$ContactsTableTableManager(
      $_db,
      $_db.contacts,
    ).filter((f) => f.importBatchId.id.sqlEquals($_itemColumn<int>('id')!));

    final cache = $_typedResult.readTableOrNull(_contactsRefsTable($_db));
    return ProcessedTableManager(
      manager.$state.copyWith(prefetchedData: cache),
    );
  }
}

class $$ImportBatchesTableFilterComposer
    extends Composer<_$AppDatabase, $ImportBatchesTable> {
  $$ImportBatchesTableFilterComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  ColumnFilters<int> get id => $composableBuilder(
    column: $table.id,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<String> get fileName => $composableBuilder(
    column: $table.fileName,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<String> get sheetName => $composableBuilder(
    column: $table.sheetName,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<int> get rowsImported => $composableBuilder(
    column: $table.rowsImported,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<int> get rowsSkipped => $composableBuilder(
    column: $table.rowsSkipped,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<DateTime> get importedAt => $composableBuilder(
    column: $table.importedAt,
    builder: (column) => ColumnFilters(column),
  );

  $$TripsTableFilterComposer get tripId {
    final $$TripsTableFilterComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.tripId,
      referencedTable: $db.trips,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$TripsTableFilterComposer(
            $db: $db,
            $table: $db.trips,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }

  Expression<bool> contactsRefs(
    Expression<bool> Function($$ContactsTableFilterComposer f) f,
  ) {
    final $$ContactsTableFilterComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.id,
      referencedTable: $db.contacts,
      getReferencedColumn: (t) => t.importBatchId,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$ContactsTableFilterComposer(
            $db: $db,
            $table: $db.contacts,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return f(composer);
  }
}

class $$ImportBatchesTableOrderingComposer
    extends Composer<_$AppDatabase, $ImportBatchesTable> {
  $$ImportBatchesTableOrderingComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  ColumnOrderings<int> get id => $composableBuilder(
    column: $table.id,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<String> get fileName => $composableBuilder(
    column: $table.fileName,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<String> get sheetName => $composableBuilder(
    column: $table.sheetName,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<int> get rowsImported => $composableBuilder(
    column: $table.rowsImported,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<int> get rowsSkipped => $composableBuilder(
    column: $table.rowsSkipped,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<DateTime> get importedAt => $composableBuilder(
    column: $table.importedAt,
    builder: (column) => ColumnOrderings(column),
  );

  $$TripsTableOrderingComposer get tripId {
    final $$TripsTableOrderingComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.tripId,
      referencedTable: $db.trips,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$TripsTableOrderingComposer(
            $db: $db,
            $table: $db.trips,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }
}

class $$ImportBatchesTableAnnotationComposer
    extends Composer<_$AppDatabase, $ImportBatchesTable> {
  $$ImportBatchesTableAnnotationComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  GeneratedColumn<int> get id =>
      $composableBuilder(column: $table.id, builder: (column) => column);

  GeneratedColumn<String> get fileName =>
      $composableBuilder(column: $table.fileName, builder: (column) => column);

  GeneratedColumn<String> get sheetName =>
      $composableBuilder(column: $table.sheetName, builder: (column) => column);

  GeneratedColumn<int> get rowsImported => $composableBuilder(
    column: $table.rowsImported,
    builder: (column) => column,
  );

  GeneratedColumn<int> get rowsSkipped => $composableBuilder(
    column: $table.rowsSkipped,
    builder: (column) => column,
  );

  GeneratedColumn<DateTime> get importedAt => $composableBuilder(
    column: $table.importedAt,
    builder: (column) => column,
  );

  $$TripsTableAnnotationComposer get tripId {
    final $$TripsTableAnnotationComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.tripId,
      referencedTable: $db.trips,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$TripsTableAnnotationComposer(
            $db: $db,
            $table: $db.trips,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }

  Expression<T> contactsRefs<T extends Object>(
    Expression<T> Function($$ContactsTableAnnotationComposer a) f,
  ) {
    final $$ContactsTableAnnotationComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.id,
      referencedTable: $db.contacts,
      getReferencedColumn: (t) => t.importBatchId,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$ContactsTableAnnotationComposer(
            $db: $db,
            $table: $db.contacts,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return f(composer);
  }
}

class $$ImportBatchesTableTableManager
    extends
        RootTableManager<
          _$AppDatabase,
          $ImportBatchesTable,
          ImportBatche,
          $$ImportBatchesTableFilterComposer,
          $$ImportBatchesTableOrderingComposer,
          $$ImportBatchesTableAnnotationComposer,
          $$ImportBatchesTableCreateCompanionBuilder,
          $$ImportBatchesTableUpdateCompanionBuilder,
          (ImportBatche, $$ImportBatchesTableReferences),
          ImportBatche,
          PrefetchHooks Function({bool tripId, bool contactsRefs})
        > {
  $$ImportBatchesTableTableManager(_$AppDatabase db, $ImportBatchesTable table)
    : super(
        TableManagerState(
          db: db,
          table: table,
          createFilteringComposer: () =>
              $$ImportBatchesTableFilterComposer($db: db, $table: table),
          createOrderingComposer: () =>
              $$ImportBatchesTableOrderingComposer($db: db, $table: table),
          createComputedFieldComposer: () =>
              $$ImportBatchesTableAnnotationComposer($db: db, $table: table),
          updateCompanionCallback:
              ({
                Value<int> id = const Value.absent(),
                Value<int?> tripId = const Value.absent(),
                Value<String> fileName = const Value.absent(),
                Value<String?> sheetName = const Value.absent(),
                Value<int> rowsImported = const Value.absent(),
                Value<int> rowsSkipped = const Value.absent(),
                Value<DateTime> importedAt = const Value.absent(),
              }) => ImportBatchesCompanion(
                id: id,
                tripId: tripId,
                fileName: fileName,
                sheetName: sheetName,
                rowsImported: rowsImported,
                rowsSkipped: rowsSkipped,
                importedAt: importedAt,
              ),
          createCompanionCallback:
              ({
                Value<int> id = const Value.absent(),
                Value<int?> tripId = const Value.absent(),
                required String fileName,
                Value<String?> sheetName = const Value.absent(),
                Value<int> rowsImported = const Value.absent(),
                Value<int> rowsSkipped = const Value.absent(),
                Value<DateTime> importedAt = const Value.absent(),
              }) => ImportBatchesCompanion.insert(
                id: id,
                tripId: tripId,
                fileName: fileName,
                sheetName: sheetName,
                rowsImported: rowsImported,
                rowsSkipped: rowsSkipped,
                importedAt: importedAt,
              ),
          withReferenceMapper: (p0) => p0
              .map(
                (e) => (
                  e.readTable<$ImportBatchesTable, ImportBatche>(table),
                  $$ImportBatchesTableReferences(db, table, e),
                ),
              )
              .toList(),
          prefetchHooksCallback: ({tripId = false, contactsRefs = false}) {
            return PrefetchHooks(
              db: db,
              explicitlyWatchedTables: [if (contactsRefs) db.contacts],
              addJoins:
                  <
                    T extends TableManagerState<
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic
                    >
                  >(state) {
                    if (tripId) {
                      state =
                          state.withJoin(
                                currentTable: table,
                                currentColumn: table.tripId,
                                referencedTable: $$ImportBatchesTableReferences
                                    ._tripIdTable(db),
                                referencedColumn: $$ImportBatchesTableReferences
                                    ._tripIdTable(db)
                                    .id,
                              )
                              as T;
                    }

                    return state;
                  },
              getPrefetchedDataCallback: (items) async {
                return [
                  if (contactsRefs)
                    await $_getPrefetchedData<
                      ImportBatche,
                      $ImportBatchesTable,
                      Contact
                    >(
                      currentTable: table,
                      referencedTable: $$ImportBatchesTableReferences
                          ._contactsRefsTable(db),
                      managerFromTypedResult: (p0) =>
                          $$ImportBatchesTableReferences(
                            db,
                            table,
                            p0,
                          ).contactsRefs,
                      referencedItemsForCurrentItem: (item, referencedItems) =>
                          referencedItems.where(
                            (e) => e.importBatchId == item.id,
                          ),
                      typedResults: items,
                    ),
                ];
              },
            );
          },
        ),
      );
}

typedef $$ImportBatchesTableProcessedTableManager =
    ProcessedTableManager<
      _$AppDatabase,
      $ImportBatchesTable,
      ImportBatche,
      $$ImportBatchesTableFilterComposer,
      $$ImportBatchesTableOrderingComposer,
      $$ImportBatchesTableAnnotationComposer,
      $$ImportBatchesTableCreateCompanionBuilder,
      $$ImportBatchesTableUpdateCompanionBuilder,
      (ImportBatche, $$ImportBatchesTableReferences),
      ImportBatche,
      PrefetchHooks Function({bool tripId, bool contactsRefs})
    >;
typedef $$ContactsTableCreateCompanionBuilder =
    ContactsCompanion Function({
      Value<int> id,
      Value<int?> tripId,
      Value<int?> stopId,
      required String name,
      required String phoneRaw,
      Value<String?> phoneE164,
      Value<String?> note,
      Value<String> category,
      Value<String> tier,
      Value<bool> callConfirmed,
      Value<DateTime?> confirmedAt,
      Value<bool> isPinned,
      Value<bool> isEmergency,
      Value<bool> hasWhatsapp,
      Value<DateTime?> lastCalledAt,
      Value<int> callCount,
      Value<int?> importBatchId,
      Value<DateTime> createdAt,
    });
typedef $$ContactsTableUpdateCompanionBuilder =
    ContactsCompanion Function({
      Value<int> id,
      Value<int?> tripId,
      Value<int?> stopId,
      Value<String> name,
      Value<String> phoneRaw,
      Value<String?> phoneE164,
      Value<String?> note,
      Value<String> category,
      Value<String> tier,
      Value<bool> callConfirmed,
      Value<DateTime?> confirmedAt,
      Value<bool> isPinned,
      Value<bool> isEmergency,
      Value<bool> hasWhatsapp,
      Value<DateTime?> lastCalledAt,
      Value<int> callCount,
      Value<int?> importBatchId,
      Value<DateTime> createdAt,
    });

final class $$ContactsTableReferences
    extends BaseReferences<_$AppDatabase, $ContactsTable, Contact> {
  $$ContactsTableReferences(super.$_db, super.$_table, super.$_typedResult);

  static $TripsTable _tripIdTable(_$AppDatabase db) =>
      db.trips.createAlias('contacts__trip_id__trips__id');

  $$TripsTableProcessedTableManager? get tripId {
    final $_column = $_itemColumn<int>('trip_id');
    if ($_column == null) return null;
    final manager = $$TripsTableTableManager(
      $_db,
      $_db.trips,
    ).filter((f) => f.id.sqlEquals($_column));
    final item = $_typedResult.readTableOrNull(_tripIdTable($_db));
    if (item == null) return manager;
    return ProcessedTableManager(
      manager.$state.copyWith(prefetchedData: [item]),
    );
  }

  static $StopsTable _stopIdTable(_$AppDatabase db) =>
      db.stops.createAlias('contacts__stop_id__stops__id');

  $$StopsTableProcessedTableManager? get stopId {
    final $_column = $_itemColumn<int>('stop_id');
    if ($_column == null) return null;
    final manager = $$StopsTableTableManager(
      $_db,
      $_db.stops,
    ).filter((f) => f.id.sqlEquals($_column));
    final item = $_typedResult.readTableOrNull(_stopIdTable($_db));
    if (item == null) return manager;
    return ProcessedTableManager(
      manager.$state.copyWith(prefetchedData: [item]),
    );
  }

  static $ImportBatchesTable _importBatchIdTable(_$AppDatabase db) => db
      .importBatches
      .createAlias('contacts__import_batch_id__import_batches__id');

  $$ImportBatchesTableProcessedTableManager? get importBatchId {
    final $_column = $_itemColumn<int>('import_batch_id');
    if ($_column == null) return null;
    final manager = $$ImportBatchesTableTableManager(
      $_db,
      $_db.importBatches,
    ).filter((f) => f.id.sqlEquals($_column));
    final item = $_typedResult.readTableOrNull(_importBatchIdTable($_db));
    if (item == null) return manager;
    return ProcessedTableManager(
      manager.$state.copyWith(prefetchedData: [item]),
    );
  }

  static MultiTypedResultKey<$CallLogsTable, List<CallLog>> _callLogsRefsTable(
    _$AppDatabase db,
  ) => MultiTypedResultKey.fromTable(
    db.callLogs,
    aliasName: 'contacts__id__call_logs__contact_id',
  );

  $$CallLogsTableProcessedTableManager get callLogsRefs {
    final manager = $$CallLogsTableTableManager(
      $_db,
      $_db.callLogs,
    ).filter((f) => f.contactId.id.sqlEquals($_itemColumn<int>('id')!));

    final cache = $_typedResult.readTableOrNull(_callLogsRefsTable($_db));
    return ProcessedTableManager(
      manager.$state.copyWith(prefetchedData: cache),
    );
  }

  static MultiTypedResultKey<$ChecklistItemsTable, List<ChecklistItem>>
  _checklistItemsRefsTable(_$AppDatabase db) => MultiTypedResultKey.fromTable(
    db.checklistItems,
    aliasName: 'contacts__id__checklist_items__contact_id',
  );

  $$ChecklistItemsTableProcessedTableManager get checklistItemsRefs {
    final manager = $$ChecklistItemsTableTableManager(
      $_db,
      $_db.checklistItems,
    ).filter((f) => f.contactId.id.sqlEquals($_itemColumn<int>('id')!));

    final cache = $_typedResult.readTableOrNull(_checklistItemsRefsTable($_db));
    return ProcessedTableManager(
      manager.$state.copyWith(prefetchedData: cache),
    );
  }
}

class $$ContactsTableFilterComposer
    extends Composer<_$AppDatabase, $ContactsTable> {
  $$ContactsTableFilterComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  ColumnFilters<int> get id => $composableBuilder(
    column: $table.id,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<String> get name => $composableBuilder(
    column: $table.name,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<String> get phoneRaw => $composableBuilder(
    column: $table.phoneRaw,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<String> get phoneE164 => $composableBuilder(
    column: $table.phoneE164,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<String> get note => $composableBuilder(
    column: $table.note,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<String> get category => $composableBuilder(
    column: $table.category,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<String> get tier => $composableBuilder(
    column: $table.tier,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<bool> get callConfirmed => $composableBuilder(
    column: $table.callConfirmed,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<DateTime> get confirmedAt => $composableBuilder(
    column: $table.confirmedAt,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<bool> get isPinned => $composableBuilder(
    column: $table.isPinned,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<bool> get isEmergency => $composableBuilder(
    column: $table.isEmergency,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<bool> get hasWhatsapp => $composableBuilder(
    column: $table.hasWhatsapp,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<DateTime> get lastCalledAt => $composableBuilder(
    column: $table.lastCalledAt,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<int> get callCount => $composableBuilder(
    column: $table.callCount,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<DateTime> get createdAt => $composableBuilder(
    column: $table.createdAt,
    builder: (column) => ColumnFilters(column),
  );

  $$TripsTableFilterComposer get tripId {
    final $$TripsTableFilterComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.tripId,
      referencedTable: $db.trips,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$TripsTableFilterComposer(
            $db: $db,
            $table: $db.trips,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }

  $$StopsTableFilterComposer get stopId {
    final $$StopsTableFilterComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.stopId,
      referencedTable: $db.stops,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$StopsTableFilterComposer(
            $db: $db,
            $table: $db.stops,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }

  $$ImportBatchesTableFilterComposer get importBatchId {
    final $$ImportBatchesTableFilterComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.importBatchId,
      referencedTable: $db.importBatches,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$ImportBatchesTableFilterComposer(
            $db: $db,
            $table: $db.importBatches,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }

  Expression<bool> callLogsRefs(
    Expression<bool> Function($$CallLogsTableFilterComposer f) f,
  ) {
    final $$CallLogsTableFilterComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.id,
      referencedTable: $db.callLogs,
      getReferencedColumn: (t) => t.contactId,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$CallLogsTableFilterComposer(
            $db: $db,
            $table: $db.callLogs,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return f(composer);
  }

  Expression<bool> checklistItemsRefs(
    Expression<bool> Function($$ChecklistItemsTableFilterComposer f) f,
  ) {
    final $$ChecklistItemsTableFilterComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.id,
      referencedTable: $db.checklistItems,
      getReferencedColumn: (t) => t.contactId,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$ChecklistItemsTableFilterComposer(
            $db: $db,
            $table: $db.checklistItems,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return f(composer);
  }
}

class $$ContactsTableOrderingComposer
    extends Composer<_$AppDatabase, $ContactsTable> {
  $$ContactsTableOrderingComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  ColumnOrderings<int> get id => $composableBuilder(
    column: $table.id,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<String> get name => $composableBuilder(
    column: $table.name,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<String> get phoneRaw => $composableBuilder(
    column: $table.phoneRaw,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<String> get phoneE164 => $composableBuilder(
    column: $table.phoneE164,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<String> get note => $composableBuilder(
    column: $table.note,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<String> get category => $composableBuilder(
    column: $table.category,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<String> get tier => $composableBuilder(
    column: $table.tier,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<bool> get callConfirmed => $composableBuilder(
    column: $table.callConfirmed,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<DateTime> get confirmedAt => $composableBuilder(
    column: $table.confirmedAt,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<bool> get isPinned => $composableBuilder(
    column: $table.isPinned,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<bool> get isEmergency => $composableBuilder(
    column: $table.isEmergency,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<bool> get hasWhatsapp => $composableBuilder(
    column: $table.hasWhatsapp,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<DateTime> get lastCalledAt => $composableBuilder(
    column: $table.lastCalledAt,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<int> get callCount => $composableBuilder(
    column: $table.callCount,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<DateTime> get createdAt => $composableBuilder(
    column: $table.createdAt,
    builder: (column) => ColumnOrderings(column),
  );

  $$TripsTableOrderingComposer get tripId {
    final $$TripsTableOrderingComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.tripId,
      referencedTable: $db.trips,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$TripsTableOrderingComposer(
            $db: $db,
            $table: $db.trips,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }

  $$StopsTableOrderingComposer get stopId {
    final $$StopsTableOrderingComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.stopId,
      referencedTable: $db.stops,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$StopsTableOrderingComposer(
            $db: $db,
            $table: $db.stops,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }

  $$ImportBatchesTableOrderingComposer get importBatchId {
    final $$ImportBatchesTableOrderingComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.importBatchId,
      referencedTable: $db.importBatches,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$ImportBatchesTableOrderingComposer(
            $db: $db,
            $table: $db.importBatches,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }
}

class $$ContactsTableAnnotationComposer
    extends Composer<_$AppDatabase, $ContactsTable> {
  $$ContactsTableAnnotationComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  GeneratedColumn<int> get id =>
      $composableBuilder(column: $table.id, builder: (column) => column);

  GeneratedColumn<String> get name =>
      $composableBuilder(column: $table.name, builder: (column) => column);

  GeneratedColumn<String> get phoneRaw =>
      $composableBuilder(column: $table.phoneRaw, builder: (column) => column);

  GeneratedColumn<String> get phoneE164 =>
      $composableBuilder(column: $table.phoneE164, builder: (column) => column);

  GeneratedColumn<String> get note =>
      $composableBuilder(column: $table.note, builder: (column) => column);

  GeneratedColumn<String> get category =>
      $composableBuilder(column: $table.category, builder: (column) => column);

  GeneratedColumn<String> get tier =>
      $composableBuilder(column: $table.tier, builder: (column) => column);

  GeneratedColumn<bool> get callConfirmed => $composableBuilder(
    column: $table.callConfirmed,
    builder: (column) => column,
  );

  GeneratedColumn<DateTime> get confirmedAt => $composableBuilder(
    column: $table.confirmedAt,
    builder: (column) => column,
  );

  GeneratedColumn<bool> get isPinned =>
      $composableBuilder(column: $table.isPinned, builder: (column) => column);

  GeneratedColumn<bool> get isEmergency => $composableBuilder(
    column: $table.isEmergency,
    builder: (column) => column,
  );

  GeneratedColumn<bool> get hasWhatsapp => $composableBuilder(
    column: $table.hasWhatsapp,
    builder: (column) => column,
  );

  GeneratedColumn<DateTime> get lastCalledAt => $composableBuilder(
    column: $table.lastCalledAt,
    builder: (column) => column,
  );

  GeneratedColumn<int> get callCount =>
      $composableBuilder(column: $table.callCount, builder: (column) => column);

  GeneratedColumn<DateTime> get createdAt =>
      $composableBuilder(column: $table.createdAt, builder: (column) => column);

  $$TripsTableAnnotationComposer get tripId {
    final $$TripsTableAnnotationComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.tripId,
      referencedTable: $db.trips,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$TripsTableAnnotationComposer(
            $db: $db,
            $table: $db.trips,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }

  $$StopsTableAnnotationComposer get stopId {
    final $$StopsTableAnnotationComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.stopId,
      referencedTable: $db.stops,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$StopsTableAnnotationComposer(
            $db: $db,
            $table: $db.stops,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }

  $$ImportBatchesTableAnnotationComposer get importBatchId {
    final $$ImportBatchesTableAnnotationComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.importBatchId,
      referencedTable: $db.importBatches,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$ImportBatchesTableAnnotationComposer(
            $db: $db,
            $table: $db.importBatches,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }

  Expression<T> callLogsRefs<T extends Object>(
    Expression<T> Function($$CallLogsTableAnnotationComposer a) f,
  ) {
    final $$CallLogsTableAnnotationComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.id,
      referencedTable: $db.callLogs,
      getReferencedColumn: (t) => t.contactId,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$CallLogsTableAnnotationComposer(
            $db: $db,
            $table: $db.callLogs,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return f(composer);
  }

  Expression<T> checklistItemsRefs<T extends Object>(
    Expression<T> Function($$ChecklistItemsTableAnnotationComposer a) f,
  ) {
    final $$ChecklistItemsTableAnnotationComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.id,
      referencedTable: $db.checklistItems,
      getReferencedColumn: (t) => t.contactId,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$ChecklistItemsTableAnnotationComposer(
            $db: $db,
            $table: $db.checklistItems,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return f(composer);
  }
}

class $$ContactsTableTableManager
    extends
        RootTableManager<
          _$AppDatabase,
          $ContactsTable,
          Contact,
          $$ContactsTableFilterComposer,
          $$ContactsTableOrderingComposer,
          $$ContactsTableAnnotationComposer,
          $$ContactsTableCreateCompanionBuilder,
          $$ContactsTableUpdateCompanionBuilder,
          (Contact, $$ContactsTableReferences),
          Contact,
          PrefetchHooks Function({
            bool tripId,
            bool stopId,
            bool importBatchId,
            bool callLogsRefs,
            bool checklistItemsRefs,
          })
        > {
  $$ContactsTableTableManager(_$AppDatabase db, $ContactsTable table)
    : super(
        TableManagerState(
          db: db,
          table: table,
          createFilteringComposer: () =>
              $$ContactsTableFilterComposer($db: db, $table: table),
          createOrderingComposer: () =>
              $$ContactsTableOrderingComposer($db: db, $table: table),
          createComputedFieldComposer: () =>
              $$ContactsTableAnnotationComposer($db: db, $table: table),
          updateCompanionCallback:
              ({
                Value<int> id = const Value.absent(),
                Value<int?> tripId = const Value.absent(),
                Value<int?> stopId = const Value.absent(),
                Value<String> name = const Value.absent(),
                Value<String> phoneRaw = const Value.absent(),
                Value<String?> phoneE164 = const Value.absent(),
                Value<String?> note = const Value.absent(),
                Value<String> category = const Value.absent(),
                Value<String> tier = const Value.absent(),
                Value<bool> callConfirmed = const Value.absent(),
                Value<DateTime?> confirmedAt = const Value.absent(),
                Value<bool> isPinned = const Value.absent(),
                Value<bool> isEmergency = const Value.absent(),
                Value<bool> hasWhatsapp = const Value.absent(),
                Value<DateTime?> lastCalledAt = const Value.absent(),
                Value<int> callCount = const Value.absent(),
                Value<int?> importBatchId = const Value.absent(),
                Value<DateTime> createdAt = const Value.absent(),
              }) => ContactsCompanion(
                id: id,
                tripId: tripId,
                stopId: stopId,
                name: name,
                phoneRaw: phoneRaw,
                phoneE164: phoneE164,
                note: note,
                category: category,
                tier: tier,
                callConfirmed: callConfirmed,
                confirmedAt: confirmedAt,
                isPinned: isPinned,
                isEmergency: isEmergency,
                hasWhatsapp: hasWhatsapp,
                lastCalledAt: lastCalledAt,
                callCount: callCount,
                importBatchId: importBatchId,
                createdAt: createdAt,
              ),
          createCompanionCallback:
              ({
                Value<int> id = const Value.absent(),
                Value<int?> tripId = const Value.absent(),
                Value<int?> stopId = const Value.absent(),
                required String name,
                required String phoneRaw,
                Value<String?> phoneE164 = const Value.absent(),
                Value<String?> note = const Value.absent(),
                Value<String> category = const Value.absent(),
                Value<String> tier = const Value.absent(),
                Value<bool> callConfirmed = const Value.absent(),
                Value<DateTime?> confirmedAt = const Value.absent(),
                Value<bool> isPinned = const Value.absent(),
                Value<bool> isEmergency = const Value.absent(),
                Value<bool> hasWhatsapp = const Value.absent(),
                Value<DateTime?> lastCalledAt = const Value.absent(),
                Value<int> callCount = const Value.absent(),
                Value<int?> importBatchId = const Value.absent(),
                Value<DateTime> createdAt = const Value.absent(),
              }) => ContactsCompanion.insert(
                id: id,
                tripId: tripId,
                stopId: stopId,
                name: name,
                phoneRaw: phoneRaw,
                phoneE164: phoneE164,
                note: note,
                category: category,
                tier: tier,
                callConfirmed: callConfirmed,
                confirmedAt: confirmedAt,
                isPinned: isPinned,
                isEmergency: isEmergency,
                hasWhatsapp: hasWhatsapp,
                lastCalledAt: lastCalledAt,
                callCount: callCount,
                importBatchId: importBatchId,
                createdAt: createdAt,
              ),
          withReferenceMapper: (p0) => p0
              .map(
                (e) => (
                  e.readTable<$ContactsTable, Contact>(table),
                  $$ContactsTableReferences(db, table, e),
                ),
              )
              .toList(),
          prefetchHooksCallback:
              ({
                tripId = false,
                stopId = false,
                importBatchId = false,
                callLogsRefs = false,
                checklistItemsRefs = false,
              }) {
                return PrefetchHooks(
                  db: db,
                  explicitlyWatchedTables: [
                    if (callLogsRefs) db.callLogs,
                    if (checklistItemsRefs) db.checklistItems,
                  ],
                  addJoins:
                      <
                        T extends TableManagerState<
                          dynamic,
                          dynamic,
                          dynamic,
                          dynamic,
                          dynamic,
                          dynamic,
                          dynamic,
                          dynamic,
                          dynamic,
                          dynamic,
                          dynamic
                        >
                      >(state) {
                        if (tripId) {
                          state =
                              state.withJoin(
                                    currentTable: table,
                                    currentColumn: table.tripId,
                                    referencedTable: $$ContactsTableReferences
                                        ._tripIdTable(db),
                                    referencedColumn: $$ContactsTableReferences
                                        ._tripIdTable(db)
                                        .id,
                                  )
                                  as T;
                        }
                        if (stopId) {
                          state =
                              state.withJoin(
                                    currentTable: table,
                                    currentColumn: table.stopId,
                                    referencedTable: $$ContactsTableReferences
                                        ._stopIdTable(db),
                                    referencedColumn: $$ContactsTableReferences
                                        ._stopIdTable(db)
                                        .id,
                                  )
                                  as T;
                        }
                        if (importBatchId) {
                          state =
                              state.withJoin(
                                    currentTable: table,
                                    currentColumn: table.importBatchId,
                                    referencedTable: $$ContactsTableReferences
                                        ._importBatchIdTable(db),
                                    referencedColumn: $$ContactsTableReferences
                                        ._importBatchIdTable(db)
                                        .id,
                                  )
                                  as T;
                        }

                        return state;
                      },
                  getPrefetchedDataCallback: (items) async {
                    return [
                      if (callLogsRefs)
                        await $_getPrefetchedData<
                          Contact,
                          $ContactsTable,
                          CallLog
                        >(
                          currentTable: table,
                          referencedTable: $$ContactsTableReferences
                              ._callLogsRefsTable(db),
                          managerFromTypedResult: (p0) =>
                              $$ContactsTableReferences(
                                db,
                                table,
                                p0,
                              ).callLogsRefs,
                          referencedItemsForCurrentItem:
                              (item, referencedItems) => referencedItems.where(
                                (e) => e.contactId == item.id,
                              ),
                          typedResults: items,
                        ),
                      if (checklistItemsRefs)
                        await $_getPrefetchedData<
                          Contact,
                          $ContactsTable,
                          ChecklistItem
                        >(
                          currentTable: table,
                          referencedTable: $$ContactsTableReferences
                              ._checklistItemsRefsTable(db),
                          managerFromTypedResult: (p0) =>
                              $$ContactsTableReferences(
                                db,
                                table,
                                p0,
                              ).checklistItemsRefs,
                          referencedItemsForCurrentItem:
                              (item, referencedItems) => referencedItems.where(
                                (e) => e.contactId == item.id,
                              ),
                          typedResults: items,
                        ),
                    ];
                  },
                );
              },
        ),
      );
}

typedef $$ContactsTableProcessedTableManager =
    ProcessedTableManager<
      _$AppDatabase,
      $ContactsTable,
      Contact,
      $$ContactsTableFilterComposer,
      $$ContactsTableOrderingComposer,
      $$ContactsTableAnnotationComposer,
      $$ContactsTableCreateCompanionBuilder,
      $$ContactsTableUpdateCompanionBuilder,
      (Contact, $$ContactsTableReferences),
      Contact,
      PrefetchHooks Function({
        bool tripId,
        bool stopId,
        bool importBatchId,
        bool callLogsRefs,
        bool checklistItemsRefs,
      })
    >;
typedef $$CallLogsTableCreateCompanionBuilder =
    CallLogsCompanion Function({
      Value<int> id,
      required int contactId,
      Value<int?> tripId,
      required String action,
      Value<DateTime> occurredAt,
    });
typedef $$CallLogsTableUpdateCompanionBuilder =
    CallLogsCompanion Function({
      Value<int> id,
      Value<int> contactId,
      Value<int?> tripId,
      Value<String> action,
      Value<DateTime> occurredAt,
    });

final class $$CallLogsTableReferences
    extends BaseReferences<_$AppDatabase, $CallLogsTable, CallLog> {
  $$CallLogsTableReferences(super.$_db, super.$_table, super.$_typedResult);

  static $ContactsTable _contactIdTable(_$AppDatabase db) =>
      db.contacts.createAlias('call_logs__contact_id__contacts__id');

  $$ContactsTableProcessedTableManager get contactId {
    final $_column = $_itemColumn<int>('contact_id')!;

    final manager = $$ContactsTableTableManager(
      $_db,
      $_db.contacts,
    ).filter((f) => f.id.sqlEquals($_column));
    final item = $_typedResult.readTableOrNull(_contactIdTable($_db));
    if (item == null) return manager;
    return ProcessedTableManager(
      manager.$state.copyWith(prefetchedData: [item]),
    );
  }

  static $TripsTable _tripIdTable(_$AppDatabase db) =>
      db.trips.createAlias('call_logs__trip_id__trips__id');

  $$TripsTableProcessedTableManager? get tripId {
    final $_column = $_itemColumn<int>('trip_id');
    if ($_column == null) return null;
    final manager = $$TripsTableTableManager(
      $_db,
      $_db.trips,
    ).filter((f) => f.id.sqlEquals($_column));
    final item = $_typedResult.readTableOrNull(_tripIdTable($_db));
    if (item == null) return manager;
    return ProcessedTableManager(
      manager.$state.copyWith(prefetchedData: [item]),
    );
  }
}

class $$CallLogsTableFilterComposer
    extends Composer<_$AppDatabase, $CallLogsTable> {
  $$CallLogsTableFilterComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  ColumnFilters<int> get id => $composableBuilder(
    column: $table.id,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<String> get action => $composableBuilder(
    column: $table.action,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<DateTime> get occurredAt => $composableBuilder(
    column: $table.occurredAt,
    builder: (column) => ColumnFilters(column),
  );

  $$ContactsTableFilterComposer get contactId {
    final $$ContactsTableFilterComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.contactId,
      referencedTable: $db.contacts,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$ContactsTableFilterComposer(
            $db: $db,
            $table: $db.contacts,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }

  $$TripsTableFilterComposer get tripId {
    final $$TripsTableFilterComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.tripId,
      referencedTable: $db.trips,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$TripsTableFilterComposer(
            $db: $db,
            $table: $db.trips,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }
}

class $$CallLogsTableOrderingComposer
    extends Composer<_$AppDatabase, $CallLogsTable> {
  $$CallLogsTableOrderingComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  ColumnOrderings<int> get id => $composableBuilder(
    column: $table.id,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<String> get action => $composableBuilder(
    column: $table.action,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<DateTime> get occurredAt => $composableBuilder(
    column: $table.occurredAt,
    builder: (column) => ColumnOrderings(column),
  );

  $$ContactsTableOrderingComposer get contactId {
    final $$ContactsTableOrderingComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.contactId,
      referencedTable: $db.contacts,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$ContactsTableOrderingComposer(
            $db: $db,
            $table: $db.contacts,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }

  $$TripsTableOrderingComposer get tripId {
    final $$TripsTableOrderingComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.tripId,
      referencedTable: $db.trips,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$TripsTableOrderingComposer(
            $db: $db,
            $table: $db.trips,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }
}

class $$CallLogsTableAnnotationComposer
    extends Composer<_$AppDatabase, $CallLogsTable> {
  $$CallLogsTableAnnotationComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  GeneratedColumn<int> get id =>
      $composableBuilder(column: $table.id, builder: (column) => column);

  GeneratedColumn<String> get action =>
      $composableBuilder(column: $table.action, builder: (column) => column);

  GeneratedColumn<DateTime> get occurredAt => $composableBuilder(
    column: $table.occurredAt,
    builder: (column) => column,
  );

  $$ContactsTableAnnotationComposer get contactId {
    final $$ContactsTableAnnotationComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.contactId,
      referencedTable: $db.contacts,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$ContactsTableAnnotationComposer(
            $db: $db,
            $table: $db.contacts,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }

  $$TripsTableAnnotationComposer get tripId {
    final $$TripsTableAnnotationComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.tripId,
      referencedTable: $db.trips,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$TripsTableAnnotationComposer(
            $db: $db,
            $table: $db.trips,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }
}

class $$CallLogsTableTableManager
    extends
        RootTableManager<
          _$AppDatabase,
          $CallLogsTable,
          CallLog,
          $$CallLogsTableFilterComposer,
          $$CallLogsTableOrderingComposer,
          $$CallLogsTableAnnotationComposer,
          $$CallLogsTableCreateCompanionBuilder,
          $$CallLogsTableUpdateCompanionBuilder,
          (CallLog, $$CallLogsTableReferences),
          CallLog,
          PrefetchHooks Function({bool contactId, bool tripId})
        > {
  $$CallLogsTableTableManager(_$AppDatabase db, $CallLogsTable table)
    : super(
        TableManagerState(
          db: db,
          table: table,
          createFilteringComposer: () =>
              $$CallLogsTableFilterComposer($db: db, $table: table),
          createOrderingComposer: () =>
              $$CallLogsTableOrderingComposer($db: db, $table: table),
          createComputedFieldComposer: () =>
              $$CallLogsTableAnnotationComposer($db: db, $table: table),
          updateCompanionCallback:
              ({
                Value<int> id = const Value.absent(),
                Value<int> contactId = const Value.absent(),
                Value<int?> tripId = const Value.absent(),
                Value<String> action = const Value.absent(),
                Value<DateTime> occurredAt = const Value.absent(),
              }) => CallLogsCompanion(
                id: id,
                contactId: contactId,
                tripId: tripId,
                action: action,
                occurredAt: occurredAt,
              ),
          createCompanionCallback:
              ({
                Value<int> id = const Value.absent(),
                required int contactId,
                Value<int?> tripId = const Value.absent(),
                required String action,
                Value<DateTime> occurredAt = const Value.absent(),
              }) => CallLogsCompanion.insert(
                id: id,
                contactId: contactId,
                tripId: tripId,
                action: action,
                occurredAt: occurredAt,
              ),
          withReferenceMapper: (p0) => p0
              .map(
                (e) => (
                  e.readTable<$CallLogsTable, CallLog>(table),
                  $$CallLogsTableReferences(db, table, e),
                ),
              )
              .toList(),
          prefetchHooksCallback: ({contactId = false, tripId = false}) {
            return PrefetchHooks(
              db: db,
              explicitlyWatchedTables: [],
              addJoins:
                  <
                    T extends TableManagerState<
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic
                    >
                  >(state) {
                    if (contactId) {
                      state =
                          state.withJoin(
                                currentTable: table,
                                currentColumn: table.contactId,
                                referencedTable: $$CallLogsTableReferences
                                    ._contactIdTable(db),
                                referencedColumn: $$CallLogsTableReferences
                                    ._contactIdTable(db)
                                    .id,
                              )
                              as T;
                    }
                    if (tripId) {
                      state =
                          state.withJoin(
                                currentTable: table,
                                currentColumn: table.tripId,
                                referencedTable: $$CallLogsTableReferences
                                    ._tripIdTable(db),
                                referencedColumn: $$CallLogsTableReferences
                                    ._tripIdTable(db)
                                    .id,
                              )
                              as T;
                    }

                    return state;
                  },
              getPrefetchedDataCallback: (items) async {
                return [];
              },
            );
          },
        ),
      );
}

typedef $$CallLogsTableProcessedTableManager =
    ProcessedTableManager<
      _$AppDatabase,
      $CallLogsTable,
      CallLog,
      $$CallLogsTableFilterComposer,
      $$CallLogsTableOrderingComposer,
      $$CallLogsTableAnnotationComposer,
      $$CallLogsTableCreateCompanionBuilder,
      $$CallLogsTableUpdateCompanionBuilder,
      (CallLog, $$CallLogsTableReferences),
      CallLog,
      PrefetchHooks Function({bool contactId, bool tripId})
    >;
typedef $$EmergencyHelplinesTableCreateCompanionBuilder =
    EmergencyHelplinesCompanion Function({
      Value<int> id,
      required String countryCode,
      Value<String?> regionCode,
      required String serviceType,
      required String label,
      required String number,
      required String sourceNote,
      Value<String?> sourceUrl,
      Value<bool> needsVerification,
      Value<String> tier,
    });
typedef $$EmergencyHelplinesTableUpdateCompanionBuilder =
    EmergencyHelplinesCompanion Function({
      Value<int> id,
      Value<String> countryCode,
      Value<String?> regionCode,
      Value<String> serviceType,
      Value<String> label,
      Value<String> number,
      Value<String> sourceNote,
      Value<String?> sourceUrl,
      Value<bool> needsVerification,
      Value<String> tier,
    });

class $$EmergencyHelplinesTableFilterComposer
    extends Composer<_$AppDatabase, $EmergencyHelplinesTable> {
  $$EmergencyHelplinesTableFilterComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  ColumnFilters<int> get id => $composableBuilder(
    column: $table.id,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<String> get countryCode => $composableBuilder(
    column: $table.countryCode,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<String> get regionCode => $composableBuilder(
    column: $table.regionCode,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<String> get serviceType => $composableBuilder(
    column: $table.serviceType,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<String> get label => $composableBuilder(
    column: $table.label,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<String> get number => $composableBuilder(
    column: $table.number,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<String> get sourceNote => $composableBuilder(
    column: $table.sourceNote,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<String> get sourceUrl => $composableBuilder(
    column: $table.sourceUrl,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<bool> get needsVerification => $composableBuilder(
    column: $table.needsVerification,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<String> get tier => $composableBuilder(
    column: $table.tier,
    builder: (column) => ColumnFilters(column),
  );
}

class $$EmergencyHelplinesTableOrderingComposer
    extends Composer<_$AppDatabase, $EmergencyHelplinesTable> {
  $$EmergencyHelplinesTableOrderingComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  ColumnOrderings<int> get id => $composableBuilder(
    column: $table.id,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<String> get countryCode => $composableBuilder(
    column: $table.countryCode,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<String> get regionCode => $composableBuilder(
    column: $table.regionCode,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<String> get serviceType => $composableBuilder(
    column: $table.serviceType,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<String> get label => $composableBuilder(
    column: $table.label,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<String> get number => $composableBuilder(
    column: $table.number,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<String> get sourceNote => $composableBuilder(
    column: $table.sourceNote,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<String> get sourceUrl => $composableBuilder(
    column: $table.sourceUrl,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<bool> get needsVerification => $composableBuilder(
    column: $table.needsVerification,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<String> get tier => $composableBuilder(
    column: $table.tier,
    builder: (column) => ColumnOrderings(column),
  );
}

class $$EmergencyHelplinesTableAnnotationComposer
    extends Composer<_$AppDatabase, $EmergencyHelplinesTable> {
  $$EmergencyHelplinesTableAnnotationComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  GeneratedColumn<int> get id =>
      $composableBuilder(column: $table.id, builder: (column) => column);

  GeneratedColumn<String> get countryCode => $composableBuilder(
    column: $table.countryCode,
    builder: (column) => column,
  );

  GeneratedColumn<String> get regionCode => $composableBuilder(
    column: $table.regionCode,
    builder: (column) => column,
  );

  GeneratedColumn<String> get serviceType => $composableBuilder(
    column: $table.serviceType,
    builder: (column) => column,
  );

  GeneratedColumn<String> get label =>
      $composableBuilder(column: $table.label, builder: (column) => column);

  GeneratedColumn<String> get number =>
      $composableBuilder(column: $table.number, builder: (column) => column);

  GeneratedColumn<String> get sourceNote => $composableBuilder(
    column: $table.sourceNote,
    builder: (column) => column,
  );

  GeneratedColumn<String> get sourceUrl =>
      $composableBuilder(column: $table.sourceUrl, builder: (column) => column);

  GeneratedColumn<bool> get needsVerification => $composableBuilder(
    column: $table.needsVerification,
    builder: (column) => column,
  );

  GeneratedColumn<String> get tier =>
      $composableBuilder(column: $table.tier, builder: (column) => column);
}

class $$EmergencyHelplinesTableTableManager
    extends
        RootTableManager<
          _$AppDatabase,
          $EmergencyHelplinesTable,
          EmergencyHelpline,
          $$EmergencyHelplinesTableFilterComposer,
          $$EmergencyHelplinesTableOrderingComposer,
          $$EmergencyHelplinesTableAnnotationComposer,
          $$EmergencyHelplinesTableCreateCompanionBuilder,
          $$EmergencyHelplinesTableUpdateCompanionBuilder,
          (
            EmergencyHelpline,
            BaseReferences<
              _$AppDatabase,
              $EmergencyHelplinesTable,
              EmergencyHelpline
            >,
          ),
          EmergencyHelpline,
          PrefetchHooks Function()
        > {
  $$EmergencyHelplinesTableTableManager(
    _$AppDatabase db,
    $EmergencyHelplinesTable table,
  ) : super(
        TableManagerState(
          db: db,
          table: table,
          createFilteringComposer: () =>
              $$EmergencyHelplinesTableFilterComposer($db: db, $table: table),
          createOrderingComposer: () =>
              $$EmergencyHelplinesTableOrderingComposer($db: db, $table: table),
          createComputedFieldComposer: () =>
              $$EmergencyHelplinesTableAnnotationComposer(
                $db: db,
                $table: table,
              ),
          updateCompanionCallback:
              ({
                Value<int> id = const Value.absent(),
                Value<String> countryCode = const Value.absent(),
                Value<String?> regionCode = const Value.absent(),
                Value<String> serviceType = const Value.absent(),
                Value<String> label = const Value.absent(),
                Value<String> number = const Value.absent(),
                Value<String> sourceNote = const Value.absent(),
                Value<String?> sourceUrl = const Value.absent(),
                Value<bool> needsVerification = const Value.absent(),
                Value<String> tier = const Value.absent(),
              }) => EmergencyHelplinesCompanion(
                id: id,
                countryCode: countryCode,
                regionCode: regionCode,
                serviceType: serviceType,
                label: label,
                number: number,
                sourceNote: sourceNote,
                sourceUrl: sourceUrl,
                needsVerification: needsVerification,
                tier: tier,
              ),
          createCompanionCallback:
              ({
                Value<int> id = const Value.absent(),
                required String countryCode,
                Value<String?> regionCode = const Value.absent(),
                required String serviceType,
                required String label,
                required String number,
                required String sourceNote,
                Value<String?> sourceUrl = const Value.absent(),
                Value<bool> needsVerification = const Value.absent(),
                Value<String> tier = const Value.absent(),
              }) => EmergencyHelplinesCompanion.insert(
                id: id,
                countryCode: countryCode,
                regionCode: regionCode,
                serviceType: serviceType,
                label: label,
                number: number,
                sourceNote: sourceNote,
                sourceUrl: sourceUrl,
                needsVerification: needsVerification,
                tier: tier,
              ),
          withReferenceMapper: (p0) => p0
              .map(
                (e) => (
                  e.readTable<$EmergencyHelplinesTable, EmergencyHelpline>(
                    table,
                  ),
                  BaseReferences<
                    _$AppDatabase,
                    $EmergencyHelplinesTable,
                    EmergencyHelpline
                  >(db, table, e),
                ),
              )
              .toList(),
          prefetchHooksCallback: null,
        ),
      );
}

typedef $$EmergencyHelplinesTableProcessedTableManager =
    ProcessedTableManager<
      _$AppDatabase,
      $EmergencyHelplinesTable,
      EmergencyHelpline,
      $$EmergencyHelplinesTableFilterComposer,
      $$EmergencyHelplinesTableOrderingComposer,
      $$EmergencyHelplinesTableAnnotationComposer,
      $$EmergencyHelplinesTableCreateCompanionBuilder,
      $$EmergencyHelplinesTableUpdateCompanionBuilder,
      (
        EmergencyHelpline,
        BaseReferences<
          _$AppDatabase,
          $EmergencyHelplinesTable,
          EmergencyHelpline
        >,
      ),
      EmergencyHelpline,
      PrefetchHooks Function()
    >;
typedef $$ChecklistItemsTableCreateCompanionBuilder =
    ChecklistItemsCompanion Function({
      Value<int> id,
      required int tripId,
      Value<int?> stopId,
      required String label,
      Value<String?> quantity,
      Value<String> sourceTags,
      Value<bool> isDone,
      Value<bool> isBlocking,
      Value<int?> contactId,
      Value<bool> isGenerated,
      Value<String?> generatorKey,
      Value<bool> isUserEdited,
      Value<int> sortOrder,
    });
typedef $$ChecklistItemsTableUpdateCompanionBuilder =
    ChecklistItemsCompanion Function({
      Value<int> id,
      Value<int> tripId,
      Value<int?> stopId,
      Value<String> label,
      Value<String?> quantity,
      Value<String> sourceTags,
      Value<bool> isDone,
      Value<bool> isBlocking,
      Value<int?> contactId,
      Value<bool> isGenerated,
      Value<String?> generatorKey,
      Value<bool> isUserEdited,
      Value<int> sortOrder,
    });

final class $$ChecklistItemsTableReferences
    extends BaseReferences<_$AppDatabase, $ChecklistItemsTable, ChecklistItem> {
  $$ChecklistItemsTableReferences(
    super.$_db,
    super.$_table,
    super.$_typedResult,
  );

  static $TripsTable _tripIdTable(_$AppDatabase db) =>
      db.trips.createAlias('checklist_items__trip_id__trips__id');

  $$TripsTableProcessedTableManager get tripId {
    final $_column = $_itemColumn<int>('trip_id')!;

    final manager = $$TripsTableTableManager(
      $_db,
      $_db.trips,
    ).filter((f) => f.id.sqlEquals($_column));
    final item = $_typedResult.readTableOrNull(_tripIdTable($_db));
    if (item == null) return manager;
    return ProcessedTableManager(
      manager.$state.copyWith(prefetchedData: [item]),
    );
  }

  static $StopsTable _stopIdTable(_$AppDatabase db) =>
      db.stops.createAlias('checklist_items__stop_id__stops__id');

  $$StopsTableProcessedTableManager? get stopId {
    final $_column = $_itemColumn<int>('stop_id');
    if ($_column == null) return null;
    final manager = $$StopsTableTableManager(
      $_db,
      $_db.stops,
    ).filter((f) => f.id.sqlEquals($_column));
    final item = $_typedResult.readTableOrNull(_stopIdTable($_db));
    if (item == null) return manager;
    return ProcessedTableManager(
      manager.$state.copyWith(prefetchedData: [item]),
    );
  }

  static $ContactsTable _contactIdTable(_$AppDatabase db) =>
      db.contacts.createAlias('checklist_items__contact_id__contacts__id');

  $$ContactsTableProcessedTableManager? get contactId {
    final $_column = $_itemColumn<int>('contact_id');
    if ($_column == null) return null;
    final manager = $$ContactsTableTableManager(
      $_db,
      $_db.contacts,
    ).filter((f) => f.id.sqlEquals($_column));
    final item = $_typedResult.readTableOrNull(_contactIdTable($_db));
    if (item == null) return manager;
    return ProcessedTableManager(
      manager.$state.copyWith(prefetchedData: [item]),
    );
  }
}

class $$ChecklistItemsTableFilterComposer
    extends Composer<_$AppDatabase, $ChecklistItemsTable> {
  $$ChecklistItemsTableFilterComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  ColumnFilters<int> get id => $composableBuilder(
    column: $table.id,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<String> get label => $composableBuilder(
    column: $table.label,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<String> get quantity => $composableBuilder(
    column: $table.quantity,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<String> get sourceTags => $composableBuilder(
    column: $table.sourceTags,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<bool> get isDone => $composableBuilder(
    column: $table.isDone,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<bool> get isBlocking => $composableBuilder(
    column: $table.isBlocking,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<bool> get isGenerated => $composableBuilder(
    column: $table.isGenerated,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<String> get generatorKey => $composableBuilder(
    column: $table.generatorKey,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<bool> get isUserEdited => $composableBuilder(
    column: $table.isUserEdited,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<int> get sortOrder => $composableBuilder(
    column: $table.sortOrder,
    builder: (column) => ColumnFilters(column),
  );

  $$TripsTableFilterComposer get tripId {
    final $$TripsTableFilterComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.tripId,
      referencedTable: $db.trips,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$TripsTableFilterComposer(
            $db: $db,
            $table: $db.trips,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }

  $$StopsTableFilterComposer get stopId {
    final $$StopsTableFilterComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.stopId,
      referencedTable: $db.stops,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$StopsTableFilterComposer(
            $db: $db,
            $table: $db.stops,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }

  $$ContactsTableFilterComposer get contactId {
    final $$ContactsTableFilterComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.contactId,
      referencedTable: $db.contacts,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$ContactsTableFilterComposer(
            $db: $db,
            $table: $db.contacts,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }
}

class $$ChecklistItemsTableOrderingComposer
    extends Composer<_$AppDatabase, $ChecklistItemsTable> {
  $$ChecklistItemsTableOrderingComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  ColumnOrderings<int> get id => $composableBuilder(
    column: $table.id,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<String> get label => $composableBuilder(
    column: $table.label,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<String> get quantity => $composableBuilder(
    column: $table.quantity,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<String> get sourceTags => $composableBuilder(
    column: $table.sourceTags,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<bool> get isDone => $composableBuilder(
    column: $table.isDone,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<bool> get isBlocking => $composableBuilder(
    column: $table.isBlocking,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<bool> get isGenerated => $composableBuilder(
    column: $table.isGenerated,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<String> get generatorKey => $composableBuilder(
    column: $table.generatorKey,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<bool> get isUserEdited => $composableBuilder(
    column: $table.isUserEdited,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<int> get sortOrder => $composableBuilder(
    column: $table.sortOrder,
    builder: (column) => ColumnOrderings(column),
  );

  $$TripsTableOrderingComposer get tripId {
    final $$TripsTableOrderingComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.tripId,
      referencedTable: $db.trips,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$TripsTableOrderingComposer(
            $db: $db,
            $table: $db.trips,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }

  $$StopsTableOrderingComposer get stopId {
    final $$StopsTableOrderingComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.stopId,
      referencedTable: $db.stops,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$StopsTableOrderingComposer(
            $db: $db,
            $table: $db.stops,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }

  $$ContactsTableOrderingComposer get contactId {
    final $$ContactsTableOrderingComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.contactId,
      referencedTable: $db.contacts,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$ContactsTableOrderingComposer(
            $db: $db,
            $table: $db.contacts,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }
}

class $$ChecklistItemsTableAnnotationComposer
    extends Composer<_$AppDatabase, $ChecklistItemsTable> {
  $$ChecklistItemsTableAnnotationComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  GeneratedColumn<int> get id =>
      $composableBuilder(column: $table.id, builder: (column) => column);

  GeneratedColumn<String> get label =>
      $composableBuilder(column: $table.label, builder: (column) => column);

  GeneratedColumn<String> get quantity =>
      $composableBuilder(column: $table.quantity, builder: (column) => column);

  GeneratedColumn<String> get sourceTags => $composableBuilder(
    column: $table.sourceTags,
    builder: (column) => column,
  );

  GeneratedColumn<bool> get isDone =>
      $composableBuilder(column: $table.isDone, builder: (column) => column);

  GeneratedColumn<bool> get isBlocking => $composableBuilder(
    column: $table.isBlocking,
    builder: (column) => column,
  );

  GeneratedColumn<bool> get isGenerated => $composableBuilder(
    column: $table.isGenerated,
    builder: (column) => column,
  );

  GeneratedColumn<String> get generatorKey => $composableBuilder(
    column: $table.generatorKey,
    builder: (column) => column,
  );

  GeneratedColumn<bool> get isUserEdited => $composableBuilder(
    column: $table.isUserEdited,
    builder: (column) => column,
  );

  GeneratedColumn<int> get sortOrder =>
      $composableBuilder(column: $table.sortOrder, builder: (column) => column);

  $$TripsTableAnnotationComposer get tripId {
    final $$TripsTableAnnotationComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.tripId,
      referencedTable: $db.trips,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$TripsTableAnnotationComposer(
            $db: $db,
            $table: $db.trips,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }

  $$StopsTableAnnotationComposer get stopId {
    final $$StopsTableAnnotationComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.stopId,
      referencedTable: $db.stops,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$StopsTableAnnotationComposer(
            $db: $db,
            $table: $db.stops,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }

  $$ContactsTableAnnotationComposer get contactId {
    final $$ContactsTableAnnotationComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.contactId,
      referencedTable: $db.contacts,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$ContactsTableAnnotationComposer(
            $db: $db,
            $table: $db.contacts,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }
}

class $$ChecklistItemsTableTableManager
    extends
        RootTableManager<
          _$AppDatabase,
          $ChecklistItemsTable,
          ChecklistItem,
          $$ChecklistItemsTableFilterComposer,
          $$ChecklistItemsTableOrderingComposer,
          $$ChecklistItemsTableAnnotationComposer,
          $$ChecklistItemsTableCreateCompanionBuilder,
          $$ChecklistItemsTableUpdateCompanionBuilder,
          (ChecklistItem, $$ChecklistItemsTableReferences),
          ChecklistItem,
          PrefetchHooks Function({bool tripId, bool stopId, bool contactId})
        > {
  $$ChecklistItemsTableTableManager(
    _$AppDatabase db,
    $ChecklistItemsTable table,
  ) : super(
        TableManagerState(
          db: db,
          table: table,
          createFilteringComposer: () =>
              $$ChecklistItemsTableFilterComposer($db: db, $table: table),
          createOrderingComposer: () =>
              $$ChecklistItemsTableOrderingComposer($db: db, $table: table),
          createComputedFieldComposer: () =>
              $$ChecklistItemsTableAnnotationComposer($db: db, $table: table),
          updateCompanionCallback:
              ({
                Value<int> id = const Value.absent(),
                Value<int> tripId = const Value.absent(),
                Value<int?> stopId = const Value.absent(),
                Value<String> label = const Value.absent(),
                Value<String?> quantity = const Value.absent(),
                Value<String> sourceTags = const Value.absent(),
                Value<bool> isDone = const Value.absent(),
                Value<bool> isBlocking = const Value.absent(),
                Value<int?> contactId = const Value.absent(),
                Value<bool> isGenerated = const Value.absent(),
                Value<String?> generatorKey = const Value.absent(),
                Value<bool> isUserEdited = const Value.absent(),
                Value<int> sortOrder = const Value.absent(),
              }) => ChecklistItemsCompanion(
                id: id,
                tripId: tripId,
                stopId: stopId,
                label: label,
                quantity: quantity,
                sourceTags: sourceTags,
                isDone: isDone,
                isBlocking: isBlocking,
                contactId: contactId,
                isGenerated: isGenerated,
                generatorKey: generatorKey,
                isUserEdited: isUserEdited,
                sortOrder: sortOrder,
              ),
          createCompanionCallback:
              ({
                Value<int> id = const Value.absent(),
                required int tripId,
                Value<int?> stopId = const Value.absent(),
                required String label,
                Value<String?> quantity = const Value.absent(),
                Value<String> sourceTags = const Value.absent(),
                Value<bool> isDone = const Value.absent(),
                Value<bool> isBlocking = const Value.absent(),
                Value<int?> contactId = const Value.absent(),
                Value<bool> isGenerated = const Value.absent(),
                Value<String?> generatorKey = const Value.absent(),
                Value<bool> isUserEdited = const Value.absent(),
                Value<int> sortOrder = const Value.absent(),
              }) => ChecklistItemsCompanion.insert(
                id: id,
                tripId: tripId,
                stopId: stopId,
                label: label,
                quantity: quantity,
                sourceTags: sourceTags,
                isDone: isDone,
                isBlocking: isBlocking,
                contactId: contactId,
                isGenerated: isGenerated,
                generatorKey: generatorKey,
                isUserEdited: isUserEdited,
                sortOrder: sortOrder,
              ),
          withReferenceMapper: (p0) => p0
              .map(
                (e) => (
                  e.readTable<$ChecklistItemsTable, ChecklistItem>(table),
                  $$ChecklistItemsTableReferences(db, table, e),
                ),
              )
              .toList(),
          prefetchHooksCallback:
              ({tripId = false, stopId = false, contactId = false}) {
                return PrefetchHooks(
                  db: db,
                  explicitlyWatchedTables: [],
                  addJoins:
                      <
                        T extends TableManagerState<
                          dynamic,
                          dynamic,
                          dynamic,
                          dynamic,
                          dynamic,
                          dynamic,
                          dynamic,
                          dynamic,
                          dynamic,
                          dynamic,
                          dynamic
                        >
                      >(state) {
                        if (tripId) {
                          state =
                              state.withJoin(
                                    currentTable: table,
                                    currentColumn: table.tripId,
                                    referencedTable:
                                        $$ChecklistItemsTableReferences
                                            ._tripIdTable(db),
                                    referencedColumn:
                                        $$ChecklistItemsTableReferences
                                            ._tripIdTable(db)
                                            .id,
                                  )
                                  as T;
                        }
                        if (stopId) {
                          state =
                              state.withJoin(
                                    currentTable: table,
                                    currentColumn: table.stopId,
                                    referencedTable:
                                        $$ChecklistItemsTableReferences
                                            ._stopIdTable(db),
                                    referencedColumn:
                                        $$ChecklistItemsTableReferences
                                            ._stopIdTable(db)
                                            .id,
                                  )
                                  as T;
                        }
                        if (contactId) {
                          state =
                              state.withJoin(
                                    currentTable: table,
                                    currentColumn: table.contactId,
                                    referencedTable:
                                        $$ChecklistItemsTableReferences
                                            ._contactIdTable(db),
                                    referencedColumn:
                                        $$ChecklistItemsTableReferences
                                            ._contactIdTable(db)
                                            .id,
                                  )
                                  as T;
                        }

                        return state;
                      },
                  getPrefetchedDataCallback: (items) async {
                    return [];
                  },
                );
              },
        ),
      );
}

typedef $$ChecklistItemsTableProcessedTableManager =
    ProcessedTableManager<
      _$AppDatabase,
      $ChecklistItemsTable,
      ChecklistItem,
      $$ChecklistItemsTableFilterComposer,
      $$ChecklistItemsTableOrderingComposer,
      $$ChecklistItemsTableAnnotationComposer,
      $$ChecklistItemsTableCreateCompanionBuilder,
      $$ChecklistItemsTableUpdateCompanionBuilder,
      (ChecklistItem, $$ChecklistItemsTableReferences),
      ChecklistItem,
      PrefetchHooks Function({bool tripId, bool stopId, bool contactId})
    >;
typedef $$WeatherSnapshotsTableCreateCompanionBuilder =
    WeatherSnapshotsCompanion Function({
      Value<int> id,
      required int stopId,
      required DateTime forDate,
      required String condition,
      Value<double?> tempMinC,
      Value<double?> tempMaxC,
      Value<double?> rainMm,
      required DateTime cachedAt,
    });
typedef $$WeatherSnapshotsTableUpdateCompanionBuilder =
    WeatherSnapshotsCompanion Function({
      Value<int> id,
      Value<int> stopId,
      Value<DateTime> forDate,
      Value<String> condition,
      Value<double?> tempMinC,
      Value<double?> tempMaxC,
      Value<double?> rainMm,
      Value<DateTime> cachedAt,
    });

final class $$WeatherSnapshotsTableReferences
    extends
        BaseReferences<_$AppDatabase, $WeatherSnapshotsTable, WeatherSnapshot> {
  $$WeatherSnapshotsTableReferences(
    super.$_db,
    super.$_table,
    super.$_typedResult,
  );

  static $StopsTable _stopIdTable(_$AppDatabase db) =>
      db.stops.createAlias('weather_snapshots__stop_id__stops__id');

  $$StopsTableProcessedTableManager get stopId {
    final $_column = $_itemColumn<int>('stop_id')!;

    final manager = $$StopsTableTableManager(
      $_db,
      $_db.stops,
    ).filter((f) => f.id.sqlEquals($_column));
    final item = $_typedResult.readTableOrNull(_stopIdTable($_db));
    if (item == null) return manager;
    return ProcessedTableManager(
      manager.$state.copyWith(prefetchedData: [item]),
    );
  }
}

class $$WeatherSnapshotsTableFilterComposer
    extends Composer<_$AppDatabase, $WeatherSnapshotsTable> {
  $$WeatherSnapshotsTableFilterComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  ColumnFilters<int> get id => $composableBuilder(
    column: $table.id,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<DateTime> get forDate => $composableBuilder(
    column: $table.forDate,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<String> get condition => $composableBuilder(
    column: $table.condition,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<double> get tempMinC => $composableBuilder(
    column: $table.tempMinC,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<double> get tempMaxC => $composableBuilder(
    column: $table.tempMaxC,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<double> get rainMm => $composableBuilder(
    column: $table.rainMm,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<DateTime> get cachedAt => $composableBuilder(
    column: $table.cachedAt,
    builder: (column) => ColumnFilters(column),
  );

  $$StopsTableFilterComposer get stopId {
    final $$StopsTableFilterComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.stopId,
      referencedTable: $db.stops,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$StopsTableFilterComposer(
            $db: $db,
            $table: $db.stops,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }
}

class $$WeatherSnapshotsTableOrderingComposer
    extends Composer<_$AppDatabase, $WeatherSnapshotsTable> {
  $$WeatherSnapshotsTableOrderingComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  ColumnOrderings<int> get id => $composableBuilder(
    column: $table.id,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<DateTime> get forDate => $composableBuilder(
    column: $table.forDate,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<String> get condition => $composableBuilder(
    column: $table.condition,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<double> get tempMinC => $composableBuilder(
    column: $table.tempMinC,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<double> get tempMaxC => $composableBuilder(
    column: $table.tempMaxC,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<double> get rainMm => $composableBuilder(
    column: $table.rainMm,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<DateTime> get cachedAt => $composableBuilder(
    column: $table.cachedAt,
    builder: (column) => ColumnOrderings(column),
  );

  $$StopsTableOrderingComposer get stopId {
    final $$StopsTableOrderingComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.stopId,
      referencedTable: $db.stops,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$StopsTableOrderingComposer(
            $db: $db,
            $table: $db.stops,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }
}

class $$WeatherSnapshotsTableAnnotationComposer
    extends Composer<_$AppDatabase, $WeatherSnapshotsTable> {
  $$WeatherSnapshotsTableAnnotationComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  GeneratedColumn<int> get id =>
      $composableBuilder(column: $table.id, builder: (column) => column);

  GeneratedColumn<DateTime> get forDate =>
      $composableBuilder(column: $table.forDate, builder: (column) => column);

  GeneratedColumn<String> get condition =>
      $composableBuilder(column: $table.condition, builder: (column) => column);

  GeneratedColumn<double> get tempMinC =>
      $composableBuilder(column: $table.tempMinC, builder: (column) => column);

  GeneratedColumn<double> get tempMaxC =>
      $composableBuilder(column: $table.tempMaxC, builder: (column) => column);

  GeneratedColumn<double> get rainMm =>
      $composableBuilder(column: $table.rainMm, builder: (column) => column);

  GeneratedColumn<DateTime> get cachedAt =>
      $composableBuilder(column: $table.cachedAt, builder: (column) => column);

  $$StopsTableAnnotationComposer get stopId {
    final $$StopsTableAnnotationComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.stopId,
      referencedTable: $db.stops,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$StopsTableAnnotationComposer(
            $db: $db,
            $table: $db.stops,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }
}

class $$WeatherSnapshotsTableTableManager
    extends
        RootTableManager<
          _$AppDatabase,
          $WeatherSnapshotsTable,
          WeatherSnapshot,
          $$WeatherSnapshotsTableFilterComposer,
          $$WeatherSnapshotsTableOrderingComposer,
          $$WeatherSnapshotsTableAnnotationComposer,
          $$WeatherSnapshotsTableCreateCompanionBuilder,
          $$WeatherSnapshotsTableUpdateCompanionBuilder,
          (WeatherSnapshot, $$WeatherSnapshotsTableReferences),
          WeatherSnapshot,
          PrefetchHooks Function({bool stopId})
        > {
  $$WeatherSnapshotsTableTableManager(
    _$AppDatabase db,
    $WeatherSnapshotsTable table,
  ) : super(
        TableManagerState(
          db: db,
          table: table,
          createFilteringComposer: () =>
              $$WeatherSnapshotsTableFilterComposer($db: db, $table: table),
          createOrderingComposer: () =>
              $$WeatherSnapshotsTableOrderingComposer($db: db, $table: table),
          createComputedFieldComposer: () =>
              $$WeatherSnapshotsTableAnnotationComposer($db: db, $table: table),
          updateCompanionCallback:
              ({
                Value<int> id = const Value.absent(),
                Value<int> stopId = const Value.absent(),
                Value<DateTime> forDate = const Value.absent(),
                Value<String> condition = const Value.absent(),
                Value<double?> tempMinC = const Value.absent(),
                Value<double?> tempMaxC = const Value.absent(),
                Value<double?> rainMm = const Value.absent(),
                Value<DateTime> cachedAt = const Value.absent(),
              }) => WeatherSnapshotsCompanion(
                id: id,
                stopId: stopId,
                forDate: forDate,
                condition: condition,
                tempMinC: tempMinC,
                tempMaxC: tempMaxC,
                rainMm: rainMm,
                cachedAt: cachedAt,
              ),
          createCompanionCallback:
              ({
                Value<int> id = const Value.absent(),
                required int stopId,
                required DateTime forDate,
                required String condition,
                Value<double?> tempMinC = const Value.absent(),
                Value<double?> tempMaxC = const Value.absent(),
                Value<double?> rainMm = const Value.absent(),
                required DateTime cachedAt,
              }) => WeatherSnapshotsCompanion.insert(
                id: id,
                stopId: stopId,
                forDate: forDate,
                condition: condition,
                tempMinC: tempMinC,
                tempMaxC: tempMaxC,
                rainMm: rainMm,
                cachedAt: cachedAt,
              ),
          withReferenceMapper: (p0) => p0
              .map(
                (e) => (
                  e.readTable<$WeatherSnapshotsTable, WeatherSnapshot>(table),
                  $$WeatherSnapshotsTableReferences(db, table, e),
                ),
              )
              .toList(),
          prefetchHooksCallback: ({stopId = false}) {
            return PrefetchHooks(
              db: db,
              explicitlyWatchedTables: [],
              addJoins:
                  <
                    T extends TableManagerState<
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic
                    >
                  >(state) {
                    if (stopId) {
                      state =
                          state.withJoin(
                                currentTable: table,
                                currentColumn: table.stopId,
                                referencedTable:
                                    $$WeatherSnapshotsTableReferences
                                        ._stopIdTable(db),
                                referencedColumn:
                                    $$WeatherSnapshotsTableReferences
                                        ._stopIdTable(db)
                                        .id,
                              )
                              as T;
                    }

                    return state;
                  },
              getPrefetchedDataCallback: (items) async {
                return [];
              },
            );
          },
        ),
      );
}

typedef $$WeatherSnapshotsTableProcessedTableManager =
    ProcessedTableManager<
      _$AppDatabase,
      $WeatherSnapshotsTable,
      WeatherSnapshot,
      $$WeatherSnapshotsTableFilterComposer,
      $$WeatherSnapshotsTableOrderingComposer,
      $$WeatherSnapshotsTableAnnotationComposer,
      $$WeatherSnapshotsTableCreateCompanionBuilder,
      $$WeatherSnapshotsTableUpdateCompanionBuilder,
      (WeatherSnapshot, $$WeatherSnapshotsTableReferences),
      WeatherSnapshot,
      PrefetchHooks Function({bool stopId})
    >;
typedef $$TravellersTableCreateCompanionBuilder =
    TravellersCompanion Function({
      Value<int> id,
      required int tripId,
      required String name,
      Value<bool> isSelf,
    });
typedef $$TravellersTableUpdateCompanionBuilder =
    TravellersCompanion Function({
      Value<int> id,
      Value<int> tripId,
      Value<String> name,
      Value<bool> isSelf,
    });

final class $$TravellersTableReferences
    extends BaseReferences<_$AppDatabase, $TravellersTable, Traveller> {
  $$TravellersTableReferences(super.$_db, super.$_table, super.$_typedResult);

  static $TripsTable _tripIdTable(_$AppDatabase db) =>
      db.trips.createAlias('travellers__trip_id__trips__id');

  $$TripsTableProcessedTableManager get tripId {
    final $_column = $_itemColumn<int>('trip_id')!;

    final manager = $$TripsTableTableManager(
      $_db,
      $_db.trips,
    ).filter((f) => f.id.sqlEquals($_column));
    final item = $_typedResult.readTableOrNull(_tripIdTable($_db));
    if (item == null) return manager;
    return ProcessedTableManager(
      manager.$state.copyWith(prefetchedData: [item]),
    );
  }

  static MultiTypedResultKey<$ExpensesTable, List<Expense>> _expensesRefsTable(
    _$AppDatabase db,
  ) => MultiTypedResultKey.fromTable(
    db.expenses,
    aliasName: 'travellers__id__expenses__paid_by_id',
  );

  $$ExpensesTableProcessedTableManager get expensesRefs {
    final manager = $$ExpensesTableTableManager(
      $_db,
      $_db.expenses,
    ).filter((f) => f.paidById.id.sqlEquals($_itemColumn<int>('id')!));

    final cache = $_typedResult.readTableOrNull(_expensesRefsTable($_db));
    return ProcessedTableManager(
      manager.$state.copyWith(prefetchedData: cache),
    );
  }

  static MultiTypedResultKey<$ExpenseSplitsTable, List<ExpenseSplit>>
  _expenseSplitsRefsTable(_$AppDatabase db) => MultiTypedResultKey.fromTable(
    db.expenseSplits,
    aliasName: 'travellers__id__expense_splits__traveller_id',
  );

  $$ExpenseSplitsTableProcessedTableManager get expenseSplitsRefs {
    final manager = $$ExpenseSplitsTableTableManager(
      $_db,
      $_db.expenseSplits,
    ).filter((f) => f.travellerId.id.sqlEquals($_itemColumn<int>('id')!));

    final cache = $_typedResult.readTableOrNull(_expenseSplitsRefsTable($_db));
    return ProcessedTableManager(
      manager.$state.copyWith(prefetchedData: cache),
    );
  }
}

class $$TravellersTableFilterComposer
    extends Composer<_$AppDatabase, $TravellersTable> {
  $$TravellersTableFilterComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  ColumnFilters<int> get id => $composableBuilder(
    column: $table.id,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<String> get name => $composableBuilder(
    column: $table.name,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<bool> get isSelf => $composableBuilder(
    column: $table.isSelf,
    builder: (column) => ColumnFilters(column),
  );

  $$TripsTableFilterComposer get tripId {
    final $$TripsTableFilterComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.tripId,
      referencedTable: $db.trips,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$TripsTableFilterComposer(
            $db: $db,
            $table: $db.trips,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }

  Expression<bool> expensesRefs(
    Expression<bool> Function($$ExpensesTableFilterComposer f) f,
  ) {
    final $$ExpensesTableFilterComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.id,
      referencedTable: $db.expenses,
      getReferencedColumn: (t) => t.paidById,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$ExpensesTableFilterComposer(
            $db: $db,
            $table: $db.expenses,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return f(composer);
  }

  Expression<bool> expenseSplitsRefs(
    Expression<bool> Function($$ExpenseSplitsTableFilterComposer f) f,
  ) {
    final $$ExpenseSplitsTableFilterComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.id,
      referencedTable: $db.expenseSplits,
      getReferencedColumn: (t) => t.travellerId,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$ExpenseSplitsTableFilterComposer(
            $db: $db,
            $table: $db.expenseSplits,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return f(composer);
  }
}

class $$TravellersTableOrderingComposer
    extends Composer<_$AppDatabase, $TravellersTable> {
  $$TravellersTableOrderingComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  ColumnOrderings<int> get id => $composableBuilder(
    column: $table.id,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<String> get name => $composableBuilder(
    column: $table.name,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<bool> get isSelf => $composableBuilder(
    column: $table.isSelf,
    builder: (column) => ColumnOrderings(column),
  );

  $$TripsTableOrderingComposer get tripId {
    final $$TripsTableOrderingComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.tripId,
      referencedTable: $db.trips,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$TripsTableOrderingComposer(
            $db: $db,
            $table: $db.trips,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }
}

class $$TravellersTableAnnotationComposer
    extends Composer<_$AppDatabase, $TravellersTable> {
  $$TravellersTableAnnotationComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  GeneratedColumn<int> get id =>
      $composableBuilder(column: $table.id, builder: (column) => column);

  GeneratedColumn<String> get name =>
      $composableBuilder(column: $table.name, builder: (column) => column);

  GeneratedColumn<bool> get isSelf =>
      $composableBuilder(column: $table.isSelf, builder: (column) => column);

  $$TripsTableAnnotationComposer get tripId {
    final $$TripsTableAnnotationComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.tripId,
      referencedTable: $db.trips,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$TripsTableAnnotationComposer(
            $db: $db,
            $table: $db.trips,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }

  Expression<T> expensesRefs<T extends Object>(
    Expression<T> Function($$ExpensesTableAnnotationComposer a) f,
  ) {
    final $$ExpensesTableAnnotationComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.id,
      referencedTable: $db.expenses,
      getReferencedColumn: (t) => t.paidById,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$ExpensesTableAnnotationComposer(
            $db: $db,
            $table: $db.expenses,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return f(composer);
  }

  Expression<T> expenseSplitsRefs<T extends Object>(
    Expression<T> Function($$ExpenseSplitsTableAnnotationComposer a) f,
  ) {
    final $$ExpenseSplitsTableAnnotationComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.id,
      referencedTable: $db.expenseSplits,
      getReferencedColumn: (t) => t.travellerId,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$ExpenseSplitsTableAnnotationComposer(
            $db: $db,
            $table: $db.expenseSplits,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return f(composer);
  }
}

class $$TravellersTableTableManager
    extends
        RootTableManager<
          _$AppDatabase,
          $TravellersTable,
          Traveller,
          $$TravellersTableFilterComposer,
          $$TravellersTableOrderingComposer,
          $$TravellersTableAnnotationComposer,
          $$TravellersTableCreateCompanionBuilder,
          $$TravellersTableUpdateCompanionBuilder,
          (Traveller, $$TravellersTableReferences),
          Traveller,
          PrefetchHooks Function({
            bool tripId,
            bool expensesRefs,
            bool expenseSplitsRefs,
          })
        > {
  $$TravellersTableTableManager(_$AppDatabase db, $TravellersTable table)
    : super(
        TableManagerState(
          db: db,
          table: table,
          createFilteringComposer: () =>
              $$TravellersTableFilterComposer($db: db, $table: table),
          createOrderingComposer: () =>
              $$TravellersTableOrderingComposer($db: db, $table: table),
          createComputedFieldComposer: () =>
              $$TravellersTableAnnotationComposer($db: db, $table: table),
          updateCompanionCallback:
              ({
                Value<int> id = const Value.absent(),
                Value<int> tripId = const Value.absent(),
                Value<String> name = const Value.absent(),
                Value<bool> isSelf = const Value.absent(),
              }) => TravellersCompanion(
                id: id,
                tripId: tripId,
                name: name,
                isSelf: isSelf,
              ),
          createCompanionCallback:
              ({
                Value<int> id = const Value.absent(),
                required int tripId,
                required String name,
                Value<bool> isSelf = const Value.absent(),
              }) => TravellersCompanion.insert(
                id: id,
                tripId: tripId,
                name: name,
                isSelf: isSelf,
              ),
          withReferenceMapper: (p0) => p0
              .map(
                (e) => (
                  e.readTable<$TravellersTable, Traveller>(table),
                  $$TravellersTableReferences(db, table, e),
                ),
              )
              .toList(),
          prefetchHooksCallback:
              ({
                tripId = false,
                expensesRefs = false,
                expenseSplitsRefs = false,
              }) {
                return PrefetchHooks(
                  db: db,
                  explicitlyWatchedTables: [
                    if (expensesRefs) db.expenses,
                    if (expenseSplitsRefs) db.expenseSplits,
                  ],
                  addJoins:
                      <
                        T extends TableManagerState<
                          dynamic,
                          dynamic,
                          dynamic,
                          dynamic,
                          dynamic,
                          dynamic,
                          dynamic,
                          dynamic,
                          dynamic,
                          dynamic,
                          dynamic
                        >
                      >(state) {
                        if (tripId) {
                          state =
                              state.withJoin(
                                    currentTable: table,
                                    currentColumn: table.tripId,
                                    referencedTable: $$TravellersTableReferences
                                        ._tripIdTable(db),
                                    referencedColumn:
                                        $$TravellersTableReferences
                                            ._tripIdTable(db)
                                            .id,
                                  )
                                  as T;
                        }

                        return state;
                      },
                  getPrefetchedDataCallback: (items) async {
                    return [
                      if (expensesRefs)
                        await $_getPrefetchedData<
                          Traveller,
                          $TravellersTable,
                          Expense
                        >(
                          currentTable: table,
                          referencedTable: $$TravellersTableReferences
                              ._expensesRefsTable(db),
                          managerFromTypedResult: (p0) =>
                              $$TravellersTableReferences(
                                db,
                                table,
                                p0,
                              ).expensesRefs,
                          referencedItemsForCurrentItem:
                              (item, referencedItems) => referencedItems.where(
                                (e) => e.paidById == item.id,
                              ),
                          typedResults: items,
                        ),
                      if (expenseSplitsRefs)
                        await $_getPrefetchedData<
                          Traveller,
                          $TravellersTable,
                          ExpenseSplit
                        >(
                          currentTable: table,
                          referencedTable: $$TravellersTableReferences
                              ._expenseSplitsRefsTable(db),
                          managerFromTypedResult: (p0) =>
                              $$TravellersTableReferences(
                                db,
                                table,
                                p0,
                              ).expenseSplitsRefs,
                          referencedItemsForCurrentItem:
                              (item, referencedItems) => referencedItems.where(
                                (e) => e.travellerId == item.id,
                              ),
                          typedResults: items,
                        ),
                    ];
                  },
                );
              },
        ),
      );
}

typedef $$TravellersTableProcessedTableManager =
    ProcessedTableManager<
      _$AppDatabase,
      $TravellersTable,
      Traveller,
      $$TravellersTableFilterComposer,
      $$TravellersTableOrderingComposer,
      $$TravellersTableAnnotationComposer,
      $$TravellersTableCreateCompanionBuilder,
      $$TravellersTableUpdateCompanionBuilder,
      (Traveller, $$TravellersTableReferences),
      Traveller,
      PrefetchHooks Function({
        bool tripId,
        bool expensesRefs,
        bool expenseSplitsRefs,
      })
    >;
typedef $$ExpensesTableCreateCompanionBuilder =
    ExpensesCompanion Function({
      Value<int> id,
      required int tripId,
      Value<int?> stopId,
      required String description,
      required int amountMinor,
      Value<String> currency,
      Value<double> rateToBase,
      Value<DateTime?> rateCapturedAt,
      required int paidById,
      Value<String?> category,
      Value<DateTime> spentAt,
    });
typedef $$ExpensesTableUpdateCompanionBuilder =
    ExpensesCompanion Function({
      Value<int> id,
      Value<int> tripId,
      Value<int?> stopId,
      Value<String> description,
      Value<int> amountMinor,
      Value<String> currency,
      Value<double> rateToBase,
      Value<DateTime?> rateCapturedAt,
      Value<int> paidById,
      Value<String?> category,
      Value<DateTime> spentAt,
    });

final class $$ExpensesTableReferences
    extends BaseReferences<_$AppDatabase, $ExpensesTable, Expense> {
  $$ExpensesTableReferences(super.$_db, super.$_table, super.$_typedResult);

  static $TripsTable _tripIdTable(_$AppDatabase db) =>
      db.trips.createAlias('expenses__trip_id__trips__id');

  $$TripsTableProcessedTableManager get tripId {
    final $_column = $_itemColumn<int>('trip_id')!;

    final manager = $$TripsTableTableManager(
      $_db,
      $_db.trips,
    ).filter((f) => f.id.sqlEquals($_column));
    final item = $_typedResult.readTableOrNull(_tripIdTable($_db));
    if (item == null) return manager;
    return ProcessedTableManager(
      manager.$state.copyWith(prefetchedData: [item]),
    );
  }

  static $StopsTable _stopIdTable(_$AppDatabase db) =>
      db.stops.createAlias('expenses__stop_id__stops__id');

  $$StopsTableProcessedTableManager? get stopId {
    final $_column = $_itemColumn<int>('stop_id');
    if ($_column == null) return null;
    final manager = $$StopsTableTableManager(
      $_db,
      $_db.stops,
    ).filter((f) => f.id.sqlEquals($_column));
    final item = $_typedResult.readTableOrNull(_stopIdTable($_db));
    if (item == null) return manager;
    return ProcessedTableManager(
      manager.$state.copyWith(prefetchedData: [item]),
    );
  }

  static $TravellersTable _paidByIdTable(_$AppDatabase db) =>
      db.travellers.createAlias('expenses__paid_by_id__travellers__id');

  $$TravellersTableProcessedTableManager get paidById {
    final $_column = $_itemColumn<int>('paid_by_id')!;

    final manager = $$TravellersTableTableManager(
      $_db,
      $_db.travellers,
    ).filter((f) => f.id.sqlEquals($_column));
    final item = $_typedResult.readTableOrNull(_paidByIdTable($_db));
    if (item == null) return manager;
    return ProcessedTableManager(
      manager.$state.copyWith(prefetchedData: [item]),
    );
  }

  static MultiTypedResultKey<$ExpenseSplitsTable, List<ExpenseSplit>>
  _expenseSplitsRefsTable(_$AppDatabase db) => MultiTypedResultKey.fromTable(
    db.expenseSplits,
    aliasName: 'expenses__id__expense_splits__expense_id',
  );

  $$ExpenseSplitsTableProcessedTableManager get expenseSplitsRefs {
    final manager = $$ExpenseSplitsTableTableManager(
      $_db,
      $_db.expenseSplits,
    ).filter((f) => f.expenseId.id.sqlEquals($_itemColumn<int>('id')!));

    final cache = $_typedResult.readTableOrNull(_expenseSplitsRefsTable($_db));
    return ProcessedTableManager(
      manager.$state.copyWith(prefetchedData: cache),
    );
  }
}

class $$ExpensesTableFilterComposer
    extends Composer<_$AppDatabase, $ExpensesTable> {
  $$ExpensesTableFilterComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  ColumnFilters<int> get id => $composableBuilder(
    column: $table.id,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<String> get description => $composableBuilder(
    column: $table.description,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<int> get amountMinor => $composableBuilder(
    column: $table.amountMinor,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<String> get currency => $composableBuilder(
    column: $table.currency,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<double> get rateToBase => $composableBuilder(
    column: $table.rateToBase,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<DateTime> get rateCapturedAt => $composableBuilder(
    column: $table.rateCapturedAt,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<String> get category => $composableBuilder(
    column: $table.category,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<DateTime> get spentAt => $composableBuilder(
    column: $table.spentAt,
    builder: (column) => ColumnFilters(column),
  );

  $$TripsTableFilterComposer get tripId {
    final $$TripsTableFilterComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.tripId,
      referencedTable: $db.trips,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$TripsTableFilterComposer(
            $db: $db,
            $table: $db.trips,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }

  $$StopsTableFilterComposer get stopId {
    final $$StopsTableFilterComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.stopId,
      referencedTable: $db.stops,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$StopsTableFilterComposer(
            $db: $db,
            $table: $db.stops,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }

  $$TravellersTableFilterComposer get paidById {
    final $$TravellersTableFilterComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.paidById,
      referencedTable: $db.travellers,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$TravellersTableFilterComposer(
            $db: $db,
            $table: $db.travellers,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }

  Expression<bool> expenseSplitsRefs(
    Expression<bool> Function($$ExpenseSplitsTableFilterComposer f) f,
  ) {
    final $$ExpenseSplitsTableFilterComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.id,
      referencedTable: $db.expenseSplits,
      getReferencedColumn: (t) => t.expenseId,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$ExpenseSplitsTableFilterComposer(
            $db: $db,
            $table: $db.expenseSplits,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return f(composer);
  }
}

class $$ExpensesTableOrderingComposer
    extends Composer<_$AppDatabase, $ExpensesTable> {
  $$ExpensesTableOrderingComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  ColumnOrderings<int> get id => $composableBuilder(
    column: $table.id,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<String> get description => $composableBuilder(
    column: $table.description,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<int> get amountMinor => $composableBuilder(
    column: $table.amountMinor,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<String> get currency => $composableBuilder(
    column: $table.currency,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<double> get rateToBase => $composableBuilder(
    column: $table.rateToBase,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<DateTime> get rateCapturedAt => $composableBuilder(
    column: $table.rateCapturedAt,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<String> get category => $composableBuilder(
    column: $table.category,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<DateTime> get spentAt => $composableBuilder(
    column: $table.spentAt,
    builder: (column) => ColumnOrderings(column),
  );

  $$TripsTableOrderingComposer get tripId {
    final $$TripsTableOrderingComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.tripId,
      referencedTable: $db.trips,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$TripsTableOrderingComposer(
            $db: $db,
            $table: $db.trips,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }

  $$StopsTableOrderingComposer get stopId {
    final $$StopsTableOrderingComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.stopId,
      referencedTable: $db.stops,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$StopsTableOrderingComposer(
            $db: $db,
            $table: $db.stops,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }

  $$TravellersTableOrderingComposer get paidById {
    final $$TravellersTableOrderingComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.paidById,
      referencedTable: $db.travellers,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$TravellersTableOrderingComposer(
            $db: $db,
            $table: $db.travellers,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }
}

class $$ExpensesTableAnnotationComposer
    extends Composer<_$AppDatabase, $ExpensesTable> {
  $$ExpensesTableAnnotationComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  GeneratedColumn<int> get id =>
      $composableBuilder(column: $table.id, builder: (column) => column);

  GeneratedColumn<String> get description => $composableBuilder(
    column: $table.description,
    builder: (column) => column,
  );

  GeneratedColumn<int> get amountMinor => $composableBuilder(
    column: $table.amountMinor,
    builder: (column) => column,
  );

  GeneratedColumn<String> get currency =>
      $composableBuilder(column: $table.currency, builder: (column) => column);

  GeneratedColumn<double> get rateToBase => $composableBuilder(
    column: $table.rateToBase,
    builder: (column) => column,
  );

  GeneratedColumn<DateTime> get rateCapturedAt => $composableBuilder(
    column: $table.rateCapturedAt,
    builder: (column) => column,
  );

  GeneratedColumn<String> get category =>
      $composableBuilder(column: $table.category, builder: (column) => column);

  GeneratedColumn<DateTime> get spentAt =>
      $composableBuilder(column: $table.spentAt, builder: (column) => column);

  $$TripsTableAnnotationComposer get tripId {
    final $$TripsTableAnnotationComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.tripId,
      referencedTable: $db.trips,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$TripsTableAnnotationComposer(
            $db: $db,
            $table: $db.trips,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }

  $$StopsTableAnnotationComposer get stopId {
    final $$StopsTableAnnotationComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.stopId,
      referencedTable: $db.stops,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$StopsTableAnnotationComposer(
            $db: $db,
            $table: $db.stops,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }

  $$TravellersTableAnnotationComposer get paidById {
    final $$TravellersTableAnnotationComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.paidById,
      referencedTable: $db.travellers,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$TravellersTableAnnotationComposer(
            $db: $db,
            $table: $db.travellers,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }

  Expression<T> expenseSplitsRefs<T extends Object>(
    Expression<T> Function($$ExpenseSplitsTableAnnotationComposer a) f,
  ) {
    final $$ExpenseSplitsTableAnnotationComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.id,
      referencedTable: $db.expenseSplits,
      getReferencedColumn: (t) => t.expenseId,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$ExpenseSplitsTableAnnotationComposer(
            $db: $db,
            $table: $db.expenseSplits,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return f(composer);
  }
}

class $$ExpensesTableTableManager
    extends
        RootTableManager<
          _$AppDatabase,
          $ExpensesTable,
          Expense,
          $$ExpensesTableFilterComposer,
          $$ExpensesTableOrderingComposer,
          $$ExpensesTableAnnotationComposer,
          $$ExpensesTableCreateCompanionBuilder,
          $$ExpensesTableUpdateCompanionBuilder,
          (Expense, $$ExpensesTableReferences),
          Expense,
          PrefetchHooks Function({
            bool tripId,
            bool stopId,
            bool paidById,
            bool expenseSplitsRefs,
          })
        > {
  $$ExpensesTableTableManager(_$AppDatabase db, $ExpensesTable table)
    : super(
        TableManagerState(
          db: db,
          table: table,
          createFilteringComposer: () =>
              $$ExpensesTableFilterComposer($db: db, $table: table),
          createOrderingComposer: () =>
              $$ExpensesTableOrderingComposer($db: db, $table: table),
          createComputedFieldComposer: () =>
              $$ExpensesTableAnnotationComposer($db: db, $table: table),
          updateCompanionCallback:
              ({
                Value<int> id = const Value.absent(),
                Value<int> tripId = const Value.absent(),
                Value<int?> stopId = const Value.absent(),
                Value<String> description = const Value.absent(),
                Value<int> amountMinor = const Value.absent(),
                Value<String> currency = const Value.absent(),
                Value<double> rateToBase = const Value.absent(),
                Value<DateTime?> rateCapturedAt = const Value.absent(),
                Value<int> paidById = const Value.absent(),
                Value<String?> category = const Value.absent(),
                Value<DateTime> spentAt = const Value.absent(),
              }) => ExpensesCompanion(
                id: id,
                tripId: tripId,
                stopId: stopId,
                description: description,
                amountMinor: amountMinor,
                currency: currency,
                rateToBase: rateToBase,
                rateCapturedAt: rateCapturedAt,
                paidById: paidById,
                category: category,
                spentAt: spentAt,
              ),
          createCompanionCallback:
              ({
                Value<int> id = const Value.absent(),
                required int tripId,
                Value<int?> stopId = const Value.absent(),
                required String description,
                required int amountMinor,
                Value<String> currency = const Value.absent(),
                Value<double> rateToBase = const Value.absent(),
                Value<DateTime?> rateCapturedAt = const Value.absent(),
                required int paidById,
                Value<String?> category = const Value.absent(),
                Value<DateTime> spentAt = const Value.absent(),
              }) => ExpensesCompanion.insert(
                id: id,
                tripId: tripId,
                stopId: stopId,
                description: description,
                amountMinor: amountMinor,
                currency: currency,
                rateToBase: rateToBase,
                rateCapturedAt: rateCapturedAt,
                paidById: paidById,
                category: category,
                spentAt: spentAt,
              ),
          withReferenceMapper: (p0) => p0
              .map(
                (e) => (
                  e.readTable<$ExpensesTable, Expense>(table),
                  $$ExpensesTableReferences(db, table, e),
                ),
              )
              .toList(),
          prefetchHooksCallback:
              ({
                tripId = false,
                stopId = false,
                paidById = false,
                expenseSplitsRefs = false,
              }) {
                return PrefetchHooks(
                  db: db,
                  explicitlyWatchedTables: [
                    if (expenseSplitsRefs) db.expenseSplits,
                  ],
                  addJoins:
                      <
                        T extends TableManagerState<
                          dynamic,
                          dynamic,
                          dynamic,
                          dynamic,
                          dynamic,
                          dynamic,
                          dynamic,
                          dynamic,
                          dynamic,
                          dynamic,
                          dynamic
                        >
                      >(state) {
                        if (tripId) {
                          state =
                              state.withJoin(
                                    currentTable: table,
                                    currentColumn: table.tripId,
                                    referencedTable: $$ExpensesTableReferences
                                        ._tripIdTable(db),
                                    referencedColumn: $$ExpensesTableReferences
                                        ._tripIdTable(db)
                                        .id,
                                  )
                                  as T;
                        }
                        if (stopId) {
                          state =
                              state.withJoin(
                                    currentTable: table,
                                    currentColumn: table.stopId,
                                    referencedTable: $$ExpensesTableReferences
                                        ._stopIdTable(db),
                                    referencedColumn: $$ExpensesTableReferences
                                        ._stopIdTable(db)
                                        .id,
                                  )
                                  as T;
                        }
                        if (paidById) {
                          state =
                              state.withJoin(
                                    currentTable: table,
                                    currentColumn: table.paidById,
                                    referencedTable: $$ExpensesTableReferences
                                        ._paidByIdTable(db),
                                    referencedColumn: $$ExpensesTableReferences
                                        ._paidByIdTable(db)
                                        .id,
                                  )
                                  as T;
                        }

                        return state;
                      },
                  getPrefetchedDataCallback: (items) async {
                    return [
                      if (expenseSplitsRefs)
                        await $_getPrefetchedData<
                          Expense,
                          $ExpensesTable,
                          ExpenseSplit
                        >(
                          currentTable: table,
                          referencedTable: $$ExpensesTableReferences
                              ._expenseSplitsRefsTable(db),
                          managerFromTypedResult: (p0) =>
                              $$ExpensesTableReferences(
                                db,
                                table,
                                p0,
                              ).expenseSplitsRefs,
                          referencedItemsForCurrentItem:
                              (item, referencedItems) => referencedItems.where(
                                (e) => e.expenseId == item.id,
                              ),
                          typedResults: items,
                        ),
                    ];
                  },
                );
              },
        ),
      );
}

typedef $$ExpensesTableProcessedTableManager =
    ProcessedTableManager<
      _$AppDatabase,
      $ExpensesTable,
      Expense,
      $$ExpensesTableFilterComposer,
      $$ExpensesTableOrderingComposer,
      $$ExpensesTableAnnotationComposer,
      $$ExpensesTableCreateCompanionBuilder,
      $$ExpensesTableUpdateCompanionBuilder,
      (Expense, $$ExpensesTableReferences),
      Expense,
      PrefetchHooks Function({
        bool tripId,
        bool stopId,
        bool paidById,
        bool expenseSplitsRefs,
      })
    >;
typedef $$ExpenseSplitsTableCreateCompanionBuilder =
    ExpenseSplitsCompanion Function({
      Value<int> id,
      required int expenseId,
      required int travellerId,
      required int shareMinor,
    });
typedef $$ExpenseSplitsTableUpdateCompanionBuilder =
    ExpenseSplitsCompanion Function({
      Value<int> id,
      Value<int> expenseId,
      Value<int> travellerId,
      Value<int> shareMinor,
    });

final class $$ExpenseSplitsTableReferences
    extends BaseReferences<_$AppDatabase, $ExpenseSplitsTable, ExpenseSplit> {
  $$ExpenseSplitsTableReferences(
    super.$_db,
    super.$_table,
    super.$_typedResult,
  );

  static $ExpensesTable _expenseIdTable(_$AppDatabase db) =>
      db.expenses.createAlias('expense_splits__expense_id__expenses__id');

  $$ExpensesTableProcessedTableManager get expenseId {
    final $_column = $_itemColumn<int>('expense_id')!;

    final manager = $$ExpensesTableTableManager(
      $_db,
      $_db.expenses,
    ).filter((f) => f.id.sqlEquals($_column));
    final item = $_typedResult.readTableOrNull(_expenseIdTable($_db));
    if (item == null) return manager;
    return ProcessedTableManager(
      manager.$state.copyWith(prefetchedData: [item]),
    );
  }

  static $TravellersTable _travellerIdTable(_$AppDatabase db) =>
      db.travellers.createAlias('expense_splits__traveller_id__travellers__id');

  $$TravellersTableProcessedTableManager get travellerId {
    final $_column = $_itemColumn<int>('traveller_id')!;

    final manager = $$TravellersTableTableManager(
      $_db,
      $_db.travellers,
    ).filter((f) => f.id.sqlEquals($_column));
    final item = $_typedResult.readTableOrNull(_travellerIdTable($_db));
    if (item == null) return manager;
    return ProcessedTableManager(
      manager.$state.copyWith(prefetchedData: [item]),
    );
  }
}

class $$ExpenseSplitsTableFilterComposer
    extends Composer<_$AppDatabase, $ExpenseSplitsTable> {
  $$ExpenseSplitsTableFilterComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  ColumnFilters<int> get id => $composableBuilder(
    column: $table.id,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<int> get shareMinor => $composableBuilder(
    column: $table.shareMinor,
    builder: (column) => ColumnFilters(column),
  );

  $$ExpensesTableFilterComposer get expenseId {
    final $$ExpensesTableFilterComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.expenseId,
      referencedTable: $db.expenses,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$ExpensesTableFilterComposer(
            $db: $db,
            $table: $db.expenses,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }

  $$TravellersTableFilterComposer get travellerId {
    final $$TravellersTableFilterComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.travellerId,
      referencedTable: $db.travellers,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$TravellersTableFilterComposer(
            $db: $db,
            $table: $db.travellers,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }
}

class $$ExpenseSplitsTableOrderingComposer
    extends Composer<_$AppDatabase, $ExpenseSplitsTable> {
  $$ExpenseSplitsTableOrderingComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  ColumnOrderings<int> get id => $composableBuilder(
    column: $table.id,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<int> get shareMinor => $composableBuilder(
    column: $table.shareMinor,
    builder: (column) => ColumnOrderings(column),
  );

  $$ExpensesTableOrderingComposer get expenseId {
    final $$ExpensesTableOrderingComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.expenseId,
      referencedTable: $db.expenses,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$ExpensesTableOrderingComposer(
            $db: $db,
            $table: $db.expenses,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }

  $$TravellersTableOrderingComposer get travellerId {
    final $$TravellersTableOrderingComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.travellerId,
      referencedTable: $db.travellers,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$TravellersTableOrderingComposer(
            $db: $db,
            $table: $db.travellers,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }
}

class $$ExpenseSplitsTableAnnotationComposer
    extends Composer<_$AppDatabase, $ExpenseSplitsTable> {
  $$ExpenseSplitsTableAnnotationComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  GeneratedColumn<int> get id =>
      $composableBuilder(column: $table.id, builder: (column) => column);

  GeneratedColumn<int> get shareMinor => $composableBuilder(
    column: $table.shareMinor,
    builder: (column) => column,
  );

  $$ExpensesTableAnnotationComposer get expenseId {
    final $$ExpensesTableAnnotationComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.expenseId,
      referencedTable: $db.expenses,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$ExpensesTableAnnotationComposer(
            $db: $db,
            $table: $db.expenses,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }

  $$TravellersTableAnnotationComposer get travellerId {
    final $$TravellersTableAnnotationComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.travellerId,
      referencedTable: $db.travellers,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$TravellersTableAnnotationComposer(
            $db: $db,
            $table: $db.travellers,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }
}

class $$ExpenseSplitsTableTableManager
    extends
        RootTableManager<
          _$AppDatabase,
          $ExpenseSplitsTable,
          ExpenseSplit,
          $$ExpenseSplitsTableFilterComposer,
          $$ExpenseSplitsTableOrderingComposer,
          $$ExpenseSplitsTableAnnotationComposer,
          $$ExpenseSplitsTableCreateCompanionBuilder,
          $$ExpenseSplitsTableUpdateCompanionBuilder,
          (ExpenseSplit, $$ExpenseSplitsTableReferences),
          ExpenseSplit,
          PrefetchHooks Function({bool expenseId, bool travellerId})
        > {
  $$ExpenseSplitsTableTableManager(_$AppDatabase db, $ExpenseSplitsTable table)
    : super(
        TableManagerState(
          db: db,
          table: table,
          createFilteringComposer: () =>
              $$ExpenseSplitsTableFilterComposer($db: db, $table: table),
          createOrderingComposer: () =>
              $$ExpenseSplitsTableOrderingComposer($db: db, $table: table),
          createComputedFieldComposer: () =>
              $$ExpenseSplitsTableAnnotationComposer($db: db, $table: table),
          updateCompanionCallback:
              ({
                Value<int> id = const Value.absent(),
                Value<int> expenseId = const Value.absent(),
                Value<int> travellerId = const Value.absent(),
                Value<int> shareMinor = const Value.absent(),
              }) => ExpenseSplitsCompanion(
                id: id,
                expenseId: expenseId,
                travellerId: travellerId,
                shareMinor: shareMinor,
              ),
          createCompanionCallback:
              ({
                Value<int> id = const Value.absent(),
                required int expenseId,
                required int travellerId,
                required int shareMinor,
              }) => ExpenseSplitsCompanion.insert(
                id: id,
                expenseId: expenseId,
                travellerId: travellerId,
                shareMinor: shareMinor,
              ),
          withReferenceMapper: (p0) => p0
              .map(
                (e) => (
                  e.readTable<$ExpenseSplitsTable, ExpenseSplit>(table),
                  $$ExpenseSplitsTableReferences(db, table, e),
                ),
              )
              .toList(),
          prefetchHooksCallback: ({expenseId = false, travellerId = false}) {
            return PrefetchHooks(
              db: db,
              explicitlyWatchedTables: [],
              addJoins:
                  <
                    T extends TableManagerState<
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic
                    >
                  >(state) {
                    if (expenseId) {
                      state =
                          state.withJoin(
                                currentTable: table,
                                currentColumn: table.expenseId,
                                referencedTable: $$ExpenseSplitsTableReferences
                                    ._expenseIdTable(db),
                                referencedColumn: $$ExpenseSplitsTableReferences
                                    ._expenseIdTable(db)
                                    .id,
                              )
                              as T;
                    }
                    if (travellerId) {
                      state =
                          state.withJoin(
                                currentTable: table,
                                currentColumn: table.travellerId,
                                referencedTable: $$ExpenseSplitsTableReferences
                                    ._travellerIdTable(db),
                                referencedColumn: $$ExpenseSplitsTableReferences
                                    ._travellerIdTable(db)
                                    .id,
                              )
                              as T;
                    }

                    return state;
                  },
              getPrefetchedDataCallback: (items) async {
                return [];
              },
            );
          },
        ),
      );
}

typedef $$ExpenseSplitsTableProcessedTableManager =
    ProcessedTableManager<
      _$AppDatabase,
      $ExpenseSplitsTable,
      ExpenseSplit,
      $$ExpenseSplitsTableFilterComposer,
      $$ExpenseSplitsTableOrderingComposer,
      $$ExpenseSplitsTableAnnotationComposer,
      $$ExpenseSplitsTableCreateCompanionBuilder,
      $$ExpenseSplitsTableUpdateCompanionBuilder,
      (ExpenseSplit, $$ExpenseSplitsTableReferences),
      ExpenseSplit,
      PrefetchHooks Function({bool expenseId, bool travellerId})
    >;
typedef $$TimelineEntriesTableCreateCompanionBuilder =
    TimelineEntriesCompanion Function({
      Value<int> id,
      required int tripId,
      Value<int?> stopId,
      required String kind,
      Value<String?> title,
      Value<String?> body,
      Value<double?> lat,
      Value<double?> lon,
      Value<double?> accuracyM,
      Value<String?> photoPaths,
      required DateTime occurredAt,
    });
typedef $$TimelineEntriesTableUpdateCompanionBuilder =
    TimelineEntriesCompanion Function({
      Value<int> id,
      Value<int> tripId,
      Value<int?> stopId,
      Value<String> kind,
      Value<String?> title,
      Value<String?> body,
      Value<double?> lat,
      Value<double?> lon,
      Value<double?> accuracyM,
      Value<String?> photoPaths,
      Value<DateTime> occurredAt,
    });

final class $$TimelineEntriesTableReferences
    extends
        BaseReferences<_$AppDatabase, $TimelineEntriesTable, TimelineEntry> {
  $$TimelineEntriesTableReferences(
    super.$_db,
    super.$_table,
    super.$_typedResult,
  );

  static $TripsTable _tripIdTable(_$AppDatabase db) =>
      db.trips.createAlias('timeline_entries__trip_id__trips__id');

  $$TripsTableProcessedTableManager get tripId {
    final $_column = $_itemColumn<int>('trip_id')!;

    final manager = $$TripsTableTableManager(
      $_db,
      $_db.trips,
    ).filter((f) => f.id.sqlEquals($_column));
    final item = $_typedResult.readTableOrNull(_tripIdTable($_db));
    if (item == null) return manager;
    return ProcessedTableManager(
      manager.$state.copyWith(prefetchedData: [item]),
    );
  }

  static $StopsTable _stopIdTable(_$AppDatabase db) =>
      db.stops.createAlias('timeline_entries__stop_id__stops__id');

  $$StopsTableProcessedTableManager? get stopId {
    final $_column = $_itemColumn<int>('stop_id');
    if ($_column == null) return null;
    final manager = $$StopsTableTableManager(
      $_db,
      $_db.stops,
    ).filter((f) => f.id.sqlEquals($_column));
    final item = $_typedResult.readTableOrNull(_stopIdTable($_db));
    if (item == null) return manager;
    return ProcessedTableManager(
      manager.$state.copyWith(prefetchedData: [item]),
    );
  }
}

class $$TimelineEntriesTableFilterComposer
    extends Composer<_$AppDatabase, $TimelineEntriesTable> {
  $$TimelineEntriesTableFilterComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  ColumnFilters<int> get id => $composableBuilder(
    column: $table.id,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<String> get kind => $composableBuilder(
    column: $table.kind,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<String> get title => $composableBuilder(
    column: $table.title,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<String> get body => $composableBuilder(
    column: $table.body,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<double> get lat => $composableBuilder(
    column: $table.lat,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<double> get lon => $composableBuilder(
    column: $table.lon,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<double> get accuracyM => $composableBuilder(
    column: $table.accuracyM,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<String> get photoPaths => $composableBuilder(
    column: $table.photoPaths,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<DateTime> get occurredAt => $composableBuilder(
    column: $table.occurredAt,
    builder: (column) => ColumnFilters(column),
  );

  $$TripsTableFilterComposer get tripId {
    final $$TripsTableFilterComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.tripId,
      referencedTable: $db.trips,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$TripsTableFilterComposer(
            $db: $db,
            $table: $db.trips,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }

  $$StopsTableFilterComposer get stopId {
    final $$StopsTableFilterComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.stopId,
      referencedTable: $db.stops,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$StopsTableFilterComposer(
            $db: $db,
            $table: $db.stops,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }
}

class $$TimelineEntriesTableOrderingComposer
    extends Composer<_$AppDatabase, $TimelineEntriesTable> {
  $$TimelineEntriesTableOrderingComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  ColumnOrderings<int> get id => $composableBuilder(
    column: $table.id,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<String> get kind => $composableBuilder(
    column: $table.kind,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<String> get title => $composableBuilder(
    column: $table.title,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<String> get body => $composableBuilder(
    column: $table.body,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<double> get lat => $composableBuilder(
    column: $table.lat,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<double> get lon => $composableBuilder(
    column: $table.lon,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<double> get accuracyM => $composableBuilder(
    column: $table.accuracyM,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<String> get photoPaths => $composableBuilder(
    column: $table.photoPaths,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<DateTime> get occurredAt => $composableBuilder(
    column: $table.occurredAt,
    builder: (column) => ColumnOrderings(column),
  );

  $$TripsTableOrderingComposer get tripId {
    final $$TripsTableOrderingComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.tripId,
      referencedTable: $db.trips,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$TripsTableOrderingComposer(
            $db: $db,
            $table: $db.trips,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }

  $$StopsTableOrderingComposer get stopId {
    final $$StopsTableOrderingComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.stopId,
      referencedTable: $db.stops,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$StopsTableOrderingComposer(
            $db: $db,
            $table: $db.stops,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }
}

class $$TimelineEntriesTableAnnotationComposer
    extends Composer<_$AppDatabase, $TimelineEntriesTable> {
  $$TimelineEntriesTableAnnotationComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  GeneratedColumn<int> get id =>
      $composableBuilder(column: $table.id, builder: (column) => column);

  GeneratedColumn<String> get kind =>
      $composableBuilder(column: $table.kind, builder: (column) => column);

  GeneratedColumn<String> get title =>
      $composableBuilder(column: $table.title, builder: (column) => column);

  GeneratedColumn<String> get body =>
      $composableBuilder(column: $table.body, builder: (column) => column);

  GeneratedColumn<double> get lat =>
      $composableBuilder(column: $table.lat, builder: (column) => column);

  GeneratedColumn<double> get lon =>
      $composableBuilder(column: $table.lon, builder: (column) => column);

  GeneratedColumn<double> get accuracyM =>
      $composableBuilder(column: $table.accuracyM, builder: (column) => column);

  GeneratedColumn<String> get photoPaths => $composableBuilder(
    column: $table.photoPaths,
    builder: (column) => column,
  );

  GeneratedColumn<DateTime> get occurredAt => $composableBuilder(
    column: $table.occurredAt,
    builder: (column) => column,
  );

  $$TripsTableAnnotationComposer get tripId {
    final $$TripsTableAnnotationComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.tripId,
      referencedTable: $db.trips,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$TripsTableAnnotationComposer(
            $db: $db,
            $table: $db.trips,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }

  $$StopsTableAnnotationComposer get stopId {
    final $$StopsTableAnnotationComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.stopId,
      referencedTable: $db.stops,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$StopsTableAnnotationComposer(
            $db: $db,
            $table: $db.stops,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }
}

class $$TimelineEntriesTableTableManager
    extends
        RootTableManager<
          _$AppDatabase,
          $TimelineEntriesTable,
          TimelineEntry,
          $$TimelineEntriesTableFilterComposer,
          $$TimelineEntriesTableOrderingComposer,
          $$TimelineEntriesTableAnnotationComposer,
          $$TimelineEntriesTableCreateCompanionBuilder,
          $$TimelineEntriesTableUpdateCompanionBuilder,
          (TimelineEntry, $$TimelineEntriesTableReferences),
          TimelineEntry,
          PrefetchHooks Function({bool tripId, bool stopId})
        > {
  $$TimelineEntriesTableTableManager(
    _$AppDatabase db,
    $TimelineEntriesTable table,
  ) : super(
        TableManagerState(
          db: db,
          table: table,
          createFilteringComposer: () =>
              $$TimelineEntriesTableFilterComposer($db: db, $table: table),
          createOrderingComposer: () =>
              $$TimelineEntriesTableOrderingComposer($db: db, $table: table),
          createComputedFieldComposer: () =>
              $$TimelineEntriesTableAnnotationComposer($db: db, $table: table),
          updateCompanionCallback:
              ({
                Value<int> id = const Value.absent(),
                Value<int> tripId = const Value.absent(),
                Value<int?> stopId = const Value.absent(),
                Value<String> kind = const Value.absent(),
                Value<String?> title = const Value.absent(),
                Value<String?> body = const Value.absent(),
                Value<double?> lat = const Value.absent(),
                Value<double?> lon = const Value.absent(),
                Value<double?> accuracyM = const Value.absent(),
                Value<String?> photoPaths = const Value.absent(),
                Value<DateTime> occurredAt = const Value.absent(),
              }) => TimelineEntriesCompanion(
                id: id,
                tripId: tripId,
                stopId: stopId,
                kind: kind,
                title: title,
                body: body,
                lat: lat,
                lon: lon,
                accuracyM: accuracyM,
                photoPaths: photoPaths,
                occurredAt: occurredAt,
              ),
          createCompanionCallback:
              ({
                Value<int> id = const Value.absent(),
                required int tripId,
                Value<int?> stopId = const Value.absent(),
                required String kind,
                Value<String?> title = const Value.absent(),
                Value<String?> body = const Value.absent(),
                Value<double?> lat = const Value.absent(),
                Value<double?> lon = const Value.absent(),
                Value<double?> accuracyM = const Value.absent(),
                Value<String?> photoPaths = const Value.absent(),
                required DateTime occurredAt,
              }) => TimelineEntriesCompanion.insert(
                id: id,
                tripId: tripId,
                stopId: stopId,
                kind: kind,
                title: title,
                body: body,
                lat: lat,
                lon: lon,
                accuracyM: accuracyM,
                photoPaths: photoPaths,
                occurredAt: occurredAt,
              ),
          withReferenceMapper: (p0) => p0
              .map(
                (e) => (
                  e.readTable<$TimelineEntriesTable, TimelineEntry>(table),
                  $$TimelineEntriesTableReferences(db, table, e),
                ),
              )
              .toList(),
          prefetchHooksCallback: ({tripId = false, stopId = false}) {
            return PrefetchHooks(
              db: db,
              explicitlyWatchedTables: [],
              addJoins:
                  <
                    T extends TableManagerState<
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic
                    >
                  >(state) {
                    if (tripId) {
                      state =
                          state.withJoin(
                                currentTable: table,
                                currentColumn: table.tripId,
                                referencedTable:
                                    $$TimelineEntriesTableReferences
                                        ._tripIdTable(db),
                                referencedColumn:
                                    $$TimelineEntriesTableReferences
                                        ._tripIdTable(db)
                                        .id,
                              )
                              as T;
                    }
                    if (stopId) {
                      state =
                          state.withJoin(
                                currentTable: table,
                                currentColumn: table.stopId,
                                referencedTable:
                                    $$TimelineEntriesTableReferences
                                        ._stopIdTable(db),
                                referencedColumn:
                                    $$TimelineEntriesTableReferences
                                        ._stopIdTable(db)
                                        .id,
                              )
                              as T;
                    }

                    return state;
                  },
              getPrefetchedDataCallback: (items) async {
                return [];
              },
            );
          },
        ),
      );
}

typedef $$TimelineEntriesTableProcessedTableManager =
    ProcessedTableManager<
      _$AppDatabase,
      $TimelineEntriesTable,
      TimelineEntry,
      $$TimelineEntriesTableFilterComposer,
      $$TimelineEntriesTableOrderingComposer,
      $$TimelineEntriesTableAnnotationComposer,
      $$TimelineEntriesTableCreateCompanionBuilder,
      $$TimelineEntriesTableUpdateCompanionBuilder,
      (TimelineEntry, $$TimelineEntriesTableReferences),
      TimelineEntry,
      PrefetchHooks Function({bool tripId, bool stopId})
    >;
typedef $$TrustedContactsTableCreateCompanionBuilder =
    TrustedContactsCompanion Function({
      Value<int> id,
      Value<int?> tripId,
      required String name,
      required String phoneE164,
      Value<bool> notifyOnArrival,
      Value<bool> escalate,
      Value<int> escalateAfterMinutes,
    });
typedef $$TrustedContactsTableUpdateCompanionBuilder =
    TrustedContactsCompanion Function({
      Value<int> id,
      Value<int?> tripId,
      Value<String> name,
      Value<String> phoneE164,
      Value<bool> notifyOnArrival,
      Value<bool> escalate,
      Value<int> escalateAfterMinutes,
    });

final class $$TrustedContactsTableReferences
    extends
        BaseReferences<_$AppDatabase, $TrustedContactsTable, TrustedContact> {
  $$TrustedContactsTableReferences(
    super.$_db,
    super.$_table,
    super.$_typedResult,
  );

  static $TripsTable _tripIdTable(_$AppDatabase db) =>
      db.trips.createAlias('trusted_contacts__trip_id__trips__id');

  $$TripsTableProcessedTableManager? get tripId {
    final $_column = $_itemColumn<int>('trip_id');
    if ($_column == null) return null;
    final manager = $$TripsTableTableManager(
      $_db,
      $_db.trips,
    ).filter((f) => f.id.sqlEquals($_column));
    final item = $_typedResult.readTableOrNull(_tripIdTable($_db));
    if (item == null) return manager;
    return ProcessedTableManager(
      manager.$state.copyWith(prefetchedData: [item]),
    );
  }
}

class $$TrustedContactsTableFilterComposer
    extends Composer<_$AppDatabase, $TrustedContactsTable> {
  $$TrustedContactsTableFilterComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  ColumnFilters<int> get id => $composableBuilder(
    column: $table.id,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<String> get name => $composableBuilder(
    column: $table.name,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<String> get phoneE164 => $composableBuilder(
    column: $table.phoneE164,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<bool> get notifyOnArrival => $composableBuilder(
    column: $table.notifyOnArrival,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<bool> get escalate => $composableBuilder(
    column: $table.escalate,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<int> get escalateAfterMinutes => $composableBuilder(
    column: $table.escalateAfterMinutes,
    builder: (column) => ColumnFilters(column),
  );

  $$TripsTableFilterComposer get tripId {
    final $$TripsTableFilterComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.tripId,
      referencedTable: $db.trips,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$TripsTableFilterComposer(
            $db: $db,
            $table: $db.trips,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }
}

class $$TrustedContactsTableOrderingComposer
    extends Composer<_$AppDatabase, $TrustedContactsTable> {
  $$TrustedContactsTableOrderingComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  ColumnOrderings<int> get id => $composableBuilder(
    column: $table.id,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<String> get name => $composableBuilder(
    column: $table.name,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<String> get phoneE164 => $composableBuilder(
    column: $table.phoneE164,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<bool> get notifyOnArrival => $composableBuilder(
    column: $table.notifyOnArrival,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<bool> get escalate => $composableBuilder(
    column: $table.escalate,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<int> get escalateAfterMinutes => $composableBuilder(
    column: $table.escalateAfterMinutes,
    builder: (column) => ColumnOrderings(column),
  );

  $$TripsTableOrderingComposer get tripId {
    final $$TripsTableOrderingComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.tripId,
      referencedTable: $db.trips,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$TripsTableOrderingComposer(
            $db: $db,
            $table: $db.trips,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }
}

class $$TrustedContactsTableAnnotationComposer
    extends Composer<_$AppDatabase, $TrustedContactsTable> {
  $$TrustedContactsTableAnnotationComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  GeneratedColumn<int> get id =>
      $composableBuilder(column: $table.id, builder: (column) => column);

  GeneratedColumn<String> get name =>
      $composableBuilder(column: $table.name, builder: (column) => column);

  GeneratedColumn<String> get phoneE164 =>
      $composableBuilder(column: $table.phoneE164, builder: (column) => column);

  GeneratedColumn<bool> get notifyOnArrival => $composableBuilder(
    column: $table.notifyOnArrival,
    builder: (column) => column,
  );

  GeneratedColumn<bool> get escalate =>
      $composableBuilder(column: $table.escalate, builder: (column) => column);

  GeneratedColumn<int> get escalateAfterMinutes => $composableBuilder(
    column: $table.escalateAfterMinutes,
    builder: (column) => column,
  );

  $$TripsTableAnnotationComposer get tripId {
    final $$TripsTableAnnotationComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.tripId,
      referencedTable: $db.trips,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$TripsTableAnnotationComposer(
            $db: $db,
            $table: $db.trips,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }
}

class $$TrustedContactsTableTableManager
    extends
        RootTableManager<
          _$AppDatabase,
          $TrustedContactsTable,
          TrustedContact,
          $$TrustedContactsTableFilterComposer,
          $$TrustedContactsTableOrderingComposer,
          $$TrustedContactsTableAnnotationComposer,
          $$TrustedContactsTableCreateCompanionBuilder,
          $$TrustedContactsTableUpdateCompanionBuilder,
          (TrustedContact, $$TrustedContactsTableReferences),
          TrustedContact,
          PrefetchHooks Function({bool tripId})
        > {
  $$TrustedContactsTableTableManager(
    _$AppDatabase db,
    $TrustedContactsTable table,
  ) : super(
        TableManagerState(
          db: db,
          table: table,
          createFilteringComposer: () =>
              $$TrustedContactsTableFilterComposer($db: db, $table: table),
          createOrderingComposer: () =>
              $$TrustedContactsTableOrderingComposer($db: db, $table: table),
          createComputedFieldComposer: () =>
              $$TrustedContactsTableAnnotationComposer($db: db, $table: table),
          updateCompanionCallback:
              ({
                Value<int> id = const Value.absent(),
                Value<int?> tripId = const Value.absent(),
                Value<String> name = const Value.absent(),
                Value<String> phoneE164 = const Value.absent(),
                Value<bool> notifyOnArrival = const Value.absent(),
                Value<bool> escalate = const Value.absent(),
                Value<int> escalateAfterMinutes = const Value.absent(),
              }) => TrustedContactsCompanion(
                id: id,
                tripId: tripId,
                name: name,
                phoneE164: phoneE164,
                notifyOnArrival: notifyOnArrival,
                escalate: escalate,
                escalateAfterMinutes: escalateAfterMinutes,
              ),
          createCompanionCallback:
              ({
                Value<int> id = const Value.absent(),
                Value<int?> tripId = const Value.absent(),
                required String name,
                required String phoneE164,
                Value<bool> notifyOnArrival = const Value.absent(),
                Value<bool> escalate = const Value.absent(),
                Value<int> escalateAfterMinutes = const Value.absent(),
              }) => TrustedContactsCompanion.insert(
                id: id,
                tripId: tripId,
                name: name,
                phoneE164: phoneE164,
                notifyOnArrival: notifyOnArrival,
                escalate: escalate,
                escalateAfterMinutes: escalateAfterMinutes,
              ),
          withReferenceMapper: (p0) => p0
              .map(
                (e) => (
                  e.readTable<$TrustedContactsTable, TrustedContact>(table),
                  $$TrustedContactsTableReferences(db, table, e),
                ),
              )
              .toList(),
          prefetchHooksCallback: ({tripId = false}) {
            return PrefetchHooks(
              db: db,
              explicitlyWatchedTables: [],
              addJoins:
                  <
                    T extends TableManagerState<
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic
                    >
                  >(state) {
                    if (tripId) {
                      state =
                          state.withJoin(
                                currentTable: table,
                                currentColumn: table.tripId,
                                referencedTable:
                                    $$TrustedContactsTableReferences
                                        ._tripIdTable(db),
                                referencedColumn:
                                    $$TrustedContactsTableReferences
                                        ._tripIdTable(db)
                                        .id,
                              )
                              as T;
                    }

                    return state;
                  },
              getPrefetchedDataCallback: (items) async {
                return [];
              },
            );
          },
        ),
      );
}

typedef $$TrustedContactsTableProcessedTableManager =
    ProcessedTableManager<
      _$AppDatabase,
      $TrustedContactsTable,
      TrustedContact,
      $$TrustedContactsTableFilterComposer,
      $$TrustedContactsTableOrderingComposer,
      $$TrustedContactsTableAnnotationComposer,
      $$TrustedContactsTableCreateCompanionBuilder,
      $$TrustedContactsTableUpdateCompanionBuilder,
      (TrustedContact, $$TrustedContactsTableReferences),
      TrustedContact,
      PrefetchHooks Function({bool tripId})
    >;

class $AppDatabaseManager {
  final _$AppDatabase _db;
  $AppDatabaseManager(this._db);
  $$TripsTableTableManager get trips =>
      $$TripsTableTableManager(_db, _db.trips);
  $$StopsTableTableManager get stops =>
      $$StopsTableTableManager(_db, _db.stops);
  $$LegsTableTableManager get legs => $$LegsTableTableManager(_db, _db.legs);
  $$PoisTableTableManager get pois => $$PoisTableTableManager(_db, _db.pois);
  $$PoiContactsTableTableManager get poiContacts =>
      $$PoiContactsTableTableManager(_db, _db.poiContacts);
  $$ImportBatchesTableTableManager get importBatches =>
      $$ImportBatchesTableTableManager(_db, _db.importBatches);
  $$ContactsTableTableManager get contacts =>
      $$ContactsTableTableManager(_db, _db.contacts);
  $$CallLogsTableTableManager get callLogs =>
      $$CallLogsTableTableManager(_db, _db.callLogs);
  $$EmergencyHelplinesTableTableManager get emergencyHelplines =>
      $$EmergencyHelplinesTableTableManager(_db, _db.emergencyHelplines);
  $$ChecklistItemsTableTableManager get checklistItems =>
      $$ChecklistItemsTableTableManager(_db, _db.checklistItems);
  $$WeatherSnapshotsTableTableManager get weatherSnapshots =>
      $$WeatherSnapshotsTableTableManager(_db, _db.weatherSnapshots);
  $$TravellersTableTableManager get travellers =>
      $$TravellersTableTableManager(_db, _db.travellers);
  $$ExpensesTableTableManager get expenses =>
      $$ExpensesTableTableManager(_db, _db.expenses);
  $$ExpenseSplitsTableTableManager get expenseSplits =>
      $$ExpenseSplitsTableTableManager(_db, _db.expenseSplits);
  $$TimelineEntriesTableTableManager get timelineEntries =>
      $$TimelineEntriesTableTableManager(_db, _db.timelineEntries);
  $$TrustedContactsTableTableManager get trustedContacts =>
      $$TrustedContactsTableTableManager(_db, _db.trustedContacts);
}
