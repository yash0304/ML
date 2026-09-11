// lib/features/emergency/data/emergency_seed.dart
//
// Tier 1 India national helplines, bundled with the binary.
//
// EVERY NUMBER HERE IS TRANSCRIBED FROM PROJECT_RUNDOWN.md §4.2, which is the
// record of the research that produced it. Nothing in this file was written
// from memory, and nothing should be. If a number is not in that table it
// does not belong here.
//
// THE NON-NEGOTIABLE: no emergency number ships without a government-domain
// source, recorded in `sourceNote` and shown in the UI. The user can see
// where a number came from and judge it for themselves. A fabricated
// ambulance number at 2am in Kongthong is worse than no number at all.

/// One bundled helpline, before it reaches the database.
class SeedHelpline {
  final String countryCode;
  final String serviceType;
  final String label;
  final String number;

  /// Shown under the number in the UI, always. Never empty.
  final String sourceNote;
  final String? sourceUrl;

  /// True for numbers widely cited on aggregator sites but NOT confirmed
  /// against a `.gov.in` source. The seeder skips these entirely — they are
  /// recorded here so the next session knows they were considered and
  /// rejected, rather than forgotten.
  ///
  /// Issue #36 verifies them and flips the flag. Nothing else should.
  final bool needsVerification;

  const SeedHelpline({
    required this.countryCode,
    required this.serviceType,
    required this.label,
    required this.number,
    required this.sourceNote,
    this.sourceUrl,
    this.needsVerification = false,
  });
}

/// Service type vocabulary. Kept as strings so a tier or service can be added
/// without a migration.
class ServiceType {
  static const all = 'all';
  static const police = 'police';
  static const fire = 'fire';
  static const ambulance = 'ambulance';
  static const women = 'women';
  static const child = 'child';
  static const tourist = 'tourist';
  static const railway = 'railway';
  static const senior = 'senior';
  static const disability = 'disability';
  static const humanRights = 'humanRights';
  static const obsceneCalls = 'obsceneCalls';
}

const _legacy = 'Legacy line, active alongside 112';
const _indiaGov = 'india.gov.in helpline directory';

/// Fourteen seeded rows, four flagged and skipped.
const emergencySeed = <SeedHelpline>[
  // --- The one number that matters most -----------------------------------
  SeedHelpline(
    countryCode: 'IN',
    serviceType: ServiceType.all,
    label: 'All emergencies',
    number: '112',
    sourceNote: '112.gov.in, Ministry of Home Affairs',
    sourceUrl: 'https://112.gov.in',
  ),

  // --- Legacy lines, still active -----------------------------------------
  SeedHelpline(
    countryCode: 'IN',
    serviceType: ServiceType.police,
    label: 'Police',
    number: '100',
    sourceNote: _legacy,
  ),
  SeedHelpline(
    countryCode: 'IN',
    serviceType: ServiceType.fire,
    label: 'Fire',
    number: '101',
    sourceNote: _legacy,
  ),
  // 102 and 108 are both ambulance lines and our source does not distinguish
  // them. Inventing a distinction would be exactly the confident-sounding
  // guess this feature exists to avoid — see ISSUE_4_Seeding.md.
  SeedHelpline(
    countryCode: 'IN',
    serviceType: ServiceType.ambulance,
    label: 'Ambulance',
    number: '102',
    sourceNote: _legacy,
  ),
  SeedHelpline(
    countryCode: 'IN',
    serviceType: ServiceType.ambulance,
    label: 'Ambulance',
    number: '108',
    sourceNote: _legacy,
  ),

  // --- Directory lines ----------------------------------------------------
  SeedHelpline(
    countryCode: 'IN',
    serviceType: ServiceType.women,
    label: 'Women helpline',
    number: '181',
    sourceNote: _indiaGov,
  ),
  SeedHelpline(
    countryCode: 'IN',
    serviceType: ServiceType.women,
    label: 'Women helpline, NCW 24x7',
    number: '14490',
    sourceNote: 'ncw.gov.in, National Commission for Women',
    sourceUrl: 'https://ncw.gov.in',
  ),
  SeedHelpline(
    countryCode: 'IN',
    serviceType: ServiceType.obsceneCalls,
    label: 'Anti-obscene calls cell',
    number: '1091',
    sourceNote: _indiaGov,
  ),
  SeedHelpline(
    countryCode: 'IN',
    serviceType: ServiceType.child,
    label: 'Child helpline',
    number: '1098',
    sourceNote: _indiaGov,
  ),
  SeedHelpline(
    countryCode: 'IN',
    serviceType: ServiceType.tourist,
    label: 'Tourist helpline',
    number: '1363',
    sourceNote: _indiaGov,
  ),
  SeedHelpline(
    countryCode: 'IN',
    serviceType: ServiceType.railway,
    label: 'Railway security and medical',
    number: '139',
    sourceNote: _indiaGov,
  ),
  SeedHelpline(
    countryCode: 'IN',
    serviceType: ServiceType.senior,
    label: 'Senior citizens',
    number: '14567',
    sourceNote: _indiaGov,
  ),
  SeedHelpline(
    countryCode: 'IN',
    serviceType: ServiceType.disability,
    label: 'Disabilities',
    number: '14456',
    sourceNote: _indiaGov,
  ),
  SeedHelpline(
    countryCode: 'IN',
    serviceType: ServiceType.humanRights,
    label: 'Human rights, NHRC',
    number: '14433',
    sourceNote: _indiaGov,
  ),

  // =========================================================================
  // FLAGGED — NOT SEEDED. Do not remove the flag without a .gov.in source.
  //
  // Widely cited on aggregator sites, not confirmed from a government domain.
  // They are listed here so a future session can see they were considered and
  // rejected rather than overlooked. Issue #36 verifies them.
  // =========================================================================
  SeedHelpline(
    countryCode: 'IN',
    serviceType: 'cyberCrime',
    label: 'Cyber crime',
    number: '1930',
    sourceNote: 'Unconfirmed against a .gov.in source',
    needsVerification: true,
  ),
  SeedHelpline(
    countryCode: 'IN',
    serviceType: 'disaster',
    label: 'Disaster management, NDMA',
    number: '1078',
    sourceNote: 'Unconfirmed against a .gov.in source',
    needsVerification: true,
  ),
  SeedHelpline(
    countryCode: 'IN',
    serviceType: 'highway',
    label: 'Highway accident',
    number: '1033',
    sourceNote: 'Unconfirmed against a .gov.in source',
    needsVerification: true,
  ),
  SeedHelpline(
    countryCode: 'IN',
    serviceType: 'health',
    label: 'Health',
    number: '104',
    sourceNote:
        'Unconfirmed against a .gov.in source, and state-operated rather '
        'than live everywhere',
    needsVerification: true,
  ),
];

// State and union territory helplines are DELIBERATELY ABSENT.
//
// There is no authoritative combined machine-readable dataset across 28 states
// and 8 union territories. Populating this from aggregator sites or from
// memory would be worse than shipping it empty. Issue #37 does it manually,
// per government portal, with provenance recorded per number.
