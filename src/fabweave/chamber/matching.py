from __future__ import annotations

import copy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from fabweave.core.timebox import TimeBox
from fabweave.core.units import finite_non_negative


def matched(reference: dict[str, float], candidate: dict[str, float], tolerances: dict[str, float]) -> bool:
    return not mismatch_reasons(reference, candidate, {key: Tolerance(absolute=value) for key, value in tolerances.items()})


@dataclass(frozen=True, slots=True)
class Tolerance:
    absolute: float = 0.0
    relative: float = 0.0

    def __post_init__(self) -> None:
        finite_non_negative(self.absolute, "absolute tolerance")
        finite_non_negative(self.relative, "relative tolerance")

    def limit_for(self, reference: float) -> float:
        return max(self.absolute, abs(reference) * self.relative)


def mismatch_reasons(
    reference: dict[str, float],
    candidate: dict[str, float],
    tolerances: dict[str, Tolerance],
) -> tuple[str, ...]:
    reasons: list[str] = []
    for key, tolerance in sorted(tolerances.items()):
        if key not in reference or key not in candidate:
            reasons.append(f"{key}:missing")
            continue
        expected, actual = reference[key], candidate[key]
        try:
            finite_non_negative(expected, f"{key} reference")
            finite_non_negative(actual, f"{key} candidate")
        except ValueError:
            reasons.append(f"{key}:non_finite_or_negative")
            continue
        limit = tolerance.limit_for(expected)
        if abs(expected - actual) > limit:
            reasons.append(f"{key}:outside_tolerance")
    return tuple(reasons)


@dataclass(frozen=True, slots=True)
class Measurement:
    """One multi-parameter chamber measurement made at a point in event time."""

    sample_id: str
    measured_at: int
    parameters: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise ValueError("sample id must be a non-empty string")
        finite_non_negative(self.measured_at, "measurement time")
        parameters = dict(self.parameters)
        for name, value in parameters.items():
            if not isinstance(name, str) or not name:
                raise ValueError("parameter names must be non-empty strings")
            finite_non_negative(value, f"{name} measurement")
        object.__setattr__(self, "parameters", MappingProxyType(parameters))

    def __deepcopy__(self, memo: dict[int, Any]) -> "Measurement":
        return Measurement(
            self.sample_id,
            self.measured_at,
            {name: copy.deepcopy(value, memo) for name, value in self.parameters.items()},
        )


# Domain language uses both “sample” and “measurement” for the same evidence row.
ChamberSample = Measurement
ChamberMeasurement = Measurement
MeasurementSample = Measurement
Sample = Measurement
Observation = Measurement


@dataclass(frozen=True, slots=True)
class ToleranceRule:
    """Versioned set of parameter tolerances effective at a point in time."""

    version: int
    effective_at: int
    tolerances: Mapping[str, Tolerance]

    def __post_init__(self) -> None:
        if self.version <= 0:
            raise ValueError("tolerance rule version must be positive")
        finite_non_negative(self.effective_at, "rule effective time")
        tolerances = dict(self.tolerances)
        for name, tolerance in tolerances.items():
            if not isinstance(name, str) or not name:
                raise ValueError("parameter names must be non-empty strings")
            if not isinstance(tolerance, Tolerance):
                raise ValueError(f"invalid tolerance for {name}")
        object.__setattr__(self, "tolerances", MappingProxyType(tolerances))

    def __deepcopy__(self, memo: dict[int, Any]) -> "ToleranceRule":
        return ToleranceRule(
            self.version,
            self.effective_at,
            {name: copy.deepcopy(value, memo) for name, value in self.tolerances.items()},
        )


ToleranceRevision = ToleranceRule
TolerancePolicy = ToleranceRule
ToleranceVersion = ToleranceRule


@dataclass(frozen=True, slots=True)
class SamplePair:
    """A golden member and the candidate member it was compared with."""

    golden: Measurement
    candidate: Measurement

    @property
    def sample_ids(self) -> tuple[str, str]:
        return self.golden.sample_id, self.candidate.sample_id

    @property
    def reference(self) -> Measurement:
        return self.golden

    @property
    def reference_sample(self) -> Measurement:
        return self.golden

    @property
    def candidate_sample(self) -> Measurement:
        return self.candidate


PairedMeasurements = SamplePair
PairedSamples = SamplePair
MeasurementPair = SamplePair
MatchPair = SamplePair


@dataclass(frozen=True, slots=True)
class ParameterDeviation:
    pair_index: int
    parameter: str
    golden_sample_id: str
    candidate_sample_id: str
    golden_value: float
    candidate_value: float
    difference: float
    allowed_limit: float
    rule_version: int

    @property
    def within_tolerance(self) -> bool:
        return abs(self.difference) <= self.allowed_limit

    @property
    def expected(self) -> float:
        return self.golden_value

    @property
    def actual(self) -> float:
        return self.candidate_value

    @property
    def gap(self) -> float:
        return abs(self.difference)


@dataclass(frozen=True, slots=True)
class ChamberMatchReport:
    """Conclusive comparison, or an explicit refusal to draw a conclusion."""

    status: str
    reasons: tuple[str, ...]
    members: tuple[SamplePair, ...]
    rule_versions: tuple[int | None, ...]
    deviations: tuple[ParameterDeviation, ...]
    window: TimeBox | None = None

    @property
    def allowed(self) -> bool:
        return self.status == "match"

    @property
    def match(self) -> bool:
        return self.allowed

    @property
    def result(self) -> str:
        return self.status

    @property
    def conclusive(self) -> bool:
        return self.status in {"match", "mismatch"}

    @property
    def pairs(self) -> tuple[SamplePair, ...]:
        return self.members

    @property
    def paired_samples(self) -> tuple[SamplePair, ...]:
        return self.members

    @property
    def samples(self) -> tuple[SamplePair, ...]:
        return self.members

    @property
    def pair_ids(self) -> tuple[tuple[str, str], ...]:
        return self.member_samples

    @property
    def member_samples(self) -> tuple[tuple[str, str], ...]:
        return tuple(pair.sample_ids for pair in self.members)

    @property
    def rule_revisions(self) -> tuple[int | None, ...]:
        return self.rule_versions

    @property
    def insufficient_evidence(self) -> bool:
        return self.status == "inconclusive"

    @property
    def evidence(self) -> Mapping[str, Any]:
        return {
            "status": self.status,
            "window": self.window,
            "members": self.members,
            "pairs": self.members,
            "member_samples": self.member_samples,
            "rule_versions": self.rule_versions,
            "rule_revisions": self.rule_versions,
            "deviations": self.deviations,
        }


ChamberComparison = ChamberMatchReport
ChamberComparisonReport = ChamberMatchReport
ChamberMatchResult = ChamberMatchReport
MatchReport = ChamberMatchReport


def _as_time_box(window: TimeBox | tuple[int, int]) -> TimeBox:
    if isinstance(window, TimeBox):
        return window
    if isinstance(window, tuple) and len(window) == 2:
        return TimeBox(window[0], window[1])
    raise ValueError("window must be a TimeBox or (start, end)")


def _as_measurement(sample: Measurement | Mapping[str, Any] | Sequence[Any]) -> Measurement:
    if isinstance(sample, Measurement):
        return Measurement(sample.sample_id, sample.measured_at, dict(sample.parameters))

    if isinstance(sample, Mapping):
        source = dict(sample)
        sample_id = source.get("sample_id", source.get("id"))
        measured_at = source.get(
            "measured_at",
            source.get("timestamp", source.get("time", source.get("measured_time"))),
        )
        parameters = source.get("parameters", source.get("values", source.get("metrics")))
    elif isinstance(sample, tuple) and len(sample) == 3:
        sample_id, measured_at, parameters = sample
    elif isinstance(sample, Sequence) and len(sample) == 3:
        sample_id, measured_at, parameters = sample[0], sample[1], sample[2]
    else:
        raise ValueError("a measurement must be a Measurement, mapping, or (id, time, parameters)")

    if sample_id is None or measured_at is None or parameters is None:
        raise ValueError("measurement id, time, and parameters are required")
    return Measurement(str(sample_id), measured_at, dict(parameters))


def _as_tolerance(value: Tolerance | Mapping[str, float] | Sequence[float]) -> Tolerance:
    if isinstance(value, Tolerance):
        return value
    if isinstance(value, Mapping):
        return Tolerance(
            absolute=float(value.get("absolute", value.get("abs", 0.0))),
            relative=float(value.get("relative", value.get("rel", 0.0))),
        )
    if isinstance(value, tuple) and len(value) == 2:
        return Tolerance(absolute=float(value[0]), relative=float(value[1]))
    raise ValueError("tolerance must be a Tolerance, mapping, or (absolute, relative)")


def _as_rule(rule: ToleranceRule | Mapping[str, Any]) -> ToleranceRule:
    if isinstance(rule, ToleranceRule):
        return rule
    if isinstance(rule, Mapping):
        source = dict(rule)
        version = source.get("version", source.get("rule_version", source.get("revision")))
        effective_at = source.get("effective_at", source.get("start"))
        tolerances = source.get("tolerances", source.get("limits"))
        if version is None or effective_at is None or tolerances is None:
            raise ValueError("tolerance rule requires version, effective_at, and tolerances")
        return ToleranceRule(version, effective_at, {key: _as_tolerance(value) for key, value in dict(tolerances).items()})

    version = getattr(rule, "version", getattr(rule, "revision", None))
    effective_at = getattr(rule, "effective_at", getattr(rule, "start", None))
    tolerances = getattr(rule, "tolerances", getattr(rule, "limits", None))
    if version is None or effective_at is None or tolerances is None:
        raise ValueError("tolerance rule requires version, effective_at, and tolerances")
    return ToleranceRule(version, effective_at, {key: _as_tolerance(value) for key, value in dict(tolerances).items()})


def _rule_at(rules: Any, instant: int) -> ToleranceRule | None:
    if isinstance(rules, ToleranceRule):
        return rules if rules.effective_at <= instant else None
    if isinstance(rules, Mapping):
        if "tolerances" in rules or "limits" in rules:
            rule = _as_rule(rules)
            return rule if rule.effective_at <= instant else None
        try:
            tolerances = {name: _as_tolerance(value) for name, value in rules.items()}
        except ValueError:
            tolerances = {}
        if tolerances:
            return ToleranceRule(1, 0, tolerances)
    if hasattr(rules, "at"):
        try:
            revision = rules.at(instant)
        except LookupError:
            return None
        value = getattr(revision, "value", revision)
        if isinstance(value, ToleranceRule):
            return value
        if isinstance(value, Mapping) and "tolerances" not in value and "limits" not in value:
            return ToleranceRule(revision.version, revision.effective_at, dict(value))
        return _as_rule(value)

    effective = [rule for rule in (_as_rule(item) for item in rules) if rule.effective_at <= instant]
    if not effective:
        return None
    return max(effective, key=lambda rule: (rule.effective_at, rule.version))


def _filter_window(
    samples: Sequence[Measurement | Mapping[str, Any] | Sequence[Any]],
    window: TimeBox | None,
) -> tuple[Measurement, ...]:
    measurements = [_as_measurement(sample) for sample in samples]
    if window is None:
        return tuple(measurements)
    return tuple(sample for sample in measurements if window.contains(sample.measured_at))


def _pairs_by_id(
    golden: Sequence[Measurement],
    candidate: Sequence[Measurement],
    max_time_delta: float | None,
) -> tuple[tuple[SamplePair, ...], list[str]]:
    reasons: list[str] = []

    golden_by_id: dict[str, list[Measurement]] = {}
    candidate_by_id: dict[str, list[Measurement]] = {}
    for sample in golden:
        golden_by_id.setdefault(sample.sample_id, []).append(sample)
    for sample in candidate:
        candidate_by_id.setdefault(sample.sample_id, []).append(sample)

    for side, grouped in (("golden", golden_by_id), ("candidate", candidate_by_id)):
        for sample_id, members in sorted(grouped.items()):
            if len(members) > 1:
                reasons.append(f"duplicate_sample_id:{side}:{sample_id}")

    if reasons:
        return (), reasons

    golden_ids = set(golden_by_id)
    candidate_ids = set(candidate_by_id)
    for sample_id in sorted(golden_ids - candidate_ids):
        reasons.append(f"unpaired_sample:golden:{sample_id}")
    for sample_id in sorted(candidate_ids - golden_ids):
        reasons.append(f"unpaired_sample:candidate:{sample_id}")
    if reasons:
        return (), reasons

    pairs: list[SamplePair] = []
    for sample_id in sorted(golden_ids & candidate_ids):
        golden_sample = golden_by_id[sample_id][0]
        candidate_sample = candidate_by_id[sample_id][0]
        gap = abs(golden_sample.measured_at - candidate_sample.measured_at)
        if max_time_delta is not None and gap > max_time_delta:
            reasons.append(f"sample_pair_time_gap:{sample_id}")
        pairs.append(SamplePair(golden_sample, candidate_sample))

    pairs.sort(key=lambda pair: (pair.candidate.measured_at, pair.golden.measured_at, pair.golden.sample_id))
    return tuple(pairs), reasons


def _pairs_by_time(
    golden: Sequence[Measurement],
    candidate: Sequence[Measurement],
    max_time_delta: float | None,
) -> tuple[tuple[SamplePair, ...], list[str]]:
    """Optimal order-preserving one-to-one nearest-time matching.

    The objective is deterministic and independent of input ordering: maximize
    the number of pairs, then minimize their total absolute time gap.  Sample ids
    only break exact ties.
    """

    golden_samples = sorted(golden, key=lambda sample: (sample.measured_at, sample.sample_id))
    candidate_samples = sorted(candidate, key=lambda sample: (sample.measured_at, sample.sample_id))
    rows = len(golden_samples)
    columns = len(candidate_samples)
    limit = float("inf") if max_time_delta is None else max_time_delta

    # Each score is (negative pair count, total gap, deterministic ids), so
    # ordinary tuple ordering maximizes coverage and then minimizes elapsed gap.
    empty = (0, 0.0, ())
    scores: list[list[tuple[int, float, tuple[tuple[str, str], ...]]]] = [
        [empty] * (columns + 1) for _ in range(rows + 1)
    ]

    for row in range(rows - 1, -1, -1):
        for column in range(columns - 1, -1, -1):
            options = [scores[row + 1][column], scores[row][column + 1]]
            golden_sample = golden_samples[row]
            candidate_sample = candidate_samples[column]
            gap = abs(golden_sample.measured_at - candidate_sample.measured_at)
            if gap <= limit:
                negative_count, total_gap, tie_break = scores[row + 1][column + 1]
                options.append(
                    (
                        negative_count - 1,
                        total_gap + gap,
                        ((golden_sample.sample_id, candidate_sample.sample_id),) + tie_break,
                    )
                )
            scores[row][column] = min(options)

    pairs: list[SamplePair] = []
    row = column = 0
    while row < rows and column < columns:
        current = scores[row][column]
        if current == scores[row + 1][column]:
            row += 1
            continue
        if current == scores[row][column + 1]:
            column += 1
            continue

        golden_sample = golden_samples[row]
        candidate_sample = candidate_samples[column]
        pairs.append(SamplePair(golden_sample, candidate_sample))
        row += 1
        column += 1

    paired_golden = {pair.golden.sample_id for pair in pairs}
    paired_candidate = {pair.candidate.sample_id for pair in pairs}
    reasons: list[str] = []
    if rows != columns or len(pairs) != rows or len(pairs) != columns:
        reasons.append(f"sample_count_mismatch:golden:{rows}:candidate:{columns}")
    for sample in golden_samples:
        if sample.sample_id not in paired_golden:
            reasons.append(f"unpaired_sample:golden:{sample.sample_id}")
    for sample in candidate_samples:
        if sample.sample_id not in paired_candidate:
            reasons.append(f"unpaired_sample:candidate:{sample.sample_id}")

    pairs.sort(key=lambda pair: (pair.candidate.measured_at, pair.golden.measured_at, pair.candidate.sample_id))
    return tuple(pairs), reasons


def _resolve_pairing_options(
    *,
    minimum_pairs: int = 1,
    pair_by: str = "auto",
    max_time_delta: float | None = None,
    **options: Any,
) -> tuple[int, str, float | None]:
    minimum_pairs = options.pop("min_pairs", minimum_pairs)
    pair_by = options.pop("matching", options.pop("match_by", pair_by))
    max_time_delta = options.pop("time_delta", options.pop("max_delta", max_time_delta))
    if options:
        name = next(iter(options))
        raise TypeError(f"unexpected keyword argument: {name}")
    return minimum_pairs, pair_by, max_time_delta


def pair_chamber_samples(
    golden_samples: Sequence[Measurement | Mapping[str, Any] | Sequence[Any]],
    candidate_samples: Sequence[Measurement | Mapping[str, Any] | Sequence[Any]],
    window: TimeBox | tuple[int, int] | None = None,
    *,
    pair_by: str = "auto",
    max_time_delta: float | None = None,
    **options: Any,
) -> tuple[SamplePair, ...]:
    """Return deterministic one-to-one pairs inside a half-open time window."""

    _, pair_by, max_time_delta = _resolve_pairing_options(
        pair_by=pair_by,
        max_time_delta=max_time_delta,
        **options,
    )
    analysis_window = None if window is None else _as_time_box(window)
    golden = _filter_window(golden_samples, analysis_window)
    candidate = _filter_window(candidate_samples, analysis_window)

    if pair_by in {"id", "sample_id"}:
        mode = "id"
    elif pair_by in {"time", "timestamp"}:
        mode = "time"
    elif pair_by == "auto":
        mode = "time"
    else:
        raise ValueError("pair_by must be one of: auto, id, time")

    if max_time_delta is not None:
        finite_non_negative(max_time_delta, "maximum time delta")
    pairing = _pairs_by_id if mode == "id" else _pairs_by_time
    pairs, reasons = pairing(golden, candidate, max_time_delta)
    if reasons:
        raise ValueError("; ".join(sorted(set(reasons))))
    return pairs


def compare_chambers(
    golden_samples: Sequence[Measurement | Mapping[str, Any] | Sequence[Any]],
    candidate_samples: Sequence[Measurement | Mapping[str, Any] | Sequence[Any]],
    tolerance_rules: Any,
    window: TimeBox | tuple[int, int] | None = None,
    *,
    minimum_pairs: int = 1,
    pair_by: str = "auto",
    max_time_delta: float | None = None,
    **options: Any,
) -> ChamberMatchReport:
    """
    Pair multi-parameter samples and evaluate each member against its golden peer.

    Averages are never substituted for paired evidence.  When the pairs or rules
    do not support a complete comparison, the returned status is ``inconclusive``.
    Tolerance selection uses the candidate sample's event time.
    """

    minimum_pairs, pair_by, max_time_delta = _resolve_pairing_options(
        minimum_pairs=minimum_pairs,
        pair_by=pair_by,
        max_time_delta=max_time_delta,
        **options,
    )
    if minimum_pairs <= 0:
        raise ValueError("minimum_pairs must be positive")
    analysis_window = None if window is None else _as_time_box(window)
    golden = _filter_window(golden_samples, analysis_window)
    candidate = _filter_window(candidate_samples, analysis_window)

    if pair_by in {"id", "sample_id"}:
        mode = "id"
    elif pair_by in {"time", "timestamp"}:
        mode = "time"
    elif pair_by == "auto":
        mode = "time"
    else:
        raise ValueError("pair_by must be one of: auto, id, time")

    if max_time_delta is not None:
        finite_non_negative(max_time_delta, "maximum time delta")
    pairing = _pairs_by_id if mode == "id" else _pairs_by_time
    members, structural_reasons = pairing(golden, candidate, max_time_delta)

    reasons = set(structural_reasons)
    if len(members) < minimum_pairs:
        reasons.add(f"insufficient_sample_pairs:{len(members)}:{minimum_pairs}")

    rule_versions: list[int | None] = []
    deviations: list[ParameterDeviation] = []

    for pair_index, pair in enumerate(members):
        rule = _rule_at(tolerance_rules, pair.candidate.measured_at)
        rule_versions.append(None if rule is None else rule.version)
        if rule is None:
            reasons.add(f"tolerance_rule_missing:{pair.candidate.measured_at}")
            continue

        golden_parameters = dict(pair.golden.parameters)
        candidate_parameters = dict(pair.candidate.parameters)
        parameter_names = sorted(rule.tolerances)
        if not parameter_names:
            reasons.add("tolerance_rule_has_no_parameters")
            continue

        for name in parameter_names:
            if name not in golden_parameters:
                reasons.add(f"{name}:missing:golden")
                continue
            if name not in candidate_parameters:
                reasons.add(f"{name}:missing:candidate")
                continue

            tolerance = rule.tolerances.get(name)
            if tolerance is None:
                reasons.add(f"{name}:tolerance_missing")
                continue

            expected = golden_parameters[name]
            actual = candidate_parameters[name]
            limit = tolerance.limit_for(expected)
            difference = actual - expected
            if abs(difference) > limit:
                reasons.add(f"{name}:outside_tolerance")
                deviations.append(
                    ParameterDeviation(
                        pair_index=pair_index,
                        parameter=name,
                        golden_sample_id=pair.golden.sample_id,
                        candidate_sample_id=pair.candidate.sample_id,
                        golden_value=expected,
                        candidate_value=actual,
                        difference=difference,
                        allowed_limit=limit,
                        rule_version=rule.version,
                    )
                )

    ordered_reasons = tuple(sorted(reasons))
    deviations.sort(
        key=lambda item: (
            item.pair_index,
            item.parameter,
            item.golden_sample_id,
            item.candidate_sample_id,
        )
    )

    structural_reason = any(
        reason != "insufficient_evidence" and not reason.endswith(":outside_tolerance")
        for reason in reasons
    )
    if structural_reason:
        reasons.update({"inconclusive", "insufficient_evidence"})
        ordered_reasons = tuple(sorted(reasons))

    if structural_reason:
        status = "inconclusive"
    elif any(reason.endswith("outside_tolerance") for reason in reasons):
        status = "mismatch"
    else:
        status = "match"

    return ChamberMatchReport(
        status=status,
        reasons=ordered_reasons,
        members=members,
        rule_versions=tuple(rule_versions),
        deviations=tuple(deviations),
        window=analysis_window,
    )


# Public aliases for callers that name this operation by its evidence shape.
pair_chamber_measurements = pair_chamber_samples
pair_samples = pair_chamber_samples
paired_samples = pair_chamber_samples
compare_chamber_samples = compare_chambers
compare_chamber_measurements = compare_chambers
compare_paired_chambers = compare_chambers
compare_paired_measurements = compare_chambers
evaluate_chamber_matching = compare_chambers
evaluate_chamber_match = compare_chambers
chamber_match_report = compare_chambers
paired_chamber_comparison = compare_chambers
