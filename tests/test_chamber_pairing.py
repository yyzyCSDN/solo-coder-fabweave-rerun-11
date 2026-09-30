from __future__ import annotations

from fabweave.chamber.matching import (
    Measurement,
    Tolerance,
    ToleranceRule,
    compare_chambers,
)
from fabweave.core.timebox import TimeBox


def test_paired_chamber_members_use_time_effective_rule():
    golden = [Measurement("s1", 10, {"pressure": 10.0, "flow": 100.0})]
    candidate = [Measurement("s1", 11, {"pressure": 11.0, "flow": 109.0})]
    rules = [
        ToleranceRule(1, 0, {"pressure": Tolerance(absolute=0.5), "flow": Tolerance(relative=0.1)}),
        ToleranceRule(2, 11, {"pressure": Tolerance(absolute=2.0), "flow": Tolerance(relative=0.1)}),
    ]

    report = compare_chambers(golden, candidate, rules, TimeBox(0, 20), pair_by="id")

    assert report.allowed
    assert report.member_samples == (("s1", "s1"),)
    assert report.rule_versions == (2,)


def test_incomplete_evidence_is_rejected_even_when_one_member_deviates():
    golden = [
        Measurement("s1", 1, {"pressure": 10.0}),
        Measurement("s2", 2, {"pressure": 10.0}),
    ]
    candidate = [
        Measurement("s1", 1, {"pressure": 12.0}),
        Measurement("s2", 2, {}),
    ]
    rules = [ToleranceRule(1, 0, {"pressure": Tolerance(absolute=1.0)})]

    report = compare_chambers(golden, candidate, rules, TimeBox(0, 10), pair_by="id")

    assert report.status == "inconclusive"
    assert "insufficient_evidence" in report.reasons
    assert "pressure:missing:candidate" in report.reasons
    assert len(report.members) == 2


def test_window_excludes_measurements_and_input_order_is_stable():
    rules = [ToleranceRule(1, 0, {"pressure": Tolerance(absolute=1.0)})]
    golden = [
        Measurement("s2", 12, {"pressure": 100.0}),
        Measurement("s1", 2, {"pressure": 10.0}),
    ]
    candidate = [
        Measurement("s1", 3, {"pressure": 10.5}),
        Measurement("s2", 13, {"pressure": 100.5}),
    ]

    report = compare_chambers(
        reversed(golden),
        reversed(candidate),
        list(reversed(rules)),
        TimeBox(0, 10),
        pair_by="time",
        max_time_delta=2,
    )

    assert report.allowed
    assert report.member_samples == (("s1", "s1"),)
    assert report.rule_versions == (1,)
