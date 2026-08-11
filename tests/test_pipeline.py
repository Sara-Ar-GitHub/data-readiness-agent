"""Tests assert that the injected faults in the demo dataset are recovered.

This is what turns the repo from a demo into evidence: the faults are known
by construction, so precision and recall of the assessment are measurable.
"""

import pandas as pd
import pytest

from dra.agent import deterministic_report
from dra.tools import lineage, physics, profiling


@pytest.fixture(scope="module")
def demo() -> pd.DataFrame:
    return pd.read_csv("examples/line3_press_shop.csv")


def test_stuck_sensor_detected(demo):
    prof = profiling.profile_dataset(demo)
    bearing = next(c for c in prof.columns if c.name == "bearing_temp_c")
    assert bearing.stuck_max_run >= 400


def test_missing_rate_detected(demo):
    prof = profiling.profile_dataset(demo)
    torque = next(c for c in prof.columns if c.name == "torque_nm")
    assert 0.15 < torque.missing_rate < 0.21


def test_constant_column_detected(demo):
    prof = profiling.profile_dataset(demo)
    assert next(c for c in prof.columns if c.name == "plant_id").is_constant


def test_duplicate_column_traced(demo):
    edges = lineage.infer_lineage(demo)
    assert any(e.target == "power_kw_duplicate" or "power_kw_duplicate" in e.source
               for e in edges)


def test_each_target_reported_once(demo):
    edges = lineage.infer_lineage(demo)
    targets = [e.target for e in edges]
    assert len(targets) == len(set(targets))


def test_negative_pressure_violates_range(demo):
    c = physics.check_range(demo, "coolant_pressure_bar", 0.0, None,
                            "Absolute pressure cannot be negative.")
    assert not c.holds and c.evidence["n_violations"] >= 25


def test_shaft_power_identity_holds(demo):
    c = physics.check_identity(
        demo, "power_kw - torque_nm * speed_rpm * 2 * pi / 60 / 1000", 0.01,
        "Shaft power equals torque times angular velocity.")
    assert c.holds


def test_meter_drift_breaks_conservation(demo):
    c = physics.check_conservation(
        demo, ["station_a_units", "station_b_units", "station_c_units"],
        "line_total_units", 0.02, "Stations must sum to the line total.")
    assert not c.holds and c.violation_rate > 0.4


def test_thermal_rate_limit_violated(demo):
    c = physics.check_rate_limit(demo, "ambient_temp_c", 1.0, "timestamp",
                                 "Ambient air cannot swing 45 C in one sample.")
    assert not c.holds


def test_report_is_serialisable(demo):
    r = deterministic_report(demo, "demo", "manufacturing")
    assert 0 <= r.overall_score <= 5
    assert r.model_dump_json()
