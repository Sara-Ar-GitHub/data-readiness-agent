"""Fault-recovery tests on the cold-chain logistics dataset.

The point of a second sector is not more coverage of the same code paths. It
is a falsifiable claim: that the assessment layer is not quietly specialised to
the press shop it was written against.

Nothing here passes a sector hint to the profiler or the quality rules. If
these tests pass, the seven readiness dimensions and the five constraint kinds
carried over to a domain with no shared vocabulary -- no torque, no shaft
power, different units, different failure modes.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from dra.agent import deterministic_report
from dra.tools import physics, profiling

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "examples" / "fleet_cold_chain.csv"


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    if not CSV.exists():
        subprocess.run([sys.executable, "examples/make_demo_logistics.py"],
                       cwd=ROOT, check=True, capture_output=True)
    return pd.read_csv(CSV)


@pytest.fixture(scope="module")
def report(df):
    return deterministic_report(df, "fleet_cold_chain", sector="transport and logistics")


# --- deterministic layer ---------------------------------------------------

def test_frozen_reefer_probe_detected(df):
    """Fault 2: return-air probe stuck for 300 samples."""
    prof = profiling.profile_dataset(df)
    col = next(c for c in prof.columns if c.name == "reefer_return_air_c")
    assert col.stuck_max_run >= 300


def test_constant_setpoint_detected(df):
    prof = profiling.profile_dataset(df)
    col = next(c for c in prof.columns if c.name == "reefer_setpoint_c")
    assert col.is_constant


def test_duplicate_consignment_ids_detected(report):
    """Fault 7. Rows are not duplicated -- only the identifier is -- so this is
    invisible to row-level duplicate detection."""
    f = next(f for f in report.findings if "consignment_id" in f.columns)
    assert f.dimension.value == "uniqueness"
    assert f.evidence["near_key_duplicate_rate"] == pytest.approx(0.03, abs=0.005)


def test_float_sensor_is_not_mistaken_for_a_business_key(report):
    """A frozen float transmitter leaves a near-unique column with repeats.
    That is a stuck-sensor finding, not a duplicate-key one."""
    uniqueness = [f for f in report.findings if f.dimension.value == "uniqueness"]
    assert not any("reefer_return_air_c" in f.columns for f in uniqueness)


def test_unit_conversion_traced_as_derived(report):
    """Fault 4: distance_mi is distance_km in other units -- a leakage risk that
    looks like an independent measurement."""
    edge = next(e for e in report.lineage if e.target == "distance_mi")
    assert "distance_km" in edge.source
    assert edge.confidence > 0.99


def test_fuel_is_traced_to_distance(report):
    """Fault 5: fuel_l is a linear function of distance, not a meter reading."""
    edge = next(e for e in report.lineage if e.target == "fuel_l")
    assert "distance_km" in edge.source


def test_vehicle_speed_is_not_read_as_rotational(df):
    """Unit inference is name-based and fallible, but `speed_kph` on a vehicle
    must not come back as rpm -- that is the exact confusion the hint exists
    to prevent."""
    prof = profiling.profile_dataset(df)
    col = next(c for c in prof.columns if c.name == "vehicle_speed_kph")
    assert col.inferred_unit == "linear_speed"


def test_telemetry_gaps_detected(report):
    assert report.profile.gap_count >= 3


# --- constraint layer, on logistics physics --------------------------------

def test_negative_payload_violates_range(df):
    """Fault 3: tare miscalibration. Not a statistical outlier -- impossible."""
    c = physics.check_range(df, "payload_kg", 0.0, None, "Payload cannot be negative.")
    assert c.holds is False
    assert 0.001 < c.violation_rate < 0.02


def test_manifest_reconciliation_breaks(df):
    """Fault 6: route pallets stop summing to the vehicle manifest."""
    c = physics.check_conservation(
        df, ["route_a_pallets", "route_b_pallets", "route_c_pallets"],
        "vehicle_manifest_pallets", 0.02,
        "Pallets loaded per route must sum to the vehicle manifest.")
    assert c.holds is False
    assert c.violation_rate > 0.3


def test_reefer_thermal_rate_limit_violated(df):
    """Fault 9: a reefer body cannot move 30 C between consecutive samples."""
    c = physics.check_rate_limit(
        df, "reefer_return_air_c", 0.05, "timestamp",
        "An insulated reefer body has thermal inertia measured in minutes.")
    assert c.holds is False


def test_odometer_reset_is_visible_but_below_the_population_threshold(df):
    """Fault 1, and a documented limitation rather than a success.

    The rate-limit check adjudicates on the *fraction* of violating intervals,
    which is the right rule for transmission glitches but the wrong one for a
    single catastrophic discontinuity: one odometer reset in 5,000 samples does
    not move a rate. The evidence carries it -- an observed maximum three orders
    of magnitude above the p99.9 -- but the verdict is still `holds`.

    Recovering this properly needs monotonicity-in-time for cumulative counters,
    which is not currently one of the five constraint kinds. Asserting the real
    behaviour here keeps that gap visible instead of letting a tuned fixture
    imply coverage the tool does not have.
    """
    c = physics.check_rate_limit(df, "distance_km", 0.05, "timestamp",
                                 "A cumulative odometer cannot decrease.")
    assert c.evidence["n_violations"] == 1
    assert c.evidence["observed_max_rate"] > 100 * c.evidence["observed_p999_rate"]
    assert c.holds is True  # the gap, asserted rather than hidden


def test_fuel_distance_identity_holds(df):
    """The identity that is *supposed* to hold. A tool that only ever confirms
    faults is not adjudicating anything."""
    c = physics.check_identity(df, "fuel_l - distance_km * 0.31", 0.01,
                               "Declared consumption is 0.31 L/km.")
    assert c.holds is True


def test_report_scores_every_dimension(report):
    assert len(report.scores) == 7
    assert 0.0 <= report.overall_score <= 5.0
