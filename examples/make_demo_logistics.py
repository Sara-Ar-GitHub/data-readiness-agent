"""Generate a synthetic cold-chain logistics dataset with planted faults.

A second sector, for a reason that is methodological rather than decorative.
A tool validated on one dataset has demonstrated that it works on that
dataset. The claim worth making is that the *abstraction* holds -- that
"physical constraint" and "derived column" mean something outside the press
shop the tool was written against.

Cold-chain distribution is a good adversarial second case because its
constraints have the same shape as a press shop's but none of the same
vocabulary. There is no torque and no shaft power. There is:

  - a temperature with thermal inertia (reefer setpoint vs return air)
  - a conserved quantity across a hierarchy (pallets per route must sum to
    the vehicle manifest)
  - an algebraic identity (fuel burned = litres per km x distance)
  - a bounded absolute quantity (payload cannot be negative)

Nothing in the assessment layer is told which sector it is looking at. If the
same seven readiness dimensions and the same five constraint kinds recover the
faults here, the abstraction is doing work.

Planted faults:
  1. GPS odometer resets to zero mid-route (invalid, breaks distance identity)
  2. reefer return-air probe frozen for a 300-sample block (stuck sensor)
  3. payload occasionally negative after a tare-weight miscalibration
  4. distance_mi is a unit conversion of distance_km (derived / leakage)
  5. fuel_l is an exact linear function of distance_km (derived column)
  6. pallets on three routes do not sum to the vehicle manifest (conservation)
  7. 3% duplicated consignment IDs (uniqueness)
  8. telemetry gaps when vehicles cross a dead zone (timeliness)
  9. reefer temperature jumps 30 C between consecutive samples (rate limit)
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
n = 5000

ts = pd.date_range("2026-03-01", periods=n, freq="30s").to_series().reset_index(drop=True)
ts = ts + pd.to_timedelta(rng.normal(0, 3, n).clip(-8, 8), unit="s")
for start in (800, 2600, 4100):  # telemetry dead zones -- fault 8
    ts.iloc[start:] = ts.iloc[start:] + pd.Timedelta(minutes=40)

speed_kph = np.clip(62 + 18 * np.sin(np.arange(n) / 220) + rng.normal(0, 6, n), 0, 110)
distance_km = np.cumsum(speed_kph * (30 / 3600))
distance_km[3200:] = np.cumsum(speed_kph[3200:] * (30 / 3600))  # fault 1: odometer reset

setpoint = np.full(n, -18.0)
return_air = setpoint + 1.4 + 0.8 * np.sin(np.arange(n) / 400) + rng.normal(0, 0.25, n)
return_air[1500:1800] = return_air[1500]          # fault 2: frozen probe
return_air[[600, 2400, 3900]] += 30               # fault 9: impossible thermal jump

payload_kg = 8200 + 900 * np.sin(np.arange(n) / 500) + rng.normal(0, 60, n)
payload_kg[rng.choice(n, 24, replace=False)] = -rng.uniform(50, 400, 24)  # fault 3

route_a = 14 + rng.integers(-2, 3, n)
route_b = 11 + rng.integers(-2, 3, n)
route_c = 9 + rng.integers(-2, 3, n)
manifest = (route_a + route_b + route_c).astype(float)
manifest[2500:] += 6                              # fault 6: manifest not reconciled

df = pd.DataFrame({
    "timestamp": ts,
    "vehicle_speed_kph": speed_kph,
    "distance_km": distance_km,
    "distance_mi": distance_km * 0.621371,        # fault 4
    "fuel_l": distance_km * 0.31,                 # fault 5
    "reefer_setpoint_c": setpoint,                # constant column
    "reefer_return_air_c": return_air,
    "payload_kg": payload_kg,
    "route_a_pallets": route_a.astype(float),
    "route_b_pallets": route_b.astype(float),
    "route_c_pallets": route_c.astype(float),
    "vehicle_manifest_pallets": manifest,
    "depot_code": rng.choice(["LUX-N", "LUX-S", "ARL"], n),
    "consignment_id": [f"C{i:06d}" for i in range(n)],
})

dup = rng.choice(n, int(0.03 * n), replace=False)  # fault 7
df.loc[dup, "consignment_id"] = df["consignment_id"].iloc[0]

df.to_csv("examples/fleet_cold_chain.csv", index=False)
print(f"Wrote examples/fleet_cold_chain.csv  ({len(df)} rows, {df.shape[1]} cols)")
