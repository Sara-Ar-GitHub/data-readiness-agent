"""Generate a synthetic industrial dataset with deliberately planted faults.

Every fault is documented here so the repo can demonstrate that the agent
recovers exactly the problems that were injected -- a small evaluation
harness rather than a demo that merely produces output.

Planted faults:
  1. torque sensor missing 18% of the time (completeness)
  2. bearing temperature frozen for a 400-sample block (stuck sensor)
  3. coolant pressure occasionally negative (physically impossible)
  4. power_kw_duplicate is an exact copy of power_kw (lineage / leakage)
  5. energy_kwh is a linear function of power_kw (derived column)
  6. sum of three line stations does not match declared total (conservation)
  7. 2% duplicated timestamps (clock reset)
  8. irregular sampling with three long gaps (timeliness)
  9. ambient temperature spikes 45 C between consecutive samples (rate limit)
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 6000

ts = pd.date_range("2026-01-01", periods=n, freq="10s").to_series().reset_index(drop=True)
jitter = pd.to_timedelta(rng.normal(0, 1.5, n).clip(-4, 4), unit="s")
ts = ts + jitter
for start in (1200, 2900, 4400):  # three historian outages
    ts.iloc[start:] = ts.iloc[start:] + pd.Timedelta(minutes=25)

speed = 1500 + 120 * np.sin(np.arange(n) / 180) + rng.normal(0, 8, n)
torque = 40 + 12 * np.sin(np.arange(n) / 95) + rng.normal(0, 1.2, n)
power = torque * speed * 2 * np.pi / 60 / 1000  # kW, exact identity

bearing_temp = 55 + 0.012 * power * 60 + rng.normal(0, 0.8, n)
bearing_temp[2000:2400] = bearing_temp[2000]  # fault 2: frozen tag

ambient = 21 + 3 * np.sin(np.arange(n) / 700) + rng.normal(0, 0.3, n)
ambient[[900, 3300, 5100]] += 45  # fault 9: impossible thermal jump

pressure = 4.2 + rng.normal(0, 0.15, n)
pressure[rng.choice(n, 30, replace=False)] = -rng.uniform(0.5, 2.0, 30)  # fault 3

st_a = 30 + rng.normal(0, 2, n)
st_b = 25 + rng.normal(0, 2, n)
st_c = 20 + rng.normal(0, 2, n)
total = st_a + st_b + st_c
total[3000:] *= 1.18  # fault 6: meter drift after a replacement

df = pd.DataFrame({
    "timestamp": ts,
    "speed_rpm": speed,
    "torque_nm": torque,
    "power_kw": power,
    "power_kw_duplicate": power,                 # fault 4
    "energy_kwh": power * (10 / 3600) * 1000,    # fault 5
    "bearing_temp_c": bearing_temp,
    "ambient_temp_c": ambient,
    "coolant_pressure_bar": pressure,
    "station_a_units": st_a,
    "station_b_units": st_b,
    "station_c_units": st_c,
    "line_total_units": total,
    "shift_code": rng.choice(["A", "B", "C"], n),
    "plant_id": "LUX-01",                        # constant column
})

mask = rng.random(n) < 0.18                      # fault 1
df.loc[mask, "torque_nm"] = np.nan

dup_idx = rng.choice(n, int(0.02 * n), replace=False)  # fault 7
df.loc[dup_idx, "timestamp"] = df["timestamp"].iloc[0]

df.to_csv("examples/line3_press_shop.csv", index=False)
print(f"Wrote examples/line3_press_shop.csv  ({len(df)} rows, {df.shape[1]} cols)")
