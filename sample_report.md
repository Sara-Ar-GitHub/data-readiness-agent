# Data readiness assessment — line3_press_shop
*Sector: manufacturing*

**Overall maturity: 3.9 / 5**

## Maturity by dimension

| Dimension | Score |
|---|---|
| completeness | 3.80 / 5 |
| validity | 3.40 / 5 |
| consistency | 5.00 / 5 |
| timeliness | 3.40 / 5 |
| uniqueness | 3.80 / 5 |
| interoperability | 5.00 / 5 |
| traceability | 1.40 / 5 |

## Dataset structure

- 6,000 rows x 15 columns
- Duplicate rows: 0.00%
- Timestamp column: `timestamp`
- Median sampling interval: 10.1 s
- Sampling regularity: 36.7%
- Gaps beyond 5 sampling periods: 3

## Inferred lineage

- `power_kw_duplicate` <- `power_kw` (duplicate, confidence 1.0000)
- `energy_kwh` <- `power_kw` (scaled (x2.778), confidence 1.0000)
- `torque_nm` <- `speed_rpm`, `power_kw` (linear combination, confidence 0.9972)

## Findings

### Major (7)

- [MAJOR] **completeness**, columns: `torque_nm` — torque_nm is missing in 17.6% of rows.
  - *Remediation:* Determine whether gaps are random or coincide with plant states; the latter biases any model trained on this tag.
- [MAJOR] **validity**, columns: `bearing_temp_c` — bearing_temp_c holds one identical value for up to 400 consecutive samples, indicating a stuck transmitter or a frozen historian tag.
  - *Remediation:* Cross-check against the instrument's maintenance log. Mask frozen segments before training rather than dropping the tag.
- [MAJOR] **timeliness**, columns: `timestamp` — Only 37% of sampling intervals are close to the median rate.
  - *Remediation:* Resample onto a fixed grid with an explicit aggregation rule, and record which samples were interpolated.
- [MAJOR] **uniqueness**, columns: `timestamp` — 2.00% of timestamps repeat.
  - *Remediation:* Usually a clock reset or a merge of overlapping exports. Resolve before treating the series as a single asset.
- [MAJOR] **traceability**, columns: `power_kw_duplicate` — power_kw_duplicate is reproducible from power_kw (duplicate, R=1.0000) and is therefore a derived quantity, not an independent measurement.
  - *Remediation:* Keep either the parents or the derivative, never both, or the model will appear accurate while learning an identity.
- [MAJOR] **traceability**, columns: `energy_kwh` — energy_kwh is reproducible from power_kw (scaled (x2.778), R=1.0000) and is therefore a derived quantity, not an independent measurement.
  - *Remediation:* Keep either the parents or the derivative, never both, or the model will appear accurate while learning an identity.
- [MAJOR] **traceability**, columns: `torque_nm` — torque_nm is reproducible from speed_rpm, power_kw (linear combination, R=0.9972) and is therefore a derived quantity, not an independent measurement.
  - *Remediation:* Keep either the parents or the derivative, never both, or the model will appear accurate while learning an identity.

### Minor (2)

- [minor] **validity**, columns: `plant_id` — plant_id never varies and carries no information.
  - *Remediation:* Drop from the feature set, or confirm the sensor is live.
- [minor] **timeliness**, columns: `timestamp` — 3 gaps exceed five sampling periods.
  - *Remediation:* Check whether gaps align with shutdowns or with historian outages; only the latter should be interpolated.

### Info (8)

- [info] **interoperability**, columns: `speed_rpm` — speed_rpm appears to carry a rotational_speed quantity, but no unit is declared anywhere in the schema.
  - *Remediation:* Attach explicit units and a semantic identifier. Undeclared units are the most common cause of silent errors when merging data across sites.
- [info] **interoperability**, columns: `torque_nm` — torque_nm appears to carry a torque quantity, but no unit is declared anywhere in the schema.
  - *Remediation:* Attach explicit units and a semantic identifier. Undeclared units are the most common cause of silent errors when merging data across sites.
- [info] **interoperability**, columns: `power_kw` — power_kw appears to carry a power quantity, but no unit is declared anywhere in the schema.
  - *Remediation:* Attach explicit units and a semantic identifier. Undeclared units are the most common cause of silent errors when merging data across sites.
- [info] **interoperability**, columns: `power_kw_duplicate` — power_kw_duplicate appears to carry a power quantity, but no unit is declared anywhere in the schema.
  - *Remediation:* Attach explicit units and a semantic identifier. Undeclared units are the most common cause of silent errors when merging data across sites.
- [info] **interoperability**, columns: `energy_kwh` — energy_kwh appears to carry a energy quantity, but no unit is declared anywhere in the schema.
  - *Remediation:* Attach explicit units and a semantic identifier. Undeclared units are the most common cause of silent errors when merging data across sites.
- [info] **interoperability**, columns: `bearing_temp_c` — bearing_temp_c appears to carry a temperature quantity, but no unit is declared anywhere in the schema.
  - *Remediation:* Attach explicit units and a semantic identifier. Undeclared units are the most common cause of silent errors when merging data across sites.
- [info] **interoperability**, columns: `ambient_temp_c` — ambient_temp_c appears to carry a temperature quantity, but no unit is declared anywhere in the schema.
  - *Remediation:* Attach explicit units and a semantic identifier. Undeclared units are the most common cause of silent errors when merging data across sites.
- [info] **interoperability**, columns: `coolant_pressure_bar` — coolant_pressure_bar appears to carry a pressure quantity, but no unit is declared anywhere in the schema.
  - *Remediation:* Attach explicit units and a semantic identifier. Undeclared units are the most common cause of silent errors when merging data across sites.

## Prioritised roadmap

| # | Action | Effort | Unlocks |
|---|---|---|---|
| 1 | Determine whether gaps are random or coincide with plant states; the latter biases any model trained on this tag. | medium | any supervised model on the affected tags |
| 2 | Resample onto a fixed grid with an explicit aggregation rule, and record which samples were interpolated. | low | time-series forecasting and drift monitoring |
| 3 | Keep either the parents or the derivative, never both, or the model will appear accurate while learning an identity. | low | leakage-free feature selection |
| 4 | Usually a clock reset or a merge of overlapping exports. Resolve before treating the series as a single asset. | low | honest train/test separation |
| 5 | Cross-check against the instrument's maintenance log. Mask frozen segments before training rather than dropping the tag. | medium | trustworthy anomaly detection |
| 6 | Check whether gaps align with shutdowns or with historian outages; only the latter should be interpolated. | low | time-series forecasting and drift monitoring |
| 7 | Drop from the feature set, or confirm the sensor is live. | medium | trustworthy anomaly detection |
