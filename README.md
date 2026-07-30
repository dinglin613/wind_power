# Kelmarsh wind-power point forecasting — data & split

Data and the exact train / validation / test split used for the point-forecasting experiments on the
Kelmarsh wind farm (2017–2020).

## Contents
- `data/kelmarsh_farm_fused_2017_2020_preprocessed_origin.csv` — hourly, preprocessed & fused SCADA + NWP.
- `data_split.py` — self-contained reproduction of the split + sliding-window construction.

## Task
24-hour-ahead power forecasting. Model input at time *t*:
- history SCADA over the last **L = 120 h**, and
- the NWP forecast over the next **H = 24 h**;
- output: power over the next **H = 24 h**.

## Split policy (`data_split.py`)
| split | years | window step |
|-------|-------|-------------|
| train | 2017, 2018 | 6 h (overlapping) |
| val   | 2019 | 24 h (non-overlapping horizons) |
| test  | 2020 | 24 h (non-overlapping horizons) |

- Each sample sits in a 7-day window `W = L_buffer(144) + H(24) = 168 h`; the model input is the **last
  L=120 h** of the 144 h buffer. A window is kept only if **every hour has `is_missing == 0`**.
- Normalization mean/std are fit on **train clean rows only** and reused for val/test (no leakage).
- Regime label per horizon hour from **measured wind speed**: extreme-low `< 3.0 m/s`,
  normal `3.0–12.5`, extreme-high `≥ 12.5 m/s`.

Window counts produced: **train 2290 · val 315 · test 254**.

## Run
```bash
pip install numpy pandas
python data_split.py
```

## Key columns
`date_time, power, wind_speed, temperature, wind_dir_sin, wind_dir_cos, is_missing` (SCADA) ·
`windspeed_100m, temperature_2m, winddirection_100m` (NWP). Wind direction is encoded as sin/cos.
