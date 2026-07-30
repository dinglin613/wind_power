"""
Train / val / test split + sliding-window construction for the Kelmarsh point-forecasting setup.

Self-contained reproduction of how the data is split and windowed (identical logic to the
project's `build_datasets`), so results are reproducible from the raw CSV alone.

Split policy
------------
- Split by YEAR:  train = {2017, 2018},  val = {2019},  test = {2020}.
- Sliding window: L=120 h look-back + H=24 h forecast horizon, inside an L_buffer=144 h history
  buffer (so the model input is the last L hours of the buffer). Total window W = L_buffer + H = 168 h.
- A window is VALID only if every hour in it has is_missing == 0 (no gap inside the 7-day window).
- Window step: train uses train_step = 6 h (overlapping windows, more data);
  val / test use step = H = 24 h (NON-overlapping horizons, so the eval set is not double-counted).
- Normalization statistics (mean/std) are computed on the TRAIN clean rows ONLY and reused for
  val/test (no test information leaks into the fit).

Run:  python data_split.py
"""
import numpy as np
import pandas as pd

CSV = "data/kelmarsh_farm_fused_2017_2020_preprocessed_origin.csv"

# ----- window / split configuration (matches the project config) -----
L, H, L_BUFFER = 120, 24, 144            # look-back, horizon, history buffer (hours)
W = L_BUFFER + H                          # 168 h total valid window
TRAIN_YEARS, VAL_YEARS, TEST_YEARS = [2017, 2018], [2019], [2020]
TRAIN_STEP = 6                            # train window step (h); val/test step = H (non-overlapping)

SCADA_FEATURES = ["wind_speed", "temperature", "wind_dir_sin", "wind_dir_cos", "power"]
NWP_FEATURES = ["windspeed_100m", "temperature_2m", "nwp_wind_dir_sin", "nwp_wind_dir_cos"]
TARGET = "power"
V_LOW, V_HIGH = 3.0, 12.5                 # cut-in / rated wind (m/s) → regime labels EL / NORMAL / EH


def valid_start_indices(is_missing: np.ndarray, step: int):
    """Start indices i such that the whole window [i, i+W) has no missing hour.
    History = last L hours of the buffer [i+L_buffer-L, i+L_buffer); horizon = [i+L_buffer, i+W)."""
    miss_cum = np.concatenate([[0], np.cumsum(is_missing)])
    idx, i = [], 0
    while i <= len(is_missing) - W:
        if miss_cum[i + W] - miss_cum[i] == 0:        # no gap in the window
            idx.append(i); i += step
        else:
            i += 1                                     # slide past the missing hour
    return np.array(idx, dtype=int)


def label_regime(wind_speed: float) -> int:
    """Physical regime by MEASURED wind: 0=extreme-low (<cut-in), 1=normal, 2=extreme-high (>=rated)."""
    if wind_speed < V_LOW:
        return 0
    if wind_speed >= V_HIGH:
        return 2
    return 1


def build_split():
    df = pd.read_csv(CSV, parse_dates=["date_time"]).sort_values("date_time").reset_index(drop=True)
    df["year"] = df["date_time"].dt.year

    # NWP wind direction -> sin/cos (if not already present)
    if "nwp_wind_dir_sin" not in df.columns and "winddirection_100m" in df.columns:
        rad = np.deg2rad(df["winddirection_100m"].values)
        df["nwp_wind_dir_sin"], df["nwp_wind_dir_cos"] = np.sin(rad), np.cos(rad)

    df_train = df[df["year"].isin(TRAIN_YEARS)].reset_index(drop=True)
    df_val = df[df["year"].isin(VAL_YEARS)].reset_index(drop=True)
    df_test = df[df["year"].isin(TEST_YEARS)].reset_index(drop=True)

    # normalization stats from TRAIN clean rows only, reused for val/test
    tr_clean = df_train[df_train["is_missing"] == 0]
    scada_mean = tr_clean[SCADA_FEATURES].mean().values
    scada_std = tr_clean[SCADA_FEATURES].std().values + 1e-8
    nwp_mean = tr_clean[NWP_FEATURES].mean().values
    nwp_std = tr_clean[NWP_FEATURES].std().values + 1e-8
    target_mean = float(tr_clean[TARGET].mean())
    target_std = float(tr_clean[TARGET].std()) + 1e-8

    splits = {}
    for name, d, step in [("train", df_train, TRAIN_STEP),
                          ("val", df_val, H), ("test", df_test, H)]:
        idx = valid_start_indices(d["is_missing"].values.astype(np.int32), step)
        splits[name] = {"df": d, "start_indices": idx}
        print(f"{name:5s}: {len(idx):5d} windows  ({d['year'].unique().tolist()})")

    return splits, dict(scada=(scada_mean, scada_std), nwp=(nwp_mean, nwp_std),
                        target=(target_mean, target_std))


def get_sample(df: pd.DataFrame, i: int):
    """Materialize one (x_scada, x_nwp, y, regimes) sample from a valid start index i.
    x_scada: (L, 5) history SCADA ; x_nwp: (H, 4) horizon NWP forecast ; y: (H,) horizon power."""
    hist = slice(i + L_BUFFER - L, i + L_BUFFER)      # last L hours of the buffer
    hor = slice(i + L_BUFFER, i + L_BUFFER + H)       # H-hour horizon
    x_scada = df[SCADA_FEATURES].values[hist]
    x_nwp = df[NWP_FEATURES].values[hor]
    x_nwp_hist = df[NWP_FEATURES].values[hist]        # NWP archive over the history window
    y = df[TARGET].values[hor]
    regimes = np.array([label_regime(w) for w in df["wind_speed"].values[hor]])
    return x_scada, x_nwp, x_nwp_hist, y, regimes


if __name__ == "__main__":
    splits, stats = build_split()
    xs, xn, xnh, y, reg = get_sample(splits["test"]["df"], splits["test"]["start_indices"][0])
    print("\nexample test sample shapes:",
          "x_scada", xs.shape, "| x_nwp", xn.shape, "| y", y.shape,
          "| regimes(EL/NM/EH counts):", np.bincount(reg, minlength=3).tolist())
