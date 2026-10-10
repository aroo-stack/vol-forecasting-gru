"""Volatility forecasting — Phase 1: data pipeline + leak-proof splits + baselines.

No neural network is built in this phase. Phase 2 (GRU) will load
data/processed.npz directly, so the arrays saved here are the contract.

Target (per ticker, per prediction date t):
    y_t = log( sqrt( mean( r_{t+1..t+5}^2 ) ) + 1e-8 )
where r is the daily log return. Uses the NEXT 5 trading days only.

Features (per date t, using data up to and including close of day t ONLY):
    - daily log return
    - absolute return
    - rolling std of returns over 5, 10, 21 days (backward-looking)
    - log volume change log(volume_t / volume_{t-1}), guarded vs zero
    - VIX close level aligned by trading date

One sample = last 40 days of features ending at t -> (40 x 7) + target y_t.

Splits are chronological by prediction date t (never shuffled):
    train: t <= 2019-12-31 | val: 2020-01-01..2021-12-31 | test: t >= 2022-01-01
(Data starts 2005-01-01, so train includes the 2008 crisis.)
The final 5 samples of train and val (per ticker) are dropped so no
5-day target window overlaps into the next split's date range.
Scaler (mean/std) is fit on TRAIN windows only, then applied to val/test.

Baselines: naive (past 5d log rv), HAR (OLS on log rv 5/10/21d, train only),
HAR-bc (HAR + 0.5 * train residual variance), GARCH(1,1) (fit once on train,
fixed params rolled forward for each test/val date).
"""

import os
import warnings

import numpy as np
import pandas as pd

np.random.seed(0)
try:
    import torch
    torch.manual_seed(0)
except Exception:
    pass
import random
random.seed(0)

# ---------------------------------------------------------------- config
TICKERS = ["SPY", "AAPL", "MSFT", "JPM", "XOM", "^VIX"]
FORECAST_TICKERS = ["SPY", "AAPL", "MSFT", "JPM", "XOM"]  # VIX is feature-only
START_DATE = "2005-01-01"   # longer history so train includes the 2008 crisis
LATE_CUTOFF = "2005-06-01"  # warn loudly if any ticker starts after this
DATA_DIR = "data"
PROCESSED_PATH = os.path.join(DATA_DIR, "processed.npz")
SCALER_PATH = os.path.join(DATA_DIR, "scaler.npz")

WINDOW = 40          # input window length (days)
HORIZON = 5          # target horizon (trading days)
EPS = 1e-8
TRAIN_END = "2019-12-31"
VAL_START, VAL_END = "2020-01-01", "2021-12-31"
TEST_START = "2022-01-01"

FEATURE_COLS = ["log_ret", "abs_ret", "std_5", "std_10", "std_21",
                "log_vol_chg", "vix_close"]

# ---------------------------------------------------------------- helpers

def safe_ticker_fname(ticker: str) -> str:
    """Map a ticker to a cache filename (^VIX -> VIX.csv)."""
    return ticker.replace("^", "").replace("/", "_") + ".csv"


def download_or_load(ticker: str, start: str = START_DATE) -> pd.DataFrame:
    """Download daily OHLCV with yfinance, or load data/<ticker>.csv cache.

    If a cache exists but starts much later than the requested start date
    (stale short-history cache), it is ignored and the full history is
    re-downloaded. A 10-day grace margin covers weekends/holidays (e.g.
    start=2005-01-01, a Saturday, vs first trading day 2005-01-03).
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, safe_ticker_fname(ticker))
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
        df = df.sort_index()
        if df.index.min() <= pd.Timestamp(start) + pd.Timedelta(days=10):
            print(f"  [cache] {ticker} <- {path}")
            return df
        print(f"  [stale cache] {ticker} starts {df.index.min().date()}, "
              f"need {start}; re-downloading ...")
    print(f"  [download] {ticker} from {start} ...")
    import yfinance as yf
    # auto_adjust=True => 'Close' is the dividend/split-adjusted close, so
    # log returns don't show artificial jumps on ex-div dates.
    df = yf.download(ticker, start=start, auto_adjust=True,
                     progress=False, timeout=60)
    if df is None or len(df) == 0:
        raise RuntimeError(f"yfinance returned no data for {ticker}")
    # yfinance may return MultiIndex columns (Price x Ticker); flatten them.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df = df.sort_index()
    df.to_csv(path)
    print(f"  [saved] {ticker} -> {path} ({len(df)} rows)")
    return df


def build_daily_frame(px: pd.DataFrame, vix: pd.DataFrame,
                       drop_na: bool = True) -> pd.DataFrame:
    """Per-ticker daily feature/target frame. All features use data <= t only.

    px: raw OHLCV for one equity ticker (must have Close, Volume).
    vix: raw OHLCV for ^VIX (we only use its Close).
    Returns DataFrame indexed by trading date with FEATURE_COLS + 'target'
    plus helper columns '_close' (for target verification) and '_log_ret'.
    Rows with any NaN are dropped.
    """
    close = px["Close"].astype(float)
    vol = px["Volume"].astype(float)
    # Daily log return at t uses closes t-1 and t only.
    log_ret = np.log(close / close.shift(1))
    abs_ret = log_ret.abs()
    # Backward-looking rolling stds: window ending at t (uses returns <= t).
    std_5 = log_ret.rolling(5, min_periods=5).std()
    std_10 = log_ret.rolling(10, min_periods=10).std()
    std_21 = log_ret.rolling(21, min_periods=21).std()
    # Guard against zero volume: clip to >=1 share before taking the ratio.
    vol_safe = vol.clip(lower=1.0)
    log_vol_chg = np.log(vol_safe / vol_safe.shift(1))
    # VIX close aligned by trading date (inner join -> drop NA). No ffill,
    # so the VIX value at t is the VIX close actually observed on date t.
    vix_close = vix["Close"].astype(float).reindex(close.index)

    frame = pd.DataFrame({
        "log_ret": log_ret, "abs_ret": abs_ret,
        "std_5": std_5, "std_10": std_10, "std_21": std_21,
        "log_vol_chg": log_vol_chg, "vix_close": vix_close,
        "_close": close,
    })
    # Target at t = log realized vol over NEXT 5 trading days t+1..t+5.
    # Written as an explicit sum of 5 forward shifts so the direction is
    # obvious: shift(-k) is the return k days AFTER t. Never uses r_t or past.
    r2 = log_ret ** 2
    fwd_mean = (r2.shift(-1) + r2.shift(-2) + r2.shift(-3)
                + r2.shift(-4) + r2.shift(-5)) / 5.0
    frame["target"] = np.log(np.sqrt(fwd_mean) + EPS)
    # Drop warmup rows (rolling windows) and the last 5 rows (no target).
    # drop_na=False keeps every row (target is NaN in the last 5); used by
    # the leakage check, which truncates data at t and compares features
    # at t (features at t never need data after t, the target does).
    if drop_na:
        frame = frame.dropna()
    return frame


def frame_to_samples(frame: pd.DataFrame, ticker: str):
    """Turn a daily frame into (40 x 7) window samples ending at each date t."""
    feats = frame[FEATURE_COLS].to_numpy(dtype=np.float64)
    dates = frame.index
    out_X, out_y, out_tick, out_date = [], [], [], []
    # Window ending at row i covers rows i-39..i, i.e. dates d_{i-39}..d_i = t.
    for i in range(WINDOW - 1, len(frame)):
        out_X.append(feats[i - WINDOW + 1:i + 1])
        out_y.append(float(frame["target"].iloc[i]))
        out_tick.append(ticker)
        out_date.append(dates[i])
    return out_X, out_y, out_tick, out_date


# ---------------------------------------------------------------- baselines

def backward_log_rv(log_ret: pd.Series, n: int) -> pd.Series:
    """Log realized vol over the PAST n days ending at t (for naive/HAR)."""
    m = (log_ret ** 2).rolling(n, min_periods=n).mean()
    return np.log(np.sqrt(m) + EPS)


def garch_forecasts(train_ret: np.ndarray, full_ret: np.ndarray,
                    test_pos: np.ndarray) -> np.ndarray:
    """5-day-ahead GARCH(1,1) forecasts in target (log-vol) units.

    Fit GARCH(1,1) ONCE on train returns*100 per ticker. Then, for every test
    date t, roll the conditional variance forward 5 steps using the FIXED
    fitted (omega, alpha, beta) and only returns observed up to and
    including t. Never refits on test data; never uses returns after t.

    Recursion: sigma²_{k+1} = omega + alpha*r_k² + beta*sigma²_k (in 100x
    units). Multi-step forecast: step 1 = filtered sigma²_{t+1}; steps 2..5
    use E[r²] = sigma², i.e. sigma²_{t+h} = omega + (alpha+beta)*sigma²_{t+h-1}.
    The 5 variances are averaged, unscaled (/100²), sqrt'd and logged to
    match y_t = log(sqrt(mean variance) + eps).
    """
    from arch import arch_model
    scaled = train_ret * 100.0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = arch_model(scaled, mean="Zero", vol="Garch", p=1, q=1,
                             rescale=False).fit(disp="off")
        omega = float(res.params["omega"])
        alpha = float(res.params["alpha[1]"])
        beta = float(res.params["beta[1]"])
    except Exception as e:
        print(f"    [garch] fit failed ({e}); falling back to train variance.")
        v = float(np.mean(scaled ** 2)) + 1e-12
        omega, alpha, beta = v * 0.05, 0.05, 0.90
    # Filtered variance over the FULL return history (in 100x units),
    # starting from the unconditional variance. For test date t at position
    # i, filtered[i+1] depends only on returns <= t. We compute the whole
    # recursion once, then read off each forecast origin (still uses <= t).
    r100 = full_ret * 100.0
    uncond = omega / max(1e-8, (1.0 - alpha - beta)) if alpha + beta < 1 else float(np.mean(r100 ** 2) + 1e-12)
    filt = np.empty(len(r100) + 1)
    filt[0] = uncond
    for k in range(len(r100)):
        filt[k + 1] = omega + alpha * r100[k] ** 2 + beta * filt[k]
    persist = alpha + beta
    preds = np.empty(len(test_pos))
    for j, i in enumerate(test_pos):
        s = filt[i + 1]  # sigma²_{t+1} given data <= t
        acc, cur = s, s
        for _ in range(2, HORIZON + 1):
            cur = omega + persist * cur
            acc += cur
        mean_var_100x = acc / HORIZON
        mean_var = mean_var_100x / (100.0 ** 2)  # undo 100x scaling
        preds[j] = float(np.log(np.sqrt(max(mean_var, 0.0)) + EPS))
    return preds


# ---------------------------------------------------------------- metrics

def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def qlike(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean QLIKE on variances: r = var_true/var_pred; r - log r - 1.

    var = (exp(y))² since y is log-vol. Standard, scale-free loss that
    penalizes under-prediction more than over-prediction.
    """
    vt = np.exp(2.0 * np.asarray(y_true, dtype=float))
    vp = np.exp(2.0 * np.asarray(y_pred, dtype=float))
    vp = np.maximum(vp, 1e-16)
    r = vt / vp
    return float(np.mean(r - np.log(np.maximum(r, 1e-16)) - 1.0))


def print_table(title: str, tickers: list, y_true_d: dict, preds_d: dict):
    """Print RMSE/QLIKE per ticker + pooled overall for one split."""
    print(f"\n=== {title} ===")
    print(f"{'ticker':<6} {'n':>6} {'model':<8} {'RMSE':>8} {'QLIKE':>10}")
    for model in ["naive", "har", "har-bc", "garch"]:
        for t in tickers + ["OVERALL"]:
            if t == "OVERALL":
                yt = np.concatenate([y_true_d[k] for k in tickers])
                yp = np.concatenate([preds_d[model][k] for k in tickers])
            else:
                yt, yp = y_true_d[t], preds_d[model][t]
            print(f"{t:<6} {len(yt):>6} {model:<8} {rmse(yp, yt):>8.4f} "
                  f"{qlike(yt, yp):>10.4f}")


def print_regime_table(tickers: list, splits: dict):
    """Mean/max of the target y (log realized vol) per split — regime check.

    splits: {name: (y_pooled, {ticker: y})}. exp(mean) is the geometric-mean
    daily-vol level, shown only to give the log numbers a human scale.
    """
    print("\n=== Realized-vol regime by split (target y = log vol) ===")
    print(f"{'split':<6} {'ticker':<6} {'n':>6} {'mean_y':>8} "
          f"{'max_y':>8} {'typ_vol%':>9}")
    for name, (y_all, y_by_t) in splits.items():
        for t in tickers + ["OVERALL"]:
            y = y_all if t == "OVERALL" else y_by_t[t]
            print(f"{name:<6} {t:<6} {len(y):>6} {np.mean(y):>8.4f} "
                  f"{np.max(y):>8.4f} {100 * np.exp(np.mean(y)):>8.3f}%")


def paired_bootstrap(y_true_d: dict, pred_a_d: dict, pred_b_d: dict,
                     tickers: list, name_a: str, name_b: str,
                     block: int = 10, n_boot: int = 2000, seed: int = 123):
    """Paired test of squared-error difference d = SE(A) - SE(B) on test data.

    d > 0 means B is better. Mean diff + 95% CI from a circular block
    bootstrap (block length 10 trading days) that resamples dates WITHIN
    each ticker (preserving autocorrelation) and pools across tickers.
    Per-ticker series are already in date order, so consecutive positions
    are consecutive prediction dates. Seed fixed for reproducibility.
    """
    rng = np.random.RandomState(seed)
    diffs = {}
    for t in tickers:
        yt = np.asarray(y_true_d[t], dtype=float)
        diffs[t] = (np.asarray(pred_a_d[t], dtype=float) - yt) ** 2 \
            - (np.asarray(pred_b_d[t], dtype=float) - yt) ** 2
    obs = float(np.mean(np.concatenate([diffs[t] for t in tickers])))
    boot = np.empty(n_boot)
    for b in range(n_boot):
        tot, cnt = 0.0, 0
        for t in tickers:
            d = diffs[t]
            n = len(d)
            starts = rng.randint(0, n, size=int(np.ceil(n / block)))
            idx = (starts[:, None] + np.arange(block)).ravel()[:n] % n
            tot += d[idx].sum()
            cnt += n
        boot[b] = tot / cnt
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"\n--- Paired SE diff on TEST: d = SE({name_a}) - SE({name_b}) "
          f"(d>0 => {name_b} better) ---")
    print(f"  pooled mean d = {obs:+.5f}, block-bootstrap 95% CI "
          f"[{lo:+.5f}, {hi:+.5f}] (L={block}, B={n_boot}, seed={seed})")
    verdict = ("significant" if (lo > 0 or hi < 0) else "not significant")
    winner = name_b if obs > 0 else name_a
    print(f"  => gap is {verdict} at 95%; point estimate favors {winner}.")
    print(f"  {'ticker':<6} {'mean_d':>9} (positive => {name_b} better)")
    for t in tickers:
        print(f"  {t:<6} {np.mean(diffs[t]):>+9.5f}")
    return obs, float(lo), float(hi)  # added for Phase 2 verdict (prints unchanged)


# ---------------------------------------------------------------- leakage checks

def check_split_order(all_dates: dict):
    for t, d in all_dates.items():
        assert d["train"].max() < d["val"].min(), f"{t}: train/val overlap"
        assert d["val"].max() < d["test"].min(), f"{t}: val/test overlap"
    print("  [check 1] split date order OK (max train < min val < ...).")


def check_scaler_train_only(X_train_raw: np.ndarray, scaler: dict):
    mu = X_train_raw.reshape(-1, X_train_raw.shape[-1]).mean(axis=0)
    sd = X_train_raw.reshape(-1, X_train_raw.shape[-1]).std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    assert np.allclose(mu, scaler["mean"]), "scaler mean not from train only!"
    assert np.allclose(sd, scaler["std"]), "scaler std not from train only!"
    print("  [check 2] scaler stats == train-window stats exactly.")


def check_recompute_features(px: pd.DataFrame, vix: pd.DataFrame,
                             frames: dict, rng_seed: int = 7):
    """Recompute features at date t from data truncated at t; must match."""
    rng = np.random.RandomState(rng_seed)
    n_ok = 0
    for _ in range(20):
        t = rng.choice(list(frames.keys()))
        frame = frames[t]
        # need a date with >=40 rows of history so the window is full
        i = rng.randint(WINDOW - 1, len(frame))
        dt = frame.index[i]
        # Delete everything after t and recompute from scratch. The target
        # at t needs returns after t, so keep all rows (drop_na=False) and
        # compare FEATURES only — features at t must not need data after t.
        px_tr = px[t][px[t].index <= dt]
        vix_tr = vix[vix.index <= dt]
        re = build_daily_frame(px_tr, vix_tr, drop_na=False)
        assert dt in re.index, f"{t} {dt}: date vanished on recompute"
        a = frame.loc[dt, FEATURE_COLS].to_numpy(float)
        b = re.loc[dt, FEATURE_COLS].to_numpy(float)
        assert np.allclose(a, b, rtol=1e-10, atol=1e-12), \
            f"{t} {dt.date()}: features changed after deleting data > t"
        n_ok += 1
    print(f"  [check 3] recompute-from-truncated-data OK ({n_ok}/20 rows).")


def check_target_uses_future_only(frames: dict, px: dict, rng_seed: int = 11):
    """Target at t must equal log-vol computed from returns t+1..t+5 only."""
    rng = np.random.RandomState(rng_seed)
    for _ in range(20):
        t = rng.choice(list(frames.keys()))
        frame = frames[t]
        i = rng.randint(0, len(frame))
        dt = frame.index[i]
        close = px[t]["Close"].astype(float)
        pos = close.index.get_loc(dt)
        fut = close.iloc[pos + 1:pos + 6]
        assert len(fut) == 5, "not 5 future bars"
        r = np.log(fut.to_numpy(float) / close.iloc[pos:pos + 5].to_numpy(float))
        expect = float(np.log(np.sqrt(np.mean(r ** 2)) + EPS))
        got = float(frame.loc[dt, "target"])
        assert abs(expect - got) < 1e-10, f"{t} {dt}: target mismatch"
    print("  [check 4] target uses exactly returns t+1..t+5 (20/20 rows).")


def manual_trace(frames: dict, px: dict, ticker: str = "SPY",
                 date: str | None = None) -> str:
    """Print + verify one full example: window dates, target dates, windows."""
    frame = frames[ticker]
    test = frame.loc[TEST_START:]
    dt = test.index[len(test) // 2] if date is None else pd.Timestamp(date)
    # snap to nearest available test date
    dt = test.index[test.index.get_indexer([dt], method="nearest")[0]]
    i = frame.index.get_loc(dt)
    close = px[ticker]["Close"].astype(float)
    pos = close.index.get_loc(dt)
    win_dates = list(frame.index[i - WINDOW + 1:i + 1])
    tgt_dates = list(close.index[pos + 1:pos + 6])
    lines = [f"\n--- MANUAL TRACE: {ticker} prediction date t = {dt.date()} ---",
             f"Input window ({WINDOW} days, last row = t):",
             f"  {win_dates[0].date()} ... {win_dates[-1].date()} "
             f"(={dt.date()}, matches t: {win_dates[-1] == dt})",
             "Target dates (next 5 trading days after t):",
             f"  {[d.date().isoformat() for d in tgt_dates]}",
             "Rolling-feature ranges at t (all backward-looking):",
             f"  std_5  uses returns {win_dates[-5].date()}..{win_dates[-1].date()} "
             f"(closes {frame.index[i-5].date()}..{dt.date()})",
             f"  std_10 uses returns {win_dates[-10].date()}..{win_dates[-1].date()}",
             f"  std_21 uses returns {win_dates[-21].date()}..{win_dates[-1].date()}",
             f"  log_vol_chg uses volumes {frame.index[i-1].date()}, {dt.date()}",
             f"  vix_close = VIX close on {dt.date()}"]
    assert win_dates[-1] == dt and all(d <= dt for d in win_dates), "input leaks future!"
    assert all(d > dt for d in tgt_dates) and len(tgt_dates) == 5, "target window wrong!"
    assert (frame.index[i-5] < dt) and (frame.index[i-1] < dt), "range error"
    lines.append("Verified: no input date after t; no target date on/before t.")
    # Show the target arithmetic explicitly.
    r = np.log(close.iloc[pos+1:pos+6].to_numpy(float)
               / close.iloc[pos:pos+5].to_numpy(float))
    lines.append(f"Target check: closes t+1..t+5 -> r={np.round(r,6).tolist()} "
                 f"-> y_t={float(np.log(np.sqrt(np.mean(r**2))+EPS)):.6f} "
                 f"(stored {float(frame.loc[dt,'target']):.6f})")
    return "\n".join(lines)


# ---------------------------------------------------------------- pipeline

def run_phase1():
    print("== Phase 1: data pipeline + leak-proof splits + baselines ==")
    print(f"TICKERS={TICKERS}  START={START_DATE}  WINDOW={WINDOW}  HORIZON={HORIZON}")

    # ---- 1) data
    print("[1/8] downloading / loading data ...")
    raw = {t: download_or_load(t) for t in TICKERS}
    vix = raw["^VIX"]
    for t in TICKERS:
        print(f"  {t}: first={raw[t].index.min().date()} "
              f"last={raw[t].index.max().date()} ({len(raw[t])} rows)")
    late = [t for t in TICKERS if raw[t].index.min() > pd.Timestamp(LATE_CUTOFF)]
    if late:
        print(f"  !!! WARNING: these tickers start later than {LATE_CUTOFF}: {late}")
    else:
        print(f"  all tickers start on or before {LATE_CUTOFF} — OK.")

    # ---- 2+3) targets + features
    print("[2/8] computing features + targets (≤t only / t+1..t+5) ...")
    frames = {t: build_daily_frame(raw[t], vix) for t in FORECAST_TICKERS}
    # keep raw px dict for verification (Close/Volume only, indexed by date)
    px = {t: raw[t] for t in FORECAST_TICKERS}
    for t, f in frames.items():
        print(f"  {t}: {len(f)} usable rows "
              f"({f.index.min().date()}..{f.index.max().date()})")

    # ---- windows + 4) chronological splits (per ticker, drop last 5 of train/val)
    print("[3/8] building 40-day windows + chronological splits ...")
    bounds = {"train": (None, TRAIN_END),
              "val": (VAL_START, VAL_END), "test": (TEST_START, None)}
    split_idx = {t: {} for t in FORECAST_TICKERS}  # ticker -> split -> list of positions
    Xs, ys, ticks, dates = {s: [] for s in bounds}, {s: [] for s in bounds}, \
        {s: [] for s in bounds}, {s: [] for s in bounds}
    for t, f in frames.items():
        X, y, tk, dt = frame_to_samples(f, t)
        dt = pd.DatetimeIndex(dt)
        for s, (lo, hi) in bounds.items():
            sel = np.ones(len(X), bool)
            if lo: sel &= dt >= pd.Timestamp(lo)
            if hi: sel &= dt <= pd.Timestamp(hi)
            idx = list(np.where(sel)[0])
            if s in ("train", "val"):
                idx = idx[:-5]  # drop final 5 so target can't overlap next split
            split_idx[t][s] = idx
            for k in idx:
                Xs[s].append(X[k]); ys[s].append(y[k])
                ticks[s].append(tk[k]); dates[s].append(dt[k])
    arrays = {}
    for s in bounds:
        order = np.argsort([pd.Timestamp(d) for d in dates[s]], kind="stable")
        arrays[s] = (np.stack(Xs[s])[order], np.array(ys[s], float)[order],
                     np.array(ticks[s], dtype=object)[order],
                     np.array([pd.Timestamp(d) for d in dates[s]])[order])
    (Xtr_raw, ytr, tktr, dttr), (Xva_raw, yva, tkva, dtva), \
        (Xte_raw, yte, tkte, dtte) = arrays["train"], arrays["val"], arrays["test"]
    print(f"  sizes: train {len(ytr)}, val {len(yva)}, test {len(yte)} "
          f"(window {WINDOW} x {len(FEATURE_COLS)})")

    # ---- scaler: fit on TRAIN only
    print("[4/8] fitting scaler on TRAIN only ...")
    flat = Xtr_raw.reshape(-1, Xtr_raw.shape[-1])
    mean, std = flat.mean(axis=0), flat.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    scaler = {"mean": mean, "std": std, "features": np.array(FEATURE_COLS)}
    np.savez(SCALER_PATH, **scaler)
    Xtr = (Xtr_raw - mean) / std
    Xva = (Xva_raw - mean) / std
    Xte = (Xte_raw - mean) / std
    np.savez(PROCESSED_PATH, X_train=Xtr, y_train=ytr,
             ticker_train=tktr,
             date_train=pd.DatetimeIndex(dttr).strftime("%Y-%m-%d").to_numpy(),
             X_val=Xva, y_val=yva,
             ticker_val=tkva,
             date_val=pd.DatetimeIndex(dtva).strftime("%Y-%m-%d").to_numpy(),
             X_test=Xte, y_test=yte,
             ticker_test=tkte,
             date_test=pd.DatetimeIndex(dtte).strftime("%Y-%m-%d").to_numpy(),
             features=np.array(FEATURE_COLS))
    print(f"  saved {PROCESSED_PATH} and {SCALER_PATH}")

    # ---- 5) baselines (same sample dates as the windows)
    print("[5/8] fitting + forecasting baselines ...")
    true, preds = {"test": {}, "val": {}}, \
        {"naive": {}, "har": {}, "har-bc": {}, "garch": {}}
    true_val = {}
    val_preds = {"naive": {}, "har": {}, "har-bc": {}, "garch": {}}
    for t in FORECAST_TICKERS:
        f = frames[t]
        lr = np.log(f["_close"] / f["_close"].shift(1))  # == log_ret incl. NaN head
        # backward log realized vols (past N days ending at t) for naive/HAR
        lrv = {n: backward_log_rv(lr, n) for n in (5, 10, 21)}
        # HAR: OLS on TRAIN sample dates only (this ticker's train dates).
        # train rows for this ticker = its train split dates present in frame
        tr_dates = pd.DatetimeIndex(dttr[np.array(tktr) == t])
        tr_idx = [f.index.get_loc(d) for d in tr_dates if d in f.index]
        A = np.column_stack([np.ones(len(tr_idx)),
                             lrv[5].iloc[tr_idx], lrv[10].iloc[tr_idx],
                             lrv[21].iloc[tr_idx]])
        coef, *_ = np.linalg.lstsq(A, f["target"].iloc[tr_idx].to_numpy(float),
                                   rcond=None)

        def har_at(dates):
            ii = [f.index.get_loc(d) for d in dates]
            B = np.column_stack([np.ones(len(ii)), lrv[5].iloc[ii],
                                 lrv[10].iloc[ii], lrv[21].iloc[ii]])
            return B @ coef

        # HAR-bc (bias-corrected HAR): same fitted HAR, plus 0.5 * variance
        # of the TRAIN residuals. Rationale: if y = HAR(x) + e with
        # e ~ (0, s²), predicting the level E[vol] rather than E[log vol]
        # adds ~0.5*s² in log space (Jensen correction for the log). The
        # residual variance s² is computed on the TRAIN split only.
        har_tr = har_at(tr_dates)
        resid_var = float(np.var(
            f["target"].loc[tr_dates].to_numpy(float) - har_tr))
        bc_shift = 0.5 * resid_var

        for split, d_all, k_all in (("test", dtte, tkte), ("val", dtva, tkva)):
            m = np.array(k_all) == t
            dd = pd.DatetimeIndex(d_all[m])
            ii = [f.index.get_loc(d) for d in dd]
            yt = f["target"].loc[dd].to_numpy(float)
            naive = lrv[5].iloc[ii].to_numpy(float)  # log rv of last 5d ending at t
            har = har_at(dd)
            har_bc = har + bc_shift
            if split == "test":
                true["test"][t] = yt
                preds["naive"][t] = naive
                preds["har"][t] = har
                preds["har-bc"][t] = har_bc
            else:
                true_val[t] = yt
                val_preds["naive"][t] = naive
                val_preds["har"][t] = har
                val_preds["har-bc"][t] = har_bc
        # GARCH(1,1): fit once on train returns; forecast each test/val date.
        # frame["log_ret"] is the return series aligned to frame dates (no NaN).
        full_ret = f["log_ret"].to_numpy(float)
        pos_of = {d: k for k, d in enumerate(f.index)}
        train_pos = np.array([pos_of[d] for d in dttr[np.array(tktr) == t]])
        test_pos = np.array([pos_of[d] for d in pd.DatetimeIndex(dtte[np.array(tkte) == t])])
        val_pos = np.array([pos_of[d] for d in pd.DatetimeIndex(dtva[np.array(tkva) == t])])
        preds["garch"][t] = garch_forecasts(full_ret[train_pos], full_ret, test_pos)
        val_preds["garch"][t] = garch_forecasts(full_ret[train_pos], full_ret, val_pos)
        print(f"  {t}: HAR coef (b0..b3) = {np.round(coef, 4).tolist()}, "
              f"train resid var = {resid_var:.4f} (HAR-bc shift +{bc_shift:.4f})")

    # ---- 6) metrics + regime + paired comparisons
    print("[6/8] metrics ...")
    print_table("TEST split (2022-01-01 onward)", FORECAST_TICKERS,
                true["test"], preds)
    print_table("VAL split (2020-01-01..2021-12-31)", FORECAST_TICKERS,
                true_val, val_preds)

    print("[7/8] regime + paired comparisons ...")
    y_by_split = {}
    for name, (yy, kk) in (("train", (ytr, tktr)), ("val", (yva, tkva)),
                           ("test", (yte, tkte))):
        y_by_split[name] = (np.asarray(yy, float),
                            {t: np.asarray(yy[np.asarray(kk) == t], float)
                             for t in FORECAST_TICKERS})
    print_regime_table(FORECAST_TICKERS, y_by_split)
    paired_bootstrap(true["test"], preds["har"], preds["garch"],
                     FORECAST_TICKERS, "HAR", "GARCH")
    paired_bootstrap(true["test"], preds["har-bc"], preds["garch"],
                     FORECAST_TICKERS, "HAR-bc", "GARCH")

    # ---- 8) leakage checks (fail loudly)
    print("\n[8/8] leakage checks ...")
    per_ticker_dates = {}
    for t in FORECAST_TICKERS:
        per_ticker_dates[t] = {
            "train": pd.DatetimeIndex(dttr[np.array(tktr) == t]),
            "val": pd.DatetimeIndex(dtva[np.array(tkva) == t]),
            "test": pd.DatetimeIndex(dtte[np.array(tkte) == t])}
    check_split_order(per_ticker_dates)
    check_scaler_train_only(Xtr_raw, scaler)
    check_recompute_features(px, vix, frames)
    check_target_uses_future_only(frames, px)
    trace = manual_trace(frames, px)
    print(trace)
    print("\nAll leakage checks passed.")
    return {"frames": frames, "trace": trace}


if __name__ == "__main__":
    run_phase1()
