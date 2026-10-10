"""Phase 4 helpers: walk-forward folds, per-fold baselines + GRU training,
walk-forward evaluation, vol-targeting demo.

Reuses vol_pipeline.py (download_or_load, build_daily_frame,
backward_log_rv, garch_forecasts, rmse, qlike), train_gru.py (TinyGRU,
fit_ols, HAR_COLS, HARX_EXTRA, train_with_fallback, predict_all,
set_all_seeds, pick_device) and robustness.py (paired_bootstrap_joint,
shared_test_dates, diff_matrix). Never writes Phase 1-3 artifacts.
"""

import numpy as np
import pandas as pd

from vol_pipeline import (FORECAST_TICKERS, backward_log_rv, garch_forecasts,
                          rmse, qlike)
from train_gru import (TinyGRU, fit_ols, HAR_COLS, HARX_EXTRA,
                       train_with_fallback, predict_all)
from robustness import paired_bootstrap_joint, shared_test_dates, diff_matrix

FOLD_YEARS = [2022, 2023, 2024, 2025, 2026]
WF_SEEDS = [0, 1, 2]


# ---------------------------------------------------------------- B1: folds
def build_folds(dates: pd.DatetimeIndex, tickers: np.ndarray,
                cal: dict):
    """Walk-forward folds over the pooled (train+val+test) sample arrays.

    Fold Y: TRAIN = date <= Dec31(Y-3), VAL = calendar years Y-2..Y-1,
    TEST = calendar year Y (2026 partial). The last 5 samples per ticker
    of TRAIN and VAL are dropped so no 5-day target crosses a boundary.
    NOTE: X features and the saved scaler are kept as-is (scaler was fit on
    data up to 2019 only, so it uses no future information for any fold).
    NOTE 2: the pooled npz arrays already carry Phase 1's own boundary
    drops (5 samples at end-2019 and end-2021). Re-dropping the last 5 of
    each fold block is therefore conservative near those old boundaries
    (up to 5 extra samples excluded, ~0.1% of train) but never wrong: after
    the drop, the block max t' always has >=5 later block samples on later
    trading days, so t'+5 trading days stays inside the block. A
    calendar-exact no-crossing assert below (using real trading calendars)
    verifies this per fold per ticker, independent of the counting.
    Returns list of {Y, masks}; asserts block ordering.
    """
    folds = []
    for Y in FOLD_YEARS:
        tr_m = dates <= pd.Timestamp(f"{Y - 3}-12-31")
        va_m = (dates >= pd.Timestamp(f"{Y - 2}-01-01")) & \
               (dates <= pd.Timestamp(f"{Y - 1}-12-31"))
        te_m = (dates >= pd.Timestamp(f"{Y}-01-01")) & \
               (dates <= pd.Timestamp(f"{Y}-12-31"))
        tr_m, va_m = tr_m.copy(), va_m.copy()
        for t in FORECAST_TICKERS:  # drop last 5 per ticker (date order)
            for m in (tr_m, va_m):
                pos = np.where(m & (tickers == t))[0]
                order = np.argsort(dates[pos])
                m[pos[order[-5:]]] = False
        for t in FORECAST_TICKERS:
            dtr = dates[tr_m & (tickers == t)]
            dva = dates[va_m & (tickers == t)]
            dte = dates[te_m & (tickers == t)]
            assert len(dtr) and len(dva) and len(dte), f"{t}/{Y}: empty block!"
            assert dtr.max() < dva.min() and dva.max() < dte.min(), \
                f"{t}/{Y}: block overlap!"
            # Calendar-exact: 5th trading day after block max < next min.
            idx = cal[t]
            for da, db, tag in ((dtr, dva, "train->val"),
                                (dva, dte, "val->test")):
                p = idx.get_loc(da.max())
                assert idx[p + 5] < db.min(), \
                    f"{t}/{Y} {tag}: target crosses boundary!"
        folds.append({"Y": Y, "train": tr_m, "val": va_m, "test": te_m})
        print(f"  fold {Y}: train<={Y - 3} (n={tr_m.sum()}) "
              f"val={Y - 2}/{Y - 1} (n={va_m.sum()}) test={Y} (n={te_m.sum()})")
    return folds


# ---------------------------------------------------------------- B2: fold baselines
def build_regressors(frames: dict):
    """Per-ticker log-ret, backward log-rvs, HAR/HAR-X regressor frames and
    the full return series (all reused across folds)."""
    out = {}
    for t in FORECAST_TICKERS:
        f = frames[t]
        lr = np.log(f["_close"] / f["_close"].shift(1))
        lrv = {n: backward_log_rv(lr, n) for n in (5, 10, 21)}
        reg = pd.DataFrame({"lrv5": lrv[5], "lrv10": lrv[10], "lrv21": lrv[21],
                            "vix_close": f["vix_close"],
                            "log_vol_chg": f["log_vol_chg"]})
        out[t] = {"frame": f, "lrv5": lrv[5], "reg": reg,
                  "full_ret": f["log_ret"].to_numpy(float),
                  "pos_of": {d: k for k, d in enumerate(f.index)}}
    return out


def fit_fold_baselines(R: dict, fold_dates: dict, fold_tk: dict):
    """HAR / HAR-X / bc versions refit on the fold's TRAIN+VAL dates.

    NOTE: refitting linear baselines on train+val slightly favours them
    (they see two extra years the GRU's early stopping only validates on);
    we keep it because OLS refits are cheap and the bias, if any, runs
    against the GRU. GARCH is likewise refit on train+val returns.
    Returns {model: {ticker: test-date preds}} + trainresid shifts.
    """
    out = {m: {} for m in ("naive", "har", "har-bc", "har-x", "har-x-bc",
                           "garch")}
    shifts = {}
    for t in FORECAST_TICKERS:
        r = R[t]
        fit_dates = pd.DatetimeIndex(
            list(fold_dates["train"][fold_tk["train"] == t]) +
            list(fold_dates["val"][fold_tk["val"] == t]))
        fi = [r["pos_of"][d] for d in fit_dates if d in r["pos_of"]]
        assert len(fi) == len(fit_dates), f"{t}: fold dates missing in frame!"
        y_fit = r["frame"]["target"].iloc[fi].to_numpy(float)
        A = np.column_stack([np.ones(len(fi))] +
                            [r["reg"][c].iloc[fi] for c in HAR_COLS])
        AX = np.column_stack([np.ones(len(fi))] +
                             [r["reg"][c].iloc[fi] for c in HAR_COLS + HARX_EXTRA])
        c_har, c_harx = fit_ols(A, y_fit), fit_ols(AX, y_fit)
        bc = 0.5 * float(np.var(y_fit - A @ c_har))    # same-fit residuals
        bcx = 0.5 * float(np.var(y_fit - AX @ c_harx))
        shifts[t] = (bc, bcx)
        dd = fold_dates["test"][fold_tk["test"] == t]
        ii = [r["pos_of"][d] for d in dd]
        B = np.column_stack([np.ones(len(ii))] +
                            [r["reg"][c].iloc[ii] for c in HAR_COLS])
        BX = np.column_stack([np.ones(len(ii))] +
                             [r["reg"][c].iloc[ii] for c in HAR_COLS + HARX_EXTRA])
        har, harx = B @ c_har, BX @ c_harx
        out["naive"][t] = r["lrv5"].iloc[ii].to_numpy(float)
        out["har"][t] = har
        out["har-bc"][t] = har + bc
        out["har-x"][t] = harx
        out["har-x-bc"][t] = harx + bcx
        out["garch"][t] = garch_forecasts(
            r["full_ret"][fi], r["full_ret"],
            np.array([r["pos_of"][d] for d in dd]))
    return out, shifts


# ---------------------------------------------------------------- B2: fold GRU
def train_fold_gru(X: dict, y: dict, fold: dict, device):
    """GRU ensemble for one fold: fold-train standardised target, early
    stopping on fold VAL, seeds 0-2. Returns test/val/train ensemble preds
    (log-vol units) + the bc shift from fold-train residuals."""
    tr_m, va_m = fold["train"], fold["val"]
    mu, sd = float(y["all"][tr_m].mean()), float(y["all"][tr_m].std())
    assert sd > 0
    ys_tr, ys_va = (y["all"][tr_m] - mu) / sd, (y["all"][va_m] - mu) / sd
    assert np.allclose(ys_tr * sd + mu, y["all"][tr_m], atol=1e-12)
    ens = {}
    for s in ("train", "val", "test"):
        ens[s] = []
    for sd_ in WF_SEEDS:
        print(f"    fold {fold['Y']} seed {sd_} ...")
        res = train_with_fallback(X["all"][tr_m], ys_tr, X["all"][va_m],
                                  ys_va, sd_, device, tag=f"fold{fold['Y']} ")
        for s, m in (("train", tr_m), ("val", va_m), ("test", fold["test"])):
            p = predict_all(res["model"], X["all"][m], device) * sd + mu
            assert np.isfinite(p).all()
            ens[s].append(p)
    ens = {s: np.mean(ens[s], axis=0) for s in ens}
    shift = 0.5 * float(np.var(y["all"][tr_m] - ens["train"]))
    return ens, shift, mu, sd


# ---------------------------------------------------------------- C: vol targeting
def vol_target_metrics(dates: pd.DatetimeIndex, forecasts: dict,
                       spy_close: pd.Series, cost_bp: float = 0.0):
    """10%-vol-target SPY strategy, one row per forecast model.

    Weight w_t (known at close t) is held through t+1: strat ret on the
    next trading day = w_t * r_{t+1} (simple returns). Turnover cost =
    cost_bp * |w_t - w_{t-1}| (start flat, w_{-1} = 0). Returns metrics +
    the dated series needed for plots and the hand trace.
    """
    rets = spy_close.pct_change()
    out, series = {}, {}
    for name, pred in forecasts.items():
        pred = np.asarray(pred, float)
        assert len(pred) == len(dates) and np.isfinite(pred).all()
        fvol = np.exp(pred) * np.sqrt(252.0)          # annualised forecast
        w = np.minimum(1.5, 0.10 / np.maximum(fvol, 1e-12))
        # Next-trading-day simple returns; weight date strictly before it.
        r_next, wdate, rdate = [], [], []
        for d, wi in zip(dates, w):
            fut = rets.index[rets.index > d]
            assert len(fut) > 0, f"no next-day return after {d.date()}!"
            r_next.append(float(rets.loc[fut[0]]))
            wdate.append(d)
            rdate.append(fut[0])
        r_next = np.array(r_next)
        assert all(wd < rd for wd, rd in zip(wdate, rdate)), "lookahead!"
        turnover = np.abs(np.diff(np.concatenate([[0.0], w])))
        sr = w * r_next - cost_bp * turnover
        eq = np.cumprod(1.0 + sr)
        ann_ret, ann_vol = float(sr.mean() * 252), float(sr.std(ddof=1)
                                                        * np.sqrt(252))
        dd = eq / np.maximum.accumulate(eq) - 1.0
        roll = pd.Series(sr, index=pd.DatetimeIndex(rdate)).rolling(21).std() \
            * np.sqrt(252.0)
        out[name] = {"ann_ret": ann_ret, "ann_vol": ann_vol,
                     "sharpe": ann_ret / ann_vol if ann_vol > 0 else np.nan,
                     "max_dd": float(dd.min()),
                     "mean_abs_dW": float(np.abs(np.diff(w)).mean())
                     if len(w) > 1 else 0.0,
                     "p95_abs_dW": float(np.quantile(np.abs(np.diff(w)), 0.95))
                     if len(w) > 1 else 0.0,
                     "rollvol_sd": float((roll - 0.10).std()),
                     "n": len(sr)}
        series[name] = {"equity": eq, "eq_dates": pd.DatetimeIndex(rdate),
                        "rollvol": roll, "w": w, "w_dates": dates,
                        "r_next": r_next, "r_dates": pd.DatetimeIndex(rdate)}
    # Buy-and-hold over the same realised days (union of strategy days).
    alldays = series[list(series)[0]]["r_dates"]
    bhr = np.array([float(rets.loc[d]) for d in alldays])
    eq = np.cumprod(1.0 + bhr)
    dd = eq / np.maximum.accumulate(eq) - 1.0
    out["buy-hold"] = {"ann_ret": float(bhr.mean() * 252),
                       "ann_vol": float(bhr.std(ddof=1) * np.sqrt(252)),
                       "sharpe": float(bhr.mean() * 252) /
                       float(bhr.std(ddof=1) * np.sqrt(252)),
                       "max_dd": float(dd.min()), "mean_abs_dW": 0.0,
                       "p95_abs_dW": 0.0, "rollvol_sd": np.nan, "n": len(bhr)}
    series["buy-hold"] = {"equity": eq, "eq_dates": alldays,
                          "rollvol": pd.Series(bhr, index=alldays).rolling(21)
                          .std() * np.sqrt(252.0)}
    return out, series
