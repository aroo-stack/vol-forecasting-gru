"""Phase 3 helpers: joint block bootstrap, shuffle control, ablations,
robustness tables, plots and the manual trace.

Reuses vol_pipeline.py (rmse, qlike, paired_bootstrap, download_or_load,
build_daily_frame, backward_log_rv, garch_forecasts) and train_gru.py
(TinyGRU, set_all_seeds, pick_device, predict_all, train_with_fallback,
compute_baselines). Never touches Phase 1/2 saved outputs.
"""

import os

import matplotlib
matplotlib.use("Agg")  # headless-safe figure writing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from vol_pipeline import FORECAST_TICKERS, FEATURE_COLS, rmse, qlike
from train_gru import (set_all_seeds, predict_all, train_with_fallback,
                       TinyGRU)

PLOT_DIR = "plots"
SHUFFLE_SEEDS = [1000, 1001, 1002, 1003, 1004]  # fixed permutation seeds
SHUFFLE_EPOCHS = 15                             # fixed budget, no early stop
ABL_SEEDS = [0, 1, 2]


# ---------------------------------------------------------------- joint bootstrap
def shared_test_dates(dates_d: dict, tickers: list) -> pd.DatetimeIndex:
    """Assert every ticker shares exactly the same test dates in the same
    order; return that common date index."""
    base = pd.DatetimeIndex(dates_d[tickers[0]])
    for t in tickers[1:]:
        assert (pd.DatetimeIndex(dates_d[t]) == base).all(), \
            f"{t}: test dates differ from {tickers[0]}!"
    return base


def diff_matrix(y_true_d: dict, pred_a_d: dict, pred_b_d: dict,
                tickers: list) -> np.ndarray:
    """D[date, ticker] = SE(A) - SE(B). Columns follow `tickers` order;
    rows follow each ticker's (common) date order. d>0 favours B."""
    cols = []
    for t in tickers:
        yt = np.asarray(y_true_d[t], dtype=float)
        a = np.asarray(pred_a_d[t], dtype=float)
        b = np.asarray(pred_b_d[t], dtype=float)
        assert len(yt) == len(a) == len(b) and np.isfinite(a).all() \
            and np.isfinite(b).all()
        cols.append((a - yt) ** 2 - (b - yt) ** 2)
    return np.column_stack(cols)


def paired_bootstrap_joint(dates: pd.DatetimeIndex, D: np.ndarray,
                           name_a: str, name_b: str,
                           block: int = 10, n_boot: int = 2000,
                           seed: int = 123):
    """Joint (date-block) paired bootstrap for d = SE(A) - SE(B).

    D is the (n_dates, n_tickers) aligned difference matrix. Uses the
    circular moving-block bootstrap over the DATE index and applies the SAME
    resampled dates to every ticker, preserving cross-ticker correlation of
    errors (the per-ticker bootstrap in Phases 1/2 resampled tickers
    independently, which breaks that correlation and narrows CIs).
    Statistic: mean over resampled dates of the mean across tickers.
    """
    assert D.shape[0] == len(dates) and D.shape[1] > 0
    assert np.isfinite(D).all()
    rng = np.random.RandomState(seed)
    n = D.shape[0]
    obs = float(D.mean())
    boot = np.empty(n_boot)
    nb = int(np.ceil(n / block))
    for b in range(n_boot):
        starts = rng.randint(0, n, size=nb)
        idx = (starts[:, None] + np.arange(block)).ravel()[:n] % n
        boot[b] = D[idx].mean()
    lo, hi = (float(x) for x in np.percentile(boot, [2.5, 97.5]))
    excl = "excludes zero" if (lo > 0 or hi < 0) else "includes zero"
    fav = name_b if obs > 0 else name_a
    print(f"  joint L={block}: mean d={obs:+.5f} 95% CI [{lo:+.5f},{hi:+.5f}] "
          f"-> {excl} (d>0 favours {name_b}; point estimate favours {fav})")
    return obs, lo, hi


# ---------------------------------------------------------------- A3: saved seeds
def evaluate_saved_seeds(X: dict, y: dict, device, seeds=(0, 1, 2, 3, 4)):
    """Reload models/gru_seed{k}.pt and recompute val/test RMSE (log-vol).

    Proves the frozen checkpoints reproduce the Phase 2 per-seed table."""
    out = {}
    for k in seeds:
        ck = torch.load(os.path.join("models", f"gru_seed{k}.pt"),
                        map_location=device, weights_only=True)
        m = TinyGRU().to(device)
        m.load_state_dict(ck["state_dict"])
        pv = predict_all(m, X["val"], device) * ck["y_sd"] + ck["y_mu"]
        pt = predict_all(m, X["test"], device) * ck["y_sd"] + ck["y_mu"]
        assert np.isfinite(pv).all() and np.isfinite(pt).all()
        out[k] = {"val_rmse": rmse(pv, y["val"]), "test_rmse": rmse(pt, y["test"]),
                  "val_pred": pv, "test_pred": pt}
    return out


# ---------------------------------------------------------------- A2: shuffle control
def run_shuffle_control(X: dict, ys: dict, y: dict, invert, device):
    """Retired-warning replacement: 5 fixed-budget trainings on permuted
    y_train (no early stopping, no val-based selection at all).

    Per shuffle seed reports (i) Pearson corr(pred, true y_val) pooled and
    after subtracting each ticker's mean of both, and (ii) model val RMSE
    vs a constant equal to the model's OWN mean val prediction (cancelling
    the train->val regime-shift bias that confounded the old test).
    PASS: mean |corr| < 0.05 (both variants) and model RMSE within 0.005
    above (not below) the matched constant. Else FAIL + LEAKAGE WARNING.
    """
    print("  5 shuffle seeds x 15 fixed epochs (no early stopping) ...")
    tkva = np.asarray(np.load("data/processed.npz",
                              allow_pickle=True)["ticker_val"])
    rows, corrs_p, corrs_d, gaps = [], [], [], []
    for k in SHUFFLE_SEEDS:
        perm = np.random.RandomState(k).permutation(len(ys["train"]))
        res = train_with_fallback(X["train"], ys["train"][perm],
                                  X["val"], ys["val"], k % 100, device,
                                  fixed_epochs=SHUFFLE_EPOCHS,
                                  tag=f"shuffle{k} ")
        p = invert(predict_all(res["model"], X["val"], device))
        yt = y["val"]
        corr_p = float(np.corrcoef(p, yt)[0, 1])
        # Ticker-demeaned: subtract each ticker's mean of preds and of y.
        pc, yc = p.copy(), yt.copy()
        for t in FORECAST_TICKERS:
            m = tkva == t
            pc[m] -= p[m].mean()
            yc[m] -= yt[m].mean()
        corr_d = float(np.corrcoef(pc, yc)[0, 1])
        r_model = rmse(p, yt)
        r_const = rmse(np.full_like(yt, p.mean()), yt)  # matched constant
        rows.append((k, corr_p, corr_d, r_model, r_const))
        corrs_p.append(abs(corr_p))
        corrs_d.append(abs(corr_d))
        gaps.append(r_const - r_model)  # >0 means model beat its own mean
        print(f"    shuffle {k}: pooled corr={corr_p:+.4f} "
              f"demeaned corr={corr_d:+.4f} rmse={r_model:.4f} "
              f"matched-const={r_const:.4f}")
    mc, md, mg = float(np.mean(corrs_p)), float(np.mean(corrs_d)), \
        float(np.mean(gaps))
    print(f"  mean|corr| pooled={mc:.4f} demeaned={md:.4f}; "
          f"mean(const-model)={mg:+.5f}")
    passed = (mc < 0.05) and (md < 0.05) and (mg < 0.005)
    if passed:
        print("  SHUFFLE CONTROL: PASS (no evidence of leakage)")
    else:
        print("  ***** SHUFFLE CONTROL: FAIL — LEAKAGE WARNING *****")
    return rows, passed


# ---------------------------------------------------------------- B2: ablations
class MLP(nn.Module):
    """Last-step MLP: 7 -> 32 -> ReLU -> Dropout -> 32 -> ReLU -> Dropout
    -> 1. Dropout after each hidden ReLU (0.1, matching the GRU head)."""

    def __init__(self, input_size=7, hidden=32, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_nn_ensemble(make_model, Xtr, ytr_s, Xva, yva_s, device, seeds, tag):
    """3-seed ensemble with the exact Phase 2 loop (standardised target,
    early stopping on val). Returns per-seed result dicts."""
    per_seed = {}
    for sd in seeds:
        print(f"  --- {tag} seed {sd} ---")
        per_seed[sd] = train_with_fallback(Xtr, ytr_s, Xva, yva_s, sd, device,
                                           make_model=make_model, tag=tag + " ")
    return per_seed


def train_gbm(Xtr_last, ytr_s, Xva_last):
    """HistGradientBoostingRegressor, FIXED settings, no tuning."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    g = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05,
                                      max_iter=200, early_stopping=False,
                                      random_state=0)
    g.fit(Xtr_last, ytr_s)
    return g.predict(Xva_last)


# ---------------------------------------------------------------- B1: tables
def yearly_rows(dates: pd.DatetimeIndex, tk: np.ndarray, yt: np.ndarray,
                preds: dict, models: list):
    """Per-calendar-year pooled RMSE + n for the given models (test only)."""
    rows = []
    for yr in sorted(set(dates.year)):
        m = dates.year == yr
        for model in models:
            p = preds[model][m]
            assert np.isfinite(p).all()
            rows.append({"year": int(yr), "model": model, "n": int(m.sum()),
                         "rmse": rmse(p, yt[m]), "qlike": qlike(yt[m], p)})
    return rows


def highvol_rows(dates: pd.DatetimeIndex, tk: np.ndarray, yt: np.ndarray,
                 preds: dict, models: list, tickers: list, frac: float = 0.10):
    """RMSE on the highest-vol `frac` of test dates (by mean target across
    tickers on that date) vs the rest. Dates shared across tickers."""
    udates = pd.DatetimeIndex(sorted(set(dates)))
    daymean = np.array([yt[dates == d].mean() for d in udates])
    cut = np.quantile(daymean, 1.0 - frac)
    hot_dates = set(udates[daymean >= cut])
    hot = np.array([d in hot_dates for d in dates])
    rows = []
    for label, m in (("top10", hot), ("rest90", ~hot)):
        for model in models:
            p = preds[model][m]
            rows.append({"group": label, "model": model, "n": int(m.sum()),
                         "rmse": rmse(p, yt[m]), "qlike": qlike(yt[m], p)})
    print(f"  high-vol cutoff: mean-target >= {cut:.4f} "
          f"({hot.sum()} samples on {len(hot_dates)} dates)")
    return rows


# ---------------------------------------------------------------- B3: plots
def _save(fig, name):
    os.makedirs(PLOT_DIR, exist_ok=True)
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")
    return path


def plot_spy_series(dates, yt, p_gru, p_harx):
    m = np.asarray(dates)  # caller passes SPY-only test vectors (date order)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(m, yt, lw=1, label="true y", color="black")
    ax.plot(m, p_gru, lw=1, label="GRU-ensemble", color="tab:blue")
    ax.plot(m, p_harx, lw=1, label="HAR-X", color="tab:orange")
    ax.set_title("SPY test period: true vs predicted log-vol")
    ax.set_xlabel("date")
    ax.set_ylabel("log vol")
    ax.legend()
    return _save(fig, "spy_test_series.png")


def plot_rmse_bars(order, values):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(order, values, color="tab:blue")
    ax.set_title("Test RMSE (overall) by model")
    ax.set_ylabel("RMSE on log-vol")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    return _save(fig, "test_rmse_bars.png")


def plot_cumsum_gain(dates, gain):
    """Cumulative sum of per-date mean [SE(HAR-X) - SE(GRU)] with year lines."""
    udates = pd.DatetimeIndex(sorted(set(dates)))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(udates, np.cumsum(gain), lw=1.5, color="tab:green")
    for yr in sorted(set(udates.year))[1:]:
        ax.axvline(pd.Timestamp(f"{yr}-01-01"), color="grey", ls="--", lw=0.8)
        ax.text(pd.Timestamp(f"{yr}-01-01"), ax.get_ylim()[1], f" {yr}",
                fontsize=8, va="top")
    ax.set_title("Cumulative gain of GRU-ensemble over HAR-X "
                 "(cumsum of mean daily SE diff)")
    ax.set_xlabel("date")
    ax.set_ylabel("cumulative SE(HAR-X) - SE(GRU)")
    return _save(fig, "cumsum_gain.png")


def plot_scatter(yt, p_gru):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(yt, p_gru, s=3, alpha=0.3)
    lo = float(min(yt.min(), p_gru.min()))
    hi = float(max(yt.max(), p_gru.max()))
    ax.plot([lo, hi], [lo, hi], color="red", lw=1, label="45-degree")
    ax.set_title("GRU-ensemble vs true y (test, all tickers)")
    ax.set_xlabel("true log-vol")
    ax.set_ylabel("predicted log-vol")
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    return _save(fig, "gru_scatter.png")
