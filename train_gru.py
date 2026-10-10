"""Volatility forecasting — Phase 2: GRU model + fair evaluation.

Reuses vol_pipeline.py for data primitives (download_or_load,
build_daily_frame), baseline primitives (backward_log_rv, garch_forecasts),
metrics (rmse, qlike, print_table) and the block bootstrap
(paired_bootstrap). Nothing in Phase 1 is reimplemented here except the
small OLS fit/predict helper (needed for the NEW HAR-X baseline) and the
per-ticker date-mask assembly, which Phase 1 keeps inline.

Contract: reads data/processed.npz exactly as Phase 1 wrote it. X stays as
is (train-only scaling); the TARGET is standardised with train-only stats.
Writes models/gru_seed{k}.pt, data/preds_gru.npz, results/phase2_results.csv.
"""

import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from vol_pipeline import (
    TICKERS, FORECAST_TICKERS, FEATURE_COLS,
    download_or_load, build_daily_frame, backward_log_rv, garch_forecasts,
    rmse, qlike, print_table, paired_bootstrap,
)

# ---------------------------------------------------------------- fixed config
# All hyperparameters below are FIXED. Nothing is tuned on the test set;
# the test set is evaluated exactly once, after all training is finished.
HIDDEN, LAYERS, DROPOUT = 32, 1, 0.1
LR, WEIGHT_DECAY, BATCH = 1e-3, 1e-5, 128
MAX_EPOCHS, PATIENCE, CLIP = 100, 10, 1.0
SEEDS = [0, 1, 2, 3, 4]
SHUFFLE_SEED = 999          # fixed seed for the label-shuffle sanity check
BOOT_BLOCK, BOOT_B, BOOT_SEED = 10, 2000, 123  # same as Phase 1
MODEL_DIR = "models"
RESULTS_PATH = os.path.join("results", "phase2_results.csv")
PREDS_PATH = os.path.join("data", "preds_gru.npz")

HAR_COLS = ["lrv5", "lrv10", "lrv21"]            # Phase 1 HAR regressors
HARX_EXTRA = ["vix_close", "log_vol_chg"]       # NEW regressors for HAR-X


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device() -> torch.device:
    """mps on Mac if it works, else CPU. Probe with a tiny op so we fall
    back to CPU rather than crashing mid-training."""
    if getattr(torch.backends, "mps", None) is not None \
            and torch.backends.mps.is_available():
        try:
            torch.zeros(2, 2, device="mps") + 1.0
            print("  device: mps")
            return torch.device("mps")
        except Exception as e:
            print(f"  WARNING: mps probe failed ({e}); using CPU.")
    print("  device: cpu")
    return torch.device("cpu")


# The single small nn.Module in the project (required by PyTorch); all
# surrounding logic stays functional. Dropout is applied explicitly before
# the head (GRU's own dropout arg only acts with >1 layer).
class TinyGRU(nn.Module):
    def __init__(self, input_size=7, hidden=32, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden, num_layers=1, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h, _ = self.gru(x)
        return self.head(self.drop(h[:, -1, :])).squeeze(-1)


def fit_ols(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least-squares coefficients for design matrix A (with intercept col)."""
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef


@torch.no_grad()
def predict_all(model: nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    """Forward pass in log-vol (standardised) units for a whole split."""
    model.eval()
    out = []
    for i in range(0, len(X), 2048):
        xb = torch.from_numpy(X[i:i + 2048].astype(np.float32)).to(device)
        out.append(model(xb).cpu().numpy())
    return np.concatenate(out)


def train_one_seed(Xtr, ytr_s, Xva, yva_s, seed, device,
                   make_model=None, fixed_epochs=None, tag="") -> dict:
    """Train with MSE on standardised y; early-stop on val loss.

    Shuffles individual train windows each epoch (windows are independent
    samples); val/test are never shuffled into training. Restores the
    best-validation weights before returning.

    make_model: optional zero-arg factory (default TinyGRU) — lets Phase 3
      ablations reuse this exact loop. fixed_epochs: if set, train exactly
      that many epochs with NO early stopping and keep the final weights
      (used by the Phase 3 shuffle control, so no model selection on true
      val labels can occur). Both default to the Phase 2 behaviour.
    """
    set_all_seeds(seed)
    g = torch.Generator().manual_seed(seed)
    train_dl = DataLoader(TensorDataset(torch.from_numpy(Xtr.astype(np.float32)),
                                        torch.from_numpy(ytr_s.astype(np.float32))),
                          batch_size=BATCH, shuffle=True, generator=g)
    Xva_t = torch.from_numpy(Xva.astype(np.float32)).to(device)
    yva_t = torch.from_numpy(yva_s.astype(np.float32)).to(device)

    model = (make_model or TinyGRU)().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()
    n_epochs = fixed_epochs or MAX_EPOCHS
    best, best_state, wait = float("inf"), None, 0
    for epoch in range(1, n_epochs + 1):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()
        if fixed_epochs is not None:
            # No validation-based selection at all: fixed budget, final
            # weights. (No val forward pass, so val labels can't influence
            # which weights are kept.)
            if epoch % 5 == 0 or epoch == n_epochs:
                with torch.no_grad():
                    xb0, yb0 = next(iter(train_dl))
                    tl = float(loss_fn(model(xb0.to(device)), yb0.to(device)))
                print(f"    {tag}seed {seed} epoch {epoch:3d}: train-batch MSE "
                      f"{tl:.5f} (fixed budget, no early stopping)", flush=True)
            continue
        model.eval()
        with torch.no_grad():
            vl = float(loss_fn(model(Xva_t), yva_t))
        if vl < best:
            best, wait = vl, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if epoch % 10 == 0 or wait == 0:
            print(f"    {tag}seed {seed} epoch {epoch:3d}: val MSE {vl:.5f} "
                  f"(best {best:.5f}, wait {wait})", flush=True)
        if wait >= PATIENCE:
            print(f"    {tag}seed {seed}: early stop at epoch {epoch} "
                  f"(best val MSE {best:.5f})")
            break
    if fixed_epochs is not None:
        model.eval()
        with torch.no_grad():
            best = float(loss_fn(model(Xva_t), yva_t))
        return {"model": model, "best_val_mse": best, "epochs": n_epochs}
    model.load_state_dict(best_state)
    return {"model": model, "best_val_mse": best, "epochs": epoch}


def train_with_fallback(Xtr, ytr_s, Xva, yva_s, seed, device,
                        make_model=None, fixed_epochs=None, tag="") -> dict:
    """Train on `device`, but redo the seed on CPU if an mps op fails."""
    try:
        return train_one_seed(Xtr, ytr_s, Xva, yva_s, seed, device,
                              make_model=make_model, fixed_epochs=fixed_epochs,
                              tag=tag)
    except RuntimeError as e:
        if "mps" in str(e).lower() and device.type != "cpu":
            print(f"    WARNING: mps failed ({e}); retrying seed {seed} on CPU.")
            return train_one_seed(Xtr, ytr_s, Xva, yva_s, seed,
                                  torch.device("cpu"), make_model=make_model,
                                  fixed_epochs=fixed_epochs, tag=tag)
        raise


# ---------------------------------------------------------------- baselines
def compute_baselines(frames: dict, npz: dict):
    """All 7 baselines, evaluated at the npz val/test dates in npz order.

    Returns dicts {split: {model: {ticker: preds}}} plus {split: {ticker:
    true y}} and the global-order (ticker, date) vectors. HAR/HAR-bc/GARCH
    follow Phase 1 exactly (same regressors, train-only fits, fixed GARCH
    params); HAR-X adds VIX level + log-vol-change at t; constant predicts
    the global train mean of y.
    """
    tk = {s: npz[f"ticker_{s}"] for s in ("train", "val", "test")}
    dt = {s: pd.DatetimeIndex(npz[f"date_{s}"]) for s in ("train", "val", "test")}
    yy = {s: np.asarray(npz[f"y_{s}"], float) for s in ("train", "val", "test")}
    const = float(yy["train"].mean())
    print(f"  constant baseline = train mean y = {const:.4f}")

    true, preds = {"val": {}, "test": {}}, \
        {m: {} for m in ("constant", "naive", "har", "har-bc",
                         "har-x", "har-x-bc", "garch")}
    worst_drift = 0.0
    for t in FORECAST_TICKERS:
        f = frames[t]
        lr = np.log(f["_close"] / f["_close"].shift(1))
        lrv = {n: backward_log_rv(lr, n) for n in (5, 10, 21)}
        reg = pd.DataFrame({"lrv5": lrv[5], "lrv10": lrv[10], "lrv21": lrv[21],
                            "vix_close": f["vix_close"],
                            "log_vol_chg": f["log_vol_chg"]})
        tr_dates = dt["train"][tk["train"] == t]
        tr_idx = [f.index.get_loc(d) for d in tr_dates if d in f.index]
        y_tr = f["target"].iloc[tr_idx].to_numpy(float)
        coef_har = fit_ols(np.column_stack(
            [np.ones(len(tr_idx))] + [reg[c].iloc[tr_idx] for c in HAR_COLS]), y_tr)
        coef_harx = fit_ols(np.column_stack(
            [np.ones(len(tr_idx))] + [reg[c].iloc[tr_idx]
                                      for c in HAR_COLS + HARX_EXTRA]), y_tr)
        # Residual variances on TRAIN only -> bias-correction shifts.
        loc = lambda dd: [f.index.get_loc(d) for d in dd]  # noqa: E731
        bc = float(np.var(y_tr - np.column_stack(
            [np.ones(len(tr_idx))] + [reg[c].iloc[tr_idx]
                                      for c in HAR_COLS]) @ coef_har))
        bcx = float(np.var(y_tr - np.column_stack(
            [np.ones(len(tr_idx))] + [reg[c].iloc[tr_idx]
                                      for c in HAR_COLS + HARX_EXTRA]) @ coef_harx))
        full_ret = f["log_ret"].to_numpy(float)
        pos_of = {d: k for k, d in enumerate(f.index)}
        train_pos = np.array([pos_of[d] for d in tr_dates])

        for s in ("val", "test"):
            m = tk[s] == t
            dd = dt[s][m]
            assert dd.isin(f.index).all(), \
                f"{t}/{s}: npz dates missing from frame — structural misalignment!"
            ii = loc(dd)
            # Single ground truth for EVERY model (GRU included) is the npz y
            # frozen by Phase 1. The frame target is only a consistency
            # diagnostic: yfinance back-adjusts history on re-download, so a
            # freshly rebuilt frame can differ from the npz by ~1e-4 (data
            # revision, not a bug). Fail only on large drift.
            yt_frame = f["target"].loc[dd].to_numpy(float)
            yt = yy[s][m]
            worst_drift = max(worst_drift, float(np.abs(yt_frame - yt).max()))
            A = np.column_stack([np.ones(len(ii))] +
                                [reg[c].iloc[ii] for c in HAR_COLS])
            AX = np.column_stack([np.ones(len(ii))] +
                                 [reg[c].iloc[ii] for c in HAR_COLS + HARX_EXTRA])
            har = A @ coef_har
            harx = AX @ coef_harx
            true[s][t] = yt
            preds["naive"].setdefault(s, {})[t] = lrv[5].iloc[ii].to_numpy(float)
            preds["har"].setdefault(s, {})[t] = har
            preds["har-bc"].setdefault(s, {})[t] = har + 0.5 * bc
            preds["har-x"].setdefault(s, {})[t] = harx
            preds["har-x-bc"].setdefault(s, {})[t] = harx + 0.5 * bcx
            preds["garch"].setdefault(s, {})[t] = garch_forecasts(
                full_ret[train_pos], full_ret,
                np.array([pos_of[d] for d in dd]))
        print(f"  {t}: HAR {np.round(coef_har, 4).tolist()} "
              f"HAR-X {np.round(coef_harx, 4).tolist()} "
              f"bc={0.5 * bc:.4f} bcx={0.5 * bcx:.4f}")
    # Constant: same value everywhere; real arrays in npz order per split.
    for s in ("val", "test"):
        preds["constant"][s] = {t: np.full((tk[s] == t).sum(), const)
                                for t in FORECAST_TICKERS}
    assert worst_drift < 0.05, \
        f"frame-vs-npz target drift {worst_drift:.4f} too large — investigate!"
    print(f"  frame-vs-npz target drift: max {worst_drift:.2e} "
          f"(>1e-3 would suggest more than routine data revision)")
    return true, preds, tk, dt, yy


def print_extra_rows(title: str, tickers: list, y_true_d: dict, extra_d: dict):
    """Print RMSE/QLIKE rows for models Phase 1's print_table doesn't know
    (its model list is hardcoded). Same layout, reuses rmse/qlike."""
    print(f"  -- {title} --")
    for model, per_t in extra_d.items():
        for t in tickers + ["OVERALL"]:
            if t == "OVERALL":
                yt = np.concatenate([y_true_d[k] for k in tickers])
                yp = np.concatenate([per_t[k] for k in tickers])
            else:
                yt, yp = y_true_d[t], per_t[t]
            print(f"{t:<6} {len(yt):>6} {model:<15} {rmse(yp, yt):>8.4f} "
                  f"{qlike(yt, yp):>10.4f}")


# ---------------------------------------------------------------- main
def run_phase2():
    print("== Phase 2: GRU model + fair evaluation ==")
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs("results", exist_ok=True)
    device = pick_device()

    # ---- 1) LOAD npz exactly as Phase 1 wrote it; X untouched.
    print("[1/7] loading data/processed.npz ...")
    npz = dict(np.load("data/processed.npz", allow_pickle=True))
    X = {s: np.asarray(npz[f"X_{s}"], float) for s in ("train", "val", "test")}
    y = {s: np.asarray(npz[f"y_{s}"], float) for s in ("train", "val", "test")}
    tk = {s: np.asarray(npz[f"ticker_{s}"]) for s in ("train", "val", "test")}
    dt = {s: pd.DatetimeIndex(np.asarray(npz[f"date_{s}"])) for s in ("train", "val", "test")}
    for s in ("train", "val", "test"):
        assert X[s].shape[1:] == (40, 7), X[s].shape
        assert np.isfinite(X[s]).all() and np.isfinite(y[s]).all()
        assert len(y[s]) == len(tk[s]) == len(dt[s]) == X[s].shape[0]
    assert dt["train"].max() < dt["val"].min()
    assert dt["val"].max() < dt["test"].min()
    print(f"  sizes: train {len(y['train'])}, val {len(y['val'])}, "
          f"test {len(y['test'])}; date order OK; no NaN/inf.")

    # Target standardisation on TRAIN only; network trains on y_s, metrics
    # always use inverted (original log-vol) predictions. Round-trip assert
    # guarantees invert(standardise(y)) == y before any metric is computed.
    y_mu, y_sd = float(y["train"].mean()), float(y["train"].std())
    assert y_sd > 0
    standardise = lambda a: (a - y_mu) / y_sd          # noqa: E731
    invert = lambda a: a * y_sd + y_mu                 # noqa: E731
    assert np.allclose(invert(standardise(y["test"])), y["test"], atol=1e-12)
    print(f"  target standardisation: train y mean={y_mu:.4f} std={y_sd:.4f}; "
          f"round trip OK.")
    ys = {s: standardise(y[s]) for s in ("train", "val", "test")}

    # ---- 2) baselines (Phase 1 code paths) + alignment asserts.
    print("[2/7] rebuilding frames + computing baselines ...")
    raw = {t: download_or_load(t) for t in TICKERS}
    frames = {t: build_daily_frame(raw[t], raw["^VIX"]) for t in FORECAST_TICKERS}
    true, preds, btk, bdt, byy = compute_baselines(frames, npz)
    for s in ("val", "test"):  # every baseline shares GRU's order exactly
        assert np.array_equal(btk[s], tk[s]) and (bdt[s] == dt[s]).all()
    print("  baseline ticker/date arrays match GRU arrays exactly, in order.")

    # ---- 3) train 5 seeds; test evaluated once at the very end.
    print("[3/7] training 5 GRU seeds ...")
    seed_val_rmse, seed_test_rmse = {}, {}
    seed_preds = {"val": {}, "test": {}}
    for seed in SEEDS:
        print(f"  --- seed {seed} ---")
        res = train_with_fallback(X["train"], ys["train"], X["val"], ys["val"],
                                  seed, device)
        # Checkpoint holds tensors + plain floats only, so torch.load with
        # weights_only=True (the default since torch 2.6) can read it back.
        torch.save({"state_dict": res["model"].state_dict(), "y_mu": y_mu,
                    "y_sd": y_sd},
                   os.path.join(MODEL_DIR, f"gru_seed{seed}.pt"))
        for s in ("val", "test"):
            p = invert(predict_all(res["model"], X[s], device))
            assert np.isfinite(p).all()
            seed_preds[s][seed] = p
            (seed_val_rmse if s == "val" else seed_test_rmse)[seed] = rmse(p, y[s])
        print(f"  seed {seed} done: epochs={res['epochs']} "
              f"val RMSE={seed_val_rmse[seed]:.4f} "
              f"(test RMSE={seed_test_rmse[seed]:.4f}, evaluated once)")
    # Ensemble = mean of the 5 models' outputs in LOG-VOL space (i.e. mean
    # of the inverted predictions, which already live in log-vol units).
    ens = {s: np.mean([seed_preds[s][k] for k in SEEDS], axis=0)
           for s in ("val", "test")}
    # GRU-ensemble-bc: 0.5 * variance of the ENSEMBLE's TRAIN residuals.
    # Reload the saved seed models and average their (inverted, log-vol)
    # train predictions; the residual variance uses TRAIN data only.
    train_preds = []
    for k in SEEDS:
        m = TinyGRU().to(device)
        m.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"gru_seed{k}.pt"),
                                     map_location=device,
                                     weights_only=True)["state_dict"])
        train_preds.append(invert(predict_all(m, X["train"], device)))
    ens_train = np.mean(train_preds, axis=0)
    gru_bc_shift = 0.5 * float(np.var(y["train"] - ens_train))
    print(f"  ensemble train resid var shift = +{gru_bc_shift:.4f}")
    ens_bc = {s: ens[s] + gru_bc_shift for s in ("val", "test")}

    print("\n=== Per-seed GRU table (RMSE on log-vol) ===")
    print(f"{'seed':<6} {'val_RMSE':>9} {'test_RMSE':>10}")
    for k in SEEDS:
        print(f"{k:<6} {seed_val_rmse[k]:>9.4f} {seed_test_rmse[k]:>10.4f}")
    for s, dd in (("val", seed_val_rmse), ("test", seed_test_rmse)):
        v = np.array([dd[k] for k in SEEDS])
        print(f"  mean±std {s} RMSE: {v.mean():.4f} ± {v.std():.4f}")

    # ---- 4) full evaluation tables (val + test, per ticker + overall).
    print("[4/7] evaluating ...")
    gru_mean = {}  # mean of per-seed RMSE/QLIKE (metric averages, not a model)
    for s in ("val", "test"):
        gru_mean[s] = {}
        for t in FORECAST_TICKERS + ["OVERALL"]:
            if t == "OVERALL":
                yt = y[s]
                rms = [rmse(seed_preds[s][k], yt) for k in SEEDS]
                qls = [qlike(yt, seed_preds[s][k]) for k in SEEDS]
            else:
                m = tk[s] == t
                yt = y[s][m]
                rms = [rmse(seed_preds[s][k][m], yt) for k in SEEDS]
                qls = [qlike(yt, seed_preds[s][k][m]) for k in SEEDS]
            gru_mean[s][t] = (float(np.mean(rms)), float(np.std(rms)),
                              float(np.mean(qls)), float(np.std(qls)))
    full = {m: preds[m] for m in ("constant", "naive", "har", "har-bc",
                                  "har-x", "har-x-bc", "garch")}
    for s in ("val", "test"):
        full_s = {m: full[m][s] for m in full}
        full_s["gru-ensemble"] = {t: ens[s][tk[s] == t] for t in FORECAST_TICKERS}
        full_s["gru-ensemble-bc"] = {t: ens_bc[s][tk[s] == t]
                                     for t in FORECAST_TICKERS}
        title = "TEST split" if s == "test" else "VAL split"
        print_table(f"{title} — baselines + GRU ensemble", FORECAST_TICKERS,
                    true[s], full_s)
        print_extra_rows(f"{title} — continued", FORECAST_TICKERS,
                         true[s], {m: full_s[m] for m in
                                   ("constant", "har-x", "har-x-bc",
                                    "gru-ensemble", "gru-ensemble-bc")})
        print(f"  -- GRU mean-of-seeds ({s}): overall RMSE "
              f"{gru_mean[s]['OVERALL'][0]:.4f}±{gru_mean[s]['OVERALL'][1]:.4f}, "
              f"QLIKE {gru_mean[s]['OVERALL'][2]:.4f}±{gru_mean[s]['OVERALL'][3]:.4f}")

    # Save predictions + tidy CSV (split, ticker, model, n, rmse, qlike).
    np.savez(PREDS_PATH,
             gru_seed_val=np.stack([seed_preds["val"][k] for k in SEEDS]),
             gru_seed_test=np.stack([seed_preds["test"][k] for k in SEEDS]),
             gru_ens_val=ens["val"], gru_ens_test=ens["test"],
             gru_ens_bc_val=ens_bc["val"], gru_ens_bc_test=ens_bc["test"],
             y_val=y["val"], y_test=y["test"],
             ticker_val=tk["val"], date_val=npz["date_val"],
             ticker_test=tk["test"], date_test=npz["date_test"],
             y_mu=np.array(y_mu), y_sd=np.array(y_sd))
    rows = []
    eval_preds = dict(full)
    eval_preds["gru-ensemble"] = {"val": {t: ens["val"][tk["val"] == t]
                                          for t in FORECAST_TICKERS},
                                  "test": {t: ens["test"][tk["test"] == t]
                                           for t in FORECAST_TICKERS}}
    eval_preds["gru-ensemble-bc"] = {
        "val": {t: ens_bc["val"][tk["val"] == t] for t in FORECAST_TICKERS},
        "test": {t: ens_bc["test"][tk["test"] == t] for t in FORECAST_TICKERS}}
    # Global-order (npz order) prediction array per model/split. Per-ticker
    # dicts must be re-scattered through the ticker masks: concatenating
    # them in ticker order would NOT match y's date-interleaved order.
    global_preds = {}
    for s in ("val", "test"):
        global_preds[s] = {}
        for model in ("constant", "naive", "har", "har-bc", "har-x",
                      "har-x-bc", "garch", "gru-ensemble", "gru-ensemble-bc"):
            g = np.empty(len(y[s]))
            for t in FORECAST_TICKERS:
                g[tk[s] == t] = eval_preds[model][s][t]
            global_preds[s][model] = g
            # Self-check: re-masked values round-trip exactly.
            for t in FORECAST_TICKERS:
                assert np.array_equal(g[tk[s] == t], eval_preds[model][s][t])
        for k in SEEDS:
            global_preds[s][f"gru-seed{k}"] = seed_preds[s][k]
    for s in ("val", "test"):
        for t in FORECAST_TICKERS + ["OVERALL"]:
            m = np.ones(len(y[s]), bool) if t == "OVERALL" else tk[s] == t
            yt = y[s][m]
            for model in ("constant", "naive", "har", "har-bc", "har-x",
                          "har-x-bc", "garch", "gru-ensemble", "gru-ensemble-bc",
                          *[f"gru-seed{k}" for k in SEEDS]):
                yp = global_preds[s][model][m]
                rows.append({"split": s, "ticker": t, "model": model,
                             "n": len(yt), "rmse": rmse(yp, yt),
                             "qlike": qlike(yt, yp)})
            rows.append({"split": s, "ticker": t, "model": "gru-mean",
                         "n": len(yt), "rmse": gru_mean[s][t][0],
                         "qlike": gru_mean[s][t][2]})
    pd.DataFrame(rows).to_csv(RESULTS_PATH, index=False)
    print(f"  saved {PREDS_PATH} and {RESULTS_PATH}")
    chk = pd.read_csv(RESULTS_PATH)
    for s in ("val", "test"):  # CSV OVERALL rows must match direct computation
        r = chk[(chk.split == s) & (chk.ticker == "OVERALL")
                & (chk.model == "gru-ensemble")].iloc[0]
        assert abs(r["rmse"] - rmse(ens[s], y[s])) < 1e-12, "CSV OVERALL bug!"
    print("  csv consistency check OK (OVERALL rows match direct computation).")

    # ---- 5) paired bootstrap on TEST (reused Phase 1 settings).
    print("[5/7] paired block-bootstrap on TEST ...")
    flat = lambda dd: {t: dd[t] for t in FORECAST_TICKERS}
    r1 = paired_bootstrap(true["test"], flat(
        {t: ens["test"][tk["test"] == t] for t in FORECAST_TICKERS}),
        preds["har"]["test"], FORECAST_TICKERS, "GRU-ensemble", "HAR",
        block=BOOT_BLOCK, n_boot=BOOT_B, seed=BOOT_SEED)
    r2 = paired_bootstrap(true["test"], flat(
        {t: ens["test"][tk["test"] == t] for t in FORECAST_TICKERS}),
        preds["har-x"]["test"], FORECAST_TICKERS, "GRU-ensemble", "HAR-X",
        block=BOOT_BLOCK, n_boot=BOOT_B, seed=BOOT_SEED)
    r3 = paired_bootstrap(true["test"], flat(
        {t: ens["test"][tk["test"] == t] for t in FORECAST_TICKERS}),
        preds["garch"]["test"], FORECAST_TICKERS, "GRU-ensemble", "GARCH",
        block=BOOT_BLOCK, n_boot=BOOT_B, seed=BOOT_SEED)
    for (obs, lo, hi), pair in zip((r1, r2, r3),
                                   ("GRU-ensemble vs HAR", "GRU-ensemble vs HAR-X",
                                    "GRU-ensemble vs GARCH")):
        state = "excludes zero" if (lo > 0 or hi < 0) else "includes zero"
        print(f"  {pair}: mean d={obs:+.5f} CI [{lo:+.5f},{hi:+.5f}] -> {state}")
    obs, lo, hi = r2  # d = SE(GRU-ens) - SE(HAR-X); d<0 favours GRU
    if hi < 0:
        verdict = "GRU-ensemble beats HAR-X significantly"
    elif lo > 0:
        verdict = "GRU-ensemble loses to HAR-X"
    else:
        verdict = "GRU-ensemble ties HAR-X"
    print(f"FINAL VERDICT: {verdict} "
          f"(GRU-ensemble vs HAR-X: mean d={obs:+.5f} CI [{lo:+.5f},{hi:+.5f}])")

    # ---- 6) sanity checks.
    print("[6/7] sanity checks ...")
    print("  label-shuffle test: training seed-0 hypers on permuted y_train ...")
    set_all_seeds(0)
    perm = np.random.RandomState(SHUFFLE_SEED).permutation(len(y["train"]))
    sh = train_with_fallback(X["train"], ys["train"][perm], X["val"], ys["val"],
                             0, device)
    shp = invert(predict_all(sh["model"], X["val"], device))
    sh_rmse = rmse(shp, y["val"])
    const_rmse = rmse(np.full_like(y["val"], float(y["train"].mean())), y["val"])
    print(f"  shuffled-train val RMSE={sh_rmse:.4f} vs constant={const_rmse:.4f}")
    if sh_rmse < const_rmse - 0.02:
        print("  ***** LEAKAGE WARNING: shuffled-label model beats constant "
              "by >0.02 — features may leak the target! *****")
    else:
        print("  label-shuffle OK (no better than constant + 0.02).")

    # ---- 7) manual trace on one real test sample.
    print("[7/7] manual trace ...")
    print(manual_trace_one(frames, npz, X, y, standardise, ens, preds, tk, dt,
                           y_mu, y_sd))
    print("\nPhase 2 done.")
    return verdict


def manual_trace_one(frames, npz, X, y, standardise, ens, preds, tk, dt,
                     y_mu, y_sd):
    """Hand-verify alignment of one test row across every array.

    Sample: SPY @ 2024-05-10. Checks the global test index against the
    frame target, the scaled last-window row against the frame features via
    the saved scaler, the standardised target arithmetic, and the ensemble
    + HAR-X prediction lookups.
    """
    from vol_pipeline import SCALER_PATH
    t, d = "SPY", pd.Timestamp("2024-05-10")
    pos = np.where((tk["test"] == t) & (dt["test"] == d))[0]
    assert len(pos) == 1, f"expected 1 match, got {len(pos)}"
    i = int(pos[0])
    f = frames[t]
    sc = dict(np.load(SCALER_PATH, allow_pickle=True))
    # NOTE: frames are rebuilt from data/*.csv, which yfinance may have
    # revised by ~1e-4 since Phase 1 froze the npz. So frame-vs-npz value
    # comparisons below are alignment diagnostics (tolerance 1e-3), while
    # index/shape/order checks are exact.
    dy = abs(y["test"][i] - f.loc[d, "target"])
    feat_back = X["test"][i][-1][0] * sc["std"][0] + sc["mean"][0]
    dfeat = abs(feat_back - f.loc[d, "log_ret"])
    lines = [f"\n--- PHASE 2 TRACE: test row i={i}, ticker={t}, date={d.date()} ---",
             f"X_test[{i}] shape = {X['test'][i].shape} (expect (40, 7))",
             f"y_test[{i}] = {y['test'][i]:.6f}; "
             f"frame target @ {d.date()} = {f.loc[d, 'target']:.6f}; "
             f"|diff| = {dy:.2e} (aligns: {dy < 1e-3})",
             f"standardised target = ({y['test'][i]:.6f}-{y_mu:.6f})/{y_sd:.6f} "
             f"= {standardise(y['test'][i]):.6f}",
             f"last window row (scaled) log_ret = {X['test'][i][-1][0]:.6f} -> "
             f"unscaled = {feat_back:.6f}; "
             f"frame log_ret = {f.loc[d, 'log_ret']:.6f}; "
             f"|diff| = {dfeat:.2e} (aligns: {dfeat < 1e-3})"]
    assert dy < 1e-3 and dfeat < 1e-3
    saved = dict(np.load(PREDS_PATH, allow_pickle=True))
    hx = preds["har-x"]["test"][t][list(
        pd.DatetimeIndex(npz["date_test"])[tk["test"] == t]).index(d)]
    lines += [
        f"GRU-ensemble pred (log-vol) = {ens['test'][i]:.6f}; "
        f"saved preds_gru.npz = {saved['gru_ens_test'][i]:.6f}; "
        f"match = {abs(ens['test'][i] - saved['gru_ens_test'][i]) < 1e-12}",
        f"HAR-X pred = {hx:.6f}; true y = {y['test'][i]:.6f}",
        f"window covers {f.index[f.index.get_loc(d) - 39].date()}..{d.date()} "
        f"(40 trading days ending at t); target uses next 5 trading days.",
        "Aligned: X row, y row, ticker/date rows, ensemble + HAR-X lookups "
        "all reference the same (SPY, 2024-05-10)."]
    assert abs(ens["test"][i] - saved["gru_ens_test"][i]) < 1e-12
    return "\n".join(lines)


if __name__ == "__main__":
    run_phase2()
