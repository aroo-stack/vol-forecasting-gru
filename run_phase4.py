"""Run Phase 4 end-to-end: python run_phase4.py (nothing else needed).

A: inference fixes (joint-bootstrap sensitivity, like-for-like ablations,
   future-corruption test) — no retraining except deterministic reproduction
   of the unsaved Phase 3 ablation predictions (asserted equal to the
   Phase 3 CSV). B: walk-forward retraining. C: vol-targeting demo.
D: checks + traces. Reads Phase 1-3 artifacts only; never rewrites them.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import robustness as R
from robustness import (shared_test_dates, diff_matrix,
                        paired_bootstrap_joint)
from train_gru import (set_all_seeds, pick_device, predict_all, TinyGRU,
                       compute_baselines)
from vol_pipeline import (TICKERS, FORECAST_TICKERS, FEATURE_COLS,
                          download_or_load, build_daily_frame, rmse, qlike)
from walkforward import (FOLD_YEARS, WF_SEEDS, build_folds, build_regressors,
                         fit_fold_baselines, train_fold_gru,
                         vol_target_metrics)

CORRUPT_N_YEAR = 4          # stratified samples per ticker-year (5*5*4=100)
CORRUPT_SEED = 7
GARBAGE_SEED = 1234
PRED_TOL = 1e-5


def per_ticker_global(arr_global, tk_arr, tickers):
    return {t: arr_global[tk_arr == t] for t in tickers}


def run_phase4():
    print("== Phase 4: validity fixes + walk-forward + vol-targeting ==")
    set_all_seeds(0)
    os.makedirs("results", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    device = pick_device()
    csv_rows = []
    add = csv_rows.append
    T = FORECAST_TICKERS

    # ---- LOAD frozen artifacts
    print("[1/10] loading frozen artifacts ...")
    npz = dict(np.load("data/processed.npz", allow_pickle=True))
    pg = dict(np.load("data/preds_gru.npz", allow_pickle=True))
    X = {s: np.asarray(npz[f"X_{s}"], float) for s in ("train", "val", "test")}
    y = {s: np.asarray(npz[f"y_{s}"], float) for s in ("train", "val", "test")}
    tk = {s: np.asarray(npz[f"ticker_{s}"]) for s in ("train", "val", "test")}
    dt = {s: pd.DatetimeIndex(np.asarray(npz[f"date_{s}"]))
          for s in ("train", "val", "test")}
    sc = dict(np.load("data/scaler.npz", allow_pickle=True))
    gru5_te = np.asarray(pg["gru_ens_test"], float)
    print(f"  test {len(y['test'])} samples; frozen ensemble loaded.")

    print("[2/10] rebuilding frames + Phase-split baselines ...")
    raw = {t: download_or_load(t) for t in TICKERS}
    frames = {t: build_daily_frame(raw[t], raw["^VIX"]) for t in T}
    true, preds, btk, bdt, _ = compute_baselines(frames, npz)
    for s in ("val", "test"):
        assert np.array_equal(btk[s], tk[s]) and (bdt[s] == dt[s]).all()
    y_te, tk_te, dt_te = y["test"], tk["test"], dt["test"]
    base = shared_test_dates({t: dt_te[tk_te == t] for t in T}, T)
    true_te = {t: y_te[tk_te == t] for t in T}
    P = {m: preds[m]["test"] for m in
         ("naive", "har", "har-bc", "har-x", "har-x-bc", "garch", "constant")}
    P["gru5"] = per_ticker_global(gru5_te, tk_te, T)

    # ---- A1: bootstrap sensitivity + leave-one-year-out
    print("[A1] joint-bootstrap sensitivity (GRU-ens vs HAR-X, vs HAR) ...")
    for a, b, na, nb in [("gru5", "har-x", "GRU-ensemble", "HAR-X"),
                         ("gru5", "har", "GRU-ensemble", "HAR")]:
        D = diff_matrix(true_te, P[a], P[b], T)
        for L in (10, 20, 40, 60):
            obs, lo, hi = paired_bootstrap_joint(base, D, na, nb, block=L,
                                                 n_boot=2000, seed=123)
            state = "includes zero" if lo <= 0 <= hi else "excludes zero"
            print(f"    L={L}: {state}")
            add({"section": "sens", "label": f"{na}_vs_{nb}", "model": "",
                 "split": "test", "ticker": "OVERALL", "n": len(y_te),
                 "rmse": np.nan, "qlike": np.nan, "L": L, "mean_d": obs,
                 "ci_lo": lo, "ci_hi": hi})
    print("--- per-ticker bootstrap, GRU-ensemble vs HAR-X, L=10 ---")
    from vol_pipeline import paired_bootstrap
    pt10 = paired_bootstrap(true_te, P["gru5"], P["har-x"], T,
                            "GRU-ensemble", "HAR-X",
                            block=10, n_boot=2000, seed=123)
    print("--- per-ticker bootstrap, GRU-ensemble vs HAR-X, L=20 ---")
    pt20 = paired_bootstrap(true_te, P["gru5"], P["har-x"], T,
                            "GRU-ensemble", "HAR-X",
                            block=20, n_boot=2000, seed=123)
    j10 = next(r for r in csv_rows if r["section"] == "sens" and
               r["label"] == "GRU-ensemble_vs_HAR-X" and r["L"] == 10)
    j20 = next(r for r in csv_rows if r["section"] == "sens" and
               r["label"] == "GRU-ensemble_vs_HAR-X" and r["L"] == 20)
    print(f"Four CIs for GRU-ensemble vs HAR-X (d<0 favours GRU):\n"
          f"  per-ticker L=10: [{pt10[1]:+.5f},{pt10[2]:+.5f}]\n"
          f"  per-ticker L=20: [{pt20[1]:+.5f},{pt20[2]:+.5f}]\n"
          f"  joint      L=10: [{j10['ci_lo']:+.5f},{j10['ci_hi']:+.5f}]\n"
          f"  joint      L=20: [{j20['ci_lo']:+.5f},{j20['ci_hi']:+.5f}]")
    obs20, lo20, hi20 = j20["mean_d"], j20["ci_lo"], j20["ci_hi"]
    a1_verdict = ("GRU-ensemble beats HAR-X significantly" if hi20 < 0 else
                  "GRU-ensemble loses to HAR-X" if lo20 > 0 else
                  "GRU-ensemble ties HAR-X")
    print(f"FINAL VERDICT (joint L=20): {a1_verdict} "
          f"(mean d={obs20:+.5f} CI [{lo20:+.5f},{hi20:+.5f}])")
    print("  leave-one-year-out mean d = SE(GRU)-SE(HAR-X):")
    yrs = dt_te.year
    for Y in sorted(set(yrs)):
        pool = np.concatenate(
            [((P["gru5"][t] - true_te[t]) ** 2 - (P["har-x"][t] - true_te[t]) ** 2)
             [np.array([d.year != Y for d in dt_te[tk_te == t]])] for t in T])
        print(f"    excl {Y}: n={len(pool)} mean d={pool.mean():+.5f}")
        add({"section": "loyo", "label": f"excl{Y}", "model": "",
             "split": "test", "ticker": "OVERALL", "n": len(pool),
             "rmse": np.nan, "qlike": np.nan, "L": np.nan,
             "mean_d": float(pool.mean()), "ci_lo": np.nan, "ci_hi": np.nan})

    # ---- A2: like-for-like ablations (deterministic reproduction)
    print("[A2] reproducing Phase 3 ablations deterministically ...")
    p3 = pd.read_csv("results/phase3_results.csv")
    vix_j = FEATURE_COLS.index("vix_close")
    keep = [j for j in range(len(FEATURE_COLS)) if j != vix_j]
    Xn = {s: X[s][..., keep] for s in ("train", "val", "test")}
    Xlast = {s: X[s][:, -1, :] for s in ("train", "val", "test")}
    y_mu, y_sd = float(pg["y_mu"]), float(pg["y_sd"])
    ys = {s: (y[s] - y_mu) / y_sd for s in ("train", "val", "test")}
    invert = lambda a: a * y_sd + y_mu  # noqa: E731
    repro = {}
    rn = R.train_nn_ensemble(lambda: TinyGRU(input_size=6), Xn["train"],
                             ys["train"], Xn["val"], ys["val"], device,
                             R.ABL_SEEDS, "repro-noVIX")
    rm = R.train_nn_ensemble(R.MLP, Xlast["train"], ys["train"],
                             Xlast["val"], ys["val"], device, R.ABL_SEEDS,
                             "repro-mlp")
    repro["gru-noVIX"] = np.mean(
        [invert(predict_all(rn[sd]["model"], Xn["test"], device))
         for sd in R.ABL_SEEDS], axis=0)
    repro["mlp"] = np.mean(
        [invert(predict_all(rm[sd]["model"], Xlast["test"], device))
         for sd in R.ABL_SEEDS], axis=0)
    repro["gbm"] = invert(R.train_gbm(Xlast["train"], ys["train"],
                                      Xlast["test"]))
    for m in ("gru-noVIX", "mlp", "gbm"):
        assert np.isfinite(repro[m]).all()
        ref = float(p3[(p3.section == "ablation") & (p3.split == "test")
                       & (p3.ticker == "OVERALL") & (p3.model == m)]
                    ["rmse"].iloc[0])
        got = rmse(repro[m], y_te)
        print(f"  {m}: reproduced RMSE {got:.6f} vs Phase 3 CSV {ref:.6f}")
        assert abs(got - ref) < 1e-6, f"{m} reproduction mismatch!"
    np.savez("data/preds_ablations.npz", gru_noVIX_test=repro["gru-noVIX"],
             mlp_test=repro["mlp"], gbm_test=repro["gbm"],
             y_test=y_te, ticker_test=tk_te, date_test=npz["date_test"])
    print("  saved data/preds_ablations.npz (NEW file; Phase 1-3 untouched).")
    gru3_te = pg["gru_seed_test"][0:3].mean(axis=0)  # 3-seed, saved preds
    A = {"gru3": per_ticker_global(gru3_te, tk_te, T),
         "gru-noVIX": per_ticker_global(repro["gru-noVIX"], tk_te, T),
         "mlp": per_ticker_global(repro["mlp"], tk_te, T),
         "gbm": per_ticker_global(repro["gbm"], tk_te, T)}
    print("  joint CIs, d = SE(A) - SE(B):")
    a2cis = {}
    for a, b, na, nb in [("gru3", "gru-noVIX", "GRU3", "noVIX"),
                         ("gru3", "mlp", "GRU3", "MLP"),
                         ("gru3", "gbm", "GRU3", "GBM"),
                         ("gru-noVIX", "har-x", "noVIX", "HAR-X"),
                         ("gru-noVIX", "har-x-bc", "noVIX", "HAR-X-bc")]:
        for L in (20, 60):
            D = diff_matrix(true_te, A[a] if a in A else P[a],
                            A[b] if b in A else P[b], T)
            print(f"--- {na} vs {nb}, L={L} ---")
            a2cis[(na, nb, L)] = paired_bootstrap_joint(
                base, D, na, nb, block=L, n_boot=2000, seed=123)
            o, lo, hi = a2cis[(na, nb, L)]
            add({"section": "abl_like", "label": f"{na}_vs_{nb}", "model": "",
                 "split": "test", "ticker": "OVERALL", "n": len(y_te),
                 "rmse": np.nan, "qlike": np.nan, "L": L, "mean_d": o,
                 "ci_lo": lo, "ci_hi": hi})
    print("VIX explains the gain"
          if not (a2cis[("noVIX", "HAR-X", 60)][2] < 0) else "VIX: no evidence")
    print("Sequence structure helps"
          if (a2cis[("GRU3", "MLP", 60)][2] < 0)
          and (a2cis[("GRU3", "GBM", 60)][2] < 0)
          else "Sequence structure: no evidence")

    # ---- A3: end-to-end future-corruption test (replaces label-shuffle)
    print("[A3] future-corruption test (100 stratified samples) ...")
    print("  (label-shuffle LEAKAGE WARNING logic retired with that test)")
    models5 = []
    for k in range(5):
        ck = torch.load(f"models/gru_seed{k}.pt",
                        map_location=device, weights_only=True)
        m = TinyGRU().to(device)
        m.load_state_dict(ck["state_dict"])
        m.eval()
        models5.append(m)

    def window_from(px_df, vix_df, t):
        fr = build_daily_frame(px_df, vix_df)
        assert t in fr.index, f"{t.date()}: date vanished!"
        w = fr.loc[:t].iloc[-40:][FEATURE_COLS].to_numpy(float)
        assert len(w) == 40
        return (w - sc["mean"]) / sc["std"]  # saved train-only scaler

    def ens_pred(w):
        with torch.no_grad():
            ps = [m(torch.from_numpy(w.astype(np.float32)).unsqueeze(0)
                    .to(device)).cpu().numpy()[0] for m in models5]
        return float(np.mean(ps) * y_sd + y_mu)

    rng_pick, rng_garb = np.random.RandomState(CORRUPT_SEED), \
        np.random.RandomState(GARBAGE_SEED)
    worst, offenders = 0.0, []
    tested = 0
    for t in T:
        for Y in sorted(set(dt_te.year)):
            cand = sorted(set(dt_te[(tk_te == t) & (dt_te.year == Y)]))
            for d in rng_pick.choice(cand, size=CORRUPT_N_YEAR, replace=False):
                d = pd.Timestamp(d)
                px_c = raw[t].copy()
                vx_c = raw["^VIX"].copy()
                after = px_c.index > d
                # Random garbage strictly AFTER t (extreme on purpose).
                px_c.loc[after, "Close"] = rng_garb.uniform(0.5, 2000.0,
                                                            after.sum())
                px_c.loc[after, "Volume"] = rng_garb.randint(
                    0, 10 ** 9, after.sum())
                for c in ("Open", "High", "Low"):
                    if c in px_c.columns:
                        px_c.loc[after, c] = rng_garb.uniform(0.5, 2000.0,
                                                              after.sum())
                vaft = vx_c.index > d
                vx_c.loc[vaft, "Close"] = rng_garb.uniform(5.0, 120.0,
                                                           vaft.sum())
                wc = window_from(px_c, vx_c, d)
                w0 = window_from(raw[t], raw["^VIX"], d)
                assert np.allclose(wc, w0, atol=1e-12), \
                    f"{t} {d.date()}: window changed under future garbage!"
                diff = abs(ens_pred(wc) - ens_pred(w0))
                worst = max(worst, diff)
                tested += 1
                if diff > PRED_TOL:
                    offenders.append((t, str(d.date()), diff))
    print(f"  {tested} samples: max |pred_corrupt - pred_clean| = {worst:.2e} "
          f"(tol {PRED_TOL:.0e})")
    add({"section": "corruption", "label": "max_abs_diff", "model": "",
         "split": "test", "ticker": "OVERALL", "n": tested, "rmse": np.nan,
         "qlike": np.nan, "L": np.nan, "mean_d": worst, "ci_lo": np.nan,
         "ci_hi": np.nan})
    if offenders:
        print("  ***** LEAKAGE WARNING: future garbage changed predictions:")
        for t, d, diff in offenders:
            print(f"    {t} {d}: diff={diff:.2e}")
    else:
        print("  future-corruption test PASSED: no post-t data reaches the "
              "GRU (windows bit-identical, predictions unchanged).")

    # ---- B: walk-forward retraining
    print("[B] walk-forward folds ...")
    X_all = np.concatenate([X[s] for s in ("train", "val", "test")])
    y_all = np.concatenate([y[s] for s in ("train", "val", "test")])
    tk_all = np.concatenate([tk[s] for s in ("train", "val", "test")])
    dt_all = pd.DatetimeIndex(
        np.concatenate([np.asarray(npz[f"date_{s}"]) for s in
                        ("train", "val", "test")]))
    folds = build_folds(dt_all, tk_all,
                        {t: raw[t].index for t in FORECAST_TICKERS})
    Rr = build_regressors(frames)
    Xd, yd = {"all": X_all}, {"all": y_all}
    wf_true, wf_pred = {}, {m: [] for m in
                            ("naive", "har", "har-bc", "har-x", "har-x-bc",
                             "garch", "gru", "gru-bc")}
    wf_tk, wf_dt, wf_y, wf_mu = [], [], [], {}
    for f in folds:
        Y = f["Y"]
        print(f"  -- fold {Y}: baselines (refit train+val) --")
        fd = {s: dt_all[f[s]] for s in ("train", "val", "test")}
        fk = {s: tk_all[f[s]] for s in ("train", "val", "test")}
        bp, _ = fit_fold_baselines(Rr, fd, fk)
        print(f"  -- fold {Y}: GRU 3 seeds --")
        ens, shift, mu, sd = train_fold_gru(Xd, yd, f, device)
        wf_mu[Y] = (mu, sd)
        te = f["test"]
        wf_tk.append(tk_all[te])
        wf_dt.append(dt_all[te])
        wf_y.append(y_all[te])
        for m in ("naive", "har", "har-bc", "har-x", "har-x-bc", "garch"):
            g = np.empty(te.sum())
            for t in T:
                g[fk["test"] == t] = bp[m][t]
            wf_pred[m].append(g)
        # GRU ensemble (+bc shift) is already in this fold's test order.
        wf_pred["gru"].append(ens["test"])
        wf_pred["gru-bc"].append(ens["test"] + shift)
        assert len(ens["test"]) == te.sum()
    wf_tk = np.concatenate(wf_tk)
    wf_dt = pd.DatetimeIndex(np.concatenate([d.values for d in wf_dt]))
    wf_y = np.concatenate(wf_y)
    for m in wf_pred:
        wf_pred[m] = np.concatenate(wf_pred[m])
        assert np.isfinite(wf_pred[m]).all() and len(wf_pred[m]) == len(wf_y)
    wbase = shared_test_dates({t: wf_dt[wf_tk == t] for t in T}, T)
    wf_true_te = {t: wf_y[wf_tk == t] for t in T}
    print(f"  pooled walk-forward test: n={len(wf_y)} over {len(wbase)} dates.")

    print("  pooled walk-forward RMSE/QLIKE (2022-2026):")
    print(f"  {'model':<10} {'ticker':<6} {'RMSE':>8} {'QLIKE':>9}")
    for m in ("naive", "har", "har-bc", "har-x", "har-x-bc", "garch", "gru",
              "gru-bc"):
        for t in T + ["OVERALL"]:
            mm = np.ones(len(wf_y), bool) if t == "OVERALL" else wf_tk == t
            r, q = rmse(wf_pred[m][mm], wf_y[mm]), qlike(wf_y[mm],
                                                        wf_pred[m][mm])
            print(f"  {m:<10} {t:<6} {r:>8.4f} {q:>9.4f}")
            add({"section": "wf_overall", "label": "", "model": m,
                 "split": "wf-test", "ticker": t, "n": int(mm.sum()),
                 "rmse": r, "qlike": q, "L": np.nan, "mean_d": np.nan,
                 "ci_lo": np.nan, "ci_hi": np.nan})
    print("  walk-forward by-year RMSE (n):")
    for Y in FOLD_YEARS:
        m = wf_dt.year == Y
        line = f"    {Y} (n={m.sum()}): "
        for mm in ("naive", "har", "har-x", "garch", "gru"):
            line += f"{mm}={rmse(wf_pred[mm][m], wf_y[m]):.4f} "
            add({"section": "wf_byyear", "label": str(Y), "model": mm,
                 "split": "wf-test", "ticker": "OVERALL", "n": int(m.sum()),
                 "rmse": rmse(wf_pred[mm][m], wf_y[m]), "qlike": np.nan,
                 "L": np.nan, "mean_d": np.nan, "ci_lo": np.nan,
                 "ci_hi": np.nan})
        print(line)
    wfP = {m: per_ticker_global(wf_pred[m], wf_tk, T)
           for m in ("gru", "gru-bc", "har-x", "har-x-bc", "har")}
    print("  joint bootstrap on pooled walk-forward test:")
    for a, b, na, nb in [("gru", "har-x", "WF-GRU", "WF-HAR-X"),
                         ("gru", "har", "WF-GRU", "WF-HAR")]:
        for L in (20, 60):
            D = diff_matrix(wf_true_te, wfP[a], wfP[b], T)
            print(f"--- {na} vs {nb}, L={L} ---")
            o, lo, hi = paired_bootstrap_joint(wbase, D, na, nb, block=L,
                                               n_boot=2000, seed=123)
            add({"section": "wf_boot", "label": f"{na}_vs_{nb}", "model": "",
                 "split": "wf-test", "ticker": "OVERALL", "n": len(wf_y),
                 "rmse": np.nan, "qlike": np.nan, "L": L, "mean_d": o,
                 "ci_lo": lo, "ci_hi": hi})
            if L == 60 and (na, nb) == ("WF-GRU", "WF-HAR-X"):
                wf_verdict = ("Walk-forward: GRU-ensemble beats HAR-X"
                              if hi < 0 else
                              "Walk-forward: GRU-ensemble loses to HAR-X"
                              if lo > 0 else
                              "Walk-forward: GRU-ensemble ties HAR-X")
    # QLIKE-style paired bootstrap (pointwise qlike diffs), L=20+60.
    def _pq(yp_, yt_):
        v = np.exp(2.0 * yt_)
        r = v / np.maximum(np.exp(2.0 * yp_), 1e-16)
        return r - np.log(np.maximum(r, 1e-16)) - 1.0
    for L in (20, 60):
        Dq = np.column_stack([_pq(wfP["gru-bc"][t], wf_true_te[t]) -
                              _pq(wfP["har-x-bc"][t], wf_true_te[t]) for t in T])
        print(f"--- WF-GRU-bc vs WF-HAR-X-bc (QLIKE points), L={L} ---")
        o, lo, hi = paired_bootstrap_joint(wbase, Dq, "WF-GRU-bc",
                                           "WF-HAR-X-bc", block=L,
                                           n_boot=2000, seed=123)
        add({"section": "wf_boot", "label": "WF-GRU-bc_vs_WF-HAR-X-bc_qlike",
             "model": "", "split": "wf-test", "ticker": "OVERALL",
             "n": len(wf_y), "rmse": np.nan, "qlike": np.nan, "L": L,
             "mean_d": o, "ci_lo": lo, "ci_hi": hi})
    print(wf_verdict)
    # Phase 2 fixed-training vs walk-forward, same years.
    p2 = pd.read_csv("results/phase2_results.csv")
    print("  Phase 2 fixed vs walk-forward test RMSE (same 2022-2026 data):")
    print(f"  {'model':<12} {'scope':<8} {'fixed':>8} {'walkfwd':>8}")
    for m_p2, m_wf in [("naive", "naive"), ("har", "har"),
                       ("har-x", "har-x"), ("garch", "garch"),
                       ("gru-ensemble", "gru")]:
        for t in T + ["OVERALL"]:
            rf = float(p2[(p2.split == "test") & (p2.ticker == t)
                          & (p2.model == m_p2)]["rmse"].iloc[0])
            mm = np.ones(len(wf_y), bool) if t == "OVERALL" else wf_tk == t
            rw = rmse(wf_pred[m_wf][mm], wf_y[mm])
            print(f"  {m_p2:<12} {t:<8} {rf:>8.4f} {rw:>8.4f}")
            add({"section": "p2_vs_wf", "label": m_p2, "model": m_wf,
                 "split": "wf-test", "ticker": t, "n": int(mm.sum()),
                 "rmse": rw, "qlike": np.nan, "L": np.nan, "mean_d": rf,
                 "ci_lo": np.nan, "ci_hi": np.nan})
    np.savez("data/preds_walkforward.npz", tickers=wf_tk, dates=wf_dt.values
             .astype("datetime64[D]").astype(str), y=wf_y,
             **{f"pred_{m}": wf_pred[m] for m in wf_pred})
    print("  saved data/preds_walkforward.npz (NEW file).")
    fig, ax = plt.subplots(figsize=(10, 4))
    Dg = diff_matrix(wf_true_te, wfP["gru"], wfP["har-x"], T)
    ax.plot(wbase, np.cumsum(Dg.mean(axis=1)), lw=1.5, color="tab:green")
    for yr in sorted(set(wbase.year))[1:]:
        ax.axvline(pd.Timestamp(f"{yr}-01-01"), color="grey", ls="--", lw=0.8)
    ax.set_title("Walk-forward cumulative gain of GRU over HAR-X")
    ax.set_xlabel("date")
    ax.set_ylabel("cumulative SE(HAR-X) - SE(GRU)")
    fig.savefig("plots/cumsum_gain_walkforward.png", dpi=130,
                bbox_inches="tight")
    plt.close(fig)
    print("  saved plots/cumsum_gain_walkforward.png")

    # ---- C: vol-targeting demo (SPY, walk-forward test preds)
    print("[C] vol-targeting demo (SPY, paper only) ...")
    m_spy = wf_tk == "SPY"
    spy_dates = wf_dt[m_spy]
    spy_fc = {m: wf_pred[m][m_spy] for m in ("naive", "har-x", "garch", "gru")}
    spy_close = raw["SPY"]["Close"].astype(float)
    for cost, tag in ((0.0, "no_cost"), (0.0002, "cost2bp")):
        met, ser = vol_target_metrics(spy_dates, spy_fc, spy_close,
                                      cost_bp=cost)
        print(f"  -- turnover cost {tag} --")
        print(f"  {'model':<10} {'ret%':>7} {'vol%':>7} {'sharpe':>7} "
              f"{'maxDD%':>7} {'|dW|':>7} {'p95|dW|':>8} {'rvSD%':>7}")
        for name in ("naive", "har-x", "garch", "gru", "buy-hold"):
            v = met[name]
            # NOTE: voltarget rows reuse generic CSV columns: rmse=ann_ret,
            # qlike=ann_vol, mean_d=sharpe, ci_lo=max_dd, ci_hi=mean|dW|.
            rvsd = v["rollvol_sd"]
            rvsd_s = f"{100 * rvsd:>6.2f}%" if rvsd == rvsd else "    nan"
            print(f"  {name:<10} {100 * v['ann_ret']:>6.2f}% "
                  f"{100 * v['ann_vol']:>6.2f}% {v['sharpe']:>7.2f} "
                  f"{100 * v['max_dd']:>6.2f}% {v['mean_abs_dW']:>7.4f} "
                  f"{v['p95_abs_dW']:>8.4f} {rvsd_s}")
            add({"section": f"voltarget_{tag}", "label": name, "model": name,
                 "split": "wf-test", "ticker": "SPY", "n": v["n"],
                 "rmse": v["ann_ret"], "qlike": v["ann_vol"], "L": np.nan,
                 "mean_d": v["sharpe"], "ci_lo": v["max_dd"],
                 "ci_hi": v["mean_abs_dW"]})
        if tag == "no_cost":
            ser_nc, met_nc = ser, met
    fig, ax = plt.subplots(figsize=(10, 4))
    for name, s in ser_nc.items():
        ax.semilogy(s["eq_dates"], s["equity"], lw=1.2, label=name)
    ax.set_title("Vol-targeting equity curves, SPY (log scale, no costs)")
    ax.set_xlabel("date")
    ax.legend()
    fig.savefig("plots/equity_curves.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4))
    for name, s in ser_nc.items():
        ax.plot(s["rollvol"].index, s["rollvol"].values, lw=1, label=name)
    ax.axhline(0.10, color="black", ls="--", lw=1, label="10% target")
    ax.set_title("Rolling 21-day realised vol vs 10% target")
    ax.set_xlabel("date")
    ax.legend()
    fig.savefig("plots/rolling_vol.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("  saved plots/equity_curves.png plots/rolling_vol.png")
    sds = {n_: met_nc[n_]["rollvol_sd"] for n_ in ("naive", "har-x", "garch",
                                                   "gru")}
    spread = max(sds.values()) - min(sds.values())
    print(f"  rolling-vol-control spread across forecasters: "
          f"{100 * spread:.2f}pp")
    if spread * 100 < 0.5:
        print("  COMMENT: all forecasters are about equal at realised-vol "
              "control; forecast quality gaps do not translate visibly.")
    else:
        print("  COMMENT: forecast quality differences translate into "
              "visibly different realised-vol control.")

    # ---- D: checks + traces
    print("[D] checks + manual traces ...")
    for arr in list(wf_pred.values()) + [wf_y]:
        assert np.isfinite(arr).all()
    print("  no NaN/inf; fold alignment asserted per ticker during build.")
    f24 = [f for f in folds if f["Y"] == 2024][0]
    print("\n--- TRACE fold Y=2024 ---")
    for s in ("train", "val", "test"):
        d = dt_all[f24[s]]
        print(f"  {s}: {d.min().date()}..{d.max().date()} n={f24[s].sum()}")
    mu24, sd24 = wf_mu[2024]
    print(f"  standardisation: train mean={mu24:.4f} std={sd24:.4f}")
    assert dt_all[f24["train"]].max() < pd.Timestamp("2024-01-01")
    assert dt_all[f24["val"]].max() < pd.Timestamp("2024-01-01")
    print("  confirmed by hand: no train/val sample dated 2024 or later.")
    i24 = int(np.where((wf_tk == "SPY") & (wf_dt == pd.Timestamp("2024-05-10")))[0][0])
    print(f"  SPY 2024-05-10: y={wf_y[i24]:.6f} "
          f"fold-GRU-ens={wf_pred['gru'][i24]:.6f} "
          f"fold-HAR-X={wf_pred['har-x'][i24]:.6f}")
    print("\n--- TRACE vol-targeting (SPY, GRU forecasts, no costs) ---")
    gser = ser_nc["gru"]
    closes = raw["SPY"]["Close"].astype(float)
    for dshow in (spy_dates[0], spy_dates[len(spy_dates) // 2], spy_dates[-1]):
        k = list(spy_dates).index(dshow)
        fc = spy_fc["gru"][k]
        w_ = gser["w"][k]
        rn_ = gser["r_next"][k]
        rd_ = gser["r_dates"][k]
        # Hand recomputation from primitives (not from the stored series).
        # Mirror the implementation exactly: it casts forecasts to float64
        # first (np.asarray(pred, float)), while the saved arrays are
        # float32 — so compare in float64 with a 1e-9 tolerance.
        fc64 = float(fc)
        w_check = min(1.5, 0.10 / max(np.exp(fc64) * np.sqrt(252.0), 1e-12))
        prev = closes.index[closes.index < rd_][-1]
        r_check = float(closes.loc[rd_] / closes.loc[prev] - 1.0)
        print(f"  t={dshow.date()}: forecast={fc:.4f} "
              f"annVol={np.exp(fc64) * np.sqrt(252):.4f} w={w_:.4f} "
              f"(recomputed {w_check:.4f}) "
              f"r_next({rd_.date()})={rn_:+.6f} (recomputed {r_check:+.6f}) "
              f"strat={w_ * rn_:+.6f}")
        assert abs(w_ - w_check) < 1e-9 and abs(rn_ - r_check) < 1e-12
        assert dshow < rd_
    print("  confirmed by hand: weight dates precede return dates; "
          "strat return = w_t * r_{t+1}.")

    pd.DataFrame(csv_rows).to_csv("results/phase4_results.csv", index=False)
    print("\nresults/phase4_results.csv written.")
    print("Phase 4 done.")


if __name__ == "__main__":
    run_phase4()
