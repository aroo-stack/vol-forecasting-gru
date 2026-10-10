"""Run Phase 3 end-to-end: python run_phase3.py (nothing else needed).

Phase 2 FIXES (joint bootstrap, new shuffle control, seed reprint) plus
Phase 3 ROBUSTNESS (by-year/high-vol tables, ablations, plots, trace).
Reads Phase 1/2 saved outputs only; never rewrites them.
"""

import os

import numpy as np
import pandas as pd

import robustness as R
from robustness import (shared_test_dates, diff_matrix,
                        paired_bootstrap_joint, evaluate_saved_seeds,
                        run_shuffle_control, MLP, train_nn_ensemble,
                        train_gbm, yearly_rows, highvol_rows)
from train_gru import (set_all_seeds, pick_device, predict_all,
                       compute_baselines, TinyGRU)
from vol_pipeline import (TICKERS, FORECAST_TICKERS, FEATURE_COLS,
                          download_or_load, build_daily_frame,
                          rmse, qlike, paired_bootstrap)


def per_ticker_global(arr_global, tk_arr, tickers):
    """Split a global-order array into per-ticker date-ordered dicts."""
    return {t: arr_global[tk_arr == t] for t in tickers}


def run_phase3():
    print("== Phase 3: robustness and ablations ==")
    set_all_seeds(0)
    os.makedirs("results", exist_ok=True)
    device = pick_device()
    csv_rows = []
    add = csv_rows.append

    # ---- LOAD (frozen Phase 1/2 outputs only)
    print("[1/9] loading frozen artifacts ...")
    npz = dict(np.load("data/processed.npz", allow_pickle=True))
    pg = dict(np.load("data/preds_gru.npz", allow_pickle=True))
    X = {s: np.asarray(npz[f"X_{s}"], float) for s in ("train", "val", "test")}
    y = {s: np.asarray(npz[f"y_{s}"], float) for s in ("train", "val", "test")}
    tk = {s: np.asarray(npz[f"ticker_{s}"]) for s in ("train", "val", "test")}
    dt = {s: pd.DatetimeIndex(np.asarray(npz[f"date_{s}"]))
          for s in ("train", "val", "test")}
    for s in ("train", "val", "test"):
        assert X[s].shape[1:] == (40, 7) and np.isfinite(X[s]).all() \
            and np.isfinite(y[s]).all()
    assert dt["train"].max() < dt["val"].min() < dt["test"].max()
    y_mu, y_sd = float(pg["y_mu"]), float(pg["y_sd"])
    assert abs(y_mu - y["train"].mean()) < 1e-12
    invert = lambda a: a * y_sd + y_mu  # noqa: E731
    ys = {s: (y[s] - y_mu) / y_sd for s in ("train", "val", "test")}
    gru_te, gru_va = np.asarray(pg["gru_ens_test"], float), \
        np.asarray(pg["gru_ens_val"], float)
    assert np.isfinite(gru_te).all() and len(gru_te) == len(y["test"])
    print(f"  train {len(y['train'])}, val {len(y['val'])}, test "
          f"{len(y['test'])}; ensemble loaded from frozen preds_gru.npz.")

    # ---- baselines via Phase 1/2 code paths (same samples/dates/order)
    print("[2/9] rebuilding frames + baselines ...")
    raw = {t: download_or_load(t) for t in TICKERS}
    frames = {t: build_daily_frame(raw[t], raw["^VIX"]) for t in FORECAST_TICKERS}
    true, preds, btk, bdt, _ = compute_baselines(frames, npz)
    for s in ("val", "test"):
        assert np.array_equal(btk[s], tk[s]) and (bdt[s] == dt[s]).all()

    T = FORECAST_TICKERS
    y_te, tk_te, dt_te = y["test"], tk["test"], dt["test"]
    base = shared_test_dates({t: dt_te[tk_te == t] for t in T}, T)
    print(f"  joint date grid: {len(base)} shared test dates "
          f"({base.min().date()}..{base.max().date()})")
    true_te = {t: y_te[tk_te == t] for t in T}
    P = {}  # per-ticker test preds for every model
    for m in ("har", "har-bc", "har-x", "har-x-bc", "garch", "naive",
              "constant"):
        P[m] = preds[m]["test"]
    P["gru"] = per_ticker_global(gru_te, tk_te, T)

    # ---- A3: per-seed table from the SAVED checkpoints
    print("[A3] re-evaluating saved checkpoints ...")
    ev = evaluate_saved_seeds(X, y, device)
    print(f"{'seed':<6} {'val_RMSE':>9} {'test_RMSE':>10}")
    for k in sorted(ev):
        print(f"{k:<6} {ev[k]['val_rmse']:>9.4f} {ev[k]['test_rmse']:>10.4f}")
        add({"section": "seeds_reloaded", "label": f"seed{k}", "model": "gru",
             "split": "val", "ticker": "OVERALL", "n": len(y["val"]),
             "rmse": ev[k]["val_rmse"], "qlike": np.nan, "L": np.nan,
             "mean_d": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
             "corr_pooled": np.nan, "corr_demeaned": np.nan,
             "rmse_matched_const": np.nan})
        # Frozen-artifact cross-check: reloaded == saved Phase 2 preds.
        assert np.allclose(ev[k]["test_pred"], pg["gru_seed_test"][k],
                           atol=1e-6), f"seed {k} test preds differ from npz!"
    assert abs(ev[4]["val_rmse"] - 0.4532) < 1e-4, "seed 4 val RMSE changed!"
    print("  confirmed: seed 4 val RMSE = "
          f"{ev[4]['val_rmse']:.4f} (Phase 2 reported 0.4532); "
          "reloaded checkpoints reproduce preds_gru.npz.")

    # ---- A1: joint block bootstrap (L=10 and L=20)
    print("[A1] joint date-block bootstrap on TEST ...")
    pairs = [("gru", "har-x", "GRU-ensemble", "HAR-X"),
             ("gru", "har", "GRU-ensemble", "HAR"),
             ("gru", "garch", "GRU-ensemble", "GARCH"),
             ("gru", "har-bc", "GRU-ensemble", "HAR-bc"),
             ("har", "garch", "HAR", "GARCH"),
             ("har-x", "har", "HAR-X", "HAR")]
    joint = {}
    for a, b, na, nb in pairs:
        D = diff_matrix(true_te, P[a], P[b], T)
        for L in (10, 20):
            print(f"--- d = SE({na}) - SE({nb}) ---")
            obs, lo, hi = paired_bootstrap_joint(base, D, na, nb, block=L,
                                                 n_boot=2000, seed=123)
            joint[(na, nb, L)] = (obs, lo, hi)
            add({"section": "bootstrap_joint", "label": f"{na}_vs_{nb}",
                 "model": "", "split": "test", "ticker": "OVERALL",
                 "n": len(y_te), "rmse": np.nan, "qlike": np.nan, "L": L,
                 "mean_d": obs, "ci_lo": lo, "ci_hi": hi,
                 "corr_pooled": np.nan, "corr_demeaned": np.nan,
                 "rmse_matched_const": np.nan})
    # The four CIs behind the verdict: per-ticker L=10 & L=20 + joint L=10
    # & L=20, all for GRU-ensemble vs HAR-X (reuses Phase 1/2 bootstrap).
    print("--- per-ticker bootstrap, GRU-ensemble vs HAR-X, L=10 ---")
    pt10 = paired_bootstrap(true_te, P["gru"], P["har-x"], T,
                            "GRU-ensemble", "HAR-X",
                            block=10, n_boot=2000, seed=123)
    print("--- per-ticker bootstrap, GRU-ensemble vs HAR-X, L=20 ---")
    pt20 = paired_bootstrap(true_te, P["gru"], P["har-x"], T,
                            "GRU-ensemble", "HAR-X",
                            block=20, n_boot=2000, seed=123)
    obs, lo, hi = joint[("GRU-ensemble", "HAR-X", 20)]  # verdict uses this
    print(f"Four CIs for GRU-ensemble vs HAR-X (d<0 favours GRU):\n"
          f"  per-ticker L=10: [{pt10[1]:+.5f},{pt10[2]:+.5f}]\n"
          f"  per-ticker L=20: [{pt20[1]:+.5f},{pt20[2]:+.5f}]\n"
          f"  joint      L=10: [{joint[('GRU-ensemble', 'HAR-X', 10)][1]:+.5f},"
          f"{joint[('GRU-ensemble', 'HAR-X', 10)][2]:+.5f}]\n"
          f"  joint      L=20: [{lo:+.5f},{hi:+.5f}]")
    verdict = ("GRU-ensemble beats HAR-X significantly" if hi < 0 else
               "GRU-ensemble loses to HAR-X" if lo > 0 else
               "GRU-ensemble ties HAR-X")
    print(f"FINAL VERDICT: {verdict} "
          f"(joint L=20: mean d={obs:+.5f} CI [{lo:+.5f},{hi:+.5f}])")

    # ---- A2: corrected shuffle control (retires the old warning)
    print("[A2] corrected label-shuffle control ...")
    print("  (old Phase 2 warning retired: it confounded leakage with the "
          "train->val regime shift)")
    srows, spok = run_shuffle_control(X, ys, y, invert, device)
    for k, cp, cd, rm, rc in srows:
        add({"section": "shuffle", "label": f"shuffle{k}", "model": "gru",
             "split": "val", "ticker": "OVERALL", "n": len(y["val"]),
             "rmse": rm, "qlike": np.nan, "L": np.nan, "mean_d": np.nan,
             "ci_lo": np.nan, "ci_hi": np.nan, "corr_pooled": cp,
             "corr_demeaned": cd, "rmse_matched_const": rc})

    # ---- B1: by-year + high-vol tables (test only)
    print("[B1] by-year and high-vol tables (TEST only) ...")
    g_te_global = gru_te  # npz order == date-interleaved global order
    G = {"har": np.empty(len(y_te)), "har-x": np.empty(len(y_te)),
         "gru": g_te_global}
    for t in T:
        for m, src in (("har", P["har"]), ("har-x", P["har-x"])):
            G[m][tk_te == t] = src[t]
    for m in G:
        assert np.isfinite(G[m]).all()
    print(f"{'year':<6} {'model':<12} {'n':>6} {'RMSE':>8}")
    for r in yearly_rows(dt_te, tk_te, y_te, G, ["har", "har-x", "gru"]):
        print(f"{r['year']:<6} {r['model']:<12} {r['n']:>6} {r['rmse']:>8.4f}")
        add({"section": "byyear", "label": str(r["year"]), "model": r["model"],
             "split": "test", "ticker": "OVERALL", "n": r["n"],
             "rmse": r["rmse"], "qlike": r["qlike"], "L": np.nan,
             "mean_d": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
             "corr_pooled": np.nan, "corr_demeaned": np.nan,
             "rmse_matched_const": np.nan})
    print(f"{'group':<7} {'model':<12} {'n':>6} {'RMSE':>8}")
    for r in highvol_rows(dt_te, tk_te, y_te, G, ["har", "har-x", "gru"], T):
        print(f"{r['group']:<7} {r['model']:<12} {r['n']:>6} {r['rmse']:>8.4f}")
        add({"section": "highvol", "label": r["group"], "model": r["model"],
             "split": "test", "ticker": "OVERALL", "n": r["n"],
             "rmse": r["rmse"], "qlike": r["qlike"], "L": np.nan,
             "mean_d": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
             "corr_pooled": np.nan, "corr_demeaned": np.nan,
             "rmse_matched_const": np.nan})

    # ---- B2: ablations (same target/loop/hypers as Phase 2, seeds 0-2)
    print("[B2] ablations ...")
    vix_j = FEATURE_COLS.index("vix_close")
    print(f"  features before: {FEATURE_COLS}")
    keep = [j for j, c in enumerate(FEATURE_COLS) if c != "vix_close"]
    print(f"  features after dropping VIX (col {vix_j}): "
          f"{[FEATURE_COLS[j] for j in keep]}")
    Xn = {s: X[s][..., keep] for s in ("train", "val", "test")}
    assert Xn["test"].shape[1:] == (40, 6), Xn["test"].shape
    novix = R.train_nn_ensemble(lambda: TinyGRU(input_size=6),
                                Xn["train"], ys["train"], Xn["val"],
                                ys["val"], device, R.ABL_SEEDS, "gru-noVIX")
    Xlast = {s: X[s][:, -1, :] for s in ("train", "val", "test")}
    assert Xlast["test"].shape[1:] == (7,)
    mlp = R.train_nn_ensemble(MLP, Xlast["train"], ys["train"],
                              Xlast["val"], ys["val"], device, R.ABL_SEEDS,
                              "mlp")
    print("  GBM (max_depth=3, lr=0.05, max_iter=200, no early stopping) ...")
    gbm_s = {"val": invert(R.train_gbm(Xlast["train"], ys["train"],
                                       Xlast["val"])),
             "test": invert(R.train_gbm(Xlast["train"], ys["train"],
                                        Xlast["test"]))}

    abl = {}  # global-order preds per ablation per split
    for s, Xs, Xl in (("val", Xn["val"], Xlast["val"]),
                      ("test", Xn["test"], Xlast["test"])):
        ens_n, ens_m = [], []
        for sd in R.ABL_SEEDS:
            ens_n.append(invert(predict_all(novix[sd]["model"], Xs, device)))
            ens_m.append(invert(predict_all(mlp[sd]["model"], Xl, device)))
        abl.setdefault("gru-noVIX", {})[s] = np.mean(ens_n, axis=0)
        abl.setdefault("mlp", {})[s] = np.mean(ens_m, axis=0)
        abl.setdefault("gbm", {})[s] = gbm_s[s]
    A = {m: per_ticker_global(abl[m]["test"], tk_te, T)
         for m in ("gru-noVIX", "mlp", "gbm")}
    print(f"{'model':<12} {'split':<5} {'ticker':<6} {'RMSE':>8} {'QLIKE':>9}")
    for m in ("gru-noVIX", "mlp", "gbm"):
        for s in ("val", "test"):
            gg = abl[m][s]
            assert np.isfinite(gg).all() and len(gg) == len(y[s])
            for t in T + ["OVERALL"]:
                mm = np.ones(len(y[s]), bool) if t == "OVERALL" else tk[s] == t
                r, q = rmse(gg[mm], y[s][mm]), qlike(y[s][mm], gg[mm])
                print(f"{m:<12} {s:<5} {t:<6} {r:>8.4f} {q:>9.4f}")
                add({"section": "ablation", "label": "", "model": m,
                     "split": s, "ticker": t, "n": int(mm.sum()),
                     "rmse": r, "qlike": q, "L": np.nan, "mean_d": np.nan,
                     "ci_lo": np.nan, "ci_hi": np.nan, "corr_pooled": np.nan,
                     "corr_demeaned": np.nan, "rmse_matched_const": np.nan})
    print("  joint L=20 CIs (d = SE(A) - SE(B), negative favours A):")
    cis = {}
    per_ticker_src = {"gru": P["gru"], "har": P["har"], "har-x": P["har-x"],
                      "gru-noVIX": A["gru-noVIX"], "mlp": A["mlp"],
                      "gbm": A["gbm"]}
    for a, b, na, nb in [("gru", "gru-noVIX", "GRU-ensemble", "GRU-noVIX"),
                         ("gru", "mlp", "GRU-ensemble", "MLP"),
                         ("gru", "gbm", "GRU-ensemble", "GBM"),
                         ("gru-noVIX", "har", "GRU-noVIX", "HAR"),
                         ("mlp", "har-x", "MLP", "HAR-X"),
                         ("gbm", "har-x", "GBM", "HAR-X")]:
        D = diff_matrix(true_te, per_ticker_src[a], per_ticker_src[b], T)
        print(f"--- {na} vs {nb} ---")
        cis[(na, nb)] = paired_bootstrap_joint(base, D, na, nb, block=20,
                                               n_boot=2000, seed=123)
        add({"section": "bootstrap_joint", "label": f"{na}_vs_{nb}",
             "model": "", "split": "test", "ticker": "OVERALL",
             "n": len(y_te), "rmse": np.nan, "qlike": np.nan, "L": 20,
             "mean_d": cis[(na, nb)][0], "ci_lo": cis[(na, nb)][1],
             "ci_hi": cis[(na, nb)][2], "corr_pooled": np.nan,
             "corr_demeaned": np.nan, "rmse_matched_const": np.nan})
    seq = (cis[("GRU-ensemble", "MLP")][2] < 0) and \
          (cis[("GRU-ensemble", "GBM")][2] < 0)
    print("Sequence structure helps" if seq else
          "Sequence structure: no evidence")
    vix = not (cis[("GRU-noVIX", "HAR")][2] < 0)
    print("VIX explains the gain" if vix else "VIX: no evidence")

    # ---- B3: plots (test split)
    print("[B3] plots ...")
    m_spy = tk_te == "SPY"
    d_spy = dt_te[m_spy]
    p1 = R.plot_spy_series(d_spy, y_te[m_spy], gru_te[m_spy], P["har-x"]["SPY"])
    bar_models = ["constant", "naive", "har", "har-bc", "har-x", "har-x-bc",
                  "garch", "gru", "gru-noVIX", "mlp", "gbm"]
    bar_src = {"constant": P["constant"], "naive": P["naive"], "har": P["har"],
               "har-bc": P["har-bc"], "har-x": P["har-x"],
               "har-x-bc": P["har-x-bc"], "garch": P["garch"], "gru": P["gru"],
               "gru-noVIX": A["gru-noVIX"], "mlp": A["mlp"], "gbm": A["gbm"]}
    bar_vals = [rmse(np.concatenate([bar_src[m][t] for t in T]),
                     np.concatenate([true_te[t] for t in T]))
                for m in bar_models]
    p2 = R.plot_rmse_bars(bar_models, bar_vals)
    Dgain = diff_matrix(true_te, P["har-x"], P["gru"], T)  # SE(HX)-SE(GRU)
    p3 = R.plot_cumsum_gain(base, Dgain.mean(axis=1))
    p4 = R.plot_scatter(y_te, gru_te)

    # ---- B4: sanity checks + manual trace
    print("[B4] sanity checks + manual trace ...")
    for arr in list(G.values()) + [gru_te] + [abl[m]["test"]
                                              for m in abl] \
            + [abl[m]["val"] for m in abl]:
        assert np.isfinite(arr).all()
    for m in ("gru-noVIX", "mlp", "gbm"):
        assert len(abl[m]["test"]) == len(y_te)
    print("  no NaN/inf; shapes and global-order alignment OK.")
    t0, d0 = "SPY", pd.Timestamp("2024-05-10")
    i = int(np.where((tk_te == t0) & (dt_te == d0))[0][0])
    print(f"\n--- PHASE 3 TRACE: test row i={i}, {t0} @ {d0.date()} ---")
    print(f"  X_test[i,-1,:] (last-step, scaled) = "
          f"{np.round(X['test'][i, -1, :], 6).tolist()}")
    print(f"  feature names = {FEATURE_COLS}")
    assert np.array_equal(Xlast["test"][i], X["test"][i, -1, :])
    print("  MLP/GBM input row == X_test[i,-1,:] exactly: True")
    print(f"  VIX column index = {vix_j} ('{FEATURE_COLS[vix_j]}')")
    assert np.array_equal(Xn["test"][i],
                          np.delete(X["test"][i], vix_j, axis=-1))
    print(f"  no-VIX input shape {Xn['test'][i].shape} excludes exactly "
          f"col {vix_j}: True")
    Dj = diff_matrix(true_te, P["gru"], P["har-x"], T)
    for dshow in (base[0], base[len(base) // 2], base[-1]):
        r_ = base.get_loc(dshow)
        print(f"  date {dshow.date()}:")
        for j, t in enumerate(T):
            yy_ = true_te[t][r_]
            pg_ = P["gru"][t][r_]
            ph_ = P["har-x"][t][r_]
            sg, sh = (pg_ - yy_) ** 2, (ph_ - yy_) ** 2
            print(f"    {t}: y={yy_:.4f} gru={pg_:.4f} harx={ph_:.4f} "
                  f"SE_gru={sg:.4f} SE_harx={sh:.4f} d={sg - sh:+.5f} "
                  f"(matrix {Dj[r_, j]:+.5f})")
            assert abs(Dj[r_, j] - (sg - sh)) < 1e-12
    # Hand-check one cell from the printed numbers above (first date, SPY):
    r_, j = 0, 0
    yy_, pg_, ph_ = true_te["SPY"][r_], P["gru"]["SPY"][r_], \
        P["har-x"]["SPY"][r_]
    print(f"  hand-check [date0, SPY]: ({pg_:.6f}-{yy_:.6f})^2 - "
          f"({ph_:.6f}-{yy_:.6f})^2 = "
          f"{(pg_ - yy_) ** 2 - (ph_ - yy_) ** 2:+.6f} vs matrix "
          f"{Dj[r_, j]:+.6f} -> aligned.")
    assert abs(Dj[r_, j] - ((pg_ - yy_) ** 2 - (ph_ - yy_) ** 2)) < 1e-12
    print("  joint-bootstrap alignment confirmed by hand.")

    pd.DataFrame(csv_rows).to_csv("results/phase3_results.csv", index=False)
    print(f"\nPlots: {p1}, {p2}, {p3}, {p4}")
    print("results/phase3_results.csv written.")
    print(f"FINAL VERDICT: {verdict}")
    print("Phase 3 done.")
    return verdict


if __name__ == "__main__":
    run_phase3()
