# vol-forecasting-gru


A 4-phase research pipeline that trains a GRU neural network to forecast 5-day-ahead realised volatility for US equities (SPY, AAPL, MSFT, JPM, XOM) and rigorously tests whether it beats classical baselines (HAR, GARCH).


## What it does
- Downloads and caches 20 years of daily OHLCV + VIX data via yfinance
- Builds leak-proof chronological train/val/test splits (train ≤ 2019, val 2020–21, test 2022+)
- Trains a 5-seed GRU ensemble and compares against naive, HAR, HAR-X, GARCH and ablation baselines
- Runs block-bootstrap significance tests, ablations (no-VIX GRU, MLP, gradient boosting), walk-forward retraining and a vol-targeting backtest


## Key finding
The GRU ensemble is slightly better than a linear HAR-X model on RMSE (0.4366 vs 0.4453), but the gap is small, regime-dependent (mostly 2022), and not visible in vol-targeting performance. Adding VIX to a linear model (HAR → HAR-X) gives a larger and more stable gain than the neural network itself.


## How to run
```bash
pip install -r requirements.txt
python run_phase1.py   # ~5 min:  downloads data, builds splits, computes baselines
python run_phase2.py   # ~15 min: trains GRU ensemble (5 seeds, mps/CPU)
python run_phase3.py   # ~25 min: ablations, robustness checks, plots
python run_phase4.py   # ~35 min: walk-forward retraining, vol-targeting demo
```


Data is cached to `data/` after the first run. Results land in `results/` and plots in `plots/` (both gitignored — regenerate locally).


## Project structure
| File | Phase | Role |
|---|---|---|
| `vol_pipeline.py` | 1 | Data download, feature engineering, leak-proof splits, baselines |
| `run_phase1.py` | 1 | Entry point: runs pipeline, prints baseline table |
| `train_gru.py` | 2 | GRU architecture, training loop, seed ensemble |
| `run_phase2.py` | 2 | Trains 5 seeds, prints full baseline + GRU comparison table |
| `robustness.py` | 3 | Ablations, joint block-bootstrap, by-year / high-vol tables, plots |
| `run_phase3.py` | 3 | Runs all Phase 3 checks and saves plots |
| `walkforward.py` | 4 | Walk-forward retraining, vol-targeting backtest |
| `run_phase4.py` | 4 | Runs Phase 4, prints walk-forward verdict |


## Results summary (test split 2022–2026, log-vol RMSE)
| Model | RMSE | QLIKE |
|---|---|---|
| Naive | 0.5763 | 1.0433 |
| HAR | 0.4645 | 0.5701 |
| HAR-X (HAR + VIX) | 0.4470 | 0.5309 |
| GARCH(1,1) | 0.4945 | 0.4266 |
| GRU ensemble | **0.4366** | 0.4703 |
| GRU ensemble-bc | 0.4502 | **0.4083** |


Walk-forward verdict (joint block-bootstrap L=60): **GRU ties HAR-X** — the gap is not significant at conservative block lengths.


## Requirements
Python 3.10+, Mac M3 recommended (MPS acceleration). Key packages: `torch`, `yfinance`, `arch`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`. See `requirements.txt` for full list.
