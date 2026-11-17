# Closed-loop IDS: re-implementation and label-poisoning study

Code for the paper extending Al Ghamdi et al., *"A lightweight closed-loop adaptive intrusion
detection framework for cyber resilience in critical infrastructure"*, Scientific Reports (2026),
doi:10.1038/s41598-026-73374-1.

It does three things:

1. **Re-implements** the paper's confirmatory SWaT protocol: 40 temporal blocks,
   5×5 StratifiedGroupKFold (seeds 42–46), the closed-loop Logistic Regression system with a
   256-label budget, the full-minus-one ablation (Exp. 9) and the batch baselines (Exp. 10).
2. **Improves the detector** (Option 2) with causal temporal features, one-hot actuator states,
   validation-tuned thresholds and k-of-n alarm persistence.
3. **Runs the poisoning study** (Option 1): random label errors, missed attacks and targeted
   poisoning of analyst labels, against three defences (filter, clip, rollback) and combinations.

## Setup

Ubuntu blocks system-wide `pip` installs, so use a virtual environment:

```bash
cd ~/Documents/GitHub/closed_loop_ids
sudo apt install python3-venv python3-full   # one-time
python3 -m venv .venv
source .venv/bin/activate                    # repeat in every new terminal
pip install --upgrade pip
pip install -r requirements.txt
```

On **Windows (PowerShell)** the same steps are:

```powershell
cd $HOME\Documents\GitHub\closed_loop_ids
python -m venv .venv
.venv\Scripts\Activate.ps1      # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt
```

In VS Code: `Ctrl+Shift+P` → "Python: Select Interpreter" → choose `./.venv`.

## Data

Source: the official iTrust **SWaT.A1 & A2 (December 2015)** release.

| File | Use |
|------|-----|
| `SWaT_Dataset_Attack_v0` | all experiments (the file used by the base paper) |
| `SWaT_Dataset_Normal_v1` | normal-operation baseline for later features. v1 = v0 without the first 30 minutes of tank drainage |

The official files are Excel. The scripts can read `.xlsx` directly, but each read takes
minutes, so convert once to CSV:

```bash
python convert_to_csv.py data/SWaT_Dataset_Attack_v0.xlsx data/attack.csv
python convert_to_csv.py data/SWaT_Dataset_Normal_v1.xlsx data/normal.csv
# if you only have Normal v0, drop the 30-minute drainage period yourself:
python convert_to_csv.py data/SWaT_Dataset_Normal_v0.xlsx data/normal.csv --drop-first 1800
```

The loader also handles tab-separated text, Latin-1 encoding and an extra title row
above the header.

`data/` and `.venv/` are listed in `.gitignore`. SWaT must not be redistributed, so never
push the data to GitHub.

Smoke test with fake data (optional, results are meaningless):

```bash
python make_synthetic.py
python run_reproduce.py --attack-csv data/synthetic_swat.csv --reps 1
```

## Step 1: check the data

```bash
python check_data.py
```

The base paper reports **449,919 rows, 54,621 attack rows and 51 variables** for the attack
file, with timestamps starting on 28 December 2015. The script marks each value OK or
DIFFERENT. If anything differs, note it in the paper. You can also check the Excel files directly:
`python check_data.py --files data/SWaT_Dataset_Attack_v0.xlsx`. If the loader raises an error (for
example, labels coded as 0/1), adapt `load_swat` in `data.py`.

## Step 2: reproduce the base paper

```bash
python run_reproduce.py --attack-csv data/attack.csv
```

Targets from the paper: full system F1 ≈ 0.75, fixed-threshold LR ≈ 0.57, batch LR ≈ 0.71,
Extra Trees ≈ 0.71. Expect differences (see assumptions below); report your numbers as
"our re-implementation".

Outputs in `results/`: per-fold CSV, summary table (mean ± sd [95% CI]) and paired tests
(Wilcoxon signed-rank and Nadeau–Bengio corrected t-test, both Holm-adjusted).

## Step 3: stronger detector

```bash
python run_reproduce.py --attack-csv data/attack.csv \
    --features improved --calibrate --persistence 3,5
```

Add one change at a time (features → calibration → persistence) to attribute each gain.

## Step 4: poisoning study

```bash
python run_poisoning.py --attack-csv data/attack.csv \
    --features raw improved --rates 0.05 0.1 0.2 0.3 \
    --defences none filter clip rollback filter+clip
```

The full grid is about 2,500 runs and takes roughly 15 minutes on a laptop.
Outputs: `poisoning_summary.csv`, `poisoning_targeted_recall.csv` (recall on the poisoned
attack segments vs the others) and `poisoning_defence_tests.csv`.

Always report the **"none" corruption row for every defence**: it shows what each defence
costs when labels are clean.

## Re-implementation assumptions (state these in the Methods section)

The base paper does not report every detail. These are the choices made here:

| # | Item | Choice |
|---|------|--------|
| A1 | Block selection | File cut into consecutive 1,000-row blocks; 20 attack-containing and 20 normal-only blocks chosen evenly spaced in time (`--block-selection spread`). `random` and `all` are also available. |
| A2 | Stream | Warm-up = 256 random events (both classes) from the training fold. The test fold is processed in time order as the stream. |
| A3 | Labels | 256 per fold in 16 equal scheduled updates, queried from the chunk just processed. Default query strategy: random. Queried events are excluded from evaluation in every configuration so evaluation sets are matched. |
| A4 | Learner | SGD, log loss, L1, alpha 1e-4. Batch phase: `optimal` learning rate with averaging. Online phase: constant step `eta0 = 0.01`, one pass per update. **eta0 is not reported in the paper.** |
| A5 | Threshold | Eq. 3 with alpha 0.30, beta 0.20, gamma 0.10, delta 0.20, tau0 0.50, clipped to [0.05, 0.95]. Threat L_t = 0.5 × deviation (no external feed in SWaT). Workload W_t = 0 by default; `--workload pumps` uses the share of running pumps. |
| A6 | Error feedback | FP and FN rates over the last 200 events using true outcomes (the paper assumes every alert outcome is confirmed). |
| A7 | Profiler | Welford mean/variance on scaled features; deviation = tanh(mean of |z| clipped at 5); active after 10 observations; label-free baseline updates gated by the 80th percentile of recent deviations (refreshed every 50 events). |
| A8 | Fixed-threshold LR | Batch-phase model, tau = 0.5, no updates. |
| A9 | Batch baselines | Trained on the whole training fold, default thresholds (0.5 / 0) unless `--calibrate`. |

**Known difference:** with W_t = 0 and no external threat feed, removing behavioural profiling
and removing context thresholding have the same effect, because the profiler only acts through
the threshold. Use `--workload pumps` if you want the two ablations to differ, and say which
you used.

## Tuning without leaking

`eta0` and `update_epochs` control how much the online model learns from each batch of 16
labels. With the default `eta0 = 0.01` the updates on SWaT are almost zero, so incremental
learning appears useless. Pick them on a **pilot subset** that you do not report:

```bash
python tune_online.py --attack-csv data/attack.csv --block-selection random --block-seed 99
```

It prints, for each setting, the F1 with and without updates and the mean relative weight
change per update. Fix one setting, then pass it to every reported run
(`--eta0 ... --update-epochs ...`).
Do the same for the defence settings in `online.Defence` (`filter_margin`, `clip_rel`,
`rollback_tol`).

## Files

| File | Purpose |
|------|---------|
| `check_data.py` | compares your SWaT files with the base paper's numbers |
| `convert_to_csv.py` | one-time Excel → CSV conversion (optionally dropping the first rows) |
| `data.py` | SWaT loading, attack segments, temporal blocks, folds |
| `features.py` | raw and improved feature sets, optional workload proxy |
| `online.py` | closed-loop system, label corruption, defences |
| `baselines.py` | batch models, threshold calibration |
| `evaluation.py` | metrics, persistence, summaries, statistical tests |
| `run_reproduce.py` | Exp. 9 ablation + Exp. 10 baselines |
| `run_poisoning.py` | poisoning × defence grid |
| `tune_online.py` | pilot tuning of the online step size and epochs |
| `make_synthetic.py` | synthetic SWaT-like file for smoke tests |

## Next steps (not yet implemented)

- WADI or HAI loader (Option 3): write a function returning the same `(X, y)` as `load_swat`.
- Residual features from a model trained on the normal-operation file.
- Diversity-aware query strategy (Exp. 11 of the base paper).
