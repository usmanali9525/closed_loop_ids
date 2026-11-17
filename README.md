# How Safe Is Learning from Analyst Feedback?

**Label errors, poisoning and lightweight defences in a closed-loop industrial intrusion detection system**

Code and experiments for a study of what happens when an adaptive intrusion detection system
(IDS) learns from analyst feedback that is sometimes wrong. It builds on the closed-loop IDS of:

> M. Al Ghamdi, I. Alghamdi and R. J. Alzahrani, "A lightweight closed-loop adaptive intrusion
> detection framework for cyber resilience in critical infrastructure," *Scientific Reports*,
> 2026. doi:10.1038/s41598-026-73374-1

Their system improves itself with labels confirmed by a human analyst, and assumes those labels
are always correct. The authors list feedback poisoning as an open limitation. This repository
tests that assumption on the SWaT water-treatment dataset.

**Author:** Usman Ali, Department of Artificial Intelligence, University of Management and
Technology, Lahore, Pakistan
**Status:** paper in preparation.

---

## What the code does

1. **Re-implements the base system** and its confirmatory SWaT protocol: 40 temporal blocks,
   5 × 5 StratifiedGroupKFold (seeds 42–46), an online Logistic Regression with a budget of
   256 analyst labels per fold, the full-minus-one ablation and six batch baselines. The
   original code link in the paper did not resolve, so the system was rebuilt from its
   description (see [Re-implementation assumptions](#re-implementation-assumptions)).
2. **Corrupts the analyst's labels** in three ways: random errors, missed attacks (attacks
   labelled "normal") and targeted poisoning of one attack segment.
3. **Tests three lightweight defences** (filter, clip, rollback) against two references:
   no defence, and a frozen model that never learns.
4. **Proposes and tests a different update rule**, memory refit: instead of one SGD step on
   each batch of 16 labels, retrain on the trusted warm-up data plus every analyst label so
   far. A trimmed variant ignores the labels the model finds hardest to explain.
5. **Repeats everything with a second feature set** (causal temporal features) as a
   robustness check.

## Main findings

Raw features, 25 matched folds, mean F1. Frozen = the same model with no updates.

| Analyst labels | No defence | Frozen | Filter | Clip | Rollback | **Memory refit** | Trimmed (0.1) |
|---|---|---|---|---|---|---|---|
| Clean | 0.648 | 0.633 | 0.647 | 0.646 | 0.647 | **0.695** | 0.657 |
| 5% random errors | 0.591 | 0.633 | 0.643 | 0.639 | 0.656 | **0.689** | 0.676 |
| 10% random errors | 0.585 | 0.633 | 0.645 | 0.625 | 0.635 | **0.672** | 0.669 |
| 20% random errors | 0.537 | 0.633 | 0.639 | 0.609 | 0.618 | 0.649 | **0.661** |
| 30% random errors | 0.485 | 0.633 | **0.639** | 0.603 | 0.632 | 0.632 | 0.634 |
| 5% missed attacks | 0.611 | 0.633 | 0.647 | 0.642 | 0.643 | **0.694** | 0.657 |
| 10% missed attacks | 0.609 | 0.633 | 0.645 | 0.639 | 0.653 | **0.693** | 0.655 |
| 20% missed attacks | 0.576 | 0.633 | 0.642 | 0.634 | 0.636 | **0.681** | 0.660 |
| 30% missed attacks | 0.541 | 0.633 | 0.642 | 0.634 | 0.630 | **0.667** | 0.654 |

1. **The original feedback loop is fragile.** Clean labels add about +0.015 F1 over the
   frozen model. With 5% wrong labels the loop falls below the frozen model on average, and
   the loss grows with the error rate (0.648 → 0.485 at 30% random errors).
2. **Error types fail differently.** Random errors double the false-alarm rate (0.14 → 0.31
   at 30%); missed-attack errors lower detection (recall 0.67 → 0.53 at 30%).
3. **Standard defences protect by limiting learning.** Filter, clip and rollback keep the
   detector near the frozen level. The filter discards the corrections the model disputes,
   which are the ones that carry the learning signal; rollback learns from fewer labels as
   quality drops (207 of 256 when clean, 76 at 30% random errors).
4. **The fragility comes from the update rule.** Memory refit retrains on the trusted
   warm-up data plus all analyst labels, so a few wrong labels are outvoted. With the raw
   features it quadruples the benefit of clean feedback (+0.062 over frozen instead of
   +0.015), lowers false alarms (0.145 → 0.092), stays at or above the original system's
   clean F1 up to 20% random errors and 30% missed attacks, and never fell below the frozen
   model in any tested scenario. Trimming helps only at high random-error rates and costs
   elsewhere, because it also discards correct labels.
5. **Its advantage shrinks with many features.** With the 353 engineered features (at most
   512 training examples per refit), memory refit is still well above the undefended loop
   and stays flat under missed-attack errors (0.617–0.618), but its clean benefit is smaller
   (+0.024 over frozen) and under random errors it falls slightly below frozen from 10%
   (0.575 vs 0.604 at 30%).
6. **Targeted poisoning is stealthy and unsolved.** Paired per segment and fold, on attacks
   the undefended system detects when clean (17 cases): poisoning one attack lowers its
   recall from 0.53 to 0.38 without defence and from 0.58 to 0.41 with memory refit
   (Wilcoxon, Holm-adjusted p ≈ 0.03), while recall on other attacks (0.68–0.71) and overall
   F1 (0.65–0.66) do not drop. Defences that barely learn keep the target near the frozen
   level (about 0.42) but lose the general benefit of feedback. No tested method combines
   both. The improved features show the same pattern (14 cases: 0.45 → 0.30 without defence,
   0.43 → 0.27 with memory refit).

Statistics: Wilcoxon signed-rank and the Nadeau–Bengio corrected resampled t-test, both
Holm-adjusted. Memory refit beats no defence in every clean, random-error and missed-attack
scenario under Wilcoxon (raw features, p ≤ 0.011). Fold-to-fold variation is large (F1
standard deviation about 0.2), so most of these differences do not pass the stricter
corrected test. Both are reported.

### Reproduction of the base paper

| | Base paper | This re-implementation |
|---|---|---|
| Attack file | 449,919 rows, 54,621 attacks | identical |
| Evaluation subset | 39,919 rows, 12,952 attacks | 39,919 rows, 11,933 attacks |
| Full system F1 | 0.753 | 0.648 |
| Fixed-threshold LR F1 | 0.572 | 0.633 |
| Gain from incremental learning | +0.170 | +0.015 (not significant) |
| Batch LR / Extra Trees / Random Forest F1 | 0.707 / 0.712 / 0.702 | 0.576 / 0.595 / 0.611 |
| No significant F1 difference among batch models | yes | yes |

The batch-model conclusions reproduce. Absolute scores are lower because the paper's block
selection was not reported and ours contains a harder set of attacks. The gain from
incremental learning depends on the online step size, which the paper does not report: with
a small step (0.01) it vanishes; on the pilot subset the best setting gives +0.02 to +0.05.

---

## Repository structure

| File | Purpose |
|---|---|
| `check_data.py` | compares your SWaT files with the base paper's numbers |
| `convert_to_csv.py` | one-time Excel → CSV conversion (optionally dropping the first rows) |
| `data.py` | SWaT loading, attack segments, temporal blocks, folds |
| `features.py` | raw and improved feature sets, optional workload proxy |
| `online.py` | closed-loop system, label corruption, defences |
| `baselines.py` | batch models, threshold calibration |
| `evaluation.py` | metrics, alarm persistence, summaries, statistical tests |
| `run_reproduce.py` | base paper's ablation (Exp. 9) and batch baselines (Exp. 10) |
| `tune_online.py` | pilot tuning of the online step size and epochs |
| `run_poisoning.py` | label corruption × defence experiments |
| `make_figures.py` | paper figures and tables |
| `make_synthetic.py` | synthetic SWaT-like file for smoke tests only |

## Setup

Python 3.10 or newer.

**Windows**

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Linux / macOS** (on Ubuntu, first `sudo apt install python3-venv python3-full`)

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional smoke test with fake data (results are meaningless):

```
python make_synthetic.py
python run_reproduce.py --attack-csv data/synthetic_swat.csv --reps 1
```

## Data

The experiments use the official **SWaT.A1 & A2 (December 2015)** release from iTrust,
Singapore University of Technology and Design. Request access at
https://itrust.sutd.edu.sg/itrust-labs_datasets/. **The data is not redistributed here**, and
`data/` is excluded by `.gitignore`.

| File | Use |
|---|---|
| `SWaT_Dataset_Attack_v0` | all experiments (the file used by the base paper) |
| `SWaT_Dataset_Normal_v1` | normal operation; v1 is v0 without the first 30 minutes of tank drainage (not used in the reported experiments) |

The official files are Excel. Convert them once (each Excel read takes minutes):

```
python convert_to_csv.py data/SWaT_Dataset_Attack_v0.xlsx data/attack.csv
python convert_to_csv.py data/SWaT_Dataset_Normal_v0.xlsx data/normal.csv --drop-first 1800
```

Then check them:

```
python check_data.py
```

The attack file must show 449,919 rows, 54,621 attack rows and 51 columns, all marked OK.
The loader also handles tab-separated text, Latin-1 encoding and an extra title row.

---

## Reproducing the results

Run the steps in order. Each command is a single line, so it works in Windows `cmd`,
PowerShell and Linux shells. Times are for an ordinary laptop.

### Step 1: reproduce the base paper (about 5–10 min)

```
python run_reproduce.py --attack-csv data/attack.csv
python run_reproduce.py --attack-csv data/attack.csv --eta0 2 --update-epochs 5 --skip-baselines
```

The first run uses the default step size (0.01), where incremental learning has no effect;
the second uses the tuned step size from Step 2.

### Step 2: tune the online learner on a pilot subset (a few minutes each)

The base paper does not report the online step size. It is tuned on a **pilot subset**
(randomly selected blocks, seed 99) that is never used for reported results.

```
python tune_online.py --attack-csv data/attack.csv --block-selection random --block-seed 99 --eta0 0.5 1 2 5 --epochs 5 10 --reps 2
python tune_online.py --attack-csv data/attack.csv --block-selection random --block-seed 99 --features improved --eta0 0.1 0.5 1 2 5 --epochs 5 --reps 2
```

Selection rule: the largest F1 gain from updates, with the false-positive rate below about
0.10; when several settings are within noise of the best, the smallest step size.

| Feature set | Chosen | Pilot gain from updates | Pilot FPR |
|---|---|---|---|
| raw | `eta0 = 2.0`, 5 epochs | +0.023 | 0.098 |
| improved | `eta0 = 1.0`, 5 epochs | +0.048 | 0.092 |

For the raw features the gain is flat from 0.5 to 2.0 and falls at 5.0. For the improved
features it is flat from 0.5 to 5.0; the tie-break rule picks 1.0. On the main subset,
`eta0 = 1` and `eta0 = 2` give the same clean F1 for the improved features (0.614).

### Step 3: choose defence parameters on the pilot subset (about 2 min)

```
python run_poisoning.py --attack-csv data/attack.csv --block-selection random --block-seed 99 --reps 1 --features raw --rates 0.1 0.3 --defences none frozen filter@0.95 filter@0.99 clip@0.02 clip@0.05 rollback@0.05 rollback@0.1 --out results_pilot
```

Rule, fixed before the main run: for each defence, the parameter with the highest mean F1
across all scenarios (clean and poisoned, equally weighted); ties go to the less aggressive
setting.

| Defence | Mean F1 (pilot) | Chosen |
|---|---|---|
| filter@0.95 / filter@0.99 | 0.749 / 0.749 | `filter@0.99` |
| clip@0.02 / clip@0.05 | 0.734 / 0.706 | `clip@0.02` |
| rollback@0.05 / rollback@0.1 | 0.755 / 0.749 | `rollback@0.05` |
| (none / frozen) | 0.727 / 0.749 | references |

Memory refit and its trimmed variant were then compared on the same pilot subset
(two repetitions, 10 folds):

```
python run_poisoning.py --attack-csv data/attack.csv --block-selection random --block-seed 99 --reps 2 --features raw --rates 0.1 0.3 --defences none frozen rollback@0.05 memory trim@0.1 trim@0.2 trim@0.3 --out results_pilot_solution
```

| Variant | Mean F1 (pilot) |
|---|---|
| memory / trim@0.1 / trim@0.2 / trim@0.3 | 0.768 / 0.767 / 0.732 / 0.722 |
| (rollback@0.05 / frozen / none) | 0.725 / 0.706 / 0.667 |

Memory refit and trim@0.1 tie; the tie-break picks memory refit. trim@0.1 is kept in the
main run to show what trimming adds.

Success criteria, fixed before the main run: (1) with clean labels, at least as good as no
defence; (2) with wrong labels, above frozen and rollback; (3) under targeted poisoning,
target recall closer to its clean value than without defence. With the raw features memory
refit meets (1), meets (2) in 7 of 8 scenarios (ties at 30% random errors) and fails (3).

### Step 4: main experiments (20–25 min each)

```
python run_poisoning.py --attack-csv data/attack.csv --features raw --eta0 2 --defences none frozen filter@0.99 clip@0.02 rollback@0.05 memory trim@0.1 --out results_solution
python run_poisoning.py --attack-csv data/attack.csv --features improved --eta0 1 --defences none frozen filter@0.99 clip@0.02 rollback@0.05 memory trim@0.1 --out results_solution_improved
```

Each run covers 11 scenarios (clean; random errors and missed attacks at 5, 10, 20 and 30%;
targeted poisoning with its clean reference) × 7 defences × 25 folds. Defence parameters
were chosen with the raw features and reused for the improved features. The online step
size only affects the SGD-based configurations; memory refit uses the warm-up learner.

With the improved features a few memory refits stop at the iteration limit
(`ConvergenceWarning`); this affects a handful of fits only.

### Step 5: figures and tables

```
python make_figures.py --folds results_solution/poisoning_folds.csv --features raw --out figures_final_raw
python make_figures.py --folds results_solution_improved/poisoning_folds.csv --features improved --out figures_final_improved
```

| Output | Content |
|---|---|
| `fig1_f1_vs_error_rate.png` | F1 vs share of wrong labels, one line per defence |
| `fig2_side_effects.png` | false alarms under random errors; recall under missed attacks |
| `fig3_labels_accepted.png` | labels each defence actually learned from |
| `fig4_targeted.png` | recall on the targeted attack, clean vs poisoned |
| `table_main.csv/.md` | F1, FPR, recall and labels used for every scenario and defence |
| `table_targeted.csv/.md` | paired targeted analysis per segment and fold |

The targeted analysis keeps only segments the undefended system detects when labels are
clean (recall ≥ 0.1, `--detectable`), because many segments are missed even then.

---

## Experimental design

**Evaluation protocol (as in the base paper).** The attack file is cut into consecutive
1,000-row blocks. Twenty attack-containing and twenty normal-only blocks form the subset.
Folds are built with 5 × 5 StratifiedGroupKFold, so each block stays inside one fold and
neighbouring seconds never cross between training and testing. Scalers are fitted on the
training fold only.

**Closed-loop system.** An L1-regularised Logistic Regression (SGD, log loss) is trained on
a 256-event warm-up batch from the training fold. The test fold is processed in time order
as a stream, in 16 chunks. In each chunk the model scores events against an adaptive
threshold (context, behavioural deviation and recent error rates), then the analyst labels
16 randomly chosen events, and the model is updated on those labels. Labelled events are
excluded from evaluation in every configuration, so evaluation sets match across
configurations.

**Label corruption** (applied only to the analyst's labels):

| Mode | What changes |
|---|---|
| `random` | each label flipped with probability r |
| `missed_attack` | each true attack label set to "normal" with probability r |
| `targeted` | all labels of one attack segment set to "normal"; repeated for each eligible segment (300+ rows, under 30% of the subset's attack rows) present in the test fold, with rate 0 as the clean reference |

**Defences:**

| Defence | Rule |
|---|---|
| `frozen` | no updates at all (lower reference) |
| `filter@m` | skip a label if the model gives the opposite class probability ≥ m |
| `clip@c` | cap the weight change per update at c × the weight norm |
| `rollback@t` | undo an update if F1 on the trusted warm-up set drops by more than t |
| `memory` | instead of one SGD step, refit on the trusted warm-up set plus every analyst label received so far |
| `trim@q` | proposed: memory refit that ignores the share q of analyst labels the model finds hardest to explain (highest loss), then refits; warm-up labels are never dropped |

Combinations use `+`, e.g. `filter@0.99+clip@0.02`.

## Re-implementation assumptions

The base paper does not report every detail. These are the choices made here:

| # | Item | Choice |
|---|---|---|
| A1 | Block selection | 20 attack and 20 normal blocks, evenly spaced in time (`--block-selection spread`) |
| A2 | Stream | warm-up = 256 random training-fold events with both classes; test fold processed in time order |
| A3 | Labels | 256 per fold in 16 equal updates, random queries from the chunk just processed; queried events excluded from evaluation |
| A4 | Learner | SGD, log loss, L1, alpha 1e-4; batch phase with `optimal` learning rate and averaging; online phase constant step, tuned on the pilot subset (raw 2.0, improved 1.0; 5 epochs per update) |
| A5 | Threshold | Eq. 3 of the base paper: alpha 0.30, beta 0.20, gamma 0.10, delta 0.20, tau0 0.50, clipped to [0.05, 0.95]; threat = 0.5 × deviation; workload 0 (`--workload pumps` available) |
| A6 | Error feedback | FP and FN rates over the last 200 events using true outcomes (as the base paper assumes) |
| A7 | Profiler | running mean and variance on scaled features; deviation = tanh(mean clipped \|z\|), clip 5; active after 10 observations; label-free updates gated by the 80th percentile of recent deviations |
| A8 | Fixed-threshold LR | warm-up model, tau = 0.5, no updates |
| A9 | Batch baselines | trained on the whole training fold, default thresholds |
| A10 | Improved features | continuous sensors, one-hot actuator states, and rolling mean, standard deviation and differences over 10, 30 and 120 s, computed within each block from past rows only (353 features) |

**Known effect of A5:** SWaT has no workload or external threat signal, so removing
behavioural profiling and removing context thresholding give identical results.

## Limitations

- One dataset (SWaT, December 2015) and one evaluation subset; results vary strongly between
  folds, and the targeted analysis rests on 17 (raw) and 14 (improved) segment-fold cases.
- Label corruption is simulated; no real analyst or adaptive attacker is involved.
- The base system is a re-implementation and some details are assumptions (A1–A10).
- Defence parameters were tuned on the raw features and reused for the improved features.
- Memory refit's robustness depends on having enough labels relative to the number of
  features; with 353 features it degrades under random errors.
- Memory refit keeps every analyst label; with a longer stream or concept drift, a bounded
  or time-weighted memory would be needed. This was not tested.
- Consistent targeted poisoning is not mitigated by any tested method.

## Citation

If you use this code, please cite the base paper above and this repository. A citation for
the accompanying paper will be added on publication.