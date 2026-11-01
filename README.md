# Freight Rate Prediction

Spotter ML assessment: predict `posted_rate` for Nov–Dec 2025 loads and a fixed Lexington → Fort Wayne December chart.


## Approach

1. **Time-forward split** — train Jan–Aug, hold out Sep–Oct (mirrors Nov–Dec).
2. **Train-only cleaning** — drop ~1.4% RPM-contaminated labels (outside 0.5–1.8× typical RPM by equipment × distance bin).
3. **Two-part model** on log(rate per mile), then `price = distance × exp(part1 + part2)`:
   - **Ridge** — market level, trend, quarter-end ramp
   - **LightGBM** — residual route/load detail (**no calendar features**)

Sep–Oct holdout: **MAE ≈ $89** full / **≈ $33** on clean rows (vs ~$109 best single GBDT).

## Repository layout

```
├── data/                  # Input CSVs (do not overwrite with predictions)
├── outputs/               # Submission artifacts
├── reports/               # Markdown + PDF validation report
├── src/                   # Library code
├── train_predict.py       # Train → predict CLI
├── score.py               # Validate outputs + December chart
├── requirements.txt
└── README.md
```

| Path | Contents |
|---|---|
| `data/train-test.csv` | Jan–Oct labeled loads |
| `data/validation.csv` | Nov–Dec features |
| `data/validation-predictions-template.csv` | Required `load_id` order |
| `data/december-chart-inputs.csv` | Fixed-lane December template |
| `outputs/validation_predictions.csv` | 12,000 predictions |
| `outputs/december_chart_inputs.csv` | Completed December chart CSV |
| `outputs/candidate_december.png` | Chart from `score.py` |
| `src/two_part.py` | Ridge + LightGBM model |
| `src/data.py` | Loading, contamination filter, features |
| `src/model.py` | GBDT baselines |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Train and predict

```bash
python train_predict.py
```

Writes under `outputs/`. Force a backend with `--model two_part|xgboost_rpm|lightgbm|xgboost|hist_gradient_boosting`.

## Score

```bash
python score.py \
  --predictions outputs/validation_predictions.csv \
  --december-predictions outputs/december_chart_inputs.csv \
  --output-dir outputs
```

