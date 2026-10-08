# Assignment 1: Polynomial Regression (BT2024013)

Polynomial regression models for the two personalised datasets:

| Problem | Inputs | Final model | 5-fold CV MSE | CV R² |
|---|---|---|---|---|
| var1: steam turbine net power score | x1–x6 | degree 5, Lasso (L1), α = 10^-2.5, 115 of 462 terms non-zero | 0.313 | 0.969 |
| var2: thermal anomaly score | x1–x3 | degree 11, Ridge (L2), λ = 0.1, 364 terms | 0.254 | 0.995 |

The full write-up is in [`report/report.pdf`](report/report.pdf).

## Approach in short
1. **Features**: every term `x1^a1 · … · xd^ad` with total degree `a1+…+ad ≤ D`.
2. **Model selection**: 5-fold cross-validation over the degree `D` and the penalty strength,
   comparing plain least squares, Ridge (L2) and Lasso (L1).
3. **Final fit**: the best model is refitted on all training rows and used to predict the test set.

## Repository layout
```
polyreg.py        feature generation, scaling, SVD ridge / least-squares solver, K-fold splits
cv_ridge.py       CV search: degree x lambda (plain least squares + ridge)
cv_lasso.py       CV search: degree x alpha (lasso + relaxed lasso)
final_model.py    fits the chosen models on all training data, writes the prediction files
self_check.py     quick 80/20 hold-out test of the final models vs simpler baselines
make_figures.py   figures used in the report
run_all.sh        runs everything in order
data/             BT2024013 train/test csv files for var1 and var2
outputs/          BT2024013_pred_var1.csv, BT2024013_pred_var2.csv, CV tables, model summaries
figures/          report figures
report/           report.tex / report.pdf
```

## Reproduce
```bash
pip install -r requirements.txt
python self_check.py       # quick test: hide 20% of train, score MSE / R^2 on it (seconds)
python final_model.py      # refit the chosen models, regenerate the prediction files (seconds)
bash run_all.sh            # full CV search (~1 h) + final models + figures
```
On Windows without bash, run the `python ...` lines inside `run_all.sh` one by one.
All random CV splits use fixed seeds.
