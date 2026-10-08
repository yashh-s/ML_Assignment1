#!/usr/bin/env bash
# Reproduces every number, table, figure and prediction file in the report.
# Total runtime ~1-1.5 h on a 2-core laptop (the CV searches dominate).
set -e

# 1) ridge / plain least squares: degree x lambda grid, 5-fold CV x 2 repeats
python cv_ridge.py --var 1 --max-degree 10
python cv_ridge.py --var 2 --max-degree 20

# 2) lasso: coarse degree x alpha grid ...
python cv_lasso.py --var 1 --degrees 3-10 --seeds 0
python cv_lasso.py --var 2 --degrees 6-20 --seeds 0
#    ... then a finer alpha grid around the var1 optimum (3 repeats)
python cv_lasso.py --var 1 --degrees 4-5 --seeds 0,1,2 --alphas=-2.8,-1.6,13 --tag _fine
python cv_lasso.py --var 1 --degrees 6-6 --seeds 0,1,2 --alphas=-2.8,-1.6,13 --tag _fine6

# 3) fit chosen models on all training data -> outputs/BT2024013_pred_var{1,2}.csv
python final_model.py

# 4) figures for the report
python make_figures.py
