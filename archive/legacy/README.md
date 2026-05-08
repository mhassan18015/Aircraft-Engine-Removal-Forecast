# Legacy notebooks and artifacts

Notebooks and model files here predate the LEAP-1A masked dataset and are kept
for provenance only. They will not run as-is and should not be re-opened for
execution. Stored metrics are also leakage-inflated and should not be cited as
baselines for the current pipeline.

## engine_removal_forecast_nn.ipynb

- **Status:** historical prototype, not runnable.
- **Architecture:** Conv1D(64, k=3, causal) -> LSTM(64, return_seq) -> LSTM(64) -> Dense(30) -> Dense(10) -> Dense(1).
- **Why archived:**
  - Loads `Data/engines2_data_cleaned_no_outliers.csv`, which no longer exists in the repo.
  - Feature list includes vibration sensors (`vib_n1_#1_bearing`, `vib_n2_*`, `zpn12p`) that are not in the LEAP-1A dataset.
  - Target column `RUL` is the pre-merge naming; the active pipeline uses `RCL`.
  - Single-engine training set + `flight_cycle` in features + time-ordered split on one engine
    -> validation MAE ~167 cycles vs train MAE ~9 cycles (the saved outputs show the leakage).
- **Lineage:** the Conv1D + LSTM template, 30-step window, LR-finder, and EarlyStopping pattern
  carried into the production [`Lstm_PM_py.ipynb`](../../Lstm_PM_py.ipynb), which fixes the data
  leakage with leave-one-engine-out splits and residual-RCL targets.

## engine_removal_forecast_ml.ipynb

- **Status:** historical tree-ensemble baseline, not runnable.
- **Models trained (top-3 from a PyCaret leaderboard):** Random Forest, XGBoost, Extra Trees
  - Saved metrics (val / test): RF 130/132 MAE R^2~0.98 | XGBoost 115/118 MAE R^2~0.98 | Extra Trees 145/146 MAE R^2~0.98
  - These metrics are **leakage-inflated**: the 80/10/10 split was random-shuffled, so rows from
    the same engine appeared in train, val, and test, and `flight_cycle` was included as a
    feature even though `RUL = max_life - flight_cycle` by construction. R^2 ~ 0.98 collapses
    under leave-one-engine-out CV.
  - GridSearchCV grids contain a single value per hyperparameter, so the "tuning" step was
    effectively just 5-fold CV with no search.
- **Why archived:**
  - Same dead CSV (`Data/engines2_data_cleaned_no_outliers.csv`).
  - Same vibration sensors absent from LEAP data.
  - `RUL` (not `RCL`) target.
- **Useful signal worth keeping (without keeping the notebook hot):** the PyCaret leaderboard
  found Random Forest, XGBoost, and Extra Trees as the three strongest classical regressors -
  worth re-trying first if a tree-ensemble baseline is ever needed on the modern 49-engine LEAP
  dataset (with `flight_cycle` excluded, engine-stratified split, and the residual-RCL target).

## artifacts/

Stale `.pkl` files trained by `engine_removal_forecast_ml.ipynb` on the dead dataset:
`best_rf_model.pkl`, `best_xgb_model.pkl`, `best_etr_model.pkl`, plus their MinMax scalers
(`scaler_rf.pkl`, `scale_xgb.pkl`, `scale_etr.pkl`). Schema mismatch with the LEAP feature
set means they cannot be loaded and used on current data. Kept for reproducibility of the
metrics quoted above, not for inference.

## If you want to revive either notebook

Modernise to the 49-engine combined dataset:
1. Read `Data/LEAB_engines_data_cleaned.csv` (49 engines, includes `*_norm` drift features and
   `engtype_*` one-hot dummies produced by the cleaning notebook).
2. **Drop `flight_cycle` from the feature list.**
3. Use engine-stratified splits (leave-one-engine-out CV), not random shuffled splits.
4. Use `RCL` as the target (residual-RCL works even better - see the LSTM notebook).
