#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from sklearn.model_selection import cross_val_predict
import optuna
from sklearn.ensemble import StackingRegressor

def run_pipeline(train_path='assets/train.csv', test_path='assets/test.csv'):
    # Load data
    print(f"Loading training data from: {train_path}")
    train = pd.read_csv(train_path)
    print(f"Loading test data from: {test_path}")
    test = pd.read_csv(test_path)


    print(f"Number of rows in train: {len(train)}")
    print(f"Number of rows in test: {len(test)}")

    # Essential classical + normalized sabermetrics features
    # Ensure all required columns exist
    essential_cols = ['R', 'RA', 'ERA', 'SOA', 'HR', 'BB', 'SO', 'E', 'FP']
    for col in essential_cols:
        if col not in train.columns:
            train[col] = np.nan
        if col not in test.columns:
            test[col] = np.nan

    # Games played (G)
    if 'G' not in train.columns:
        train['G'] = np.nan
    if 'G' not in test.columns:
        test['G'] = np.nan

    # Run differential
    train['run_diff'] = train['R'] - train['RA']
    test['run_diff'] = test['R'] - test['RA']

    # Runs per game (RPG)
    train['RPG'] = train['R'] / train['G']
    test['RPG'] = test['R'] / test['G']

    # Calculate league average runs per game (mlb_rpg) for each year
    train['team_rpg'] = train['R'] / train['G']
    test['team_rpg'] = test['R'] / test['G']
    if 'yearID' in train.columns:
        mlb_rpg_by_year = train.groupby('yearID').apply(lambda x: x['R'].sum() / x['G'].sum())
        train = train.merge(mlb_rpg_by_year.rename('mlb_rpg'), left_on='yearID', right_index=True)
    else:
        train['mlb_rpg'] = train['team_rpg'].mean()
    if 'yearID' in test.columns:
        if 'mlb_rpg' in train.columns:
            test = test.merge(train[['yearID', 'mlb_rpg']].drop_duplicates('yearID'), on='yearID', how='left')
            test['mlb_rpg'] = test['mlb_rpg'].fillna(train['mlb_rpg'].mean())
        else:
            test['mlb_rpg'] = train['team_rpg'].mean()
    else:
        test['mlb_rpg'] = train['team_rpg'].mean()

    if 'mlb_rpg' not in train.columns:
        train['mlb_rpg'] = train['team_rpg'].mean()
    if 'mlb_rpg' not in test.columns:
        test['mlb_rpg'] = train['team_rpg'].mean()

    # RD_adj = run_diff * (mlb_rpg / team_rpg)
    train['RD_adj'] = train['run_diff'] * (train['mlb_rpg'] / train['team_rpg'])
    test['RD_adj'] = test['run_diff'] * (test['mlb_rpg'] / test['team_rpg'])

    # Normalized RPG
    train['RPG_norm'] = train['RPG'] / train['mlb_rpg']
    test['RPG_norm'] = test['RPG'] / test['mlb_rpg']

    # RD_eff = run_diff per game / RPG
    train['RD_eff'] = (train['run_diff'] / train['G']) / (train['RPG'] + 1e-5)
    test['RD_eff'] = (test['run_diff'] / test['G']) / (test['RPG'] + 1e-5)

    # Target variable
    target_col = 'W'

    # Core features for baseline configuration
    core_features = ['R', 'RA', 'ERA', 'SOA', 'HR', 'BB', 'SO', 'E', 'FP', 'RD_adj', 'RPG_norm', 'RD_eff']
    # Only keep features that exist in both train and test
    core_features = [f for f in core_features if f in train.columns and f in test.columns]

    X_train = train[core_features].copy()
    y_train = train[target_col].copy()
    X_test = test[core_features].copy()

    # Fill NaNs with 0 (safe default for missing stats)
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=core_features)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=core_features)

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_dir = os.path.join('submission', timestamp)
    os.makedirs(submission_dir, exist_ok=True)

    # Ridge regression (with Optuna tuning)
    def tune_ridge(X, y, n_trials=20):
        def objective(trial):
            alpha = trial.suggest_loguniform('alpha', 1e-3, 10)
            model = Ridge(alpha=alpha, random_state=42)
            preds = cross_val_predict(model, X, y, cv=5, method='predict')
            return mean_absolute_error(y, preds)
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        return study.best_params['alpha']

    print("Tuning Ridge Regression with Optuna...")
    ridge_alpha = tune_ridge(X_train_scaled, y_train)
    print(f"Best Ridge alpha: {ridge_alpha}")

    ridge_final = Ridge(alpha=ridge_alpha, random_state=42)
    ridge_final.fit(X_train_scaled, y_train)
    ridge_preds_train = ridge_final.predict(X_train_scaled)
    ridge_preds_test = ridge_final.predict(X_test_scaled)

    # LightGBM (with Optuna tuning)
    def tune_lgb(X, y, n_trials=20):
        def objective(trial):
            param = {
                'objective': 'regression',
                'metric': 'mae',
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 15, 255),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
                'verbosity': -1,
                'seed': 42
            }
            cv_results = lgb.cv(
                param,
                lgb.Dataset(X, label=y),
                nfold=5,
                metrics='mae',
                seed=42,
                stratified=False
            )
            # dynamically get key containing 'l1'
            mae_key = [k for k in cv_results.keys() if 'l1' in k][0]
            return min(cv_results[mae_key])
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        return study.best_params

    print("Tuning LightGBM with Optuna...")
    lgb_params = tune_lgb(X_train_scaled, y_train)
    print(f"Best LightGBM params: {lgb_params}")
    lgb_final = lgb.LGBMRegressor(**lgb_params)
    lgb_final.fit(X_train_scaled, y_train)
    lgb_preds_train = lgb_final.predict(X_train_scaled)
    lgb_preds_test = lgb_final.predict(X_test_scaled)

    # Stacking with Ridge + LightGBM
    estimators = [
        ('ridge', ridge_final),
        ('lgb', lgb_final)
    ]
    stacking_model = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=1.0, random_state=42),
        cv=5,
        n_jobs=-1,
        passthrough=True
    )
    stacking_model.fit(X_train_scaled, y_train)
    stacking_preds_train = stacking_model.predict(X_train_scaled)
    stacking_preds_test = stacking_model.predict(X_test_scaled)

    # Clip predictions to reasonable bounds
    stacking_preds_test_rounded = np.round(stacking_preds_test).astype(int)
    stacking_preds_test_rounded = np.clip(stacking_preds_test_rounded, 40, 120)

    # Save stacking model submission
    if 'ID' in test.columns:
        test_index = test['ID']
        print("Test data contains 'ID' column for submission index.")
    else:
        test_index = test.index
        print("Test data does not contain 'ID' column; using index for submission.")
    submission_stacking = pd.DataFrame({'ID': test_index, 'W': stacking_preds_test_rounded})
    submission_stacking_path = os.path.join(submission_dir, 'submission_stacking.csv')
    submission_stacking.to_csv(submission_stacking_path, index=False)
    print(f"Saved stacking ensemble submission file to: {submission_stacking_path}")

    # OOF MAE by decade for stacking model
    oof_df = train[['yearID']].copy() if 'yearID' in train.columns else pd.DataFrame(index=train.index)
    oof_df['W_actual'] = y_train
    oof_df['W_pred'] = stacking_preds_train
    oof_df = oof_df.dropna(subset=['W_pred'])
    if 'yearID' in oof_df.columns:
        oof_df['decade'] = (oof_df['yearID'] // 10) * 10
        mae_by_decade = oof_df.groupby('decade').apply(
            lambda x: mean_absolute_error(x['W_actual'], x['W_pred'])
        )
        print("Stacking OOF MAE by decade:")
        print(mae_by_decade)
        oof_mae_decade_path = os.path.join(submission_dir, 'oof_mae_by_decade.csv')
        mae_by_decade.to_csv(oof_mae_decade_path)
        print(f"Saved OOF MAE by decade to: {oof_mae_decade_path}")

    print("Ridge, LightGBM tuning, and stacking ensemble pipeline finished successfully. Output files are saved.")

if __name__ == "__main__":
    run_pipeline()