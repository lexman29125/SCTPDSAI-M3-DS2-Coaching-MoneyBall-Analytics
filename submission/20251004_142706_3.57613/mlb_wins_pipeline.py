#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold

def run_pipeline(train_path='assets/train.csv', test_path='assets/test.csv'):
    # Load data
    print(f"Loading training data from: {train_path}")
    train = pd.read_csv(train_path)
    print(f"Loading test data from: {test_path}")
    test = pd.read_csv(test_path)

    print(f"Number of rows in train: {len(train)}")
    print(f"Number of rows in test: {len(test)}")

    # Basic feature engineering: create a feature for run differential (R - RA)
    train['run_diff'] = train['R'] - train['RA']
    test['run_diff'] = test['R'] - test['RA']

    # Additional feature engineering
    # Innings pitched (IP) - assuming IP column exists, else set to NaN
    if 'IP' not in train.columns:
        train['IP'] = np.nan
    if 'IP' not in test.columns:
        test['IP'] = np.nan

    # Games played (G) - assuming column exists
    if 'G' not in train.columns:
        train['G'] = np.nan
    if 'G' not in test.columns:
        test['G'] = np.nan

    # Runs per game (RPG)
    train['RPG'] = train['R'] / train['G']
    test['RPG'] = test['R'] / test['G']

    # Runs allowed per game (RAPG)
    train['RAPG'] = train['RA'] / train['G']
    test['RAPG'] = test['RA'] / test['G']

    # Run differential per game (RD_per_game)
    train['RD_per_game'] = train['run_diff'] / train['G']
    test['RD_per_game'] = test['run_diff'] / test['G']

    # Calculate league average runs per game (mlb_rpg) for each year
    train['team_rpg'] = train['R'] / train['G']
    test['team_rpg'] = test['R'] / test['G']
    # For train, group by yearID to get mlb_rpg
    if 'yearID' in train.columns:
        mlb_rpg_by_year = train.groupby('yearID').apply(lambda x: x['R'].sum() / x['G'].sum())
        train = train.merge(mlb_rpg_by_year.rename('mlb_rpg'), left_on='yearID', right_index=True)
    else:
        train['mlb_rpg'] = train['team_rpg'].mean()
    if 'yearID' in test.columns:
        # For test, merge mlb_rpg from train by yearID if possible, else use mean
        if 'mlb_rpg' in train.columns:
            test = test.merge(train[['yearID', 'mlb_rpg']].drop_duplicates('yearID'), on='yearID', how='left')
            test['mlb_rpg'] = test['mlb_rpg'].fillna(train['mlb_rpg'].mean())
        else:
            test['mlb_rpg'] = train['team_rpg'].mean()
    else:
        test['mlb_rpg'] = train['team_rpg'].mean()

    # Ensure mlb_rpg is present before computing RD_adj
    if 'mlb_rpg' not in train.columns:
        train['mlb_rpg'] = train['team_rpg'].mean()
    if 'mlb_rpg' not in test.columns:
        test['mlb_rpg'] = train['team_rpg'].mean()

    # RD_adj = run_diff * (mlb_rpg / team_rpg)
    train['RD_adj'] = train['run_diff'] * (train['mlb_rpg'] / train['team_rpg'])
    test['RD_adj'] = test['run_diff'] * (test['mlb_rpg'] / test['team_rpg'])

    # Pythagorean predicted wins
    # Using exponent 2 for simplicity
    train['Pythag_W'] = (train['R']**2) / (train['R']**2 + train['RA']**2) * train['G']
    test['Pythag_W'] = (test['R']**2) / (test['R']**2 + test['RA']**2) * test['G']

    # Pitching and batting rates per game
    for col in ['SO', 'BB', 'H', 'HR', 'SOA']:
        if col not in train.columns:
            train[col] = np.nan
        if col not in test.columns:
            test[col] = np.nan

    train['SO_per_game'] = train['SO'] / train['G']
    test['SO_per_game'] = test['SO'] / test['G']

    train['BB_per_game'] = train['BB'] / train['G']
    test['BB_per_game'] = test['BB'] / test['G']

    train['H_per_game'] = train['H'] / train['G']
    test['H_per_game'] = test['H'] / test['G']

    train['HR_per_game'] = train['HR'] / train['G']
    test['HR_per_game'] = test['HR'] / test['G']

    train['SOA_per_game'] = train['SOA'] / train['G']
    test['SOA_per_game'] = test['SOA'] / test['G']

    # Add lag features for previous season W, R, and RA
    if 'yearID' in train.columns and 'teamID' in train.columns:
        train = train.sort_values(['teamID', 'yearID'])
        train['W_lag1'] = train.groupby('teamID')['W'].shift(1)
        train['R_lag1'] = train.groupby('teamID')['R'].shift(1)
        train['RA_lag1'] = train.groupby('teamID')['RA'].shift(1)
        train['W_lag1_missing'] = train['W_lag1'].isna().astype(int)
        train['R_lag1_missing'] = train['R_lag1'].isna().astype(int)
        train['RA_lag1_missing'] = train['RA_lag1'].isna().astype(int)
        train['W_lag1'] = train['W_lag1'].fillna(0)
        train['R_lag1'] = train['R_lag1'].fillna(0)
        train['RA_lag1'] = train['RA_lag1'].fillna(0)
    else:
        train['W_lag1'] = np.nan
        train['R_lag1'] = np.nan
        train['RA_lag1'] = np.nan
        train['W_lag1_missing'] = 1
        train['R_lag1_missing'] = 1
        train['RA_lag1_missing'] = 1

    if 'yearID' in test.columns and 'teamID' in test.columns:
        test = test.sort_values(['teamID', 'yearID'])
        last_season = train[['teamID', 'yearID', 'W', 'R', 'RA']].copy()
        last_season['yearID'] += 1
        last_season = last_season.rename(columns={'W': 'W_lag1', 'R': 'R_lag1', 'RA': 'RA_lag1'})
        test = test.merge(last_season[['teamID', 'yearID', 'W_lag1', 'R_lag1', 'RA_lag1']], on=['teamID', 'yearID'], how='left')
        test['W_lag1_missing'] = test['W_lag1'].isna().astype(int)
        test['R_lag1_missing'] = test['R_lag1'].isna().astype(int)
        test['RA_lag1_missing'] = test['RA_lag1'].isna().astype(int)
        test['W_lag1'] = test['W_lag1'].fillna(0)
        test['R_lag1'] = test['R_lag1'].fillna(0)
        test['RA_lag1'] = test['RA_lag1'].fillna(0)
    else:
        test['W_lag1'] = np.nan
        test['R_lag1'] = np.nan
        test['RA_lag1'] = np.nan
        test['W_lag1_missing'] = 1
        test['R_lag1_missing'] = 1
        test['RA_lag1_missing'] = 1

    # Add interaction terms such as Pythag_W * ERA and RD_per_game * FP
    for col in ['ERA', 'FP']:
        if col not in train.columns:
            train[col] = np.nan
        if col not in test.columns:
            test[col] = np.nan

    train['Pythag_W_ERA'] = train['Pythag_W'] * train['ERA']
    test['Pythag_W_ERA'] = test['Pythag_W'] * test['ERA']

    train['RD_per_game_FP'] = train['RD_per_game'] * train['FP']
    test['RD_per_game_FP'] = test['RD_per_game'] * test['FP']

    # Target variable
    target_col = 'W'

    # Select features for base models
    base_features_ridge = [
        'Pythag_W', 'run_diff', 'RD_per_game', 'RD_per_game_FP',
        'R_lag1', 'RA_lag1', 'W_lag1',
        'Pythag_W_ERA', 'RD_adj',
        'RPG', 'RAPG',
        'HR_per_game', 'SO_per_game', 'BB_per_game'
    ]
    base_features_gbm = [
        'Pythag_W', 'run_diff', 'RD_per_game', 'RD_per_game_FP',
        'R_lag1', 'RA_lag1', 'W_lag1',
        'Pythag_W_ERA', 'RD_adj',
        'RPG', 'RAPG',
        'HR_per_game', 'SO_per_game', 'BB_per_game',
        'ERA', 'FP', 'SOA_per_game', 'H_per_game'
    ]
    print(f"Features used for Ridge base model: {base_features_ridge}")
    print(f"Features used for GBM base model: {base_features_gbm}")

    # Impute NaNs in features used for modeling before any modeling
    for col in set(base_features_ridge + base_features_gbm):
        train[col] = train[col].fillna(0)
        test[col] = test[col].fillna(0)

    # Prepare index for test submission (assuming 'ID' column exists)
    if 'ID' in test.columns:
        test_index = test['ID']
        print("Test data contains 'ID' column for submission index.")
    else:
        test_index = test.index
        print("Test data does not contain 'ID' column; using index for submission.")

    # Prepare folds for cross-validation: KFold with shuffle and fixed random state
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Initialize out-of-fold predictions for base models
    oof_preds_ridge = np.zeros(len(train))
    oof_preds_gbm = np.zeros(len(train))

    # Initialize test predictions for base models
    test_preds_ridge = np.zeros(len(test))
    test_preds_gbm = np.zeros(len(test))

    # Ridge Regression base model
    print("\nTraining Ridge Regression base model with 5-fold CV for OOF predictions...")
    ridge_alpha = 1.0  # fixed alpha for base model
    for train_idx, val_idx in kf.split(train):
        X_train = train.iloc[train_idx][base_features_ridge]
        y_train = train.iloc[train_idx][target_col]
        X_val = train.iloc[val_idx][base_features_ridge]
        ridge = Ridge(alpha=ridge_alpha, random_state=42)
        ridge.fit(X_train, y_train)
        oof_preds_ridge[val_idx] = ridge.predict(X_val)
        test_preds_ridge += ridge.predict(test[base_features_ridge]) / n_splits

    mae_ridge = mean_absolute_error(train[target_col], oof_preds_ridge)
    print(f"Ridge base model OOF MAE: {mae_ridge:.4f}")

    # Gradient Boosting base model
    print("\nTraining Gradient Boosting base model with 5-fold CV for OOF predictions...")
    for train_idx, val_idx in kf.split(train):
        X_train = train.iloc[train_idx][base_features_gbm]
        y_train = train.iloc[train_idx][target_col]
        X_val = train.iloc[val_idx][base_features_gbm]
        gbm = GradientBoostingRegressor(random_state=42, n_estimators=100, max_depth=3)
        gbm.fit(X_train, y_train)
        oof_preds_gbm[val_idx] = gbm.predict(X_val)
        test_preds_gbm += gbm.predict(test[base_features_gbm]) / n_splits

    mae_gbm = mean_absolute_error(train[target_col], oof_preds_gbm)
    print(f"Gradient Boosting base model OOF MAE: {mae_gbm:.4f}")

    # Prepare stacking dataset
    print("\nPreparing stacking dataset...")
    train_stack = pd.DataFrame({
        'ridge_pred': oof_preds_ridge,
        'gbm_pred': oof_preds_gbm,
        'W_actual': train[target_col]
    })
    test_stack = pd.DataFrame({
        'ridge_pred': test_preds_ridge,
        'gbm_pred': test_preds_gbm
    })

    # Train meta-model (Ridge) on stacking dataset with CV
    print("\nTraining stacking meta-model (Ridge)...")
    oof_preds_meta = np.zeros(len(train))
    test_preds_meta = np.zeros(len(test))
    for train_idx, val_idx in kf.split(train_stack):
        X_train = train_stack.iloc[train_idx][['ridge_pred', 'gbm_pred']]
        y_train = train_stack.iloc[train_idx]['W_actual']
        X_val = train_stack.iloc[val_idx][['ridge_pred', 'gbm_pred']]
        meta_model = Ridge(alpha=1.0, random_state=42)
        meta_model.fit(X_train, y_train)
        oof_preds_meta[val_idx] = meta_model.predict(X_val)
        test_preds_meta += meta_model.predict(test_stack) / n_splits

    mae_meta = mean_absolute_error(train_stack['W_actual'], oof_preds_meta)
    print(f"Stacking meta-model OOF MAE: {mae_meta:.4f}")

    # Clip predictions to reasonable bounds
    test_preds_meta_rounded = np.round(test_preds_meta).astype(int)
    test_preds_meta_rounded = np.clip(test_preds_meta_rounded, 40, 120)

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_dir = os.path.join('submission', timestamp)
    os.makedirs(submission_dir, exist_ok=True)

    # Save OOF predictions for meta-model
    oof_df = train[['yearID']].copy() if 'yearID' in train.columns else pd.DataFrame(index=train.index)
    oof_df['W_actual'] = train[target_col]
    oof_df['W_pred'] = oof_preds_meta
    oof_df = oof_df.dropna(subset=['W_pred'])
    oof_path = os.path.join(submission_dir, 'oof_predictions_stacking.csv')
    oof_df.to_csv(oof_path, index=False)
    print(f"Saved stacking OOF predictions to: {oof_path}")

    # OOF evaluation by decade if yearID exists
    if 'yearID' in oof_df.columns:
        oof_df['decade'] = (oof_df['yearID'] // 10) * 10
        mae_by_decade = oof_df.groupby('decade').apply(
            lambda x: mean_absolute_error(x['W_actual'], x['W_pred'])
        )
        print("Stacking OOF MAE by decade:")
        print(mae_by_decade)

    # Plot OOF predicted vs actual wins (stacking)
    plt.figure(figsize=(8, 8))
    plt.scatter(oof_df['W_actual'], oof_df['W_pred'], alpha=0.5, edgecolors='k')
    plt.plot([oof_df['W_actual'].min(), oof_df['W_actual'].max()],
             [oof_df['W_actual'].min(), oof_df['W_actual'].max()],
             'r--', lw=2)
    plt.xlabel('Actual Wins')
    plt.ylabel('OOF Predicted Wins')
    plt.title('OOF Predicted vs Actual Wins (Stacking)')
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(submission_dir, 'oof_pred_vs_actual_stacking.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved stacking OOF predicted vs actual plot to: {plot_path}")

    # Save final submission file
    submission = pd.DataFrame({'ID': test_index, 'W': test_preds_meta_rounded})
    submission_path = os.path.join(submission_dir, 'submission_stacking.csv')
    submission.to_csv(submission_path, index=False)
    print(f"Saved stacking submission file to: {submission_path}")

    # Save meta-model coefficients as feature importances
    fi_path = os.path.join(submission_dir, 'meta_model_feature_importances.csv')
    meta_coefs = pd.DataFrame({'feature': ['ridge_pred', 'gbm_pred'], 'coefficient': meta_model.coef_})
    meta_coefs.to_csv(fi_path, index=False)
    print(f"Saved stacking meta-model feature importances (coefficients) to: {fi_path}")

    print("Stacking pipeline finished successfully. Output files are saved.")

if __name__ == "__main__":
    run_pipeline()