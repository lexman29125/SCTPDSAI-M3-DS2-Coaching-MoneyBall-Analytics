#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

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

    # Pythagorean predicted wins
    # Using exponent 2 for simplicity
    train['Pythag_W'] = (train['R']**2) / (train['R']**2 + train['RA']**2) * train['G']
    test['Pythag_W'] = (test['R']**2) / (test['R']**2 + test['RA']**2) * test['G']

    # Pitching and batting rates per game
    # Assuming columns: SO (strikeouts), ERA, H, HR, BB, SOA (strikeouts against), etc.
    # We'll create SO_per_game and BB_per_game if columns exist
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
        # Create missing flags for lag features
        train['W_lag1_missing'] = train['W_lag1'].isna().astype(int)
        train['R_lag1_missing'] = train['R_lag1'].isna().astype(int)
        train['RA_lag1_missing'] = train['RA_lag1'].isna().astype(int)
        # Fill NaNs with zeros or median if preferred (here zero)
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
        # For test set lag features, merge with train to get last season stats
        last_season = train[['teamID', 'yearID', 'W', 'R', 'RA']].copy()
        last_season['yearID'] += 1
        last_season = last_season.rename(columns={'W': 'W_lag1', 'R': 'R_lag1', 'RA': 'RA_lag1'})
        test = test.merge(last_season[['teamID', 'yearID', 'W_lag1', 'R_lag1', 'RA_lag1']], on=['teamID', 'yearID'], how='left')
        # Create missing flags for lag features
        test['W_lag1_missing'] = test['W_lag1'].isna().astype(int)
        test['R_lag1_missing'] = test['R_lag1'].isna().astype(int)
        test['RA_lag1_missing'] = test['RA_lag1'].isna().astype(int)
        # Fill NaNs with zeros or median if preferred (here zero)
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
    # Ensure ERA and FP exist, else fill with NaN
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

    # Select numeric features for modeling
    numeric_features = [
        'R', 'RA', 'run_diff', 'RPG', 'RAPG', 'RD_per_game', 'Pythag_W',
        'IP', 'SO_per_game', 'BB_per_game', 'H_per_game', 'HR_per_game', 'SOA_per_game',
        'W_lag1', 'R_lag1', 'RA_lag1',
        'W_lag1_missing', 'R_lag1_missing', 'RA_lag1_missing',
        'Pythag_W_ERA', 'RD_per_game_FP'
    ]

    print(f"Features used for modeling: {numeric_features}")

    # Prepare train and test feature sets
    train_fe = train.copy()
    test_fe = test.copy()

    # Impute NaNs in features used for modeling (numeric_features) before any modeling
    for col in numeric_features:
        train_fe[col] = train_fe[col].fillna(0)
        test_fe[col] = test_fe[col].fillna(0)

    # Prepare index for test submission (assuming 'ID' column exists)
    if 'ID' in test_fe.columns:
        test_index = test_fe['ID']
        print("Test data contains 'ID' column for submission index.")
    else:
        test_index = test_fe.index
        print("Test data does not contain 'ID' column; using index for submission.")

    # Prepare folds for cross-validation: simple rolling-year split by year if available
    if 'yearID' in train_fe.columns:
        years = sorted(train_fe['yearID'].unique())
        splits = []
        for i in range(len(years)-1):
            train_years = years[:i+1]
            val_year = years[i+1]
            splits.append((train_years, val_year))
    else:
        splits = []

    # For Ridge, use all numeric_features, or optionally select top features based on absolute correlation with target
    # For consistency, use all features, but you can select top ones by correlation if desired.
    top_features = numeric_features.copy()

    # Impute NaNs in top_features (again, for safety)
    for col in top_features:
        train_fe[col] = train_fe[col].fillna(0)
        test_fe[col] = test_fe[col].fillna(0)

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_dir = os.path.join('submission', timestamp)
    os.makedirs(submission_dir, exist_ok=True)

    # Ridge Regression as the only model: OOF with rolling-year CV, MAE, retrain, and test prediction
    # Initially use all features for OOF prediction to get feature importances
    print("\nTraining Ridge Regression (primary model) with rolling-year CV for OOF predictions...")
    oof_ridge_primary = np.full(len(train_fe), np.nan)
    for train_years, val_year in splits:
        train_mask = train_fe['yearID'].isin(train_years)
        val_mask = (train_fe['yearID'] == val_year)
        X_train = train_fe.loc[train_mask, top_features]
        y_train = train_fe.loc[train_mask, target_col]
        X_val = train_fe.loc[val_mask, top_features]
        ridge = Ridge(random_state=42)
        ridge.fit(X_train, y_train)
        oof_ridge_primary[val_mask] = ridge.predict(X_val)

    # Save OOF predictions for Ridge as primary
    oof_ridge_primary_df = train_fe[['yearID']].copy() if 'yearID' in train_fe.columns else pd.DataFrame(index=train_fe.index)
    oof_ridge_primary_df['W_actual'] = train_fe[target_col]
    oof_ridge_primary_df['W_pred'] = oof_ridge_primary
    oof_ridge_primary_df = oof_ridge_primary_df.dropna(subset=['W_pred'])
    oof_path = os.path.join(submission_dir, 'oof_predictions.csv')
    oof_ridge_primary_df.to_csv(oof_path, index=False)
    print(f"Saved OOF predictions to: {oof_path}")

    # OOF evaluation: MAE overall and by era/decade (Ridge)
    mae_overall = mean_absolute_error(oof_ridge_primary_df['W_actual'], oof_ridge_primary_df['W_pred'])
    print(f"Overall OOF MAE: {mae_overall:.4f}")
    if 'yearID' in oof_ridge_primary_df.columns:
        oof_ridge_primary_df['decade'] = (oof_ridge_primary_df['yearID'] // 10) * 10
        mae_by_decade = oof_ridge_primary_df.groupby('decade').apply(
            lambda x: mean_absolute_error(x['W_actual'], x['W_pred'])
        )
        print("OOF MAE by decade:")
        print(mae_by_decade)

    # Plot OOF predicted vs actual wins (Ridge)
    plt.figure(figsize=(8, 8))
    plt.scatter(oof_ridge_primary_df['W_actual'], oof_ridge_primary_df['W_pred'], alpha=0.5, edgecolors='k')
    plt.plot([oof_ridge_primary_df['W_actual'].min(), oof_ridge_primary_df['W_actual'].max()],
             [oof_ridge_primary_df['W_actual'].min(), oof_ridge_primary_df['W_actual'].max()],
             'r--', lw=2)
    plt.xlabel('Actual Wins')
    plt.ylabel('OOF Predicted Wins')
    plt.title('OOF Predicted vs Actual Wins (Ridge)')
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(submission_dir, 'oof_pred_vs_actual.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved OOF predicted vs actual plot to: {plot_path}")

    # Retrain Ridge on full train set and predict test set
    print("\nTraining Ridge Regression on full train set and predicting test set for submission...")
    ridge_final = Ridge(random_state=42)
    ridge_final.fit(train_fe[top_features], train_fe[target_col])
    test_preds_ridge = ridge_final.predict(test_fe[top_features])
    test_preds_rounded = np.round(test_preds_ridge).astype(int)
    test_preds_rounded = np.clip(test_preds_rounded, 40, 120)
    submission = pd.DataFrame({'ID': test_index, 'W': test_preds_rounded})
    submission_path = os.path.join(submission_dir, 'submission.csv')
    submission.to_csv(submission_path, index=False)
    print(f"Saved submission file to: {submission_path}")

    # Save Ridge coefficients as feature importances
    fi_path = os.path.join(submission_dir, 'feature_importances.csv')
    ridge_coefs = pd.DataFrame({'feature': top_features, 'coefficient': ridge_final.coef_})
    ridge_coefs.to_csv(fi_path, index=False)
    print(f"Saved Ridge feature importances (coefficients) to: {fi_path}")

    # Use a fixed set of top 5 features for Ridge Regression
    fixed_top5_features = [
        'Pythag_W',
        'run_diff',
        'RD_per_game_FP',
        'RD_per_game',
        'R_lag1'
    ]
    print(f"\nUsing fixed top 5 features for Ridge Regression: {fixed_top5_features}")

    # Now restrict OOF prediction generation and final model training/prediction to these fixed top 5 features
    print("\nRe-training Ridge Regression with fixed top 5 features for OOF predictions...")
    oof_ridge_top5 = np.full(len(train_fe), np.nan)
    for train_years, val_year in splits:
        train_mask = train_fe['yearID'].isin(train_years)
        val_mask = (train_fe['yearID'] == val_year)
        X_train = train_fe.loc[train_mask, fixed_top5_features]
        y_train = train_fe.loc[train_mask, target_col]
        X_val = train_fe.loc[val_mask, fixed_top5_features]
        ridge_top5 = Ridge(random_state=42)
        ridge_top5.fit(X_train, y_train)
        oof_ridge_top5[val_mask] = ridge_top5.predict(X_val)

    # Save OOF predictions for Ridge with fixed top 5 features
    oof_ridge_top5_df = train_fe[['yearID']].copy() if 'yearID' in train_fe.columns else pd.DataFrame(index=train_fe.index)
    oof_ridge_top5_df['W_actual'] = train_fe[target_col]
    oof_ridge_top5_df['W_pred'] = oof_ridge_top5
    oof_ridge_top5_df = oof_ridge_top5_df.dropna(subset=['W_pred'])
    oof_top5_path = os.path.join(submission_dir, 'oof_predictions.csv')
    oof_ridge_top5_df.to_csv(oof_top5_path, index=False)
    print(f"Saved OOF predictions with fixed top 5 features to: {oof_top5_path}")

    # OOF evaluation: MAE overall and by era/decade (Ridge fixed top 5)
    mae_overall_top5 = mean_absolute_error(oof_ridge_top5_df['W_actual'], oof_ridge_top5_df['W_pred'])
    print(f"Overall OOF MAE with fixed top 5 features: {mae_overall_top5:.4f}")
    if 'yearID' in oof_ridge_top5_df.columns:
        oof_ridge_top5_df['decade'] = (oof_ridge_top5_df['yearID'] // 10) * 10
        mae_by_decade_top5 = oof_ridge_top5_df.groupby('decade').apply(
            lambda x: mean_absolute_error(x['W_actual'], x['W_pred'])
        )
        print("OOF MAE by decade with fixed top 5 features:")
        print(mae_by_decade_top5)

    # Plot OOF predicted vs actual wins (Ridge fixed top 5)
    plt.figure(figsize=(8, 8))
    plt.scatter(oof_ridge_top5_df['W_actual'], oof_ridge_top5_df['W_pred'], alpha=0.5, edgecolors='k')
    plt.plot([oof_ridge_top5_df['W_actual'].min(), oof_ridge_top5_df['W_actual'].max()],
             [oof_ridge_top5_df['W_actual'].min(), oof_ridge_top5_df['W_actual'].max()],
             'r--', lw=2)
    plt.xlabel('Actual Wins')
    plt.ylabel('OOF Predicted Wins')
    plt.title('OOF Predicted vs Actual Wins (Ridge - Fixed Top 5 Features)')
    plt.grid(True)
    plt.tight_layout()
    plot_top5_path = os.path.join(submission_dir, 'oof_pred_vs_actual.png')
    plt.savefig(plot_top5_path)
    plt.close()
    print(f"Saved OOF predicted vs actual plot with fixed top 5 features to: {plot_top5_path}")

    # Retrain Ridge on full train set with fixed top 5 features and predict test set
    print("\nTraining Ridge Regression on full train set with fixed top 5 features and predicting test set for submission...")
    ridge_final_top5 = Ridge(random_state=42)
    ridge_final_top5.fit(train_fe[fixed_top5_features], train_fe[target_col])
    test_preds_ridge_top5 = ridge_final_top5.predict(test_fe[fixed_top5_features])
    test_preds_rounded_top5 = np.round(test_preds_ridge_top5).astype(int)
    test_preds_rounded_top5 = np.clip(test_preds_rounded_top5, 40, 120)
    submission_top5 = pd.DataFrame({'ID': test_index, 'W': test_preds_rounded_top5})
    submission_top5_path = os.path.join(submission_dir, 'submission.csv')
    submission_top5.to_csv(submission_top5_path, index=False)
    print(f"Saved submission file with fixed top 5 features to: {submission_top5_path}")

    print("Pipeline finished successfully. Output files are saved.")

if __name__ == "__main__":
    run_pipeline()