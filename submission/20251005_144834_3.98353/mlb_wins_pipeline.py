#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
# --- Additional imports for stacking and tuning ---
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold, cross_val_predict
import optuna
from sklearn.ensemble import StackingRegressor
import warnings

def run_pipeline(train_path='assets/train.csv', test_path='assets/test.csv'):
    # Load data
    print(f"Loading training data from: {train_path}")
    train = pd.read_csv(train_path)
    print(f"Loading test data from: {test_path}")
    test = pd.read_csv(test_path)

    # --- Add rolling lag features: rolling mean of W, R, RA over previous 2 seasons for each teamID ---
    # Use per-team median imputation and missingness flags for lags
    if 'yearID' in train.columns and 'teamID' in train.columns:
        train = train.sort_values(['teamID', 'yearID'])
        # Rolling mean over previous 2 seasons (lags 1 and 2)
        train['W_lag2_mean'] = train.groupby('teamID')['W'].transform(lambda x: x.shift(1).rolling(2, min_periods=1).mean())
        train['R_lag2_mean'] = train.groupby('teamID')['R'].transform(lambda x: x.shift(1).rolling(2, min_periods=1).mean())
        train['RA_lag2_mean'] = train.groupby('teamID')['RA'].transform(lambda x: x.shift(1).rolling(2, min_periods=1).mean())
        # Missingness flags
        train['W_lag2_mean_missing'] = train['W_lag2_mean'].isna().astype(int)
        train['R_lag2_mean_missing'] = train['R_lag2_mean'].isna().astype(int)
        train['RA_lag2_mean_missing'] = train['RA_lag2_mean'].isna().astype(int)
        # Per-team median imputation
        for col in ['W_lag2_mean', 'R_lag2_mean', 'RA_lag2_mean']:
            medians = train.groupby('teamID')[col].transform('median')
            train[col] = train[col].fillna(medians)
        # If still NaN (all NaN for team), fill with global median
        for col in ['W_lag2_mean', 'R_lag2_mean', 'RA_lag2_mean']:
            train[col] = train[col].fillna(train[col].median())
    else:
        train['W_lag2_mean'] = 0
        train['R_lag2_mean'] = 0
        train['RA_lag2_mean'] = 0
        train['W_lag2_mean_missing'] = 1
        train['R_lag2_mean_missing'] = 1
        train['RA_lag2_mean_missing'] = 1

    # For test: merge rolling lag features from train, aligned by teamID and yearID
    if 'yearID' in test.columns and 'teamID' in test.columns:
        lag2_df = train[['teamID', 'yearID', 'W_lag2_mean', 'R_lag2_mean', 'RA_lag2_mean',
                         'W_lag2_mean_missing', 'R_lag2_mean_missing', 'RA_lag2_mean_missing']].copy()
        lag2_df['yearID'] += 1
        test = test.merge(
            lag2_df,
            on=['teamID', 'yearID'],
            how='left'
        )
        for col in ['W_lag2_mean', 'R_lag2_mean', 'RA_lag2_mean']:
            # Per-team median imputation
            medians = train.groupby('teamID')[col].median()
            test[col] = test.apply(lambda row: medians[row['teamID']] if pd.isna(row[col]) and row['teamID'] in medians else row[col], axis=1)
            test[col] = test[col].fillna(train[col].median())
        for col in ['W_lag2_mean_missing', 'R_lag2_mean_missing', 'RA_lag2_mean_missing']:
            test[col] = test[col].fillna(1).astype(int)
    else:
        test['W_lag2_mean'] = 0
        test['R_lag2_mean'] = 0
        test['RA_lag2_mean'] = 0
        test['W_lag2_mean_missing'] = 1
        test['R_lag2_mean_missing'] = 1
        test['RA_lag2_mean_missing'] = 1

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

    # Add lag features for previous season W, R, and RA (with per-team median imputation and missingness)
    if 'yearID' in train.columns and 'teamID' in train.columns:
        train = train.sort_values(['teamID', 'yearID'])
        train['W_lag1'] = train.groupby('teamID')['W'].shift(1)
        train['R_lag1'] = train.groupby('teamID')['R'].shift(1)
        train['RA_lag1'] = train.groupby('teamID')['RA'].shift(1)
        train['W_lag1_missing'] = train['W_lag1'].isna().astype(int)
        train['R_lag1_missing'] = train['R_lag1'].isna().astype(int)
        train['RA_lag1_missing'] = train['RA_lag1'].isna().astype(int)
        # Per-team median imputation
        for col in ['W_lag1', 'R_lag1', 'RA_lag1']:
            medians = train.groupby('teamID')[col].transform('median')
            train[col] = train[col].fillna(medians)
        for col in ['W_lag1', 'R_lag1', 'RA_lag1']:
            train[col] = train[col].fillna(train[col].median())
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
        for col in ['W_lag1', 'R_lag1', 'RA_lag1']:
            medians = train.groupby('teamID')[col].median()
            test[col] = test.apply(lambda row: medians[row['teamID']] if pd.isna(row[col]) and row['teamID'] in medians else row[col], axis=1)
            test[col] = test[col].fillna(train[col].median())
        test['W_lag1_missing'] = test['W_lag1'].isna().astype(int)
        test['R_lag1_missing'] = test['R_lag1'].isna().astype(int)
        test['RA_lag1_missing'] = test['RA_lag1'].isna().astype(int)
    else:
        test['W_lag1'] = np.nan
        test['R_lag1'] = np.nan
        test['RA_lag1'] = np.nan
        test['W_lag1_missing'] = 1
        test['R_lag1_missing'] = 1
        test['RA_lag1_missing'] = 1

    # Remove weak or redundant interaction terms and keep only strong predictors and their clean features
    # Ensure all needed columns exist in both train and test
    for col in ['RD_per_game','ERA','FP','RD_adj','Pythag_W','W_lag1','W_lag2_mean','R_lag1','RA_lag1','SO_per_game','BB_per_game']:
        if col not in train.columns:
            train[col] = np.nan
        if col not in test.columns:
            test[col] = np.nan

    # --- Modern sabermetric features ---
    # Ensure HBP and SF columns exist in both train and test, fill with 0 if missing
    for col in ['HBP', 'SF']:
        if col not in train.columns:
            train[col] = 0
        if col not in test.columns:
            test[col] = 0
    # On-base percentage (OBP)
    train['OBP'] = (train['H'] + train['BB'] + train['HBP']) / (train['AB'] + train['BB'] + train['HBP'] + train['SF'] + 1e-5)
    test['OBP'] = (test['H'] + test['BB'] + test['HBP']) / (test['AB'] + test['BB'] + test['HBP'] + test['SF'] + 1e-5)

    # Slugging percentage (SLG)
    train['SLG'] = ((train['H'] - train['2B'] - train['3B'] - train['HR']) + 2*train['2B'] + 3*train['3B'] + 4*train['HR']) / (train['AB'] + 1e-5)
    test['SLG'] = ((test['H'] - test['2B'] - test['3B'] - test['HR']) + 2*test['2B'] + 3*test['3B'] + 4*test['HR']) / (test['AB'] + 1e-5)

    # OPS (On-base Plus Slugging)
    train['OPS'] = train['OBP'] + train['SLG']
    test['OPS'] = test['OBP'] + test['SLG']

    # Safe IP calculation
    train['IP'] = train['IPouts'] / 3
    test['IP'] = test['IPouts'] / 3

    # Fielding Independent Pitching (FIP) with 3.1 constant and safe denominator
    train['FIP'] = ((13 * train['HRA'] + 3 * train['BBA'] - 2 * train['SOA']) / (train['IP'] + 1e-5)) + 3.1
    test['FIP'] = ((13 * test['HRA'] + 3 * test['BBA'] - 2 * test['SOA']) / (test['IP'] + 1e-5)) + 3.1

    print("Added modern sabermetric features: OBP, SLG, OPS, FIP")

    # Replace infinite and NaN values
    train.replace([np.inf, -np.inf], np.nan, inplace=True)
    test.replace([np.inf, -np.inf], np.nan, inplace=True)
    train.fillna(0, inplace=True)
    test.fillna(0, inplace=True)
    print("Cleaned dataset: replaced NaN and infinite values.")

    # Target variable
    target_col = 'W'

    # --- Feature sets for stacking models ---
    # Ridge: top classical predictors
    ridge_features = [
        'RD', 'RD_per_game', 'Pythag_W', 'ERA', 'SO_per_game', 'BB_per_game'
    ]
    # LightGBM: 12–15 features, including modern sabermetrics and era-normalized RD
    lgb_features = [
        'RD', 'RD_per_game', 'RD_adj', 'Pythag_W', 'ERA', 'SO_per_game', 'BB_per_game',
        'OBP', 'SLG', 'OPS', 'FIP', 'W_lag1', 'W_lag1_missing', 'W_lag2_mean', 'W_lag2_mean_missing'
    ]
    # CatBoost: same as LGBM plus era indicators (if present)
    era_cols = [col for col in train.columns if col.startswith('era_') or col.startswith('decade_')]
    cat_features = lgb_features + era_cols

    # Make sure all features exist in train and test
    for featlist in [ridge_features, lgb_features, cat_features]:
        for col in featlist:
            if col not in train.columns:
                train[col] = 0
            if col not in test.columns:
                test[col] = 0

    # --- CatBoost: Ensure era/decade columns are string/object and passed as categorical ---
    # Identify all categorical columns (era/decade)
    catboost_cat_cols = [col for col in cat_features if col in era_cols]
    # Ensure categorical columns are string/object dtype
    for col in catboost_cat_cols:
        train[col] = train[col].astype(str)
        test[col] = test[col].astype(str)
    # Ensure numeric columns are float
    for col in cat_features:
        if col not in catboost_cat_cols:
            train[col] = train[col].astype(float)
            test[col] = test[col].astype(float)

    # Prepare training and test data for each model
    X_train_ridge = train[ridge_features].copy()
    X_test_ridge = test[ridge_features].copy()
    X_train_lgb = train[lgb_features].copy()
    X_test_lgb = test[lgb_features].copy()
    X_train_cat = train[cat_features].copy()
    X_test_cat = test[cat_features].copy()
    y_train = train[target_col].copy()

    # --- Prepare X_train_cat and X_test_cat: clean numeric columns and ensure dtypes ---
    # Identify numeric columns as those not in catboost_cat_cols
    numeric_cols = [c for c in X_train_cat.columns if c not in catboost_cat_cols]
    # For each numeric column, replace 'False' with 0 and 'True' with 1, then coerce to float, fillna 0
    for col in numeric_cols:
        X_train_cat[col] = (
            X_train_cat[col]
            .replace({'False': 0, 'True': 1})
            .apply(pd.to_numeric, errors='coerce')
            .fillna(0)
            .astype(float)
        )
        X_test_cat[col] = (
            X_test_cat[col]
            .replace({'False': 0, 'True': 1})
            .apply(pd.to_numeric, errors='coerce')
            .fillna(0)
            .astype(float)
        )
    # For categorical columns, ensure string dtype
    for col in catboost_cat_cols:
        X_train_cat[col] = X_train_cat[col].astype(str)
        X_test_cat[col] = X_test_cat[col].astype(str)

    # --- Preprocess categorical columns for stacking (LightGBM compatibility) ---
    # Identify categorical columns as catboost_cat_cols
    from sklearn.preprocessing import OneHotEncoder
    # Fit OneHotEncoder on all categorical columns in train
    ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
    X_train_cat_ohe_arr = ohe.fit_transform(X_train_cat[catboost_cat_cols])
    X_test_cat_ohe_arr = ohe.transform(X_test_cat[catboost_cat_cols])
    # Get feature names for OHE columns
    ohe_feature_names = ohe.get_feature_names_out(catboost_cat_cols)
    # Create DataFrames for OHE columns, align index with input
    X_train_cat_ohe = pd.DataFrame(X_train_cat_ohe_arr, columns=ohe_feature_names, index=X_train_cat.index)
    X_test_cat_ohe = pd.DataFrame(X_test_cat_ohe_arr, columns=ohe_feature_names, index=X_test_cat.index)
    # Select numeric columns (already float, no object)
    X_train_numeric = X_train_cat[numeric_cols]
    X_test_numeric = X_test_cat[numeric_cols]
    # Concatenate numeric with OHE categorical columns
    X_train_stacking = pd.concat([X_train_numeric, X_train_cat_ohe], axis=1)
    X_test_stacking = pd.concat([X_test_numeric, X_test_cat_ohe], axis=1)

    # Scale features for each model
    scaler_ridge = StandardScaler()
    X_train_ridge_scaled = scaler_ridge.fit_transform(X_train_ridge)
    X_test_ridge_scaled = scaler_ridge.transform(X_test_ridge)
    scaler_lgb = StandardScaler()
    X_train_lgb_scaled = scaler_lgb.fit_transform(X_train_lgb)
    X_test_lgb_scaled = scaler_lgb.transform(X_test_lgb)
    # CatBoost: scale only numeric columns, leave categorical as is, and keep as DataFrame (not numpy array)
    cat_num_cols = [c for c in cat_features if c not in catboost_cat_cols]
    scaler_cat = StandardScaler()
    X_train_cat_scaled = X_train_cat.copy()
    X_test_cat_scaled = X_test_cat.copy()
    # Only scale numeric columns, leave categorical columns untouched
    X_train_cat_scaled[cat_num_cols] = scaler_cat.fit_transform(X_train_cat[cat_num_cols])
    X_test_cat_scaled[cat_num_cols] = scaler_cat.transform(X_test_cat[cat_num_cols])

    # Train Ridge Regression model (classical top predictors)
    lr = Ridge(alpha=1.0, random_state=42)
    lr.fit(X_train_ridge_scaled, y_train)
    train_preds = lr.predict(X_train_ridge_scaled)
    test_preds = lr.predict(X_test_ridge_scaled)
    mae = mean_absolute_error(y_train, train_preds)
    rmse = math.sqrt(mean_squared_error(y_train, train_preds))
    r2 = r2_score(y_train, train_preds)
    print(f"Ridge Regression (top predictors) training MAE: {mae:.4f}")
    print(f"Ridge Regression training RMSE: {rmse:.4f}")
    print(f"Ridge Regression training R^2: {r2:.4f}")

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_dir = os.path.join('submission', timestamp)
    os.makedirs(submission_dir, exist_ok=True)

    # Feature importance from Ridge coefficients
    coef_df = pd.DataFrame({
        'feature': ridge_features,
        'coefficient': lr.coef_
    })
    coef_df['abs_coefficient'] = coef_df['coefficient'].abs()
    coef_df = coef_df.sort_values(by='abs_coefficient', ascending=False)
    print("Top features by absolute coefficient (Ridge):")
    print(coef_df.head(10)[['feature', 'coefficient']])
    feature_importances_path = os.path.join(submission_dir, 'ridge_feature_importances.csv')
    coef_df[['feature', 'coefficient']].to_csv(feature_importances_path, index=False)
    print(f"Saved Ridge feature importances to: {feature_importances_path}")

    # Clip predictions to reasonable bounds
    test_preds_rounded = np.round(test_preds).astype(int)
    test_preds_rounded = np.clip(test_preds_rounded, 40, 120)

    # Save training predictions (OOF not applicable here, so save train preds)
    oof_df = train[['yearID']].copy() if 'yearID' in train.columns else pd.DataFrame(index=train.index)
    oof_df['W_actual'] = y_train
    oof_df['W_pred'] = train_preds
    oof_df = oof_df.dropna(subset=['W_pred'])
    oof_path = os.path.join(submission_dir, 'train_predictions_ridge.csv')
    oof_df.to_csv(oof_path, index=False)
    print(f"Saved Ridge training predictions to: {oof_path}")

    # Evaluation by decade if yearID exists
    if 'yearID' in oof_df.columns:
        oof_df['decade'] = (oof_df['yearID'] // 10) * 10
        mae_by_decade = oof_df.groupby('decade').apply(
            lambda x: mean_absolute_error(x['W_actual'], x['W_pred'])
        )
        print("Ridge Regression training MAE by decade:")
        print(mae_by_decade)

    # Plot predicted vs actual wins (training data)
    plt.figure(figsize=(8, 8))
    plt.scatter(oof_df['W_actual'], oof_df['W_pred'], alpha=0.5, edgecolors='k')
    plt.plot([oof_df['W_actual'].min(), oof_df['W_actual'].max()],
             [oof_df['W_actual'].min(), oof_df['W_actual'].max()],
             'r--', lw=2)
    plt.xlabel('Actual Wins')
    plt.ylabel('Predicted Wins')
    plt.title('Predicted vs Actual Wins (Linear Regression Training Data)')
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(submission_dir, 'pred_vs_actual_linear_regression_train.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved predicted vs actual plot to: {plot_path}")

    # Prepare index for test submission (assuming 'ID' column exists)
    if 'ID' in test.columns:
        test_index = test['ID']
        print("Test data contains 'ID' column for submission index.")
    else:
        test_index = test.index
        print("Test data does not contain 'ID' column; using index for submission.")

    # Save final submission file
    submission = pd.DataFrame({'ID': test_index, 'W': test_preds_rounded})
    submission_path = os.path.join(submission_dir, 'submission_linear_regression.csv')
    submission.to_csv(submission_path, index=False)
    print(f"Saved submission file to: {submission_path}")

    print("Linear Regression pipeline finished successfully. Output files are saved.")

    # --- Suppress LightGBM warnings globally ---
    warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
    import sys
    import contextlib
    @contextlib.contextmanager
    def suppress_stdout_stderr():
        with open(os.devnull, 'w') as devnull:
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            try:
                sys.stdout = devnull
                sys.stderr = devnull
                yield
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

    # --- Define Optuna tuning functions ---
    def tune_ridge(X, y, n_trials=20):
        from sklearn.model_selection import cross_val_predict
        def objective(trial):
            alpha = trial.suggest_loguniform('alpha', 1e-3, 10)
            model = Ridge(alpha=alpha, random_state=42)
            scores = cross_val_predict(model, X, y, cv=5, method='predict')
            return mean_absolute_error(y, scores)
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        return study.best_params['alpha']

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
            with suppress_stdout_stderr():
                cv_results = lgb.cv(
                    param,
                    lgb.Dataset(X, label=y),
                    nfold=5,
                    metrics='mae',
                    seed=42,
                    stratified=False
                )
            mae_key = [k for k in cv_results.keys() if 'l1' in k][0]
            return min(cv_results[mae_key])
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        return study.best_params

    def tune_catboost(X, y, n_trials=20):
        from sklearn.model_selection import cross_val_predict
        def objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 100, 500),
                'depth': trial.suggest_int('depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'loss_function': 'MAE',
                'verbose': False,
                'random_seed': 42
            }
            # Pass cat_features indices for CatBoost
            cat_feat_indices = [X.columns.get_loc(c) for c in catboost_cat_cols]
            model = CatBoostRegressor(**params, cat_features=cat_feat_indices)
            # Suppress CatBoost type warnings about auto conversion
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, module="catboost")
                # Pass DataFrame to CatBoost, never numpy array!
                preds = cross_val_predict(model, X, y, cv=5, method='predict')
            return mean_absolute_error(y, preds)
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        best_trial = study.best_trial
        best_params = best_trial.params
        best_params.update({'loss_function':'MAE','verbose':False,'random_seed':42})
        # Add cat_features for final estimator
        best_params['cat_features'] = [X.columns.get_loc(c) for c in catboost_cat_cols]
        return best_params

    # --- OOF predictions for stacking and evaluation by decade ---
    print("Tuning Ridge Regression with Optuna...")
    ridge_alpha = tune_ridge(X_train_ridge_scaled, y_train)
    print(f"Best Ridge alpha: {ridge_alpha}")
    print("Tuning LightGBM with Optuna...")
    lgb_params = tune_lgb(X_train_lgb_scaled, y_train)
    print(f"Best LightGBM params: {lgb_params}")
    print("Tuning CatBoost with Optuna...")
    # For CatBoost, pass DataFrame (not numpy array) so categorical columns are preserved
    cat_params = tune_catboost(X_train_cat, y_train)
    print(f"Best CatBoost params: {cat_params}")

    ridge_final = Ridge(alpha=ridge_alpha, random_state=42)
    lgb_final = lgb.LGBMRegressor(**lgb_params)
    # Remove cat_features from cat_params for CatBoostRegressor instantiation
    catboost_cat_indices = cat_params.pop('cat_features', None)
    cat_final = CatBoostRegressor(**cat_params, cat_features=catboost_cat_indices)

    # ---- CatBoost OOF predictions for stacking ----
    print("Generating CatBoost OOF predictions for stacking...")
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cat_oof = np.zeros(len(X_train_cat))
    cat_test_preds = np.zeros((len(X_test_cat), n_splits))
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_cat)):
        X_tr, X_val = X_train_cat.iloc[tr_idx], X_train_cat.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        model = CatBoostRegressor(**cat_params, cat_features=catboost_cat_indices)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="catboost")
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
        cat_oof[val_idx] = model.predict(X_val)
        cat_test_preds[:, fold] = model.predict(X_test_cat)
    cat_test_pred_mean = cat_test_preds.mean(axis=1)
    # Normalize CatBoost OOF predictions before adding to stacking features
    oof_cat = cat_oof
    test_cat_preds = cat_test_pred_mean
    oof_cat_norm = (oof_cat - np.mean(oof_cat)) / np.std(oof_cat)
    X_train_stacking['cat_oof'] = oof_cat_norm
    X_test_stacking['cat_oof'] = (test_cat_preds - np.mean(oof_cat)) / np.std(oof_cat)
    print("Normalized CatBoost OOF predictions added to stacking features.")

    # ---- Stacking: LightGBM base model only, CatBoost OOF as feature ----
    from sklearn.linear_model import LassoCV
    estimators = [
        ('lgb', lgb_final)
    ]

    stacking_model = StackingRegressor(
        estimators=estimators,
        final_estimator=LassoCV(cv=5, random_state=42),
        passthrough=True,
        n_jobs=-1
    )

    # OOF predictions for stacking model
    from sklearn.model_selection import cross_val_predict
    with suppress_stdout_stderr():
        oof_preds = cross_val_predict(stacking_model, X_train_stacking, y_train, cv=5, method='predict', n_jobs=-1)
    # Save OOF predictions with yearID, teamID, actual W, predicted W
    oof_stack_df = pd.DataFrame({
        'yearID': train['yearID'] if 'yearID' in train.columns else np.nan,
        'teamID': train['teamID'] if 'teamID' in train.columns else np.nan,
        'W_actual': y_train,
        'W_pred': oof_preds
    })
    oof_stack_path = os.path.join(submission_dir, 'oof_predictions_stacking.csv')
    oof_stack_df.to_csv(oof_stack_path, index=False)
    print(f"Saved OOF predictions for stacking to: {oof_stack_path}")

    # OOF MAE by decade
    if 'yearID' in oof_stack_df.columns:
        oof_stack_df['decade'] = (oof_stack_df['yearID'] // 10) * 10
        mae_by_decade = oof_stack_df.groupby('decade').apply(
            lambda x: mean_absolute_error(x['W_actual'], x['W_pred'])
        )
        print("Stacking OOF MAE by decade:")
        print(mae_by_decade)
        # Save to CSV
        mae_by_decade_csv = os.path.join(submission_dir, 'oof_mae_by_decade.csv')
        mae_by_decade.to_csv(mae_by_decade_csv, header=['mae'])
        print(f"Saved OOF MAE by decade to: {mae_by_decade_csv}")

    # Fit stacking model on full train
    with suppress_stdout_stderr():
        stacking_model.fit(X_train_stacking, y_train)
        stacking_preds = stacking_model.predict(X_test_stacking)
    stacking_preds_rounded = np.round(stacking_preds).astype(int)
    stacking_preds_rounded = np.clip(stacking_preds_rounded, 40, 120)

    # Save stacking model submission
    submission_stacking = pd.DataFrame({'ID': test_index, 'W': stacking_preds_rounded})
    submission_stacking_path = os.path.join(submission_dir, 'submission_stacking.csv')
    submission_stacking.to_csv(submission_stacking_path, index=False)
    print(f"Saved stacking ensemble submission file to: {submission_stacking_path}")

    print("Ridge, LightGBM tuning, and stacking ensemble pipeline finished successfully. Output files are saved.")

if __name__ == "__main__":
    run_pipeline()