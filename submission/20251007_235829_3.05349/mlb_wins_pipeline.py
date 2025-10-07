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

def run_pipeline(train_path='assets/train.csv', test_path='assets/test.csv'):
    # Load data
    print(f"Loading training data from: {train_path}")
    train = pd.read_csv(train_path)
    print(f"Loading test data from: {test_path}")
    test = pd.read_csv(test_path)

    # --- Add rolling lag features: rolling mean of W, R, RA over previous 2 seasons for each teamID ---
    if 'yearID' in train.columns and 'teamID' in train.columns:
        train = train.sort_values(['teamID', 'yearID'])
        # Rolling mean over previous 2 seasons (lags 1 and 2)
        train['W_lag2_mean'] = train.groupby('teamID')['W'].transform(lambda x: x.shift(1).rolling(2, min_periods=1).mean())
        train['R_lag2_mean'] = train.groupby('teamID')['R'].transform(lambda x: x.shift(1).rolling(2, min_periods=1).mean())
        train['RA_lag2_mean'] = train.groupby('teamID')['RA'].transform(lambda x: x.shift(1).rolling(2, min_periods=1).mean())
        train['W_lag2_mean'] = train['W_lag2_mean'].fillna(0)
        train['R_lag2_mean'] = train['R_lag2_mean'].fillna(0)
        train['RA_lag2_mean'] = train['RA_lag2_mean'].fillna(0)
    else:
        train['W_lag2_mean'] = 0
        train['R_lag2_mean'] = 0
        train['RA_lag2_mean'] = 0

    # For test: merge rolling lag features from train, aligned by teamID and yearID
    if 'yearID' in test.columns and 'teamID' in test.columns:
        # Prepare rolling lag features from train, shift yearID by +1 (so that for test year, lag is from previous train years)
        lag2_df = train[['teamID', 'yearID', 'W_lag2_mean', 'R_lag2_mean', 'RA_lag2_mean']].copy()
        lag2_df['yearID'] += 1
        test = test.merge(
            lag2_df,
            on=['teamID', 'yearID'],
            how='left'
        )
        test['W_lag2_mean'] = test['W_lag2_mean'].fillna(0)
        test['R_lag2_mean'] = test['R_lag2_mean'].fillna(0)
        test['RA_lag2_mean'] = test['RA_lag2_mean'].fillna(0)
    else:
        test['W_lag2_mean'] = 0
        test['R_lag2_mean'] = 0
        test['RA_lag2_mean'] = 0

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

    # --- New advanced interaction features ---
    # Ensure all needed columns exist in both train and test
    for col in ['RD_per_game','ERA','FP','RD_adj','Pythag_W','W_lag1','W_lag2_mean','R_lag1','RA_lag1','RPG','HR_per_game','H_per_game','SO_per_game','BB_per_game','SOA_per_game','RA_per_game','mlb_rpg','G']:
        if col not in train.columns:
            train[col] = np.nan
        if col not in test.columns:
            test[col] = np.nan

    train['RD_ERA'] = train['RD_per_game'] * train['ERA']
    test['RD_ERA'] = test['RD_per_game'] * test['ERA']

    train['RD_FP'] = train['RD_per_game'] * train['FP']
    test['RD_FP'] = test['RD_per_game'] * test['FP']

    train['RDadj_FP'] = train['RD_adj'] * train['FP']
    test['RDadj_FP'] = test['RD_adj'] * test['FP']

    train['Pythag_ERA'] = train['Pythag_W'] * train['ERA']
    test['Pythag_ERA'] = test['Pythag_W'] * test['ERA']

    train['Pythag_FP'] = train['Pythag_W'] * train['FP']
    test['Pythag_FP'] = test['Pythag_W'] * test['FP']

    train['Wlag1_RD'] = train['W_lag1'] * train['RD_per_game']
    test['Wlag1_RD'] = test['W_lag1'] * test['RD_per_game']

    train['Wlag2_Pythag'] = train['W_lag2_mean'] * train['Pythag_W']
    test['Wlag2_Pythag'] = test['W_lag2_mean'] * test['Pythag_W']

    train['R_lag1_RA_lag1'] = train['R_lag1'] * train['RA_lag1']
    test['R_lag1_RA_lag1'] = test['R_lag1'] * test['RA_lag1']

    train['RPG_ERA'] = train['RPG'] * train['ERA']
    test['RPG_ERA'] = test['RPG'] * test['ERA']

    train['HR_H'] = train['HR_per_game'] * train['H_per_game']
    test['HR_H'] = test['HR_per_game'] * test['H_per_game']

    train['SO_BB_ratio'] = train['SO_per_game'] / (train['BB_per_game']+1e-5)
    test['SO_BB_ratio'] = test['SO_per_game'] / (test['BB_per_game']+1e-5)

    train['RAPG_SOA'] = train['RA_per_game'] * train['SOA_per_game']
    test['RAPG_SOA'] = test['RA_per_game'] * test['SOA_per_game']

    train['RPG_norm'] = train['RPG'] / train['mlb_rpg']
    test['RPG_norm'] = test['RPG'] / test['mlb_rpg']

    train['RD_eff'] = train['RD_per_game'] / (train['RPG']+1e-5)
    test['RD_eff'] = test['RD_per_game'] / (test['RPG']+1e-5)

    # Target variable
    target_col = 'W'

    # Define default features exactly as specified in DATA_DESCRIPTION.md
    default_features = [
        'yearID', 'teamID', 'lgID', 'G', 'W', 'L', 'R', 'AB', 'H', '2B', '3B', 'HR',
        'BB', 'SO', 'SB', 'CS', 'HBP', 'SF', 'RA', 'ER', 'ERA', 'CG', 'SHO', 'SV',
        'IP', 'HA', 'HRA', 'BBA', 'SOA', 'E', 'DP', 'FP', 'attendance', 'BPF', 'PPF',
        'teamIDBR', 'teamIDlahman45', 'teamIDretro'
    ]

    # Filter features to those present in both train and test
    available_features = [f for f in default_features if f in train.columns and f in test.columns]

    # Prepare training and test data
    X_train = train[available_features].copy()
    y_train = train[target_col].copy()
    X_test = test[available_features].copy()

    # Identify one-hot encoded era and decade columns (if any)
    # They start with 'era_' or 'decade_'
    era_decade_cols = [col for col in X_train.columns if col.startswith('era_') or col.startswith('decade_')]
    # Features to scale are those not in era_decade_cols
    features_to_scale = [col for col in X_train.columns if col not in era_decade_cols]

    # Fill NaNs with 0 for scaling features
    X_train[features_to_scale] = X_train[features_to_scale].fillna(0)
    X_test[features_to_scale] = X_test[features_to_scale].fillna(0)

    # For era_decade_cols fill NaNs with 0 as well (usually one-hot encoded, but just in case)
    if era_decade_cols:
        X_train[era_decade_cols] = X_train[era_decade_cols].fillna(0)
        X_test[era_decade_cols] = X_test[era_decade_cols].fillna(0)

    # Scale features except era and decade one-hot columns
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[features_to_scale] = scaler.fit_transform(X_train[features_to_scale])
    X_test_scaled[features_to_scale] = scaler.transform(X_test[features_to_scale])

    # Train Linear Regression model
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)

    # Predict on training and test data
    train_preds = lr.predict(X_train_scaled)
    test_preds = lr.predict(X_test_scaled)

    # Evaluate on training data
    mae = mean_absolute_error(y_train, train_preds)
    rmse = math.sqrt(mean_squared_error(y_train, train_preds))
    r2 = r2_score(y_train, train_preds)
    print(f"Linear Regression training MAE: {mae:.4f}")
    print(f"Linear Regression training RMSE: {rmse:.4f}")
    print(f"Linear Regression training R^2: {r2:.4f}")

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_dir = os.path.join('submission', timestamp)
    os.makedirs(submission_dir, exist_ok=True)

    # --- Copy current pipeline file to output folder for reproducibility ---
    import shutil
    current_script_path = os.path.realpath(__file__)
    script_copy_path = os.path.join(submission_dir, 'mlb_wins_pipeline.py')
    shutil.copy2(current_script_path, script_copy_path)
    print(f"Copied pipeline script to: {script_copy_path}")

    # Feature importance from Linear Regression coefficients
    coef_df = pd.DataFrame({
        'feature': available_features,
        'coefficient': lr.coef_
    })
    coef_df['abs_coefficient'] = coef_df['coefficient'].abs()
    coef_df = coef_df.sort_values(by='abs_coefficient', ascending=False)
    print("Top 10 features by absolute coefficient (Linear Regression):")
    print(coef_df.head(10)[['feature', 'coefficient']])

    # Save all feature importances (coefficients) to CSV
    feature_importances_path = os.path.join(submission_dir, 'feature_importances.csv')
    coef_df[['feature', 'coefficient']].to_csv(feature_importances_path, index=False)
    print(f"Saved all feature importances to: {feature_importances_path}")

    # Save top 10 coefficients to CSV
    top_10_path = os.path.join(submission_dir, 'top_10_coefficients.csv')
    coef_df.head(10)[['feature', 'coefficient']].to_csv(top_10_path, index=False)
    print(f"Saved top 10 coefficients to: {top_10_path}")

    # Clip predictions to reasonable bounds
    test_preds_rounded = np.round(test_preds).astype(int)
    test_preds_rounded = np.clip(test_preds_rounded, 40, 120)

    # Save training predictions (OOF not applicable here, so save train preds)
    oof_df = train[['yearID']].copy() if 'yearID' in train.columns else pd.DataFrame(index=train.index)
    oof_df['W_actual'] = y_train
    oof_df['W_pred'] = train_preds
    oof_df = oof_df.dropna(subset=['W_pred'])
    oof_path = os.path.join(submission_dir, 'train_predictions_linear_regression.csv')
    oof_df.to_csv(oof_path, index=False)
    print(f"Saved training predictions to: {oof_path}")

    # Evaluation by decade if yearID exists
    if 'yearID' in oof_df.columns:
        oof_df['decade'] = (oof_df['yearID'] // 10) * 10
        mae_by_decade = oof_df.groupby('decade').apply(
            lambda x: mean_absolute_error(x['W_actual'], x['W_pred'])
        )
        print("Linear Regression training MAE by decade:")
        print(mae_by_decade)
        # Save MAE by decade to CSV in output folder
        mae_by_decade_df = mae_by_decade.reset_index()
        mae_by_decade_df.columns = ['decade', 'MAE']
        mae_by_decade_path = os.path.join(submission_dir, 'oof_mae_by_decade.csv')
        mae_by_decade_df.to_csv(mae_by_decade_path, index=False)
        print(f"Saved MAE by decade to: {mae_by_decade_path}")

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

    # --- Additional steps for Ridge and LightGBM feature importance and retraining ---

    # Train Ridge regression on all features
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_train_scaled, y_train)
    ridge_coefs = np.abs(ridge.coef_)

    # Train LightGBM regressor on all features
    lgb_train = lgb.Dataset(X_train_scaled, label=y_train)
    params = {
        'objective': 'regression',
        'metric': 'l2',
        'verbosity': -1,  # Suppress warnings
        'seed': 42,
        'force_row_wise': True  # For warning suppression and compatibility
    }
    # Suppress LightGBM terminal warnings
    try:
        with lgb.basic.silent():
            lgb_model = lgb.train(params, lgb_train, num_boost_round=100)
    except AttributeError:
        # For older LightGBM versions, fallback to direct call
        lgb_model = lgb.train(params, lgb_train, num_boost_round=100)
    lgb_importance = lgb_model.feature_importance(importance_type='gain')
    lgb_features = X_train_scaled.columns

    # Create DataFrame for feature importances
    ridge_imp_df = pd.DataFrame({'feature': X_train_scaled.columns, 'ridge_importance': ridge_coefs})
    lgb_imp_df = pd.DataFrame({'feature': lgb_features, 'lgb_importance': lgb_importance})

    # Merge importances
    combined_imp_df = pd.merge(ridge_imp_df, lgb_imp_df, on='feature', how='inner')

    # Normalize importances
    combined_imp_df['ridge_norm'] = combined_imp_df['ridge_importance'] / combined_imp_df['ridge_importance'].max()
    combined_imp_df['lgb_norm'] = combined_imp_df['lgb_importance'] / combined_imp_df['lgb_importance'].max()

    # Combined score as sum of normalized importances
    combined_imp_df['combined_score'] = combined_imp_df['ridge_norm'] + combined_imp_df['lgb_norm']

    # Sort by combined score descending
    combined_imp_df = combined_imp_df.sort_values(by='combined_score', ascending=False)

    # Select top 12-15 features (choose 15)
    top_features_count = 15
    top_features = combined_imp_df.head(top_features_count)['feature'].tolist()

    # Save combined top features to CSV
    top_features_path = os.path.join(submission_dir, 'top_features_combined.csv')
    combined_imp_df.head(top_features_count)[['feature', 'combined_score']].to_csv(top_features_path, index=False)
    print(f"Saved combined top features to: {top_features_path}")

    # Prepare data with top features only
    X_train_top = X_train_scaled[top_features]
    X_test_top = X_test_scaled[top_features]


    # --- Scale lag and interaction features before stacking ---
    # Identify lag and interaction features
    lag_interaction_cols = [col for col in X_train_top.columns if 'lag' in col or 'RD' in col or 'Pythag' in col
                            or col in [
                                'RD_ERA','RD_FP','RDadj_FP','Pythag_ERA','Pythag_FP','Wlag1_RD','Wlag2_Pythag',
                                'R_lag1_RA_lag1','RPG_ERA','HR_H','SO_BB_ratio','RAPG_SOA','RPG_norm','RD_eff'
                            ]]
    # Remove duplicates
    lag_interaction_cols = list(dict.fromkeys(lag_interaction_cols))
    if lag_interaction_cols:
        scaler_lag = StandardScaler()
        X_train_top[lag_interaction_cols] = scaler_lag.fit_transform(X_train_top[lag_interaction_cols])
        X_test_top[lag_interaction_cols] = scaler_lag.transform(X_test_top[lag_interaction_cols])
    else:
        print("No lag or interaction features found to scale.")

    # --- Define Optuna tuning functions ---
    def tune_ridge(X, y, n_trials=20):
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

    def tune_catboost(X, y, n_trials=20):
        def objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 100, 500),
                'depth': trial.suggest_int('depth', 3, 10),
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
                'loss_function': 'MAE',
                'verbose': False,
                'random_seed': 42
            }
            model = CatBoostRegressor(**params)
            preds = cross_val_predict(model, X, y, cv=5, method='predict')
            return mean_absolute_error(y, preds)
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        best_trial = study.best_trial
        best_params = best_trial.params
        best_params.update({'loss_function':'MAE','verbose':False,'random_seed':42})
        return best_params

    # --- Stacking with Ridge + LightGBM + CatBoost ---
    print("Tuning Ridge Regression with Optuna...")
    ridge_alpha = tune_ridge(X_train_top, y_train)
    print(f"Best Ridge alpha: {ridge_alpha}")
    print("Tuning LightGBM with Optuna...")
    lgb_params = tune_lgb(X_train_top, y_train)
    print(f"Best LightGBM params: {lgb_params}")
    print("Tuning CatBoost with Optuna...")
    cat_params = tune_catboost(X_train_top, y_train)
    print(f"Best CatBoost params: {cat_params}")

    ridge_final = Ridge(alpha=ridge_alpha, random_state=42)
    lgb_final = lgb.LGBMRegressor(**lgb_params)
    cat_final = CatBoostRegressor(**cat_params)

    estimators = [
        ('ridge', ridge_final),
        ('lgb', lgb_final),
        ('cat', cat_final)
    ]

    stacking_model = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=1.0, random_state=42),
        cv=5,
        n_jobs=-1,
        passthrough=True
    )

    stacking_model.fit(X_train_top, y_train)
    stacking_preds = stacking_model.predict(X_test_top)
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