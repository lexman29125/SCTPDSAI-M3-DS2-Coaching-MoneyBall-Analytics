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
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold, cross_val_predict
import optuna
from sklearn.ensemble import StackingRegressor
import random

def set_global_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)

# Helper: Create lag features (rolling mean and lag1)
def create_lag_features(df, group_cols=['teamID'], time_col='yearID', lags=[1,2], fillna_value=0):
    df = df.sort_values(group_cols + [time_col])
    # Rolling mean of W, R, RA over previous 2 seasons
    for col in ['W', 'R', 'RA']:
        df[f'{col}_lag2_mean'] = df.groupby(group_cols)[col].transform(lambda x: x.shift(1).rolling(2, min_periods=1).mean())
        df[f'{col}_lag2_mean'] = df[f'{col}_lag2_mean'].fillna(fillna_value)
        df[f'{col}_lag1'] = df.groupby(group_cols)[col].shift(1)
        df[f'{col}_lag1_missing'] = df[f'{col}_lag1'].isna().astype(int)
        df[f'{col}_lag1'] = df[f'{col}_lag1'].fillna(fillna_value)
    return df

# Helper: Merge lag features for test set from train
def merge_lag_features_to_test(test, train, lags=['W_lag2_mean','R_lag2_mean','RA_lag2_mean','W_lag1','R_lag1','RA_lag1']):
    lag2_df = train[['teamID', 'yearID'] + [c for c in lags if c in train.columns]].copy()
    lag2_df['yearID'] += 1
    test = test.merge(
        lag2_df,
        on=['teamID', 'yearID'],
        how='left'
    )
    for c in lags:
        if c in test.columns:
            test[c] = test[c].fillna(0)
    # Add missing indicators for lag1
    for col in ['W', 'R', 'RA']:
        lag1_col = f'{col}_lag1'
        test[f'{col}_lag1_missing'] = test[lag1_col].isna().astype(int)
        test[lag1_col] = test[lag1_col].fillna(0)
    return test

# Helper: Create interaction features
def create_interaction_features(df):
    if 'Pythag_W' in df.columns and 'ERA' in df.columns:
        df['Pythag_W_ERA'] = df['Pythag_W'] * df['ERA']
    if 'RD_per_game' in df.columns and 'FP' in df.columns:
        df['RD_per_game_FP'] = df['RD_per_game'] * df['FP']
    return df

# Helper: Feature scaling (StandardScaler) and NaN handling
def scale_numeric_features(X_train, X_test, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = []
    # Only consider exclude_cols that are actually present in both X_train and X_test
    exclude_cols_filtered = [col for col in exclude_cols if col in X_train.columns and col in X_test.columns]
    features_to_scale = [col for col in X_train.columns if col not in exclude_cols_filtered]
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    # Fill NaNs with 0
    X_train_scaled[features_to_scale] = X_train_scaled[features_to_scale].fillna(0)
    X_test_scaled[features_to_scale] = X_test_scaled[features_to_scale].fillna(0)
    scaler = StandardScaler()
    X_train_scaled[features_to_scale] = scaler.fit_transform(X_train_scaled[features_to_scale])
    X_test_scaled[features_to_scale] = scaler.transform(X_test_scaled[features_to_scale])
    # Fill NaNs in excluded cols too (usually one-hot)
    if exclude_cols_filtered:
        X_train_scaled[exclude_cols_filtered] = X_train_scaled[exclude_cols_filtered].fillna(0)
        X_test_scaled[exclude_cols_filtered] = X_test_scaled[exclude_cols_filtered].fillna(0)
    return X_train_scaled, X_test_scaled

# Helper: Remove zero-variance columns
def remove_zero_variance_features(X_train, X_test):
    stds = X_train.std(axis=0)
    keep_cols = stds[stds > 0].index.tolist()
    return X_train[keep_cols], X_test[keep_cols]

def run_pipeline(train_path='assets/train.csv', test_path='assets/test.csv'):
    set_global_seed(42)
    # Load data
    print(f"Loading training data from: {train_path}")
    train = pd.read_csv(train_path)
    print(f"Loading test data from: {test_path}")
    test = pd.read_csv(test_path)

    # --- Add lag features ---
    if 'yearID' in train.columns and 'teamID' in train.columns:
        train = create_lag_features(train)
    else:
        for col in ['W_lag2_mean', 'R_lag2_mean', 'RA_lag2_mean', 'W_lag1', 'R_lag1', 'RA_lag1']:
            train[col] = 0
        for col in ['W_lag1_missing', 'R_lag1_missing', 'RA_lag1_missing']:
            train[col] = 1
    if 'yearID' in test.columns and 'teamID' in test.columns:
        test = merge_lag_features_to_test(test, train)
    else:
        for col in ['W_lag2_mean', 'R_lag2_mean', 'RA_lag2_mean', 'W_lag1', 'R_lag1', 'RA_lag1']:
            test[col] = 0
        for col in ['W_lag1_missing', 'R_lag1_missing', 'RA_lag1_missing']:
            test[col] = 1

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

    # Add interaction terms (only if meaningful)
    for col in ['ERA', 'FP']:
        if col not in train.columns:
            train[col] = np.nan
        if col not in test.columns:
            test[col] = np.nan
    train = create_interaction_features(train)
    test = create_interaction_features(test)

    # Target variable
    target_col = 'W'

    # --- Feature selection: start with expert base features ---
    expert_base_features = [
        'R', 'RA', 'ERA', 'OBP', 'OPS', 'RD', 'RD_per_game', 'RD_adj', 'Pythag_W', 'SOA', 'BBA', 'E', 'FP'
    ]
    # Add only lag/interaction features with meaningful signal (based on SHAP/importance, here we pick a few)
    lag_interaction_candidates = [
        'W_lag2_mean','R_lag2_mean','RA_lag2_mean','W_lag1','R_lag1','RA_lag1',
        'W_lag1_missing','R_lag1_missing','RA_lag1_missing',
        'Pythag_W_ERA','RD_per_game_FP'
    ]
    # Only include lag/interaction features if present in data
    lag_interaction_features = [f for f in lag_interaction_candidates if f in train.columns and f in test.columns]
    # Use only those with strong signal (for brevity, we include all; in practice, filter by importance)

    # Era/decade one-hot columns
    era_decade_cols = [col for col in train.columns if col.startswith('era_') or col.startswith('decade_')]
    # Compose final feature list
    all_features = expert_base_features + lag_interaction_features + era_decade_cols
    # Remove duplicates and keep only those present in both train and test
    all_features = [f for f in dict.fromkeys(all_features) if f in train.columns and f in test.columns]
    # Prepare X/y
    X_train = train[all_features].copy()
    y_train = train[target_col].copy()
    X_test = test[all_features].copy()
    # Scale numeric features (era/decade one-hot columns excluded from scaling)
    X_train_scaled, X_test_scaled = scale_numeric_features(X_train, X_test, exclude_cols=era_decade_cols)
    # Remove zero-variance features
    X_train_scaled, X_test_scaled = remove_zero_variance_features(X_train_scaled, X_test_scaled)

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
    # Copy this pipeline file to submission_dir
    import shutil
    pipeline_src = os.path.abspath(__file__)
    pipeline_dst = os.path.join(submission_dir, os.path.basename(pipeline_src))
    shutil.copy2(pipeline_src, pipeline_dst)
    print(f"Copied pipeline file to: {pipeline_dst}")

    # Feature importance from Linear Regression coefficients
    coef_df = pd.DataFrame({
        'feature': X_train_scaled.columns,
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
        # Save OOF MAE by decade to CSV
        oof_mae_by_decade_path = os.path.join(submission_dir, 'oof_mae_by_decade.csv')
        mae_by_decade_df = mae_by_decade.reset_index()
        mae_by_decade_df.columns = ['decade', 'MAE']
        mae_by_decade_df.to_csv(oof_mae_by_decade_path, index=False)
        print(f"Saved OOF MAE by decade to: {oof_mae_by_decade_path}")

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

    # --- Feature importance for Ridge and LightGBM ---
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_train_scaled, y_train)
    ridge_coefs = np.abs(ridge.coef_)
    lgb_train = lgb.Dataset(X_train_scaled, label=y_train)
    params = {
        'objective': 'regression',
        'metric': 'l2',
        'verbosity': -1,
        'seed': 42,
        'force_row_wise': True,
        'min_data_in_leaf': 10
    }
    # Remove zero-variance columns for LGBM
    X_train_lgb, X_test_lgb = remove_zero_variance_features(X_train_scaled, X_test_scaled)
    try:
        with lgb.basic.silent():
            lgb_model = lgb.train(params, lgb.Dataset(X_train_lgb, label=y_train), num_boost_round=100)
    except AttributeError:
        lgb_model = lgb.train(params, lgb.Dataset(X_train_lgb, label=y_train), num_boost_round=100)
    lgb_importance = lgb_model.feature_importance(importance_type='gain')
    lgb_features = X_train_lgb.columns
    ridge_imp_df = pd.DataFrame({'feature': X_train_scaled.columns, 'ridge_importance': ridge_coefs})
    lgb_imp_df = pd.DataFrame({'feature': lgb_features, 'lgb_importance': lgb_importance})
    combined_imp_df = pd.merge(ridge_imp_df, lgb_imp_df, on='feature', how='inner')
    combined_imp_df['ridge_norm'] = combined_imp_df['ridge_importance'] / combined_imp_df['ridge_importance'].max()
    combined_imp_df['lgb_norm'] = combined_imp_df['lgb_importance'] / (combined_imp_df['lgb_importance'].max() if combined_imp_df['lgb_importance'].max() > 0 else 1)
    combined_imp_df['combined_score'] = combined_imp_df['ridge_norm'] + combined_imp_df['lgb_norm']
    combined_imp_df = combined_imp_df.sort_values(by='combined_score', ascending=False)
    top_features_count = 15
    top_features = combined_imp_df.head(top_features_count)['feature'].tolist()
    # Save combined top features to CSV
    top_features_path = os.path.join(submission_dir, 'top_features_combined.csv')
    combined_imp_df.head(top_features_count)[['feature', 'combined_score']].to_csv(top_features_path, index=False)
    print(f"Saved combined top features to: {top_features_path}")
    # Prepare data with top features only
    X_train_top = X_train_scaled[top_features]
    X_test_top = X_test_scaled[top_features]

    # --- Optuna tuning helpers ---
    def tune_ridge(X, y, n_trials=20):
        def objective(trial):
            alpha = trial.suggest_float('alpha', 1e-3, 10, log=True)
            model = Ridge(alpha=alpha, random_state=42)
            scores = cross_val_predict(model, X, y, cv=5, method='predict')
            return mean_absolute_error(y, scores)
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
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
            mae_key = [k for k in cv_results.keys() if 'l1' in k][0]
            return min(cv_results[mae_key])
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials)
        return study.best_params
    def tune_catboost(X, y, n_trials=20):
        def objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 100, 500),
                'depth': trial.suggest_int('depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'loss_function': 'MAE',
                'verbose': False,
                'random_seed': 42
            }
            model = CatBoostRegressor(**params)
            preds = cross_val_predict(model, X, y, cv=5, method='predict')
            return mean_absolute_error(y, preds)
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials)
        best_trial = study.best_trial
        best_params = best_trial.params
        best_params.update({'loss_function':'MAE','verbose':False,'random_seed':42})
        return best_params

    # --- Stacking Ensemble: Ridge + LGBM + CatBoost, Ridge as final estimator ---
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
    lgb_final = lgb.LGBMRegressor(**lgb_params, random_state=42)
    cat_final = CatBoostRegressor(**cat_params)
    estimators = [
        ('ridge', ridge_final),
        ('lgb', lgb_final),
        ('cat', cat_final)
    ]

    # Custom stacking: Normalize OOF predictions before final estimator
    def get_oof_predictions(model, X, y, cv):
        oof = np.zeros(len(y))
        for train_idx, test_idx in cv.split(X, y):
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            oof[test_idx] = model.predict(X.iloc[test_idx])
        return oof
    # 5-fold CV for OOF preds
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    ridge_oof = get_oof_predictions(ridge_final, X_train_top, y_train, kf)
    lgb_oof = get_oof_predictions(lgb_final, X_train_top, y_train, kf)
    cat_oof = get_oof_predictions(cat_final, X_train_top, y_train, kf)
    # Normalize each OOF prediction
    def normalize(x):
        return (x - np.mean(x)) / (np.std(x) + 1e-8)
    ridge_oof_norm = normalize(ridge_oof)
    lgb_oof_norm = normalize(lgb_oof)
    cat_oof_norm = normalize(cat_oof)
    # Stack as new features
    X_stack = pd.DataFrame({
        'ridge': ridge_oof_norm,
        'lgb': lgb_oof_norm,
        'cat': cat_oof_norm
    })
    # Final estimator: Ridge
    final_ridge = Ridge(alpha=1.0, random_state=42)
    final_ridge.fit(X_stack, y_train)
    # Train base models on full data for test prediction
    ridge_final.fit(X_train_top, y_train)
    lgb_final.fit(X_train_top, y_train)
    cat_final.fit(X_train_top, y_train)
    ridge_test = ridge_final.predict(X_test_top)
    lgb_test = lgb_final.predict(X_test_top)
    cat_test = cat_final.predict(X_test_top)
    ridge_test_norm = normalize(ridge_test)
    lgb_test_norm = normalize(lgb_test)
    cat_test_norm = normalize(cat_test)
    X_stack_test = pd.DataFrame({
        'ridge': ridge_test_norm,
        'lgb': lgb_test_norm,
        'cat': cat_test_norm
    })
    stacking_preds = final_ridge.predict(X_stack_test)
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