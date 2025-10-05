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
from sklearn.model_selection import KFold, cross_val_predict
import optuna
from sklearn.ensemble import StackingRegressor
from catboost import CatBoostRegressor, Pool
import warnings

def run_pipeline(train_path='assets/train.csv', test_path='assets/test.csv'):
    # Load data
    print(f"Loading training data from: {train_path}")
    train = pd.read_csv(train_path)
    print(f"Loading test data from: {test_path}")
    test = pd.read_csv(test_path)

    # Remove lag features (reset to baseline: no lag features)

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

    # Remove lag features (reset to baseline: no lag features)

    # Add era normalization and sabermetric features only (core features, no lag/interactions)
    for col in ['ERA', 'FP', 'SOA', 'HR', 'BB', 'SO', 'E', 'R', 'RA', 'RD_adj', 'RPG_norm', 'RD_eff']:
        if col not in train.columns:
            train[col] = np.nan
        if col not in test.columns:
            test[col] = np.nan

    # Target variable
    target_col = 'W'

    # Baseline core features (12-15 classical + normalized sabermetrics, no lags/interactions)
    core_features = [
        'R', 'RA', 'ERA', 'SOA', 'HR', 'BB', 'SO', 'E', 'FP', 'RD_adj', 'RPG_norm', 'RD_eff'
    ]
    # Filter features to those present in both train and test
    available_features = [f for f in core_features if f in train.columns and f in test.columns]

    # Prepare training and test data
    X_train = train[available_features].copy()
    y_train = train[target_col].copy()
    X_test = test[available_features].copy()

    # Fill NaNs with 0 for features
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    # Scale all features except those that will be treated as categorical by CatBoost
    # Here, all core features are numeric, so scale all
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

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

    # --- Ridge, LightGBM & CatBoost Stacking (baseline, no lag/interactions, core features only) ---
    def tune_ridge(X, y, n_trials=20):
        def objective(trial):
            alpha = trial.suggest_loguniform('alpha', 1e-3, 10)
            model = Ridge(alpha=alpha, random_state=42)
            preds = cross_val_predict(model, X, y, cv=5, method='predict')
            return mean_absolute_error(y, preds)
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
            mae_key = [k for k in cv_results.keys() if 'l1' in k][0]
            return min(cv_results[mae_key])
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        return study.best_params

    print("Tuning Ridge Regression with Optuna...")
    ridge_alpha = tune_ridge(X_train_scaled, y_train)
    print(f"Best Ridge alpha: {ridge_alpha}")
    print("Tuning LightGBM with Optuna...")
    warnings.filterwarnings('ignore', category=UserWarning)
    lgb_params = tune_lgb(X_train_scaled, y_train)
    warnings.filterwarnings('default', category=UserWarning)
    print(f"Best LightGBM params: {lgb_params}")

    ridge_final = Ridge(alpha=ridge_alpha, random_state=42)
    lgb_final = lgb.LGBMRegressor(**lgb_params)
    cat_features = []  # No categorical features in core features; all numeric

    catboost_final = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.03,
        depth=6,
        loss_function='MAE',
        random_seed=42,
        verbose=False
    )

    estimators = [
        ('ridge', ridge_final),
        ('lgb', lgb_final),
        ('catboost', catboost_final)
    ]

    # For stacking with CatBoost, we need to override fit and predict to handle categorical features properly
    class StackingRegressorWithCatBoost(StackingRegressor):
        def fit(self, X, y, **fit_params):
            # Fit base estimators
            for name, est in self.estimators:
                if name == 'catboost':
                    pool = Pool(X, y, cat_features=cat_features)
                    est.fit(pool)
                else:
                    est.fit(X, y)
            # Fit final estimator on meta-features
            meta_features = self._get_meta_features(X)
            self.final_estimator.fit(meta_features, y)
            return self

        def _get_meta_features(self, X):
            meta_features = []
            for name, est in self.estimators:
                if name == 'catboost':
                    preds = est.predict(Pool(X, cat_features=cat_features))
                else:
                    preds = est.predict(X)
                meta_features.append(preds)
            meta_features = np.array(meta_features).T
            if self.passthrough:
                meta_features = np.hstack((X, meta_features))
            return meta_features

        def predict(self, X):
            meta_features = self._get_meta_features(X)
            return self.final_estimator.predict(meta_features)

    stacking_model = StackingRegressorWithCatBoost(
        estimators=estimators,
        final_estimator=Ridge(alpha=1.0, random_state=42),
        cv=5,
        n_jobs=-1,
        passthrough=True
    )

    stacking_model.fit(X_train_scaled, y_train)
    stacking_preds = stacking_model.predict(X_test_scaled)
    stacking_preds_rounded = np.round(stacking_preds).astype(int)
    stacking_preds_rounded = np.clip(stacking_preds_rounded, 40, 120)

    # Save stacking model submission
    submission_stacking = pd.DataFrame({'ID': test_index, 'W': stacking_preds_rounded})
    submission_stacking_path = os.path.join(submission_dir, 'submission_stacking.csv')
    submission_stacking.to_csv(submission_stacking_path, index=False)
    print(f"Saved stacking ensemble submission file to: {submission_stacking_path}")

    # OOF MAE by decade reporting for stacking
    # We perform cross_val_predict for Ridge and LGB only, then average with CatBoost OOF predictions
    # Instead, for simplicity, do cross_val_predict with stacking_model ignoring cat_features for OOF
    # But cat_features are empty, so we can do cross_val_predict directly

    oof_preds = cross_val_predict(stacking_model, X_train_scaled, y_train, cv=5, method='predict')
    oof_df_stacking = train[['yearID']].copy() if 'yearID' in train.columns else pd.DataFrame(index=train.index)
    oof_df_stacking['W_actual'] = y_train
    oof_df_stacking['W_pred'] = oof_preds
    if 'yearID' in oof_df_stacking.columns:
        oof_df_stacking['decade'] = (oof_df_stacking['yearID'] // 10) * 10
        mae_by_decade = oof_df_stacking.groupby('decade').apply(
            lambda x: mean_absolute_error(x['W_actual'], x['W_pred'])
        )
        print("Stacking OOF MAE by decade:")
        print(mae_by_decade)
        oof_mae_path = os.path.join(submission_dir, 'oof_mae_by_decade.csv')
        mae_by_decade.to_csv(oof_mae_path)
        print(f"Saved OOF MAE by decade to: {oof_mae_path}")

    print("Ridge, LightGBM, CatBoost tuning, and stacking ensemble pipeline finished successfully. Output files are saved.")

if __name__ == "__main__":
    run_pipeline()