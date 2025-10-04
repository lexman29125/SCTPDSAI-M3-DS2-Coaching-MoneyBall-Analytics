#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import math
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_absolute_error

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

    # Prepare index for test submission (assuming 'ID' column exists)
    if 'ID' in test_fe.columns:
        test_index = test_fe['ID']
        print("Test data contains 'ID' column for submission index.")
    else:
        test_index = test_fe.index
        print("Test data does not contain 'ID' column; using index for submission.")

    # Define LightGBM parameters
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'seed': 42
    }

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

    n_boost_round = 1000
    early_stopping_rounds = 50

    best_iterations = []
    fold = 0
    # Prepare OOF predictions array
    oof_preds = np.full(len(train_fe), np.nan)
    feature_importances = pd.DataFrame()
    feature_importances['feature'] = numeric_features

    for train_years, val_year in splits:
        train_mask = train_fe['yearID'].isin(train_years)
        val_mask = (train_fe['yearID'] == val_year)
        X_train = train_fe.loc[train_mask, numeric_features]
        y_train = train_fe.loc[train_mask, target_col]
        X_val = train_fe.loc[val_mask, numeric_features]
        y_val = train_fe.loc[val_mask, target_col]
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val)
        model = lgb.train(
            lgb_params,
            dtrain,
            num_boost_round=n_boost_round,
            valid_sets=[dtrain, dval],
            valid_names=['train', 'valid'],
            callbacks=[lgb.early_stopping(early_stopping_rounds), lgb.log_evaluation(0)]
        )
        best_iterations.append(model.best_iteration)
        # Save OOF predictions
        oof_preds[val_mask] = model.predict(X_val, num_iteration=model.best_iteration)
        # Save feature importances
        fold_importance = model.feature_importance(importance_type='gain')
        feature_importances[f'fold_{fold}'] = fold_importance
        fold += 1

    # Average feature importances across folds
    feature_importances['importance_mean'] = feature_importances.loc[:, feature_importances.columns.str.startswith('fold_')].mean(axis=1)
    feature_importances = feature_importances.sort_values(by='importance_mean', ascending=False)

    # Save feature importances
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_dir = os.path.join('submission', timestamp)
    os.makedirs(submission_dir, exist_ok=True)
    fi_path = os.path.join(submission_dir, 'feature_importances.csv')
    feature_importances[['feature', 'importance_mean']].to_csv(fi_path, index=False)
    print(f"Saved feature importances to: {fi_path}")

    # Select top 12-15 features based on importance_mean for final model
    top_features = feature_importances['feature'].head(15).tolist()
    print(f"Top 15 features selected for final model training: {top_features}")

    # Placeholder: LightGBM hyperparameter tuning with Optuna (MAE objective)
    # ---------------------------------------------------------------
    # import optuna
    # def objective(trial):
    #     param = {
    #         'objective': 'regression',
    #         'metric': 'mae',
    #         'verbosity': -1,
    #         'boosting_type': 'gbdt',
    #         'seed': 42,
    #         'num_leaves': trial.suggest_int('num_leaves', 20, 100),
    #         'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
    #         'feature_fraction': trial.suggest_uniform('feature_fraction', 0.6, 1.0),
    #         'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.6, 1.0),
    #         'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
    #         'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 100),
    #     }
    #     cv_results = lgb.cv(param, lgb.Dataset(train_fe[top_features], label=train_fe[target_col]),
    #                        nfold=5, early_stopping_rounds=50, metrics='mae', seed=42)
    #     return min(cv_results['l1-mean'])
    # study = optuna.create_study(direction='minimize')
    # study.optimize(objective, n_trials=50)
    # best_params = study.best_params
    # lgb_params.update(best_params)
    # ---------------------------------------------------------------

    # Fit final model on full training data using top features
    print("\nTraining final model on full train set...")
    # Use mean best_iteration from previous folds if available, else fallback
    if fold > 0:
        avg_best_iter = int(np.mean(best_iterations))
    else:
        avg_best_iter = n_boost_round

    dtrain_full = lgb.Dataset(train_fe[top_features], label=train_fe[target_col])
    final_model = lgb.train(
        lgb_params,
        dtrain_full,
        num_boost_round=avg_best_iter,
        callbacks=[lgb.log_evaluation(0)]
    )

    # Placeholder: Stacking with Ridge and CatBoost
    # ---------------------------------------------------------------
    # from sklearn.linear_model import Ridge
    # from catboost import CatBoostRegressor
    # from sklearn.model_selection import KFold
    # stacking_features = np.zeros((len(train_fe), 2))  # Ridge and CatBoost predictions
    # kf = KFold(n_splits=5, shuffle=True, random_state=42)
    # for train_idx, val_idx in kf.split(train_fe):
    #     X_train_stack = train_fe.iloc[train_idx][top_features]
    #     y_train_stack = train_fe.iloc[train_idx][target_col]
    #     X_val_stack = train_fe.iloc[val_idx][top_features]
    #
    #     ridge = Ridge()
    #     ridge.fit(X_train_stack, y_train_stack)
    #     stacking_features[val_idx, 0] = ridge.predict(X_val_stack)
    #
    #     cat = CatBoostRegressor(verbose=0, random_seed=42)
    #     cat.fit(X_train_stack, y_train_stack)
    #     stacking_features[val_idx, 1] = cat.predict(X_val_stack)
    #
    # # Use stacking features as additional inputs or for blending predictions
    # ---------------------------------------------------------------

    # Predict test set and save submission with error handling
    try:
        print("Predicting test set...")
        test_preds = final_model.predict(test_fe[top_features], num_iteration=final_model.best_iteration)
        print(f"Number of predictions made: {len(test_preds)}")
        # Round & clip to realistic bounds - adjust as dataset dictates
        test_preds_rounded = np.round(test_preds).astype(int)
        test_preds_rounded = np.clip(test_preds_rounded, 40, 120)

        # Create submission
        submission = pd.DataFrame({'ID': test_index, 'W': test_preds_rounded})
        submission_path = os.path.join(submission_dir, 'submission.csv')
        submission.to_csv(submission_path, index=False)
        print(f"Saved submission file to: {submission_path}")
    except Exception as e:
        print(f"Error during prediction or saving submission: {e}")

    # Save OOF predictions
    oof_df = train_fe[['yearID']].copy() if 'yearID' in train_fe.columns else pd.DataFrame(index=train_fe.index)
    oof_df['W_actual'] = train_fe[target_col]
    oof_df['W_pred'] = oof_preds
    oof_df = oof_df.dropna(subset=['W_pred'])
    oof_path = os.path.join(submission_dir, 'oof_predictions.csv')
    oof_df.to_csv(oof_path, index=False)
    print(f"Saved OOF predictions to: {oof_path}")

    # OOF evaluation: MAE overall and by era/decade
    mae_overall = mean_absolute_error(oof_df['W_actual'], oof_df['W_pred'])
    print(f"Overall OOF MAE: {mae_overall:.4f}")

    if 'yearID' in oof_df.columns:
        # Define decade as yearID floored to nearest 10
        oof_df['decade'] = (oof_df['yearID'] // 10) * 10
        mae_by_decade = oof_df.groupby('decade').apply(
            lambda x: mean_absolute_error(x['W_actual'], x['W_pred'])
        )
        print("OOF MAE by decade:")
        print(mae_by_decade)

    # Plot OOF predicted vs actual wins
    plt.figure(figsize=(8, 8))
    plt.scatter(oof_df['W_actual'], oof_df['W_pred'], alpha=0.5, edgecolors='k')
    plt.plot([oof_df['W_actual'].min(), oof_df['W_actual'].max()],
             [oof_df['W_actual'].min(), oof_df['W_actual'].max()],
             'r--', lw=2)
    plt.xlabel('Actual Wins')
    plt.ylabel('OOF Predicted Wins')
    plt.title('OOF Predicted vs Actual Wins')
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(submission_dir, 'oof_pred_vs_actual.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved OOF predicted vs actual plot to: {plot_path}")

    print("Pipeline finished successfully. Output files are saved.")

if __name__ == "__main__":
    run_pipeline()