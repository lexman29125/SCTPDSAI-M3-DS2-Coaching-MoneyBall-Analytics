#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
import lightgbm as lgb

# Check for optuna availability
try:
    import optuna
except ImportError:
    print("The 'optuna' module is required to run this script.")
    print("Please install it using: pip install optuna")
    import sys
    sys.exit(1)

def run_pipeline(train_path='assets/train.csv', test_path='assets/test.csv', top_feature_count=30):
    # -------------------
    # Load Data
    # -------------------
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print(f"Training rows: {len(train_df)}, Test rows: {len(test_df)}")

    # -------------------
    # Feature Engineering
    # -------------------
    for df in [train_df, test_df]:
        df['run_diff'] = df['R'] - df['RA']
        df['R_per_game'] = df['R'] / df['G']
        df['RA_per_game'] = df['RA'] / df['G']
        df['RD_per_game'] = df['run_diff'] / df['G']
        df['Pythag_W'] = df['R']**2 / (df['R']**2 + df['RA']**2) * df['G']
        # Interaction features
        for col in ['ERA','FP']:
            if col not in df.columns:
                df[col] = 0
        df['Pythag_W_ERA'] = df['Pythag_W'] * df['ERA']
        df['RD_per_game_FP'] = df['RD_per_game'] * df['FP']

    # Lag features
    train_df = train_df.sort_values(['teamID','yearID'])
    for lag_col in ['W','R','RA']:
        train_df[f'{lag_col}_lag1'] = train_df.groupby('teamID')[lag_col].shift(1).fillna(0)
        train_df[f'{lag_col}_lag1_missing'] = train_df.groupby('teamID')[lag_col].shift(1).isna().astype(int)

    # Merge lag features to test
    if 'teamID' in train_df.columns and 'teamID' in test_df.columns:
        last_season = train_df[['teamID','yearID','W','R','RA']].copy()
        last_season['yearID'] += 1
        last_season.rename(columns={'W':'W_lag1','R':'R_lag1','RA':'RA_lag1'}, inplace=True)
        test_df = test_df.merge(last_season[['teamID','yearID','W_lag1','R_lag1','RA_lag1']], on=['teamID','yearID'], how='left')
        for col in ['W_lag1','R_lag1','RA_lag1']:
            test_df[col] = test_df[col].fillna(0)
            test_df[f'{col}_missing'] = test_df[col].isna().astype(int)
    else:
        print("Warning: 'teamID' not found in dataset, skipping lag feature merge for test set.")

    # Ensure per-game columns exist
    per_game_cols = ['SO_per_game', 'BB_per_game', 'H_per_game', 'HR_per_game', 'SOA_per_game']
    for col in per_game_cols:
        if col not in train_df.columns:
            train_df[col] = 0
        if col not in test_df.columns:
            test_df[col] = 0

    # -------------------
    # Feature List
    # -------------------
    features = ['R','RA','run_diff','R_per_game','RA_per_game','RD_per_game','Pythag_W',
                'Pythag_W_ERA','RD_per_game_FP','W_lag1','R_lag1','RA_lag1',
                'W_lag1_missing','R_lag1_missing','RA_lag1_missing','SO_per_game','BB_per_game',
                'H_per_game','HR_per_game','SOA_per_game','FP','ERA','CG','SV','IPouts','AB']

    # Ensure all features exist in train_df and test_df, else create with 0
    for feature in features:
        if feature not in train_df.columns:
            train_df[feature] = 0
        if feature not in test_df.columns:
            test_df[feature] = 0

    X_train = train_df[features].fillna(0)
    y_train = train_df['W']
    X_test = test_df[features].fillna(0)

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=features)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=features)

    # -------------------
    # Ridge Hyperparameter Tuning
    # -------------------
    def ridge_objective(trial):
        alpha = trial.suggest_loguniform('alpha', 0.01, 1000)
        oof_preds = np.zeros(len(X_train_scaled))
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for tr_idx, val_idx in kf.split(X_train_scaled):
            model = Ridge(alpha=alpha, random_state=42)
            model.fit(X_train_scaled.iloc[tr_idx], y_train.iloc[tr_idx])
            oof_preds[val_idx] = model.predict(X_train_scaled.iloc[val_idx])
        return mean_absolute_error(y_train, oof_preds)

    study = optuna.create_study(direction='minimize')
    study.optimize(ridge_objective, n_trials=25)
    best_alpha = study.best_params['alpha']
    print(f"Tuned Ridge alpha: {best_alpha}")
    ridge_model = Ridge(alpha=best_alpha, random_state=42)
    ridge_model.fit(X_train_scaled, y_train)
    ridge_importances = np.abs(ridge_model.coef_)

    # -------------------
    # LightGBM Hyperparameter Tuning
    # -------------------
    def lgb_objective(trial):
        params = {
            'objective':'regression',
            'metric':'mae',
            'learning_rate': trial.suggest_float('learning_rate',0.01,0.3),
            'num_leaves': trial.suggest_int('num_leaves',10,256),
            'max_depth': trial.suggest_int('max_depth',3,15),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf',10,100),
            'verbose':-1,
            'seed':42
        }
        oof_preds = np.zeros(len(X_train_scaled))
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for tr_idx, val_idx in kf.split(X_train_scaled):
            lgb_train = lgb.Dataset(X_train_scaled.iloc[tr_idx], y_train.iloc[tr_idx])
            lgb_val = lgb.Dataset(X_train_scaled.iloc[val_idx], y_train.iloc[val_idx], reference=lgb_train)
            model = lgb.train(params, lgb_train, valid_sets=[lgb_val])
            oof_preds[val_idx] = model.predict(X_train_scaled.iloc[val_idx])
        return mean_absolute_error(y_train, oof_preds)

    study_lgb = optuna.create_study(direction='minimize')
    study_lgb.optimize(lgb_objective, n_trials=25)
    print(f"Best LGB params: {study_lgb.best_params}")
    lgb_params = study_lgb.best_params
    lgb_params.update({'objective':'regression','metric':'mae','verbose':-1,'seed':42})
    lgb_model = lgb.train(lgb_params, lgb.Dataset(X_train_scaled, label=y_train), num_boost_round=500, valid_sets=[lgb.Dataset(X_train_scaled, label=y_train)])
    lgb_importances = lgb_model.feature_importance(importance_type='gain')

    # -------------------
    # Feature Selection
    # -------------------
    importance_df = pd.DataFrame({
        'feature': features,
        'ridge_importance': ridge_importances,
        'lgb_importance': lgb_importances
    })
    importance_df['ridge_norm'] = importance_df['ridge_importance']/importance_df['ridge_importance'].max()
    importance_df['lgb_norm'] = importance_df['lgb_importance']/importance_df['lgb_importance'].max()
    importance_df['combined_score'] = importance_df['ridge_norm'] + importance_df['lgb_norm']
    importance_df = importance_df.sort_values(by='combined_score', ascending=False)
    top_features = importance_df.head(top_feature_count)['feature'].tolist()
    print(f"Top {top_feature_count} features: {top_features}")

    X_train_top = X_train_scaled[top_features]
    X_test_top = X_test_scaled[top_features]

    # -------------------
    # Stacking Ensemble
    # -------------------
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    ridge_oof = np.zeros(len(X_train_top))
    lgb_oof = np.zeros(len(X_train_top))
    for tr_idx, val_idx in kf.split(X_train_top):
        # Ridge fold
        ridge_fold = Ridge(alpha=best_alpha, random_state=42)
        ridge_fold.fit(X_train_top.iloc[tr_idx], y_train.iloc[tr_idx])
        ridge_oof[val_idx] = ridge_fold.predict(X_train_top.iloc[val_idx])
        # LGB fold
        lgb_fold = lgb.train(lgb_params, lgb.Dataset(X_train_top.iloc[tr_idx], y_train.iloc[tr_idx]), num_boost_round=500, valid_sets=[lgb.Dataset(X_train_top.iloc[val_idx], y_train.iloc[val_idx])])
        lgb_oof[val_idx] = lgb_fold.predict(X_train_top.iloc[val_idx])

    # Meta-model
    stack_X = pd.DataFrame({'ridge': ridge_oof, 'lgb': lgb_oof})
    meta_model = Ridge(alpha=1.0, random_state=42)
    meta_model.fit(stack_X, y_train)

    # -------------------
    # Train Final Models for Test Prediction
    # -------------------
    ridge_model.fit(X_train_top, y_train)
    lgb_model = lgb.train(lgb_params, lgb.Dataset(X_train_top, label=y_train), num_boost_round=500, valid_sets=[lgb.Dataset(X_train_top, label=y_train)])
    stack_test = pd.DataFrame({
        'ridge': ridge_model.predict(X_test_top),
        'lgb': lgb_model.predict(X_test_top)
    })
    y_pred = meta_model.predict(stack_test)

    # -------------------
    # Save Outputs
    # -------------------
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_dir = f'submission_{ts}'
    os.makedirs(submission_dir, exist_ok=True)

    # Save feature importance
    importance_df.to_csv(os.path.join(submission_dir, "feature_importance.csv"), index=False)

    # Save top coefficients from Ridge model
    top_coefs = pd.DataFrame({
        'feature': features,
        'coefficient': ridge_model.coef_
    }).sort_values(by='coefficient', key=abs, ascending=False).head(30)
    top_coefs.to_csv(os.path.join(submission_dir, "top_30_ridge_coefficients.csv"), index=False)

    # Save OOF predictions for stacking
    oof_preds_df = pd.DataFrame({
        'ridge_oof': ridge_oof,
        'lgb_oof': lgb_oof,
        'actual': y_train.values
    })
    oof_preds_df.to_csv(os.path.join(submission_dir, "oof_predictions.csv"), index=False)

    # Save submission
    submission = pd.DataFrame({'ID': test_df['ID'], 'W': y_pred})
    submission.to_csv(os.path.join(submission_dir, "submission.csv"), index=False)

    print(f"Pipeline complete. Outputs saved to {submission_dir}")

if __name__ == "__main__":
    run_pipeline()