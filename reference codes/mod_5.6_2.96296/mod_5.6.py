import os, time, warnings
os.chdir(os.path.dirname(__file__))

import numpy as np
import pandas as pd

from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import Ridge, ElasticNet, HuberRegressor, LinearRegression, Lasso
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor

# Optional LightGBM (kept, like your good runs)
HAS_LGBM = False
try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

# ====== CONFIG ======
SEEDS = [42, 7, 99]   # try [42,7,99,123,202] for a final push (slower)
N_SPLITS = 5
PYTH_EXP_GRID = [1.7, 1.8, 1.9, 2.0]  # fast grid that matched your best

def log(msg): print(msg, flush=True)

def clip_and_round(preds, G=None, delta=0.0):
    p = preds + float(delta)
    if G is not None:
        p = np.clip(p, 0, G)
    return np.rint(p).astype(int)

def make_strat_labels(y, win_bins=None, n_splits=5, max_bins=12):
    labels = None
    if win_bins is not None:
        vc = pd.Series(win_bins).value_counts()
        if len(vc) >= 2 and vc.min() >= n_splits:
            labels = win_bins.astype(int)
    if labels is None:
        for q in range(max_bins, 2, -1):
            try:
                cand = pd.qcut(y, q=q, labels=False, duplicates="drop").astype(int)
            except Exception:
                continue
            vc = pd.Series(cand).value_counts()
            if len(vc) >= 2 and vc.min() >= n_splits:
                labels = cand
                break
    return labels

# ---------- 1) load ----------
t0 = time.time()
log("Step 1/7: Loading data…")
train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")
y = train["W"].astype(float)

exclude = {"W", "win_bins"}
common = [c for c in train.columns if c in test.columns and c not in exclude]

# ---------- 2) Pythag tuning (once) + features ----------
def build_pyth(df: pd.DataFrame, exp: float):
    rp = np.power(df["R"], exp)
    rap = np.power(df["RA"], exp)
    win_pct = rp / (rp + rap + 1e-12)
    wins = win_pct * df["G"]
    return win_pct, wins

def tune_pyth_exp(df: pd.DataFrame, y: pd.Series, splits, exp_grid):
    best_exp, best_mae = 1.83, float("inf")
    for e in exp_grid:
        oof = np.zeros(len(df))
        pw = build_pyth(df, e)[1].to_numpy()
        for tr_idx, va_idx in splits:
            lr = LinearRegression()
            lr.fit(pw[tr_idx].reshape(-1, 1), y.iloc[tr_idx])
            oof[va_idx] = lr.predict(pw[va_idx].reshape(-1, 1))
        mae = mean_absolute_error(y, oof)
        if mae < best_mae:
            best_mae, best_exp = mae, e
    return best_exp, best_mae

# Use a fixed split just for tuning pyth exponent (seed=42)
strat_for_pyth = make_strat_labels(y, train["win_bins"] if "win_bins" in train.columns else None, n_splits=N_SPLITS)
kf_pyth = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42) if strat_for_pyth is not None else KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
splits_pyth = list(kf_pyth.split(train[common], strat_for_pyth)) if strat_for_pyth is not None else list(kf_pyth.split(train[common]))

log("Step 2/7: Tuning Pyth exponent…")
best_exp, pyth_oof_mae = tune_pyth_exp(train[common], y, splits_pyth, PYTH_EXP_GRID)
log(f" - Pyth exp = {best_exp} (calibrated pyth-only OOF MAE={pyth_oof_mae:.4f})")

def add_features(df: pd.DataFrame, pyth_exp: float) -> pd.DataFrame:
    df = df.copy()
    # core per-game
    df["R_pg"]  = df["R"]  / df["G"]
    df["RA_pg"] = df["RA"] / df["G"]
    df["run_diff"]    = df["R"] - df["RA"]
    df["run_diff_pg"] = df["run_diff"] / df["G"]
    # tuned Pythag
    p_pct, p_wins = build_pyth(df, pyth_exp)
    df["pyth_win_pct"] = p_pct
    df["pyth_wins"]    = p_wins
    # OBP/SLG/OPS (+TB fix) + Runs Created
    for col in ["HBP","SF"]:
        if col not in df.columns: df[col] = 0.0
    df["OBP_num"] = df["H"] + df["BB"] + df["HBP"]
    df["OBP_den"] = df["AB"] + df["BB"] + df["HBP"] + df["SF"]
    df["OBP"]     = df["OBP_num"] / df["OBP_den"].replace(0, 1)
    singles = df["H"] - df["2B"] - df["3B"] - df["HR"]
    df["TB"]  = singles + 2*df["2B"] + 3*df["3B"] + 4*df["HR"]
    df["SLG"] = df["TB"] / df["AB"].replace(0, 1)
    df["OPS"] = df["OBP"] + df["SLG"]
    df["RC"]    = (df["H"] + df["BB"]) * df["TB"] / (df["AB"] + df["BB"]).replace(0, 1)
    df["RC_pg"] = df["RC"] / df["G"]
    # park interactions
    if "BPF" in df.columns:
        df["OPS_BPF"]       = df["OPS"] * df["BPF"]
        df["run_diff_BPF"]  = df["run_diff_pg"] * df["BPF"]
        df["pyth_wins_BPF"] = df["pyth_wins"] * df["BPF"]
        df["RC_BPF"]        = df["RC_pg"] * df["BPF"]
    if "PPF" in df.columns:
        df["OPS_PPF"]       = df["OPS"] * df["PPF"]
        df["run_diff_PPF"]  = df["run_diff_pg"] * df["PPF"]
        df["pyth_wins_PPF"] = df["pyth_wins"] * df["PPF"]
        df["RC_PPF"]        = df["RC_pg"] * df["PPF"]
    # stabilizer
    df["ratio_R_RA"] = (df["R"] + 1) / (df["RA"] + 1)
    return df.fillna(0)

log("Step 3/7: Building features…")
X_train_full = add_features(train[common], best_exp)
X_test_full  = add_features(test[common],  best_exp)

# ---------- 3b) preprocessor ----------
def split_columns_for_scaling(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    dont_scale_prefixes = ("era_", "decade_")
    dont_scale = [c for c in numeric_cols if c.startswith(dont_scale_prefixes)]
    if "teamID" in numeric_cols: dont_scale.append("teamID")
    to_scale = [c for c in numeric_cols if c not in dont_scale]
    return to_scale, dont_scale

to_scale, dont_scale = split_columns_for_scaling(X_train_full)
pre = ColumnTransformer(
    transformers=[
        ("scale", StandardScaler(), to_scale),
        ("keep",  "passthrough",    dont_scale),
    ],
    remainder="drop",
)

focus_cols = [c for c in [
    "R_pg","RA_pg","run_diff_pg","pyth_wins","OPS","OBP","SLG","BPF","PPF","G","ratio_R_RA","RC_pg"
] if c in X_train_full.columns]

# Containers to accumulate across seeds
all_oof = { }
all_test = { }
model_names = ["ridge","enet","huber","gbr","hgbr","poly_ridge_focus","pyth_cal"]
if HAS_LGBM: model_names.append("lgbm")
for m in model_names:
    all_oof[m]  = np.zeros(len(X_train_full))
    all_test[m] = np.zeros(len(X_test_full))

# ---------- 4) multi-seed CV bagging ----------
log("Step 4/7: Multi-seed CV training…")
for si, seed in enumerate(SEEDS, 1):
    log(f" - Seed {seed} ({si}/{len(SEEDS)})")

    # build CV for this seed
    strat = make_strat_labels(y, train["win_bins"] if "win_bins" in train.columns else None, n_splits=N_SPLITS)
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed) if strat is not None else KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    splits = list(kf.split(X_train_full, strat)) if strat is not None else list(kf.split(X_train_full))

    # models for this seed (set random_state where applicable)
    ridge = Pipeline([("pre", pre), ("ridge", Ridge(alpha=10.0))])
    enet  = Pipeline([("pre", pre), ("enet",  ElasticNet(alpha=0.001, l1_ratio=0.2, max_iter=60000, tol=1e-4, random_state=seed))])
    huber = Pipeline([("pre", pre), ("huber", HuberRegressor(epsilon=1.35, alpha=1e-3, max_iter=20000, tol=1e-6))])
    gbr   = Pipeline([("pre", pre), ("gbr",   GradientBoostingRegressor(n_estimators=600, learning_rate=0.05, max_depth=3, random_state=seed))])
    hgbr  = Pipeline([("pre", pre), ("hgbr",  HistGradientBoostingRegressor(
        learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=10,
        early_stopping=True, validation_fraction=0.1, random_state=seed
    ))])

    models = {
        "ridge": ridge,
        "enet": enet,
        "huber": huber,
        "gbr": gbr,
        "hgbr": hgbr,
        "poly_ridge_focus": None,  # special
        "pyth_cal": None           # special
    }
    if HAS_LGBM:
        lgbm = Pipeline([("pre", pre), ("lgbm", LGBMRegressor(
            n_estimators=1500, learning_rate=0.03, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=-1
        ))])
        models["lgbm"] = lgbm

    # seed-local oof & test
    oof_seed = {m: np.zeros(len(X_train_full)) for m in models.keys()}
    test_seed = {m: np.zeros(len(X_test_full)) for m in models.keys()}

    # OOF
    for fold, (tr_idx, va_idx) in enumerate(splits, 1):
        log(f"   · Fold {fold}/{N_SPLITS}")
        Xtr, Xva = X_train_full.iloc[tr_idx], X_train_full.iloc[va_idx]
        ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]

        for name, mdl in models.items():
            if name == "poly_ridge_focus":
                mdl_f = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), Ridge(alpha=1.0))
                mdl_f.fit(Xtr[focus_cols].to_numpy(), ytr)
                oof_seed[name][va_idx] = mdl_f.predict(Xva[focus_cols].to_numpy())
            elif name == "pyth_cal":
                lr = LinearRegression().fit(Xtr[["pyth_wins"]].to_numpy(), ytr)
                oof_seed[name][va_idx] = lr.predict(Xva[["pyth_wins"]].to_numpy())
            else:
                mdl.fit(Xtr, ytr)
                p = mdl.predict(Xva)
                oof_seed[name][va_idx] = p if isinstance(p, np.ndarray) else p.to_numpy()

    # Full-fit → test for this seed (so trees vary with seed)
    for name, mdl in models.items():
        if name == "poly_ridge_focus":
            mdl_f = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), Ridge(alpha=1.0))
            mdl_f.fit(X_train_full[focus_cols].to_numpy(), y)
            test_seed[name] = mdl_f.predict(X_test_full[focus_cols].to_numpy())
        elif name == "pyth_cal":
            lr = LinearRegression().fit(X_train_full[["pyth_wins"]].to_numpy(), y)
            test_seed[name] = lr.predict(X_test_full[["pyth_wins"]].to_numpy())
        else:
            mdl.fit(X_train_full, y)
            tp = mdl.predict(X_test_full)
            test_seed[name] = tp if isinstance(tp, np.ndarray) else tp.to_numpy()

    # accumulate
    for m in models.keys():
        all_oof[m]  += oof_seed[m]
        all_test[m] += test_seed[m]

# average across seeds
for m in all_oof.keys():
    all_oof[m]  /= len(SEEDS)
    all_test[m] /= len(SEEDS)

# ---------- 5) Report per-model OOF and build stacks ----------
log("Step 5/7: Per-model OOF MAE (seed-averaged, pre-round)…")
model_oof_mae = {m: mean_absolute_error(y, all_oof[m]) for m in all_oof.keys()}
for name in sorted(model_oof_mae, key=model_oof_mae.get):
    log(f"  {name:16s}: {model_oof_mae[name]:.4f}")

model_list = list(all_oof.keys())
oof_stack  = np.column_stack([all_oof[m]  for m in model_list])
test_stack = np.column_stack([all_test[m] for m in model_list])

# candidates (same as your v5 style)
cands = []
# simple & median
cands.append(("simpleAvg", None, oof_stack.mean(axis=1), test_stack.mean(axis=1)))
cands.append(("medianAvg", None, np.median(oof_stack, axis=1), np.median(test_stack, axis=1)))
# linear meta
meta_lr = LinearRegression().fit(oof_stack, y)
cands.append(("stackLR", meta_lr, meta_lr.predict(oof_stack), meta_lr.predict(test_stack)))
# ridge meta (small grid)
best_ridge = None; best_ridge_oof = None; best_ridge_pred = None; best_ridge_alpha = None
for alpha in [1e-6, 1e-4, 1e-3, 1e-2, 5e-2, 0.1, 0.2, 0.5, 1.0]:
    rid = Ridge(alpha=alpha).fit(oof_stack, y)
    oof_r = rid.predict(oof_stack)
    if (best_ridge_oof is None) or (mean_absolute_error(y, oof_r) < mean_absolute_error(y, best_ridge_oof)):
        best_ridge, best_ridge_oof, best_ridge_pred, best_ridge_alpha = rid, oof_r, rid.predict(test_stack), alpha
cands.append(("stackRidge", best_ridge, best_ridge_oof, best_ridge_pred))
# sparse non-negative
lasso = Lasso(alpha=1e-3, positive=True, max_iter=100000).fit(oof_stack, y)
cands.append(("stackLassoPos", lasso, lasso.predict(oof_stack), lasso.predict(test_stack)))
# convex blend top-5 (coarse grid)
ranked = [m for m,_ in sorted(model_oof_mae.items(), key=lambda kv: kv[1])]
top5 = ranked[:5]
oof_top  = np.column_stack([all_oof[m]  for m in top5])
test_top = np.column_stack([all_test[m] for m in top5])
def convex_grid_best(oof_mat, test_mat, y_true, step=0.05):
    k = oof_mat.shape[1]; ticks = int(round(1/step))
    best_mae, best_w = None, None
    def gen(prefix, remaining, slots):
        if slots == 1: yield prefix+[remaining]
        else:
            for v in range(remaining+1): yield from gen(prefix+[v], remaining-v, slots-1)
    for parts in gen([], ticks, k):
        w = np.array(parts, dtype=float)/ticks
        pred = (oof_mat*w).sum(axis=1)
        mae = mean_absolute_error(y_true, pred)
        if (best_mae is None) or (mae < best_mae): best_mae, best_w = mae, w
    return best_mae, best_w, (test_mat*best_w).sum(axis=1)
convex_oof_mae, convex_w, convex_test_pred = convex_grid_best(oof_top, test_top, y, step=0.05)
cands.append(("convexBlend", {"models": top5, "weights": convex_w}, (oof_top*convex_w).sum(axis=1), convex_test_pred))

# choose best by pre-round OOF (your original recipe)
best_name, best_model, best_oof_pred, best_test_pred = min(cands, key=lambda c: mean_absolute_error(y, c[2]))
pre_round_oof_mae = mean_absolute_error(y, best_oof_pred)
log(f"Step 6/7: Ensemble choice (pre-round): {best_name} (OOF MAE={pre_round_oof_mae:.4f})")
if best_name == "stackLR":
    log("  weights: " + str(dict(zip(model_list, np.round(best_model.coef_, 3)))))
elif best_name == "stackRidge":
    log(f"  ridge alpha={best_ridge_alpha} | weights: " + str(dict(zip(model_list, np.round(best_model.coef_, 3)))))
elif best_name == "stackLassoPos":
    log("  nn-weights: " + str(dict(zip(model_list, np.round(best_model.coef_, 3)))))
elif best_name == "convexBlend":
    log("  convex over: " + str(top5))
    log("  weights: " + str(np.round(convex_w, 3).tolist()))

# ---------- 6) round-shift delta (global) ----------
log("Step 7/7: Learning round-shift delta…")
G_train = X_train_full["G"].to_numpy() if "G" in X_train_full.columns else None
G_test  = X_test_full["G"].to_numpy()  if "G" in X_test_full.columns  else None

best_delta, best_round_mae = 0.0, float("inf")
for delta in np.arange(-0.5, 0.501, 0.01):
    rr = clip_and_round(best_oof_pred, G_train, delta=delta)
    mae = mean_absolute_error(y, rr)
    if mae < best_round_mae:
        best_round_mae, best_delta = mae, float(delta)

preds = clip_and_round(best_test_pred, G_test, delta=best_delta)
log(f" - Chosen delta: {best_delta:+.2f} (OOF MAE after round={best_round_mae:.4f})")

# ---------- save ----------
out_path = os.path.abspath(f"moneyball_v58_bag{len(SEEDS)}_{best_name}_round.csv")
pd.DataFrame({"ID": test["ID"], "W": preds}).to_csv(out_path, index=False)
log("✅ Saved: " + out_path)
log(f"Total runtime: {time.time()-t0:.1f}s")
