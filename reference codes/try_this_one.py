# overall 2.995
import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import RidgeCV
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

# ==============================================================================
# 1. LOAD DATASETS AND SETUP
# ==============================================================================
print("--- 1. LOADING AND PREPARING DATASETS ---")
#df_train = pd.read_csv('train.csv')
#df_test = pd.read_csv('test.csv')

# --- Combine for unified imputation/FE/Clustering ---
# We'll work on copies for safety
df_all = pd.concat([df_train.drop('W', axis=1), df_test], ignore_index=True)
id_col = df_test['ID'] # Save the IDs for the submission file


# ==============================================================================
# 2. FEATURE ENGINEERING (6 SURVIVING FEATURES)
# ==============================================================================
# The features PA, wOBA, K_rate_off, and BB_rate_off are EXCLUDED
# as they require the missing HBP and SF columns.

# Calculate IPouts for all pitching features
IP = df_all['IPouts'] / 3.0
# Ensure IP has no zeros for division
IP = IP.replace(0, np.finfo(float).eps) 

# --- 2a. Core Engineered Features (BA, WHIP) ---
df_all['BA'] = df_all['H'] / (df_all['AB'].replace(0, np.finfo(float).eps))
df_all['WHIP'] = (df_all['BBA'] + df_all['HA']) / IP

# --- 2b. New Advanced Features (Reduced Set) ---
df_all['ISO'] = (df_all['2B'] + 2*df_all['3B'] + 3*df_all['HR']) / df_all['AB'].replace(0, np.finfo(float).eps)
df_all['BABIP'] = (df_all['H'] - df_all['HR']) / (df_all['AB'] - df_all['SO'].fillna(0) - df_all['HR']).replace(0, np.finfo(float).eps)
df_all['FIP_CORE'] = (13*df_all['HRA'] + 3*df_all['BBA'] - 2*df_all['SOA']) / IP
df_all['K_BB_ratio_pitch'] = df_all['SOA'] / df_all['BBA'].replace(0, np.finfo(float).eps)


# Impute NaNs using Mean from the entire combined dataset
numerical_cols = df_all.select_dtypes(include=np.number).columns
for col in numerical_cols:
    df_all[col] = df_all[col].fillna(df_all[col].mean())


# ==============================================================================
# 3. UNSUPERVISED FEATURE CREATION (K-MEANS CLUSTERING, K=5)
# ==============================================================================
print("\n--- 2. K-MEANS CLUSTERING (K=5) ---")
# Use the set of engineered rate stats for clustering
clustering_features = ['FIP_CORE', 'BA', 'WHIP', 'ISO', 'BABIP']

# Train K-means ONLY on the original training data portion (for proper methodology)
X_cluster_train = df_all.iloc[:len(df_train)][clustering_features]
scaler_cluster = StandardScaler()
X_cluster_train_scaled = scaler_cluster.fit_transform(X_cluster_train)
kmeans = KMeans(n_clusters=5, random_state=42, n_init='auto')
kmeans.fit(X_cluster_train_scaled)

# Predict cluster IDs for ALL data (train and test)
X_cluster_all_scaled = scaler_cluster.transform(df_all[clustering_features])
df_all['cluster_id'] = kmeans.predict(X_cluster_all_scaled)


# ==============================================================================
# 4. SPLIT DATA BACK AND TRAIN FINAL PIPELINE
# ==============================================================================

# Define the FINAL features list based on what's available
default_features = [
    'G', 'R', 'AB', 'H', '2B', '3B', 'HR', 'BB', 'SO', 'SB',
    'RA', 'ER', 'ERA', 'CG', 'SHO', 'SV', 'IPouts', 'HA', 'HRA', 'BBA', 'SOA',
    'E', 'DP', 'FP', 'mlb_rpg', 'attendance', 'BPF', 'PPF',
    'era_1', 'era_2', 'era_3', 'era_4', 'era_5', 'era_6', 'era_7', 'era_8',
    'decade_1910', 'decade_1920', 'decade_1930', 'decade_1940', 'decade_1950',
    'decade_1960', 'decade_1970', 'decade_1980', 'decade_1990', 'decade_2000', 'decade_2010'
]
engineered_features = ['BA', 'WHIP', 'ISO', 'BABIP', 'FIP_CORE', 'K_BB_ratio_pitch', 'cluster_id']

# Create a master list of all columns to select
final_cols = [col for col in (default_features + engineered_features) if col in df_all.columns]
final_cols = list(pd.Index(final_cols).drop_duplicates()) # Ensure uniqueness

# Split data back into final training features (X) and test features (X_test_final)
X = df_all.iloc[:len(df_train)][final_cols]
y = df_train['W']
X_test_final = df_all.iloc[len(df_train):][final_cols]


# Define features for the pipeline
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
# cluster_id must be treated as categorical for the one-hot encoding step
categorical_features = ['cluster_id'] 

preprocessor = ColumnTransformer(
    transformers=[
        ('num_scale', StandardScaler(), numerical_features),
        ('cat_ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ],
    remainder='drop'
)

# Define the RidgeCV model and pipeline
alphas = np.logspace(-2, 2, 100)
ridge_cv = RidgeCV(alphas=alphas, cv=5)
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', ridge_cv)
])

# Train the pipeline on the full training set
print("\n--- 4. TRAINING FINAL MODEL AND MAKING PREDICTIONS ---")
model_pipeline.fit(X, y)


# ==============================================================================
# 5. PREDICTION AND SUBMISSION FILE GENERATION
# ==============================================================================

# Predict on the holdout test.csv data
final_predictions = model_pipeline.predict(X_test_final)

# Create the submission DataFrame
submission_df = pd.DataFrame({
    'ID': id_col,
    'W': final_predictions.round().astype(int)
})

# --- FILENAME GENERATION ---
# 1. Generate a formatted timestamp (e.g., '20250930_075250')
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# 2. Create the unique filename
filename = f'submission_{timestamp}.csv'

# Save to the unique submission file
submission_df.to_csv(filename, index=False)
print(f"\nFinal predictions made on test.csv and saved to '{filename}'.")
print("Submission format (first 5 rows):")
print(submission_df.head().to_markdown(index=False))