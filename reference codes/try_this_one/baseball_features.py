import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import RidgeCV
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

# ==============================================================================
# 1. LOAD DATA AND FEATURE ENGINEERING
# ==============================================================================
print("--- 1. LOADING DATA AND FEATURE ENGINEERING ---")
df = pd.read_csv('train.csv')

# Feature Engineering: BA and WHIP
df['BA'] = df['H'] / (df['AB'].replace(0, np.finfo(float).eps))
IP = df['IPouts'] / 3.0
df['WHIP'] = (df['BBA'] + df['HA']) / IP.replace(0, np.finfo(float).eps)

# ==============================================================================
# 2. K-MEANS CLUSTERING (K=5)
# ==============================================================================
print("\n--- 2. K-MEANS CLUSTERING (K=5) ---")

# Features used for clustering
clustering_features = ['R', 'RA', 'BA', 'WHIP']
X_cluster = df[clustering_features]

# Standardize the clustering features
scaler_cluster = StandardScaler()
X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)

# Apply K-Means with K=5
kmeans = KMeans(n_clusters=5, random_state=42, n_init='auto')
df['cluster_id'] = kmeans.fit_predict(X_cluster_scaled)

print(f"Created 'cluster_id' feature with 5 unique team archetypes.")

# ==============================================================================
# 3. RIDGE REGRESSION (WITH CLUSTER FEATURE)
# ==============================================================================

# Define all available features
default_features = [
    'G', 'R', 'AB', 'H', '2B', '3B', 'HR', 'BB', 'SO', 'SB',
    'RA', 'ER', 'ERA', 'CG', 'SHO', 'SV', 'IPouts', 'HA', 'HRA', 'BBA', 'SOA',
    'E', 'DP', 'FP', 'mlb_rpg',
    'R_per_game', 'RA_per_game', 'CS', 'HBP', 'SF', 'attendance', 'BPF', 'PPF',
    'era_1', 'era_2', 'era_3', 'era_4', 'era_5', 'era_6', 'era_7', 'era_8',
    'decade_1910', 'decade_1920', 'decade_1930', 'decade_1940', 'decade_1950',
    'decade_1960', 'decade_1970', 'decade_1980', 'decade_1990', 'decade_2000', 'decade_2010'
]
target = 'W'

# Full features set: existing + engineered + cluster ID
features = [col for col in default_features if col in df.columns]
features.extend(['BA', 'WHIP', 'cluster_id'])
X = df[[f for f in features if f in df.columns and f != target]]
y = df[target]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify numerical and categorical features for separate pre-processing
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = ['cluster_id']

# Create a preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        # Impute NaNs with mean and Scale numerical features
        ('num', StandardScaler(), numerical_features),
        # One-hot encode the categorical cluster_id
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ],
    remainder='passthrough'
)

# Define the model pipeline: Preprocessor -> RidgeCV
alphas = np.logspace(-2, 2, 100)
ridge_cv = RidgeCV(alphas=alphas, cv=5)

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', ridge_cv)
])

# Train the pipeline
print("\n--- 3. TRAINING RIDGECV MODEL (WITH CLUSTER FEATURE) ---")
model_pipeline.fit(X_train, y_train)

# Optimal Alpha Selection (retrieved from the regressor step in the pipeline)
optimal_alpha = model_pipeline['regressor'].alpha_
print(f"Optimal Alpha found by RidgeCV: {optimal_alpha:.4f}")

# Make predictions
pipeline_test_preds = model_pipeline.predict(X_test)

# Evaluate Ridge Regression with Cluster Feature
pipeline_test_mae = mean_absolute_error(y_test, pipeline_test_preds)

print(f"\nFINAL MODEL PERFORMANCE (RidgeCV + BA/WHIP + K=5 Cluster):")
print(f"  Test MAE: {pipeline_test_mae:.4f}")