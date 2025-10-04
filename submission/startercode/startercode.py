# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# Load the pre-processed train and test datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Display basic information about the datasets
print(f"Training set shape: {train_df.shape}")
print(f"Test set shape: {test_df.shape}")

# Select only the default features from DATA_DESCRIPTION.md
default_features = [
    # Basic Statistics
    'G', 'R', 'AB', 'H', '2B', '3B', 'HR', 'BB', 'SO', 'SB', 'CS', 'HBP', 'SF',
    'RA', 'ER', 'ERA', 'CG', 'SHO', 'SV', 'IPouts', 'HA', 'HRA', 'BBA', 'SOA',
    'E', 'DP', 'FP', 'attendance', 'BPF', 'PPF',
    
    # Derived Features
    'R_per_game', 'RA_per_game', 'mlb_rpg',
    
    # Era Indicators
    'era_1', 'era_2', 'era_3', 'era_4', 'era_5', 'era_6', 'era_7', 'era_8',
    
    # Decade Indicators
    'decade_1910', 'decade_1920', 'decade_1930', 'decade_1940', 'decade_1950',
    'decade_1960', 'decade_1970', 'decade_1980', 'decade_1990', 'decade_2000', 'decade_2010'
]

# Filter features that exist in both datasets
available_features = [col for col in default_features if col in train_df.columns and col in test_df.columns]
print(f"Number of available default features: {len(available_features)}")

# Separate features and target variable
X_train = train_df[available_features]
y_train = train_df['W']
X_test = test_df[available_features]


# Scale features
# Identify columns to exclude from scaling (one-hot encoded and label columns)
one_hot_cols = [col for col in X_train.columns if col.startswith(('era_', 'decade_'))]
other_cols = [col for col in X_train.columns if col not in one_hot_cols]

# Scale only non-one-hot features
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[other_cols] = scaler.fit_transform(X_train[other_cols])
X_test_scaled[other_cols] = scaler.transform(X_test[other_cols])

# Linear Regression Model
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
lr_train_preds = lr.predict(X_train_scaled)
lr_test_preds = lr.predict(X_test_scaled)

# Evaluate Linear Regression on training data only
lr_train_mae = mean_absolute_error(y_train, lr_train_preds)
lr_train_rmse = np.sqrt(mean_squared_error(y_train, lr_train_preds))
lr_train_r2 = r2_score(y_train, lr_train_preds)

print(f"Linear Regression Performance (Training Data):")
print(f"  Training MAE: {lr_train_mae:.4f}")
print(f"  Training RMSE: {lr_train_rmse:.4f}")
print(f"  Training R²: {lr_train_r2:.4f}")

# Feature importance from Linear Regression
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': lr.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Visualize actual vs predicted values on training data
plt.figure(figsize=(10, 6))
plt.scatter(y_train, lr_train_preds, alpha=0.5)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
plt.xlabel('Actual Wins')
plt.ylabel('Predicted Wins')
plt.title('Linear Regression: Actual vs Predicted Wins (Training Data)')
plt.grid(True, alpha=0.3)

# Add residual plot on training data
plt.figure(figsize=(10, 6))
residuals = y_train - lr_train_preds
plt.scatter(lr_train_preds, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Wins')
plt.ylabel('Residuals')
plt.title('Linear Regression: Residual Plot (Training Data)')
plt.grid(True, alpha=0.3)
plt.show()

# Create submission directory if it does not exist
os.makedirs('submission', exist_ok=True)

# Prepare submission dataframe with rounded predictions and test set indices as ID
submission_df = pd.DataFrame({
    'ID': test_df.index,
    'W': np.round(lr_test_preds).astype(int)
})

# Save to CSV
submission_df.to_csv('submission/submission.csv', index=False)