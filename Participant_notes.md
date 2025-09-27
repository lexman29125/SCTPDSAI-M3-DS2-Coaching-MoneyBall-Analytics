# Effective Modeling Techniques in Baseball Analytics

This document highlights key modeling techniques from the Baseball Analytics tutorial that can help Kaggle contestants improve their predictive models.


## 1. Feature Engineering

**Creating Domain-Specific Features:**
- **Era and Decade Classification**: Categorizing years into baseball eras and decades helps capture historical context and trends.
- **Derived Performance Metrics**: Creating advanced baseball statistics like:
  - Batting metrics (eg. BA (Batting Average): H / AB)
  - Pitching metrics (eg WHIP (Walks + Hits per Inning Pitched): (BBA + HA) / (IPouts / 3)
  - Per-game normalization of key statistics ** important **

## 2. Unsupervised Learning for Feature Creation (Covered later in Lesson 3.5)

**K-means Clustering**:
- Using clustering to identify team archetypes
- Determining optimal cluster count with silhouette scores
- Adding cluster labels as features to improve model performance

**Principal Component Analysis**:
- Reducing feature space while preserving variance
- Addressing multicollinearity in baseball statistics
- Transforming correlated features into orthogonal principal components
- Visualizing team positioning in 2D/3D feature space
- Setting explained variance threshold to determine optimal number of components
- Using PCA components as input features for downstream models


## 3. Feature Scaling and Preprocessing

**Smart Scaling Strategy**:
- Selectively scaling only appropriate features
- Not scaling one-hot encoded columns and selectively for categorical features
- Saving scalers for future use

## 4. Model Selection and Regularization

**Comparing Multiple Regression Techniques**:
- Linear Regression as baseline
- Ridge Regression (L2 regularization) to handle multicollinearity
- Lasso Regression (L1 regularization) for feature selection
- ElasticNet (combined L1+L2) for balanced regularization

**Hyperparameter Tuning with Cross-Validation**:
- Using `RidgeCV` and `LassoCV` with logarithmic alpha ranges
- 5-fold cross-validation for robust parameter selection

## 5. Comprehensive Model Evaluation

**Multiple Evaluation Metrics**:
- Mean Absolute Error (MAE) as primary metric
- Root Mean Squared Error (RMSE) to penalize larger errors
- R² to measure explained variance
- Training vs. test set performance comparison to detect overfitting

**Feature Importance Analysis**:
- Ranking features by coefficient magnitude
- Identifying the most predictive variables

**Visual Evaluation**:
- Actual vs. predicted plots
- Residual analysis to check for patterns in errors

## 6. Team Collaboration Strategies

**Key Team Roles**:
- **Pipeline and Data Engineering**: Handles data cleaning and preprocessing
- **Feature Engineering & Analysis**: Creates domain-specific hypothesis and features
- **Modelling**: Implements ML algorithms, architecture and pipeline
- **Ensembling**: Combines models for optimal predictions

**Collaboration Essentials**:
- Use version control
- Document preprocessing steps
- Create reusable code
- Track experiments systematically
  - Maintain experiment logs with hyperparameters, metrics, and timestamps
  - Document failed approaches to avoid repeating unsuccessful strategies
  - Compare model iterations with consistent evaluation criteria
- Cross-validate each other's work
