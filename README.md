# MAFS 5440 Project 1: Home Credit Default Risk Prediction

This project addresses the Kaggle Home Credit Default Risk competition through systematic feature engineering and model optimization

## Project Overview

**Data Preparation**
- `data_exploration.py` - Conducts initial exploratory data analysis, examining data distributions, missing value patterns, and correlations to inform feature engineering strategies.

**Baseline Establishment**
- `basic_model.py` - Implements a baseline LightGBM model with minimal preprocessing, establishing the performance benchmark (AUC: 0.74622) for subsequent improvements.

**Feature Engineering**
- `advanced_feature_engineering.py` - Develops domain-informed features including debt ratios, payment behaviors, and credit utilization patterns from auxiliary tables.
- `optimization_features.py` - Creates aggregation features and temporal trend indicators, capturing historical payment patterns and behavioral evolution.

**Model Development**
- `enhanced_model.py` - Main modeling script that trains LightGBM with engineered features. This script is reused across experiments by manually adjusting data paths to load different feature sets.

**Model Optimization**
- `feature_selection.py` - Implements automated feature selection using importance-based filtering and recursive elimination to reduce model complexity while maintaining performance.
- `hyperparameter_tuning.py` - Applies Bayesian optimization to fine-tune model hyperparameters, achieving marginal but consistent improvements.

**Ensemble Experiments**
- `ensemble_model.py` - Explores various ensemble strategies including weighted averaging and stacking, revealing that well-tuned individual models outperform simple ensemble approaches.

## Notes

- Original datasets (`DataLibrary/`) and intermediate feature files (`output/enhanced/`) are excluded due to size constraints.
- All submission files and experimental outputs are preserved in `output/submission/` for reproducibility verification.

## Results

Private Leaderboard Score: **0.79280** (Top 15%, ranked 1070/7180)