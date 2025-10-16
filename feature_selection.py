# feature_selection.py

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')


def load_best_params():
    """Load best parameters from tuning"""
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 67, 'max_depth': 13, 'learning_rate': 0.010004219424732138,
        'feature_fraction': 0.6329129931469809, 'bagging_fraction': 0.8062944304248291, 'bagging_freq': 2,
        'min_child_samples': 136, 'min_child_weight': 0.02793882913795377, 'reg_alpha': 0.8658581250048688,
        'reg_lambda': 18.76093655187994, 'scale_pos_weight': 9.374436274260821
    }
    return params


def calculate_feature_importance(X, y, params, n_folds=3):
    """
    Calculate feature importance using CV
    """
    print("=" * 70)
    print("Calculating Feature Importance")
    print("=" * 70)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    feature_importance_df = pd.DataFrame()

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\nFold {fold + 1}/{n_folds}")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=3000,
            valid_sets=[lgb_val],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
        )

        fold_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importance(importance_type='gain'),
            'fold': fold + 1
        })

        feature_importance_df = pd.concat([feature_importance_df, fold_importance])

    # Aggregate
    importance_agg = feature_importance_df.groupby('feature')['importance'].agg([
        'mean', 'std', 'min', 'max'
    ]).reset_index()
    importance_agg = importance_agg.sort_values('mean', ascending=False)

    return importance_agg


def remove_low_importance_features(X_train, X_test, importance_df, threshold_percentile=25):
    """
    Remove features below importance threshold
    """
    print("\n" + "=" * 70)
    print(f"Removing Low Importance Features (Bottom {threshold_percentile}%)")
    print("=" * 70)

    # Calculate threshold
    threshold = np.percentile(importance_df['mean'], threshold_percentile)

    # Get features to keep
    features_to_keep = importance_df[importance_df['mean'] > threshold]['feature'].tolist()
    features_to_remove = importance_df[importance_df['mean'] <= threshold]['feature'].tolist()

    print(f"\nOriginal features: {len(X_train.columns)}")
    print(f"Features to keep: {len(features_to_keep)}")
    print(f"Features to remove: {len(features_to_remove)}")
    print(f"Threshold: {threshold:.2f}")

    print(f"\nBottom 10 features being removed:")
    bottom_features = importance_df.nsmallest(10, 'mean')[['feature', 'mean']]
    print(bottom_features.to_string(index=False))

    # Filter
    X_train_filtered = X_train[features_to_keep]
    X_test_filtered = X_test[features_to_keep]

    return X_train_filtered, X_test_filtered, features_to_keep


def remove_correlated_features(X_train, X_test, threshold=0.95):
    """
    Remove highly correlated features
    """
    print("\n" + "=" * 70)
    print(f"Removing Highly Correlated Features (threshold={threshold})")
    print("=" * 70)

    # Calculate correlation matrix
    corr_matrix = X_train.corr().abs()

    # Upper triangle
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # Find features with correlation > threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    print(f"\nFound {len(to_drop)} highly correlated features to remove")

    if to_drop:
        print("\nHighly correlated features:")
        for col in to_drop[:10]:  # Show first 10
            correlated_with = upper[col][upper[col] > threshold].index.tolist()
            print(f"  {col} correlated with: {correlated_with}")

    X_train_filtered = X_train.drop(columns=to_drop)
    X_test_filtered = X_test.drop(columns=to_drop)

    print(f"\nFeatures after removing correlations: {len(X_train_filtered.columns)}")

    return X_train_filtered, X_test_filtered


def train_with_selected_features(X_train, y_train, X_test, test_ids, params):
    """
    Train model with selected features
    """
    print("\n" + "=" * 70)
    print("Training with Selected Features (5-Fold CV)")
    print("=" * 70)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"\nFold {fold + 1}/5")

        X_train_fold = X_train.iloc[train_idx]
        X_val_fold = X_train.iloc[val_idx]
        y_train_fold = y_train.iloc[train_idx]
        y_val_fold = y_train.iloc[val_idx]

        lgb_train = lgb.Dataset(X_train_fold, y_train_fold)
        lgb_val = lgb.Dataset(X_val_fold, y_val_fold, reference=lgb_train)

        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=3000,
            valid_sets=[lgb_train, lgb_val],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100),
                lgb.log_evaluation(period=200)
            ]
        )

        oof_preds[val_idx] = model.predict(X_val_fold, num_iteration=model.best_iteration)
        test_preds += model.predict(X_test, num_iteration=model.best_iteration) / 5

        fold_score = roc_auc_score(y_val_fold, oof_preds[val_idx])
        fold_scores.append(fold_score)
        print(f"Fold {fold + 1} AUC: {fold_score:.5f}")

    overall_auc = roc_auc_score(y_train, oof_preds)

    print("\n" + "=" * 70)
    print("Results with Feature Selection")
    print("=" * 70)
    print(f"\nFold scores: {[f'{s:.5f}' for s in fold_scores]}")
    print(f"Mean AUC: {np.mean(fold_scores):.5f} (+/- {np.std(fold_scores):.5f})")
    print(f"Overall OOF AUC: {overall_auc:.5f}")

    return oof_preds, test_preds, overall_auc


def main():
    import os
    os.makedirs('output', exist_ok=True)

    print("\n" + "=" * 70)
    print("FEATURE SELECTION PIPELINE")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv('output/enhanced/app_train_optimized.csv')
    test_df = pd.read_csv('output/enhanced/app_test_optimized.csv')

    y_train = train_df['TARGET']
    test_ids = test_df['SK_ID_CURR']

    X_train = train_df.drop(columns=['SK_ID_CURR', 'TARGET'])
    X_test = test_df.drop(columns=['SK_ID_CURR'])

    # Encode categorical
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        print(f"Encoding {len(categorical_cols)} categorical features...")
        for col in categorical_cols:
            le = LabelEncoder()
            X_train[col] = X_train[col].fillna('missing')
            X_test[col] = X_test[col].fillna('missing')
            all_cats = pd.concat([X_train[col], X_test[col]]).unique()
            le.fit(all_cats)
            X_train[col] = le.transform(X_train[col])
            X_test[col] = le.transform(X_test[col])

    # Fill missing
    for col in X_train.columns:
        if X_train[col].isnull().any():
            median_val = X_train[col].median()
            X_train[col] = X_train[col].fillna(median_val)
            X_test[col] = X_test[col].fillna(median_val)

    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

    print(f"Initial shape: {X_train.shape}")

    # Load best params
    params = load_best_params()

    # Calculate feature importance
    importance_df = calculate_feature_importance(X_train, y_train, params, n_folds=3)
    importance_df.to_csv('output/feature_importance_detailed.csv', index=False)
    print("\n✓ Feature importance saved to: output/feature_importance_detailed.csv")

    # Remove low importance features
    X_train_filtered, X_test_filtered, selected_features = remove_low_importance_features(
        X_train, X_test, importance_df, threshold_percentile=20
    )

    # Remove correlated features
    X_train_final, X_test_final = remove_correlated_features(
        X_train_filtered, X_test_filtered, threshold=0.95
    )

    # Save selected features
    pd.DataFrame({'feature': X_train_final.columns}).to_csv(
        'output/selected_features.csv', index=False
    )
    print(f"\n✓ Selected features saved to: output/selected_features.csv")

    # Train with selected features
    oof_preds, test_preds, overall_auc = train_with_selected_features(
        X_train_final, y_train, X_test_final, test_ids, params
    )

    # Generate submission
    submission = pd.DataFrame({
        'SK_ID_CURR': test_ids,
        'TARGET': test_preds
    })
    submission.to_csv('output/submission/feature_selected_submission.csv', index=False)

    print("\n" + "=" * 70)
    print("FEATURE SELECTION COMPLETE!")
    print("=" * 70)
    print(f"\nOriginal features: {X_train.shape[1]}")
    print(f"Final features: {X_train_final.shape[1]}")
    print(f"Features removed: {X_train.shape[1] - X_train_final.shape[1]}")
    print(f"\nOOF AUC: {overall_auc:.5f}")
    print("\n✓ Submission saved to: output/feature_selected_submission.csv")


if __name__ == '__main__':
    main()
