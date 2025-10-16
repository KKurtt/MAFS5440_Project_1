# ensemble_model.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
import os
import gc

warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)


def load_optimized_data():
    """
    Load optimized training and test data
    """
    print("=" * 70)
    print("Loading Optimized Data for Ensemble")
    print("=" * 70)

    train_df = pd.read_csv('output/enhanced/app_train_optimized.csv')
    test_df = pd.read_csv('output/enhanced/app_test_optimized.csv')

    print(f"\nTraining data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    print(f"\nTarget distribution:")
    print(train_df['TARGET'].value_counts())
    print(f"Default rate: {train_df['TARGET'].mean():.2%}")

    return train_df, test_df


def prepare_data(train_df, test_df):
    """
    Prepare data for modeling
    """
    print("\n" + "=" * 70)
    print("Preparing Data")
    print("=" * 70)

    y_train = train_df['TARGET']
    test_ids = test_df['SK_ID_CURR']

    X_train = train_df.drop(columns=['SK_ID_CURR', 'TARGET'])
    X_test = test_df.drop(columns=['SK_ID_CURR'])

    # Encode categorical variables
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()

    if categorical_cols:
        print(f"\nEncoding {len(categorical_cols)} categorical features...")
        for col in categorical_cols:
            le = LabelEncoder()
            X_train[col] = X_train[col].fillna('missing')
            X_test[col] = X_test[col].fillna('missing')
            all_categories = pd.concat([X_train[col], X_test[col]]).unique()
            le.fit(all_categories)
            X_train[col] = le.transform(X_train[col])
            X_test[col] = le.transform(X_test[col])

    # Handle missing values
    for col in X_train.columns:
        if X_train[col].isnull().any():
            if X_train[col].dtype in ['int64', 'float64']:
                median_val = X_train[col].median()
                X_train[col] = X_train[col].fillna(median_val)
                X_test[col] = X_test[col].fillna(median_val)

    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)
    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_test.median())

    print(f"\n✓ Final feature count: {X_train.shape[1]}")
    print(f"✓ Training samples: {len(X_train)}")
    print(f"✓ Test samples: {len(X_test)}")

    return X_train, y_train, X_test, test_ids


def get_lgb_params():
    """LightGBM parameters"""
    # return {
    #     'objective': 'binary',
    #     'metric': 'auc',
    #     'boosting_type': 'gbdt',
    #     'num_leaves': 40,
    #     'max_depth': 8,
    #     'learning_rate': 0.02,
    #     'feature_fraction': 0.8,
    #     'bagging_fraction': 0.8,
    #     'bagging_freq': 5,
    #     'min_child_samples': 70,
    #     'min_child_weight': 0.001,
    #     'reg_alpha': 0.1,
    #     'reg_lambda': 10,
    #     'scale_pos_weight': 11,
    #     'verbose': -1,
    #     'random_state': 42,
    #     'n_jobs': -1
    # }
    return {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',

        'num_leaves': 67, 'max_depth': 13, 'learning_rate': 0.010004219424732138,
        'feature_fraction': 0.6329129931469809, 'bagging_fraction': 0.8062944304248291, 'bagging_freq': 2,
        'min_child_samples': 136, 'min_child_weight': 0.02793882913795377, 'reg_alpha': 0.8658581250048688,
        'reg_lambda': 18.76093655187994, 'scale_pos_weight': 9.374436274260821
    }


def get_xgb_params():
    """XGBoost parameters"""
    return {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 7,
        'learning_rate': 0.02,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 70,
        'reg_alpha': 0.1,
        'reg_lambda': 10,
        'scale_pos_weight': 11,
        'tree_method': 'hist',
        'random_state': 42,
        'n_jobs': -1
    }


def get_cat_params():
    """CatBoost parameters"""
    return {
        'iterations': 3000,
        'learning_rate': 0.02,
        'depth': 8,
        'l2_leaf_reg': 10,
        'random_seed': 42,
        'verbose': 0,
        'task_type': 'CPU',
        'scale_pos_weight': 11,
        'early_stopping_rounds': 100
    }


def train_lgb_fold(X_train, y_train, X_val, y_val, X_test):
    """Train LightGBM for one fold"""
    params = get_lgb_params()

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=3000,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
    )

    val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    test_pred = model.predict(X_test, num_iteration=model.best_iteration)

    return val_pred, test_pred, model


def train_xgb_fold(X_train, y_train, X_val, y_val, X_test):
    """Train XGBoost for one fold"""
    params = get_xgb_params()

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test)

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=3000,
        evals=[(dval, 'valid')],
        early_stopping_rounds=100,
        verbose_eval=0
    )

    val_pred = model.predict(dval)
    test_pred = model.predict(dtest)

    return val_pred, test_pred, model


def train_cat_fold(X_train, y_train, X_val, y_val, X_test):
    """Train CatBoost for one fold"""
    params = get_cat_params()

    model = CatBoostClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        verbose=False
    )

    val_pred = model.predict_proba(X_val)[:, 1]
    test_pred = model.predict_proba(X_test)[:, 1]

    return val_pred, test_pred, model


def train_ensemble_cv(X, y, X_test, n_splits=5):
    """
    Train ensemble with cross-validation
    """
    print("\n" + "=" * 70)
    print(f"Training Ensemble Model ({n_splits}-Fold CV)")
    print("=" * 70)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Initialize prediction arrays
    lgb_oof = np.zeros(len(X))
    xgb_oof = np.zeros(len(X))
    cat_oof = np.zeros(len(X))

    lgb_test = np.zeros(len(X_test))
    xgb_test = np.zeros(len(X_test))
    cat_test = np.zeros(len(X_test))

    # Store fold scores
    lgb_scores = []
    xgb_scores = []
    cat_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'=' * 50}")
        print(f"Fold {fold + 1}/{n_splits}")
        print(f"{'=' * 50}")

        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        print(f"Train size: {len(X_train_fold)}, Validation size: {len(X_val_fold)}")

        # Train LightGBM
        print("\n  Training LightGBM...")
        lgb_val_pred, lgb_test_pred, lgb_model = train_lgb_fold(
            X_train_fold, y_train_fold, X_val_fold, y_val_fold, X_test
        )
        lgb_oof[val_idx] = lgb_val_pred
        lgb_test += lgb_test_pred / n_splits
        lgb_auc = roc_auc_score(y_val_fold, lgb_val_pred)
        lgb_scores.append(lgb_auc)
        print(f"    LightGBM AUC: {lgb_auc:.5f}")

        del lgb_model
        gc.collect()

        # Train XGBoost
        print("  Training XGBoost...")
        xgb_val_pred, xgb_test_pred, xgb_model = train_xgb_fold(
            X_train_fold, y_train_fold, X_val_fold, y_val_fold, X_test
        )
        xgb_oof[val_idx] = xgb_val_pred
        xgb_test += xgb_test_pred / n_splits
        xgb_auc = roc_auc_score(y_val_fold, xgb_val_pred)
        xgb_scores.append(xgb_auc)
        print(f"    XGBoost AUC: {xgb_auc:.5f}")

        del xgb_model
        gc.collect()

        # Train CatBoost
        print("  Training CatBoost...")
        cat_val_pred, cat_test_pred, cat_model = train_cat_fold(
            X_train_fold, y_train_fold, X_val_fold, y_val_fold, X_test
        )
        cat_oof[val_idx] = cat_val_pred
        cat_test += cat_test_pred / n_splits
        cat_auc = roc_auc_score(y_val_fold, cat_val_pred)
        cat_scores.append(cat_auc)
        print(f"    CatBoost AUC: {cat_auc:.5f}")

        del cat_model, X_train_fold, X_val_fold, y_train_fold, y_val_fold
        gc.collect()

    # Calculate OOF scores
    lgb_oof_score = roc_auc_score(y, lgb_oof)
    xgb_oof_score = roc_auc_score(y, xgb_oof)
    cat_oof_score = roc_auc_score(y, cat_oof)

    print("\n" + "=" * 70)
    print("Individual Model Results")
    print("=" * 70)

    print(f"\nLightGBM:")
    print(f"  Fold scores: {[f'{s:.5f}' for s in lgb_scores]}")
    print(f"  Mean: {np.mean(lgb_scores):.5f} (+/- {np.std(lgb_scores):.5f})")
    print(f"  OOF AUC: {lgb_oof_score:.5f}")

    print(f"\nXGBoost:")
    print(f"  Fold scores: {[f'{s:.5f}' for s in xgb_scores]}")
    print(f"  Mean: {np.mean(xgb_scores):.5f} (+/- {np.std(xgb_scores):.5f})")
    print(f"  OOF AUC: {xgb_oof_score:.5f}")

    print(f"\nCatBoost:")
    print(f"  Fold scores: {[f'{s:.5f}' for s in cat_scores]}")
    print(f"  Mean: {np.mean(cat_scores):.5f} (+/- {np.std(cat_scores):.5f})")
    print(f"  OOF AUC: {cat_oof_score:.5f}")

    return lgb_oof, xgb_oof, cat_oof, lgb_test, xgb_test, cat_test


def find_best_ensemble_weights(y_true, lgb_oof, xgb_oof, cat_oof):
    """
    Find optimal ensemble weights using grid search
    """
    print("\n" + "=" * 70)
    print("Finding Optimal Ensemble Weights")
    print("=" * 70)

    best_score = 0
    best_weights = None

    # Grid search over weights
    weight_ranges = np.arange(0.0, 1.01, 0.05)

    results = []

    print("\nSearching weight combinations...")
    for lgb_w in weight_ranges:
        for xgb_w in weight_ranges:
            for cat_w in weight_ranges:
                # Weights must sum to 1
                if abs(lgb_w + xgb_w + cat_w - 1.0) < 0.01:
                    ensemble_pred = lgb_w * lgb_oof + xgb_w * xgb_oof + cat_w * cat_oof
                    score = roc_auc_score(y_true, ensemble_pred)

                    results.append({
                        'lgb_w': lgb_w,
                        'xgb_w': xgb_w,
                        'cat_w': cat_w,
                        'auc': score
                    })

                    if score > best_score:
                        best_score = score
                        best_weights = (lgb_w, xgb_w, cat_w)

    results_df = pd.DataFrame(results).sort_values('auc', ascending=False)

    print("\nTop 10 Weight Combinations:")
    print(results_df.head(10).to_string(index=False))

    print(f"\nBest Ensemble Weights:")
    print(f"  LightGBM: {best_weights[0]:.2f}")
    print(f"  XGBoost:  {best_weights[1]:.2f}")
    print(f"  CatBoost: {best_weights[2]:.2f}")
    print(f"  Ensemble OOF AUC: {best_score:.5f}")

    # Save weight search results
    results_df.to_csv('output/ensemble_weight_search.csv', index=False)
    print("\n✓ Weight search results saved to: output/ensemble_weight_search.csv")

    return best_weights, best_score


def plot_ensemble_comparison(y_true, lgb_oof, xgb_oof, cat_oof, ensemble_oof):
    """
    Plot comparison of individual models and ensemble
    """
    print("\n" + "=" * 70)
    print("Creating Ensemble Comparison Plots")
    print("=" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    models = [
        ('LightGBM', lgb_oof),
        ('XGBoost', xgb_oof),
        ('CatBoost', cat_oof),
        ('Ensemble', ensemble_oof)
    ]

    for idx, (name, preds) in enumerate(models):
        ax = axes[idx // 2, idx % 2]

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, preds)
        auc_score = roc_auc_score(y_true, preds)

        ax.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC (AUC = {auc_score:.4f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
        ax.set_title(f'{name} ROC Curve', fontsize=13, fontweight='bold')
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('output/ensemble_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Ensemble comparison saved to: output/ensemble_comparison.png")
    plt.close()

    # Performance comparison bar plot
    fig, ax = plt.subplots(figsize=(10, 6))

    model_names = ['LightGBM', 'XGBoost', 'CatBoost', 'Ensemble']
    scores = [
        roc_auc_score(y_true, lgb_oof),
        roc_auc_score(y_true, xgb_oof),
        roc_auc_score(y_true, cat_oof),
        roc_auc_score(y_true, ensemble_oof)
    ]

    colors = ['steelblue', 'green', 'orange', 'red']
    bars = ax.bar(model_names, scores, color=colors, alpha=0.7)

    # Add value labels
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{score:.5f}',
                ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax.set_ylabel('AUC Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim([min(scores) - 0.005, max(scores) + 0.005])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('output/model_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Model comparison saved to: output/model_comparison.png")
    plt.close()


def generate_ensemble_submission(test_preds, test_ids, filename='ensemble_submission.csv'):
    """
    Generate ensemble submission file
    """
    print("\n" + "=" * 70)
    print("Generating Ensemble Submission")
    print("=" * 70)

    submission = pd.DataFrame({
        'SK_ID_CURR': test_ids,
        'TARGET': test_preds
    })

    submission_path = f'output/{filename}'
    submission.to_csv(submission_path, index=False)

    print(f"\n✓ Submission saved to: {submission_path}")
    print(f"  Shape: {submission.shape}")
    print(f"\nSubmission Preview:")
    print(submission.head(10))
    print(f"\nPrediction Statistics:")
    print(f"  Min: {test_preds.min():.5f}")
    print(f"  Max: {test_preds.max():.5f}")
    print(f"  Mean: {test_preds.mean():.5f}")
    print(f"  Median: {np.median(test_preds):.5f}")

    return submission


def save_ensemble_summary(lgb_scores, xgb_scores, cat_scores,
                          best_weights, ensemble_score):
    """
    Save ensemble training summary
    """
    summary_path = 'output/ensemble_summary.txt'

    with open(summary_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("ENSEMBLE MODEL SUMMARY\n")
        f.write("=" * 70 + "\n\n")

        f.write("Individual Model Scores:\n")
        f.write("-" * 70 + "\n")
        f.write(f"LightGBM:\n")
        for i, score in enumerate(lgb_scores, 1):
            f.write(f"  Fold {i}: {score:.5f}\n")
        f.write(f"  Mean: {np.mean(lgb_scores):.5f} (+/- {np.std(lgb_scores):.5f})\n\n")

        f.write(f"XGBoost:\n")
        for i, score in enumerate(xgb_scores, 1):
            f.write(f"  Fold {i}: {score:.5f}\n")
        f.write(f"  Mean: {np.mean(xgb_scores):.5f} (+/- {np.std(xgb_scores):.5f})\n\n")

        f.write(f"CatBoost:\n")
        for i, score in enumerate(cat_scores, 1):
            f.write(f"  Fold {i}: {score:.5f}\n")
        f.write(f"  Mean: {np.mean(cat_scores):.5f} (+/- {np.std(cat_scores):.5f})\n\n")

        f.write("=" * 70 + "\n")
        f.write("Ensemble Configuration:\n")
        f.write("=" * 70 + "\n")
        f.write(f"Best Weights:\n")
        f.write(f"  LightGBM: {best_weights[0]:.2f}\n")
        f.write(f"  XGBoost:  {best_weights[1]:.2f}\n")
        f.write(f"  CatBoost: {best_weights[2]:.2f}\n\n")
        f.write(f"Ensemble OOF AUC: {ensemble_score:.5f}\n")

    print(f"\n✓ Ensemble summary saved to: {summary_path}")


if __name__ == '__main__':
    os.makedirs('output', exist_ok=True)

    print("\n" + "=" * 70)
    print("ENSEMBLE MODEL TRAINING PIPELINE")
    print("=" * 70)

    # Load data
    train_df, test_df = load_optimized_data()

    # Prepare data
    X_train, y_train, X_test, test_ids = prepare_data(train_df, test_df)

    del train_df, test_df
    gc.collect()

    # Train ensemble with CV
    lgb_oof, xgb_oof, cat_oof, lgb_test, xgb_test, cat_test = train_ensemble_cv(
        X_train, y_train, X_test, n_splits=5
    )

    # Find best ensemble weights
    best_weights, ensemble_score = find_best_ensemble_weights(
        y_train, lgb_oof, xgb_oof, cat_oof
    )

    # Generate ensemble predictions
    ensemble_oof = (best_weights[0] * lgb_oof +
                    best_weights[1] * xgb_oof +
                    best_weights[2] * cat_oof)

    ensemble_test = (best_weights[0] * lgb_test +
                     best_weights[1] * xgb_test +
                     best_weights[2] * cat_test)

    # Plot comparisons
    plot_ensemble_comparison(y_train, lgb_oof, xgb_oof, cat_oof, ensemble_oof)

    # Generate submission
    submission = generate_ensemble_submission(ensemble_test, test_ids, 'submission/ensemble_tune_submission.csv')

    # Save summary
    lgb_scores = [roc_auc_score(y_train.iloc[val_idx], lgb_oof[val_idx])
                  for _, val_idx in StratifiedKFold(5, shuffle=True, random_state=42).split(X_train, y_train)]
    xgb_scores = [roc_auc_score(y_train.iloc[val_idx], xgb_oof[val_idx])
                  for _, val_idx in StratifiedKFold(5, shuffle=True, random_state=42).split(X_train, y_train)]
    cat_scores = [roc_auc_score(y_train.iloc[val_idx], cat_oof[val_idx])
                  for _, val_idx in StratifiedKFold(5, shuffle=True, random_state=42).split(X_train, y_train)]

    save_ensemble_summary(lgb_scores, xgb_scores, cat_scores, best_weights, ensemble_score)

    print("\n" + "=" * 70)
    print("ENSEMBLE TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nFinal Results:")
    print(f"  Single LightGBM OOF AUC: {roc_auc_score(y_train, lgb_oof):.5f}")
    print(f"  Ensemble OOF AUC: {ensemble_score:.5f}")
    print(f"  Improvement: {ensemble_score - roc_auc_score(y_train, lgb_oof):+.5f}")

    print("\nNext step: Submit ensemble_submission.csv to Kaggle")
