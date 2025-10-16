import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import warnings
import os
import gc

warnings.filterwarnings('ignore')

# Set style for visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)


def load_enhanced_data():
    """
    Load enhanced training and test data
    """
    print("=" * 70)
    print("Loading Enhanced Data")
    print("=" * 70)

    train_df = pd.read_csv('output/enhanced/app_train_enhanced.csv')
    test_df = pd.read_csv('output/enhanced/app_test_enhanced.csv')

    print(f"\nTraining data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    print(f"\nTarget distribution:")
    print(train_df['TARGET'].value_counts())
    print(f"Default rate: {train_df['TARGET'].mean():.2%}")

    return train_df, test_df


def load_optimized_data():
    """
    Load optimized training and test data
    """
    print("=" * 70)
    print("Loading Optimized Data")
    print("=" * 70)

    train_df = pd.read_csv('output/enhanced/app_train_optimized.csv')
    test_df = pd.read_csv('output/enhanced/app_test_optimized.csv')

    print(f"\nTraining data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    print(f"\nTarget distribution:")
    print(train_df['TARGET'].value_counts())
    print(f"Default rate: {train_df['TARGET'].mean():.2%}")

    return train_df, test_df


def prepare_data_for_modeling(train_df, test_df):
    """
    Prepare data for machine learning models

    Parameters:
    -----------
    train_df : pd.DataFrame
        Training dataset
    test_df : pd.DataFrame
        Test dataset

    Returns:
    --------
    tuple : (X_train, y_train, X_test, test_ids, feature_names)
    """
    print("\n" + "=" * 70)
    print("Preparing Data for Modeling")
    print("=" * 70)

    # Separate target and ID
    y_train = train_df['TARGET']
    test_ids = test_df['SK_ID_CURR']

    # Drop ID and target from features
    X_train = train_df.drop(columns=['SK_ID_CURR', 'TARGET'])
    X_test = test_df.drop(columns=['SK_ID_CURR'])

    print(f"\nInitial feature count: {X_train.shape[1]}")

    # Encode categorical variables
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()

    if categorical_cols:
        print(f"\nEncoding {len(categorical_cols)} categorical features...")

        for col in categorical_cols:
            le = LabelEncoder()

            # Fill missing values
            X_train[col] = X_train[col].fillna('missing')
            X_test[col] = X_test[col].fillna('missing')

            # Fit on all unique categories
            all_categories = pd.concat([X_train[col], X_test[col]]).unique()
            le.fit(all_categories)

            X_train[col] = le.transform(X_train[col])
            X_test[col] = le.transform(X_test[col])

    # Handle numeric missing values
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns

    print(f"Handling missing values in {len(numeric_cols)} numeric features...")

    for col in numeric_cols:
        if X_train[col].isnull().any():
            median_val = X_train[col].median()
            X_train[col] = X_train[col].fillna(median_val)
            X_test[col] = X_test[col].fillna(median_val)

    # Replace infinite values
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    # Fill any remaining NaN
    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_test.median())

    # Ensure same columns in train and test
    assert X_train.shape[1] == X_test.shape[1], "Train and test have different number of features!"

    feature_names = X_train.columns.tolist()

    print(f"\n✓ Final feature count: {len(feature_names)}")
    print(f"✓ Training samples: {len(X_train)}")
    print(f"✓ Test samples: {len(X_test)}")

    return X_train, y_train, X_test, test_ids, feature_names


def get_optimized_lgbm_params():
    """
    Get optimized LightGBM parameters

    Returns:
    --------
    dict : Model parameters
    """
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 67, 'max_depth': 13, 'learning_rate': 0.010004219424732138,
        'feature_fraction': 0.6329129931469809, 'bagging_fraction': 0.8062944304248291, 'bagging_freq': 2,
        'min_child_samples': 136, 'min_child_weight': 0.02793882913795377, 'reg_alpha': 0.8658581250048688,
        'reg_lambda': 18.76093655187994, 'scale_pos_weight': 9.374436274260821
        # 'num_leaves': 40,
        # 'max_depth': 8,
        # 'learning_rate': 0.02,
        # 'feature_fraction': 0.8,
        # 'bagging_fraction': 0.8,
        # 'bagging_freq': 5,
        # 'min_child_samples': 70,
        # 'min_child_weight': 0.001,
        # 'reg_alpha': 0.1,
        # 'reg_lambda': 10,
        # 'scale_pos_weight': 11,  # 282686/24825 ≈ 11
        # 'verbose': -1,
        # 'random_state': 42,
        # 'n_jobs': -1
    }
    return params


def train_with_kfold_cv(X, y, X_test, n_splits=5, params=None):
    """
    Train LightGBM with K-Fold cross-validation

    Parameters:
    -----------
    X : pd.DataFrame
        Training features
    y : pd.Series
        Training target
    X_test : pd.DataFrame
        Test features
    n_splits : int
        Number of folds for cross-validation
    params : dict
        Model parameters

    Returns:
    --------
    tuple : (oof_predictions, test_predictions, feature_importance_df, fold_scores)
    """
    print("\n" + "=" * 70)
    print(f"Training LightGBM with {n_splits}-Fold Cross-Validation")
    print("=" * 70)

    if params is None:
        params = get_optimized_lgbm_params()

    # Print parameters
    print("\nModel Parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")

    # Initialize arrays for predictions
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))

    # Initialize feature importance dataframe
    feature_importance_df = pd.DataFrame()

    # Initialize stratified k-fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Store fold scores
    fold_scores = []

    # Training loop
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'=' * 50}")
        print(f"Fold {fold + 1}/{n_splits}")
        print(f"{'=' * 50}")

        # Split data
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}")

        # Create datasets
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

        # Train model
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

        # Out-of-fold predictions
        oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)

        # Calculate fold AUC
        fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
        fold_scores.append(fold_auc)
        print(f"\nFold {fold + 1} AUC: {fold_auc:.5f}")

        # Test predictions (averaged across folds)
        test_preds += model.predict(X_test, num_iteration=model.best_iteration) / n_splits

        # Feature importance
        fold_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importance(importance_type='gain'),
            'fold': fold + 1
        })
        feature_importance_df = pd.concat([feature_importance_df, fold_importance], axis=0)

        # Clear memory
        del X_train, X_val, y_train, y_val, lgb_train, lgb_val, model
        gc.collect()

    # Calculate overall OOF AUC
    overall_auc = roc_auc_score(y, oof_preds)

    print("\n" + "=" * 70)
    print("Cross-Validation Results")
    print("=" * 70)
    print(f"\nFold AUC scores:")
    for i, score in enumerate(fold_scores, 1):
        print(f"  Fold {i}: {score:.5f}")
    print(f"\nMean AUC: {np.mean(fold_scores):.5f} (+/- {np.std(fold_scores):.5f})")
    print(f"Overall OOF AUC: {overall_auc:.5f}")

    return oof_preds, test_preds, feature_importance_df, fold_scores


def plot_feature_importance(feature_importance_df, top_n=30):
    """
    Plot aggregated feature importance across folds

    Parameters:
    -----------
    feature_importance_df : pd.DataFrame
        Feature importance from all folds
    top_n : int
        Number of top features to display
    """
    print("\n" + "=" * 70)
    print("Analyzing Feature Importance")
    print("=" * 70)

    # Aggregate importance across folds
    importance_agg = feature_importance_df.groupby('feature')['importance'].agg(['mean', 'std']).reset_index()
    importance_agg = importance_agg.sort_values('mean', ascending=False).head(top_n)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))

    y_pos = np.arange(len(importance_agg))
    ax.barh(y_pos, importance_agg['mean'], xerr=importance_agg['std'],
            color='steelblue', alpha=0.8, error_kw={'ecolor': 'red', 'capsize': 3})

    ax.set_yticks(y_pos)
    ax.set_yticklabels(importance_agg['feature'])
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance (Gain)', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Most Important Features (with std across folds)',
                 fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig('output/enhanced_feature_importance.png', dpi=300, bbox_inches='tight')
    print("\n✓ Feature importance plot saved to: output/enhanced_feature_importance.png")
    plt.show()

    # Print top features
    print(f"\nTop {top_n} Important Features:")
    print(importance_agg.to_string(index=False))

    # Save to CSV
    importance_agg.to_csv('output/feature_importance_summary.csv', index=False)
    print("\n✓ Feature importance saved to: output/feature_importance_summary.csv")

    return importance_agg


def plot_roc_curve(y_true, y_pred):
    """
    Plot ROC curve

    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted probabilities
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot(fpr, tpr, color='darkorange', lw=2,
            label=f'ROC curve (AUC = {auc_score:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
            label='Random Classifier')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve',
                 fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('output/roc_curve.png', dpi=300, bbox_inches='tight')
    print("\n✓ ROC curve saved to: output/roc_curve.png")
    plt.show()


def analyze_predictions(y_true, y_pred):
    """
    Analyze prediction distribution

    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted probabilities
    """
    print("\n" + "=" * 70)
    print("Prediction Analysis")
    print("=" * 70)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Prediction distribution by actual label
    ax = axes[0]
    df_pred = pd.DataFrame({'True_Label': y_true, 'Prediction': y_pred})

    df_pred[df_pred['True_Label'] == 0]['Prediction'].hist(
        bins=50, alpha=0.6, label='No Default (0)', ax=ax, color='green'
    )
    df_pred[df_pred['True_Label'] == 1]['Prediction'].hist(
        bins=50, alpha=0.6, label='Default (1)', ax=ax, color='red'
    )

    ax.set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Prediction Distribution by True Label', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. Prediction quantiles
    ax = axes[1]
    quantiles = np.percentile(y_pred, [10, 25, 50, 75, 90])
    labels = ['10th', '25th', '50th', '75th', '90th']

    ax.bar(labels, quantiles, color='steelblue', alpha=0.7)
    ax.set_xlabel('Percentile', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted Probability', fontsize=12, fontweight='bold')
    ax.set_title('Prediction Quantiles', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (label, value) in enumerate(zip(labels, quantiles)):
        ax.text(i, value + 0.01, f'{value:.3f}', ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('output/prediction_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✓ Prediction analysis plot saved to: output/prediction_analysis.png")
    plt.show()

    # Print statistics
    print("\nPrediction Statistics:")
    print(f"  Min: {y_pred.min():.5f}")
    print(f"  Max: {y_pred.max():.5f}")
    print(f"  Mean: {y_pred.mean():.5f}")
    print(f"  Median: {np.median(y_pred):.5f}")
    print(f"  Std: {y_pred.std():.5f}")

    print("\nPrediction Distribution by Actual Label:")
    print(f"\nNo Default (0):")
    no_default_preds = df_pred[df_pred['True_Label'] == 0]['Prediction']
    print(f"  Mean: {no_default_preds.mean():.5f}")
    print(f"  Median: {no_default_preds.median():.5f}")

    print(f"\nDefault (1):")
    default_preds = df_pred[df_pred['True_Label'] == 1]['Prediction']
    print(f"  Mean: {default_preds.mean():.5f}")
    print(f"  Median: {default_preds.median():.5f}")


def generate_submission(test_preds, test_ids, filename='enhanced_submission.csv'):
    """
    Generate submission file for Kaggle

    Parameters:
    -----------
    test_preds : array-like
        Test predictions
    test_ids : array-like
        Test IDs
    filename : str
        Output filename
    """
    print("\n" + "=" * 70)
    print("Generating Submission File")
    print("=" * 70)

    submission = pd.DataFrame({
        'SK_ID_CURR': test_ids,
        'TARGET': test_preds
    })

    # Save submission
    submission_path = f'output/{filename}'
    submission.to_csv(submission_path, index=False)

    print(f"\n✓ Submission file saved to: {submission_path}")
    print(f"  Shape: {submission.shape}")
    print(f"\nSubmission Preview:")
    print(submission.head(10))

    print(f"\nSubmission Statistics:")
    print(f"  Min: {test_preds.min():.5f}")
    print(f"  Max: {test_preds.max():.5f}")
    print(f"  Mean: {test_preds.mean():.5f}")
    print(f"  Median: {np.median(test_preds):.5f}")

    return submission


def save_training_summary(fold_scores, overall_auc, feature_count):
    """
    Save training summary to text file
    """
    summary_path = 'output/training_summary.txt'

    with open(summary_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("ENHANCED MODEL TRAINING SUMMARY\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Total Features: {feature_count}\n")
        f.write(f"Number of Folds: {len(fold_scores)}\n\n")

        f.write("Fold Scores:\n")
        for i, score in enumerate(fold_scores, 1):
            f.write(f"  Fold {i}: {score:.5f}\n")

        f.write(f"\nMean AUC: {np.mean(fold_scores):.5f}\n")
        f.write(f"Std AUC: {np.std(fold_scores):.5f}\n")
        f.write(f"Overall OOF AUC: {overall_auc:.5f}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("Model Parameters:\n")
        f.write("=" * 70 + "\n")
        params = get_optimized_lgbm_params()
        for key, value in params.items():
            f.write(f"{key}: {value}\n")

    print(f"\n✓ Training summary saved to: {summary_path}")


if __name__ == '__main__':
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)

    print("\n" + "=" * 70)
    print("ENHANCED MODEL TRAINING PIPELINE")
    print("=" * 70)

    # Step 1: Load data
    # train_df, test_df = load_enhanced_data()
    train_df, test_df = load_optimized_data()

    # Step 2: Prepare data
    X_train, y_train, X_test, test_ids, feature_names = prepare_data_for_modeling(
        train_df, test_df
    )

    # Clear memory
    del train_df, test_df
    gc.collect()

    # Step 3: Train model with K-Fold CV
    oof_preds, test_preds, feature_importance_df, fold_scores = train_with_kfold_cv(
        X_train, y_train, X_test, n_splits=5
    )

    # Step 4: Calculate overall metrics
    overall_auc = roc_auc_score(y_train, oof_preds)

    # Step 5: Visualizations and Analysis
    print("\n" + "=" * 70)
    print("Generating Visualizations and Analysis")
    print("=" * 70)

    # Feature importance
    importance_summary = plot_feature_importance(feature_importance_df, top_n=30)

    # ROC curve
    plot_roc_curve(y_train, oof_preds)

    # Prediction analysis
    analyze_predictions(y_train, oof_preds)

    # Step 6: Generate submission
    # submission = generate_submission(test_preds, test_ids, 'submission/enhanced_submission.csv')
    submission = generate_submission(test_preds, test_ids, 'submission/optimized_tuned_submission.csv')

    # Step 7: Save training summary
    save_training_summary(fold_scores, overall_auc, len(feature_names))

    # Step 8: Compare with baseline
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print("\nBaseline Model (from previous run):")
    print("  Features: 129")
    print("  Validation AUC: 0.7633")
    print("  Public Leaderboard: 0.7462")

    print(f"\nEnhanced Model (current):")
    print(f"  Features: {len(feature_names)}")
    print(f"  CV Mean AUC: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})")
    print(f"  Overall OOF AUC: {overall_auc:.4f}")

    improvement = overall_auc - 0.7633
    print(f"\nImprovement over baseline: {improvement:+.4f} ({improvement / 0.7633 * 100:+.2f}%)")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)

    print("\nGenerated files in output/ directory:")
    print("  1. enhanced_submission.csv - Kaggle submission file")
    print("  2. enhanced_feature_importance.png - Feature importance visualization")
    print("  3. feature_importance_summary.csv - Detailed feature importance")
    print("  4. roc_curve.png - ROC curve")
    print("  5. prediction_analysis.png - Prediction distribution analysis")
    print("  6. training_summary.txt - Complete training summary")

    print("\n" + "=" * 70)
    print("Next Steps:")
    print("=" * 70)
    print("1. Submit enhanced_submission.csv to Kaggle")
    print("2. Compare public leaderboard score with baseline (0.7462)")
    print("3. Analyze top features for insights")
    print("4. Consider further optimizations if needed")
    print("=" * 70)
