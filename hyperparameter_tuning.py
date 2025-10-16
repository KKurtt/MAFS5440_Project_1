# hyperparameter_tuning.py

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import optuna
from optuna import Trial
import warnings
import os
import gc

warnings.filterwarnings('ignore')


def load_and_prepare_data():
    """
    Load and prepare data for tuning
    """
    print("=" * 70)
    print("Loading Data for Hyperparameter Tuning")
    print("=" * 70)

    train_df = pd.read_csv('output/enhanced/app_train_optimized.csv')

    y_train = train_df['TARGET']
    X_train = train_df.drop(columns=['SK_ID_CURR', 'TARGET'])

    # Encode categorical
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        print(f"\nEncoding {len(categorical_cols)} categorical features...")
        for col in categorical_cols:
            le = LabelEncoder()
            X_train[col] = X_train[col].fillna('missing')
            le.fit(X_train[col])
            X_train[col] = le.transform(X_train[col])

    # Handle missing values
    for col in X_train.columns:
        if X_train[col].isnull().any():
            X_train[col] = X_train[col].fillna(X_train[col].median())

    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_train = X_train.fillna(X_train.median())

    print(f"\n✓ Data shape: {X_train.shape}")
    print(f"✓ Target distribution: {y_train.value_counts().to_dict()}")

    return X_train, y_train


def objective(trial: Trial, X, y, n_folds=3, use_gpu=True):
    """
    Optuna objective function for LightGBM

    Using 3-fold CV for faster optimization
    """

    # Define hyperparameter search space
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'random_state': 42,

        # Parameters to optimize
        'num_leaves': trial.suggest_int('num_leaves', 20, 80),
        'max_depth': trial.suggest_int('max_depth', 6, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 0.95),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 0.95),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 30, 150),
        'min_child_weight': trial.suggest_float('min_child_weight', 0.001, 0.1, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 20.0),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 8, 14)
    }

    # GPU settings
    if use_gpu:
        params['device'] = 'gpu'
        params['gpu_platform_id'] = 0
        params['gpu_device_id'] = 0
        # GPU specific parameters for better performance
        params['max_bin'] = 63  # GPU works better with smaller max_bin
    else:
        params['n_jobs'] = -1

    # Cross-validation
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        lgb_train = lgb.Dataset(X_train_fold, y_train_fold)
        lgb_val = lgb.Dataset(X_val_fold, y_val_fold, reference=lgb_train)

        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=2000,
            valid_sets=[lgb_val],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=0)
            ]
        )

        preds = model.predict(X_val_fold, num_iteration=model.best_iteration)
        score = roc_auc_score(y_val_fold, preds)
        scores.append(score)

        del X_train_fold, X_val_fold, y_train_fold, y_val_fold, lgb_train, lgb_val, model
        gc.collect()

    return np.mean(scores)


def check_gpu_availability():
    """
    Check if GPU is available for LightGBM
    """
    try:
        # Try to create a simple GPU dataset
        test_data = lgb.Dataset(np.random.rand(100, 10), np.random.randint(0, 2, 100))
        params = {'device': 'gpu', 'verbosity': -1}
        lgb.train(params, test_data, num_boost_round=1)
        print("✓ GPU is available and will be used for training")
        return True
    except Exception as e:
        print(f"⚠ GPU not available: {str(e)}")
        print("  Falling back to CPU training")
        return False


def optimize_hyperparameters(X, y, n_trials=100, use_gpu=True):
    """
    Run Optuna optimization
    """
    print("\n" + "=" * 70)
    print(f"Starting Hyperparameter Optimization ({n_trials} trials)")
    print("=" * 70)

    if use_gpu:
        gpu_available = check_gpu_availability()
        if not gpu_available:
            use_gpu = False

    compute_mode = "GPU" if use_gpu else "CPU"
    print(f"\nCompute mode: {compute_mode}")
    print("This may take 1-3 hours depending on your hardware...")
    print("Using 3-fold CV for faster optimization\n")

    # Create study
    study = optuna.create_study(
        direction='maximize',
        study_name='lgb_optimization',
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    # Optimize
    study.optimize(
        lambda trial: objective(trial, X, y, n_folds=3, use_gpu=use_gpu),
        n_trials=n_trials,
        show_progress_bar=True
    )

    # Results
    print("\n" + "=" * 70)
    print("Optimization Results")
    print("=" * 70)

    print(f"\nBest trial:")
    print(f"  Value (AUC): {study.best_value:.5f}")

    print(f"\nBest parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Save results
    trials_df = study.trials_dataframe()
    trials_df.to_csv('output/optuna_trials.csv', index=False)
    print(f"\n✓ All trials saved to: output/optuna_trials.csv")

    # Save best parameters
    best_params = study.best_params.copy()
    best_params['objective'] = 'binary'
    best_params['metric'] = 'auc'
    best_params['boosting_type'] = 'gbdt'
    best_params['verbosity'] = -1
    best_params['random_state'] = 42

    import json
    with open('output/best_lgb_params.json', 'w') as f:
        json.dump(best_params, f, indent=4)
    print(f"✓ Best parameters saved to: output/best_lgb_params.json")

    return study.best_params, study.best_value, use_gpu


def plot_optimization_history(trials_df):
    """
    Plot optimization history
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style('whitegrid')

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 1. Optimization history
    ax = axes[0, 0]
    ax.plot(trials_df['number'], trials_df['value'], 'o-', alpha=0.6)
    ax.axhline(y=trials_df['value'].max(), color='r', linestyle='--',
               label=f'Best: {trials_df["value"].max():.5f}')
    ax.set_xlabel('Trial', fontweight='bold')
    ax.set_ylabel('AUC Score', fontweight='bold')
    ax.set_title('Optimization History', fontweight='bold', fontsize=13)
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. Best score over time
    ax = axes[0, 1]
    best_scores = trials_df['value'].cummax()
    ax.plot(trials_df['number'], best_scores, 'g-', linewidth=2)
    ax.set_xlabel('Trial', fontweight='bold')
    ax.set_ylabel('Best AUC Score', fontweight='bold')
    ax.set_title('Best Score Over Time', fontweight='bold', fontsize=13)
    ax.grid(alpha=0.3)

    # 3. Parameter importance (learning_rate)
    ax = axes[1, 0]
    param_col = 'params_learning_rate'
    if param_col in trials_df.columns:
        ax.scatter(trials_df[param_col], trials_df['value'], alpha=0.5)
        ax.set_xlabel('Learning Rate', fontweight='bold')
        ax.set_ylabel('AUC Score', fontweight='bold')
        ax.set_title('Learning Rate vs AUC', fontweight='bold', fontsize=13)
        ax.grid(alpha=0.3)

    # 4. Parameter importance (num_leaves)
    ax = axes[1, 1]
    param_col = 'params_num_leaves'
    if param_col in trials_df.columns:
        ax.scatter(trials_df[param_col], trials_df['value'], alpha=0.5, c='orange')
        ax.set_xlabel('Num Leaves', fontweight='bold')
        ax.set_ylabel('AUC Score', fontweight='bold')
        ax.set_title('Num Leaves vs AUC', fontweight='bold', fontsize=13)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('output/optimization_history.png', dpi=300, bbox_inches='tight')
    print("\n✓ Optimization history plot saved to: output/optimization_history.png")
    plt.close()


def train_with_best_params(X, y, X_test, test_ids, best_params, use_gpu=True):
    """
    Train final model with best parameters using 5-fold CV
    """
    print("\n" + "=" * 70)
    print("Training Final Model with Best Parameters (5-Fold CV)")
    print("=" * 70)

    # Add required parameters
    params = best_params.copy()
    params['objective'] = 'binary'
    params['metric'] = 'auc'
    params['boosting_type'] = 'gbdt'
    params['verbosity'] = -1
    params['random_state'] = 42

    # GPU settings
    if use_gpu:
        params['device'] = 'gpu'
        params['gpu_platform_id'] = 0
        params['gpu_device_id'] = 0
        params['max_bin'] = 63
    else:
        params['n_jobs'] = -1

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\nFold {fold + 1}/5")

        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

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

        del X_train_fold, X_val_fold, y_train_fold, y_val_fold, lgb_train, lgb_val, model
        gc.collect()

    overall_auc = roc_auc_score(y, oof_preds)

    print("\n" + "=" * 70)
    print("Final Results")
    print("=" * 70)
    print(f"\nFold scores: {[f'{s:.5f}' for s in fold_scores]}")
    print(f"Mean AUC: {np.mean(fold_scores):.5f} (+/- {np.std(fold_scores):.5f})")
    print(f"Overall OOF AUC: {overall_auc:.5f}")

    # Generate submission
    os.makedirs('output/submission', exist_ok=True)
    submission = pd.DataFrame({
        'SK_ID_CURR': test_ids,
        'TARGET': test_preds
    })
    submission.to_csv('output/submission/tuned_submission.csv', index=False)
    print(f"\n✓ Tuned submission saved to: output/submission/tuned_submission.csv")

    return oof_preds, test_preds, fold_scores, overall_auc


if __name__ == '__main__':
    os.makedirs('output', exist_ok=True)

    print("\n" + "=" * 70)
    print("HYPERPARAMETER TUNING PIPELINE")
    print("=" * 70)

    # Load and prepare data
    X_train, y_train = load_and_prepare_data()

    # Run optimization with GPU if available
    best_params, best_value, use_gpu = optimize_hyperparameters(
        X_train, y_train, n_trials=100, use_gpu=True
    )

    # Plot optimization history
    trials_df = pd.read_csv('output/optuna_trials.csv')
    plot_optimization_history(trials_df)

    # Load test data
    print("\n" + "=" * 70)
    print("Loading Test Data")
    print("=" * 70)

    test_df = pd.read_csv('output/app_test_optimized.csv')
    test_ids = test_df['SK_ID_CURR']
    X_test = test_df.drop(columns=['SK_ID_CURR'])

    # Encode categorical in test
    categorical_cols = X_test.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        for col in categorical_cols:
            le = LabelEncoder()
            X_test[col] = X_test[col].fillna('missing')
            all_cats = pd.concat([X_train[col], X_test[col]]).unique()
            le.fit(all_cats)
            X_test[col] = le.transform(X_test[col])

    for col in X_test.columns:
        if X_test[col].isnull().any():
            X_test[col] = X_test[col].fillna(X_test[col].median())

    X_test = X_test.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.fillna(X_test.median())

    # Train final model
    oof_preds, test_preds, fold_scores, overall_auc = train_with_best_params(
        X_train, y_train, X_test, test_ids, best_params, use_gpu=use_gpu
    )

    print("\n" + "=" * 70)
    print("HYPERPARAMETER TUNING COMPLETE!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  1. output/optuna_trials.csv - All optimization trials")
    print("  2. output/best_lgb_params.json - Best parameters")
    print("  3. output/optimization_history.png - Visualization")
    print("  4. output/submission/tuned_submission.csv - Submission with tuned params")
    print("\nNext: Submit tuned_submission.csv to Kaggle!")