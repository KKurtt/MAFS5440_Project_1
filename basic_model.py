import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)


def simple_feature_engineering(df):
    """
    Create new features from existing ones

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe

    Returns:
    --------
    df : pd.DataFrame
        Dataframe with engineered features
    """
    df = df.copy()

    print("\nFeature Engineering:")

    # 1. Age in years
    df['AGE_YEARS'] = -df['DAYS_BIRTH'] / 365
    print("✓ Created AGE_YEARS feature")

    # 2. Employment duration in years
    df['EMPLOYMENT_YEARS'] = -df['DAYS_EMPLOYED'] / 365
    df['EMPLOYMENT_YEARS'] = df['EMPLOYMENT_YEARS'].replace({np.inf: 0, -np.inf: 0})
    print("✓ Created EMPLOYMENT_YEARS feature")

    # 3. Income to credit ratio
    df['INCOME_CREDIT_RATIO'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    print("✓ Created INCOME_CREDIT_RATIO feature")

    # 4. Annuity to income ratio
    df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    print("✓ Created ANNUITY_INCOME_RATIO feature")

    # 5. Credit to goods price ratio
    df['CREDIT_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    print("✓ Created CREDIT_GOODS_RATIO feature")

    # 6. External source combinations
    ext_sources = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    df['EXT_SOURCE_MEAN'] = df[ext_sources].mean(axis=1)
    df['EXT_SOURCE_MAX'] = df[ext_sources].max(axis=1)
    df['EXT_SOURCE_MIN'] = df[ext_sources].min(axis=1)
    print("✓ Created external source combination features")

    # 7. Days employed anomaly flag (365243 is a known anomaly value)
    df['DAYS_EMPLOYED_ANOMALY'] = (df['DAYS_EMPLOYED'] == 365243).astype(int)
    print("✓ Created DAYS_EMPLOYED_ANOMALY feature")

    return df


def prepare_data_for_modeling(train_df, test_df, target='TARGET'):
    """
    Prepare data for machine learning models

    Parameters:
    -----------
    train_df : pd.DataFrame
        Training dataset
    test_df : pd.DataFrame
        Test dataset
    target : str
        Name of target variable

    Returns:
    --------
    tuple : (X_train, y_train, X_test)
        Processed features and target
    """

    # Separate features and target
    y_train = train_df[target]
    X_train = train_df.drop(columns=[target, 'SK_ID_CURR'])
    X_test = test_df.drop(columns=['SK_ID_CURR'])

    # Encode categorical variables
    categorical_cols = X_train.select_dtypes(include=['object']).columns

    print(f"\nEncoding {len(categorical_cols)} categorical features...")

    for col in categorical_cols:
        le = LabelEncoder()

        # Fill missing values
        X_train[col] = X_train[col].fillna('missing')
        X_test[col] = X_test[col].fillna('missing')

        # Fit on all unique categories from both train and test
        all_categories = pd.concat([X_train[col], X_test[col]]).unique()
        le.fit(all_categories)

        X_train[col] = le.transform(X_train[col])
        X_test[col] = le.transform(X_test[col])

    # Fill missing values in numeric columns
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns

    print(f"Imputing missing values in {len(numeric_cols)} numeric features...")

    for col in numeric_cols:
        median_val = X_train[col].median()
        X_train[col] = X_train[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)

    # Replace infinite values
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_test.median())

    print(f"\nFinal feature set: {X_train.shape[1]} features")

    return X_train, y_train, X_test


def train_and_evaluate_models(X_tr, y_tr, X_val, y_val):
    """
    Train multiple models and compare their performance

    Parameters:
    -----------
    X_tr, y_tr : Training data and labels
    X_val, y_val : Validation data and labels

    Returns:
    --------
    tuple : (best_model, results_dataframe)
    """

    results = {}
    models = {}

    print("\n" + "=" * 50)
    print("Model Training and Evaluation")
    print("=" * 50)

    # 1. Logistic Regression
    print("\n1. Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    lr.fit(X_tr, y_tr)
    lr_pred = lr.predict_proba(X_val)[:, 1]
    lr_auc = roc_auc_score(y_val, lr_pred)
    results['Logistic Regression'] = lr_auc
    models['Logistic Regression'] = lr
    print(f"   Validation AUC: {lr_auc:.4f}")

    # 2. Random Forest
    print("\n2. Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    rf.fit(X_tr, y_tr)
    rf_pred = rf.predict_proba(X_val)[:, 1]
    rf_auc = roc_auc_score(y_val, rf_pred)
    results['Random Forest'] = rf_auc
    models['Random Forest'] = rf
    print(f"   Validation AUC: {rf_auc:.4f}")

    # 3. LightGBM
    print("\n3. Training LightGBM...")
    lgb_train = lgb.Dataset(X_tr, y_tr)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42,
        'is_unbalance': True
    }

    lgb_model = lgb.train(
        params,
        lgb_train,
        num_boost_round=1000,
        valid_sets=[lgb_train, lgb_val],
        valid_names=['train', 'valid'],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )

    lgb_pred = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
    lgb_auc = roc_auc_score(y_val, lgb_pred)
    results['LightGBM'] = lgb_auc
    models['LightGBM'] = lgb_model
    print(f"   Validation AUC: {lgb_auc:.4f}")

    # Summary of results
    print("\n" + "=" * 50)
    print("Model Performance Comparison")
    print("=" * 50)
    results_df = pd.DataFrame(list(results.items()), columns=['Model', 'AUC'])
    results_df = results_df.sort_values('AUC', ascending=False)
    print(results_df.to_string(index=False))

    # Get best model
    best_model_name = results_df.iloc[0]['Model']
    best_model = models[best_model_name]

    print(f"\nBest Model: {best_model_name}")

    return best_model, results_df, models


def plot_feature_importance(model, feature_names, top_n=20):
    """
    Plot feature importance from trained model

    Parameters:
    -----------
    model : trained model
        LightGBM model
    feature_names : list
        List of feature names
    top_n : int
        Number of top features to display

    Returns:
    --------
    importance_df : pd.DataFrame
        DataFrame with feature importance
    """
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importance(importance_type='gain')
    })
    importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(importance_df)), importance_df['importance'], color='steelblue')
    plt.yticks(range(len(importance_df)), importance_df['feature'])
    plt.xlabel('Feature Importance (Gain)', fontsize=12)
    plt.title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('output/feature_importance.png', dpi=300, bbox_inches='tight')
    print("\nFeature importance plot saved as 'feature_importance.png'")
    plt.show()

    return importance_df


def generate_submission(model, X_test, test_ids, filename='submission.csv'):
    """
    Generate submission file for Kaggle

    Parameters:
    -----------
    model : trained model
    X_test : test features
    test_ids : test set IDs
    filename : output filename
    """
    print(f"\nGenerating predictions for test set...")

    if isinstance(model, lgb.Booster):
        predictions = model.predict(X_test, num_iteration=model.best_iteration)
    else:
        predictions = model.predict_proba(X_test)[:, 1]

    submission = pd.DataFrame({
        'SK_ID_CURR': test_ids,
        'TARGET': predictions
    })

    submission.to_csv(filename, index=False)
    print(f"Submission file saved as '{filename}'")
    print(f"Shape: {submission.shape}")
    print(f"\nPreview:")
    print(submission.head())

    return submission


if __name__ == '__main__':
    # Load main application data
    print("=" * 50)
    print("Loading Data...")
    print("=" * 50)

    app_train = pd.read_csv('DataLibrary/application_train.csv')
    app_test = pd.read_csv('DataLibrary/application_test.csv')

    app_train_fe = simple_feature_engineering(app_train)
    app_test_fe = simple_feature_engineering(app_test)

    print(f"\nNumber of features after engineering: {app_train_fe.shape[1]}")

    # Prepare data
    X_train, y_train, X_test = prepare_data_for_modeling(app_train_fe, app_test_fe)

    # Create validation set
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    print(f"\nTraining set: {X_tr.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")

    # Train models
    best_model, model_results, all_models = train_and_evaluate_models(X_tr, y_tr, X_val, y_val)

    # Plot feature importance (if best model is LightGBM)
    if isinstance(best_model, lgb.Booster):
        feature_importance = plot_feature_importance(best_model, X_train.columns, top_n=20)
        print("\nTop 20 Important Features:")
        print(feature_importance.to_string(index=False))

    # Generate submission
    test_ids = app_test['SK_ID_CURR']
    submission = generate_submission(best_model, X_test, test_ids, 'output/submission/baseline_submission.csv')
