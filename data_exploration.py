import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)


def data_quality_report(df, name="Dataset"):
    """
    Generate comprehensive data quality report

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to analyze
    name : str
        Name of the dataset for display

    Returns:
    --------
    missing_table : pd.DataFrame
        Table showing missing value statistics
    """
    print(f"\n{'=' * 50}")
    print(f"{name} Data Quality Report")
    print(f"{'=' * 50}")

    # Basic information
    print(f"\nDataset dimensions: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

    # Missing value analysis
    missing = df.isnull().sum()
    missing_pct = 100 * missing / len(df)
    missing_table = pd.DataFrame({
        'Missing_Count': missing,
        'Missing_Percentage': missing_pct
    })
    missing_table = missing_table[missing_table['Missing_Count'] > 0].sort_values(
        'Missing_Percentage', ascending=False
    )

    print(f"\nMissing values summary ({len(missing_table)} features with missing data):")
    print(missing_table.head(20))

    # Data types
    print(f"\nData types distribution:")
    print(df.dtypes.value_counts())

    return missing_table


def analyze_key_features(df):
    """
    Analyze key features in the dataset

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to analyze

    Returns:
    --------
    tuple : (numeric_features, categorical_features)
        Lists of numeric and categorical feature names
    """
    print(f"\n{'=' * 50}")
    print("Key Features Analysis")
    print(f"{'=' * 50}")

    # Identify numeric features
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
    numeric_features = [col for col in numeric_features if col != 'TARGET']

    print(f"\nNumber of numeric features: {len(numeric_features)}")

    # Identify categorical features
    categorical_features = df.select_dtypes(include=['object']).columns
    print(f"Number of categorical features: {len(categorical_features)}")

    # Analyze important features
    key_features = {
        'AMT_INCOME_TOTAL': 'Total Income',
        'AMT_CREDIT': 'Credit Amount',
        'AMT_ANNUITY': 'Loan Annuity',
        'DAYS_BIRTH': 'Age in Days',
        'DAYS_EMPLOYED': 'Employment Duration in Days',
        'EXT_SOURCE_1': 'External Source 1',
        'EXT_SOURCE_2': 'External Source 2',
        'EXT_SOURCE_3': 'External Source 3'
    }

    print("\nStatistics of important features:")
    for feat, desc in key_features.items():
        if feat in df.columns:
            print(f"\n{desc} ({feat}):")
            print(df[feat].describe())

    return numeric_features, categorical_features


def visualize_target_distribution(df):
    """
    Create comprehensive visualizations for target variable analysis

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset containing target variable and features
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Target variable distribution
    ax = axes[0, 0]
    df['TARGET'].value_counts().plot(kind='bar', ax=ax, color=['green', 'red'])
    ax.set_title('Target Variable Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Default Status (0=No Default, 1=Default)')
    ax.set_ylabel('Count')
    ax.set_xticklabels(['No Default', 'Default'], rotation=0)

    # 2. Income distribution by target
    ax = axes[0, 1]
    df[df['AMT_INCOME_TOTAL'] < 1000000].boxplot(
        column='AMT_INCOME_TOTAL', by='TARGET', ax=ax
    )
    ax.set_title('Income Distribution by Target')
    ax.set_xlabel('Default Status')
    ax.set_ylabel('Income Amount')
    plt.sca(ax)
    plt.xticks([1, 2], ['No Default', 'Default'])

    # 3. Credit amount distribution
    ax = axes[0, 2]
    df.boxplot(column='AMT_CREDIT', by='TARGET', ax=ax)
    ax.set_title('Credit Amount Distribution by Target')
    ax.set_xlabel('Default Status')
    ax.set_ylabel('Credit Amount')
    plt.sca(ax)
    plt.xticks([1, 2], ['No Default', 'Default'])

    # 4. Age distribution
    ax = axes[1, 0]
    df['AGE_YEARS'] = -df['DAYS_BIRTH'] / 365
    df.boxplot(column='AGE_YEARS', by='TARGET', ax=ax)
    ax.set_title('Age Distribution by Target')
    ax.set_xlabel('Default Status')
    ax.set_ylabel('Age (Years)')
    plt.sca(ax)
    plt.xticks([1, 2], ['No Default', 'Default'])

    # 5. External sources comparison
    ax = axes[1, 1]
    ext_data = []
    labels = []
    for source in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']:
        if source in df.columns:
            for target in [0, 1]:
                data = df[df['TARGET'] == target][source].dropna()
                ext_data.append(data)
                labels.append(f'{source[-1]}_T{target}')

    ax.boxplot(ext_data, labels=labels)
    ax.set_title('External Sources by Target')
    ax.set_xlabel('Source_Target')
    ax.set_ylabel('Score')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # 6. Correlation heatmap
    ax = axes[1, 2]
    corr_features = ['TARGET', 'AMT_INCOME_TOTAL', 'AMT_CREDIT',
                     'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    corr_features = [f for f in corr_features if f in df.columns]
    corr = df[corr_features].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', ax=ax, cmap='coolwarm', center=0)
    ax.set_title('Feature Correlation Matrix')

    plt.tight_layout()
    plt.savefig('output/eda_visualizations.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'output/eda_visualizations.png'")
    plt.show()


if __name__ == '__main__':
    # Load main application data
    print("=" * 50)
    print("Loading Data...")
    print("=" * 50)

    app_train = pd.read_csv('DataLibrary/application_train.csv')
    app_test = pd.read_csv('DataLibrary/application_test.csv')

    print(f"\nTraining set shape: {app_train.shape}")
    print(f"Test set shape: {app_test.shape}")
    print(f"\nTarget variable distribution:")
    print(app_train['TARGET'].value_counts())
    print(f"\nDefault rate: {app_train['TARGET'].mean():.2%}")

    # Execute data quality check
    missing_report = data_quality_report(app_train, "Training Set")

    # Target Variable and Key Features Analysis
    numeric_cols, categorical_cols = analyze_key_features(app_train)

    # Execute visualization
    visualize_target_distribution(app_train)
