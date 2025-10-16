import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')


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

    print("\nBasic Feature Engineering:")

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


def aggregate_bureau_data(bureau, bureau_balance):
    """
    Aggregate bureau and bureau_balance data

    Key insights to extract:
    - Number of previous credits
    - Active/closed credits ratio
    - Average/max overdue amounts
    - Credit utilization patterns
    - Payment behavior from bureau_balance
    """

    print("\nAggregating Bureau Data...")

    # ===== Bureau Table Aggregation =====
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({
        'SK_ID_BUREAU': 'count',  # Number of previous credits - 修正列名
        'DAYS_CREDIT': ['min', 'max', 'mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_SUM': ['sum', 'mean', 'max'],
        'AMT_CREDIT_SUM_DEBT': ['sum', 'mean', 'max'],
        'AMT_CREDIT_SUM_OVERDUE': ['sum', 'mean', 'max'],
        'CNT_CREDIT_PROLONG': ['sum', 'mean', 'max'],
    })

    # Flatten column names
    bureau_agg.columns = ['_'.join(col).strip() for col in bureau_agg.columns.values]
    bureau_agg.rename(columns={'SK_ID_BUREAU_count': 'BUREAU_LOAN_COUNT'}, inplace=True)

    # Calculate derived features
    bureau_agg['BUREAU_DEBT_CREDIT_RATIO'] = (
            bureau_agg['AMT_CREDIT_SUM_DEBT_sum'] / bureau_agg['AMT_CREDIT_SUM_sum']
    )

    # Active credits
    bureau_active = bureau[bureau['CREDIT_ACTIVE'] == 'Active'].groupby('SK_ID_CURR').size()
    bureau_agg['BUREAU_ACTIVE_LOANS'] = bureau_active

    # Closed credits
    bureau_closed = bureau[bureau['CREDIT_ACTIVE'] == 'Closed'].groupby('SK_ID_CURR').size()
    bureau_agg['BUREAU_CLOSED_LOANS'] = bureau_closed

    # Active/Total ratio
    bureau_agg['BUREAU_ACTIVE_RATIO'] = (
            bureau_agg['BUREAU_ACTIVE_LOANS'] / bureau_agg['BUREAU_LOAN_COUNT']
    )

    # Credit type diversity
    credit_types = bureau.groupby('SK_ID_CURR')['CREDIT_TYPE'].nunique()
    bureau_agg['BUREAU_CREDIT_TYPE_COUNT'] = credit_types

    # ===== Bureau Balance Aggregation =====
    # First, merge bureau_balance with bureau to get SK_ID_CURR
    bureau_balance = bureau_balance.merge(
        bureau[['SK_ID_CURR', 'SK_ID_BUREAU']],  # 修正列名
        on='SK_ID_BUREAU',  # 修正列名
        how='left'
    )

    # Aggregate bureau_balance
    # Convert STATUS to numeric (DPD categories)
    status_map = {'C': 0, 'X': 0, '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5}
    bureau_balance['STATUS_NUMERIC'] = bureau_balance['STATUS'].map(status_map)

    bureau_balance_agg = bureau_balance.groupby('SK_ID_CURR').agg({
        'MONTHS_BALANCE': ['min', 'max', 'size'],  # Duration of credit history
        'STATUS_NUMERIC': ['mean', 'max', 'sum'],  # Payment behavior
    })

    bureau_balance_agg.columns = ['_'.join(col).strip() for col in bureau_balance_agg.columns.values]
    bureau_balance_agg.rename(columns={'MONTHS_BALANCE_size': 'BUREAU_BALANCE_MONTHS'}, inplace=True)

    # Count of each status (proportion of different payment statuses)
    status_counts = pd.get_dummies(bureau_balance['STATUS'], prefix='BUREAU_STATUS')
    status_counts['SK_ID_CURR'] = bureau_balance['SK_ID_CURR']
    status_counts = status_counts.groupby('SK_ID_CURR').sum()

    # Calculate proportions instead of counts (more meaningful)
    status_total = status_counts.sum(axis=1)
    for col in status_counts.columns:
        status_counts[col] = status_counts[col] / status_total
    status_counts.columns = [f'{col}_RATIO' for col in status_counts.columns]

    # Merge bureau and bureau_balance aggregations
    bureau_final = bureau_agg.merge(bureau_balance_agg, left_index=True, right_index=True, how='left')
    bureau_final = bureau_final.merge(status_counts, left_index=True, right_index=True, how='left')

    # Reset index to make SK_ID_CURR a column
    bureau_final.reset_index(inplace=True)

    # Fill NaN values for clients without bureau_balance data
    bureau_final = bureau_final.fillna(0)

    print(f"✓ Created {len(bureau_final.columns) - 1} bureau features")

    return bureau_final


def aggregate_previous_applications(prev_app):
    """
    Aggregate previous application data

    Key patterns:
    - Approved/rejected ratio
    - Average loan amount changes
    - Time patterns
    """

    print("\nAggregating Previous Applications...")

    prev_agg = prev_app.groupby('SK_ID_CURR').agg({
        'SK_ID_PREV': 'count',
        'AMT_CREDIT': ['mean', 'max', 'sum', 'min'],
        'AMT_APPLICATION': ['mean', 'max', 'min'],
        'AMT_DOWN_PAYMENT': ['mean', 'max'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'max', 'sum'],
        'RATE_DOWN_PAYMENT': ['mean', 'max'],
    })

    prev_agg.columns = ['_'.join(col).strip() for col in prev_agg.columns.values]
    prev_agg.rename(columns={'SK_ID_PREV_count': 'PREV_APP_COUNT'}, inplace=True)

    # Application status aggregations
    approved = prev_app[prev_app['NAME_CONTRACT_STATUS'] == 'Approved'].groupby('SK_ID_CURR').size()
    prev_agg['PREV_APPROVED_COUNT'] = approved
    prev_agg['PREV_APPROVAL_RATE'] = prev_agg['PREV_APPROVED_COUNT'] / prev_agg['PREV_APP_COUNT']

    refused = prev_app[prev_app['NAME_CONTRACT_STATUS'] == 'Refused'].groupby('SK_ID_CURR').size()
    prev_agg['PREV_REFUSED_COUNT'] = refused
    prev_agg['PREV_REFUSAL_RATE'] = prev_agg['PREV_REFUSED_COUNT'] / prev_agg['PREV_APP_COUNT']

    canceled = prev_app[prev_app['NAME_CONTRACT_STATUS'] == 'Canceled'].groupby('SK_ID_CURR').size()
    prev_agg['PREV_CANCELED_COUNT'] = canceled

    # Product type analysis
    cash_loans = prev_app[prev_app['NAME_CONTRACT_TYPE'] == 'Cash loans'].groupby('SK_ID_CURR').size()
    prev_agg['PREV_CASH_LOAN_COUNT'] = cash_loans

    # Reset index
    prev_agg.reset_index(inplace=True)

    print(f"✓ Created {len(prev_agg.columns) - 1} previous application features")

    return prev_agg


def aggregate_installments_payments(inst_pay):
    """
    Extract payment behavior patterns

    Key metrics:
    - Payment punctuality
    - Late payment patterns
    - Payment amounts vs expected
    """

    print("\nAggregating Installments & Payments...")

    # Calculate payment difference and delay
    inst_pay['PAYMENT_DIFF'] = inst_pay['AMT_PAYMENT'] - inst_pay['AMT_INSTALMENT']
    inst_pay['PAYMENT_DELAY'] = inst_pay['DAYS_ENTRY_PAYMENT'] - inst_pay['DAYS_INSTALMENT']
    inst_pay['LATE_PAYMENT'] = (inst_pay['PAYMENT_DELAY'] > 0).astype(int)
    inst_pay['UNDERPAYMENT'] = (inst_pay['PAYMENT_DIFF'] < 0).astype(int)

    inst_agg = inst_pay.groupby('SK_ID_CURR').agg({
        'NUM_INSTALMENT_NUMBER': ['count', 'max'],
        'PAYMENT_DIFF': ['mean', 'max', 'min', 'sum', 'std'],
        'PAYMENT_DELAY': ['mean', 'max', 'min', 'std'],
        'LATE_PAYMENT': ['sum', 'mean'],
        'UNDERPAYMENT': ['sum', 'mean'],
        'AMT_PAYMENT': ['sum', 'mean', 'max', 'min'],
        'AMT_INSTALMENT': ['sum', 'mean', 'max'],
    })

    inst_agg.columns = ['_'.join(col).strip() for col in inst_agg.columns.values]
    inst_agg.rename(columns={'NUM_INSTALMENT_NUMBER_count': 'INST_COUNT'}, inplace=True)

    # Calculate payment ratio
    inst_agg['INST_PAYMENT_RATIO'] = (
            inst_agg['AMT_PAYMENT_sum'] / inst_agg['AMT_INSTALMENT_sum']
    )

    # Reset index
    inst_agg.reset_index(inplace=True)

    print(f"✓ Created {len(inst_agg.columns) - 1} installment payment features")

    return inst_agg


def aggregate_pos_cash_balance(pos):
    """
    Extract POS and cash loan balance patterns
    """

    print("\nAggregating POS Cash Balance...")

    pos_agg = pos.groupby('SK_ID_CURR').agg({
        'MONTHS_BALANCE': ['min', 'max', 'size'],
        'SK_DPD': ['mean', 'max', 'sum'],
        'SK_DPD_DEF': ['mean', 'max', 'sum'],
        'CNT_INSTALMENT': ['mean', 'max'],
        'CNT_INSTALMENT_FUTURE': ['mean', 'max', 'sum'],
    })

    pos_agg.columns = ['_'.join(col).strip() for col in pos_agg.columns.values]
    pos_agg.rename(columns={'MONTHS_BALANCE_size': 'POS_MONTHS_COUNT'}, inplace=True)

    # Contract status counts
    pos_active = pos[pos['NAME_CONTRACT_STATUS'] == 'Active'].groupby('SK_ID_CURR').size()
    pos_agg['POS_ACTIVE_COUNT'] = pos_active

    pos_completed = pos[pos['NAME_CONTRACT_STATUS'] == 'Completed'].groupby('SK_ID_CURR').size()
    pos_agg['POS_COMPLETED_COUNT'] = pos_completed

    # Calculate completion rate
    pos_agg['POS_COMPLETION_RATE'] = (
            pos_agg['POS_COMPLETED_COUNT'] / pos_agg['POS_MONTHS_COUNT']
    )

    # Reset index
    pos_agg.reset_index(inplace=True)

    print(f"✓ Created {len(pos_agg.columns) - 1} POS cash balance features")

    return pos_agg


def aggregate_credit_card_balance(cc):
    """
    Extract credit card usage patterns
    """

    print("\nAggregating Credit Card Balance...")

    cc_agg = cc.groupby('SK_ID_CURR').agg({
        'MONTHS_BALANCE': ['min', 'max', 'size'],
        'AMT_BALANCE': ['mean', 'max', 'min', 'sum'],
        'AMT_CREDIT_LIMIT_ACTUAL': ['mean', 'max', 'min'],
        'AMT_DRAWINGS_CURRENT': ['mean', 'max', 'sum'],
        'AMT_DRAWINGS_ATM_CURRENT': ['mean', 'max', 'sum'],
        'AMT_DRAWINGS_POS_CURRENT': ['mean', 'max', 'sum'],
        'AMT_PAYMENT_CURRENT': ['mean', 'max', 'sum'],
        'CNT_DRAWINGS_CURRENT': ['mean', 'max', 'sum'],
        'SK_DPD': ['mean', 'max', 'sum'],
        'SK_DPD_DEF': ['mean', 'max', 'sum'],
    })

    cc_agg.columns = ['_'.join(col).strip() for col in cc_agg.columns.values]
    cc_agg.rename(columns={'MONTHS_BALANCE_size': 'CC_MONTHS_COUNT'}, inplace=True)

    # Calculate utilization metrics
    cc_agg['CC_UTILIZATION_MEAN'] = (
            cc_agg['AMT_BALANCE_mean'] / cc_agg['AMT_CREDIT_LIMIT_ACTUAL_mean']
    )
    cc_agg['CC_UTILIZATION_MAX'] = (
            cc_agg['AMT_BALANCE_max'] / cc_agg['AMT_CREDIT_LIMIT_ACTUAL_max']
    )

    # Payment to drawings ratio
    cc_agg['CC_PAYMENT_DRAWING_RATIO'] = (
            cc_agg['AMT_PAYMENT_CURRENT_sum'] / cc_agg['AMT_DRAWINGS_CURRENT_sum']
    )

    # Reset index
    cc_agg.reset_index(inplace=True)

    print(f"✓ Created {len(cc_agg.columns) - 1} credit card features")

    return cc_agg


def create_advanced_features(df):
    """
    Create more sophisticated features
    """
    df = df.copy()

    print("\nCreating Advanced Features...")

    # 1. Polynomial features for important ratios
    if all(col in df.columns for col in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']):
        df['EXT_SOURCE_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
        df['EXT_SOURCE_WEIGHTED'] = (
                0.4 * df['EXT_SOURCE_1'].fillna(0) +
                0.4 * df['EXT_SOURCE_2'].fillna(0) +
                0.2 * df['EXT_SOURCE_3'].fillna(0)
        )
        print("✓ Created external source polynomial features")

    # 2. Interaction features
    if 'CNT_FAM_MEMBERS' in df.columns:
        df['INCOME_PER_FAMILY_MEMBER'] = df['AMT_INCOME_TOTAL'] / (df['CNT_FAM_MEMBERS'] + 1)
        print("✓ Created income per family member feature")

    df['CREDIT_INCOME_PERCENT'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']

    if 'AMT_ANNUITY' in df.columns:
        df['ANNUITY_CREDIT_PERCENT'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
        print("✓ Created annuity/credit ratio feature")

    # 3. Age-related features
    if all(col in df.columns for col in ['EMPLOYMENT_YEARS', 'AGE_YEARS']):
        df['WORKING_AGE_RATIO'] = df['EMPLOYMENT_YEARS'] / df['AGE_YEARS']
        df['INCOME_PER_AGE'] = df['AMT_INCOME_TOTAL'] / df['AGE_YEARS']
        print("✓ Created age-related interaction features")

    # 4. Document completeness
    doc_cols = [col for col in df.columns if 'FLAG_DOCUMENT' in col]
    if doc_cols:
        df['DOCUMENT_COUNT'] = df[doc_cols].sum(axis=1)
        print(f"✓ Created document count feature from {len(doc_cols)} documents")

    # 5. Contact information completeness
    contact_cols = ['FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE',
                    'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL']
    contact_cols = [col for col in contact_cols if col in df.columns]
    if contact_cols:
        df['CONTACT_INFO_COUNT'] = df[contact_cols].sum(axis=1)
        print(f"✓ Created contact info count feature")

    # 6. Regional risk features
    if all(col in df.columns for col in ['REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY']):
        df['REGION_RISK_SCORE'] = df['REGION_RATING_CLIENT'] * df['REGION_RATING_CLIENT_W_CITY']
        print("✓ Created regional risk score")

    # 7. Bureau-based features (if available)
    if 'BUREAU_LOAN_COUNT' in df.columns and 'AMT_CREDIT' in df.columns:
        df['AVG_CREDIT_PER_BUREAU_LOAN'] = df['AMT_CREDIT'] / (df['BUREAU_LOAN_COUNT'] + 1)
        print("✓ Created bureau-based features")

    # 8. Previous application features (if available)
    if 'PREV_APP_COUNT' in df.columns:
        df['APPROVAL_SUCCESS_RATE'] = df.get('PREV_APPROVAL_RATE', 0)
        print("✓ Enhanced previous application features")

    return df


def advanced_missing_value_handling(df):
    """
    Advanced missing value handling with domain knowledge
    """
    df = df.copy()

    print("\nAdvanced Missing Value Handling...")

    # 1. Create missing value indicators for important features
    important_features = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
                          'AMT_ANNUITY', 'OWN_CAR_AGE', 'OCCUPATION_TYPE']

    missing_count = 0
    for col in important_features:
        if col in df.columns and df[col].isnull().any():
            df[f'{col}_MISSING'] = df[col].isnull().astype(int)
            missing_count += 1

    if missing_count > 0:
        print(f"✓ Created {missing_count} missing value indicators")

    # 2. Domain-specific imputation
    # For car age: missing likely means no car
    if 'OWN_CAR_AGE' in df.columns:
        df['OWN_CAR_AGE'] = df['OWN_CAR_AGE'].fillna(0)
        print("✓ Imputed OWN_CAR_AGE (missing = no car)")

    # 3. Group-based imputation for building features
    building_features = [col for col in df.columns if any(
        x in col for x in ['APARTMENTS', 'YEARS_BUILD', 'ELEVATORS', 'FLOORSMAX']
    )]

    if building_features and 'REGION_RATING_CLIENT' in df.columns:
        for col in building_features:
            if df[col].isnull().any():
                df[col] = df.groupby('REGION_RATING_CLIENT')[col].transform(
                    lambda x: x.fillna(x.median())
                )
        print(f"✓ Applied region-based imputation to {len(building_features)} building features")

    return df


if __name__ == '__main__':
    # Create output directory if it doesn't exist
    import os

    os.makedirs('output', exist_ok=True)

    # Load  data
    print("=" * 70)
    print("Loading Data")
    print("=" * 70)

    app_train = pd.read_csv('DataLibrary/application_train.csv')
    app_test = pd.read_csv('DataLibrary/application_test.csv')
    bureau = pd.read_csv('DataLibrary/bureau.csv')
    bureau_balance = pd.read_csv('DataLibrary/bureau_balance.csv')
    prev_app = pd.read_csv('DataLibrary/previous_application.csv')
    pos_cash = pd.read_csv('DataLibrary/POS_CASH_balance.csv')
    credit_card = pd.read_csv('DataLibrary/credit_card_balance.csv')
    installments = pd.read_csv('DataLibrary/installments_payments.csv')

    # Apply basic feature engineering
    print("\n" + "=" * 50)
    print("Basic Feature Engineering")
    print("=" * 50)

    app_train_fe = simple_feature_engineering(app_train)
    app_test_fe = simple_feature_engineering(app_test)

    # Aggregate features from additional tables
    print("\n" + "=" * 50)
    print("Aggregating Features from Additional Tables")
    print("=" * 50)

    bureau_features = aggregate_bureau_data(bureau, bureau_balance)
    prev_features = aggregate_previous_applications(prev_app)
    inst_features = aggregate_installments_payments(installments)
    pos_features = aggregate_pos_cash_balance(pos_cash)
    cc_features = aggregate_credit_card_balance(credit_card)

    # Merge with main data
    print("\n" + "=" * 50)
    print("Merging All Features")
    print("=" * 50)

    app_train_enhanced = app_train_fe.merge(bureau_features, on='SK_ID_CURR', how='left')
    app_test_enhanced = app_test_fe.merge(bureau_features, on='SK_ID_CURR', how='left')

    app_train_enhanced = app_train_enhanced.merge(prev_features, on='SK_ID_CURR', how='left')
    app_test_enhanced = app_test_enhanced.merge(prev_features, on='SK_ID_CURR', how='left')

    app_train_enhanced = app_train_enhanced.merge(inst_features, on='SK_ID_CURR', how='left')
    app_test_enhanced = app_test_enhanced.merge(inst_features, on='SK_ID_CURR', how='left')

    app_train_enhanced = app_train_enhanced.merge(pos_features, on='SK_ID_CURR', how='left')
    app_test_enhanced = app_test_enhanced.merge(pos_features, on='SK_ID_CURR', how='left')

    app_train_enhanced = app_train_enhanced.merge(cc_features, on='SK_ID_CURR', how='left')
    app_test_enhanced = app_test_enhanced.merge(cc_features, on='SK_ID_CURR', how='left')

    # Create advanced features
    print("\n" + "=" * 50)
    print("Creating Advanced Features")
    print("=" * 50)

    app_train_enhanced = create_advanced_features(app_train_enhanced)
    app_test_enhanced = create_advanced_features(app_test_enhanced)

    # Advanced missing value handling
    app_train_enhanced = advanced_missing_value_handling(app_train_enhanced)
    app_test_enhanced = advanced_missing_value_handling(app_test_enhanced)

    # Save enhanced datasets
    train_output_path = 'output/enhanced/app_train_enhanced.csv'
    test_output_path = 'output/enhanced/app_test_enhanced.csv'

    app_train_enhanced.to_csv(train_output_path, index=False)
    app_test_enhanced.to_csv(test_output_path, index=False)

    # Save feature names
    feature_names = [col for col in app_train_enhanced.columns if col not in ['SK_ID_CURR', 'TARGET']]
    feature_df = pd.DataFrame({'feature_name': feature_names})
    feature_df.to_csv('output/enhanced/feature_names.csv', index=False)

    # Generate summary report
    print("\n" + "=" * 50)
    print("FEATURE ENGINEERING SUMMARY")
    print("=" * 50)

    print(f"\nOriginal features: {app_train.shape[1]}")
    print(f"Final features: {app_train_enhanced.shape[1]}")
    print(f"New features created: {app_train_enhanced.shape[1] - app_train.shape[1]}")

    print("\nFeature breakdown:")
    print(f"  - Basic engineered features: ~7")
    print(f"  - Bureau features: {len(bureau_features.columns) - 1}")
    print(f"  - Previous application features: {len(prev_features.columns) - 1}")
    print(f"  - Installment features: {len(inst_features.columns) - 1}")
    print(f"  - POS cash features: {len(pos_features.columns) - 1}")
    print(f"  - Credit card features: {len(cc_features.columns) - 1}")
    print(f"  - Advanced features: ~10")
    print(f"  - Missing indicators: variable")

    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Run model training script with enhanced features")
    print("2. Compare performance with baseline model")
    print("3. Analyze feature importance of new features")
