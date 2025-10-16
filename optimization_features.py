# optimization_features.py

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')


def create_time_series_features(bureau, bureau_balance):
    """
    Create time-series based features from bureau and bureau_balance data
    Extract trends and patterns over time
    """
    print("\nCreating Time Series Features from Bureau...")

    # Sort by date
    bureau = bureau.sort_values(['SK_ID_CURR', 'DAYS_CREDIT'])

    # ===== Bureau Time Series Features =====

    # Recent vs old credit behavior
    bureau['CREDIT_RECENCY'] = bureau.groupby('SK_ID_CURR')['DAYS_CREDIT'].rank(pct=True)

    # Recent credits (last 2 years)
    recent_credits = bureau[bureau['DAYS_CREDIT'] >= -730].groupby('SK_ID_CURR').agg({
        'AMT_CREDIT_SUM': ['sum', 'mean', 'max'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'SK_ID_BUREAU': 'count'
    })
    recent_credits.columns = ['RECENT_' + '_'.join(col).strip() for col in recent_credits.columns.values]
    recent_credits.rename(columns={'RECENT_SK_ID_BUREAU_count': 'RECENT_CREDIT_COUNT'}, inplace=True)

    # Old credits (more than 2 years ago)
    old_credits = bureau[bureau['DAYS_CREDIT'] < -730].groupby('SK_ID_CURR').agg({
        'AMT_CREDIT_SUM': ['sum', 'mean'],
        'SK_ID_BUREAU': 'count'
    })
    old_credits.columns = ['OLD_' + '_'.join(col).strip() for col in old_credits.columns.values]
    old_credits.rename(columns={'OLD_SK_ID_BUREAU_count': 'OLD_CREDIT_COUNT'}, inplace=True)

    # Credit growth trend
    bureau_trend = bureau.groupby('SK_ID_CURR').agg({
        'AMT_CREDIT_SUM': lambda x: x.iloc[-1] - x.iloc[0] if len(x) > 1 else 0,
        'CREDIT_DAY_OVERDUE': lambda x: x.iloc[-1] - x.iloc[0] if len(x) > 1 else 0
    })
    bureau_trend.columns = ['CREDIT_AMOUNT_TREND', 'OVERDUE_TREND']

    # ===== Bureau Balance Time Series Features =====

    # Merge bureau_balance with bureau to get SK_ID_CURR
    bureau_balance = bureau_balance.merge(
        bureau[['SK_ID_CURR', 'SK_ID_BUREAU']],
        on='SK_ID_BUREAU',
        how='left'
    )

    # Sort by time
    bureau_balance = bureau_balance.sort_values(['SK_ID_CURR', 'SK_ID_BUREAU', 'MONTHS_BALANCE'])

    # Convert STATUS to numeric
    status_map = {'C': 0, 'X': 0, '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5}
    bureau_balance['STATUS_NUM'] = bureau_balance['STATUS'].map(status_map).fillna(0)

    # Recent payment status (last 6 months)
    recent_balance = bureau_balance[bureau_balance['MONTHS_BALANCE'] >= -6].groupby('SK_ID_CURR').agg({
        'STATUS_NUM': ['mean', 'max', 'sum'],
        'MONTHS_BALANCE': 'count'
    })
    recent_balance.columns = ['RECENT_STATUS_' + '_'.join(col).strip() for col in recent_balance.columns.values]
    recent_balance.rename(columns={'RECENT_STATUS_MONTHS_BALANCE_count': 'RECENT_MONTHS_COUNT'}, inplace=True)

    # Old payment status (more than 12 months ago)
    old_balance = bureau_balance[bureau_balance['MONTHS_BALANCE'] < -12].groupby('SK_ID_CURR').agg({
        'STATUS_NUM': ['mean', 'max'],
        'MONTHS_BALANCE': 'count'
    })
    old_balance.columns = ['OLD_STATUS_' + '_'.join(col).strip() for col in old_balance.columns.values]
    old_balance.rename(columns={'OLD_STATUS_MONTHS_BALANCE_count': 'OLD_MONTHS_COUNT'}, inplace=True)

    # Payment behavior trend (getting better or worse)
    def status_trend(group):
        if len(group) < 6:
            return 0
        # Sort by time
        group = group.sort_values('MONTHS_BALANCE')
        # Compare recent vs old average status
        recent_avg = group.tail(6)['STATUS_NUM'].mean()
        old_avg = group.head(6)['STATUS_NUM'].mean()
        return recent_avg - old_avg  # Positive = getting worse

    balance_trend = bureau_balance.groupby('SK_ID_CURR').apply(status_trend).reset_index()
    balance_trend.columns = ['SK_ID_CURR', 'PAYMENT_STATUS_TREND']

    # Count consecutive months with good status (C or X)
    def count_consecutive_good_months(group):
        group = group.sort_values('MONTHS_BALANCE', ascending=False)
        count = 0
        for status in group['STATUS']:
            if status in ['C', 'X', '0']:
                count += 1
            else:
                break
        return count

    consecutive_good = bureau_balance.groupby('SK_ID_CURR').apply(count_consecutive_good_months).reset_index()
    consecutive_good.columns = ['SK_ID_CURR', 'CONSECUTIVE_GOOD_MONTHS']

    # ===== Merge All Time Series Features =====

    bureau_ts = recent_credits.merge(old_credits, left_index=True, right_index=True, how='outer')
    bureau_ts = bureau_ts.merge(bureau_trend, left_index=True, right_index=True, how='outer')
    bureau_ts = bureau_ts.merge(recent_balance, left_index=True, right_index=True, how='outer')
    bureau_ts = bureau_ts.merge(old_balance, left_index=True, right_index=True, how='outer')

    bureau_ts.reset_index(inplace=True)

    # Merge trend features
    bureau_ts = bureau_ts.merge(balance_trend, on='SK_ID_CURR', how='left')
    bureau_ts = bureau_ts.merge(consecutive_good, on='SK_ID_CURR', how='left')

    # Calculate additional ratios
    bureau_ts['RECENT_TO_OLD_CREDIT_RATIO'] = (
            bureau_ts['RECENT_CREDIT_COUNT'] / (bureau_ts['OLD_CREDIT_COUNT'] + 1)
    )

    bureau_ts['STATUS_IMPROVEMENT'] = (
            bureau_ts['OLD_STATUS_STATUS_NUM_mean'] - bureau_ts['RECENT_STATUS_STATUS_NUM_mean']
    )  # Positive = improving

    bureau_ts = bureau_ts.fillna(0)

    print(f"✓ Created {len(bureau_ts.columns) - 1} time series features")

    return bureau_ts


def create_payment_behavior_features(inst_pay):
    """
    Advanced payment behavior patterns
    Focus on consistency and trends
    """
    print("\nCreating Advanced Payment Behavior Features...")

    inst_pay = inst_pay.sort_values(['SK_ID_CURR', 'DAYS_INSTALMENT'])

    # Payment consistency
    inst_pay['PAYMENT_DIFF'] = inst_pay['AMT_PAYMENT'] - inst_pay['AMT_INSTALMENT']
    inst_pay['PAYMENT_DELAY'] = inst_pay['DAYS_ENTRY_PAYMENT'] - inst_pay['DAYS_INSTALMENT']
    inst_pay['LATE_PAYMENT'] = (inst_pay['PAYMENT_DELAY'] > 0).astype(int)
    inst_pay['UNDERPAYMENT'] = (inst_pay['PAYMENT_DIFF'] < 0).astype(int)

    # Recent payment behavior (last 12 months)
    recent_payments = inst_pay[inst_pay['DAYS_INSTALMENT'] >= -365].groupby('SK_ID_CURR').agg({
        'PAYMENT_DIFF': ['mean', 'std', 'min', 'max'],
        'PAYMENT_DELAY': ['mean', 'max', 'sum'],
        'LATE_PAYMENT': ['sum', 'mean'],
        'UNDERPAYMENT': ['sum', 'mean'],
        'NUM_INSTALMENT_NUMBER': 'count'
    })
    recent_payments.columns = ['RECENT_PAY_' + '_'.join(col).strip() for col in recent_payments.columns.values]

    # Old payment behavior (more than 12 months ago)
    old_payments = inst_pay[inst_pay['DAYS_INSTALMENT'] < -365].groupby('SK_ID_CURR').agg({
        'PAYMENT_DELAY': ['mean', 'max'],
        'LATE_PAYMENT': 'mean',
        'NUM_INSTALMENT_NUMBER': 'count'
    })
    old_payments.columns = ['OLD_PAY_' + '_'.join(col).strip() for col in old_payments.columns.values]

    # Payment trend (improving or deteriorating)
    payment_groups = inst_pay.groupby('SK_ID_CURR')

    def payment_delay_trend(group):
        if len(group) < 6:
            return 0
        mid = len(group) // 2
        first_half_delay = group.iloc[:mid]['PAYMENT_DELAY'].mean()
        second_half_delay = group.iloc[mid:]['PAYMENT_DELAY'].mean()
        return second_half_delay - first_half_delay  # Positive = getting worse

    delay_trend = payment_groups.apply(payment_delay_trend).reset_index()
    delay_trend.columns = ['SK_ID_CURR', 'PAYMENT_DELAY_TREND']

    # Payment consistency (standard deviation of delays)
    def payment_consistency(group):
        return group['PAYMENT_DELAY'].std()

    consistency = payment_groups.apply(payment_consistency).reset_index()
    consistency.columns = ['SK_ID_CURR', 'PAYMENT_DELAY_STD']

    # Last payment behavior (most recent 3 payments)
    def last_payment_quality(group):
        if len(group) < 3:
            return 0
        last_3 = group.nlargest(3, 'DAYS_INSTALMENT')
        return last_3['LATE_PAYMENT'].mean()

    last_payments = payment_groups.apply(last_payment_quality).reset_index()
    last_payments.columns = ['SK_ID_CURR', 'LAST_3_PAYMENTS_LATE_RATE']

    # Merge all
    payment_features = recent_payments.reset_index()
    payment_features = payment_features.merge(old_payments.reset_index(), on='SK_ID_CURR', how='left')
    payment_features = payment_features.merge(delay_trend, on='SK_ID_CURR', how='left')
    payment_features = payment_features.merge(consistency, on='SK_ID_CURR', how='left')
    payment_features = payment_features.merge(last_payments, on='SK_ID_CURR', how='left')

    # Calculate improvement metrics
    payment_features['PAYMENT_IMPROVEMENT'] = (
            payment_features['OLD_PAY_PAYMENT_DELAY_mean'] -
            payment_features['RECENT_PAY_PAYMENT_DELAY_mean']
    )  # Positive = improving

    payment_features = payment_features.fillna(0)

    print(f"✓ Created {len(payment_features.columns) - 1} payment behavior features")

    return payment_features


def create_credit_card_velocity_features(cc):
    """
    Credit card usage velocity and patterns
    """
    print("\nCreating Credit Card Velocity Features...")

    cc = cc.sort_values(['SK_ID_CURR', 'MONTHS_BALANCE'])

    # Calculate utilization
    cc['UTILIZATION'] = cc['AMT_BALANCE'] / (cc['AMT_CREDIT_LIMIT_ACTUAL'] + 1)

    # Recent activity (last 6 months)
    recent_cc = cc[cc['MONTHS_BALANCE'] >= -6].groupby('SK_ID_CURR').agg({
        'AMT_DRAWINGS_CURRENT': ['sum', 'mean', 'max'],
        'AMT_PAYMENT_CURRENT': ['sum', 'mean', 'max'],
        'AMT_BALANCE': ['mean', 'max'],
        'UTILIZATION': ['mean', 'max'],
        'SK_DPD': ['max', 'sum', 'mean'],
        'SK_DPD_DEF': ['max', 'sum'],
        'MONTHS_BALANCE': 'count'
    })
    recent_cc.columns = ['CC_RECENT_' + '_'.join(col).strip() for col in recent_cc.columns.values]

    # Old activity (6-12 months ago)
    old_cc = cc[(cc['MONTHS_BALANCE'] >= -12) & (cc['MONTHS_BALANCE'] < -6)].groupby('SK_ID_CURR').agg({
        'AMT_DRAWINGS_CURRENT': ['mean', 'max'],
        'AMT_PAYMENT_CURRENT': 'mean',
        'UTILIZATION': 'mean',
        'SK_DPD': 'mean'
    })
    old_cc.columns = ['CC_OLD_' + '_'.join(col).strip() for col in old_cc.columns.values]

    # Spending acceleration
    cc_groups = cc.groupby('SK_ID_CURR')

    def spending_trend(group):
        if len(group) < 3:
            return 0
        # Compare recent vs older spending
        recent = group.nlargest(3, 'MONTHS_BALANCE')['AMT_DRAWINGS_CURRENT'].mean()
        older = group.nsmallest(3, 'MONTHS_BALANCE')['AMT_DRAWINGS_CURRENT'].mean()
        return (recent - older) / (older + 1)  # Percentage change

    spending_trend_df = cc_groups.apply(spending_trend).reset_index()
    spending_trend_df.columns = ['SK_ID_CURR', 'CC_SPENDING_TREND']

    # Balance utilization trend
    def utilization_trend(group):
        if len(group) < 3:
            return 0
        group = group.sort_values('MONTHS_BALANCE')
        recent_util = group.iloc[-3:]['UTILIZATION'].mean()
        older_util = group.iloc[:3]['UTILIZATION'].mean()
        return recent_util - older_util

    utilization_trend_df = cc_groups.apply(utilization_trend).reset_index()
    utilization_trend_df.columns = ['SK_ID_CURR', 'CC_UTILIZATION_TREND']

    # Payment to drawings ratio trend
    def payment_drawing_trend(group):
        if len(group) < 3:
            return 1
        group = group.sort_values('MONTHS_BALANCE')
        recent_ratio = (group.iloc[-3:]['AMT_PAYMENT_CURRENT'].sum() /
                        (group.iloc[-3:]['AMT_DRAWINGS_CURRENT'].sum() + 1))
        older_ratio = (group.iloc[:3]['AMT_PAYMENT_CURRENT'].sum() /
                       (group.iloc[:3]['AMT_DRAWINGS_CURRENT'].sum() + 1))
        return recent_ratio - older_ratio

    payment_draw_trend_df = cc_groups.apply(payment_drawing_trend).reset_index()
    payment_draw_trend_df.columns = ['SK_ID_CURR', 'CC_PAYMENT_DRAWING_TREND']

    # DPD trend (delinquency getting worse or better)
    def dpd_trend(group):
        if len(group) < 3:
            return 0
        group = group.sort_values('MONTHS_BALANCE')
        recent_dpd = group.iloc[-3:]['SK_DPD'].mean()
        older_dpd = group.iloc[:3]['SK_DPD'].mean()
        return recent_dpd - older_dpd

    dpd_trend_df = cc_groups.apply(dpd_trend).reset_index()
    dpd_trend_df.columns = ['SK_ID_CURR', 'CC_DPD_TREND']

    # Merge all
    cc_features = recent_cc.reset_index()
    cc_features = cc_features.merge(old_cc.reset_index(), on='SK_ID_CURR', how='left')
    cc_features = cc_features.merge(spending_trend_df, on='SK_ID_CURR', how='left')
    cc_features = cc_features.merge(utilization_trend_df, on='SK_ID_CURR', how='left')
    cc_features = cc_features.merge(payment_draw_trend_df, on='SK_ID_CURR', how='left')
    cc_features = cc_features.merge(dpd_trend_df, on='SK_ID_CURR', how='left')

    # Calculate improvement metrics
    cc_features['CC_UTILIZATION_IMPROVEMENT'] = (
            cc_features['CC_OLD_UTILIZATION_mean'] - cc_features['CC_RECENT_UTILIZATION_mean']
    )  # Positive = improving

    cc_features['CC_SPENDING_ACCELERATION'] = (
            cc_features['CC_RECENT_AMT_DRAWINGS_CURRENT_mean'] -
            cc_features['CC_OLD_AMT_DRAWINGS_CURRENT_mean']
    )

    cc_features = cc_features.fillna(0)

    print(f"✓ Created {len(cc_features.columns) - 1} credit card velocity features")

    return cc_features


def create_application_history_features(prev_app):
    """
    Previous application patterns and changes over time
    """
    print("\nCreating Application History Features...")

    prev_app = prev_app.sort_values(['SK_ID_CURR', 'DAYS_DECISION'])

    # Application frequency
    prev_groups = prev_app.groupby('SK_ID_CURR')

    # Time between applications
    def avg_time_between_apps(group):
        if len(group) < 2:
            return 0
        time_diffs = group['DAYS_DECISION'].diff().abs()
        return time_diffs.mean()

    time_between = prev_groups.apply(avg_time_between_apps).reset_index()
    time_between.columns = ['SK_ID_CURR', 'AVG_TIME_BETWEEN_APPS']

    # Loan amount changes
    def loan_amount_trend(group):
        if len(group) < 2:
            return 0
        return (group.iloc[-1]['AMT_CREDIT'] - group.iloc[0]['AMT_CREDIT']) / (group.iloc[0]['AMT_CREDIT'] + 1)

    amount_trend = prev_groups.apply(loan_amount_trend).reset_index()
    amount_trend.columns = ['SK_ID_CURR', 'PREV_LOAN_AMOUNT_TREND']

    # Recent application success rate (last 3 applications)
    def recent_approval_rate(group):
        if len(group) < 1:
            return 0
        recent = group.nlargest(min(3, len(group)), 'DAYS_DECISION')
        approved = (recent['NAME_CONTRACT_STATUS'] == 'Approved').sum()
        return approved / len(recent)

    recent_approval = prev_groups.apply(recent_approval_rate).reset_index()
    recent_approval.columns = ['SK_ID_CURR', 'RECENT_APPROVAL_RATE']

    # Old application success rate (excluding last 3)
    def old_approval_rate(group):
        if len(group) <= 3:
            return 0
        old = group.nsmallest(len(group) - 3, 'DAYS_DECISION')
        if len(old) == 0:
            return 0
        approved = (old['NAME_CONTRACT_STATUS'] == 'Approved').sum()
        return approved / len(old)

    old_approval = prev_groups.apply(old_approval_rate).reset_index()
    old_approval.columns = ['SK_ID_CURR', 'OLD_APPROVAL_RATE']

    # Application amount changes
    def amount_volatility(group):
        if len(group) < 2:
            return 0
        return group['AMT_APPLICATION'].std() / (group['AMT_APPLICATION'].mean() + 1)

    volatility = prev_groups.apply(amount_volatility).reset_index()
    volatility.columns = ['SK_ID_CURR', 'PREV_AMOUNT_VOLATILITY']

    # Down payment trend
    def down_payment_trend(group):
        if len(group) < 2:
            return 0
        recent_dp = group.nlargest(3, 'DAYS_DECISION')['AMT_DOWN_PAYMENT'].mean()
        old_dp = group.nsmallest(3, 'DAYS_DECISION')['AMT_DOWN_PAYMENT'].mean()
        return (recent_dp - old_dp) / (old_dp + 1)

    dp_trend = prev_groups.apply(down_payment_trend).reset_index()
    dp_trend.columns = ['SK_ID_CURR', 'DOWN_PAYMENT_TREND']

    # Merge all
    app_history = time_between.merge(amount_trend, on='SK_ID_CURR', how='outer')
    app_history = app_history.merge(recent_approval, on='SK_ID_CURR', how='outer')
    app_history = app_history.merge(old_approval, on='SK_ID_CURR', how='outer')
    app_history = app_history.merge(volatility, on='SK_ID_CURR', how='outer')
    app_history = app_history.merge(dp_trend, on='SK_ID_CURR', how='outer')

    # Calculate improvement metrics
    app_history['APPROVAL_RATE_IMPROVEMENT'] = (
            app_history['RECENT_APPROVAL_RATE'] - app_history['OLD_APPROVAL_RATE']
    )  # Positive = improving

    app_history = app_history.fillna(0)

    print(f"✓ Created {len(app_history.columns) - 1} application history features")

    return app_history


def create_pos_cash_trend_features(pos):
    """
    POS cash balance trends over time
    """
    print("\nCreating POS Cash Trend Features...")

    pos = pos.sort_values(['SK_ID_CURR', 'MONTHS_BALANCE'])

    # Recent behavior (last 6 months)
    recent_pos = pos[pos['MONTHS_BALANCE'] >= -6].groupby('SK_ID_CURR').agg({
        'SK_DPD': ['mean', 'max', 'sum'],
        'SK_DPD_DEF': ['mean', 'max'],
        'CNT_INSTALMENT_FUTURE': 'mean'
    })
    recent_pos.columns = ['POS_RECENT_' + '_'.join(col).strip() for col in recent_pos.columns.values]

    # Old behavior
    old_pos = pos[pos['MONTHS_BALANCE'] < -6].groupby('SK_ID_CURR').agg({
        'SK_DPD': ['mean', 'max'],
        'SK_DPD_DEF': 'mean'
    })
    old_pos.columns = ['POS_OLD_' + '_'.join(col).strip() for col in old_pos.columns.values]

    # DPD trend
    pos_groups = pos.groupby('SK_ID_CURR')

    def dpd_trend(group):
        if len(group) < 3:
            return 0
        group = group.sort_values('MONTHS_BALANCE')
        recent_dpd = group.iloc[-3:]['SK_DPD'].mean()
        old_dpd = group.iloc[:3]['SK_DPD'].mean()
        return recent_dpd - old_dpd

    dpd_trend_df = pos_groups.apply(dpd_trend).reset_index()
    dpd_trend_df.columns = ['SK_ID_CURR', 'POS_DPD_TREND']

    # Merge
    pos_features = recent_pos.reset_index()
    pos_features = pos_features.merge(old_pos.reset_index(), on='SK_ID_CURR', how='left')
    pos_features = pos_features.merge(dpd_trend_df, on='SK_ID_CURR', how='left')

    # Improvement metric
    pos_features['POS_DPD_IMPROVEMENT'] = (
            pos_features['POS_OLD_SK_DPD_mean'] - pos_features['POS_RECENT_SK_DPD_mean']
    )

    pos_features = pos_features.fillna(0)

    print(f"✓ Created {len(pos_features.columns) - 1} POS cash trend features")

    return pos_features


def create_cross_table_features(app_train, bureau_features, prev_features, inst_features):
    """
    Create features by combining information across tables
    """
    print("\nCreating Cross-Table Features...")

    cross_features = app_train[['SK_ID_CURR', 'AMT_INCOME_TOTAL', 'AMT_CREDIT']].copy()

    # Merge aggregated features
    cross_features = cross_features.merge(
        bureau_features[['SK_ID_CURR', 'BUREAU_LOAN_COUNT', 'BUREAU_ACTIVE_LOANS']],
        on='SK_ID_CURR', how='left'
    )

    cross_features = cross_features.merge(
        prev_features[['SK_ID_CURR', 'PREV_APP_COUNT', 'PREV_APPROVED_COUNT']],
        on='SK_ID_CURR', how='left'
    )

    cross_features = cross_features.merge(
        inst_features[['SK_ID_CURR', 'INST_COUNT', 'LATE_PAYMENT_sum']],
        on='SK_ID_CURR', how='left'
    )

    cross_features = cross_features.fillna(0)

    # Create interaction features
    cross_features['TOTAL_CREDIT_EXPERIENCE'] = (
            cross_features['BUREAU_LOAN_COUNT'] + cross_features['PREV_APP_COUNT']
    )

    cross_features['BUREAU_TO_PREV_RATIO'] = (
            cross_features['BUREAU_LOAN_COUNT'] / (cross_features['PREV_APP_COUNT'] + 1)
    )

    cross_features['LATE_PAYMENT_RATIO'] = (
            cross_features['LATE_PAYMENT_sum'] / (cross_features['INST_COUNT'] + 1)
    )

    cross_features['CREDIT_ACTIVITY_SCORE'] = (
            cross_features['PREV_APPROVED_COUNT'] + cross_features['BUREAU_ACTIVE_LOANS']
    )

    cross_features['CREDIT_PER_INCOME'] = (
            cross_features['AMT_CREDIT'] / cross_features['AMT_INCOME_TOTAL']
    )

    cross_features['DEBT_PER_EXPERIENCE'] = (
            cross_features['AMT_CREDIT'] / (cross_features['TOTAL_CREDIT_EXPERIENCE'] + 1)
    )

    # Keep only new features
    cross_features = cross_features[['SK_ID_CURR', 'TOTAL_CREDIT_EXPERIENCE',
                                     'BUREAU_TO_PREV_RATIO', 'LATE_PAYMENT_RATIO',
                                     'CREDIT_ACTIVITY_SCORE', 'CREDIT_PER_INCOME',
                                     'DEBT_PER_EXPERIENCE']]

    print(f"✓ Created {len(cross_features.columns) - 1} cross-table features")

    return cross_features


def reduce_ext_source_dominance(df):
    """
    Create alternative features to reduce EXT_SOURCE_MEAN dominance
    """
    print("\nCreating Alternative High-Value Features...")

    df = df.copy()

    # Weighted combinations with other important features
    if all(col in df.columns for col in ['EXT_SOURCE_MEAN', 'AGE_YEARS', 'ANNUITY_INCOME_RATIO']):
        df['RISK_SCORE_1'] = (
                0.5 * df['EXT_SOURCE_MEAN'].fillna(0.5) +
                0.3 * (df['AGE_YEARS'] / 100) +
                0.2 * (1 - df['ANNUITY_INCOME_RATIO'].clip(0, 1).fillna(0.2))
        )

    if all(col in df.columns for col in ['EXT_SOURCE_MEAN', 'BUREAU_DEBT_CREDIT_RATIO', 'LATE_PAYMENT_mean']):
        df['RISK_SCORE_2'] = (
                0.4 * df['EXT_SOURCE_MEAN'].fillna(0.5) +
                0.3 * (1 - df['BUREAU_DEBT_CREDIT_RATIO'].clip(0, 1).fillna(0.3)) +
                0.3 * (1 - df['LATE_PAYMENT_mean'].clip(0, 1).fillna(0))
        )

    # Non-linear transformations of EXT_SOURCE
    for col in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']:
        if col in df.columns:
            df[f'{col}_SQUARED'] = df[col].fillna(0) ** 2
            df[f'{col}_CUBED'] = df[col].fillna(0) ** 3
            df[f'{col}_SQRT'] = np.sqrt(df[col].fillna(0))
            df[f'{col}_LOG'] = np.log1p(df[col].fillna(0))

    # Interaction between EXT_SOURCE and other key features
    if all(col in df.columns for col in ['EXT_SOURCE_MEAN', 'AMT_INCOME_TOTAL']):
        df['EXT_SOURCE_INCOME_INTERACTION'] = (
                df['EXT_SOURCE_MEAN'].fillna(0.5) * np.log1p(df['AMT_INCOME_TOTAL'])
        )

    if all(col in df.columns for col in ['EXT_SOURCE_MEAN', 'AMT_CREDIT']):
        df['EXT_SOURCE_CREDIT_INTERACTION'] = (
                df['EXT_SOURCE_MEAN'].fillna(0.5) * np.log1p(df['AMT_CREDIT'])
        )

    print(f"✓ Created alternative scoring features to balance EXT_SOURCE dominance")

    return df


if __name__ == '__main__':
    import os

    os.makedirs('output', exist_ok=True)

    print("=" * 70)
    print("OPTIMIZATION FEATURE ENGINEERING")
    print("=" * 70)

    # Load existing enhanced data
    print("\nLoading existing enhanced data...")
    app_train = pd.read_csv('output/enhanced/app_train_enhanced.csv')
    app_test = pd.read_csv('output/enhanced/app_test_enhanced.csv')

    print(f"Current train shape: {app_train.shape}")
    print(f"Current test shape: {app_test.shape}")

    # Load raw additional tables
    print("\n" + "=" * 70)
    print("Loading Raw Additional Tables")
    print("=" * 70)

    bureau = pd.read_csv('DataLibrary/bureau.csv')
    print(f"✓ Loaded bureau: {bureau.shape}")

    bureau_balance = pd.read_csv('DataLibrary/bureau_balance.csv')
    print(f"✓ Loaded bureau_balance: {bureau_balance.shape}")

    prev_app = pd.read_csv('DataLibrary/previous_application.csv')
    print(f"✓ Loaded previous_application: {prev_app.shape}")

    installments = pd.read_csv('DataLibrary/installments_payments.csv')
    print(f"✓ Loaded installments: {installments.shape}")

    credit_card = pd.read_csv('DataLibrary/credit_card_balance.csv')
    print(f"✓ Loaded credit_card: {credit_card.shape}")

    pos_cash = pd.read_csv('DataLibrary/POS_CASH_balance.csv')
    print(f"✓ Loaded POS_CASH: {pos_cash.shape}")

    # Create new optimization features
    print("\n" + "=" * 70)
    print("Creating Optimization Features")
    print("=" * 70)

    bureau_ts = create_time_series_features(bureau, bureau_balance)
    payment_features = create_payment_behavior_features(installments)
    cc_velocity = create_credit_card_velocity_features(credit_card)
    app_history = create_application_history_features(prev_app)
    pos_trends = create_pos_cash_trend_features(pos_cash)

    # For cross-table features, we need some aggregated data
    # Load from enhanced data (assuming they exist)
    bureau_agg = app_train[['SK_ID_CURR', 'BUREAU_LOAN_COUNT', 'BUREAU_ACTIVE_LOANS']].copy()
    prev_agg = app_train[['SK_ID_CURR', 'PREV_APP_COUNT', 'PREV_APPROVED_COUNT']].copy()
    inst_agg = app_train[['SK_ID_CURR', 'INST_COUNT', 'LATE_PAYMENT_sum']].copy()

    cross_features_train = create_cross_table_features(app_train, bureau_agg, prev_agg, inst_agg)
    cross_features_test = create_cross_table_features(app_test, bureau_agg, prev_agg, inst_agg)

    # Merge all new features with existing data
    print("\n" + "=" * 70)
    print("Merging Optimization Features")
    print("=" * 70)

    feature_list = [
        (bureau_ts, 'bureau time series'),
        (payment_features, 'payment behavior'),
        (cc_velocity, 'credit card velocity'),
        (app_history, 'application history'),
        (pos_trends, 'POS cash trends')
    ]

    for new_features, name in feature_list:
        print(f"\nMerging {name} features...")
        app_train = app_train.merge(new_features, on='SK_ID_CURR', how='left')
        app_test = app_test.merge(new_features, on='SK_ID_CURR', how='left')
        print(f"Shape after merge: {app_train.shape}")

    # Merge cross-table features
    print(f"\nMerging cross-table features...")
    app_train = app_train.merge(cross_features_train, on='SK_ID_CURR', how='left')
    app_test = app_test.merge(cross_features_test, on='SK_ID_CURR', how='left')
    print(f"Shape after merge: {app_train.shape}")

    # Create alternative features to reduce EXT_SOURCE dominance
    print("\n" + "=" * 70)
    print("Balancing Feature Importance")
    print("=" * 70)

    app_train = reduce_ext_source_dominance(app_train)
    app_test = reduce_ext_source_dominance(app_test)

    # Fill NaN and replace inf
    app_train = app_train.replace([np.inf, -np.inf], np.nan)
    app_test = app_test.replace([np.inf, -np.inf], np.nan)

    app_train = app_train.fillna(0)
    app_test = app_test.fillna(0)

    # Save optimized datasets
    print("\n" + "=" * 70)
    print("Saving Optimized Datasets")
    print("=" * 70)

    app_train.to_csv('output/enhanced/app_train_optimized.csv', index=False)
    app_test.to_csv('output/enhanced/app_test_optimized.csv', index=False)

    print(f"\n✓ Optimized train shape: {app_train.shape}")
    print(f"✓ Optimized test shape: {app_test.shape}")
    print(f"✓ Features increased from 284 to {app_train.shape[1]}")
    print(f"✓ New features added: {app_train.shape[1] - 284}")

    # Summary
    print("\n" + "=" * 70)
    print("OPTIMIZATION SUMMARY")
    print("=" * 70)
    print(f"\nFeature additions:")
    print(f"  - Bureau time series: {len(bureau_ts.columns) - 1}")
    print(f"  - Payment behavior: {len(payment_features.columns) - 1}")
    print(f"  - Credit card velocity: {len(cc_velocity.columns) - 1}")
    print(f"  - Application history: {len(app_history.columns) - 1}")
    print(f"  - POS cash trends: {len(pos_trends.columns) - 1}")
    print(f"  - Cross-table features: {len(cross_features_train.columns) - 1}")
    print(f"  - Alternative scoring: ~20")

    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE!")
    print("=" * 70)
    print("\nNext step: Run enhanced_model.py with optimized data")