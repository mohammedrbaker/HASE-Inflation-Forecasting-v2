# Explainable Horizon-Adaptive Stacked Learning for U.S. Inflation Forecasting with SHAP
# Baker, M.R., Buyrukoğlu, S., & Jihad, K.H.

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_regression


def create_features(df):
    df = df.copy()

    df['Inflation_Rate'] = df['PCE'].pct_change(12) * 100

    # Unemployment gap: LEVELS (Phillips Curve specification)
    df['UNRATE_Trend'] = df['UNRATE'].rolling(36, min_periods=12).mean()
    df['UNRATE_Gap']   = df['UNRATE'] - df['UNRATE_Trend']

    df['INDPRO_YoY']      = df['INDPRO'].pct_change(12) * 100
    df['Output_Gap']      = df['INDPRO_YoY'] - df['INDPRO_YoY'].rolling(24).mean()
    df['M2_YoY']          = df['M2SL'].pct_change(12) * 100
    df['M2_Growth_Accel'] = df['M2_YoY'].diff(3)
    df['Oil_YoY']         = df['WTI'].pct_change(12) * 100
    df['Oil_Volatility']  = df['WTI'].pct_change().rolling(12).std() * 100
    df['Real_Rate']        = df['PolicyRate'] - df['Inflation_Rate']
    df['Real_Rate_Change'] = df['Real_Rate'].diff(3)
    df['Term_Spread']      = df['TermSpread'] if 'TermSpread' in df.columns else 0.0

    for window in [3, 6, 9, 12]:
        df[f'Inflation_Momentum_{window}M'] = df['Inflation_Rate'].rolling(window).mean()
        df[f'Inflation_Velocity_{window}M'] = df['Inflation_Rate'].diff(window)

    df['Inflation_Vol_12M'] = df['Inflation_Rate'].rolling(12).std()

    for col in ['Inflation_Rate', 'UNRATE_Gap', 'Oil_YoY', 'M2_YoY', 'Real_Rate']:
        for lag in [1, 2, 3, 6, 9, 12]:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)

    df['Inter_Inf_Unrate']   = df['Inflation_Rate'] * df['UNRATE']
    df['Inter_Inf_Oil']      = df['Inflation_Rate'] * df['Oil_YoY']
    df['Inter_RealRate_Gap'] = df['Real_Rate'] * df['Output_Gap']

    raw_levels = ['PCE', 'UNRATE', 'INDPRO', 'M2SL', 'WTI', 'GS10', 'TB3MS', 'TermSpread']
    df = df.drop(columns=[c for c in raw_levels if c in df.columns])
    df = df.dropna()

    print(f"Features: {df.shape[1]} stationary predictors, {df.shape[0]} observations")
    return df


def create_supervised_dataset(df, target_col='Inflation_Rate', horizon=1):
    df = df.copy()
    df['Target'] = df[target_col].shift(-horizon)
    df = df.dropna()
    return df.drop(columns=['Target']), df['Target']


class MutualInfoFeatureSelector:
    def __init__(self, k=25, random_state=42):
        self.k = k
        self.random_state = random_state
        self.selected_features_ = None

    def fit_transform(self, X, y):
        # random_state=42 ensures deterministic MI selection across all scripts
        scorer = lambda X_, y_: mutual_info_regression(X_, y_, random_state=self.random_state)
        selector = SelectKBest(score_func=scorer, k=self.k)
        X_new = selector.fit_transform(X, y)
        self.selected_features_ = X.columns[selector.get_support()]
        return pd.DataFrame(X_new, index=X.index, columns=self.selected_features_)

    def transform(self, X):
        return X[self.selected_features_]
