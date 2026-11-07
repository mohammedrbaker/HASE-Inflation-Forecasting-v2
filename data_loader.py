# Explainable Horizon-Adaptive Stacked Learning for U.S. Inflation Forecasting with SHAP
# Baker, M.R., Buyrukoğlu, S., & Jihad, K.H.

import pandas as pd
import numpy as np
import os

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Data_Cache')  # Local data folder (sibling of this script)


def load_series(name):
    path = os.path.join(CACHE_DIR, f"{name}.csv")
    if "_vintage" in name:
        df = pd.read_csv(path)
        df['date'] = pd.to_datetime(df['date'])
        df['realtime_start'] = pd.to_datetime(df['realtime_start'])
        # NOTE (R3 B.12): .last() after sorting by realtime_start selects the most
        # recently available vintage for each observation date (latest revision).
        # This is NOT a first-release design; it uses the most recently revised value
        # available in the ALFRED archive as of the download date.
        s = df.sort_values('realtime_start').groupby('date')['value'].last()
    else:
        df = pd.read_csv(path, header=None, index_col=0)
        df.index = pd.to_datetime(df.index)
        s = df.iloc[:, 0]
    return s


def load_macro_dataset(use_realtime=True, publication_delays=True, **kwargs):
    # NOTE (R3 B.12): use_realtime and publication_delays are retained for API
    # compatibility but are not implemented. Data uses latest-vintage ALFRED values
    # without publication-lag shifts. See manuscript Section 2.1 for details.
    series = {
        'PCE':        load_series('PCE_vintage'),
        'UNRATE':     load_series('UNRATE_vintage'),
        'INDPRO':     load_series('INDPRO_vintage'),
        'M2SL':       load_series('M2SL_vintage'),
        'PolicyRate': load_series('PolicyRate_market'),
        'WTI':        load_series('WTI_market'),
        'GS10':       load_series('GS10_market'),
        'TB3MS':      load_series('TB3MS_market'),
    }
    df = pd.DataFrame(series)
    df = df.resample('ME').last()
    df = df.ffill().dropna()
    # Target & Transformations
    df['Inflation_Rate'] = df['PCE'].pct_change(12) * 100
    df['TermSpread']     = df['GS10'] - df['TB3MS']
    # Long-horizon trend filtering establishing the stable post-1995 active modeling window:
    # A 120-month rolling mean sets the active baseline to 1995:12 (362 observations),
    # which yields 327 complete observations (1998:11-2026:01) after feature-engineering burn-in,
    # providing exactly 170 training observations (1998:11-2012:12).
    df['UNRATE_Gap']     = df['UNRATE'] - df['UNRATE'].rolling(120).mean()
    df = df.dropna()
    print(f"Data loaded: {df.shape[0]} observations ({df.index[0].date()} to {df.index[-1].date()})")
    return df



