# Explainable Horizon-Adaptive Stacked Learning for U.S. Inflation Forecasting with SHAP
# Baker, M.R., Buyrukoğlu, S., & Jihad, K.H.

import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from data_loader import load_macro_dataset
from feature_engineering import create_features, create_supervised_dataset, MutualInfoFeatureSelector
from models import HASE


def compute_r2(y_true, y_pred):
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan


def run_rolling_validation():
    df_raw = load_macro_dataset()
    df_f   = create_features(df_raw)
    df_f   = df_f[df_f.index.notnull()].sort_index()

    cutoff_years = list(range(2005, 2024))
    HORIZONS     = [1, 3, 6]
    results      = []

    for H in HORIZONS:
        print(f"\nHorizon H={H}")
        X_all, y_all = create_supervised_dataset(df_f, 'Inflation_Rate', horizon=H)
        X_all, y_all = X_all.sort_index(), y_all.sort_index()

        for yr in cutoff_years:
            X_tr = X_all[X_all.index <= pd.Timestamp(f'{yr}-12-31')]
            y_tr = y_all[y_all.index <= pd.Timestamp(f'{yr}-12-31')]
            X_te = X_all[(X_all.index >= pd.Timestamp(f'{yr+1}-01-01')) &
                         (X_all.index <= pd.Timestamp(f'{yr+1}-12-31'))]
            y_te = y_all[(y_all.index >= pd.Timestamp(f'{yr+1}-01-01')) &
                         (y_all.index <= pd.Timestamp(f'{yr+1}-12-31'))]

            if len(X_te) < 5 or len(X_tr) < 80:
                continue

            try:
                sel    = MutualInfoFeatureSelector(k=25)
                X_tr_s = sel.fit_transform(X_tr, y_tr)
                X_te_s = sel.transform(X_te)

                hase = HASE(horizon=H)
                hase.fit(X_tr_s, y_tr)
                r2_hase = compute_r2(y_te.values, hase.predict(X_te_s))

                # NOTE (R3 B.10): Fixed from dynamic AR(1) iterated path to direct H-step OLS,
                # consistent with AR1DirectWrapper in models.py (B.4 fix).
                # Old code: AutoReg(...).predict(start=len(y_tr), ...) chains from own
                # predictions (iterated path), which is incorrect for H>1 direct forecasting.
                # New code: fits OLS of y[t+H] on y[t] over training window, then applies
                # to the inflation feature in each test-period feature row.
                from sklearn.linear_model import LinearRegression
                y_tr_arr = y_tr.values
                _X_ar = y_tr_arr[:-H].reshape(-1, 1) if H < len(y_tr_arr) else y_tr_arr.reshape(-1, 1)
                _y_ar = y_tr_arr[H:]               if H < len(y_tr_arr) else y_tr_arr
                ar1_lr = LinearRegression().fit(_X_ar, _y_ar)
                INFL_CANDS = ['Inflation_Rate', 'Inflation_Momentum_3M',
                              'Inflation_Momentum_6M', 'Inflation_Momentum_9M',
                              'Inflation_Momentum_12M']
                _infl_col = next((c for c in INFL_CANDS if c in X_te.columns), None)
                if _infl_col is not None:
                    ar1_preds = ar1_lr.predict(X_te[[_infl_col]].values)
                else:
                    ar1_preds = np.full(len(X_te), ar1_lr.predict([[y_tr_arr[-1]]])[0])
                r2_ar1 = compute_r2(y_te.values, ar1_preds)

                from xgboost import XGBRegressor
                xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3,
                                   random_state=42, n_jobs=-1, verbosity=0)
                xgb.fit(X_tr_s, y_tr.values)
                r2_xgb = compute_r2(y_te.values, xgb.predict(X_te_s))

                results.append({'Horizon': f'H={H}', 'Train End': f'{yr}-12',
                                'HASE_R2': r2_hase, 'AR1_R2': r2_ar1, 'XGB_R2': r2_xgb})
                print(f"  Origin {yr}: HASE={r2_hase:.3f}, AR1={r2_ar1:.3f}, XGB={r2_xgb:.3f}")
            except Exception as e:
                print(f"  Origin {yr} failed: {e}")

    df_res = pd.DataFrame(results)

    eras = {
        'Great Moderation (2006-2007)':        ['2005-12', '2006-12'],
        'Global Financial Crisis (2008-2009)':  ['2007-12', '2008-12'],
        'Post-Pandemic Shock (2021-2022)':      ['2020-12', '2021-12'],
    }

    summary_rows = []
    for h in [f'H={x}' for x in HORIZONS]:
        df_h = df_res[df_res['Horizon'] == h]
        for era_name, subset in eras.items():
            df_era = df_h[df_h['Train End'].isin(subset)]
            if len(df_era) > 0:
                summary_rows.append({
                    'Horizon': h, 'Economic Era': era_name,
                    'HASE':    f"{df_era['HASE_R2'].mean():.3f}",
                    'AR(1)':   f"{df_era['AR1_R2'].mean():.3f}",
                    'XGBoost': f"{df_era['XGB_R2'].mean():.3f}",
                })

    df_era_table = pd.DataFrame(summary_rows)
    print("\n--- Rolling-Origin Summary (Table C.1) ---")
    print(df_era_table.to_string(index=False))
    df_era_table.to_csv("rolling_origin_summary.csv", index=False)
    return df_era_table


if __name__ == "__main__":
    run_rolling_validation()
