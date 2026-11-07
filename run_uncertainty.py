# Explainable Horizon-Adaptive Stacked Learning for U.S. Inflation Forecasting with SHAP
# Baker, M.R., Buyrukoğlu, S., & Jihad, K.H.

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from data_loader import load_macro_dataset
from feature_engineering import create_features, create_supervised_dataset, MutualInfoFeatureSelector
from models import HASE

OUTPUT_DIR = 'results/3_Uncertainty'
os.makedirs(f"{OUTPUT_DIR}/Figures", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/Tables",  exist_ok=True)


def generate_prediction_intervals(model, X_train, y_train, X_test, alpha=0.10):
    X_arr, y_arr = np.array(X_train), np.array(y_train)
    n     = len(X_arr)
    n_cal = max(int(n * 0.25), 10)
    X_cal, y_cal  = X_arr[-n_cal:], y_arr[-n_cal:]
    residuals     = np.abs(y_cal - model.predict(X_cal))
    q_level       = min(np.ceil((1 - alpha) * (n_cal + 1)) / n_cal, 1.0)
    half_width    = np.quantile(residuals, q_level)
    point         = model.predict(np.array(X_test))
    return {'point': point, 'lower': point - half_width, 'upper': point + half_width}


def evaluate_coverage(y_true, intervals):
    covered = (y_true >= intervals['lower']) & (y_true <= intervals['upper'])
    return {'target_coverage': 0.90, 'empirical_coverage': float(covered.mean()),
            'coverage_gap': float(covered.mean()) - 0.90}


def run_uncertainty_quantification():
    df_raw   = load_macro_dataset(use_realtime=True, publication_delays=True)
    df_feats = create_features(df_raw)
    all_results = []

    for H in [1, 3, 6, 12]:
        print(f"\n  H={H}")
        X, y        = create_supervised_dataset(df_feats, 'Inflation_Rate', horizon=H)
        X_train     = X[X.index <  '2013-01-01']
        y_train     = y[y.index <  '2013-01-01']
        X_test      = X[X.index >= '2013-01-01']
        y_test      = y[y.index >= '2013-01-01']

        selector    = MutualInfoFeatureSelector(k=25)
        X_train_sel = selector.fit_transform(X_train, y_train)
        X_test_sel  = selector.transform(X_test)

        model = HASE(horizon=H, random_state=42)
        model.fit(X_train_sel, y_train)

        intervals = generate_prediction_intervals(model, X_train_sel, y_train, X_test_sel)
        coverage  = evaluate_coverage(y_test.values, intervals)
        print(f"    Coverage: {coverage['empirical_coverage']:.1%} (target 90%)")

        df_intervals = pd.DataFrame({
            'Date':      X_test.index,
            'Actual':    y_test.values,
            'Predicted': intervals['point'],
            'Lower_90':  intervals['lower'],
            'Upper_90':  intervals['upper'],
        })
        df_intervals.to_csv(
            os.path.join(OUTPUT_DIR, 'Tables', f'H_{H}_intervals.csv'), index=False)

        plt.figure(figsize=(14, 6))
        plt.plot(df_intervals['Date'], df_intervals['Actual'], 'k-', label='Actual', linewidth=2)
        plt.plot(df_intervals['Date'], df_intervals['Predicted'], 'r-', label='HASE', linewidth=2)
        plt.fill_between(df_intervals['Date'], df_intervals['Lower_90'], df_intervals['Upper_90'],
                         alpha=0.3, color='lightblue', label='90% Interval')
        plt.xlabel('Date'); plt.ylabel('PCE Inflation (%)'); plt.grid(True, alpha=0.3)
        plt.title(f'HASE — 90% Prediction Intervals (H={H})', fontsize=14, fontweight='bold')
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'Figures', f'Intervals_H{H}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        all_results.append({
            'Horizon': f'H={H}',
            'Target_Coverage':    coverage['target_coverage'],
            'Empirical_Coverage': coverage['empirical_coverage'],
            'Coverage_Gap':       coverage['coverage_gap'],
            'Avg_Interval_Width': float((intervals['upper'] - intervals['lower']).mean()),
        })

    df_cov = pd.DataFrame(all_results)
    df_cov.to_csv(os.path.join(OUTPUT_DIR, 'Tables', 'Coverage_Summary.csv'), index=False)
    print(f"\nDone. Output saved to: {OUTPUT_DIR}/")
    return df_cov


if __name__ == "__main__":
    run_uncertainty_quantification()
