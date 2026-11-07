# run_dm_tests.py
# Diebold-Mariano Statistical Significance Tests with Newey-West HAC and HLN Correction
# R3 Peer-Review Resolution: Comment B.5
#
# References:
#   Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy.
#     Journal of Business & Economic Statistics, 13(3), 253-263.
#   Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of prediction
#     mean squared errors. International Journal of Forecasting, 13(2), 281-291.
#   Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite,
#     heteroskedasticity and autocorrelation consistent covariance matrix.
#     Econometrica, 55(3), 703-708.

import os, sys
os.environ['LOKY_MAX_CPU_COUNT'] = '1'
import ast
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

# Ensure local imports resolve
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from data_loader import load_macro_dataset
from feature_engineering import create_features, create_supervised_dataset, MutualInfoFeatureSelector
from models import HASE, get_model_roster
from run_experiment import train_statistical_baseline

TEST_START = '2013-01-01'
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results_B4')
os.makedirs(RESULTS_DIR, exist_ok=True)


def diebold_mariano_test(e_ref, e_comp, h, loss_type='squared'):
    """
    Computes the Diebold-Mariano (1995) test with Newey-West (1987) HAC
    autocovariance estimator and the Harvey-Leybourne-Newbold (1997)
    small-sample correction.

    Parameters:
    -----------
    e_ref : array-like
        Forecast errors of reference model (e_ref = y_true - y_pred_ref).
        Here: HASE ensemble.
    e_comp : array-like
        Forecast errors of competing benchmark model (e_comp = y_true - y_pred_comp).
    h : int
        Forecast horizon in months (h in {1, 3, 6, 12}).
    loss_type : str, default='squared'
        Loss function: 'squared' (MSE) or 'absolute' (MAE).

    Returns:
    --------
    dict with:
        T             : Sample size (number of out-of-sample forecast origins)
        bandwidth     : HAC lag truncation parameter q = max(0, h - 1)
        mean_d        : Sample mean of loss differential d_t = Loss(ref) - Loss(comp)
        DM_stat       : Standard Diebold-Mariano z-statistic
        p_val_norm    : Two-sided p-value from standard normal N(0, 1)
        DM_HLN        : Harvey-Leybourne-Newbold corrected test statistic
        p_val_HLN     : Two-sided p-value from Student's t(T - 1) distribution
    """
    e_ref = np.asarray(e_ref)
    e_comp = np.asarray(e_comp)
    T = len(e_ref)

    # 1. Loss differential: d_t = Loss(ref) - Loss(comp)
    # Note: d_t < 0 indicates the reference model (HASE) has smaller loss
    if loss_type == 'squared':
        d = e_ref**2 - e_comp**2
    elif loss_type == 'absolute':
        d = np.abs(e_ref) - np.abs(e_comp)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    mean_d = float(np.mean(d))

    # 2. HAC Newey-West spectral variance with Bartlett kernel
    # Under optimal h-step forecasts, forecast errors follow an MA(h-1) process.
    # The bandwidth is set to q = h - 1.
    q = max(0, h - 1)
    gamma0 = float(np.var(d, ddof=0))
    gamma_sum = 0.0

    for k in range(1, q + 1):
        gamma_k = float(np.mean((d[k:] - mean_d) * (d[:-k] - mean_d)))
        bartlett_weight = 1.0 - (k / (q + 1.0))
        gamma_sum += bartlett_weight * gamma_k

    long_run_var = gamma0 + 2.0 * gamma_sum

    if long_run_var <= 1e-12:
        return {
            'T': T, 'bandwidth': q, 'mean_d': 0.0,
            'DM_stat': 0.0, 'p_val_norm': 1.0,
            'DM_HLN': 0.0, 'p_val_HLN': 1.0
        }

    # 3. Standard DM statistic
    var_d_bar = long_run_var / T
    dm_stat = mean_d / np.sqrt(var_d_bar)
    p_val_norm = 2.0 * (1.0 - stats.norm.cdf(abs(dm_stat)))

    # 4. Harvey-Leybourne-Newbold (1997) small-sample correction
    # Correction factor: sqrt([T + 1 - 2h + h(h-1)/T] / T)
    hln_factor = np.sqrt(max(0.0, (T + 1 - 2*h + (h * (h - 1)) / T) / T))
    dm_hln = dm_stat * hln_factor

    # Two-sided p-value evaluated against Student's t distribution with (T - 1) df
    p_val_hln = 2.0 * (1.0 - stats.t.cdf(abs(dm_hln), df=T - 1))

    return {
        'T': T,
        'bandwidth': q,
        'mean_d': mean_d,
        'DM_stat': dm_stat,
        'p_val_norm': p_val_norm,
        'DM_HLN': dm_hln,
        'p_val_HLN': p_val_hln
    }


def main():
    print("=" * 75)
    print("  DIEBOLD-MARIANO TEST SUITE (HAC Bandwidth q=h-1, HLN Corrected)")
    print("  Reference Model: HASE Ensemble")
    print("=" * 75)

    print("\nLoading macro dataset...")
    df_raw = load_macro_dataset(use_realtime=True, publication_delays=True)
    df_feats = create_features(df_raw)

    dm_results = []

    # Paths to tuned hyperparameters
    tuned_dir = os.path.join(SCRIPT_DIR, '..', '..', '1_Model_Performance', 'Tables')
    if not os.path.exists(tuned_dir):
        tuned_dir = os.path.join(SCRIPT_DIR, 'results_B4')

    for H in [1, 3, 6, 12]:
        print(f"\nEvaluating Horizon H={H}...")
        X, y = create_supervised_dataset(df_feats, 'Inflation_Rate', horizon=H)
        X_train = X[X.index < TEST_START]
        y_train = y[y.index < TEST_START]
        X_test = X[X.index >= TEST_START]
        y_test = y[y.index >= TEST_START]
        T_h = len(y_test)

        selector = MutualInfoFeatureSelector(k=25, random_state=42)
        X_train_sel = selector.fit_transform(X_train, y_train)
        X_test_sel = selector.transform(X_test)

        # Load tuned parameters
        tp_path = os.path.join(tuned_dir, f'H_{H}_tuned_params.csv')
        if os.path.exists(tp_path):
            tp_df = pd.read_csv(tp_path)
            tuned_params = {col: ast.literal_eval(tp_df[col].iloc[0]) for col in tp_df.columns}
        else:
            tuned_params = {}

        # 1. Predictions for HASE (Reference)
        hase = HASE(horizon=H, tuned_params=tuned_params)
        hase.fit(X_train_sel, y_train)
        y_pred_hase = hase.predict(X_test_sel)
        e_hase = y_test.values - y_pred_hase
        rmse_hase = np.sqrt(mean_squared_error(y_test, y_pred_hase))

        # Add Reference row
        dm_results.append({
            'Horizon': f'H={H}',
            'Model': 'HASE model (Ours)',
            'T': T_h,
            'Bandwidth_q': max(0, H - 1),
            'RMSE': round(rmse_hase, 4),
            'DM_Stat': '-',
            'DM_HLN': '-',
            'p_value_HLN': '-',
            'Significance': 'Ref',
            'Superiority': 'Reference'
        })

        # 2. Competitor models
        competitors = {}

        # Statistical baselines
        for bname in ['RandomWalk', 'AR1_Direct', 'VAR']:
            competitors[bname] = train_statistical_baseline(bname, X_train_sel, y_train, X_test_sel)

        # Standalone ML models
        roster = get_model_roster(tuned_params, horizon=H)
        roster['RandomForest'].n_jobs = 1
        for mname in ['ElasticNet', 'RandomForest', 'XGBoost', 'LightGBM']:
            m = roster[mname]
            m.fit(X_train_sel, y_train)
            competitors[mname] = m.predict(X_test_sel)

        # Stacking RF-KNN
        rf_base = RandomForestRegressor(
            n_estimators=roster['RandomForest'].n_estimators,
            max_depth=roster['RandomForest'].max_depth,
            min_samples_split=roster['RandomForest'].min_samples_split,
            random_state=42, n_jobs=1
        )
        knn_base = KNeighborsRegressor(n_neighbors=5, n_jobs=1)
        stacking = StackingRegressor(
            estimators=[('rf', rf_base), ('knn', knn_base)],
            final_estimator=RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1),
            cv=5, n_jobs=1
        )
        stacking.fit(X_train_sel, y_train)
        competitors['Stacking (RF-KNN)'] = stacking.predict(X_test_sel)

        # Order of evaluation in Table 10
        eval_order = [
            'RandomWalk', 'AR1_Direct', 'VAR', 'ElasticNet',
            'RandomForest', 'XGBoost', 'LightGBM', 'Stacking (RF-KNN)'
        ]

        for mname in eval_order:
            y_pred_comp = competitors[mname]
            e_comp = y_test.values - y_pred_comp
            rmse_comp = np.sqrt(mean_squared_error(y_test, y_pred_comp))

            dm_dict = diebold_mariano_test(e_hase, e_comp, h=H, loss_type='squared')

            dm_hln = dm_dict['DM_HLN']
            p_val = dm_dict['p_val_HLN']

            # Significance attribution
            # In a two-sided test, rejection depends on magnitude: p < 0.05 (|DM| >= crit)
            # Sign indicates direction:
            # DM < 0: HASE has lower loss (HASE outperforms)
            # DM > 0: Competitor has lower loss (Competitor outperforms)
            if p_val < 0.01:
                sig = '***'
            elif p_val < 0.05:
                sig = '**'
            elif p_val < 0.10:
                sig = '*'
            else:
                sig = 'ns'

            if sig in ['***', '**']:
                winner = 'HASE' if dm_hln < 0 else mname
            elif sig == '*':
                winner = f"{'HASE' if dm_hln < 0 else mname} (10%)"
            else:
                winner = 'Tied'

            dm_results.append({
                'Horizon': f'H={H}',
                'Model': mname,
                'T': T_h,
                'Bandwidth_q': max(0, H - 1),
                'RMSE': round(rmse_comp, 4),
                'DM_Stat': round(dm_dict['DM_stat'], 3),
                'DM_HLN': round(dm_hln, 3),
                'p_value_HLN': f"{p_val:.4e}" if p_val < 1e-4 else f"{p_val:.4f}",
                'Significance': sig,
                'Superiority': winner
            })

            print(f"  {mname:<18}: RMSE={rmse_comp:.4f} | DM_HLN={dm_hln:+.3f} | p={p_val:.4f} ({sig}) -> {winner}")

    df_out = pd.DataFrame(dm_results)
    out_csv = os.path.join(RESULTS_DIR, 'Diebold_Mariano_Tests_Corrected.csv')
    df_out.to_csv(out_csv, index=False)
    print(f"\nAll DM test results saved to: {out_csv}")


if __name__ == '__main__':
    main()
