# run_baselines_only.py
# R3 B.4 correction: re-evaluate ONLY the three corrected statistical benchmarks
# (AR1_Direct, RandomWalk, VAR) using the frozen ML tuned params and original
# feature selection.  ML model results (ElasticNet, RF, XGBoost, LightGBM,
# Stacking RF-KNN, HASE) are read from the existing CSVs and NEVER re-run.
#
# Run from the Zenodo Upload Code_V2 directory:
#   cd "e:\Economy\V10_Final_Project\Final Results V11\R3\Zenodo Upload Code_V2"
#   python run_baselines_only.py

import os, sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── paths ────────────────────────────────────────────────────────────────────
ZENODO_DIR   = os.path.dirname(os.path.abspath(__file__))
RESULTS_ROOT = r'e:\Economy\V10_Final_Project\Final Results V11\1_Model_Performance\Tables'
OUTPUT_DIR   = os.path.join(ZENODO_DIR, 'results_B4')
os.makedirs(OUTPUT_DIR, exist_ok=True)

sys.path.insert(0, ZENODO_DIR)
from data_loader import load_macro_dataset
from feature_engineering import create_features, create_supervised_dataset, MutualInfoFeatureSelector
from run_experiment import train_statistical_baseline

TEST_START = '2013-01-01'

# ── load data once ────────────────────────────────────────────────────────────
print("Loading data...")
df_raw   = load_macro_dataset(use_realtime=True, publication_delays=True)
df_feats = create_features(df_raw)
print(f"  Feature matrix: {df_feats.shape}")

summary_rows = []

for H in [1, 3, 6, 12]:
    print(f"\n{'='*55}\n  HORIZON H={H}\n{'='*55}")

    # ── supervised split ───────────────────────────────────────────────────
    X, y     = create_supervised_dataset(df_feats, 'Inflation_Rate', horizon=H)
    X_train  = X[X.index <  TEST_START]
    y_train  = y[y.index <  TEST_START]
    X_test   = X[X.index >= TEST_START]
    y_test   = y[y.index >= TEST_START]

    # ── reproduce exact same MI feature selection as original run ─────────
    selector    = MutualInfoFeatureSelector(k=25, random_state=42)
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel  = selector.transform(X_test)

    # Confirm feature set matches original
    orig_feats_path = os.path.join(RESULTS_ROOT, f'H_{H}_selected_features.csv')
    orig_feats = pd.read_csv(orig_feats_path)['Feature'].tolist()
    new_feats  = list(selector.selected_features_)
    if set(orig_feats) == set(new_feats):
        print(f"  Feature set: MATCHES original run ({len(new_feats)} features)")
    else:
        diff_added   = set(new_feats)  - set(orig_feats)
        diff_removed = set(orig_feats) - set(new_feats)
        print(f"  WARNING: Feature set differs!")
        print(f"    Added  : {diff_added}")
        print(f"    Removed: {diff_removed}")

    # Which inflation column does AR1_Direct get at this horizon?
    from models import AR1DirectWrapper
    cols = list(selector.selected_features_)
    infl_idx_used = next(
        (i for i, c in enumerate(cols) if c in AR1DirectWrapper.INFL_CANDIDATES), 0)
    infl_col_used = cols[infl_idx_used] if infl_idx_used < len(cols) else 'FALLBACK'
    print(f"  AR1_Direct uses: col[{infl_idx_used}] = '{infl_col_used}'")

    # ── run only the three statistical benchmarks ─────────────────────────
    bench_rows = []
    for name in ['AR1_Direct', 'RandomWalk', 'VAR']:
        try:
            y_pred = train_statistical_baseline(name, X_train_sel, y_train, X_test_sel)
            row = {
                'Model': name,
                'RMSE': float(np.sqrt(mean_squared_error(y_test, y_pred))),
                'MAE':  float(mean_absolute_error(y_test, y_pred)),
                'R2':   float(r2_score(y_test, y_pred)),
            }
            bench_rows.append(row)
            print(f"  {name:15s}: RMSE={row['RMSE']:.4f}  MAE={row['MAE']:.4f}  R2={row['R2']:.4f}")
        except Exception as e:
            print(f"  {name} FAILED: {e}")

    # ── load frozen ML results ────────────────────────────────────────────
    frozen_path = os.path.join(RESULTS_ROOT, f'H_{H}_metrics.csv')
    frozen_df   = pd.read_csv(frozen_path)

    # Drop old AR1 / VAR rows (they were iterated, now replaced)
    ml_df = frozen_df[~frozen_df['Model'].isin(['AR1', 'VAR'])].copy()
    ml_df = ml_df[['Model', 'RMSE', 'MAE', 'R2']]

    # Combine new benchmarks with frozen ML rows
    bench_df  = pd.DataFrame(bench_rows)
    combined  = pd.concat([bench_df, ml_df], ignore_index=True)
    combined  = combined.sort_values('RMSE').reset_index(drop=True)
    combined['Rank'] = range(1, len(combined) + 1)
    combined['Horizon'] = H

    out_path = os.path.join(OUTPUT_DIR, f'H_{H}_metrics_B4.csv')
    combined.to_csv(out_path, index=False)
    print(f"\n  Results saved: {out_path}")
    print(combined[['Model', 'RMSE', 'MAE', 'R2', 'Rank']].to_string(index=False))

    summary_rows.append(combined)

# ── cross-horizon summary ─────────────────────────────────────────────────────
print(f"\n{'='*55}\n  CROSS-HORIZON SUMMARY\n{'='*55}")
all_df = pd.concat(summary_rows, ignore_index=True)
pivot  = all_df.pivot_table(index='Model', columns='Horizon', values='RMSE')
print("\nRMSE by horizon:")
print(pivot.to_string())

avg_rank = all_df.groupby('Model')['Rank'].mean().sort_values()
print("\nAverage rank across horizons:")
print(avg_rank.to_string())

summary_path = os.path.join(OUTPUT_DIR, 'Summary_B4_corrected.csv')
all_df.to_csv(summary_path, index=False)
print(f"\nFull summary saved: {summary_path}")
print("\nDONE. Paste the RMSE/MAE/R2 numbers above into the manuscript.")
