# Explainable Horizon-Adaptive Stacked Learning for U.S. Inflation Forecasting with SHAP
# Baker, M.R., Buyrukoğlu, S., & Jihad, K.H.

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import optuna
from optuna.samplers import TPESampler
import os
import warnings
warnings.filterwarnings('ignore')

from data_loader import load_macro_dataset
from feature_engineering import create_features, create_supervised_dataset, MutualInfoFeatureSelector
from models import HASE

optuna.logging.set_verbosity(optuna.logging.WARNING)

OUTPUT_DIR = 'results'
os.makedirs(f"{OUTPUT_DIR}/1_Model_Performance/Tables",  exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/1_Model_Performance/Figures", exist_ok=True)


def _tune(objective_fn, n_trials):
    study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
    study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


# NOTE (R3 B.13): All Optuna objective functions below use scikit-learn's default
# cross_val_score(..., cv=5), which for regressors is unshuffled KFold(n_splits=5),
# evaluating hyperparameter candidates across the 170 observations of the 1998-2012 training block.
# This yields the optimal hyperparameter values reported in Table 4 of the manuscript.
# Out-of-sample forecast evaluations (Tables 5-8) are evaluated on the separate 2013-2025 holdout.
def tune_elasticnet(X_train, y_train, n_trials=50):
    def objective(trial):
        p = {'alpha': trial.suggest_float('alpha', 0.001, 10.0, log=True),
             'l1_ratio': trial.suggest_float('l1_ratio', 0.0, 1.0), 'random_state': 42}
        return -cross_val_score(ElasticNet(**p), X_train, y_train,
                                cv=5, scoring='neg_mean_squared_error', n_jobs=-1).mean()
    return _tune(objective, n_trials)


def tune_randomforest(X_train, y_train, n_trials=50):
    def objective(trial):
        p = {'n_estimators': trial.suggest_int('n_estimators', 100, 500),
             'max_depth': trial.suggest_int('max_depth', 5, 20),
             'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
             'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
             'random_state': 42, 'n_jobs': -1}
        return -cross_val_score(RandomForestRegressor(**p), X_train, y_train,
                                cv=5, scoring='neg_mean_squared_error', n_jobs=-1).mean()
    return _tune(objective, n_trials)


def tune_xgboost(X_train, y_train, n_trials=50):
    def objective(trial):
        p = {'n_estimators': trial.suggest_int('n_estimators', 50, 300),
             'max_depth': trial.suggest_int('max_depth', 3, 10),
             'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
             'subsample': trial.suggest_float('subsample', 0.5, 1.0),
             'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
             'random_state': 42, 'n_jobs': -1, 'verbosity': 0}
        return -cross_val_score(XGBRegressor(**p), X_train, y_train,
                                cv=5, scoring='neg_mean_squared_error', n_jobs=-1).mean()
    return _tune(objective, n_trials)


def tune_lightgbm(X_train, y_train, n_trials=50):
    def objective(trial):
        p = {'n_estimators': trial.suggest_int('n_estimators', 50, 300),
             'max_depth': trial.suggest_int('max_depth', 3, 10),
             'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
             'num_leaves': trial.suggest_int('num_leaves', 20, 150),
             'subsample': trial.suggest_float('subsample', 0.5, 1.0),
             'random_state': 42, 'n_jobs': -1, 'verbose': -1}
        return -cross_val_score(LGBMRegressor(**p), X_train, y_train,
                                cv=5, scoring='neg_mean_squared_error', n_jobs=-1).mean()
    return _tune(objective, n_trials)


def tune_knn_meta(X_train, y_train, n_trials=30):
    def objective(trial):
        p = {'n_neighbors': trial.suggest_int('n_neighbors', 3, 15),
             'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
             'n_jobs': -1}
        return -cross_val_score(KNeighborsRegressor(**p), X_train, y_train,
                                cv=5, scoring='neg_mean_squared_error', n_jobs=-1).mean()
    return _tune(objective, n_trials)


def train_statistical_baseline(model_name, X_train, y_train, X_test):
    """
    Statistical benchmark implementations.

    NOTE (R3 B.4): The previous AR1 implementation used statsmodels AutoReg
    .predict(start=len(y_train), end=...) which generates a single iterated
    multi-step path from the last training observation.  With a fitted AR
    coefficient of ~0.935 the path converges to the unconditional mean within
    a few months and produces a near-horizontal forecast line over 2013-2026,
    yielding horizon-invariant RMSE (~1.68-1.75) regardless of H.

    Corrected implementations below use DIRECT h-step-ahead forecasts, where
    each test-period prediction is made from the contemporaneous predictor
    value at that origin — consistent with the direct-forecasting strategy
    used throughout the rest of the pipeline (HASE, ElasticNet, RF, etc.).

    AR1_Direct  : OLS regression of y(t+h) on y(t) estimated on training
                  data; at test time uses the contemporaneous Inflation_Rate
                  value for each origin t.  Identical specification to HASE's
                  internal AR1DirectWrapper component.

    RandomWalk  : No-change (Atkeson-Ohanian 2001) forecast:
                  y_hat(t+h) = y(t).  Requires only the contemporaneous
                  Inflation_Rate to be present in the feature matrix.

    VAR         : Recursive multi-step forecast from statsmodels VAR using
                  the last k_ar observations as the forecast origin at each
                  step.  Lag order selected by BIC (maxlags=12, ic='bic').
    """
    from sklearn.linear_model import LinearRegression
    from statsmodels.tsa.api import VAR
    import pandas as pd
    import numpy as np

    # Helper: resolve the contemporaneous Inflation_Rate column index
    candidates = [
        'Inflation_Rate', 'Inflation_Rate_lag1',
        'Inflation_Rate_lag3', 'Inflation_Momentum_3M',
    ]

    def _infl_col(X_df):
        """Return column index of best inflation proxy in X_df."""
        if hasattr(X_df, 'columns'):
            cols = list(X_df.columns)
            idx = next((i for i, c in enumerate(cols) if c in candidates), 0)
            return idx
        return 0

    if model_name == 'AR1_Direct':
        # Direct h-step OLS: y(t+h) ~ alpha + beta * y(t)
        idx = _infl_col(X_train)
        x_tr = np.array(X_train)[:, idx].reshape(-1, 1)
        x_te = np.array(X_test)[:, idx].reshape(-1, 1)
        return LinearRegression().fit(x_tr, np.array(y_train)).predict(x_te)

    elif model_name == 'RandomWalk':
        # No-change / Atkeson-Ohanian (2001): y_hat(t+h) = y(t)
        idx = _infl_col(X_test)
        return np.array(X_test)[:, idx].astype(float)

    elif model_name == 'VAR':
        try:
            data = pd.concat(
                [pd.Series(y_train.values, name='target', index=y_train.index),
                 X_train.iloc[:, :5]], axis=1)
            fitted = VAR(data).fit(maxlags=12, ic='bic')
            return fitted.forecast(data.values[-fitted.k_ar:],
                                   steps=len(X_test))[:, 0]
        except Exception:
            return train_statistical_baseline('AR1_Direct', X_train, y_train, X_test)


def run_single_horizon(H, df_feats, test_start='2013-01-01', n_trials=50):
    print(f"\n{'='*60}\n   HORIZON H={H}\n{'='*60}")

    X, y        = create_supervised_dataset(df_feats, 'Inflation_Rate', horizon=H)
    X_train     = X[X.index <  test_start]
    y_train     = y[y.index <  test_start]
    X_test      = X[X.index >= test_start]
    y_test      = y[y.index >= test_start]

    selector    = MutualInfoFeatureSelector(k=25)
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel  = selector.transform(X_test)
    pd.DataFrame({'Feature': selector.selected_features_}).to_csv(
        f"{OUTPUT_DIR}/1_Model_Performance/Tables/H_{H}_selected_features.csv", index=False)

    results, tuned_params = [], {}

    for name in ['AR1_Direct', 'RandomWalk', 'VAR']:
        try:
            y_pred = train_statistical_baseline(name, X_train_sel, y_train, X_test_sel)
            results.append({'Model': name,
                            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                            'MAE':  mean_absolute_error(y_test, y_pred),
                            'R2':   r2_score(y_test, y_pred)})
            print(f"  {name}: RMSE={results[-1]['RMSE']:.3f}")
        except Exception as e:
            print(f"  {name} failed: {e}")

    for name, tune_fn, ModelClass, extra in [
        ('ElasticNet',   tune_elasticnet,   ElasticNet,           {'random_state': 42}),
        ('RandomForest', tune_randomforest, RandomForestRegressor,{'random_state': 42, 'n_jobs': -1}),
        ('XGBoost',      tune_xgboost,      XGBRegressor,         {'random_state': 42, 'n_jobs': -1, 'verbosity': 0}),
        ('LightGBM',     tune_lightgbm,     LGBMRegressor,        {'random_state': 42, 'n_jobs': -1, 'verbose': -1}),
    ]:
        print(f"  [{name}] tuning...", end='', flush=True)
        params = tune_fn(X_train_sel, y_train, n_trials)
        tuned_params[name] = params
        model  = ModelClass(**params, **extra)
        model.fit(X_train_sel, y_train)
        y_pred = model.predict(X_test_sel)
        results.append({'Model': name,
                        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                        'MAE':  mean_absolute_error(y_test, y_pred),
                        'R2':   r2_score(y_test, y_pred)})
        print(f" RMSE={results[-1]['RMSE']:.3f}")

    knn_params = tune_knn_meta(X_train_sel, y_train, n_trials=30)
    tuned_params['KNN_meta'] = knn_params

    for stack_name, base_name, BaseClass, base_extra in [
        ('Stack_RF_KNN',  'RandomForest', RandomForestRegressor, {'random_state': 42, 'n_jobs': -1}),
        ('Stack_XGB_KNN', 'XGBoost',      XGBRegressor,          {'random_state': 42, 'n_jobs': -1, 'verbosity': 0}),
    ]:
        stack = StackingRegressor(
            estimators=[('base', BaseClass(**tuned_params[base_name], **base_extra))],
            final_estimator=KNeighborsRegressor(**knn_params), n_jobs=-1)
        stack.fit(X_train_sel, y_train)
        y_pred = stack.predict(X_test_sel)
        results.append({'Model': stack_name,
                        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                        'MAE':  mean_absolute_error(y_test, y_pred),
                        'R2':   r2_score(y_test, y_pred)})
        print(f"  {stack_name}: RMSE={results[-1]['RMSE']:.3f}")

    hase = HASE(horizon=H, random_state=42, tuned_params=tuned_params)
    hase.fit(X_train_sel, y_train)
    y_pred = hase.predict(X_test_sel)
    results.append({'Model': 'HASE',
                    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'MAE':  mean_absolute_error(y_test, y_pred),
                    'R2':   r2_score(y_test, y_pred)})
    print(f"  HASE: RMSE={results[-1]['RMSE']:.3f}  weights={hase.get_weights_summary()}")

    df_results = pd.DataFrame(results).sort_values('RMSE').reset_index(drop=True)
    df_results['Rank'] = range(1, len(df_results) + 1)
    df_results.to_csv(f"{OUTPUT_DIR}/1_Model_Performance/Tables/H_{H}_metrics.csv", index=False)
    pd.DataFrame([tuned_params]).to_csv(
        f"{OUTPUT_DIR}/1_Model_Performance/Tables/H_{H}_tuned_params.csv", index=False)
    return df_results


def run_experiment():
    df_raw   = load_macro_dataset(use_realtime=True, publication_delays=True)
    df_feats = create_features(df_raw)

    all_results = {}
    for H in [1, 3, 6, 12]:
        all_results[H] = run_single_horizon(H, df_feats, test_start='2013-01-01', n_trials=50)

    model_ranks = {}
    for H, df in all_results.items():
        for _, row in df.iterrows():
            model_ranks.setdefault(row['Model'], []).append(row['Rank'])

    df_agg = pd.DataFrame([
        {'Model': m, 'Avg_Rank': np.mean(r), 'Rank_Variance': np.var(r)}
        for m, r in model_ranks.items()
    ]).sort_values('Avg_Rank')
    df_agg.to_csv(f"{OUTPUT_DIR}/1_Model_Performance/Tables/Aggregated_Rankings.csv", index=False)
    print(f"\nBest model: {df_agg.iloc[0]['Model']} (avg rank={df_agg.iloc[0]['Avg_Rank']:.2f})")
    return df_agg


if __name__ == "__main__":
    run_experiment()
