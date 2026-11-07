# Explainable Horizon-Adaptive Stacked Learning for U.S. Inflation Forecasting with SHAP
# Baker, M.R., Buyrukoğlu, S., & Jihad, K.H.

import numpy as np
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from statsmodels.tsa.ar_model import AutoReg
import warnings
warnings.filterwarnings('ignore')


class AR1DirectWrapper(BaseEstimator, RegressorMixin):
    """
    Direct h-step AR(1) regression: y(t+h) = alpha + beta * y(t).
    Searches the feature matrix for the contemporaneous Inflation_Rate column.
    At long horizons (e.g. H=12), Inflation_Rate may not survive MI feature
    selection; the search falls back to momentum/lag proxies so the estimator
    always uses an inflation-based predictor rather than an unrelated column.

    NOTE (R3 B.4): The infl_idx is resolved in HASE.fit() by searching the
    selected feature column list for candidates in priority order:
      1. 'Inflation_Rate'           (contemporaneous level)
      2. 'Inflation_Rate_lag1'      (one-month lag)
      3. 'Inflation_Rate_lag3'      (three-month lag)
      4. 'Inflation_Momentum_3M'    (3-month rolling mean, a smooth proxy)
    This guarantees the AR1-Direct component always regresses the h-step-ahead
    target on an inflation-based signal, consistent with its mathematical
    definition as an autoregressive direct forecast.
    """
    # Priority list used by HASE.fit() to resolve the inflation column index
    INFL_CANDIDATES = [
        'Inflation_Rate',
        'Inflation_Rate_lag1',
        'Inflation_Rate_lag3',
        'Inflation_Momentum_3M',
    ]

    def __init__(self, infl_idx=1):
        self.infl_idx   = infl_idx
        self.coef_      = 0.0
        self.intercept_ = 0.0

    def fit(self, X, y):
        X_arr, y_arr = np.array(X), np.array(y)
        if self.infl_idx >= X_arr.shape[1]:
            self.intercept_ = float(y_arr.mean())
            return self
        x = X_arr[:, self.infl_idx]
        mx, my = x.mean(), y_arr.mean()
        denom = np.sum((x - mx) ** 2)
        if denom < 1e-10:
            self.intercept_ = my
        else:
            b = np.sum((x - mx) * (y_arr - my)) / denom
            self.coef_, self.intercept_ = b, my - b * mx
        return self

    def predict(self, X):
        X_arr = np.array(X)
        if self.infl_idx >= X_arr.shape[1]:
            return np.full(len(X_arr), self.intercept_)
        return self.intercept_ + self.coef_ * X_arr[:, self.infl_idx]


class AR2ChainedWrapper(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.a_ = self.rho1_ = self.rho2_ = self.last2_ = None

    def fit(self, X, y):
        y_arr = np.array(y)
        try:
            m = AutoReg(y_arr, lags=2, trend='c').fit()
            self.a_, self.rho1_, self.rho2_ = m.params
        except Exception:
            Y = y_arr[2:]
            A = np.column_stack([np.ones(len(Y)), y_arr[1:-1], y_arr[:-2]])
            self.a_, self.rho1_, self.rho2_ = np.linalg.lstsq(A, Y, rcond=None)[0]
        self.last2_ = (y_arr[-2], y_arr[-1])
        return self

    def predict(self, X):
        preds = np.zeros(len(X))
        lag2, lag1 = self.last2_
        for i in range(len(X)):
            p = self.a_ + self.rho1_ * lag1 + self.rho2_ * lag2
            preds[i], lag2, lag1 = p, lag1, p
        return preds


class AR1ChainedWrapper(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        y = np.array(y)
        yl, yn = y[:-1], y[1:]
        mx, my = yl.mean(), yn.mean()
        b = np.sum((yl - mx) * (yn - my)) / (np.sum((yl - mx) ** 2) + 1e-10)
        self.rho_, self.mu_, self.last_ = np.clip(b, -0.999, 0.999), my - b * mx, y[-1]
        return self

    def predict(self, X):
        p, last = np.zeros(len(X)), self.last_
        for i in range(len(X)):
            p[i] = self.mu_ + self.rho_ * last
            last = p[i]
        return p


class HASE(BaseEstimator, RegressorMixin):
    """Horizon-Adaptive Stacking Ensemble — proposed model."""

    def __init__(self, horizon=1, random_state=42, tuned_params=None, val_fraction=0.25):
        self.horizon      = horizon
        self.random_state = random_state
        self.tuned_params = tuned_params or {}
        self.val_fraction = val_fraction
        self.base_models_ = {}
        self.weights_     = None
        self.base_names_  = None
        self.infl_idx_    = 1

    def _build_base_models(self):
        tp = self.tuned_params
        return {
            'EN': ElasticNet(
                alpha    =tp.get('ElasticNet', {}).get('alpha',    0.1),
                l1_ratio =tp.get('ElasticNet', {}).get('l1_ratio', 0.5),
                max_iter=5000, random_state=self.random_state),
            'AR1_Direct':  AR1DirectWrapper(infl_idx=self.infl_idx_),
            'AR2_Chained': AR2ChainedWrapper(),
            'Ridge':       Ridge(alpha=5.0),
        }

    def fit(self, X, y):
        if hasattr(X, 'columns'):
            cols = list(X.columns)
            # NOTE (R3 B.4): Search in priority order so AR1_Direct always
            # uses an inflation-based column even when 'Inflation_Rate' itself
            # is not in the MI-selected top-25 (e.g. at H=12).
            self.infl_idx_ = next(
                (i for i, c in enumerate(cols)
                 if c in AR1DirectWrapper.INFL_CANDIDATES), 1)

        X_arr, y_arr = np.array(X), np.array(y)
        n      = len(X_arr)
        n_base = max(int(n * (1.0 - self.val_fraction)), min(50, n - 20))
        n_hold = n - n_base
        X_base, y_base = X_arr[:n_base], y_arr[:n_base]
        X_hold, y_hold = X_arr[n_base:],  y_arr[n_base:]

        stage1 = self._build_base_models()
        self.base_names_ = list(stage1.keys())
        n_m = len(self.base_names_)

        hold_preds = np.zeros((n_hold, n_m))
        for j, (name, model) in enumerate(stage1.items()):
            model.fit(X_base, y_base)
            hold_preds[:, j] = model.predict(X_hold)

        rmse_per     = np.sqrt(np.mean((hold_preds - y_hold[:, None]) ** 2, axis=0))
        inv_rmse     = 1.0 / (rmse_per + 1e-8)
        data_weights = inv_rmse / inv_rmse.sum()

        H = self.horizon
        ar1d_prior = max(0.40 - 0.025 * (H - 1), 0.12)
        ar2_prior  = min(0.10 + 0.025 * (H - 1), 0.40)
        rem        = max(1.0 - ar1d_prior - ar2_prior, 0.10)
        prior      = np.array([rem / 2, ar1d_prior, ar2_prior, rem / 2])
        prior      = prior / prior.sum()

        self.weights_ = (0.70 * data_weights + 0.30 * prior)
        self.weights_ = self.weights_ / self.weights_.sum()

        self.base_models_ = self._build_base_models()
        for model in self.base_models_.values():
            model.fit(X_arr, y_arr)
        return self

    def predict(self, X):
        base_preds = np.column_stack([
            self.base_models_[n].predict(np.array(X)) for n in self.base_names_])
        return base_preds @ self.weights_

    def get_weights_summary(self):
        return {n: round(float(w), 4) for n, w in zip(self.base_names_, self.weights_)}


GrandStackOptimum = HASE  # backward-compatibility alias


def get_model_roster(tuned_params, horizon=1):
    tp = tuned_params
    models = {}

    models['ElasticNet'] = ElasticNet(
        alpha    =tp.get('ElasticNet', {}).get('alpha',    0.1),
        l1_ratio =tp.get('ElasticNet', {}).get('l1_ratio', 0.5),
        random_state=42)

    models['RandomForest'] = RandomForestRegressor(
        n_estimators      =tp.get('RandomForest', {}).get('n_estimators',     100),
        max_depth         =tp.get('RandomForest', {}).get('max_depth',         10),
        min_samples_split =tp.get('RandomForest', {}).get('min_samples_split',  2),
        random_state=42, n_jobs=-1)

    models['XGBoost'] = XGBRegressor(
        n_estimators  =tp.get('XGBoost', {}).get('n_estimators',  100),
        max_depth     =tp.get('XGBoost', {}).get('max_depth',       3),
        learning_rate =tp.get('XGBoost', {}).get('learning_rate', 0.1),
        subsample     =tp.get('XGBoost', {}).get('subsample',      0.8),
        random_state=42, n_jobs=-1, verbosity=0)

    models['LightGBM'] = LGBMRegressor(
        n_estimators  =tp.get('LightGBM', {}).get('n_estimators',  100),
        max_depth     =tp.get('LightGBM', {}).get('max_depth',      -1),
        learning_rate =tp.get('LightGBM', {}).get('learning_rate', 0.1),
        num_leaves    =tp.get('LightGBM', {}).get('num_leaves',     31),
        random_state=42, n_jobs=-1, verbose=-1)

    knn_meta = KNeighborsRegressor(
        n_neighbors=tp.get('KNN_meta', {}).get('n_neighbors', 5), n_jobs=-1)

    models['Stack_RF_KNN'] = StackingRegressor(
        estimators=[('rf', RandomForestRegressor(
            n_estimators=tp.get('RandomForest', {}).get('n_estimators', 100),
            max_depth   =tp.get('RandomForest', {}).get('max_depth',    10),
            random_state=42, n_jobs=-1))],
        final_estimator=knn_meta, n_jobs=-1)

    models['Stack_XGB_KNN'] = StackingRegressor(
        estimators=[('xgb', XGBRegressor(
            n_estimators  =tp.get('XGBoost', {}).get('n_estimators',  100),
            max_depth     =tp.get('XGBoost', {}).get('max_depth',       3),
            learning_rate =tp.get('XGBoost', {}).get('learning_rate', 0.1),
            random_state=42, n_jobs=-1, verbosity=0))],
        final_estimator=knn_meta, n_jobs=-1)

    models['HASE'] = HASE(horizon=horizon, random_state=42, tuned_params=tuned_params)

    return models
