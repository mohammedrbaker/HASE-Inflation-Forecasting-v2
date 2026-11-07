# Explainable Horizon-Adaptive Stacked Learning for U.S. Inflation Forecasting with SHAP

**Baker, Mohammed Rashad** — University of Kirkuk, Iraq  
**Buyrukoğlu, Selim** — Kayseri University, Turkey  
**Jihad, Kamal H.** — University of Kirkuk, Iraq

---

## Requirements

```
Python >= 3.9
pip install -r requirements.txt
```

---

## Data Setup

All data are free and publicly available.

**Step 1 — Download ALFRED vintage files** from https://alfred.stlouisfed.org  
Download the following series and save each as `<ID>_vintage.csv`:

| Series ID | Description |
|-----------|-------------|
| PCE | Personal Consumption Expenditures |
| UNRATE | Unemployment Rate |
| INDPRO | Industrial Production Index |
| M2SL | M2 Money Stock |

**Step 2 — Download FRED market files** from https://fred.stlouisfed.org  
Download the following series and save each as `<ID>_market.csv`:

| Series ID | Description |
|-----------|-------------|
| FEDFUNDS → save as PolicyRate_market.csv | Federal Funds Rate |
| DCOILWTICO → save as WTI_market.csv | WTI Crude Oil Price |
| GS10 → save as GS10_market.csv | 10-Year Treasury Yield |
| TB3MS → save as TB3MS_market.csv | 3-Month T-Bill Rate |

**Step 3** — Place all 8 CSV files in one folder, then open `data_loader.py` and set:

```python
CACHE_DIR = 'path/to/your/data/folder'
```

---

## How to Run

Run the scripts in this order:

```bash
# Main experiment — all models, all horizons (Tables 5–9)
python run_experiment.py

# Rolling-origin validation (Table C.1)
python run_rolling_validation.py

# Prediction intervals (Figure 7)
python run_uncertainty.py

# SHAP feature importance (Figures 3–6)
python run_shap_analysis.py
```

All results are saved to a `results/` folder created automatically.

---

## File Overview

| File | Purpose |
|------|---------|
| `data_loader.py` | Loads ALFRED/FRED data |
| `feature_engineering.py` | Builds 55 stationary predictors |
| `models.py` | HASE model + 8 benchmarks |
| `run_experiment.py` | Main experiment with Optuna tuning |
| `run_rolling_validation.py` | Rolling-origin validation |
| `run_uncertainty.py` | 90% prediction intervals |
| `run_shap_analysis.py` | SHAP explainability analysis |

---

## Citation

```
Baker, M.R., Buyrukoğlu, S., & Jihad, K.H. (2026).
Explainable Horizon-Adaptive Stacked Learning for U.S. Inflation Forecasting with SHAP.
https://doi.org/10.5281/zenodo.24025941
        
        
https://doi.org/10.5281/zenodo.24025941
        
        
        
        
        
        
```

---

## License

MIT License
